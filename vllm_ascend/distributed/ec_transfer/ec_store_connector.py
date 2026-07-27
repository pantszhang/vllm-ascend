#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""ECStoreConnector — EC connector backed by pluggable EC backends.

Configuration::

    {
        "ec_connector": "ECStoreConnector",
        "ec_role": "ec_both",
        "ec_connector_module_path": "vllm_ascend.distributed.ec_transfer.ec_store_connector",
        "ec_connector_extra_config": {
            "backend": "memcache",
            "lease_ttl_ms": 300000
        }
    }

Supports ``ec_producer``, ``ec_consumer``, and ``ec_both`` roles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tensor_model_parallel_rank,
)
from vllm.logger import init_logger

from vllm_ascend.distributed.ec_transfer.ec_backend import (
    create_ec_backend,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

    from vllm_ascend.distributed.ec_transfer.ec_backend.backend import ECBackend

logger = init_logger(__name__)

# Read lease TTL (ms). Must cover the duration of a single batch_copy(G2L)
# plus the subsequent remove_lease call (typically < 1 s for a 14 MB blob).
_DEFAULT_LEASE_TTL_MS = 300_000  # 5 minutes

# Transfer direction constants (mirrors _MmcDirect in memcache_backend).
_COPY_L2G = 0  # Local → Global (save)
_COPY_G2L = 1  # Global → Local (load)


@dataclass
class ECStoreConnectorMetadata(ECConnectorMetadata):
    """Per-step metadata sent from scheduler to worker.

    ``loads`` is the list of mm_hashes whose encoder embeddings
    should be fetched from the backend before the model forward.
    """

    loads: list[str] = field(default_factory=list)


class ECStoreConnector(ECConnectorBase):
    """Pluggable EC connector for encoder-cache offloading.

    Uses the EC backend registry (``ec_backend_map``) so the storage
    layer (memcache, mooncake, …) is selected through configuration
    rather than hard-coded.
    """

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)

        extra = vllm_config.ec_transfer_config.ec_connector_extra_config
        self._backend_name = extra.get("backend", "memcache").lower()
        self._lease_ttl_ms: int = extra.get("lease_ttl_ms", _DEFAULT_LEASE_TTL_MS)

        self._model_name = vllm_config.model_config.model.rstrip("/").split("/")[-1]
        self._dtype = vllm_config.model_config.dtype
        self._element_size = self._dtype.itemsize if hasattr(self._dtype, "itemsize") else 2

        # Try to derive encoder output hidden size from the model config.
        # Falls back to model hidden_size if vision_config isn't available.
        self._encoder_hidden_size = self._get_encoder_hidden_size(vllm_config)

        self._store: "ECBackend | None" = None

        self._tp_rank = get_tensor_model_parallel_rank()
        try:
            self._pcp_rank = get_pcp_group().rank_in_group
        except Exception:
            self._pcp_rank = 0

        # ---- Role-specific state ----
        # Scheduler
        self._pending_loads: dict[str, Any] = {}

        if role == ECConnectorRole.WORKER:
            self._store = create_ec_backend(
                self._backend_name, local_rank=self._resolve_local_rank()
            )
        elif role == ECConnectorRole.SCHEDULER:
            self._store = create_ec_backend(
                self._backend_name,
                local_rank=0,
                init_bm=False,
            )

    # ==================================================================
    # Scheduler-side methods
    # ==================================================================

    def has_cache_item(self, identifier: str) -> bool:
        if not self._is_consumer:
            return False
        assert self._store is not None
        key = self._make_key(identifier)
        result = self._store.batch_is_exist([key])
        return bool(result and result[0] == 1)

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        if not self._is_consumer:
            return
        mm_hash = request.mm_features[index].identifier
        if self.has_cache_item(mm_hash):
            self._pending_loads[mm_hash] = True

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECStoreConnectorMetadata:
        meta = ECStoreConnectorMetadata(loads=list(self._pending_loads))
        self._pending_loads.clear()
        return meta

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        return True

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    # ==================================================================
    # Worker-side methods
    # ==================================================================

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        if not self._is_consumer:
            return

        metadata: ECStoreConnectorMetadata = self._get_connector_metadata()  # type: ignore[assignment]
        if not metadata.loads:
            return
        assert self._store is not None

        for mm_hash in metadata.loads:
            key = self._make_key(mm_hash)
            try:
                key_infos = self._store.batch_get_key_info([key])
                ki = key_infos[0] if key_infos else None
                if ki is None:
                    logger.warning(
                        "EC load: key_info returned empty for mm_hash=%s", mm_hash
                    )
                    continue

                # Fetch size and GVA from the key-info object.
                sizes = ki.size()
                blob_size = sizes if isinstance(sizes, int) else (sizes[0] if sizes else 0)
                gva = ki.gva_list()[0] if blob_size and blob_size > 0 else 0

                if gva <= 0:
                    logger.warning(
                        "EC load: invalid gva=%s for mm_hash=%s (size=%s); skipping",
                        gva,
                        mm_hash,
                        blob_size,
                    )
                    continue

                # Reconstruct shape from byte size.
                token_bytes = self._encoder_hidden_size * self._element_size
                if token_bytes <= 0 or blob_size % token_bytes != 0:
                    logger.error(
                        "EC load: invalid shape params for mm_hash=%s "
                        "hidden_size=%d element_size=%d blob_size=%d token_bytes=%d",
                        mm_hash,
                        self._encoder_hidden_size,
                        self._element_size,
                        blob_size,
                        token_bytes,
                    )
                    continue
                num_tokens = blob_size // token_bytes
                if num_tokens <= 0:
                    logger.error(
                        "EC load: computed zero tokens for mm_hash=%s blob_size=%d token_bytes=%d",
                        mm_hash,
                        blob_size,
                        token_bytes,
                    )
                    continue

                logger.debug(
                    "EC load: mm_hash=%s size=%d num_tokens=%d hidden_size=%d",
                    mm_hash,
                    blob_size,
                    num_tokens,
                    self._encoder_hidden_size,
                )

                # Allocate destination buffer on NPU.
                buffer = torch.empty(
                    num_tokens,
                    self._encoder_hidden_size,
                    dtype=self._dtype,
                    device="npu",
                )

                # Acquire read lease to prevent eviction during copy.
                lease_res = self._store.batch_add_lease([key], self._lease_ttl_ms)
                lease_ok = lease_res and lease_res[0] == 0
                if not lease_ok:
                    logger.warning(
                        "EC load: lease failed for mm_hash=%s result=%s; "
                        "proceeding without lease guarantee",
                        mm_hash, lease_res,
                    )

                try:
                    # DMA from store to NPU.
                    copy_res = self._store.batch_copy(
                        [gva],
                        [buffer.data_ptr()],
                        [blob_size],
                        _COPY_G2L,
                    )
                    if copy_res != 0:
                        logger.error(
                            "EC load: batch_copy(G2L) failed for mm_hash=%s res=%d",
                            mm_hash, copy_res,
                        )
                finally:
                    if lease_ok:
                        self._store.batch_remove_lease([key])

                encoder_cache[mm_hash] = buffer
                logger.debug(
                    "EC load: success for mm_hash=%s shape=(%d,%d)",
                    mm_hash,
                    num_tokens,
                    self._encoder_hidden_size,
                )

            except Exception:
                logger.exception(
                    "EC load: unexpected error for mm_hash=%s", mm_hash
                )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        if not self._is_producer:
            return

        # Only one TP/PCP rank writes — all ranks hold identical encoder output.
        if self._tp_rank != 0 or self._pcp_rank != 0:
            return

        tensor = encoder_cache.get(mm_hash)
        if tensor is None:
            return

        key = self._make_key(mm_hash)
        data_ptr = tensor.data_ptr()
        data_bytes = tensor.numel() * tensor.element_size()

        assert self._store is not None

        # Dedup: skip if already stored by a concurrent request.
        exists = self._store.batch_is_exist([key])
        if exists and exists[0] == 1:
            logger.debug("EC save: mm_hash=%s already exists; skipping", mm_hash)
            return

        try:
            gva_list = self._store.batch_alloc([key], [data_bytes])
            gva = gva_list[0]

            copy_res = self._store.batch_copy(
                [gva],
                [data_ptr],
                [data_bytes],
                _COPY_L2G,
            )
            if copy_res != 0:
                logger.error(
                    "EC save: batch_copy(L2G) failed for mm_hash=%s res=%d",
                    mm_hash,
                    copy_res,
                )
            else:
                logger.debug(
                    "EC save: mm_hash=%s bytes=%d shape=%s",
                    mm_hash,
                    data_bytes,
                    list(tensor.shape),
                )
        except Exception:
            logger.exception("EC save: unexpected error for mm_hash=%s", mm_hash)

    # ==================================================================
    # Helpers
    # ==================================================================

    def _make_key(self, mm_hash: str) -> str:
        """Build the unique store key for a multimodal hash.

        Format: ``encoder@{model_name}@{mm_hash}``
        """
        return f"encoder@{self._model_name}@{mm_hash}"

    @staticmethod
    def _resolve_local_rank() -> int:
        import os
        try:
            from vllm import envs
            return envs.LOCAL_RANK
        except Exception:
            return int(os.environ.get("LOCAL_RANK", 0))

    @staticmethod
    def _get_encoder_hidden_size(vllm_config: "VllmConfig") -> int:
        """Derive encoder output hidden size from the model config."""
        hf_config = vllm_config.model_config.hf_config

        vision_config = getattr(hf_config, "vision_config", None)
        if vision_config is not None and hasattr(vision_config, "out_hidden_size"):
            hidden_size = vision_config.out_hidden_size
            # Qwen3-VL deepstack: multiscale features are concatenated
            # along the hidden‑size axis.
            deepstack_indexes = getattr(
                vision_config, "deepstack_visual_indexes", None
            )
            if deepstack_indexes:
                hidden_size *= 1 + len(deepstack_indexes)
            return int(hidden_size)

        # Fallback: most VL models project embeddings to text hidden_size.
        return vllm_config.model_config.hidden_size
