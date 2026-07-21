# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECMemCacheWorker — worker-side EC connector delegate.

Performs NPU↔MemCache data transfer using MemCache's GVA API:
- Save: batch_alloc → batch_copy(L2G) — NPU HBM → MemCache.
- Load: batch_get_key_info → batch_copy(G2L) — MemCache → NPU HBM.

Uses a dedicated NPU stream for loads to overlap with compute.
"""

from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.common import (
    ECMemCacheConnectorMetadata,
    _get_encoder_cache_hidden_dim,
    _make_memcache_key,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.metrics import (
    ECMemCacheMetrics,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ECMemCacheWorker:
    """Worker-side delegate for ECMemCacheConnector.

    - Producer role: copies encoder_cache[mm_hash] (NPU HBM) → MemCache GVA.
    - Consumer role: copies MemCache GVA → encoder_cache (NPU HBM).
    - A dedicated load_stream overlaps MemCache→NPU copies with compute.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        self._is_producer: bool = ec_config.is_ec_producer
        self._is_consumer: bool = ec_config.is_ec_consumer
        self._dtype = vllm_config.model_config.dtype
        self._vllm_config = vllm_config

        # Reuse existing MemcacheBackend pattern
        from vllm.distributed.parallel_state import get_world_group

        local_rank = get_world_group().local_rank
        self._store = MemcacheBackend(
            vllm_config.parallel_config, local_rank=local_rank, init_bm=True
        )

        # Dedicated stream for async MemCache → NPU copies
        self._load_stream = current_platform.Stream()

        # Metrics (zero-overhead when disabled)
        enable_metrics: bool = ec_config.get_from_extra_config(
            "ec_memcache_metrics_enable", False
        )
        self._metrics = ECMemCacheMetrics(enable=enable_metrics)

    # ========== Producer: Save ==========

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        connector_metadata: ECMemCacheConnectorMetadata,
    ) -> None:
        """Save encoder_cache[mm_hash] from NPU HBM → MemCache via GVA.

        Uses batch_alloc to reserve space, then batch_copy(L2G) for
        direct NPU-to-MemCache transfer.
        """
        if not self._is_producer:
            return
        if mm_hash not in connector_metadata.saves:
            return

        tensor = encoder_cache[mm_hash]
        key = _make_memcache_key(mm_hash)
        size = tensor.numel() * tensor.element_size()

        gva = self._store.batch_alloc([key], [size])
        if not gva or gva[0] == 0:
            self._metrics.record_alloc_failure(mm_hash)
            return

        ret = self._store.batch_copy(
            [gva[0]],
            [tensor.data_ptr()],
            [size],
            direct=MmcDirect.COPY_L2G.value,
        )
        if ret != 0:
            self._metrics.record_copy_error(mm_hash, "L2G")
            return

        self._metrics.record_save(mm_hash, size, "DRAM")

    # ========== Consumer: Load ==========

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        connector_metadata: ECMemCacheConnectorMetadata,
    ) -> None:
        """Load embeddings from MemCache → encoder_cache (NPU HBM).

        Uses a dedicated load_stream so copies overlap with compute.
        Uses batch_get_key_info → batch_copy(G2L) for direct transfer.
        """
        if not self._is_consumer:
            return
        if not connector_metadata.loads:
            return

        element_size = torch.empty(0, dtype=self._dtype).element_size()

        with current_platform.stream(self._load_stream):
            for mm_hash, num_tokens in connector_metadata.loads.items():
                # HBM hit — already in encoder_cache from this or prior step
                if mm_hash in encoder_cache:
                    self._metrics.record_hbm_hit(mm_hash)
                    continue

                key = _make_memcache_key(mm_hash)
                key_infos = self._store.batch_get_key_info([key], flag=1)
                if (
                    key_infos is None
                    or len(key_infos) == 0
                    or key_infos[0].size() == 0
                ):
                    self._metrics.record_memcache_miss(mm_hash)
                    continue

                gva_list = key_infos[0].gva_list()
                if not gva_list or gva_list[0] == 0:
                    self._metrics.record_memcache_miss(mm_hash)
                    continue

                hidden_dim = _get_encoder_cache_hidden_dim(self._vllm_config)
                expected_size = num_tokens * hidden_dim * element_size

                tensor = torch.empty(
                    num_tokens,
                    hidden_dim,
                    dtype=self._dtype,
                    device=current_platform.device_type,
                )

                ret = self._store.batch_copy(
                    [gva_list[0]],
                    [tensor.data_ptr()],
                    [expected_size],
                    direct=MmcDirect.COPY_G2L.value,
                )
                if ret != 0:
                    self._metrics.record_copy_error(mm_hash, "G2L")
                    continue

                type_list = key_infos[0].type_list()
                media = "SSD" if type_list and 2 in type_list else "DRAM"
                self._metrics.record_memcache_hit(mm_hash, media)
                encoder_cache[mm_hash] = tensor

        # Ensure loads complete before compute reads the tensors
        current_platform.current_stream().wait_stream(self._load_stream)

    def shutdown(self) -> None:
        """Synchronize streams on shutdown."""
        if hasattr(self, "_load_stream"):
            self._load_stream.synchronize()
