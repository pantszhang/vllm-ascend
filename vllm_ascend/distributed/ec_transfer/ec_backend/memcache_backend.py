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
"""ECMemcacheBackend — memcache_hybrid-based EC backend.

Thin wrapper over ``memcache_hybrid.DistributedObjectStore``, exposing
only the subset of the API needed for encoder-cache offloading.
"""

import threading
from enum import Enum
from typing import Any

from vllm.logger import init_logger

from vllm_ascend.distributed.ec_transfer.ec_backend.backend import ECBackend
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

logger = init_logger(__name__)


class _MmcDirect(Enum):
    """Transfer direction constants (mirrors memcache_hybrid)."""

    COPY_L2G = 0  # Local → Global (save)
    COPY_G2L = 1  # Global → Local (load)


class ECMemcacheBackend(ECBackend):
    """Memcache-hybrid backend for encoder-cache offloading.

    Parameters:
        local_rank: NPU device local rank (scheduler side uses 0).
        init_bm: Whether to run the barrier-marker all-gather on A2.
        lazy_init: If True, defer ``DistributedObjectStore`` init until
            the first ``batch_alloc`` or ``batch_add_lease`` call.
    """

    def __init__(
        self,
        local_rank: int = 0,
        init_bm: bool = True,
        lazy_init: bool = False,
    ):
        self._local_rank = local_rank
        self._init_bm = init_bm
        self._is_a2 = get_ascend_device_type() in {AscendDeviceType.A2}
        self._lazy_init = lazy_init and not self._is_a2

        self.store: Any | None = None
        self._store_initialized = False
        self._store_init_lock = threading.Lock()

        if not self._lazy_init:
            self.store = self._setup_store()
            self._store_initialized = True

    # ------------------------------------------------------------------
    # Factory: scheduler-side lightweight client
    # ------------------------------------------------------------------

    @classmethod
    def create_scheduler_client(cls, **kwargs) -> "ECMemcacheBackend":
        """Lightweight client for the scheduler process.

        Uses ``local_rank=0`` and ``init_bm=False`` — no GPUDirect / NCCL
        barrier-marker initialization.  Only ``batch_is_exist`` is
        guaranteed to work.
        """
        return cls(local_rank=0, init_bm=False)

    # ------------------------------------------------------------------
    # Per-use API (inline with existing memcache patterns)
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> None:
        """Bring up the store if lazy-init was requested."""
        if self._store_initialized:
            return
        with self._store_init_lock:
            if self._store_initialized:
                return
            logger.info(
                "Initializing ECMemcacheBackend store. local_rank=%d",
                self._local_rank,
            )
            self.store = self._setup_store()
            self._store_initialized = True

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        if self._lazy_init and not self._store_initialized:
            logger.debug(
                "ECMemcacheBackend.batch_is_exist before store init; "
                "treating %d keys as missing.",
                len(keys),
            )
            return [0] * len(keys)
        assert self.store is not None
        return self.store.batch_is_exist(keys)

    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        self.ensure_initialized()
        assert self.store is not None
        return self.store.batch_alloc(keys, sizes)

    def batch_get_key_info(self, keys: list[str]):
        self.ensure_initialized()
        assert self.store is not None
        return self.store.batch_get_key_info(keys)

    def batch_copy(
        self,
        gvas: list[int],
        addrs: list[int],
        sizes: list[int],
        direction: int,
    ) -> int:
        self.ensure_initialized()
        assert self.store is not None
        try:
            res = self.store.batch_copy(gvas, addrs, sizes, direction)
            if res != 0:
                dir_name = (
                    "save(L2G)" if direction == _MmcDirect.COPY_L2G.value else "load(G2L)"
                )
                logger.error(
                    "ECMemcacheBackend.batch_copy %s FAILED res=%d",
                    dir_name,
                    res,
                )
            return res
        except Exception as e:
            logger.error(
                "ECMemcacheBackend.batch_copy failed: type=%s error=%s",
                type(e).__name__,
                e,
            )
            return -1

    def batch_add_lease(self, keys: list[str], ttl_ms: int) -> list[int]:
        self.ensure_initialized()
        assert self.store is not None
        return self.store.batch_add_lease(keys, ttl_ms)

    def batch_remove_lease(self, keys: list[str]) -> int:
        self.ensure_initialized()
        assert self.store is not None
        return self.store.batch_remove_lease(keys)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_store(self):
        try:
            from memcache_hybrid import DistributedObjectStore  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "Please install memcache by following the instructions at "
                "https://gitee.com/ascend/memfabric_hybrid "
                "to use EC memcache backend."
            ) from e

        # On A2 the store init broadcasts a barrier-marker tensor.
        if self._init_bm and self._is_a2:
            import torch
            import torch.distributed

            tmp_tensor = torch.zeros(1, device="npu")
            world_size = torch.distributed.get_world_size()
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(world_size)
            ]
            torch.distributed.all_gather(output_tensor_list, tmp_tensor)

        store = DistributedObjectStore()
        try:
            res = store.init(self._local_rank, init_bm=self._init_bm)
        except ValueError as e:
            logger.error(
                "ECMemcacheBackend: config loading failed. error=%s. "
                "Check MMC_LOCAL_CONFIG_PATH and memcache config.",
                e,
            )
            raise
        except Exception as exc:
            logger.error(
                "ECMemcacheBackend: store init failed. error=%s. "
                "Check memcache setup and dependencies.",
                exc,
            )
            raise

        if res != 0:
            raise RuntimeError(
                f"ECMemcacheBackend: DistributedObjectStore.init returned {res}"
            )
        return store
