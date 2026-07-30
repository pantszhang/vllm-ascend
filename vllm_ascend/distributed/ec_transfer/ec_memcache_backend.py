#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Minimal memcache backend for encoder embedding offload.

Wraps ``memcache_hybrid.DistributedObjectStore`` directly — no dependency
on the KV-pool backend module.
"""

from __future__ import annotations

import time

import torch
import torch.distributed
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

logger = init_logger(__name__)

# Wait for memcache internal threads after store init (same as KV pool backend)
_STORE_INIT_WAIT_S = 0.1


class EcMemcacheBackend:
    """Lightweight memcache wrapper for embedding storage.

    Only exposes the six methods needed by ``EncoderCacheStore``:
    ``exists``, ``batch_alloc``, ``batch_get_key_info``, ``batch_add_lease``,
    ``batch_remove_lease``, and ``batch_copy``.
    """

    def __init__(self, local_rank: int):
        self._local_rank = local_rank
        self._is_a2 = get_ascend_device_type() in {AscendDeviceType.A2}
        self._store = self._init_store()

    def _init_store(self):
        try:
            from memcache_hybrid import DistributedObjectStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install memcache_hybrid to use embedding memcache offload. "
                "See https://gitee.com/ascend/memfabric_hybrid"
            ) from e

        # A2 devices need an all_gather warmup before store init
        if self._is_a2:
            tmp = torch.zeros(1, device="npu")
            out = [torch.empty_like(tmp) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(out, tmp, group=get_world_group().device_group)

        store = DistributedObjectStore()
        res = store.init(self._local_rank, init_bm=True)
        if res != 0:
            raise RuntimeError(
                f"DistributedObjectStore.init failed with code {res}. "
                f"Check memcache configuration and environment."
            )
        time.sleep(_STORE_INIT_WAIT_S)
        return store

    _COPY_L2G = 0
    _COPY_G2L = 1

    # ---- EncoderCacheStore needs only these ----

    def exists(self, keys: list[str]) -> list[int]:
        return self._store.batch_is_exist(keys)

    def put(self, keys: list[str], addrs: list[int], sizes: list[int]):
        """Store data from local NPU memory to memcache.

        Uses the layered API that works with all protocols (host_shm, sdma, rdma).
        """
        return self._store.batch_put_from_layers(keys, addrs, sizes, self._COPY_L2G)

    def get(self, keys: list[str], addrs: list[int], sizes: list[int]):
        """Load data from memcache to local NPU memory.

        Uses the layered API that works with all protocols (host_shm, sdma, rdma).
        """
        return self._store.batch_get_into_layers(keys, addrs, sizes, self._COPY_G2L)
