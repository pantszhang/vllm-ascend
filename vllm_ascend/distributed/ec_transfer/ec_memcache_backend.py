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

Uses batch_put_from_layers / batch_get_into_layers (batch APIs).
Single-medium config (DRAM only) means direction validation is skipped
by HYBM, so register_buffer is not needed.
"""

from __future__ import annotations

import time

import torch
import torch.distributed
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import init_logger
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

logger = init_logger(__name__)

_STORE_INIT_WAIT_S = 0.1

_COPY_L2G = 0  # SMEMB_COPY_L2G
_COPY_G2L = 1  # SMEMB_COPY_G2L

_MEDIA_NAMES = {0: "HBM", 1: "DRAM", 2: "SSD"}

def _media_name(key_info) -> str:
    """Return human-readable media type from a KeyInfo object."""
    try:
        types = key_info.type_list()
        if types:
            return _MEDIA_NAMES.get(types[0], str(types[0]))
    except Exception:
        pass
    return "?"


class EcMemcacheBackend:
    """Lightweight memcache wrapper for embedding storage.

    Exposes three methods: ``exists``, ``put``, and ``get``.
    """

    def __init__(self, local_rank: int):
        self._local_rank = local_rank
        self._is_a2 = get_ascend_device_type() in {AscendDeviceType.A2}
        self._store = self._init_store()
        # statistics
        self._cnt_stores: int = 0
        self._cnt_hits: dict[str, int] = {"HBM": 0, "DRAM": 0, "SSD": 0}
        self._cnt_misses: int = 0

    def _init_store(self):
        try:
            from memcache_hybrid import DistributedObjectStore  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Please install memcache_hybrid to use embedding memcache offload. "
                "See https://gitee.com/ascend/memfabric_hybrid"
            ) from e

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

    # ---- public API ----

    def exists(self, keys: list[str]) -> list[int]:
        return self._store.batch_is_exist(keys)

    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Store *tensor* under *key* via batch_put_from_layers."""
        logger.info("EC DEBUG put() called: key=%s", key)
        addr = tensor.data_ptr()
        nbytes = tensor.nbytes
        results = self._store.batch_put_from_layers(
            [key],
            [[addr]],
            [[nbytes]],
            _COPY_L2G,
        )
        ret = results[0] if results else -1
        if ret != 0:
            raise RuntimeError(
                f"EcMemcacheBackend.put: batch_put_from_layers(L2G) failed "
                f"ret={ret} key={key} addr=0x{addr:x} nbytes={nbytes}"
            )
        self._cnt_stores += 1
        logger.info("EC memcache STORE: key=%s nbytes=%d %s",
                     key, nbytes, self._stats())

    def get(
        self, key: str, elem_size: int, hidden_dim: int, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Load data for *key* via batch_get_key_info + batch_get_into_layers."""
        key_infos = self._store.batch_get_key_info([key])
        ki = key_infos[0]
        if ki.size() == 0:
            self._cnt_misses += 1
            logger.info("EC memcache MISS: key=%s %s", key, self._stats())
            return None
        nbytes = ki.size()
        media = _media_name(ki)
        self._cnt_hits[media] = self._cnt_hits.get(media, 0) + 1
        num_tokens = nbytes // elem_size // hidden_dim
        tensor = torch.empty(
            num_tokens, hidden_dim, dtype=dtype, device="npu"
        )
        addr = tensor.data_ptr()
        results = self._store.batch_get_into_layers(
            [key],
            [[addr]],
            [[nbytes]],
            _COPY_G2L,
        )
        ret = results[0] if results else -1
        if ret != 0:
            raise RuntimeError(
                f"EcMemcacheBackend.get: batch_get_into_layers(G2L) failed "
                f"ret={ret} key={key} addr=0x{addr:x} nbytes={nbytes}"
            )
        logger.info(
            "EC memcache HIT: key=%s nbytes=%d tokens=%d media=%s %s",
            key, nbytes, num_tokens, media, self._stats(),
        )
        return tensor

    def _stats(self) -> str:
        return (
            f"[stores={self._cnt_stores} "
            f"hits={sum(self._cnt_hits.values())} "
            f"hbm_hits={self._cnt_hits.get('HBM',0)} "
            f"dram_hits={self._cnt_hits.get('DRAM',0)} "
            f"misses={self._cnt_misses}]"
        )
