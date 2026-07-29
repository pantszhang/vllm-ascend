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

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import vllm.envs as vllm_envs
from vllm.logger import logger
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

from vllm_ascend.distributed.ec_transfer.ec_store_client import (
    EncoderCacheStoreClient,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


class EncoderCacheManagerWithStore(EncoderCacheManager):
    """Encoder cache manager backed by remote memcache store.

    Differences from upstream ``EncoderCacheManager``:

    - ``check_and_update_cache`` queries memcache via ZMQ when the local
      hot-cache misses.
    - No slot capacity tracking / eviction — memcache manages its own
      storage lifecycle.
    - ``get_freed_mm_hashes`` always returns an empty list.
    """

    def __init__(self, cache_size: int):
        super().__init__(cache_size)
        # Connect to the worker's EncoderCacheStore REP server.
        # Path is deterministic from VLLM_RPC_BASE_PATH (available in all
        # processes — same as pool_scheduler.py).
        dp_rank = int(os.environ.get("DATA_PARALLEL_RANK", "0"))
        socket_path = (
            f"ipc://{vllm_envs.VLLM_RPC_BASE_PATH}"
            f"/ec_lookup_dp_rank{dp_rank}"
        )
        self._ec_store_client = EncoderCacheStoreClient(socket_path)

    # ---- overrides ----

    def check_and_update_cache(self, request: "Request", input_id: int) -> bool:
        mm_hash = request.mm_features[input_id].identifier

        # 1. Local hot cache (nanosecond)
        if mm_hash in self.cached:
            self.cached[mm_hash].add(request.request_id)
            logger.info("EC lookup LOCAL_HIT: mm_hash=%s", mm_hash)
            return True

        # 2. ZMQ → memcache exists query (global truth)
        if self._ec_store_client.exists(mm_hash):
            self.cached[mm_hash] = {request.request_id}
            logger.info("EC lookup MEMCACHE_HIT: mm_hash=%s", mm_hash)
            return True

        logger.info("EC lookup MISS (will compute): mm_hash=%s", mm_hash)
        return False

    def allocate(self, request: "Request", input_id: int) -> None:
        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()
        self.cached[mm_hash].add(request_id)
        self.request_cached_ids.setdefault(request_id, set()).add(input_id)

    def free(self, request: "Request") -> None:
        for input_id in list(self.get_cached_input_ids(request)):
            self.free_encoder_input(request, input_id)

    def free_encoder_input(self, request: "Request", input_id: int) -> None:
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        if req_id in self.request_cached_ids:
            self.request_cached_ids[req_id].discard(input_id)
            if not self.request_cached_ids[req_id]:
                del self.request_cached_ids[req_id]
        if mm_hash in self.cached:
            self.cached[mm_hash].discard(req_id)
            if not self.cached[mm_hash]:
                del self.cached[mm_hash]

    def get_freed_mm_hashes(self) -> list[str]:
        # memcache manages eviction; scheduler never instructs the worker
        # to free encoder cache entries.
        return []

    def can_allocate(
        self,
        request: "Request",
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        num_embeds = request.get_num_encoder_embeds(input_id)
        # Only check compute budget; memcache manages storage capacity.
        if num_embeds > encoder_compute_budget:
            return False
        return True
