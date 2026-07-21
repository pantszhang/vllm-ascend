# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECMemCacheScheduler — scheduler-side EC connector delegate.

Tracks which mm_hashes are cached in MemCache, manages LRU eviction
with a configurable max entry count, and decides per-step save/load sets.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.step_tracker import (
    StepTracker,
)
from vllm.logger import init_logger
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.common import (
    ECMemCacheConnectorMetadata,
    _make_memcache_key,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.metrics import (
    ECMemCacheMetrics,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECMemCacheScheduler:
    """Scheduler delegate for ECMemCacheConnector.

    Responsibilities:
    - Track which mm_hashes exist in MemCache (LRU ordered dict).
    - Decide which embeddings to save (producer) and load (consumer).
    - Evict oldest entries when exceeding max_cached_entries.
    - Coordinate with StepTracker for async transfer completion.
    """

    def __init__(self, vllm_config: "VllmConfig") -> None:
        ec_config = vllm_config.ec_transfer_config
        assert ec_config is not None
        self._is_producer: bool = ec_config.is_ec_producer
        self._is_consumer: bool = ec_config.is_ec_consumer

        # MemCache client — scheduler has no NPU, contributes no memory.
        self._store = MemcacheBackend.create_scheduler_client(
            vllm_config.parallel_config
        )

        # LRU tracking: OrderedDict[mm_hash, num_tokens]
        self._cache_entries: OrderedDict[str, int] = OrderedDict()
        self._max_cached_entries: int = ec_config.get_from_extra_config(
            "ec_memcache_max_cached_entries", 10000
        )

        # Step synchronizers
        max_batches = vllm_config.max_concurrent_batches
        self._ready_tracker = StepTracker(max_batches)
        self._unpin_tracker = StepTracker(max_batches)

        # Pending per-step state
        self._pending_saves: dict[str, int] = {}
        self._pending_loads: dict[str, int] = {}

        # Metrics
        enable_metrics: bool = ec_config.get_from_extra_config(
            "ec_memcache_metrics_enable", False
        )
        self._metrics = ECMemCacheMetrics(enable=enable_metrics)

    # ========== Public API ==========

    def has_cache_item(self, mm_hash: str) -> bool:
        """Check whether MemCache holds an embedding for *mm_hash*.

        As a side-effect, promotes *mm_hash* to the LRU tail on hit.
        """
        if mm_hash in self._cache_entries:
            self._cache_entries.move_to_end(mm_hash)
            return True
        return False

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        """Record save/load intent after the scheduler allocates encoder cache.

        Producer: new mm_hash → pending_saves + ready_tracker.
        Consumer: cached mm_hash → pending_loads + unpin_tracker.
        """
        mm_hash = request.mm_features[index].identifier
        num_tokens = request.get_num_encoder_embeds(index)

        if self._is_producer and mm_hash not in self._cache_entries:
            self._pending_saves[mm_hash] = num_tokens
            self._ready_tracker.add(mm_hash, request.request_id)

        if self._is_consumer and mm_hash in self._cache_entries:
            self._pending_loads[mm_hash] = num_tokens
            self._unpin_tracker.add(mm_hash, request.request_id)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECMemCacheConnectorMetadata:
        """Build per-step metadata for worker consumption.

        Processes completed transfers via StepTracker, then packages
        the pending saves/loads into ECMemCacheConnectorMetadata.
        """
        finished = (
            scheduler_output.finished_req_ids
            if scheduler_output is not None
            else set()
        )

        # Mark entries whose GPU→MemCache DMA is complete
        for key in self._ready_tracker.step(finished):
            if key in self._pending_saves:
                self._cache_entries[key] = self._pending_saves[key]
            self._evict_if_needed()

        # Unpin entries whose MemCache→GPU DMA is complete
        for _key in self._unpin_tracker.step(finished):
            pass  # No pin/unpin needed — MemCache manages access

        meta = ECMemCacheConnectorMetadata()
        if self._is_producer:
            meta.saves = dict(self._pending_saves)
            self._pending_saves.clear()
        if self._is_consumer:
            meta.loads = dict(self._pending_loads)
            self._pending_loads.clear()
        return meta

    def shutdown(self) -> None:
        """Drain pending state on shutdown."""
        self._pending_loads.clear()
        self._pending_saves.clear()
        for _key in self._unpin_tracker.drain_all():
            pass
        for _key in self._ready_tracker.drain_all():
            pass
        self._is_producer = False
        self._is_consumer = False

    # ========== Internal ==========

    def _evict_if_needed(self) -> None:
        """LRU eviction: remove oldest entries when over max_cached_entries."""
        while len(self._cache_entries) > self._max_cached_entries:
            old_key, _ = self._cache_entries.popitem(last=False)
            try:
                self._store.remove(_make_memcache_key(old_key))
            except Exception:
                logger.debug(
                    "EC eviction: failed to remove key for %s", old_key, exc_info=True
                )
            self._metrics.record_remove(old_key, 0)
            logger.debug("EC LRU eviction: removed %s from MemCache", old_key)
