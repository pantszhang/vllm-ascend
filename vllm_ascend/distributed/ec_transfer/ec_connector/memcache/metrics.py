# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECMemCacheMetrics — observability for EC MemCache Connector.

Tracks hit rates by storage medium, capacity statistics, error counts,
and encoder recompute savings. Zero overhead when disabled.
"""

import time
from vllm.logger import init_logger

logger = init_logger(__name__)


class ECMemCacheMetrics:
    """Observability collector for EC MemCache Connector.

    Args:
        enable: When False (default), all record methods are no-ops.
    """

    def __init__(self, enable: bool = False):
        self._enable = enable
        if not enable:
            return

        # Hit/medium counters
        self.hbm_hits = 0
        self.memcache_hits = 0
        self.memcache_misses = 0
        self.saves = 0
        self.loads = 0
        self.dram_loads = 0
        self.ssd_loads = 0
        self.dram_promotions = 0  # SSD → DRAM

        # Capacity
        self.current_cached_entries = 0
        self.total_embedding_bytes = 0
        self.max_embedding_size_bytes = 0

        # Errors
        self.alloc_failures = 0
        self.copy_errors = 0
        self.save_skipped = 0

        # Benefit
        self.encoder_recomputes_avoided = 0

        # Age tracking
        self._save_times: dict[str, float] = {}

    # ========== Record methods ==========

    def record_save(self, mm_hash: str, size_bytes: int, media: str) -> None:
        if not self._enable:
            return
        self.saves += 1
        self.current_cached_entries += 1
        self.total_embedding_bytes += size_bytes
        if size_bytes > self.max_embedding_size_bytes:
            self.max_embedding_size_bytes = size_bytes
        self._save_times[mm_hash] = time.time()
        logger.debug(
            "EC save [%s]: size=%.1fMB → %s, entries=%d, total=%.1fMB",
            mm_hash,
            size_bytes / 1e6,
            media,
            self.current_cached_entries,
            self.total_embedding_bytes / 1e6,
        )

    def record_memcache_hit(self, mm_hash: str, media: str) -> None:
        if not self._enable:
            return
        self.memcache_hits += 1
        self.encoder_recomputes_avoided += 1
        age = time.time() - self._save_times.get(mm_hash, 0)
        if media == "DRAM":
            self.dram_loads += 1
        elif media == "SSD":
            self.ssd_loads += 1
            self.dram_promotions += 1
        logger.info(
            "EC MemCache hit [%s]: medium=%s, age=%.1fs, avoided_recompute",
            mm_hash,
            media,
            age,
        )

    def record_memcache_miss(self, mm_hash: str) -> None:
        if not self._enable:
            return
        self.memcache_misses += 1
        logger.debug("EC MemCache miss [%s]", mm_hash)

    def record_hbm_hit(self, mm_hash: str) -> None:
        if not self._enable:
            return
        self.hbm_hits += 1
        self.encoder_recomputes_avoided += 1
        age = time.time() - self._save_times.get(mm_hash, 0)
        logger.debug("EC HBM hit [%s]: age=%.1fs", mm_hash, age)

    def record_alloc_failure(self, mm_hash: str) -> None:
        if not self._enable:
            return
        self.alloc_failures += 1
        logger.warning("EC alloc failure [%s]: MemCache pool full", mm_hash)

    def record_copy_error(self, mm_hash: str, direction: str) -> None:
        if not self._enable:
            return
        self.copy_errors += 1
        logger.error("EC copy error [%s]: direction=%s", mm_hash, direction)

    def record_save_skipped(self, mm_hash: str, reason: str) -> None:
        if not self._enable:
            return
        self.save_skipped += 1
        logger.debug("EC save skipped [%s]: %s", mm_hash, reason)

    def record_remove(self, mm_hash: str, size_bytes: int) -> None:
        if not self._enable:
            return
        self.current_cached_entries = max(0, self.current_cached_entries - 1)
        self.total_embedding_bytes = max(0, self.total_embedding_bytes - size_bytes)
        self._save_times.pop(mm_hash, None)

    # ========== Summary ==========

    def log_summary(self) -> None:
        if not self._enable:
            return
        total_mc = self.memcache_hits + self.memcache_misses
        mc_rate = f"{self.memcache_hits / total_mc:.1%}" if total_mc else "N/A"
        total_lookups = self.hbm_hits + self.memcache_hits + self.memcache_misses
        overall_rate = (
            f"{(self.hbm_hits + self.memcache_hits) / total_lookups:.1%}"
            if total_lookups
            else "N/A"
        )
        logger.info(
            "EC MemCache summary | "
            "hit: mc=%s overall=%s | "
            "media: dram=%d ssd=%d promotions=%d | "
            "capacity: entries=%d total=%.1fMB max=%.1fMB | "
            "errors: alloc_fail=%d copy_err=%d skip=%d | "
            "benefit: saved=%d saves=%d",
            mc_rate,
            overall_rate,
            self.dram_loads,
            self.ssd_loads,
            self.dram_promotions,
            self.current_cached_entries,
            self.total_embedding_bytes / 1e6,
            self.max_embedding_size_bytes / 1e6,
            self.alloc_failures,
            self.copy_errors,
            self.save_skipped,
            self.encoder_recomputes_avoided,
            self.saves,
        )

    def reset(self) -> None:
        """Reset all counters. Preserves enable state."""
        enabled = self._enable
        self.__init__(enable=enabled)
