# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECMemCacheMetrics -- observability for EC Score Encoder MemCache integration.

Tracks hit rates by storage medium, promotion/demotion churn, capacity
statistics, and error counts. Zero overhead when disabled.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


class ECMemCacheMetrics:
    """Observability collector for EC Score Encoder Cache + MemCache.

    Args:
        enable: When False (default), all record methods are no-ops.
        log_interval_steps: Periodic summary log every N steps (default 300).
    """

    def __init__(self, enable: bool = False, log_interval_steps: int = 300):
        self._enable = enable
        self._log_interval = log_interval_steps
        if not enable:
            return

        # Hit counters by medium
        self.hbm_hits = 0
        self.memcache_hits = 0
        self.memcache_misses = 0
        self.dram_hits = 0
        self.ssd_hits = 0

        # Promotion / demotion churn
        self.promotions = 0  # MemCache -> NPU
        self.demotions_from_npu = 0  # NPU -> MemCache
        self.dram_promotions = 0  # SSD -> DRAM (rewarm)

        # Save / remove
        self.saves = 0
        self.removes = 0

        # Capacity
        self.memcache_cached_entries = 0
        self.total_embedding_bytes = 0
        self.max_embedding_bytes = 0

        # Errors
        self.alloc_failures = 0
        self.copy_errors = 0

        # Benefit
        self.encoder_recomputes_avoided = 0

        self._step_last_logged = 0

    # -- Record methods --

    def record_hbm_hit(self, mm_hash: str) -> None:
        if not self._enable:
            return
        self.hbm_hits += 1
        self.encoder_recomputes_avoided += 1

    def record_memcache_hit(self, mm_hash: str, media: str) -> None:
        if not self._enable:
            return
        self.memcache_hits += 1
        self.encoder_recomputes_avoided += 1
        if media == "SSD":
            self.ssd_hits += 1
            self.dram_promotions += 1
        else:
            self.dram_hits += 1

    def record_memcache_miss(self, mm_hash: str) -> None:
        if not self._enable:
            return
        self.memcache_misses += 1

    def record_promotion(self, mm_hash: str) -> None:
        if not self._enable:
            return
        self.promotions += 1

    def record_demotion_from_npu(self, mm_hash: str) -> None:
        if not self._enable:
            return
        self.demotions_from_npu += 1

    def record_save(self, mm_hash: str, size_bytes: int) -> None:
        if not self._enable:
            return
        self.saves += 1
        self.memcache_cached_entries += 1
        self.total_embedding_bytes += size_bytes
        if size_bytes > self.max_embedding_bytes:
            self.max_embedding_bytes = size_bytes

    def record_remove(self, mm_hash: str, size_bytes: int) -> None:
        if not self._enable:
            return
        self.removes += 1
        self.memcache_cached_entries = max(0, self.memcache_cached_entries - 1)
        self.total_embedding_bytes = max(0, self.total_embedding_bytes - size_bytes)

    def record_alloc_failure(self, mm_hash: str) -> None:
        if not self._enable:
            return
        self.alloc_failures += 1

    def record_copy_error(self, mm_hash: str, direction: str) -> None:
        if not self._enable:
            return
        self.copy_errors += 1

    def record_npu_usage(self, npu_used: int, npu_total: int) -> None:
        """Snapshot NPU slot usage ratio for the next log_summary."""
        if not self._enable:
            return
        self._npu_used = npu_used
        self._npu_total = npu_total

    # -- Periodic summary --

    def log_summary_if_needed(self, step: int) -> None:
        if not self._enable:
            return
        if step - self._step_last_logged < self._log_interval:
            return
        self._step_last_logged = step

        total = self.memcache_hits + self.memcache_misses
        mc_rate = f"{self.memcache_hits / total:.1%}" if total else "N/A"
        all_lookups = self.hbm_hits + total
        overall_rate = (
            f"{(self.hbm_hits + self.memcache_hits) / all_lookups:.1%}"
            if all_lookups
            else "N/A"
        )

        npu_used = getattr(self, "_npu_used", 0)
        npu_total = getattr(self, "_npu_total", 1)
        npu_pct = f"{npu_used}/{npu_total}({npu_used / max(1, npu_total):.1%})"

        logger.info(
            "[ECMemCacheMetrics] step=%d | hit: hbm=%d mc=%s overall=%s | "
            "media: dram=%d ssd=%d rewarm=%d | "
            "capacity: npu_used=%s mc_entries=%d mc_bytes=%.1fMB max=%.1fMB | "
            "churn: promotions=%d demotions=%d removes=%d | "
            "errors: alloc_fail=%d copy_err=%d | "
            "benefit: avoided=%d saves=%d",
            step,
            self.hbm_hits,
            mc_rate,
            overall_rate,
            self.dram_hits,
            self.ssd_hits,
            self.dram_promotions,
            npu_pct,
            self.memcache_cached_entries,
            self.total_embedding_bytes / 1e6,
            self.max_embedding_bytes / 1e6,
            self.promotions,
            self.demotions_from_npu,
            self.removes,
            self.alloc_failures,
            self.copy_errors,
            self.encoder_recomputes_avoided,
            self.saves,
        )

    def reset(self) -> None:
        """Reset all counters. Preserves enable state."""
        enabled = self._enable
        self.__init__(enable=enabled, log_interval_steps=self._log_interval)
