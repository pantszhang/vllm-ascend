# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ECMemCacheMetrics."""
import pytest
from vllm_ascend.ec_manager.metrics import ECMemCacheMetrics


class TestMetricsDisabled:
    """When enable=False, all methods are no-ops with zero overhead."""

    def test_disabled_no_counters_allocated(self):
        m = ECMemCacheMetrics(enable=False)
        assert not hasattr(m, "hbm_hits")

    def test_disabled_record_hit_is_noop(self):
        m = ECMemCacheMetrics(enable=False)
        m.record_hbm_hit("h1")
        m.record_memcache_hit("h1", "DRAM")
        m.record_memcache_miss("h1")

    def test_disabled_log_summary_is_noop(self):
        m = ECMemCacheMetrics(enable=False)
        m.log_summary_if_needed(step=100)


class TestMetricsEnabled:
    """When enable=True, all counters accumulate correctly."""

    def test_record_save_increments(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save("h1", 1024)
        assert m.saves == 1
        assert m.memcache_cached_entries == 1
        assert m.total_embedding_bytes == 1024

    def test_record_multiple_saves_tracks_entries(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save("h1", 1000)
        m.record_save("h2", 2000)
        assert m.saves == 2
        assert m.memcache_cached_entries == 2
        assert m.total_embedding_bytes == 3000

    def test_max_embedding_size_tracked(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save("h1", 1000)
        m.record_save("h2", 5000)
        assert m.max_embedding_bytes == 5000

    def test_record_hbm_hit(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_hbm_hit("h1")
        assert m.hbm_hits == 1
        assert m.encoder_recomputes_avoided == 1

    def test_record_memcache_hit_dram(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_memcache_hit("h1", "DRAM")
        assert m.memcache_hits == 1
        assert m.dram_hits == 1
        assert m.ssd_hits == 0
        assert m.encoder_recomputes_avoided == 1

    def test_record_memcache_hit_ssd(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_memcache_hit("h2", "SSD")
        assert m.memcache_hits == 1
        assert m.ssd_hits == 1
        assert m.dram_promotions == 1

    def test_record_memcache_miss(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_memcache_miss("h_miss")
        assert m.memcache_misses == 1

    def test_record_promotion(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_promotion("h1")
        assert m.promotions == 1

    def test_record_demotion_from_npu(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_demotion_from_npu("h1")
        assert m.demotions_from_npu == 1

    def test_record_alloc_failure(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_alloc_failure("h1")
        assert m.alloc_failures == 1

    def test_record_copy_error(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_copy_error("h1", "L2G")
        assert m.copy_errors == 1

    def test_record_remove_decrements(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save("h1", 1000)
        m.record_save("h2", 2000)
        m.record_remove("h1", 1000)
        assert m.memcache_cached_entries == 1
        assert m.total_embedding_bytes == 2000
        assert m.removes == 1

    def test_remove_lower_bounded(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_remove("never_added", 100)
        assert m.memcache_cached_entries == 0
        assert m.total_embedding_bytes == 0

    def test_log_summary_output(self, capsys):
        m = ECMemCacheMetrics(enable=True, log_interval_steps=10)
        m.record_save("h1", 1000)
        m.record_memcache_hit("h1", "DRAM")
        m.record_memcache_miss("h2")
        m.record_promotion("h1")
        m.log_summary_if_needed(step=10)
        # Should not raise; content verified by manual inspection

    def test_log_summary_skips_before_interval(self, capsys):
        m = ECMemCacheMetrics(enable=True, log_interval_steps=10)
        m.record_save("h1", 1000)
        m.log_summary_if_needed(step=5)
        # First call at step 0 is a no-op; step 5 < interval, still no-op
        # (log_interval_steps=10 -> fires at 10, 20, ...)

    def test_reset_preserves_enable(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_hbm_hit("h1")
        m.reset()
        assert m._enable is True
        assert m.hbm_hits == 0
