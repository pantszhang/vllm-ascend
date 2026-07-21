"""Tests for ECMemCacheMetrics."""
import pytest
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.metrics import (
    ECMemCacheMetrics,
)


class TestMetricsDisabled:
    """When enable=False, all methods are no-ops with zero overhead."""

    def test_disabled_no_attribute_init(self):
        m = ECMemCacheMetrics(enable=False)
        # No internal attributes should be initialized
        assert not hasattr(m, "hbm_hits")

    def test_disabled_record_save_does_nothing(self):
        m = ECMemCacheMetrics(enable=False)
        m.record_save("hash1", 1024, "DRAM")

    def test_disabled_log_summary_does_nothing(self):
        m = ECMemCacheMetrics(enable=False)
        m.log_summary()

    def test_disabled_record_hit_does_nothing(self):
        m = ECMemCacheMetrics(enable=False)
        m.record_memcache_hit("hash1", "DRAM")
        m.record_memcache_miss("hash1")
        m.record_hbm_hit("hash1")


class TestMetricsEnabled:
    """When enable=True, all counters accumulate correctly."""

    def test_record_save_increments_counters(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save("h1", 1024, "DRAM")
        assert m.saves == 1
        assert m.current_cached_entries == 1
        assert m.total_embedding_bytes == 1024

    def test_record_multiple_saves_tracks_entries(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save("h1", 1000, "DRAM")
        m.record_save("h2", 2000, "DRAM")
        assert m.saves == 2
        assert m.current_cached_entries == 2
        assert m.total_embedding_bytes == 3000

    def test_max_embedding_size_tracked(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save("h1", 1000, "DRAM")
        m.record_save("h2", 5000, "DRAM")
        assert m.max_embedding_size_bytes == 5000

    def test_record_memcache_hit_dram(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_memcache_hit("h1", "DRAM")
        assert m.memcache_hits == 1
        assert m.dram_loads == 1
        assert m.ssd_loads == 0
        assert m.dram_promotions == 0
        assert m.encoder_recomputes_avoided == 1

    def test_record_memcache_hit_ssd(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_memcache_hit("h1", "SSD")
        assert m.memcache_hits == 1
        assert m.ssd_loads == 1
        assert m.dram_promotions == 1

    def test_record_memcache_miss(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_memcache_miss("h1")
        assert m.memcache_misses == 1

    def test_record_hbm_hit(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_hbm_hit("h1")
        assert m.hbm_hits == 1
        assert m.encoder_recomputes_avoided == 1

    def test_record_alloc_failure(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_alloc_failure("h1")
        assert m.alloc_failures == 1

    def test_record_copy_error(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_copy_error("h1", "L2G")
        assert m.copy_errors == 1

    def test_record_save_skipped(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save_skipped("h1", "already_exists")
        assert m.save_skipped == 1

    def test_record_remove_decrements_entries(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save("h1", 1000, "DRAM")
        m.record_save("h2", 2000, "DRAM")
        m.record_remove("h1", 1000)
        assert m.current_cached_entries == 1
        assert m.total_embedding_bytes == 2000

    def test_record_remove_bounds_lower(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_remove("nonexistent", 100)
        assert m.current_cached_entries == 0
        assert m.total_embedding_bytes == 0

    def test_log_summary_runs_without_error(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save("h1", 1000, "DRAM")
        m.record_memcache_hit("h1", "DRAM")
        m.record_memcache_miss("h2")
        m.log_summary()

    def test_save_tracks_timestamp_for_age(self):
        m = ECMemCacheMetrics(enable=True)
        m.record_save("h1", 1000, "DRAM")
        assert "h1" in m._save_times
