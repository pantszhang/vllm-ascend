"""Tests for ECMemCacheScheduler."""
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.scheduler import (
    ECMemCacheScheduler,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.common import (
    ECMemCacheConnectorMetadata,
)


class FakeECConfig:
    """Minimal fake of ECTransferConfig for testing."""
    def __init__(self, is_producer=True, is_consumer=True, extra_config=None):
        self._is_producer = is_producer
        self._is_consumer = is_consumer
        self._extra = extra_config or {}

    @property
    def is_ec_producer(self):
        return self._is_producer

    @property
    def is_ec_consumer(self):
        return self._is_consumer

    def get_from_extra_config(self, key, default):
        return self._extra.get(key, default)


class FakeVllmConfig:
    """Minimal fake of VllmConfig for scheduler testing."""
    def __init__(self, extra_config=None):
        self.ec_transfer_config = FakeECConfig(extra_config=extra_config)
        self.max_concurrent_batches = 4


class FakeRequest:
    """Minimal fake of Request for scheduler testing."""
    def __init__(self, request_id, mm_hash, num_encoder_embeds):
        self.request_id = request_id
        self._mm_hash = mm_hash
        self._num_encoder_embeds = num_encoder_embeds

    @property
    def mm_features(self):
        return [FakeMMFeature(self._mm_hash)]

    def get_num_encoder_embeds(self, index):
        return self._num_encoder_embeds


class FakeMMFeature:
    def __init__(self, identifier):
        self.identifier = identifier


class FakeSchedulerOutput:
    def __init__(self, finished_req_ids=None):
        self.finished_req_ids = finished_req_ids or set()


@pytest.fixture
def mock_store():
    """Mock MemcacheBackend for scheduler testing."""
    with patch(
        "vllm_ascend.distributed.ec_transfer.ec_connector.memcache.scheduler.MemcacheBackend"
    ) as mock_cls:
        mock_instance = MagicMock()
        mock_cls.create_scheduler_client.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def scheduler(mock_store):
    """Create an ECMemCacheScheduler with mocked store."""
    cfg = FakeVllmConfig(extra_config={"ec_memcache_max_cached_entries": 3})
    return ECMemCacheScheduler(cfg)


class TestHasCacheItem:
    def test_unknown_hash_returns_false(self, scheduler):
        assert scheduler.has_cache_item("unknown") is False

    def test_known_hash_returns_true(self, scheduler):
        scheduler._cache_entries["hash_a"] = 100
        assert scheduler.has_cache_item("hash_a") is True

    def test_lru_touch_on_hit(self, scheduler):
        scheduler._cache_entries["hash_a"] = 1
        scheduler._cache_entries["hash_b"] = 2
        scheduler._cache_entries["hash_c"] = 3
        assert next(iter(scheduler._cache_entries)) == "hash_a"
        scheduler.has_cache_item("hash_a")
        keys = list(scheduler._cache_entries.keys())
        assert keys[-1] == "hash_a"


class TestUpdateStateAfterAlloc:
    def test_producer_new_hash_adds_to_pending_saves(self, scheduler):
        req = FakeRequest("req_1", "hash_new", 50)
        scheduler.update_state_after_alloc(req, 0)
        assert "hash_new" in scheduler._pending_saves
        assert scheduler._pending_saves["hash_new"] == 50

    def test_producer_existing_hash_not_duplicated(self, scheduler):
        scheduler._cache_entries["hash_old"] = 50
        req = FakeRequest("req_1", "hash_old", 50)
        scheduler.update_state_after_alloc(req, 0)
        assert "hash_old" not in scheduler._pending_saves

    def test_consumer_cached_hash_adds_to_pending_loads(self, scheduler):
        scheduler._cache_entries["hash_cached"] = 100
        req = FakeRequest("req_2", "hash_cached", 100)
        scheduler.update_state_after_alloc(req, 0)
        assert "hash_cached" in scheduler._pending_loads
        assert scheduler._pending_loads["hash_cached"] == 100

    def test_consumer_uncached_hash_no_load(self, scheduler):
        req = FakeRequest("req_3", "hash_uncached", 100)
        scheduler.update_state_after_alloc(req, 0)
        assert "hash_uncached" not in scheduler._pending_loads


class TestBuildConnectorMeta:
    def test_returns_metadata_with_saves_and_loads(self, scheduler):
        scheduler._pending_saves = {"h1": 1}
        scheduler._pending_loads = {"h2": 2}
        meta = scheduler.build_connector_meta(FakeSchedulerOutput())
        assert isinstance(meta, ECMemCacheConnectorMetadata)
        assert meta.saves == {"h1": 1}
        assert meta.loads == {"h2": 2}

    def test_resets_pending_after_build(self, scheduler):
        scheduler._pending_saves = {"h1": 1}
        scheduler._pending_loads = {"h2": 2}
        scheduler.build_connector_meta(FakeSchedulerOutput())
        assert scheduler._pending_saves == {}
        assert scheduler._pending_loads == {}


class TestLRUEviction:
    def test_eviction_triggered_when_over_max(self, scheduler, mock_store):
        scheduler._cache_entries["h1"] = 100
        scheduler._cache_entries["h2"] = 100
        scheduler._cache_entries["h3"] = 100
        scheduler._pending_saves["h4"] = 50
        scheduler._ready_tracker.add("h4", "req_4")

        so = FakeSchedulerOutput(finished_req_ids={"req_4"})
        scheduler.build_connector_meta(so)

        assert "h4" in scheduler._cache_entries
        assert len(scheduler._cache_entries) <= 3
