# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ScoreEncoderCacheManager MemCache integration."""
from collections import OrderedDict
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from vllm_ascend.ec_manager.score_ec_manager import (
    CacheEntry,
    ScoreEncoderCacheManager,
)


class FakeScoreCacheConfig:
    enabled = True
    use_memcache = True
    cpu_cache_slots = 1000
    max_clock = 15
    clock_decay_every = 64
    watermark = 0.2
    promote_percentile = 0.2
    memcache_meta_url = "tcp://127.0.0.1:5000"
    memcache_config_store_url = "tcp://127.0.0.1:6000"
    memcache_protocol = "host_shm"
    memcache_dram_size = "4GB"
    memcache_ssd_size = "0GB"
    memcache_ssd_path = ""
    memcache_evict_high = 80
    memcache_evict_low = 70
    memcache_metrics_enable = False
    memcache_metrics_interval = 300


class FakeRequest:
    def __init__(self, request_id, mm_hash, num_embeds):
        self.request_id = request_id
        self._mm_hash = mm_hash
        self._num_embeds = num_embeds
        self.mm_features = [FakeMMFeature(mm_hash)]

    def get_num_encoder_embeds(self, input_id):
        return self._num_embeds


class FakeMMFeature:
    def __init__(self, identifier):
        self.identifier = identifier


class FakeVllmConfig:
    def __init__(self):
        self.model_config = MagicMock()
        self.model_config.hf_config = MagicMock()
        self.model_config.hf_config.vision_config = MagicMock()
        self.model_config.hf_config.vision_config.num_heads = 16
        self.model_config.hf_config.vision_config.hidden_size = 1024
        self.model_config.hf_config.vision_config.intermediate_size = 4096
        self.parallel_config = MagicMock()


@pytest.fixture
def fake_store():
    with patch(
        "vllm_ascend.ec_manager.score_ec_manager.MemcacheBackend"
    ) as mock_cls:
        instance = MagicMock()
        instance.remove.return_value = 0
        instance.batch_is_exist.return_value = [1]
        mock_cls.create_scheduler_client.return_value = instance
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def score_mgr(fake_store):
    with patch(
        "vllm_ascend.ec_manager.score_ec_manager.get_score_encoder_cache_config",
        return_value=FakeScoreCacheConfig,
    ):
        return ScoreEncoderCacheManager(
            cache_size=100, vllm_config=FakeVllmConfig()
        )


class TestMemcacheEviction:
    def test_can_allocate_calls_remove_on_eviction(self, score_mgr, fake_store):
        score_mgr.cpu_cache["h1"] = CacheEntry("h1", 1, 10, 50, 100.0)
        score_mgr.cpu_cache["h2"] = CacheEntry("h2", 1, 10, 50, 100.0)
        score_mgr.cpu_freeable["h1"] = score_mgr.cpu_cache["h1"]
        score_mgr.cpu_freeable["h2"] = score_mgr.cpu_cache["h2"]
        score_mgr.cpu_num_freeable_slots = 100
        score_mgr.cpu_num_free_slots = 0

        req = FakeRequest("r_new", "h_new", 60)
        score_mgr.can_allocate(req, 0, encoder_compute_budget=1000, num_embeds_to_schedule=0)

        fake_store.remove.assert_called()
        assert "h1" not in score_mgr.cpu_cache
        assert "h1" not in score_mgr.cpu_freeable

    def test_remove_failure_is_graceful(self, score_mgr, fake_store):
        fake_store.remove.side_effect = Exception("connection lost")
        score_mgr.cpu_cache["h1"] = CacheEntry("h1", 1, 10, 10, 100.0)
        score_mgr.cpu_freeable["h1"] = score_mgr.cpu_cache["h1"]
        score_mgr.cpu_num_freeable_slots = 10
        score_mgr.cpu_num_free_slots = 0

        req = FakeRequest("r_new", "h_new", 10)
        score_mgr.can_allocate(req, 0, encoder_compute_budget=1000, num_embeds_to_schedule=0)
        # Still freed from metadata even if remove fails
        assert "h1" not in score_mgr.cpu_cache
        assert score_mgr.cpu_num_free_slots >= 10

    def test_non_memcache_mode_skips_remove(self, score_mgr, fake_store):
        score_mgr._use_memcache = False
        score_mgr.cpu_cache["h1"] = CacheEntry("h1", 1, 10, 10, 100.0)
        score_mgr.cpu_freeable["h1"] = score_mgr.cpu_cache["h1"]
        score_mgr.cpu_num_freeable_slots = 10
        score_mgr.cpu_num_free_slots = 0

        req = FakeRequest("r_new", "h_new", 10)
        score_mgr.can_allocate(req, 0, encoder_compute_budget=1000, num_embeds_to_schedule=0)
        fake_store.remove.assert_not_called()
