"""Tests for ECMemCacheConnectorMetadata and helpers."""
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.common import (
    ECMemCacheConnectorMetadata,
    _make_memcache_key,
)


class TestMetadata:
    def test_default_metadata_empty(self):
        meta = ECMemCacheConnectorMetadata()
        assert meta.saves == {}
        assert meta.loads == {}

    def test_metadata_saves_populated(self):
        meta = ECMemCacheConnectorMetadata()
        meta.saves = {"hash1": 100}
        meta.loads = {"hash2": 200}
        assert meta.saves == {"hash1": 100}
        assert meta.loads == {"hash2": 200}

    def test_metadata_saves_loads_independent(self):
        meta = ECMemCacheConnectorMetadata()
        meta.saves = {"a": 1}
        meta.loads = {"b": 2}
        assert "a" in meta.saves
        assert "b" in meta.loads
        assert "a" not in meta.loads


class TestMakeMemcacheKey:
    def test_make_key_prefix(self):
        key = _make_memcache_key("abc123")
        assert key == "ec_abc123"

    def test_make_key_empty_hash(self):
        key = _make_memcache_key("")
        assert key == "ec_"

    def test_make_key_special_chars(self):
        key = _make_memcache_key("hash/with:special")
        assert key.startswith("ec_")
        assert len(key) > 3
