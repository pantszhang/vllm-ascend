"""Tests for ECMemCacheWorker."""
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.worker import (
    ECMemCacheWorker,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.common import (
    ECMemCacheConnectorMetadata,
)


class FakeKeyInfo:
    """Fake KeyInfo returned by MemCache batch_get_key_info."""
    def __init__(self, size, gva_list, type_list=None):
        self._size = size
        self._gva_list = gva_list or []
        self._type_list = type_list or [1]  # MEDIA_DRAM

    def size(self):
        return self._size

    def gva_list(self):
        return self._gva_list

    def type_list(self):
        return self._type_list


class FakeECConfig:
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


@pytest.fixture
def mock_store():
    with patch(
        "vllm_ascend.distributed.ec_transfer.ec_connector.memcache.worker.MemcacheBackend"
    ) as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def worker(mock_store):
    with patch(
        "vllm_ascend.distributed.ec_transfer.ec_connector.memcache.worker.current_platform"
    ) as mock_plat:
        mock_plat.Stream.return_value = MagicMock()
        mock_plat.device_type = "npu"
        mock_plat.current_stream.return_value = MagicMock()
        mock_plat.stream.return_value = MagicMock()

        vllm_config = MagicMock()
        type(vllm_config).ec_transfer_config = PropertyMock(
            return_value=FakeECConfig()
        )
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_config = None
        vllm_config.model_config.get_inputs_embeds_size.return_value = 4096

        return ECMemCacheWorker(vllm_config)


class TestSaveCaches:
    def test_not_producer_skips(self, worker, mock_store):
        worker._is_producer = False
        worker.save_caches({"h1": torch.tensor([1.0])}, "h1", MagicMock())
        mock_store.batch_alloc.assert_not_called()

    def test_hash_not_in_saves_skips(self, worker, mock_store):
        meta = ECMemCacheConnectorMetadata()
        meta.saves = {}
        worker.save_caches({}, "h1", meta)
        mock_store.batch_alloc.assert_not_called()

    def test_save_calls_batch_alloc_and_copy(self, worker, mock_store):
        mock_store.batch_alloc.return_value = [12345]
        mock_store.batch_copy.return_value = 0
        tensor = torch.zeros(10, 4096, dtype=torch.float16, device="cpu")
        meta = ECMemCacheConnectorMetadata()
        meta.saves = {"h1": 10}

        worker.save_caches({"h1": tensor}, "h1", meta)

        mock_store.batch_alloc.assert_called_once()
        mock_store.batch_copy.assert_called_once()

    def test_save_alloc_failure_handled(self, worker, mock_store):
        mock_store.batch_alloc.return_value = [0]  # alloc failed
        tensor = torch.zeros(10, 4096, dtype=torch.float16, device="cpu")
        meta = ECMemCacheConnectorMetadata()
        meta.saves = {"h1": 10}

        worker.save_caches({"h1": tensor}, "h1", meta)
        mock_store.batch_copy.assert_not_called()


class TestStartLoadCaches:
    def test_not_consumer_skips(self, worker, mock_store):
        worker._is_consumer = False
        worker.start_load_caches({}, MagicMock())
        mock_store.batch_get_key_info.assert_not_called()

    def test_empty_loads_skips(self, worker, mock_store):
        meta = ECMemCacheConnectorMetadata()
        meta.loads = {}
        worker.start_load_caches({}, meta)
        mock_store.batch_get_key_info.assert_not_called()

    def test_already_in_encoder_cache_hits_hbm(self, worker, mock_store):
        worker._is_consumer = True
        tensor = torch.zeros(10, 4096, dtype=torch.float16, device="cpu")
        meta = ECMemCacheConnectorMetadata()
        meta.loads = {"h1": 10}

        worker.start_load_caches({"h1": tensor}, meta)
        mock_store.batch_get_key_info.assert_not_called()
