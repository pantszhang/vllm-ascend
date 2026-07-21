# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECMemCacheConnector — MemCache-backed EC connector.

A role-routed shell: one instance per process. The scheduler delegate
owns MemCache metadata tracking and LRU eviction; the worker delegate
owns GVA-based NPU↔MemCache transfers.
"""

from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.common import (
    ECMemCacheConnectorMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECMemCacheConnector(ECConnectorBase):
    """EC connector that offloads encoder cache to MemCache.

    Producer role: after the model encoder runs, save_caches() copies
    encoder outputs from NPU HBM → MemCache via GVA direct transfer.

    Consumer role: before the model forward pass, start_load_caches()
    copies cached embeddings from MemCache → NPU HBM via GVA.

    ec_both role: both paths run in sequence.
    """

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        super().__init__(vllm_config=vllm_config, role=role)

        self.connector_worker = None
        self.connector_scheduler = None

        if role == ECConnectorRole.WORKER:
            self.connector_worker = self._make_worker(vllm_config)
        elif role == ECConnectorRole.SCHEDULER:
            self.connector_scheduler = self._make_scheduler(vllm_config)
        else:
            raise ValueError(f"Unknown ECConnectorRole: {role}")

    # ── Construction seams ──

    def _make_worker(self, vllm_config: "VllmConfig"):
        from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.worker import (
            ECMemCacheWorker,
        )

        return ECMemCacheWorker(vllm_config)

    def _make_scheduler(self, vllm_config: "VllmConfig"):
        from vllm_ascend.distributed.ec_transfer.ec_connector.memcache.scheduler import (
            ECMemCacheScheduler,
        )

        return ECMemCacheScheduler(vllm_config)

    # ── Worker-side forwarders ──

    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMemCacheConnectorMetadata)
        self.connector_worker.start_load_caches(encoder_cache, metadata)

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECMemCacheConnectorMetadata)
        self.connector_worker.save_caches(encoder_cache, mm_hash, metadata)

    # ── Scheduler-side forwarders ──

    def has_cache_item(self, identifier: str) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_cache_item(identifier)

    def ensure_cache_available(
        self, request: "Request", num_computed_tokens: int
    ) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.ensure_cache_available(
            request, num_computed_tokens
        )

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_state_after_alloc(request, index)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> ECMemCacheConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    # ── Shared ──

    def shutdown(self) -> None:
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
