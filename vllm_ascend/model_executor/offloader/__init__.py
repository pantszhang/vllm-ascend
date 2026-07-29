"""Ascend-specific model parameter offloading."""

from vllm_ascend.model_executor.offloader.base import create_offloader
from vllm_ascend.model_executor.offloader.prefetch import NPUPrefetchOffloader

__all__ = [
    "NPUPrefetchOffloader",
    "create_offloader",
]
