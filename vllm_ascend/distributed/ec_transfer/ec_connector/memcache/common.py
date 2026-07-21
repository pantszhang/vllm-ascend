# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared types and helpers for ECMemCacheConnector."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Prefix to avoid key collision with KV cache keys in MemCache.
_MEMCACHE_KEY_PREFIX = "ec_"


@dataclass
class ECMemCacheConnectorMetadata(ECConnectorMetadata):
    """Per-step scheduler → worker payload for ECMemCacheConnector.

    Populated by ECMemCacheScheduler.build_connector_meta();
    consumed by ECMemCacheWorker via bind_connector_metadata().

    Unlike ECCPUConnectorMetadata, this carries only {mm_hash: num_tokens}
    rather than block IDs — MemCache manages addresses internally via GVA.
    """

    saves: dict[str, int] = field(default_factory=dict)
    loads: dict[str, int] = field(default_factory=dict)


def _make_memcache_key(mm_hash: str) -> str:
    """Generate a MemCache key from a multimodal hash.

    The 'ec_' prefix prevents collisions with KV cache pool keys.
    """
    return f"{_MEMCACHE_KEY_PREFIX}{mm_hash}"


def _get_encoder_cache_hidden_dim(vllm_config: "VllmConfig") -> int:
    """Return the per-token hidden dimension for encoder cache entries.

    For most models this equals the LLM's hidden size. Qwen3-VL (and any
    future model with deepstack visual encoding) is an exception: the ViT
    concatenates its own output with features from N decoder layers before
    storing in encoder_cache, producing a wider tensor.
    """
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = (
        getattr(hf_config, "vision_config", None) if hf_config is not None else None
    )
    if vision_config is not None:
        out_hidden_size = getattr(vision_config, "out_hidden_size", None)
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", None)
        if out_hidden_size is not None and deepstack_indexes:
            return out_hidden_size * (1 + len(deepstack_indexes))
    return model_config.get_inputs_embeds_size()
