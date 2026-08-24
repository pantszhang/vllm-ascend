# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared probe-fallback helper for GDN CANN-operator integration.

Each replacement point probes a ``torch_npu`` interface with a minimal smoke
call and falls back to the mainline implementation when unavailable. The
``auto``/``on``/``off`` environment variable lets us force either side for
same-environment A/B comparison (see the design spec, section 6).
"""

from __future__ import annotations

import os
from typing import Callable

import torch_npu


class ProbeResult:
    """Probe outcome. ``fn`` is the selected torch_npu interface (None if unavailable)."""

    __slots__ = ("available", "fn")

    def __init__(self, available: bool, fn: Callable | None = None):
        self.available = available
        self.fn = fn


_cache: dict[str, ProbeResult] = {}


def probe_cann_interface(
    env_var: str,
    candidate_names: tuple[str, ...],
    smoke_test: Callable[[Callable], None],
) -> ProbeResult:
    """Probe candidate ``torch_npu`` interfaces in order, cached per env_var.

    Raises ``ValueError`` for unknown modes and ``RuntimeError`` when ``on`` is
    forced but every candidate fails the smoke test (fail fast, never silently
    fall back while the operator is explicitly required).
    """
    if env_var in _cache:
        return _cache[env_var]

    mode = os.environ.get(env_var, "auto").strip().lower()
    if mode == "off":
        res = ProbeResult(False)
        _cache[env_var] = res
        return res
    if mode not in ("auto", "on"):
        raise ValueError(f"{env_var} must be auto/on/off, got: {mode!r}")

    selected: Callable | None = None
    for name in candidate_names:
        fn = getattr(torch_npu, name, None)
        if fn is None:
            continue
        try:
            smoke_test(fn)
            selected = fn
            break
        except Exception:
            continue

    if mode == "on" and selected is None:
        raise RuntimeError(
            f"{env_var}=on but none of {candidate_names} passed the smoke test; "
            "the installed CANN package may not provide the implementation yet."
        )
    res = ProbeResult(selected is not None, selected)
    _cache[env_var] = res
    return res
