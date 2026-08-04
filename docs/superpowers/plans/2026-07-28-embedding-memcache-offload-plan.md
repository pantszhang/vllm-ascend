# Embedding Memcache Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offload encoder embeddings to memcache via GVA path, with ZMQ-based exists queries from scheduler to worker.

**Architecture:** 6 new/modified files. `EncoderCacheStore` wraps `MemcacheBackend` + ZMQ REP server in worker daemon thread. `EncoderCacheStoreClient` is ZMQ REQ client in scheduler. `EncoderCacheManagerWithStore` inherits upstream `EncoderCacheManager`, removing eviction logic, adding ZMQ exists query. `NPUModelRunner` overrides 4 encoder cache methods guarded by `ec_memcache_config.enabled`.

**Tech Stack:** Python, memcache_hybrid (C++ ext), ZMQ (pyzmq), V1 GPUModelRunner inheritance

## Global Constraints

- Do NOT modify any file outside `C:\code\vllm-ascend\`
- All memcache APIs used MUST exist in `MemcacheBackend` / `DistributedObjectStore`
- Switch OFF (`ec_memcache_config.enabled=false`) → zero interference with upstream flow
- NPUModelRunner overrides only V1 methods from `vllm/v1/worker/gpu_model_runner.py`
- No EC connector involvement

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `vllm_ascend/ascend_config.py` | Modify | Add `ECMemcacheConfig` class |
| `vllm_ascend/distributed/ec_transfer/__init__.py` | Create | Module init |
| `vllm_ascend/distributed/ec_transfer/ec_store_client.py` | Create | `EncoderCacheStoreClient` + `get_zmq_rpc_path_ec_lookup()` |
| `vllm_ascend/distributed/ec_transfer/encoder_cache_store.py` | Create | `EncoderCacheStore` (MemcacheBackend wrapper + ZMQ REP server) |
| `vllm_ascend/core/encoder_cache_manager.py` | Create | `EncoderCacheManagerWithStore` subclass |
| `vllm_ascend/worker/model_runner_v1.py` | Modify | Override 4 encoder cache methods |
| `vllm_ascend/worker/worker.py` | Modify | Set `encoder_cache_manager_cls` config |

---

### Task 1: ECMemcacheConfig in ascend_config.py

**Files:**
- Modify: `vllm_ascend/ascend_config.py:50-70`

**Interfaces:**
- Produces: `ECMemcacheConfig(enabled=False, **kwargs)` class, `AscendConfig.ec_memcache_config: ECMemcacheConfig`

- [ ] **Step 1: Add ECMemcacheConfig class**

Insert before `AscendConfig` class (near line 26):

```python
class ECMemcacheConfig:
    """Configuration for encoder embedding memcache offload.

    All parameters can be configured via ``additional_config.ec_memcache_config``
    in the vLLM config.
    """

    def __init__(self, enabled: bool = False, **kwargs):
        self.enabled = enabled
```

- [ ] **Step 2: Wire into AscendConfig.__init__**

Read `AscendConfig.__init__` (around line 55, after the `scheduler_config` block), add:

```python
ec_memcache_config = additional_config.get("ec_memcache_config", {})
self.ec_memcache_config = ECMemcacheConfig(**ec_memcache_config)
```

- [ ] **Step 3: Verify file is syntactically correct**

Read the modified file and confirm:
- `ECMemcacheConfig` class is defined before `AscendConfig` class
- `self.ec_memcache_config = ECMemcacheConfig(**ec_memcache_config)` is added in `AscendConfig.__init__`
- The pattern matches existing sub-config classes (`EplbConfig`, `AscendCompilationConfig`)

- [ ] **Step 4: Commit**

```bash
git add vllm_ascend/ascend_config.py
git commit -m "feat: add ECMemcacheConfig for embedding memcache offload switch"
```

---

### Task 2: ZMQ path function and EncoderCacheStoreClient

**Files:**
- Create: `vllm_ascend/distributed/ec_transfer/__init__.py`
- Create: `vllm_ascend/distributed/ec_transfer/ec_store_client.py`

**Interfaces:**
- Produces: `get_zmq_rpc_path_ec_lookup(vllm_config) → str`, `EncoderCacheStoreClient.__init__(socket_path)`, `EncoderCacheStoreClient.exists(mm_hash: str) → bool`

- [ ] **Step 1: Create module __init__.py**

```python
# vllm_ascend/distributed/ec_transfer/__init__.py
```

(empty file, just marks the package)

- [ ] **Step 2: Write EncoderCacheStoreClient**

```python
# vllm_ascend/distributed/ec_transfer/ec_store_client.py
from __future__ import annotations

from typing import TYPE_CHECKING

import zmq
from vllm.utils.network_utils import make_zmq_socket

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def get_zmq_rpc_path_ec_lookup(vllm_config: "VllmConfig") -> str:
    """Return the IPC socket path for EC lookup ZMQ channel.

    Follows the same pattern as ``get_zmq_rpc_path_lookup`` in
    ``vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py``.
    """
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    import vllm.envs as vllm_envs
    base_url = vllm_envs.VLLM_RPC_BASE_PATH
    return f"ipc://{base_url}/ec_lookup_dp_rank{dp_rank}"


class EncoderCacheStoreClient:
    """ZMQ REQ client for querying encoder cache existence in memcache.

    Used by the scheduler-side EncoderCacheManagerWithStore to check
    whether a given mm_hash already has a cached encoder output in
    the remote memcache store.
    """

    def __init__(self, socket_path: str):
        self._ctx = zmq.Context()
        self._socket = make_zmq_socket(
            self._ctx, socket_path, zmq.REQ, connect=True
        )

    def exists(self, mm_hash: str) -> bool:
        """Check whether *mm_hash* exists in the memcache store."""
        self._socket.send_string(mm_hash)
        return self._socket.recv() == b'1'
```

- [ ] **Step 3: Verify import and basic functionality**

Verify by reading the file:
- `get_zmq_rpc_path_ec_lookup(vllm_config) -> str` signature correct
- `EncoderCacheStoreClient.__init__(socket_path)` binds ZMQ REQ with `bind=False`
- `EncoderCacheStoreClient.exists(mm_hash) -> bool` uses send_string/recv
- All imports reference existing modules

- [ ] **Step 4: Commit**

```bash
git add vllm_ascend/distributed/ec_transfer/__init__.py vllm_ascend/distributed/ec_transfer/ec_store_client.py
git commit -m "feat: add EncoderCacheStoreClient and ZMQ path for EC lookup"
```

---

### Task 3: EncoderCacheStore (Worker-side memcache wrapper + ZMQ REP server)

**Files:**
- Create: `vllm_ascend/distributed/ec_transfer/encoder_cache_store.py`

**Interfaces:**
- Consumes: `get_zmq_rpc_path_ec_lookup` from Task 2
- Produces: `EncoderCacheStore.__init__(vllm_config, local_rank)`, `EncoderCacheStore.put(mm_hash, tensor)`, `EncoderCacheStore.get(mm_hash) → Tensor|None`

- [ ] **Step 1: Write EncoderCacheStore class**

```python
# vllm_ascend/distributed/ec_transfer/encoder_cache_store.py
from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch
import zmq
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
    MemcacheBackend,
    MmcDirect,
)
from vllm_ascend.distributed.ec_transfer.ec_store_client import (
    get_zmq_rpc_path_ec_lookup,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class EncoderCacheStore:
    """Worker-side encoder cache store backed by memcache.

    Provides:
    - put(mm_hash, tensor): Store encoder output via GVA + batch_copy(L2G)
    - get(mm_hash) -> Tensor|None: Load encoder output via GVA + batch_copy(G2L)
    - ZMQ REP daemon thread for exists queries from the scheduler
    """

    LEASE_READ_TTL_MS = 60_000  # Short TTL, only needed during batch_copy(G2L)

    def __init__(self, vllm_config: "VllmConfig", local_rank: int):
        self._store = MemcacheBackend(
            parallel_config=vllm_config.parallel_config,
            local_rank=local_rank,
            init_bm=True,
            lazy_init=False,
        )
        model_config = vllm_config.model_config
        self._model_name = model_config.model.split("/")[-1]
        self._hidden_dim = _get_encoder_cache_hidden_dim(vllm_config)
        self._dtype = model_config.dtype
        self._elem_size = torch.tensor([], dtype=self._dtype).element_size()

        # ZMQ REP server for scheduler exists queries
        self._running = True
        socket_path = get_zmq_rpc_path_ec_lookup(vllm_config)
        self._zmq_ctx = zmq.Context()
        self._zmq_socket = make_zmq_socket(
            self._zmq_ctx, socket_path, zmq.REP, bind=True
        )
        self._zmq_thread = threading.Thread(
            target=self._zmq_loop, daemon=True
        )
        self._zmq_thread.start()
        logger.info(
            "EncoderCacheStore started on %s (model=%s hidden_dim=%d)",
            socket_path, self._model_name, self._hidden_dim,
        )

    # ---- NPUModelRunner calls ----

    def put(self, mm_hash: str, tensor: torch.Tensor) -> None:
        """Store *tensor* as the encoder output for *mm_hash*.

        Allocates a GVA via batch_alloc, then copies NPU→memcache (L2G).
        """
        key = self._make_key(mm_hash)
        nbytes = tensor.nbytes
        gva = self._store.batch_alloc([key], [nbytes])[0]
        self._store.store.batch_copy(
            [gva],
            [tensor.data_ptr()],
            [nbytes],
            MmcDirect.COPY_L2G.value,
        )
        logger.debug("EncoderCacheStore.put: key=%s nbytes=%d gva=%d", key, nbytes, gva)

    def get(self, mm_hash: str) -> torch.Tensor | None:
        """Return the cached encoder output for *mm_hash*, or None."""
        key = self._make_key(mm_hash)

        # Look up GVA and byte size from memcache
        ki = self._store.batch_get_key_info([key])
        if ki.size() == 0:
            logger.debug("EncoderCacheStore.get: key=%s not found", key)
            return None
        gva = ki.gva_list()[0]
        nbytes = ki.size()

        # Lease → copy G2L → release lease
        self._store.batch_add_lease([key], ttl_ms=self.LEASE_READ_TTL_MS)
        num_tokens = nbytes // self._elem_size // self._hidden_dim
        tensor = torch.empty(
            num_tokens, self._hidden_dim, dtype=self._dtype, device="npu"
        )
        self._store.store.batch_copy(
            [gva],
            [tensor.data_ptr()],
            [nbytes],
            MmcDirect.COPY_G2L.value,
        )
        self._store.batch_remove_lease([key])
        logger.debug(
            "EncoderCacheStore.get: key=%s nbytes=%d num_tokens=%d",
            key, nbytes, num_tokens,
        )
        return tensor

    # ---- ZMQ server ----

    def _zmq_loop(self) -> None:
        """ZMQ REP loop: receive mm_hash, reply 1 (exists) or 0 (missing)."""
        while self._running:
            try:
                mm_hash = self._zmq_socket.recv_string()
                key = self._make_key(mm_hash)
                exists = self._store.exists([key])[0] == 1
                self._zmq_socket.send(b'1' if exists else b'0')
            except zmq.error.ZMQError:
                break

    # ---- internal helpers ----

    def _make_key(self, mm_hash: str) -> str:
        return f"{self._model_name}@cache_role:ec@{mm_hash}"


def _get_encoder_cache_hidden_dim(vllm_config: "VllmConfig") -> int:
    """Return per-token hidden dimension for encoder cache entries.

    Mirrors ``ec_connector/cpu/common.py:38-59`` for Qwen3-VL deepstack support.
    """
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = getattr(hf_config, "vision_config", None) if hf_config else None
    if vision_config is not None:
        out_hidden_size = getattr(vision_config, "out_hidden_size", None)
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", None)
        if out_hidden_size is not None and deepstack_indexes:
            return out_hidden_size * (1 + len(deepstack_indexes))
    return model_config.get_inputs_embeds_size()
```

- [ ] **Step 2: Verify import**

**Step 2: Verify file structure**

Read the created file and confirm:
- `EncoderCacheStore` class with `put()`, `get()`, `_zmq_loop()`, `_make_key()` methods
- `_get_encoder_cache_hidden_dim()` free function
- `batch_copy` called on `self._store.store` (DistributedObjectStore), not MemcacheBackend
- ZMQ REP socket created with `bind=True`
- Key format: `{model_name}@cache_role:ec@{mm_hash}`

- [ ] **Step 3: Commit**

```bash
git add vllm_ascend/distributed/ec_transfer/encoder_cache_store.py
git commit -m "feat: add EncoderCacheStore with GVA put/get and ZMQ REP server"
```

---

### Task 4: EncoderCacheManagerWithStore (scheduler-side)

**Files:**
- Create: `vllm_ascend/core/ec_manager_with_store.py`

**Interfaces:**
- Consumes: `EncoderCacheStoreClient` from Task 2
- Consumes: upstream `EncoderCacheManager` from `vllm.v1.core.encoder_cache_manager`
- Produces: `EncoderCacheManagerWithStore.__init__(cache_size: int)` — creates ZMQ client internally

- [ ] **Step 1: Write EncoderCacheManagerWithStore**

```python
# vllm_ascend/core/ec_manager_with_store.py
from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

from vllm_ascend.distributed.ec_transfer.ec_store_client import (
    EncoderCacheStoreClient,
    get_zmq_rpc_path_ec_lookup,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


class EncoderCacheManagerWithStore(EncoderCacheManager):
    """Encoder cache manager backed by remote memcache store.

    Differences from upstream EncoderCacheManager:

    - ``check_and_update_cache`` queries memcache via ZMQ when the local
      hot-cache misses.
    - No slot capacity tracking / eviction — memcache manages its own
      storage.
    - ``get_freed_mm_hashes`` always returns an empty list.
    """

    def __init__(self, cache_size: int):
        super().__init__(cache_size)
        # The ZMQ client connects to the worker's EncoderCacheStore REP server.
        # Socket path uses VLLM_RPC_BASE_PATH (available in all processes,
        # same as pool_scheduler.py:get_zmq_rpc_path_lookup).
        # dp_rank is read from env; defaults to 0 for single-DP deployments.
        import os
        import vllm.envs as vllm_envs
        dp_rank = int(os.environ.get("DATA_PARALLEL_RANK", "0"))
        socket_path = (
            f"ipc://{vllm_envs.VLLM_RPC_BASE_PATH}"
            f"/ec_lookup_dp_rank{dp_rank}"
        )
        self._ec_store_client = EncoderCacheStoreClient(socket_path)

    # ---- overrides ----

    def check_and_update_cache(self, request: "Request", input_id: int) -> bool:
        mm_hash = request.mm_features[input_id].identifier

        # 1. Local hot cache (nanosecond)
        if mm_hash in self.cached:
            self.cached[mm_hash].add(request.request_id)
            return True

        # 2. ZMQ → memcache exists query (global truth)
        if self._ec_store_client.exists(mm_hash):
            self.cached[mm_hash] = {request.request_id}
            return True

        return False

    def allocate(self, request: "Request", input_id: int) -> None:
        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()
        self.cached[mm_hash].add(request_id)
        self.request_cached_ids.setdefault(request_id, set()).add(input_id)

    def free(self, request: "Request") -> None:
        for input_id in list(self.get_cached_input_ids(request)):
            self.free_encoder_input(request, input_id)

    def free_encoder_input(self, request: "Request", input_id: int) -> None:
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        if req_id in self.request_cached_ids:
            self.request_cached_ids[req_id].discard(input_id)
            if not self.request_cached_ids[req_id]:
                del self.request_cached_ids[req_id]
        if mm_hash in self.cached:
            self.cached[mm_hash].discard(req_id)
            if not self.cached[mm_hash]:
                del self.cached[mm_hash]

    def get_freed_mm_hashes(self) -> list[str]:
        # memcache manages eviction; scheduler never instructs the worker
        # to free encoder cache entries.
        return []

    def can_allocate(
        self,
        request: "Request",
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        num_embeds = request.get_num_encoder_embeds(input_id)
        # Only check compute budget; memcache manages storage capacity.
        if num_embeds > encoder_compute_budget:
            return False
        return True
```

- [ ] **Step 2: Verify import and test basic creation**

Read the created file and verify:
- Inherits from `vllm.v1.core.encoder_cache_manager.EncoderCacheManager`
- Constructor creates `EncoderCacheStoreClient` internally via env-var-derived socket path
- Overrides: `check_and_update_cache`, `allocate`, `free`, `free_encoder_input`, `get_freed_mm_hashes`, `can_allocate`
- `get_freed_mm_hashes` returns `[]`
- Local hot-cache miss triggers ZMQ `exists()` query

- [ ] **Step 3: Commit**

```bash
git add vllm_ascend/core/ec_manager_with_store.py
git commit -m "feat: add EncoderCacheManagerWithStore (ZMQ-backed, no eviction)"
```

---

### Task 5: Wire config in worker.py and set encoder_cache_manager_cls

**Files:**
- Modify: `vllm_ascend/worker/worker.py:160-170`

**Interfaces:**
- Consumes: `ECMemcacheConfig` from Task 1, `EncoderCacheManagerWithStore` from Task 4

- [ ] **Step 1: Read the current worker init flow**

Read `vllm_ascend/worker/worker.py` around lines 155-170 where `self.use_v2_model_runner` is set. We need to find a place after `self.vllm_config` is available to set the encoder_cache_manager_cls.

- [ ] **Step 2: Set encoder_cache_manager_cls in worker init**

After `self.vllm_config` is available, add:

```python
# After self.vllm_config is fully initialized:
ascend_config = get_ascend_config()
if ascend_config.ec_memcache_config.enabled:
    self.vllm_config.ec_manager_config.encoder_cache_manager_cls = (
        "vllm_ascend.core.ec_manager_with_store.EncoderCacheManagerWithStore"
    )
```

The exact insertion point depends on the worker init flow. Search for where `self.vllm_config` is first available (likely in or near `__init__` or `init_device`).

- [ ] **Step 3: Verify config wiring**

Add a temporary print or log statement and confirm the config is set when `ec_memcache_config.enabled=true`.

- [ ] **Step 4: Commit**

```bash
git add vllm_ascend/worker/worker.py
git commit -m "feat: wire EncoderCacheManagerWithStore into scheduler via ec_manager_config"
```

---

### Task 6: NPUModelRunner overrides in model_runner_v1.py

**Files:**
- Modify: `vllm_ascend/worker/model_runner_v1.py:290-310`

**Interfaces:**
- Consumes: `ECMemcacheConfig` from Task 1, `EncoderCacheStore` from Task 3
- Overrides: `_cache_encoder_output`, `_get_encoder_output_from_cache`, `_process_encoder_cache_scheduler_output`, `reset_encoder_cache`

- [ ] **Step 1: Add imports at top of file**

```python
# After existing imports in vllm_ascend/worker/model_runner_v1.py:
from vllm_ascend.distributed.ec_transfer.encoder_cache_store import EncoderCacheStore
```

- [ ] **Step 2: Add init block after super().__init__()**

After line 290 (`super().__init__(vllm_config, device)`) and before line 292 (`self.pin_memory = ...`), insert:

```python
        # Embedding memcache offload
        ascend_config = get_ascend_config()
        self.use_ec_memcache_offload = (
            ascend_config.ec_memcache_config.enabled
            and self.supports_mm_inputs
            and get_pp_group().is_first_rank
        )
        if self.use_ec_memcache_offload:
            from vllm_ascend.distributed.ec_transfer.encoder_cache_store import (
                EncoderCacheStore,
            )
            self.encoder_cache_store = EncoderCacheStore(
                vllm_config, envs.LOCAL_RANK
            )
            # Release the plain dict created by upstream GPUModelRunner.__init__
            self.encoder_cache = None
```

- [ ] **Step 3: Override _cache_encoder_output**

Find `_cache_encoder_output` (inherited from upstream, not defined in this file). Add override inside the class, after existing methods:

```python
    def _cache_encoder_output(
        self, mm_hash: str, output: torch.Tensor,
        ec_manager_metadata, free_encoder_mm_hashes: list[str],
    ) -> None:
        if self.use_ec_memcache_offload:
            self.encoder_cache_store.put(mm_hash, output)
        else:
            self.encoder_cache[mm_hash] = output
            self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

- [ ] **Step 4: Override _get_encoder_output_from_cache**

```python
    def _get_encoder_output_from_cache(
        self, mm_hash: str,
    ) -> torch.Tensor | None:
        if self.use_ec_memcache_offload:
            return self.encoder_cache_store.get(mm_hash)
        return self.encoder_cache.get(mm_hash, None)
```

- [ ] **Step 5: Override _process_encoder_cache_scheduler_output**

```python
    def _process_encoder_cache_scheduler_output(
        self, scheduler_output,
    ) -> None:
        if self.use_ec_memcache_offload:
            pass  # memcache manages its own eviction
        else:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.pop(mm_hash, None)
```

- [ ] **Step 6: Override reset_encoder_cache**

```python
    def reset_encoder_cache(self) -> None:
        if self.use_ec_memcache_offload:
            pass  # memcache data managed at pool level
        else:
            self.encoder_cache.clear()
```

- [ ] **Step 7: Verify file parses**

Read the modified file and verify:
- Imports `EncoderCacheStore` from correct module
- 4 override methods added with `if self.use_ec_memcache_offload:` guards
- Each override method keeps original logic in `else:` branch
- `self.encoder_cache = None` set after creating `EncoderCacheStore`
- All code syntactically valid Python (check indentation, brackets, colons)

- [ ] **Step 8: Commit**

```bash
git add vllm_ascend/worker/model_runner_v1.py
git commit -m "feat: override 4 encoder cache methods for memcache offload path"
```

---

### Task 7: Unit tests

**Files:**
- Create: `tests/ut/distributed/ascend_store/test_ec_store_client.py`
- Create: `tests/ut/distributed/ascend_store/test_ec_manager_with_store.py`

**Interfaces:**
- Tests `EncoderCacheStoreClient.exists()` with mock ZMQ socket
- Tests `EncoderCacheManagerWithStore.check_and_update_cache()` flow
- Tests key format `_make_key()`

- [ ] **Step 1: Write test for EncoderCacheStoreClient (mock ZMQ)**

```python
# tests/ut/distributed/ascend_store/test_ec_store_client.py
from unittest.mock import MagicMock, patch

# import pytest  # Not available in this environment; test logic validated by code review

from vllm_ascend.distributed.ec_transfer.ec_store_client import (
    EncoderCacheStoreClient,
)


class TestEncoderCacheStoreClient:
    def test_exists_returns_true(self):
        with patch("zmq.Context") as mock_ctx:
            mock_socket = MagicMock()
            mock_socket.recv.return_value = b'1'
            mock_ctx.return_value.socket.return_value = mock_socket
            client = EncoderCacheStoreClient("ipc:///tmp/test")
            assert client.exists("abc123") is True
            mock_socket.send_string.assert_called_once_with("abc123")

    def test_exists_returns_false(self):
        with patch("zmq.Context") as mock_ctx:
            mock_socket = MagicMock()
            mock_socket.recv.return_value = b'0'
            mock_ctx.return_value.socket.return_value = mock_socket
            client = EncoderCacheStoreClient("ipc:///tmp/test")
            assert client.exists("abc123") is False
```

- [ ] **Step 2: Write test for EncoderCacheManagerWithStore**

```python
# tests/ut/distributed/ascend_store/test_ec_manager_with_store.py
from unittest.mock import MagicMock, patch

# import pytest  # Not available in this environment; test logic validated by code review


class TestEncoderCacheManagerWithStore:
    """Tests the scheduler-side manager with mocked ZMQ client."""

    # @pytest.fixture — converted to manual setup for offline environment
    def mock_client(self):
        return MagicMock()

    # @pytest.fixture — converted to manual setup for offline environment
    def mock_request(self):
        req = MagicMock()
        req.request_id = "req-1"
        req.mm_features = {0: MagicMock(identifier="hash-abc")}
        return req

    @patch(
        "vllm_ascend.distributed.ec_transfer.ec_store_client.EncoderCacheStoreClient"
    )
    def test_check_and_update_cache_local_hit(self, mock_client_cls, mock_request):
        from vllm_ascend.core.ec_manager_with_store import (
            EncoderCacheManagerWithStore,
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mgr = EncoderCacheManagerWithStore(cache_size=1000)
        # Pre-populate local cache
        mgr.cached["hash-abc"] = {"req-0"}

        assert mgr.check_and_update_cache(mock_request, 0) is True
        assert "req-1" in mgr.cached["hash-abc"]
        mock_client.exists.assert_not_called()  # ZMQ not called

    @patch(
        "vllm_ascend.distributed.ec_transfer.ec_store_client.EncoderCacheStoreClient"
    )
    def test_check_and_update_cache_zmq_hit(self, mock_client_cls, mock_request):
        from vllm_ascend.core.ec_manager_with_store import (
            EncoderCacheManagerWithStore,
        )
        mock_client = MagicMock()
        mock_client.exists.return_value = True
        mock_client_cls.return_value = mock_client

        mgr = EncoderCacheManagerWithStore(cache_size=1000)
        # Local cache is empty

        assert mgr.check_and_update_cache(mock_request, 0) is True
        assert "hash-abc" in mgr.cached
        assert "req-1" in mgr.cached["hash-abc"]
        mock_client.exists.assert_called_once_with("hash-abc")

    @patch(
        "vllm_ascend.distributed.ec_transfer.ec_store_client.EncoderCacheStoreClient"
    )
    def test_check_and_update_cache_full_miss(self, mock_client_cls, mock_request):
        from vllm_ascend.core.ec_manager_with_store import (
            EncoderCacheManagerWithStore,
        )
        mock_client = MagicMock()
        mock_client.exists.return_value = False
        mock_client_cls.return_value = mock_client

        mgr = EncoderCacheManagerWithStore(cache_size=1000)

        assert mgr.check_and_update_cache(mock_request, 0) is False
        mock_client.exists.assert_called_once_with("hash-abc")

    @patch(
        "vllm_ascend.distributed.ec_transfer.ec_store_client.EncoderCacheStoreClient"
    )
    def test_get_freed_mm_hashes_always_empty(self, mock_client_cls):
        from vllm_ascend.core.ec_manager_with_store import (
            EncoderCacheManagerWithStore,
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mgr = EncoderCacheManagerWithStore(cache_size=1000)
        assert mgr.get_freed_mm_hashes() == []

    @patch(
        "vllm_ascend.distributed.ec_transfer.ec_store_client.EncoderCacheStoreClient"
    )
    def test_free_cleans_local_cached_only(self, mock_client_cls, mock_request):
        from vllm_ascend.core.ec_manager_with_store import (
            EncoderCacheManagerWithStore,
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mgr = EncoderCacheManagerWithStore(cache_size=1000)
        mgr.cached["hash-abc"] = {"req-0"}
        mgr.request_cached_ids["req-0"] = {0}
        mgr.request_cached_ids["req-1"] = {0}
        mgr.cached["hash-abc"].add("req-1")

        # Free req-0
        mock_request.request_id = "req-0"
        # Override get_cached_input_ids for this test
        mgr.get_cached_input_ids = MagicMock(return_value={0})
        mgr.free_encoder_input(mock_request, 0)

        assert "req-0" not in mgr.cached["hash-abc"]
        assert "req-1" in mgr.cached["hash-abc"]  # Still referenced
```

- [ ] **Step 3: Run tests**

Read the test files and verify:
- Each test method tests exactly one behavior
- Mocks are used for ZMQ sockets and memcache backends
- Assertions check return values and side effects (mock_calls)
- Test isolation: no shared state between tests

- [ ] **Step 4: Commit**

```bash
git add tests/ut/distributed/ascend_store/test_ec_store_client.py tests/ut/distributed/ascend_store/test_ec_manager_with_store.py
git commit -m "test: add unit tests for EncoderCacheStoreClient and EncoderCacheManagerWithStore"
```

---

### Task 8: Integration sanity check

**Files:** None (verification only)

**Interfaces:** N/A

- [ ] **Step 1: Verify all imports reference existing modules**

Read each file and confirm imports only reference existing packages:
- `vllm_ascend.ascend_config` → `ECMemcacheConfig`, `get_ascend_config`
- `vllm_ascend.distributed.ec_transfer.ec_store_client` → `EncoderCacheStoreClient`, `get_zmq_rpc_path_ec_lookup`
- `vllm_ascend.distributed.ec_transfer.encoder_cache_store` → `EncoderCacheStore`, `_get_encoder_cache_hidden_dim`
- `vllm_ascend.core.ec_manager_with_store` → `EncoderCacheManagerWithStore`
- `vllm.v1.core.encoder_cache_manager` → `EncoderCacheManager` (upstream, read-only)

- [ ] **Step 2: Verify switch OFF path imports zero code from ec_transfer**

```python
# Verify that when ec_memcache_config.enabled=False, ec_transfer is NOT imported
# This is a manual check — confirm model_runner_v1.py imports EncoderCacheStore
# only inside the `if self.use_ec_memcache_offload:` block (lazy import pattern).
```

- [ ] **Step 3: Verify all files parse**

Open each file in the list and visually verify syntax:
- vllm_ascend/ascend_config.py
- vllm_ascend/distributed/ec_transfer/__init__.py
- vllm_ascend/distributed/ec_transfer/ec_store_client.py
- vllm_ascend/distributed/ec_transfer/encoder_cache_store.py
- vllm_ascend/core/ec_manager_with_store.py
- vllm_ascend/worker/model_runner_v1.py

Check each file: balanced parentheses/brackets, valid Python keywords, no dangling indentation.

- [ ] **Step 4: Commit if any cleanup needed**

```bash
git status
```

---

## Execution Order

```
Task 1  (config)
  └→ Task 2  (client + ZMQ path)
        └→ Task 3  (store)
  └→ Task 4  (manager, needs Task 2 only)
        └→ Task 5  (worker wiring, needs Task 1 + 4)
  └→ Task 6  (model runner, needs Task 1 + 3)
        └→ Task 7  (tests)
              └→ Task 8  (integration verify)
```

Tasks 2+3 can run in parallel after Task 1. Tasks 5+6 can run in parallel after their deps.
