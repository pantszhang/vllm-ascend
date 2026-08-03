# EC Memcache: 统计口径修复 & memcache 淘汰一致性修复

## 问题 1：compute_rate 超过 100%（双重 STORE + HIT 时多余 STORE）

### 现象

```
EC memcache STORE: gets=3 stores=4 offload_hit_rate=-33.3% compute_rate=133.3%
```

`compute_rate = stores / gets * 100` 超过 100%，而且 `hits=3 misses=0` 但 `stores=4`。

### 根因

`model_runner_v1.py` 中有两处冗余的 memcache 写入：

**双重 STORE**（`_cache_encoder_output`）：

```python
def _cache_encoder_output(self, mm_hash, output, ...):
    self.encoder_cache[mm_hash] = output      # ① _EcMemcacheDict.__setitem__
                                               #    → _store.put() → _cnt_stores+1
    self.encoder_cache_store.put(mm_hash, output)  # ② 直接再写一次
                                               #    → _store.put() → _cnt_stores+1
```

第①行 `self.encoder_cache[mm_hash] = output` 触发 `_EcMemcacheDict.__setitem__` 已经写入了 memcache，第②行又写了一次。每次实际 store 被计数两次。

**HIT 时多余 STORE**（`_get_encoder_output_from_cache`）：

```python
def _get_encoder_output_from_cache(self, mm_hash):
    tensor = self.encoder_cache_store.get(mm_hash)  # 从 memcache 读出来
    if tensor is not None:
        self.encoder_cache[mm_hash] = tensor  # 又存回去 → _cnt_stores+1
```

memcache 命中后不应该再 store，但这个 `__setitem__` 触发了写入计数。

**LOCAL_HIT 死代码**：

`_EcMemcacheDict.__setitem__` 对非 tmp_ key 只写 memcache，不写 local dict。导致 `_get_encoder_output_from_cache` 第一段检查 `mm_hash in self.encoder_cache` 永远为 False。

### 修复（commit `003cc2011`）

1. 删除 `_cache_encoder_output` 中的 `self.encoder_cache_store.put()` — 只保留 `self.encoder_cache[mm_hash]=output`
2. 删除 `_get_encoder_output_from_cache` 中 HIT 路径的 `self.encoder_cache[mm_hash]=tensor`
3. 重排 `_get_encoder_output_from_cache` 逻辑 — memcache 模式直接查 memcache

---

## 问题 2：memcache 淘汰导致 assert 崩溃

### 现象

```
15:56:10  DP0 MISS → Worker STORE   ← 计算并存入 memcache ✓
15:56:10  Worker HIT                ← _gather 能取到 ✓

15:59:34  DP0 MEMCACHE_HIT          ← ZMQ exists()=True ✓
15:59:34  Worker HIT                ← memcache 还有 ✓

         ~~~ 10分钟，新增 ~400 条数据，memcache LRU 淘汰 ~~~

16:09:13  DP0 LOCAL_HIT             ← self.cached 里还有，调度器说"不用算了"
16:09:14  Worker MISS               ← memcache 已被淘汰！
16:09:14  AssertionError: Encoder cache miss  ← 💥 崩溃
```

### 根因

**调度器和 memcache 各管各的淘汰，没有同步。**

```
调度器 self.cached           memcache                local dict
─────────────────           ────────                ──────────
15:56 allocate()→{"07fe"}    STORE→{"07fe"}          (空)

15:59 MEMCACHE_HIT           有 {"07fe"}             (空)
     {"07fe"} 还在 ✓

16:09 LOCAL_HIT              LRU淘汰，{"07fe"}没了    (空)
     {"07fe"} 还在 ✓
     告诉Worker"有缓存"                              ← 没有！
                                          ↓
                    Worker: encoder_cache.get("07fe")
                    → memcache MISS
                    → local dict MISS
                    → return None → ASSERT 💥
```

上游 `_gather_mm_embeddings`（`gpu_model_runner.py:3138`）有硬断言：

```python
encoder_output = self.encoder_cache.get(mm_hash, None)
assert encoder_output is not None  # 调度器承诺有的，就必须有
```

在原生 vllm 里 `self.encoder_cache` 是普通 dict，淘汰完全由调度器引用计数控制，这个 assert 永远成立。但 memcache 引入后，LRU 淘汰打破了调度器的承诺。

三个具体原因叠加：

1. **`_EcMemcacheDict.__setitem__`** 只写 memcache，不写 local dict → local dict 永远是空的
2. **`get_freed_mm_hashes()`** 永远返回 `[]` → 调度器不通知 Worker 淘汰 → 即使写了 local dict 也会内存泄漏
3. **`free_encoder_input()`** 删掉 `self.cached` 后不通知外界 → `self.freed` 永远是空的

### 修复（commit `22449a640`）

**数据面** — `_EcMemcacheDict` 写 local dict 作为兜底：

```python
def __setitem__(self, key, value):
    if isinstance(key, str) and not key.startswith("tmp_"):
        _store.put(key, value)              # memcache（跨 worker 共享）
        dict.__setitem__(self, key, value)  # local dict（兜底/快速路径）

def get(self, key, default=None):
    if isinstance(key, str) and not key.startswith("tmp_"):
        if key in self:                            # ① local dict（免 G2L copy）
            return super().get(key)
        tensor = _store.get(key)                   # ② memcache
        if tensor is not None:
            dict.__setitem__(self, key, tensor)    # 回填 local dict
            return tensor
    return super().get(key, default)
```

**调度面** — 恢复正常的引用计数淘汰：

```python
# ec_manager_with_store.py

def free_encoder_input(self, request, input_id):
    ...
    if not self.cached[mm_hash]:
        del self.cached[mm_hash]
        self.freed.append(mm_hash)   # 通知 Worker 可以清理了

def get_freed_mm_hashes(self):
    freed = self.freed
    self.freed = []
    return freed                      # 不再是永远 []
```

**Worker 面** — 始终清理 local dict，但不动 memcache：

```python
def _process_encoder_cache_scheduler_output(self, scheduler_output):
    for mm_hash in scheduler_output.free_encoder_mm_hashes:
        self.encoder_cache.pop(mm_hash, None)  # 只清 local dict
    # memcache 不动 — 留给跨 worker 共享
```

### 最终架构

```
                   调度器                        Worker
              ┌──────────────┐          ┌──────────────────────┐
 request →    │ self.cached  │          │ _EcMemcacheDict      │
              │  (引用计数)    │          │                      │
              │              │          │  local dict  memcache │
              │  free→freed  │──pop──→  │  ┌──────┐   ┌──────┐│
              │              │          │  │ refs │   │数据共享││
              │              │          │  └──────┘   └──────┘│
              └──────────────┘          │    ↑            ↑    │
                                        │ 调度器淘汰    LRU淘汰 │
                                        │ 同步控制      独立管理│
                                        └──────────────────────┘
```

- **local dict**：存 Python 引用（指向 NPU 原始 tensor），生命周期和调度器引用计数完全同步。保证"调度器承诺有的，local dict 就一定还有"。作为 memcache 淘汰时的安全兜底。
- **memcache**：独立 LRU 管理，调度器不干预。跨 worker 共享仍然有效。如果没被淘汰，get 走 local dict 快速路径（免 G2L copy）。
- **两者互不干扰**：memcache 淘汰不影响 local dict；local dict 的 pop 不影响 memcache。
