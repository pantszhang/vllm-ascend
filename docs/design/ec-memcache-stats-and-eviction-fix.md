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

---

## 问题 3：local dict 快速路径导致统计全为 0

### 现象

```
EC memcache STORE: gets=0 stores=83 hits=0 hbm_hits=0 dram_hits=0 misses=0
```

`gets`、`hits`、`misses` 始终为 0，只有 `stores` 在增长。

### 根因

问题 2 的修复中，`_EcMemcacheDict.get()` 加入了 local dict 快速路径：

```python
def get(self, key, default=None):
    if isinstance(key, str) and not key.startswith("tmp_"):
        if key in self:                            # ① local dict 快速路径
            return super().get(key)                 # ← 直接返回！
        tensor = _store.get(key)                   # ② memcache（计数器在这里）
        ...
```

`__setitem__` 写入 local dict 后，下一次 `get()` 在步骤 ① 就命中了，**直接返回，跳过了步骤 ②**。而 `_cnt_gets`、`_cnt_hits`、`_cnt_misses` 全都在 `EcMemcacheBackend.get()`（步骤 ②）里统计。local dict 快速路径绕过了计数器，导致统计全为 0。

```
第一次请求:
  __setitem__ → memcache ✓ + local dict ✓

第二次请求（同一个 key）:
  get() → if key in self: 命中! → 直接返回  ← 没调用 _store.get()
                                          ← 计数器全不更新
```

### 修复（commit `f1e49f864`）

去掉 local dict 快速路径，**始终先走 `_store.get()`** 保证计数器更新。local dict 只在 memcache miss 时作为兜底：

```python
def get(self, key, default=None):
    if isinstance(key, str) and not key.startswith("tmp_"):
        try:
            tensor = _store.get(key)              # ① 始终先走 memcache（更新计数器）
            if tensor is not None:
                dict.__setitem__(self, key, tensor)  # 回填 local dict
                return tensor
        except Exception as e:
            ...
        # Memcache miss — local dict 兜底（memcache 被淘汰的情况）
        if key in self:                            # ② local dict 兜底
            return super().get(key)
    return super().get(key, default)
```

关键：**local dict 从"快速路径"降级为"兜底路径"**。计数器准确性优先，local dict 只处理 memcache 淘汰的异常情况。

这个改动不会导致问题 2 的 assert 崩溃复发——local dict 仍然在 memcache miss 后兜底。调度器承诺有的 key，一定能取到：
- memcache 有 → ① 命中 → 返回 ✓
- memcache 被淘汰 → ① miss → ② local dict 命中 → 返回 ✓
- 都没有 → 返回 None（但调度器不会在这种情况下说 LOCAL_HIT）

---

## 问题 4：misses 和 hits 语义混乱，compute_rate > 100%

### 现象

```
EC memcache STORE: gets=3 stores=4 hits=3 misses=0 offload_hit_rate=-33.3% compute_rate=133.3%
EC memcache STORE: gets=3 stores=10 hits=3 misses=0 offload_hit_rate=-233.3% compute_rate=333.3%
```

`compute_rate > 100%`，`offload_hit_rate < 0%`。

### 根因

**`put()` 和 `get()` 在同一个 step 内的执行顺序不同**：

```
_execute_mm_encoder    → put() → stores++    ← 先跑
_gather_mm_embeddings  → get() → gets++      ← 后跑
```

在 `_execute_mm_encoder` 阶段 stores 已经涨上去了，gets 还没动。旧公式 `compute_rate = stores / gets * 100` 必然超过 100%。

更深层的问题是 **misses 和 hits 的语义混乱**：

```
旧代码:
  put()  → 无 miss 记录               (调度器判定的 MISS 没被记录)
  get()  → ki.size()>0 → hits+1      (刚存进去的也算 HIT)
  get()  → ki.size()==0 → misses+1   (只有 memcache 淘汰才触发)

结果：一张新图在 get() 时永远 HIT（刚 put 进去的），所以 misses=0。
     但 stores 已经 +1 了，compute_rate = stores / gets 自然 > 100%。
```

核心矛盾：新图的"调度器判定 MISS → 计算 → store"这条路径，在 get() 看来永远是 HIT（数据已存入）。**HIT 有双重含义——"之前就有的缓存"和"刚算出来存进去的"。**

### 修复（commit `c2899fe64` → `a7ecb9011`）

**最终方案**：引入 `_fresh` 集合标记本 step 刚 put 的 key。

```
put(key)                           get(key)
  ├─ memcache                        ├─ key in _fresh?  ──→ MISS  (本 step 刚存的)
  ├─ local dict                      ├─ key in self?    ──→ HIT   (之前就缓存的)
  └─ _fresh.add(key)                 └─ memcache        ──→ HIT/MISS (其他 worker)
```

**`_EcMemcacheDict` 完整代码**：

```python
class _EcMemcacheDict(dict):
    _fresh: set = set()  # keys stored by put() in the current step

    def __setitem__(self, key, value):
        if isinstance(key, str) and not key.startswith("tmp_"):
            try:
                _store.put(key, value)
            except Exception as e:
                logger.warning(...)
            dict.__setitem__(self, key, value)
            _EcMemcacheDict._fresh.add(key)       # 标记"刚存的"
        else:
            super().__setitem__(key, value)

    def get(self, key, default=None):
        if isinstance(key, str) and not key.startswith("tmp_"):
            # ① 本 step 刚 put 的 → 逻辑 MISS
            if key in _EcMemcacheDict._fresh:
                _store.record_get_miss()
                _EcMemcacheDict._fresh.discard(key)
                return dict.__getitem__(self, key)
            # ② 之前就缓存的 → 逻辑 HIT
            if key in self:
                _store.record_local_hit()
                return dict.__getitem__(self, key)
            # ③ 其他 worker / 之前的 session → memcache
            try:
                tensor = _store.get(key)
                if tensor is not None:
                    dict.__setitem__(self, key, tensor)
                    return tensor
            except Exception as e:
                logger.warning(...)
            # ④ memcache miss (被淘汰) — 罕见
        return super().get(key, default)
```

**`EcMemcacheBackend` 新增方法**：

```python
def record_get_miss(self) -> None:
    """Record a logical get + miss (data was just stored or evicted)."""
    self._cnt_gets += 1
    self._cnt_misses += 1

def record_local_hit(self) -> None:
    """Record a get that hit in local dict (previously cached)."""
    self._cnt_gets += 1
    self._cnt_hits["local"] = self._cnt_hits.get("local", 0) + 1
```

**`_stats()` 公式**：

```python
total_hits = sum(self._cnt_hits.values())  # 包含 "local" 命中
offload_rate = total_hits / self._cnt_gets * 100
compute_rate = self._cnt_misses / self._cnt_gets * 100
```

### 最终语义

| 指标 | 含义 | 来源 |
|------|------|------|
| `gets` | 收到多少次 cache 查询 | `_gather_mm_embeddings` |
| `hits` | 之前就缓存好的（local + memcache） | `record_local_hit()` + memcache HIT |
| `misses` | 本 step 刚算的 / memcache 淘汰的 | `record_get_miss()` + memcache MISS |
| `stores` | 总共写入了多少次 memcache | `put()` |

**保证**：`gets = hits + misses` 恒成立，两个比率永远在 0~100%。

### 数值示例

请求 1（7 张新图 + 3 张已缓存）：
```
_execute_mm_encoder:  7 puts → _fresh={7 keys}, stores=7
_gather_mm_embeddings:
  7 keys in _fresh → gets+7, misses+7
  3 keys in self   → gets+3, hits+3
结果: gets=10, hits=3, misses=7, stores=7
      offload_hit_rate=30%, compute_rate=70%
```

请求 2（相同 10 张图）：
```
_execute_mm_encoder:  0 puts → _fresh={}, stores=7
_gather_mm_embeddings:
  10 keys in self → gets+10, hits+10
结果: gets=20, hits=13, misses=7, stores=7
      offload_hit_rate=65%, compute_rate=35%
```

请求 N 后：`offload_hit_rate → 100%`，`compute_rate → 0%`。
