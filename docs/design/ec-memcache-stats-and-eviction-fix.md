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

### 修复（commit `605052c95`）

**目标语义**：
- `gets`：收到多少次 encoder cache 查询（`_gather_mm_embeddings` 中的 `encoder_cache.get()` 调用）
- `hits`：在 memcache 中直接命中（数据之前就存在）
- `misses`：memcache 中找不到（被 LRU 淘汰）
- 保证：`gets = hits + misses`

**不再用 `stores` 参与比率计算**。`stores` 只作为独立指标展示（总共写入了多少次 memcache）。

```
_EcMemcacheDict.get() 的最终逻辑：

    def get(self, key, default=None):
        if isinstance(key, str) and not key.startswith("tmp_"):
            try:
                tensor = _store.get(key)      # ① 始终走 memcache
                                               #   gets+1, hit 或 miss+1
                if tensor is not None:
                    dict.__setitem__(self, key, tensor)  # 回填 local dict
                    return tensor
            except Exception:
                ...
            # ② memcache miss（淘汰）→ local dict 兜底 → assert 不炸
            if key in self:
                return super().get(key)
        return super().get(key, default)
```

`_stats()` 公式：

```python
if self._cnt_gets > 0:
    offload_rate = total_hits / self._cnt_gets * 100
    compute_rate = self._cnt_misses / self._cnt_gets * 100
```

**`compute_rate` 的含义变了**：不再是"多少图需要计算"，而是"多少 memcache 查询命中失败（被淘汰）"。大部分时间 `compute_rate ≈ 0%`（memcache 很少淘汰），`offload_hit_rate ≈ 100%`。

要了解实际计算量，看 `stores` 指标即可——每 store 一次代表调度器判定了一次 MISS 并计算了一张新图。
