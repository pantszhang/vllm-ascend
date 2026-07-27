# EC Memcache Encoder Cache 卸载 — 设计文档

## Context

当前 vllm-ascend 的 memcache 基础设施仅服务于 KV Cache 传输（通过 `AscendStoreConnector` → `MemcacheBackend`），多模态 encoder 输出的 embedding 只在进程内 `dict[str, torch.Tensor]` 中缓存，无法跨请求复用。本设计新增基于 memcache 的 encoder embedding 缓存卸载，在不修改 vllm 上游和 `memcache_hybrid` 的前提下，全部改动落在 vllm-ascend 仓库中。

**核心约束：**
- vllm 上游：零改动
- memcache_hybrid：零改动
- EC 与 KV 的 backend 配置独立解耦
- 未开启功能时所有原路径不受影响
- memcache 不写死，支持通过配置扩展其他 backend

## 一、架构

```
┌─ vllm 上游（不动）─────────────────────────────────────────────┐
│  ECConnectorBase              Scheduler.ec_connector           │
│  ECConnectorModelRunnerMixin  ECTransferConfig                 │
│  ECConnectorFactory           ECConnectorOutput                │
└───────────────────────────────────────────────────────────────┘
                           │ extends
┌─ vllm-ascend（4 新建 + 1 修改）────────────────────────────────┐
│                                                                │
│ 新建:                                                          │
│  ec_transfer/                                         │
│  ├── ec_store_connector.py         ← ECStoreConnector          │
│  └── ec_backend/                                              │
│      ├── __init__.py               ← ec_backend_map           │
│      ├── backend.py                ← ECBackend (抽象)          │
│      └── memcache_backend.py       ← ECMemcacheBackend        │
│                                                                │
│ 修改 (1 行):                                                    │
│  worker/model_runner_v1.py         ← ec_both 判断修复          │
└────────────────────────────────────────────────────────────────┘
```

**设计原则：**
- ECBackend 与 KV Backend 平级独立，共享模式（backend_map + 动态 import）但不共享实例
- ECStoreConnector 内部通过 `role` 区分 Scheduler/Worker，两端各自创建 ECBackend 客户端
- 所有 `memcache_hybrid.*` import 仅在 `ec_backend/memcache_backend.py` 文件中出现
- 配置驱动：`ec_connector_extra_config.get("backend")` 选择存储后端

## 二、新建文件

### 2.1 `vllm_ascend/distributed/ec_transfer/ec_backend/backend.py`

EC 专用存储后端抽象，仅定义 EC 需要的接口（与 KV `Backend` 无继承关系）：

```python
class ECBackend(ABC):

    @abstractmethod
    def batch_is_exist(self, keys: list[str]) -> list[int]:
        """返回 [1] 表示存在，[0] 表示不存在。"""

    @abstractmethod
    def batch_alloc(self, keys: list[str], sizes: list[int]) -> list[int]:
        """为每个 key 分配空间，返回 GVA 列表。"""

    @abstractmethod
    def batch_get_key_info(self, keys: list[str]):
        """返回 key_info 对象，可调用 .size() 和 .gva_list()。"""

    @abstractmethod
    def batch_copy(
        self, gvas: list[int], addrs: list[int], sizes: list[int], direction: int
    ) -> int:
        """DMA 拷贝。direction: 0=L2G(save), 1=G2L(load)。返回 0 表示成功。"""

    @abstractmethod
    def batch_add_lease(self, keys: list[str], ttl_ms: int) -> list[int]:
        """获取读租约，防驱逐。"""

    @abstractmethod
    def batch_remove_lease(self, keys: list[str]) -> int:
        """释放读租约。"""

    @classmethod
    def create_scheduler_client(cls, **kwargs) -> "ECBackend":
        """创建 scheduler 端轻量客户端，只需支持 batch_is_exist。"""
```

### 2.2 `vllm_ascend/distributed/ec_transfer/ec_backend/__init__.py`

```python
# 注册表：字符串名 → {name, path}
ec_backend_map = {
    "memcache": {
        "name": "ECMemcacheBackend",
        "path": "vllm_ascend.distributed.ec_transfer.ec_backend.memcache_backend",
    },
}

def create_ec_backend(name: str, **kwargs) -> ECBackend:
    """根据 name 从 ec_backend_map 动态 import 并构造 ECBackend 实例。"""
```

### 2.3 `vllm_ascend/distributed/ec_transfer/ec_backend/memcache_backend.py`

包装 `memcache_hybrid.DistributedObjectStore`，实现 `ECBackend` 所有方法。

**使用的 memcache_hybrid API（全部在 KV `MemcacheBackend` 中有调用记录）：**

| API | 用途 |
|-----|------|
| `DistributedObjectStore()` | 构造 |
| `store.init(local_rank, init_bm)` | 初始化，连接 MetaService |
| `store.batch_is_exist(keys)` | Scheduler 端命中检查 + Worker 端存前去重 |
| `store.batch_alloc(keys, sizes)` | 为 encoder output 分配 GVA |
| `store.batch_get_key_info(keys)` | Worker 端获取 GVA + size |
| → `ki.size()` / `ki.gva_list()[0]` | 从 key_info 对象取字节数和 GVA |
| `store.batch_copy(gvas, addrs, sizes, direction)` | DMA 传输 |
| `store.batch_add_lease(keys, ttl_ms)` | 读租约（防驱逐） |
| `store.batch_remove_lease(keys)` | 释放租约 |
| `MmcDirect.COPY_L2G (0)` | save 方向 |
| `MmcDirect.COPY_G2L (1)` | load 方向 |

**Scheduler 端轻量客户端：** `create_scheduler_client()` 返回 `ECMemcacheBackend(local_rank=0, init_bm=False)`，只使用 `batch_is_exist`。

**不需要的方法（KV 专用，EC 不暴露）：** `register_buffer`、`put`（key路径）、`get`（key路径）、`batch_put_from_layers`、`batch_get_into_layers`。

**待验证：** A2 上动态创建的 encoder embedding tensor 是否需 `register_buffer`。若需要，在 `save_caches` 前调用。

### 2.4 `vllm_ascend/distributed/ec_transfer/ec_store_connector.py`

```python
class ECStoreConnector(ECConnectorBase):
    """基于 ECBackend 的 ECConnector 实现。

    内部结构：role 分发 + backend 委托
    ┌─ role == SCHEDULER ───────────────────┐
    │  ECStoreScheduler                     │
    │   has_cache_item → backend.exists     │
    │   update_state_after_alloc            │
    │   build_connector_meta                │
    └───────────────────────────────────────┘
    ┌─ role == WORKER ──────────────────────┐
    │  ECStoreWorker                        │
    │   save_caches → alloc + copy(L2G)     │
    │   start_load_caches → info + lease +  │
    │                       copy(G2L)       │
    └───────────────────────────────────────┘
    """
```

**Scheduler 端实现：**

| 方法 | 逻辑 |
|------|------|
| `has_cache_item(identifier)` | `backend.batch_is_exist([key])[0] == 1` |
| `update_state_after_alloc(request, index)` | 记录需要 load 的 mm_hash（同 ECExampleConnector 模式） |
| `build_connector_meta(scheduler_output)` | 构建 metadata，包含 loads 列表 |
| `ensure_cache_available(request, num)` | 返回 True |
| `request_finished(request)` | 返回 False, None |

**Worker 端实现：**

`save_caches(encoder_cache, mm_hash)`:
1. `is_producer` 且 `tp_rank == 0` 且 `pcp_rank == 0` 才执行
2. 从 `encoder_cache[mm_hash]` 取 tensor
3. 计算 `key = f"encoder@{model_name}@{mm_hash}"`
4. `batch_is_exist([key])` → 已存在则跳过（去重）
5. `batch_alloc([key], [tensor.numel() * tensor.element_size()])` → gva
6. `batch_copy([gva], [tensor.data_ptr()], [size], COPY_L2G)`

`start_load_caches(encoder_cache, **kwargs)`:
1. `is_consumer` 才执行
2. 从 metadata 获取需要 load 的 mm_hash 列表
3. 对每个 key: `batch_get_key_info([key])` → size, gva
4. `num_tokens = size // (hidden_size * element_size)`
5. `buffer = torch.empty(num_tokens, hidden_size, dtype=..., device="npu")`
6. `batch_add_lease([key], lease_ttl_ms)`
7. `batch_copy([gva], [buffer.data_ptr()], [size], COPY_G2L)`
8. `batch_remove_lease([key])`
9. `encoder_cache[mm_hash] = buffer`

**Key 格式：** `"encoder@{model_name}@{mm_hash}"` — 不含 `tp_rank`，所有 TP rank 共享同一份缓存。

**Metadata 格式：** 包含 `loads: list[str]`（需要 load 的 mm_hash 列表），通过 `ECStoreConnectorMetadata` 在 Scheduler → Worker 间传递。

## 三、修改已有文件

### 3.1 `vllm_ascend/worker/model_runner_v1.py`（改 1 行）

**位置：** 第 2080 行附近

```python
# 改前
if has_ec_transfer() and get_ec_transfer().is_producer:

# 改后
ec = get_ec_transfer()
ec_config = self.vllm_config.ec_transfer_config
extra = ec_config.ec_connector_extra_config if ec_config else {}
is_ec_memcache_both = (extra.get("backend", "") == "memcache" and ec.is_consumer)

if has_ec_transfer() and ec.is_producer and not is_ec_memcache_both:
```

**各场景行为矩阵：**

| 配置 | backend | is_producer | is_consumer | is_ec_memcache_both | 行为 |
|------|---------|-------------|-------------|---------------------|------|
| 无 EC | - | - | - | - | 原路径 ✓ |
| ec_producer + memcache | memcache | True | False | **False** | 原路径（纯 EP producer 提前返回）✓ |
| ec_producer + other | other | True | False | **False** | 原路径 ✓ |
| ec_both + memcache | memcache | True | True | **True** | 新路径（不走提前返回，load cache + encoder 跳过已缓存 + LLM forward）✓ |
| ec_both + other | other | True | True | **False** | 原路径 ✓ |
| ec_consumer | any | False | True | **False** | 原路径 ✓ |

## 四、流程

### 4.1 初始化

```
vLLM 启动 ──→ 解析 --ec-transfer-config
  │
  ├─→ ECConnectorFactory.create_connector("ECStoreConnector", role=SCHEDULER)
  │     │
  │     ├─→ ECStoreConnector.__init__(vllm_config, role=SCHEDULER)
  │     │     │
  │     │     ├─ 读取 ec_connector_extra_config.get("backend")  → "memcache"
  │     │     ├─ create_ec_backend("memcache", scheduler_mode=True)
  │     │     │     │
  │     │     │     └─→ ECMemcacheBackend(local_rank=0, init_bm=False)
  │     │     │           │
  │     │     │           └─→ DistributedObjectStore().init(0, init_bm=False)
  │     │     │                 │
  │     │     │                 └─ 连接 MetaService tcp://x.x.x.x:5000
  │     │     │                    (轻量，不注册 GPU buffer，只用 batch_is_exist)
  │     │     │
  │     │     └─ self._store → ECMemcacheBackend (scheduler 轻量客户端)
  │     │
  │     └─ Scheduler.ec_connector = ECStoreConnector 实例
  │
  ├─→ ECConnectorFactory.create_connector("ECStoreConnector", role=WORKER)
  │     │
  │     └─→ ECStoreConnector.__init__(vllm_config, role=WORKER)
  │           │
  │           ├─ 读取 ec_connector_extra_config.get("backend")  → "memcache"
  │           ├─ create_ec_backend("memcache", local_rank=n)
  │           │     │
  │           │     └─→ ECMemcacheBackend(local_rank=n, init_bm=True)
  │           │           │
  │           │           └─→ DistributedObjectStore().init(n, init_bm=True)
  │           │                 │
  │           │                 └─ 连接 MetaService，注册 GPU 通信
  │           │
  │           └─ self._store → ECMemcacheBackend (worker 完整客户端)
  │
  └─ 初始化完成

依赖关系:
  MetaService ← 由 start_memcache.sh 预先启动
  mmc-local.conf ← 由 start_memcache.sh 生成，MMC_LOCAL_CONFIG_PATH 环境变量指定
```

### 4.2 请求 1：首次遇到图片 (Cache Miss)

```
用户请求 (image: dog.jpg, mm_hash="abc123")
  │
  ▼
┌─ Scheduler ────────────────────────────────────────────────────────┐
│                                                                     │
│  1. ec_connector.has_cache_item("abc123")                          │
│       │                                                             │
│       ▼                                                             │
│     ECStoreConnector (Scheduler 端)                                 │
│       │                                                             │
│       ├─ 构造 key: "encoder@{model_name}@abc123"                   │
│       │                                                             │
│       └─ ECMemcacheBackend (scheduler 轻量客户端)                   │
│            │                                                        │
│            └─ batch_is_exist(["encoder@model@abc123"])              │
│                 │                                                   │
│                 └─→ [0] ←── miss!                                  │
│                                                                     │
│  2. update_state_after_alloc(request, index)                       │
│       → 无 cache 命中，正常调度 encoder 计算                         │
│                                                                     │
│  3. build_connector_meta(scheduler_output)                          │
│       → ECStoreConnectorMetadata(loads=[])  ← 空列表                │
│                                                                     │
│  4. scheduler_output.ec_connector_metadata = metadata               │
│                                                                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ metadata + scheduler_output
                                   ▼
┌─ Worker ────────────────────────────────────────────────────────────┐
│                                                                     │
│  ★ model_runner_v1.py: is_producer=True, is_consumer=True (ec_both)│
│    is_ec_memcache_both = True → 不走提前返回，正常执行 LLM forward   │
│                                                                     │
│  1. bind_connector_metadata(metadata)                               │
│       → metadata.loads = []   ← 无需加载                             │
│                                                                     │
│  2. start_load_caches(encoder_cache)                                │
│       │                                                             │
│       ▼                                                             │
│     ECStoreConnector (Worker 端)                                    │
│       │  metadata.loads 为空 → skip                                 │
│       └─ 不触发 memcache 操作                                        │
│                                                                     │
│  3. _execute_mm_encoder(scheduler_output)                          │
│       │                                                             │
│       ├─ encoder_cache.get("abc123") → None (不在 cache 中)           │
│       │                                                             │
│       └─ vision encoder forward                                    │
│            │                                                        │
│            └─ tensor: [1732, 4096], bfloat16, 14.2 MB               │
│                                                                     │
│  4. _cache_encoder_output("abc123", tensor, ...)                    │
│       │                                                             │
│       ├─ encoder_cache["abc123"] = tensor   ← 进程内缓存              │
│       │                                                             │
│       └─ maybe_save_ec_to_connector(encoder_cache, "abc123")        │
│            │                                                        │
│            ▼                                                       │
│          ECStoreConnector.save_caches(encoder_cache, "abc123")      │
│            │                                                        │
│            │  ★ 仅 tp_rank==0 && pcp_rank==0 执行 ★                │
│            │                                                        │
│            ▼                                                       │
│          ECMemcacheBackend (Worker 完整客户端)                       │
│            │                                                        │
│            ├─ ① batch_is_exist(["encoder@model@abc123"])            │
│            │      → [0]   ← 去重检查，无人存过                        │
│            │                                                        │
│            ├─ ② size = 1732 × 4096 × 2 = 14,188,544 bytes          │
│            │    batch_alloc(["encoder@model@abc123"], [14188544])    │
│            │      → [gva_0x7f...]   ← memcache 分配空间             │
│            │                                                        │
│            └─ ③ batch_copy(                                        │
│                     gvas     = [gva_0x7f...],                       │
│                     addrs    = [tensor.data_ptr()],                 │
│                     sizes    = [14188544],                          │
│                     direction= COPY_L2G (0)                         │
│                   )                                                 │
│                   → NPU ──SDMA──→ memcache DRAM                     │
│                                                                     │
│  5. LLM forward (使用 encoder output)                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 请求 2：同一张图片再次出现 (Cache Hit)

```
用户请求 (image: dog.jpg, mm_hash="abc123")  ← 同一张图
  │
  ▼
┌─ Scheduler ────────────────────────────────────────────────────────┐
│                                                                     │
│  1. ec_connector.has_cache_item("abc123")                          │
│       │                                                             │
│       ▼                                                             │
│     ECMemcacheBackend (scheduler 轻量客户端)                         │
│       │                                                             │
│       └─ batch_is_exist(["encoder@model@abc123"])                   │
│            │                                                        │
│            └─→ [1] ←── HIT! ✓                                      │
│                                                                     │
│  2. update_state_after_alloc(request, index)                       │
│       → 记录 "abc123" 需要从 cache 加载                              │
│       → self._pending_loads["abc123"] = ...                        │
│                                                                     │
│  3. build_connector_meta(scheduler_output)                          │
│       → ECStoreConnectorMetadata(loads=["abc123"])                  │
│                                                                     │
│  4. scheduler_output.ec_connector_metadata = metadata               │
│                                                                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ metadata + scheduler_output
                                   ▼
┌─ Worker ────────────────────────────────────────────────────────────┐
│                                                                     │
│  ★ is_ec_memcache_both = True → 不提前返回                           │
│                                                                     │
│  1. bind_connector_metadata(metadata)                               │
│       → metadata.loads = ["abc123"]                                 │
│                                                                     │
│  2. start_load_caches(encoder_cache)                                │
│       │                                                             │
│       ▼                                                             │
│     ECStoreConnector (Worker 端)                                    │
│       │                                                             │
│       │ 遍历 metadata.loads: ["abc123"]                              │
│       │                                                             │
│       ▼                                                             │
│     ECMemcacheBackend (Worker 完整客户端)                             │
│       │                                                             │
│       ├─ ① batch_get_key_info(["encoder@model@abc123"])             │
│       │      │                                                      │
│       │      ├─ ki.size() → 14,188,544  bytes                      │
│       │      └─ ki.gva_list()[0] → gva_0x7f...                     │
│       │                                                             │
│       ├─ ② num_tokens = 14188544 // (4096 × 2) = 1732              │
│       │    buffer = torch.empty(1732, 4096, bfloat16, "npu")        │
│       │                                                             │
│       ├─ ③ batch_add_lease(                                        │
│       │        ["encoder@model@abc123"],                            │
│       │        lease_ttl_ms = 300000                                │
│       │      )                                                      │
│       │      → 加锁！memcache 不会驱逐此 blob                        │
│       │                                                             │
│       ├─ ④ batch_copy(                                             │
│       │        gvas     = [gva_0x7f...],                            │
│       │        addrs    = [buffer.data_ptr()],                      │
│       │        sizes    = [14188544],                               │
│       │        direction= COPY_G2L (1)                              │
│       │      )                                                      │
│       │      → memcache DRAM ──SDMA──→ NPU buffer                  │
│       │                                                             │
│       └─ ⑤ batch_remove_lease(["encoder@model@abc123"])            │
│            → 解锁                                                   │
│                                                                     │
│       encoder_cache["abc123"] = buffer                              │
│                                                                     │
│  3. _execute_mm_encoder(scheduler_output)                          │
│       │                                                             │
│       └─ encoder_cache.get("abc123") → tensor ← 已存在！             │
│          → 跳过 vision encoder forward  ←── 省时！                   │
│                                                                     │
│  4. LLM forward (使用从 memcache 加载的 embedding)                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.4 组件职责总结

```
                    Scheduler 端              Worker 端
                    ─────────────             ─────────
ECStoreConnector   has_cache_item()          save_caches()
                   update_state_after_alloc() start_load_caches()
                   build_connector_meta()

ECMemcacheBackend  batch_is_exist()          batch_alloc()
(scheduler 轻量)                              batch_copy(L2G)
                                              batch_get_key_info()
                                              batch_copy(G2L)
                                              batch_add/remove_lease()

Key 格式:           "encoder@{model_name}@{mm_hash}"
```

## 五、配置

### 5.1 memcache 基础设施（部署层）

```bash
# 和 KV memcache 完全一样
source scripts/memcache/env_memcache.sh   # MMC_LOCAL_CONFIG_PATH, hugepages, HCCL
./scripts/memcache/start_memcache.sh      # MetaService + mmc-local.conf
```

### 5.2 vLLM 启动参数

```json
{
  "--ec-transfer-config": {
    "ec_connector": "ECStoreConnector",
    "ec_role": "ec_both",
    "ec_connector_module_path": "vllm_ascend.distributed.ec_transfer.ec_store_connector",
    "ec_connector_extra_config": {
      "backend": "memcache",
      "lease_ttl_ms": 300000
    }
  }
}
```

`ec_connector_extra_config` 字段说明：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `backend` | `"memcache"` | 存储后端名称，对应 `ec_backend_map` 的 key |
| `lease_ttl_ms` | `300000` (5 分钟) | 读租约 TTL，覆盖单次 load 耗时 |

**不在 EC 配置中的内容：** 容量（`dram_size`、`ssd_size`）、驱逐水位（`evict_high`/`evict_low`）、传输协议（`protocol`）— 这些都是 memcache 基础设施层配置，由 `start_memcache.sh` + `mmc-local.conf` 管理。

## 六、不改动的范围

| 范围 | 状态 |
|------|------|
| vllm 上游（ECConnectorBase, ECConnectorFactory, Scheduler, ECConnectorModelRunnerMixin） | 不动 |
| memcache_hybrid 包 | 不动 |
| KV backend（memcache_backend.py, backend.py, backend/__init__.py） | 不动 |
| `pool_worker.py` / `pool_scheduler.py` | 不动 |
| `model_runner_v1.py` 除 1 行修改外 | 不动 |
| `start_memcache.sh` / `env_memcache.sh` | 不动 |

## 七、Encoder Output 兼容性

EC memcache 存储的是 `encoder_cache[mm_hash]` 对应的 tensor，来自 `model.embed_multimodal()` 返回的 encoder 输出，格式始终为 `[num_tokens, hidden_size]` 的 2D tensor。

**Qwen2.5-VL：** `hidden_size = vision_config.out_hidden_size`。无 deepstack，标准格式。

**Qwen3-VL deepstack：** `hidden_size = vision_config.out_hidden_size * (1 + len(deepstack_visual_indexes))`。多尺度特征沿 dim=-1 拼接，对外仍是单个 `[N, H]` tensor。deepstack 的分拆发生在 `_compute_deepstack_embeds()` 中（`torch.split` 沿 dim=-1 切开），这是 cache 下游行为，缓存层不感知。

**结论：所有 VL 模型端到端兼容，无需特殊判断。** cache 黑盒搬运 `[N, H]` tensor，H 的含义由模型自行解释。后续新模型只要 encoder 输出仍是 2D tensor 即自动兼容。

## 八、待验证项

1. `ki.size()` 返回类型 — 从 `pool_worker.py:1242` 使用 `ki.size()` 和 `if sizes and sizes > 0` 判断来看，可能是 list 或 int。如果是 list，取 `ki.size()[0]`。
2. A2 架构上动态创建的 encoder embedding tensor 是否需调用 `store.register_buffer(ptr, size)`。
3. GVA 被驱逐后 `batch_get_key_info` 的行为 — 是否返回 0 / 空，还是报错。

## 九、验证方式

1. **单元测试：** 测试 `ECMemcacheBackend` 的方法（mock `DistributedObjectStore`），参考 `tests/ut/distributed/ascend_store/test_backend.py` 的模式
2. **集成测试：** 启动 Memcache MetaService，启动 vllm 服务（`ec_role=ec_both`），使用同一张图片发送两次请求，验证第二次请求跳过 encoder 计算
3. **配置验证：** 确认 `ec_connector_extra_config.backend` 未设置或设为非 memcache 值时，`model_runner_v1.py` 走原路径（通过日志或 `is_ec_memcache_both` 为 False 判断）
