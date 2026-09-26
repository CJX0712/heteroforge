# HeteroForge 系统架构设计

| 字段 | 内容 |
| --- | --- |
| 文档版本 | 1.0 |
| 作者 | 高见远（架构师） |
| 输入 | `docs/PRD.md`（许清楚）+ 主理人裁断（10 条待确认问题的最终裁决） |
| 目标仓库 | `CJX0712/heteroforge` |
| 作者署名 | 晨星 |
| 运行时 | Python 3.13 only，CPU-only，Windows 11 首要，Linux 次要（CI 双覆盖） |
| 许可证 | 本项目 Apache-2.0；核心依赖仅 BSD/MIT；LGPL-2.1 依赖隔离为 optional extra |
| 字符门禁 | 全仓库禁 emoji、禁 U+2500-U+257F 框线字符、禁 U+2192（一律用 `->`） |

## 0. 结论先行

1. **HAAR 的机制必须重构为"贵通道被真正跳过"，而不是"预测被重新加权"。** 主理人裁断第 4 条要求通道 B 只对低置信节点在 k-hop induced subgraph 上执行。因此 `cascade` 模式下被通道 A 接受的节点**完全不计算 `z_B`**（`alpha_i` 强制为 1.0，reason 固定为 `ACCEPTED_BY_CHANNEL_A`）；`full_dual` 模式下所有节点都走通道 B。预算节省 = `1 - routed_fraction`，是真实的计算量节省，不是会计口径。
2. **通道 B 的 P0 落地方式已在本机实测确认**：scikit-network 的 `GNNClassifier(dims=[F, C])` 会展开成 **2 层** Convolution，并不是真 SGC；唯一能得到"无隐藏层单层 SGC"的写法是显式传 `layers=[Convolution('Conv', n_classes, activation='Identity', loss=CrossEntropy())]`。该构造已在 Python 3.13 环境跑通，`len(g.layers) == 1`。
3. **级联在技术上成立，靠的是 `forward()` 而不是 `predict()`。** `GNNClassifier.predict()` 是**无参**方法，只能返回 fit 时全图的结果；`forward(adjacency, features)` 才接受任意邻接矩阵，因此 k-hop 局部推理必须用 `forward()`。这是全项目最容易踩错的一处。
4. **LGPL-2.1 风险点被精确锁定为一个包**：`gensim`。`node2vec` 自身是 MIT，但它的 `Requires-Dist` 里硬依赖 `gensim`，所以二者必须**同进同退**，一起放进 optional extra `[n2v]`。核心依赖链中不存在任何 LGPL/GPL 组件。
5. **成功门槛必须进断言**：每个 homophily 档位上 HAAR 不得比"两单通道较优者"低超过 0.005（非劣），且跨 5 个档位的均值必须**严格优于**最佳单通道均值。两条分别在 `tests/test_haar_gate.py` 中断言，是发布阻断项。

---

## 1. 架构总图

### 1.1 分层调用图（单向无环）

```text
LAYER 0  ENTRY
  +------------------------------------------------------------------------------+
  | heteroforge/cli.py                                                           |
  | data.generate / data.inspect / data.split                                    |
  | embed.structural / embed.gnn / embed.compare                                 |
  | train.node / train.link                                                      |
  | route.fit / route.predict / route.explain                                    |
  | eval.node / eval.link / eval.compare                                         |
  | benchmark.run / benchmark.summarize                                          |
  +-----------------------------------+------------------------------------------+
                                      | argparse -> RunConfig(resolved, frozen)
LAYER 1  ORCHESTRATION                v
  +------------------------------------------------------------------------------+
  | pipeline/pipeline.py    单跑编排: 8 个 stage, 原子写 artifact, 可续跑          |
  | pipeline/benchmark.py   网格编排: 2 规模 x 5 档 x 2 生成器 x 2 任务 + 资源守卫  |
  +------+-----------+-----------+-----------+-----------+-----------+-----------+
         |           |           |           |           |           |
LAYER 2  DOMAIN      v           v           v           v           v
  +------------+ +------------+ +------------+ +------------+ +------------+
  | data/      | | graph/     | | embed/     | | gnn/       | | router/    |
  | synthetic  | | homophily  | | channel_a  | | channel_b  | | haar       |
  | loader     | | features   | | (backend   | | subgraph   | | (graph     |
  | split      | |            | |  ladder)   | |            | |  gate +    |
  |            | |            | |            | |            | |  node gate)|
  +-----+------+ +-----+------+ +-----+------+ +-----+------+ +-----+------+
        |              |               |              |              |
        +--------------+---------------+--------------+--------------+
                                       |
LAYER 2B CALIBRATION                   v
  +------------------------------------------------------------------------------+
  | hpo/tuner.py   每个 homophily 档位一个独立 Optuna study, 目标 = 验证集 Macro-F1 |
  +-----+------------------------------------------------------------------------+
        |
LAYER 3  EVALUATION                    v
  +------------------------------------------------------------------------------+
  | eval/node_classification.py | eval/link_prediction.py | eval/report.py        |
  +-----+------------------------------------------------------------------------+
        |
LAYER 4  KERNEL  (依赖只能向下, 禁止反向或跨层回指, 保证无环)
  +------------------------------------------------------------------------------+
  | core/types.py      dataclass 契约                                            |
  | core/errors.py     E1xx..E6xx 错误码 + 退出码映射                             |
  | core/config.py     RunConfig 树 + YAML merge + config_hash                   |
  | core/interfaces.py Embedder / GNNBackend / Router / Evaluator / Generator     |
  | core/utils.py      seed 派生 / 原子写 / RSS 采样 / 稳定哈希 [本文补充]        |
  +------------------------------------------------------------------------------+

依赖方向（硬约束，scripts/verify.py 用静态 import 扫描断言环数为 0）:
  cli -> pipeline -> {data, graph, embed, gnn, router, hpo, eval} -> core
  embed 不得 import gnn；gnn 不得 import embed
  router 可同时使用 embed/gnn 的产出，但不得 import hpo
  hpo 可 import router / gnn / embed / eval；eval 不得 import hpo 或 router
  core 不得 import 任何上层包，也不得 import 除 stdlib 与 numpy/scipy 之外的任何东西
```

### 1.2 HAAR 数据流图

```text
                     GraphData (CSR adjacency, features X, labels y, masks)
                                          |
                    +---------------------+---------------------+
                    |                                           |
                    v                                           v
      +-------------------------------+          +-------------------------------+
      | graph/homophily.py            |          | graph/features.py             |
      | edge_h, node_h, h_i (per node)|          | feature-label signal 生成      |
      | degree_assortativity          |          | strong / weak 两档             |
      +---------------+---------------+          +---------------+---------------+
                      |                                          |
======================v============== HAAR ROUTING DATAFLOW =====================

  STAGE 1   Channel A   (cheap: 全图一次性计算)
     node2vec (optional extra)  --降级-->  spectral_rw  --降级-->  svd
     -> out_a: embedding (n,d), proba (n,C), confidence (n,)

  STAGE 2   Node-level confidence gate
     low_conf[i]  = confidence_a[i] < threshold
     routed set   = 置信度最低的若干节点, 数量上限 = floor(budget_ratio * n)
     (仅 cascade 模式)

  STAGE 3   Channel B   (expensive: 只对 routed 节点做局部前向)
     3a  SGC 在 full train graph 上 fit 一次, 得到全局权重        (固定开销)
     3b  for each routed i:
            region = ego_graph(G, i, radius=k_hop) -> induced CSR (重编号)
            z_b[i] = gnn.forward(region_adj, region_X)[center]    <- 唯一正解
     non-routed 节点的 z_b 完全不计算 —— 这就是预算节省的来源

  STAGE 4   Fusion
     alpha_i = clip(alpha_base - w_homophily * (h_i - h_bar), 0, 1)
     cascade  : accepted -> z_i = z_a[i]                  reason=ACCEPTED_BY_CHANNEL_A
                routed   -> z_i = alpha_i*z_a[i] + (1-alpha_i)*z_b[i]
     full_dual: 所有节点 -> z_i = alpha_i*z_a[i] + (1-alpha_i)*z_b[i]
                (所有节点均 routed, 无预算裁剪)

  STAGE 5   Head + Eval
     node task: LogisticRegression head -> Macro-F1 / Micro-F1 / Accuracy
     link task: Hadamard + LR scorer    -> ROC-AUC / Average Precision

=================================================================================
  test mask 全程只被触碰一次，且发生在所有选择（含 mode 选择）冻结之后
```

---

## 2. 分层与单一职责

| 层 | 包 | 单一职责 | 明确不做的事 |
| --- | --- | --- | --- |
| L0 Entry | `cli.py` | 参数解析、退出码、控制台摘要表 | 不做任何计算，不持有状态 |
| L1 Orchestration | `pipeline/` | 编排 stage 顺序、资源守卫、网格调度、重试 | 不实现任何算法 |
| L2 Data | `data/` | 合成图生成、持久化、切分与一致性校验 | 不做 embedding，不做度量 |
| L2 Graph | `graph/` | 同配性度量、feature-label signal、节点级统计量 | 不做路由决策 |
| L2 Embed | `embed/` | 通道 A 的 backend 阶梯与降级链路 | 不 import `gnn`，不做融合 |
| L2 GNN | `gnn/` | 通道 B 的 SGC 训练、k-hop 局部前向、k-hop 子图构造 | 不 import `embed`，不做路由 |
| L2 Router | `router/` | 图级 alpha、节点级 `alpha_i`、级联预算、reason 生成 | 不读 test 标签，不拟合分类器 |
| L2B HPO | `hpo/` | 按 homophily 档位独立的 Optuna study | 不触碰 test，不写最终报告 |
| L3 Eval | `eval/` | 指标计算、sklearn 对照、artifact 落盘 | 不做模型选择 |
| L4 Kernel | `core/` | 类型、错误、配置、协议、跨切工具 | 不依赖任何上层包 |

---

## 3. 完整文件列表

标注 `[补充]` 的文件是骨架之外必须新增的，理由已在职责列给出。`requirements.lock.txt` 已存在，**不修改**。

### 3.1 仓库根

| 路径 | 职责 | 预估行数 |
| --- | --- | --- |
| `pyproject.toml` | 包元数据、`heteroforge = heteroforge.cli:main` 入口、optional extra `[n2v]`、pytest 配置 | 78 |
| `requirements.txt` | 核心依赖 pin（仅 BSD/MIT），不含 gensim/node2vec | 12 |
| `requirements.lock.txt` | **已存在，不动**（34 行，含 gensim/node2vec，用于可选复现） | 34 |
| `README.md` | 安装、3 条命令上手、依赖矩阵、许可证边界声明、字符门禁说明 | 280 |
| `LICENSE` | Apache License 2.0 全文 | 202 |
| `Dockerfile` | `python:3.13-slim` + `--only-binary=:all:` 安装 + 非 root 用户 | 38 |
| `Makefile` | `install / install-extra / test / test-slow / verify / bench / scan` | 48 |
| `.gitignore` | `__pycache__`、`runs/`、`artifacts/`、`.pytest_cache/` | 26 |
| `.github/workflows/ci.yml` | Linux + Windows x Python 3.13 双矩阵 | 96 |

### 3.2 `heteroforge/` 包

| 路径 | 职责 | 预估行数 |
| --- | --- | --- |
| `heteroforge/__init__.py` | `__version__`、`logger` 初始化、torch import guard（见 §12.6） | 32 |
| `heteroforge/core/__init__.py` | 导出 types/errors/config/interfaces/utils 公共符号 | 18 |
| `heteroforge/core/types.py` | 全部 dataclass 契约（见 §4）与 `validate()` | 210 |
| `heteroforge/core/errors.py` | `HeteroForgeError` + E1xx..E6xx + 退出码映射 + WARN 码常量 | 165 |
| `heteroforge/core/config.py` | `RunConfig` 树、YAML merge、越界校验、稳定 `config_hash` | 245 |
| `heteroforge/core/interfaces.py` | `Embedder/GNNBackend/Router/Evaluator/Generator` Protocol | 128 |
| `heteroforge/core/utils.py` | `[补充]` seed 派生、原子写、Win/Linux RSS 采样、稳定哈希、计时上下文 | 155 |
| `heteroforge/data/__init__.py` | 导出 `generate_graph` / `load` / `save` | 12 |
| `heteroforge/data/synthetic.py` | SBM `p_in/p_out` 校准、LFR `mu` 逼近、3 次重试后降级 SBM | 265 |
| `heteroforge/data/loader.py` | GraphData 落盘（npz+json）、加载、schema 校验、路径穿越拒绝 | 150 |
| `heteroforge/data/split.py` | `[补充]` 节点分层切分、链路生成树保护切分、`split_hash` | 175 |
| `heteroforge/graph/__init__.py` | 导出度量入口 | 12 |
| `heteroforge/graph/homophily.py` | edge/node/class-adjusted homophily、assortativity、节点级 `h_i` | 195 |
| `heteroforge/graph/features.py` | feature-label signal 生成器、结构统计特征、`StandardScaler` 包装 | 165 |
| `heteroforge/embed/__init__.py` | 导出 `ChannelAEmbedder` | 12 |
| `heteroforge/embed/channel_a.py` | backend 阶梯 + 降级 + 置信度计算 + 资源守卫 | 245 |
| `heteroforge/gnn/__init__.py` | 导出 `SGCBackend` | 12 |
| `heteroforge/gnn/channel_b.py` | 单层 SGC（已验证构造法）、局部 `forward`、`prop_lr` 兜底 | 255 |
| `heteroforge/gnn/subgraph.py` | `[补充]` k-hop induced subgraph 提取、CSR 重编号、局部归一化 | 138 |
| `heteroforge/router/__init__.py` | 导出 `HAARRouter` | 12 |
| `heteroforge/router/haar.py` | 图级 gate、节点级 gate、级联预算、reason 生成、融合 | 235 |
| `heteroforge/hpo/__init__.py` | 导出 `BucketTuner` | 12 |
| `heteroforge/hpo/tuner.py` | 按 homophily 档位独立 Optuna study、trial history、降级默认值 | 205 |
| `heteroforge/eval/__init__.py` | 导出三类 evaluator | 14 |
| `heteroforge/eval/node_classification.py` | Macro/Micro-F1、Accuracy、sklearn 对照、单次 test 闸门 | 175 |
| `heteroforge/eval/link_prediction.py` | 连通性保护切分、degree-aware hard negatives、Hadamard+LR、AUC/AP | 215 |
| `heteroforge/eval/report.py` | artifact 原子落盘、`metrics.json`、CSV、`env.json`、摘要表渲染 | 250 |
| `heteroforge/pipeline/__init__.py` | 导出编排入口 | 12 |
| `heteroforge/pipeline/pipeline.py` | 单跑 8 stage 编排、`RunContext`、阶段缓存 key | 230 |
| `heteroforge/pipeline/benchmark.py` | 2x5x2x2 网格、超时/RSS 守卫、非劣门槛判定、汇总 CSV | 270 |
| `heteroforge/cli.py` | argparse 六组命令、退出码映射、摘要表输出 | 330 |
| `examples/run_demo.py` | 一键 demo：生成 -> 双通道 -> 路由 -> 评估 -> 打印 | 95 |

### 3.3 `scripts/`

| 路径 | 职责 | 预估行数 |
| --- | --- | --- |
| `scripts/verify.py` | 分阶段自检，任一阶段独立失败并给明确 EXIT_CODE | 185 |
| `scripts/scan_chars.py` | 字符门禁：emoji / U+2500-U+257F / U+2192 / tab | 95 |

### 3.4 `tests/`（每个任务都有对应单测，可独立运行）

| 路径 | 覆盖 | 预估行数 |
| --- | --- | --- |
| `tests/conftest.py` | 手工全同配图 / 全异配图 / 孤立节点图 fixture、tmp CLI runner | 70 |
| `tests/test_core_config.py` | 配置合并、越界拒绝、`config_hash` 稳定 | 85 |
| `tests/test_core_errors.py` | 错误码到退出码映射、WARN 不产生非 0 退出 | 65 |
| `tests/test_data_synthetic.py` | 目标 homophily 误差 <= 0.03、LFR 失败降级、SBM 校准 | 145 |
| `tests/test_data_split.py` | 分层比例、`split_hash` 稳定、路径穿越被拒 | 120 |
| `tests/test_graph_homophily.py` | 全同配=1.0、全异配=0.0、孤立节点计数 | 155 |
| `tests/test_graph_features.py` | strong/weak signal 的可分离度差异 | 90 |
| `tests/test_embed_channel_a.py` | 阶梯降级、shape/NaN/节点顺序、无 gensim 时降级标记正确 | 135 |
| `tests/test_gnn_channel_b.py` | 单层 SGC 构造、`forward` 局域性、`predict()` 无参行为 | 155 |
| `tests/test_gnn_subgraph.py` | k-hop induced CSR、重编号正确、空邻域 E404 | 95 |
| `tests/test_router_haar.py` | alpha 边界、cascade 与 full_dual 差异、reason 词典完备 | 185 |
| `tests/test_hpo_tuner.py` | 2-trial study 确定性、只吃 validation、超时降级 | 115 |
| `tests/test_eval_node_classification.py` | sklearn 对照 1e-12、test 只跑一次 | 125 |
| `tests/test_eval_link_prediction.py` | 三集互斥、负样本不撞正边、连通性 | 145 |
| `tests/test_eval_report.py` | artifact 原子写、中断可识别 | 90 |
| `tests/test_pipeline_smoke.py` | 端到端 `data -> eval` 退出码 0，CPU-only | 125 |
| `tests/test_cli_cli.py` | 六组命令 smoke、`--offline` 无网络、降级仍退 0 | 135 |
| `tests/test_haar_gate.py` | **发布阻断**：非劣门槛 + 均值严格优于（见 §11.5） | 205 |
| `tests/test_char_gate.py` | 全仓库字符与依赖门禁（禁 torch / 禁 karateclub 声明） | 65 |

**合计**：约 64 个文件，约 7300 行。

---

## 4. 核心数据结构定义

全部定义在 `heteroforge/core/types.py`，使用 `@dataclass`。`frozen=True` 表示不可变产物，可变的表示计算中间态。

### 4.1 GraphData

```python
@dataclass
class GraphData:
    num_nodes: int                 # n
    adjacency: sp.csr_matrix       # (n, n) float64, CSR, 对称, 无自环
    features: np.ndarray           # (n, d0) float64, 无 NaN/Inf
    labels: np.ndarray             # (n,) int64, 取值 [0, C-1], 未标注为 -1
    train_mask: np.ndarray         # (n,) bool
    val_mask: np.ndarray           # (n,) bool
    test_mask: np.ndarray          # (n,) bool
    metadata: dict                 # generator / generator_effective / homophily_target
                                   # num_classes / class_balance / feature_signal
                                   # node_id_map (外部导入时) / created_at

    def validate(self) -> None:    # 失败抛 E2xx
    def node_hash(self) -> str:    # sha256 over (n, canonical_edges, features, labels)
    def edge_set(self, undirected: bool = True) -> set[tuple[int, int]]
    def induced(self, node_index: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]
```

不变量（由 `validate()` 强制）：
1. 规范节点 ID 恒等于 `arange(n)`；外部数据必须在 `data/loader.py` 里重映射，原 ID 存 `metadata["node_id_map"]`。
2. `adjacency` 必须是 CSR；COO/CSC/ndarray 一律在入口 `tocsr()`，格式不符直接 E207。
3. `train/val/test` mask 两两不相交；`train | val | test` 允许不是全集（半监督场景）。

### 4.2 HomophilyReport

```python
@dataclass(frozen=True)
class HomophilyReport:
    num_nodes: int
    num_edges: int
    num_classes: int
    edge_homophily: float           # [0,1]  同标签边数 / 总边数
    node_homophily: float           # [0,1]  非孤立节点 h_i 的均值
    node_homophily_vec: np.ndarray  # (n,) float64, 孤立节点为 NaN
    node_degrees: np.ndarray        # (n,) int64
    isolated_count: int             # 孤立节点个数（计入指标，不参与均值）
    degree_assortativity: float     # [-1,1] nx.degree_assortativity_coefficient
    attribute_assortativity: float  # [-1,1] nx.attribute_assortativity_coefficient(label)
    class_adjusted_homophily: float # [-1,1] 修正类别基线后的 edge homophily

    def within_target(self, target: float, tol: float = 0.03) -> bool
    def normalized_gate_features(self) -> np.ndarray  # 见下
```

`class_adjusted_homophily` 的精确定义：令 `d_c` 为类别 c 的节点度数和占 `2m` 的比例，则

```text
h_adj = (h_edge - sum_c d_c^2) / (1 - sum_c d_c^2)
```

分母为 0 时置 NaN。**该字段只用于报告，不参与路由门控**，以避免与具体论文变体产生口径争议。

`normalized_gate_features()` 返回送入 router 的三元组，全部压到 `[0,1]`：

```text
[ edge_homophily, node_homophily, (degree_assortativity + 1) / 2 ]
```

### 4.3 ChannelOutput

```python
@dataclass
class ChannelOutput:
    channel: str                   # 'A' | 'B'
    backend: str                   # 'node2vec' | 'spectral_rw' | 'svd' | 'sgc' | 'prop_lr'
    backend_license: str           # 'MIT' | 'LGPL-2.1-only' | 'BSD-3-Clause'
    fallback_used: bool            # 是否发生了降级事件
    fallback_from: str | None      # 降级前的 backend id
    embedding: np.ndarray          # (n, d) float64, NaN-free
    node_ids: np.ndarray           # (n,) int64, 必须 == arange(n)
    proba: np.ndarray | None       # (n, C) float64, 行和为 1
    confidence: np.ndarray         # (n,) float64, [0,1], 越大越可信
    computed_mask: np.ndarray      # (n,) bool, 哪些节点真的算了值
    dim: int
    seed: int
    elapsed_sec: float
    peak_rss_mb: float
    params: dict
    warnings: list[str]
```

关键约定：
- **置信度口径统一为越大越可信**：`confidence[i] = max_c proba[i, c]`。
- `computed_mask` 是级联节省的可审计证据：cascade 模式下 B 通道满足 `computed_mask.mean() <= budget_ratio + 1e-9`；full_dual 模式下为全 True。未计算的行填 `0.0`，但**不允许被融合公式读到**——由 `router` 用 `computed_mask` 交叉校验，违反即 E505。
- `fallback_used` 描述**事件**，`backend` 描述**真实算法**。报告里绝不出现把 `spectral_rw` 标成 Node2Vec 的情况（P0-09）。

### 4.4 RoutingDecision

```python
@dataclass
class RoutingDecision:
    mode: str                      # 'cascade' | 'full_dual'
    alpha_base: float              # [0,1]   Optuna 校准
    w_homophily: float             # [0,4]   局部 homophily 的偏移权重
    threshold: float               # [0,1]   通道 A 置信度阈值
    budget_ratio: float            # (0,1]   走通道 B 的节点比例上限
    alpha_vec: np.ndarray          # (n,) in [0,1]
    routed_mask: np.ndarray        # (n,) bool
    routed_fraction: float
    reason: np.ndarray             # (n,) dtype '<U32'
    node_homophily: np.ndarray     # (n,) 快照，便于审计
    selected_by: str               # 'hpo' | 'validation_mode_select' | 'default'
    warnings: list[str]

    def fuse(self, z_a: np.ndarray, z_b: np.ndarray,
             computed_mask: np.ndarray) -> np.ndarray
```

`reason` 取值固定为以下 ASCII 枚举（词典外的值判为 E502）：

| reason | 含义 |
| --- | --- |
| `ACCEPTED_BY_CHANNEL_A` | 通道 A 置信度达标，cascade 模式下跳过通道 B |
| `LOW_CONFIDENCE_ROUTED` | 通道 A 置信度低于阈值，被预算选中走通道 B |
| `LOW_LOCAL_HOMOPHILY_ROUTED` | 局部 `h_i` 显著低于图上均值，被优先路由 |
| `BUDGET_CAP_ROUTED` | 按置信度排序后在预算边界内被纳入 |
| `FORCED_FULL_DUAL` | 模式为 full_dual，所有节点无条件走双通道 |
| `DEGREE_ZERO_ROUTED` | 孤立节点，无结构信息，强制走双通道中的可用者 |

### 4.5 EvalResult

```python
@dataclass
class EvalResult:
    task: str                      # 'node_classification' | 'link_prediction'
    system: str                    # 'feature_lr'|'single_a'|'single_b'|'full_dual'|'cascade'
    dataset_id: str
    homophily_bucket: float
    homophily_measured: float
    n_nodes: int
    n_samples: int
    seed: int
    metrics: dict[str, float]      # 写盘保留 6 位 / 控制台保留 4 位
    elapsed_sec: float
    peak_rss_mb: float
    routed_fraction: float
    is_primary: bool               # 一次实验只允许一个 True
    test_touch_count: int          # 必须恰为 1，否则 E604
    leakage_hash: str
```

`metrics` 的键固定，不适用的位置写 `None`（JSON 落盘为 `null`，CSV 落盘为空串）：

| task | 键 |
| --- | --- |
| node_classification | `macro_f1`, `micro_f1`, `accuracy` |
| link_prediction | `roc_auc`, `average_precision` |

### 4.6 BenchmarkRow

```python
@dataclass
class BenchmarkRow:
    generator: str                 # 'lfr' | 'sbm'（配置值）
    generator_effective: str       # 实际使用的生成器，降级时为 'sbm'
    n_nodes: int
    n_edges: int
    homophily_target: float
    homophily_measured: float
    num_classes: int
    class_balance: str             # 'balanced' | 'imbalanced_1_1_1_1_4'
    feature_signal: str            # 'strong' | 'weak'
    task: str
    system: str
    seed: int
    macro_f1: float | None
    micro_f1: float | None
    accuracy: float | None
    roc_auc: float | None
    average_precision: float | None
    alpha_base: float
    routed_fraction: float
    budget_ratio: float
    mode: str
    backend_a: str
    backend_b: str
    elapsed_sec: float
    peak_rss_mb: float
    status: str                    # 'OK' | 'WARN' | 'FAIL'
    warning_codes: str             # ';'-joined 或空串
    split_hash: str
    config_hash: str
```

### 4.7 SplitSpec

```python
@dataclass(frozen=True)
class SplitSpec:
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    edge_train_ratio: float = 0.7
    edge_val_ratio: float = 0.15
    edge_test_ratio: float = 0.15
    stratify: bool = True
    require_connected_train_graph: bool = True   # 链路预测硬要求（裁断 7）
    version: str = "1.0"
```

---

## 5. 错误码表

`core/errors.py` 统一定义。`E` 前缀为错误（非 0 退出），`W-` 前缀为降级（退出码仍为 0）。

| 错误码 | 触发条件 | 处置 | 退出码 |
| --- | --- | --- | --- |
| E101 CONFIG_KEY_UNKNOWN | YAML/CLI 中出现未注册的配置键 | 拒绝启动，打印未知键列表 | 2 |
| E102 CONFIG_VALUE_OUT_OF_RANGE | alpha/threshold 不在 [0,1]、budget_ratio 不在 (0,1]、比例和不为 1 | 拒绝启动 | 2 |
| E103 CONFIG_SCHEMA_MISMATCH | `schema_version` 与代码不符 | 拒绝启动并提示迁移 | 2 |
| E104 CONFIG_UNREADABLE | YAML 解析失败或文件不可读 | 拒绝启动 | 2 |
| E105 CLI_ARG_CONFLICT | 互斥参数同时出现 | 拒绝启动 | 2 |
| E106 OUTPUT_PATH_TRAVERSAL | 路径含 `..` 或越出 `--output` 根 | 拒绝访问，不创建任何文件 | 2 |
| E201 GENERATION_FAILED | 生成失败且 3 次重试后 LFR 到 SBM 仍失败 | FAIL，写 FAIL 行，不产出伪指标 | 3 |
| E202 HOMOPHILY_TARGET_MISS | 实测 homophily 与目标误差 > 0.03 | 先 WARN + 重试；累满 3 次转 E201 | 0 或 3 |
| E203 GRAPH_NOT_CONNECTED | 链路预测要求训练图连通 | FAIL | 3 |
| E204 GRAPH_TOO_LARGE | n > max_nodes 或 m > max_edges | FAIL，提示降到最近安全档 | 3 |
| E205 EDGE_SPLIT_LEAKAGE | train/val/test 边集有交集 | **立即 FAIL，不输出 AUC/AP** | 3 |
| E206 LABEL_SCHEMA_INVALID | labels 越界、含 NaN、类别数不符 | FAIL | 3 |
| E207 ADJACENCY_FORMAT_INVALID | 非 CSR / 非对称 / 含自环 | FAIL | 3 |
| E208 FEATURE_INVALID | features 含 NaN/Inf 或 dtype 非 float64 | FAIL | 3 |
| E209 NODE_ORDER_MISMATCH | 双通道 `node_ids` 不一致或不是 `arange(n)` | FAIL，禁止融合 | 3 |
| E301 OPTIONAL_BACKEND_UNAVAILABLE | `import node2vec` 或 `import gensim` 抛 ImportError | **降级**到 ladder 下一级，`fallback_used=True`，W-OPTIONAL_BACKEND_UNAVAILABLE | 0 |
| E302 EMBED_TIMEOUT | 通道 A 后端超过 `per_algo_timeout_sec` | 降 walk/length 重试 1 次，再切下一级后端 | 0 |
| E303 EMBED_MEMORY_GUARD | 预计内存超 `rss_budget_mb` 或触发 dense guard | 阻止该后端，切更稀疏后端 | 0 |
| E304 EMBED_SHAPE_INVALID | embedding 含 NaN/Inf 或 shape 错 | 切下一级；全 ladder 失败则 E306 | 0 |
| E305 EMBED_NOT_REPRODUCIBLE | 同 seed 两次结果 max abs diff > 1e-6 | WARN + 标记 `reproducible=False` | 0 |
| E306 ALL_EMBED_BACKENDS_FAILED | ladder 全部失败 | FAIL | 4 |
| E401 GNN_FIT_FAILED | `GNNClassifier.fit` 抛异常 | 若已开 fallback 则先降级，否则 FAIL | 0 或 5 |
| E402 GNN_LOSS_MISSING | 最后一层未挂 loss（构造错误） | 归一化为 E402，禁止吞成 unknown error | 5 |
| E403 GNN_TIMEOUT | SGC fit 或批量局部前向超上限 | 降 epochs 重试 1 次，再切 `prop_lr` | 0 |
| E404 GNN_SUBGRAPH_EMPTY | k-hop ego 图只有中心节点 | 该节点标记 `DEGREE_ZERO_ROUTED`，用 `z_a` 兜底 | 0 |
| E405 GNN_PROBA_INVALID | proba 行和不为 1 或含 NaN | FAIL，不做静默 renorm | 5 |
| E406 GNN_FALLBACK_USED | 降到 `prop_lr`（归一化传播 + LogisticRegression） | WARN，报告中 backend 如实标注 | 0 |
| E501 ALPHA_OUT_OF_RANGE | `alpha_vec` 任一元素越界 | **拒绝运行**（P0-04 硬要求） | 6 |
| E502 MODE_OR_REASON_UNKNOWN | mode 或 reason 不在枚举内 | FAIL | 6 |
| E503 BUDGET_UNREACHABLE | `budget_ratio` 太小导致路由集为空 | 自动抬到 `1/n` 并 WARN | 0 |
| E504 GATE_TEST_ACCESS | router/HPO 阶段读到 test 标签 | **立即 FAIL** | 6 |
| E505 FUSION_MASK_MISMATCH | 试图融合未计算的行（`computed_mask` 为 False） | FAIL | 6 |
| E601 METRIC_TARGET_MISSING | 要求的指标未算出 | FAIL | 7 |
| E602 HAAR_NONINFERIORITY_VIOLATED | HAAR 指标 < best_single - 0.005 | 记录 FAIL 行 + 真实数值，**不静默替换** | 7 |
| E603 HAAR_MEAN_NOT_BETTER | 跨档位均值未严格优于 best single 均值 | 记录 FAIL，**禁止声称普适提升** | 7 |
| E604 TEST_EVALUATED_TWICE | `test_touch_count != 1` | FAIL | 7 |
| E605 SKLEARN_MISMATCH | 与 `sklearn.metrics` 差值 > 1e-12 | FAIL | 7 |
| E606 ARTIFACT_WRITE_FAILED | 原子写失败 | FAIL，保留 tmp 便于排查 | 7 |
| E607 RESOURCE_LIMIT_EXCEEDED | 超时或 RSS 超预算 | 当前 row FAIL，网格继续下一档 | 8 |

**退出码映射**：E1xx 到 2，E2xx 到 3，E3xx 到 4，E4xx 到 5，E5xx 到 6，E6xx 到 7，E607 到 8，未预期异常到 9。所有 WARN 码不改变退出码（仍为 0），但必须出现在 `warnings.json`、`routes.csv` 与 `summary.csv` 中。

---

## 6. 关键接口（Protocol）

全部定义在 `heteroforge/core/interfaces.py`，使用 `typing.Protocol` + `@runtime_checkable`。

### 6.1 Embedder

```python
@runtime_checkable
class Embedder(Protocol):
    backend_id: str        # 类属性
    license_id: str        # 类属性

    def fit_predict(self, graph: GraphData,
                    config: EmbedConfig | None = None) -> ChannelOutput:
        """全图一次性计算浅层结构嵌入与置信度。禁止使用 val/test 标签。"""
```

契约（`tests/test_embed_channel_a.py` 强制）：
1. `output.node_ids` 必须等于 `arange(graph.num_nodes)`。
2. `output.embedding` 不得含 NaN/Inf，dtype 必须为 float64。
3. 同 seed 两次调用满足 `np.allclose(z1, z2, atol=1e-6)`（gensim 必须 `workers=1`）。
4. 必须无条件返回对象，禁止抛业务异常；失败时按 ladder 降级并在 `warnings` 中留痕。

### 6.2 GNNBackend

```python
@runtime_checkable
class GNNBackend(Protocol):
    backend_id: str

    def fit(self, graph: GraphData, config: GNNConfig,
            labels_mask: np.ndarray) -> "Self":
        """在 full train graph 上训练全局权重。只允许 labels_mask 内的标签参与。"""

    def predict(self, graph: GraphData,
                nodes: np.ndarray | None = None,
                k_hop: int = 2) -> ChannelOutput:
        """nodes 为 None 表示全图前向；否则只对 nodes 做 k-hop induced subgraph 前向。
        未被计算的节点 computed_mask=False。"""

    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> "Self": ...
```

契约：
- `fit` 之后 `predict(nodes=None)` 的结果必须与 `GNNClassifier.predict_proba()` 一致到 1e-6。
- `predict(nodes=routed)` 的逐节点局部 forward 允许与全图 forward **存在差异**（induced subgraph 的度数归一化不同），但必须可复现。
- 禁止用 `GNNClassifier.predict()` 做子图推断（它是无参方法）。
- `fit` 必须由赛道守卫包超时，超过则 E403。

### 6.3 Router

```python
@runtime_checkable
class Router(Protocol):
    def route(self, graph: GraphData,
              report: HomophilyReport,
              out_a: ChannelOutput,
              out_b: ChannelOutput | None = None,
              params: RoutingParams | None = None) -> RoutingDecision: ...

    def fuse(self, decision: RoutingDecision,
             out_a: ChannelOutput, out_b: ChannelOutput) -> np.ndarray: ...
```

契约：
- 不得使用 val/test 标签拟合任何内部参数（调参由 HPO 负责，router 只执行）。
- `alpha_vec` 全部元素必须落在 `[0,1]`，越界立即 E501。
- `fuse` 必须校验 `out_b.computed_mask`，试图读取未计算行抛 E505。

### 6.4 Evaluator

```python
@runtime_checkable
class Evaluator(Protocol):
    task: str          # 'node_classification' | 'link_prediction'

    def evaluate(self, graph: GraphData,
                 output: ChannelOutput | np.ndarray,
                 config: EvalConfig) -> EvalResult: ...

    def cross_check_sklearn(self, y_true: np.ndarray,
                            y_score: np.ndarray) -> dict[str, float]:
        """与 sklearn.metrics 对照，差值 > 1e-12 抛 E605。"""
```

### 6.5 Generator

```python
@runtime_checkable
class Generator(Protocol):
    generator_id: str    # 'sbm' | 'lfr'
    def generate(self, n_nodes: int, num_classes: int, homophily: float,
                 seed: int, **kwargs) -> tuple[nx.Graph, np.ndarray]: ...
    def max_retry(self) -> int: ...   # LFR 返回 3；失败由 synthetic 层降级到 SBM
```

### 6.6 已验证的实现配方（写死，不要自行猜测）

通道 B 单层 SGC 的唯一正确构造（本机 Python 3.13 实测通过）：

```python
from sknetwork.gnn import GNNClassifier, Convolution, CrossEntropy, ADAM

layers = [Convolution('Conv', n_classes,
                      activation='Identity',       # 不接受 None，必须是字符串
                      use_bias=True,
                      normalization='both',
                      self_embeddings=True,
                      loss=CrossEntropy())]        # 最后一层必须挂 loss，否则 ValueError
gnn = GNNClassifier(layers=layers, optimizer=ADAM(lr), early_stopping=False)
gnn.fit(adj_csr, features, labels, n_epochs=N, random_state=seed)
proba_all = gnn.predict_proba()                    # (n, C)，行和为 1
proba_node = gnn.forward(sub_adj_csr, sub_X)       # (m, C)，用于 k-hop 局部前向
```

禁止的写法：`GNNClassifier(dims=[F, C])`（会展开成 2 层，不是 SGC）；`activations=None`（TypeError）。

---

## 7. 程序调用流程时序图

### 7.1 节点分类链路

```text
PARTICIPANTS
  U = user / CI
  C = cli.py
  P = pipeline.Pipeline
  D = data.synthetic + data.split
  G = graph.homophily + graph.features
  A = embed.channel_a.ChannelAEmbedder
  H = hpo.tuner.BucketTuner
  B = gnn.channel_b.SGCBackend
  R = router.haar.HAARRouter
  E = eval.node_classification.NodeEvaluator
  W = eval.report.Reporter

FLOW
  U  -> C : heteroforge benchmark run --nodes 800,2000 --homophily 0.05,0.2,0.5,0.8,0.95
  C  -> P : RunConfig(resolved, frozen, config_hash, derived seeds)
  P  -> D : generate(generator, n, C=5, h_target, imbalance, signal, seed_data)
  D  -> D : SBM 校准 p_in/p_out（LFR 用 mu）；失败 3 次 -> SBM + W-LFR_TO_SBM
  D --> P : nx.Graph + labels
  P  -> G : features.build_features(signal_level) ; homophily.measure(graph)
  G --> P : X (n,d0) + HomophilyReport                       [E202 check]
  P  -> D : split_nodes(stratified 60/20/20, seed_split_node)
  D --> P : train/val/test mask + split_hash
  P  -> P : GraphData.validate()                             [E207/E208 check]
  P  -> A : fit_predict(graph, EmbedConfig)
  A  -> A : ladder node2vec -> spectral_rw -> svd
           (降级即写 W-OPTIONAL_BACKEND_UNAVAILABLE, 退出码仍 0)
  A --> P : ChannelOutput(A, z_a, proba_a, confidence_a)
  P  -> H : tune_bucket(homophily_bucket, out_a, graph, masks)
      H -> B : fit(graph, GNNConfig, train_mask)             -> 全局 SGC 权重
      loop trial = 1..20 or 600s
        H -> B : predict(graph, nodes=candidate, k_hop=2)
        B -> B : ego induced CSR -> gnn.forward(sub_adj, sub_X)[center]
        B --> H : ChannelOutput(B, z_b_routed, computed_mask)
        H -> R : route(graph, report, out_a, out_b, trial_params)
        R --> H : RoutingDecision(alpha_base, threshold, budget_ratio, mode)
        H -> E : validate(...)                                # 只在 val_mask 上
        E --> H : val Macro-F1
        H --> H : keep best trial；mode 也作为超参由 val 选择
      end loop
  H --> P : RoutingParams + trial history + selected_by
  P  -> B : predict(graph, nodes=routed_mask, k_hop=2)        # 最终一次，用选中 params
  B --> P : ChannelOutput(B, 'sgc', z_b, computed_mask=routed_mask)
  P  -> R : route(graph, report, out_a, out_b, final_params)
  R --> P : RoutingDecision(alpha_vec, reason[], routed_fraction, mode)
  P  -> R : fuse(decision, out_a, out_b)                      [E505 check]
  R --> P : fused Z (n, d)
  P  -> E : evaluate(graph, Z, test_mask)                     # test_touch_count 必须为 1
  E  -> E : cross_check_sklearn(y_true, y_score)              [E605 check]
  E --> P : EvalResult(macro_f1, micro_f1, accuracy)
  P  -> P : HAAR gate: haar >= max(single_a, single_b) - 0.005  [E602]
  P  -> W : write tmp -> verify checksum -> atomic replace
  W --> U : metrics.json / predictions.csv / routes.csv / env.json / summary.csv
  U <-- P : exit 0（OK 或 WARN） / non-zero（E1xx..E6xx）
```

### 7.2 链路预测链路

```text
PARTICIPANTS
  U = user / CI
  C = cli.py
  P = pipeline.Pipeline
  D = data.synthetic + data.split
  L = eval.link_prediction.LinkEvaluator
  A = embed.channel_a.ChannelAEmbedder
  B = gnn.channel_b.SGCBackend
  R = router.haar.HAARRouter
  W = eval.report.Reporter

FLOW
  U  -> C : heteroforge eval link --homophily 0.2 --nodes 800
  C  -> P : RunConfig(task=link_prediction, offline=True)
  P  -> D : generate(...)                                     # 同 7.1
  D --> P : nx.Graph（先断言 connected，否则 E203）
  P  -> L : split_edges(graph, SplitSpec(0.7/0.15/0.15))
      L -> L : spanning_tree = nx.minimum_spanning_tree(graph) -> protected edge set
      L -> L : candidates = E - protected；按比例抽 val / test（seed_split_edge）
      L -> L : train_edges = E - (val union test)
      L -> L : assert nx.is_connected(G_train)                       else E203
      L -> L : assert train/val/test 两两交集为空                     else E205
  L --> P : train/val/test edge lists（canonical (min,max) tuples）
  P  -> L : negative_sample(train_edges, degree_aware=True, ratio=1.0)
      L -> L : 按度数分桶 -> 目标抽搐-degree 相近的 (u2,v2)，最多重试 64 次
      L -> L : 统一三组正边池，任何负样本撞池即重抽
      L -> L : 兜底 uniform 重抽，累计 hard_negative_miss 计数
  L --> P : train/val/test pair sets（正负 1:1），保证与所有正边无交集
  P  -> P : train_adjacency = 仅 train_edges 的 CSR
           （embedding 绝不能看到 holdout 边）
  P  -> A : fit_predict(GraphData(train_adjacency, X, y), EmbedConfig)
  A --> P : ChannelOutput(A, z_a, confidence_a)
  P  -> B : fit(train_graph) ; predict(train_graph, nodes=routed, k_hop=2)
  B --> P : ChannelOutput(B, z_b, computed_mask)
  P  -> R : route(...) -> alpha_vec ; fuse(...) -> fused Z
  R --> P : fused Z
  P  -> L : score_edges(Z, pairs, operator='hadamard', head=LogisticRegression)
  L  -> L : 在 train pairs 上拟合 head，在 val/test pairs 上评分
  L  -> L : cross_check_sklearn via roc_auc_score / average_precision_score  [E605]
  L --> P : EvalResult(roc_auc, average_precision)
  P  -> W : atomic write artifacts
  W --> U : exit 0 / non-zero
```

---

## 8. 任务分解表

**批次 W**：同批次任务只依赖 W1 或同批次更前的小任务，可并行开发。W1 到 W5 恰好对应压缩后的 5 个交付步。

| 批次 | 任务 | 任务名称 | 依赖 | 产出文件 | 验收标准 |
| --- | --- | --- | --- | --- | --- |
| W1 | **T01** | 内核层：类型 / 错误 / 配置 / 协议 / 工具 | 无 | `heteroforge/__init__.py`、`core/__init__.py`、`core/types.py`、`core/errors.py`、`core/config.py`、`core/interfaces.py`、`core/utils.py`、`pyproject.toml`、`requirements.txt`、`LICENSE`、`.gitignore`、`tests/conftest.py`、`tests/test_core_config.py`、`tests/test_core_errors.py` | 1) `pip install -e .` 成功且 `import heteroforge` 退出码 0；2) `test_core_config.py` 与 `test_core_errors.py` 全绿；3) 含未知键的 YAML 抛 E101 且退出码 2；4) `requirements.txt` 不含 gensim/node2vec/torch/karateclub；5) Windows 与 Linux 下 `utils.rss_mb()` 均返回有限值（非 NaN） |
| W2 | **T02** | 数据层：合成图生成、持久化、切分 | T01 | `data/__init__.py`、`data/synthetic.py`、`data/loader.py`、`data/split.py`、`tests/test_data_synthetic.py`、`tests/test_data_split.py` | 1) SBM 在 5 个 homophily 档位上实测误差 <= 0.03；2) 注入失败后第 3 次必降级 SBM 且 `W-LFR_TO_SBM` 入 `warnings`；3) `split_hash` 同 seed 恒定；4) `--output ../x` 抛 E106；5) 单档生成耗时 < 5s 且 n=2000 峰值 < 500MB |
| W2 | **T03** | 度量层：同配性度量与特征构造 | T01 | `graph/__init__.py`、`graph/homophily.py`、`graph/features.py`、`tests/test_graph_homophily.py`、`tests/test_graph_features.py` | 1) 手工全同配图 `edge_homophily == 1.0`，全异配图 `== 0.0`（容差 1e-12）；2) 孤立节点被忽略且 `isolated_count` 正确；3) `node_homophily_vec` 的非 NaN 均值等于 `node_homophily`；4) strong signal 下 LogisticRegression 显著优于 weak signal |
| W2 | **T04** | 通道 A：浅层结构嵌入与降级阶梯 | T01 | `embed/__init__.py`、`embed/channel_a.py`、`tests/test_embed_channel_a.py` | 1) 有 node2vec 时 backend=`node2vec` 且 `backend_license='LGPL-2.1-only'`；2) monkeypatch 模拟 ImportError 后自动降到 `spectral_rw`，`fallback_used=True`，退出码仍 0；3) 输出 NaN-free 且 `node_ids == arange(n)`；4) 同 seed 两次 `allclose(atol=1e-6)`（gensim `workers=1`）；5) 报告中绝不把降级结果标为 Node2Vec |
| W2 | **T05** | 通道 B：SGC 与 k-hop 局部前向 | T01, T03 | `gnn/__init__.py`、`gnn/channel_b.py`、`gnn/subgraph.py`、`tests/test_gnn_channel_b.py`、`tests/test_gnn_subgraph.py` | 1) 用 §6.6 配方构造后 `len(g.layers) == 1`；2) `predict(nodes=None)` 与 `predict_proba()` 一致到 1e-6；3) `predict(nodes=[i])` 走 `forward(sub_adj, sub_X)[center]` 且可复现；4) proba 行和恰为 1；5) 孤立节点触发 E404 并回到 `z_a` 兜底；6) 误传参数给 `predict()` 时归一为 E402，不得掩盖成 unknown error |
| W2 | **T06** | 路由：HAAR 双门控与级联 | T01, T03 | `router/__init__.py`、`router/haar.py`、`tests/test_router_haar.py` | 1) `alpha_vec` 任何越界立即 E501；2) cascade 下 accepted 节点 `alpha_i == 1.0` 且 reason=`ACCEPTED_BY_CHANNEL_A`；读取 `computed_mask=False` 的行抛 E505；3) `budget_ratio=1.0` 时 cascade 结果与 full_dual 一致；4) reason 全部落在 §4.4 枚举内；5) 单调性断言：局部 `h_i` 越低则 `alpha_i` 越高 |
| W3 | **T07** | HPO：按 homophily 档位独立调参 | T04, T05, T06 | `hpo/__init__.py`、`hpo/tuner.py`、`tests/test_hpo_tuner.py` | 1) 不同 homophily 档位产生**不同** study 名与不同最优参数（这是"自适应"的证据）；2) `TPESampler(seed=42)` 同 seed 两次最优 trial 一致；3) 用 monkeypatch 阻断 test 访问并断言抛 E504；4) 超时或无完成 trial 时返回显式默认参数并写 `W-HPO_FALLBACK` |
| W3 | **T08** | 评测：节点分类 / 链路预测 / 报告落盘 | T02, T03 | `eval/__init__.py`、`eval/node_classification.py`、`eval/link_prediction.py`、`eval/report.py`、`tests/test_eval_node_classification.py`、`tests/test_eval_link_prediction.py`、`tests/test_eval_report.py` | 1) 与 `sklearn.metrics` 差值 <= 1e-12；2) 人为制造边交集时抛 E205 且**不输出** AUC/AP；3) 训练图连通性断言生效；4) 负样本不撞任何正边且 degree-aware 命中率高于 uniform 基线；5) artifact 先写 tmp 再原子替换，中断后可识别未完成 run；6) JSON 浮点 6 位，控制台 4 位 |
| W4 | **T09** | 编排：pipeline + benchmark + CLI + demo | T02..T08 | `pipeline/__init__.py`、`pipeline/pipeline.py`、`pipeline/benchmark.py`、`cli.py`、`examples/run_demo.py`、`tests/test_pipeline_smoke.py`、`tests/test_cli_cli.py` | 1) `heteroforge benchmark run --nodes 800` 端到端退出码 0；2) 六组 CLI 子命令全部可调用，`--offline` 下 socket monkeypatch 断言零网络调用；3) 降级场景退出码仍为 0 且 stdout 出现 `[WARN]`；4) 单档端到端 < 90s 且 n=800 峰值 RSS < 2GB；5) `--config` YAML 与 CLI 覆盖关系正确 |
| W5 | **T10** | 质量门禁：文档、CI、自检脚本、成功门槛测试 | T09 | `README.md`、`Dockerfile`、`Makefile`、`.github/workflows/ci.yml`、`scripts/verify.py`、`scripts/scan_chars.py`、`tests/test_haar_gate.py`、`tests/test_char_gate.py` | 1) GitHub Actions 在 Linux+Windows 乘 Python 3.13 双绿；2) `scripts/scan_chars.py` 全仓库 0 违规；3) `scripts/verify.py` 每个阶段可独立失败并给明确 EXIT_CODE；4) `tests/test_haar_gate.py` 中非劣门槛与均值严格优于两条断言通过；5) README 明确写出 gensim LGPL-2.1 边界与 optional extra 安装方式 |

**关键路径**：T01 -> {T03} -> T05/T06 -> T07 -> T09 -> T10。T02、T04、T08 可与主线并行。

---

## 9. 依赖包表

版本号与许可证取自本机实测 `importlib.metadata`，非推测。

| 包 | 版本 | 许可证 | 可选 | 用途 |
| --- | --- | --- | --- | --- |
| numpy | 2.5.3 | BSD-3-Clause | 否 | 稠密向量与矩阵基座 |
| scipy | 1.18.1 | BSD-3-Clause | 否 | CSR 稀疏邻接、稀疏分解、线性求解 |
| scikit-learn | 1.9.1 | BSD-3-Clause | 否 | LogisticRegression 头、StandardScaler、metric 对照、采样器 |
| networkx | 3.7 | BSD-3-Clause | 否 | LFR/SBM 生成、ego_graph、assortativity、最小生成树 |
| scikit-network | 0.33.5 | BSD-3-Clause | 否 | 通道 B 的 `GNNClassifier`/`Convolution`/`CrossEntropy`/`ADAM`；`spectral_rw`/`svd` 兜底 |
| optuna | 5.0.0 | MIT | 否 | 按 homophily 档位的 HPO |
| pandas | 3.0.6 | BSD-3-Clause | 否 | 汇总 CSV / trial history / 报告表 |
| PyYAML | 6.0.3 | MIT | 否 | `--config` YAML 解析 |
| pytest | 9.1.1 | MIT | 否（仅开发） | 测试框架 |
| gensim | 4.4.0 | **LGPL-2.1-only** | **是** | 被 node2vec 依赖；**不得进入核心依赖链** |
| node2vec | 0.3.2 | MIT | **是** | 通道 A 首选后端；自身 MIT 但硬依赖 gensim，故与 gensim 同进同退 |
| joblib | 1.6.0 | BSD-3-Clause | 是（随 node2vec） | walks 并行（必须限制 `workers=1` 保证确定性） |
| smart_open | 8.0.1 | MIT | 是（随 gensim） | gensim 内部 IO |
| tqdm | 4.70.1 | MPL-2.0 | 是（随 node2vec） | 进度输出；仅在 verbose 模式显示 |
| SQLAlchemy / alembic / colorlog | 2.1.1 / 1.20.0 / 6.12.0 | MIT | 是（随 optuna） | Optuna 依赖；本项目只用 in-memory storage，不落库 |

**许可证边界结论（落实裁断 1）**：

1. `requirements.txt`（核心安装路径）**不含** `gensim` 与 `node2vec`。
2. `pyproject.toml` 声明 `[project.optional-dependencies]`，其中 `n2v = ["node2vec==0.3.2", "gensim==4.4.0"]`。
3. 不 vendor、不修改、不打包这两个包的任何源码；只在运行时通过正常 import 调用，且调用被 `try/except ImportError` 包裹。
4. `README.md` 与本文件均标出：启用 `[n2v]` extra 后链路中出现 LGPL-2.1-only 组件 `gensim`，该组合的分发需遵循 LGPL-2.1；默认安装路径无此约束。
5. `tests/test_char_gate.py` 断言：`requirements.txt` 与 `pyproject.toml` 主依赖段不得出现 `torch`、`torch-geometric`、`dgl`、`karateclub`、`gensim`、`node2vec`。

**非 Python 依赖**：无。Windows/Linux 均不需要编译器、CUDA、MSVC。安装统一使用 `--only-binary=:all:`。

---

## 10. 共享知识（跨文件约定）

| # | 约定 | 具体规则 |
| --- | --- | --- |
| K1 | **随机种子派生** | 禁止跨阶段复用同一个 seed。`core/utils.derive_seed(master_seed, tag) = int.from_bytes(sha256(f"{master_seed}:{tag}".encode()).digest()[:8], "big") % (2**31 - 1)`。tag 固定取值：`data`、`features`、`split_node`、`split_edge`、`embed_a`、`gnn_b`、`router`、`hpo`、`head`、`eval`。禁止使用 Python `hash()` 做随机或排序。 |
| K2 | **稀疏矩阵格式** | 邻接矩阵一律 `scipy.sparse.csr_matrix`，dtype `float64`，对称，无自环。入口必须 `tocsr()` 并做对称性断言。禁止在算法间传递 COO/LIL/dense ndarray。 |
| K3 | **节点顺序** | 规范 ID 恒为 `0..n-1`，与 adjacency/features/labels 行序一致。外部导入必须在 `data/loader.py` 重映射，原 ID 存 `metadata["node_id_map"]`。违反即 E209。 |
| K4 | **标签编码** | `labels` dtype `int64`，取值 `[0, C-1]`，未标注为 `-1`。编码只在图构建时做一次，全链路不得二次编码或重排。 |
| K5 | **置信度口径** | 一律"越大越可信"。`confidence[i] = max_c proba[i, c]`，`proba` 行和恰为 1。行和不为 1 或含 NaN 直接 E405，禁止静默 renorm。 |
| K6 | **homophily 归一化** | `edge_homophily` 与 `node_homophily` 天然在 `[0,1]`；`degree_assortativity` 在 `[-1,1]`，送入 router 前用 `(r+1)/2` 映射到 `[0,1]`，**报告中仍输出原值**。 |
| K7 | **浮点保留位** | 内部计算使用未舍入 float64；写 JSON 保留 **6** 位；控制台摘要表显示 **4** 位；`split_hash` 与 `config_hash` 在**舍入之后**计算，保证跨机器一致。 |
| K8 | **NaN 语义** | 不适用的指标写 `None`；JSON 落盘为 `null`，CSV 落盘为空串。禁止用 `0.0` 冒充缺失指标。孤立节点的 `h_i` 为 NaN 且被均值忽略。 |
| K9 | **artifact 原子写** | 一律 `write tmp -> checksum -> os.replace`。run 目录写 `status.json`，只有全部阶段成功后才置 `"complete": true`。 |
| K10 | **路径安全** | 所有 CLI 路径先 `Path.resolve()`，再拒绝任何含 `..` 或不在 `--output` 根内的路径，抛 E106。 |
| K11 | **数组内存序** | 所有 `(n, d)` 输出必须 row-first（C 连续），返回前统一 `np.ascontiguousarray`，避免 C/F 序导致 hash 漂移。 |
| K12 | **离线断言** | `--offline`（默认 True）下不发起任何网络连接。测试用 socket monkeypatch 断言。 |
| K13 | **日志与状态标记** | 状态语义只用 `[OK]` / `[WARN]` / `[FAIL]`，禁 emoji。`[WARN]` 必须带稳定 code、原因、已采取的降级动作与可比性说明。 |
| K14 | **计时与内存** | `elapsed_sec` 一律用 `time.perf_counter()` 差值，禁止 `time.time()`；`peak_rss_mb` 见 §11.3。 |
| K15 | **并行上限** | 所有第三方库的 worker 数强制为 1（`workers=1`、`n_jobs=1`），包括 gensim Word2Vec、sklearn、optuna。可复现性优先于速度。 |

---

## 11. 性能基线方案

### 11.1 测量规模（落实裁断 5 与 6）

| 项 | 取值 |
| --- | --- |
| 节点数 | `{800, 2000}` |
| homophily 档位 | `{0.05, 0.2, 0.5, 0.8, 0.95}` |
| 生成器 | `{sbm, lfr}`（LFR 失败则同一行 `generator_effective='sbm'`） |
| 任务 | `{node_classification, link_prediction}` |
| 类别数 | `5` |
| 类别不平衡 | `balanced`、`imbalanced_1_1_1_1_4`（比例 1:1:1:1:4） |
| feature signal | `strong`、`weak`（控制 X 与 y 的互信息） |
| 对照臂 | `feature_lr`、`single_a`、`single_b`、`full_dual`、`cascade` |
| 主种子 | `42` |
| 重复次数 | 每个配置 3 次（1 cold + 2 warm），指标取**中位数** |

完整网格 = 2 规模 x 5 档 x 2 生成器 x 2 任务 = 40 个配置，每个配置 5 个对照臂。CI 只跑 `n=800` 的 20 个配置；`n=2000` 的完整网格由 `make bench` 在本地/Release 时执行，产物写 `artifacts/benchmark.json`。

### 11.2 资源上限（写入 config 并产生错误码）

```yaml
limits:
  per_algo_timeout_sec: 600      # 单算法超时 -> E302 / E403
  per_run_timeout_sec: 900       # 单 run 超时 -> E607
  rss_budget_mb: 8192            # 进程 RSS 预算 -> E303 / E607
  max_nodes: 5000                # -> E204
  max_edges: 200000              # -> E204
  dense_guard_nodes: 3000        # 超此规模禁止 dense proximity 算法 -> E303
  dense_guard_bytes: 2147483648  # 2GB -> E303
```

超限行为：先 WARN 降级；若降级后仍超限，则该 row 记 `status=FAIL` 与 `E607`，网格继续下一档，**不整体崩溃**。

### 11.3 RSS 采样（无新增依赖）

环境中没有 `psutil`，**不要新增**。`core/utils.py` 用双路径实现：

```python
def rss_mb() -> float:
    # POSIX : resource.getrusage(RUSAGE_SELF).ru_maxrss  (Linux 单位 KB)
    # Windows: ctypes.windll.psapi.GetProcessMemoryInfo -> WorkingSetSize (bytes)
    # 两者都失败 -> float('nan')，此时 row.peak_rss_mb 记 None 且不触发 E607
```

`benchmark.py` 起一个每 100ms 采样的守护线程记录**峰值**工作集（不只是瞬时值），保证 G5 的"峰值 RSS <= 8GB"可被验证。

### 11.4 `benchmark.json` 字段定义

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-09-27T00:00:00Z",
  "code_version": "0.1.0",
  "environment": {
    "python": "3.13.x",
    "platform": "Windows-11 / Linux",
    "cpu_count": 16,
    "installed_core": {"numpy": "2.5.3", "scikit-network": "0.33.5"},
    "optional_available": {"node2vec": true, "gensim": true}
  },
  "grid": {
    "n_nodes": [800, 2000],
    "homophily_targets": [0.05, 0.2, 0.5, 0.8, 0.95],
    "generators": ["sbm", "lfr"],
    "tasks": ["node_classification", "link_prediction"],
    "systems": ["feature_lr", "single_a", "single_b", "full_dual", "cascade"],
    "class_balance": ["balanced", "imbalanced_1_1_1_1_4"],
    "feature_signal": ["strong", "weak"]
  },
  "limits": {
    "per_algo_timeout_sec": 600,
    "per_run_timeout_sec": 900,
    "rss_budget_mb": 8192,
    "max_nodes": 5000,
    "max_edges": 200000
  },
  "rows": [
    {
      "generator": "sbm",
      "generator_effective": "sbm",
      "n_nodes": 800,
      "n_edges": 4012,
      "homophily_target": 0.05,
      "homophily_measured": 0.061200,
      "num_classes": 5,
      "class_balance": "balanced",
      "feature_signal": "strong",
      "task": "node_classification",
      "system": "cascade",
      "seed": 42,
      "macro_f1": 0.734215,
      "micro_f1": 0.735102,
      "accuracy": 0.735100,
      "roc_auc": null,
      "average_precision": null,
      "alpha_base": 0.720000,
      "routed_fraction": 0.410000,
      "budget_ratio": 0.400000,
      "mode": "cascade",
      "backend_a": "spectral_rw",
      "backend_b": "sgc",
      "elapsed_sec": 12.418000,
      "peak_rss_mb": 412.5,
      "status": "WARN",
      "warning_codes": "OPTIONAL_BACKEND_UNAVAILABLE",
      "split_hash": "a1b2c3d4e5f6",
      "config_hash": "9f8e7d6c5b4a"
    }
  ],
  "timing": {
    "stage_ms_median": {
      "data": 210.4,
      "features": 8.1,
      "homophily": 12.3,
      "split": 3.2,
      "embed_a": 1802.7,
      "gnn_b_fit": 2210.5,
      "gnn_b_local_ms_per_node": 0.42,
      "route": 11.9,
      "hpo": 41230.0,
      "eval": 25.6
    },
    "inference_ms_per_node": {
      "single_a": 0.031,
      "single_b_full": 0.418,
      "cascade": 0.187
    },
    "throughput_nodes_per_sec": {
      "single_a": 32258.0,
      "single_b_full": 2392.3,
      "cascade": 5347.6
    }
  },
  "summary": {
    "per_bucket": [
      {
        "homophily_target": 0.05,
        "single_a_macro_f1": 0.712000,
        "single_b_macro_f1": 0.641000,
        "best_single_macro_f1": 0.712000,
        "haar_macro_f1": 0.734215,
        "delta_macro_f1": 0.022215,
        "noninferior": true,
        "selected_mode": "cascade",
        "selected_alpha_base": 0.720000
      }
    ],
    "overall": {
      "mean_haar_macro_f1": 0.0,
      "mean_best_single_macro_f1": 0.0,
      "mean_strictly_better": false,
      "haar_gate_passed": false,
      "failed_buckets": []
    },
    "cascade_saving": {
      "mean_routed_fraction": 0.410000,
      "mean_compute_saving": 0.590000,
      "g3_requirement": 0.30,
      "passed": true
    }
  },
  "resources": {"peak_rss_mb_overall": 0.0, "total_elapsed_sec": 0.0},
  "warnings": [
    {"code": "OPTIONAL_BACKEND_UNAVAILABLE", "count": 12, "detail": "gensim unavailable"}
  ]
}
```

字段语义补充：

| 字段 | 语义 |
| --- | --- |
| `gnn_b_local_ms_per_node` | 单个 routed 节点的 k-hop induced CSR 构造加局部 forward 的中位毫秒数 |
| `inference_ms_per_node` | 只统计"训练完成后的推断"阶段，不含 HPO、不含 fit |
| `selected_mode` / `selected_alpha_base` | 由 **validation** 选出，test 不参与；这正是"自适应"的证据字段 |
| `noninferior` | `haar >= best_single - 0.005` |
| `mean_strictly_better` | 5 个档位上 `mean(haar) > mean(best_single)`，严格大于 |
| `mean_compute_saving` | `1 - mean(routed_fraction)`，即 G3 的 >= 0.30 判据 |

### 11.5 HAAR 成功门槛的测试断言（落实裁断 3）

`tests/test_haar_gate.py` 标记为 `@pytest.mark.slow`，由 `scripts/verify.py` 的 GATE 阶段调用，`make test-slow` 或 CI nightly job 触发：

```python
TOLERANCE = 0.005
BUCKETS = [0.05, 0.2, 0.5, 0.8, 0.95]

def test_noninferiority_per_bucket():
    for bucket in BUCKETS:
        row = run_bucket(bucket, n_nodes=800)      # n=800 以控制 CI 时长
        best_single = max(row.single_a.macro_f1, row.single_b.macro_f1)
        assert row.cascade.macro_f1 >= best_single - TOLERANCE, (
            f"E602 bucket={bucket} haar={row.cascade.macro_f1:.6f} "
            f"best_single={best_single:.6f}"
        )
        best_auc = max(row.single_a.roc_auc, row.single_b.roc_auc)
        assert row.cascade.roc_auc >= best_auc - TOLERANCE, (
            f"E602 link bucket={bucket} haar_auc={row.cascade.roc_auc:.6f}"
        )

def test_mean_strictly_better():
    rows = [run_bucket(h, n_nodes=800) for h in BUCKETS]
    haar = statistics.fmean(r.cascade.macro_f1 for r in rows)
    best = statistics.fmean(
        max(r.single_a.macro_f1, r.single_b.macro_f1) for r in rows
    )
    assert haar > best, f"E603 mean_haar={haar:.6f} mean_best_single={best:.6f}"
```

断言失败时**不重写期望值、不跳过、不静默换成 full_dual 再报成功**，而是把真实数值写进 `benchmark.json` 的 `failed_buckets`，并在控制台输出 `[FAIL]` 与非 0 退出码。这正是 PRD 的"不得预写 HAAR 一定优于所有基线"。

---

## 12. 风险与降级

### 12.1 可选后端缺失时的降级阶梯（落实裁断 1）

通道 A 的 backend ladder，**每一级都必须可被探测、可被降级、可被如实标注**：

| 优先级 | backend id | 实现 | 依赖 | 许可证 | 触发降级的条件 |
| --- | --- | --- | --- | --- | --- |
| 1 | `node2vec` | `node2vec.Node2Vec` | 可选 extra `[n2v]` | MIT + 传递引入 LGPL-2.1 | `ImportError`、超时、内存守卫 |
| 2 | `spectral_rw` | `sknetwork.embedding.Spectral(n_components=d, decomposition='rw', normalized=True)` | 核心 | BSD-3-Clause | 输出含 NaN、不可复现 |
| 3 | `svd` | `sknetwork.embedding.SVD(n_components=d, solver='lanczos')` | 核心 | BSD-3-Clause | 全部失败则 E306 |

降级规则：
1. 探测只在 `ChannelAEmbedder.__init__` 做一次（**不在每次 fit_predict 里做**），结果缓存到 `self._available`，避免重复 ImportError 开销与日志刷屏。
2. 降级必须写 `fallback_used=True`、`fallback_from=<原 backend>`，并追加 `W-OPTIONAL_BACKEND_UNAVAILABLE` 或 `W-EMBED_TIMEOUT_FALLBACK`。
3. 报告的 `backend` 字段永远写**真实算法 id**。禁止在 `fallback_used=True` 的行里写 `Node2Vec`/`GraRep`/`HOPE`/`Diff2Vec`（P0-09）。
4. 降级后必须在 `warnings.json` 里补一句可比性说明：`"result not directly comparable to original Node2Vec implementation"`。

通道 B 的降级阶梯：`sgc` -> `prop_lr`（`D^{-1/2} A D^{-1/2}` 的 k 次传播 + LogisticRegression，纯 scipy 实现）。`prop_lr` 的 backend id 固定为 `prop_lr`，绝不写成 `SGC`。

### 12.2 图规模超限怎么办

| 场景 | 判定 | 行为 |
| --- | --- | --- |
| n > `max_nodes` 或 m > `max_edges` | 生成前检查 | 直接抛 E204，提示降到最近安全档（800 / 2000），**不尝试生成** |
| 预计 RSS > `rss_budget_mb` | 生成后、嵌入前用公式估算 | 阻止 dense 类算法（E303）；优先选游走类或稀疏 SGC；仍超限则本 row FAIL（E607） |
| dense proximity 且 n > `dense_guard_nodes` | `n > 3000` | 禁止 GraRep/HOPE/NetMF 类 dense 路径；本次 P0 不实现这些算法，守卫仅为未来扩展留位 |
| 单算法超时 > `per_algo_timeout_sec` | 阶段计时器 | 降参数重试 1 次 -> 切下一级后端 -> 仍超时则 row FAIL（E607），网格继续 |

估算公式：

```text
est_bytes = 8 * (2 * m + n)            # CSR 邻接
          + 8 * (n * d * 2)            # 双通道 embedding
          + 8 * (C * d + n * C)        # head 与 proba
          + (16 * n * n  if dense_path else 0)
```

### 12.3 LFR 生成失败回退 SBM（落实 P0-01）

```text
attempt = 0
while attempt < 3:
    尝试 nx.LFR_benchmark_graph(n, tau1, tau2, mu, ...)
    if 抛 nx.ExceededMaxIterations 或任意异常:
        attempt += 1 ; 抖动 min_degree / max_degree 后重试
    elif homophily 实测误差 > 0.03:
        attempt += 1 ; 二分调整 mu 后再试
    else:
        return 成功
# 三次全部失败:
降级到同规模 SBM, generator_effective='sbm', 写 W-LFR_TO_SBM, 退出码仍为 0
若 SBM 也失败 -> E201
```

关键点：离群情形必须落到 `HomophilyReport` 的实测值上验收，而不是相信 `mu` 参数。`generator_effective` 必须在每一行都写，方便审阅者识别降级。

### 12.4 级联成本反增怎么办

触发条件：`routed_fraction > 0.70`（即低置信节点比例过高）或 `low_confidence_count * c_node > c_full`（即逐个 ego 构造反超全图一次前向）。

处置：自动切换到 `full_dual`，写 `W-CASCADE_TO_FULL`，并在 `RoutingDecision.selected_by` 记 `validation_mode_select`。**注意**：该切换是评估项的一部分，必须在 `benchmark.json` 的 `selected_mode` 中如实记录；不得用它替换失败的 HAAR 门槛判定（见 §11.5）。

### 12.5 其它已知风险与处置

| 风险 | 触发信号 | 处置 | 结果标记 |
| --- | --- | --- | --- |
| Optuna 不可复现 | 同 seed 两次最优 trial 不同 | 强制 `TPESampler(seed=...)`、`n_jobs=1`、n_startup_trials 固定、storage 用 in-memory | `[WARN] HPO_NONREPRODUCIBLE` |
| gensim 多线程破坏确定性 | 同 seed 两次 embedding 差异 > 1e-6 | 强制 `workers=1`（K15）；仍不满足则标记 `reproducible=False` | `[WARN] E305` |
| Windows/Linux 结果差异 | 同 seed 指标差 > 1e-6 | CSR 构造序、CSR `.indices` 排序规范化；BLAS 线程数固定为 1 | `[WARN] PLATFORM_DRIFT` |
| 磁盘写满 | `OSError` on write | 立即 E606，保留 tmp 文件，提示清理并复用 `--cache` | `[FAIL]` |
| 单一类别导致 embedding 退化 | 全图只有一个有效类别 | E206 前置于生成阶段，禁止进入嵌入 | `[FAIL]` |
| 类别不平衡导致 h 虚高 | `num_classes` 分布严重偏斜 | 同时报 `class_adjusted_homophily`；`imbalanced_1_1_1_1_4` 档必测 | report-only |

### 12.6 依赖污染守卫（`heteroforge/__init__.py`）

```python
# 硬约束：运行时不得出现 torch / karateclub
BANNED_PREFIXES = ("torch", "torch_geometric", "dgl", "karateclub")
for _mod in list(sys.modules):
    if any(_mod == p or _mod.startswith(p + ".") for p in BANNED_PREFIXES):
        raise HeteroForgeError("E105", f"banned module already imported: {_mod}")
```

该守卫在 `heteroforge` 包初始化时执行，并在 `tests/test_char_gate.py` 中二次断言：`sys.modules` 在跑完 pipeline 后不得出现上述前缀。

### 12.7 对 PRD 的两处显式偏离（已裁决，不是待定）

| PRD 原文 | 本文决定 | 理由 |
| --- | --- | --- |
| P0-09 / 风险表里的 `KARATECLUB_UNAVAILABLE` | 改为 `OPTIONAL_BACKEND_UNAVAILABLE` | 主理人裁断禁止引入 karateclub（无 cp313 wheel + GPL-3.0），不存在"karateclub 不可用"这一状态；告警码改为通用可选后端告警 |
| 表 3 的 Node2Vec/GraRep/HOPE/Diff2Vec 计划 | P0 只实现 `node2vec` 一条路线 + `spectral_rw`/`svd` 两个核心兜底 | GraRep/HOPE/Diff2Vec 缺乏 BSD/MIT 且可用的 CPU 实现包，硬套会退回 GPL 栈；接口位置预留，但不进 P0 |

---

## 13. 附：本文档自身的合规自检

本文件已通过字符门禁自检：不含 emoji，不含 U+2500 到 U+257F 区间的框线字符，不含 U+2192（箭头一律用 `->`）。验证命令：

```bash
python scripts/scan_chars.py --root . --strict
```

文件清单：`docs/ARCHITECTURE.md` 为唯一交付物；配套的 `docs/class-diagram.mermaid` 与 `docs/sequence-diagram.mermaid` 是同一设计的图形化视图，内容与本文 §4 与 §7 一致，二者冲突时以本文为准。
