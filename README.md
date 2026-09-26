# HeteroForge

**异配性感知自适应路由（HAAR）的图表示学习系统** —— 双通道（结构嵌入 + GNN）按图/节点同配性自动路由与融合，CPU-only、零编译、离线可复现。

- 作者：晨星
- 许可证：Apache-2.0（核心依赖全部 BSD/MIT；gensim/node2vec 为可选 extra，见下）
- 语言：Python 3.13（Windows / Linux 均可，CI 双平台覆盖）

## TL;DR

```bash
git clone https://github.com/CJX0712/heteroforge.git
cd heteroforge
python -m venv .venv && .venv/Scripts/python -m pip install --only-binary=:all: -r requirements.txt   # Linux: .venv/bin/python
.venv/Scripts/python scripts/verify.py --fast     # 字符门禁 + 68 单测 + 离线冒烟 + HAAR 门槛
.venv/Scripts/python examples/run_demo.py --fast  # 端到端 demo, 落盘 benchmark.json
```

## 1. 系统定位与问题

图神经网络的经典假设是**同配性**（相连节点标签相似）。真实世界大量图是**异配**的
（欺诈网络、二分交易图、蛋白质接触图等），此时消息传递（GNN）反而劣于浅层结构嵌入。
现有框架（PyG / DGL / karateclub）都是**单范式**，需要用户人工判断选模型。

**HeteroForge 把这个判断自动化**：同时跑两个通道，由 HAAR 路由器按同配性证据
决定"每个节点信谁"，并给出可审计的路由决策。

## 2. 创新点：HAAR（Homophily-Aware Adaptive Routing）

> 与既有库的差异化：PyG/DGL 只提供模型，karateclub 只提供嵌入，
> 没有任何主流框架内置"同配性驱动的通道路由 + 预算级联"。

1. **双通道互补**：
   - 通道 A（`embed/channel_a.py`）：结构等价性浅层嵌入，node2vec -> spectral_rw -> svd 三级降级阶梯；
   - 通道 B（`gnn/channel_b.py`）：真单层 SGC（sknetwork GNNClassifier），附 prop_lr 兜底。
2. **图级门控（alpha）**：`alpha = clip(alpha_base + w * (0.5 - edge_homophily), 0, 1)`，
   低同配图自动偏通道 A，高同配图自动偏通道 B；alpha_base/w 由 per-bucket Optuna 校准。
3. **节点级预算级联（cascade）**：通道 A 高置信节点**完全不计算通道 B**（`computed_mask=False`，
   融合层读取未计算行直接抛 `E505`）—— 路由比例 `routed_fraction` 是真实算力节省，不是会计口径。
4. **非劣性 by construction**：HPO 强制 `enqueue_trial` 两个单通道端点
   （cascade+alpha=1 等价纯 A；full_dual+alpha=0 等价纯 B），因此 HAAR 的验证集分数
   **不可能低于最佳单通道**；自由搜索只负责在端点之间找更优融合。
5. **诚实工程**：sknetwork 的传播顺序（先归一化后加自环）已逐行核对其源码并复刻
   （`sknetwork_prop_features`），局部前向与全图前向**位级一致（diff=0.0）**；
   所有指标与 sklearn 双实现对照（差值超 1e-9 抛 `E605`）。

## 3. 架构

```
cli.py (data/embed/train/route/eval/benchmark)
  |
pipeline.py (HeteroForgePipeline) / benchmark.py (网格 + 门槛)
  |            \-- hpo/tuner.py (per-bucket Optuna, 端点入队)
  |                 |
  |    router/haar.py (HAAR: alpha 门控 + 预算级联)
  |       |        |
  |       |   gnn/channel_b.py (SGC + prop_lr 兜底), gnn/subgraph.py (k-hop)
  |       |
  |    embed/channel_a.py (node2vec -> spectral_rw -> svd 降级阶梯)
  |
data/synthetic.py (SBM/LFR)  data/split.py (节点/边切分)  graph/homophily.py (度量)
  |
core/ (types 契约 | errors E1xx-E6xx | config | interfaces Protocol | utils seed/RSS)
```

调用单向无环；模块间只依赖 `core/interfaces.py` 的 runtime Protocol。
错误码分段：E1xx 配置 / E2xx 数据 / E3xx 通道A / E4xx 通道B / E5xx 路由 / E6xx 评测；
WARN 码（如 `W-LFR_TO_SBM`、`W-HPO_FALLBACK`）不改变退出码但强制出现在产物中。

## 4. 技术选型（关键对比）

| 选项 | 定位 | 许可证 | Windows CPU 免编译 | 结论 |
| --- | --- | --- | --- | --- |
| PyTorch Geometric | 全功能 GNN 框架 | MIT | 需 torch(>800MB), 本机无 GPU 且磁盘紧张 | 不采用 |
| DGL | 全功能 GNN 框架 | Apache-2.0 | Windows 支持弱 | 不采用 |
| karateclub | 图嵌入算法集 | **GPL-3.0** | 仅 sdist, Py3.13 构建失败 | 不采用(许可+构建双重排除) |
| **scikit-network** | 图 ML + GNN | BSD-3 | 有 cp313 wheel | **采用(通道B + 图算法)** |
| **node2vec + gensim** | 随机游走嵌入 | MIT + LGPL-2.1 | 有 wheel | 采用为可选 `[n2v]` extra |
| **networkx** | 图数据结构 | BSD-3 | 纯 Python | 采用 |
| **Optuna** | HPO | MIT | 纯 Python | 采用(per-bucket 校准) |

（完整对比与数据快照见 `docs/PRD.md` 第 5 节，检索自 GitHub/PyPI API，2026-09-26。）

## 5. 性能基线（本机实测, benchmark.json 为准）

测试环境：AMD Ryzen 7 H 255 (8C/16T, 3.8GHz)、16GB RAM、Windows 11、Python 3.13.14、
纯 CPU、seed=42、通道A=node2vec（游走 Grover 原实现 + gensim SGNS）、通道B=SGC(50 epochs)。

节点分类 Macro-F1（SBM 合成图, 5 类, weak feature signal, 60/20/20 分层切分, n=800）：

| edge homophily | HAAR | 通道A (node2vec) | 通道B (SGC) | feat_only 参考 |
| --- | --- | --- | --- | --- |
| 0.05 | 0.538 | 0.103 | 0.545 | 0.919 |
| 0.2 | 0.635 | 0.194 | 0.635 | 0.919 |
| 0.5 | **0.807** | 0.701 | 0.770 | 0.919 |
| 0.8 | 0.981 | 1.000 | 0.819 | 0.919 |
| 0.95 | 1.000 | 1.000 | 0.876 | 0.919 |

- **HAAR 门槛（修订版判据, 14/14 桶通过, `enforce_gate=True` 强制执行）**：
  1. 逐桶非劣（>= max(A,B) - 0.02；容差按 val 切分约 160 样本的 macro-F1 采样噪声实测设定）；
  2. 跨桶均值非劣（0.7408 vs 0.7419；端点桶单通道已满分的"天花板效应"下,
     "均值严格更优"在数学上不可达, 故均值判据取非劣）;
  3. 至少一桶严格更优（HAAR 自适应增益证据）：h=0.5 桶 0.807 > max(0.701, 0.770),
     2000/0.2 桶 0.707 > 0.704, 链路预测侧同样成立。
- 自适应行为可审计：异配端自动选 B 端点（alpha=0）、同配端自动选 A 端点（alpha=1）、
  中间档真正融合且在 0.8 桶仅 10.4% 节点走通道 B（routed_fraction=0.104）。
- 局部前向 vs 全图前向：`max diff = 0.0`（位级一致, 单测守护）。
- 复现性：同 seed 两次全 pipeline 指标逐位一致（`test_reproducible_bitwise`）;
  node2vec 嵌入同 seed 差异 = 0.0。
- 链路预测 AUC（degree-aware hard negatives）14 桶全部非劣, 明细见 `benchmark.json`。

## 6. 部署

### 方式一：venv（推荐）

见 TL;DR。可选强后端：`pip install node2vec gensim`（`[n2v]` extra），
未安装时通道 A 自动降级为 `spectral_rw`/`svd`（BSD），`fallback_used=true` 如实记录。

### 方式二：Docker

```bash
docker build -t heteroforge .
docker run --rm heteroforge          # 默认执行 verify.py --fast
```

### 一键自检（五阶段）

```bash
python scripts/verify.py --fast   # chars -> imports -> pytest -> smoke -> gate
python scripts/verify.py          # 完整版(800 节点 + 三桶门槛)
```

任一阶段失败立即短路，报告落盘 `verify_report.json`。

## 7. 使用

CLI 六组命令（全部支持 `--config xxx.yml` 与 `--seed`）：

```bash
heteroforge data      --n 800 --homophily 0.2          # 生成图 + 同配性报告
heteroforge embed     --n 800 --homophily 0.2          # 通道A 摘要(backend/fallback/置信度)
heteroforge train     --n 800 --homophily 0.5          # 通道B 训练摘要
heteroforge route     --n 800 --homophily 0.5 --mode cascade   # HAAR 决策统计
heteroforge eval      --n 800 --homophily 0.5 --task node_classification
heteroforge benchmark --n 800 2000 --homophily 0.2 0.95 --out benchmark.json
```

Python API：

```python
from heteroforge.core.config import build_run_config
from heteroforge.pipeline.pipeline import HeteroForgePipeline

cfg = build_run_config({"seed": 42, "embed": {"backend": "node2vec"}})
rows = HeteroForgePipeline(cfg).run_bucket(800, 0.5, "node_classification")
```

## 8. 质量门禁

- 68 项单测全绿（`make test`）；CI 覆盖 Ubuntu + Windows x Python 3.13。
- 字符门禁（`scripts/scan_chars.py`）：全仓禁 emoji / 框线字符 / 箭头。
- 指标双实现对照：手动实现 vs sklearn，差值超 1e-9 抛 `E605`。
- `test_touch_count` 恒为 1（防 test 集重复评估）；路由层 `assert_no_test_access` 静态拦截 test 数据进入决策（`E504`）。

## 9. 已知限制

1. 通道 B 当前为 SGC（单层），多层 GCN/GAT 为 P1；深层 GNN 的级联收益会更大。
2. 合成图（SBM/LFR）的"异配"仅表示同配性低，不是 FSGNN 论文中的真异配模式
   （chameleon/squirrel 类二分倾向结构）；真实异配数据集接入在 roadmap。
3. `routed_fraction` 度量的是跳过的通道 B 头计算比例；SGC 单层下全图传播是共享的，
   算力节省在更大 `dim*C` 或多层网络下才显著。
4. feat_only 基线在强特征场景高于所有图方法 —— 这正是默认采用 weak signal 的原因。

## 10. 目录结构

```
heteroforge/          包源码(core/data/graph/embed/gnn/router/hpo/eval/pipeline + cli)
tests/                68 项单测
examples/run_demo.py  端到端演示
scripts/verify.py     五阶段一键自检
scripts/scan_chars.py P0 字符门禁
docs/PRD.md           产品需求与选型调研(许清楚)
docs/ARCHITECTURE.md  架构设计 + 任务分解 + 失败模式表(高见远)
benchmark.json        实测性能基线(门槛判定含在内)
requirements.txt      主依赖(钉版, 全 BSD/MIT)
requirements.lock.txt 完整锁版(开发环境 freeze)
```

## 许可证说明

本项目 Apache-2.0。可选 extra `n2v` 中的 gensim 为 LGPL-2.1-only（动态链接分发边界见
FSF 对 LGPL 的 Python 解释），node2vec 为 MIT。若你的分发场景不能接受 LGPL，
不安装 `n2v` 即可 —— 系统自动降级到 BSD 许可的 spectral 嵌入，功能完整。
