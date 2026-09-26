# HeteroForge 产品需求文档

## 0. 项目信息

| 字段 | 内容 |
| --- | --- |
| Language | 中文，关键术语、指标与 CLI 命令保留 English |
| Programming Language | Python 3.13，Windows 11，CPU-only |
| Project Name | `heteroforge` |
| 作者署名 | 晨星 |
| 目标仓库 | `CJX0712/heteroforge` |
| 选题来源 | 从 12 个候选域真随机抽中 B. GNN，种子 SHA-256 为 `58d56e97c1b1c656` |
| 核心任务 | 节点分类 Node Classification、链路预测 Link Prediction |
| 固定技术栈 | networkx、karateclub、scikit-network、Optuna、scikit-learn、numpy、scipy、pandas、pytest |
| 硬件约束 | AMD Ryzen 7 H 255、16GB RAM、无独立 GPU、C 盘剩余 18GB；禁止引入 torch 等大于 1GB 的依赖 |

**原始需求复述**：交付一个可在 Windows 11 和 Python 3.13 上离线运行的 CPU-only 图学习 CLI 系统，以合成同配图和异配图为主要评测载体，通过 HAAR 双通道自适应路由，在可控计算预算下完成节点分类与链路预测，并输出可复现、可比较的指标与实验产物。

## 1. 结论先行

1. HeteroForge 必须以 `networkx` 负责图生成与结构操作，以 `scikit-network` 负责轻量消息传递与稀疏图算法，以 `karateclub` 算法适配层负责浅层结构嵌入；不得把 PyG、DGL 或 CogDL 引入运行时依赖。
2. HAAR 的产品价值不是宣称单一模型始终最优，而是在不同 homophily 档位自动选择或融合结构通道与消息传递通道，并对低置信节点启用预算级联。
3. Python 3.13 是首要兼容风险：`scikit-network 0.33.5` 有 `cp313-win_amd64` wheel；`karateclub 1.3.3` 只有 source distribution。P0 必须提供适配器与不冒充算法等价的离线 fallback。
4. 成功门槛以可复现、无数据泄漏、CPU 资源可控为先；性能结论必须由 synthetic homophily grid 实测产生，不得预写“HAAR 一定优于所有基线”。

## 2. 产品定义

### 2.1 产品目标

| ID | 目标 | 可量化成功标准 |
| --- | --- | --- |
| G1 | 在纯 CPU 环境提供完整图学习闭环 | 在目标 Windows 11 / Python 3.13 环境中，`data -> embed -> train -> route -> eval` 冒烟流程退出码为 0，运行时不安装或导入 torch |
| G2 | 验证 HAAR 跨 homophily 的鲁棒性 | 在 6 个 homophily 档位上，HAAR 平均 Macro-F1 相对较优单通道的下降不超过 1 个百分点，并至少在 3 个档位取得正增益；未达到时必须如实标记实验未通过 |
| G3 | 降低昂贵通道计算量 | cascade 模式相对 full-dual 模式减少至少 30% 的昂贵通道目标节点数，同时 Macro-F1 下降不超过 1 个百分点；若门槛未达，自动回退 full-dual |
| G4 | 保证实验可复现 | 相同输入、配置与 `random_state` 重跑时，数据 split 哈希一致，核心指标绝对差小于等于 `1e-6` |
| G5 | 控制单机资源 | 默认 benchmark 在 16GB RAM 下峰值 RSS 小于等于 8GB；触发预计内存上限、超时或磁盘预算时必须提前拒绝或降级，不得导致系统级 OOM |

### 2.2 目标用户与场景

| 用户 | 核心问题 | 典型场景 | 价值 |
| --- | --- | --- | --- |
| 图学习研究者 | 同配与异配条件下模型表现难横向比较 | 生成 homophily grid，比较 shallow embedding、SGC 与 HAAR | 获得固定随机种子、统一 split 与可追溯 benchmark |
| 推荐工程师 | 用户物品图可能局部异配，单一消息传递策略不稳定 | 在候选边上执行 link prediction，检查不同路由权重 | 用 CPU 快速建立可解释 baseline |
| 风控工程师 | 欺诈节点常与正常节点连接，平均 homophily 掩盖局部风险 | 查看 node homophily、置信度与逐节点路由原因 | 找到低置信、局部异配节点并使用更稳健通道 |
| 无 GPU 的中小企业 | 无法承担 CUDA、torch 与大模型依赖 | 在普通 Windows 工作站离线训练、评估和导出报告 | 低安装成本、低资源占用、可离线交付 |

### 2.3 用户故事

1. 作为图学习研究者，我希望按目标 homophily 生成 LFR 与 SBM 图，以便系统比较同配到异配的性能变化。
2. 作为图学习研究者，我希望同时查看 edge homophily、node homophily 与 assortativity，以便避免单一度量误判图结构。
3. 作为图学习研究者，我希望对 Channel A 和 Channel B 使用相同 split 与 seed，以便公平比较两类归纳偏置。
4. 作为图学习研究者，我希望用 Optuna 校准融合权重 α 与级联阈值，以便减少人工试参。
5. 作为推荐工程师，我希望获得 AUC、AP 与负采样配置，以便评估候选连接排序能力。
6. 作为风控工程师，我希望导出每个节点的局部 homophily、置信度、路由通道与最终预测，以便审计高风险节点。
7. 作为无 GPU 企业用户，我希望一条 CLI 命令完成 benchmark，以便不搭建深度学习环境也能形成基线。
8. 作为离线环境用户，我希望禁用联网并使用本地合成数据与缓存依赖，以便在隔离网络中完成演示和验证。
9. 作为算法工程师，我希望通过统一 adapter 添加新的 shallow embedding，以便扩展算法而不改动评测协议。
10. 作为复现实验审阅者，我希望报告包含配置、seed、依赖版本、split 哈希与 artifact 路径，以便复核结果。
11. 作为运维人员，我希望任务在预计超内存或超时前给出 `[WARN]` 或 `[FAIL]`，以便保护共享工作站。

### 2.4 范围边界

| In Scope | Out of Scope |
| --- | --- |
| 合成图生成、同配性度量、双通道嵌入、HAAR 路由、HPO、节点分类、链路预测、CLI、离线 fallback、benchmark | CUDA/GPU、torch 运行时、分布式训练、Web UI、动态图、知识图谱推理、超大图在线服务、生产推荐系统集成 |

## 3. HAAR 产品机制

### 3.1 定义

**HAAR = Homophily-Aware Adaptive Routing，异配性感知自适应路由。**

- Channel A，Structural Equivalence：使用 Node2Vec、GraRep、HOPE 或 Diff2Vec 等浅层结构嵌入，重点捕获角色相似性、高阶邻近性或扩散邻域；预期在异配图上更稳健。
- Channel B，Neural Message Passing：使用 scikit-network 的 SGC 或 GNNClassifier，利用邻域传播；预期在同配图上更有效。
- Graph-level gate：根据 edge homophily、node homophily、assortativity 与验证集表现，由 Optuna 校准 α。
- Node-level gate：根据局部 node homophily、预测置信度与预算阈值生成 `alpha_i` 和 route reason。
- Cascade：先执行便宜通道；仅对低置信节点集合执行昂贵通道。若低置信节点比例超过配置阈值，切换为 full-dual，避免局部子图开销反而更高。

统一维度后的融合定义为：

`z_i = alpha_i * z_A_i + (1 - alpha_i) * z_B_i`

约束：`alpha_i` 必须位于 `[0, 1]`。较低局部 homophily 原则上提高 Channel A 权重，但最终值只能由 validation set 校准，test set 不得参与路由拟合。

### 3.2 数据流

```text
synthetic graph -> homophily metrics -> cheap channel -> confidence gate
                                      -> accepted nodes -> output
                                      -> low-confidence nodes -> expensive channel
                                                              -> calibrated fusion -> output
all predictions -> node/link evaluation -> report artifacts
```

## 4. 需求池

优先级定义：P0 为 Must Have，P1 为 Should Have，P2 为 Could Have。

### 4.1 P0 Must Have

| ID | 需求 | 验收标准 |
| --- | --- | --- |
| P0-01 | 合成图生成 | 必须支持 `networkx.LFR_benchmark_graph` 与 `networkx.stochastic_block_model`；接受 `nodes`、`classes`、目标 homophily、`random_state`；生成后实测 homophily 与目标误差小于等于 0.03，LFR 三次失败后降级 SBM 并记录原因 |
| P0-02 | 同配性度量 | 必须输出 edge homophily、node homophily、attribute assortativity coefficient 与节点级 `h_i`；在手工构造的全同配、全异配、孤立节点图上有单测，孤立节点处理策略固定为忽略并报告计数 |
| P0-03 | 双通道嵌入 | 必须产生维度一致的 `z_A`、`z_B` 和节点 ID 映射；Channel A 至少支持 Node2Vec 与一种矩阵类算法，Channel B 至少支持 SGC；输出含算法、维度、seed、耗时、峰值内存、backend |
| P0-04 | HAAR 路由 | 必须支持 graph-level α、node-level `alpha_i`、confidence threshold 与 cascade/full-dual 两种模式；所有节点均有 route、reason、confidence；`alpha_i` 越界时拒绝运行 |
| P0-05 | HPO | 必须使用 Optuna 对 α、置信阈值与至少两个通道参数调优；默认 20 trials 或 600 秒先到即停；sampler 固定 seed；只使用 validation objective，并保存 best trial 与 trial history |
| P0-06 | 节点分类评估 | 必须输出 Macro-F1、Micro-F1、Accuracy 与样本数；使用固定、分层 60/20/20 split；test 只评估一次；指标与 `sklearn.metrics` 对照误差小于等于 `1e-12` |
| P0-07 | 链路预测评估 | 必须输出 ROC-AUC 与 Average Precision；训练图、validation edge、test edge 互斥；负样本不得与任一正边重合；embedding 只基于训练邻接矩阵；默认正负比 1:1 |
| P0-08 | CLI | 必须提供 `heteroforge data/embed/train/route/eval/benchmark` 六组命令；支持 `--config`、`--seed`、`--output`、`--offline`；成功、降级、失败分别返回退出码 0、0、非 0，降级必须出 `[WARN]` |
| P0-09 | 离线兜底 | 必须在 `--offline` 下不发起网络请求；karateclub 不可导入或算法超时时，切换到明确标记为 `fallback_spectral` 的 scipy/scikit-learn 稀疏结构基线；报告不得把 fallback 标成 Node2Vec、GraRep、HOPE 或 Diff2Vec |
| P0-10 | 可复现产物 | 每次运行必须保存 resolved config、metrics JSON、predictions CSV、route decisions CSV、environment JSON 与 split hash；同 seed 重跑满足 G4 |
| P0-11 | CPU 资源守卫 | 运行前必须估算内存与算法复杂度；默认 RSS 预算 8GB、单算法超时 600 秒；超预算时降维、减少 walks 或切换算法，并将原计划与降级动作写入报告 |
| P0-12 | 测试与质量门禁 | pytest 必须覆盖生成、度量、split 防泄漏、路由边界、fallback 与 CLI smoke；核心测试全通过；源码与文档扫描不得出现 torch import 或大依赖声明 |

P0 共 12 条。

### 4.2 P1 Should Have

| ID | 需求 | 验收标准 |
| --- | --- | --- |
| P1-01 | 批量 benchmark | 一条命令覆盖 2 种生成器、6 个 homophily 档位、2 个任务，并输出汇总 CSV |
| P1-02 | 节点路由审计 | 可按节点 ID 查询 `h_i`、双通道置信度、`alpha_i`、预算决策与预测结果 |
| P1-03 | Embedding adapter | 新算法仅需实现 `fit_transform(graph, features, seed)` 协议；contract tests 自动验证 shape、NaN、节点顺序与复现性 |
| P1-04 | 缓存与续跑 | 以数据 hash、算法、参数、seed 组成 cache key；中断后跳过已完成且校验和一致的阶段 |
| P1-05 | 报告导出 | 输出 Markdown 与 CSV 摘要，包含单通道、full-dual、cascade 三组对照及资源消耗 |

### 4.3 P2 Could Have

| ID | 需求 | 验收标准 |
| --- | --- | --- |
| P2-01 | 外部数据导入 | 支持本地 edge list、node features、labels，完成 schema 校验且不联网 |
| P2-02 | 多 seed 置信区间 | 支持至少 5 个 seeds，输出均值、标准差与 95% bootstrap CI |
| P2-03 | 静态图表 | 从汇总 CSV 生成 homophily-metric 曲线 PNG；图表失败不影响核心 JSON/CSV |
| P2-04 | 实验注册表 | 以 run ID 索引配置、指标与 artifact，可比较两次运行差异 |

## 5. 竞品与选型调研

### 5.1 调研口径

- 检索日期：2026-09-26。
- 近 6 个月窗口：2026-03-26 至 2026-09-26。
- 关键词组 A：`graph learning framework GitHub stars latest commit license`。
- 关键词组 B：`PyPI Windows CPython 3.13 wheel torch dependency`。
- 关键词组 C：`heterophily homophily metrics H2GCN GPRGNN LINKX FSGNN`。
- Star 取 GitHub REST API `stargazers_count`；最近提交取默认分支 commits API。`pushed_at` 与默认分支提交不一致时单独说明。
- Windows CPU 免编译按目标环境 Python 3.13 判定，不以“理论支持 Windows”代替 wheel 证据。

### 5.2 表 1：图学习框架横评

| 框架 | 定位 | Star 量级 | 最近默认分支提交 | 近 6 个月活跃度 | 许可证 | Windows CPU / Py3.13 免编译 | 是否依赖 torch | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch Geometric | PyTorch 上的完整 GNN 研究框架 | 24.1k，精确值 24099 | 2026-09-01 | 活跃 | MIT | 核心包有通用 wheel，但完整使用需另装 PyTorch；不满足本项目体积约束 | 是 | 功能最全，但因 torch 与磁盘预算反选 |
| DGL | 多后端图深度学习与采样框架 | 14.3k，精确值 14286 | 2025-07-31 | 不活跃 | Apache-2.0 | 2.2.1 仅提供 CPython 3.8 至 3.12 的 Windows wheel，无 cp313 | 是，当前主后端 | Python 3.13 与体积约束均不满足 |
| karateclub | NetworkX 生态的浅层无监督图嵌入 | 2.3k，精确值 2286 | 2024-07-17 | 不活跃 | GPL-3.0 | 1.3.3 仅有 source distribution，Python 3.13 免编译不可确认，按不可用处理 | 否 | 算法覆盖契合 Channel A，但必须经 adapter、fallback 与许可证审查 |
| scikit-network | scipy sparse 上的图算法与轻量 GNN | 0.6k，精确值 633 | 2025-11-19 | 默认分支不活跃；仓库 `pushed_at` 为 2026-09-01 | BSD-3-Clause | 0.33.5 有 `cp313-cp313-win_amd64.whl` | 否 | 与 CPU、稀疏矩阵、Python 3.13 约束最匹配，作为 Channel B 首选 |
| networkx | 通用图建模、生成与算法基座 | 17.3k，精确值 17289 | 2026-09-25 | 活跃 | BSD-3-Clause | 3.7 有 `py3-none-any.whl` | 否 | 作为数据模型、LFR/SBM 生成与结构操作底座 |
| StellarGraph | TensorFlow/Keras 图机器学习库 | 3.1k，精确值 3059 | 2021-10-29 | 不活跃；仓库未 archived | Apache-2.0 | 1.2.1 要求 Python `>=3.6,<3.9`，不支持 3.13 | 否，依赖 TensorFlow | 版本与依赖栈均不适配，反选 |
| GraKeL | 图核 Graph Kernel 工具库 | 0.6k，精确值 647 | 2026-09-17 | 活跃 | BSD-3-Clause | 0.1.11 要求 Python `>=3.9,<3.13`，仅到 cp312 Windows wheel | 否 | 维护活跃但任务偏图级 kernel，且不支持 Python 3.13，反选 |
| CogDL | 多模型图深度学习研究工具箱 | 1.8k，精确值 1821 | 2023-11-08 | 不活跃 | MIT | 包本体为通用 wheel，但显式依赖 torch 且依赖链重 | 是 | 与 CPU-only 小体积目标冲突，反选 |

**表 1 选型结论**：最终选择 `karateclub + scikit-network` 承担双通道算法面，并以 `networkx` 承担图数据层；不选 PyG/DGL 的决定性原因是 torch 体积、Python 3.13 wheel 与纯 CPU 安装约束，而 karateclub 的兼容性和 GPL-3.0 风险必须由 adapter、fallback 与许可证决策化解。

### 5.3 表 2：同配性度量与异配图处理方法

| 方法 | 论文出处 | 核心思想 | CPU 可复现性 |
| --- | --- | --- | --- |
| Edge Homophily | Zhu et al., *Beyond Homophily in Graph Neural Networks*, NeurIPS 2020 | 同标签边数除以总边数；直观但受类别不平衡与高阶结构影响 | 高，单次边扫描，时间 O(m)，内存 O(1) 或 O(n) |
| Node Homophily | Zhu et al., NeurIPS 2020；后续 heterophily 评测广泛采用 | 对每个节点计算同标签邻居比例，再对节点平均；可直接形成节点级 gate 特征 | 高，时间 O(m)，需定义孤立节点策略 |
| Attribute Assortativity | Newman, *Mixing patterns in networks*, Physical Review E, 2003 | 基于 mixing matrix 的归一化同类连接相关性，较 edge ratio 更能修正类别基线 | 高，networkx 可直接计算；时间通常 O(m + c²) |
| H2GCN | Zhu et al., NeurIPS 2020 | 分离 ego 与 neighbor embedding，显式使用 1-hop/2-hop 邻域并组合中间表示 | 中，小图 CPU 可复现；2-hop 邻接可能膨胀，原实现栈不纳入运行依赖 |
| GPR-GNN | Chien et al., *Adaptive Universal Generalized PageRank Graph Neural Network*, ICLR 2021 | 学习多跳传播系数，将 GPR 滤波在同配和异配间自适应调整 | 中，稀疏传播可跑 CPU，但参考实现依赖 torch；本项目仅借鉴可学习 hop 权重思想 |
| LINKX | Lim et al., *Large Scale Learning on Non-Homophilous Graphs*, NeurIPS 2021 | 分别编码 adjacency 与 node features，再以 MLP 融合，避免强制邻域平滑 | 低至中，完整复现需要神经网络训练且 adjacency row 维度高；不纳入 P0 |
| FSGNN | Maurya et al., *Simplifying approach to node classification in Graph Neural Networks*, Journal of Computational Science, 2022 | 预计算多跳特征，对不同 hop 独立变换并学习 feature selection 权重 | 中，预计算适合 CPU，但参考神经实现依赖深度学习框架；本项目仅借鉴 hop/channel 选择思想 |

**表 2 选型结论**：P0 用 edge homophily、node homophily 与 assortativity 形成 HAAR 可解释路由特征，并吸收 H2GCN、GPR-GNN、LINKX、FSGNN 的分离与自适应思想，但不引入其 torch 参考实现。

### 5.4 表 3：图嵌入算法选型

复杂度记号：`n` 为节点数，`m` 为边数，`r` 为每节点 walk 数，`l` 为 walk 长度，`w` 为窗口，`k` 为 negative samples，`d` 为 embedding 维度，`K` 为最大阶数。下表为典型实现的数量级，具体值受稀疏化与 SVD 实现影响。

| 算法 | 捕获的结构性质 | 典型时间与内存特征 | 适用范围 |
| --- | --- | --- | --- |
| Node2Vec | 通过 `p/q` 偏置在 BFS-like 社区邻近与 DFS-like 结构角色之间权衡 | 采样约 O(nrl)，SGNS 训练约 O(nrlwkd)；内存随 walk corpus 与 nd 增长 | Channel A 默认候选；适合社区和角色混合、稀疏中小图 |
| DeepWalk | 无偏随机游走共现，偏重局部社区与短程邻近 | 采样约 O(nrl)，训练约 O(nrlwkd)；比 Node2Vec 参数更少 | Node2Vec 的 `p=q=1` 退化基线；适合作为超时降级 |
| GraRep | 显式分解 1 到 K 阶 transition/proximity，捕获全局高阶邻近 | `T^K` 可能稠密到 O(n²) 存储；逐阶 truncated SVD 成本高 | 小图、多尺度高阶邻近；必须受 dense-memory guard 限制 |
| HOPE | Katz、Rooted PageRank 等高阶 proximity，保留有向图非对称传递性 | 广义 SVD；显式 proximity 最坏 O(n²) 内存，dense 求解可达 O(n³) | 有向链路预测、小中型图；不适合无约束大图 |
| Diff2Vec | 扩散子图与 Euler tour 覆盖局部邻域，强调局部结构与图距离 | 序列生成随扩散规模和重复数近线性增长，后续训练同序列嵌入 | 较稠密图、局部角色；作为 Node2Vec 的结构通道备选 |
| NetMF | 显式构造并分解 DeepWalk 的隐式矩阵，提供确定性矩阵视角 | exact 版本可能 O(n²) 内存与高阶分解；large-window approximation 可降成本 | 研究对照与解释性分析；P0 不作为大图默认 |

**表 3 选型结论**：P0 优先 Node2Vec，超时降级 DeepWalk 或 `fallback_spectral`；Diff2Vec 作为稠密图备选，GraRep/HOPE 仅在规模守卫允许时启用，NetMF 保留为研究基线。

## 6. CLI 与输出设计

### 6.1 命令树

```text
heteroforge
+-- data
|   +-- generate       生成 LFR 或 SBM 合成图
|   +-- inspect        输出规模、连通性、类别分布与 homophily
|   +-- split          生成并固化 node/edge split
+-- embed
|   +-- structural     运行 Channel A
|   +-- gnn            运行 Channel B
|   +-- compare        校验节点顺序、维度与资源消耗
+-- train
|   +-- node           训练节点分类器
|   +-- link           训练链路预测 scorer
+-- route
|   +-- fit            Optuna 校准 alpha、alpha_i 规则与 threshold
|   +-- predict        执行 cascade 或 full-dual 路由
|   +-- explain        查询逐节点路由理由
+-- eval
|   +-- node           输出 Macro-F1、Micro-F1、Accuracy
|   +-- link           输出 AUC、AP
|   +-- compare        对比 single-A、single-B、full-dual、cascade
+-- benchmark
    +-- run            运行 synthetic homophily grid
    +-- summarize      汇总指标、耗时、RSS 与降级状态
```

### 6.2 关键命令示例

```bash
heteroforge data generate --model sbm --nodes 2000 --homophily 0.30 --seed 42 --output runs/h030/data
heteroforge embed structural --algo node2vec --input runs/h030/data --dim 64 --seed 42 --offline
heteroforge embed gnn --model sgc --input runs/h030/data --dim 64 --seed 42
heteroforge route fit --channel-a runs/h030/a.npy --channel-b runs/h030/b.npy --trials 20 --timeout 600
heteroforge train node --route runs/h030/route.json --seed 42
heteroforge eval node --run runs/h030
heteroforge benchmark run --generators lfr,sbm --homophily 0.05,0.15,0.30,0.50,0.70,0.90 --seed 42 --offline
```

### 6.3 控制台输出

因全文字符门禁禁止 emoji，状态语义统一使用 ASCII 标记 `[OK]`、`[WARN]`、`[FAIL]`。

```text
STATUS | STAGE       | METRIC          | VALUE   | TARGET     | ARTIFACT
[OK]   | data        | homophily_edge  | 0.3021  | 0.30+/-0.03| runs/h030/data.json
[OK]   | channel_b   | elapsed_sec     | 4.18    | <=600      | runs/h030/b.npy
[WARN] | channel_a   | backend         | fallback_spectral | karateclub unavailable | runs/h030/warnings.json
[OK]   | route       | routed_fraction | 0.41    | report-only | runs/h030/routes.csv
[OK]   | eval_node   | macro_f1        | 0.7342  | report-only | runs/h030/metrics.json
```

输出规则：

- stdout 只输出摘要表；详细日志写入 run directory。
- `[WARN]` 必须包含稳定的 warning code、原因、采取的降级与结果可比性说明。
- `[FAIL]` 必须给出非 0 exit code，不生成伪成功 metrics。
- 数值指标保留 4 位小数，原始全精度值保存在 JSON。
- 所有表格都必须显示 `seed`、backend、数据 hash 与配置 hash。

## 7. 评测方案

### 7.1 数据矩阵

| 维度 | 默认配置 |
| --- | --- |
| 生成器 | `LFR_benchmark_graph`、`stochastic_block_model` |
| Homophily 档位 | `0.05, 0.15, 0.30, 0.50, 0.70, 0.90` |
| 默认规模 | smoke: `n=500`；benchmark: `n=2000`；扩展档由资源守卫决定 |
| 类别数 | 默认 4，可配置 2 至 10 |
| Random state | 默认 42；数据、split、embedding、classifier、Optuna 分别派生固定子 seed |
| 特征 | 基于类别中心加固定噪声生成，同时报告 feature-label signal，避免把结构收益与特征强度混淆 |

LFR 使用 `mu` 作为目标 homophily 的近似控制量，生成后必须以实测指标验收；若误差超限则最多重试 3 次。SBM 通过校准 `p_in/p_out` 命中目标 homophily。任何生成失败或降级都进入 report。

### 7.2 节点分类协议

- Split：stratified train/validation/test 为 60%/20%/20%，固定 seed。
- Metrics：Macro-F1 为 HPO 主目标，同时报告 Micro-F1 与 Accuracy。
- Baselines：feature-only Logistic Regression、Channel A、Channel B、fixed-alpha 0.5、HAAR full-dual、HAAR cascade。
- 防泄漏：scaler、projection、classifier、α 与 threshold 只在 train/validation 拟合；test 在选择完成后只运行一次。
- 结果表必须同时给出性能、训练秒数、推理秒数、峰值 RSS 与昂贵通道路由比例。

### 7.3 链路预测协议

- 正边按 70%/15%/15% 切分，先固定必要生成树边以保持训练图可用，再抽 validation/test。
- 无向边统一规范为 `(min(u,v), max(u,v))`；三个 split 不重叠。
- 负样本从全图非边采样，默认正负比 1:1，且不得与任何正边重合。
- Embedding 仅使用 train graph；候选边特征默认采用 Hadamard product，并以 Logistic Regression 评分。
- Metrics：ROC-AUC 与 Average Precision，均保存未四舍五入原值。

### 7.4 HAAR 验证假设

| 假设 | 判据 | 未通过动作 |
| --- | --- | --- |
| H1: Channel A 在低 homophily 更稳健 | 在 0.05、0.15 档位至少一个 structural 方法优于 Channel B | 保留真实结果，重新检查特征信号与算法参数，不改 test 数据 |
| H2: Channel B 在高 homophily 更有效 | 在 0.70、0.90 档位 SGC/GNNClassifier 至少一项优于 Channel A | 检查传播深度与归一化，仍失败则标记假设不成立 |
| H3: HAAR 跨档位稳健 | 满足 G2 | 输出失败档位和 selected α，不声称普适提升 |
| H4: Cascade 节省计算 | 满足 G3 | 自动建议 full-dual，并报告级联无收益 |

## 8. 非功能要求

| 类别 | 要求 |
| --- | --- |
| 性能 | 默认算法单次超时 600 秒；默认进程 RSS 预算 8GB；dense proximity 算法默认 `n<=3000` |
| 可靠性 | 所有阶段原子写 artifact，先写临时文件，校验通过后替换；中断后可识别不完整 run |
| 可观测性 | 每阶段记录 start/end、elapsed、RSS、input/output hash、backend、warning code |
| 可移植性 | Windows 11 / CPython 3.13 为阻断性 CI 环境；不得要求 CUDA、MSVC 或本地编译器 |
| 可复现性 | 所有随机源显式 seed；禁止依赖 Python hash 随机顺序；节点索引映射必须固化 |
| 安全性 | CLI 路径限制在用户指定 output 下；拒绝 `..` 路径穿越；不下载任意代码或模型 |
| 许可证 | 发布前生成依赖许可证清单；karateclub 的 GPL-3.0 使用和分发方式必须经明确决策 |

## 9. 风险与降级策略

| 风险 | 触发信号 | 默认处置 | 结果标记 |
| --- | --- | --- | --- |
| torch/PyG/DGL 无法安装或超体积 | 依赖解析出现 torch，或下载预算超过 1GB | 拒绝安装；继续使用 scikit-network + scipy sparse；若 Channel B 不可用则降级为 normalized adjacency propagation + Logistic Regression | `[WARN] BACKEND_FALLBACK` |
| karateclub 在 Python 3.13 安装失败 | import smoke 失败、仅 sdist、编译链缺失 | 不现场编译；adapter 切换 `fallback_spectral`，保留相同 embedding contract，并注明不可与原算法直接等价比较 | `[WARN] KARATECLUB_UNAVAILABLE` |
| karateclub 算法超时 | 达到 600 秒或预测将超预算 | Node2Vec 降低 walks/length 后最多重试 1 次；仍超时则 DeepWalk；再失败则 `fallback_spectral` | `[WARN] EMBED_TIMEOUT_FALLBACK` |
| 内存不足 | 预计 RSS 大于 8GB，或 dense matrix 预计大于 2GB | 阻止 GraRep/HOPE/NetMF；优先随机游走或 sparse SGC；benchmark 图规模降到最近安全档 | `[WARN] MEMORY_GUARD` |
| LFR 不收敛 | NetworkX 抛出迭代失败或实测 homophily 超差 | 最多 3 次调整合法参数重试，随后以同规模 SBM 替代，并保留 generator 字段 | `[WARN] LFR_TO_SBM` |
| 节点级局部嵌入开销不降反升 | 低置信节点比例超过阈值或局部子图覆盖超过 70% 节点 | 改用 full-dual，避免重复构图 | `[WARN] CASCADE_TO_FULL` |
| 链路预测数据泄漏 | split 边交集非空、embedding 输入包含 holdout edge | 立即失败，不输出 AUC/AP | `[FAIL] EDGE_LEAKAGE` |
| HPO 过拟合或超时 | test 被访问、trial 超时、validation 波动过大 | test 访问即失败；超时返回 best completed trial；无完成 trial 则使用显式 default config | `[WARN] HPO_FALLBACK` 或 `[FAIL] TEST_LEAKAGE` |
| GPL-3.0 分发风险 | 计划复制、修改或打包 karateclub 代码 | 禁止直接 vendor；先确定项目许可证和调用边界 | `[FAIL] LICENSE_DECISION_REQUIRED` |

## 10. 待确认问题

1. 项目仓库计划采用哪种许可证？若使用 MIT/Apache-2.0，如何处理 karateclub 的 GPL-3.0 组合与分发边界？
2. Python 3.13 是否为唯一运行时，还是允许为 karateclub 提供隔离的 Python 3.11 sidecar environment？默认建议不引入 sidecar，使用 adapter fallback。
3. HAAR 的正式成功门槛是否采用 G2/G3，还是以“相对最佳单通道不劣于 1 个百分点”为唯一门槛？
4. Node-level cascade 对 Channel A 是否允许使用低置信节点的 k-hop induced subgraph，还是必须保持全图 embedding 的严格可比性？
5. benchmark 最大图规模、单次运行时长与峰值 RSS 的验收机型是否固定为当前 Ryzen 7 H 255 / 16GB？
6. 节点分类默认类别数、类别不平衡程度与 feature-label signal 应固定在哪些档位？
7. 链路预测是否要求训练图保持连通，以及负样本是否需要 degree-aware hard negatives？
8. HPO 是分别为每个 homophily 档位调参，还是只在中间档位调参后跨档位复用？前者上限更高，后者更能检验泛化。
9. `GNNClassifier` 与 SGC 的主次关系是否固定为 SGC P0、GNNClassifier P1，以降低 Python 3.13 兼容风险？
10. CI 是否只要求 Windows + Python 3.13，还是需要额外覆盖 Linux + Python 3.13 以验证跨平台行为？

## 11. 资料来源

### 11.1 框架维护与兼容性

- GitHub Repository REST API，访问日期 2026-09-26：`https://api.github.com/repos/pyg-team/pytorch_geometric`
- GitHub Repository REST API，访问日期 2026-09-26：`https://api.github.com/repos/dmlc/dgl`
- GitHub Repository REST API，访问日期 2026-09-26：`https://api.github.com/repos/benedekrozemberczki/karateclub`
- GitHub Repository REST API，访问日期 2026-09-26：`https://api.github.com/repos/sknetwork-team/scikit-network`
- GitHub Repository REST API，访问日期 2026-09-26：`https://api.github.com/repos/networkx/networkx`
- GitHub Repository REST API，访问日期 2026-09-26：`https://api.github.com/repos/stellargraph/stellargraph`
- GitHub Repository REST API，访问日期 2026-09-26：`https://api.github.com/repos/ysig/GraKeL`
- GitHub Repository REST API，访问日期 2026-09-26：`https://api.github.com/repos/THUDM/CogDL`
- PyPI JSON API，访问日期 2026-09-26：`https://pypi.org/pypi/torch-geometric/json`、`https://pypi.org/pypi/dgl/json`、`https://pypi.org/pypi/karateclub/json`、`https://pypi.org/pypi/scikit-network/json`、`https://pypi.org/pypi/networkx/json`、`https://pypi.org/pypi/stellargraph/json`、`https://pypi.org/pypi/grakel/json`、`https://pypi.org/pypi/cogdl/json`

### 11.2 方法与论文

- Zhu et al., *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs*, NeurIPS 2020：`https://proceedings.neurips.cc/paper/2020/file/58ae23d878a47004366189884c2f8440-Paper.pdf`
- Chien et al., *Adaptive Universal Generalized PageRank Graph Neural Network*, ICLR 2021 / arXiv:2006.07988：`https://arxiv.org/pdf/2006.07988`
- Lim et al., *Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods*, NeurIPS 2021：`https://proceedings.neurips.cc/paper/2021/file/ae816a80e4c1c56caa2eb4e1819cbb2f-Paper.pdf`
- Maurya et al., *Simplifying approach to node classification in Graph Neural Networks*, Journal of Computational Science 2022，DOI `10.1016/j.jocs.2022.101695`。
- Luan et al., *Revisiting Heterophily For Graph Neural Networks*, NeurIPS 2022：`https://proceedings.neurips.cc/paper_files/paper/2022/file/092359ce5cf60a80e882378944bf1be4-Paper-Conference.pdf`
- Goyal and Ferrara, *Graph Embedding Techniques, Applications, and Performance: A Survey*：`https://arxiv.org/pdf/1705.02801`
- Hamilton et al., *Representation Learning on Graphs: Methods and Applications*：`https://arxiv.org/pdf/1709.05584`
- Rozemberczki and Sarkar, *Fast Sequence Based Embedding with Diffusion Graphs*：`https://arxiv.org/pdf/2001.07463.pdf`
