# JITServe: SLO-aware LLM Serving with Imprecise Request Information (`NSDI 2026` [CCF-A])
- 链接：https://www.usenix.org/conference/nsdi26/presentation/zhang-wei
- 代码：https://github.com/UIUC-MLSys/JITServe.git
- 简述：面向多样化 SLO 的 LLM 请求调度系统，利用不精确的请求信息（响应长度上界、执行依赖图匹配）做保守但可逐步松弛的调度，通过 Grouped Margin Goodput Maximization (GMAX) 算法在满足各请求 SLO 的前提下最大化 service goodput，最终实现在 vLLM 上的轻量级集成（~2800 行代码）。

# 1. 该领域存在的一个问题

现有 LLM 服务系统（vLLM, Sarathi-Serve, Autellix）的调度策略聚焦于聚合指标优化——最小化平均 E2EL、最大化总吞吐、或仅优化 latency-sensitive 请求的 TTFT/TBT。但随着 LLM 应用从简单 chatbot 扩展到 agentic workflow、deep research、multi-agent system，请求的 SLO 需求变得高度多样化：

- **Latency-sensitive**（流式交互）：关注 TTFT 和 TBT，如 chatbot 的逐 token 流式输出
- **Deadline-sensitive**（完整响应截止）：关注 E2EL，必须在 deadline 前返回完整响应，如触发外部 tool 的 agent 请求
- **Compound**（多步依赖）：包含多个依赖 LLM 调用，整体 E2EL 需满足 deadline，如 deep research、multi-agent 协调

文章做了用户调研（550+ LLM 用户、百万级请求分析、与两个头部 LLM 服务商沟通），发现即使在同一个应用类别内（如 code generation），用户对延迟模式的偏好也高度分化（38.1% 流式 vs 30.5% 完整响应 vs 31.4% 视情况而定）。

现有调度器在这种多样性下会导致严重的 **service goodput 损失**——一个比 deadline 快 0.5 秒完成的响应如果下游 tool 需要 1 分钟执行，那么这个"过早"完成没有实际价值，却消耗了可以分配给其他更紧急请求的带宽。论文从理论上证明了 EDF 和 SJF 这类传统调度策略在 goodput 指标上可以做到**任意差**（通过构造对手 workload 使 competitive ratio 趋于无穷）。

# 2. 解决问题的思路和分析实验

## 基本思路

**Just-in-Time (JIT) 调度**：不追求精确预测未来（因为不可靠），而是用"保守的上界估计 + 在线逐步松弛"策略——先按最坏情况预留带宽（确保不违反 SLO），随着生成进展逐步释放多余带宽给其他请求。

这个思路要 work，需要三个前提：
- **前提1**：能对响应长度和依赖关系做出**可靠的上界估计**（可以高估但绝不能低估，低估导致 SLO violation）
- **前提2**：有一个调度算法能在"不精确信息"下**最大化 goodput**，且有可证明的性能保证
- **前提3**：调度算法能处理 batch 层面的**长度异构性**——请求输入长度不同时，放在同一 batch 会拖慢所有人

## 分析实验

### 实验1：用户调研——SLO 多样化是真实需求还是伪命题？

论文做了 550+ LLM 用户调研 + 百万级请求分析 + 两个头部 LLM 服务商访谈（§2.1, Appendix A）。

**结论**：
- 请求明确分为三类：latency-sensitive（关注 TTFT/TBT）、deadline-sensitive（关注 E2EL）、compound（多步依赖）
- 即使同一应用类别内部，偏好也高度分化（如 code generation：38.1% 流式、30.5% 完整响应、31.4% 视情况）
- **分集群处理不可行**：SLO 需求在请求粒度就不同，甚至同一请求的不同 phase 会切换类型（如 reasoning phase → deadline-sensitive，streaming phase → latency-sensitive）

→ **设计决策**：需要一个**统一的、SLO-aware 的调度器**，不能只优化单一 SLO 类型。

### 实验2：响应长度预测——为什么精确预测不靠谱？

对比了三种预测方法（Figure 2b, 5b）：Gemini 自预测、fine-tuned BERT、fine-tuned Llama3。结果：**三者都有大幅误差，且倾向于低估**——低估意味着按预测分配带宽会导致 SLO violation。

但换个角度看：**预测上界比预测精确值更可行**。QRF（Quantile Regression Forest，2006 年的经典方法）天然输出分位数，每次预测仅需 7ms（BERT 的 7× 快），且在线 refine 时随生成 token 增加逐步收敛（Figure 5b）。

→ **设计决策**：用 QRF 做**上界估计**（可以保守但不能低估），在生成过程中周期性 refine（每 50 tokens 重跑一次），逐步释放被过度预留的带宽。

### 实验3：现有调度器在多样化 SLO 下的表现——为什么需要新算法？

在混合三种 SLO 的 workload 上测试（Figure 3）：Sarathi-Serve（专为 latency-sensitive 优化，TTFT/TBT 优秀但 deadline-sensitive 的 TTLT 极差）、Autellix（专为 compound 优化，平均 E2EL 好但 SLO 违规率 >90%）。

关键对照：**Autellix + precise info（oracle） vs Autellix**——有了精确信息后 SLO 达成率提升 41%，说明信息不确定性的影响非常大。

论文更从理论上证明了 EDF 和 SJF 在 goodput 指标下的 competitive ratio 是无穷大（Appendix E.1），各用 5 行对手构造：EDF 可被"大量 tight-deadline 低价值请求重复抢占高价值长任务"攻击；SJF 可被"大量极短低价值请求占据所有 slot"攻击。

→ **设计决策**：新调度算法**不能是 EDF/SJF 的简单变体**，需要显式考虑 goodput/bandwidth 比值，且需要有 **provable guarantee**。

### 实验4：异构输入长度对 batch 效率的影响

即使使用 Flash Decoding，将输入长度差异大的请求放在同一 batch 仍然显著拖慢 per-layer 执行速度（Figure 8）。这是 LLM batching 特有的问题——不同于传统任务调度，不仅考虑"调度谁"还要考虑"跟谁一起"。

→ **设计决策**：GMAX 的**第二维**——按输入长度分组。在优先级候选池中用 sliding window 选长度最接近的 B 个请求形成 batch。这解释了为什么 GMAX 不是简单的"按 priority 取 top-B"。

### 实验5：调度延迟 scalability

Figure 9 验证 GMAX 可以在 20ms 内完成数千并发请求的调度决策——这对在线服务至关重要。

→ **设计决策**：O(N log N) 复杂度足够，不需要近似或降级。

# 3. 如何实现

JITServe 是 vLLM 上的一个 policy module，分三块：Request Analyzer（信息预估）→ SLO-Aware Scheduler（调度决策）→ SLO Tracker（在线监控与修正）。

## 3.1 Request Analyzer：不精确但可靠的信息预估

### QRF 响应长度上界估计

**为什么用 QRF 而不是 fine-tuned BERT/Llama3？**
- BERT/Llama3 做的是精确值预测，误差大且**偏差不可控**（有时低估有时高估）。低估 fatal，高估只是浪费。QRF 天然输出分位数，可以指定预测 90%/95% 上界，"宁可高估"的偏差方向可控
- 轻量：每次预测 7ms（BERT 的 1/7），支持在线每 50 tokens refine 一次
- 随着已生成的 token 增多，上界逐步收紧（Figure 5b），自然实现"保守→松弛"

### Pattern Graph 依赖匹配

**为什么不用文本相似度匹配？**
- 下游请求的 prompt 在调度时还不存在（它依赖上游请求的输出）
- 只用图结构（节点=LLM/tool 调用，标注 input/output length 和 model/tool 类型），不用原始文本
- 增量匹配：随着新请求逐步暴露更多 stage，匹配精度自然提升（Figure 7b）
- Compact + 高效：每个 pattern <0.2KB，匹配 500 个历史 graph <5ms（Figure 7a）

**子 deadline 分配**：用累计时间占比 φ(s) = t_≤s / t_total（而非单 stage 占比 t_s/t_total），因为历史信息可以分担单个 stage 的估计噪声，实验验证这个选择显著优于其他公式（Appendix B）。

## 3.2 GMAX 算法：二维调度

GMAX 要同时解决两个维度：**维度1**——每个请求分多少带宽（单请求调度）；**维度2**——选哪些请求放同一 batch（batch composition）。

### 维度1：Priority 公式的推导逻辑

```
Priority(r) = goodput(r) / tgen(r) = goodput per unit generation time
```

**为什么是这个公式？**
- 论文证明了最优单请求调度是 NP-hard（Appendix D.1，归约到 Multiple Knapsack），需要高效的近似
- goodput/tgen 本质是"单位生成时间的 goodput 密度"——生成时间越短、goodput 越高的请求优先调度，这是对 knapsack 问题的经典贪心近似（density-ordered greedy）
- 论文证明了这种贪心策略有 **competitive ratio ≥ 1/8.56**（Theorem 4.1, Appendix E.2），即最坏情况下也不会比 offline optimal 差超过 8.56 倍

**为什么用 frame 级（Δ 时间窗口）而不是 per-token 调度？**
- 每帧重新评估优先级，允许随着 QRF refine 动态调整带宽分配
- 某帧的剩余带宽自动结转到下一帧，给低优先级请求"赶工"的机会
- 对 best-effort 请求，每帧给 priority 加一个小的 δ，防止饥饿

### 维度2：长度分组的必要性

如果只按 priority 取 top-B，会导致 batch 内输入长度差异大，batch 执行被最长输入拖慢（见实验4）。GMAX 的做法（Algorithm 1）：
1. 过滤：只保留 priority ≥ p × 第 B 大 priority 的请求（p ≈ 0.95，可自动在线调）
2. 排序：按 input length 排序
3. 滑动窗口：找长度最接近的连续 B 个请求中 ∑priority 最大的组

p 控制"高 priority vs 长度均匀"的 tradeoff——p 越小候选池越大，长度更均匀但可能混入低 priority 请求。论文让 p 在线自动探索收敛，无需人工调参。

### Preemption 的成本建模

Preemption 有代价（KV cache stall），JITServe 不做盲目的时间片轮转，而是显式建模 gain vs cost：只有当新请求的 goodput 增量 > 抢占导致的 stall 时间内可生成的 token 量时才执行。这保证了抢占的净收益。

## 3.3 侵入性：低

~2800 行代码作为一个 vLLM policy module，不改 vLLM 核心循环，保留 chunked-prefill、PagedAttention、prefix caching。QRF 和 graph matcher 通过 gRPC 异步通信，不在 GPU critical path 上。API 兼容 OpenAI 格式，仅新增 SLO 相关参数。

# 4. 方法

## 核心洞察
**预测上界 + 在线松弛 比 预测精确值 + 一次性决策 更可行**。这在概念上简单，但此前社区的主流思路是"把响应长度预测得更准"（如 S3、TetriInfer、u-Serve 用 BERT/Llama3 做精确预测）。JITServe 论证了这条路线不 work（误差太大且偏差方向不可控），并给出了一套完整的替代方案。

第二个洞察是将 **SLO-aware scheduling 从传统网络/存储领域引入 LLM serving**，但关键不同在于：LLM 的"任务时长"在调度时尚不可知。论文用 QRF 和 pattern graph 解决了这个 LLM 特有的不确定性挑战。

## 与之前工作的关键区别
- **vs Autellix**：Autellix 的 PLAS 本质是 SJF 近似（优先调度"看起来最短"的请求，优化平均 E2EL），不感知 SLO。JITServe 显式建模 goodput，调度目标是最大化 SLO 达成率而非最小化平均延迟
- **vs SLOs-Serve**：都用 DP/优化方法做 multi-SLO，但 SLOs-Serve 的 DP 复杂度高、预设资源分配刚性。GMAX 用贪心 + 滑动窗口的轻量设计，O(N log N) + online 自调参，实际性能更好
- **vs LTR**：LTR 预测响应长度的相对排序（不预测具体值），但仍然是 SJF 逻辑。JITServe 预测上界（具体数值），用于计算最小带宽需求，这是 goodput 最大化的前提

## 理论贡献
- **NP-hardness 证明**（Appendix D.1）：即使有完整未来信息，最优单请求调度也是 Multiple Knapsack 归约，NP-hard
- **Competitive ratio bound**（Appendix E.2）：GMAX 的 competitive ratio ≥ 1/8.56 of offline optimal，用 amortized analysis + credit charging 方法证明。1/8.56 是理论 worst-case，实验中 JITServe 达到 oracle 的 91-97%
- **EDF/SJF 的非竞争性证明**（Appendix E.1）：各用 5 行对手构造即可使 competitive ratio 趋于无穷

# 5. 实验评审

## Baseline 是否合理
- vLLM (FCFS), Sarathi-Serve (chunked prefill, TTFT/TBT 优化), Autellix (PLAS, 针对 compound workload 的 SJF 近似), LTR (Learn-to-Rank, 预测响应长度排序), SLOs-Serve (DP-based multi-SLO 调度)
- baseline 阵容合理，覆盖了 aggregate throughput 优化、latency-sensitive 专项优化、compound 专项优化、multi-SLO 优化等方向
- 特别值得注意：**Autellix w/ precise info**（oracle 变体）作为一个 upper bound 对比，以及 JITServe*（自己带 perfect info 的变体）用于衡量 prediction 带来的性能损失

## 环境
- 真实 GPU：16×A100，多个模型（Llama-3.1-8B, Qwen2.5-14B, Qwen3-30B-MoE-A3B, Llama-3.1-70B），覆盖 dense/MoE 和不同规模
- 每个实验 10K+ 请求，至少 1 小时在线部署窗口
- 请求到达使用 Microsoft 真实 LLM serving trace（主要实验），辅以 Poisson distribution 消融
- SLO 设定来自实测 DeepSeek API 的 P95 latency（增加了实践可信度）

## Workload 多样性
- 4 种应用：Chatbot, Deep Research, Agentic Code Generation, Math Reasoning
- 三种请求模式混合（默认 1:1:1），做了不同混合比例的 sensitivity 实验
- 考虑了不同 SLO tightness 和不同负载水平
- 缺失：纯离线批处理场景（batch API）、长 context 场景（128K+）、embedding 类非生成 workload

## 消融实验
消融实验设计合理：
- JITServe*（perfect info oracle）：接近完美信息性能，差距 3-9%，说明预测质量足够好
- JITServe w/o Request Analyzer（回退到平均响应长度）：goodput 显著下降
- JITServe w/o GMAX（替换为 SJF + QRF 信息）：goodput 下降，说明调度算法本身独立贡献
- 多模型扩展、SLO tightness 变化、workload composition 变化的 sensitivity 实验

## 声称收益的成立条件

- **QRF 预测器需要针对模型做 offline profiling**：不同模型的响应长度分布不同，QRF 需要相应训练
- **Pattern graph matching 依赖历史数据的质量和覆盖度**：如果新 workload 的执行模式与历史完全不同，匹配精度会下降（论文在 §7 承认了这一点，但声称 conservative initialization + online refinement 可以缓解）
- **Preemption 开销评估依赖离线 profiling 的 I/O 带宽**：在不同 GPU 架构上需要重新评估
- **competitive ratio 1/8.56 是理论 bound**：这是 worst-case guarantee，实际性能（near-oracle 3-9% gap）远好于这个 bound

# 6. 局限与边界

## 什么场景可能不 work

1. **请求量不足以填满 batch**：GMAX 的优先级排序和长度分组在 batch 不满时退化为简单的优先级队列，与 SJF 的差距缩小
2. **所有请求 SLO 都极宽松**：如果所有请求的 deadline 都远大于实际生成时间，goodput 最大化退化为 throughput 最大化，JITServe 的调度 overhead 反而可能略微降低绝对吞吐（Figure 14 显示比 Sarathi-Serve 低 2-4%）
3. **QRF 预测器需要冷启动**：新的模型/domain 下预测器需要积累足够的训练数据。论文没有详细讨论冷启动策略
4. **Pattern graph 匹配在全新 workflow 上失效**：依赖历史 pattern，如果用户提交全新的 compound workflow，初始匹配误差大，可能误导早期调度
5. **跨节点的分布式 serving**：论文仅在单节点内 TP 场景评估，没有 PP 跨节点的实验。跨节点时网络延迟的不确定性可能影响 deadline 估计

## 隐藏代价

- **QRF 训练和维护**：虽然推理轻量（7ms），但训练需要累积数据，对每个 LLM 模型分别训练
- **Pattern graph 存储和匹配**：随时间增长需要清理（论文用 0.9/小时的 decay 做 eviction）
- **gRPC 通信**：虽然 paper 声称 negligible，但在极端高并发下可能成为瓶颈
- **Preemption 的 KV cache 开销**：论文建模了 preemption cost 并用 threshold 控制，但实际 KV cache 的保存/恢复在不同硬件上差异大

## 工程化门槛

- **低**。这是 JITServe 相比于 NanoFlow 的一个重要优势。2800 行代码增量，不改 vLLM 核心，API 兼容。这意味着被社区采纳的门槛显著低于 NanoFlow 那种 16K 行的独立引擎。
- QRF 和 pattern graph matcher 作为独立微服务运行，解耦良好

# 7. 工业对齐

## vLLM/SGLang 有无对应 Issue/RFC
- vLLM 社区最近在讨论 multi-SLO support 和 request prioritization，JITServe 的方向与之对齐。2026 年 NSDI 的论文，目前太新，暂无公开讨论
- SGLang 的 cache-aware scheduling 关注 prefix caching 带来的 scheduling 优化，与 JITServe 的 SLO-aware 方向正交但可叠加

## 工业界公开讨论
- 论文与两个头部 LLM 服务商进行了讨论（未具名），用户调研覆盖 550+ 用户
- 论文使用了 DeepSeek API 的 P95 latency 作为 SLO 设定参考
- Microsoft 的 LLM serving trace 用于请求到达建模
- Google/Cisco 作者参与（Nikhil Sarda@Google, Myungjin Lee@Cisco Research），增加了工业落地的可信度
- **JITServe 解决的问题在工业界是真实且紧迫的**：OpenAI/Anthropic 的 batch API（差异化定价）、agent workflow 的 timeout 处理，都直接对应论文中的 deadline-sensitive 和 compound 请求场景

## 开源实现成熟度
- 代码已开源：https://github.com/UIUC-MLSys/JITServe.git
- 基于 vLLM，继承了 vLLM 的生态
- 2800 行代码相对轻量，但作为 NSDI 2026 的新论文，需要观察后续社区活跃度

## 与相关技术的关系

| 技术 | 关系 | 说明 |
|------|------|------|
| Continuous Batching | 互补 | JITServe 决定"batch 里放哪些请求、放多少"，CB 负责"怎么执行这些请求" |
| Chunked Prefill | 互补 | JITServe 继承 vLLM 的 chunked prefill，用于控制 prefill 的 TTFT |
| PD Disaggregation | 互补/部分重叠 | Disaggregation 按 phase 分离资源，JITServe 按 SLO 分配带宽。两者可组合，但 Disaggregation 可能减少单 cluster 内的 SLO 混合度 |
| Prefix Caching | 可叠加 | JITServe 已继承 vLLM 的 prefix caching。更智能的做法是把 prefix cache hit 的预期加速纳入 GMAX 的优先级计算（论文未做） |
| NanoFlow | 不同优化层级 | NanoFlow 优化 operation 级别的 kernel overlap，JITServe 优化 request 级别的 SLO-aware 调度。理论上可以叠在一起用 |
| Autellix | 替代关系 | JITServe 直接对标并超越了 Autellix 的 PLAS 调度，同样的 compound workload 优化目标，但更 general（支持三种 SLO 类型） |

---

**一句话总结**：JITServe 将 SLO-aware scheduling 引入 LLM serving，通过 QRF 预测响应长度上界 + pattern graph 匹配依赖关系，以"保守估计、逐步松弛"的策略在最大化 service goodput 的同时保持低侵入性（vLLM + 2800 行），在 1.4×-6.3× goodput 提升之外，更重要的贡献是系统化地论证了"为什么现有聚合指标优化的调度器在多样化 SLO 下会失败"。

**证据强度**：[强]
- 真实 GPU + 多模型 + 多种 workload + 长时间在线部署实验
- 用户调研 + 工业服务商讨论 + 真实 trace 驱动
- Baseline 阵容全面
- 理论分析严格（NP-hardness proof + competitive ratio bound）
- Ablation 消融合理，sensitivity 分析覆盖多个维度
- 轻微不足：无多节点 PP 实验；QRF 冷启动策略未展开；长 context 场景未评估

**值得关注 / 可以跳过**：
- **值得关注**。JITServe 解决的是一个真实且日益紧迫的问题——随着 agentic workflow 的普及，多样化的 SLO 需求会越来越突出。论文的"goodput 最大化"框架、QRF 上界预测、GMAX 算法三部分都值得认真阅读。相比于 NanoFlow 的"改写整个引擎"，JITServe 的"vLLM + 2800 行"策略更务实，被社区采纳的可能性更高。
- 建议关注：GMAX 算法是否可以独立抽取出来，作为 vLLM 的 scheduler plugin 使用？QRF 预测在你的 workload 上的实际 accuracy 如何？SGLang 社区是否会做类似的多 SLO 支持？这将决定这个方向的生态走向。
