# Paper Title
**Conference | Year | Rating**
- 1. Problem（作者想解决什么问题）
- 2. Gap（现有方法为什么不够）
- 3. Idea（作者的核心思想）
- 4. Evidence（如何证明）
- 5. Takeaway（我的理解）


# Past-Future Scheduler for LLM Serving under SLA Guarantees

**ASPLOS 2025 | A**
  基于 LightLLM 实现调度器原型。

- 存在的问题：LLM serving 中，调度器需要同时满足 SLA 和提高 GPU 利用率，但输出长度和未来 KV cache 占用在请求到达时未知。现有方法通常走两个极端：
    - 保守调度：按最大输出长度或高分位估计未来 KV 占用，能减少 eviction/SLA violation，但会低估可用显存，导致 batch size 偏小、吞吐浪费；
    - 激进调度：按较短输出或当前显存占用接纳更多请求，GPU 利用率更高，但一旦运行中请求生成更长，就可能触发 KV cache 不足、request eviction，进而破坏
      SLA。

- 文章提出了 Past-Future Scheduler：调度器利用历史请求的输出长度分布，动态预测当前 running requests 未来还会生成多少
  token，并据此估计未来 KV cache 峰值。在调度 waiting队列的请求时，它会先预测候选请求的输出长度，再模拟该请求加入 running batch 后的未来 KV cache 占用情况，只有当未来峰值显存仍满足约束时，才允许该请求进入 batch。这样可以在避免 eviction 的同时，比保守策略接纳更多请求。

Past-Future Scheduler 的重点不是追求最高 raw throughput，而是在 SLA 约束下提高goodput。实验
  显示，它在多数负载下能取得更高 goodput，同时避免激进调度中因 KV cache 不足导致的 eviction 和 SLA violation。


# JITServe: SLO-aware LLM Serving with Imprecise Request Information
**NSDI 2026 | A**
  基于vLLM改动，代码已开源在https://github.com/UIUC-MLSys/JITServe/，基本没人使用

- 存在的问题：现有 LLM serving scheduler 大多优化throughput、平均延迟或单一类型SLO，难以同时处理多种应用级SLO。请求的输出长度、依赖关系等都不明确，导致传统FCFS，SJF-like调度很难稳定最大化 goodput。

- 文章提出了 JITServe：核心思想是 Just-in-Time scheduling，每个请求分配“刚好满足 SLO 所需”的 serving bandwidth，把剩余 capacity 留给其他请求。具体做法包括：
    - 用 QRF（Quantile Regression Forest） 预测输出长度的上界
    - 随着生成进行，每隔一段 token 重新预测，逐步放松保守估计；
    - 对 compound requests，用 pattern graph matching 估计动态依赖关系；
    - 用 GMAX（Grouped Margin Goodput Maximization） 给请求排序和组 batch：优先服务 margin goodput 高的请求，同时把 input length 相近的请求组成
        batch，减少 batch 内长度不均带来的执行效率损失。


- 实验设置：使用 Llama-3.1-8B、Qwen2.5-14B、Qwen3-30B-MoE-A3B、Llama-3.1-70B，在 16 张 A100 上测试。workload 包括 chatbot、deep research、agentic
    code generation、math reasoning。对比对象包括 vLLM、Sarathi-Serve、Autellix、LTR、SLOs-Serve。

- 实验效果：JITServe 相比现有方法提升 1.4×–6.3× service goodput，或者在相同 goodput 下节省 28.5%–83.2% 资源。论文还说它距离 oracle 只差 3%–9%，throughput overhead 很小。




# Beyond Prediction
**ICML 2026| CCF-A**
针对Tail Latency的调度策略，参考Strongly tail-optimal scheduling in the light-tailed M/G/1这篇文章提出的排队算法。


# SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads
**OSDI 2025 | CCF-A**
文章针对请求到达时加载模型的服务方式，提出了SubNetAct



TetriInfer: A General Efficient Scheduling Framework for LLM Inference
LoongServe (SOSP'25) —— 长上下文场景下的调度与并行。

| **SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads** | NSDI 2025 | A | 不可预测负载的细粒度推理服务 |

| **BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching** | OSDI 2025 | A | Serverless 模型弹性扩缩，GPU 计算网络替代 host 缓存，层级别活迁移，尾延迟降低 94% (SJTU, Huawei) |


**PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Model HPCA 2026 | A**
面向 reasoning-based LLM / CoT 模型的感知调度器。

- 存在的问题：现有 LLM serving scheduler 通常把 decode 阶段看成同质的 token-by-token generation，只关注整体 TTFT、TPOT、throughput 或 KV cache 约束。但 reasoning-based LLM 的输出可以分成 reasoning phase 和 answering phase：前者用于生成思考过程，通常较长且用户未必看到最终答案；后者才是用户真正关心的可见回答。如果调度器不区分这两个阶段，就可能让长 reasoning 请求占住 GPU，导致其他请求的最终答案迟迟出不来；或者过度抢占 answering 阶段，破坏流式输出体验。

- 文章提出了 PASCAL：核心思想是 phase-aware scheduling，也就是根据请求所处阶段采用不同调度策略。对 reasoning phase，调度器更积极地推进，请求尽快越过长思考阶段，降低最终答案的等待时间；对 answering phase，则使用更温和的控制策略。

- 关键机制：
  - 将 reasoning-based LLM 的生成过程划分为 reasoning phase 和 answering phase
  - 对 reasoning phase 赋予更高优先级，降低TTFT
  - 对 answering phase 使用 controlled preemption，避免回答阶段被频繁打断
  - 使用 token pacing 控制回答阶段的输出节奏，维持 QoE/SLO
  - 在 phase boundary 处做动态迁移，缓解实例间负载不均和阶段间干扰
  - 结合 instance-level placement 和 intra-instance execution 两层调度
