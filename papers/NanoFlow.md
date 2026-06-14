# NanoFlow: Towards Optimal Large Language Model Serving Throughput  (`OSDI 2025` [CCF-A])  
- 链接：https://www.usenix.org/conference/osdi25/presentation/zhu-kan  
- 代码：https://github.com/efeslab/Nanoflow
- 简述：从设备内执行调度角度入手，把输入切成更小的 `nano-batches`，自动决定批大小、顺序和资源分配，尽量将计算、访存和通信overlap，来提升GPU的利用率，进而提升整体 serving throughput。 在当时可以实现比vLLM快4.18×，比TensorRT-LLM快1.91×。这里没和SGLang比较。


# 1. 文章发现该领域存在的问题

在 TP 多 GPU 场景下，LLM 推理的一次 iteration 包含三类异构操作：compute-bound 的 GEMM(矩阵乘)， memory-bound 的 decode attention ， network-bound 的 AllReduce/AllGather。现有引擎（vLLM, SGLang, TensorRT-LLM）将这些操作**顺序执行**。单个操作对其瓶颈资源的利用率大约为80%，但因为不同操作的瓶颈资源不同，且 compute 在所有操作之间是断续的——GEMM 运行时GPU利用率高，attention 运行时利用率低——导致**平均利用率只有 40%**。

# 2. 解决问题的思路和分析实验

## 基本思路
将输入的 Batch 拆成多个更小的 **nano-batch**，每个 nano-batch 有自己的 **nano-operation**（与原 operation 执行相同计算，只是数据量变小）。由于不同 nano-batch 之间没有数据依赖，原本按顺序执行的异构操作——计算密集型（GEMM）、内存密集型（decode attention）、通信密集型（AllReduce/AllGather）——可以**同时在 GPU 上并行执行**，填补原有顺序执行产生的 pipeline bubble，从而提高GPU资源的利用率。

代价：nano-batch 需要重复加载模型权重（原本一份 batch 只需加载一次），增加了 memory I/O。

这个思路，需要两个前提：
- **前提1**：额外增加的 memory I/O 可以被 overlap "隐藏"掉。只有大部分时间为计算密集型，才可行，因为 compute 本来就在空等，memory I/O 多做一些不拖慢总时间。
- **前提2**：能找到一个实际可用的 nano-batch 切分方案（数量、大小、顺序、GPU 资源分配），同时处理好 kernel 间的干扰。

## 分析实验---Cost model的建立
这是论文最核心的分析贡献，感兴趣可以阅读原文。作者**先建了一个 cost model，再拿到真实 GPU 上验证**。


    每个 GPU iteration 的耗时可以从三个视角来算：

    - **内存视角**：如果把 GPU 显存里的所有数据（模型权重 + KV-cache）全部读一遍需要多久？ T_mem = MemSize / MemBW
    - **计算视角**：GEMM 操作做完需要多久？T_compute ≈ 2 × B_dense × P_model / Compute。其中 B_dense 是 GEMM 的 token batch size（prefill tokens + decode tokens 汇总），P_model 是参数量
    - **通信视角**：TP 的 AllReduce/AllGather 传完数据要多久？T_net ≈ 4 × (N_GPU - 1) × B_dense × D_model × S_type × L / NetBW

    三个时间中**最长的那个就是瓶颈资源**。

## 2.3分析实验1---通信不是瓶颈
T_net / T_compute 这个比值中，D_model 项在分子出现一次而在 P_model中出现两次，所以对 4096+ hidden dim 的大模型，这个比值天然远小于 1。加上 NVLink 的高带宽（900 GB/s），通信时间远小于计算时间。

    <div align="center">
    <img src="../images/ntct.png" >

    上图是 T_net vs T_compute 的热力图，越接近黄色越 compute-bound，越接近蓝色越 network-bound。可以看到几乎所有模型+workload 组合都是黄色——**通信不是瓶颈**。

## 2.4 分析实验2---内存也不能算是瓶颈：

    ```
    T_R = T_mem / T_compute ≈ (Compute / MemBW) × (MemSize / P_model) × (1 / (2 × B_dense))
    ```

    T_R > 1 → memory-bound；T_R < 1 → compute-bound

    这里面有三个因子：
    - Compute/MemBW：硬件决定，各代 GPU 大致稳定在 150-300 范围内（Table 1）
    - MemSize / P_model：模型越大，这个比值越小
    - 1 / (2 × B_dense)：**batch size 越大，T_R 越小，越偏向 compute-bound**

    **GQA 是改变游戏规则的关键**。没有 GQA 时，每个 attention head 都有独立的 KV-cache，KV-cache 吃满显存，B_dense 只能到 ~256。有了 GQA（多个 Q head 共享 KV head），同样的显存可以装更多的 decode 请求，B_dense 可以达到 ~2048（对于 LLaMA-2-70B on 8×A100）。B_dense 直接大了 8 倍，T_R 除以 8，**从 memory-bound 翻转到 compute-bound**。

    <div align="center">
    <img src="../images/ctmt.png" >

    上图是 T_R 热力图（五个模型 × 多种 workload），越接近黄色越 compute-bound，越接近绿色越 memory-bound。**结论一目了然：除了 LLaMA-3-8B + 长 decode（512-1024 tokens）这个组合外，其余全部是黄色——compute-bound**。尤其在 70B 级别模型上，T_R 远小于 1（compute/memory 比值 > 2）。

## 2.5实验验证和结论
    作者在 LLaMA-2-70B + 8×A100 + B_dense=2048 上做了 profiling。结果：
    - 所有 op 的 T_compute 之和 ≈ 114ms，T_mem ≈ 45ms，T_net ≈ 31ms
    - 实际测得的总时间与 T_compute 估计值基本吻合（除了 prefill attention 有 kernel launch overhead）
    - **compute 确实是瓶颈**，耗时是 memory 的 2.5 倍

实验结论

    - 既然 compute 是瓶颈而 memory 带宽有余量，**nano-batch 重复加载权重的额外 memory I/O 可以被 overlap 吸收，不拖慢总时间**。
    - 最优吞吐公式简化为：Throughput_optimal = Compute / (2 × P_model)，**仅取决于 GPU 总算力和模型参数量**，与显存大小、带宽、序列长度无关。
    - 对比最优值，现有系统仅达到 22-38%（vLLM 22%, TensorRT-LLM 37.8%），**gap 的来源就是GPU资源在顺序执行中被浪费了**。

# 3.NanoFlow如何实现(简述)

NanoFlow 核心分为两部分：自动搜索引擎（offline 生成 pipeline）+ 运行时（online 执行 pipeline）。

分为三部分： profiling → Stage I（pipeline 结构搜索）→ Stage II（资源分配优化）。
- 1. 分析三类kernel在无干扰和两两并行相互干扰时的性能表现。
- 2. 结合上面无干扰的分析结果，尽量消除compute bubble
- 3. 通过实验得出的R->P的映射表格，来修正执行时间

## 示例

- **70B 模型**（LLaMA-2-70B, TP=8）：KQV 阶段三种资源都参与，切 4 个 nano-op；后续 GEMM 为主，切 2 个。Decode attention 分到 R=0.4（牺牲 GEMM 40% 性能），达到独立运行的 80% 性能
- **8B 模型**：单卡，无网络通信，只切 2 个 nano-op，decode attention 与 FFN 的 Up/Gate/Down 做 overlap
- **MoE 模型**：FFN 用 grouped-GEMM + gate routing，auto-search 自动适配出有效 pipeline
运行时:
- **Batch formation**：优先 decode 请求，按 token 粒度 chunk prefill 填满固定 dense batch size（类似 SarathiServe），保证每轮 iteration 的 compute 负载稳定
- **异步调度**：CPU 侧 batch formation 与 GPU 执行 overlap，用 1 个 iteration 的延迟（多生成一个 token，开销 <1%）换取 batch formation 延迟的隐藏
- **KV-cache offloading**：KQV 生成后立即 offload KV vectors 到 CPU/SSD（不等请求完成），利用 FFN 计算时间做 D2H copy。Offload 回 GPU 时先拷到连续 buffer 再 scatter 到 PagedAttention 的分页目的地（7-10× 带宽提升）

# 4.具体实现细节(可跳过)
## 4.1Kernel 性能 Profiling（前置步骤）

在为具体模型做 auto-search 之前，需要先 profiling 三类 kernel 的两种行为：

**无干扰性能**（各 kernel 独占 GPU）：以 128 为batch size，从 128 到 max dense batch size，穷举 thread block数 /  warp数 / tile size 的组合，记录每种(kernel, batch_size)下的最优实现和耗时。

**有干扰性能**（两两并行）：核心难点——如何量化"kernel A 和 B 同时跑时互相拖慢多少"？

NanoFlow 的做法：
- 以 GEMM 性能为**资源占用的 proxy R**：GEMM 与 GEMV 并行时，GEMM 降到独立运行时的 40% → R_GEMM=0.4，假设 GEMV 拿到剩余资源 R_GEMV=0.6
- 归一化定义 **P**：每个 kernel 在干扰下的性能 / 其独立运行时的最优性能
- 对 GEMM，P = R（定义如此）；对 GEMV/Network，P 捕获了"给了 R 资源后实际能达到的性能比例"
- 这样就建立了 **R → P 的映射关系**（Table 3），本质是"用多少 GEMM 性能换多少 GEMV/Network 性能"的汇率表

搜索空间压缩：
- GEMV/Network kernel 只测 8-128 thread blocks（128 就饱和了），排除耗时更长但用了更多 thread blocks 的低效 GEMM kernel
- 只做 **pairwise 干扰**（GEMM-GEMV, GEMM-Network），不做三 kernel 同时干扰
- 最终约 100 对 GEMM-GEMV 组合，按 GEMM 性能降序排列，丢弃"伤 GEMM 多但帮 GEMV 少"的劣质组合，保留 Pareto 最优 trade-off 点
- **关键结论**：R→P 映射在不同 GEMM shape 和 batch size 下 std < 5%，一张表可覆盖所有场景

## 4.2 Stage I：Pipeline 结构搜索（MILP，忽略干扰）

**输入**：模型 op 依赖图 + max dense batch size + 无干扰 kernel profiling 结果

**优化目标**：消除 compute 的 pipeline bubble，最小化总执行时间

**核心约束**：
- **切分约束**：每种 op 至少切 2 个 nano-op，从 2 开始尝试，有 compute bubble 就增加切分数
- **依赖约束**：两个 nano-op 有依赖 ⇔ 其父 op 有依赖 **且** 输入 batch 范围有交集。**这是 nano-batching 创造 overlap 机会的关键——反过来，只要两个 nano-op 的 batch 不交叠，即使父 op 有依赖，它们也可以并行执行**
- **重叠约束**：只用不同资源瓶颈的 op 做重叠（compute 跟 compute 重叠无意义）
- **网络 op 变换**：AG 可以等价转换为 AR（通过调整权重划分方式），auto-search 会探索所有等价变换

搜索约 10 分钟，优先找可行解而非可证明最优解。

## 4.3 Stage II：资源分配优化（MILP，考虑干扰）

保持 Stage I 输出的 nano-batch 数量/大小/顺序不变，用实验得出的表格中的 R→P 映射修正执行时间。

- **资源约束**：任意时刻 ∑R ≤ 1.0
- **时间修正**：实际执行时间 = D_best / P（R 越小 → P 越低 → 时间越长）
- 输出最终 pipeline：每个 nano-op 的 batch range、kernel 实现、资源分配 R、CUDA stream 上的执行时序

不是"通用调度算法"，而是"给定具体 (模型, GPU) 配置后自动生成最优执行方案的工具链"。两阶段 MILP + interference profiling 的方法论扎实但高度工程化，换模型或换 GPU 需要重新跑 profiling 和搜索。

# 5. 实验部分
## 环境
- 真实 GPU：8×A100 80GB SXM with NVLink，是合理的实验平台
- 但**所有实验都是单节点**。对于声称要解决 "planet-scale serving" 问题的论文来说，缺乏多节点实验是一个明显的 gap。跨节点的 PP 场景下，网络通信模式更复杂（IB/RoCE vs. NVLink），T_net 会显著增大，可能改变 compute/network 的瓶颈判断。
- 没有在 H100/H200/B200 上的实验。虽然 Table 1 分析了硬件比例趋势并声称 "ratios are stable"，但缺乏实证支持。
## Workload 多样性
- 三个数据集（ShareGPT, LMSYS-Chat-1M, Splitwise）都是对话场景
- 缺失：非对话 workload（batch embedding、文档摘要、代码生成、RAG 长文档 QA）
- 缺失：长 context 场景。随着 context window 扩展到 128K+，KV-cache 的 memory pressure 会显著改变瓶颈判断。论文自己承认 LLaMA-3-8B + 长 decode（512-1024 tokens）时 T_R ≈ 1（compute 和 memory 同等瓶颈）
## 消融实验
- nano-batch 不 overlap：13.2% 性能下降 → nano-batching 本身有开销
- overlap network kernel 贡献 1.07×，overlap network + memory kernel 贡献 1.17×
- **结论**：1.91× 的总加速比中，overlap 的核心技术贡献约为 1.17-1.2×，其余来自更好的 kernel 实现（CUTLASS vs. 各框架自带的 GEMM）和异步调度
## 延迟评估
- 低负载下 NanoFlow 延迟**略高于** TensorRT-LLM（因为维护了大 batch），高负载下在 200ms SLO 内能处理 **1.64×** 更多请求
- **P99 latency 只比平均值高 1.07×**——固定 dense batch size 使每轮 iteration 耗时稳定，意外地带来了良好的 tail latency。这对吞吐优先系统是加分项
## 多模型结果
在 LLaMA-3-70B, LLaMA-3-8B, QWen2-72B, Deepseek-67B, Mixtral 8×7B 上均验证，达到 50%-72% 最优吞吐，平均比 vLLM 快 2.66×。其中：
- MoE 模型（Mixtral）受益最大（FFN 计算量更大 → compute-bound 更强 → overlap 空间更大）
- 8B 单卡场景也有收益但显著低于 TP 场景（两种资源 overlap vs 三种资源 overlap）
## 声称收益的成立条件
- 大 batch size（B_dense ~2048）：只有在批量推理或高请求率在线服务中成立
- GQA 模型：GQA 减少了 KV-cache memory pressure，使得更大 decode batch 成为可能。非 GQA 模型下 batch size 会小得多
- 相对短的 context：论文的 avg input 在 102-1155 tokens、avg output 在 211-322 tokens
- 至少需要 TP：单卡能装下的模型（如 8B）没有网络通信，overlap 空间有限

# 6. 局限与边界

## 什么场景可能不适用

1. **低请求率在线服务**：batch size 不够大时 compute-bound 前提被削弱。论文 §6.3 的低请求率区间中，NanoFlow 延迟略高于 TensorRT-LLM。

2. **长 context 推理**：KV-cache 膨胀→decode batch 变小→GEMM 计算密度下降→memory 可能重新成为瓶颈。这是 NanoFlow 的一个基本假设风险——随着长 context 越来越重要，compute-bound 的结论可能越来越不成立。

3. **小模型（<8B）**：单卡装得下，没有网络通信。compute/memory 两种资源的 overlap 空间远小于三种资源。论文的 8B pipeline 只用 2 个 nano-batch。

4. **非 NVIDIA GPU**：整套实现基于 CUDA/CUTLASS/MSCCL++。kernel interference 的特性在不同 GPU 架构上完全不同，需要重新 profiling。

5. **与 Prefix Caching 的交互不明确**：SGLang 的 RadixAttention 通过 prefix caching 大幅减少 prefill 计算量。如果 prefill 被缓存命中"吃掉"了，pipeline 中 compute 的比例会降低，可能改变 nano-batch 的分配策略。论文完全没有讨论这个交互。

## 代价
- **工程复杂度**：~16K 行代码（CUDA + Python），MILP 求解器依赖，需要对每种模型/GPU 组合做 profiling
- **静态 pipeline 的僵化**：pipeline 是为固定 batch size 生成的。在真实服务中 batch size 随请求到达模式动态变化。论文没有讨论动态调整 pipeline 的机制
- **Interference model 的简化**：只 profile 了 pairwise interference（GEMM-GEMV, GEMM-Network），假设三相干扰可以从 pairwise 推导。这个假设在 3 个 kernel 同时运行时（尤其是 memory controller 和 NVLink 同时抢占 L2 cache 时）可能不准确



# 7. 业界内进展 截止为2026年6月

## vLLM/SGLang 有无对应 Issue/RFC

**SGLang 已采用 NanoFlow 的异步调度**。SGLang v0.4（2024.12）的 "Zero-Overhead Batch Scheduler" 直接引用了 NanoFlow 的思路——CPU 侧提前一个 batch 做调度准备（memory allocation、prefix matching、radix cache 操作），GPU 执行当前 batch 时 CPU 已经在准备下一个 batch。Nsight profiling 确认 GPU 无空闲间隔，获得 1.1× 吞吐提升。Blog 原文："This idea is simple and has been proposed in NanoFlow."

但 SGLang **只采用了异步调度这一部分**，没有采用 nano-batching + MILP 的完整方案。

**DeepSeek V3 独立验证了 SM 划分方向**。DeepSeek V3/R1 的推理系统使用 ~20 SM 专门做通信（warp specialization + custom PTX），其余 SM 做计算；dual micro-batch 策略将 batch 一拆二交替执行，与 NanoFlow 的 nano-batching 概念上同源。这不是"受到 NanoFlow 启发"，而是独立收敛到相同结论——说明 SM 级别资源划分是正确方向。

vLLM 社区暂无公开讨论 NanoFlow。vLLM 目前的优化重点在 multi-step scheduling、disaggregated serving、FP8 量化等方向，与 NanoFlow 的优化层级不同。

## 工业界公开讨论
- **DeepSeek V3**：推理系统（DeepEP + profile-data）使用了本质上相同的 SM 级别资源划分——~20 SM 留给通信，其余做计算，dual micro-batch 做 compute-communication overlap。独立验证了方向
- **SGLang v0.4**：直接引用了 NanoFlow 的异步调度思路，已在生产中使用
- **NVIDIA**：TensorRT-LLM 尚未公开引入类似的 SM 级别 kernel overlap。但 DeepSeek 的做法已经证明了这在实际部署中有效，NVIDIA 是否会跟进值得关注


## 与相关技术的正交性分析

| 技术 | 关系 | 说明 |
|------|------|------|
| Continuous Batching | 互补 | NanoFlow 在 nano-batch 层面做 overlap，CB 在 request 层面做动态批处理。两者关注不同层级 |
| Chunked Prefill | 部分重叠 | NanoFlow 的固定 dense batch + token 粒度 chunk 策略实现了类似效果，但动机不同（为了保持 pipeline 稳定而非降低 prefill 延迟） |
| PD Disaggregation | 可能削弱动机 | 如果 compute 是瓶颈，分离 prefill/decode 到不同实例不能解决根本瓶颈。但如果分离后单实例 batch 变小，compute 瓶颈可能减弱 |
| Prefix Caching | 未讨论 | Prefix cache hit 减少 prefill 计算量，可能改变 compute/memory 平衡 |
| FP8 / 量化 | 未讨论 | FP8 下 compute 吞吐翻倍但 memory bandwidth 不变，可能改变 compute-bound 前提 |

# 8. 简评：
- 1. NanoFlow 的核心贡献是系统论证了"LLM 推理是 compute-bound"并首次在 GPU operation 级别通过 nano-batching 实现异构资源的细粒度并行。1.91× 的总体加速比中，overlap 本身贡献 ~1.2×。作为独立引擎未被直接采用，但其核心思想已在工业界分裂传播：SGLang v0.4 采用了异步调度，DeepSeek V3 独立验证了 SM 级别资源划分方向。cost model 分析框架和 intra-device parallelism 方向已被证明是正确的。
- 2. 值得关注：SM 级别资源划分已成为工业级推理优化的主流方向之一
- 3. 作为一个独立引擎的工程化门槛仍然高，且收益主要在大 batch 离线场景。但如果你在做推理引擎开发，应该关注它的核心思想（尤其是异步调度和 kernel overlap），这些已经被 SGLang 和 DeepSeek 证明可行
