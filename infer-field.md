# 大模型推理调度近期进展清单

更新时间：2026-04-17

这份清单聚焦 **LLM inference serving / scheduling**，重点不是模型算法本身，而是线上推理系统里更接近工程和系统研究的问题：

- 如何在 `prefill` 和 `decode` 两个阶段之间做调度
- 如何利用 `KV cache / prefix cache` 做 cache-aware 调度
- 如何在多实例、多 GPU、多模型场景下做重调度、迁移和负载均衡
- 如何在吞吐、时延、公平性、SLO 之间做权衡

如果只看一个结论，过去两年的主线大致是：

1. 从 `continuous batching + PagedAttention` 这类基础能力，走向更细粒度的运行时调度。
2. 从单实例队列调度，走向 `prefill/decode` 解耦、跨实例重调度、cache-aware 路由。
3. 从“把 GPU 跑满”，走向“按 TTFT / TPOT / tail latency / fairness / cost` 这些服务指标做系统优化。
4. 2025 年以后，越来越多工作开始把视角扩展到 `多模型池化`、`agentic / intercepted inference`、`serverless` 和 `大规模生产环境`。

---

这个才是你文章的“主菜”，包括：

输入阶段
batching
dynamic batching
执行调度
prefill / decode
scheduling
性能优化
prefix caching
chunked prefill
speculative decoding
系统级问题
tail latency
admission control
fairness

## 先看这 6 篇

如果你想先建立主线理解，推荐按这个顺序读：

1. **PagedAttention / vLLM**  
   论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention*  
   链接：https://arxiv.org/abs/2309.06180  
   为什么看：这是现代 LLM serving 的基础设施论文之一，很多后续调度工作默认都建立在它提出的 KV cache 管理思路上。

2. **Sarathi-Serve**  
   论文：*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*  
   链接：https://arxiv.org/abs/2403.02310  
   为什么看：把问题非常明确地表述成 `prefill` 和 `decode` 的调度矛盾，并提出 `chunked prefills + stall-free scheduling`。

3. **DistServe**  
   论文：*DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*  
   链接：https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin  
   补充代码页：https://github.com/LLMServe/DistServe  
   为什么看：把 `prefill/decode` 从“同机同队列”进一步推进到“物理解耦”，是 2024 年最关键的方向之一。
   
1. **FlashAttention**
   论文：FLASHATTENTION: fast and memory-efficient exact attention with IO-awareness
   链接：https://dl.acm.org/doi/10.5555/3600270.3601459
   为什么看：减少了Attention计算中IO次数，成为了行业的基石

4. **Llumnix**  
   论文：*Llumnix: Dynamic Scheduling for Large Language Model Serving*  
   链接：https://arxiv.org/abs/2406.03243  
   为什么看：关注异构请求、优先级、尾延迟和在线迁移，本质上是把操作系统里的 runtime rescheduling 思路带到 LLM serving。

5. **Preble**  
   论文：*Preble: Efficient Distributed Prompt Scheduling for LLM Serving*  
   链接：https://arxiv.org/abs/2407.00023  
   为什么看：代表了 `prefix-aware / prompt-sharing-aware` 分布式调度方向，不再把每个请求当成完全独立的工作负载。

6. **Inside vLLM (2025 博客)**  
   博客：*Inside vLLM: Anatomy of a High-Throughput LLM Inference System*  
   链接：https://vllm.ai/blog/anatomy-of-vllm  
   为什么看：不是论文，但它把 `scheduler / paged attention / chunked prefill / prefix caching / speculative decoding / disaggregated P/D` 串成了一张比较完整的图。

---

## 论文清单

## A. 基础设施与 baseline

- **Orca: A Distributed Serving System for Transformer-Based Generative Models**  
  来源：OSDI 2022  
  链接：https://www.usenix.org/conference/osdi22/presentation/yu  
  推荐理由：早期把 iteration-level scheduling 和 continuous batching 明确化的代表作，很多后续工作默认以它为 baseline 或背景。

- **Efficient Memory Management for Large Language Model Serving with PagedAttention**  
  来源：SOSP 2023 / arXiv 2023  
  链接：https://arxiv.org/abs/2309.06180  
  推荐理由：现代 LLM serving 的“地基”，重点是 KV cache 的分页式管理，直接影响后续调度空间。

- **Fairness in Serving Large Language Models**  
  来源：arXiv 2024  
  链接：https://arxiv.org/abs/2401.00588  
  推荐理由：如果你关心的不只是吞吐，而是多用户场景下的服务公平性，这篇值得读。它把“谁先服务、谁被饿死、短请求是否被长请求压住”问题提到台面上。

## B. Prefill / Decode 调度与解耦

- **Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills**  
  来源：arXiv 2023  
  链接：https://arxiv.org/abs/2308.16369  
  推荐理由：先看思想版，再看 Sarathi-Serve 的系统化版本会更顺。

- **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**  
  来源：arXiv 2024，OSDI 2024 session  
  链接：https://arxiv.org/abs/2403.02310  
  推荐理由：调度视角非常强，核心是避免 prefill 把 decode 卡死，同时维持高吞吐。

- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**  
  来源：OSDI 2024  
  链接：https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin  
  推荐理由：从“同队列优化”走向“不同阶段分配不同资源”，对后续工业界系统影响很大。

- **P/D-Serve: Serving Disaggregated Large Language Model at Scale**  
  来源：arXiv 2024  
  链接：https://arxiv.org/abs/2408.08147  
  推荐理由：更偏工业系统与大规模部署，强调解耦之后在生产环境里如何组织、编排、扩缩容。

## C. Cache-aware / Prefix-aware 调度

- **Prompt Cache: Modular Attention Reuse for Low-Latency Inference**  
  来源：arXiv 2023  
  链接：https://arxiv.org/abs/2311.04934  
  推荐理由：如果你想理解“可复用前缀为什么会改变调度目标函数”，这篇很适合做入口。

- **Preble: Efficient Distributed Prompt Scheduling for LLM Serving**  
  来源：arXiv 2024  
  链接：https://arxiv.org/abs/2407.00023  
  推荐理由：分布式 prompt scheduling 的代表作，核心不是单机 batch，而是“共享前缀 + 分布式放置 + 负载均衡”三者联动。

- **Hydragen: High-Throughput LLM Inference with Shared Prefixes**  
  来源：2024  
  链接：https://arxiv.org/abs/2402.05099  
  推荐理由：虽然更偏 shared-prefix execution，但和 prefix-aware scheduling 是同一条技术脉络，适合并读。

## D. 在线重调度、迁移、异构请求

- **Llumnix: Dynamic Scheduling for Large Language Model Serving**  
  来源：OSDI 2024 / arXiv 2024  
  链接：https://arxiv.org/abs/2406.03243  
  推荐理由：核心价值在于 `runtime rescheduling + live migration`，对尾延迟、优先级请求、隔离性很有启发。

- **ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models**  
  来源：OSDI 2024  
  链接：https://yeqi-huang.com/publication/serverlessllm/  
  推荐理由：如果你关心 serverless 或冷启动，这篇很关键；它把 locality、迁移、模型加载时间一起纳入调度问题。

- **INFERCEPT: Efficient Intercept Support for Augmented Large Language Model Inference**  
  来源：ICML 2024 / arXiv 2024  
  链接：https://arxiv.org/abs/2402.01869  
  推荐理由：面向 tool use / agent workflow。对今天的 agentic serving 很有参考价值，因为请求会被外部调用中断再恢复，传统推理调度模型不够用了。

## E. KV cache 管理与长上下文相关

- **InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management**  
  来源：arXiv 2024  
  链接：https://arxiv.org/abs/2406.19707  
  推荐理由：更聚焦 KV cache 的动态管理和长文本生成，在长上下文服务里很实用。

- **ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition**  
  来源：arXiv 2024  
  链接：https://arxiv.org/abs/2402.15220  
  推荐理由：和 prefix-aware KV 组织、两阶段划分相关，适合作为连接“算子优化”和“调度优化”的一篇。

## F. 2025 之后值得留意的新方向

- **Aegaeon: Effective GPU Pooling for Concurrent LLM Serving on the Market**  
  来源：SOSP 2025  
  链接：https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf  
  补充介绍：https://www.alibabacloud.com/blog/602623  
  推荐理由：很值得关注。它把问题从“单模型多请求”推进到“多模型并发 + GPU 池化 + token-level 资源复用”，更接近模型市场和企业托管场景。

- **Structure Prediction and Opportunity-Cost Scheduler for LLM Inference**  
  来源：Pattern Recognition, online 2026-03-23  
  链接：https://www.sciencedirect.com/science/article/pii/S0031320326005650  
  推荐理由：这是很新的工作，可信度还要看后续影响力，但思路上有意思：不用脆弱的长度估计，而是引入结构预测和机会成本来做调度。

---

## 博客与工程资料

- **Inside vLLM: Anatomy of a High-Throughput LLM Inference System**  
  来源：vLLM Blog，2025-09-05  
  链接：https://vllm.ai/blog/anatomy-of-vllm  
  推荐理由：最适合把论文里的概念和真实系统实现连起来。

- **vLLM 2024 Retrospective and 2025 Vision**  
  来源：vLLM Blog，2025-01-10  
  链接：https://blog.vllm.ai/2025/01/10/vllm-2024-wrapped-2025-vision  
  推荐理由：偏路线图，能快速看到社区和工业界把优化重点放在哪些方向上。

- **SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs**  
  来源：LMSYS Blog，2024-12-04  
  链接：https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/  
  推荐理由：工程含量高，尤其是 `zero-overhead batch scheduler` 和 `cache-aware load balancer` 两个点，非常贴近“近期进展”。

- **Achieving Faster Open-Source Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM)**  
  来源：LMSYS Blog，2024-07-25  
  链接：https://www.lmsys.org/blog/2024-07-25-sglang-llama3/  
  推荐理由：偏 benchmark 和系统设计，适合理解 SGLang 的 runtime 取舍。

- **NVIDIA TensorRT-LLM Now Accelerates Encoder-Decoder Models with In-Flight Batching**  
  来源：NVIDIA Technical Blog，2024-12-11  
  链接：https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-accelerates-encoder-decoder-models-with-in-flight-batching/  
  推荐理由：`in-flight batching` 是工业系统里非常关键的一类在线批处理机制，这篇是看工程实践的好入口。

- **Run High-Performance LLM Inference Kernels from NVIDIA Using FlashInfer**  
  来源：NVIDIA Technical Blog，2025-06-13  
  链接：https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/  
  推荐理由：虽然更偏 kernel/operator，但 2025 年以后调度优化越来越需要和底层 kernel 能力协同看，这篇能帮你补齐这一层。

---

## 一个简化的“近期趋势图”

- **阶段一：基础 serving 原语成熟**  
  代表：Orca、PagedAttention、continuous batching。

- **阶段二：围绕 `prefill/decode` 的运行时调度成为主战场**  
  代表：Sarathi、Sarathi-Serve、DistServe。

- **阶段三：调度开始显式利用状态信息**  
  代表：Prompt Cache、Preble、cache-aware load balancer。

- **阶段四：从单实例走向跨实例重调度和迁移**  
  代表：Llumnix、ServerlessLLM。

- **阶段五：从单模型 serving 走向多模型与 agentic serving**  
  代表：Aegaeon、INFERCEPT。

我的判断：**最近最值得跟的两个方向** 是：

- `prefill/decode disaggregation` 及其进一步的大规模编排
- `cache-aware / state-aware scheduling`，尤其是 prefix sharing、KV locality、live migration、multi-model pooling

---

## 如果你要系统读一遍

### 路线 1：学术主线

1. Orca
2. PagedAttention
3. Sarathi / Sarathi-Serve
4. DistServe
5. Llumnix
6. Preble
7. INFERCEPT
8. Aegaeon

### 路线 2：工程主线

1. Inside vLLM
2. vLLM prefix caching 设计文档：https://docs.vllm.ai/design/prefix_caching.html
3. SGLang v0.4 blog
4. TensorRT-LLM in-flight batching blog
5. FlashInfer blog

---

## 备注

- 严格聚焦“LLM 推理调度”的综述目前并不多，更多是散落在系统论文、OSDI/SOSP/MLSys 论文和项目博客里。
- 如果你希望补一篇更宽泛的“推理系统/优化综述”，可以另外单独找 inference optimization survey；但对“调度”这个子问题，**读系统论文 + 读 vLLM/SGLang/TensorRT-LLM 工程博客** 往往更有效。
- 上面有两类材料我刻意混在一起：  
  一类是学术论文，适合看问题定义和方法；  
  一类是工程博客，适合看哪些想法已经真正进入主流 runtime。

