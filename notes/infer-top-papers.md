LLM 推理调度阅读提纲

# 1. 这个方向在研究什么

核心不是模型本身，而是 `serving runtime` 在做哪几件具体事：

- 怎么把请求拼成批次
- 怎么管理 KV cache
- 怎么协调 prefill / decode
- 怎么在多模型 / 多 adapter 之间共享 GPU
- 怎么把调度粒度继续下沉到执行层，或上提到工作流

所以读文献时，最好不要按空泛主题分，而要按“它到底改了哪一步操作”来分。

---

# 2. 按具体操作分点

## 2.1 把请求拼成批次

这是最早的一层调度问题：谁先上 GPU，什么时候插入新请求，如何减少队首阻塞。

- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) — **OSDI 2022**  
  关键词：`iteration-level scheduling`、`selective batching`  
  它回答的是：生成式模型为什么不能沿用传统 request-level batching。
- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) — **OSDI 2024**  
  关键词：`chunked prefill`、`stall-free scheduling`  
  它回答的是：怎么在不把 decode 卡死的情况下插入新的长 prompt。
- [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao) — **OSDI 2024**  
  关键词：`runtime rescheduling`、`migration`  
  它回答的是：队列已经排坏了之后，能不能把活请求迁走重排。
- [SOLA](https://proceedings.mlsys.org/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html) — **MLSys 2025**  
  关键词：`TTFT/TPOT balancing`、`state-aware scheduling`  
  它回答的是：调度目标不只是吞吐，还包括 SLO。

## 2.2 把 KV cache 管起来

这是最核心的一层。不是“顺便利用缓存”，而是 `cache` 本身变成了调度对象。

- [vLLM / PagedAttention](https://doi.org/10.1145/3600006.3613165) — **SOSP 2023**  
  关键词：`KV paging`、`continuous batching`  
  这是行业基石，应该放在最前面读。它回答的是：KV cache 为什么必须像操作系统分页一样管理。
- [Preble](https://openreview.net/forum?id=meKEKDhdnx) — **ICLR 2025**  
  关键词：`prefix locality`、`distributed prompt scheduling`  
  它回答的是：prefix 复用一旦进入分布式系统，路由和调度要怎么改。
- [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) — **FAST 2025**  
  关键词：`KVCache-centric architecture`、`memory hierarchy`  
  它回答的是：如果把 KV 当成核心资产，整个平台应该怎么围着它设计。
- [EPIC](https://proceedings.mlr.press/v267/hu25j.html) — **ICML 2025**  
  关键词：`position-independent caching`  
  它回答的是：cache 复用能不能从“公共前缀”扩展到更一般的 chunk 复用。

## 2.3 把 prefill 和 decode 拆开

这是第二条主线。因为这两个阶段的资源需求完全不同，所以后来大家开始物理解耦。

- [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) — **OSDI 2024**  
  关键词：`P/D disaggregation`、`goodput`  
  它回答的是：prefill 和 decode 为什么不该共用同一组 GPU。
- [InfiniGen](https://www.usenix.org/conference/osdi24/presentation/lee) — **OSDI 2024**  
  关键词：`dynamic KV offloading`  
  它回答的是：拆开之后，KV fetch / offload 的代价怎么控制。
- [FlowKV](https://arxiv.org/abs/2504.03775)  
  关键词：`KV transfer scheduling`  
  它回答的是：如果瓶颈变成 KV 传输，调度目标也要跟着改。
- [P/D-Serve](https://arxiv.org/abs/2408.08147)  
  关键词：`large-scale P/D orchestration`  
  它回答的是：P/D 解耦放到更大规模集群里，编排问题会怎么变化。
- [Kairos](https://arxiv.org/abs/2605.02329)  
  关键词：`prefill urgency`、`decode slack`  
  它回答的是：如果还不想彻底物理解耦，能不能先在调度策略上显式区分两类 phase。

## 2.4 把多模型 / 多 adapter 塞进同一个资源池

这是平台化之后的问题：不是只服务一个模型，而是几十个模型、几千个 adapter。

- [MuxServe](https://proceedings.mlr.press/v235/duan24a.html) — **ICML 2024**  
  关键词：`multi-model multiplexing`  
  它回答的是：多个模型怎样共用 GPU 的算力和显存。
- [Punica](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html) — **MLSys 2024**  
  关键词：`multi-tenant LoRA serving`  
  它回答的是：不同 LoRA adapter 能不能一起 batch。
- [SLoRA](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html) — **MLSys 2024**  
  关键词：`thousands of adapters`、`unified memory pool`  
  它回答的是：adapter 数量上去之后，内存池怎么统一管理。
- [Aegaeon](https://dl.acm.org/doi/10.1145/3731569.3764815) — **SOSP 2025**  
  关键词：`token-level autoscaling`、`GPU pooling`  
  它回答的是：模型市场里 bursty workload 下，GPU 池化怎么做。
- [Chameleon](https://research.ibm.com/publications/chameleon-adaptive-caching-and-scheduling-for-many-adapter-llm-inference-environments) — **MICRO 2025**  
  关键词：`adapter caching`、`adapter-aware scheduling`  
  它回答的是：adapter load latency 能不能直接被调度器显式利用。
- [ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu) — **OSDI 2024**  
  关键词：`checkpoint locality`、`startup-aware scheduling`  
  它回答的是：平台调度除了在线请求，还要不要把模型冷启动成本一起算进去。

## 2.5 把调度继续往执行层压

当请求级调度做得差不多，问题就会继续下沉到长上下文、MoE、激活迁移这些更细粒度的执行问题。

- [LoongServe](https://arxiv.org/abs/2404.09526)  
  关键词：`elastic sequence parallelism`
- [Medha](https://arxiv.org/abs/2409.17264)  
  关键词：`fine-grained preemption`
- [Janus](https://arxiv.org/abs/2512.13525)  
  关键词：`MoE execution disaggregation`
- [FinDEP](https://arxiv.org/abs/2512.21487)  
  关键词：`expert parallel scheduling`

这里先知道问题在往下沉就够了，不用一开始读太深。

## 2.6 把调度往上提到 Agent / workflow

这是最新的一条线。我建议先知道方向，不建议前期投入太多时间。

- [Autellix](https://arxiv.org/abs/2502.13965)
- [SAGA](https://arxiv.org/abs/2605.00528)

它们关心的已经不是单个请求，而是多步工作流里的 KV 复用和执行编排。

---

# 3. 推荐阅读顺序

如果想抓主线，但又不想太薄，我建议读下面 16 篇：

1. [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
2. [vLLM / PagedAttention](https://doi.org/10.1145/3600006.3613165)
3. [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
4. [Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao)
5. [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
6. [Preble](https://openreview.net/forum?id=meKEKDhdnx)
7. [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin)
8. [InfiniGen](https://www.usenix.org/conference/osdi24/presentation/lee)
9. [EPIC](https://proceedings.mlr.press/v267/hu25j.html)
10. [FlowKV](https://arxiv.org/abs/2504.03775)
11. [MuxServe](https://proceedings.mlr.press/v235/duan24a.html)
12. [Punica](https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html)
13. [SLoRA](https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html)
14. [Chameleon](https://research.ibm.com/publications/chameleon-adaptive-caching-and-scheduling-for-many-adapter-llm-inference-environments)
15. [ServerlessLLM](https://www.usenix.org/conference/osdi24/presentation/fu)
16. [Aegaeon](https://dl.acm.org/doi/10.1145/3731569.3764815)

如果时间更少，压缩成 6 篇也行：

1. Orca
2. vLLM / PagedAttention
3. Sarathi-Serve
4. DistServe
5. Preble
6. Aegaeon

这 6 篇基本就把这条线从“怎么 batch”到“怎么管 cache”再到“怎么做平台化资源池”串起来了。

如果你想继续加，但又不想失控，我建议上限控制在 20 篇以内，再往上就会开始失焦。

---

# 5. 近期工程博客里最值得补的

如果只补工程材料，我建议看这几篇：

- [vLLM Router](https://blog.vllm.ai/2025/12/13/vllm-router-release.html) : router statefulness
- [Inside vLLM's New KV Offloading Connector](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)  : KV offloading
- [vLLM Large Scale Serving](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) : large-scale disaggregated serving
- [HiSparse](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/) ：long-context memory hierarchy
- [Optimizing GLM4-MoE for Production](https://www.lmsys.org/blog/2026-01-21-novita-glm4/) :MoE production optimization
- [DeepSeek-V4 on Day 0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/) : sparse / MoE / overlap / PD engineering convergence



