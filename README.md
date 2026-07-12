# 大模型推理学习笔记

这是一个关于大模型推理的学习笔记仓库，用来记录相关知识、阅读理解与实践过程中的整理和总结。

目前内容包括：

- [大模型推理基础](./notes/infer-basic.md)
- [大模型推理顶会](./notes/infer-top-papers.md)
- [vLLM 中 LLaMA 解析](./notes/vllm-llama-code.md)
- [LatencyPrism 解读](./notes/LatencyPrism.md) 没啥参考价值，可以不看，他们自己内部的工具，只开源了一小部分
- [调度领域近况](./notes/infer-field.md)
- [Aleksa Gordić---详解vLLM](./notes/vllm-anatomy-translated.md)


# 大模型推理顶会论文分享：

## 调度领域：

- 1.[NanoFlow](./papers/NanoFlow.md): Towards Optimal Large Language Model Serving Throughput  (`OSDI 2025` [CCF-A])：从设备内执行调度角度入手，把输入切成更小的 `nano-batches`，自动决定批大小、执行顺序和资源分配，尽量将计算、访存和通信overlap，来提升GPU的利用率，进而提升整体 serving throughput。 在当时可以实现比vLLM快4.18×，比TensorRT-LLM快1.91×。这里没和SGLang比较。文章中实验部分可以细看，损失模型的建立值得学习和复用。

- 2.[JITServe](./papers/JITServe.md): Just-In-Time Scheduling for Large Language Model Serving (`NSID 2026` [CCF-A])：从 SLO-aware 调度角度入手，针对不同类型 LLM 请求（latency-sensitive、deadline-sensitive、compound requests），通过请求分析预测剩余工作量和资源需求，并提出 GMAX 调度算法动态选择 batch，在满足 SLO 的同时最大化 serving goodput。同时支持多模型部署、动态抢占以及复杂 Agent 工作流。

- 3.[Past-Future Scheduler](./papers/Past-Future-Scheduler.md): Past-Future Scheduler for LLM Serving under SLA Guarantees (`ASPLOS 2025` [CCF-A])：针对 LLM serving 中请求输出长度和未来 KV Cache 占用未知的问题，提出了一种基于历史请求分布的预测调度方法。该方法预测 waiting/running 请求未来生成长度，并估计加入新请求后的KV Cache峰值，从而动态决定是否将请求加入batch，在避免eviction和SLA violation的同时提升有效吞吐（goodput）。文章提出了 LightLLM 框架，实验重点可以关注输出长度预测、未来显存估计以及 SLA 约束下 goodput 优化。





同时，我建立了一个微信群，方便大家交流、讨论和分享学习经验。有需要可以添加我的微信：whhsleeping
