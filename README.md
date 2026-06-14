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

- 1.[NanoFlow](./papers/NanoFlow.md): Towards Optimal Large Language Model Serving Throughput  (`OSDI 2025` [CCF-A])：从设备内执行调度角度入手，把输入切成更小的 `nano-batches`，自动决定批大小、顺序和资源分配，尽量将计算、访存和通信overlap，来提升GPU的利用率，进而提升整体 serving throughput。 在当时可以实现比vLLM快4.18×，比TensorRT-LLM快1.91×。这里没和SGLang比较。文章中实验部分可以细看，损失模型的建立值得学习和复用。

- 2.

同时，我建立了一个微信群，方便大家交流、讨论和分享学习经验。有需要可以添加我的微信：whhsleeping
