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


| **Past-Future Scheduler for LLM Serving under SLA Guarantees** | ASPLOS 2025 | A | 结合历史模式和未来预测的 SLA 保障 |
| **TAPAS: Thermal- and Power-Aware Scheduling in Cloud Platforms** | ASPLOS 2025 | A | 热/功耗感知调度，区分 prefill/decode |
| **SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads** | NSDI 2025 | A | 不可预测负载的细粒度推理服务 |

| **WaferLLM: Large Language Model Inference at Wafer Scale** | OSDI 2025 | A | 首个晶圆级 LLM 推理系统（Cerebras WSE-2），PLMR 模型 + MeshGEMM，比 A100 集群快 10-20× (Edinburgh, MSR) |
| **BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching** | OSDI 2025 | A | Serverless 模型弹性扩缩，GPU 计算网络替代 host 缓存，层级别活迁移，尾延迟降低 94% (SJTU, Huawei) |
| **DecDEC: A Systems Approach to Advancing Low-Bit LLM Quantization** | OSDI 2025 | A | 3/4-bit 低比特量化质量提升，动态获取残差矩阵改善 on-device 推理 (Seoul National Univ) |


同时，我建立了一个微信群，方便大家交流、讨论和分享学习经验。有需要可以添加我的微信：whhsleeping
