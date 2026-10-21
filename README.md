# 大模型推理学习笔记

记录大模型推理相关的知识整理、论文阅读、源码分析与实践实验。

主要关注 LLM Inference、LLM Serving、请求调度与推理性能优化，通过阅读论文、分析开源框架和动手实验，理解大模型推理系统的设计原理与优化方法。

## 基础知识

- [大模型推理基础](./notes/infer-basic.md)：整理大模型推理的基本原理与核心技术。
- [大模型推理流程](./notes/infer-process.md)：从矩阵规模变化的角度梳理大模型推理流程。
- [vLLM 中 LLaMA 解析](./notes/vllm-llama-code.md)：分析 vLLM 中 LLaMA 模型的实现。
- [Anatomy of vLLM：vLLM 详解](./notes/vllm-anatomy-translated.md)：学习 vLLM 的系统设计与实现原理。
- [大模型推理领域近况](./notes/infer-field.md)：整理推理系统领域的技术进展。

## 论文阅读：推理调度

整理大模型推理系统中的请求调度与性能优化研究，重点关注输出长度预测、SLO 约束、尾延迟优化与请求切分等方向。

**1. 基于输出长度预测的调度**

这类方法尝试预测请求的输出长度或剩余工作量，并据此调整请求优先级或决定请求准入，以改善服务效率。

- [Past-Future Scheduler](./papers/Past-Future-Scheduler.md)：Past-Future Scheduler for LLM Serving under SLA Guarantees（ASPLOS 2025，CCF-A）。基于历史请求分布预测未来 Token 需求，并结合 KV Cache 容量约束进行请求准入，优化 SLA 约束下的有效吞吐量（goodput）。
- [JITServe](./papers/JITServe.md)：Just-In-Time Scheduling for Large Language Model Serving（NSDI 2026，CCF-A）。面向不同类型的请求与服务目标，预测剩余工作量和资源需求，通过 GMAX 调度算法动态选择 Batch，以提升满足 SLO 的有效吞吐量。

**2. 基于排队论的调度**

这类方法借鉴经典排队理论与操作系统调度思想，并结合 LLM 推理的执行特点设计调度策略。

- [Beyond Prediction](./papers/Beyond-Prediction.md)：Beyond Prediction: Tail-Aware Scheduling for LLM Inference（ICML 2026，CCF-A）。研究面向尾延迟的调度方法，通过调节请求工作量对调度优先级的影响，在不同调度目标之间进行权衡。

**3. 面向特定推理阶段的调度**

这类方法针对特定模型或推理阶段的特点设计专门的调度策略。

- [PASCAL](./papers/PASCAL.md)：A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Model（HPCA 2026，CCF-A）。针对推理模型的 Reasoning 与 Answering 阶段，研究阶段感知的请求调度策略。

**4. 基于请求切分的调度**

这类方法通过将请求或执行任务切分为更细粒度的单元，优化 GPU 资源利用率与整体吞吐量。

- [NanoFlow](./papers/NanoFlow.md)：Towards Optimal Large Language Model Serving Throughput（OSDI 2025，CCF-A）。将输入请求切分为更小的 nano-batches，联合优化批大小、执行顺序与资源分配，以提升 GPU 利用率和整体 Serving 吞吐量。

## 源码分析与实践

**1. LightLLM 源码分析**

- [LightLLM 调度器源码分析：基于历史请求长度分布的调度机制](./Lightllm/lightllm.md)：结合 Past-Future Scheduler 相关研究，分析请求分组、Batch 构建与 Admission Control，重点研究历史输出长度分布、未来 Token 需求估计及 KV Cache 容量约束。

**2. LightLLM 调度实验**

- [LightLLM 调度源码分析与实践：基于历史请求长度分布的调度优化](./Lightllm/LightLLM-sche.md)：修改 LightLLM 的请求调度策略，对比 FCFS 与 SJF 在不同请求到达速率下的表现，分析吞吐量、平均延迟、TTFT、ITL 与不同分位数延迟，并结合请求输出长度分布讨论实验结果。

## 交流与反馈

欢迎交流大模型推理系统、推理框架实现与请求调度相关的技术问题。

建立了一个微信群，方便大家交流与分享学习经验。有需要可以添加微信：whhsleeping。