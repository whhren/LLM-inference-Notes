# 大模型推理学习笔记

记录大模型推理相关的知识整理、论文阅读、源码分析与实践实验。

主要关注 LLM Inference、LLM Serving、请求调度与推理性能优化，通过阅读论文、分析开源框架和动手实验，理解大模型推理系统的设计原理与优化方法。

## 基础知识

- [大模型推理基础](./notes/infer-basic.md)：整理大模型推理的基本原理与核心技术。
- [大模型推理流程](./notes/infer-process.md)：从矩阵规模变化的角度梳理大模型推理流程。
- [vLLM 中 LLaMA 解析](./notes/vllm-llama-code.md)：分析 vLLM 中 LLaMA 模型的实现。
- [Anatomy of vLLM：vLLM 详解](./notes/vllm-anatomy-translated.md)：学习 vLLM 的系统设计与实现原理。
- [大模型推理领域近况](./notes/infer-field.md)：整理推理系统领域的技术进展。

# 论文阅读
整理大模型推理系统中的请求调度与性能优化研究，重点关注输出长度预测、SLO 约束、尾延迟优化与请求切分等方向。

## 推理调度相关


**基于输出长度预测的调度**

这类方法尝试预测请求的输出长度或剩余工作量，并据此调整请求优先级或决定请求准入，以改善服务效率。

- [Past-Future Scheduler](./papers/Past-Future-Scheduler.md)：Past-Future Scheduler for LLM Serving under SLA Guarantees（ASPLOS 2025，CCF-A）。基于历史请求分布预测未来 Token 需求，并结合 KV Cache 容量约束请求准入。

- [TIE](./papers/TIE.md)：Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions（ICML 2026，CCF-A）。使用 Log-t 分布建模请求的输出长度，通过 DeBERTa 预测分布参数，并结合期望与 CVaR 评估长尾风险，动态调整请求优先级，以降低队头阻塞并改善推理性能。


- [JITServe](./papers/JITServe.md)：Just-In-Time Scheduling for Large Language Model Serving（NSDI 2026，CCF-A）。面向不同类型的请求与服务目标，预测剩余工作量和资源需求，通过 GMAX 调度算法动态选择 Batch，以提升满足 SLO 的有效吞吐量。

- [SageSched](./papers/SageSched.md):SageSched: Efficient LLM Scheduling Confronting Demand Uncertainty and Hybridity（NeurIPS 2026，CCF-A）。文章使用历史请求分布，获取当前这个请求的输出长度分布，再通过建模获取计算成本分布，最后通过Gittins准则计算指数，确定请求执行优先级。

- [PiLLM](./papers/PiLLM.md):PiLLM: Resource-Efficient LLM Inference Using Workload Prediction（EuroSys 2026，CCF-A）。通过历史请求长度统计预测 Batch的计算和显存需求，并将其转换为 FLOPs 和 Memory 估计。在此基础上，分别进行 GPU 实例内的 Batch 级显存管理，以及跨 GPU 实例的资源规划和动态请求分配，从而提高 GPU 利用率并保证 SLO。

**从排队理论出发的调度策略**

- [Beyond Prediction](./papers/Beyond-Prediction.md)：Beyond Prediction: Tail-Aware Scheduling for LLM Inference（ICML 2026，CCF-A）。研究面向尾延迟的调度方法，使用$\gamma$ -boost算法，能让调度器在先来先到和短作业优先策略中间不断平衡。不过没有源码，具体实现比较模糊，很多细节都没有。

- [FastServe](./papers/fastserve.md):FastServe: Iteration-Level Preemptive Scheduling for Large Language Model  Inference（NSDI 2026， ccfa）。文章在请求级别调度层面引入多级反馈队列，和KV缓存管理的策略。文章比较老，23年就写完了，实验中比较的框架也比较老。引入多级反馈队列了，会出现显存问题，感觉没太大意义。

**面向特定推理阶段的调度**

这类方法针对特定模型或推理阶段的特点设计专门的调度策略。

- [PASCAL](./papers/PASCAL.md)：A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Model（HPCA 2026，CCF-A）。针对推理模型的 Reasoning 与 Answering 阶段，研究阶段感知的请求调度策略。

**基于Batch切分的调度**

这类方法通过将请求或执行任务切分为更细粒度的单元，优化 GPU 资源利用率与整体吞吐量。

- [NanoFlow](./papers/NanoFlow.md)：Towards Optimal Large Language Model Serving Throughput（OSDI 2025，CCF-A）。将输入请求切分为更小的 nano-batches，联合优化批大小、执行顺序与资源分配，以提升 GPU 利用率和整体 Serving 吞吐量。



# 源码分析与实践

**1. LightLLM 源码分析**

- [LightLLM 调度器源码分析：基于历史请求长度分布的调度机制](./Lightllm/lightllm.md)：结合 Past-Future Scheduler 相关研究，分析请求分组、Batch 构建与 Admission Control，重点研究历史输出长度分布、未来 Token 需求估计及 KV Cache 容量约束。

**2. LightLLM 调度实验**

- [LightLLM 调度源码分析与实践：基于历史请求长度分布的调度优化](./Lightllm/LightLLM-sche.md)：修改 LightLLM 的请求调度策略，对比 FCFS 与 SJF 在不同请求到达速率下的表现，分析吞吐量、平均延迟、TTFT、ITL 与不同分位数延迟，并结合请求输出长度分布讨论实验结果。


## 交流与反馈
欢迎交流大模型推理系统、推理框架实现与请求调度相关的技术问题。

建立了一个微信群，方便大家交流与分享学习经验。有需要可以添加微信：whhsleeping。