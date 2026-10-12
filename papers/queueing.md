**Queueing, Predictions, and LLMs: Challenges and Open Problems| 2025 Stochastic Systems**

# 术语
- FIFO
- Shortest Job First (SJF) 知道准确的长度
- Shortest Predicted Job First (SPJF)
- Preemptive Shortest Job First (PSJF)
- Preemptive Shortest Predicted Job First (PSPJF)
- Shortest Remaining Processing Time (SRPT)
- Shortest Predicted Remaining Processing Time (SPRPT)


# queueing 传统方式的策略
统方法在不知道任务长度时通常使用 FIFO，在知道确切任务长度时使用 SJF/SRPT。对于大模型推理，输出长度通常未知，需要通过预测模型估计。排队论研究中，为了分析预测型调度算法，通常假设给定真实长度 x，预测长度 y 服从某种数学分布，例如指数分布或均匀分布。
**open questions**
- 最优调度策略 Gittins 是否能实际使用

其他策略
- 1-bit策略
不去预测，直接将请求划分为短或者长，短的放在队列前面。需要考虑是否可以抢占。

# 大模型推理中目前所用到的预测方式
大致分为这几类：
## 长度预测(神经网络)
用LLM预测长度
- Response length perception and sequence scheduling: an llm-empowered llm inference pipeline.  (NeurIPS 2023)
用BERT预测长度
- S3: increasing GPU utilization during generative inference for higher throughput. (NeurIPS 2023)
## 直接分类：分为几类，每类都是一个区间
- Enabling efficient batch serving for LMaaS via generation length prediction. arXiv 
- Dynamollm: designing llm inference clusters for performance and energy efficiency.(HPCA 2025)
## 回归预测：
Power-aware deep learning model serving with μ-Serve   (OSDI 2024)
Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction arXiv 
## Learning-to-Rank approach：
短作业优先
- Efficient LLM Scheduling by Learning to Rank (arXiv 2024)
## Multi-Level Feedback Queue (MLFQ): FastServe

## 长度预测(历史输出长度的概率分布)
- Past-Future Scheduler for LLM Serving under SLA Guarantees (`ASPLOS 2025` [CCF-A])：针对 LLM serving 中请求输出长度和未来 KV Cache 占用未知的问题，提出了一种基于历史请求分布的预测调度方法。该方法预测请求未来生成长度，并估计加入新请求后的KV Cache峰值，从而动态决定是否将请求加入batch，在避免eviction和SLA violation的同时提升有效吞吐（goodput）。文章提出了 LightLLM 框架，实验重点可以关注输出长度预测、未来显存估计以及 SLA 约束下 goodput 优化。


## tail
在模型输出最外层再加一层网络，输出长度
- Don’t Stop Me Now: Embedding Based Scheduling for LLMs (ICLR 2025)
# 针对prefix的调度
目前没有