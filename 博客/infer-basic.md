# 1. 文章定位

目前主流开源推理框架里，`vLLM` 与 `SGLang` 都很值得学习。考虑到 vLLM 社区活跃、资料丰富、工程实现公开透明，这篇文章会以 vLLM 所代表的技术路线为主线。

本文目标不是讲源码细节，而是帮你建立一张“推理系统技术地图”：

- 近几年最核心的推理优化技术是什么
- 每项技术是为了解决什么问题
- 这些技术在系统里是如何落地的

文章会尽量不陷入过深的算子细节，但每个部分都会给出可继续深入的资料。

其他推荐的系统性资料：

- `ai-infra-learning`：系统介绍大模型工程基础与实践
- 项目地址：https://github.com/cr7258/ai-infra-learning

> 声明：本文是个人学习总结，可能存在理解偏差，请辩证阅读。未经允许，不得转载。

---

# 2. 什么是大模型推理

## 2.1 大模型是什么

从架构演进看，Transformer（出自 *Attention Is All You Need*）是现代大模型的基础。常见结构可以粗分为三类：

- `Encoder-only`
- `Decoder-only`
- `Encoder-Decoder`

当前主流生成式大模型主要采用 `Decoder-only`。核心原因是它更适合自回归生成：每次基于已有上下文预测下一个 token。

推荐阅读：

- 论文：*Attention Is All You Need*  
  https://arxiv.org/abs/1706.03762

## 2.2 大模型推理是什么

推理（Inference）可以理解为：模型在参数固定的前提下，接收输入并生成输出。

和训练相比，推理有几个关键差异：

- 训练要反向传播并更新参数；推理只做前向计算
- 训练关注收敛和总吞吐；推理更关注用户体验与服务稳定性
- 推理请求长度、到达时间、输出长度高度不确定，调度难度更高

对生成式模型来说，推理通常分两阶段：

- `Prefill`（输入阶段，先处理完整 prompt 并建立KV缓存）
- `Decode`（生成阶段，逐 token 输出答案）

这两个阶段的计算特征并不一样：`prefill` 更像“把整段输入一次性处理完”，通常更适合大块计算；`decode` 更像“每次只往前走一小步，但要重复很多轮”，所以虽然单步更轻，却对交互体验更敏感。

对应的核心指标：

- `TTFT (Time To First Token)`：首 token 延迟
- `TPOT (Time Per Output Token)`：每个输出 token 的平均延迟
- `Throughput`：单位时间处理 token 或请求数
- `Tail Latency (P95/P99)`：最慢那批请求的延迟表现。比如 `P95 = 2秒`，表示 95% 的请求能在 2 秒内完成，剩下 5% 会更慢。这个指标越高，说明系统越容易出现“少数用户等很久”的情况。
- `Memory Efficiency`：显存利用率（尤其 KV Cache）

一句话总结：推理系统优化，本质是在“延迟-吞吐-显存-公平性”之间做工程权衡。

推荐阅读：

- 博客：*Inside vLLM: Anatomy of a High-Throughput LLM Inference System*  
  https://vllm.ai/blog/anatomy-of-vllm
- 论文：*Orca: A Distributed Serving System for Transformer-Based Generative Models*  
  https://www.usenix.org/conference/osdi22/presentation/yu

---

# 3. 模型推理如何优化

这一部分按两条线展开：

- `Model-side`：模型结构与算子相关优化
- `System-side`：服务系统与调度相关优化

## 3.1 Model-side：如何服务好一个模型

参数规模极大的 Transformer 中，最关键的计算热点通常在：

- Attention
- MLP
- KV Cache（跨 token 持续占用显存的历史状态缓存）

这里有一个很容易被忽略、但非常重要的前提：在自回归生成里，模型每生成一个新 token，都要依赖前面所有 token 的历史信息。如果每次都从头重算，代价会非常高，所以系统会把这些历史计算结果缓存下来，这就是 KV Cache。也正因为如此，KV Cache 会直接成为推理系统里的核心资源。

为什么模型侧优化主要集中在这三块？

- Attention 会随着上下文变长而越来越重，尤其容易带来中间结果和访存压力
- MLP 在很多模型里占据了大量计算量，往往是纯算力热点
- KV Cache 会随着“上下文长度 x 并发请求数”一起增长，直接影响显存上限

当上下文变长时，这三个问题往往会同时变严重：Attention 计算更重，KV Cache 膨胀更快，系统可承载并发也更容易下降。所以很多推理系统问题，在长上下文场景下会被进一步放大。

换句话说，模型侧优化主要是在解决两件事：一是“算得更快”，二是“占得更省”。

### 3.1.1 如何管理 KV Cache

`PagedAttention` 借鉴了操作系统分页思想，把 KV Cache 按固定大小的小块来管理，减少显存碎片。

如何实现（简述）：

- 不再要求一条请求的整段 KV 缓存连续放在一大片显存里
- 而是把它拆成很多固定大小的小块，需要多少就分配多少
- 系统额外维护一张映射表，记录“这一段内容现在放在哪些小块里”
- 请求结束后，这些小块可以单独回收，也更容易复用给别的请求

工程意义：它把“很难管理的一整段大缓存”变成了“很多可灵活调度的小块缓存”，所以更省显存，也更适合高并发场景。

推荐阅读：

- 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention*  
  https://arxiv.org/abs/2309.06180
- 博客：*Inside vLLM: Anatomy of a High-Throughput LLM Inference System*  
  https://vllm.ai/blog/anatomy-of-vllm

### 3.1.2 Attention 机制优化

自回归生成时，序列越长，Attention 的中间结果就越大。很多时候真正拖慢速度的，不是公式本身，而是这些中间结果需要反复写入显存、再从显存读回来。`FlashAttention` 的关键价值就在这里：它主要优化的是“数据搬运”，而不是改掉 Attention 的数学定义。

如何实现（简述）：

- 不一次性处理完整的 attention 矩阵，而是拆成很多小块分批计算
- 每算完一小块，就尽量立刻在 GPU 更快的片上缓存里继续完成后续步骤
- 不再把那张很大的中间分数矩阵完整地写回显存
- 这样就少了很多“写到显存再读回来”的动作

工程意义：`FlashAttention` 的提升主要来自“少搬中间结果”，因此通常既能提速，也能降低显存压力。

推荐阅读：

- 论文：*FLASHATTENTION: Fast and Memory-Efficient Exact Attention with IO-Awareness*  
  https://arxiv.org/abs/2205.14135
- 论文：*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*  
  https://arxiv.org/abs/2307.08691

### 3.1.3 MLP 优化（以 MoE 为例）

在很多模型里，MLP 同样是主要算力开销来源。`MoE (Mixture of Experts)` 的核心是“稀疏激活”：不是每次都让全部参数参与计算，而是只让少数几个“专家”处理当前 token。

如何实现（简述）：

- 先增加一个很小的路由模块，判断当前 token 更适合交给哪些专家处理
- 每个 token 只会被分配给少数几个专家，而不是所有专家
- 系统把去往同一专家的 token 聚在一起计算，再把结果拼回原顺序
- 为了避免某几个专家总是过载，训练和服务时通常还要做负载均衡

工程挑战：

- token 在专家之间分发和回收结果，会带来额外通信与重排开销
- 如果很多 token 总是挤到少数专家上，就会出现热点和长尾延迟问题

参考：

- HuggingFace（中文）：https://huggingface.co/blog/zh/moe
- 论文：*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*  
  https://arxiv.org/abs/2101.03961
- DeepSpeed MoE Inference Tutorial：https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/

## 3.2 分布式中如何加速计算

当模型或服务规模继续变大时，单卡通常会先遇到三类问题：

- 模型参数太大，一张卡放不下
- KV Cache 太大，并发一上来显存就吃紧
- 即使能放下，单卡吞吐也不够，扛不住线上流量

这时就需要并行化。可以把不同并行方式理解成：它们都在“把大任务拆开”，但拆分角度不一样。

常见并行方式：

- `Tensor Parallelism (TP)`：按张量维度切分算子
- `Pipeline Parallelism (PP)`：按层切分模型并流水执行
- `Data Parallelism (DP)`：多副本并行处理不同请求
- `Expert Parallelism (EP)`：主要用于 `MoE`，把不同专家分布到不同设备上

如何实现（简述）：

- `TP` 的思路是把同一层的计算拆到多张卡上，算完后再把结果合起来
- `PP` 的思路是把不同层放到不同设备上，让多个请求像流水线一样依次经过这些层
- `DP` 的思路是直接复制出多个完整模型副本，让不同请求去不同副本上执行
- `EP` 的思路是把不同专家放到不同设备上，让每张卡只负责部分专家，而不是容纳整套专家参数

需要注意：

- `TP` 更适合解决“单层太大，一张卡算不下”的问题
- `PP` 更适合解决“模型层数太多，整模型放不下”的问题
- `DP` 更适合解决“模型能放下，但流量太大，需要复制副本扩吞吐”的问题
- `EP` 更适合解决“MoE 专家太多，单卡放不下全部专家”的问题

对于普通稠密模型，通常重点关注 `TP / PP / DP`。对于 `MoE` 模型，`EP` 也很关键，因为它直接决定专家如何分布以及 token 如何在专家之间流动。这些并行策略主要解决“算力与容量扩展”，在线体验（TTFT、P99）仍高度依赖上层调度。

参考资料：

- Megatron Core Parallelism Guide：https://docs.nvidia.com/megatron-core/developer-guide/0.17.0/user-guide/parallelism-guide.html
- DeepSpeed MoE Inference Tutorial：https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/
- Megatron Core MoE 文档：https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html

## 3.3 System-side：推理系统如何优化输入与执行

如果把一个请求放到真实推理系统里看，它大致会经历这样一条生命周期：

- 请求先进入等待队列
- 调度器决定它什么时候进入 batch
- 先执行 `prefill`，把输入 prompt 处理完并建立 KV Cache
- 再进入 `decode` 循环，逐 token 生成输出
- 请求结束后，系统再回收它占用的 KV Cache 和调度状态

这样看就比较容易理解：`batching` 决定“哪些请求一起跑”，`scheduling` 决定“谁先跑、跑多久”，`prefix caching` 决定“哪些计算可以不必重做”。

### 3.3.1 Batching 策略

为什么推理系统几乎一定会做 batching？

- 因为单个请求往往喂不满 GPU，如果每次只跑一个请求，很多算力会被浪费
- 但 batching 也不是越大越好；batch 越大，吞吐通常越高，但等待拼批的时间也可能越长
- 所以 batching 的核心问题不是“要不要拼”，而是“怎么拼，拼到什么程度”

可以把常见 batching 方式理解成一条逐步演进的路线：

- `Static Batching`：先攒够固定数量的请求，再一起执行。实现最简单，但对线上波动流量不友好；请求来得快时可能还行，请求来得慢时就容易出现“为了等人上车而空耗时间”的情况。
- `Dynamic Batching`：不再死等固定批大小，而是在一个时间窗内尽量多收集请求，再一起执行。它比 static 更灵活，但本质上仍然是在“先等一会，再开一批”。
- `Continuous Batching`：这是现在最重要的一类做法。系统不会等当前 batch 全部结束后再装下一批，而是让已经在生成中的请求继续跑，同时把新请求持续插入进来。这样 GPU 更不容易空转，也更适合生成式模型这种“每个请求长度都不一样”的场景。
- `Selective Batching`：当请求差异很大时，系统还会进一步按请求特征分桶，比如把长度相近、阶段相近的请求更倾向于放在一起，减少彼此干扰。

如何实现（简述）：

- 最简单的实现是：设一个 batch 大小，攒满再执行
- 再往前一步，就是允许系统在一个很短的等待窗口里动态拼批
- 到 continuous batching 时，调度器会持续观察等待队列，只要 GPU 还有空余，就把合适的新请求塞进当前运行中的 batch
- 对生成式模型来说，这样做特别重要，因为不同请求的输出长度不同；如果非要整批一起进、一起出，短请求就会被长请求拖住

如果只记一个结论，这一节最值得记住的是：`continuous batching` 之所以重要，不是因为它名字新，而是因为它更符合 LLM 推理“请求到达时间不同、输出长度不同、decode 过程持续很久”这几个现实特点。

参考资料：

- Orca（continuous batching 的经典早期系统）：https://www.usenix.org/conference/osdi22/presentation/yu
- Inside vLLM：https://vllm.ai/blog/anatomy-of-vllm
- NVIDIA TensorRT-LLM In-Flight Batching Blog：https://developer.nvidia.com/blog/nvidia-tensorrt-llm-now-accelerates-encoder-decoder-models-with-in-flight-batching/

### 3.3.2 调度核心：Prefill 与 Decode

推理调度最核心的矛盾之一：

- `Prefill` 计算密集、单次开销大
- `Decode` 单步短但次数多，对交互体验高度敏感

这背后其实对应着一个更普遍的系统矛盾：吞吐和延迟经常天然冲突。

- 如果想把吞吐做高，系统往往会倾向于多攒一些请求再一起跑
- 如果想把延迟做低，系统又会倾向于尽快响应眼前的请求
- `Decode` 之所以更敏感，是因为它直接决定用户是否能持续看到输出；哪怕单步只慢一点，累积起来体感也会很明显

如何实现（简述）：

- 把很重的 prefill 拆成更小的片段，避免它一次占住 GPU 太久
- 调度器通常会优先保证 decode 有稳定的执行机会，因为它更直接影响用户体感
- 在更复杂的系统里，还会把 prefill 和 decode 放到不同实例上，减少互相干扰

这类策略本质上是在控制“谁先占 GPU、占多久、何时让出”。

这里有两个特别有代表性的方向：

`Chunked Prefill`：

- 它的核心思路不是让 prefill 变少，而是让 prefill 不要一次占用 GPU 太久
- 做法是把一个很长的输入拆成多个较小片段，分几轮完成
- 这样 decode 请求就有机会在这些片段之间穿插执行，减少“首字出来了，后面却突然卡住”的情况

`Disaggregated Prefill/Decode`：

- 如果同一套资源里同时做 prefill 和 decode，二者还是会相互干扰
- 更进一步的系统会把它们放到不同资源池，甚至不同实例上
- 这样做的好处是：prefill 可以追求更高吞吐，decode 可以单独追求更稳的交互延迟
- 代价是系统架构会更复杂，请求和缓存的流转也更难管理

参考资料：

- Sarathi-Serve：https://arxiv.org/abs/2403.02310
- DistServe：https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- P/D-Serve：https://arxiv.org/abs/2408.08147

### 3.3.3 Prefix Caching 与 Cache-aware Scheduling

`Prefix Caching`：

- 核心思路：复用共享前缀的 KV，跳过重复 prefill
- 很多请求的开头部分其实是一样的，比如系统提示词或公共上下文
- 系统会识别这些“已经算过的前缀”，把对应缓存记下来
- 下次再遇到相同前缀时，直接复用已有结果，而不是从头再算一遍

它不只是“省一点算力”，还会改变调度逻辑。因为当两个请求都在排队时，那个能直接命中前缀缓存的请求，往往可以更快完成，所以系统有时会显式利用这种缓存命中信息来做调度或路由。

这也引出了另一个很有代表性的方向：`Cache-aware Scheduling / Routing`。

- 传统调度更像只看队列长度、优先级、显存是否够
- cache-aware 调度会进一步看“这个请求能不能复用已有缓存”
- 如果某个请求命中前缀缓存，它的实际成本就更低，系统可能会优先把它放到能命中缓存的实例上
- 所以在更先进的系统里，缓存不只是局部优化手段，还会成为调度决策的一部分

参考资料：

- vLLM Prefix Caching 设计文档：https://docs.vllm.ai/en/stable/design/prefix_caching.html
- Preble：https://arxiv.org/abs/2407.00023

### 3.3.4 Speculative Decoding

`Speculative Decoding`：

- 核心思路：先由小模型（draft）生成多个候选 token
- 再由大模型一次性检查这些候选 token 是否合理
- 如果大模型认可，就一次接收多个 token；如果不认可，再退回到分歧点继续正常生成

它和 prefix caching 的区别在于：prefix caching 是“不要重复做已经做过的计算”，speculative decoding 是“先用便宜路径猜一部分，再让大模型批量确认”，两者思路不同，但目标都是减少无效等待。

参考资料：

- Speculative Sampling：https://arxiv.org/abs/2302.01318
- Draft & Verify：https://arxiv.org/abs/2309.08168

### 3.3.5 Admission Control 与 Fairness

线上服务不只追求平均性能，还要考虑稳定性与多租户体验。

如何实现（简述）：

- Admission Control：当队列太长或资源太紧时，不让新请求无限制涌入，而是限流、排队或降级
- Fairness：避免某些大请求或高频用户长期占住资源，让其他请求一直等不到机会
- Tail 控制：对特别长、特别重的请求做隔离或限额，避免它们把整体延迟拖坏

因此，推理系统优化最终是一个服务治理问题，而不仅是算子问题。

参考资料：

- Fairness in Serving Large Language Models：https://arxiv.org/abs/2401.00588
- Llumnix：https://arxiv.org/abs/2406.03243
