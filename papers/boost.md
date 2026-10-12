**Strongly Tail-Optimal Scheduling in the Light-Tailed M/G/1 | SIGMETRICS| CCF-B**

# 存在的问题

针对 heavy-tailed 和 light-tailed 任务长度分布，传统调度策略都存在不足：

- **FCFS（First Come First Serve）**：不会因为未来到达的请求而延迟当前请求，因此尾延迟（tail latency）较稳定，但无法优先处理短任务，平均响应时间较长
- **SRPT / SJF（Shortest Remaining Processing Time / Shortest Job First）**：优先处理短任务，可显著降低平均响应时间。但在 **light-tailed** 场景下，长任务会不断被未来到达的短任务抢占，导致尾延迟恶化。

针对上述问题，**Nudge**调度算法首次提出在FCFS基础上进行修改，以改善尾延迟：

- 将任务划分为 **短任务（short）** 和 **长任务（long）** 两类
- 当短任务到达时，允许它最多越过前面的 **K 个长任务**

理论上证明，经过合理参数设置后，Nudge的tail constant优于 FCFS。但是文章认为Nudge存在不足：
- 只将任务分为长或短台粗粒度了
- Nudge 只考虑了任务的到达顺序，没有考虑任务之间的到达时间间隔

# Boost 调度算法

任务 $s$ 在时刻 $a$ 到达后，Boost 不直接按照真实到达时间排序，而是根据 **boosted arrival time** 进行调度：


$$a' = a - b(s)$$

其中，$b(s)$ 为任务的 Boost 值。Boost 会优先调度 **boosted arrival time** 更早的任务。

- 当 $b(s)=0$ 时，Boost退化为FCFS。
- 当短任务具有更大的 $b(s)$ 时，其到达时间会被"提前"，因此会获得更高的调度优先级；$b(l)$ 越大，策略越接近SRPT。
- 不同的 $b(s)$ 对应不同的 Boost 策略，而论文提出的 $\gamma$ -Boost 则采用理论推导得到的最优 Boost 函数，在 light-tailed 场景下能够达到 **Strongly Tail-Optimal**。

## $\gamma$-Boost

Boost 本身是一类调度算法，其性能由 Boost 函数 $b(s)$ 决定。论文进一步推导出了理论最优的 Boost 函数，并将采用该函数的策略称为 $\gamma$-Boost。

在全信息场景下（任务大小已知），$\gamma$-Boost 的 Boost 函数为：

$$b_\gamma(s)=\frac{1}{\gamma}\log\left(\frac{1}{1-e^{-\gamma s}}\right)$$

对于每个任务，根据其大小计算对应的 $b_\gamma(s)$，再计算

$$a' = a - b_\gamma(s)$$

最后按照 **boosted arrival time** 从小到大进行调度。

由于短任务对应更大的 Boost 值，其到达时间会被提前得更多，因此能够优先获得服务。而长任务的 Boost 值较小，不会像 SRPT 那样被无限期推迟，从而在优先短任务和保护长任务之间取得平衡。论文证明 γ-Boost 能够达到 **Strongly Tail-Optimal**，即具有理论最优的尾延迟性能。