# 从矩阵维度变化理解大模型推理：以 Llama 3.2 1B 为例

这篇笔记从一个真实的小模型出发，沿着输入、Attention、KV Cache的顺序，梳理推理过程中矩阵的形状变化过程。


# 1. 模型与示例

我们使用 Llama 3.2 1B 作为例子。其主要结构参数如下：

| 参数 | 数值 |
|---|---:|
| Hidden Size | 2048 |
| Transformer 层数 | 16 |
| Query Heads | 32 |
| KV Heads | 8 |
| Head Dimension | 64 |
| MLP Intermediate Size | 8192 |

因为 `32 × 64 = 2048`，所以 Q 的总维度是 2048；因为 `8 × 64 = 512`，所以 K、V 的总维度是 512。该模型使用 GQA（Grouped-Query Attention）：每 4 个 Query Head 共享一个 KV Head。

Llama 3.2 1B 是 Decoder-only 模型，模型大致结构如下：
- Embedding → 多层 Transformer（Attention + MLP）→ 输出层

这里和下面省略RMSNorm的计算过程，只涉及其他核心计算过程。RMSNrom不改变矩阵大小。

# 2. Token ID 变成 Embedding
假设有八个请求，长短不一，最长的为6，总共有30个token。分词器先把文本转换成 Token ID。每个 Token ID 通过 Embedding 查表，变成一个 2048 维向量。

对请求 A 来说：

```text
Token ID：[A1, A2, A3]
Embedding：[3, 2048]
```

8 个请求合计有 30 个有效 Token，所以采用**变长打包（packed）**表示时，输入可以组织成：

```text
X：[30, 2048]
```

另一种直观方式是把每个请求补齐到最长长度 6：

```text
X：[8, 6, 2048]
```

这会预留 `8 × 6 = 48` 个位置，其中 30 个是有效 Token，18 个是 Padding。下面我们以padding后的大小为例子

# 3.  Q\K\V计算流程
计算公式如下：
- $$Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V$$

- $$O = \operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$

- $$Y = \operatorname{Concat}(O_1,O_2,\ldots,O_{32})W_O$$

其中：

- $X$：输入矩阵
- $W_Q、W_K、W_V$：模型训练得到的 Q、K、V 投影参数矩阵。
- $M$：注意力掩码，用于屏蔽未来 Token 和无效的 Padding 位置。
- $d_k$：每个注意力头的维度，Llama 3.2 1B 中为 64。
- $W_O$：Attention 输出投影的参数矩阵。
- $O_i$：第 $i$ 个注意力头的输出。
- $\operatorname{Concat}$：将所有注意力头的输出拼接起来。
- $Y$：经过多头拼接和输出投影后的 Attention 输出。





## 3.1 获取QKV 

为方便展示，先把 Padding 形式的前两个维度合并：

```text
[8, 6, 2048] → [48, 2048]
```

通过上述公式计算得到：

```text
Q：[48, 2048] → [8, 6, 32, 64]
K：[48,  512] → [8, 6,  8, 64]
V：[48,  512] → [8, 6,  8, 64]
```

为了方便描述 Attention 计算，再把维度调整为 `[请求数, 头数, Token 数, 每头维度]`： 

```text
Q：[8, 32, 6, 64]
K：[8,  8, 6, 64]
V：[8,  8, 6, 64]
```

接着对 Q 和 K 应用 RoPE（旋转位置编码），以注入位置信息。RoPE 不改变矩阵形状；V 不做 RoPE。



## 3.2 QK 之间如何计算

模型有 32 个 Query Heads，却只有 8 个 KV Heads。每 4 个 Q 头使用同一组 K、V：

```text
Q0、Q1、Q2、Q3   → K0、V0
Q4、Q5、Q6、Q7   → K1、V1
...
Q28、Q29、Q30、Q31 → K7、V7
```

这些 Q 头共享的是 K、V 数据，不是 Attention 的计算结果。由于每个 Q 头的 Q 不同，它们算出的注意力权重和输出也不同。



先以 A 的一个 Query Head为例。A 有 3 个有效 Token；如果按 Padding 的逻辑形式展示，则长度上限是 6：

```text
Q0：[6, 64]
K0：[6, 64]
K0ᵀ：[64, 6]
```

计算得到：

```text
[6, 64] × [64, 6] → [6, 6]
```
这里计算出来是
$$
Q_0 K_0^\top
$$

对于请求 A，每个 Query Head 都会产生一个 `[6,6]` 的注意力分数矩阵，因此共有 32 个。对于整个 Batch，注意力分数矩阵的逻辑形状为 `[8,32,6,6]`。 

## 3.3 Mask、Softmax计算

对分数进行缩放，并使用两类 Mask：

- **Causal Mask（因果掩码）**：不允许当前位置关注未来 Token。
- **Padding Mask**：不允许有效 Token 把 Padding 位置当成有效上下文。

Mask 和 Softmax 不改变分数矩阵的形状：

$$
QK^\top \in \mathbb{R}^{8 \times 32 \times 6 \times 6}
$$

$$
A = \operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)
\in \mathbb{R}^{8 \times 32 \times 6 \times 6}
$$

其中，\(M\) 表示因果掩码和 Padding 掩码，\(d_k=64\)。

这里的公式是 Attention 的简化表示。实际计算时，每个 Query Head 使用 GQA 中与其对应的 KV Head 进行计算，每 4 个 Query Head 共享一组 K、V。
## 3.4 计算 Attention 输出


原始 V 矩阵的形状为：

$$
V\in\mathbb{R}^{8\times8\times6\times64}
$$

由于模型使用 GQA，每 4 个 Query Head 共享一个 KV Head。因此，计算时和QK计算类似，会得到32个`[6, 64]`.

对于单个请求、单个注意力头，计算过程为：

$$
[6,6]\times[6,64]\rightarrow[6,64]
$$

所有请求和 Query Head 完成计算后，得到 Attention 输出：

$$
O\in\mathbb{R}^{8\times32\times6\times64}
$$

接下来，调整维度顺序，并合并 32 个注意力头：

$$
[8,32,6,64]\rightarrow[8,6,32,64]\rightarrow[8,6,2048]
$$

最后，通过 Attention 输出投影矩阵 $W_O$：

$$
Y=OW_O
$$

$$
[8,6,2048]\times[2048,2048]\rightarrow[8,6,2048]
$$

至此，Attention 子层计算完成，输出 $Y$ 将进入后续的残差连接与 MLP。

## 3.5 KV缓存是什么

在上面的计算中我们得到了QKV矩阵`[请求数, 头数, Token 数, 每头维度]`： 

```text
Q：[8, 32, 3, 64]
K：[8,  8, 3, 64]
V：[8,  8, 3, 64]
```
**每个请求每一层都有自己的 K Cache 和 V Cache。** 第 1 层保存的是 Token 在第 1 层算出来的 K、V；第 2 层保存的是这些 Token 在第 2 层算出来的 K、V。不同层的 K、V 不能互相替代。

对单个请求 A，假设它的有 3 个 Token。Prefill 结束后，每一层分别保存：

```text
 K Cache：[8, 3, 64]
 V Cache：[8, 3, 64]
```


如果把所有层算上，实际框架不一定这样连续存储。可以写成：

```text
K：[16, 8, 3, 64]
V：[16, 8, 3, 64]
```
这就是我们需要缓存的对象。目前这个KV只是这个请求通过prefill阶段得到的，接下来就是decode阶段，我们会对这个KV缓存进行复用。

## 3.6 缓存如何复用
对上述的请求A，进入decode阶段会得到新的QKV`[头数, Token 数, 每头维度]`
```text
Q：[32, 1, 64]
K：[8, 1, 64]
V：[8, 1, 64]
```
然后把新得到的KV都加到历史的KV后面得到：
```text
K：[8, 4, 64]
V：[8, 4, 64]
```

当前 Token 的 Q 与历史 K Cache 及当前 Token 的 K 共同计算 Attention，再与对应的 V Cache 相乘，得到当前 Token 的 Attention 输出。Q 只用于当前轮计算，不需要缓存；只有 K、V 会不断追加。

## 3.7 KV 缓存的计算

如果 K、V 使用 FP16/BF16（每个占 2 字节），Llama 3.2 1B 每个 Token 的 KV Cache 大约占：

```text
16 层 × 2（K 和 V）× 8 个 KV Heads × 64 维 × 2 Bytes
= 32,768 Bytes
≈ 32 KiB / Token
```

因此，30 个有效输入 Token 的 KV Cache 合计约为 `30 × 32 KiB = 960 KiB`（约 0.94 MiB）。这只是缓存本身的估算，不含模型权重、激活、临时工作区等其他显存开销。


# 4 残差连接与 MLP

## 4.1 Attention 残差连接

Attention 输出与进入 Attention 子层前的隐藏状态相加得到$H$：


$$
[8,6,2048] + [8,6,2048]
\rightarrow [8,6,2048]
$$

## 4.2 RMSNorm 与 Gate，Up 投影

先进行 RMSNorm矩阵大小不变，然后通过两个线性层分别计算 Gate 和 Up：

$$
N = \operatorname{RMSNorm}(H)
$$

$$
G = NW_{\text{gate}},\qquad U = NW_{\text{up}}
$$

其中，$W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{2048\times8192}$。

矩阵形状变化：

$$
[8,6,2048]\rightarrow[8,6,8192]
$$

因此，$G$ 和 $U$ 的形状均为 $[8,6,8192]$。

## 4.3 激活与逐元素相乘

Gate 分支经过 SiLU 激活，再与 Up 分支逐元素相乘：

$$
Z = \operatorname{SiLU}(G)\odot U
$$

$$
[8,6,8192]\odot[8,6,8192]
\rightarrow[8,6,8192]
$$

其中，$\odot$ 表示逐元素相乘。

## 4.4 Down 投影与残差连接

将中间结果投影回 2048 维：

$$
D = ZW_{\text{down}}
$$

其中，$W_{\text{down}}\in\mathbb{R}^{8192\times2048}$。

矩阵形状变化：

$$
[8,6,8192]\rightarrow[8,6,2048]
$$

最后，与 MLP 子层的输入 $H$ 进行残差相加：

$$
H_{next} = H + D
$$

$$
[8,6,2048]+[8,6,2048]
\rightarrow[8,6,2048]
$$

至此，一个 Transformer 层执行完毕，输出 $H_{next}$ 将作为下一层的输入。

以上过程重复 16 层。最后经过最终 RMSNorm 和词表投影（LM Head），得到下一 Token 的 logits，再通过采样或贪心策略选出输出 Token。




