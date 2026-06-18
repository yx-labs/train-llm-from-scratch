可以。现在既然 0-1 已经从单一 “Shape Reader” 变成了 **阶梯挑战关卡**，背景知识讲解也应该相应改掉：不要再在关卡开始前一次性讲完 Tensor、Rank、Shape、Token、Embedding、Hidden Tensor，而是改成 **分段解锁式知识讲解**。

核心原则是：

> 玩家每遇到一个新操作，才讲它所需的最小知识；
> 玩家通过操作验证后，再补充它在 LLM 里的真实意义。

也就是把原来的：

```text
先讲完整知识 → 再进入画布做题
```

改成：

```text
极短总引导
→ Stage 1 知识卡 → 操作 → 验证 → 小结
→ Stage 2 知识卡 → 操作 → 验证 → 小结
→ ...
→ 最终综合任务
```

这样会更像游戏关卡，而不是教程课件。

---

# 0-1 背景知识讲解新版结构

## 原始版本的问题

之前的背景知识讲解是连续知识卡：

```text
LLM 不直接读取文字
什么是 Tensor
什么是 Rank
什么是 Shape
Shape 不等于语义
Token IDs 是 [B,T]
Embedding 增加 C
Hidden Tensor 是 [B,T,C]
```

这些内容本身是对的，但如果玩家还没有动手，信息量会太大。

尤其是新手会遇到两个问题：

```text
1. 还没看到画布和节点，不知道这些知识要用在哪里。
2. 还没犯错，不知道为什么 Shape / Axis Semantics 很重要。
```

所以新版要改成 **按挑战阶段分发知识**。

---

# 1. 新版整体节奏

0-1 关卡开始时，只给玩家一个非常短的总引导。然后每个阶梯挑战开始前，弹出一个 **Concept Capsule / 概念胶囊**。

每个 Concept Capsule 只包含：

```text
1 个核心概念
1 张小图
1 个本阶段要用的操作工具
1 句任务目标
```

每个阶段完成后，再出现一个 **Debrief / 阶段小结**，告诉玩家刚才的操作对应 LLM 里的什么真实机制。

---

## 新版知识讲解分布

```text
进入关卡
├─ 总引导：你将修复一条从文本到 hidden tensor 的数据链路
│
├─ 0-1A Tensor Object
│  ├─ 讲：Tensor 是结构化数字
│  ├─ 做：把不同数据对象送入 Inspector
│  └─ 小结：模型内部所有数据都以 tensor 形式流动
│
├─ 0-1B Rank Scanner
│  ├─ 讲：Rank 是轴数量
│  ├─ 做：扫描 tensor 有几个轴
│  └─ 小结：rank 决定 tensor 是几维结构
│
├─ 0-1C Shape Caliper
│  ├─ 讲：Shape 是每个轴的长度
│  ├─ 做：测量 axis length，拼出 shape
│  └─ 小结：shape 是 tensor 的结构尺寸
│
├─ 0-1D Shape ≠ Semantics
│  ├─ 讲：shape 只说明大小，不说明意义
│  ├─ 做：尝试把未标注 tensor 接给下游模块
│  └─ 小结：模块需要 shape contract
│
├─ 0-1E Token Grid Builder
│  ├─ 讲：token_ids 通常是 [B,T]
│  ├─ 做：文本批次 → token 表格
│  └─ 小结：B 是样本，T 是 token 位置
│
├─ 0-1F Embedding Expansion
│  ├─ 讲：Embedding 把每个 token 扩展成 C 维向量
│  ├─ 做：[B,T] → [B,T,C]
│  └─ 小结：C 是每个 token 的特征通道
│
├─ 0-1G Hidden Contract Repair
│  ├─ 讲：Hidden Tensor 是模型内部工作状态
│  ├─ 做：用 Probe 恢复 hidden[B,T,C]
│  └─ 小结：hidden[B,T,C] 是 Transformer 基础激活形状
│
├─ 0-1H Consumer Validation
│  ├─ 讲：B/T/C 是下游模块的合同
│  ├─ 做：把 B/T/C 接给 Batch Viewer / Mask / Linear
│  └─ 小结：轴语义决定模块如何读取 tensor
│
└─ 0-1X Hidden Test Gauntlet
   ├─ 讲：不要靠数字大小猜语义
   ├─ 做：不同 shape 下恢复 B/T/C
   └─ 小结：掌握 shape contract 的迁移能力
```

---

# 2. 关卡开场总引导

## 目的

开场不要讲太多，只告诉玩家：

```text
你即将修复一条 LLM 内部数据链路。
你会逐步解锁 Tensor、Rank、Shape、Token Grid、Embedding、Hidden Tensor。
```

不要在这里一次性讲完 B/T/C。

---

## 开场界面标题

```text
0-1 Shape Reader
Tensor Debugger Bootcamp
```

## 开场文案

```text
LLM 的内部不是直接流动文字，而是流动一组组 tensor。

在这个训练关里，你会从最基础的数据对象开始，
一步步修复一条模型数据链路：

Text
→ Token IDs
→ Embedding
→ Hidden Tensor
→ Shape Contract
→ Downstream Tests

最终目标：
让系统识别并验证 hidden[B,T,C]。
```

## 玩家按钮

```text
Start Bootcamp
```

点击后进入 0-1A。

---

# 3. 0-1A 背景知识：Tensor Object

## 这一阶段玩家要学会

```text
Tensor 是模型内部流动的结构化数字。
```

不讲 Rank，不讲 Shape，不讲 B/T/C。

---

## Stage 开始知识卡

### 标题

```text
Tensor：模型里的数字对象
```

### 文案

```text
Tensor 可以理解为一组排列好的数字。

它可以是：
一个数字，
一排数字，
一张数字表，
或者一叠数字表。

在模型内部，几乎所有数据都会以 tensor 的形式流动。
```

### 配图建议

画面从左到右：

```text
3.14
→ [0.2, -0.7, 1.4]
→ 2D number grid
→ stacked number grids
```

### 本阶段任务提示

```text
把不同的数据对象拖入 Tensor Inspector，
确认它们都可以被系统识别为 tensor。
```

---

## 画布内即时提示

当玩家拖入 scalar：

```text
Inspector:
这是一个只有一个数值的 tensor。
```

拖入 vector：

```text
Inspector:
这是一排数字，也是一种 tensor。
```

拖入 matrix：

```text
Inspector:
这是一张数字表，也是一种 tensor。
```

拖入 3D tensor：

```text
Inspector:
这是一叠数字表。后续模型里会经常看到这种结构。
```

---

## 阶段完成小结

```text
Stage Complete: Tensor Object

你已经确认：
Tensor 不是某一种固定形状，
而是模型中用于承载数字数据的统一对象。

下一步：
我们要判断一个 tensor 有几个轴。
```

解锁：

```text
Rank Scanner
```

---

# 4. 0-1B 背景知识：Rank Scanner

## 这一阶段玩家要学会

```text
Rank = tensor 的轴数量。
```

不讲 shape 长度，只讲“有几个轴”。

---

## Stage 开始知识卡

### 标题

```text
Rank：tensor 有几个轴
```

### 文案

```text
Rank 表示 tensor 有几个可以索引的方向。

一个数字没有轴，rank = 0。
一排数字有 1 个轴，rank = 1。
一张表有 2 个轴，rank = 2。
一叠表有 3 个轴，rank = 3。
```

### 配图建议

```text
rank 0：点
rank 1：线
rank 2：面
rank 3：体
```

### 本阶段任务提示

```text
使用 Rank Scanner 扫描不同 tensor，
给它们贴上正确的 rank 标记。
```

---

## 画布内即时提示

扫描 vector：

```text
Rank Scanner:
检测到 1 条轴。
这个 tensor 的 rank 是 1。
```

扫描 matrix：

```text
Rank Scanner:
检测到 2 条轴。
这个 tensor 的 rank 是 2。
```

扫描 3D tensor：

```text
Rank Scanner:
检测到 3 条轴。
这个 tensor 的 rank 是 3。
```

---

## 失败提示

如果玩家给 3D tensor 贴 rank 2：

```text
Rank mismatch.

你标记的是 rank 2，
但扫描器检测到 3 条独立轴。

Rank 只表示轴的数量，
不是数值大小，也不是数据内容。
```

---

## 阶段完成小结

```text
Stage Complete: Rank Scanner

你已经知道：
Rank 描述 tensor 有几个轴。

下一步：
我们要测量每个轴有多长。
```

解锁：

```text
Shape Caliper
```

---

# 5. 0-1C 背景知识：Shape Caliper

## 这一阶段玩家要学会

```text
Shape = 每个轴的长度，按轴顺序排列。
```

---

## Stage 开始知识卡

### 标题

```text
Shape：tensor 的结构尺寸
```

### 文案

```text
Shape 告诉我们每个轴有多长。

例如：

shape = [2,4,8]

表示这个 tensor 有 3 个轴：
Axis 0 长度是 2，
Axis 1 长度是 4，
Axis 2 长度是 8。
```

### 配图建议

一个 3D 数据盒，三个方向分别标：

```text
Axis 0 length = 2
Axis 1 length = 4
Axis 2 length = 8
```

### 本阶段任务提示

```text
使用 Shape Caliper 测量三条轴，
拼出这个 tensor 的 shape。
```

---

## 画布内即时提示

玩家测量 Axis 0：

```text
Axis 0 measured:
length = 2
```

玩家测量 Axis 1：

```text
Axis 1 measured:
length = 4
```

玩家测量 Axis 2：

```text
Axis 2 measured:
length = 8
```

拼出 shape：

```text
float32[2,4,8]
```

---

## 失败提示

如果玩家把顺序放错：

```text
Shape order mismatch.

Shape 的数字顺序必须跟 axis 顺序一致。

Axis 0 的长度应该放在第 1 个位置，
Axis 1 的长度应该放在第 2 个位置，
Axis 2 的长度应该放在第 3 个位置。
```

---

## 阶段完成小结

```text
Stage Complete: Shape Caliper

你已经知道：
Rank 表示有几个轴。
Shape 表示每个轴有多长。

但现在还缺一个关键问题：
这些轴分别代表什么？
```

解锁：

```text
Semantic Inspector
```

---

# 6. 0-1D 背景知识：Shape 不等于语义

## 这一阶段玩家要学会

```text
Shape 只说明大小，不说明每个轴的工程意义。
```

这是整个 0-1 的关键转折。

---

## Stage 开始知识卡

### 标题

```text
Shape 不等于语义
```

### 文案

```text
float32[2,4,8] 只告诉你这个 tensor 的大小。

它没有告诉你：
哪个轴是样本？
哪个轴是 token 位置？
哪个轴是特征通道？

在模型里，下游模块需要知道轴的意义。
这叫 axis semantics。
```

### 配图建议

同一个 tensor：

```text
float32[2,4,8]
Axis 0: ?
Axis 1: ?
Axis 2: ?
```

旁边出现三个下游模块：

```text
Batch Viewer needs sample axis
Causal Mask needs token-position axis
Linear needs feature axis
```

### 本阶段任务提示

```text
尝试把只有 shape、没有轴语义的 tensor 连接到下游模块，
观察为什么它会失败。
```

---

## 画布内即时提示

玩家把 `float32[2,4,8]` 接到 Linear：

```text
Connection blocked.

Linear 不只需要知道 tensor 是 rank 3。
它还需要知道哪个轴是 feature/channel。
```

玩家把 tensor 接到 Causal Mask：

```text
Connection blocked.

Causal Mask 需要 token-position axis。
当前 tensor 的轴语义未知。
```

---

## 失败其实是教学点

这一阶段可以设计成“必然失败一次”。
玩家第一次连接下游模块必然被拒绝，然后系统解锁 Axis Tags。

提示：

```text
这不是数值错误，而是合同缺失。

下游模块需要一个带语义的 shape contract，
例如：

hidden[B,T,C]
```

---

## 阶段完成小结

```text
Stage Complete: Semantic Missing

你已经知道：
Shape 说明大小。
Axis Semantics 说明每个轴的用途。

下一步：
我们从 token_ids[B,T] 开始建立 B 和 T 的语义。
```

解锁：

```text
Axis Tags: B / T
Token Grid Builder
```

---

# 7. 0-1E 背景知识：Token Grid Builder

## 这一阶段玩家要学会

```text
token_ids 通常是 [B,T]。
B 是 batch。
T 是 token position。
```

---

## Stage 开始知识卡

### 标题

```text
Token IDs：从文字到二维表格
```

### 文案

```text
一句文字会先被切成 token，
每个 token 会变成一个整数 ID。

当多条文本一起送进模型时，
token ids 会组成一个二维 tensor：

token_ids[B,T]

B 表示 batch 中的样本。
T 表示每条样本里的 token 位置。
```

### 配图建议

```text
sample 0: "we train llm"
sample 1: "shape tells truth"

↓

token_ids[B,T]

          T0      T1      T2      T3
B0       502    2841    9172       0
B1      1042    7191    3910       0
```

### 本阶段任务提示

```text
连接 Text Batch → Tokenizer → Token Grid，
生成 token_ids[B,T]。
然后给二维表格标出 B 轴和 T 轴。
```

---

## 画布内即时提示

点击一行：

```text
Row selected:
这是一条独立样本。
它对应 B 轴上的一个位置。
```

点击一列：

```text
Column selected:
这是所有样本的同一个 token 位置。
它对应 T 轴上的一个位置。
```

---

## 失败提示

如果玩家把横轴标成 B：

```text
Axis semantic error.

你把 token 位置标记成了 batch。

Batch axis 应该分离不同样本。
Token-position axis 应该保持每条样本内部的顺序。
```

---

## 阶段完成小结

```text
Stage Complete: Token Grid

你已经构建了：

token_ids[B,T]

B 让模型一次处理多条样本。
T 保存每条样本内部的 token 顺序。

下一步：
token id 还只是整数。
模型会把每个 token id 变成一条向量。
```

解锁：

```text
Embedding Lookup
```

---

# 8. 0-1F 背景知识：Embedding Expansion

## 这一阶段玩家要学会

```text
Embedding 把每个 token id 变成 C 维向量。
于是 [B,T] 变成 [B,T,C]。
```

---

## Stage 开始知识卡

### 标题

```text
Embedding：给每个 token 增加特征通道
```

### 文案

```text
token id 只是一个整数。

Embedding Lookup 会用 token id 去查表，
把每个 token 转换成一条向量。

这条向量的长度叫 C，
也就是 channel / embedding dimension。
```

### 配图建议

```text
token_id = 9172
     ↓ lookup
embedding[9172] = [0.12, -0.08, 0.31, 0.44, ...]
```

再展示：

```text
token_ids[B,T]
      ↓ embedding
hidden[B,T,C]
```

### 本阶段任务提示

```text
连接 Token Grid 和 Embedding Lookup，
观察每个 token 如何扩展成 C 维 feature vector。
```

---

## 画布内即时提示

玩家点击一个 token id：

```text
token_ids[0,2] = 9172
```

Embedding Table 高亮：

```text
row 9172 selected
```

输出：

```text
embedding vector:
float32[C]
```

提示：

```text
这个 C 维向量会放回 sample 0、token position 2 的位置。
```

---

## 失败提示

如果玩家把 C 理解成新的 token 位置：

```text
Expansion error.

C 不是新的时间步。
C 是每个 token 内部的 feature channel。

token 的位置仍然由 T 表示。
token 内部的向量长度由 C 表示。
```

如果玩家混淆 V 和 C：

```text
Axis clarification.

V 是 vocabulary size，用来选择哪一行 embedding。
C 是每一行 embedding vector 的长度。
```

这里可以轻讲 V，但不要展开太深。

---

## 阶段完成小结

```text
Stage Complete: Embedding Expansion

你已经看到：

token_ids[B,T]
通过 Embedding Lookup
变成 hidden[B,T,C]

B：样本
T：token 位置
C：每个 token 的特征通道

下一步：
模型板上的 hidden tensor 丢失了这些轴标签。
你需要修复它。
```

解锁：

```text
Hidden Tensor
Batch Probe
Token Position Probe
Channel Probe
Axis Tag C
```

---

# 9. 0-1G 背景知识：Hidden Contract Repair

## 这一阶段玩家要学会

```text
Hidden Tensor 是模型内部的中间表示。
常见合同是 hidden[B,T,C]。
```

---

## Stage 开始知识卡

### 标题

```text
Hidden Tensor：模型内部的工作状态
```

### 文案

```text
Hidden Tensor 不是最终输出，
也不是原始 token id。

它是模型内部正在处理的中间表示。

在 Transformer 中，常见的 hidden tensor 是：

hidden[B,T,C]

每个 token 在每一层里，
都会带着一条 C 维向量继续向前流动。
```

### 配图建议

简化管线：

```text
token_ids[B,T]
      ↓ Embedding
hidden[B,T,C]
      ↓ Transformer Block
hidden[B,T,C]
      ↓ LM Head
logits[B,T,V]
```

重点标出：

```text
hidden 在多层中继续流动，shape 通常保持 [B,T,C]。
```

### 本阶段任务提示

```text
使用 Probe 观察 Hidden Tensor 的三个轴，
把 B / T / C 标签贴到正确位置。
```

---

## 画布内即时提示

### Batch Probe 点到 B 轴

```text
Probe Result:
这个轴分离不同样本。

slice 0: sample 0
slice 1: sample 1

可能语义：B / batch
```

### Token Position Probe 点到 T 轴

```text
Probe Result:
这个轴按照 token 顺序变化。

T0 → T1 → T2 → T3

可能语义：T / token position
```

### Channel Probe 点到 C 轴

```text
Probe Result:
这个轴包含连续 float feature values。

hidden[0,2,:] = [-0.04, 0.11, -0.38, ...]

可能语义：C / channel
```

早期可以显示“可能语义”，后续关卡去掉，改成纯观察证据。

---

## 失败提示

### B/T 标反

```text
Contract failed: B and T are swapped.

你标记为 B 的轴具有 token 顺序。
你标记为 T 的轴分离独立样本。

Batch axis 不应该参与 token-to-token attention。
Token-position axis 才能用于 causal mask。
```

### T/C 标反

```text
Contract failed: T and C are swapped.

你标记为 T 的轴包含连续 feature values。
你标记为 C 的轴沿 token 位置变化。

Attention 需要 T。
Linear / Projection 需要 C。
```

---

## 阶段完成小结

```text
Stage Complete: Hidden Contract Repair

你已经恢复：

hidden[B,T,C]

B：一批中的样本
T：每条样本里的 token 位置
C：每个 token 的特征通道

下一步：
我们要证明这些标签不是装饰，
而是真的能被下游模块使用。
```

解锁：

```text
Axis Decoder
Consumer Modules
```

---

# 10. 0-1H 背景知识：Consumer Validation

## 这一阶段玩家要学会

```text
B/T/C 是模块连接合同，不是视觉标签。
```

---

## Stage 开始知识卡

### 标题

```text
Axis Semantics 是模块合同
```

### 文案

```text
下游模块读取 tensor 时，
不只关心 shape 的数字大小，
还关心每个轴代表什么。

Batch Viewer 需要 B。
Causal Mask 需要 T。
Linear Projection 需要 C。

如果轴语义错误，模块可能仍然收到数据，
但它会在错误的方向上计算。
```

### 配图建议

```text
hidden[B,T,C]
   │   │   │
   │   │   └─ Linear Projection
   │   └───── Causal Mask
   └───────── Batch Viewer
```

### 本阶段任务提示

```text
把 Hidden Tensor 的 B/T/C 语义端口
连接到正确的下游模块。
```

---

## 画布内即时提示

### 连接 B → Batch Viewer

```text
Batch Viewer:
显示 batch 中的独立样本。

✓ B axis accepted
```

### 连接 T → Causal Mask

```text
Causal Mask:
使用 T 构建 [T,T] 下三角 mask。

✓ T axis accepted
```

### 连接 C → Linear Projection

```text
Linear Projection:
使用 C 作为输入 feature dimension。

weight shape: [C,O]
output shape: [B,T,O]

✓ C axis accepted
```

---

## 失败提示

如果 B 接到 Causal Mask：

```text
Consumer contract failed.

Causal Mask received B axis.

为什么失败：
B 分离不同样本。
不同样本之间不应该互相 attention。

Causal Mask 必须使用 T，
因为 T 表示同一句话内部的 token 顺序。
```

如果 T 接到 Linear：

```text
Consumer contract failed.

Linear received T axis.

为什么失败：
Linear Projection 投影的是每个 token 的 feature vector。
它应该消费 C，而不是 token position。
```

---

## 阶段完成小结

```text
Stage Complete: Consumer Validation

你已经验证：

B 被样本查看器使用。
T 被 attention / causal mask 使用。
C 被 linear / projection 使用。

Axis Semantics 决定 tensor 可以进入哪些模块。
```

解锁：

```text
Hidden Test Gauntlet
```

---

# 11. 0-1X 背景知识：Hidden Test Gauntlet

## 这一阶段玩家要学会

```text
不要靠数字大小猜轴语义。
B/T/C 是行为和合同，不是固定尺寸。
```

---

## Stage 开始知识卡

### 标题

```text
B/T/C 不是固定数字
```

### 文案

```text
在一个 tiny model 里，
C 可能只有 4。

在一个长上下文样本里，
T 可能是 16、128、甚至更长。

所以不能靠“哪个数字最大”判断 C，
也不能靠固定位置死记 B/T/C。

你需要观察：
哪个轴分离样本，
哪个轴保持 token 顺序，
哪个轴存储 feature values。
```

### 配图建议

展示三个不同 case：

```text
hidden[2,4,8]
hidden[1,16,4]
hidden[4,3,32]
```

并提示：

```text
同样是 hidden[B,T,C]，
尺寸可以变化。
```

### 本阶段任务提示

```text
连续修复多个未知 hidden tensor。
使用 Probe 判断轴语义，
通过所有隐藏测试。
```

---

## 隐藏测试中的即时提示

如果玩家按最大维度猜 C，遇到：

```text
hidden[1,16,4]
```

系统失败提示：

```text
Size-based guess failed.

你把长度 16 的轴标成了 C，
但它实际表现为 token sequence。

Axis length alone does not define meaning.
Use probe behavior and downstream contracts.
```

---

## 最终完成小结

```text
0-1 Complete: Shape Reader

你已经掌握：

Tensor 是模型内部的结构化数字。
Rank 表示轴数量。
Shape 表示每个轴的长度。
Shape 不等于轴语义。
token_ids 通常是 [B,T]。
Embedding 会把每个 token 扩展成 C 维向量。
Hidden Tensor 通常是 hidden[B,T,C]。
B/T/C 决定下游模块如何读取 tensor。

下一关：
0-2 MatMul Gate
你将学习 Linear 如何消费 C 轴。
```

---

# 12. 可选支线挑战的背景知识适配

如果 0-1 做得更完整，可以加入支线。支线的讲解也不要提前放入主线，而是玩家点击时再讲。

---

## Bonus A：Slice Drill

### 知识卡标题

```text
Tensor Slice：读取 tensor 的一部分
```

### 文案

```text
当一个 tensor 是 hidden[B,T,C] 时，
我们可以选择其中一个值、一条向量、一张切片，或者一个平面。

hidden[b,t,c] 是一个数值。
hidden[b,t,:] 是一个 token 的完整 C 维向量。
hidden[b,:,:] 是一条样本的所有 token 表示。
hidden[:,t,:] 是所有样本在同一个 token 位置上的表示。
hidden[:,:,c] 是某个 channel 的整体分布。
```

### 任务提示

```text
根据目标索引，在 3D tensor 上选出对应切片。
```

### 小结

```text
你现在不仅能读 shape，
还能理解 hidden tensor 的索引方式。
```

---

## Bonus B：Shape Mutation

### 知识卡标题

```text
Shape 可以变化，合同保持不变
```

### 文案

```text
B、T、C 的具体数值可以变化。

batch size 改变 B。
context length 改变 T。
embedding dimension 改变 C。

但只要语义不变，
它仍然是 hidden[B,T,C]。
```

### 任务提示

```text
根据任务要求调整 B/T/C，
生成指定 shape 的 hidden tensor。
```

### 小结

```text
你已经理解：
shape 是可变尺寸，
contract 是稳定语义。
```

---

## Bonus C：Consumer Puzzle

### 知识卡标题

```text
不同模块读取不同轴
```

### 文案

```text
同一个 hidden[B,T,C]，
不同模块会关注不同的轴。

Batch Viewer 看 B。
Sequence Player 看 T。
Causal Mask 用 T 生成 [T,T]。
Linear Projection 用 C 做矩阵乘法。
Channel Histogram 统计 C。
```

### 任务提示

```text
把每个模块连接到它需要的语义轴。
```

### 小结

```text
你已经能通过模块需求反推轴语义。
这是后续调试 Attention 和 Linear 的基础。
```

---

# 13. 新版知识讲解 UI 形式

为了适配阶梯挑战，建议把知识讲解分成 4 种 UI。

---

## 13.1 Concept Capsule / 概念胶囊

每个 Stage 开始时出现，面积不要太大。

结构：

```text
标题
一句核心概念
一张小示意图
本阶段工具
本阶段目标
```

例如：

```text
Rank：tensor 有几个轴

Rank 表示 tensor 有几个可以索引的方向。

Tool unlocked:
Rank Scanner

Mission:
扫描对象并标记它的 rank。
```

---

## 13.2 Inspector Note / 检查器说明

玩家操作时，右侧 Inspector 提供即时解释。

例如：

```text
Axis 1 selected

Observed behavior:
- values advance in token order
- aligned with token_ids columns
- used by causal mask

Likely semantic:
T / token position
```

这是“操作中的知识”。

---

## 13.3 Failure Lesson / 失败教学

失败时不要弹“错误”，而要解释为什么错。

结构：

```text
检测到什么错误
为什么会失败
相关概念
下一步建议
```

例如：

```text
Causal Mask received B axis.

为什么失败：
B 分离不同样本。
Attention 不应该跨样本发生。

下一步：
寻找沿 token 顺序变化的轴。
```

这是“犯错后的知识”。

---

## 13.4 Debrief / 阶段小结

每个 Stage 通过后，给 3 行总结。

结构：

```text
你刚刚做了什么
它对应什么概念
下一阶段会用到它的哪一部分
```

例如：

```text
你刚刚测量了 tensor 的三个轴。

这说明 shape 是按 axis order 记录的结构尺寸。

下一步：
我们会发现 shape 还不足以让下游模块运行。
```

---

# 14. 知识文本密度控制

为了避免再次变成教程，建议每类文本有严格长度限制。

```text
Concept Capsule:
最多 80–120 字

Inspector Note:
最多 3 条 bullet

Failure Lesson:
最多 4 行

Debrief:
最多 3 行

More Info:
可展开，不默认显示
```

“More Info” 里可以放更完整解释，例如 dtype、Vocab Size、Embedding Table、hidden activation 等，不阻塞主流程。

---

# 15. 0-1 的完整知识解锁顺序

这是最终推荐的知识解锁表。

| 阶段   | 解锁概念           | 不要提前讲的内容               |
| ---- | -------------- | ---------------------- |
| 开场   | 模型内部流动 tensor  | 不讲 rank/shape 细节       |
| 0-1A | Tensor 是结构化数字  | 不讲轴                    |
| 0-1B | Rank 是轴数量      | 不讲轴长度                  |
| 0-1C | Shape 是轴长度     | 不讲 B/T/C               |
| 0-1D | Shape 不等于语义    | 不讲 hidden 细节           |
| 0-1E | token_ids[B,T] | 不讲 attention           |
| 0-1F | Embedding 增加 C | 不讲 QKV                 |
| 0-1G | hidden[B,T,C]  | 不讲完整 Transformer Block |
| 0-1H | B/T/C 是模块合同    | 不讲 MatMul 细节           |
| 0-1X | 语义可迁移          | 不讲训练 loop              |

这张表可以作为内容制作约束，避免第一关信息过载。

---

# 16. 新版 0-1 背景知识总结

新版 0-1 的背景知识不再是一次性教程，而是：

```text
每个阶段讲一个最小概念；
每个概念马上对应一个操作；
每次操作都通过工程测试验证；
每次失败都变成一次概念解释；
每次通过都解锁下一层知识。
```

最终玩家不是“读懂了 hidden[B,T,C]”，而是经历了：

```text
看见 tensor
扫描 rank
测量 shape
发现 shape 不够
构建 token_ids[B,T]
通过 embedding 增加 C
修复 hidden[B,T,C]
验证 B/T/C 被下游模块正确消费
通过不同 shape 的隐藏测试
```

这才适合现在的阶梯挑战模式，也更接近《图灵完备》的学习体验：
**不是先上课再做题，而是在修机器的过程中逐步学会系统规则。**
