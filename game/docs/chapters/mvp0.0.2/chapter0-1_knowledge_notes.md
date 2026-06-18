可以。0-1 的背景知识可以比之前再往前铺一层：**先让玩家知道张量是模型内部流动的数据形态，再讲 shape 是张量的“结构说明书”，最后才引出 hidden tensor 是 Transformer 中最常见的中间激活数据。**

这一段不要做成长篇文字教程，建议做成 **8 张短知识卡 + 3 个微交互**。每张卡都只解决一个概念，最后自然进入画布任务。

---

# 0-1 Shape Reader：背景知识讲解完整设计

## 这一段的教学目标

玩家进入画布之前，至少要建立 5 个基础概念：

```text
1. LLM 内部处理的是数字，不是文字。
2. 张量 Tensor 是一组有结构的数字。
3. Rank 表示张量有几个轴。
4. Shape 表示每个轴有多长。
5. Hidden Tensor 是模型内部流动的中间表示，常见形状是 [B,T,C]。
```

但这些概念不能只靠文字解释，要让玩家看到：

```text
文字 → token id → embedding vector → hidden tensor
```

并且看到：

```text
hidden[b,t,c]
```

到底是在访问哪一个样本、哪一个 token、哪一个特征值。

---

# 1. 背景知识讲解的整体形式

建议进入 0-1 后，先进入一个独立的 **Bootcamp Briefing / 训练营简报界面**。

界面不是普通 PPT，而是一个小型交互教学台。

画面中间是可旋转的数据展示区；右侧是概念卡；底部是“下一步”按钮。

流程为：

```text
知识卡 1：LLM 不直接读取文字
知识卡 2：什么是 Tensor
知识卡 3：Rank：张量有几个轴
知识卡 4：Shape：每个轴有多长
知识卡 5：Axis：每个轴代表什么
知识卡 6：Token IDs 的 shape 是 [B,T]
知识卡 7：Embedding 把 token 变成向量
知识卡 8：Hidden Tensor 是 [B,T,C]
→ 进入画布
→ 任务弹窗
```

每张卡最多停留 10–20 秒。
玩家可以跳过，但第一次进入建议默认播放。

---

# 2. 知识卡 1：LLM 不直接读取文字

## 标题

```text
LLM 不直接读取文字
```

## 玩家可见文案

```text
你输入的是文字，但模型内部处理的是数字。

一句话会先被拆成 token，
每个 token 再被转换成一个整数 ID。
```

## 画面表现

左侧是一句文字：

```text
we train llm
```

文字经过一个 Tokenizer 小机器，变成 token 方块：

```text
[we] [train] [llm]
```

每个 token 方块翻转后露出 ID：

```text
[we: 502]
[train: 2841]
[llm: 9172]
```

最后形成一条整数序列：

```text
token_ids = [502, 2841, 9172]
```

## 微交互

玩家点击文字条，文字被切成 token。
再点击 token，token 翻转成 ID。

## 这一卡的目的

让玩家知道：

```text
模型不是直接吃自然语言，而是吃 token id。
```

这为后面 `token_ids[B,T]` 做铺垫。

---

# 3. 知识卡 2：什么是 Tensor

## 标题

```text
Tensor：模型里的数字容器
```

## 玩家可见文案

```text
Tensor 可以理解为一组排列整齐的数字。

它可以是一个数字，
也可以是一排数字，
也可以是一张表，
还可以是一叠表。
```

## 画面表现

中间展示 4 种数据形态：

```text
Scalar 标量
3.14
```

```text
Vector 向量
[0.2, -0.7, 1.4, 0.5]
```

```text
Matrix 矩阵
[
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6]
]
```

```text
3D Tensor 三维张量
一叠矩阵
```

视觉上可以从一个点逐渐展开：

```text
点 → 线 → 面 → 体
```

## 微交互

玩家拖动一个滑杆：

```text
Rank 0 → Rank 1 → Rank 2 → Rank 3
```

画面随之变化：

```text
一个数字 → 一排数字 → 一张表 → 一叠表
```

## 这一卡的目的

让玩家理解：

```text
Tensor 本质上是结构化数字数组。
```

不要一开始就讲“高维空间”，这里要保持工程感。

---

# 4. 知识卡 3：Rank 是轴的数量

## 标题

```text
Rank：张量有几个轴
```

## 玩家可见文案

```text
Rank 表示一个 tensor 有几个方向可以索引。

一个数字没有轴，rank = 0。
一排数字有 1 个轴，rank = 1。
一张表有 2 个轴，rank = 2。
一叠表有 3 个轴，rank = 3。
```

## 画面表现

展示 rank 的层级：

```text
rank 0: scalar
shape: []
```

```text
rank 1: vector
shape: [4]
```

```text
rank 2: matrix
shape: [3,4]
```

```text
rank 3: tensor
shape: [2,3,4]
```

这里可以把轴画成发光坐标边：

```text
rank 1：只有 X 轴
rank 2：X + Y 轴
rank 3：X + Y + Z 轴
```

## 微交互

玩家点击不同 rank 的对象，Inspector 显示：

```text
rank: 2
axis count: 2
shape: [3,4]
```

## 这一卡的目的

让玩家把“rank”和“shape”区分开：

```text
rank = 有几个轴
shape = 每个轴有多长
```

---

# 5. 知识卡 4：Shape 是每个轴的长度

## 标题

```text
Shape：张量的结构说明书
```

## 玩家可见文案

```text
Shape 告诉我们 tensor 的每个轴有多长。

例如 shape = [2,4,8]，
表示这个 tensor 有 3 个轴：

第 0 轴长度是 2，
第 1 轴长度是 4，
第 2 轴长度是 8。
```

## 画面表现

显示一个 3D 数据盒：

```text
shape = [2,4,8]
```

三个轴分别标注：

```text
Axis 0 length = 2
Axis 1 length = 4
Axis 2 length = 8
```

但此时不要直接写 B/T/C，只写：

```text
Axis 0
Axis 1
Axis 2
```

画面中：

* Axis 0 方向只有 2 层；
* Axis 1 方向有 4 个位置；
* Axis 2 方向有 8 个特征格。

## 微交互

玩家把鼠标放到 shape 数字上：

```text
[2, 4, 8]
 ↑
```

高亮第 0 轴。

再放到 4：

```text
[2, 4, 8]
    ↑
```

高亮第 1 轴。

再放到 8：

```text
[2, 4, 8]
       ↑
```

高亮第 2 轴。

## 这一卡的目的

让玩家理解：

```text
shape 不是一个普通数字列表，
它对应 tensor 的实际结构。
```

---

# 6. 知识卡 5：Shape 不等于语义

这是 0-1 最重要的一张卡。

## 标题

```text
Shape 只说明大小，不说明意义
```

## 玩家可见文案

```text
[2,4,8] 只告诉我们有三个轴，
长度分别是 2、4、8。

但它没有告诉我们：
哪个轴是样本？
哪个轴是 token 位置？
哪个轴是特征通道？

这些意义需要由 shape contract 标注。
```

## 画面表现

同一个 3D 数据盒显示：

```text
float32[2,4,8]
```

三个轴都是问号：

```text
Axis 0: ?
Axis 1: ?
Axis 2: ?
```

旁边出现两种错误理解：

```text
可能是 [B,T,C]
也可能是 [T,B,C]
也可能是 [B,C,T]
```

但是只有一种符合下游模块的合同。

## 微交互

玩家点击“Show Consumers / 查看下游模块”。

画面出现两个半透明模块：

```text
Causal Mask needs T
Linear Projection needs C
Batch Viewer needs B
```

这时玩家会看到：

```text
轴的意义会影响它能连接到哪里。
```

## 这一卡的目的

直接引出 0-1 的玩法核心：

```text
玩家不是识别数字大小，而是恢复轴语义。
```

---

# 7. 知识卡 6：Token IDs 的 shape 是 [B,T]

## 标题

```text
Token IDs 通常是 [B,T]
```

## 玩家可见文案

```text
当多句话一起进入模型时，
token id 会排成一个二维 tensor。

B 表示 batch 中有几条样本。
T 表示每条样本有几个 token 位置。
```

## 示例

假设 batch 里有两句话：

```text
sample 0: "we train llm"
sample 1: "shape tells truth"
```

Tokenizer 后得到：

```text
token_ids[B,T]

[
  [502, 2841, 9172, 0],
  [1042, 7191, 3910, 0]
]
```

shape 是：

```text
[2,4]
```

其中：

```text
B = 2    两条样本
T = 4    每条样本四个 token 位置
```

## 画面表现

二维表格：

```text
          T0      T1      T2      T3
B0       502    2841    9172       0
B1      1042    7191    3910       0
```

用颜色强调：

* 横向是 T，表示 token 顺序；
* 纵向是 B，表示不同样本。

## 微交互

玩家点击一行：

```text
B0 = sample 0
```

整行高亮。

点击一列：

```text
T2 = 每个样本的第 2 个 token 位置
```

整列高亮。

## 这一卡的目的

让玩家先理解：

```text
token_ids 是 [B,T]
```

这样下一张卡引入 C 轴就很自然。

---

# 8. 知识卡 7：Embedding 把 token 变成向量

## 标题

```text
Embedding：把 token ID 变成向量
```

## 玩家可见文案

```text
token id 只是一个整数。

Embedding Lookup 会用这个整数去查表，
把每个 token 转换成一个向量。
```

## 示例

```text
token_id = 502
```

查 embedding table 后得到：

```text
embedding[502] = [0.12, -0.08, 0.31, 0.44, ...]
```

这个向量的长度就是：

```text
C = channel / embedding dimension
```

## 画面表现

左边是 token id：

```text
502
```

中间是 Embedding Table：

```text
row 502 被点亮
```

右边输出一条向量：

```text
[0.12, -0.08, 0.31, 0.44, ...]
```

视觉上可以表现为：

```text
token 方块 → 查表矩阵墙 → feature bars
```

## 微交互

玩家点击 token id，Embedding Table 对应行被点亮，然后拉出一条 feature vector。

## 这一卡的目的

让玩家理解 C 轴的来源：

```text
每个 token 不再只是一个整数，
而是变成了 C 维特征向量。
```

---

# 9. 知识卡 8：Hidden Tensor 是 [B,T,C]

## 标题

```text
Hidden Tensor：模型内部流动的工作状态
```

## 玩家可见文案

```text
Embedding 后，每个 token 都有一个 C 维向量。

如果我们有 B 条样本，
每条样本有 T 个 token，
每个 token 有 C 个特征值，

那么得到的 hidden tensor 就是：

hidden[B,T,C]
```

## 示例

```text
B = 2
T = 4
C = 8
```

所以：

```text
hidden.shape = [2,4,8]
```

它可以理解为：

```text
2 条样本
每条 4 个 token
每个 token 8 个特征值
```

## 画面表现

从上一张二维 token 表扩展成三维 tensor：

原来：

```text
token_ids[B,T]
```

现在每个格子展开成一条 C 维向量：

```text
hidden[B,T,C]
```

视觉上：

```text
二维 token 表
每个 token 格子向外长出一排 feature bars
最后形成一个 3D 数据盒
```

## 关键索引示例

在画面上点击一个具体位置：

```text
hidden[0,2,:]
```

显示：

```text
第 0 条样本
第 2 个 token
这个 token 的完整 C 维向量
```

再点击一个具体小格：

```text
hidden[0,2,5] = -0.381
```

显示：

```text
第 0 条样本
第 2 个 token
第 5 个 channel 的数值
```

## 这一卡的目的

让玩家真正看懂：

```text
hidden[B,T,C]
```

不是一个抽象符号，而是一个可以索引、可以切片、可以传给下游模块的真实数据结构。

---

# 10. 额外补充：为什么叫 Hidden Tensor

这部分可以做成知识卡 8 的展开说明，或者放在 Inspector 的“More”里。

## 玩家可见文案

```text
Hidden 不是“隐藏起来不能看”的意思。

在 Transformer 里，hidden tensor 通常指模型内部的中间表示。
它不是原始 token id，也不是最后输出的文字，
而是每一层模型正在处理的激活状态。
```

可以补充：

```text
输入阶段：
token_ids[B,T]

Embedding 后：
hidden[B,T,C]

经过 Transformer Block 后：
hidden 仍然通常保持 [B,T,C]，
但每个 token 的向量已经融合了上下文信息。

最后经过 LM Head：
logits[B,T,V]
```

其中：

```text
V = vocabulary size
```

## 画面表现

一条完整但简化的管线：

```text
token_ids[B,T]
      ↓ Embedding
hidden[B,T,C]
      ↓ Transformer Block 0
hidden[B,T,C]
      ↓ Transformer Block 1
hidden[B,T,C]
      ↓ LM Head
logits[B,T,V]
```

重点强调：

```text
shape 可能保持 [B,T,C]，
但里面的数值会在每层不断变化。
```

---

# 11. 0-1 背景知识讲解里的核心概念表

可以在讲解结束时给一个“小抄”，后面也可以常驻在右侧术语面板。

| 概念             | 玩家理解版本           | 工程表达                          |
| -------------- | ---------------- | ----------------------------- |
| Tensor         | 一组有结构的数字         | typed n-dimensional array     |
| Dtype          | 数字的类型            | int / float32 / bf16          |
| Rank           | 有几个轴             | number of dimensions          |
| Shape          | 每个轴的长度           | `[2,4,8]`                     |
| Axis           | 张量的一个方向          | axis 0 / axis 1 / axis 2      |
| B              | batch 轴          | independent samples           |
| T              | token position 轴 | sequence length               |
| C              | channel 轴        | embedding / feature dimension |
| Hidden Tensor  | 模型内部的中间激活        | `float32[B,T,C]`              |
| Shape Contract | 带语义的 shape 约定    | `hidden[B,T,C]`               |

---

# 12. 知识讲解结束后的过渡语

知识卡结束后，不要直接进画布，可以显示一句任务过渡。

## 过渡弹窗

标题：

```text
Briefing Complete
```

正文：

```text
你已经知道：

Tensor 是模型内部流动的数字结构。
Shape 描述 tensor 的轴长度。
Hidden Tensor 是 Transformer 中常见的中间表示。

现在，一台训练板上的 Hidden Tensor 丢失了轴标签。
进入 Workbench，恢复它的 shape contract。
```

按钮：

```text
Enter Workbench
```

---

# 13. 进入画布后的任务弹窗文案

进入画布后再弹一次任务说明，但这次不讲概念，只布置任务。

## 标题

```text
Repair Mission: Hidden Tensor Contract
```

## 正文

```text
当前模型板已经生成了一个 Hidden Tensor：

float32[2,4,8]

但它的三个轴标签丢失了。

你需要通过探针观察它的结构，
判断哪个轴是 B、哪个轴是 T、哪个轴是 C，
并将它修复为：

hidden[B,T,C]
```

## 任务步骤

```text
1. 连接 Text Batch → Tokenizer → Embedding Lookup → Hidden Tensor
2. 使用 Probe 观察 Hidden Tensor 的三个轴
3. 将 B / T / C 标签拖到正确轴槽
4. 将 Hidden Tensor 连接到 Axis Decoder
5. 运行 Shape Tests
```

## 通过条件

```text
- 数据流完整
- hidden tensor rank = 3
- 三个轴语义正确
- downstream contract 通过
- hidden tests 通过
```

按钮：

```text
Start Repair
```

---

# 14. Hidden Tensor 的更形象化设计

为了让玩家在画布中真正理解 hidden tensor，建议不要只画成普通立方体，而是设计成 **“样本层 × token 格 × 特征条”** 的结构。

## 14.1 总体形象：三维数据柜

可以像一个半透明数据柜：

```text
B 轴：一层一层的样本抽屉
T 轴：每层里面从左到右的 token 格
C 轴：每个 token 格里的一排 feature 条
```

也就是：

```text
B = 有几层抽屉
T = 每层抽屉里有几个 token 格
C = 每个 token 格里有多少条特征值
```

这个比单纯立方体更直观。

---

## 14.2 展开视图

默认是 compact view：

```text
Hidden Tensor
float32[2,4,8]
```

点击展开后变成：

```text
Batch Layer 0:
[token 0: feature vector]
[token 1: feature vector]
[token 2: feature vector]
[token 3: feature vector]

Batch Layer 1:
[token 0: feature vector]
[token 1: feature vector]
[token 2: feature vector]
[token 3: feature vector]
```

每个 token 格里面有 8 个小条：

```text
C0 C1 C2 C3 C4 C5 C6 C7
```

这样玩家可以直观看到：

```text
[2,4,8] = 2 层 × 4 个 token × 8 个特征
```

---

## 14.3 三种切片视图

玩家使用 Probe 时，Hidden Tensor 切换到对应切片视图。

### B Slice View

```text
hidden[0,:,:]
hidden[1,:,:]
```

解释：

```text
每一个 B 切片是一条完整样本的所有 token 表示。
```

画面：

两张独立的数据板。

---

### T Slice View

```text
hidden[:,0,:]
hidden[:,1,:]
hidden[:,2,:]
hidden[:,3,:]
```

解释：

```text
每一个 T 切片表示每条样本在同一个 token 位置上的向量。
```

画面：

同一位置的 token 在不同 batch 样本中被高亮。

---

### C Slice View

```text
hidden[:,:,0]
hidden[:,:,1]
...
hidden[:,:,7]
```

解释：

```text
每一个 C 切片表示所有样本、所有 token 在同一个特征通道上的数值。
```

画面：

像热力图一样显示某个 channel 在整个 batch-token 平面上的分布。

---

# 15. 推荐的讲解节奏：从生活直觉到工程表达

虽然游戏硬核，但第一关概念可以按“直觉 → 工程表达”的方式讲。

例如：

## 直觉版

```text
一个 batch 像一叠作业本。
每本作业本是一句话。
每句话有多个 token。
每个 token 不是一个数字，而是一排特征值。
```

## 工程版

```text
batch size = B
sequence length = T
embedding dimension = C

hidden.shape = [B,T,C]
```

## 索引版

```text
hidden[b,t,c]
```

表示：

```text
第 b 条样本
第 t 个 token
第 c 个特征值
```

这样玩家能一步步从直觉进入真实表达。

---

# 16. 可以直接放进游戏里的完整文本版本

下面是一版可以直接拆成 UI 文案的讲解稿。

---

## Card 1

```text
LLM 不直接读取文字

你输入的是文字，
但模型内部处理的是数字。

文字会先被切成 token，
每个 token 再被转换成整数 ID。
```

---

## Card 2

```text
Tensor 是模型里的数字容器

Tensor 可以理解为一组排列整齐的数字。

一个数字是一种 tensor。
一排数字是一种 tensor。
一张数字表也是 tensor。
一叠数字表仍然是 tensor。
```

---

## Card 3

```text
Rank 表示有几个轴

一个数字没有轴，rank = 0。
一排数字有 1 个轴，rank = 1。
一张表有 2 个轴，rank = 2。
一叠表有 3 个轴，rank = 3。
```

---

## Card 4

```text
Shape 表示每个轴的长度

shape = [2,4,8]

这表示：
第 0 轴长度是 2，
第 1 轴长度是 4，
第 2 轴长度是 8。

shape 说明 tensor 的结构大小。
```

---

## Card 5

```text
Shape 不等于语义

[2,4,8] 只告诉你每个轴有多长，
但没有告诉你每个轴代表什么。

在模型里，轴的语义非常重要。
下游模块必须知道哪个轴是 batch，
哪个轴是 token position，
哪个轴是 channel。
```

---

## Card 6

```text
Token IDs 通常是 [B,T]

B 是 batch，表示一次送进模型的多条样本。
T 是 token position，表示每条样本里的 token 位置。

例如：
2 条样本，每条 4 个 token，
token_ids.shape = [2,4]
也可以写成 token_ids[B,T]。
```

---

## Card 7

```text
Embedding 把 token ID 变成向量

token id 只是一个整数。

Embedding Lookup 会用 token id 查表，
把每个 token 变成一条 C 维向量。

C 是 channel，
也可以理解为每个 token 的特征维度。
```

---

## Card 8

```text
Hidden Tensor 通常是 [B,T,C]

当 B 条样本中的每个 token，
都被转换成 C 维向量后，
我们就得到了 hidden tensor。

hidden[B,T,C] 表示：

B：有几条样本
T：每条样本有几个 token 位置
C：每个 token 有多少个特征值
```

---

## Card 9，可选展开

```text
为什么叫 Hidden Tensor？

Hidden 指模型内部的中间表示。

它不是原始文字，
也不是最后输出的答案，
而是模型在每一层中持续更新的工作状态。

Transformer Block 会不断读取和改写 hidden tensor，
但它通常仍然保持 [B,T,C] 的形状。
```

---

# 17. 最后进入画布的引导语

```text
现在你已经知道：

Tensor 是模型内部流动的数字结构。
Shape 描述每个轴的长度。
Hidden Tensor 是 Transformer 中常见的中间激活。

接下来，你需要修复一个丢失轴标签的 hidden tensor。

目标合同：
hidden[B,T,C]
```

按钮：

```text
Enter Tensor Workbench
```

---

# 18. 这一段和后续玩法的连接

这一套知识讲解结束后，玩家进入画布，看到的就不是陌生符号，而是刚刚讲过的内容：

```text
Text Batch
Tokenizer
Embedding Lookup
Hidden Tensor float32[2,4,8]
Axis 0 ?
Axis 1 ?
Axis 2 ?
```

玩家会自然知道：

```text
我现在要做的不是猜答案，
而是把刚刚学到的 shape contract 修回去。
```

而且它会直接服务于后续关卡：

```text
0-2 MatMul Gate：
Linear 需要吃 C 轴。

0-3 Transpose Trap：
Attention 需要围绕 T 轴构建 [T,T]。

0-4 Broadcast Add：
bias[C] 和 pos_emb[T,C] 要正确对齐到 hidden[B,T,C]。
```

所以 0-1 的背景讲解要把重点落在一句话上：

```text
Shape 告诉你数据有多大，
Axis Semantics 告诉你数据该怎么用。
```

这句话可以作为 0-1 的核心教学标语，显示在关卡标题下方。
