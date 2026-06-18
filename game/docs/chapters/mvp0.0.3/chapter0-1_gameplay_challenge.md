进阶建议： **0-1 不要只做一个“Shape Reader”任务**，而是拆成一串非常短的阶梯挑战。它们都发生在同一个关卡里，但像《图灵完备》的早期关卡一样，每一步只教一个小概念，玩家每掌握一个，就解锁下一个小工具。

也就是说，0-1 不是一道题，而是一个 **Tensor Debugger 入门执照考试**。

---

# 0-1 Shape Reader：阶梯挑战设计

## 关卡总目标

让玩家最终掌握：

```text
Tensor 是模型内部流动的结构化数字。
Rank 表示张量有几个轴。
Shape 表示每个轴有多长。
Axis Semantics 表示每个轴的工程意义。
Token IDs 通常是 [B,T]。
Embedding 会把每个 token 扩展成 C 维向量。
Hidden Tensor 通常是 hidden[B,T,C]。
B/T/C 不是装饰，而是决定下游模块如何读取这个张量。
```

所以 0-1 可以拆成：

```text
0-1A：什么是 Tensor
0-1B：识别 Rank
0-1C：读取 Shape
0-1D：理解 Shape 不等于语义
0-1E：构建 token_ids[B,T]
0-1F：从 token_ids 扩展到 hidden[B,T,C]
0-1G：给 Hidden Tensor 贴 B/T/C 轴标签
0-1H：用下游模块验证 B/T/C 是否正确
0-1X：隐藏测试 / 泛化挑战
```

其中 A 到 H 是主线，X 是最后考核。每个挑战都遵循：

```text
概念讲解 → 实际操作 → 考核验证
```

但概念讲解一定要非常短，操作和验证才是主体。

---

# 一、整体关卡流程

进入 0-1 后，玩家不是直接看到完整大画布，而是进入一个逐步解锁的节点画布。

初始画布很小，只有：

```text
[Data Object] → [Inspector] → [Shape Gate]
```

随着挑战推进，逐步扩展成：

```text
Text Batch
    ↓
Tokenizer
    ↓
token_ids[B,T]
    ↓
Embedding Lookup
    ↓
hidden[B,T,C]
    ↓
Axis Decoder
    ↓
Shape Tests
```

也就是，玩家不是一开始面对一堆概念，而是亲手把这条链路搭出来。

---

# 二、阶梯挑战总览

| 挑战                       | 玩家学到什么            | 主要操作                                    | 验证方式                      |
| ------------------------ | ----------------- | --------------------------------------- | ------------------------- |
| 0-1A Tensor Object       | Tensor 是结构化数字     | 把数字对象送入 Inspector                       | Inspector 能读取 dtype/value |
| 0-1B Rank Scanner        | Rank = 轴数量        | 给对象安装轴扫描器                               | 判断 rank 是否正确              |
| 0-1C Shape Caliper       | Shape = 每个轴长度     | 用量尺测量各轴长度                               | 生成正确 shape tag            |
| 0-1D Semantic Missing    | Shape 不等于语义       | 给同 shape 张量尝试接不同模块                      | 模块因语义缺失而拒绝                |
| 0-1E Token Grid          | token_ids 是 [B,T] | 把多条文本整理成二维 token 表                      | 通过 batch/sequence 检查      |
| 0-1F Embedding Expansion | Embedding 增加 C 轴  | 把每个 token 展开成 feature vector            | 输出 rank 从 2 变 3           |
| 0-1G Hidden Contract     | hidden 是 [B,T,C]  | 使用探针给三轴贴标签                              | hidden contract 成立        |
| 0-1H Consumer Validation | B/T/C 会被不同模块消费    | 把 B/T/C 接给 Batch Viewer / Mask / Linear | 下游模块全部运行                  |
| 0-1X Hidden Tests        | 泛化掌握              | 换不同尺寸的 hidden tensor                    | 隐藏测试通过                    |

---

# 三、0-1A：Tensor Object / 张量对象

## 教学目标

让玩家知道：

```text
Tensor 是模型里的数字容器。
它可以是一个数字、一排数字、一张表、一叠表。
```

这一关不讲 B/T/C，只讲“数据对象可以有结构”。

---

## 画布状态

画布上有三个节点：

```text
[Raw Number Objects] → [Tensor Inspector] → [Type Check]
```

左侧工具箱里有几种数据对象：

```text
Scalar: 3.14
Vector: [0.2, -0.7, 1.4]
Matrix: [[1,2,3], [4,5,6]]
Tensor Block: stack of matrices
```

---

## 玩家操作

玩家需要把这些数据对象拖进 Tensor Inspector。

每拖入一个对象，Inspector 显示：

```text
dtype: float32
rank: ?
shape: ?
sample values: ...
```

一开始 rank 和 shape 是锁住的，只显示：

```text
This is a tensor-like object.
```

---

## 验证

玩家把 4 种对象都送入 Inspector 后通过。

测试结果：

```text
✓ scalar accepted
✓ vector accepted
✓ matrix accepted
✓ 3D tensor accepted
```

---

## 通关反馈

系统解锁：

```text
Tool Unlocked: Rank Scanner
```

提示：

```text
下一步：不是所有 tensor 都长得一样。
我们需要知道它有几个轴。
```

---

# 四、0-1B：Rank Scanner / 识别 Rank

## 教学目标

让玩家掌握：

```text
Rank = tensor 的轴数量。
rank 0：没有轴
rank 1：一条轴
rank 2：两条轴
rank 3：三条轴
```

这里依然不要求玩家背定义，而是让玩家用工具扫描。

---

## 画布状态

新增一个 Rank Scanner 节点：

```text
[Tensor Object] → [Rank Scanner] → [Rank Gate]
```

Rank Scanner 会在数据对象周围显示发光轴。

例如：

```text
Vector:
Axis 0 only

Matrix:
Axis 0 + Axis 1

3D Tensor:
Axis 0 + Axis 1 + Axis 2
```

---

## 玩家操作

玩家需要把不同 tensor 拖到 Rank Scanner，然后从工具栏里拖一个 Rank Stamp 贴上去：

```text
rank 0
rank 1
rank 2
rank 3
```

注意，这里虽然有“贴标签”，但不是选择题，因为玩家是在观察一个实际对象。

---

## 验证

Rank Gate 会检查：

```text
object.rank == assigned_rank
```

失败时不要显示“错了”，而是显示：

```text
Rank mismatch:
You assigned rank 2,
but the scanner detected 3 independent axes.
```

---

## 通关反馈

解锁：

```text
Tool Unlocked: Shape Caliper
```

提示：

```text
知道有几个轴还不够。
我们还需要知道每个轴有多长。
```

---

# 五、0-1C：Shape Caliper / 读取 Shape

## 教学目标

让玩家理解：

```text
Shape = 每个轴的长度。
shape [2,4,8] 表示 rank 3，三个轴长度分别是 2、4、8。
```

---

## 画布状态

出现一个 3D 数据盒：

```text
Unknown Tensor
float32[?, ?, ?]
```

旁边有一个 Shape Caliper 工具。

数据盒实际尺寸是：

```text
[2,4,8]
```

但玩家看不到标签，只能用量尺测。

---

## 玩家操作

玩家用 Shape Caliper 分别拖到三条轴上。

测量结果显示：

```text
Axis 0 length = 2
Axis 1 length = 4
Axis 2 length = 8
```

然后玩家需要把 shape tag 拼出来：

```text
[2,4,8]
```

不是选项，而是把三个测量结果放进 Shape Gate 的三个槽位：

```text
shape slot 0: 2
shape slot 1: 4
shape slot 2: 8
```

---

## 验证

Shape Gate 检查：

```text
rank == 3
shape == [2,4,8]
```

失败反馈：

```text
Shape order mismatch:
Axis 1 length is 4,
but you placed 8 in slot 1.
Shape order follows axis order.
```

---

## 通关反馈

解锁：

```text
Concept Unlocked: Axis Order
```

提示：

```text
shape [2,4,8] 说明了大小。
但它还没有说明每个轴代表什么。
```

---

# 六、0-1D：Shape 不等于语义

这是 0-1 的核心转折点。

## 教学目标

让玩家真正意识到：

```text
shape [2,4,8] 只说明这个张量的大小。
它不说明哪个轴是 batch、哪个轴是 sequence、哪个轴是 channel。
```

也就是：

```text
shape ≠ axis semantics
```

---

## 画布状态

画布中有两个外形完全一样的 tensor：

```text
Tensor A: float32[2,4,8]
Tensor B: float32[2,4,8]
```

但它们内部语义不同。

例如：

```text
Tensor A: hidden[B,T,C]
Tensor B: something[T,B,C]
```

玩家一开始只能看到 shape，看不到语义。

旁边有三个下游模块：

```text
Batch Viewer
Causal Mask Preview
Linear Probe
```

---

## 玩家操作

玩家尝试把 `float32[2,4,8]` 直接接入下游模块。

模块会拒绝：

```text
Connection blocked:
Shape is known, but axis semantics are missing.
```

这时系统给出一句概念提示：

```text
下游模块不只需要知道大小。
它们还需要知道哪个轴代表什么。
```

---

## 关键交互

玩家需要打开 Semantic Inspector。

它会显示：

```text
Axis 0: unknown
Axis 1: unknown
Axis 2: unknown
```

此时还不要求玩家贴 B/T/C，只是让玩家看到“语义槽位”这个概念。

---

## 验证

玩家必须给一个 tensor 添加“unresolved semantic contract”标记：

```text
float32[2,4,8]
axis: ?, ?, ?
```

Shape Tests 通过：

```text
✓ shape detected
✓ semantics unresolved
✓ downstream modules require semantic contract
```

这一步看似简单，但很重要。它告诉玩家：

> 后面的任务不是猜 [2,4,8]，而是恢复 [B,T,C]。

---

## 通关反馈

解锁：

```text
Tool Unlocked: Axis Tags
[B] [T] [C]
```

提示：

```text
下一步，我们要先从 token_ids[B,T] 开始理解 B 和 T。
```

---

# 七、0-1E：Token Grid Builder / 构建 token_ids[B,T]

## 教学目标

让玩家掌握：

```text
Text Batch → Tokenizer → token_ids[B,T]
```

并理解：

```text
B = batch 中有几条样本
T = 每条样本里的 token 位置
```

---

## 画布状态

出现文本批次：

```text
sample 0: "we train llm"
sample 1: "shape tells truth"
```

玩家有三个节点：

```text
Text Batch → Tokenizer → Token Grid
```

但 Token Grid 一开始是空的。

---

## 玩家操作

玩家连接：

```text
Text Batch.output → Tokenizer.input
Tokenizer.output → Token Grid.input
```

运行一次 trace 后，Token Grid 生成二维表：

```text
token_ids[B,T]

          T0      T1      T2      T3
B0       502    2841    9172       0
B1      1042    7191    3910       0
```

玩家需要把两个轴标签贴上去：

```text
vertical axis → B
horizontal axis → T
```

这里比直接问 B/T 更有操作性，因为玩家能看到：

* 行是不同样本；
* 列是 token 位置。

---

## 微挑战

系统会要求玩家完成两个小任务：

### 任务 1：选出 sample 1

玩家点击 B1 行。

验证：

```text
✓ selected one independent sample
```

### 任务 2：选出所有样本的 T2 token

玩家点击 T2 列。

验证：

```text
✓ selected token position 2 across batch
```

这一步非常重要，因为它让玩家理解：

```text
B 是样本维度。
T 是位置维度。
```

不是只记住字母。

---

## 失败反馈

如果玩家把 B/T 标反：

```text
Axis semantic error:
You marked token positions as batch.

Batch axis should separate independent samples.
Token axis should preserve sequence order.
```

---

## 通关反馈

解锁：

```text
Concept Unlocked: token_ids[B,T]
Tool Unlocked: Token Position Probe
```

提示：

```text
token_ids 只有 [B,T]。
但模型不会直接使用整数 ID 计算。
每个 token 还要变成一条向量。
```

---

# 八、0-1F：Embedding Expansion / 从 [B,T] 到 [B,T,C]

## 教学目标

让玩家掌握：

```text
Embedding Lookup 会把每个 token id 变成 C 维向量。
```

所以：

```text
token_ids[B,T] → hidden[B,T,C]
```

这是 hidden tensor 的来源。

---

## 画布状态

上一阶段的 Token Grid 保留在画布上：

```text
token_ids[B,T]
```

新增节点：

```text
Embedding Table
Embedding Lookup
Hidden Tensor Builder
```

画布结构：

```text
Token Grid[B,T] → Embedding Lookup → Hidden Tensor
Embedding Table[V,C] ───────────────┘
```

这里顺带埋一个后面会学的概念：

```text
V = vocab size
C = embedding dimension
```

但 0-1 不展开 V，只让玩家知道 C 是“每个 token 的向量长度”。

---

## 玩家操作

玩家连接 Token Grid 到 Embedding Lookup。

点击某个 token id，比如：

```text
token_ids[0,2] = 9172
```

Embedding Table 中第 9172 行亮起，输出一条向量：

```text
embedding[9172] = [0.12, -0.08, 0.31, 0.44, ...]
```

玩家需要把这条向量放入对应 token 格子的 C 轴槽中。

第一次可以手动演示一个 token。
后面点击 Auto Fill，让所有 token 都完成 embedding lookup。

最终生成：

```text
hidden[B,T,C]
```

形象上，原本二维 token 表里的每个格子向外长出一排 feature bars，变成三维数据柜。

---

## 验证

系统检查：

```text
✓ input token_ids shape = [B,T]
✓ embedding table shape = [V,C]
✓ each token id maps to one C-dimensional vector
✓ output hidden rank = 3
✓ output hidden shape = [B,T,C]
```

---

## 失败反馈

如果玩家把 C 当成新的 token 位置：

```text
Expansion error:
C is not another sequence axis.
C stores feature values inside each token representation.
```

如果玩家试图把 embedding table 的 V 轴当 C：

```text
Axis confusion:
V indexes vocabulary rows.
C is the length of each embedding vector.
```

这里可以只做轻提示，V 不要在 0-1 讲太深。

---

## 通关反馈

解锁：

```text
Object Unlocked: Hidden Tensor
Tool Unlocked: Channel Probe
```

提示：

```text
现在我们得到了 hidden tensor。
但它的轴标签在模型板上丢失了。
下一步：恢复 hidden[B,T,C] 合同。
```

---

# 九、0-1G：Hidden Contract Repair / 修复 hidden[B,T,C]

这是你原本 0-1 的核心任务，但现在它已经有了前面的铺垫。

## 教学目标

让玩家掌握：

```text
hidden[B,T,C]
```

其中：

```text
B：独立样本
T：token 位置
C：每个 token 的特征通道
```

并能用探针识别每个轴。

---

## 画布状态

系统生成一个 Hidden Tensor：

```text
Hidden Tensor
float32[2,4,8]
Axis 0: ?
Axis 1: ?
Axis 2: ?
```

现在它的 shape 已知，但语义缺失。

旁边有三个工具：

```text
Batch Probe
Token Position Probe
Channel Probe
```

还有三个标签：

```text
[B] Batch
[T] Token Position
[C] Channel
```

---

## 玩家操作

### 第一步：探测 B 轴

使用 Batch Probe 点击某条轴。

如果点对，画面显示：

```text
Axis slice 0: sample 0
Axis slice 1: sample 1
```

如果点错，显示观察证据而不是直接判错。

---

### 第二步：探测 T 轴

使用 Token Position Probe。

如果点对，画面显示：

```text
T0 → T1 → T2 → T3
```

并同步显示 token：

```text
we → train → llm → <pad>
```

---

### 第三步：探测 C 轴

使用 Channel Probe。

如果点对，画面显示：

```text
hidden[0,2,:]
= [-0.04, 0.11, -0.38, 0.27, ...]
```

并显示 feature bars。

---

### 第四步：贴标签

玩家把标签拖到轴槽：

```text
Axis 0 → B
Axis 1 → T
Axis 2 → C
```

Hidden Tensor 显示为：

```text
float32[B,T,C]
```

---

## 验证

点击 Run Local Tests：

```text
✓ rank = 3
✓ B axis separates samples
✓ T axis preserves token order
✓ C axis stores feature values
✓ contract resolved: hidden[B,T,C]
```

---

## 失败反馈

如果 T 和 C 反了：

```text
FAIL: Token axis and channel axis are swapped.

Observed:
Axis marked T contains continuous feature values.
Axis marked C changes with token position.

Why it matters:
Attention builds [T,T] token-to-token scores.
Linear projects along C.
```

并在画布上显示两个失败预览：

```text
Causal Mask Preview: tried to build [C,C] instead of [T,T]
Linear Probe: tried to consume T instead of C
```

这样玩家会知道为什么错。

---

## 通关反馈

解锁：

```text
Tool Unlocked: Axis Decoder
```

提示：

```text
仅仅贴好标签还不够。
下游模块也要验证这些标签是否能工作。
```

---

# 十、0-1H：Consumer Validation / 下游模块验证

这是把“知识”变成“工程直觉”的关键。

## 教学目标

让玩家知道：

```text
B/T/C 不是名字，而是下游模块的连接合同。
```

具体来说：

```text
Batch Viewer 需要 B
Causal Mask 需要 T
Linear Probe 需要 C
```

---

## 画布状态

Hidden Tensor 已经显示：

```text
hidden[B,T,C]
```

新增三个下游模块：

```text
Batch Viewer
Causal Mask Preview
Linear Probe
```

每个模块都有一个语义输入端口：

```text
Batch Viewer.input: B
Causal Mask.input: T
Linear Probe.input: C
```

Hidden Tensor 有三个语义输出端口：

```text
B axis port
T axis port
C axis port
```

---

## 玩家操作

玩家连线：

```text
Hidden.B → Batch Viewer.B
Hidden.T → Causal Mask.T
Hidden.C → Linear Probe.C
```

---

## 每个模块的验证表现

### Batch Viewer

输入 B 后显示：

```text
sample 0
sample 1
```

验证：

```text
✓ batch samples remain independent
```

### Causal Mask Preview

输入 T 后生成：

```text
T × T lower triangular mask
```

画面显示：

```text
current token can see previous tokens
future tokens blocked
```

验证：

```text
✓ mask shape = [T,T]
```

### Linear Probe

输入 C 后显示：

```text
input feature dimension = C
weight shape = [C,O]
output shape = [B,T,O]
```

验证：

```text
✓ Linear consumes C
```

---

## 失败反馈

如果玩家把 B 接给 Causal Mask：

```text
FAIL: Causal Mask received B axis.

Why this fails:
Batch axis separates independent samples.
Attention should not connect different samples.
Causal Mask must operate over token positions T.
```

画面上可以显示一个很直观的错误预览：

```text
sample 0 token attends to sample 1 token
```

然后标红：

```text
Cross-sample attention is illegal.
```

---

## 通关反馈

显示：

```text
Shape Contract Verified

hidden[B,T,C] can now be consumed by:
✓ Batch Viewer
✓ Causal Mask
✓ Linear Projection
```

解锁：

```text
Next Challenge: 0-2 MatMul Gate
```

---

# 十一、0-1X：Hidden Test Gauntlet / 隐藏测试挑战

这是最终考核，防止玩家只是记住了当前 `[2,4,8]`。

## 教学目标

让玩家真正掌握：

```text
B/T/C 是语义，不是固定数字。
```

比如：

```text
[2,4,8] 不一定是 [B,T,C]
[4,2,8] 也可能是 [B,T,C]
[1,16,32] 也可能是 [B,T,C]
```

玩家要靠数据行为和下游合同判断，而不是靠数字大小猜。

---

## 画布状态

系统连续给出三台损坏的 Hidden Tensor：

```text
Case A: float32[1,8,16]
Case B: float32[4,3,32]
Case C: float32[2,12,6]
```

每个 case 都随机隐藏轴标签。

例如 Case B 内部可能是：

```text
Axis 0 = B
Axis 1 = T
Axis 2 = C
```

也可能故意打乱展示方向：

```text
Axis 0 = T
Axis 1 = B
Axis 2 = C
```

玩家不能只看数字，需要用 Probe 判断。

---

## 玩家操作

每个 case 的流程非常短：

```text
Probe axes
→ assign B/T/C
→ connect consumers
→ run tests
```

每个 case 限制：

```text
Probe Budget: 5
Reference Hint: 1
```

不强制失败，但影响评分。

---

## 验证

每个 case 检查：

```text
✓ B separates independent samples
✓ T aligns with token order
✓ C contains feature vector values
✓ Causal Mask uses T
✓ Linear uses C
```

最终显示：

```text
Hidden Tests Passed: 3/3
```

---

## 高级失败反馈

如果玩家始终根据“最大的数字就是 C”来猜，系统可以在隐藏测试里专门打破这个策略。

例如：

```text
shape = [2,16,4]
```

这里 T=16，C=4。

如果玩家把 16 当 C，系统提示：

```text
Size-based guess failed.

Axis length alone does not define meaning.
A long axis may be sequence length.
A short axis may be channel dimension in a tiny model.

Use probes to inspect behavior.
```

这句话非常重要。

---

# 十二、推荐把 0-1 做成“主线 + 可选挑战”

如果全部都放进主线，第一关可能太长。建议分成两层：

## 主线必做

```text
0-1A Tensor Object
0-1B Rank Scanner
0-1C Shape Caliper
0-1E Token Grid
0-1F Embedding Expansion
0-1G Hidden Contract Repair
0-1H Consumer Validation
```

## 可选挑战

```text
0-1D Shape 不等于语义
0-1X Hidden Test Gauntlet
0-1S Slice Drill
0-1BOSS Random Axis Repair
```

也可以把 D 保留在主线，因为它是最关键的概念转折；X 作为通关后的 Challenge Badge。

---

# 十三、增加一个可选挑战：Slice Drill / 切片训练

这个挑战非常适合帮助玩家理解 `hidden[b,t,c]`。

## 教学目标

让玩家掌握：

```text
hidden[b,t,c]
hidden[b,t,:]
hidden[b,:,:]
hidden[:,t,:]
hidden[:,:,c]
```

不用讲太多代码，只让玩家通过“切片任务”理解。

---

## 画布状态

一个已经标好的：

```text
hidden[B,T,C]
```

系统给出目标卡片：

```text
Find the full vector of sample 0, token 2.
```

工程表达：

```text
hidden[0,2,:]
```

---

## 玩家操作

玩家在 3D tensor 上选择：

```text
B = 0
T = 2
C = all
```

Tensor 展开一条 feature vector。

---

## 阶梯任务

### 任务 1：选一个数值

```text
Select hidden[1,3,5]
```

玩家选择：

```text
sample 1
token 3
channel 5
```

### 任务 2：选一个 token 的完整向量

```text
Select hidden[0,2,:]
```

### 任务 3：选一整条样本

```text
Select hidden[1,:,:]
```

### 任务 4：选所有样本的同一 token 位置

```text
Select hidden[:,2,:]
```

### 任务 5：选一个 channel 平面

```text
Select hidden[:,:,4]
```

---

## 验证

系统检查切片 shape：

```text
hidden[1,3,5] → scalar
hidden[0,2,:] → [C]
hidden[1,:,:] → [T,C]
hidden[:,2,:] → [B,C]
hidden[:,:,4] → [B,T]
```

这会让玩家对 B/T/C 形成非常强的空间直觉。

---

# 十四、再增加一个可选挑战：Shape Mutation / 形状变体

## 教学目标

让玩家理解：

```text
B、T、C 可以变化，但合同结构不变。
```

---

## 画布状态

系统有一个 Tensor Resizer 节点，可以调整：

```text
B = batch size
T = sequence length
C = channel dimension
```

玩家拖动三个旋钮：

```text
B: 1 → 8
T: 4 → 16
C: 8 → 64
```

Hidden Tensor 实时变化形状：

```text
hidden[1,4,8]
hidden[4,4,8]
hidden[4,16,8]
hidden[4,16,64]
```

---

## 玩家任务

系统给任务：

```text
Create a hidden tensor for:
3 samples,
each with 6 token positions,
each token with 12 feature channels.
```

玩家设置：

```text
B = 3
T = 6
C = 12
```

输出：

```text
hidden[3,6,12]
```

---

## 验证

```text
✓ B = 3
✓ T = 6
✓ C = 12
✓ contract = hidden[B,T,C]
```

失败反馈：

```text
You created hidden[6,3,12].
This swaps batch and token position.
```

---

# 十五、再增加一个可选挑战：Consumer Puzzle / 模块消费谜题

## 教学目标

强化：

```text
不同模块关心不同轴。
```

---

## 画布状态

给玩家 5 个模块：

```text
Batch Viewer
Sequence Player
Causal Mask
Linear Projection
Channel Histogram
```

给一个 hidden[B,T,C]。

---

## 玩家任务

把每个模块接到它需要的轴：

```text
Batch Viewer → B
Sequence Player → T
Causal Mask → T
Linear Projection → C
Channel Histogram → C
```

---

## 验证

每个模块运行一个微动画：

```text
Batch Viewer: 显示不同样本
Sequence Player: 播放 token 顺序
Causal Mask: 生成 [T,T]
Linear Projection: 使用 [C,O]
Channel Histogram: 统计 feature distribution
```

这比单独背 B/T/C 更牢固。

---

# 十六、0-1 的评分设计

通关不要只有 Pass / Fail，可以有工程评分。

```text
Rank C：通过 visible tests
Rank B：通过 hidden tests
Rank A：少于 8 次 probe 通过
Rank S：无 hint、无错误连接、一次通过 hidden tests
```

评分维度：

```text
Correctness：合同是否正确
Generalization：隐藏测试是否通过
Probe Efficiency：探针使用次数
Debug Quality：错误修复次数
Trace Understanding：是否查看过失败 trace
```

这会给硬核玩家复玩动力，但不会阻碍新手通关。

---

# 十七、0-1 的知识掌握路径

可以把 0-1 的学习目标分成四层。

## 第一层：识别

玩家能看懂：

```text
tensor
rank
shape
axis
```

对应挑战：

```text
0-1A / 0-1B / 0-1C
```

---

## 第二层：构建

玩家能搭出：

```text
text → token_ids[B,T] → hidden[B,T,C]
```

对应挑战：

```text
0-1E / 0-1F
```

---

## 第三层：解释

玩家能说明：

```text
B 是 batch
T 是 token position
C 是 channel
```

对应挑战：

```text
0-1G
```

---

## 第四层：迁移

玩家能在不同尺寸、不同轴顺序、不同下游模块中仍然判断正确。

对应挑战：

```text
0-1H / 0-1X / Slice Drill / Consumer Puzzle
```

这样一来，0-1 不是“教会一个符号”，而是让玩家建立可迁移的 shape 直觉。

---

# 十八、推荐的最终 0-1 结构

我建议正式版本里，0-1 可以这样安排：

```text
0-1 Shape Reader
├─ Briefing：什么是 Tensor / Shape / Hidden Tensor
├─ Stage 1：Rank Scanner
├─ Stage 2：Shape Caliper
├─ Stage 3：Token Grid Builder
├─ Stage 4：Embedding Expansion
├─ Stage 5：Hidden Contract Repair
├─ Stage 6：Consumer Validation
└─ Final：Hidden Test Gauntlet
```

可选支线：

```text
Bonus A：Slice Drill
Bonus B：Shape Mutation
Bonus C：Consumer Puzzle
```

如果担心第一关太长，可以把它拆成两个关卡：

```text
0-1 Shape Reader
学习 tensor / rank / shape / hidden[B,T,C]

0-1B Shape Contract
学习 B/T/C 消费关系、hidden tests、consumer validation
```

但从体验上，我更推荐放在一个关卡里，以检查点形式推进。这样玩家会感觉自己是在修复同一台机器，而不是上多个小课。

---

# 十九、这一套阶梯挑战的核心价值

这样设计后，玩家不会只是知道：

```text
B = batch
T = sequence
C = channel
```

而是会形成完整工程直觉：

```text
我知道 tensor 是结构化数字。
我知道 rank 和 shape 的区别。
我知道 shape 只说明大小，不说明语义。
我知道 token_ids 是 [B,T]。
我知道 embedding 会给每个 token 增加 C 维向量。
我知道 hidden tensor 是 [B,T,C]。
我知道 B/T/C 会决定它能不能进入 Batch Viewer、Causal Mask、Linear。
我知道不能靠数字大小猜轴语义，必须看数据行为和下游合同。
```

这才是后面学习 Linear、MatMul、Transpose、Attention 的基础。
