你现在这版视觉基底是对的：**界面已经有“工程调试台”的感觉**，Tensor Board、Inspector、Test Runner、Chapter 进度这些都成立。问题主要出在底部的交互逻辑：现在玩家做的是“看概念 → 选答案”，而不是“发现问题 → 动手修复 → 运行验证”。

建议把 Chapter 0 从“选择题教程”改成：

# Chapter 0：Tensor Bootcamp / 张量工程入门

核心体验不是答题，而是成为一个 **Shape Debugger**。
玩家面对的是一个被破坏的模型板，任务是把缺失的 shape 语义、axis 标注、矩阵连接、broadcast 规则修好，让测试样例通过。

---

# 1. 当前问题判断

你现在的 0-1 Shape Reader 大概是这样：

> 给 hidden[B,T,C]，问 B/T/C 分别表示什么。
> 玩家在底部卡片里选择 batch / sequence / channel。

这个交互的问题是：

**第一，答案太显性。**
玩家看到选项后是在做知识回忆，不是在操作模型。

**第二，概念和画布脱节。**
中间有很漂亮的 Tensor Workbench，但真正决策发生在底部选择题区域。画布像插图，不像玩法场。

**第三，验证没有工程感。**
Run Tests 只是检查答案，不像《图灵完备》那样用输入输出、线路、错误信号让玩家知道哪里错了。

**第四，Inspector 只是说明面板。**
它现在告诉玩家很多信息，但玩家不能把这些信息“拿去修机器”。

所以不要推翻 UI，而是把现在这些元素重新定义成游戏对象。

---

# 2. 新的基本循环：概念 → 操作 → 验证

每个小关卡都采用同一个结构：

```text
1. Concept Brief / 概念简报
   用极短文本说明当前工程规则。

2. Broken Board / 故障模型板
   给玩家一个缺失、接错、shape 不匹配或测试失败的模型结构。

3. Interactive Repair / 实际操作
   玩家拖拽标签、接线、旋转矩阵、插入模块、设置 transpose、放置 mask、调整 broadcast。

4. Probe & Trace / 探针观察
   玩家运行局部数据流，观察 shape、dtype、sample values、axis pattern。

5. Autograder / 考核验证
   系统用公开测试 + 隐藏测试验证结构是否正确。

6. Unlock / 解锁新工具
   通过后解锁下一种真实 LLM 构件。
```

重点是：
**概念讲解不再是题目前的一段说明，而是玩家修机器时必须用到的工程规则。**

---

# 3. 把选择题改成“形状修复玩法”

以你截图里的 **0-1 Shape Reader** 为例，原来是问：

```text
Axis B 表示什么？
Axis T 表示什么？
Axis C 表示什么？
```

改成下面这种玩法。

---

## 0-1 Shape Reader：修复 Hidden Tensor 的轴标签

### 关卡设定

系统给玩家一个已经生成好的 `hidden tensor`，但它的三个轴标签丢失了。

画布中央显示：

```text
Hidden Tensor
float32[?, ?, ?]
```

这个张量接下来要进入 `Axis Decoder` 和 `Shape Tests`，但因为轴标签缺失，下游模块拒绝运行。

错误提示：

```text
Shape contract incomplete:
hidden tensor rank = 3, but axis semantics are unknown.
Required contract: hidden[B,T,C]
```

---

## 玩家目标

玩家不是回答选择题，而是要把三个标签拖到 Hidden Tensor 的三个轴槽上：

```text
B = batch
T = sequence/token position
C = channel/embedding feature
```

画面上 Hidden Tensor 的 3D 盒子有三个可交互轴：

```text
Axis 0 slot: empty
Axis 1 slot: empty
Axis 2 slot: empty
```

左侧或底部提供三个实体标签芯片：

```text
[B] Batch
[T] Token Position
[C] Channel
```

玩家需要把它们拖到正确轴上。

---

## 操作工具

为了让这件事不是“猜”，需要给玩家三个探针工具。

### 1. Batch Probe / 批次探针

点击某个轴后，画布显示这个轴上的不同切片。

如果这是 B 轴，玩家会看到：

```text
sample 0: "the cat sat"
sample 1: "hello world !"
sample 2: "a small model"
```

它们是不同训练样本。

视觉表现：

* B 轴切开后是一叠独立样本卡。
* 每张卡内部都有自己的 token 序列。
* Inspector 显示：`axis separates independent examples`

---

### 2. Time Probe / 序列探针

点击某个轴后，系统沿这个轴播放 token 位置。

如果这是 T 轴，玩家会看到：

```text
position 0 → position 1 → position 2 → position 3 ...
```

每一步对应句子里的一个 token。

视觉表现：

* token 位置像时间步一样从左到右亮起。
* causal mask 预览只会沿这个轴生成下三角。
* Inspector 显示：`axis is used by causal attention`

---

### 3. Channel Probe / 通道探针

点击某个轴后，系统显示这个轴上的连续特征值。

如果这是 C 轴，玩家会看到：

```text
[-0.04, 0.11, -0.38, 0.27, ...]
```

它不是 token，也不是样本，而是每个 token 的 embedding feature。

视觉表现：

* C 轴不是文字序列，而是一排细密的特征条。
* Linear 层会消费这个轴。
* Inspector 显示：`axis is consumed by Linear / MLP / Attention projection`

---

## 通过条件

当玩家拖完 B/T/C 标签后，点击 Run Tests。

系统执行：

```text
Check 1: tensor rank == 3
Check 2: axis labels assigned
Check 3: B axis separates samples
Check 4: T axis matches token positions
Check 5: C axis enters Linear as feature dimension
```

如果正确：

```text
PASS
hidden contract resolved: float32[B,T,C]
Axis Decoder unlocked
Shape Inspector upgraded
```

如果错误，不要说“答案错误”，而是显示工程失败：

```text
FAIL: C axis assigned to token position.
Linear expected feature axis C, but received sequence axis T.
Downstream projection cannot consume this tensor.
```

这样玩家学到的不是答案，而是为什么这个轴必须是 C。

---

# 4. 底部选择题区域怎么改

你截图里底部现在是：

```text
Axis B
Axis T
Axis C
每个下面是选项卡
```

建议改成 **Repair Console / 修复控制台**。

---

## 旧结构

```text
Axis B: batch / token position / channel
Axis T: sequence / vocab size / attention heads
Axis C: channels / batch / time step count
```

## 新结构

```text
Repair Console
- Available Axis Tags: [B] [T] [C]
- Probe Tools: Batch Probe / Time Probe / Channel Probe
- Contract Target: hidden[B,T,C]
- Hidden Tests: 3 pending
```

底部不再让玩家点选项，而是：

1. 选择探针；
2. 点击画布上的轴；
3. 观察证据；
4. 拖标签到轴槽；
5. 运行测试。

这样底部区域就从“答题区”变成了“调试工具区”。

---

# 5. Chapter 0 建议改成四个硬核小关

你现在 Chapter 0 已经有：

```text
0-1 Shape Reader
0-2 MatMul Gate
0-3 Transpose Trap
0-4 Broadcast Add
```

这个结构很好，只需要把每关都变成“修复机器”。

---

# 0-1 Shape Reader：识别 [B,T,C]

## 概念

```text
LLM 中最常见的激活张量是 hidden[B,T,C]。
B 是 batch，T 是 token position，C 是 embedding channel。
```

## 操作

玩家使用 Probe，给三条轴贴上 B/T/C 标签。

## 验证

系统检查：

```text
hidden.rank == 3
axis_labels == [B,T,C]
Linear consumes C
Causal Mask consumes T
Batch Loader consumes B
```

## 解锁

```text
Shape Inspector
Axis Tags
Tensor Probe
```

---

# 0-2 MatMul Gate：修复 Linear 的内维度

## 概念

Linear 层不是随便接矩阵，它要求最后一个维度 C 和权重矩阵内维对齐。

真实规则：

```text
input:  [B,T,C]
weight: [C,O]
output: [B,T,O]
```

## 故障

画布上有一个 Linear Gate，但权重板方向错了。

当前错误：

```text
input:  [B,T,C]
weight: [O,C]
error: inner dimension mismatch
```

## 操作

玩家需要：

* 旋转 Weight Plate；
* 或插入 Transpose Switch；
* 或选择正确权重形状 `[C,O]`；
* 把 output axis 标成 `O`。

## 玩法表现

权重矩阵是一块可旋转的 3D 板：

```text
[C,O]   正确
[O,C]   错误，需要 transpose
```

如果玩家接错，接口物理上不能吸附，或者吸附后亮红灯。

## 验证

```text
input last dim == weight first dim
output shape == [B,T,O]
reference matmul allclose within 1e-5
```

## 解锁

```text
Linear Module
Weight Plate
Transpose Switch preview
```

---

# 0-3 Transpose Trap：修复 QKᵀ

## 概念

Attention score 不是 `Q @ K`，而是：

```text
scores = Q @ K.transpose(-2, -1)
```

如果：

```text
Q: [B,H,T,D]
K: [B,H,T,D]
```

那么：

```text
Kᵀ:     [B,H,D,T]
scores: [B,H,T,T]
```

## 故障

玩家看到：

```text
Q: [B,H,T,D]
K: [B,H,T,D]
Q @ K = error
```

或者系统错误地生成了：

```text
[B,H,T,D] @ [B,H,T,D]
```

## 操作

玩家需要把 `Transpose Switch` 放在 K 路径上，并设置：

```text
swap last two axes: T <-> D
```

然后接入 MatMul Gate。

## 玩法表现

K 张量是一块 3D/4D 折叠板，玩家操作一个“轴交换开关”：

```text
before: K[B,H,T,D]
after:  Kᵀ[B,H,D,T]
```

画布上会显示 `T × T` attention score 热力图板。

## 验证

```text
K last two axes swapped
scores shape == [B,H,T,T]
attention score matrix square over token positions
reference allclose within tolerance
```

## 解锁

```text
Transpose Switch
Attention Score Board
```

---

# 0-4 Broadcast Add：修复 Broadcast 加法

## 概念

LLM 里经常发生 broadcast：

```text
hidden[B,T,C] + bias[C]        -> [B,T,C]
hidden[B,T,C] + pos_emb[T,C]   -> [B,T,C]
```

但不是所有形状都能加。

## 故障

玩家需要把三块数据板相加：

```text
token_hidden: [B,T,C]
pos_emb:      [T,C]
bias:         [C]
```

但当前 board 报错：

```text
Cannot align axes.
```

## 操作

玩家需要把 `pos_emb[T,C]` 对齐到 hidden 的 T/C 轴，把 `bias[C]` 对齐到 C 轴。

可以提供一个 Broadcast Rail：

```text
[B,T,C]
[  T,C]
[    C]
```

玩家需要把小张量贴到正确的轴槽上。

## 玩法表现

Broadcast 不是选择题，而是物理对齐：

```text
hidden:  [B][T][C]
pos_emb:    [T][C]
bias:          [C]
```

玩家拖错位置时，轴槽红灯。

## 验证

```text
broadcast result shape == [B,T,C]
no illegal expansion on semantic axis
reference add allclose
```

## 解锁

```text
Broadcast Rail
Bias Add
Position Add
```

---

# 6. 把“概念讲解”变成三层信息，而不是长教程

你需要保留概念讲解，但不要一次性塞给玩家。建议分三层。

---

## 第一层：任务简报，只讲一句规则

例如 0-1：

```text
hidden[B,T,C] 是 Transformer 中最常见的激活形状。
B 是样本，T 是 token 位置，C 是每个 token 的特征通道。
```

只显示在关卡开始或顶部任务栏。

---

## 第二层：操作时的即时解释

玩家点击 B Probe 时，Inspector 不是直接告诉答案，而是显示观察证据：

```text
This axis separates independent samples.
Each slice has its own token sequence.
This is usually the batch axis.
```

玩家点击 T Probe 时：

```text
This axis is ordered.
Causal mask will be built along this dimension.
This is usually the sequence axis.
```

玩家点击 C Probe 时：

```text
Values on this axis are continuous features.
Linear layers consume this dimension.
This is usually the channel axis.
```

---

## 第三层：失败后的针对性提示

只有失败后才讲更明确的概念。

例如玩家把 T 和 C 标反：

```text
You assigned C to the token-position axis.

Why this fails:
- C is the feature/channel axis.
- Linear layers project C into another feature space.
- Causal attention operates over T, not C.

Try probing which axis changes with token position.
```

这样概念讲解就不会像教材，而像调试反馈。

---

# 7. 增加游戏感的关键机制

不要做时间压力。这个类型更适合“工程挑战感”。
游戏感应该来自 **限制、反馈、验证、解锁和优化**。

---

## 机制 A：Shape Contract / 形状契约

每个模块都有输入输出契约。

例如 Linear：

```text
accepts: float[B,T,C]
weight:  float[C,O]
returns: float[B,T,O]
```

Attention Score：

```text
accepts:
Q: float[B,H,T,D]
Kᵀ: float[B,H,D,T]

returns:
scores: float[B,H,T,T]
```

玩家不是“选答案”，而是要让线路满足契约。

视觉表现：

* 合法端口蓝光吸附；
* shape 不匹配红光拒绝；
* dtype 不匹配黄光警告；
* 契约满足后模块盖章通过。

---

## 机制 B：Probe Budget / 探针预算

为了增加策略感，可以给每关一个非强制预算。

例如：

```text
Probe used: 3 / 5
Hints used: 0
Reference runs: 1 / 3
```

通过不受影响，但评分受影响：

```text
PASS
Rank: A
Probe efficiency: 80%
Reference calls: 1
```

这会让懂的人追求少用探针，不懂的人也能慢慢试。

---

## 机制 C：Hidden Tests / 隐藏测试

不要只用当前屏幕上的固定 shape。
每关都应该有隐藏测试。

例如 0-1：

```text
Visible case:
hidden[2,4,8]

Hidden cases:
hidden[1,8,16]
hidden[4,3,32]
hidden[3,12,64]
```

如果玩家只是猜当前数字，隐藏测试会失败。
如果玩家理解了 B/T/C 语义，就能通过。

---

## 机制 D：Reference Trace / 参考实现对比

硬核感来自“我搭出来的东西能和参考实现对齐”。

每关验证时显示：

```text
Your output:
shape: [B,T,O]
max error: 0.000003

Reference:
shape: [B,T,O]

PASS allclose atol=1e-5
```

这比“答对了”更有工程满足感。

---

## 机制 E：模块解锁

Chapter 0 每关通过后解锁一个真实构件：

```text
0-1 解锁 Shape Inspector
0-2 解锁 Linear Gate
0-3 解锁 Transpose Switch
0-4 解锁 Broadcast Rail
```

解锁后的构件会在 Chapter 1/2 真正用到。

这样 Chapter 0 不是考试，而是“新手训练营”。

---

# 8. 当前 UI 的具体改造建议

基于你截图，我建议这样改。

---

## 左侧 Chapter 面板

现在左侧已经不错，但可以把每个关卡状态从“警告三角”改成工程状态：

```text
0-1 Shape Reader
Status: Contract incomplete

0-2 MatMul Gate
Status: Locked

0-3 Transpose Trap
Status: Locked

0-4 Broadcast Add
Status: Locked
```

关卡卡片上显示：

```text
Concept: Axis semantics
Tool: Probe
Reward: Shape Inspector
```

---

## 中央 Canvas

这是最重要的交互区。

现在 hidden tensor 是一个静态块。
建议给它增加三个可交互轴槽：

```text
Axis 0: [ empty ]
Axis 1: [ empty ]
Axis 2: [ empty ]
```

玩家拖 B/T/C 标签上去后，张量块表面直接显示：

```text
float32[B,T,C]
```

如果标错，先不马上报错，等测试时再给失败反馈。

---

## 右侧 Tensor Inspector

现在 Inspector 信息很多，但偏说明书。
建议改成“证据面板”。

选中某条轴后显示：

```text
Selected axis: axis 1
length: 4
observed pattern:
- ordered token positions
- used by causal mask
- affects attention matrix size

Possible semantic: T
Confidence: high
```

注意这里可以给“possible semantic”，但不要直接替玩家完成操作。
或者在早期关卡给 confidence，后期取消。

---

## 底部 Test Runner

现在底部选择题区域改成两部分：

左边：Concept Brief
右边：Repair Console

例如：

```text
Concept Brief
hidden[B,T,C] is a 3D activation tensor.
B = independent samples.
T = token positions.
C = feature channels.

Repair Console
Available tags: [B] [T] [C]
Tools: Batch Probe / Time Probe / Channel Probe
Target contract: hidden[B,T,C]
```

下面保留 Run Tests，但测试结果要具体：

```text
Visible checks:
✓ rank is 3
✓ all axes labeled
✗ Linear consumed T instead of C

Hidden checks:
not run
```

---

# 9. 0-1 的完整玩家流程示例

可以按这个节奏做。

---

## Step 1：进入关卡

顶部任务：

```text
0-1 Shape Reader
Restore the missing axis labels of hidden tensor.
Target contract: hidden[B,T,C]
```

简报弹出 3 行：

```text
A Transformer activation is often shaped as [B,T,C].
B stores independent samples.
T stores token positions.
C stores per-token features.
```

玩家点击 Start。

---

## Step 2：观察故障

画布里数据流停在 Hidden Tensor。

错误灯亮：

```text
Axis Decoder blocked:
hidden tensor has rank 3, but axes are unlabeled.
```

---

## Step 3：使用探针

玩家选择 Time Probe，点击某个轴。

画布播放该轴切片：

```text
pos 0: token id 101
pos 1: token id 205
pos 2: token id 432
...
```

Inspector 显示：

```text
This axis is ordered and matches token position.
Causal mask will operate along this axis.
```

玩家把 `[T]` 标签拖到这个轴上。

---

## Step 4：继续标注

玩家选择 Batch Probe，找到 batch 轴，拖 `[B]`。
玩家选择 Channel Probe，找到 channel 轴，拖 `[C]`。

Hidden Tensor 从：

```text
float32[?, ?, ?]
```

变成：

```text
float32[B,T,C]
```

线路重新亮起。

---

## Step 5：运行测试

玩家点击 Run Tests。

测试面板显示：

```text
Visible tests
✓ hidden rank is 3
✓ B axis separates samples
✓ T axis matches token positions
✓ C axis consumed by Linear

Hidden tests
✓ hidden[1,8,16]
✓ hidden[4,3,32]
✓ hidden[2,12,64]

PASS
```

---

## Step 6：解锁

弹出奖励：

```text
Tool unlocked: Shape Inspector
Contract unlocked: float[B,T,C]
Next: MatMul Gate
```

然后让玩家进入 0-2。

这个循环就完整了：

```text
概念讲解 → 实际观察 → 操作修复 → 测试验证 → 解锁推进
```

---

# 10. 让关卡更像《图灵完备》的关键

《图灵完备》的爽感不来自选择题，而来自：

```text
我搭了一个结构
我跑了测试
我看见失败
我定位问题
我修复
我通过
```

所以你这套游戏也要始终保持：

```text
玩家产物 = 一个可运行结构
不是一个答案
```

在 Chapter 0，玩家产物可以是：

```text
已标注的 tensor contract
已修复的 matmul gate
已设置的 transpose path
已对齐的 broadcast add
```

到 Chapter 1 以后，玩家产物会升级成：

```text
Embedding pipeline
Single attention head
Multi-head attention
Transformer block
Tiny GPT
Training loop
Sampler
SFT pipeline
```

这就是从“教程”变成“构建游戏”的根本区别。

---

# 11. 可以立即落地的实现清单

你现在这个 MVP 不需要大改架构，可以按下面顺序做。

## 第一阶段：把 0-1 从选择题改成拖拽标注

增加：

```text
axis_slots: 3
axis_tags: B/T/C
drag_to_assign
test_axis_labels
```

原来的选项卡可以暂时隐藏，或者作为 Hint 模式保留。

---

## 第二阶段：增加 Probe 工具

先做最简单版本：

```text
Batch Probe：点击轴后显示 batch slices
Time Probe：点击轴后播放 token positions
Channel Probe：点击轴后显示 feature bars / sample values
```

不需要一开始做复杂数值，只要视觉反馈明确。

---

## 第三阶段：增加 Shape Contract Engine

每个模块定义：

```ts
inputContract
outputContract
axisSemantics
dtype
rank
```

测试时检查：

```ts
rankMatches
axisLabelsMatch
dtypeMatches
consumerMatches
```

---

## 第四阶段：把错误反馈改成工程失败

错误不写：

```text
Wrong answer
```

而写：

```text
Linear expected channel axis C, but received sequence axis T.
```

或者：

```text
Causal mask requires token-position axis T.
Current tensor has no T axis assigned.
```

---

## 第五阶段：0-2 / 0-3 / 0-4 都按同一模式重做

```text
0-2：旋转 / 转置权重板
0-3：给 K 路径插入 transpose switch
0-4：把小张量拖到 broadcast rail 的正确轴槽
```

---

# 12. 一个关卡配置模板

你后面可以用这种数据结构组织关卡。

```ts
const level_0_1 = {
  id: "0-1",
  title: "Shape Reader",
  concept: {
    brief: [
      "hidden[B,T,C] is the standard activation tensor in a Transformer.",
      "B separates independent samples.",
      "T stores token positions.",
      "C stores per-token feature channels."
    ],
    unlockTerms: ["batch", "sequence", "channel", "rank", "dtype"]
  },

  board: {
    nodes: [
      { id: "text_batch", type: "TextBatch", output: "text[B]" },
      { id: "token_ids", type: "TokenIDs", output: "int[B,T]" },
      { id: "hidden", type: "HiddenTensor", output: "float32[?,?,?]", axisLabels: [null, null, null] },
      { id: "axis_decoder", type: "AxisDecoder", input: "float32[B,T,C]" },
      { id: "shape_tests", type: "ShapeTests", input: "bool[checks]" }
    ]
  },

  interactions: [
    "probe_axis",
    "drag_axis_tag",
    "run_visible_tests",
    "run_hidden_tests"
  ],

  availableTags: ["B", "T", "C"],

  probes: [
    {
      id: "batch_probe",
      description: "Reveals independent sample slices."
    },
    {
      id: "time_probe",
      description: "Animates token positions."
    },
    {
      id: "channel_probe",
      description: "Shows per-token feature values."
    }
  ],

  checks: [
    {
      id: "rank_check",
      rule: "hidden.rank === 3"
    },
    {
      id: "axis_check",
      rule: "hidden.axisLabels === ['B','T','C']"
    },
    {
      id: "consumer_check",
      rule: "linear.consumesAxis === 'C'"
    }
  ],

  rewards: [
    "Shape Inspector",
    "hidden[B,T,C] contract"
  ]
};
```

这个模板的重点是：
关卡不再配置“问题和选项”，而是配置“故障板、可操作对象、测试规则、解锁奖励”。

---

# 13. 最小改版目标

如果你想快速把当前 MVP 从“教程感”改成“游戏感”，第一版只做这三个变化就够明显：

## 变化 1：选择题变成拖拽标签

不要让玩家点答案。
让玩家把 `[B] [T] [C]` 标签贴到 Hidden Tensor 的三个轴上。

## 变化 2：选项解释变成探针观察

不要直接问“B 是什么”。
让玩家用 Probe 观察哪个轴像 batch、哪个轴像 sequence、哪个轴像 channel。

## 变化 3：答题验证变成工程测试

不要显示“3 checks pending”。
显示：

```text
Contract tests
- rank test
- axis semantic test
- downstream consumer test
- hidden shape test
```

通过后让线路亮起来，模块解锁。

---

# 14. 这一章的设计目标

Chapter 0 的目标不是让玩家背下：

```text
B = batch
T = sequence
C = channel
```

而是让玩家形成工程直觉：

```text
看到 [B,T,C]，我知道这是 batch 里的 token 序列，每个 token 有 C 维特征。
看到 Linear，我知道它吃的是 C。
看到 Attention，我知道它主要在 T × T 上工作。
看到 Broadcast，我知道我要对齐语义轴。
看到 shape mismatch，我知道应该检查 rank、axis、dtype 和 transpose。
```

这才是后续 Chapter 1/2/3 能继续硬核化的基础。
