可以。0-1 关卡最好不要再做成“B/T/C 选择题”，而是做成一个完整的 **“张量合同修复任务”**。

玩家打开关卡后，先获得基础知识，但不是被动看教程；进入画布后，玩家要真的在节点系统里 **连接数据流、观察 tensor、标注轴语义、运行测试、修复失败**。整个体验应该接近《图灵完备》的第一批关卡：简单，但已经有“我在修一个真实系统”的感觉。

下面是 0-1 关卡的完整玩法设计。

---

# 0-1 Shape Reader：张量合同修复

## 关卡一句话

玩家进入 Tensor Bootcamp 的第一台训练机，发现 `Hidden Tensor` 的轴语义丢失，导致后续模块无法判断哪些维度是 batch、token position、channel。玩家需要通过观察、拖拽、连线和测试，恢复 `hidden[B,T,C]` 这个基础张量合同。

---

# 1. 关卡目标

## 教学目标

让玩家真正理解：

```text
hidden[B,T,C]
```

不是一个要背的符号，而是 Transformer 中非常核心的激活张量结构。

其中：

```text
B = batch，多个样本
T = token position，一句话里的 token 序列位置
C = channel，每个 token 的特征通道 / embedding 维度
```

但在玩法里，重点不是让玩家“选出答案”，而是让玩家明白：

```text
B 被 Batch Loader / Sample Viewer 使用
T 被 Causal Mask / Attention 使用
C 被 Linear / MLP / Projection 使用
```

也就是说，玩家要建立的是工程直觉：

> 一个轴的名字不是装饰，而是决定它可以进入哪些模块。

---

# 2. 关卡核心循环

0-1 的完整循环应该是：

```text
背景知识讲解
→ 进入画布
→ 任务弹窗
→ 关闭弹窗
→ 节点拖拽与连线
→ 使用探针观察 tensor
→ 给 tensor 轴贴标签
→ 连接到验证模块
→ 运行测试
→ 根据错误反馈修复
→ 通过隐藏测试
→ 解锁下一关
```

这套循环对应你提到的：

```text
概念讲解 → 实际操作 → 考核验证
```

区别是它不再像教程答题，而是变成一个小型工程维修任务。

---

# 3. 开场流程设计

## 3.1 点击关卡卡片

玩家在左侧 Chapter 0 选择：

```text
0-1 Shape Reader
Axis semantics
```

点击后不要直接进入画布，而是先进入一段很短的背景知识讲解。

---

## 3.2 背景知识讲解页

背景知识讲解详细内容参考 [knowledge_notes](./chapter0-1_knowledge_notes.md)

---

# 4. 进入画布后的任务弹窗

玩家进入画布后，先弹出任务说明，不要马上让玩家操作。

## 弹窗标题

```text
0-1 Shape Reader
Repair the Hidden Tensor Contract
```

## 弹窗正文

```text
Bootcamp 的第一台模型板已经启动，但 Hidden Tensor 的轴标签丢失了。

后续模块无法判断：
- 哪个轴是 batch
- 哪个轴是 token position
- 哪个轴是 channel

你的任务：
1. 恢复节点之间的数据流
2. 观察 Hidden Tensor 的三个轴
3. 将 B / T / C 标签贴到正确轴上
4. 连接到 Axis Decoder
5. 运行 Shape Tests
```

## 成功条件

```text
PASS 条件：
- Hidden Tensor rank = 3
- 三个轴都有语义标签
- B 轴能分离不同样本
- T 轴能对应 token 序列位置
- C 轴能被 Linear/Projection 消费
```

## 弹窗按钮

```text
Start Repair
```

关闭弹窗后进入真正画布。

---

# 5. 画布整体布局

画布仍然是节点拖拽连线式，但要把节点从“示意图”变成“可操作机器”。

## 5.1 默认画布结构

玩家进入后，中央画布上有一条半完成的数据管线：

```text
Text Batch → Tokenizer → Embedding Lookup → Hidden Tensor → Axis Decoder → Shape Tests
```

但是其中有几个故障：

```text
1. Hidden Tensor 的轴标签为空
2. Hidden Tensor 到 Axis Decoder 的合同线断开
3. Shape Tests 当前无法运行
```

画面可以这样布置：

```text
[Text Batch] ──blue──> [Tokenizer] ──blue──> [Embedding Lookup] ──blue──> [Hidden Tensor ? ? ?]
                                                                                  │
                                                                                  ╳ contract missing
                                                                                  │
                                                                        [Axis Decoder] ──green──> [Shape Tests]
```

节点本身可以拖拽，但这一关不要求玩家重新搭整条模型，只让玩家修复关键部分。

---

# 6. 节点设计

## 6.1 Text Batch 节点

作用：提供一批文本样本。

显示内容：

```text
Text Batch
text[B]

sample 0: "we train llm"
sample 1: "shape tells truth"
```

端口：

```text
output: text[B]
```

视觉：

像一个小型文本卡片堆，每张卡代表一个 batch 样本。

---

## 6.2 Tokenizer 节点

作用：把文本转换为 token id。

显示内容：

```text
Tokenizer
text[B] → int[B,T]
```

端口：

```text
input: text[B]
output: token_ids[B,T]
```

视觉：

文本卡片进入切割机，变成 token 方块。

---

## 6.3 Embedding Lookup 节点

作用：把 token id 查表变成 hidden tensor。

显示内容：

```text
Embedding Lookup
int[B,T] → float32[?, ?, ?]
```

端口：

```text
input: token_ids[B,T]
output: hidden tensor
```

视觉：

一个矩阵墙。token id 像地址一样点亮某些行，然后输出连续数值块。

---

## 6.4 Hidden Tensor 节点

这是本关主角。

显示内容初始为：

```text
Hidden Tensor
float32[?, ?, ?]
```

它不是普通方块，而是一个更形象的 3D 张量物体。

---

# 7. Hidden Tensor 的可视化设计

这一关是否有游戏感，关键就在 Hidden Tensor 的表现。

## 7.1 基础形象

Hidden Tensor 可以表现成一个半透明的 3D 数据盒：

```text
        C axis ?
       ↑
       │   feature bars
       │
       └────────→ T axis ?
      /
     /
B axis ?
```

它像一个透明数据仓库，内部由很多小格子组成。

推荐表现：

* 半透明立方体 / 长方体。
* 三个轴都有发光边框，但一开始都是问号。
* 每个格子不是完全随机噪点，而是能看出数据结构。
* 表面显示 dtype 和 shape：

```text
float32[2,4,8]
axis: unknown
```

## 7.2 三个轴的视觉差异

虽然标签未知，但玩家用探针观察后，会看到三个轴有不同“行为”。

### B 轴视觉：样本堆叠

当玩家探测 B 轴时，Tensor 会切成几张“样本切片”。

```text
B=0: "we train llm"
B=1: "shape tells truth"
```

表现：

* 整个 tensor 沿一个方向切成多层。
* 每层对应一个独立句子。
* 每层内部都有自己的 token 序列。
* 不同层之间不互相连续。

玩家直观感受：

> 这个轴是在存放多条独立样本。

---

### T 轴视觉：token 时间线

当玩家探测 T 轴时，Tensor 会沿 token 序列方向播放。

```text
T=0 → T=1 → T=2 → T=3
```

表现：

* 一个扫描光标沿这个轴移动。
* 下方同步显示对应 token：

```text
T0: we
T1: train
T2: llm
T3: <eos>
```

* 如果打开 Causal Mask Preview，会出现一个 `T × T` 下三角预览。

玩家直观感受：

> 这个轴是句子里的 token 位置。

---

### C 轴视觉：连续特征条

当玩家探测 C 轴时，不显示文字，而显示每个 token 内部的一排连续数值。

```text
hidden[0,2,:] = [-0.04, 0.11, -0.38, 0.27, ...]
```

表现：

* 选中一个 token 后，C 轴展开成一排 feature bars。
* 每个小条是一个 float32 数值。
* 这些值会被 Linear / Projection 模块消费。

玩家直观感受：

> 这个轴不是样本，也不是时间，而是每个 token 的特征维度。

---

# 8. 核心交互一：节点拖拽与连线

0-1 虽然重点是识别轴，但必须让玩家先做一点“修机器”的动作。

## 8.1 第一小目标：恢复数据流

关卡开始时，Text Batch、Tokenizer、Embedding Lookup 已经在画布上，但连接线断了一段。

玩家需要连接：

```text
Text Batch.output → Tokenizer.input
Tokenizer.output → Embedding Lookup.input
Embedding Lookup.output → Hidden Tensor.input
```

为什么要做这一步？

因为这会告诉玩家：

```text
文字不是直接变 hidden
而是 text → token ids → hidden tensor
```

这是一个实际操作，而不是看说明。

## 8.2 连线反馈

合法连接：

```text
text[B] → text[B]
```

线变蓝，端口吸附。

非法连接：

```text
text[B] → float32[?,?,?]
```

线变红，并显示：

```text
Port mismatch:
Text Batch outputs text[B],
but Hidden Tensor expects embedding activations.
```

这样从一开始就建立“类型与 shape 合同”的规则。

---

# 9. 核心交互二：探针观察

玩家恢复数据流后，Hidden Tensor 被生成，但轴标签仍然未知。

这时画布弹出一个小目标：

```text
Objective Updated:
Inspect the Hidden Tensor axes.
```

底部 Repair Console 出现三个工具：

```text
Batch Probe
Time Probe
Channel Probe
```

## 9.1 探针不是提示，而是观察工具

玩家选择一个 Probe，然后点击 Hidden Tensor 的某条轴。

比如点击 Axis 0，右侧 Inspector 会显示观察结果。

### 如果用 Batch Probe 点到了正确轴

```text
Probe Result:
This axis separates independent samples.

axis slice 0:
"we train llm"

axis slice 1:
"shape tells truth"
```

画布表现：

* Tensor 被切成两层。
* 每层上方出现 sample label。
* 这一轴旁边出现一个“疑似 B”的弱提示。

### 如果用 Batch Probe 点错了轴

例如点到了 T 轴：

```text
Probe Result:
This axis changes token position inside the same sample.
It does not separate independent examples.
```

画布表现：

* 扫描光标沿句子移动。
* 系统不说“错”，而是给观察证据。

这样玩家不是被判错，而是在形成判断。

---

## 9.2 三种探针的作用

### Batch Probe

用于判断哪个轴是 B。

反馈关键词：

```text
independent samples
separate examples
batch slices
```

### Time Probe

用于判断哪个轴是 T。

反馈关键词：

```text
ordered positions
token sequence
causal mask axis
```

### Channel Probe

用于判断哪个轴是 C。

反馈关键词：

```text
continuous features
float values
consumed by Linear
```

---

# 10. 核心交互三：拖拽轴标签

探测后，玩家需要把标签芯片拖到 Hidden Tensor 的三个轴槽上。

底部 Repair Console 显示：

```text
Available Axis Tags:
[B] Batch
[T] Token Position
[C] Channel
```

Hidden Tensor 三条轴上有空槽：

```text
Axis 0: [ empty ]
Axis 1: [ empty ]
Axis 2: [ empty ]
```

玩家拖拽后：

```text
Axis 0: [B]
Axis 1: [T]
Axis 2: [C]
```

此时 Hidden Tensor 节点显示从：

```text
float32[?, ?, ?]
```

变成：

```text
float32[B,T,C]
```

但注意，**不要在拖放瞬间告诉玩家对错**。
应该让玩家点击 Run Tests 后再验证，保持工程调试感。

---

# 11. 核心交互四：合同连线

为了让它更像节点游戏，而不是贴标签小游戏，建议加入一个“Axis Decoder”节点。

## 11.1 Axis Decoder 节点

Axis Decoder 有三个语义输入端口：

```text
B input
T input
C input
```

Hidden Tensor 标注完成后，会出现三个语义输出端口：

```text
B axis port
T axis port
C axis port
```

玩家需要把它们连到 Axis Decoder：

```text
Hidden Tensor.B → Axis Decoder.B
Hidden Tensor.T → Axis Decoder.T
Hidden Tensor.C → Axis Decoder.C
```

这样玩家的最终产物不是一个静态答案，而是一个可运行的合同线路。

## 11.2 连错时的表现

如果玩家把 T 接到 C：

```text
Hidden Tensor.T → Axis Decoder.C
```

连接可以暂时成立，但会在测试时失败。

失败提示：

```text
FAIL: Axis Decoder received token-position axis at channel input.

Why this fails:
Linear layers consume C, not T.
C is the per-token feature dimension.
```

这会强化概念。

---

# 12. 核心交互五：运行测试

玩家完成连线后，点击：

```text
Run Tests
```

测试分三层。

---

## 12.1 第一层：Visible Checks

立即显示的基础检查：

```text
✓ Data flow connected
✓ Hidden Tensor rank = 3
✓ All axes labeled
✓ Axis Decoder connected
```

这些是结构检查。

---

## 12.2 第二层：Behavior Checks

运行一个可视化 trace：

```text
Text Batch
→ Tokenizer
→ Embedding Lookup
→ Hidden Tensor
→ Axis Decoder
→ Shape Tests
```

数据沿蓝线流动。

每经过一个节点，节点底部显示实际 shape：

```text
Text Batch: text[B]
Tokenizer: int[B,T]
Embedding Lookup: float32[B,T,C]
Axis Decoder: contract resolved
```

然后验证：

```text
✓ B axis separates samples
✓ T axis matches token positions
✓ C axis contains continuous features
```

---

## 12.3 第三层：Hidden Tests

隐藏测试不是换题，而是换数据尺寸：

```text
hidden[1,8,16]
hidden[4,3,32]
hidden[2,12,6]
```

目标是防止玩家只根据尺寸大小猜。

结果显示：

```text
✓ hidden[1,8,16]
✓ hidden[4,3,32]
✓ hidden[2,12,6]
```

如果失败，提示：

```text
Hidden test failed:
Your contract depends on axis size, but axis meaning must remain stable when B/T/C values change.
```

---

# 13. 失败反馈设计

失败反馈要具体、工程化，不要说“答案错误”。

## 13.1 B 和 T 标反

提示：

```text
FAIL: Batch axis and token-position axis are swapped.

Observed:
- Axis marked B is ordered like a token sequence.
- Axis marked T separates independent samples.

Why this matters:
Causal Mask needs T to build a [T,T] attention matrix.
Batch axis should not be used for attention.
```

画面：

* 错误的 T 轴变红。
* Causal Mask Preview 出现错误：在 batch 之间画了 attention。
* 系统显示“不同样本之间不应该互相 attend”。

---

## 13.2 T 和 C 标反

提示：

```text
FAIL: Sequence axis and channel axis are swapped.

Observed:
- Axis marked T contains continuous feature values.
- Axis marked C contains token positions.

Why this matters:
Linear layers project C.
Attention builds relationships over T.
```

画面：

* Linear Preview 尝试消费 T，结果报红。
* Attention Preview 变成 `C × C`，提示这不是 token-to-token attention。

---

## 13.3 C 和 B 标反

提示：

```text
FAIL: Channel axis and batch axis are swapped.

Observed:
- Axis marked C separates independent text samples.
- Axis marked B contains feature values.

Why this matters:
Feature channels belong to each token.
Batch samples should remain independent during forward pass.
```

---

# 14. 通关反馈

通过后不要只显示 PASS，要让玩家看到“机器被修好”。

## 14.1 画布反馈

* Hidden Tensor 三个轴变成稳定颜色。
* 数据线从蓝色变成亮蓝。
* Axis Decoder 点亮。
* Shape Tests 输出绿色。
* 整条 pipeline 完整运行一次。

## 14.2 通关弹窗

标题：

```text
Contract Restored
```

正文：

```text
Hidden Tensor contract resolved:

float32[B,T,C]

B: independent samples
T: token positions
C: per-token feature channels

The board can now send activations into Linear and Attention modules.
```

奖励：

```text
Unlocked:
- Shape Inspector
- Axis Tags
- Tensor Probe
- 0-2 MatMul Gate
```

评分：

```text
Rank A
Probes used: 4
Hints used: 0
Hidden tests passed: 3/3
```

---

# 15. 底部区域重设计

你现在底部是选择题区域，可以改成 **Repair Console**。

## 15.1 Repair Console 布局

```text
┌──────────────────────────────────────────────────────────────┐
│ Objective                                                     │
│ Restore hidden tensor contract: float32[B,T,C]                │
├──────────────────────────────────────────────────────────────┤
│ Tools                                                         │
│ [Batch Probe] [Time Probe] [Channel Probe]                    │
├──────────────────────────────────────────────────────────────┤
│ Axis Tags                                                     │
│ [B Batch] [T Token Position] [C Channel]                      │
├──────────────────────────────────────────────────────────────┤
│ Checklist                                                     │
│ □ data flow connected                                         │
│ □ hidden tensor generated                                     │
│ □ axes labeled                                                │
│ □ Axis Decoder connected                                      │
│ □ tests passed                                                │
└──────────────────────────────────────────────────────────────┘
```

这样底部就从“题目答案区”变成了“工具台”。

---

# 16. 右侧 Inspector 重设计

右侧 Inspector 不应该直接告诉答案，而应该显示观察证据。

## 未选择时

```text
Tensor Inspector

Select a tensor or axis to inspect:
- dtype
- rank
- shape
- sample values
- observed axis behavior
```

## 选择 Hidden Tensor 时

```text
Hidden Tensor
dtype: float32
rank: 3
shape: [2,4,8]
contract: unresolved

axis labels:
Axis 0: ?
Axis 1: ?
Axis 2: ?
```

## 选择某个轴后

```text
Selected Axis: Axis 1
length: 4

Observed behavior:
- values advance with token position
- aligned with token_ids[B,T]
- candidate for causal attention

Likely role:
Token Position axis
```

早期关卡可以显示 `Likely role`，后期去掉，变成纯证据。

---

# 17. Tensor 形象化方向

为了让 tensor 更“可视化和形象化”，建议做成三层表现。

## 17.1 远看：数据盒

默认视图中，tensor 是一个完整的数据盒。

```text
Hidden Tensor
float32[2,4,8]
```

玩家可以看到：

* rank = 三个轴。
* shape = 三个尺寸。
* dtype = float32。
* 标签是否完整。

## 17.2 中看：切片结构

使用探针后，tensor 可以被切开。

比如 B Probe：

```text
Sample Slice 0
Sample Slice 1
```

T Probe：

```text
Token Position 0 → 1 → 2 → 3
```

C Probe：

```text
Feature 0...7
```

## 17.3 近看：单个数值

点击一个小格子，可以看到：

```text
hidden[b=0, t=2, c=5] = -0.381
```

再点击一整条 C 轴：

```text
hidden[0,2,:] = [-0.04, 0.11, -0.38, 0.27, ...]
```

这会让玩家逐渐理解：

```text
tensor 不是抽象盒子
而是一组有结构的数值
```

---

# 18. 0-1 的完整玩家流程

下面是玩家实际体验的一遍流程。

---

## Step 1：背景知识

玩家看到 3 张知识卡：

```text
LLM 不直接读取文字
shape 只说明大小
hidden[B,T,C] 是 Transformer 的基础激活格式
```

点击 Enter Workbench。

---

## Step 2：任务弹窗

系统说明：

```text
Hidden Tensor 的轴标签丢失。
修复它，让后续模块能读取正确的 B/T/C 合同。
```

点击 Start Repair。

---

## Step 3：进入节点画布

玩家看到损坏管线：

```text
Text Batch → Tokenizer → Embedding Lookup → Hidden Tensor ? ? ? → Axis Decoder → Shape Tests
```

目标提示：

```text
Objective 1:
Restore data flow into Hidden Tensor.
```

---

## Step 4：连接数据线

玩家连接 Text Batch、Tokenizer、Embedding Lookup、Hidden Tensor。

成功后：

```text
Hidden Tensor generated:
float32[2,4,8]
contract unresolved
```

目标更新：

```text
Objective 2:
Inspect the three axes.
```

---

## Step 5：使用探针

玩家选择 Time Probe，点击一个轴。

系统显示 token 序列扫描。

玩家判断这是 T，把 `[T]` 标签拖过去。

玩家选择 Batch Probe，看到样本切片，把 `[B]` 标签拖过去。

玩家选择 Channel Probe，看到 feature bars，把 `[C]` 标签拖过去。

Hidden Tensor 显示：

```text
float32[B,T,C]
```

目标更新：

```text
Objective 3:
Connect semantic axes to Axis Decoder.
```

---

## Step 6：连接合同线

玩家把 Hidden Tensor 的三个语义端口连接到 Axis Decoder：

```text
B → B input
T → T input
C → C input
```

目标更新：

```text
Objective 4:
Run Shape Tests.
```

---

## Step 7：运行测试

玩家点击 Run Tests。

数据流开始 trace：

```text
Text Batch → Tokenizer → Embedding Lookup → Hidden Tensor → Axis Decoder → Shape Tests
```

测试面板显示：

```text
Visible Checks:
✓ Data flow connected
✓ Hidden Tensor rank = 3
✓ Axis labels assigned
✓ Axis Decoder connected

Behavior Checks:
✓ B separates independent samples
✓ T matches token positions
✓ C contains feature channels

Hidden Checks:
✓ hidden[1,8,16]
✓ hidden[4,3,32]
✓ hidden[2,12,6]
```

通关。

---

# 19. 关卡状态机

可以按这个状态实现。

```ts
type LevelState =
  | "knowledge_intro"
  | "mission_modal"
  | "canvas_entered"
  | "data_flow_repair"
  | "tensor_generated"
  | "axis_probe"
  | "axis_tagging"
  | "contract_wiring"
  | "visible_testing"
  | "hidden_testing"
  | "completed";
```

每个阶段有一个清晰目标。

```ts
const objectives = {
  data_flow_repair: "Connect Text Batch, Tokenizer, Embedding Lookup, and Hidden Tensor.",
  tensor_generated: "Inspect the Hidden Tensor axes.",
  axis_probe: "Use probes to discover axis behavior.",
  axis_tagging: "Assign B, T, C tags to the Hidden Tensor.",
  contract_wiring: "Connect semantic axes to Axis Decoder.",
  visible_testing: "Run visible shape tests.",
  hidden_testing: "Pass hidden shape cases."
};
```

---

# 20. 关卡数据结构示例

可以这样组织 0-1 的关卡配置。

```ts
const level_0_1 = {
  id: "0-1",
  title: "Shape Reader",
  chapter: "Tensor Bootcamp",
  mode: "Build + Trace",

  knowledgeCards: [
    {
      title: "LLM 不直接读取文字",
      body: "文本会先被切成 token，再变成数字序列，最后进入模型内部的 hidden tensor。"
    },
    {
      title: "shape 只说明大小",
      body: "[2,4,8] 只说明张量有三个轴。每个轴代表什么，需要语义标签。"
    },
    {
      title: "hidden[B,T,C]",
      body: "B 是 batch，T 是 token position，C 是每个 token 的 feature channel。"
    }
  ],

  mission: {
    title: "Repair the Hidden Tensor Contract",
    body: "Hidden Tensor 的轴标签丢失。恢复 B/T/C 语义，让 Axis Decoder 和 Shape Tests 可以运行。",
    success: [
      "Hidden Tensor rank = 3",
      "All axes labeled",
      "B separates samples",
      "T matches token positions",
      "C enters Linear/Projection modules"
    ]
  },

  nodes: [
    {
      id: "text_batch",
      type: "TextBatch",
      output: "text[B]",
      initialPosition: [80, 240]
    },
    {
      id: "tokenizer",
      type: "Tokenizer",
      input: "text[B]",
      output: "int[B,T]",
      initialPosition: [280, 240]
    },
    {
      id: "embedding_lookup",
      type: "EmbeddingLookup",
      input: "int[B,T]",
      output: "float32[?,?,?]",
      initialPosition: [500, 240]
    },
    {
      id: "hidden_tensor",
      type: "TensorBlock",
      shape: [2, 4, 8],
      dtype: "float32",
      axisLabels: [null, null, null],
      initialPosition: [740, 220]
    },
    {
      id: "axis_decoder",
      type: "AxisDecoder",
      requiredContract: "float32[B,T,C]",
      initialPosition: [1020, 240]
    },
    {
      id: "shape_tests",
      type: "ShapeTests",
      initialPosition: [1240, 240]
    }
  ],

  tools: [
    "BatchProbe",
    "TimeProbe",
    "ChannelProbe",
    "AxisTagB",
    "AxisTagT",
    "AxisTagC"
  ],

  checks: [
    {
      id: "data_flow_connected",
      label: "Data flow connected"
    },
    {
      id: "rank_is_3",
      label: "Hidden Tensor rank = 3"
    },
    {
      id: "axes_labeled",
      label: "All axes labeled"
    },
    {
      id: "axis_contract",
      label: "Axis contract is B,T,C"
    },
    {
      id: "decoder_connected",
      label: "Axis Decoder connected"
    }
  ],

  hiddenTests: [
    { shape: [1, 8, 16] },
    { shape: [4, 3, 32] },
    { shape: [2, 12, 6] }
  ],

  rewards: [
    "Shape Inspector",
    "Tensor Probe",
    "Axis Contract: hidden[B,T,C]",
    "Next Level: 0-2 MatMul Gate"
  ]
};
```

---

# 21. 让它更有“游戏感”的几个小细节

## 21.1 数据流运行动画

点击 Run Tests 后，不要直接出结果。
让 token 小方块沿节点线流动：

```text
Text → Token IDs → Hidden Tensor → Axis Decoder → Tests
```

每一步节点亮起，并显示 shape。

---

## 21.2 合同线和数据线分颜色

建议：

```text
蓝色：activation / tensor data
紫色：axis contract / semantic labels
绿色：test / validation
橙色：warning / repair path
红色：failed connection
```

这样玩家能一眼区分“数据流”和“语义合同”。

---

## 21.3 局部通过反馈

玩家每完成一个小目标，给轻反馈：

```text
Data flow restored
Hidden Tensor generated
Axis label assigned
Decoder input complete
Visible tests ready
```

不要等最后才反馈。

---

## 21.4 提示系统做成 Debug Hint

不要叫 Hint，可以叫：

```text
Debug Probe
Reference Trace
Contract Hint
```

提示内容也要像工程调试：

```text
The failed module is Axis Decoder.
It expected C at input 3, but received an axis used by causal mask.
```

---

# 22. 0-1 的最终体验目标

这一关结束时，玩家应该不是记住了一道题，而是产生下面的直觉：

```text
hidden[B,T,C] 是 Transformer 的基础激活张量。
B 是样本维度，不参与 token 间 attention。
T 是 token 位置，attention 和 causal mask 围绕它工作。
C 是每个 token 的特征通道，Linear / MLP 会消费它。
shape contract 是节点能否连接和运行的基础。
```

这就为后面的 0-2 MatMul Gate、0-3 Transpose Trap、0-4 Broadcast Add 打下了非常自然的基础。

0-2 里玩家会马上用到：

```text
Linear 接收 hidden[B,T,C]
权重是 [C,O]
输出是 [B,T,O]
```

这时 0-1 学到的 C 轴就不再是一个概念，而是后续玩法里的真实接口。
