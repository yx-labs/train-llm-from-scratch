# LLM Complete — 0-4 Broadcast Add 可执行详细设计文档

> 章节定位：Chapter 0 / Tensor Bootcamp 的第四关。  
> 前置关卡：0-1 Shape Reader，玩家已经掌握 `Tensor / Rank / Shape / Axis Semantics / hidden[B,T,C]`；0-2 MatMul Gate，玩家已经掌握 `MatMul / Linear Projection / C→O`；0-3 Transpose Trap，玩家已经掌握 `Transpose / QKᵀ / scores[B,H,T,T]`。  
> 本关目标：让玩家通过节点画布、Broadcast Rail、轴对齐器、Ghost Expansion 可视化、Add Gate 和隐藏测试，真正理解 **Broadcast Add 不是随便相加，而是在满足轴对齐合同的前提下，将小张量逻辑扩展到大张量上执行 elementwise add**。

---

## 0. 设计总述

### 0.1 关卡一句话

玩家需要修复一台损坏的 **Broadcast Add Station**。系统会给出多种 LLM 中常见的加法场景：

```text
hidden[B,T,C] + bias[C]        → hidden[B,T,C]
tok_emb[B,T,C] + pos_emb[T,C] → hidden[B,T,C]
scores[B,H,T,T] + mask[1,1,T,T] → masked_scores[B,H,T,T]
residual_a[B,T,C] + residual_b[B,T,C] → hidden[B,T,C]
```

玩家要在节点画布上对齐语义轴，配置广播规则，并通过 Add Gate 与 Reference Checker 验证结果。

这一关的核心不是让玩家背“broadcast 会自动扩展”，而是让玩家亲手发现：

```text
Broadcast Add 的本质是：
在每个输出位置上，从大张量和小张量中找到对应值，然后相加。

小张量缺少的轴，会被逻辑上重复。
长度为 1 的轴，也可以被逻辑上扩展。
但轴语义必须正确。
```

---

### 0.2 设计原则

0-4 延续 0-1 / 0-2 / 0-3 的组织结构：

```text
Concept Capsule / 最小概念讲解
→ Broken Board / 故障画布
→ Repair Operation / 玩家修复
→ Probe & Trace / 探针观察
→ Autograder / 可验证测试
→ Debrief / 阶段小结
→ Tool Unlock / 工具解锁
```

本关不做选择题，不把 broadcast 讲成魔法自动机制，而是让玩家形成工程直觉：

```text
Add 是逐元素运算。
同 shape 张量可以直接逐元素相加。
不同 shape 的张量只有在满足 broadcast contract 时才能相加。
Broadcast 会从右侧轴开始对齐，也可以用语义轴明确对齐。
缺失轴相当于长度为 1 的轴。
长度为 1 的轴可以逻辑扩展到目标长度。
Broadcast 通常不复制真实数据，而是复用同一组值。
Bias Add、Position Add、Mask Add 都是 LLM 中真实存在的 broadcast 场景。
Shape 能 broadcast 不代表语义正确，必须检查 axis semantics。
```

---

### 0.3 本关学习成果

玩家通关后应该能自然理解：

```text
1. Elementwise Add 要求输出的每个 cell 都来自两个输入的对应 cell 相加。
2. Same-shape Add 是最简单的逐元素加法：[B,T,C] + [B,T,C] → [B,T,C]。
3. Broadcast Add 允许小张量在缺失轴或长度为 1 的轴上逻辑重复。
4. bias[C] 可以加到 hidden[B,T,C] 的每个 token vector 上。
5. pos_emb[T,C] 可以加到 tok_emb[B,T,C] 的每个 batch 样本上。
6. causal mask[1,1,T,T] 或 mask[T,T] 可以加到 scores[B,H,T,T] 上。
7. B/H/T/C/O 等轴语义决定了 broadcast 是否安全。
8. `[B,T,C] + [T]` 通常不是合法的 position add，因为 `[T]` 会按右对齐尝试匹配 C。
9. `[B,T,C] + [B,C]` 在某些尺寸相等时可能 shape 通过，但语义错误。
10. Broadcast Debug 的顺序是：检查 rank → 检查 axis alignment → 检查 length=1 / missing axes → 检查语义 → 数值验证。
```

---

## 1. 与前置关卡的衔接

### 1.1 从 0-1 继承的知识

0-1 中，玩家已经理解：

```text
hidden[B,T,C]
B = batch
T = token position
C = channel / feature dimension
```

0-4 会继续使用这个 hidden tensor，但增加一个新的问题：

```text
当一个小张量只有 [C] 或 [T,C] 时，
它如何与 hidden[B,T,C] 相加？
```

如果玩家只是知道 shape，不知道轴语义，就很容易把 `[T]`、`[C]`、`[B]` 混淆。

---

### 1.2 从 0-2 继承的知识

0-2 中，玩家已经搭过：

```text
hidden[B,T,C] @ weight[C,O] → projected[B,T,O]
```

真实 Linear 通常还会加一个 bias：

```text
projected[B,T,O] + bias[O] → projected_with_bias[B,T,O]
```

0-4 会让玩家修复这个 bias add，理解 `bias[O]` 会被加到每个 batch、每个 token 的输出向量上。

---

### 1.3 从 0-3 继承的知识

0-3 中，玩家已经得到：

```text
scores[B,H,T,T] = Q[B,H,T,D] @ Kᵀ[B,H,D,T]
```

在真正进入 softmax 前，causal mask 会被加到 scores 上：

```text
masked_scores[B,H,T,T] = scores[B,H,T,T] + causal_mask[1,1,T,T]
```

0-4 会提前铺垫这个机制。玩家会看到 mask 不是替换 scores，而是通过 broadcast add 把非法位置变成一个很大的负数。

---

### 1.4 本关在 Chapter 0 中的位置

Chapter 0 的四个基础能力到此闭环：

```text
0-1 Shape Reader      读懂张量结构和语义轴
0-2 MatMul Gate       学会内维度消费与输出轴生成
0-3 Transpose Trap    学会轴顺序改变与 QKᵀ
0-4 Broadcast Add     学会小张量如何安全加到大张量上
```

通过 0-4 后，玩家已经具备进入 Attention Block / Transformer Block 的最低 shape 调试能力。

---

## 2. 本关整体结构

### 2.1 阶梯挑战总览

| 阶段 | 名称 | 玩家学到什么 | 主要操作 | 验证方式 |
|---|---|---|---|---|
| 0-4A | Add Cell | 单个输出值来自两个值相加 | 连接两个 scalar / cell | 数值 exact match |
| 0-4B | Same-Shape Add | 同 shape 张量逐元素相加 | 对齐两个 `[T,C]` 或 `[B,T,C]` | 输出 shape 与数值匹配 |
| 0-4C | Broadcast Rule Lab | 缺失轴 / 1 轴如何逻辑扩展 | 使用 Broadcast Rail 对齐轴 | 广播映射正确 |
| 0-4D | Bias Add | `bias[C/O]` 如何加到每个 token vector | 把 bias strip 接到 C/O 轴 | 每个 B/T 位置复用 bias |
| 0-4E | Position Add | `pos_emb[T,C]` 如何加到 `tok_emb[B,T,C]` | 对齐 T/C，广播 B | 输出 `hidden[B,T,C]` |
| 0-4F | Mask Add | `mask[1,1,T,T]` 如何加到 `scores[B,H,T,T]` | 对齐 score row/col T，广播 B/H | 非法位置被加负值 |
| 0-4G | Broadcast Trap Debugger | shape 可广播但语义可能错 | 调试无效 / 误导 case | 结构 + 语义 + 数值检查 |
| 0-4X | Broadcast Gauntlet | 泛化掌握 broadcast add | 多 case 限制探针修复 | 隐藏测试全部通过 |

---

### 2.2 关卡主流程

```text
开场总引导
→ 0-4A Add Cell
→ 0-4B Same-Shape Add
→ 0-4C Broadcast Rule Lab
→ 0-4D Bias Add
→ 0-4E Position Add
→ 0-4F Mask Add
→ 0-4G Broadcast Trap Debugger
→ 0-4X Broadcast Gauntlet
→ Chapter 0 总结与后续 Attention 章节预告
```

每个阶段开始前弹出 Concept Capsule，阶段结束后弹出 Debrief。失败时不显示“回答错误”，而显示工程化失败原因。

---

## 3. 关卡开场总引导

### 3.1 开场界面标题

```text
0-4 Broadcast Add
Repair the Add Station
```

---

### 3.2 开场文案

```text
你已经学会读取 hidden[B,T,C]，
也学会了通过 MatMul 和 Transpose 生成新张量。

现在，模型板上出现了另一种常见操作：Add。

在 LLM 中，Add 不只是两个相同大小的张量相加。
很多时候，一个小张量会被广播到大张量上：

hidden[B,T,C] + bias[C]
tok_emb[B,T,C] + pos_emb[T,C]
scores[B,H,T,T] + mask[1,1,T,T]

你的任务是修复 Broadcast Add Station，
让每一次加法都满足 shape contract 和 axis semantics。
```

按钮：

```text
Start Broadcast Bootcamp
```

---

### 3.3 开场图示

```text
Same Shape Add
A[B,T,C] + B[B,T,C] → Out[B,T,C]

Broadcast Bias Add
hidden[B,T,C] + bias[C] → hidden[B,T,C]

Position Add
tok_emb[B,T,C] + pos_emb[T,C] → hidden[B,T,C]

Mask Add
scores[B,H,T,T] + mask[1,1,T,T] → masked_scores[B,H,T,T]
```

画面建议：四条小型管线并排展示，每条都流入同一个 Add Gate。

---

### 3.4 开场知识介绍卡片扩展

#### Card 0-4-Intro-1：Add 是逐元素运算

**玩家可见文案：**

```text
Add Gate 会为输出 tensor 的每个 cell 计算一个值。

最简单的情况是两个输入 shape 完全一样：
A[i] + B[i] = Out[i]
```

**图示：**

```text
A:   [1, 2, 3]
B:   [4, 5, 6]
Out: [5, 7, 9]
```

**微交互：** 玩家点击 `Out[1]`，系统高亮 `A[1]` 和 `B[1]`，显示 `2 + 5 = 7`。

---

#### Card 0-4-Intro-2：Broadcast 让小张量参与大张量计算

**玩家可见文案：**

```text
Broadcast 允许一个小张量在某些轴上逻辑重复。

例如 bias[C] 可以加到 hidden[B,T,C] 的每个 token 上。
它不是复制出 B*T 份真实数据，而是在计算时复用同一条 bias。
```

**图示：**

```text
hidden[B,T,C]
+ bias[    C]
= out[B,T,C]
```

**画面表现：** `bias[C]` 是一条短 feature strip，被投影到 hidden 数据柜中每个 token 格子上，形成半透明 ghost strips。

---

#### Card 0-4-Intro-3：本关要修复什么

**玩家可见文案：**

```text
本关的 Add Station 已经损坏。
它会把 bias 接到错误轴，
把 position embedding 当成 channel bias，
或者让 mask 在 batch/head 轴上错误扩展。

你要用 Broadcast Rail 和 Axis Inspector 修复这些错误。
```

**任务桥接语：**

```text
第一步，从最小的 Add Cell 开始。
```

---

## 4. 核心视觉语言

### 4.1 Tensor Tile / 张量格

用于表示 elementwise add 中的单个数值。

视觉建议：

```text
- 每个 cell 是一个小方格，内含数值。
- 被追踪的 cell 高亮。
- 两个输入 cell 通过细线汇入输出 cell。
```

用途：

```text
0-4A Add Cell
0-4B Same-Shape Add
0-4G Trap Debugger 的 cell trace
```

---

### 4.2 Tensor Cabinet / 张量数据柜

沿用 0-1 的 3D 数据柜，但在本关加入 broadcast 预览。

```text
B 轴：样本层
T 轴：token 格
C/O 轴：feature 条
H 轴：head 层
T_row/T_col：attention score matrix 的行列 token 轴
```

当一个小张量被广播时，数据柜中出现半透明 ghost copies，但 Inspector 明确说明：

```text
Ghost expansion is logical. No real data copy is required.
```

---

### 4.3 Broadcast Rail / 广播轨道

这是本关核心交互工具。

Broadcast Rail 是一条对齐槽，显示目标 shape 与输入 shape：

```text
Target: [B][T][C]
Input :       [C]
Result: [B][T][C]
```

或：

```text
Target: [B][T][C]
Input :    [T][C]
Result: [B][T][C]
```

玩家需要把小张量拖到正确的语义槽上。对齐正确时出现绿色 ghost expansion；对齐错误时出现红色冲突。

---

### 4.4 Axis Alignment Ruler / 轴对齐尺

用于解释从右对齐和语义对齐的关系。

默认规则视图：

```text
[B][T][C]
      [C]
```

语义安全视图：

```text
hidden[B][T][C]
bias         [C]
```

对于容易混淆的 case，系统会要求玩家切换到 semantic alignment 模式，手动声明小张量的轴语义。

---

### 4.5 Add Gate / 加法门

Add Gate 有两个输入：

```text
A input
B input
```

以及一个输出：

```text
Out
```

节点面板显示：

```text
A: float32[B,T,C]
B: float32[C]
Broadcast Plan: B expands over B,T
Output: float32[B,T,C]
```

当 Add Gate 运行时，数据流不是一次性通过，而是显示：

```text
1. align axes
2. build broadcast plan
3. trace sample cell
4. run full add
5. compare reference
```

---

### 4.6 Bias Strip / 偏置条

`bias[C]` 或 `bias[O]` 表现为一条 feature strip：

```text
bias[C] = [b0, b1, b2, ...]
```

它会被贴到每个 token vector 的 C/O 轴上。

---

### 4.7 Position Sheet / 位置表

`pos_emb[T,C]` 表现为一张二维 sheet：

```text
每一行是一个 token position
每一行有 C 个 feature 值
```

它会沿 B 轴广播。

---

### 4.8 Mask Plate / 遮罩板

`mask[1,1,T,T]` 或 `mask[T,T]` 表现为一张下三角/上三角区域分明的板：

```text
0      allowed
-∞     blocked
```

它会沿 B/H 轴广播到每个 batch 和 head 的 score matrix 上。

---

### 4.9 颜色建议

```text
蓝色：主 activation / hidden / scores
紫色：broadcasted small tensor / ghost expansion
绿色：合法 broadcast plan
黄色：需要语义确认的可疑 broadcast
红色：非法 broadcast / 语义错误
灰色：逻辑重复的 ghost copy
橙色：mask 负值区域
```

---

## 5. 0-4A：Add Cell / 单格加法

### 5.1 阶段目标

让玩家理解：

```text
Add 的最小单元是两个数值相加。
```

本阶段不讲 broadcast，只讲 elementwise add 的 cell trace。

---

### 5.2 Concept Capsule

```text
Add Gate 的输出中，每一个 cell 都来自两个输入 cell 的相加。

如果 A 和 B 的 shape 一样，
Out 的每个位置都可以直接找到 A 和 B 的对应位置。
```

工具解锁：

```text
Add Gate
Cell Trace Probe
```

任务：

```text
连接两个数值 cell，生成正确的输出 cell。
```

---

#### 5.2.1 挑战前知识介绍卡片扩展

##### Card 0-4A-1：一个输出值来自两个输入值

**玩家可见文案：**

```text
加法最小的单位是一个数值。

如果输入是：
A = 2.0
B = 5.0

输出就是：
Out = 7.0
```

**图示：**

```text
[A: 2.0] ─┐
          + → [Out: 7.0]
[B: 5.0] ─┘
```

**微交互：** 玩家拖动 A 和 B 到 Add Gate，输出 cell 点亮。

---

##### Card 0-4A-2：Elementwise 表示逐位置计算

**玩家可见文案：**

```text
当输入不是一个数，而是一排数时，
Add Gate 会在每个位置分别执行一次加法。

位置 0 加位置 0，
位置 1 加位置 1，
位置 2 加位置 2。
```

**图示：**

```text
A[0] + B[0] → Out[0]
A[1] + B[1] → Out[1]
A[2] + B[2] → Out[2]
```

---

##### Card 0-4A-3：先会追踪一个 cell，再扩展到整个 tensor

**玩家可见文案：**

```text
本关会反复使用 Cell Trace。

当一个 broadcast 出错时，
最可靠的调试方式就是追踪某个输出 cell 到底用了哪些输入值。
```

**任务桥接语：**

```text
先让一个 Add Cell 通过测试。
```

---

### 5.3 画布初始状态

```text
[Scalar A: 2.0]    [Scalar B: 5.0]
       \             /
        \           /
         [Add Gate]
              ↓
       [Output Cell ?]
```

Add Gate 初始未连接。

---

### 5.4 玩家操作

1. 将 `Scalar A` 连接到 Add Gate 的 A input。
2. 将 `Scalar B` 连接到 Add Gate 的 B input。
3. 点击 `Run Cell Trace`。
4. 输出 cell 显示计算：

```text
2.0 + 5.0 = 7.0
```

---

### 5.5 交互反馈

合法连接：

```text
float32[] → Add input
```

运行时动画：

```text
A cell 和 B cell 同时飞入 Add Gate，输出 cell 亮起。
```

Inspector：

```text
Operation: elementwise add
A value: 2.0
B value: 5.0
Output: 7.0
```

---

### 5.6 验证规则

```text
✓ A input connected
✓ B input connected
✓ dtype compatible
✓ output = A + B
✓ output shape = []
```

---

### 5.7 失败反馈

#### 只连接一个输入

```text
Add Gate incomplete.

Elementwise Add requires two input values.
Connect both A and B inputs.
```

#### dtype 不兼容

```text
Dtype mismatch.

A is float32, but B is text.
Add Gate only accepts numeric tensors.
```

#### 输出被手动改值

```text
Reference mismatch.

Output cell must be computed from A + B.
Manual output value does not match reference.
```

---

### 5.8 阶段小结

```text
Stage Complete: Add Cell

你已经看到：Add 的本质是逐位置的数值相加。
下一步，我们会把一个 cell 扩展到一整个同 shape tensor。
```

解锁：

```text
Tensor Add Grid
```

---

## 6. 0-4B：Same-Shape Add / 同形状张量相加

### 6.1 阶段目标

让玩家理解：

```text
如果两个 tensor shape 完全相同，可以直接逐元素相加。
```

这一步为后续 residual add 和 broadcast add 做基础。

---

### 6.2 Concept Capsule

```text
Same-shape Add 是最简单的 tensor add。

A[T,C] + B[T,C] → Out[T,C]

输出保留同样的 shape，
每个输出 cell 都来自同位置的 A cell 和 B cell。
```

工具解锁：

```text
Grid Trace Probe
Shape Match Gate
```

任务：

```text
修复一张 [T,C] + [T,C] 的 Add Board。
```

---

#### 6.2.1 挑战前知识介绍卡片扩展

##### Card 0-4B-1：同 shape 可以直接对齐

**玩家可见文案：**

```text
如果两个 tensor 的 shape 完全一样，
它们的每个位置都能一一对应。

A[T,C] 和 B[T,C]
可以直接逐元素相加。
```

**图示：**

```text
A[0,0] + B[0,0] → Out[0,0]
A[0,1] + B[0,1] → Out[0,1]
A[1,0] + B[1,0] → Out[1,0]
```

---

##### Card 0-4B-2：输出 shape 不会变

**玩家可见文案：**

```text
Same-shape Add 不会产生新轴，
也不会消耗某个轴。

A[T,C] + B[T,C]
输出仍然是 [T,C]。
```

**微交互：** 玩家点击输出 shape `[T,C]`，系统高亮两个输入的 T/C 轴。

---

##### Card 0-4B-3：Residual Add 就是同 shape add

**玩家可见文案：**

```text
Transformer 里经常出现 residual add。

例如：
main_path[B,T,C] + residual[B,T,C]

这类加法通常要求两个输入 shape 完全相同。
```

**任务桥接语：**

```text
先修复一个小型 same-shape add。
```

---

### 6.3 画布初始状态

```text
A Tensor: float32[T=2,C=3]
B Tensor: float32[T=2,C=3]

A ──?──┐
       [Add Gate] → Out[?,?]
B ──?──┘
```

输出节点未声明 shape。

---

### 6.4 玩家操作

1. 将 A 和 B 连接到 Add Gate。
2. 使用 Shape Match Gate 检查两个输入 shape 是否相同。
3. 将输出合同设置为：

```text
Out[T,C]
```

4. 使用 Grid Trace Probe 检查一个输出 cell：

```text
Out[1,2] = A[1,2] + B[1,2]
```

---

### 6.5 可视化表现

- 两个 `[T,C]` 矩阵板叠在一起。
- 点击输出 cell，两个输入中同坐标 cell 高亮。
- 运行时每个位置出现一条短线汇入输出。

---

### 6.6 验证规则

```text
✓ input A shape == input B shape
✓ output shape == input shape
✓ axis labels preserved
✓ each output cell uses same coordinate from A and B
✓ numeric output allclose reference
```

---

### 6.7 失败反馈

#### 输出 shape 设置错

```text
Output contract error.

Same-shape Add does not create a new axis.
Expected output shape: [T,C]
Received: [T,O]
```

#### cell 来源错位

```text
Cell trace failed.

Out[1,2] should use A[1,2] and B[1,2].
Your trace used B[0,2].
```

#### 轴语义被换掉

```text
Axis semantics changed unexpectedly.

Add should preserve T and C.
It should not transpose or project the tensor.
```

---

### 6.8 阶段小结

```text
Stage Complete: Same-Shape Add

你已经掌握最简单的 tensor add：
同 shape、同坐标、同语义轴逐元素相加。

下一步，我们会让一个更小的 tensor 加到大 tensor 上。
```

解锁：

```text
Broadcast Rail
Ghost Expansion View
```

---

## 7. 0-4C：Broadcast Rule Lab / 广播规则实验室

### 7.1 阶段目标

让玩家理解：

```text
Broadcast 允许小张量在缺失轴或长度为 1 的轴上逻辑扩展。
```

这一阶段先讲通用规则，不绑定具体 LLM 模块。

---

### 7.2 Concept Capsule

```text
Broadcast Add 会先尝试把两个输入对齐到同一个输出 shape。

一个轴可以 broadcast 的条件：
- 两边长度相同；或者
- 其中一边长度为 1；或者
- 小张量缺少这个轴，可以视作长度为 1。

但在本游戏里，还要检查 axis semantics。
```

工具解锁：

```text
Broadcast Rail
Axis Alignment Ruler
Ghost Expansion Probe
```

任务：

```text
把小张量对齐到目标 shape，并生成合法 broadcast plan。
```

---

#### 7.2.1 挑战前知识介绍卡片扩展

##### Card 0-4C-1：缺失轴可以被看作长度 1

**玩家可见文案：**

```text
当你把 [C] 加到 [B,T,C] 上时，
[C] 缺少 B 和 T 轴。

Broadcast 会把它理解成：
[1,1,C]

然后沿 B 和 T 逻辑重复。
```

**图示：**

```text
Target: [B][T][C]
Input :       [C]
View  : [1][1][C]
```

---

##### Card 0-4C-2：长度为 1 的轴可以扩展

**玩家可见文案：**

```text
如果某个轴长度是 1，
它可以被逻辑扩展到目标长度。

例如：
[1,T,C] 可以加到 [B,T,C]。
```

**图示：**

```text
[1,T,C]
 ↓ expand B
[B,T,C]
```

---

##### Card 0-4C-3：Broadcast 不是复制粘贴

**玩家可见文案：**

```text
Broadcast 通常不会真的复制出很多份数据。

它只是告诉 Add Gate：
当访问缺失轴或长度为 1 的轴时，
重复使用同一个值。
```

**微交互：** 玩家打开 Ghost Expansion，看到许多半透明副本，但 Inspector 显示：

```text
real storage: [C]
logical view: [B,T,C]
```

---

### 7.3 画布初始状态

三个练习 case 依次出现：

#### Case A：vector 到 matrix

```text
A[T,C]
B[C]
Target: [T,C]
```

#### Case B：matrix 到 3D tensor

```text
A[B,T,C]
B[T,C]
Target: [B,T,C]
```

#### Case C：singleton axis

```text
A[B,T,C]
B[1,T,1]
Target: [B,T,C]
```

每个 case 都需要玩家在 Broadcast Rail 上对齐输入轴。

---

### 7.4 玩家操作

1. 将大张量放入 Target Rail。
2. 将小张量放入 Input Rail。
3. 调整对齐方式：

```text
right-align
semantic-align
insert singleton axis
```

4. 点击 `Preview Broadcast`。
5. 观察 ghost expansion。
6. 运行 `Broadcast Plan Tests`。

---

### 7.5 可视化表现

Broadcast Rail 示例：

```text
Target: [B][T][C]
Input :    [T][C]
Plan  : [*][T][C]
```

`*` 表示沿该轴逻辑扩展。

对于 singleton：

```text
Target: [B][T][C]
Input : [1][T][1]
Plan  : [*][T][*]
```

Ghost Expansion：

- 小张量真实体是实线。
- 扩展副本是半透明虚线。
- 被复用的值有共享来源线。

---

### 7.6 验证规则

```text
✓ output rank equals target rank
✓ each input axis aligned to valid target axis
✓ every unmatched target axis is broadcastable
✓ every singleton axis expands legally
✓ semantic axes are compatible
✓ sample cell trace matches reference
```

---

### 7.7 失败反馈

#### 长度不匹配且都不是 1

```text
Broadcast failed: incompatible axis length.

Target axis T has length 4.
Input axis has length 3.
Neither side is length 1.
```

#### 缺失轴位置错了

```text
Broadcast alignment error.

You aligned [C] to T instead of C.
The values would repeat along channel and vary along token position,
which is not a valid bias-style broadcast.
```

#### 只看 shape，语义不对

```text
Semantic broadcast warning.

Shape can broadcast,
but the input axis labeled T is aligned to target C.

Broadcast is shape-valid but semantic-invalid.
```

---

### 7.8 阶段小结

```text
Stage Complete: Broadcast Rule Lab

你已经掌握：
Broadcast 会建立一个 logical view。
缺失轴或长度为 1 的轴可以重复。
但 axis semantics 必须正确。

下一步，我们把这个规则应用到 Linear Bias Add。
```

解锁：

```text
Bias Strip
Linear Bias Add Board
```

---

## 8. 0-4D：Bias Add / 偏置广播

### 8.1 阶段目标

让玩家理解：

```text
Linear 输出 projected[B,T,O] 可以加 bias[O]。
```

也可以支持 `hidden[B,T,C] + bias[C]` 作为同类练习，但本阶段默认使用 0-2 的 Linear 输出轴 `O`，与前置关卡形成衔接。

---

### 8.2 Concept Capsule

```text
Linear Projection 通常会有 bias。

projected[B,T,O] + bias[O] → projected_with_bias[B,T,O]

bias[O] 会被加到每个 batch、每个 token 的 O 维输出向量上。
```

工具解锁：

```text
Bias Strip
Bias Broadcast Probe
```

任务：

```text
把 bias[O] 安全广播到 projected[B,T,O] 上。
```

---

#### 8.2.1 挑战前知识介绍卡片扩展

##### Card 0-4D-1：Bias 是每个输出通道的偏移量

**玩家可见文案：**

```text
Linear 输出的每个 O 通道都可以有一个 bias。

如果 O = 4，
bias[O] 就是一条长度为 4 的向量。

它会加到每个 token 的输出向量上。
```

**图示：**

```text
projected[b,t,:] + bias[:] → out[b,t,:]
```

---

##### Card 0-4D-2：B 和 T 是广播轴

**玩家可见文案：**

```text
bias[O] 没有 B 轴，也没有 T 轴。

这表示同一条 bias 会被复用到：
每个样本，
每个 token position。
```

**图示：**

```text
projected[B][T][O]
bias             [O]
```

---

##### Card 0-4D-3：Bias 必须对齐到 feature/output 轴

**玩家可见文案：**

```text
bias[O] 只能对齐到 O 轴。

如果把 bias 误接到 T 轴，
模型会给不同 token position 加偏移，
而不是给不同 output feature 加偏移。
```

**任务桥接语：**

```text
修复一台 bias 被接错轴的 Linear Add Board。
```

---

### 8.3 画布初始状态

```text
[Projected Tensor]
float32[B=2,T=3,O=4]

[Bias Strip]
float32[O=4]

Projected ───┐
             [Broadcast Add Gate] → Out[?]
Bias ── wrong axis? ┘
```

当前错误状态：bias 被系统默认拖到了 T 轴槽，Add Gate 显示黄色警告。

---

### 8.4 玩家操作

1. 使用 Axis Inspector 查看 projected 的三个轴：B/T/O。
2. 拖动 `bias[O]` 到 Broadcast Rail 的 O 槽。
3. 预览 ghost expansion：bias 沿 B/T 复制。
4. 运行 Cell Trace：

```text
out[1,2,3] = projected[1,2,3] + bias[3]
```

5. 运行 Reference Checker。

---

### 8.5 可视化表现

- `bias[O]` 是一条紫色 feature strip。
- 每个 token 格子的 O 轴上出现一条半透明 bias strip。
- 点击任一输出 token，系统显示同一条 bias 来源。
- 输出张量保持 `[B,T,O]`。

---

### 8.6 验证规则

```text
✓ bias rank == 1
✓ bias axis semantic == O
✓ projected feature axis == O
✓ broadcast plan expands over B and T only
✓ output shape == [B,T,O]
✓ out[b,t,o] == projected[b,t,o] + bias[o]
✓ numeric allclose reference
```

---

### 8.7 失败反馈

#### bias 接到 T 轴

```text
Bias axis mismatch.

bias[O] was aligned to token-position axis T.

Why this fails:
Bias belongs to the output feature axis.
It should shift each O channel,
not each token position.
```

#### bias 长度与 O 不匹配

```text
Bias length mismatch.

Projected output axis O has length 4,
but bias length is 3.

A bias vector must have exactly one value per output channel.
```

#### 输出 shape 改变

```text
Output contract error.

Bias Add should preserve [B,T,O].
It should not create a new axis or remove an axis.
```

---

### 8.8 阶段小结

```text
Stage Complete: Bias Add

你已经修复：
projected[B,T,O] + bias[O] → projected_with_bias[B,T,O]

Bias 会沿 B/T 逻辑复用，
但必须对齐到 O 轴。

下一步，我们会把 [T,C] 的 position embedding 加到 [B,T,C] 上。
```

解锁：

```text
Position Sheet
Position Broadcast Board
```

---

## 9. 0-4E：Position Add / 位置向量广播

### 9.1 阶段目标

让玩家理解：

```text
tok_emb[B,T,C] + pos_emb[T,C] → hidden[B,T,C]
```

这一步回到 0-1 / 0-4 的 LLM 主线，让玩家理解位置编码/位置嵌入如何通过 broadcast add 加入 token embedding。

---

### 9.2 Concept Capsule

```text
Token embedding 只表示 token 本身。
Position embedding 表示 token 在序列中的位置。

pos_emb[T,C] 会加到每个 batch 样本的对应 token position 上。
```

工具解锁：

```text
Position Sheet
Token Position Alignment Probe
```

任务：

```text
把 pos_emb[T,C] 安全加到 tok_emb[B,T,C] 上。
```

---

#### 9.2.1 挑战前知识介绍卡片扩展

##### Card 0-4E-1：token embedding 还不知道位置

**玩家可见文案：**

```text
tok_emb[B,T,C] 给每个 token 一个 C 维向量。

但同一个 token 在不同位置出现时，
模型还需要知道它处在哪个 token position。
```

**图示：**

```text
same token vector
+ different position vector
= position-aware hidden vector
```

---

##### Card 0-4E-2：pos_emb 是 [T,C]

**玩家可见文案：**

```text
pos_emb[T,C] 中，
每一行对应一个 token position，
每一行也是 C 维向量。

它没有 B 轴，
因为所有 batch 样本共享同一套位置向量。
```

**图示：**

```text
T0: [C values]
T1: [C values]
T2: [C values]
```

---

##### Card 0-4E-3：位置向量必须对齐 T 和 C

**玩家可见文案：**

```text
pos_emb[T,C] 不能只对齐到 C。
它的 T 轴必须对齐 token position，
C 轴必须对齐 feature channel。

然后它沿 B 轴广播。
```

**任务桥接语：**

```text
修复一个 position sheet 被错误对齐的 Embedding Add Board。
```

---

### 9.3 画布初始状态

```text
[Token Embedding]
float32[B=2,T=4,C=6]

[Position Embedding]
float32[T=4,C=6]

Token Emb ──┐
            [Broadcast Add Gate] → Hidden Tensor[?]
Pos Emb ────┘
```

当前错误：Position Sheet 的 T 轴未对齐，系统只把 `[T,C]` 当作 `[?,C]` 右对齐，导致 T 轴语义缺失。

---

### 9.4 玩家操作

1. 打开 Semantic Alignment 模式。
2. 将 `pos_emb.T` 对齐到 `tok_emb.T`。
3. 将 `pos_emb.C` 对齐到 `tok_emb.C`。
4. 让 B 轴显示为 broadcast expansion。
5. 运行 Cell Trace：

```text
hidden[b,t,c] = tok_emb[b,t,c] + pos_emb[t,c]
```

6. 使用 Batch Probe 检查同一个 `pos_emb[t,:]` 被用于不同 B。

---

### 9.5 可视化表现

- `tok_emb[B,T,C]` 是多层 token 数据柜。
- `pos_emb[T,C]` 是一张二维位置 sheet。
- 对齐后，position sheet 像透明贴膜一样覆盖到每个 batch layer 上。
- 点击 `b=0,t=2` 与 `b=1,t=2`，两个位置共享同一条 `pos_emb[2,:]`。

---

### 9.6 验证规则

```text
✓ tok_emb contract == [B,T,C]
✓ pos_emb contract == [T,C]
✓ pos_emb.T aligned to tok_emb.T
✓ pos_emb.C aligned to tok_emb.C
✓ B axis is broadcasted
✓ output contract == hidden[B,T,C]
✓ hidden[b,t,c] == tok_emb[b,t,c] + pos_emb[t,c]
✓ numeric allclose reference
```

---

### 9.7 失败反馈

#### pos_emb 的 T 轴被当作 B

```text
Position axis mismatch.

pos_emb.T was aligned to batch axis B.

Why this fails:
Position embedding varies by token position,
not by independent sample.
All batch samples should share the same position table.
```

#### pos_emb 只作为 bias[C] 广播

```text
Position information lost.

You broadcast pos_emb as if it were bias[C].
This ignores token position T.

pos_emb[T,C] must vary across T.
```

#### C 轴不匹配

```text
Channel mismatch.

tok_emb.C has length 6,
pos_emb.C has length 8.

Token and position vectors must have the same channel dimension before add.
```

---

### 9.8 阶段小结

```text
Stage Complete: Position Add

你已经修复：
tok_emb[B,T,C] + pos_emb[T,C] → hidden[B,T,C]

pos_emb 沿 B 广播，
但必须同时对齐 T 和 C。

下一步，我们会把 broadcast add 用到 Attention Mask 上。
```

解锁：

```text
Mask Plate
Attention Score Add Board
```

---

## 10. 0-4F：Mask Add / 注意力遮罩广播

### 10.1 阶段目标

让玩家理解：

```text
scores[B,H,T,T] + mask[1,1,T,T] → masked_scores[B,H,T,T]
```

这一步连接 0-3 的 `scores[B,H,T,T]`，为后续 softmax / causal attention 做准备。

---

### 10.2 Concept Capsule

```text
Attention scores 是每个 query token 对每个 key token 的分数。

在 decoder-only LLM 中，未来 token 不允许被看到。
Causal mask 会在非法位置加上一个很大的负数。

这个 mask 通常沿 B 和 H 轴广播。
```

工具解锁：

```text
Mask Plate
Score Matrix Inspector
Negative Infinity Meter
```

任务：

```text
把 causal mask 正确广播到 scores[B,H,T,T]。
```

---

#### 10.2.1 挑战前知识介绍卡片扩展

##### Card 0-4F-1：scores 有两个 T 轴

**玩家可见文案：**

```text
scores[B,H,T,T] 中的两个 T 不是重复写错。

第一个 T 是 query token 位置。
第二个 T 是 key token 位置。

每个 cell 表示：
某个 query token 对某个 key token 的分数。
```

**图示：**

```text
scores[b,h,query_t,key_t]
```

---

##### Card 0-4F-2：mask 让非法位置变得不可能

**玩家可见文案：**

```text
Causal mask 不删除 cell。
它会给非法位置加上一个很大的负数。

这样经过 softmax 后，
这些位置的概率会接近 0。
```

**图示：**

```text
allowed: score + 0
blocked: score + (-∞)
```

---

##### Card 0-4F-3：mask 沿 B 和 H 广播

**玩家可见文案：**

```text
同一个 causal mask 可以用于每个 batch 样本，
也可以用于每个 attention head。

所以 mask 可以是：
mask[1,1,T,T]

它会广播到：
scores[B,H,T,T]
```

**任务桥接语：**

```text
修复一个被错误广播到 head 轴的 mask board。
```

---

### 10.3 画布初始状态

```text
[Scores]
float32[B=2,H=3,Tq=4,Tk=4]

[Causal Mask]
float32[1,1,Tq=4,Tk=4]

Scores ──┐
         [Broadcast Add Gate] → Masked Scores[?]
Mask ────┘
```

当前错误：mask 的两个 T 轴没有被声明为 query/key 语义，系统无法确定它是 `[Tq,Tk]` 还是 `[Tk,Tq]`。

---

### 10.4 玩家操作

1. 使用 Score Matrix Inspector 确认 scores 的两个 T 轴：

```text
Tq = query token axis
Tk = key token axis
```

2. 将 mask 的 Tq 对齐到 scores 的 Tq。
3. 将 mask 的 Tk 对齐到 scores 的 Tk。
4. 将 mask 的两个 singleton axes `[1,1]` 对齐到 B/H。
5. 运行 Mask Preview：上三角 blocked 区域变橙。
6. 运行 Cell Trace：

```text
masked_scores[b,h,q,k] = scores[b,h,q,k] + mask[0,0,q,k]
```

---

### 10.5 可视化表现

- scores 是一叠热力图，每个 B/H 一个 score board。
- mask 是一张半透明三角板。
- 对齐正确后，mask 复制到每个 B/H score board 上。
- 允许区域加 0，非法区域加 `-1e9` 或 `-∞`。
- 点击一个非法 cell，Negative Infinity Meter 显示该位置会在 softmax 后趋近 0。

---

### 10.6 验证规则

```text
✓ scores contract == [B,H,Tq,Tk]
✓ mask contract == [1,1,Tq,Tk] or [Tq,Tk]
✓ singleton axes align to B/H
✓ mask Tq aligns to scores Tq
✓ mask Tk aligns to scores Tk
✓ output shape == [B,H,Tq,Tk]
✓ allowed cells add 0
✓ blocked cells add negative large value
✓ numeric allclose reference
```

---

### 10.7 失败反馈

#### mask Tq/Tk 反了

```text
Mask orientation error.

The mask query/key axes are swapped.

Why this fails:
Causal mask depends on whether key position is in the future of query position.
Swapping Tq and Tk changes which cells are blocked.
```

#### mask 被沿 T 扩展而不是 B/H

```text
Broadcast alignment error.

Singleton axes of mask should expand over B and H.
Your plan expands over token axes.

This changes the causal pattern and fails reference check.
```

#### B/H 轴被当作 token 轴

```text
Semantic error.

Batch/head axes are parallel containers.
They should not define causal visibility.
Causal mask operates over token axes Tq and Tk.
```

#### 使用乘法 mask 而不是加法 mask

```text
Operation mismatch.

This board expects additive mask:
score + mask_value

Multiplicative masking will not match the reference implementation for this stage.
```

---

### 10.8 阶段小结

```text
Stage Complete: Mask Add

你已经修复：
scores[B,H,T,T] + mask[1,1,T,T] → masked_scores[B,H,T,T]

mask 沿 B/H 广播，
并在 token-to-token score matrix 上控制哪些位置可见。

下一步，你会调试一些 shape 看似合法但语义错误的 broadcast 陷阱。
```

解锁：

```text
Broadcast Trap Debugger
```

---

## 11. 0-4G：Broadcast Trap Debugger / 广播陷阱调试器

### 11.1 阶段目标

让玩家理解：

```text
Broadcast bug 不一定是 shape mismatch。
有些 case shape 可以通过，但语义或数值是错的。
```

这一阶段是本关的核心硬核调试部分。

---

### 11.2 Concept Capsule

```text
Broadcast Debug 不能只看 shape。

你需要同时检查：
1. 轴是否能按规则 broadcast
2. 小张量是否对齐到正确语义轴
3. 某个输出 cell 是否使用了正确输入值
4. reference 数值是否一致
```

工具解锁：

```text
Broadcast Debugger
Semantic Warning Lens
Reference Cell Checker
```

任务：

```text
修复三种 broadcast trap。
```

---

#### 11.2.1 挑战前知识介绍卡片扩展

##### Card 0-4G-1：非法 broadcast 会直接报 shape 错

**玩家可见文案：**

```text
如果两个轴长度不同，且都不是 1，
broadcast 会失败。

例如：
[B,T,C] + [T_wrong,C]
当 T_wrong 不等于 T 时，不能相加。
```

---

##### Card 0-4G-2：可 broadcast 也可能语义错

**玩家可见文案：**

```text
有些错误不会立刻报 shape 错。

例如当 B 和 T 长度刚好相等时，
把 batch 轴当成 token 轴可能仍然能跑，
但模型语义已经错了。
```

---

##### Card 0-4G-3：Cell Trace 是最后的真相

**玩家可见文案：**

```text
当你不确定 broadcast 是否正确时，
追踪一个输出 cell。

例如：
out[b,t,c]
应该使用：
hidden[b,t,c] + bias[c]

如果它使用了 bias[t]，就是错的。
```

**任务桥接语：**

```text
进入 Broadcast Trap Debugger，修复三块损坏的 Add Board。
```

---

### 11.3 Trap Case 设计

#### Trap A：Right-Align Trap

错误输入：

```text
hidden[B,T,C] + token_bias[T]
```

玩家期望把 `[T]` 加到 token position 上，但默认 right-align 会尝试把 `[T]` 对齐到 C。

如果 `T != C`，直接报 shape 错。  
如果 `T == C`，shape 通过但语义错。

正确修复方式：

```text
明确 reshape / unsqueeze 为 token_bias[1,T,1]
hidden[B,T,C] + token_bias[1,T,1]
```

---

#### Trap B：Batch-Token Swap Trap

错误输入：

```text
hidden[B,T,C] + sample_bias[B,1,1]
```

被错误对齐成：

```text
[1,B,1]
```

当 `B == T` 时 shape 通过，但语义错。

正确修复方式：

```text
sample_bias[B,1,1] 对齐到 B 轴
```

---

#### Trap C：Position Sheet as Bias Trap

错误输入：

```text
pos_emb[T,C]
```

被错误压扁或错误对齐成：

```text
bias[C]
```

导致所有 token position 使用同一条位置向量，丢失位置信息。

正确修复方式：

```text
保留 pos_emb 的 T 轴和 C 轴，并沿 B 广播。
```

---

### 11.4 玩家操作

每个 trap 的流程：

```text
1. 读取 Add Gate 错误报告
2. 打开 Broadcast Plan
3. 检查每个输入轴对齐到哪个输出轴
4. 使用 Cell Trace 验证一个输出 cell
5. 插入 Unsqueeze / Axis Tag / Semantic Alignment
6. 运行 Reference Checker
```

---

### 11.5 可视化表现

- 错误的 broadcast plan 显示为黄色，而不是立即红色，表示“shape 可能合法但语义危险”。
- Semantic Warning Lens 会高亮：

```text
Input axis T aligned to output C
Input axis B aligned to output T
```

- Cell Trace 显示错误来源：

```text
Expected: token_bias[t]
Actual: token_bias[c]
```

---

### 11.6 验证规则

```text
✓ broadcast plan shape-valid
✓ semantic alignment valid
✓ no accidental axis swap
✓ no missing position axis
✓ sample cell trace matches intended formula
✓ full numeric output matches reference
```

---

### 11.7 失败反馈

#### 只修 shape，不修语义

```text
Shape passed, semantic failed.

Your broadcast plan is legal under raw shape rules,
but axis semantics do not match the intended operation.

Open Semantic Warning Lens to inspect axis alignment.
```

#### 误用 squeeze / flatten

```text
Information lost.

You removed an axis that carries semantic meaning.
Flattening pos_emb[T,C] into [T*C] cannot be used for position add.
```

#### 使用 repeat 真实复制

```text
Inefficient repair.

Your result is numerically correct,
but you created real copies instead of a broadcast view.

This passes correctness but loses efficiency score.
Use Ghost Expansion / logical broadcast instead.
```

---

### 11.8 阶段小结

```text
Stage Complete: Broadcast Trap Debugger

你已经看到：
Broadcast bug 有两类。
一种是 shape 不合法，会直接失败。
另一种是 shape 合法但语义错误，需要通过 axis semantics 和 cell trace 发现。

下一步进入最终隐藏测试。
```

解锁：

```text
Broadcast Gauntlet
```

---

## 12. 0-4X：Broadcast Gauntlet / 最终隐藏测试

### 12.1 阶段目标

让玩家证明自己能把 Broadcast Add 规则迁移到不同 shape、不同轴顺序、不同 LLM 场景中。

---

### 12.2 Concept Capsule

```text
最终测试不会只考一种 shape。

你会遇到：
- bias add
- position add
- mask add
- singleton axis expansion
- shape 看似通过但语义错误的陷阱

不要靠数字大小猜。
使用 Broadcast Plan、Axis Semantics 和 Cell Trace。
```

---

#### 12.2.1 挑战前知识介绍卡片扩展

##### Card 0-4X-1：Broadcast Contract 要能泛化

**玩家可见文案：**

```text
真正掌握 broadcast，
不是记住 [B,T,C] + [C]。

而是面对任意 shape，
都能判断：
哪些轴对齐，
哪些轴扩展，
哪些轴语义不允许。
```

---

##### Card 0-4X-2：隐藏测试会包含等尺寸陷阱

**玩家可见文案：**

```text
有些隐藏测试会让 B、T、C 中的数值相等。

这会让错误轴对齐在 shape 上看起来合法。
最终只有 semantic checks 和 numeric reference 能抓住错误。
```

---

##### Card 0-4X-3：推荐调试顺序

**玩家可见文案：**

```text
推荐顺序：
1. 看目标输出 contract
2. 看两个输入的 axis semantics
3. 建立 broadcast plan
4. 追踪一个输出 cell
5. 再跑完整 reference
```

---

### 12.3 测试 Case 设计

#### Case A：标准 bias add

```text
projected[B=2,T=3,O=4] + bias[O=4]
→ [2,3,4]
```

验证：

```text
out[b,t,o] = projected[b,t,o] + bias[o]
```

---

#### Case B：position add

```text
tok_emb[B=3,T=5,C=6] + pos_emb[T=5,C=6]
→ [3,5,6]
```

验证：

```text
hidden[b,t,c] = tok_emb[b,t,c] + pos_emb[t,c]
```

---

#### Case C：singleton broadcast

```text
hidden[B=4,T=3,C=8] + scale[1,T=3,1]
→ [4,3,8]
```

验证：

```text
out[b,t,c] = hidden[b,t,c] + scale[0,t,0]
```

---

#### Case D：mask add

```text
scores[B=2,H=4,T=6,T=6] + mask[1,1,6,6]
→ [2,4,6,6]
```

验证：

```text
masked[b,h,q,k] = scores[b,h,q,k] + mask[0,0,q,k]
```

---

#### Case E：equal-dimension trap

```text
hidden[B=4,T=4,C=4] + candidate[4]
```

系统不会告诉 candidate 是 `[B]`、`[T]` 还是 `[C]`。玩家必须通过 Probe 和 Consumer Contract 判断。

---

#### Case F，可选困难版：wrong semantic but shape-valid

```text
scores[B=2,H=2,T=4,T=4] + mask[1,1,Tk,Tq]
```

shape 看起来相同，但 mask 的 query/key 轴反了。

---

### 12.4 玩家操作

每个 case 有统一流程：

```text
Inspect target output contract
→ Inspect input axes
→ Build broadcast plan
→ Preview ghost expansion
→ Trace one sample cell
→ Run visible tests
→ Run hidden reference tests
```

---

### 12.5 限制与评分

```text
Probe Budget: 12
Reference Runs: 5
Hints: optional
```

评分：

```text
Rank C：通过 visible tests
Rank B：通过 hidden tests
Rank A：少于 8 次 probe，通过全部 hidden tests
Rank S：无 hint、无 semantic warning、reference 一次通过
```

效率奖励：

```text
Logical Broadcast Bonus：没有使用真实 repeat/copy
Cell Trace Bonus：至少追踪一个关键 cell 后再运行 reference
```

---

### 12.6 失败反馈

#### 过度依赖 shape

```text
Size-based guess failed.

This case was designed so multiple axes have the same length.
Raw shape cannot determine axis semantics.
Use Probe and Consumer Contract.
```

#### 使用真实 repeat 通过数值但效率差

```text
Correct but inefficient.

Your output matches reference,
but you materialized broadcast copies.

Broadcast should create a logical view when possible.
Rank reduced from A to B.
```

#### mask orientation 错误

```text
Mask orientation failed hidden test.

Shape was correct,
but query/key axes were swapped.
Blocked cells do not match causal pattern.
```

---

### 12.7 最终通关小结

```text
0-4 Complete: Broadcast Add

你已经掌握：
- Add 是逐元素运算。
- Same-shape Add 保持原 shape。
- Broadcast Add 通过缺失轴或 singleton axis 逻辑扩展。
- bias[C/O] 沿 B/T 广播。
- pos_emb[T,C] 沿 B 广播。
- mask[1,1,T,T] 沿 B/H 广播。
- shape 合法不等于语义正确。
- Cell Trace 是调试 broadcast 的关键工具。

Chapter 0 Complete。
下一章将开始搭建真正的 Attention / Transformer Block。
```

---

## 13. 可选支线挑战

### 13.1 Bonus A：Unsqueeze Lab / 插入长度为 1 的轴

#### 目标

让玩家理解：

```text
unsqueeze 可以显式增加一个长度为 1 的轴，
从而控制 broadcast 对齐方向。
```

#### 知识卡文案

```text
有时候 [T] 默认会右对齐到 C，导致错误。

如果你想让它对齐到 T，
可以把它变成 [1,T,1]。

这不是改变数据内容，
而是明确它的轴语义。
```

#### 操作

玩家把：

```text
token_bias[T]
```

通过 Unsqueeze Node 改成：

```text
token_bias[1,T,1]
```

然后加到：

```text
hidden[B,T,C]
```

#### 验证

```text
out[b,t,c] = hidden[b,t,c] + token_bias[t]
```

---

### 13.2 Bonus B：No-Copy Broadcast / 逻辑广播效率挑战

#### 目标

让玩家理解 broadcast view 与真实 repeat 的区别。

#### 知识卡文案

```text
Broadcast 通常不需要复制数据。

真实 repeat 会占用更多内存。
Logical broadcast 只是在计算时复用同一份值。
```

#### 操作

玩家面对两个方案：

```text
方案 A：Repeat bias → real tensor[B,T,C]
方案 B：Broadcast view bias[C] → logical tensor[B,T,C]
```

两者数值都正确，但评分不同。

#### 验证

```text
Correctness: both pass
Memory score: broadcast view wins
```

---

### 13.3 Bonus C：Residual Add Preview / 残差连接预览

#### 目标

提前连接 Transformer Block 的 residual add。

#### 知识卡文案

```text
Residual Add 通常不是 broadcast，
而是同 shape add。

x[B,T,C] + attn_out[B,T,C] → y[B,T,C]

它要求两个路径的 shape 完全一致。
```

#### 操作

玩家连接：

```text
main_path[B,T,C]
residual[B,T,C]
```

通过 Add Gate。

#### 验证

```text
shape exact match
axis exact match
numeric add allclose
```

---

### 13.4 Bonus D：Mask Value Experiment / mask 数值实验

#### 目标

让玩家理解 additive mask 中 `-∞` 或大负数的作用。

#### 知识卡文案

```text
mask 的 blocked cell 通常加一个很大的负数。

这样 softmax 后，
这些位置的概率会接近 0。
```

#### 操作

玩家调整 mask value：

```text
0
-10
-100
-1e9
```

观察 softmax preview。

#### 验证

```text
blocked probability < threshold
allowed probability remains normalized
```

---

## 14. 画布与 UI 设计

### 14.1 主画布布局

默认画布采用左到右数据流：

```text
[Input Tensor A] ─┐
                  [Broadcast Rail] → [Add Gate] → [Output Tensor] → [Reference Checker]
[Input Tensor B] ─┘
```

Stage 0-4D / 0-4E / 0-4F 会在画布上显示对应领域节点：

```text
0-4D: Projected Tensor + Bias Strip
0-4E: Token Embedding + Position Sheet
0-4F: Scores + Mask Plate
```

---

### 14.2 Repair Console 内容

#### 0-4A

```text
Objective:
Connect two scalar cells and run Add Gate.

Tools:
[Cell Trace Probe]

Checks:
□ A connected
□ B connected
□ output = A + B
```

#### 0-4B

```text
Objective:
Align two same-shape tensors and perform elementwise add.

Tools:
[Grid Trace Probe] [Shape Match Gate]

Checks:
□ shapes match
□ axes preserved
□ output allclose reference
```

#### 0-4C

```text
Objective:
Build a valid broadcast plan.

Tools:
[Broadcast Rail] [Axis Alignment Ruler] [Ghost Expansion]

Checks:
□ rank expanded legally
□ singleton axes valid
□ semantic alignment valid
□ sample cell trace correct
```

#### 0-4D

```text
Objective:
Broadcast bias[O] over projected[B,T,O].

Tools:
[Bias Strip] [Bias Broadcast Probe]

Checks:
□ bias aligned to O
□ expanded over B/T
□ output [B,T,O]
```

#### 0-4E

```text
Objective:
Broadcast pos_emb[T,C] over token embedding batch.

Tools:
[Position Sheet] [Token Position Alignment Probe]

Checks:
□ T aligned to T
□ C aligned to C
□ B broadcasted
```

#### 0-4F

```text
Objective:
Add causal mask to scores[B,H,T,T].

Tools:
[Mask Plate] [Score Matrix Inspector] [Negative Infinity Meter]

Checks:
□ singleton axes over B/H
□ query/key axes aligned
□ blocked cells receive negative value
```

---

### 14.3 Inspector 状态示例

#### 选中 Broadcast Rail

```text
Broadcast Rail
Target: float32[B,T,C]
Input : float32[C]
Plan  : [B*, T*, C]

* = logical expansion
Semantic status: valid
Real copy: no
```

#### 选中 Bias Strip

```text
Bias Strip
shape: [O]
dtype: float32
semantic axis: O / output channel
broadcast target: projected[B,T,O]
expands over: B, T
```

#### 选中 Position Sheet

```text
Position Embedding
shape: [T,C]
T axis: token position
C axis: channel
broadcast target: tok_emb[B,T,C]
expands over: B
```

#### 选中 Mask Plate

```text
Causal Mask
shape: [1,1,Tq,Tk]
axes:
- singleton B
- singleton H
- query token Tq
- key token Tk
blocked value: -1e9
```

#### 选中 Add Gate

```text
Add Gate
operation: elementwise add
A: float32[B,T,C]
B: float32[C]
output: float32[B,T,C]
reference: pending
cell trace: available
```

#### 选中 Reference Checker

```text
Reference Checker
visible tests: 5/5 passed
hidden tests: 2/3 pending
max abs error: 0.000000
semantic warnings: none
```

---

## 15. Autograder 设计

### 15.1 结构检查

```text
- Add Gate 是否有两个输入
- Broadcast Rail 是否连接到 Add Gate
- Output Tensor 是否连接到 Reference Checker
- 必要的 Axis Tag / Unsqueeze Node 是否存在
- 是否使用了本阶段允许的工具
```

---

### 15.2 Shape 检查

```text
- 输入 rank 是否可对齐
- 每个轴长度是否相等或其中一边为 1
- 缺失轴是否可以插入 singleton
- output shape 是否等于 broadcasted shape
- same-shape add 是否保持原 shape
```

---

### 15.3 语义检查

```text
- bias[C/O] 是否对齐到 C/O
- pos_emb[T,C] 是否对齐到 T/C
- mask[Tq,Tk] 是否对齐到 scores 的 query/key axes
- singleton axes 是否扩展到正确语义轴
- 是否发生 B/T/H/C/O 语义错接
```

---

### 15.4 数值检查

对于每个输出 cell，reference 规则如下：

```text
same-shape:
out[i,j,k] = A[i,j,k] + B[i,j,k]

bias:
out[b,t,o] = projected[b,t,o] + bias[o]

position:
hidden[b,t,c] = tok_emb[b,t,c] + pos_emb[t,c]

mask:
masked[b,h,q,k] = scores[b,h,q,k] + mask[0,0,q,k]

singleton scale:
out[b,t,c] = hidden[b,t,c] + scale[0,t,0]
```

检查方式：

```text
max_abs_error < 1e-5
allclose(reference, player_output)
```

---

### 15.5 反作弊 / 泛化检查

```text
- 隐藏测试更换 B/T/C/O/H 的具体数值
- 隐藏测试包含 B == T == C 的等尺寸陷阱
- 隐藏测试打乱可视化轴方向，但保留语义标签
- 如果玩家用 hard-coded axis index 而不是 semantic contract，隐藏测试失败
- 如果玩家真实 repeat 而非 broadcast view，正确性通过但效率分降低
```

---

### 15.6 效率检查，可选

```text
- 是否 materialize broadcast copies
- 额外内存占用估算
- 是否使用 logical broadcast view
```

评分影响：

```text
Correct but materialized: Max rank B
Logical broadcast: Eligible for A/S
```

---

## 16. 关卡状态机

```ts
type Level04State =
  | "intro"
  | "stage_0_4A_add_cell"
  | "stage_0_4B_same_shape_add"
  | "stage_0_4C_broadcast_rule_lab"
  | "stage_0_4D_bias_add"
  | "stage_0_4E_position_add"
  | "stage_0_4F_mask_add"
  | "stage_0_4G_trap_debugger"
  | "stage_0_4X_gauntlet"
  | "complete";
```

每个阶段内部状态：

```ts
type StageState =
  | "concept_card"
  | "canvas_entered"
  | "board_broken"
  | "repairing"
  | "preview_broadcast"
  | "cell_trace"
  | "visible_tests"
  | "hidden_tests"
  | "debrief"
  | "unlocked_next";
```

---

## 17. 关卡配置示例

```ts
const level_0_4 = {
  id: "0-4",
  title: "Broadcast Add",
  chapter: "Tensor Bootcamp",
  mode: "Build + Align + Trace",

  prerequisites: ["0-1", "0-2", "0-3"],

  learningOutcomes: [
    "elementwise_add",
    "same_shape_add",
    "broadcast_missing_axes",
    "broadcast_singleton_axes",
    "bias_add",
    "position_add",
    "mask_add",
    "semantic_broadcast_debug"
  ],

  stages: [
    {
      id: "0-4A",
      title: "Add Cell",
      conceptCards: ["0-4A-1", "0-4A-2", "0-4A-3"],
      tools: ["AddGate", "CellTraceProbe"],
      nodes: [
        { id: "scalar_a", type: "Scalar", value: 2.0, dtype: "float32" },
        { id: "scalar_b", type: "Scalar", value: 5.0, dtype: "float32" },
        { id: "add_gate", type: "AddGate" },
        { id: "out", type: "OutputCell" }
      ],
      checks: ["inputs_connected", "dtype_numeric", "scalar_add_exact"]
    },

    {
      id: "0-4D",
      title: "Bias Add",
      conceptCards: ["0-4D-1", "0-4D-2", "0-4D-3"],
      tools: ["BroadcastRail", "BiasStrip", "CellTraceProbe"],
      tensors: [
        { id: "projected", shape: [2, 3, 4], axes: ["B", "T", "O"], dtype: "float32" },
        { id: "bias", shape: [4], axes: ["O"], dtype: "float32" }
      ],
      targetOutput: { shape: [2, 3, 4], axes: ["B", "T", "O"] },
      checks: [
        "bias_aligned_to_O",
        "broadcast_over_B_T",
        "output_shape_matches",
        "cell_trace_bias",
        "numeric_allclose"
      ]
    },

    {
      id: "0-4F",
      title: "Mask Add",
      conceptCards: ["0-4F-1", "0-4F-2", "0-4F-3"],
      tools: ["MaskPlate", "ScoreMatrixInspector", "NegativeInfinityMeter"],
      tensors: [
        { id: "scores", shape: [2, 3, 4, 4], axes: ["B", "H", "Tq", "Tk"], dtype: "float32" },
        { id: "mask", shape: [1, 1, 4, 4], axes: ["1", "1", "Tq", "Tk"], dtype: "float32" }
      ],
      targetOutput: { shape: [2, 3, 4, 4], axes: ["B", "H", "Tq", "Tk"] },
      checks: [
        "mask_singleton_over_B_H",
        "mask_Tq_Tk_aligned",
        "blocked_cells_negative",
        "numeric_allclose"
      ]
    }
  ],

  scoring: {
    visibleTestsRequired: true,
    hiddenTestsRequired: true,
    maxRankWithMaterializedRepeat: "B",
    sRankRequires: ["no_hint", "no_semantic_warning", "one_reference_run"]
  }
};
```

---

## 18. 失败信息库

### 18.1 Incompatible Broadcast Length

```text
Broadcast failed: incompatible axis length.

Target axis {target_axis} has length {target_len}.
Input axis {input_axis} has length {input_len}.

Broadcast requires equal length or one side length = 1.
```

---

### 18.2 Bias Aligned to Wrong Axis

```text
Bias alignment error.

bias[{bias_axis}] was aligned to {wrong_axis}.
Expected alignment: {expected_axis}.

Bias should shift feature/output channels,
not batch or token positions.
```

---

### 18.3 Position Axis Lost

```text
Position information lost.

pos_emb[T,C] was treated as bias[C].
This removes variation across token positions.

Keep T aligned to token-position axis.
```

---

### 18.4 Mask Query/Key Swapped

```text
Mask orientation error.

The mask query axis and key axis are swapped.
Shape may still match,
but future-token blocking pattern is wrong.
```

---

### 18.5 Shape Passed But Semantic Failed

```text
Shape passed, semantic failed.

Raw broadcast rules allow this alignment,
but axis semantics do not match the intended operation.

Inspect Broadcast Plan and run Cell Trace.
```

---

### 18.6 Accidental Equal-Dimension Trap

```text
Equal-dimension trap detected.

Several axes have the same numeric length.
You cannot infer semantics from size.
Use probes and consumer contracts.
```

---

### 18.7 Materialized Repeat Warning

```text
Correct but inefficient.

You repeated the tensor into a full-size copy.
A logical broadcast view would produce the same values with less memory.
```

---

### 18.8 Operation Mismatch

```text
Operation mismatch.

This stage expects additive broadcast:
A + B

The submitted graph uses {operation}.
```

---

## 19. 数值与示例数据建议

### 19.1 0-4A 推荐数据

```text
A = 2.0
B = 5.0
Out = 7.0
```

追加一个负数 case：

```text
A = -1.5
B = 0.25
Out = -1.25
```

---

### 19.2 0-4B 推荐数据

```text
A[T=2,C=3] =
[[1,2,3],
 [4,5,6]]

B[T=2,C=3] =
[[10,20,30],
 [40,50,60]]

Out =
[[11,22,33],
 [44,55,66]]
```

---

### 19.3 0-4D 推荐数据

```text
projected[B=1,T=2,O=3] =
[
  [[1, 2, 3],
   [4, 5, 6]]
]

bias[O=3] = [0.1, 0.2, 0.3]

out =
[
  [[1.1, 2.2, 3.3],
   [4.1, 5.2, 6.3]]
]
```

---

### 19.4 0-4E 推荐数据

```text
tok_emb[B=2,T=3,C=2]
pos_emb[T=3,C=2]

验证重点：
同一个 t 的 pos_emb[t,:]
在 b=0 和 b=1 中被复用。
```

---

### 19.5 0-4F 推荐默认 shape

```text
scores[B=2,H=2,T=4,T=4]
mask[1,1,4,4]

mask rule:
mask[0,0,q,k] = 0      if k <= q
mask[0,0,q,k] = -1e9   if k > q
```

---

### 19.6 0-4X 必须包含的陷阱 shape

```text
hidden[B=4,T=4,C=4] + candidate[4]
```

这会测试玩家是否依赖数字大小猜轴。

另一个建议：

```text
hidden[B=2,T=16,C=4] + token_bias[T=16]
```

这里 T 比 C 大，打破“最大维度就是 C”的错误策略。

---

## 20. 与后续关卡的连接

### 20.1 连接到 Attention Softmax

0-4F 的 Mask Add 直接进入后续：

```text
scores[B,H,T,T]
+ causal_mask[1,1,T,T]
→ masked_scores[B,H,T,T]
→ softmax over Tk
→ attention_weights[B,H,T,T]
```

玩家会在下一章继续学习：

```text
softmax 应该沿 key token axis Tk 做归一化。
```

---

### 20.2 连接到 Transformer Block

0-4B / Bonus C 的 Residual Add 会在 Transformer Block 中再次出现：

```text
x[B,T,C] + attn_out[B,T,C]
x[B,T,C] + mlp_out[B,T,C]
```

玩家已经知道：Residual Add 是 same-shape add，不应 broadcast。

---

### 20.3 连接到 LayerNorm / Bias / MLP

很多模块会用到 broadcast：

```text
LayerNorm gamma[C] / beta[C]
MLP bias[4C]
LM Head bias[V]
```

0-4 建立的 bias broadcast 直觉会在后续重复使用。

---

### 20.4 连接到框架实现

后续技术视图中可以展示类 PyTorch 写法：

```python
x = tok_emb + pos_emb          # [B,T,C] + [T,C]
x = linear(x) + bias           # [B,T,O] + [O]
scores = scores + causal_mask  # [B,H,T,T] + [1,1,T,T]
```

但在 0-4 主流程中不要求玩家写代码，只通过节点和 trace 理解。

---

## 21. MVP 实现建议

### 21.1 MVP 必做范围

建议第一版 0-4 MVP 包含：

```text
0-4A Add Cell
0-4B Same-Shape Add
0-4C Broadcast Rule Lab 的 Case A/B
0-4D Bias Add
0-4E Position Add
0-4X 简化 Gauntlet
```

0-4F Mask Add 可以作为强推荐项，因为它与 0-3 衔接紧密；如果工期紧张，可以作为 0-4.5 或下一章 Attention 前置关。

---

### 21.2 MVP 可暂缓内容

```text
- 真实 no-copy memory profiler
- Negative Infinity Meter 的 softmax 动态预览
- Mask Value Experiment
- 复杂 axis permutation 隐藏测试
- 支线 Bonus 全部内容
```

---

### 21.3 最小可玩闭环

最小闭环必须包含：

```text
Concept Card
→ 画布节点
→ Broadcast Rail 对齐
→ Ghost Expansion 预览
→ Cell Trace
→ Run Tests
→ 失败反馈
→ Debrief
```

如果没有 Cell Trace，本关容易变成“看 shape 选对齐方式”的教程感；Cell Trace 是游戏感和工程感的关键。

---

### 21.4 数据系统简化建议

MVP 不需要完整 tensor 数值计算引擎，可以用：

```text
小尺寸真实数组
reference 函数
allclose 检查
```

例如：

```ts
function broadcastAdd(a, b, plan) {
  // 只支持本关小尺寸 shape
  // 根据 plan 计算每个 output cell 对应输入 index
}
```

---

## 22. 开发任务拆分

### 22.1 玩法逻辑

```text
- Add Gate 节点
- Broadcast Rail 对齐系统
- Axis semantic mapping
- Singleton axis insertion / unsqueeze
- Broadcast plan generation
- Cell trace index mapping
- Reference checker
- Hidden test runner
```

---

### 22.2 UI / 可视化

```text
- Tensor Tile / Grid 可视化
- 3D Tensor Cabinet ghost expansion
- Bias Strip 可视化
- Position Sheet 可视化
- Mask Plate 可视化
- Broadcast Rail 对齐槽
- Semantic Warning Lens
- Cell Trace 动画
```

---

### 22.3 内容系统

```text
- Concept Card 配置
- Stage Debrief 配置
- Failure Lesson 配置
- Stage Objective 配置
- Tool Unlock 配置
```

---

### 22.4 Autograder / 数值系统

```text
- same-shape add reference
- bias add reference
- position add reference
- mask add reference
- broadcast shape validator
- semantic contract validator
- equal-dimension trap validator
- materialized repeat detector，可选
```

---

### 22.5 QA 测试点

```text
- [B,T,C] + [C] 正确
- [B,T,C] + [T,C] 正确
- [B,H,T,T] + [1,1,T,T] 正确
- [B,T,C] + [T] 在 T != C 时报错
- [B,T,C] + [T] 在 T == C 时 semantic warning
- mask Tq/Tk 反转会被数值测试抓住
- 真实 repeat 通过数值但效率降级
```

---

## 23. 最终体验目标

玩家完成 0-4 后，应该形成下面的工程直觉：

```text
Add 是逐元素运算。
同 shape add 是最简单情况。
Broadcast add 需要先建立合法的 broadcast plan。
小张量缺失的轴或长度为 1 的轴可以逻辑扩展。
Bias 加到 feature/output 轴。
Position embedding 加到 token position + channel 轴。
Causal mask 加到 score matrix 的 query/key token 轴。
Broadcast shape 合法不等于语义正确。
遇到 broadcast bug，要用 axis alignment 和 cell trace 调试。
```

这关结束后，Chapter 0 的张量基本功完整闭环。玩家已经具备进入后续 Attention / Transformer Block 章节的能力：

```text
能读 shape
能做 matmul
能修 transpose
能做 broadcast add
能通过 reference tests 验证数值
```

---

# 附录 A：本关核心术语表

| 术语 | 玩家理解版本 | 工程表达 |
|---|---|---|
| Add | 两个数值相加 | elementwise addition |
| Same-shape Add | 两个同 shape 张量逐位置相加 | `A.shape == B.shape` |
| Broadcast | 小张量逻辑扩展到目标 shape | broadcasting rules |
| Singleton Axis | 长度为 1 的轴，可扩展 | axis length = 1 |
| Missing Axis | 小张量没有的目标轴 | implicit singleton axis |
| Broadcast Plan | 每个输入轴如何映射到输出轴 | axis mapping |
| Ghost Expansion | 广播的可视化副本 | logical view, no real copy |
| Bias | 每个 feature/output 通道的偏移 | `bias[C]` / `bias[O]` |
| Position Embedding | 每个 token 位置的向量 | `pos_emb[T,C]` |
| Causal Mask | 阻止看到未来 token 的加性遮罩 | `mask[1,1,T,T]` |
| Semantic Warning | shape 合法但轴语义可疑 | semantic-invalid broadcast |
| Cell Trace | 追踪输出 cell 的输入来源 | index mapping trace |

---

# 附录 B：本关一句话标语

```text
Broadcast 不是魔法扩展，而是带语义轴合同的逻辑复用。
```

备选：

```text
Shape tells whether add can run; semantics tells whether add is right.
```

中文版本：

```text
Shape 决定能不能加，语义决定加得对不对。
```

---

# 附录 C：Concept Card 内容配置规范

## C.1 文本长度规则

```text
Concept Capsule：最多 120 字
挑战前扩展卡：每张最多 120–180 字
Inspector Note：最多 3 条 bullet
Failure Lesson：最多 4 行
Debrief：最多 3 行
```

---

## C.2 出现时机

```text
首次进入 Stage：显示 Concept Capsule
玩家连续失败 2 次：显示相关 Failure Lesson
玩家点击 More Info：显示扩展知识卡
Stage 完成：显示 Debrief
```

---

## C.3 可跳过规则

```text
第一次游玩：Concept Capsule 默认显示，可跳过
复玩：默认折叠为标题条
隐藏测试：不显示概念卡，只显示目标和工具
```

---

# 附录 D：可选支线挑战知识卡片

## D.1 Bonus A：Unsqueeze Lab 知识卡

### Card Bonus-A-1：Unsqueeze 不是改变内容

```text
Unsqueeze 会增加一个长度为 1 的轴。

例如：
[T] → [1,T,1]

数值没有变，
但 broadcast 对齐方式变得明确。
```

---

## D.2 Bonus B：No-Copy Broadcast 知识卡

### Card Bonus-B-1：Broadcast View 比 Repeat 更省

```text
Repeat 会创建真实副本。
Broadcast View 只在计算时复用原值。

两者数值可以相同，
但内存成本不同。
```

---

## D.3 Bonus C：Residual Add 知识卡

### Card Bonus-C-1：Residual Add 通常不需要 broadcast

```text
Residual Add 要求两条路径 shape 完全相同。

x[B,T,C] + y[B,T,C]

如果某条路径丢了轴，
它不是 residual add，而是错误结构。
```

---

## D.4 Bonus D：Mask Value Experiment 知识卡

### Card Bonus-D-1：为什么是很大的负数

```text
Softmax 会把更大的 score 分配更高概率。

当非法位置加上很大的负数后，
它们的概率会接近 0。

这就是 additive mask 的作用。
```

---

# 附录 E：建议埋点与数据分析

```text
stage_start / stage_complete
concept_card_opened / skipped
broadcast_plan_created
semantic_warning_triggered
cell_trace_used
reference_run_count
hidden_test_fail_reason
materialized_repeat_used
axis_alignment_changed
hint_used
rank_awarded
```

重点观察：

```text
1. 玩家是否在 0-4C 理解 missing axis / singleton axis。
2. 玩家是否在 0-4E 把 pos_emb 当成 bias[C]。
3. 玩家是否在 0-4F 频繁交换 Tq/Tk。
4. 玩家是否依赖 shape 数值猜语义。
5. Cell Trace 是否降低后续失败率。
```

---

# 附录 F：0-4 到后续章节的素材复用清单

```text
Broadcast Rail → 后续 LayerNorm / Bias / Mask / Sampler 章节复用
Ghost Expansion → 后续 KV Cache batch/head 可视化复用
Mask Plate → Attention Softmax 章节复用
Cell Trace → 全部数值调试关卡复用
Semantic Warning Lens → 后续高阶 shape bug 复用
No-Copy Broadcast → 性能/显存优化章节复用
```

---

# 附录 G：最终交付检查表

```text
□ 每个阶段都有 Concept Capsule
□ 每个阶段都有 Broken Board 初始状态
□ 每个阶段都有明确玩家操作
□ 每个阶段都有 Cell Trace 或 Broadcast Preview
□ 每个阶段都有 visible tests
□ 0-4X 有 hidden tests
□ 失败反馈区分 shape error / semantic error / numeric error / efficiency warning
□ UI 能区分数据线、broadcast ghost、semantic warning、test result
□ 0-4F 能正确表现 query/key T 轴
□ 章节完成后能自然过渡到 Attention Softmax
```
