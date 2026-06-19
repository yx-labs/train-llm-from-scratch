# LLM Complete — 0-3 Transpose Trap 可执行详细设计文档

> 章节定位：Chapter 0 / Tensor Bootcamp 的第三关。  
> 前置关卡：0-1 Shape Reader，玩家已经掌握 `Tensor / Rank / Shape / Axis Semantics / hidden[B,T,C]`；0-2 MatMul Gate，玩家已经掌握 `MatMul / Linear Projection / 内维度对齐 / weight[C,O] / projected[B,T,O]`。  
> 本关目标：让玩家通过节点画布、轴交换器、张量姿态可视化、Q/K score board 和隐藏数值测试，真正理解 **Transpose 不是装饰性旋转，而是改变轴顺序；Attention score 必须使用 Q @ Kᵀ 才能得到 token-to-token 的 [T,T] 矩阵**。

---

## 0. 设计总述

### 0.1 关卡一句话

玩家需要修复一台损坏的 **Attention Score Board**。系统已经生成了 Query 和 Key：

```text
Q[B,H,T,D]
K[B,H,T,D]
```

但当前板子错误地尝试执行：

```text
Q @ K
```

导致内维度不匹配，或者在某些特殊尺寸下 shape 看似通过但数值错误。玩家要插入并配置 **Transpose Switch**，只交换 K 的最后两个轴：

```text
K[B,H,T,D] → Kᵀ[B,H,D,T]
```

最终让 MatMul Gate 生成真实的 attention scores：

```text
Q[B,H,T,D] @ Kᵀ[B,H,D,T] → scores[B,H,T,T]
```

这一关的核心不是让玩家背 `QKᵀ` 公式，而是让玩家在画布上亲手发现：

```text
为了让每个 query token 和每个 key token 都比较一次，
K 必须把 token axis T 转到输出列方向。
```

---

### 0.2 设计原则

0-3 延续 0-1 / 0-2 的组织结构：

```text
Concept Capsule / 最小概念讲解
→ Broken Board / 故障画布
→ Repair Operation / 玩家修复
→ Probe & Trace / 探针观察
→ Autograder / 可验证测试
→ Debrief / 阶段小结
→ Tool Unlock / 工具解锁
```

本关不做选择题，不要求玩家只靠符号记忆，而是让玩家形成工程直觉：

```text
Transpose 会交换 tensor 的轴顺序。
Matrix transpose 会把 [R,C] 变成 [C,R]。
高阶 tensor 的 transpose 通常只交换指定轴，而不是把整个 tensor 乱序。
MatMul 需要内维度对齐。
Q 和 K 都是 [T,D] 时，不能直接 Q @ K。
Q @ Kᵀ 会得到 [T,T] token-to-token score matrix。
在多头 attention 中，B 和 H 是 batch/head 维度，应该保留；只交换 K 的 T 和 D。
Shape 正确不代表数值正确，最终必须对齐 reference implementation。
```

---

### 0.3 本关学习成果

玩家通关后应该能自然理解：

```text
1. Transpose 是轴交换，不是简单视觉旋转。
2. 2D matrix A[R,C] 转置后是 Aᵀ[C,R]，并满足 A[i,j] = Aᵀ[j,i]。
3. MatMul 的内维度必须匹配：A[M,N] @ B[N,P] → C[M,P]。
4. 如果 K 是 [T,D]，为了与 Q[T,D] 相乘，需要 Kᵀ[D,T]。
5. Q[T,D] @ Kᵀ[D,T] → scores[T,T]。
6. scores 的行是 query token，列是 key token。
7. 在 multi-head attention 中，Q/K 的真实形状是 [B,H,T,D]。
8. 对 K 做 transpose 时，应只交换最后两个轴 T 和 D：K.transpose(-2,-1)。
9. B/H 不是要参与 transpose 的内维度，它们只是并行携带轴。
10. 当 T == D 时，错误的 Q @ K 可能 shape 看起来通过，但数值和轴语义仍然错误。
11. Attention Score Board 必须通过 shape、axis semantics、reference 数值三重验证。
```

---

## 1. 与 0-2 的衔接

0-2 结束时，玩家已经能修复一个 Linear Projection：

```text
hidden[B,T,C] @ weight[C,O] → projected[B,T,O]
```

并理解：

```text
MatMul 的关键是内维度对齐。
C 被消费，O 被生成。
权重方向错误时，需要旋转或 transpose。
```

0-3 的第一个引导要把问题自然抛给玩家：

> 你已经会让一个 token vector 通过权重板被投影。  
> Attention 的下一步不是把特征投影到 O，而是让 token 与 token 相互比较。  
> 这个比较矩阵来自 `Q @ Kᵀ`。  
> 本关要解决的就是那个小小的、但极其关键的 `ᵀ`。

从 0-2 到 0-3 的知识推进关系：

```text
0-1：读懂 hidden[B,T,C]
0-2：用 MatMul 投影 C 轴，生成新特征轴 O
0-3：用 Transpose 修复 Q/K 的方向，生成 token-to-token scores[B,H,T,T]
0-4：理解 Broadcast Add，处理 bias、positional add、mask bias 等广播加法
后续 Attention 章节：scores → scale → mask → softmax → weights @ V
```

---

## 2. 本关整体结构

### 2.1 阶梯挑战总览

| 阶段 | 名称 | 玩家学到什么 | 主要操作 | 验证方式 |
|---|---|---|---|---|
| 0-3A | Matrix Flip | 2D transpose 会交换行列轴 | 用 Transpose Switch 把 matrix[R,C] 变成 matrixᵀ[C,R] | shape 和索引映射 `A[i,j]=Aᵀ[j,i]` 通过 |
| 0-3B | Inner-Dim Repair | transpose 可以修复 MatMul 的方向 | 将 `A[M,N]` 与错误方向的 `B[P,N]` 对齐 | 输出 shape `[M,P]` + reference 数值通过 |
| 0-3C | Higher-Rank Axis Swap | 高阶 tensor 只交换指定轴 | 对 `[B,T,D]` / `[B,H,T,D]` 配置 `swap(-2,-1)` | 携带轴 B/H 保留，T/D 交换 |
| 0-3D | Single-Head QK Score | `Q[T,D] @ Kᵀ[D,T] → [T,T]` | 给单头 Q/K 插入 Kᵀ 并连接 Score Board | score matrix 行列语义正确 |
| 0-3E | Multi-Head Transpose Trap | 真实 attention 是 `[B,H,T,D] @ [B,H,D,T]` | 只转置 K 的最后两个轴，不动 B/H | scores[B,H,T,T] 通过 |
| 0-3F | Trap Debugger | 识别常见错误：转置 Q、转错轴、没转置、全轴反转 | 用 Trace 和 Inspector 修复多个故障板 | shape + axis + numeric 三重通过 |
| 0-3X | Transpose Gauntlet | 泛化到不同 B/H/T/D，尤其 T==D 陷阱 | 连续修复随机 QKᵀ 电路 | hidden tests 全部通过 |

可选支线：

```text
Bonus A：Index Mirror / 手动追踪 A[i,j] 与 Aᵀ[j,i]
Bonus B：Axis Permutation Lab / 练习 permute 与 transpose 的区别
Bonus C：T == D Numeric Trap / shape 通过但数值失败的专项训练
Bonus D：Attention Preview / 预览 scores → softmax → weights @ V
```

---

### 2.2 关卡主流程

```text
关卡开场总引导
→ 0-3A Matrix Flip
→ 0-3B Inner-Dim Repair
→ 0-3C Higher-Rank Axis Swap
→ 0-3D Single-Head QK Score
→ 0-3E Multi-Head Transpose Trap
→ 0-3F Trap Debugger
→ 0-3X Transpose Gauntlet
→ 通关总结
```

---

## 3. 关卡开场总引导

### 3.1 开场界面标题

```text
0-3 Transpose Trap
QKᵀ Repair Bootcamp
```

### 3.2 开场文案

```text
上一关中，你修复了 MatMul Gate。
你已经知道：矩阵乘法要求内维度对齐。

现在，Attention Score Board 出现了新的故障。
它试图用 Q 和 K 直接相乘，但方向不对。

本关目标：
插入并配置 Transpose Switch，
让 Q[B,H,T,D] 与 Kᵀ[B,H,D,T] 正确相乘，
输出 scores[B,H,T,T]。
```

### 3.3 开场图示

```text
错误线路：
Q[B,H,T,D] ─────┐
                 MatMul Gate  →  ERROR / wrong scores
K[B,H,T,D] ─────┘

正确线路：
Q[B,H,T,D] ───────────────┐
                           MatMul Gate → scores[B,H,T,T]
K[B,H,T,D] → Transpose ───┘
             swap T,D
             Kᵀ[B,H,D,T]
```

重点高亮：

```text
只交换 K 的最后两个轴：T ↔ D
B/H 保持不变
输出的两个 T 分别表示 query token 和 key token
```

按钮：

```text
Start Transpose Bootcamp
```

---

### 3.4 开场知识介绍卡片扩展

0-3 开场只建立方向，不抢先讲完整 Attention。建议使用 3 张可跳过短卡，每张 15–25 秒。

#### Card 0-3-Intro-1：上一关留下的问题

**标题**

```text
MatMul 要求方向正确
```

**玩家可见文案**

```text
0-2 中，你已经修复了 MatMul Gate。

它的规则很简单：
左边的最后一轴，必须对齐右边的倒数第二轴。

现在，Attention 里的 Q 和 K 都是 [T,D]。
直接相乘时，方向并不适合生成 token-to-token score。
```

**公式 / 图示**

```text
Q[T,D]
K[T,D]

Q @ K      ❌ 方向错误
Q @ Kᵀ     ✅ 得到 [T,T]
```

**画面表现**

两块矩阵板并排进入 MatMul Gate。Q 的 D 边缘发光，K 的 T 边缘错误地对着 Q 的 D，接口红灯。

**微交互**

玩家点击 `Q @ K`，MatMul Gate 震动并显示 `inner dimension mismatch / orientation mismatch`。

**任务桥接语**

```text
本关第一步：学会什么是 transpose。
```

---

#### Card 0-3-Intro-2：Transpose 是轴交换

**标题**

```text
Transpose：交换轴，不是装饰性旋转
```

**玩家可见文案**

```text
Transpose 会改变 tensor 的轴顺序。

对一个 2D matrix 来说，
行轴和列轴会交换：

A[R,C] → Aᵀ[C,R]
```

**公式 / 图示**

```text
A[i,j] = Aᵀ[j,i]
```

**画面表现**

一个矩阵板沿对角线翻折，行标签变成列标签，列标签变成行标签。某个高亮格子 `A[1,2]` 移动到 `Aᵀ[2,1]`。

**微交互**

玩家拖动一个对角线翻折手柄，观察行列标签互换。

**任务桥接语**

```text
接下来，你会用 Transpose Switch 修复错误方向的矩阵。
```

---

#### Card 0-3-Intro-3：为什么 Attention 要 QKᵀ

**标题**

```text
Attention score 是 token-to-token 比较表
```

**玩家可见文案**

```text
每个 query token 都要和每个 key token 比较一次。

如果一句话有 T 个 token，
那么 score board 应该是 T × T。

这正是：
Q[T,D] @ Kᵀ[D,T] → scores[T,T]
```

**公式 / 图示**

```text
scores[i,j] = dot(Q token i, K token j)
```

**画面表现**

左侧一排 query token，顶部一排 key token，中间生成一个 T×T score grid。每个格子来自一个 query 向量和一个 key 向量的点积。

**微交互**

玩家点击 score[i,j]，系统高亮 Q[i,:] 和 K[j,:] 两条 feature vector。

**任务桥接语**

```text
最终任务：让每个 token 对每个 token 都生成一个分数。
```

---

## 4. 核心视觉语言

### 4.1 Matrix Plate / 矩阵板

用于 2D matrix。表现为带行列标签的刚性数据板。

```text
A[R,C]
R axis：行方向
C axis：列方向
```

视觉规则：

```text
行轴颜色：青蓝
列轴颜色：紫色
被 transpose 后，颜色随语义轴移动，而不是固定在屏幕方向。
```

---

### 4.2 Transpose Switch / 转置开关

本关核心工具。它不是普通旋转按钮，而是一个明确的轴交换装置。

属性面板：

```text
Module: Transpose Switch
Input:  tensor[..., A, B]
Mode:   swap last two axes
Output: tensor[..., B, A]

Axis controls:
- swap axis -2 and -1
- lock carry axes
```

视觉表现：

```text
输入张量进入开关
两个被选中的轴用交叉轨道交换位置
未选中的 carry axes 直线通过
输出端口显示新 shape
```

---

### 4.3 Axis Lock / 携带轴锁

用于高阶 tensor。B/H 这类并行维度应该保持不动。

视觉表现：

```text
B axis: locked pass-through
H axis: locked pass-through
T axis: swap track
D axis: swap track
```

如果玩家试图交换 B/H，模块会黄灯警告：

```text
Warning: You are moving a carry axis.
Attention scores should preserve B and H.
```

---

### 4.4 Q / K Tensor Dock

Q 和 K 不只是普通矩阵，而是带语义标签的 tensor dock。

```text
Q[B,H,T,D]
K[B,H,T,D]
```

轴含义：

```text
B = batch / 样本
H = attention head / 头
T = token position / token 位置
D = head dimension / 每个 head 的特征维度
```

视觉表现：

- B：样本层叠抽屉。
- H：多头切片叠层。
- T：token 序列方向。
- D：每个 token/head 内部的 feature bars。

---

### 4.5 Score Board / 注意力分数板

输出 `scores[B,H,T,T]` 的核心可视化。

每个 head 有一张 T×T 热力图：

```text
rows: query token index
cols: key token index
cell[i,j] = dot(Q[i,:], K[j,:])
```

视觉规则：

```text
第一条 T：query token axis
第二条 T：key token axis
```

尽管两个轴长度都叫 T，但语义不同，UI 中建议显示：

```text
Tq = query token
Tk = key token
```

也可以标为：

```text
scores[B,H,T_query,T_key]
```

用于降低玩家对两个 T 的混淆。

---

### 4.6 颜色建议

```text
B / batch：蓝色
H / head：青绿色
T / token position：金色
D / head_dim：紫色
Transpose action：橙色交叉轨道
MatMul inner match：亮白/青光
错误轴：红色
Carry axes：低亮灰蓝
Score board：冷色热力图，选中格子用白色描边
```

---

## 5. 0-3A：Matrix Flip / 二维矩阵转置

### 5.1 阶段目标

让玩家掌握最小概念：

```text
对 2D matrix 来说，transpose 会交换行轴和列轴。
A[R,C] → Aᵀ[C,R]
并且 A[i,j] = Aᵀ[j,i]。
```

这一步不讲 Q/K，也不讲 attention，只讲轴交换和索引映射。

---

### 5.2 Concept Capsule

```text
Transpose 会交换一个 matrix 的两个轴。

如果 A 的 shape 是 [R,C]，
那么 Aᵀ 的 shape 是 [C,R]。

一个格子的数值不会凭空改变，
它只是从 A[i,j] 移到 Aᵀ[j,i]。
```

---

#### 5.2.1 挑战前知识介绍卡片扩展

##### Card 0-3A-1：Matrix 有行轴和列轴

**标题**

```text
Matrix 是 rank 2 tensor
```

**玩家可见文案**

```text
一个 matrix 有两个轴：
行轴 R 和列轴 C。

shape [3,4] 表示：
3 行，4 列。
```

**公式 / 图示**

```text
A[R,C]
A[1,2] = 第 1 行，第 2 列的数值
```

**画面表现**

显示一块 3×4 的矩阵板。行方向高亮 R，列方向高亮 C。点击一个格子显示 `A[1,2]`。

**微交互**

玩家点击任意格子，Inspector 显示 row index、column index、value。

**任务桥接语**

```text
现在我们要交换这两个轴。
```

---

##### Card 0-3A-2：Transpose 交换行列

**标题**

```text
A[R,C] → Aᵀ[C,R]
```

**玩家可见文案**

```text
Transpose 会把行轴变成列轴，
把列轴变成行轴。

原来的 A[i,j]，
会出现在 Aᵀ[j,i]。
```

**公式 / 图示**

```text
A = shape [3,4]
Aᵀ = shape [4,3]

A[1,2] = Aᵀ[2,1]
```

**画面表现**

矩阵板沿对角线翻折，左侧的行标签移动到底部，顶部的列标签移动到左侧。

**微交互**

拖动 Transpose Switch，观察选中格子的坐标镜像。

**任务桥接语**

```text
你的任务：把输入 matrix 通过 Transpose Switch，输出正确的 Aᵀ。
```

---

##### Card 0-3A-3：Transpose 不是随机重排

**标题**

```text
数值不变，位置变了
```

**玩家可见文案**

```text
Transpose 不会改变每个格子的数值。

它只改变这些数值在轴上的位置。

所以验证 transpose 时，
不仅要检查 shape，
还要检查每个值是否到了正确坐标。
```

**公式 / 图示**

```text
for all i,j:
Aᵀ[j,i] == A[i,j]
```

**画面表现**

几个彩色格子带着数字飞到转置后的镜像位置，数字内容保持不变。

**微交互**

玩家点击 Reference Overlay，系统用虚线连接 `A[i,j]` 和 `Aᵀ[j,i]`。

**任务桥接语**

```text
Autograder 会同时检查 shape 和坐标映射。
```

---

### 5.3 画布初始状态

```text
[Matrix Source A[R,C]] ──broken──> [Transpose Switch] ──> [Matrix Output ?]
```

默认数据：

```text
A.shape = [3,4]
A = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
```

输出目标：

```text
Aᵀ.shape = [4,3]
Aᵀ = [
  [1,5,9],
  [2,6,10],
  [3,7,11],
  [4,8,12]
]
```

---

### 5.4 玩家操作

1. 将 Matrix Source 连接到 Transpose Switch。
2. 设置 Transpose Switch：

```text
swap axis 0 and axis 1
```

3. 将 Transpose Switch 输出连接到 Matrix Output。
4. 点击 `Run Local Tests`。

---

### 5.5 交互反馈

正确配置时：

```text
Input:  float32[R,C] = [3,4]
Switch: swap axis 0 <-> axis 1
Output: float32[C,R] = [4,3]
```

画面上：

- 两条轴出现橙色交叉轨道。
- 格子沿对角线翻折。
- `A[1,2]` 高亮到 `Aᵀ[2,1]`。

---

### 5.6 验证规则

```text
Check 1: output.rank == 2
Check 2: output.shape == [C,R]
Check 3: for all i,j, output[j,i] == input[i,j]
Check 4: axis labels swapped: R ↔ C
```

---

### 5.7 失败反馈

#### 未转置

```text
FAIL: Matrix was passed through without transpose.

Expected output shape: [4,3]
Received output shape: [3,4]

Transpose should swap the row and column axes.
```

#### 只改 shape 标签但没有移动数值

```text
FAIL: Shape label changed, but values were not transposed.

A[1,2] should appear at Aᵀ[2,1].
The output kept the value at the original coordinate.

Transpose is an axis operation, not a relabel-only operation.
```

#### 方向理解反了

```text
FAIL: Axis mapping mismatch.

You mapped A[i,j] to output[i,j].
Expected mapping is A[i,j] → output[j,i].
```

---

### 5.8 阶段小结

```text
Stage Complete: Matrix Flip

你已经验证：
Transpose 会交换 axis order。

A[R,C] 变成 Aᵀ[C,R]，
并且 A[i,j] = Aᵀ[j,i]。

下一步：
我们要用 transpose 修复一个 MatMul 的方向问题。
```

解锁：

```text
Tool Unlocked: Transpose Switch
Concept Unlocked: Axis Swap
```

---

## 6. 0-3B：Inner-Dim Repair / 用转置修复 MatMul 内维度

### 6.1 阶段目标

让玩家把 0-2 的 MatMul 规则和 0-3A 的 Transpose 规则连接起来：

```text
MatMul 要求内维度对齐：
A[M,N] @ B[N,P] → C[M,P]

如果 B 被存成 [P,N]，
就需要 Bᵀ[N,P]。
```

这一步仍然不讲 Q/K。它是从普通矩阵乘法进入 Attention 之前的桥梁。

---

### 6.2 Concept Capsule

```text
Transpose 经常用于修复矩阵方向。

如果 A[M,N] 要乘以 B，
B 的第一轴必须是 N。

当 B 当前是 [P,N] 时，
需要先转置成 [N,P]。
```

---

#### 6.2.1 挑战前知识介绍卡片扩展

##### Card 0-3B-1：MatMul 的内维度规则

**标题**

```text
A[M,N] @ B[N,P]
```

**玩家可见文案**

```text
MatMul 会让左矩阵的最后一轴，
和右矩阵的第一轴对齐。

只有这两个轴长度相同，
MatMul Gate 才能运行。
```

**公式 / 图示**

```text
A[M,N] @ B[N,P] → out[M,P]
        ↑   ↑
      inner dimensions must match
```

**画面表现**

MatMul Gate 中间有两个内维度接口，N 对 N 时接口发白光，N 对 P 时红灯。

**微交互**

玩家拖动 B 板方向，观察接口从红到绿。

**任务桥接语**

```text
现在有一块方向错误的 B 板，需要用 transpose 修复。
```

---

##### Card 0-3B-2：方向错误不一定是数据错误

**标题**

```text
同一批数值，轴顺序不同，计算意义就不同
```

**玩家可见文案**

```text
一块权重板可能存成 [P,N]，
但 MatMul 需要 [N,P]。

数值没有坏，
只是轴方向不适合当前计算。
```

**公式 / 图示**

```text
B_stored[P,N]
B_for_matmul = B_storedᵀ[N,P]
```

**画面表现**

B 板标记为 `stored orientation`，经过 Transpose Switch 后变成 `compute orientation`。

**微交互**

点击“Storage View / Compute View”切换显示。

**任务桥接语**

```text
你要让 MatMul Gate 看到计算方向，而不是存储方向。
```

---

##### Card 0-3B-3：Shape 通过后还要数值通过

**标题**

```text
MatMul 修好后必须对齐 reference
```

**玩家可见文案**

```text
这一关的测试不会只看 shape。

系统会把你的输出和参考实现逐格比较。

正确修复应该同时满足：
shape 正确，数值正确，轴语义正确。
```

**公式 / 图示**

```text
max_abs_error(your_out, reference_out) < 1e-5
```

**画面表现**

输出矩阵与 reference 矩阵重叠，差异单元格用红色标出。

**微交互**

玩家点击一个红色格子，Trace 显示它来自哪两个向量的点积。

**任务桥接语**

```text
Run Tests 会同时检查 shape 和 numeric result。
```

---

### 6.3 画布初始状态

```text
[A Matrix M,N] ───────────┐
                           [MatMul Gate] ──> [Output]
[B Stored P,N] ──broken───┘
```

默认数据：

```text
A[M,N] = [2,3]
B_stored[P,N] = [4,3]
```

目标：

```text
B_compute = B_storedᵀ[N,P] = [3,4]
A[M,N] @ B_compute[N,P] → out[M,P] = [2,4]
```

---

### 6.4 玩家操作

1. 将 A 接入 MatMul Gate 左端。
2. 将 B_stored 接入 Transpose Switch。
3. 配置 Transpose Switch：

```text
swap axis 0 and axis 1
```

4. 将输出 B_compute 接入 MatMul Gate 右端。
5. 运行测试。

---

### 6.5 可视化表现

- B_stored 作为 `[P,N]` 板，P 轴朝外，N 轴朝右。
- 经过 transpose 后，变成 `[N,P]`。
- MatMul Gate 的内维度 N-N 吸附成白色。
- 输出板显示 `[M,P]`。

---

### 6.6 验证规则

```text
Check 1: B passed through Transpose Switch
Check 2: B_compute.shape == [N,P]
Check 3: A.shape[-1] == B_compute.shape[-2]
Check 4: output.shape == [M,P]
Check 5: output allclose reference within 1e-5
```

---

### 6.7 失败反馈

#### 没有插入 transpose

```text
FAIL: Inner dimension mismatch.

A shape: [2,3]
B shape: [4,3]

MatMul expected B first axis = 3,
but received 4.

Try transposing B before MatMul.
```

#### 转置了 A 而不是 B

```text
FAIL: Wrong operand transposed.

You transposed A, but the stored orientation problem is on B.

Expected:
A[M,N] @ Bᵀ[N,P]
```

#### 转置后数值仍错

```text
FAIL: Numeric mismatch.

Shape is correct, but output values differ from reference.

Check whether you transposed before MatMul,
and whether the correct matrix was transposed.
```

---

### 6.8 阶段小结

```text
Stage Complete: Inner-Dim Repair

你已经用 transpose 修复了 MatMul 的方向问题。

这一步会在 Attention 中再次出现：
K 的方向需要从 [T,D] 改成 [D,T]，
才能与 Q[T,D] 相乘。
```

解锁：

```text
Tool Upgrade: Transpose Switch supports named axes
Concept Unlocked: Compute Orientation
```

---

## 7. 0-3C：Higher-Rank Axis Swap / 高阶张量轴交换

### 7.1 阶段目标

让玩家理解：

```text
在 rank > 2 的 tensor 中，transpose 不一定交换所有轴。
常见操作是只交换最后两个轴。

K[B,H,T,D].transpose(-2,-1) → Kᵀ[B,H,D,T]
```

这一步重点建立 carry axes 的概念：B/H 不参与矩阵乘法方向修复，只是并行携带。

---

### 7.2 Concept Capsule

```text
高阶 tensor 可以只交换指定轴。

对 [B,H,T,D] 来说，
B 和 H 是携带轴，应该保持不变。

Transpose Trap 的核心是：
只交换 T 和 D。
```

---

#### 7.2.1 挑战前知识介绍卡片扩展

##### Card 0-3C-1：高阶 tensor 有携带轴

**标题**

```text
不是每个轴都参与 MatMul
```

**玩家可见文案**

```text
在 [B,H,T,D] 中：
B 表示 batch，
H 表示 head，
T 表示 token position，
D 表示每个 head 的 feature dimension。

当我们计算 QKᵀ 时，
真正参与内维度对齐的是 T 和 D。
B 和 H 只是并行携带。
```

**公式 / 图示**

```text
Q[B,H,T,D]
K[B,H,T,D]

carry axes: B,H
matmul axes: T,D
```

**画面表现**

B/H 两条轴被画成锁定轨道，T/D 两条轴进入 Transpose Switch 的交叉轨道。

**微交互**

玩家点击 B/H 轴，Inspector 显示 `carry axis: preserve order`。

**任务桥接语**

```text
接下来，你只能交换指定轴，不能乱动所有轴。
```

---

##### Card 0-3C-2：swap(-2,-1) 的意思

**标题**

```text
transpose(-2, -1) = 交换最后两个轴
```

**玩家可见文案**

```text
很多框架会用负数索引表示从后往前数。

-1 是最后一轴。
-2 是倒数第二轴。

所以 transpose(-2, -1)
就是交换最后两个轴。
```

**公式 / 图示**

```text
K[B,H,T,D]
        ↑ ↑
       -2 -1

K.transpose(-2,-1) = K[B,H,D,T]
```

**画面表现**

shape 标签 `[B,H,T,D]` 下方显示索引：`0,1,2,3` 和 `-4,-3,-2,-1`。T/D 被橙色框选。

**微交互**

玩家点击 `-2` 和 `-1`，UI 高亮 T 与 D 轴。

**任务桥接语**

```text
本阶段要配置一个只交换最后两轴的 Transpose Switch。
```

---

##### Card 0-3C-3：全轴反转是错误操作

**标题**

```text
不要把 [B,H,T,D] 反成 [D,T,H,B]
```

**玩家可见文案**

```text
Transpose Trap 不是把所有轴倒过来。

我们只需要让 K 的 D 轴和 Q 的 D 轴对齐，
再把 K 的 T 轴留给输出列。

B 和 H 必须保持原来的位置。
```

**公式 / 图示**

```text
正确：K[B,H,T,D] → Kᵀ[B,H,D,T]
错误：K[B,H,T,D] → K[D,T,H,B]
```

**画面表现**

错误的全轴反转会让 B/H 轨道断裂，Score Board 的 batch/head 层无法对齐。

**微交互**

玩家尝试勾选 `reverse all axes`，系统弹出警告但允许继续，以便后续测试失败。

**任务桥接语**

```text
本关的 Autograder 会检查 B/H 是否被保留。
```

---

### 7.3 画布初始状态

先给一个 rank 3 的练习，再进入 rank 4。

#### 练习 A：rank 3

```text
Tensor X[B,T,D] → Transpose Switch → Xᵀ[B,D,T]
```

#### 练习 B：rank 4

```text
Tensor K[B,H,T,D] → Transpose Switch → Kᵀ[B,H,D,T]
```

默认参数：

```text
B = 2
H = 3
T = 4
D = 5
```

---

### 7.4 玩家操作

1. 选择 Transpose Switch。
2. 设置轴交换模式：

```text
swap axes: -2, -1
```

或通过可视化界面选择：

```text
swap T ↔ D
lock B
lock H
```

3. 将 K 接入 switch。
4. 观察输出 shape 是否为：

```text
[B,H,D,T]
```

---

### 7.5 可视化表现

- B/H 轴沿直线路径通过 switch，不交叉。
- T/D 轴在 switch 中交叉交换。
- 输出形状标签动态更新：

```text
before: K[B,H,T,D]
after:  Kᵀ[B,H,D,T]
```

- Inspector 显示：

```text
axis map:
output axis 0 ← input axis 0 (B)
output axis 1 ← input axis 1 (H)
output axis 2 ← input axis 3 (D)
output axis 3 ← input axis 2 (T)
```

---

### 7.6 验证规则

```text
Check 1: output.rank == input.rank
Check 2: output.axis[0] == B
Check 3: output.axis[1] == H
Check 4: output.axis[2] == D
Check 5: output.axis[3] == T
Check 6: output.shape == [B,H,D,T]
Check 7: value mapping: out[b,h,d,t] == in[b,h,t,d]
```

---

### 7.7 失败反馈

#### 全轴反转

```text
FAIL: All axes were reversed.

Received: [D,T,H,B]
Expected: [B,H,D,T]

B and H are carry axes.
They should remain in place during K.transpose(-2,-1).
```

#### 交换了 H 和 T

```text
FAIL: Head axis was moved.

H represents independent attention heads.
Transpose Trap should only swap T and D.

Expected axis map:
[B,H,T,D] → [B,H,D,T]
```

#### 只改标签不改值

```text
FAIL: Axis labels changed, but value mapping is wrong.

Expected:
out[b,h,d,t] = in[b,h,t,d]
```

---

### 7.8 阶段小结

```text
Stage Complete: Higher-Rank Axis Swap

你已经掌握：
高阶 tensor 的 transpose 可以只交换指定轴。

Attention 中最常见的操作是：
K.transpose(-2,-1)

它把 K[B,H,T,D] 变成 Kᵀ[B,H,D,T]，
保留 B/H，只交换 T/D。
```

解锁：

```text
Tool Unlocked: Axis Lock
Object Unlocked: Q/K Tensor Dock
```

---

## 8. 0-3D：Single-Head QK Score / 单头 QKᵀ 分数板

### 8.1 阶段目标

让玩家理解 Attention score 的最小结构：

```text
Q[T,D] @ Kᵀ[D,T] → scores[T,T]
```

其中：

```text
scores[i,j] = dot(Q[i,:], K[j,:])
```

这一步先不加入 B/H，避免信息过载。

---

### 8.2 Concept Capsule

```text
一个 token 的 Query 向量，
要和每个 token 的 Key 向量比较一次。

所以 T 个 query token × T 个 key token，
会得到 T×T 的 score board。
```

---

#### 8.2.1 挑战前知识介绍卡片扩展

##### Card 0-3D-1：Q 和 K 都是 token vector 表

**标题**

```text
Q[T,D] 和 K[T,D]
```

**玩家可见文案**

```text
单头 attention 中，
每个 token 都有一条 Query 向量，
也有一条 Key 向量。

如果有 T 个 token，
每条向量长度是 D，
那么 Q 和 K 的 shape 都是 [T,D]。
```

**公式 / 图示**

```text
Q[token, feature] = Q[T,D]
K[token, feature] = K[T,D]
```

**画面表现**

两个 T×D 的矩阵板。行方向显示 token0/token1/token2，列方向显示 D0/D1/D2。

**微交互**

点击 `Q[2,:]` 高亮 token 2 的 Query vector；点击 `K[1,:]` 高亮 token 1 的 Key vector。

**任务桥接语**

```text
要比较 Q token 2 和 K token 1，就要对这两条 D 维向量做点积。
```

---

##### Card 0-3D-2：为什么输出是 T×T

**标题**

```text
每个 query token 对每个 key token 打分
```

**玩家可见文案**

```text
如果有 T 个 query token，
也有 T 个 key token，
那么每个 query 都要对所有 key 生成一个分数。

所以输出不是 [T,D]，
而是 [T,T]。
```

**公式 / 图示**

```text
scores[i,j] = dot(Q[i,:], K[j,:])

rows = query token i
cols = key token j
```

**画面表现**

左侧 query token 列表，顶部 key token 列表，中间出现 T×T 的空 score board。

**微交互**

玩家把 Q token i 和 K token j 拖到某个 cell，系统生成该 cell 的 dot product。

**任务桥接语**

```text
为了生成完整 T×T board，K 需要被转置。
```

---

##### Card 0-3D-3：Kᵀ 让 Key token 成为列

**标题**

```text
Kᵀ[D,T] 把 key token 放到输出列方向
```

**玩家可见文案**

```text
K 原本是 [T,D]：
每一行是一个 key token 的 D 维向量。

转置后，Kᵀ 是 [D,T]。
这样 MatMul 可以让 Q 的 D 与 Kᵀ 的 D 对齐，
并把 K 的 T 留在输出列上。
```

**公式 / 图示**

```text
Q[T,D] @ Kᵀ[D,T] → scores[T,T]
```

**画面表现**

K 板经过 Transpose Switch 后，token 轴转到顶部列方向。MatMul Gate 输出方形 score board。

**微交互**

玩家点击输出第 j 列，系统高亮原始 K 的第 j 个 token vector。

**任务桥接语**

```text
你的任务：插入 Kᵀ，让 Score Board 变成 T×T。
```

---

### 8.3 画布初始状态

```text
[Q Tensor T,D] ───────────┐
                           [MatMul Gate] ──> [Score Board ?]
[K Tensor T,D] ──broken───┘
```

默认参数：

```text
T = 4
D = 3
Q.shape = [4,3]
K.shape = [4,3]
```

目标：

```text
Kᵀ.shape = [3,4]
scores.shape = [4,4]
```

---

### 8.4 玩家操作

1. 将 Q 接到 MatMul Gate 左端。
2. 将 K 接到 Transpose Switch。
3. 配置：

```text
swap axis 0 and axis 1
```

4. 将 Kᵀ 接到 MatMul Gate 右端。
5. 将 MatMul 输出连接到 Score Board。
6. 点击某个 score cell，查看它对应的 Q/K 向量。
7. 运行测试。

---

### 8.5 可视化表现

- Q 行方向 T 保留为 output rows。
- K 经过转置后，原来的 T 变成 output columns。
- MatMul Gate 中间 D-D 对齐。
- Score Board 显示：

```text
scores[T_query,T_key]
```

- 选中 `scores[2,1]` 时：

```text
scores[2,1] = dot(Q[2,:], K[1,:])
```

---

### 8.6 验证规则

```text
Check 1: K passed through Transpose Switch
Check 2: K_t.shape == [D,T]
Check 3: Q.shape[-1] == K_t.shape[-2]
Check 4: scores.shape == [T,T]
Check 5: scores row axis semantic == query token
Check 6: scores column axis semantic == key token
Check 7: scores[i,j] allclose dot(Q[i,:], K[j,:])
```

---

### 8.7 失败反馈

#### 直接 Q @ K

```text
FAIL: K was not transposed.

Q shape: [T,D] = [4,3]
K shape: [T,D] = [4,3]

MatMul needs Q last axis D to meet Kᵀ first axis D.
Try K.transpose(0,1).
```

#### 转置了 Q

```text
FAIL: Q was transposed instead of K.

Expected:
Q[T,D] @ Kᵀ[D,T]

You built:
Qᵀ[D,T] @ K[T,D]

This creates a feature-to-feature matrix, not token-to-token scores.
```

#### 输出不是 T×T

```text
FAIL: Score Board contract mismatch.

Attention scores should compare every query token with every key token.
Expected shape: [T,T]
```

---

### 8.8 阶段小结

```text
Stage Complete: Single-Head QK Score

你已经修复了单头 attention 的核心方向：

Q[T,D] @ Kᵀ[D,T] → scores[T,T]

scores 的每个格子都是：
dot(query token i, key token j)。

下一步：
我们把这个过程扩展到 batch 和 multi-head。
```

解锁：

```text
Object Unlocked: Score Board
Contract Unlocked: scores[T_query,T_key]
```

---

## 9. 0-3E：Multi-Head Transpose Trap / 多头 QKᵀ 方向陷阱

### 9.1 阶段目标

让玩家掌握真实 LLM/Transformer 中的 Q/K 形状：

```text
Q[B,H,T,D]
K[B,H,T,D]
Kᵀ[B,H,D,T]
scores[B,H,T,T]
```

重点是：

```text
B/H 保留，T/D 交换。
```

---

### 9.2 Concept Capsule

```text
真实 attention 会同时处理多个样本和多个 head。

B 和 H 是并行维度。
它们不参与 dot product，应该直接保留。

每个 batch、每个 head 内部，
都执行一次 Q[T,D] @ Kᵀ[D,T]。
```

---

#### 9.2.1 挑战前知识介绍卡片扩展

##### Card 0-3E-1：B/H 是并行层

**标题**

```text
每个 batch、每个 head 都有一张 score board
```

**玩家可见文案**

```text
Q[B,H,T,D] 可以理解为：
B 个样本，
每个样本里有 H 个 attention head，
每个 head 有 T 个 token，
每个 token 有 D 维向量。

MatMul 会在每个 B/H 切片里独立计算。
```

**公式 / 图示**

```text
for b in B:
  for h in H:
    scores[b,h] = Q[b,h] @ K[b,h]ᵀ
```

**画面表现**

B/H 两层像文件夹一样展开，每个文件夹里都有一个 Q[T,D] 与 K[T,D] 的小板。

**微交互**

玩家切换 head 页签，看到每个 head 都有自己的 T×T score board。

**任务桥接语**

```text
你要修复的是所有 B/H 切片共享的轴规则。
```

---

##### Card 0-3E-2：只交换 K 的最后两轴

**标题**

```text
K.transpose(-2,-1)
```

**玩家可见文案**

```text
对 K[B,H,T,D] 来说，
最后两个轴是 T 和 D。

我们只需要交换它们：

K[B,H,T,D] → Kᵀ[B,H,D,T]

B 和 H 不应该移动。
```

**公式 / 图示**

```text
K.transpose(-2,-1)
axis map:
B → B
H → H
T → last axis
D → third axis
```

**画面表现**

B/H 轨道直线通过，T/D 轨道交叉。

**微交互**

玩家拖动一个“swap last two axes”拨杆，输出 shape 从 `[B,H,T,D]` 动态变成 `[B,H,D,T]`。

**任务桥接语**

```text
请配置 Transpose Switch 的 axis map。
```

---

##### Card 0-3E-3：scores[B,H,T,T] 的两个 T

**标题**

```text
输出里有两个 T，但意义不同
```

**玩家可见文案**

```text
scores[B,H,T,T] 的两个 T 分别表示：

第一个 T：query token index
第二个 T：key token index

它们长度相同，
但在 Score Board 上是行和列。
```

**公式 / 图示**

```text
scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:])
```

**画面表现**

Score Board 行标题显示 `T_query`，列标题显示 `T_key`。当鼠标悬停 cell[i,j]，Q token i 和 K token j 同时高亮。

**微交互**

玩家点击一个 cell，Inspector 显示 b/h/i/j 四个坐标及 dot product 来源。

**任务桥接语**

```text
Autograder 会检查两个 T 的行列语义是否正确。
```

---

### 9.3 画布初始状态

```text
[Q Tensor B,H,T,D] ─────────┐
                             [Batched MatMul Gate] ──> [Score Board]
[K Tensor B,H,T,D] ──broken─┘
```

当前故障：

```text
K path has no transpose.
Batched MatMul cannot resolve the contract:
Expected right operand: [B,H,D,T]
Received right operand: [B,H,T,D]
```

默认参数：

```text
B = 2
H = 2
T = 4
D = 3
```

---

### 9.4 玩家操作

1. 将 Q 接到 Batched MatMul Gate 左端。
2. 将 K 接入 Transpose Switch。
3. 锁定 B/H 轴。
4. 设置交换轴：

```text
swap -2 and -1
// or swap T and D
```

5. 将 Kᵀ 接到 Batched MatMul Gate 右端。
6. 将输出接到 Score Board。
7. 运行 Visible Tests。

---

### 9.5 可视化表现

- Q 与 K 都显示为 4D 折叠张量。
- B/H 作为分页/层叠切片。
- 当前选中的 `(b,h)` 切片展开为 Q[T,D] 和 K[T,D]。
- K 通过 Transpose Switch 后，变成 Kᵀ[D,T]。
- Score Board 生成 `(b,h)` 对应的 T×T 热力图。
- UI 显示总输出：

```text
scores[B,H,T_query,T_key]
```

---

### 9.6 验证规则

```text
Check 1: Q contract == [B,H,T,D]
Check 2: K input contract == [B,H,T,D]
Check 3: K_transposed contract == [B,H,D,T]
Check 4: MatMul inner axis == D
Check 5: output contract == [B,H,T_query,T_key]
Check 6: B axis preserved
Check 7: H axis preserved
Check 8: for all b,h,i,j:
         scores[b,h,i,j] == dot(Q[b,h,i,:], K[b,h,j,:])
Check 9: output allclose reference within 1e-5
```

---

### 9.7 失败反馈

#### B/H 被交换

```text
FAIL: Carry axes were moved.

Expected K transpose:
[B,H,T,D] → [B,H,D,T]

Received:
[H,B,D,T]

B and H are parallel carry axes.
They must remain aligned with Q.
```

#### T/D 没交换

```text
FAIL: K still has token axis before feature axis.

Received right operand: [B,H,T,D]
Expected right operand: [B,H,D,T]

Batched MatMul needs D to match Q's last axis.
```

#### Q 被转置

```text
FAIL: Query tensor was transposed.

Q's token axis should become the output row axis.
Transposing Q changes the output into feature-to-token or feature-to-feature space.

Expected:
Q[B,H,T,D] @ Kᵀ[B,H,D,T]
```

#### 输出两个 T 语义反了

```text
FAIL: Score Board axes are swapped.

Expected rows = query tokens, columns = key tokens.
Your board maps rows to key tokens.

Check the operand order: Q should be on the left, Kᵀ on the right.
```

---

### 9.8 阶段小结

```text
Stage Complete: Multi-Head Transpose Trap

你已经修复真实 attention score 的核心线路：

Q[B,H,T,D] @ Kᵀ[B,H,D,T] → scores[B,H,T,T]

B/H 被保留，
T/D 在 K 上交换，
输出每个 batch、每个 head 的 token-to-token score board。
```

解锁：

```text
Tool Unlocked: Score Inspector
Contract Unlocked: scores[B,H,T_query,T_key]
```

---

## 10. 0-3F：Trap Debugger / 转置陷阱调试器

### 10.1 阶段目标

让玩家识别并修复 Transpose Trap 的常见错误类型，而不是只完成一条固定线路。

常见错误：

```text
1. 没有转置 K。
2. 转置了 Q。
3. 转置了 K 但交换了错误轴。
4. 把所有轴反转。
5. 交换了 B/H carry axes。
6. Q/K 操作数顺序反了。
7. T == D 时 shape 看似正确，但数值错。
```

---

### 10.2 Concept Capsule

```text
Transpose bug 最危险的地方在于：
有些错误会立刻 shape mismatch，
有些错误却能通过 shape 检查。

所以调试顺序应该是：
先看 axis contract，
再看 score board 语义，
最后看 reference numeric result。
```

---

#### 10.2.1 挑战前知识介绍卡片扩展

##### Card 0-3F-1：不是所有 transpose bug 都会报 shape 错

**标题**

```text
Shape error 只是最简单的错误
```

**玩家可见文案**

```text
如果 T 和 D 不相等，
错误的 Q @ K 往往会直接 shape mismatch。

但如果 T == D，
错误线路可能产生同样大小的输出。

这时必须检查 axis semantics 和 numeric reference。
```

**公式 / 图示**

```text
T = D = 4
Q[T,D] @ K[T,D] 可能得到 [T,D]
shape 看起来像 [T,T]
但它不是 Q @ Kᵀ
```

**画面表现**

两个 Score Board 外形都为 4×4，一个通过 shape 灯，另一个 reference 灯红。

**微交互**

玩家切换 `Shape View` 和 `Value View`，观察 shape 一样但 cell 值不同。

**任务桥接语**

```text
本阶段会给你几个故障板，你要用 Trace 找出真正的 bug。
```

---

##### Card 0-3F-2：调试时先看轴合同

**标题**

```text
先检查 axis contract
```

**玩家可见文案**

```text
QKᵀ 的正确合同是：

Q:  [B,H,T,D]
Kᵀ: [B,H,D,T]
out:[B,H,T,T]

如果任何一个轴位置不对，
后面的计算就不可信。
```

**公式 / 图示**

```text
axis map for Kᵀ:
B ← B
H ← H
D ← D
T ← T

out row T ← Q.T
out col T ← K.T
```

**画面表现**

Axis Contract Panel 显示每条轴的来源和去向，错误轴用红色箭头。

**微交互**

玩家点击红色轴箭头，画布自动高亮导致它错位的节点。

**任务桥接语**

```text
使用 Axis Trace 修复第一个故障板。
```

---

##### Card 0-3F-3：再检查单个 cell 的来源

**标题**

```text
一个 score cell 应该来自两条 token 向量
```

**玩家可见文案**

```text
正确的 scores[b,h,i,j] 来自：

Q[b,h,i,:]
和
K[b,h,j,:]

如果某个 cell 来自 feature index 或错误 head，
就说明 transpose 或 operand order 有问题。
```

**公式 / 图示**

```text
scores[b,h,i,j]
= Σ_d Q[b,h,i,d] * K[b,h,j,d]
```

**画面表现**

选中一个 score cell，Q 的第 i 行和 K 的第 j 行发光，D 维特征条一一相乘。

**微交互**

玩家点击 `Trace Cell`，查看该 cell 的完整乘加来源。

**任务桥接语**

```text
如果 shape 看起来对，就用 Cell Trace 找数值错误。
```

---

### 10.3 画布初始状态

本阶段不是固定一个板，而是连续三个故障板。

```text
Fault Board A：No K Transpose
Fault Board B：Q Transposed
Fault Board C：All Axes Reversed
Fault Board D：T == D Numeric Trap，可选困难
```

每个故障板都包含：

```text
Q Tensor
K Tensor
Transpose Switch / 或缺失
Batched MatMul Gate
Score Board
Reference Checker
```

---

### 10.4 玩家操作

每个故障板流程：

```text
1. Run Trace
2. 查看 Axis Contract Panel
3. 查看 Score Board cell 来源
4. 修改 Transpose Switch 或线路
5. Run Tests
```

允许操作：

```text
- 插入 / 删除 Transpose Switch
- 修改 swap axes
- 锁定 carry axes
- 调整 operand order
- 将 Q/K 接回正确端口
```

---

### 10.5 可视化表现

- 错误线路不会直接“禁止”，而是允许玩家运行，让他看到失败。
- Axis Contract Panel 给出结构错误。
- Score Cell Trace 给出数值来源错误。
- Reference Checker 显示最大误差。

示例 Inspector：

```text
Selected: scores[0,1,2,3]
Expected source:
Q[0,1,2,:] · K[0,1,3,:]

Actual source:
Q[0,1,:,2] · K[0,1,:,3]

Diagnosis:
Feature axis D and token axis T were swapped incorrectly.
```

---

### 10.6 验证规则

每个故障板必须通过：

```text
Check 1: Q on left operand
Check 2: K passes through correct Transpose Switch
Check 3: axis map equals [B,H,T,D] → [B,H,D,T]
Check 4: output scores contract == [B,H,T_query,T_key]
Check 5: selected cell traces match Q token i and K token j
Check 6: output allclose reference
```

---

### 10.7 失败反馈

#### 操作数顺序反了

```text
FAIL: Operand order reversed.

You computed Kᵀ @ Q.

This changes which token axis becomes rows and columns.
Expected:
rows = query token from Q
columns = key token from K
```

#### T == D 下 shape 通过但数值失败

```text
FAIL: Shape passed, but numeric reference failed.

This is the Transpose Trap.
Because T == D, the wrong operation produced the same visible shape.

But scores[i,j] should equal dot(Q[i,:], K[j,:]).
Run Cell Trace to inspect one failed cell.
```

#### 轴语义错误

```text
FAIL: Output axis semantics invalid.

The output has two axes of length T,
but they do not map to query-token and key-token correctly.

Expected:
out[..., i, j] = dot(Q[..., i, :], K[..., j, :])
```

---

### 10.8 阶段小结

```text
Stage Complete: Trap Debugger

你已经学会调试 transpose bug：

先检查 shape，
再检查 axis contract，
最后检查 reference values。

尤其要注意：
shape 通过不代表 QKᵀ 正确。
```

解锁：

```text
Tool Unlocked: Cell Trace
Tool Unlocked: Reference Checker Strict Mode
```

---

## 11. 0-3X：Transpose Gauntlet / 最终隐藏测试

### 11.1 阶段目标

让玩家把 Transpose Trap 的知识迁移到不同尺寸、不同故障类型中。

最终验证玩家不是只记住当前案例，而是真正掌握：

```text
Q[B,H,T,D] @ K.transpose(-2,-1)[B,H,D,T] → scores[B,H,T,T]
```

---

### 11.2 Concept Capsule

```text
最终测试会改变 B/H/T/D。

有些 case 会让 T != D，错误线路直接 shape mismatch。
有些 case 会让 T == D，错误线路可能 shape 通过但数值失败。

不要靠尺寸猜。
用 axis contract 和 reference trace 调试。
```

---

#### 11.2.1 挑战前知识介绍卡片扩展

##### Card 0-3X-1：Transpose Contract 要能泛化

**标题**

```text
不同尺寸，同一个规则
```

**玩家可见文案**

```text
B、H、T、D 的具体数值可以变化。

但 QKᵀ 的合同不变：

Q[B,H,T,D]
Kᵀ[B,H,D,T]
scores[B,H,T,T]
```

**公式 / 图示**

```text
for any B,H,T,D:
K_t = K.transpose(-2,-1)
scores = Q @ K_t
```

**画面表现**

多个不同尺寸的 Q/K 工件进入同一个 Transpose Switch 模板，输出统一的 score board 合同。

**微交互**

玩家拖动 B/H/T/D 尺寸滑杆，合同标签保持不变。

**任务桥接语**

```text
你将连续修复多个随机故障板。
```

---

##### Card 0-3X-2：隐藏测试会包含 T == D

**标题**

```text
小心 T == D 陷阱
```

**玩家可见文案**

```text
当 T 和 D 相等时，
错误操作可能产生同样大小的矩阵。

这会骗过只看 shape 的测试。

最终测试会检查数值来源：
each score cell must come from Q token i and K token j。
```

**公式 / 图示**

```text
wrong: Q[T,D] @ K[T,D]
right: Q[T,D] @ Kᵀ[D,T]

when T == D, shape may look identical
```

**画面表现**

两个 4×4 的 score board：一个 shape 绿灯，一个 numeric 红灯。玩家点击红色单元格后看到错误来源。

**微交互**

点击 `Show Difference`，两个 board 的差值热力图展开。

**任务桥接语**

```text
最终测试必须通过 numeric reference。
```

---

##### Card 0-3X-3：推荐调试顺序

**标题**

```text
Transpose Debug Checklist
```

**玩家可见文案**

```text
遇到失败时，按这个顺序检查：

1. Q 是否在左侧？
2. K 是否经过 transpose？
3. 是否只交换了最后两个轴？
4. B/H 是否保留？
5. Score cell 是否来自正确 Q/K token？
```

**公式 / 图示**

```text
Q left
K → transpose(-2,-1)
out = [B,H,T_query,T_key]
```

**画面表现**

Repair Console 中显示一张可勾选 Debug Checklist。

**微交互**

每修好一项，Checklist 对应条目变绿。

**任务桥接语**

```text
开始最终测试。
```

---

### 11.3 测试 Case 设计

#### Case A：标准非等尺寸

```text
B=2, H=2, T=4, D=3
故障：K 未转置
预期：shape mismatch，玩家插入 transpose
```

#### Case B：carry axes 被移动

```text
B=3, H=2, T=5, D=4
故障：axis map [B,H,T,D] → [H,B,D,T]
预期：玩家锁定 B/H，只交换 T/D
```

#### Case C：T == D 数值陷阱

```text
B=1, H=3, T=4, D=4
故障：Q @ K 直接相乘，shape 看起来可能通过
预期：numeric reference 失败，玩家用 Cell Trace 发现 K 没转置
```

#### Case D：操作数顺序反了

```text
B=2, H=1, T=6, D=3
故障：Kᵀ @ Q
预期：shape 或 axis semantics 失败；玩家把 Q 接回左端，Kᵀ 接右端
```

#### Case E，可选困难版：重复 head / 随机轴标签隐藏

```text
B=2, H=4, T=3, D=5
故障：H 和 T 标签被交换，视觉 shape 不明显
预期：玩家使用 Axis Probe 识别 H 与 T
```

---

### 11.4 玩家操作

每个 case 的流程：

```text
1. Inspect Q/K contracts
2. Repair K transpose path
3. Lock carry axes
4. Connect Q left, Kᵀ right
5. Run visible tests
6. Run hidden numeric tests
```

---

### 11.5 限制与评分

不强制时间压力，但加入工程评分：

```text
Probe used: 0 / 8
Cell Trace used: 0 / 5
Failed runs: 0
Reference overlay used: 0 / 3
```

评分建议：

```text
Rank C：通过 visible tests
Rank B：通过 hidden tests
Rank A：少于 3 次失败运行
Rank S：无 hint、全部 case 一次通过 numeric reference
```

---

### 11.6 失败反馈

#### 过度依赖 shape

```text
Hidden test failed: shape-only repair is insufficient.

Your output shape matches [B,H,T,T],
but score values do not match Q @ Kᵀ.

Use Cell Trace to verify:
scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:])
```

#### 反复移动 carry axes

```text
Hidden test failed: carry axes changed across cases.

B and H are not part of K transpose.
They must pass through unchanged for every case.
```

#### 将第二个 T 当成 D

```text
Hidden test failed: output column axis must be key token T.

The D axis is consumed inside dot product.
It should not appear in scores[B,H,T,T].
```

---

### 11.7 最终通关小结

```text
0-3 Complete: Transpose Trap

你已经掌握：

Transpose 是轴交换。
K.transpose(-2,-1) 会把 K[B,H,T,D] 变成 Kᵀ[B,H,D,T]。
QKᵀ 的输出是 token-to-token score board：scores[B,H,T,T]。
B/H 是 carry axes。
T/D 是需要在 K 上交换的 axes。
Shape 正确不等于数值正确。

下一关：
0-4 Broadcast Add
你将学习 bias、positional embedding 和 mask bias 如何通过 broadcast 加到张量上。
```

---

## 12. 可选支线挑战

### 12.1 Bonus A：Index Mirror / 索引镜像训练

#### 目标

让玩家手动验证 transpose 的坐标映射：

```text
A[i,j] = Aᵀ[j,i]
```

#### 操作

系统给一个矩阵 A 和几个目标格子：

```text
Find where A[2,3] appears after transpose.
Find the source of Aᵀ[1,4].
```

玩家在矩阵板上点击对应坐标。

#### 验证

```text
✓ A[2,3] maps to Aᵀ[3,2]
✓ Aᵀ[1,4] comes from A[4,1]
```

#### 小结

```text
你已经能从索引层面理解 transpose，
这有助于排查 shape 正确但数值错误的问题。
```

---

### 12.2 Bonus B：Axis Permutation Lab / 轴排列实验室

#### 目标

区分 `transpose` 和更一般的 `permute`。

#### 知识卡文案

```text
Transpose 通常表示交换两个轴。
Permute 可以重新排列多个轴。

本关主线只需要 K.transpose(-2,-1)，
但理解 permute 能帮助你识别全轴反转错误。
```

#### 操作

给定：

```text
X[B,H,T,D]
```

任务：

```text
1. swap T/D → [B,H,D,T]
2. move H after T → [B,T,H,D]
3. reverse all axes → [D,T,H,B]
```

玩家通过轴轨道排列器完成。

#### 验证

检查 axis map 与 value map。

---

### 12.3 Bonus C：T == D Numeric Trap / 等尺寸数值陷阱

#### 目标

专门训练玩家不要只看 shape。

#### 操作

系统提供：

```text
Q[4,4]
K[4,4]
```

两条线路都能生成 4×4：

```text
wrong: Q @ K
right: Q @ Kᵀ
```

玩家需要用 Cell Trace 证明哪条是正确 attention score。

#### 验证

```text
scores[i,j] == dot(Q[i,:], K[j,:])
```

#### 小结

```text
当轴长度相等时，shape 检查无法发现所有错误。
必须检查 axis semantics 和数值来源。
```

---

### 12.4 Bonus D：Attention Preview / 后续 Attention 预览

#### 目标

让玩家看到 QKᵀ 只是 Attention 的第一步。

#### 画布

```text
QKᵀ scores[B,H,T,T]
      ↓ scale by sqrt(D)
scaled_scores[B,H,T,T]
      ↓ causal mask
masked_scores[B,H,T,T]
      ↓ softmax
attention_weights[B,H,T,T]
      ↓ @ V[B,H,T,D]
context[B,H,T,D]
```

#### 文案

```text
本关只修复 QKᵀ。
后续关卡会继续处理 scale、mask、softmax 和 weights @ V。
```

#### 验证

玩家只需要点击每个后续节点，查看它消费的 shape，不要求修复。

---

## 13. 画布与 UI 设计

### 13.1 主画布布局

推荐默认布局：

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Left: Sources                Center: Repair Path        Right: Tests │
│                                                                     │
│ [Q Tensor Dock] ───────────────┐                                    │
│                                │                                    │
│                                [Batched MatMul Gate] ── [Score Board]│
│                                │                         │          │
│ [K Tensor Dock] → [Transpose Switch] ───────────────────┘          │
│                                                                     │
│ [Reference Checker]      [Axis Contract Panel]       [Trace Console]│
└─────────────────────────────────────────────────────────────────────┘
```

底部 Repair Console 根据阶段变化。

---

### 13.2 Repair Console 内容

#### 0-3A

```text
Objective:
Transpose matrix A[R,C] into Aᵀ[C,R].

Tools:
[Transpose Switch]
[Index Probe]

Checklist:
□ input connected
□ axis 0/1 swapped
□ output shape [C,R]
□ value map A[i,j] = out[j,i]
```

#### 0-3C

```text
Objective:
Swap only the last two axes of K[B,H,T,D].

Tools:
[Transpose Switch]
[Axis Lock]
[Axis Inspector]

Axis Map:
B → B  locked
H → H  locked
T → output axis 3
D → output axis 2
```

#### 0-3E

```text
Objective:
Build scores[B,H,T,T] from Q and K.

Required path:
Q[B,H,T,D] → MatMul.left
K[B,H,T,D] → Transpose(-2,-1) → MatMul.right

Checklist:
□ Q on left
□ K transposed
□ B/H preserved
□ output [B,H,T_query,T_key]
□ numeric reference passed
```

#### 0-3F / 0-3X

```text
Debug Checklist:
□ Q is left operand
□ K is right operand after transpose
□ transpose swaps only -2 and -1
□ carry axes preserved
□ score cell source is Q token i × K token j
□ reference allclose passed
```

---

### 13.3 Inspector 状态示例

#### 选中 Transpose Switch

```text
Module: Transpose Switch
Input:  K[B,H,T,D]
Mode:   swap selected axes
Swap:   axis -2 (T) ↔ axis -1 (D)
Output: K_t[B,H,D,T]

Carry axes:
B locked
H locked

Value map:
out[b,h,d,t] = in[b,h,t,d]
```

#### 选中 Q Tensor

```text
Tensor: Q
shape: [B,H,T,D]
dtype: float32
axis semantics:
B = batch
H = attention head
T = query token position
D = head feature dimension

MatMul role:
left operand
output row source: T
inner axis: D
```

#### 选中 K Tensor

```text
Tensor: K
shape: [B,H,T,D]
dtype: float32
axis semantics:
B = batch
H = attention head
T = key token position
D = head feature dimension

Required transform:
transpose last two axes → [B,H,D,T]
```

#### 选中 Score Board

```text
Tensor: scores
shape: [B,H,T_query,T_key]
dtype: float32

cell meaning:
scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:])

selected cell:
b=0, h=1, query token=2, key token=3
```

#### 选中 Reference Checker

```text
Reference:
K_t = K.transpose(-2,-1)
scores_ref = Q @ K_t

current max_abs_error: 0.000002
status: PASS
```

---

## 14. Autograder 设计

### 14.1 结构检查

```text
1. Q node exists and is connected to MatMul.left。
2. K node exists and passes through Transpose Switch。
3. Transpose Switch output is connected to MatMul.right。
4. MatMul output is connected to Score Board。
5. Reference Checker receives both current output and reference output。
```

---

### 14.2 Shape 检查

```text
Q.shape == [B,H,T,D]
K.shape == [B,H,T,D]
K_t.shape == [B,H,D,T]
MatMul output.shape == [B,H,T,T]
```

对单头阶段：

```text
Q.shape == [T,D]
K.shape == [T,D]
K_t.shape == [D,T]
output.shape == [T,T]
```

---

### 14.3 语义检查

```text
K_t.axisMap == {
  out[0] <- in[0] B,
  out[1] <- in[1] H,
  out[2] <- in[3] D,
  out[3] <- in[2] T
}

scores.axis[0] == B
scores.axis[1] == H
scores.axis[2] == T_query_from_Q
scores.axis[3] == T_key_from_K
```

---

### 14.4 数值检查

```text
reference = matmul(Q, transpose(K, -2, -1))
assert allclose(user_scores, reference, atol=1e-5, rtol=1e-5)
```

单个 cell 检查：

```text
for sampled b,h,i,j:
  expected = sum_d Q[b,h,i,d] * K[b,h,j,d]
  actual = scores[b,h,i,j]
  assert abs(expected - actual) < tolerance
```

---

### 14.5 反作弊 / 泛化检查

为了防止玩家写死 shape 或靠视觉猜：

```text
hidden cases include:
- T != D
- T == D
- B = 1
- H = 1
- H > 1
- T smaller than D
- T larger than D
```

必须验证 axis semantics 和数值，不只验证 shape。

---

## 15. 关卡状态机

```ts
type Level03State =
  | "intro"
  | "stage_0_3A_matrix_flip"
  | "stage_0_3B_inner_dim_repair"
  | "stage_0_3C_higher_rank_axis_swap"
  | "stage_0_3D_single_head_qk"
  | "stage_0_3E_multi_head_trap"
  | "stage_0_3F_trap_debugger"
  | "stage_0_3X_gauntlet"
  | "completed";
```

每个阶段的局部状态：

```ts
type StageState =
  | "concept_card"
  | "board_loaded"
  | "repairing"
  | "trace_running"
  | "visible_tests"
  | "hidden_tests"
  | "debrief"
  | "unlocked_next";
```

---

## 16. 关卡配置示例

```ts
const level_0_3 = {
  id: "0-3",
  title: "Transpose Trap",
  chapter: "Tensor Bootcamp",
  mode: "Build + Trace + Debug",

  prerequisites: ["0-1", "0-2"],

  introCards: [
    "0-3-Intro-1",
    "0-3-Intro-2",
    "0-3-Intro-3"
  ],

  stages: [
    {
      id: "0-3A",
      title: "Matrix Flip",
      conceptCards: ["0-3A-1", "0-3A-2", "0-3A-3"],
      targetContract: "A[R,C] -> A_t[C,R]",
      nodes: ["MatrixSource", "TransposeSwitch", "MatrixOutput", "ReferenceChecker"],
      tools: ["TransposeSwitch", "IndexProbe"],
      checks: ["rank", "shape", "axis_map", "value_map"]
    },
    {
      id: "0-3B",
      title: "Inner-Dim Repair",
      conceptCards: ["0-3B-1", "0-3B-2", "0-3B-3"],
      targetContract: "A[M,N] @ B_stored[P,N].T[N,P] -> out[M,P]",
      nodes: ["MatrixA", "MatrixBStored", "TransposeSwitch", "MatMulGate", "Output", "ReferenceChecker"],
      tools: ["TransposeSwitch", "MatMulTrace"],
      checks: ["inner_dim", "shape", "numeric"]
    },
    {
      id: "0-3C",
      title: "Higher-Rank Axis Swap",
      conceptCards: ["0-3C-1", "0-3C-2", "0-3C-3"],
      targetContract: "K[B,H,T,D] -> K_t[B,H,D,T]",
      nodes: ["KTensor", "TransposeSwitch", "AxisContractPanel"],
      tools: ["AxisLock", "AxisInspector"],
      checks: ["carry_axes", "swap_last_two", "value_map"]
    },
    {
      id: "0-3D",
      title: "Single-Head QK Score",
      conceptCards: ["0-3D-1", "0-3D-2", "0-3D-3"],
      targetContract: "Q[T,D] @ K_t[D,T] -> scores[T,T]",
      nodes: ["QTensor", "KTensor", "TransposeSwitch", "MatMulGate", "ScoreBoard"],
      tools: ["ScoreInspector"],
      checks: ["score_shape", "score_semantics", "numeric"]
    },
    {
      id: "0-3E",
      title: "Multi-Head Transpose Trap",
      conceptCards: ["0-3E-1", "0-3E-2", "0-3E-3"],
      targetContract: "Q[B,H,T,D] @ K_t[B,H,D,T] -> scores[B,H,T,T]",
      nodes: ["QTensor4D", "KTensor4D", "TransposeSwitch", "BatchedMatMulGate", "ScoreBoard", "ReferenceChecker"],
      tools: ["AxisLock", "CellTrace"],
      checks: ["carry_axes", "axis_semantics", "numeric"]
    },
    {
      id: "0-3F",
      title: "Trap Debugger",
      conceptCards: ["0-3F-1", "0-3F-2", "0-3F-3"],
      cases: ["no_transpose", "q_transposed", "all_axes_reversed", "t_equals_d_numeric"],
      tools: ["AxisTrace", "CellTrace", "ReferenceOverlay"],
      checks: ["repair_all_cases"]
    },
    {
      id: "0-3X",
      title: "Transpose Gauntlet",
      conceptCards: ["0-3X-1", "0-3X-2", "0-3X-3"],
      cases: [
        { B: 2, H: 2, T: 4, D: 3, fault: "no_k_transpose" },
        { B: 3, H: 2, T: 5, D: 4, fault: "carry_axes_swapped" },
        { B: 1, H: 3, T: 4, D: 4, fault: "t_equals_d_wrong_matmul" },
        { B: 2, H: 1, T: 6, D: 3, fault: "operand_order_reversed" }
      ],
      checks: ["visible", "hidden", "numeric", "cell_trace"]
    }
  ],

  rewards: [
    "Transpose Switch",
    "Axis Lock",
    "Score Board",
    "Cell Trace",
    "Contract: QK^T -> scores[B,H,T,T]",
    "Next Level: 0-4 Broadcast Add"
  ]
};
```

---

## 17. 失败信息库

### 17.1 No K Transpose

```text
K is not transposed.

Q expects the right operand's first matmul axis to be D.
Current K is [B,H,T,D].
Expected K_t is [B,H,D,T].

Fix:
Insert Transpose Switch on K path and swap axes -2 and -1.
```

---

### 17.2 Q Transposed Instead of K

```text
Wrong tensor transposed.

Q should stay as [B,H,T,D] so its T axis becomes score rows.
K should become [B,H,D,T] so its T axis becomes score columns.

Fix:
Remove transpose from Q path.
Apply transpose to K path.
```

---

### 17.3 Wrong Axis Swapped

```text
Wrong axes swapped.

Expected K axis map:
[B,H,T,D] → [B,H,D,T]

Your axis map moved an axis that should be preserved.
B and H are carry axes.
Only T and D should be swapped.
```

---

### 17.4 Carry Axis Moved

```text
Carry axis moved.

B/H align Q and K across batch and head.
If B or H moves, scores can mix samples or heads.

Fix:
Lock B and H in the Transpose Switch.
```

---

### 17.5 Operand Order Reversed

```text
Operand order reversed.

Expected:
Q @ K_t

Received:
K_t @ Q

Score rows should come from query tokens.
Score columns should come from key tokens.
```

---

### 17.6 Shape Passed But Numeric Failed

```text
Shape passed, but numeric reference failed.

This usually happens when T == D.
The output shape can look correct even when the operation is wrong.

Run Cell Trace:
scores[b,h,i,j] must equal dot(Q[b,h,i,:], K[b,h,j,:]).
```

---

### 17.7 Score Axis Semantics Invalid

```text
Score Board axis semantics invalid.

Expected output:
scores[B,H,T_query,T_key]

The first T must come from Q.
The second T must come from K.
```

---

## 18. 数值与示例数据建议

### 18.1 0-3A 推荐数据

使用整数矩阵，便于玩家看坐标映射：

```text
A = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
```

参考输出：

```text
A_t = [
  [1,5,9],
  [2,6,10],
  [3,7,11],
  [4,8,12]
]
```

---

### 18.2 0-3D 推荐数据

选小尺寸便于 Cell Trace：

```text
T = 3
D = 2

Q = [
  [1, 0],
  [0, 1],
  [1, 1]
]

K = [
  [2, 0],
  [0, 3],
  [1, 1]
]

K_t = [
  [2,0,1],
  [0,3,1]
]

scores = Q @ K_t = [
  [2,0,1],
  [0,3,1],
  [2,3,2]
]
```

这个数据非常适合解释：

```text
scores[2,1] = dot([1,1], [0,3]) = 3
```

---

### 18.3 0-3E 推荐默认 shape

```text
B = 2
H = 2
T = 4
D = 3
```

原因：

```text
B/H/T/D 都大于 1，能暴露 carry axis 错误。
T != D，未转置时会明显 shape mismatch。
```

---

### 18.4 0-3X 必须包含 T == D

```text
B = 1
H = 3
T = 4
D = 4
```

目的：

```text
让 shape-only 判断失效。
必须检查 numeric reference 和 cell trace。
```

---

## 19. 与后续关卡的连接

### 19.1 连接到 0-4 Broadcast Add

0-3 输出：

```text
scores[B,H,T,T]
```

0-4 可以继续引入：

```text
scores[B,H,T,T] + mask_bias[1,1,T,T] → masked_scores[B,H,T,T]
```

或：

```text
hidden[B,T,C] + bias[C] → hidden[B,T,C]
hidden[B,T,C] + pos_emb[T,C] → hidden[B,T,C]
```

0-3 的 T/T 轴语义会帮助玩家理解为什么 mask bias 的 shape 是 `[1,1,T,T]` 或 `[T,T]` 可以 broadcast。

---

### 19.2 连接到后续 Attention 章节

0-3 只完成：

```text
QKᵀ scores
```

后续 Attention 章节可以扩展：

```text
scores[B,H,T,T]
→ scale by 1/sqrt(D)
→ causal mask
→ softmax over key token axis
→ attention_weights[B,H,T,T]
→ attention_weights @ V[B,H,T,D]
→ context[B,H,T,D]
```

0-3 必须给玩家建立：

```text
scores 的行是 query token。
scores 的列是 key token。
```

因为 softmax 和 causal mask 都依赖这个行列语义。

---

## 20. MVP 实现建议

如果当前 MVP 时间有限，0-3 可先实现主线核心：

```text
MVP-1：0-3A Matrix Flip
- 2D matrix transpose
- shape + value map test

MVP-2：0-3D Single-Head QK Score
- Q[T,D], K[T,D]
- K transpose
- score[T,T]
- cell trace

MVP-3：0-3E Multi-Head Transpose Trap
- Q/K [B,H,T,D]
- transpose(-2,-1)
- output [B,H,T,T]
- reference checker

MVP-4：0-3X Hidden Test
- 至少包含一个 T==D shape-pass numeric-fail case
```

0-3B / 0-3C / 0-3F 可以作为中间教学增强，逐步补全。

---

## 21. 开发任务拆分

### 21.1 玩法逻辑

```text
- 实现 TransposeSwitch 节点
- 支持 axis swap 配置
- 支持 named axis map：B/H/T/D
- 支持 negative axis index：-2/-1
- 实现 AxisLock 逻辑
- 实现 MatMulGate 对 batch/carry axes 的广播式批量乘法检查
- 实现 ScoreBoard 节点
- 实现 CellTrace：选中 score cell，回溯 Q/K 来源
```

---

### 21.2 UI / 可视化

```text
- Matrix Plate 翻折动画
- 高阶 tensor axis track 可视化
- B/H carry axis lock 视觉
- T/D 交叉轨道视觉
- Score Board 热力图
- Score cell 来源高亮
- Difference heatmap：user output vs reference
- Axis Contract Panel
```

---

### 21.3 内容系统

```text
- Concept Card 配置
- Stage Debrief 配置
- Failure Lesson 配置
- Hidden Test Case 配置
- Debug Checklist 配置
- Tool Unlock 配置
```

---

### 21.4 Autograder / 数值系统

```text
- Tensor shape evaluator
- Axis semantic evaluator
- Axis map evaluator
- Value mapping evaluator for transpose
- Reference matmul implementation
- Allclose checker
- Sampled cell checker
- T==D trap checker
```

---

### 21.5 QA 测试点

```text
- 2D transpose value map 正确
- rank 4 transpose(-2,-1) 正确
- B/H 不被误交换
- Q @ K_t 输出 [B,H,T,T]
- Q_t @ K 被正确判错
- K_t @ Q 被正确判错
- T==D 时 shape 通过但 numeric fail 能被捕获
- Score cell trace 来源正确
- hidden tests 随机 B/H/T/D 通过
```

---

## 22. 最终体验目标

0-3 的目标不是让玩家记住：

```text
Attention 要 QKᵀ
```

而是让玩家形成可迁移的工程直觉：

```text
我知道 transpose 是轴交换。
我知道矩阵转置会把 [R,C] 变成 [C,R]。
我知道在 QKᵀ 中，K 的 T/D 轴必须交换。
我知道 B/H 是 carry axes，不能动。
我知道输出 scores[B,H,T,T] 的两个 T 分别来自 Q 和 K。
我知道 shape 通过不代表数值正确。
我知道如何用 Cell Trace 验证 scores[b,h,i,j] 的来源。
```

这会直接支撑后续 Attention、Causal Mask、Softmax 和 `weights @ V` 的学习。

---

# 附录 A：本关核心术语表

| 术语 | 玩家理解版本 | 工程表达 |
|---|---|---|
| Transpose | 交换轴 | `tensor.transpose(axis_a, axis_b)` |
| Matrix Transpose | 行列互换 | `A[R,C] → Aᵀ[C,R]` |
| Axis Map | 输出轴来自哪个输入轴 | `out[b,h,d,t] = in[b,h,t,d]` |
| Carry Axis | 不参与当前内维度计算、并行保留的轴 | B/H in `[B,H,T,D]` |
| Q | Query tensor | `Q[B,H,T,D]` |
| K | Key tensor | `K[B,H,T,D]` |
| Kᵀ | Key 的最后两轴转置 | `K.transpose(-2,-1)` → `[B,H,D,T]` |
| D | head_dim | 每个 head 的 feature 维度 |
| H | attention heads | 多头 attention 的 head 轴 |
| scores | query-key 分数矩阵 | `scores[B,H,T_query,T_key]` |
| Cell Trace | 追踪某个输出格子的来源 | `scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:])` |
| T == D Trap | shape 检查容易被骗的等尺寸陷阱 | wrong operation may pass shape check |

---

# 附录 B：本关一句话标语

```text
Transpose 不是转一下面板，而是交换张量的轴。
```

备用标语：

```text
QKᵀ 的那个 ᵀ，决定了 score board 是 token-to-token，而不是 feature-to-feature。
```

---

# 附录 C：Concept Card 内容配置规范

## C.1 文本长度规则

```text
标题：不超过 18 个字
主文案：60–120 字
公式：1–3 行
任务桥接语：1 句
```

## C.2 出现时机

```text
Intro Cards：关卡进入时，可跳过
Stage Concept Cards：每个挑战第一次进入时显示
Failure Lessons：仅失败后显示
Debrief：阶段通过后显示
More Info：可展开，不自动打断玩家
```

## C.3 可跳过规则

```text
第一次进入：显示全部 Concept Card
重玩关卡：只显示标题 + 目标，可展开详细知识
隐藏测试：不再弹概念卡，只显示 Debug Checklist
```

---

# 附录 D：可选支线挑战知识卡片

## D.1 Bonus A：Index Mirror 知识卡

### Card Bonus-A-1：坐标也会被转置

```text
Transpose 不只是 shape 改变。

每个值的坐标也会镜像：

A[i,j] → Aᵀ[j,i]

如果只改 shape 标签，
数值位置没有跟着移动，
那不是正确的 transpose。
```

---

## D.2 Bonus B：Axis Permutation Lab 知识卡

### Card Bonus-B-1：Transpose 与 Permute

```text
Transpose 通常交换两个轴。
Permute 可以重新排列多个轴。

本关主线只需要：
K.transpose(-2,-1)

也就是只交换 K 的最后两个轴。
```

---

## D.3 Bonus C：T == D Numeric Trap 知识卡

### Card Bonus-C-1：当 shape 骗过你

```text
如果 T 和 D 相等，
错误的 Q @ K 可能也输出同样大小的矩阵。

这时 shape test 不够。
你必须检查：

scores[i,j] 是否真的等于 dot(Q[i,:], K[j,:])。
```

---

## D.4 Bonus D：Attention Preview 知识卡

### Card Bonus-D-1：QKᵀ 只是 Attention 的第一步

```text
QKᵀ 生成的是 raw scores。

真正的 attention 还要经过：
scale、mask、softmax，
最后再乘以 V。

本关先确保 scores 的方向正确。
```

---

# 附录 E：建议埋点与数据分析

为了验证 0-3 的教学效果，建议记录：

```text
- 玩家第一次是否把 K 接入 Transpose Switch
- 玩家是否误转置 Q
- 玩家是否移动 B/H carry axes
- 玩家在 T==D case 中是否只通过 shape 判断
- Cell Trace 使用次数
- Reference Checker 使用次数
- 首次通过 0-3E 所需时间
- 0-3X 每个 case 的失败类型分布
```

这些数据能帮助判断玩家是否真的理解了 transpose trap，而不是背流程。

