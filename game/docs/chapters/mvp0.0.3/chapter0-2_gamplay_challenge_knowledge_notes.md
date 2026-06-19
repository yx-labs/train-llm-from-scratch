# LLM Complete — 0-2 MatMul Gate 可执行详细设计文档

> 章节定位：Chapter 0 / Tensor Bootcamp 的第二关。  
> 前置关卡：0-1 Shape Reader，玩家已经掌握 `Tensor / Rank / Shape / Axis Semantics / hidden[B,T,C]`。  
> 本关目标：让玩家通过节点画布、张量可视化、权重板旋转、合同验证和数值测试，真正理解 **MatMul / Linear Projection 如何消费 C 轴并生成 O 轴**。

---

## 0. 设计总述

### 0.1 关卡一句话

玩家需要修复一台损坏的 **MatMul Gate**。输入是上一关修好的：

```text
hidden[B,T,C]
```

玩家要连接正确的权重板，理解矩阵乘法的内维度匹配规则，处理权重方向问题，最终让系统输出：

```text
projected[B,T,O]
```

这一关的核心不是“选择正确公式”，而是让玩家在画布上亲手完成：

```text
hidden[B,T,C] @ weight[C,O] → projected[B,T,O]
```

并通过可视化 trace 和 autograder 证明结果正确。

---

### 0.2 设计原则

0-2 延续 0-1 的组织结构：

```text
Concept Capsule / 最小概念讲解
→ Broken Board / 故障画布
→ Repair Operation / 玩家修复
→ Probe & Trace / 探针观察
→ Autograder / 可验证测试
→ Debrief / 阶段小结
→ Tool Unlock / 工具解锁
```

本关不做选择题，不要求玩家背公式，而是让玩家形成工程直觉：

```text
MatMul 的关键是内维度对齐。
Linear 消费 hidden 的 C 轴。
输出保留 B/T 轴，并把 C 替换成 O。
Weight 的方向非常重要。
错误的 transpose 会导致 shape mismatch 或数值错误。
```

---

### 0.3 本关学习成果

玩家通关后应该能自然理解：

```text
1. 向量点积 [C] · [C] → scalar。
2. 单个 token 的特征向量可以被投影：[C] @ [C,O] → [O]。
3. 一句话中的所有 token 都可以共享同一个权重投影：[T,C] @ [C,O] → [T,O]。
4. 一个 batch 中的所有样本也可以共享同一个权重投影：[B,T,C] @ [C,O] → [B,T,O]。
5. C 是被 Linear 消费的输入特征轴。
6. O 是 Linear 生成的新输出特征轴。
7. 权重 `[C,O]` 和 `[O,C]` 不是同一个计算方向。
8. PyTorch 等框架中常见的权重存储方向 `[O,C]` 需要在计算时转置。
9. Shape 通过不等于数值正确，最终必须与 reference implementation 数值对齐。
```

---

## 1. 与 0-1 的衔接

0-1 结束时，玩家已经修复：

```text
hidden[B,T,C]
```

并理解：

```text
B = batch / 独立样本
T = token position / token 序列位置
C = channel / 每个 token 的特征通道
```

0-2 的第一个引导要把问题自然抛给玩家：

> 现在我们已经知道每个 token 有一条 C 维特征向量。  
> 接下来，模型要用 Linear Projection 把这条向量变成新的特征空间。  
> 这一步的核心计算就是 MatMul。

从 0-1 到 0-2 的知识推进关系：

```text
0-1：读懂 hidden[B,T,C]
0-2：用 weight[C,O] 投影 hidden 的 C 轴
0-3：用 MatMul 结果构建 Q/K/V，并解决 QKᵀ 的 transpose 问题
0-4：理解 bias、position embedding 等 broadcast add
```

---

## 2. 本关整体结构

### 2.1 阶梯挑战总览

| 阶段 | 名称 | 玩家学到什么 | 主要操作 | 验证方式 |
|---|---|---|---|---|
| 0-2A | Dot Cell | 点积会消费两个同长度 C 向量 | 对齐两个 C 向量，计算一个 scalar | 单格数值对齐 reference |
| 0-2B | Token Projection | `[C] @ [C,O] → [O]` | 将单个 token vector 接入 weight plate | 输出 O 维向量 shape 正确 |
| 0-2C | Sequence Projection | `[T,C] @ [C,O] → [T,O]` | 对每个 token 复用同一权重板 | T 被保留，C 被消费 |
| 0-2D | Batch Projection | `[B,T,C] @ [C,O] → [B,T,O]` | 将 hidden tensor 接入 MatMul Gate | B/T 保留，O 生成 |
| 0-2E | Weight Orientation Trap | 权重方向会影响能否相乘 | 旋转或转置 `[O,C]` 权重板 | 内维度匹配 + 数值正确 |
| 0-2F | Linear Module Assembly | Linear = MatMul projection，bias 暂不展开 | 搭建完整 Linear Projection 节点链 | 输出合同通过 |
| 0-2X | MatMul Gauntlet | 泛化到不同 B/T/C/O 和权重方向 | 连续修复多个随机 MatMul Gate | hidden tests 全部通过 |

可选支线：

```text
Bonus A：Cell Inspector / 手算单个输出值
Bonus B：PyTorch Weight Storage / [O,C] 存储方向说明
Bonus C：QKV Projection Preview / 为 Attention 关卡做预告
Bonus D：FLOPs Meter / 理解 B*T*C*O 计算量
```

---

### 2.2 关卡主流程

```text
关卡开场总引导
→ 0-2A Dot Cell
→ 0-2B Token Projection
→ 0-2C Sequence Projection
→ 0-2D Batch Projection
→ 0-2E Weight Orientation Trap
→ 0-2F Linear Module Assembly
→ 0-2X MatMul Gauntlet
→ 通关总结
```

---

## 3. 关卡开场总引导

### 3.1 开场界面标题

```text
0-2 MatMul Gate
Linear Projection Bootcamp
```

### 3.2 开场文案

```text
上一关中，你修复了 hidden[B,T,C]。

现在，每个 token 都带着一条 C 维 feature vector。
模型需要把这些特征投影到新的特征空间。

这个过程叫 Linear Projection，核心计算是 MatMul。

本关目标：
修复 MatMul Gate，让 hidden[B,T,C] 通过 weight[C,O]，输出 projected[B,T,O]。
```

### 3.3 开场图示

```text
hidden[B,T,C]
      │
      │ Linear / MatMul
      ▼
projected[B,T,O]
```

重点高亮：

```text
C 被消费
O 被生成
B/T 被保留
```

按钮：

```text
Start MatMul Bootcamp
```


### 3.4 开场知识介绍卡片扩展

0-2 开场只做方向建立，不抢先讲完所有 MatMul 细节。建议使用 3 张可跳过的短卡片，每张卡控制在 15–25 秒。

#### Card 0-2-Intro-1：从 `hidden[B,T,C]` 到新特征空间

**标题**

```text
上一关修好的 hidden[B,T,C] 要继续被加工
```

**玩家可见文案**

```text
0-1 中，你已经修复了 hidden[B,T,C]。

现在，每个 token 都带着一条 C 维特征向量。
模型接下来要把这条向量投影到新的特征空间。
```

**画面表现**

```text
hidden[B,T,C]
B/T 网格保持不动
每个 token 格子里的 C 维 feature strip 被高亮
```

**微交互**

玩家点击一个 token 格子，展开：

```text
hidden[b,t,:] = [C0, C1, C2, ...]
```

右侧 Inspector 显示：

```text
This is one token feature vector.
Linear will consume this C axis.
```

**过渡语**

```text
下一张：Linear Projection 的核心其实是 MatMul。
```

---

#### Card 0-2-Intro-2：Linear Projection = 用权重板改变特征维度

**标题**

```text
Linear Projection：把 C 变成 O
```

**玩家可见文案**

```text
Linear Projection 会读取一个 C 维输入向量，
再通过一块 weight plate，生成 O 维输出向量。

在本关里，我们只关注核心计算：MatMul。
```

**公式显示**

```text
hidden[..., C] @ weight[C, O] → projected[..., O]
```

**画面表现**

```text
Feature Strip[C] → MatMul Gate ← Weight Plate[C,O]
                    ↓
              Output Strip[O]
```

**微交互**

鼠标悬停 `C`：

```text
C = input feature channels，来自 hidden tensor。
```

鼠标悬停 `O`：

```text
O = output feature channels，由 weight plate 生成。
```

**过渡语**

```text
下一张：MatMul 最容易出错的地方是内维度和权重方向。
```

---

#### Card 0-2-Intro-3：本关要修复什么

**标题**

```text
MatMul Gate 的三类故障
```

**玩家可见文案**

```text
本关你会修复三类真实工程问题：

1. 内维度没有对齐
2. 输出轴标注错误
3. weight plate 方向反了
```

**画面表现**

三个故障预览并排显示：

```text
C ≠ C          → inner dimension mismatch
[B,O,T]        → output contract error
weight[O,C]    → needs transpose
```

**本关目标**

```text
最终修复：
hidden[B,T,C] @ weight[C,O] → projected[B,T,O]
```

**按钮**

```text
Enter MatMul Workbench
```


---

## 4. 核心视觉语言

### 4.1 Feature Vector / 特征条

一个 token 的 C 维向量表现为一条水平特征条：

```text
hidden[b,t,:]
[C0][C1][C2][C3][C4][C5][C6][C7]
```

每个格子是一个 float value。选中某个格子时 Inspector 显示：

```text
hidden[b=0, t=2, c=5] = -0.381
```

### 4.2 Weight Plate / 权重板

权重矩阵表现为一块可旋转的矩形板：

```text
weight[C,O]

C axis ↓
O axis →
```

每一列可以理解为一个 output channel 的权重向量：

```text
W[:,0] → output channel O0
W[:,1] → output channel O1
...
```

### 4.3 MatMul Gate / 矩阵乘法门

MatMul Gate 有两个输入端口和一个输出端口：

```text
Left input:  activation[..., C]
Right input: weight[C, O]
Output:      activation[..., O]
```

Gate 中间有一个 **Inner Dimension Dock / 内维度接口**。

合法对齐时：

```text
C  ── matches ── C
```

非法对齐时：

```text
C  ── mismatch ── O
```

### 4.4 Output Tensor / 输出张量柜

输出形象为一个新的数据柜：

```text
projected[B,T,O]
```

视觉上 B/T 方向保持不变，C 方向替换为 O 方向。可以用动画表现：

```text
C 轴被 MatMul Gate 吞入
O 轴从 Gate 另一侧生成
```

### 4.5 颜色建议

| 元素 | 颜色语义 |
|---|---|
| B 轴 | 样本层颜色，稳定保留 |
| T 轴 | token 序列颜色，稳定保留 |
| C 轴 | 输入 feature 颜色，被消费 |
| O 轴 | 输出 feature 颜色，新生成 |
| 合法内维度 | 蓝白色吸附光 |
| 维度错误 | 红色断裂光 |
| Transpose 修复 | 紫色旋转光 |
| 数值 reference 通过 | 绿色校验光 |

---

## 5. 0-2A：Dot Cell / 点积单元

### 5.1 阶段目标

玩家掌握：

```text
两个同长度向量可以做 dot product。
[C] · [C] → scalar
```

这是所有 MatMul 的最小单元。

### 5.2 Concept Capsule

标题：

```text
Dot Product：两个 C 向量相乘求和
```

文案：

```text
一个 output value 来自一次点积。

模型会取一个 token 的 C 维特征向量，
再取一条同样长度的 C 维权重向量，
逐项相乘后求和，得到一个数字。
```

公式：

```text
out[o] = Σ hidden[c] * weight[c,o]
```

本阶段任务：

```text
对齐两个 C 维向量，生成一个 output scalar。
```


#### 5.2.1 挑战前知识介绍卡片扩展

0-2A 是本关的最小计算单元。知识卡不要一开始就讲矩阵乘法，而是先让玩家看懂“一个输出数值从哪里来”。

##### Card 0-2A-1：一个输出值来自一次点积

**标题**

```text
一个 output cell 是怎样算出来的？
```

**玩家可见文案**

```text
MatMul 看起来像整块矩阵相乘，
但它最小的计算单元其实是 dot product。

一条输入向量和一条权重向量，
逐项相乘，再全部加起来，得到一个数字。
```

**画面表现**

```text
hidden[C]       weight[C]
[2, -1, 3]  ·  [4, 5, -2]

2*4 + (-1)*5 + 3*(-2) = -3
```

**微交互**

玩家点击 `Run Mini Dot`，两个向量从左到右逐项高亮：

```text
C0 × C0 → partial sum
C1 × C1 → partial sum
C2 × C2 → final scalar
```

**任务桥接**

```text
现在，把两个 C 维向量接到 Dot Cell，生成一个 scalar。
```

---

##### Card 0-2A-2：点积要求两个向量长度相同

**标题**

```text
Dot Product 需要相同的 C 长度
```

**玩家可见文案**

```text
点积是一一配对计算。

如果左边有 8 个 feature，右边也必须有 8 个 weight。
长度不一致，就没有办法逐项相乘。
```

**画面表现**

合法：

```text
[C=8] · [C=8] → scalar
```

非法：

```text
[C=8] · [C=6] → mismatch
```

**微交互**

拖动一个错误长度的 `Weight Column[C=6]` 到 Dot Cell，Gate 的 Inner Dock 红灯闪烁，显示：

```text
Inner length mismatch: 8 vs 6
```

**任务桥接**

```text
用 Shape Inspector 确认两个输入都是同长度 [C]。
```

---

##### Card 0-2A-3：Dot Cell 是 MatMul 的一个格子

**标题**

```text
Dot Cell 是 MatMul 的最小格子
```

**玩家可见文案**

```text
之后你会看到更大的矩阵和张量。

但每一个输出格子，
本质上仍然来自一次 dot product。
```

**画面表现**

```text
MatMul Output Grid
[ cell ][ cell ][ cell ]
   ↑
每个 cell 都由一个 Dot Cell 计算
```

**任务桥接**

```text
先通过 Dot Cell，下一步再把多个 Dot Cell 并排组成 O 维输出。
```


---

### 5.3 画布初始状态

```text
[Token Feature Vector]      [Weight Column]
      hidden[C]                  w[C]
          │                       │
          └─────? Dot Cell ?──────┘
                    │
                 output[?]
```

节点：

```text
Token Feature Vector: float32[C]
Weight Column:        float32[C]
Dot Cell:             expects [C] and [C]
Scalar Output:         float32[]
```

### 5.4 玩家操作

玩家需要：

1. 将 `Token Feature Vector` 接入 Dot Cell 左端口；
2. 将 `Weight Column` 接入 Dot Cell 右端口；
3. 点击 `Run Dot Trace`；
4. 观察逐项相乘动画：

```text
h0*w0 + h1*w1 + h2*w2 + ... + hC*wC
```

### 5.5 交互反馈

合法连接后：

```text
Inner length matched: C = 8
```

Trace 时，两个向量格子逐项高亮：

```text
C0 × C0
C1 × C1
C2 × C2
...
SUM → scalar
```

### 5.6 验证规则

```text
Check 1: left.rank == 1
Check 2: right.rank == 1
Check 3: left.shape[0] == right.shape[0]
Check 4: output.rank == 0
Check 5: abs(output - reference_dot) < 1e-5
```

### 5.7 失败反馈

如果向量长度不一致：

```text
Dot Cell failed.

Left vector length = 8
Right vector length = 6

Dot product requires both vectors to have the same length.
```

如果玩家接入的是 `[T,C]` 而不是 `[C]`：

```text
Dot Cell expected a single feature vector [C],
but received a matrix [T,C].

Use Slice Probe to select one token vector hidden[b,t,:].
```

### 5.8 阶段小结

```text
Stage Complete: Dot Cell

你已经看到：
一个输出数值来自一次 C 维点积。

下一步：
如果我们同时需要 O 个输出数值，就需要 O 条权重向量。
这就是 [C] @ [C,O] → [O]。
```

解锁：

```text
Weight Plate
Single Token Projection Gate
```

---

## 6. 0-2B：Token Projection / 单 token 投影

### 6.1 阶段目标

玩家掌握：

```text
[C] @ [C,O] → [O]
```

即：一个 token 的 C 维特征向量，通过权重矩阵，变成 O 维输出向量。

---

### 6.2 Concept Capsule

标题：

```text
Weight Plate：一次生成 O 个输出通道
```

文案：

```text
一块 weight[C,O] 可以看成 O 条并排的 C 维权重向量。

每一列生成一个 output channel。
所以一个 token vector [C] 通过 weight[C,O] 后，会变成 [O]。
```

图示：

```text
hidden[C] @ weight[C,O] → projected[O]
```

本阶段任务：

```text
把一个 token 的 feature vector 投影成 O 维 output vector。
```


#### 6.2.1 挑战前知识介绍卡片扩展

0-2B 从“一个 scalar”扩展到“一条 O 维输出向量”。这里要讲清楚 Weight Plate 的列与 output channel 的关系。

##### Card 0-2B-1：O 个输出 = O 条权重向量

**标题**

```text
从一个输出值，到 O 个输出值
```

**玩家可见文案**

```text
如果只有一条权重向量，
一个 token vector 只能生成一个 scalar。

如果有 O 条权重向量并排放在一起，
同一个 token vector 就能生成 O 个输出值。
```

**画面表现**

```text
hidden[C]
   │
   ├─ dot weight[:,0] → out[0]
   ├─ dot weight[:,1] → out[1]
   ├─ dot weight[:,2] → out[2]
   └─ ...
```

**微交互**

玩家拖动 `O` 滑杆：

```text
O = 1 → 一列 weight → 一个输出值
O = 4 → 四列 weight → 四个输出值
```

**任务桥接**

```text
接下来，把一个 token vector 投影成 output[O]。
```

---

##### Card 0-2B-2：Weight Plate 的方向

**标题**

```text
weight[C,O]：C 向下，O 向右
```

**玩家可见文案**

```text
在本关的计算约定里，Weight Plate 的形状是 [C,O]。

C 轴负责和输入向量对齐。
O 轴负责生成输出通道。
```

**画面表现**

```text
weight[C,O]

        O0   O1   O2
C0     w00  w01  w02
C1     w10  w11  w12
C2     w20  w21  w22
```

**微交互**

悬停一列 `O1`，整列 `weight[:,1]` 高亮，旁边显示：

```text
This column produces output channel O1.
```

**任务桥接**

```text
把 weight 的 C 轴对准 MatMul Gate 的 inner dock。
```

---

##### Card 0-2B-3：C 被消费，O 被生成

**标题**

```text
输出不再是 C，而是 O
```

**玩家可见文案**

```text
MatMul 会沿 C 轴做点积。

因此 C 在输出中消失，
新的输出轴由 weight 的 O 轴生成。
```

**公式显示**

```text
[C] @ [C,O] → [O]
```

**画面表现**

```text
Input Feature Strip[C] 被 Gate 吞入
Output Strip[O] 从 Gate 另一侧生成
```

**任务桥接**

```text
完成后，请确认输出合同是 [O]，不是 [C]。
```


---

### 6.3 画布初始状态

```text
[Token Vector]
 float32[C]
      │
      ▼
[MatMul Gate] ◄──── [Weight Plate]
                     float32[C,O]
      │
      ▼
[Output Vector]
 float32[?]
```

### 6.4 玩家操作

玩家需要：

1. 将 `Token Vector[C]` 接入 MatMul Gate 左端口；
2. 将 `Weight Plate[C,O]` 接入右端口；
3. 将输出端连接到 `Output Vector`；
4. 运行测试。

### 6.5 可视化表现

MatMul Gate 内部展开成 O 个 Dot Cell：

```text
out[0] = dot(hidden[C], weight[:,0])
out[1] = dot(hidden[C], weight[:,1])
out[2] = dot(hidden[C], weight[:,2])
...
```

视觉上：

```text
一个输入 feature strip
同时扫过多列 weight
生成一条 output strip
```

### 6.6 验证规则

```text
Check 1: input.shape == [C]
Check 2: weight.shape == [C,O]
Check 3: inner dimension C matches
Check 4: output.shape == [O]
Check 5: output values allclose reference
```

### 6.7 失败反馈

如果权重是 `[O,C]`：

```text
MatMul failed: inner dimension mismatch.

Input vector: [C=8]
Weight plate: [O=4, C=8]

For this gate, weight must be oriented as [C,O].
Try rotating or transposing the weight plate.
```

如果玩家把输出标成 `[C]`：

```text
Output contract failed.

C was consumed by MatMul.
The new axis is O, not C.

Expected output: [O]
```

### 6.8 阶段小结

```text
Stage Complete: Token Projection

你已经完成：
[C] @ [C,O] → [O]

C 是输入特征轴，被 MatMul 消费。
O 是输出特征轴，由 weight 的第二个轴生成。
```

解锁：

```text
Sequence Projection Board
Carrier Axis Visualizer
```

---

## 7. 0-2C：Sequence Projection / 序列投影

### 7.1 阶段目标

玩家掌握：

```text
[T,C] @ [C,O] → [T,O]
```

每个 token 都有一条 C 维向量。Linear 对每个 token 独立应用同一块 weight。

---

### 7.2 Concept Capsule

标题：

```text
T 轴会被保留
```

文案：

```text
一句话有 T 个 token。
每个 token 都有一条 C 维 feature vector。

Linear Projection 会对每个 token 都执行同一套 [C] @ [C,O]。

因此：
T 轴保留，C 轴变成 O 轴。
```

图示：

```text
sequence[T,C] @ weight[C,O] → sequence[T,O]
```

本阶段任务：

```text
将一整条 token sequence 投影到 O 维输出空间。
```


#### 7.2.1 挑战前知识介绍卡片扩展

0-2C 引入第一个 carrier axis：T。核心是让玩家理解 Linear 不会混合 token 位置，而是对每个 token 共享同一块权重。

##### Card 0-2C-1：一句话是一排 token vector

**标题**

```text
Sequence Tensor = 多个 token vector 排成一行
```

**玩家可见文案**

```text
一个 token 是 [C]。

一整句话有 T 个 token，
所以它的 hidden slice 可以看成 [T,C]：
每一行是一个 token 的 C 维向量。
```

**画面表现**

```text
T0: [C0 C1 C2 ...]
T1: [C0 C1 C2 ...]
T2: [C0 C1 C2 ...]
```

**微交互**

玩家点击 `T1`，该行展开为：

```text
hidden[t=1, :] = one token feature vector [C]
```

**任务桥接**

```text
接下来，把每一个 token vector 都送过同一块 Weight Plate。
```

---

##### Card 0-2C-2：T 轴是携带轴，不参与点积

**标题**

```text
T 被保留，C 被投影
```

**玩家可见文案**

```text
Linear Projection 不会把不同 token 位置混在一起。

它会对 T0、T1、T2 ... 分别执行同一个 [C] @ [C,O]。
所以 T 轴会保留到输出。
```

**公式显示**

```text
[T,C] @ [C,O] → [T,O]
```

**画面表现**

```text
T0: [C] → [O]
T1: [C] → [O]
T2: [C] → [O]
```

**微交互**

点击 `Carrier Axis Visualizer`，T 轴以保护框形式穿过 MatMul Gate，显示：

```text
T is carried, not consumed.
```

**任务桥接**

```text
运行 Sequence Trace，观察每个 token 行独立通过 MatMul。
```

---

##### Card 0-2C-3：同一块权重被复用

**标题**

```text
每个 token 使用同一块 weight
```

**玩家可见文案**

```text
Linear 的参数不是给每个 token 单独准备一份。

同一块 weight[C,O] 会被所有 token 位置共享。
这让模型在不同位置上使用同一套特征变换规则。
```

**画面表现**

```text
T0 ─┐
T1 ─┼── shared weight[C,O] → T0/T1/T2 outputs
T2 ─┘
```

**任务桥接**

```text
请不要复制多块 weight。把整条 sequence 接入同一个 MatMul Gate。
```


---

### 7.3 画布初始状态

```text
[Sequence Tensor]
 float32[T,C]
      │
      ▼
[MatMul Gate] ◄──── [Weight Plate]
                     float32[C,O]
      │
      ▼
[Projected Sequence]
 float32[?,?]
```

### 7.4 玩家操作

玩家需要：

1. 连接 `Sequence Tensor[T,C]` 到 Gate；
2. 连接 `Weight Plate[C,O]`；
3. 在 Output Contract 中标注输出轴：

```text
Axis 0: T
Axis 1: O
```

4. 运行 `Sequence Trace`。

### 7.5 可视化表现

Sequence Tensor 以表格形式呈现：

```text
T0: [C0 C1 C2 ...]
T1: [C0 C1 C2 ...]
T2: [C0 C1 C2 ...]
...
```

Trace 动画：

- MatMul Gate 逐行扫描每个 token；
- 每一行 `[C]` 通过同一块 `weight[C,O]`；
- 输出对应一行 `[O]`；
- 最终得到 `[T,O]`。

### 7.6 验证规则

```text
Check 1: input.shape == [T,C]
Check 2: weight.shape == [C,O]
Check 3: input last axis matches weight first axis
Check 4: output.shape == [T,O]
Check 5: T axis preserved
Check 6: output values allclose reference
```

### 7.7 失败反馈

如果玩家把 T 当作被消费轴：

```text
Projection failed.

You consumed T, but Linear should consume C.

T represents token positions and must be preserved.
C represents per-token features and should be projected into O.
```

如果输出标成 `[O,T]`：

```text
Output axis order mismatch.

Input sequence order T is preserved as the leading axis.
Expected output shape: [T,O]
```

### 7.8 阶段小结

```text
Stage Complete: Sequence Projection

你已经完成：
[T,C] @ [C,O] → [T,O]

Linear 对每个 token 独立应用同一块 weight。
T 轴不参与点积，只是被携带到输出。
```

解锁：

```text
Batch Projection Board
Carrier Axes: B/T
```

---

## 8. 0-2D：Batch Projection / 批量投影

### 8.1 阶段目标

玩家掌握完整关卡核心公式：

```text
hidden[B,T,C] @ weight[C,O] → projected[B,T,O]
```

B 和 T 是 carrier axes，被保留；C 是 inner dimension，被消费；O 是新轴。

---

### 8.2 Concept Capsule

标题：

```text
B/T 被携带，C 被消费
```

文案：

```text
hidden[B,T,C] 中，
B 表示多条样本，
T 表示每条样本的 token 位置，
C 表示每个 token 的 feature vector。

MatMul 只在 C 轴上做点积。
B 和 T 会原样保留到输出。
```

公式：

```text
hidden[B,T,C] @ weight[C,O] → projected[B,T,O]
```

本阶段任务：

```text
修复完整 Batch MatMul Gate。
```


#### 8.2.1 挑战前知识介绍卡片扩展

0-2D 是本关主公式第一次完整出现。知识卡要把 B/T carrier axes、C consumed axis、O generated axis 讲清楚。

##### Card 0-2D-1：Batch 只是并行样本

**标题**

```text
B 轴也会被保留
```

**玩家可见文案**

```text
hidden[B,T,C] 中，B 表示一次送入模型的多条样本。

Linear 会对每条样本、每个 token 都执行同样的 C→O 投影。
它不会把不同样本混在一起。
```

**画面表现**

```text
B0: sequence[T,C] → sequence[T,O]
B1: sequence[T,C] → sequence[T,O]
```

**微交互**

玩家切换 B0/B1 样本层，看到两层都使用同一块 Weight Plate，但数据彼此独立。

**任务桥接**

```text
这一次，输入将是完整的 hidden[B,T,C]。
```

---

##### Card 0-2D-2：本关核心公式

**标题**

```text
hidden[B,T,C] @ weight[C,O] → projected[B,T,O]
```

**玩家可见文案**

```text
MatMul 只消耗最后的 C 轴。

B 和 T 只是携带轴，会原样保留。
weight 的 O 轴会成为输出的新 feature axis。
```

**公式拆解**

```text
输入：hidden[B,T,C]
权重：weight[C,O]
输出：projected[B,T,O]
```

**画面表现**

```text
B/T 网格位置固定
每个格子内部：C feature strip → O feature strip
```

**任务桥接**

```text
请修复完整 Batch MatMul Gate，并标注输出为 [B,T,O]。
```

---

##### Card 0-2D-3：三类轴角色

**标题**

```text
Carrier / Consumed / Generated
```

**玩家可见文案**

```text
在这个 MatMul 里，轴有三种角色：

B/T：Carrier axes，被保留。
C：Consumed axis，被点积消耗。
O：Generated axis，由 weight 生成。
```

**画面表现**

三色标注：

```text
B/T：穿过 Gate
C：进入 Inner Dock 后消失
O：从 Weight Plate 右侧生成
```

**任务桥接**

```text
在 Gate 的轴角色面板中确认：B/T carried，C consumed，O generated。
```


---

### 8.3 画布初始状态

画布显示一条损坏管线：

```text
[Hidden Tensor]
 float32[B,T,C]
      │
      ▼
[MatMul Gate] ◄──── [Weight Plate]
                     float32[C,O]
      │
      ╳ output contract missing
      ▼
[Projected Tensor]
 float32[?,?,?]
```

### 8.4 玩家操作

玩家需要：

1. 将 hidden 的 activation output 接入 MatMul Gate；
2. 将 weight plate 接入 MatMul Gate；
3. 在 Gate 的 carrier axis 面板中确认：

```text
Carried axes: B, T
Consumed axis: C
Generated axis: O
```

4. 给 Projected Tensor 贴轴标签：

```text
Axis 0 → B
Axis 1 → T
Axis 2 → O
```

5. 连接到 Output Contract Gate。

### 8.5 可视化表现

Hidden Tensor 以三维数据柜出现：

```text
B 层：样本层
T 列：token 位置
C 条：特征通道
```

运行时：

- 对每个 `(b,t)` 位置，取出一条 `[C]` feature vector；
- 送入 Weight Plate；
- 生成 `[O]` output vector；
- 填回同样的 `(b,t)` 位置；
- 最终形成 `[B,T,O]`。

画面强调：

```text
B/T 网格位置不动
每个格子里的 C 条被换成 O 条
```

### 8.6 验证规则

```text
Check 1: input contract == hidden[B,T,C]
Check 2: weight compute shape == [C,O]
Check 3: inner dimension C matches
Check 4: output rank == 3
Check 5: output contract == projected[B,T,O]
Check 6: B/T axes preserved in order
Check 7: C not present in output
Check 8: O generated from weight second axis
Check 9: output values allclose reference
```

### 8.7 失败反馈

如果玩家输出写成 `[B,O,T]`：

```text
Output axis order mismatch.

B and T are carrier axes from the input.
Their order should remain [B,T].

Expected: [B,T,O]
Received: [B,O,T]
```

如果玩家把 C 保留到输出：

```text
Output contract failed.

C was consumed by MatMul.
It should not appear in the output tensor.

Expected generated axis: O
```

如果玩家把 O 接到 input side：

```text
MatMul contract failed.

O is not an input feature axis.
O is generated by the weight plate's output axis.
```

### 8.8 阶段小结

```text
Stage Complete: Batch Projection

你已经完成本关核心结构：

hidden[B,T,C] @ weight[C,O] → projected[B,T,O]

B/T 被保留。
C 被消费。
O 被生成。
```

解锁：

```text
Weight Orientation Switch
Transpose Repair Tool
```

---

## 9. 0-2E：Weight Orientation Trap / 权重方向陷阱

### 9.1 阶段目标

玩家掌握：

```text
weight[C,O] 和 weight[O,C] 是不同方向。
如果权重存储为 [O,C]，计算时需要转置成 [C,O]。
```

这一阶段为真实工程做准备，因为很多框架中 `Linear` 的权重存储方向是 `[out_features, in_features]`。

---

### 9.2 Concept Capsule

标题：

```text
Weight Orientation：存储方向和计算方向
```

文案：

```text
MatMul Gate 的计算方向需要：

activation[..., C] @ weight[C,O]

但有些权重板可能以 [O,C] 的方向存储。
这时不能直接接入 Gate，必须先转置。
```

图示：

```text
Stored weight[O,C]
      ↓ transpose
Compute weight[C,O]
```

本阶段任务：

```text
修复方向错误的 weight plate。
```


#### 9.2.1 挑战前知识介绍卡片扩展

0-2E 要重点处理真实工程里的“存储方向”和“计算方向”差异。这里容易成为玩家第一次感受到硬核调试价值的挑战。

##### Card 0-2E-1：权重方向不是视觉装饰

**标题**

```text
weight[C,O] 和 weight[O,C] 不等价
```

**玩家可见文案**

```text
一块矩阵板旋转 90 度后，shape 会改变。

对于 MatMul Gate 来说，
它需要右侧输入的第一个轴是 C，
这样才能和 activation 的最后一个 C 对齐。
```

**对比图**

```text
正确计算方向：
activation[...,C] @ weight[C,O]

错误方向：
activation[...,C] @ weight[O,C]
```

**微交互**

玩家拖动 `[O,C]` Weight Plate 靠近 MatMul Gate，Inner Dock 红灯显示：

```text
expected first axis: C
received first axis: O
```

**任务桥接**

```text
先不要运行测试，先检查 Weight Plate 的方向。
```

---

##### Card 0-2E-2：存储方向 vs 计算方向

**标题**

```text
Stored Weight 不一定能直接计算
```

**玩家可见文案**

```text
真实框架里，Linear 的 weight 经常以 [O,C] 存储。

但计算 x[...,C] @ weight 时，
需要使用 weight.T，也就是 [C,O]。
```

**公式显示**

```text
stored_weight[O,C]
compute_weight = transpose(stored_weight)  # [C,O]
output = hidden[...,C] @ compute_weight[C,O]
```

**画面表现**

```text
Stored Weight[O,C]
      │ Transpose Switch
      ▼
Compute Weight[C,O]
```

**任务桥接**

```text
插入 Transpose Switch，让 C 轴对准 MatMul Gate。
```

---

##### Card 0-2E-3：转置修复的是轴顺序，不是数值魔法

**标题**

```text
Transpose 只是交换轴
```

**玩家可见文案**

```text
Transpose 不会创造新参数，
也不会改变矩阵里的数值集合。

它只是交换矩阵的两个轴，
让原本的 [O,C] 变成计算需要的 [C,O]。
```

**画面表现**

一块权重板翻转，格子位置重新排列，旁边显示：

```text
before: weight[o,c]
after:  weight_T[c,o]
```

**任务桥接**

```text
修复方向后，仍然要通过 Reference Checker 验证数值。
```


---

### 9.3 画布初始状态

```text
[Hidden Tensor]
 float32[B,T,C]
      │
      ▼
[MatMul Gate] ◄──── [Stored Weight Plate]
                     float32[O,C]
      │
      ▼
[Projected Tensor]
```

Gate 当前报错：

```text
Inner dimension mismatch:
activation last axis = C
weight first axis = O
```

### 9.4 玩家操作

玩家可以选择两种修复方式：

#### 方案 A：旋转权重板

直接在 Weight Plate 上点击 Rotate / Transpose，使其从：

```text
[O,C]
```

变成：

```text
[C,O]
```

#### 方案 B：插入 Transpose Switch

在权重路径上插入：

```text
Transpose Switch: swap axis 0 and axis 1
```

使数据流变成：

```text
Stored Weight[O,C]
      ↓ transpose
Compute Weight[C,O]
      ↓
MatMul Gate
```

### 9.5 可视化表现

权重板像一块实体板：

- `[O,C]` 时，O 轴朝向 Gate 的 inner dock，红灯；
- 转置后，C 轴对准 Gate 的 inner dock，蓝光吸附；
- O 轴转到输出方向，形成新的 output axis。

### 9.6 验证规则

```text
Check 1: stored_weight.shape in {[C,O], [O,C]}
Check 2: if stored_weight == [O,C], transpose_switch.enabled == true
Check 3: compute_weight.shape == [C,O]
Check 4: MatMul inner dimension matches C
Check 5: output.shape == [B,T,O]
Check 6: output values allclose reference using correct orientation
```

### 9.7 失败反馈

如果没有转置：

```text
Weight orientation error.

MatMul Gate expected weight[C,O],
but received weight[O,C].

The inner dimension must match activation C.
Insert Transpose Switch or rotate the Weight Plate.
```

如果玩家错误转置了原本已经是 `[C,O]` 的权重：

```text
Unnecessary transpose detected.

Original weight was already [C,O].
After transpose it became [O,C], causing inner dimension mismatch.
```

如果 shape 正确但数值不对：

```text
Numeric reference failed.

Shape contract passed, but output values do not match reference.
Check whether the weight plate was transposed in the correct direction.
```

### 9.8 阶段小结

```text
Stage Complete: Weight Orientation Trap

你已经知道：
MatMul 不只看数字大小，还要看轴方向。

计算方向需要 weight[C,O]。
如果权重以 [O,C] 存储，必须转置后再计算。
```

解锁：

```text
Linear Assembly Board
Reference Trace
```

---

## 10. 0-2F：Linear Module Assembly / 组装 Linear Projection

### 10.1 阶段目标

玩家将前面所有知识组合起来，搭建一个完整 Linear Projection 模块。

本阶段主公式：

```text
projected[B,T,O] = hidden[B,T,C] @ weight[C,O]
```

可选显示但不展开：

```text
+ bias[O]
```

Bias 的广播加法留到 0-4 Broadcast Add 重点学习。

---

### 10.2 Concept Capsule

标题：

```text
Linear Projection：把 C 投影到 O
```

文案：

```text
在 Transformer 中，Linear Projection 会反复出现。

它读取每个 token 的 C 维向量，
通过一块 weight，
生成新的 O 维向量。

本关只处理 MatMul。
Bias 的广播加法会在后续关卡学习。
```

图示：

```text
hidden[B,T,C]
      │
      ▼
MatMul Gate + weight[C,O]
      │
      ▼
projected[B,T,O]
```

本阶段任务：

```text
从零组装一个可通过 reference trace 的 Linear Projection。
```


#### 10.2.1 挑战前知识介绍卡片扩展

0-2F 是综合组装关。知识卡要少讲新概念，多强调“把前面的工具合成一个可验证模块”。

##### Card 0-2F-1：Linear 是 Transformer 的常用构件

**标题**

```text
Linear Projection 会反复出现
```

**玩家可见文案**

```text
Transformer 里到处都有 Linear Projection。

它会用于生成 Q/K/V，
用于 MLP，
也用于最后的 vocabulary logits。

本关先掌握最基础版本：
hidden[B,T,C] @ weight[C,O] → projected[B,T,O]
```

**画面表现**

简化预览：

```text
Attention Q/K/V Projection
MLP Expansion
LM Head
```

三者下面都标：

```text
uses MatMul / Linear
```

**任务桥接**

```text
现在从空白画布组装一个完整 Linear Projection。
```

---

##### Card 0-2F-2：本关暂时锁定 Bias

**标题**

```text
本关只做 MatMul，不做 Bias Add
```

**玩家可见文案**

```text
真实 Linear 常常包含 bias：

x @ W + b

但 bias[O] 需要 broadcast 规则。
这个内容会在 0-4 Broadcast Add 中学习。

本关只要求你完成 MatMul Projection。
```

**画面表现**

```text
MatMul Gate: unlocked
Bias Add Socket: locked / coming in 0-4
```

**任务桥接**

```text
如果看到 Bias 插槽，不要连接它。先把 MatMul 路径修好。
```

---

##### Card 0-2F-3：Shape 正确不等于数值正确

**标题**

```text
最后必须通过 Reference Check
```

**玩家可见文案**

```text
一个结构可能输出了正确 shape，
但数值仍然可能是错的。

常见原因是 weight 方向错、transpose 错，
或者接入了错误的输入。

所以完整模块必须通过 reference implementation 对比。
```

**画面表现**

```text
Your output[B,T,O]
Reference output[B,T,O]
max_abs_error → PASS / FAIL
```

**任务桥接**

```text
组装完成后，把输出连接到 Reference Checker。
```


---

### 10.3 画布初始状态

玩家拥有模块：

```text
Hidden Tensor Source
Weight Plate
Optional Transpose Switch
MatMul Gate
Output Contract Gate
Reference Checker
```

画布上是一块空白工作区，只有目标提示：

```text
Build Linear Projection:
hidden[B,T,C] → projected[B,T,O]
```

### 10.4 玩家操作

玩家需要：

1. 拖入 `Hidden Tensor Source`；
2. 拖入 `Weight Plate`；
3. 检查 weight orientation；
4. 必要时插入 `Transpose Switch`；
5. 拖入 `MatMul Gate`；
6. 连接 activation 和 weight；
7. 拖入 `Projected Tensor Output`；
8. 标注输出合同 `[B,T,O]`；
9. 连接到 `Reference Checker`；
10. 运行测试。

### 10.5 Bias 处理策略

为了防止 0-2 与 0-4 重叠，本关的 Bias 模块建议以锁定插槽形式出现：

```text
Bias Add Socket: locked
Coming in 0-4 Broadcast Add
```

但可以给一条轻提示：

```text
Real Linear often includes bias[O].
This level focuses on MatMul only.
```

### 10.6 验证规则

```text
Structural checks:
✓ hidden connected
✓ weight connected
✓ MatMul Gate connected
✓ output contract assigned

Shape checks:
✓ hidden == [B,T,C]
✓ compute_weight == [C,O]
✓ output == [B,T,O]

Semantic checks:
✓ B/T carried from input
✓ C consumed
✓ O generated

Numeric checks:
✓ max_abs_error < 1e-5
✓ no NaN
✓ no illegal broadcast
```

### 10.7 失败反馈

如果玩家忘记连接 Reference Checker：

```text
Output not verified.

A Linear module is not complete until its output is checked against reference values.
Connect projected[B,T,O] to Reference Checker.
```

如果输出 shape 正确但数值不对：

```text
Reference mismatch.

Your output shape is correct,
but values differ from reference.

Possible causes:
- Weight plate transposed incorrectly
- Wrong input tensor selected
- Output axes reordered after MatMul
```

如果玩家把 Bias 插槽强行连接：

```text
Bias Add is locked in this level.

Bias requires broadcast rules.
You will unlock it in 0-4 Broadcast Add.
```

### 10.8 阶段小结

```text
Stage Complete: Linear Module Assembly

你已经从零搭建了：

hidden[B,T,C] @ weight[C,O] → projected[B,T,O]

这就是 Transformer 中大量 Linear Projection 的基础。
```

解锁：

```text
MatMul Hidden Test Gauntlet
QKV Projection Preview
```

---

## 11. 0-2X：MatMul Gauntlet / 最终隐藏测试

### 11.1 阶段目标

玩家证明自己理解的是规则，而不是记住了某一个具体 shape。

测试不同：

```text
B/T/C/O 尺寸变化
weight orientation 变化
输出合同变化
数值 reference 变化
```

---

### 11.2 Concept Capsule

标题：

```text
MatMul Contract 必须能泛化
```

文案：

```text
你不能靠固定数字通关。

有时 C 很小，有时 O 很大。
有时 weight 已经是 [C,O]，有时是 [O,C]。

你需要始终判断：
哪个轴被消费，哪个轴被保留，哪个轴被生成。
```

本阶段任务：

```text
连续修复 3 个未知 MatMul Gate。
```


#### 11.2.1 挑战前知识介绍卡片扩展

0-2X 是最终考核。知识卡的作用是提醒玩家不要靠固定数字、固定方向或固定套路通关。

##### Card 0-2X-1：MatMul Contract 要能泛化

**标题**

```text
不要记数字，要读合同
```

**玩家可见文案**

```text
前面的例子常用 B=2、T=4、C=8、O=6。

但真实模型里这些数字会不断变化。
你不能靠位置或大小猜答案。
你要判断每个轴的角色。
```

**画面表现**

```text
Case A: hidden[2,4,8]  @ weight[8,6]
Case B: hidden[1,8,4]  @ stored_weight[12,4]
Case C: hidden[4,3,2]  @ weight[2,16]
```

**任务桥接**

```text
每个 case 都先 inspect，再连接，再验证。
```

---

##### Card 0-2X-2：三种隐藏陷阱

**标题**

```text
最终测试会故意打破直觉猜测
```

**玩家可见文案**

```text
隐藏测试会包含三类陷阱：

1. C 可能很小，O 可能很大。
2. Weight 可能以 [O,C] 存储。
3. Shape 看似正确，但数值可能失败。
```

**画面表现**

三个 warning 标签：

```text
Size Trap
Orientation Trap
Numeric Trap
```

**任务桥接**

```text
不要急着接线。先确认 activation、weight orientation 和 expected output。
```

---

##### Card 0-2X-3：推荐调试顺序

**标题**

```text
MatMul Debug Checklist
```

**玩家可见文案**

```text
每个未知 MatMul Gate 都按这个顺序检查：

1. activation 最后一个轴是不是 C？
2. weight 计算方向是不是 [C,O]？
3. B/T carrier axes 是否保留？
4. 输出是否是 [B,T,O]？
5. reference 数值是否通过？
```

**画面表现**

右侧显示可勾选 Debug Checklist。

**任务桥接**

```text
准备好后，开始连续修复 3 个未知 MatMul Gate。
```


---

### 11.3 测试 Case 设计

#### Case A：标准方向

```text
hidden[B=2,T=4,C=8]
weight[C=8,O=6]
expected output[B=2,T=4,O=6]
```

#### Case B：权重存储方向反了

```text
hidden[B=1,T=8,C=4]
stored_weight[O=12,C=4]
requires transpose
expected output[B=1,T=8,O=12]
```

#### Case C：防止最大维度猜测

```text
hidden[B=4,T=3,C=2]
weight[C=2,O=16]
expected output[B=4,T=3,O=16]
```

这里故意让 `O` 很大、`C` 很小，让玩家不能靠“最大的是 C”判断。

#### Case D，可选困难版

```text
hidden[B=3,T=12,C=5]
stored_weight[O=7,C=5]
requires transpose
expected output[B=3,T=12,O=7]
```

---

### 11.4 玩家操作

每个 Case 的流程：

```text
Inspect hidden contract
Inspect weight orientation
Repair orientation if needed
Connect MatMul Gate
Assign output contract
Run reference checker
```

### 11.5 限制与评分

```text
Probe budget: 6 per case
Reference runs: 3 per case
Hints: optional, affects rank only
```

评分：

```text
Rank C：visible tests passed
Rank B：all hidden cases passed
Rank A：≤ 2 failed runs
Rank S：no hints + all cases first reference pass
```

### 11.6 失败反馈

如果玩家靠固定位置猜测：

```text
Axis-order assumption failed.

You assumed the last axis of weight is always the inner dimension.
But this stored weight is [O,C].

Inspect weight orientation before connecting.
```

如果玩家靠数字大小猜测：

```text
Size-based guess failed.

C can be smaller than O.
O can be larger than C.
MatMul is defined by axis roles, not by axis size.
```

如果 shape 全对但数值错：

```text
Shape passed, numeric reference failed.

This usually means the weight orientation is wrong,
even though the final output shape appears plausible.
```

### 11.7 最终通关小结

```text
0-2 Complete: MatMul Gate

你已经掌握：

Dot product 是 MatMul 的最小单元。
Linear 消费 C 轴并生成 O 轴。
B/T 是 carrier axes，会保留到输出。
权重方向决定能否正确计算。
Shape 正确不代表数值正确，最终需要 reference check。

下一关：
0-3 Transpose Trap
你将使用 MatMul 构建 attention scores：Q @ Kᵀ。
```

---

## 12. 可选支线挑战

### 12.1 Bonus A：Cell Inspector / 单格数值计算

#### 目标

让玩家理解输出张量中的一个具体值如何计算：

```text
projected[b,t,o] = Σ hidden[b,t,c] * weight[c,o]
```

#### 操作

系统给出目标：

```text
Find projected[0,2,3]
```

玩家需要在可视化张量里选择：

```text
hidden[0,2,:]
weight[:,3]
```

然后 Dot Cell 展示逐项乘法求和。

#### 验证

```text
selected_hidden_slice == [C]
selected_weight_column == [C]
output_scalar == reference
```

#### 小结

```text
你现在能从单个输出值追溯回输入向量和权重列。
```

---

### 12.2 Bonus B：PyTorch Weight Storage / 框架存储方向

#### 目标

解释真实工程中常见的：

```text
nn.Linear(in_features=C, out_features=O)
weight stored as [O,C]
compute as x @ weight.T
```

#### 知识卡文案

```text
在许多深度学习框架里，Linear 的 weight 以 [O,C] 存储。

但计算时，输入 x[...,C] 需要乘上 weight.T[C,O]。

所以：
存储方向不一定等于计算方向。
```

#### 操作

玩家把 `Stored Weight[O,C]` 通过 `Transpose Switch` 变成 `Compute Weight[C,O]`。

#### 验证

```text
storage_shape == [O,C]
compute_shape == [C,O]
output_shape == [B,T,O]
```

---

### 12.3 Bonus C：QKV Projection Preview

#### 目标

为 0-3 / Attention 做预告：Transformer 会用多个 Linear Projection 从 hidden 生成 Q/K/V。

#### 画布

```text
hidden[B,T,C]
   ├─ Wq[C,D] → Q[B,T,D]
   ├─ Wk[C,D] → K[B,T,D]
   └─ Wv[C,D] → V[B,T,D]
```

#### 文案

```text
Attention 不是直接用 hidden 计算。
它会先通过 Linear Projection 得到 Q、K、V。

下一关，你会学习为什么 Attention 需要 Q @ Kᵀ。
```

#### 验证

```text
Q.shape == [B,T,D]
K.shape == [B,T,D]
V.shape == [B,T,D]
all three projections use hidden C axis
```

---

### 12.4 Bonus D：FLOPs Meter / 计算量仪表

#### 目标

让硬核玩家理解矩阵乘法成本：

```text
approx multiply-adds = B * T * C * O
```

#### 操作

玩家调整：

```text
B / T / C / O
```

观察 FLOPs Meter 和 memory preview 变化。

#### 小结

```text
Linear Projection 的计算量会随 B、T、C、O 成比例增长。
这解释了为什么 context length 和 hidden size 会影响模型运行成本。
```

---

## 13. 画布与 UI 设计

### 13.1 主画布布局

默认布局从左到右：

```text
[Hidden Source] → [MatMul Gate] → [Projected Output] → [Reference Checker]
                         ▲
                         │
                 [Weight Plate / Transpose]
```

底部是 Repair Console：

```text
Objective
Tools
Axis Labels
Checklist
Run Tests
```

右侧是 Inspector：

```text
Selected node / tensor / axis / weight plate
shape
axis semantics
orientation
sample values
reference error
```

---

### 13.2 Repair Console 内容

不同阶段底部工具变化：

#### 0-2A

```text
Tools:
[Vector Probe] [Dot Trace]
Tags:
[C]
Checklist:
□ left vector connected
□ right vector connected
□ inner length matched
□ scalar output generated
```

#### 0-2B / 0-2C

```text
Tools:
[Weight Plate] [MatMul Gate] [Output Axis Tag O]
Checklist:
□ activation connected
□ weight connected
□ C matched
□ output O generated
```

#### 0-2E

```text
Tools:
[Rotate Weight] [Transpose Switch] [Orientation Probe]
Checklist:
□ stored orientation inspected
□ compute orientation = [C,O]
□ output values verified
```

---

### 13.3 Inspector 状态示例

#### 选中 Hidden Tensor

```text
Hidden Tensor
contract: hidden[B,T,C]
dtype: float32
shape: [2,4,8]

B: batch samples
T: token positions
C: input feature channels
```

#### 选中 Weight Plate

```text
Weight Plate
stored shape: [O,C]
compute shape: unresolved
orientation: needs transpose

Expected by MatMul Gate:
[C,O]
```

#### 选中 MatMul Gate

```text
MatMul Gate
left input:  [B,T,C]
right input: [C,O]
inner match: C = C
output:      [B,T,O]

Carrier axes: B,T
Consumed axis: C
Generated axis: O
```

#### 选中 Reference Checker

```text
Reference Checker
max abs error: 0.000003
mean abs error: 0.000001
tolerance: 1e-5
status: PASS
```

---

## 14. Autograder 设计

### 14.1 结构检查

```ts
structuralChecks = [
  "activation_connected",
  "weight_connected",
  "matmul_gate_connected",
  "output_connected",
  "reference_checker_connected"
]
```

### 14.2 Shape 检查

```ts
shapeChecks = [
  "activation.rank >= 1",
  "weight.rank == 2",
  "activation.lastAxis == C",
  "computeWeight.shape == [C,O]",
  "output.shape == activation.carrierAxes + [O]"
]
```

### 14.3 语义检查

```ts
semanticChecks = [
  "C is consumed",
  "O is generated",
  "B is preserved if present",
  "T is preserved if present",
  "carrier axis order is preserved"
]
```

### 14.4 数值检查

```ts
numericChecks = [
  "output = matmul(activation, computeWeight)",
  "max_abs_error < 1e-5",
  "no_nan",
  "no_inf"
]
```

Reference 伪代码：

```ts
function referenceLinear(hidden: Tensor, weight: Tensor): Tensor {
  // educational compute convention
  // hidden: [B,T,C]
  // weight: [C,O]
  // output: [B,T,O]
  for b in B:
    for t in T:
      for o in O:
        out[b,t,o] = 0
        for c in C:
          out[b,t,o] += hidden[b,t,c] * weight[c,o]
  return out
}
```

带 storage orientation 的版本：

```ts
function resolveComputeWeight(storedWeight, orientation) {
  if orientation == "C,O": return storedWeight
  if orientation == "O,C": return transpose(storedWeight)
}
```

---

## 15. 关卡状态机

```ts
type MatMulGateState =
  | "intro"
  | "stage_dot_cell_intro"
  | "stage_dot_cell_repair"
  | "stage_dot_cell_test"
  | "stage_token_projection_intro"
  | "stage_token_projection_repair"
  | "stage_token_projection_test"
  | "stage_sequence_projection_intro"
  | "stage_sequence_projection_repair"
  | "stage_sequence_projection_test"
  | "stage_batch_projection_intro"
  | "stage_batch_projection_repair"
  | "stage_batch_projection_test"
  | "stage_orientation_trap_intro"
  | "stage_orientation_trap_repair"
  | "stage_orientation_trap_test"
  | "stage_linear_assembly_intro"
  | "stage_linear_assembly_repair"
  | "stage_linear_assembly_test"
  | "stage_gauntlet"
  | "completed";
```

每个阶段必须有：

```ts
type StageDefinition = {
  id: string;
  title: string;
  conceptCapsule: ConceptCapsule;
  boardPreset: BoardPreset;
  allowedTools: ToolId[];
  objectives: Objective[];
  checks: CheckDefinition[];
  debrief: Debrief;
  unlocks: UnlockId[];
}
```

---

## 16. 关卡配置示例

```ts
const level_0_2 = {
  id: "0-2",
  title: "MatMul Gate",
  chapter: "Tensor Bootcamp",
  mode: "Build + Trace + Reference Check",

  prerequisites: [
    "0-1 completed",
    "hidden[B,T,C] contract unlocked",
    "Shape Inspector unlocked",
    "Axis Tags B/T/C unlocked"
  ],

  globalConcepts: [
    "dot product",
    "inner dimension",
    "carrier axes",
    "linear projection",
    "weight orientation",
    "reference numeric check"
  ],

  stages: [
    {
      id: "0-2A",
      title: "Dot Cell",
      targetFormula: "[C] · [C] -> scalar",
      tools: ["VectorProbe", "DotTrace"],
      checks: ["vector_length_match", "scalar_output", "numeric_reference"]
    },
    {
      id: "0-2B",
      title: "Token Projection",
      targetFormula: "[C] @ [C,O] -> [O]",
      tools: ["WeightPlate", "MatMulGate", "OutputAxisTagO"],
      checks: ["inner_C_match", "output_O", "numeric_reference"]
    },
    {
      id: "0-2C",
      title: "Sequence Projection",
      targetFormula: "[T,C] @ [C,O] -> [T,O]",
      tools: ["CarrierAxisVisualizer"],
      checks: ["T_preserved", "C_consumed", "O_generated"]
    },
    {
      id: "0-2D",
      title: "Batch Projection",
      targetFormula: "[B,T,C] @ [C,O] -> [B,T,O]",
      tools: ["HiddenTensorSource", "OutputContractGate"],
      checks: ["BT_preserved", "output_BTO", "numeric_reference"]
    },
    {
      id: "0-2E",
      title: "Weight Orientation Trap",
      targetFormula: "[B,T,C] @ transpose([O,C]) -> [B,T,O]",
      tools: ["TransposeSwitch", "OrientationProbe"],
      checks: ["compute_weight_CO", "orientation_correct", "numeric_reference"]
    },
    {
      id: "0-2F",
      title: "Linear Module Assembly",
      targetFormula: "Linear(hidden) -> projected[B,T,O]",
      tools: ["ReferenceChecker"],
      checks: ["full_pipeline_connected", "shape_reference", "numeric_reference"]
    }
  ],

  hiddenTests: [
    { id: "case_A", hidden: [2,4,8], weight: [8,6], orientation: "C,O" },
    { id: "case_B", hidden: [1,8,4], weight: [12,4], orientation: "O,C" },
    { id: "case_C", hidden: [4,3,2], weight: [2,16], orientation: "C,O" },
    { id: "case_D", hidden: [3,12,5], weight: [7,5], orientation: "O,C", optional: true }
  ],

  rewards: [
    "MatMul Gate",
    "Weight Plate",
    "Transpose Switch",
    "Reference Checker",
    "Output Axis O",
    "Next Level: 0-3 Transpose Trap"
  ]
};
```

---

## 17. 失败信息库

### 17.1 Inner Dimension Mismatch

```text
MatMul failed: inner dimension mismatch.

Left input last axis: C = {leftC}
Right input first axis: {rightAxis} = {rightSize}

MatMul requires:
activation[..., C] @ weight[C, O]
```

### 17.2 Consumed Wrong Axis

```text
Projection failed.

You consumed {axisName}, but Linear should consume C.

C is the feature vector inside each token.
B and T should be preserved as carrier axes.
```

### 17.3 Output Contract Error

```text
Output contract failed.

Expected: [B,T,O]
Received: {receivedShape}

C should disappear after MatMul.
O should appear as the new feature axis.
```

### 17.4 Weight Orientation Error

```text
Weight orientation error.

MatMul Gate expected compute weight [C,O].
Received stored weight [O,C].

Insert Transpose Switch or rotate the Weight Plate.
```

### 17.5 Shape Passed But Numeric Failed

```text
Reference mismatch.

Your output shape is correct,
but the values do not match the reference implementation.

Likely causes:
- Wrong weight orientation
- Wrong output axis order
- Incorrect transpose
- Wrong input slice
```

### 17.6 Size-Based Guess Failed

```text
Size-based guess failed.

C is not always the largest dimension.
O is not always smaller than C.

Use axis roles and MatMul contract, not dimension size.
```

---

## 18. 数值与示例数据建议

为了便于玩家手算和验证，0-2A / 0-2B 前期不要使用太大的 C/O。

### 18.1 0-2A 推荐数据

```text
hidden[C=3] = [2, -1, 3]
weight[C=3] = [4, 5, -2]

output = 2*4 + (-1)*5 + 3*(-2)
       = 8 - 5 - 6
       = -3
```

### 18.2 0-2B 推荐数据

```text
hidden[C=3] = [2, -1, 3]
weight[C,O] = [
  [ 4,  1],
  [ 5, -2],
  [-2,  3]
]

output[O=2] = [
  2*4 + (-1)*5 + 3*(-2),
  2*1 + (-1)*(-2) + 3*3
]
= [-3, 13]
```

前两关可以用整数，后续再用 float32，降低理解负担。

### 18.3 0-2D 推荐默认 shape

```text
B = 2
T = 4
C = 8
O = 6

hidden: float32[2,4,8]
weight: float32[8,6]
output: float32[2,4,6]
```

---

## 19. 与后续关卡的连接

### 19.1 连接到 0-3 Transpose Trap

0-2 结尾的 QKV Preview 提醒玩家：

```text
hidden[B,T,C]
   ├─ Linear → Q[B,T,D]
   ├─ Linear → K[B,T,D]
   └─ Linear → V[B,T,D]
```

0-3 将继续学习：

```text
Q[B,H,T,D] @ Kᵀ[B,H,D,T] → scores[B,H,T,T]
```

也就是：

```text
0-2 学的是 Linear MatMul。
0-3 学的是 Attention Score MatMul。
```

### 19.2 连接到 0-4 Broadcast Add

0-2 中 Bias Add 只显示锁定插槽：

```text
bias[O] + projected[B,T,O]
```

但暂不让玩家操作。0-4 再正式学习：

```text
projected[B,T,O] + bias[O] → projected[B,T,O]
```

---

## 20. MVP 实现建议

如果开发资源有限，0-2 的最小可玩版本可以只做 4 个阶段：

```text
1. Dot Cell
2. Token Projection [C] @ [C,O] → [O]
3. Batch Projection [B,T,C] @ [C,O] → [B,T,O]
4. Weight Orientation Trap [O,C] → transpose → [C,O]
```

必须实现的系统：

```text
- MatMul Gate 节点
- Weight Plate 可视化
- Inner dimension shape checker
- Output contract checker
- Transpose Switch
- Reference numeric checker
- Failure message system
```

可以延后实现的系统：

```text
- FLOPs Meter
- QKV Preview
- PyTorch Storage Bonus
- 手算 Cell Inspector 的完整交互
- 多 case gauntlet 的评分系统
```

---

## 21. 开发任务拆分

### 21.1 玩法逻辑

```text
[ ] 实现 MatMul shape inference
[ ] 实现 carrier axis propagation
[ ] 实现 consumed/generated axis 标记
[ ] 实现 weight orientation 状态
[ ] 实现 transpose switch
[ ] 实现 reference matmul 数值计算
[ ] 实现 allclose 检查
[ ] 实现 hidden test case runner
```

### 21.2 UI / 可视化

```text
[ ] Feature Vector Strip
[ ] Weight Plate 2D/2.5D 组件
[ ] MatMul Gate 节点
[ ] Inner Dimension Dock 动画
[ ] Output Tensor Cabinet
[ ] Orientation Probe 面板
[ ] Reference Checker 面板
[ ] 失败高亮与红线反馈
```

### 21.3 内容系统

```text
[ ] Concept Capsule 配置
[ ] Debrief 配置
[ ] Failure Lesson 配置
[ ] Stage Objective 配置
[ ] Hidden Test 配置
[ ] Reward Unlock 配置
```

### 21.4 QA 测试点

```text
[ ] 正确 [C,O] 权重可以通过
[ ] 错误 [O,C] 不转置无法通过
[ ] [O,C] 转置后可以通过
[ ] 输出 [B,T,O] 轴顺序错误会失败
[ ] shape 正确但数值错误会失败
[ ] B/T/C/O 不同尺寸下仍然能通过
[ ] 玩家不能通过固定数字位置绕过规则
```

---

## 22. 最终体验目标

0-2 结束时，玩家应该有这样的直觉：

```text
看到 hidden[B,T,C]，我知道 Linear 会处理每个 token 的 C 维向量。
看到 weight[C,O]，我知道 C 是输入通道，O 是输出通道。
看到 hidden[B,T,C] @ weight[C,O]，我知道输出是 [B,T,O]。
看到 weight[O,C]，我知道它可能需要 transpose。
看到 shape 通过但数值失败，我知道要检查方向和 reference。
```

这就是后续学习 Attention 的基础。因为 Attention 里的 Q/K/V、QKᵀ、head split，本质上都依赖玩家已经掌握：

```text
MatMul 是轴合同驱动的计算。
```

---

# 附录 A：本关核心术语表

| 术语 | 玩家理解版本 | 工程表达 |
|---|---|---|
| Dot Product | 两条同长度向量逐项相乘求和 | `[C] · [C] → scalar` |
| MatMul | 批量做很多点积 | `[..., C] @ [C, O] → [..., O]` |
| Inner Dimension | 两边必须相同、被消耗的维度 | `C` |
| Carrier Axes | 不参与点积、保留到输出的轴 | `B, T` |
| Output Axis | 权重矩阵生成的新维度 | `O` |
| Weight Plate | Linear 的参数矩阵 | `[C,O]` or stored `[O,C]` |
| Transpose | 交换矩阵两个轴 | `[O,C] → [C,O]` |
| Linear Projection | 用 MatMul 改变 feature dimension | `[B,T,C] → [B,T,O]` |
| Reference Check | 与标准实现数值对比 | allclose tolerance |

---

# 附录 B：本关一句话标语

```text
MatMul consumes C, preserves B/T, and creates O.
```

中文版：

```text
矩阵乘法消耗 C 轴，保留 B/T 轴，并生成 O 轴。
```


---

# 附录 C：Concept Card 内容配置规范

为了让 0-2 的知识介绍卡片能直接进入内容制作管线，建议每张卡使用统一结构：

```ts
type ConceptCard = {
  id: string;
  stageId: string;
  title: string;
  visibleText: string[];
  formula?: string[];
  visualSpec: string[];
  microInteraction?: string[];
  missionBridge: string;
  unlockTerms?: string[];
  moreInfo?: string[];
};
```

## C.1 文本长度规则

```text
标题：不超过 18 字
正文：2–4 行，每行尽量不超过 24 个汉字
公式：单独显示，不和正文混排
任务桥接：1 句话，直接引导进入操作
More Info：可展开，不默认显示
```

## C.2 出现时机

每个挑战阶段建议使用：

```text
Stage Enter：显示 Concept Card 1
首次操作失败：显示对应 Failure Lesson
首次操作成功：显示 Mini Debrief
Stage Complete：显示 Debrief Card
```

## C.3 可跳过规则

```text
第一次进入关卡：默认显示所有 Concept Cards
复玩同一阶段：只显示标题和任务桥接，可展开详情
S/A 评分挑战模式：默认隐藏知识卡，只保留 Debug Checklist
```

---

# 附录 D：可选支线挑战知识卡片

## D.1 Bonus A：Cell Inspector 知识卡

### Card Bonus-A-1：追踪一个具体输出值

**标题**

```text
projected[b,t,o] 从哪里来？
```

**玩家可见文案**

```text
输出张量里的一个具体数值，
来自一个 token 的 C 维向量，
和 weight 的第 o 列做点积。
```

**公式**

```text
projected[b,t,o] = Σ hidden[b,t,c] * weight[c,o]
```

**任务桥接**

```text
选择 hidden[b,t,:] 和 weight[:,o]，计算 projected[b,t,o]。
```

---

## D.2 Bonus B：PyTorch Weight Storage 知识卡

### Card Bonus-B-1：框架里的 Linear weight 常以 [O,C] 存储

**标题**

```text
真实框架的权重方向
```

**玩家可见文案**

```text
很多框架中，Linear 的 weight 以 [out_features, in_features] 存储。
也就是 [O,C]。

计算时通常会用 weight.T，变成 [C,O]。
```

**公式**

```text
nn.Linear(C, O).weight.shape == [O,C]
y = x[...,C] @ weight.T[C,O]
```

**任务桥接**

```text
把 Stored Weight[O,C] 通过 Transpose Switch 转成 Compute Weight[C,O]。
```

---

## D.3 Bonus C：QKV Projection Preview 知识卡

### Card Bonus-C-1：Attention 前需要先生成 Q/K/V

**标题**

```text
Q/K/V 也是 Linear Projection
```

**玩家可见文案**

```text
Attention 不会直接使用 hidden。

它会先用三块不同的 weight，
把 hidden 投影成 Q、K、V。
```

**图示**

```text
hidden[B,T,C]
   ├─ Wq[C,D] → Q[B,T,D]
   ├─ Wk[C,D] → K[B,T,D]
   └─ Wv[C,D] → V[B,T,D]
```

**任务桥接**

```text
用三块 MatMul Gate 预览 Q/K/V 的生成。
```

---

## D.4 Bonus D：FLOPs Meter 知识卡

### Card Bonus-D-1：MatMul 的成本来自 B*T*C*O

**标题**

```text
矩阵乘法为什么贵？
```

**玩家可见文案**

```text
Linear 会对每个 batch、每个 token、每个 output channel，
都做一次 C 维点积。

所以计算量大致随 B、T、C、O 相乘增长。
```

**公式**

```text
approx multiply-adds = B * T * C * O
```

**任务桥接**

```text
调整 B/T/C/O，观察 FLOPs Meter 如何变化。
```
