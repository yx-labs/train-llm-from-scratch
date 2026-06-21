# Chapter 4-3 QK Score: QKScore

## 1. 组件真实用途

QKScore 是 attention score board 的核心组件。它把 query 和 key 的 token 表示转成 token-to-token 分数：

```text
scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:])
```

在图上，这不是 `Q @ K`，而是：

```text
Q @ transpose(K, -2, -1)
```

本关要让玩家理解：只交换 K 的最后两个轴，让 `D` 成为 MatMul 的内维，让 key token `T` 成为输出列。

## 2. 前置组件

- `component.linear.v1`

玩家已经知道 Linear / MatMulGate 如何消费内维。QKScore 复用已认证的 `component.matmul_gate.v1`，但本关禁止使用 `component.qk_score.v1` 本身。

## 3. 本关新增能力

- `TransposeSwitch`：只交换 K 的 `T` 与 `D`。
- `component.matmul_gate.v1`：复用已认证 MatMulGate，而不是重新造乘法。
- `ScoreBoard`：检查输出必须是 attention score board。
- `CellTrace`：展示某个 score cell 如何由 Q/K 行 dot 出来。
- `ReferenceChecker`：提供数值参考，防止 shape-only 通过。

`CellTrace` 是解释探针。它不是可复用组件的一部分，但它帮助玩家把一个 `[B,H,T,T]` 单元和具体 Q/K 行关联起来。

## 4. 具体案例

Visible case:

```text
Q: float32[B=1,H=1,T=3,D=2]
K: float32[B=1,H=1,T=3,D=2]

K^T: float32[B=1,H=1,D=2,T=3]
scores = Q @ K^T
scores: float32[B=1,H=1,T=3,T=3]
```

一个具体单元的含义：

```text
scores[0,0,1,2] = dot(Q[0,0,1,:], K[0,0,2,:])
```

## 5. 初始错误图

画布只给出：

- `q: InputTensor`
- `k: InputTensor`

没有 transpose、没有 matmul、没有 score board、没有 reference。玩家必须从组件库拖入实现节点和探针节点。

## 6. 目标内部实现

目标子图：

```text
q.out -> qk_matmul.left
k.out -> k_transpose.x
k_transpose.out -> qk_matmul.right
qk_matmul.out -> score_board.scores
score_board.out -> reference.x
score_board.out -> cell_trace.scores
q.out -> cell_trace.q
k.out -> cell_trace.k
```

其中：

- `k_transpose.moduleId = TransposeSwitch`
- `k_transpose.axisA = -2`
- `k_transpose.axisB = -1`
- `qk_matmul.moduleId = component.matmul_gate.v1`
- `score_board.expectedAxes = [B,H,T,T]`

## 7. 玩家操作

1. 拖入 `TransposeSwitch`，设置默认 `swap(-2,-1)`。
2. 拖入已认证的 `MatMulGate v1`。
3. 将 `q` 接到 matmul left。
4. 将 `k` 先接入 transpose，再把 transpose 输出接到 matmul right。
5. 将 matmul 输出接到 `ScoreBoard`。
6. 接入 `ReferenceChecker` 和 `CellTrace`。
7. 运行“检查当前任务”，再运行“提交认证”。

## 8. 错误路径

- `k.out -> qk_matmul.right`：没有 K transpose，结构断言失败；在某些 T/D 下还会 shape mismatch。
- 转置 Q：输出语义错误，hidden allclose 失败。
- reverse 全部轴：B/H carrier axes 被破坏，ScoreBoard axis 失败。
- 只接 ScoreBoard 不接 Reference：形状可能过，但数值证明缺失。
- 不接 CellTrace：玩家失去单元级解释，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 `k_transpose: TransposeSwitch`。
- 必须使用 `qk_matmul: component.matmul_gate.v1`。
- 必须经过 `ScoreBoard`、`ReferenceChecker` 和 `CellTrace`。
- `score_board` 输出必须是 `float32[B,H,T,T]`。
- 输出必须 allclose 到 `referenceQKScores(Q, K)`。

## 10. Hidden / Mutation 测试

Hidden case A：`T == D`。

```text
Q/K: [B=1,H=1,T=3,D=3]
```

这个 case 用来防止“形状看起来能乘但语义错”的旁路。

Hidden case B：B/H carrier axes 变体。

```text
Q/K: [B=2,H=2,T=4,D=3]
```

这个 case 检查 transpose 只作用在最后两轴，不能移动 B/H。

公开认证让玩家选择 `B/H/T/D` 和 seed。系统生成 Q/K/reference，不要求玩家手写高维 tensor。

## 11. 认证后接口

认证后得到：

```text
component.qk_score.v1
inputs:
  q: float32[B,H,T,D]
  k: float32[B,H,T,D]
output:
  scores: float32[B,H,T,T]
```

## 12. 后续调用

Scale、CausalMask、Softmax 和 WeightedSum 都依赖这个 score board。如果 QKScore 只做 shape 合约，后续 attention 会在数值上悄悄错；因此本关必须是 behavior + contract + trace 三层证明。
