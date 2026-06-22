# Chapter 4-4 Scale: ScaleBySqrtD

## 1. 组件真实用途

ScaleBySqrtD 把 QKScore 的分数除以 `sqrt(D)`：

```text
scaled = scores / sqrt(D)
```

它的作用是让 dot product 的幅度随 head width D 变化时保持稳定。玩家不需要背推导，但必须知道 scale factor 来自 D，不能硬编码 visible case 的数字。

## 2. 前置组件

- `component.qk_score.v1`
- `component.scalar_cell.v1`
- `component.add_scalar.v1`

本关复用 score board，并构造一个由 D 计算出来的标量 scale。

## 3. 本关新增能力

- `HeadWidthProbe`：读取 score 来源中的 D。
- `SqrtGate`：计算 `sqrt(D)`。
- `ReciprocalGate`：计算 `1 / sqrt(D)`。
- `BroadcastMultiply`：把 rank-0 scale 乘到 `[B,H,T,T]` 分数上。
- `ReferenceChecker`：检查 scale 数值。

`HeadWidthProbe` 是探针和参数来源。它解释为什么当前 case 的 factor 是 0.5，而不是任意常数。

## 4. 具体案例

Visible case:

```text
scores[B=1,H=1,T=3,T=3]
D = 4
scale = 1 / sqrt(4) = 0.5
```

示例单元：

```text
scores[0,0,1,2] = 6.0
scaled[0,0,1,2] = 3.0
```

## 5. 初始错误图

画布给出：

- `scores: QKScoreOutput[B,H,T,T]`
- `head_width: HeadWidthProbe`
- `scaled_out: ScoreContract`
- `reference: ReferenceChecker`

缺少 sqrt、reciprocal 和 broadcast multiply。

## 6. 目标内部实现

```text
head_width.out -> sqrt.x
sqrt.out -> reciprocal.x
scores.out -> scale_mul.tensor
reciprocal.out -> scale_mul.scalar
scale_mul.out -> scaled_out.x
scaled_out.out -> reference.x
```

其中：

- `sqrt.moduleId = SqrtGate`
- `reciprocal.moduleId = ReciprocalGate`
- `scale_mul.moduleId = BroadcastMultiply`
- `scaled_out.expectedAxes = [B,H,T,T]`

## 7. 玩家操作

1. 拖入 `SqrtGate` 和 `ReciprocalGate`。
2. 将 `HeadWidthProbe` 输出接入 sqrt，再接 reciprocal。
3. 拖入 `BroadcastMultiply`。
4. 将 scores 与 scalar scale 相乘。
5. 接入合约和 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- 硬编码 `0.5`：visible case 可能过，D=9 hidden 失败。
- 乘以 `sqrt(D)`：数值放大，allclose 失败。
- 除以 T 而不是 D：当 T=D 时才可能混过，变体失败。
- 只检查 shape：scale 不改变 shape，必须有 reference。
- 直接使用预制 ScaleBySqrtD：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 `SqrtGate`、`ReciprocalGate`、`BroadcastMultiply`。
- scale 来自 `HeadWidthProbe`。
- 输出 axes 保持 `[B,H,T,T]`。
- 输出 allclose 到 `scores / sqrt(D)`。

## 10. Hidden / Mutation 测试

Hidden case A：`D=9`。

```text
scale = 1/3
```

Hidden case B：`T == D`。

```text
T=4, D=4
```

这个 case 会配合另一个变体区分“拿 T 做 scale”和“拿 D 做 scale”。

Hidden case C：非平方 D。

```text
D=5
```

认证使用浮点容差检查，不要求玩家手算。

## 11. 认证后接口

```text
component.scale_by_sqrt_d.v1
inputs:
  scores: float32[B,H,T,T]
  d: int[]
output:
  scaled: float32[B,H,T,T]
```

## 12. 后续调用

CausalMask 和 Softmax 都接在 scaled scores 后面。如果 scale 被硬编码，模型在不同 head width 下会表现完全不稳定。
