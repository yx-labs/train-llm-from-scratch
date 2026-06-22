# Chapter 3-0 Linear Core: LinearCore

## 1. 组件真实用途

LinearCore 是不带 bias 的线性层主体：

```text
core = hidden[B,T,C] @ weight[C,O]
```

它把 token hidden state 从 C 维投影到 O 维。本关要让玩家复用已认证的 `MatMulGate`，并理解 weight 的 C 轴必须和 hidden 的 C 轴对齐。

## 2. 前置组件

- `component.matmul_gate.v1`
- `component.axis_tensor.v1`

本关不允许使用 `LinearCore` 或 `Linear` 作为答案节点。

## 3. 本关新增能力

- `WeightPlate`：提供参数矩阵 `float32[C,O]`。
- `component.matmul_gate.v1`：执行最后一维 C 与第一维 C 的乘法。
- `ProjectionContract`：检查输出 axis 变为 `[B,T,O]`。
- `ReferenceChecker`：检查数值等于 matrix multiply。
- `ColumnTrace`：展示一个 output channel 是 hidden 与 weight 列的 dot product。

`ColumnTrace` 是探针，帮助玩家理解 O 不是新的 token 轴。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=2,C=3]
weight[C=3,O=2]

hidden[0,0,:] = [1.0, 2.0, -1.0]
weight[:,0] = [0.5, 0.0, 1.0]

core[0,0,0] = 1.0*0.5 + 2.0*0.0 + (-1.0)*1.0 = -0.5
```

输出合约：

```text
float32[B,T,O] = [1,2,2]
```

## 5. 初始错误图

画布给出：

- `hidden: InputTensor[B,T,C]`
- `weight: WeightPlate[C,O]`
- `core_out: ProjectionContract`
- `reference: ReferenceChecker`
- `column_trace: ColumnTrace`

缺少 MatMulGate。hidden 或 weight 直接接合约都会失败。

## 6. 目标内部实现

```text
hidden.out -> matmul.left
weight.out -> matmul.right
matmul.out -> core_out.x
core_out.out -> reference.x
matmul.out -> column_trace.core
hidden.out -> column_trace.hidden
weight.out -> column_trace.weight
```

其中：

- `matmul.moduleId = component.matmul_gate.v1`
- `core_out.expectedAxes = [B,T,O]`

## 7. 玩家操作

1. 拖入已认证 `MatMulGate v1`。
2. 将 hidden 接 left，weight 接 right。
3. 连接 ProjectionContract 和 ReferenceChecker。
4. 连接 ColumnTrace，查看某个 output channel 的 dot 过程。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- 交换 hidden 和 weight：可能形状不匹配，或输出 axes 错。
- 使用普通 MatrixMultiply 而不是认证 MatMulGate：结构断言失败。
- 把 O 当作 token T：shape 可能碰巧一样，hidden axis 变体失败。
- 只接 ProjectionContract：缺少数值证明。
- 拖入预制 Linear：目标组件 shortcut，被 palette 禁止。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 `component.matmul_gate.v1`。
- 输出 dtype 为 `float32`。
- 输出 axes 为 `[B,T,O]`。
- 输出 allclose 到 `matmul(hidden, weight)`。
- ColumnTrace 必须接入 hidden、weight 和 core。

## 10. Hidden / Mutation 测试

Hidden case A：`T == O`。

```text
hidden: [B=1,T=2,C=3]
weight: [C=3,O=2]
```

这个 case 防止玩家把 O 当 T。

Hidden case B：不同 C/O。

```text
hidden: [B=2,T=3,C=4]
weight: [C=4,O=5]
expected: [B=2,T=3,O=5]
```

Hidden case C：错误 weight axis。

```text
weight: [O,C]
```

认证必须失败，并提示 MatMul 内维 C 没有对齐。

## 11. 认证后接口

```text
component.linear_core.v1
inputs:
  hidden: float32[B,T,C]
  weight: float32[C,O]
output:
  core: float32[B,T,O]
```

## 12. 后续调用

BiasAdd 会接在 LinearCore 后面，Linear Module 会把二者封装起来。Q/K/V projection 和 MLP 的核心都从这个组件开始。
