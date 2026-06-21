# Chapter 3-2 Linear Module

## 1. 组件真实用途

Linear 是第一个由多个已认证组件组合出的完整层：

```text
Linear(hidden, weight, bias) = MatMul(hidden, weight) + bias
```

本关不是拖一个 Linear 节点，而是把 `component.matmul_gate.v1`、`BroadcastRail` 和 `AddGate` 组合成可认证的 `component.linear.v1`。

## 2. 前置组件

- `component.matmul_gate.v1`

本关不允许使用预制 Linear。MatMulGate 必须来自组件库，证明玩家之前构建的资产可以被复用。

## 3. 本关新增能力

- `component.matmul_gate.v1`：执行 `hidden[B,T,C] @ weight[C,O]`。
- `BroadcastRail`：把 `bias[O]` 扩展到 `matmul[B,T,O]` 的 B/T 位置。
- `AddGate`：把 matmul 输出和 broadcast bias 相加。
- `OutputContractGate` + `ReferenceChecker`：证明 shape 和数值。

## 4. 具体案例

Visible case 使用前面已经出现过的 hidden 和 weight：

```text
hidden[B=1,T=2,C=3]
weight[C=3,O=2]
bias[O=2] = [0.2, -0.1]
```

MatMul 结果：

```text
matmul[token0] = [-1.0, 1.4]
matmul[token1] = [ 2.55, 1.5]
```

加 bias 后：

```text
linear[token0] = [-0.8, 1.3]
linear[token1] = [ 2.75, 1.4]
```

输出合约：

```text
float32[B,T,O] = [1,2,2]
```

## 5. 初始错误图

画布给出：

- `hidden: InputTensor`
- `weight: WeightPlate`
- `bias: InputTensor`
- `linear_out: OutputContractGate`
- `reference: ReferenceChecker`

缺少 matmul、broadcast、add 三个内部实现节点。

## 6. 目标内部实现

```text
hidden.out -> matmul.left
weight.out -> matmul.right
matmul.out -> bias_broadcast.target
bias.out -> bias_broadcast.small
matmul.out -> linear_add.left
bias_broadcast.out -> linear_add.right
linear_add.out -> linear_out.x
linear_out.out -> reference.x
```

其中：

- `matmul.moduleId = component.matmul_gate.v1`
- `bias_broadcast.moduleId = BroadcastRail`
- `bias_broadcast.alignAxes = [O]`
- `linear_add.moduleId = AddGate`

## 7. 玩家操作

1. 拖入 `MatMulGate v1`。
2. 拖入 `BroadcastRail`。
3. 拖入 `AddGate`。
4. 先完成 matmul，再把 bias 按 O 轴 broadcast 到 matmul 输出形状。
5. 将 matmul 和 broadcast bias 输入 AddGate。
6. 将 AddGate 输出接入合约和 reference。
7. 检查当前任务，再提交认证。

## 8. 错误路径

- 只做 MatMul：shape 是 `[B,T,O]`，但数值少了 bias，allclose 失败。
- bias 沿 B 或 T 对齐：shape 可能可广播，但语义错，hidden 变体失败。
- 直接 `bias -> AddGate`：AddGate 要求两侧 shape 完全一致，未 broadcast 会 shape mismatch。
- 使用原生 `MatMulGate` 而不是 `component.matmul_gate.v1`：没有体现组件复用。
- 使用预制 Linear：目标组件不能出现在自己的构建关 palette。

## 9. 测试设计

当前任务：

- `linear_out` 必须是 `float32[B=1,T=2,O=2]`。
- 输出必须 allclose 到 `matmul(hidden, weight) + bias`。

Hidden / certification：

- hidden 使用 `[B=2,T=3,C=4]`。
- weight 使用 `[C=4,O=3]`。
- bias 使用 `[O=3]`。
- 输出必须是 `[B=2,T=3,O=3]`。
- 公开认证允许选择 B/T/C/O 和 seed，系统生成输入和 reference。

## 10. 认证后接口

```text
component.linear.v1
inputs:
  hidden: float32[B,T,C]
  weight: float32[C,O]
  bias: float32[O]
output:
  out: float32[B,T,O]
```

## 11. 后续调用

Q/K/V projection、MLP up/down projection、LM head 都是 Linear 的变体。Linear 文档必须强调“组件复用 + 内部图保留”，否则玩家会回到拖答案节点的旧体验。
