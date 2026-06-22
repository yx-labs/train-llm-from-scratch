# Chapter 3-1 Bias Add: BiasAdd

## 1. 组件真实用途

BiasAdd 把一个 output-channel bias 加到每个 batch、每个 token 的 LinearCore 输出上：

```text
out[b,t,o] = core[b,t,o] + bias[o]
```

本关核心不是“加法”，而是让玩家理解 bias 只沿 O 轴对齐，B/T 是广播出来的。

## 2. 前置组件

- `component.linear_core.v1`
- `component.add_scalar.v1`

本关不允许使用预制 `BiasAdd` 或完整 `Linear`。

## 3. 本关新增能力

- `BiasVector`：提供 `float32[O]` 参数。
- `BroadcastRail`：把 `[O]` 扩展到 `[B,T,O]`。
- `AddGate`：要求两侧 shape/axes 一致后逐元素相加。
- `BroadcastProbe`：显示 bias 值如何复制到每个 token。
- `ReferenceChecker`：检查 `core + bias`。

## 4. 具体案例

Visible case:

```text
core[B=1,T=2,O=2] =
  [[[-1.0, 1.4],
    [ 2.5, 1.5]]]

bias[O=2] = [0.2, -0.1]
```

期望：

```text
out =
  [[[-0.8, 1.3],
    [ 2.7, 1.4]]]
```

## 5. 初始错误图

画布给出：

- `core: LinearCoreOutput[B,T,O]`
- `bias: BiasVector[O]`
- `bias_out: OutputContractGate`
- `broadcast_probe: BroadcastProbe`
- `reference: ReferenceChecker`

缺少 broadcast 和 add。直接把 bias 接到 AddGate 或合约都会失败。

## 6. 目标内部实现

```text
core.out -> broadcast.target
bias.out -> broadcast.small
broadcast.out -> add.right
core.out -> add.left
add.out -> bias_out.x
broadcast.out -> broadcast_probe.x
bias_out.out -> reference.x
```

其中：

- `broadcast.alignAxes = [O]`
- `bias_out.expectedAxes = [B,T,O]`

## 7. 玩家操作

1. 拖入 `BroadcastRail`。
2. 设置 bias 对齐轴为 `O`。
3. 将 core 作为 target，bias 作为 small。
4. 拖入 `AddGate`，把 core 和 broadcast bias 相加。
5. 连接输出合约、probe 和 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- bias 直接进 AddGate：shape `[O]` 与 `[B,T,O]` 不一致。
- broadcast 对齐 T：当 `T == O` 时可能迷惑，hidden 变体失败。
- 对 bias 做 sum/reduce：数值错，ReferenceChecker 失败。
- 使用完整 Linear：shortcut，结构断言失败。
- 只连合约不连 reference：shape-only 不能认证。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 `BroadcastRail` 和 `AddGate`。
- `BroadcastRail.alignAxes = [O]`。
- 输出 axes 为 `[B,T,O]`。
- 输出 allclose 到 `core + bias[O]`。
- BroadcastProbe 展示同一个 bias 在不同 token 位置重复出现。

## 10. Hidden / Mutation 测试

Hidden case A：`T == O`。

```text
core: [B=1,T=3,O=3]
bias: [O=3]
```

这个 case 防止用长度猜 axis。

Hidden case B：batch 变体。

```text
core: [B=2,T=4,O=2]
bias: [O=2]
```

Hidden case C：错误 dtype。

```text
bias: int[O]
```

认证必须失败，bias 参数必须是 float32。

## 11. 认证后接口

```text
component.bias_add.v1
inputs:
  core: float32[B,T,O]
  bias: float32[O]
output:
  out: float32[B,T,O]
```

## 12. 后续调用

Linear Module 会把 LinearCore 和 BiasAdd 合并成一个更高级组件。MLP、Attention projection、LM head 都依赖这个广播语义。
