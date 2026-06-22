# Chapter 3-4 MLP Up Projection: Linear C to 4C

## 1. 组件真实用途

MLP Up Projection 把 hidden channel 从 C 扩展到 4C：

```text
up = Linear(hidden, W_up, b_up)
up: float32[B,T,4C]
```

这个扩展给激活函数提供更宽的中间空间。

## 2. 前置组件

- `component.linear.v1`
- `component.parameter_matrix.v1`

本关复用 Linear，但必须证明输出宽度来自 `4 * C`，不能硬编码 visible case 的 O。

## 3. 本关新增能力

- `WidthMultiplierProbe`：根据输入 C 计算目标 O=4C。
- `UpWeightContract`：检查 weight axes `[C,4C]`。
- `component.linear.v1`：执行投影。
- `UpProjectionContract`：检查输出 `[B,T,4C]`。
- `ReferenceChecker`：检查数值。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=2,C=2]
W_up[C=2,O=8]
b_up[O=8]
```

期望输出：

```text
up[B=1,T=2,4C=8]
```

## 5. 初始错误图

画布给出：

- `hidden: HiddenSource[B,T,C]`
- `w_up: WeightPlate[C,4C]`
- `b_up: BiasVector[4C]`
- `up_out: UpProjectionContract`
- `width_probe: WidthMultiplierProbe`
- `reference: ReferenceChecker`

缺少 Linear 和 weight contract。

## 6. 目标内部实现

```text
hidden.out -> width_probe.hidden
w_up.out -> weight_contract.weight
width_probe.out -> weight_contract.expected_width
hidden.out -> linear.hidden
weight_contract.out -> linear.weight
b_up.out -> linear.bias
linear.out -> up_out.x
up_out.out -> reference.x
```

## 7. 玩家操作

1. 拖入 `WidthMultiplierProbe`。
2. 拖入 `UpWeightContract`，验证 O=4C。
3. 拖入 `Linear v1`。
4. 接入 hidden、W_up、b_up。
5. 接入输出合约和 reference。
6. 检查当前任务并提交认证。

## 8. 错误路径

- O 硬编码为 8：C=3 hidden case 失败。
- 使用普通 Linear 但不验证 width：结构断言失败。
- weight 转置成 `[4C,C]`：Linear 内维错。
- 输出 `[B,T,C]`：没有完成 up projection。
- 使用预制 MLPUp：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须使用 `component.linear.v1`。
- 必须存在 WidthMultiplierProbe 和 UpWeightContract。
- 输出 axes 为 `[B,T,4C]`。
- 输出 allclose 到 Linear reference。

## 10. Hidden / Mutation 测试

Hidden case A：`C=3`。

```text
O must be 12
```

Hidden case B：`B=2,T=1`。

检查 B/T carrier axes 保留。

Hidden case C：错误 bias 长度。

```text
b_up[O=8] when C=3
```

必须失败。

## 11. 认证后接口

```text
component.mlp_up.v1
inputs:
  hidden: float32[B,T,C]
  weight: float32[C,4C]
  bias: float32[4C]
output:
  up: float32[B,T,4C]
```

## 12. 后续调用

Activation 会消费 up projection 输出。Down projection 之后再回到 C 维。
