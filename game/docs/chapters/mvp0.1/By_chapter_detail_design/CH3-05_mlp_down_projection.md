# Chapter 3-5 MLP Down Projection: Linear 4C to C

## 1. 组件真实用途

MLP Down Projection 把激活后的 4C 中间表示压回 C：

```text
down = Linear(activated, W_down, b_down)
down: float32[B,T,C]
```

它让 MLP 输出可以和原 hidden 做 residual add。

## 2. 前置组件

- `component.mlp_up.v1`
- `component.relu.v1`
- `component.linear.v1`

本关复用 Linear，但输入宽度是 4C，输出宽度是 C。

## 3. 本关新增能力

- `DownWeightContract`：检查 weight axes `[4C,C]`。
- `ChannelReturnProbe`：检查输出 channel 回到原 C。
- `component.linear.v1`：执行投影。
- `DownProjectionContract`：检查 `[B,T,C]`。
- `ReferenceChecker`：检查数值。

## 4. 具体案例

Visible case:

```text
activated[B=1,T=2,4C=8]
W_down[4C=8,C=2]
b_down[C=2]
```

期望输出：

```text
down[B=1,T=2,C=2]
```

## 5. 初始错误图

画布给出：

- `activated: ActivationOutput[B,T,4C]`
- `w_down: WeightPlate[4C,C]`
- `b_down: BiasVector[C]`
- `down_out: DownProjectionContract`
- `return_probe: ChannelReturnProbe`
- `reference: ReferenceChecker`

缺少 DownWeightContract 和 Linear。

## 6. 目标内部实现

```text
activated.out -> return_probe.input
activated.out -> weight_contract.activation
w_down.out -> weight_contract.weight
weight_contract.out -> linear.weight
activated.out -> linear.hidden
b_down.out -> linear.bias
linear.out -> down_out.x
linear.out -> return_probe.output
down_out.out -> reference.x
```

## 7. 玩家操作

1. 拖入 `DownWeightContract`。
2. 验证 weight 是 `[4C,C]`。
3. 拖入 `Linear v1`。
4. 接入 activated、W_down、b_down。
5. 接入 ChannelReturnProbe、合约和 reference。
6. 检查当前任务并提交认证。

## 8. 错误路径

- weight 用 `[C,4C]`：内维错。
- 输出仍是 4C：不能 residual add。
- 省略 bias：shape 对但数值错。
- 使用 up projection 权重：数值错。
- 使用预制 MLPDown：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须使用 `component.linear.v1`。
- 必须存在 DownWeightContract。
- 输出 axes 为 `[B,T,C]`。
- 输出 allclose 到 Linear reference。
- ChannelReturnProbe 显示 4C -> C。

## 10. Hidden / Mutation 测试

Hidden case A：`C=3`。

```text
activated width = 12
output width = 3
```

Hidden case B：`T=4`。

B/T carrier axes 必须保留。

Hidden case C：`4C == T`。

防止把 time axis 当 channel axis。

## 11. 认证后接口

```text
component.mlp_down.v1
inputs:
  activated: float32[B,T,4C]
  weight: float32[4C,C]
  bias: float32[C]
output:
  down: float32[B,T,C]
```

## 12. 后续调用

MLP Module 会把 up、activation、down 串起来。DownProjection 是 MLP 回到 residual stream 的出口。
