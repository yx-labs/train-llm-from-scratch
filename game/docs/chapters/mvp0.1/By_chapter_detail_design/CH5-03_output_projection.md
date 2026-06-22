# Chapter 5-3 Output Projection: Attention Output Linear

## 1. 组件真实用途

OutputProjection 把多头 concat 的 context 再投影一次：

```text
out = Linear(context[B,T,C], W_o[C,C], b_o[C])
```

它让各 head 的信息在通道维重新混合，并回到 residual stream 的 C 维。

## 2. 前置组件

- `component.head_concat.v1`
- `component.linear.v1`

本关复用 Linear，但要求输入和输出都是 C。

## 3. 本关新增能力

- `SquareWeightContract`：检查 W_o 是 `[C,C]`。
- `component.linear.v1`：执行输出投影。
- `ResidualWidthProbe`：确认输出可与 residual 相加。
- `ReferenceChecker`：检查数值。

## 4. 具体案例

Visible case:

```text
context[B=1,T=2,C=6]
W_o[C=6,C=6]
b_o[C=6]
out[B=1,T=2,C=6]
```

## 5. 初始错误图

画布给出 context、W_o、b_o、output_out、residual_width_probe、reference。缺少 SquareWeightContract 和 Linear。

## 6. 目标内部实现

```text
w_o.out -> square_weight.weight
context.out -> square_weight.context
context.out -> linear.hidden
square_weight.out -> linear.weight
b_o.out -> linear.bias
linear.out -> output_out.x
linear.out -> residual_width_probe.x
output_out.out -> reference.x
```

## 7. 玩家操作

1. 拖入 `SquareWeightContract`。
2. 拖入 `Linear v1`。
3. 接入 context、W_o、b_o。
4. 连接 residual width probe 和 reference。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- 跳过 output projection：shape 对但数值错。
- W_o 不是 `[C,C]`：不能回到 residual width。
- 使用 MLP down projection：语义错，参数角色不对。
- 漏接 bias：reference 失败。
- 使用预制 OutputProjection：shortcut。

## 9. 测试设计

- Visible：C=6，输出 allclose 到 Linear reference。
- Hidden A：C=8。
- Hidden B：W_o 设成 identity，输出应等于 context+bias。
- Hidden C：非方阵 W_o 必须失败。

## 10. 认证后接口

```text
component.attention_output_projection.v1
inputs:
  context: float32[B,T,C]
  weight: float32[C,C]
  bias: float32[C]
output:
  out: float32[B,T,C]
```

## 11. 后续调用

MultiHeadAttention 会把 split、parallel heads、concat 和 output projection 串成完整子层核心。
