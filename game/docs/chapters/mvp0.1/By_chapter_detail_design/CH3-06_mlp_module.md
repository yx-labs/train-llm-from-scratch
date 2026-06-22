# Chapter 3-6 MLP Module: MLP

## 1. 组件真实用途

MLP Module 把 up projection、activation 和 down projection 组合成一个完整前馈层：

```text
up = MLPUp(hidden)
act = ReLU(up)
out = MLPDown(act)
```

它输入和输出都是 `[B,T,C]`，但内部经过 `[B,T,4C]` 的宽层。

## 2. 前置组件

- `component.mlp_up.v1`
- `component.relu.v1`
- `component.mlp_down.v1`

本关是组件编排关，不允许拖预制 MLP。

## 3. 本关新增能力

- `MLPStageContract`：检查 up/activation/down 三段顺序。
- `ActivationProbe`：展示非线性前后的变化。
- `ResidualReadyProbe`：确认输出回到 `[B,T,C]`。
- `ReferenceChecker`：端到端数值检查。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=2,C=2]
up width = 4C = 8
```

其中一个中间值：

```text
up[0,1,3] = -0.7
act[0,1,3] = 0.0
```

最终输出：

```text
mlp_out[B=1,T=2,C=2]
```

## 5. 初始错误图

画布给出：

- `hidden: HiddenSource[B,T,C]`
- `weights: MLPWeightPack`
- `mlp_out: MLPOutputContract`
- `activation_probe: ActivationProbe`
- `residual_probe: ResidualReadyProbe`
- `reference: ReferenceChecker`

缺少 Up、ReLU、Down 三个已认证组件。

## 6. 目标内部实现

```text
hidden.out -> up.hidden
weights.w_up -> up.weight
weights.b_up -> up.bias
up.out -> relu.x
relu.out -> down.activated
weights.w_down -> down.weight
weights.b_down -> down.bias
down.out -> mlp_out.x
up.out -> activation_probe.before
relu.out -> activation_probe.after
down.out -> residual_probe.x
mlp_out.out -> reference.x
```

## 7. 玩家操作

1. 拖入 `MLPUp v1`。
2. 拖入 `ReLU v1`。
3. 拖入 `MLPDown v1`。
4. 按 Up -> ReLU -> Down 顺序连接。
5. 接入 ActivationProbe 和 ResidualReadyProbe。
6. 检查当前任务并提交认证。

## 8. 错误路径

- 漏掉 ReLU：shape 完全一样，但数值不同。
- Down 直接接 hidden：跳过 up/activation，结构失败。
- Up 和 Down 权重接反：维度或 reference 失败。
- 输出停在 4C：不能进入 residual。
- 使用预制 MLP：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 Up、ReLU、Down 三段。
- 三段必须按顺序连通。
- 输出 axes 为 `[B,T,C]`。
- ActivationProbe 必须显示至少一个负值被截断。
- 输出 allclose 到 end-to-end MLP reference。

## 10. Hidden / Mutation 测试

Hidden case A：全正 up 激活。

ReLU 不改变值，但结构仍必须存在。

Hidden case B：`C=3`。

内部宽度必须是 12。

Hidden case C：无 bias 旁路。

系统设置非零 bias，漏接 bias 会 reference 失败。

## 11. 认证后接口

```text
component.mlp.v1
inputs:
  hidden: float32[B,T,C]
  weights: MLPWeightPack
output:
  out: float32[B,T,C]
```

## 12. 后续调用

Transformer block 会把 LayerNorm、Attention、Residual、MLP 串起来。MLP Module 是第二条主要子层。
