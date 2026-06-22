# Chapter 8-4 Backward Trace: BackwardPass

## 1. 组件真实用途

BackwardTrace 从标量 loss 反向追踪到每个 trainable parameter 的 gradient：

```text
loss -> dParam[P]
```

MVP0.1 不要求玩家手推全部微分，但要让图中每个可训练参数都有梯度路径。

## 2. 前置组件

- `component.forward_pass.v1`
- `component.loss_mean.v1`
- `component.parameter_matrix.v1`

## 3. 本关新增能力

- `GradSeed`：给 loss 放置 dLoss=1。
- `BackpropTraceGate`：沿 forward graph 反向传播。
- `TrainableParamCollector`：收集 trainable 参数。
- `GradientContract`：检查每个参数梯度 shape 匹配。
- `GradientProbe`：显示某个参数的梯度 checksum。

## 4. 具体案例

Visible case:

```text
loss: float32[]
params:
  tok_embed.weight[V,C]
  lm_head.weight[C,V]
```

期望每个 trainable 参数都有同形 gradient。

## 5. 初始错误图

画布给出 loss、forward_trace、params、grad_out、probe、reference。缺少 GradSeed、collector、backprop。

## 6. 目标内部实现

```text
loss -> grad_seed
params -> trainable_collector
grad_seed + forward_trace + trainable_collector -> backprop_trace
backprop_trace -> gradient_contract
gradient_contract -> grad_out
gradient_contract -> gradient_probe
grad_out -> reference
```

## 7. 错误路径

- 只给 loss 一个梯度，不连 params：没有可更新对象。
- frozen 参数也产出 gradient：collector 错。
- gradient shape 与 parameter 不同：contract 失败。
- 梯度全 0：visible reference/finite check 失败。

## 8. 测试设计

- Visible：两个参数梯度 shape 检查。
- Hidden A：包含 frozen 参数，必须无 gradient。
- Hidden B：额外 bias 参数。
- Hidden C：loss 为 NaN 时必须失败。

## 9. 认证后接口

```text
component.backward_pass.v1
inputs:
  loss: float32[]
  forward_trace: ForwardTrace
  params: Parameter[]
output:
  grads: Gradient[P]
```

## 10. 后续调用

Optimizer 使用 gradients 更新参数。
