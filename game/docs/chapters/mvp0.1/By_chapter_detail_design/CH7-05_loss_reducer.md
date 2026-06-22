# Chapter 7-5 Loss Reducer: MaskedLossMean

## 1. 组件真实用途

LossReducer 把 `[B,T]` 的 token losses 按有效 target mask 求平均：

```text
loss = sum(losses * mask) / sum(mask)
```

padding 位置不能参与训练 loss。

## 2. 前置组件

- `component.cross_entropy.v1`
- `component.attention_mask.v1`

## 3. 本关新增能力

- `TargetMaskShift`：将 input mask 对齐到 target positions。
- `MaskMultiply`：无效位置 loss 置 0。
- `SumReduce`：求 masked loss 总和和 mask 总数。
- `SafeDivide`：防止除以 0。
- `LossMeanProbe`：显示分子/分母。

## 4. 具体案例

Visible case:

```text
losses = [[0.2, 0.5, 1.0]]
target_mask = [[1,1,0]]
mean = (0.2 + 0.5) / 2 = 0.35
```

## 5. 初始错误图

画布给出 losses、attention_mask、loss_out、mean_probe、reference。缺少 mask shift、multiply、sum、divide。

## 6. 目标内部实现

```text
attention_mask -> target_mask_shift
losses + target_mask -> mask_multiply
mask_multiply -> sum_loss
target_mask -> sum_count
sum_loss + sum_count -> safe_divide
safe_divide -> loss_out
safe_divide -> mean_probe
loss_out -> reference
```

## 7. 错误路径

- 对所有位置平均：padding loss 污染结果。
- mask 不右移：target 对齐错。
- 分母用 T 而不是有效数量：reference 失败。
- 全 mask 时产生 NaN：SafeDivide 失败。

## 8. 测试设计

- Visible：mask 掉最后一位。
- Hidden A：无 padding。
- Hidden B：只有一个有效 token。
- Hidden C：全 padding，返回 guarded loss 0 并 warning。

## 9. 认证后接口

```text
component.loss_mean.v1
inputs:
  losses: float32[B,T]
  attention_mask: bool[B,T+1]
output:
  loss: float32[]
```

## 10. 后续调用

BackwardTrace 从这个标量 loss 开始传播梯度。
