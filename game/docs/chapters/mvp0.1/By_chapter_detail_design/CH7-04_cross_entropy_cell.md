# Chapter 7-4 Cross Entropy Cell: CrossEntropy

## 1. 组件真实用途

CrossEntropy 计算一个位置的 next-token loss：

```text
loss = -log_softmax(logits)[target_id]
```

它把 logits 和正确 target id 变成非负标量。

## 2. 前置组件

- `component.logits_board.v1`
- `component.target_shift.v1`
- `component.softmax_last_dim.v1`

## 3. 本关新增能力

- `LogSoftmaxGate`：沿 V 轴计算 log probability。
- `TargetGatherGate`：按 target id 取正确列。
- `NegateGate`：转成 loss。
- `LossCellProbe`：展示目标 id、logit、loss。
- `ReferenceChecker`：数值参考。

## 4. 具体案例

Visible case:

```text
logits for one position = [2.0, 1.0, 0.0]
target_id = 0
loss = -log(exp(2)/(exp(2)+exp(1)+exp(0)))
```

## 5. 初始错误图

画布给出 logits_row、target_id、loss_out、cell_probe、reference。缺少 logsoftmax/gather/negate。

## 6. 目标内部实现

```text
logits_row -> log_softmax
log_softmax + target_id -> target_gather
target_gather -> negate
negate -> loss_out
negate -> cell_probe
loss_out -> reference
```

## 7. 错误路径

- 对 logits 直接取负：没有 softmax 归一化。
- 取最大 logit 而不是 target id：预测对错被忽略。
- softmax 后再 log 不稳定：大数 hidden 可能溢出。
- target 越界没有报错。

## 8. 测试设计

- Visible：三类 logits。
- Hidden A：target 不是 argmax，loss 应较大。
- Hidden B：logits 加常数，loss 不变。
- Hidden C：target id 超过 V 必须失败。

## 9. 认证后接口

```text
component.cross_entropy.v1
inputs:
  logits: float32[V]
  target_id: int[]
output:
  loss: float32[]
```

## 10. 后续调用

LossReducer 会把每个位置的 loss 结合 mask 做平均。
