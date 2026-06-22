# Chapter 9-2 Next Logits: ForwardLastToken

## 1. 组件真实用途

NextLogits 对当前上下文运行模型，并取最后一个有效位置的 logits：

```text
logits_all[B,T,V] -> next_logits[V]
```

## 2. 前置组件

- `component.context_crop.v1`
- `component.forward_pass.v1`
- `component.logits_board.v1`

## 3. 本关新增能力

- `ForwardPassCall`：运行模型。
- `LastValidIndexGate`：从 mask 找最后有效位置。
- `GatherLastLogits`：取该位置的 V 维向量。
- `NextLogitProbe`：显示 top few logits。
- `ReferenceChecker`：检查取值。

## 4. 具体案例

Visible case:

```text
context_ids = [[12,4,7,0]]
mask = [[1,1,1,0]]
last_valid = 2
next_logits = logits[0,2,:]
```

## 5. 初始错误图

画布给出 context、mask、model、next_logits_out、probe、reference。缺少 forward、last index、gather。

## 6. 目标内部实现

```text
context + mask + model -> forward_call
mask -> last_valid_index
forward_call.logits + last_valid_index -> gather_last_logits
gather_last_logits -> next_logits_out
gather_last_logits -> probe
next_logits_out -> reference
```

## 7. 错误路径

- 取最后槽位 T-1：padding prompt 失败。
- 取所有 logits `[B,T,V]`：Sampler 需要 `[V]`。
- 不传 mask：forward 行为错。
- 用 argmax id 代替 logits：Sampler 无法调整策略。

## 8. 测试设计

- Visible：last_valid=2。
- Hidden A：无 padding，last=T-1。
- Hidden B：B=1 但 T 变化。
- Hidden C：全 mask 必须失败。

## 9. 认证后接口

```text
component.forward_last_token.v1
inputs:
  context_ids: int[B,T]
  context_mask: bool[B,T]
  weights: TinyModelWeights
output:
  logits: float32[V]
```

## 10. 后续调用

Sampler 把 next logits 转成下一个 token id。
