# Chapter 9-3 Sampler: SampleNextToken

## 1. 组件真实用途

Sampler 根据 logits、temperature、top-k 和随机 seed 选择下一个 token：

```text
prob = softmax(logits / temperature)
candidate = top_k(prob)
next_id = sample(candidate, seed)
```

## 2. 前置组件

- `component.forward_last_token.v1`
- `component.softmax_last_dim.v1`

## 3. 本关新增能力

- `TemperatureScaleGate`：按 temperature 缩放 logits。
- `TopKGate`：只保留 top-k 候选。
- `SeededCategoricalGate`：按 seed 采样。
- `SamplerProbe`：显示候选概率和选中 id。
- `ReferenceChecker`：检查确定性 seed 输出。

## 4. 具体案例

Visible case:

```text
logits = [1.0, 3.0, 2.0, -1.0]
temperature = 1.0
top_k = 2
seed = 42
```

候选只来自 id 1 和 id 2。

## 5. 初始错误图

画布给出 logits、temperature、top_k、seed、sample_out、probe、reference。缺少 scale/topk/sample。

## 6. 目标内部实现

```text
logits + temperature -> temperature_scale
temperature_scale + top_k -> top_k_gate
top_k_gate + seed -> seeded_categorical
seeded_categorical -> sample_out
seeded_categorical -> sampler_probe
sample_out -> reference
```

## 7. 错误路径

- 永远 argmax：temperature/seed 失效。
- top-k 后不重新归一化：概率错。
- temperature=0 不处理：必须使用 greedy guard。
- 输出概率而非 id：AppendToken 需要 int scalar。

## 8. 测试设计

- Visible：seed 固定输出。
- Hidden A：top_k=1 等价 greedy。
- Hidden B：temperature=0 guard。
- Hidden C：同 seed 可重复，不同 seed 可变化。

## 9. 认证后接口

```text
component.sample_next_token.v1
inputs:
  logits: float32[V]
  temperature: float32[]
  top_k: int[]
  seed: int[]
output:
  next_id: int[]
```

## 10. 后续调用

AppendToken 会把 next_id 加到生成 buffer 尾部。
