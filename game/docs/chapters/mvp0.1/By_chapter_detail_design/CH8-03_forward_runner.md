# Chapter 8-3 Forward Runner: ForwardPass

## 1. 组件真实用途

ForwardRunner 把 batch 输入送进 tiny model，得到 logits：

```text
input_ids[B,T] -> model -> logits[B,T,V]
```

它复用 tokenizer 后的模型组件，不重新实现内部层。

## 2. 前置组件

- `component.batch_builder.v1`
- `component.hidden_init.v1`
- `component.transformer_stack.v1`
- `component.final_layernorm.v1`
- `component.lm_head.v1`

## 3. 本关新增能力

- `ModelGraphCall`：调用已组装模型子图。
- `InputMaskRouter`：把 attention mask 与 ids 一起传入。
- `ForwardTraceProbe`：显示 embedding、stack、logits 三段。
- `ReferenceChecker`：端到端 logits reference。

## 4. 具体案例

Visible case:

```text
input_ids[B=1,T=3]
logits[B=1,T=3,V=13]
```

## 5. 初始错误图

画布给出 input_ids、model weights、mask、logits_out、trace、reference。缺少模型调用和 mask router。

## 6. 目标内部实现

```text
input_ids + mask -> input_mask_router
input_mask_router + weights -> model_graph_call
model_graph_call.logits -> logits_out
model_graph_call.trace -> forward_trace
logits_out -> reference
```

## 7. 错误路径

- 只运行 embedding，不经过 stack/lm head。
- 丢掉 attention mask：padding/future 行为错。
- 输出 last-token logits `[V]`：本关需要 `[B,T,V]`。
- 使用固定 reference logits：seed/weights 变体失败。

## 8. 测试设计

- Visible：B=1,T=3。
- Hidden A：B=2。
- Hidden B：padding mask 包含 false。
- Hidden C：不同 weights seed，logits 必须变化。

## 9. 认证后接口

```text
component.forward_pass.v1
inputs:
  input_ids: int[B,T]
  mask: bool[B,T]
  weights: TinyModelWeights
output:
  logits: float32[B,T,V]
```

## 10. 后续调用

BackwardTrace 从 forward logits 和 loss 开始建立梯度路径。
