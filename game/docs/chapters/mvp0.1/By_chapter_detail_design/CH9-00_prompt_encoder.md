# Chapter 9-0 Prompt Encoder: PromptEncode

## 1. 组件真实用途

PromptEncoder 把用户 prompt 变成模型输入 ids 和 mask：

```text
prompt -> TextInput -> Tokenizer -> input_ids[B,T], attention_mask[B,T]
```

它复用训练时构造好的 tokenizer，不重新定义词表。

## 2. 前置组件

- `component.text_input.v1`
- `component.tokenizer.v1`

## 3. 本关新增能力

- `PromptTextGate`：验证 prompt 是 raw_text。
- `TokenizerCall`：调用已认证 tokenizer。
- `PromptEncodeProbe`：显示 prompt、pieces、ids、mask。
- `ReferenceChecker`：检查 tokenizer 输出。

## 4. 具体案例

Visible case:

```text
prompt = "we train"
ids = tokenizer(prompt)
```

## 5. 初始错误图

画布给出 prompt、tokenizer、encoded_out、probe、reference。缺少 text gate 和 tokenizer call。

## 6. 目标内部实现

```text
prompt -> prompt_text_gate
prompt_text_gate -> tokenizer_call.text
tokenizer -> tokenizer_call.tokenizer
tokenizer_call.ids/mask -> encoded_out
tokenizer_call -> probe
encoded_out -> reference
```

## 7. 错误路径

- 手写 ids：prompt 变体失败。
- 使用训练 batch builder：生成时只有一个 prompt。
- 丢掉 mask：padding 位置不可见信息缺失。
- prompt 被 trim/lowercase：reference 失败。

## 8. 测试设计

- Visible：`"we train"`。
- Hidden A：未知词，必须走 `<unk>`。
- Hidden B：带前导空格 piece。
- Hidden C：空 prompt 产生 guarded warning。

## 9. 认证后接口

```text
component.prompt_encode.v1
inputs:
  prompt: raw_text[]
  tokenizer: component.tokenizer.v1
outputs:
  ids: int[B,T]
  mask: bool[B,T]
```

## 10. 后续调用

ContextCrop 会把 prompt ids 裁到模型上下文长度。
