# Chapter 1-8 Tokenizer Component

## 1. 组件真实用途

TokenizerComponent 把一段或多段 raw text 变成模型可读的两个输出：

```text
ids:  int[B,T]
mask: mask[B,T]
```

它不是一个黑盒答案节点，而是由 Splitter、MergeForge、VocabLookup、Padder 和 AttentionMaskBuilder 组合出来的文本入口组件。

## 2. 前置组件

- `component.splitter.v1`
- `component.merge_forge.v1`
- `component.vocab_lookup.v1`
- `component.padder.v1`
- `component.attention_mask_builder.v1`

本关不允许使用 `component.tokenizer.v1`。

## 3. 建议内部节点

- `TextInput`
- `component.splitter.v1`
- `component.merge_forge.v1`
- `component.vocab_lookup.v1`
- `Padder`
- `AttentionMaskBuilder`
- `EmbeddingReadyProbe`

## 4. 具体案例

Visible case:

```text
texts:
  B0 = "we train llm."
  B1 = "tokenizers are useful!"

maxT = 8
special tokens = <bos>, <eos>, <pad>
```

期望：

- 每一行输出固定长度 T=8。
- 内容 tokens 后必须有 `<eos>`。
- padding 位置的 mask 为 0。
- 非 padding 位置的 mask 为 1。

## 5. 初始错误图

画布只给出 raw text 输入、两个输出合约和探针：

- `ids_out: TypeContractGate(int[B,T])`
- `mask_out: TypeContractGate(mask[B,T])`
- `embedding_probe: EmbeddingReadyProbe`

玩家必须构建完整 pipeline。

## 6. 目标内部实现

```text
text.out -> splitter.text
splitter.pieces -> merge.pieces
merge.pieces -> lookup.pieces
lookup.ids -> padder.ids
padder.ids -> ids_out.x
padder.ids -> mask_builder.ids
mask_builder.mask -> mask_out.x
ids_out.out -> embedding_probe.ids
```

## 7. 错误路径

- 只输出 pieces：EmbeddingReadyProbe 要求 int IDs。
- 不加 `<eos>`：短句边界不明确。
- 不 padding：batch 无法形成矩形 `[B,T]`。
- mask 全 1：padding 位置会被 attention 当成有效内容。
- 字符级切分导致 T 超预算：`token_budget` 失败。

## 8. 测试设计

当前任务：

- `ids` 是 `int[B=2,T=8]`。
- `mask` 是 `mask[B=2,T=8]`。
- `EmbeddingReadyProbe` 通过。
- `eos_preserved`、`mask_pad`、`token_budget` 同时通过。

Hidden cases：

- unknown glyph 需要 fallback。
- 长文本需要截断但仍保留 EOS。
- 短文本需要 right padding。

## 9. 认证变体

公开认证让玩家选择：

- tokenizer strategy：word / subword / char
- fallback：unk / char / none
- maxT：6 / 8 / 10

认证判断同一内部图能否在不同文本 fixture 下保持 IDs、mask、预算和 EOS 规则。

## 10. 认证后接口

```text
component.tokenizer.v1
input:
  text: raw_text[B]
outputs:
  ids: int[B,T]
  attention_mask: mask[B,T]
```

## 11. 后续调用

EmbeddingLookup 只接受 `int[B,T]`。TokenizerComponent 认证后，文本任务才真正进入模型张量世界。
