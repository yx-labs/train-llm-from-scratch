# Chapter 9-5 Decode: TokenDecode

## 1. 组件真实用途

TokenDecode 使用 vocab 的反向映射把 token ids 拼回文本：

```text
ids -> pieces -> text
```

它必须保留 piece 中的空格和标点。

## 2. 前置组件

- `component.vocab_table.v1`
- `component.append_token.v1`

## 3. 本关新增能力

- `InverseVocabGate`：构造 id -> piece 映射。
- `IdToPieceMapGate`：逐 id 查 piece。
- `PieceJoinGate`：拼接 pieces。
- `DecodeProbe`：显示每个 id 的 piece。
- `ReferenceChecker`：检查文本。

## 4. 具体案例

Visible case:

```text
ids = [12,4,7]
12 -> "we"
4 -> " train"
7 -> " llm"
text = "we train llm"
```

## 5. 初始错误图

画布给出 ids、vocab、decode_out、probe、reference。缺少 inverse/map/join。

## 6. 目标内部实现

```text
vocab -> inverse_vocab
ids + inverse_vocab -> id_to_piece
id_to_piece -> piece_join
piece_join -> decode_out
id_to_piece -> decode_probe
decode_out -> reference
```

## 7. 错误路径

- 按 id 数字直接转字符串：输出 `"12 4 7"`。
- trim piece：前导空格丢失。
- unknown id 不处理：必须输出 `<unk>` 或失败策略。
- 跳过 inverse vocab：无法证明 id 语义。

## 8. 测试设计

- Visible：`we train llm`。
- Hidden A：包含 `<unk>`。
- Hidden B：piece 有标点。
- Hidden C：未知 id 按策略处理。

## 9. 认证后接口

```text
component.token_decode.v1
inputs:
  ids: int[N]
  vocab: vocab_table
output:
  text: raw_text[]
```

## 10. 后续调用

TinyChatLoop 使用 Decode 展示生成结果。
