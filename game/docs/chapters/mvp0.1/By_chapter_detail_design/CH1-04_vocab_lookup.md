# Chapter 1-4 Vocab Lookup

## 1. 组件真实用途

VocabLookup 把 `token_piece[T]` 映射成 `int[B,T]` token IDs。它是文本进入模型前的类型边界：字符串片段必须变成整数索引。

本关核心认知：

```text
piece surface -> vocab row id
"tokenizers" -> 12
"are"        -> 9
"useful"     -> 15
"!"          -> 4
```

## 2. 前置组件

- `component.merge_forge.v1`
- `component.vocab_table.v1`

本关不允许使用 `component.vocab_lookup.v1`。玩家要搭出 table lookup、fallback 和 ID buffer。

## 3. 建议内部节点

- `PieceInput`：提供 `token_piece[T]`。
- `VocabTable`：提供 piece -> id 映射。
- `LookupGate`：逐个 piece 查表。
- `FallbackResolver`：处理 unknown piece。
- `IdBuffer`：把 id 排成 `int[B,T]`。
- `TypeContractGate`：检查输出。

## 4. 具体案例

Visible case:

```text
pieces = ["tokenizers", "are", "useful", "!"]

vocab:
  tokenizers -> 12
  are        -> 9
  useful     -> 15
  !          -> 4
  <unk>      -> 1

expected ids[B=1,T=4] = [[12, 9, 15, 4]]
```

## 5. 初始错误图

画布只给出 pieces source、vocab table、id contract、reference。没有 lookup 和 fallback。

如果玩家直接把 pieces 接到 int contract，dtype 失败；如果拖答案组件，违反构建规则。

## 6. 目标内部实现

```text
pieces.out -> lookup.pieces
vocab.out -> lookup.table
lookup.ids -> fallback.ids
lookup.oov -> fallback.oov
fallback.out -> id_buffer.ids
id_buffer.out -> ids_out.x
ids_out.out -> reference.x
```

## 7. 错误路径

- 输出 piece 字符串而不是 int：dtype fail。
- 对 unknown piece 直接报错：hidden OOV case blocked。
- fallback 全部映射到 `<unk>`：visible 中已知词的行为失败。
- 忽略顺序：ids shape 正确，但序列和 reference 不一致。

## 8. 测试设计

当前任务：

- 结构断言：必须经过 lookup 和 id buffer。
- dtype/shape：`int[B=1,T=4]`。
- 行为：ID 序列精确等于 `[[12,9,15,4]]`。

Hidden cases：

- 包含 unknown piece：`["we", "train", "glyph?"]`，unknown 必须映射 `<unk>`。
- vocab 行顺序变化：不能依赖数组 index，必须按 surface lookup。

## 9. 认证变体

公开认证提供三个 fixture：

- all known pieces
- one unknown piece
- punctuation-heavy pieces

系统生成 pieces/vocab/reference。玩家不手写完整 vocab，但必须保留 lookup + fallback 机制。

## 10. 认证后接口

```text
component.vocab_lookup.v1
inputs:
  pieces: token_piece[T]
  vocab: vocab_table[V]
output:
  ids: int[B,T]
```

## 11. 后续调用

TokenBuffer、Padder、EmbeddingLookup 都要求整数 token IDs。VocabLookup 是从文本符号世界进入 tensor 世界的边界。
