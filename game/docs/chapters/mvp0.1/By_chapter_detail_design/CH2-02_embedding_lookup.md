# Chapter 2-2 Embedding Lookup

## 1. 组件真实用途

EmbeddingLookup 把 token IDs 映射成 hidden vectors：

```text
ids[B,T] + table[V,C] -> hidden[B,T,C]
```

每个整数 id 都是 vocab table 的一行索引。输出不是计算出来的新数，而是从 table 中 gather 对应行。

## 2. 前置组件

- `component.tokenizer.v1`
- `component.embedding_table.v1`

本关不允许使用 `component.embedding_lookup.v1`。

## 3. 建议内部节点

- `TokenIdInput`
- `EmbeddingTable`
- `RowGather`
- `PadRowPolicy`
- `HiddenBuffer`
- `TypeContractGate`
- `ReferenceChecker`

## 4. 具体案例

Visible case:

```text
ids[B=1,T=4] = [[2, 5, 1, 0]]

table[V=6,C=3]:
  row0 <pad> = [0.0, 0.0, 0.0]
  row1 <unk> = [0.1, 0.1, 0.1]
  row2 we    = [0.2, 0.0, 0.5]
  row5 llm   = [0.7, 0.3, 0.2]

expected hidden:
  T0 -> table[2]
  T1 -> table[5]
  T2 -> table[1]
  T3 -> table[0]
```

输出合约：

```text
float32[B,T,C] = [1,4,3]
```

## 5. 初始错误图

画布给出 ids、table、hidden contract、reference。缺少 row gather 和 hidden buffer。

直接把 ids 接到 hidden contract 会 dtype 失败；直接输出 table 会 shape `[V,C]` 错。

## 6. 目标内部实现

```text
ids.out -> row_gather.ids
table.out -> row_gather.table
row_gather.rows -> hidden_buffer.rows
hidden_buffer.out -> hidden_out.x
hidden_out.out -> reference.x
```

## 7. 错误路径

- 把 id 当作数值特征直接转 float：shape 可能接近，但不等于 table rows。
- 忽略 batch 维：只输出 `[T,C]`。
- PAD id 没有拿到 zero row：mask 之后仍可能泄漏 padding 信息。
- id 越界不报错：hidden case 应 blocked 或 fallback 到 `<unk>`，由关卡策略明确。

## 8. 测试设计

当前任务：

- `hidden_out` 必须是 `float32[B,T,C]`。
- 行为断言：每个输出 row 必须等于对应 `table[id]`。
- allclose 到 reference。

Hidden cases：

- B=2 多行 batch。
- C 宽度变化。
- ids 中包含 PAD 和 `<unk>`。

## 9. 认证变体

公开认证选择：

- B/T 尺寸
- C 宽度
- seed
- 是否包含 PAD

系统生成 ids/table/reference。认证只改变输入，不改变 row gather 图。

## 10. 认证后接口

```text
component.embedding_lookup.v1
inputs:
  ids: int[B,T]
  table: float32[V,C]
output:
  hidden: float32[B,T,C]
```

## 11. 后续调用

PositionEmbedding 和 Linear 都会消费 `hidden[B,T,C]`。EmbeddingLookup 是 token IDs 到模型 hidden state 的边界。
