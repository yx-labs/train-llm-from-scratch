# Chapter 2-1 Embedding Table: EmbeddingTable[V,C]

## 1. 组件真实用途

EmbeddingTable 把 ParameterMatrix 解释成 token id 到向量的查找表：

```text
table[id] -> float32[C]
```

它本身还不执行 lookup，只保证第 0 维是 vocab id，第二维是 hidden channel。

## 2. 前置组件

- `component.parameter_matrix.v1`
- `component.vocab_table.v1`

VocabTable 定义 id 空间，ParameterMatrix 提供每个 id 的向量。

## 3. 本关新增能力

- `VocabAxisContract`：检查矩阵第一轴 V 与 vocab id 空间一致。
- `ChannelAxisContract`：检查第二轴 C。
- `EmbeddingTableTag`：把 parameter role 标为 token embedding。
- `RowPreviewProbe`：显示某个 id 对应的向量。
- `ReferenceChecker`：检查表身份和行对齐。

## 4. 具体案例

Visible case:

```text
vocab ids = {0,1,4,7,12}
embedding weight shape = [V=13,C=3]
```

因为最大 id 是 12，所以 V 至少要覆盖 row 12。稀疏词表允许中间 id 不被使用。

## 5. 初始错误图

画布给出：

- `param: ParameterMatrix[V,C]`
- `vocab: VocabTable`
- `table_out: EmbeddingTableContract`
- `row_probe: RowPreviewProbe`
- `reference: ReferenceChecker`

缺少 vocab/channel axis contract 和 table tag。

## 6. 目标内部实现

```text
param.out -> vocab_axis.x
vocab.out -> vocab_axis.vocab
vocab_axis.out -> channel_axis.x
channel_axis.out -> table_tag.x
table_tag.out -> table_out.x
table_tag.out -> row_probe.table
table_out.out -> reference.x
```

## 7. 玩家操作

1. 查看 vocab 最大 id 和参数矩阵 V。
2. 拖入 `VocabAxisContract`，连接 vocab 与 param。
3. 拖入 `ChannelAxisContract`。
4. 用 `EmbeddingTableTag` 标记 token embedding。
5. 连接 RowPreviewProbe 和 reference。
6. 检查当前任务并提交认证。

## 8. 错误路径

- V 小于最大 vocab id + 1：lookup row 越界。
- 把矩阵转成 `[C,V]`：shape 可能相近，row 语义错。
- 忽略 vocab：无法证明 id 与 row 对齐。
- 用 frozen 参数：训练阶段无法更新 embedding。
- row probe 不接：玩家无法看到 id->row 关系。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 VocabAxisContract 和 EmbeddingTableTag。
- V 必须覆盖 vocab 最大 id。
- axes 为 `[V,C]`。
- RowPreviewProbe 对 id 12 显示 row 12。

## 10. Hidden / Mutation 测试

Hidden case A：最大 id 增大。

```text
vocab contains id 20
V must be >= 21
```

Hidden case B：转置参数。

```text
param axes = [C,V]
```

必须失败。

Hidden case C：不同 C。

认证只要求 C 一致，不要求固定为 3。

## 11. 认证后接口

```text
component.embedding_table.v1
inputs:
  param: parameter_matrix[V,C]
  vocab: vocab_table
output:
  table: embedding_table[V,C]
```

## 12. 后续调用

EmbeddingLookup 会根据 token ids 从这张表 gather rows。EmbeddingTable 必须先保证 row 语义，否则 lookup 数值会系统性错位。
