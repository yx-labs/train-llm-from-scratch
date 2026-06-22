# Chapter 2-4 Position Embedding: PositionEmbedding[T,C]

## 1. 组件真实用途

PositionEmbedding 根据 position ids 从位置参数表中取出每个位置的向量：

```text
position_ids[T] -> position_vectors[T,C]
```

它和 token embedding 类似，但 lookup key 是位置索引，不是 token id。

## 2. 前置组件

- `component.position_ids.v1`
- `component.parameter_matrix.v1`
- `component.embedding_lookup.v1`

本关复用 gather 行为，但要求 table role 是 position embedding。

## 3. 本关新增能力

- `PositionTableContract`：检查 table axes `[Tmax,C]`。
- `GatherRows`：按 position ids 取行。
- `PositionEmbeddingContract`：输出 axes `[T,C]`。
- `PositionTraceProbe`：展示 position 2 取 table row 2。
- `ReferenceChecker`：检查 gather 结果。

## 4. 具体案例

Visible case:

```text
position_ids = [0,1,2]
position_table[Tmax=5,C=3]
```

期望：

```text
position_vectors[0,:] = table[0,:]
position_vectors[1,:] = table[1,:]
position_vectors[2,:] = table[2,:]
```

## 5. 初始错误图

画布给出：

- `position_ids: PositionIds[T]`
- `position_table: ParameterMatrix[Tmax,C]`
- `pos_emb_out: PositionEmbeddingContract`
- `trace: PositionTraceProbe`
- `reference: ReferenceChecker`

缺少 PositionTableContract 和 GatherRows。

## 6. 目标内部实现

```text
position_table.out -> table_contract.x
position_ids.out -> gather.ids
table_contract.out -> gather.table
gather.out -> pos_emb_out.x
gather.out -> trace.x
pos_emb_out.out -> reference.x
```

## 7. 玩家操作

1. 检查 position ids 范围。
2. 拖入 `PositionTableContract`。
3. 拖入 `GatherRows`。
4. 连接 ids 和 table。
5. 接入 trace、合约和 reference。
6. 检查当前任务并提交认证。

## 8. 错误路径

- 用 token embedding table：role 错。
- position 从 1 开始：所有 row 偏移，reference 失败。
- table 长度小于最大 position：越界失败。
- 输出 `[B,T,C]`：本关只输出 `[T,C]`，batch broadcast 在下一关。
- 直接复制 table 前 T 行但忽略 ids：乱序 ids hidden 失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 PositionTableContract 和 GatherRows。
- 输出 axes 为 `[T,C]`。
- 每行等于 table[position_id]。
- TraceProbe 能定位至少一个 position-row 对应。

## 10. Hidden / Mutation 测试

Hidden case A：乱序 ids。

```text
position_ids = [2,0,1]
```

必须按 ids gather，而不是默认前 T 行。

Hidden case B：Tmax 变大。

```text
T=4, Tmax=8
```

Hidden case C：越界。

```text
position_ids includes 6, table Tmax=5
```

必须失败。

## 11. 认证后接口

```text
component.position_embedding.v1
inputs:
  position_ids: int[T]
  table: parameter_matrix[Tmax,C]
output:
  position_vectors: float32[T,C]
```

## 12. 后续调用

HiddenInit 会把 token embedding `[B,T,C]` 和 position embedding `[T,C]` 相加。这里必须保持 `[T,C]`，方便下一关清楚地学习 broadcast。
