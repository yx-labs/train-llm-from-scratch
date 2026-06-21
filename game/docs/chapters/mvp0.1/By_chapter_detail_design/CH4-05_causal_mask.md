# Chapter 4-5 Causal Mask

## 1. 组件真实用途

CausalMask 生成 decoder-only attention 的未来屏蔽矩阵。它保证 query token 只能看自己和过去的 key token，不能看未来。

核心规则：

```text
mask[i,j] = 0       if key j <= query i
mask[i,j] = -10000  if key j > query i
```

## 2. 前置组件

- `component.qk_score.v1`

本关不允许使用 `component.causal_mask.v1`。玩家要搭出 position grid、future comparator 和 mask value writer。

## 3. 建议内部节点

- `ScoreShapeProbe`：从 scores 或 T length 读取 T。
- `QueryIndexGrid`：生成 query row index。
- `KeyIndexGrid`：生成 key column index。
- `FutureComparator`：判断 `key > query`。
- `MaskValueWriter`：未来位置写入大负数，其余写 0。
- `MaskContractGate`：检查 `float32[T,T]` 或 broadcast-ready `[B,H,T,T]`。

## 4. 具体案例

Visible case T=4:

```text
mask =
[
  [0, -10000, -10000, -10000],
  [0,      0, -10000, -10000],
  [0,      0,      0, -10000],
  [0,      0,      0,      0]
]
```

这个矩阵后续会加到 QK scores 上，让 softmax 后未来位置接近 0。

## 5. 初始错误图

画布给出 T source、mask contract、reference。没有 query/key grid，也没有 comparator。

如果直接拖一个下三角答案节点，玩家不会理解 row/column 语义。

## 6. 目标内部实现

```text
t_source.out -> query_grid.t
t_source.out -> key_grid.t
query_grid.out -> future_compare.query
key_grid.out -> future_compare.key
future_compare.out -> mask_writer.future
mask_writer.out -> mask_out.x
mask_out.out -> reference.x
```

## 7. 错误路径

- 方向反了：屏蔽过去而不是未来。
- 用 1/0 mask 直接加到 scores：加性 mask 需要 0 和大负数。
- 只生成 `[T]`：缺少 query-key 二维关系。
- T 变大时硬编码 4x4：hidden T=6 失败。

## 8. 测试设计

当前任务：

- shape：`float32[T,T]`。
- 行为：所有 `j > i` 的位置必须 <= `-9999`。
- 行为：所有 `j <= i` 的位置必须等于 0。

Hidden cases：

- T=6。
- orientation trap：检查 `[query,key]`，不是 `[key,query]`。
- broadcast variant：mask 能扩展到 `[B,H,T,T]`。

## 9. 认证变体

公开认证允许选择：

- T length
- masked value：`-10000` / `-1e9`
- output mode：`[T,T]` or broadcast-ready `[B,H,T,T]`

系统生成 reference mask，不要求玩家手写矩阵。

## 10. 认证后接口

```text
component.causal_mask.v1
input:
  t or scores: shape source
output:
  mask: float32[T,T] or float32[B,H,T,T]
```

## 11. 后续调用

MaskApply 会把 causal mask 加到 QK scores 上。Softmax 关会验证未来位置的概率是否接近 0，因此 CausalMask 必须是行为正确的矩阵，而不是只满足 shape。
