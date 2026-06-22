# Chapter 2-5 Hidden Init: Token + Position Add

## 1. 组件真实用途

HiddenInit 把 token embedding 和 position embedding 相加，生成 transformer block 的初始 hidden states：

```text
hidden[b,t,c] = token_emb[b,t,c] + pos_emb[t,c]
```

本关核心是把 `[T,C]` position vectors 正确 broadcast 到 `[B,T,C]`。

## 2. 前置组件

- `component.embedding_lookup.v1`
- `component.position_embedding.v1`
- `component.bias_add.v1`

玩家已经见过 bias broadcast。本关复用 broadcast 思维，但对齐轴是 `[T,C]`。

## 3. 本关新增能力

- `PositionBroadcast`：把 `[T,C]` 扩展到 `[B,T,C]`。
- `AddGate`：逐元素相加。
- `HiddenContract`：检查输出 `[B,T,C]`。
- `TokenPositionTrace`：展示某个 hidden cell 的两个来源。
- `ReferenceChecker`：检查数值。

## 4. 具体案例

Visible case:

```text
token_emb[B=1,T=3,C=2]
pos_emb[T=3,C=2]
```

示例：

```text
token_emb[0,1,:] = [0.4, -0.2]
pos_emb[1,:] = [0.1, 0.3]
hidden[0,1,:] = [0.5, 0.1]
```

## 5. 初始错误图

画布给出：

- `token_emb: TokenEmbeddingOutput[B,T,C]`
- `pos_emb: PositionEmbeddingOutput[T,C]`
- `hidden_out: HiddenContract`
- `trace: TokenPositionTrace`
- `reference: ReferenceChecker`

缺少 PositionBroadcast 和 AddGate。

## 6. 目标内部实现

```text
token_emb.out -> pos_broadcast.target
pos_emb.out -> pos_broadcast.small
token_emb.out -> add.left
pos_broadcast.out -> add.right
add.out -> hidden_out.x
add.out -> trace.hidden
token_emb.out -> trace.token
pos_emb.out -> trace.position
hidden_out.out -> reference.x
```

其中：

- `pos_broadcast.alignAxes = [T,C]`
- `hidden_out.expectedAxes = [B,T,C]`

## 7. 玩家操作

1. 拖入 `PositionBroadcast`。
2. 将 token_emb 作为 target，pos_emb 作为 small。
3. 设置对齐轴 `[T,C]`。
4. 用 `AddGate` 相加。
5. 连接 trace、contract 和 reference。
6. 检查当前任务并提交认证。

## 8. 错误路径

- 对齐 `[C]`：每个位置使用同一 position vector，reference 失败。
- 把 position 加到 batch 轴：B 变体失败。
- concat token 和 position：shape 变成 `[B,T,2C]`。
- 忘记 position：shape 对但数值少一项。
- 直接使用预制 HiddenInit：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 PositionBroadcast 和 AddGate。
- 输出 axes 为 `[B,T,C]`。
- 输出 allclose 到 `token_emb + pos_emb`。
- TraceProbe 能显示 token 与 position 两个加数。

## 10. Hidden / Mutation 测试

Hidden case A：`B=2`。

同一 position embedding 应广播到两个 batch。

Hidden case B：`T=1`。

只加 position 0。

Hidden case C：位置向量乱序。

如果上一关 position ids 是 `[2,0,1]`，HiddenInit 必须使用传入的 position vectors，不重新生成默认顺序。

## 11. 认证后接口

```text
component.hidden_init.v1
inputs:
  token_emb: float32[B,T,C]
  pos_emb: float32[T,C]
output:
  hidden: float32[B,T,C]
```

## 12. 后续调用

Linear、Attention 和 MLP 都消费 hidden `[B,T,C]`。HiddenInit 是文本数据第一次进入模型隐藏空间。
