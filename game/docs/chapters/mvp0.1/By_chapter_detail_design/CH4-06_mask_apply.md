# Chapter 4-6 Mask Apply: MaskAdd

## 1. 组件真实用途

MaskAdd 把 causal mask 转成 additive bias，并加到 scaled scores 上：

```text
masked_scores[i,j] = scaled_scores[i,j] + mask_bias[i,j]

mask_bias[i,j] = 0       if j <= i
mask_bias[i,j] = -10000  if j > i
```

本关要明确：attention mask 在 softmax 之前是“加一个很大的负数”，不是把分数乘成 0。

## 2. 前置组件

- `component.scale_by_sqrt_d.v1`
- `component.causal_mask.v1`
- `component.bias_add.v1`

玩家已经能生成 causal mask。本关把 bool mask 转成可参与数值计算的 float32 bias。

## 3. 本关新增能力

- `MaskBiasGate`：把 bool/int mask 转成 `0 / -10000`。
- `MaskBroadcast`：把 `[T,T]` mask 扩展到 `[B,H,T,T]`。
- `AddGate`：将 bias 加到 scores。
- `MaskedCellProbe`：展示一个被遮挡 future cell 的变化。
- `ReferenceChecker`：检查数值。

## 4. 具体案例

Visible case:

```text
scaled_scores[B=1,H=1,T=4,T=4]
causal_mask[T=4,T=4]
```

对于 query token 1：

```text
allowed: key 0, key 1
blocked: key 2, key 3
```

所以：

```text
masked_scores[0,0,1,2] = scaled_scores[0,0,1,2] - 10000
masked_scores[0,0,1,1] = scaled_scores[0,0,1,1]
```

## 5. 初始错误图

画布给出：

- `scaled_scores: ScoreBoard[B,H,T,T]`
- `causal_mask: CausalMask[T,T]`
- `masked_out: ScoreContract`
- `masked_probe: MaskedCellProbe`
- `reference: ReferenceChecker`

缺少 mask-to-bias、broadcast 和 add。

## 6. 目标内部实现

```text
causal_mask.out -> mask_bias.mask
mask_bias.out -> mask_broadcast.small
scaled_scores.out -> mask_broadcast.target
mask_broadcast.out -> add.right
scaled_scores.out -> add.left
add.out -> masked_out.x
add.out -> masked_probe.scores
masked_out.out -> reference.x
```

其中：

- `mask_bias.blockValue = -10000`
- `mask_broadcast.alignAxes = [T,T]`
- `masked_out.expectedAxes = [B,H,T,T]`

## 7. 玩家操作

1. 拖入 `MaskBiasGate`。
2. 将 causal mask 转成 float32 bias。
3. 拖入 `MaskBroadcast`，把 `[T,T]` 对齐到 score board 的后两维。
4. 用 `AddGate` 把 mask bias 加到 scaled scores。
5. 连接 MaskedCellProbe 和 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- 用乘法 mask：blocked cell 变成 0，softmax 后仍可能得到概率。
- 把 past token mask 掉：orientation 错，hidden cell 检查失败。
- broadcast 到 B/H：shape 可能可扩展，但语义错。
- blockValue 用 `-1`：softmax 仍会给 future token 可见概率。
- 只接合约：shape 不变，必须有数值 reference。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 `MaskBiasGate`、`MaskBroadcast`、`AddGate`。
- blocked cell 必须减少 `10000`。
- allowed cell 必须保持原值。
- 输出 axes 保持 `[B,H,T,T]`。
- MaskedCellProbe 必须接入输出。

## 10. Hidden / Mutation 测试

Hidden case A：`T=1`。

```text
没有 future token，所有 cell 保持原值。
```

Hidden case B：`T=5`。

检查多个 future cell 都被 mask。

Hidden case C：B/H 变体。

```text
scores: [B=2,H=3,T=4,T=4]
mask: [T=4,T=4]
```

认证必须证明 mask 只沿 B/H broadcast，不改变后两维 orientation。

## 11. 认证后接口

```text
component.mask_add.v1
inputs:
  scores: float32[B,H,T,T]
  causal_mask: bool[T,T]
output:
  masked_scores: float32[B,H,T,T]
```

## 12. 后续调用

Softmax 会把 masked scores 转成概率。只有 additive mask 才能让 future token 的概率接近 0。
