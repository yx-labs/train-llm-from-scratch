# Chapter 6-3 Attention Sublayer: LN -> MHA -> Residual

## 1. 组件真实用途

AttentionSublayer 是 transformer block 的第一半：

```text
normed = LayerNorm(hidden)
attn = MultiHeadAttention(normed)
out = ResidualAdd(hidden, attn)
```

这是 pre-norm 结构，LayerNorm 作用在 branch 输入上，residual 使用原 hidden。

## 2. 前置组件

- `component.layernorm.v1`
- `component.multi_head_attention.v1`
- `component.residual_add.v1`

禁止拖预制 AttentionSublayer。

## 3. 本关新增能力

- `PreNormPathProbe`：显示 residual 路径没有经过 LayerNorm。
- `SublayerStageContract`：检查 LN -> MHA -> Residual 顺序。
- `ReferenceChecker`：端到端 reference。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=3,C=6]
H=3
```

期望输出仍为：

```text
float32[B,T,C]
```

## 5. 初始错误图

画布给出 hidden、LN 参数、MHA weights、mask、sublayer_out、path_probe、reference。缺少 LN/MHA/Residual。

## 6. 目标内部实现

```text
hidden -> layernorm.x
layernorm.y -> mha.hidden
mask -> mha.causal_mask
mha.out -> residual_add.branch
hidden -> residual_add.residual
residual_add.out -> sublayer_out
hidden + layernorm + residual_add -> path_probe
sublayer_out -> reference
```

## 7. 玩家操作

1. 拖入 LayerNorm、MHA、ResidualAdd。
2. 将 LayerNorm 输出接入 MHA。
3. 将 MHA 输出作为 branch。
4. 将原 hidden 作为 residual。
5. 接 PathProbe 和 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- residual 也接 normed：pre-norm 结构错。
- MHA 直接吃原 hidden：reference 失败。
- 漏掉 residual：shape 对但数值错。
- mask 不接 MHA：future leak。
- 顺序变成 MHA -> LN -> residual：StageContract 失败。

## 9. 测试设计

- Visible：端到端 allclose。
- Hidden A：LayerNorm gamma/beta 非默认。
- Hidden B：future leak trap。
- Hidden C：branch 全零时输出应等于 hidden。

## 10. 认证后接口

```text
component.attention_sublayer.v1
inputs:
  hidden: float32[B,T,C]
  weights: AttentionSublayerWeights
  mask: bool[T,T]
output:
  out: float32[B,T,C]
```

## 11. 后续调用

TransformerBlock 会把 AttentionSublayer 接到 MLPSublayer 前面。
