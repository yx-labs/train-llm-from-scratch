# Chapter 6-4 MLP Sublayer: LN -> MLP -> Residual

## 1. 组件真实用途

MLPSublayer 是 transformer block 的第二半：

```text
normed = LayerNorm(hidden)
branch = MLP(normed)
out = ResidualAdd(hidden, branch)
```

它与 AttentionSublayer 结构相同，但 branch 组件换成 MLP。

## 2. 前置组件

- `component.layernorm.v1`
- `component.mlp.v1`
- `component.residual_add.v1`

禁止使用预制 MLPSublayer。

## 3. 本关新增能力

- `PreNormPathProbe`：确认 residual 是原 hidden。
- `MLPStageProbe`：展示 MLP 内部宽度 4C。
- `SublayerStageContract`：检查 LN -> MLP -> Residual。
- `ReferenceChecker`：端到端 reference。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=2,C=3]
MLP internal width = 12
out[B=1,T=2,C=3]
```

## 5. 初始错误图

画布给出 hidden、LN 参数、MLP weights、mlp_sublayer_out、path_probe、reference。缺少 LN/MLP/Residual。

## 6. 目标内部实现

```text
hidden -> layernorm.x
layernorm.y -> mlp.hidden
mlp.out -> residual_add.branch
hidden -> residual_add.residual
residual_add.out -> mlp_sublayer_out
mlp + residual_add -> stage_probe
mlp_sublayer_out -> reference
```

## 7. 玩家操作

1. 拖入 LayerNorm、MLP、ResidualAdd。
2. 按 pre-norm 顺序连接。
3. 将原 hidden 接 residual。
4. 接 MLPStageProbe 和 reference。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- MLP 吃原 hidden 而非 normed：reference 失败。
- residual 接 normed：结构错。
- 漏掉 MLP activation：MLP reference 失败。
- 输出 4C：ResidualAdd shape 失败。
- 用 AttentionSublayer 代替：branch 语义错。

## 9. 测试设计

- Visible：端到端 allclose。
- Hidden A：MLP branch 全零，输出等于 hidden。
- Hidden B：gamma/beta 非默认。
- Hidden C：C=4，内部宽度 16。

## 10. 认证后接口

```text
component.mlp_sublayer.v1
inputs:
  hidden: float32[B,T,C]
  weights: MLPSublayerWeights
output:
  out: float32[B,T,C]
```

## 11. 后续调用

TransformerBlock 把 AttentionSublayer 和 MLPSublayer 串联起来。
