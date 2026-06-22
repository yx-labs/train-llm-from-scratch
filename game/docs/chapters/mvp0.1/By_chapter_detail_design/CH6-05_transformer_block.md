# Chapter 6-5 Transformer Block: Block

## 1. 组件真实用途

TransformerBlock 串联 attention sublayer 和 MLP sublayer：

```text
h1 = AttentionSublayer(hidden)
h2 = MLPSublayer(h1)
```

输出仍是 `[B,T,C]`，可以继续进入下一层 block。

## 2. 前置组件

- `component.attention_sublayer.v1`
- `component.mlp_sublayer.v1`

本关是 block 级编排，不允许拖预制 Block。

## 3. 本关新增能力

- `BlockStageContract`：检查 Attention -> MLP 顺序。
- `ResidualStreamProbe`：展示 h0/h1/h2 的变化。
- `ReferenceChecker`：端到端 block reference。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=3,C=6]
block_out[B=1,T=3,C=6]
```

ResidualStreamProbe 展示三段：

```text
h0 input
h1 after attention
h2 after mlp
```

## 5. 初始错误图

画布给出 hidden、block weights、mask、block_out、stream_probe、reference。缺少两个 sublayer。

## 6. 目标内部实现

```text
hidden -> attention_sublayer.hidden
mask -> attention_sublayer.mask
attention_sublayer.out -> mlp_sublayer.hidden
mlp_sublayer.out -> block_out
hidden + attention_sublayer + mlp_sublayer -> stream_probe
block_out -> reference
```

## 7. 玩家操作

1. 拖入 AttentionSublayer。
2. 拖入 MLPSublayer。
3. 将 attention 输出接到 MLP 输入。
4. 接入 StreamProbe 和 reference。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- MLP 和 Attention 顺序反了：reference 失败。
- 两个 sublayer 都吃原 hidden：结构断言失败。
- 漏掉 mask：future leak。
- 输出 h1 而非 h2：数值错。
- 使用预制 Block：shortcut。

## 9. 测试设计

- Visible：端到端 allclose。
- Hidden A：attention branch zero，block 仍执行 MLP。
- Hidden B：MLP branch zero，输出等于 h1。
- Hidden C：T=1，无 future mask 干扰。

## 10. 认证后接口

```text
component.transformer_block.v1
inputs:
  hidden: float32[B,T,C]
  weights: BlockWeights
  mask: bool[T,T]
output:
  out: float32[B,T,C]
```

## 11. 后续调用

BlockStack 会重复使用同一个 Block 结构多次。Block 的价值是封装一层完整 transformer 计算。
