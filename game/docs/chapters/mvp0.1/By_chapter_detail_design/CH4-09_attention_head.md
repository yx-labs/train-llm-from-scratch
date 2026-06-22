# Chapter 4-9 Attention Head: SingleHeadAttention

## 1. 组件真实用途

SingleHeadAttention 把前面 Chapter 4 的组件组装成一个完整 attention head：

```text
q = QLinear(hidden)
k = KLinear(hidden)
v = VLinear(hidden)
scores = QKScore(q,k)
scaled = ScaleBySqrtD(scores,D)
masked = MaskAdd(scaled, causal_mask)
prob = SoftmaxLastDim(masked)
context = AttentionApply(prob,v)
```

本关是一次组件编排关。玩家不再造内部数学细节，而是证明这些已认证组件可以按正确顺序组合。

## 2. 前置组件

- `component.q_linear.v1`
- `component.k_linear.v1`
- `component.v_linear.v1`
- `component.qk_score.v1`
- `component.scale_by_sqrt_d.v1`
- `component.causal_mask.v1`
- `component.mask_add.v1`
- `component.softmax_last_dim.v1`
- `component.attention_apply.v1`

本关禁止使用预制 `SingleHeadAttention`。

## 3. 本关新增能力

- `AttentionPipelineContract`：检查 Q/K/V/score/prob/context 的完整链路。
- `StageProbe`：允许玩家在每个阶段查看 shape、axes 和小样本。
- `LeakProbe`：检查 future token 是否被 causal mask 阻断。
- `ReferenceChecker`：端到端数值参考。

`StageProbe` 是教学探针。它不是目标组件的一部分，但能减少“这么多节点不知道错在哪”的困惑。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=3,C=4]
D=2
causal mask: token i 只能看 j <= i
```

期望输出：

```text
context[B=1,H=1,T=3,D=2]
```

LeakProbe 展示：

```text
token 0 的 prob 只能落在 key 0
token 1 的 prob 只能落在 key 0,1
token 2 可以看 0,1,2
```

## 5. 初始错误图

画布给出：

- `hidden: HiddenSource[B,T,C]`
- `weights: QKVWeightPack`
- `causal_mask: CausalMask[T,T]`
- `head_out: ContextContract`
- `stage_probe: StageProbe`
- `leak_probe: LeakProbe`
- `reference: ReferenceChecker`

没有任何 Q/K/V projection 或 attention pipeline。玩家必须从组件库拖入已认证组件。

## 6. 目标内部实现

```text
hidden.out -> q_linear.hidden
hidden.out -> k_linear.hidden
hidden.out -> v_linear.hidden
weights.wq -> q_linear.weight
weights.bq -> q_linear.bias
weights.wk -> k_linear.weight
weights.bk -> k_linear.bias
weights.wv -> v_linear.weight
weights.bv -> v_linear.bias
q_linear.q -> qk_score.q
k_linear.k -> qk_score.k
qk_score.scores -> scale.scores
weights.d -> scale.d
scale.scaled -> mask_add.scores
causal_mask.out -> mask_add.causal_mask
mask_add.masked_scores -> softmax.masked_scores
softmax.prob -> attention_apply.prob
v_linear.v -> attention_apply.value
attention_apply.context -> head_out.x
attention_apply.context -> stage_probe.context
softmax.prob -> leak_probe.prob
head_out.out -> reference.x
```

## 7. 玩家操作

1. 拖入 QLinear、KLinear、VLinear。
2. 接入 hidden 和各自权重。
3. 拖入 QKScore、Scale、MaskAdd、Softmax、AttentionApply。
4. 按 pipeline 顺序连接每一段。
5. 接 StageProbe 和 LeakProbe，确认每个阶段的 shape。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- Q/K 接反：shape 可能过，score 数值错。
- 漏掉 scale：context 数值偏差，reference 失败。
- softmax 在 mask 前执行：future leak，LeakProbe 失败。
- 用 V 参与 QKScore：role contract 失败。
- 跳过 StageProbe：结构断言失败，避免玩家黑盒连线。
- 使用预制 AttentionHead：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须按 Q/K/V -> QKScore -> Scale -> MaskAdd -> Softmax -> AttentionApply 顺序连通。
- Q/K/V role contract 必须正确。
- LeakProbe 检查 future probability 小于 `1e-4`。
- 输出 axes 为 `[B,H,T,D]`。
- 输出 allclose 到 end-to-end attention reference。

## 10. Hidden / Mutation 测试

Hidden case A：`T=1`。

只有一个 token，mask 不应破坏输出。

Hidden case B：`T=4,D=3`。

检查 scale 随 D 变化，mask 随 T 变化。

Hidden case C：future leak trap。

系统把 future value 设置成极大值。如果 mask 或 softmax 顺序错，context 会明显偏移。

## 11. 认证后接口

```text
component.single_head_attention.v1
inputs:
  hidden: float32[B,T,C]
  qkv_weights: QKVWeightPack
  causal_mask: bool[T,T]
output:
  context: float32[B,H,T,D]
```

## 12. 后续调用

Chapter 5 会把单头扩展成多头。SingleHeadAttention 是后续 Multi-Head Attention 的最小可解释版本，必须保留内部 stage 可展开能力，不能变成黑盒答案节点。
