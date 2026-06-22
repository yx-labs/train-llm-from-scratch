# Chapter 5-4 Multi-Head Attention: MHA

## 1. 组件真实用途

MultiHeadAttention 把 hidden 投影成 Q/K/V，多头并行 attention，再合并并输出投影：

```text
q/k/v = Linear(hidden)
heads = ParallelHeads(split(q), split(k), split(v))
context = HeadConcat(heads)
out = OutputProjection(context)
```

本关是 Chapter 5 的组件编排关。

## 2. 前置组件

- `component.q_linear.v1`
- `component.k_linear.v1`
- `component.v_linear.v1`
- `component.head_split.v1`
- `component.parallel_heads.v1`
- `component.head_concat.v1`
- `component.attention_output_projection.v1`

禁止使用预制 MHA。

## 3. 本关新增能力

- `MHAStageContract`：检查 q/k/v、split、parallel、concat、output 的顺序。
- `HeadCountProbe`：显示 H、D、C 的关系。
- `NoLeakProbe`：检查 causal mask 仍然生效。
- `ReferenceChecker`：端到端多头 reference。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=3,C=6]
H=3
D=2
out[B=1,T=3,C=6]
```

HeadCountProbe 展示：`6 channels = 3 heads * 2 dims`。

## 5. 初始错误图

画布给出 hidden、qkv weights、head count、causal mask、mha_out、stage probe、reference。缺少所有内部组件。

## 6. 目标内部实现

```text
hidden -> q_linear/k_linear/v_linear
q_linear/k_linear/v_linear -> head_split(q/k/v)
split q/k/v + mask -> parallel_heads
parallel_heads -> head_concat
head_concat + W_o/b_o -> output_projection
output_projection -> mha_out -> reference
```

StageContract 必须能追踪整条路径。

## 7. 玩家操作

1. 拖入 Q/K/V projection。
2. 对 Q/K/V 分别接 HeadSplit。
3. 接 ParallelHeads。
4. 接 HeadConcat。
5. 接 OutputProjection。
6. 连接 HeadCountProbe、NoLeakProbe、reference。
7. 检查当前任务，再提交认证。

## 8. 错误路径

- 只跑单头 attention：H 轴没有真实参与。
- Q/K/V split 只接其中一个：结构失败。
- concat 前漏掉某个 head：reference 失败。
- output projection 被省略：shape 对但数值错。
- mask 未传入 ParallelHeads：NoLeakProbe 失败。

## 9. 测试设计

- Visible：H=3,D=2。
- Hidden A：H=2,D=4,C=8。
- Hidden B：future value 设置极大，检查 NoLeak。
- Hidden C：某个 head sentinel 改变，只影响对应 concat 段。

## 10. 认证后接口

```text
component.multi_head_attention.v1
inputs:
  hidden: float32[B,T,C]
  weights: MHAWeightPack
  causal_mask: bool[T,T]
  head_count: int[]
output:
  out: float32[B,T,C]
```

## 11. 后续调用

Transformer block 会把 MHA 放在 LayerNorm 和 residual 之间。MHA 必须保留内部可展开结构，方便玩家定位 attention 错误。
