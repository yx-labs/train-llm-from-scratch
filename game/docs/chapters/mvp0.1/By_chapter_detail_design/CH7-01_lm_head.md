# Chapter 7-1 LM Head: LMHead

## 1. 组件真实用途

LMHead 把 hidden channel 投影到 vocabulary 维度：

```text
logits = Linear(normed[B,T,C], W_lm[C,V], b_lm[V])
```

每个 token 位置都会得到一个对全 vocab 的打分向量。

## 2. 前置组件

- `component.final_layernorm.v1`
- `component.linear.v1`
- `component.vocab_table.v1`

## 3. 本关新增能力

- `VocabWidthContract`：检查输出 V 覆盖 vocab id 空间。
- `LMWeightContract`：检查 weight axes `[C,V]`。
- `component.linear.v1`：执行投影。
- `LogitTraceProbe`：展示某个 vocab id 的 logit 来源。
- `ReferenceChecker`：检查数值。

## 4. 具体案例

Visible case:

```text
normed[B=1,T=2,C=4]
vocab V=13
W_lm[C=4,V=13]
logits[B=1,T=2,V=13]
```

## 5. 初始错误图

画布给出 normed、W_lm、b_lm、vocab、lm_out、trace、reference。缺少 vocab/weight contract 和 Linear。

## 6. 目标内部实现

```text
vocab -> vocab_width
w_lm -> lm_weight_contract
normed -> linear.hidden
lm_weight_contract -> linear.weight
b_lm -> linear.bias
linear.out -> lm_out
linear.out -> trace
lm_out -> reference
```

## 7. 错误路径

- 输出 C 维而不是 V 维：不能采样 token。
- V 小于最大 vocab id：部分 token 没有 logit。
- weight 转置 `[V,C]`：内维错。
- 复用 embedding table 未转置：如果设计 weight tying，必须显式 TieWeightGate；本关默认独立 W_lm。
- 漏 bias：reference 失败。

## 8. 测试设计

- Visible：V=13。
- Hidden A：V=21。
- Hidden B：C=3。
- Hidden C：vocab 最大 id 超过 V，必须失败。

## 9. 认证后接口

```text
component.lm_head.v1
inputs:
  normed: float32[B,T,C]
  weight: float32[C,V]
  bias: float32[V]
output:
  logits: float32[B,T,V]
```

## 10. 后续调用

LogitsBoard 会把 logits 与 vocab 语义绑定，CrossEntropy 和 Sampler 都消费这张 board。
