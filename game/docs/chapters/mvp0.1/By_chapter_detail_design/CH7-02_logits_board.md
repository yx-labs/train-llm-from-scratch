# Chapter 7-2 Logits Board: Logits[B,T,V]

## 1. 组件真实用途

LogitsBoard 给 LMHead 输出绑定 `[B,T,V]` 语义，并连接 vocab id 到 logit 列。

它不改变数值，只证明最后一维确实是 vocabulary axis。

## 2. 前置组件

- `component.lm_head.v1`
- `component.vocab_table.v1`

## 3. 本关新增能力

- `VocabAxisBinder`：把最后一维绑定为 V。
- `LogitColumnProbe`：查看某个 token id 的 logit。
- `LogitsContract`：检查 dtype/axes。
- `ReferenceChecker`：检查不改变 LMHead 输出。

## 4. 具体案例

Visible case:

```text
logits[B=1,T=2,V=13]
id 7 = " llm"
logits[0,1,7] 表示位置 1 下一个 token 为 " llm" 的分数
```

## 5. 初始错误图

画布给出 logits_source、vocab、board_out、column_probe、reference。缺少 VocabAxisBinder。

## 6. 目标内部实现

```text
logits_source -> vocab_axis_binder.logits
vocab -> vocab_axis_binder.vocab
vocab_axis_binder -> board_out
vocab_axis_binder -> column_probe
board_out -> reference
```

## 7. 错误路径

- 把 T 当 V：shape 可能相似，但 vocab lookup 失败。
- 对 logits 做 softmax：LogitsBoard 不负责概率化。
- 裁剪 vocab 列：reference 失败。
- 不接 vocab：无法解释 V 轴。

## 8. 测试设计

- Visible：id 7 column probe。
- Hidden A：稀疏 vocab 最大 id 20。
- Hidden B：T==V 小尺寸陷阱。
- Hidden C：logits 值必须原样保留。

## 9. 认证后接口

```text
component.logits_board.v1
inputs:
  logits: float32[B,T,V]
  vocab: vocab_table
output:
  board: logits[B,T,V]
```

## 10. 后续调用

训练时 CrossEntropy 读取目标 id 对应列；生成时 Sampler 读取最后一个位置的 V 维分布。
