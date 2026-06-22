# Chapter 4-7 Softmax: SoftmaxLastDim

## 1. 组件真实用途

SoftmaxLastDim 把 masked scores 的最后一维转成概率分布：

```text
prob[i,j] = exp(score[i,j] - row_max[i]) / sum_k exp(score[i,k] - row_max[i])
```

每个 query row 的概率和必须是 1，被 mask 的 future token 概率应接近 0。

## 2. 前置组件

- `component.mask_add.v1`
- `component.sum_reduce.v1`

本关重点是沿最后一个 key 轴做归一化，不是对整张 score board 归一化。

## 3. 本关新增能力

- `RowMaxReduce`：按最后一维求 row max。
- `SubtractBroadcast`：做数值稳定的 `score - row_max`。
- `ExpGate`：逐元素指数。
- `SumReduce`：按最后一维求和。
- `DivideBroadcast`：逐行除以 row sum。
- `RowSumProbe`：展示每行概率和。

## 4. 具体案例

Visible case 关注一个 query row：

```text
masked_scores[0,0,1,:] = [2.0, 1.0, -10000, -10000]
```

期望：

```text
softmax row ~= [0.7311, 0.2689, 0.0, 0.0]
row_sum ~= 1.0
```

## 5. 初始错误图

画布给出：

- `masked_scores: MaskedScoreBoard[B,H,T,T]`
- `prob_out: ProbabilityContract`
- `row_sum_probe: RowSumProbe`
- `reference: ReferenceChecker`

缺少 max、subtract、exp、sum、divide。

## 6. 目标内部实现

```text
masked_scores.out -> row_max.x
masked_scores.out -> shift.left
row_max.out -> shift.right
shift.out -> exp.x
exp.out -> row_sum.x
exp.out -> divide.left
row_sum.out -> divide.right
divide.out -> prob_out.x
divide.out -> row_sum_probe.prob
prob_out.out -> reference.x
```

其中：

- `row_max.axis = last`
- `row_sum.axis = last`
- `prob_out.expectedAxes = [B,H,T,T]`

## 7. 玩家操作

1. 拖入 `RowMaxReduce`，设置 axis 为最后一维。
2. 用 `SubtractBroadcast` 做 row-wise shift。
3. 拖入 `ExpGate`。
4. 用 `SumReduce` 沿最后一维求和。
5. 用 `DivideBroadcast` 得到概率。
6. 连接 RowSumProbe 和 reference。
7. 检查当前任务，再提交认证。

## 8. 错误路径

- 对整张 `[T,T]` 求一个 sum：每行不再独立，RowSumProbe 失败。
- 沿 query 轴求 sum：列归一化，语义错。
- 不减 row max：visible 可能过，大数 hidden 溢出。
- 被 mask cell 仍有明显概率：reference 失败。
- 使用预制 SoftmaxLastDim：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 max-shift-exp-sum-divide 结构。
- sum/reduce axis 必须是最后一维 key axis。
- 每个 row sum allclose 到 1。
- masked future 概率小于 `1e-4`。
- 输出 allclose 到 stable softmax reference。

## 10. Hidden / Mutation 测试

Hidden case A：score 平移不变。

```text
row A = [2,1,0]
row B = [102,101,100]
```

两行 softmax 应相同，验证 row max shift。

Hidden case B：`T=5`。

检查每一行都独立归一化。

Hidden case C：全 mask 保护。

对于非法全 mask row，系统提供 guard policy：输出全 0 并给出警告，不允许 NaN。

## 11. 认证后接口

```text
component.softmax_last_dim.v1
inputs:
  masked_scores: float32[B,H,T,T]
output:
  prob: float32[B,H,T,T]
```

## 12. 后续调用

WeightedSum 会用这些概率读取 V。Softmax 如果按错轴，shape 仍然完全正确，但 attention 行为会彻底错误，因此本关必须是行为认证。
