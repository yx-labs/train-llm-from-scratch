# Chapter 6-0 Mean and Variance: ChannelStats

## 1. 组件真实用途

ChannelStats 在每个 token 的 C 维上计算 mean 和 variance：

```text
mean[b,t] = avg_c x[b,t,c]
var[b,t] = avg_c (x[b,t,c] - mean[b,t])^2
```

LayerNorm 会用这两个统计量标准化 hidden。

## 2. 前置组件

- `component.sum_reduce.v1`
- `component.axis_tensor.v1`

本关重点是沿 C 轴 reduce，而不是沿 T 或 B。

## 3. 本关新增能力

- `MeanReduce`：沿 C 轴求均值。
- `CenterBroadcast`：把 mean 广播回 `[B,T,C]`。
- `SquareGate`：平方中心化结果。
- `VarianceReduce`：沿 C 轴求均值。
- `StatsProbe`：展示某个 token 的 mean/var。

## 4. 具体案例

Visible case:

```text
x[0,1,:] = [1,3,5]
mean[0,1] = 3
var[0,1] = ((-2)^2 + 0^2 + 2^2)/3 = 2.6667
```

## 5. 初始错误图

画布给出 x、stats_out、stats_probe、reference。缺少 mean/center/square/variance。

## 6. 目标内部实现

```text
x -> mean_reduce
x + mean_reduce -> center_broadcast/subtract
centered -> square
square -> variance_reduce
mean_reduce + variance_reduce -> stats_out
stats_out -> stats_probe
stats_out -> reference
```

## 7. 玩家操作

1. 拖入 `MeanReduce`，axis 设为 C。
2. 将 mean broadcast 回原 shape 并 subtract。
3. 平方 centered 值。
4. 拖入 `VarianceReduce`，axis 仍为 C。
5. 连接 stats probe 和 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- 沿 T 求 mean：shape 可能可广播但语义错。
- 用 sample variance 除以 C-1：reference 失败。
- 漏掉中心化直接平方 x：variance 错。
- reduce 掉 B/T：stats shape 错。
- 使用预制 LayerNorm：shortcut。

## 9. 测试设计

- Visible：C=3，检查 mean/var。
- Hidden A：C=4。
- Hidden B：T==C，防止轴混淆。
- Hidden C：常量向量，variance 应为 0。

## 10. 认证后接口

```text
component.channel_stats.v1
inputs:
  x: float32[B,T,C]
outputs:
  mean: float32[B,T]
  variance: float32[B,T]
```

## 11. 后续调用

LayerNorm 使用 mean/variance 进行标准化。ChannelStats 是 LayerNorm 的可解释核心。
