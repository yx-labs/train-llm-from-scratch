# Chapter 5-1 Parallel Heads: ParallelHeads

## 1. 组件真实用途

ParallelHeads 对每个 head 独立运行 attention：

```text
context[:,h,:,:] = SingleHeadAttention(q[:,h,:,:], k[:,h,:,:], v[:,h,:,:])
```

每个 head 共享流程，但不共享 Q/K/V 数据。

## 2. 前置组件

- `component.head_split.v1`
- `component.single_head_attention.v1`

本关复用单头组件，但必须把 H 轴保留下来。

## 3. 本关新增能力

- `HeadMapGate`：沿 H 轴应用单头 attention。
- `PerHeadMaskBroadcast`：把 causal mask 广播到每个 head。
- `HeadIsolationProbe`：显示 head 0/1 的不同输出。
- `ReferenceChecker`：检查每个 head 的独立 reference。

## 4. 具体案例

Visible case:

```text
q/k/v: float32[B=1,H=2,T=3,D=2]
mask: bool[T,T]
out: float32[B=1,H=2,T=3,D=2]
```

HeadIsolationProbe 展示：head 0 的 value 被改变时，head 1 输出不应变化。

## 5. 初始错误图

画布给出 q/k/v heads、causal mask、`heads_out`、isolation probe、reference。缺少 HeadMap 和 mask broadcast。

## 6. 目标内部实现

```text
mask.out -> mask_broadcast.mask
q.out -> head_map.q
k.out -> head_map.k
v.out -> head_map.v
mask_broadcast.out -> head_map.mask
head_map.out -> heads_out.x
head_map.out -> isolation_probe.x
heads_out.out -> reference.x
```

## 7. 玩家操作

1. 拖入 `PerHeadMaskBroadcast`。
2. 拖入 `HeadMapGate`，选择内部组件 `SingleHeadAttention v1`。
3. 接入 q/k/v 与 mask。
4. 连接 isolation probe 和 reference。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- 把 H 合并到 B：shape 可能可执行，但 head 身份丢失。
- 所有 head 共用 head 0 的数据：hidden isolation 失败。
- mask 未广播到每个 head：H>1 失败。
- reduce 掉 H 轴：输出不能 concat。
- 使用预制 MultiHeadAttention：shortcut。

## 9. 测试设计

- Visible：H=2，两个 head 的 reference 分别检查。
- Hidden A：H=3。
- Hidden B：只修改 head 1 value，head 0 输出必须不变。
- Hidden C：T=1，mask broadcast 不应引入错误。

## 10. 认证后接口

```text
component.parallel_heads.v1
inputs:
  q: float32[B,H,T,D]
  k: float32[B,H,T,D]
  v: float32[B,H,T,D]
  mask: bool[T,T]
output:
  context_heads: float32[B,H,T,D]
```

## 11. 后续调用

HeadConcat 会把各 head 的 D 重新拼成 C。ParallelHeads 必须保持 H 轴，不能提前合并。
