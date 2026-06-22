# Chapter 4-1 K Projection: KLinear

## 1. 组件真实用途

KLinear 把 hidden state 投影成 key。Key 表示“每个历史 token 提供什么可匹配的地址”：

```text
k = Linear(hidden, Wk, bk)
k: float32[B,H=1,T,D]
```

它和 QLinear shape 相同，但 role 和参数不同。这个关卡专门防止玩家把“形状相同”误认为“语义相同”。

## 2. 前置组件

- `component.linear.v1`
- `component.q_linear.v1`

玩家刚构造过 QLinear。本关复用同样的 Linear 机制，但必须使用 K 权重和 key role。

## 3. 本关新增能力

- `component.linear.v1`：投影 hidden。
- `HeadAxisLift`：补出 H 轴。
- `ProjectionRoleTag`：标记 role 为 `key`。
- `MemoryProbe`：展示 key 将作为可被查询的 token memory。
- `ReferenceChecker`：检查使用的是 Wk/bk。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=3,C=3]
Wk[C=3,D=2]
bk[D=2]
k: [B=1,H=1,T=3,D=2]
```

MemoryProbe 展示：

```text
k[0,0,j,:] 是第 j 个 token 被其他 token 查询时使用的 key。
```

## 5. 初始错误图

画布给出：

- `hidden: HiddenSource[B,T,C]`
- `wk: WeightPlate[C,D]`
- `bk: BiasVector[D]`
- `k_out: ProjectionContract`
- `memory_probe: MemoryProbe`
- `reference: ReferenceChecker`

缺少 Linear、HeadAxisLift 和 key role。

## 6. 目标内部实现

```text
hidden.out -> linear.hidden
wk.out -> linear.weight
bk.out -> linear.bias
linear.out -> head_lift.x
head_lift.out -> role_tag.x
role_tag.out -> k_out.x
role_tag.out -> memory_probe.x
k_out.out -> reference.x
```

其中：

- `role_tag.role = key`
- `k_out.expectedAxes = [B,H,T,D]`

## 7. 玩家操作

1. 拖入 `Linear v1`。
2. 接入 hidden、Wk、bk。
3. 插入 `HeadAxisLift`。
4. 使用 `ProjectionRoleTag` 标记 key。
5. 连接 MemoryProbe、合约和 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- 复用上一关 Wq：shape 对，reference 失败。
- role 仍然是 query：QKScore 结构断言失败。
- 忘记 H 轴：不能接入 QKScore。
- 把 key 转置提前做掉：KLinear 只负责生成 `[B,H,T,D]`，转置属于 QKScore。
- 使用预制 KLinear：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须使用 `component.linear.v1`。
- role 必须是 `key`。
- 输出 axes 必须是 `[B,H,T,D]`。
- 输出 allclose 到 `Linear(hidden,Wk,bk)` 后插入 H 轴。
- MemoryProbe 必须接到 key 输出。

## 10. Hidden / Mutation 测试

Hidden case A：Q/K 参数交换。

```text
Wq != Wk
bk != bq
```

Hidden case B：`T == D`。

```text
k: [B=1,H=1,T=3,D=3]
```

这个 case 防止玩家提前转置或靠 shape 混过去。

## 11. 认证后接口

```text
component.k_linear.v1
inputs:
  hidden: float32[B,T,C]
  weight: float32[C,D]
  bias: float32[D]
output:
  k: float32[B,H,T,D]
```

## 12. 后续调用

QKScore 会在内部转置 K 的最后两个轴。本关要明确：KLinear 输出仍然是 token-major 的 `[B,H,T,D]`，不要把 QKScore 的工作提前做掉。
