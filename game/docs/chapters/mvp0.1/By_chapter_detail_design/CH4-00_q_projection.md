# Chapter 4-0 Q Projection: QLinear

## 1. 组件真实用途

QLinear 把 hidden state 投影成 query。Query 表示“当前位置要去问什么问题”：

```text
q = Linear(hidden, Wq, bq)
q: float32[B,H=1,T,D]
```

本关不是再造 Linear，而是复用 `component.linear.v1`，再加上 query role 和 head axis。

## 2. 前置组件

- `component.linear.v1`
- `component.axis_tensor.v1`

本关禁止使用预制 `QLinear`。

## 3. 本关新增能力

- `component.linear.v1`：执行 hidden `[B,T,C]` 到 `[B,T,D]` 的投影。
- `HeadAxisLift`：插入单头轴 `H=1`，得到 `[B,H,T,D]`。
- `ProjectionRoleTag`：标记输出 role 为 `query`。
- `RoleProbe`：说明 query 将作为 QKScore 的 left 输入。
- `ReferenceChecker`：检查数值等于 Linear(hidden,Wq,bq)。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=2,C=3]
Wq[C=3,D=2]
bq[D=2]

linear_out: [B=1,T=2,D=2]
q: [B=1,H=1,T=2,D=2]
```

RoleProbe 展示：

```text
q[0,0,token_i,:] 会和所有 key token 做 dot。
```

## 5. 初始错误图

画布给出：

- `hidden: HiddenSource[B,T,C]`
- `wq: WeightPlate[C,D]`
- `bq: BiasVector[D]`
- `q_out: ProjectionContract`
- `role_probe: RoleProbe`
- `reference: ReferenceChecker`

缺少 Linear、HeadAxisLift 和 role 标注。

## 6. 目标内部实现

```text
hidden.out -> linear.hidden
wq.out -> linear.weight
bq.out -> linear.bias
linear.out -> head_lift.x
head_lift.out -> role_tag.x
role_tag.out -> q_out.x
role_tag.out -> role_probe.x
q_out.out -> reference.x
```

其中：

- `linear.moduleId = component.linear.v1`
- `head_lift.axis = H`
- `role_tag.role = query`
- `q_out.expectedAxes = [B,H,T,D]`

## 7. 玩家操作

1. 拖入 `Linear v1`。
2. 接入 hidden、Wq、bq。
3. 拖入 `HeadAxisLift`，添加 `H=1`。
4. 拖入 `ProjectionRoleTag`，选择 `query`。
5. 接入合约、RoleProbe 和 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- 使用 K 或 V 权重：shape 一样，但 reference 数值失败。
- 不加 H 轴：后续 QKScore 接口不匹配。
- role 标成 key：shape 对，语义错，RoleProbe/hidden role 测试失败。
- 直接复制 hidden：D 轴不是 C 轴，数值和 shape 都不对。
- 使用预制 QLinear：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须使用 `component.linear.v1`。
- 必须存在 `HeadAxisLift`。
- role 必须是 `query`。
- 输出 axes 为 `[B,H,T,D]`。
- 输出 allclose 到 `Linear(hidden,Wq,bq)` 后插入 H 轴。

## 10. Hidden / Mutation 测试

Hidden case A：`C != D`。

```text
hidden: [B=2,T=3,C=4]
Wq: [C=4,D=3]
```

Hidden case B：Q/K 权重刻意不同。

```text
Wq != Wk
```

这会抓出把 KLinear 当 QLinear 复用的错误。

## 11. 认证后接口

```text
component.q_linear.v1
inputs:
  hidden: float32[B,T,C]
  weight: float32[C,D]
  bias: float32[D]
output:
  q: float32[B,H,T,D]
```

## 12. 后续调用

QKScore 的 left 输入必须是 query。QLinear 的重点是“同一个 Linear 计算结果，带上了不同语义角色”，这为 K/V projection 做铺垫。
