# Chapter 4-2 V Projection: VLinear

## 1. 组件真实用途

VLinear 把 hidden state 投影成 value。Value 是 attention 最后要混合的内容：

```text
v = Linear(hidden, Wv, bv)
v: float32[B,H=1,T,D]
```

Q 和 K 参与打分，V 不参与 QKScore。它会在 WeightedSum 里被 attention probabilities 加权求和。

## 2. 前置组件

- `component.linear.v1`
- `component.q_linear.v1`
- `component.k_linear.v1`

本关复用 Linear，但强调 value role 与 Q/K role 的区别。

## 3. 本关新增能力

- `component.linear.v1`：投影 hidden。
- `HeadAxisLift`：补出单头轴。
- `ProjectionRoleTag`：标记 role 为 `value`。
- `ValueProbe`：展示 value 是被加权汇聚的内容向量。
- `ReferenceChecker`：检查使用 Wv/bv。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=3,C=3]
Wv[C=3,D=2]
bv[D=2]
v: [B=1,H=1,T=3,D=2]
```

ValueProbe 展示：

```text
context[token_i,:] = sum_j attention[i,j] * v[token_j,:]
```

这里只解释 V 的去向，不要求玩家已经会 WeightedSum。

## 5. 初始错误图

画布给出：

- `hidden: HiddenSource[B,T,C]`
- `wv: WeightPlate[C,D]`
- `bv: BiasVector[D]`
- `v_out: ProjectionContract`
- `value_probe: ValueProbe`
- `reference: ReferenceChecker`

缺少 Linear、HeadAxisLift 和 value role。

## 6. 目标内部实现

```text
hidden.out -> linear.hidden
wv.out -> linear.weight
bv.out -> linear.bias
linear.out -> head_lift.x
head_lift.out -> role_tag.x
role_tag.out -> v_out.x
role_tag.out -> value_probe.x
v_out.out -> reference.x
```

其中：

- `role_tag.role = value`
- `v_out.expectedAxes = [B,H,T,D]`

## 7. 玩家操作

1. 拖入 `Linear v1`。
2. 接入 hidden、Wv、bv。
3. 插入 `HeadAxisLift`。
4. 用 `ProjectionRoleTag` 标记 value。
5. 接入 ValueProbe、合约和 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- 复用 Wk 或 Wq：shape 对，数值错。
- role 标成 key：WeightedSum 不接受 key role。
- 提前做 QKScore 或 softmax：VLinear 只负责生成 V。
- 不加 H 轴：后续 WeightedSum 接口不匹配。
- 使用预制 VLinear：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须使用 `component.linear.v1`。
- role 必须是 `value`。
- 输出 axes 必须是 `[B,H,T,D]`。
- 输出 allclose 到 `Linear(hidden,Wv,bv)` 后插入 H 轴。
- ValueProbe 必须能显示每个 token 的 value 向量。

## 10. Hidden / Mutation 测试

Hidden case A：V 参数与 Q/K 全部不同。

```text
Wv != Wq
Wv != Wk
```

Hidden case B：value 内容符号变化。

```text
hidden seed changes sign pattern
```

这个 case 防止玩家只复制 key 或 query。

## 11. 认证后接口

```text
component.v_linear.v1
inputs:
  hidden: float32[B,T,C]
  weight: float32[C,D]
  bias: float32[D]
output:
  v: float32[B,H,T,D]
```

## 12. 后续调用

WeightedSum 会消费 attention probabilities 和 V。VLinear 文档必须让玩家知道：V 是“被读出的内容”，不是“参与打分的地址”。
