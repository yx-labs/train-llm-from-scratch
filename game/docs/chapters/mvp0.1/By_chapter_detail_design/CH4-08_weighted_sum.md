# Chapter 4-8 Weighted Sum: AttentionApply

## 1. 组件真实用途

WeightedSum 用 attention probabilities 读取 value：

```text
context[b,h,i,d] = sum_j prob[b,h,i,j] * v[b,h,j,d]
```

它是 attention 真正“看向过去 token 并取出内容”的一步。

## 2. 前置组件

- `component.softmax_last_dim.v1`
- `component.v_linear.v1`
- `component.matmul_gate.v1`

本关复用 MatMulGate，但要让玩家正确对齐 key token 轴 `j`。

## 3. 本关新增能力

- `AttentionProbContract`：检查 prob axes 为 `[B,H,Tq,Tk]`。
- `ValueContract`：检查 value axes 为 `[B,H,Tk,D]`。
- `component.matmul_gate.v1`：执行 `prob @ value`。
- `ContextContract`：检查输出 `[B,H,Tq,D]`。
- `MixTrace`：展示一个 context cell 是多个 value cell 的加权和。

## 4. 具体案例

Visible case:

```text
prob[B=1,H=1,Tq=2,Tk=3]
v[B=1,H=1,Tk=3,D=2]
```

对于 query token 1：

```text
prob[1,:] = [0.2, 0.3, 0.5]
v[:,0] = [10, 20, 40]

context[1,0] = 0.2*10 + 0.3*20 + 0.5*40 = 28
```

## 5. 初始错误图

画布给出：

- `prob: AttentionProbSource`
- `value: ValueSource`
- `context_out: ContextContract`
- `mix_trace: MixTrace`
- `reference: ReferenceChecker`

缺少 MatMulGate 和两个输入合约。

## 6. 目标内部实现

```text
prob.out -> prob_contract.x
value.out -> value_contract.x
prob_contract.out -> matmul.left
value_contract.out -> matmul.right
matmul.out -> context_out.x
matmul.out -> mix_trace.context
prob_contract.out -> mix_trace.prob
value_contract.out -> mix_trace.value
context_out.out -> reference.x
```

其中：

- `matmul.moduleId = component.matmul_gate.v1`
- `prob_contract.expectedAxes = [B,H,Tq,Tk]`
- `value_contract.expectedAxes = [B,H,Tk,D]`
- `context_out.expectedAxes = [B,H,Tq,D]`

## 7. 玩家操作

1. 拖入 `AttentionProbContract` 和 `ValueContract`。
2. 拖入已认证 `MatMulGate v1`。
3. 将 prob 接 left，value 接 right。
4. 将 matmul 输出接 ContextContract、MixTrace 和 reference。
5. 检查一个 context cell 的加权和。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- prob 和 value 接反：inner axis 不匹配，或输出 axis 错。
- 逐元素相乘不 reduce：shape 不会得到 `[B,H,Tq,D]`。
- 把 Tq 当 Tk：当自注意力 Tq=Tk 时容易混过，cross hidden 失败。
- 忽略 ValueContract：可能把 key tensor 接成 value。
- 只过 shape：必须 allclose 到 weighted sum reference。

## 9. Visible 测试

Visible 测试要求：

- 必须使用 `component.matmul_gate.v1`。
- 必须经过 prob/value 两个 contract。
- 输出 axes 为 `[B,H,Tq,D]`。
- 输出 allclose 到 `sum_j prob[i,j] * value[j,d]`。
- MixTrace 必须接入 prob、value、context。

## 10. Hidden / Mutation 测试

Hidden case A：cross attention shape。

```text
Tq=2
Tk=3
```

这个 case 防止把 Tq/Tk 混为一个 T。

Hidden case B：one-hot probability。

```text
prob row = [0,1,0]
context row 应等于 value[1,:]
```

Hidden case C：multi-head carrier。

```text
B=2,H=2,Tq=3,Tk=3,D=4
```

认证检查 B/H 不能被 reduce。

## 11. 认证后接口

```text
component.attention_apply.v1
inputs:
  prob: float32[B,H,Tq,Tk]
  value: float32[B,H,Tk,D]
output:
  context: float32[B,H,Tq,D]
```

## 12. 后续调用

AttentionHead 会把 Q/K/V、score、scale、mask、softmax 和 WeightedSum 串起来。WeightedSum 是玩家看到 attention 产生新 hidden 内容的第一步。
