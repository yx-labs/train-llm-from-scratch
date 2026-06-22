# Chapter 3-3 Activation: ReLU

## 1. 组件真实用途

ReLU 给 MLP 增加非线性：

```text
relu(x) = max(x, 0)
```

它逐元素处理 tensor，不改变 shape、axis 或 dtype。

## 2. 前置组件

- `component.linear.v1`
- `component.typed_tensor.v1`

玩家已经会构造线性输出。本关让玩家看到线性后为什么需要逐元素门。

## 3. 本关新增能力

- `ZeroScalar`：提供 float32 0。
- `BroadcastZero`：把 0 扩展到输入形状。
- `ElementwiseMaxGate`：逐元素取较大值。
- `ActivationContract`：检查输出 shape/axes 不变。
- `SignTraceProbe`：显示负数被截断、正数保留。

## 4. 具体案例

Visible case:

```text
x[B=1,T=2,O=3] =
  [[[-1.0, 0.0, 2.0],
    [ 3.0,-0.5, 1.0]]]
```

期望：

```text
relu(x) =
  [[[0.0,0.0,2.0],
    [3.0,0.0,1.0]]]
```

## 5. 初始错误图

画布给出：

- `x: LinearOutput[B,T,O]`
- `relu_out: ActivationContract`
- `sign_trace: SignTraceProbe`
- `reference: ReferenceChecker`

缺少 zero broadcast 和 max gate。

## 6. 目标内部实现

```text
zero.out -> broadcast_zero.small
x.out -> broadcast_zero.target
x.out -> max.left
broadcast_zero.out -> max.right
max.out -> relu_out.x
max.out -> sign_trace.y
relu_out.out -> reference.x
```

## 7. 玩家操作

1. 拖入 `ZeroScalar` 和 `BroadcastZero`。
2. 将 zero broadcast 到输入形状。
3. 拖入 `ElementwiseMaxGate`。
4. 将 x 与 broadcast zero 逐元素 max。
5. 接入 trace、contract 和 reference。
6. 检查当前任务并提交认证。

## 8. 错误路径

- 用 absolute value：负数变正，reference 失败。
- 只把负数置空但 dtype 变 bool：dtype 错。
- reduce 成一个最大值：shape 错。
- leaky ReLU：负数不是 0，reference 失败。
- 使用预制 ReLU：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 ElementwiseMaxGate。
- 输出 shape/axes 与输入完全一致。
- 负数输出 0，0 保持 0，正数保持原值。
- SignTraceProbe 能展示至少一个负数和一个正数。

## 10. Hidden / Mutation 测试

Hidden case A：全负数。

输出应全 0。

Hidden case B：全正数。

输出应等于输入。

Hidden case C：含极小负数。

```text
x = -0.0001
```

输出仍应为 0。

## 11. 认证后接口

```text
component.relu.v1
inputs:
  x: float32[...]
output:
  y: float32[...]
```

## 12. 后续调用

MLP 会在 up projection 和 down projection 之间使用 activation。ReLU 是 MVP0.1 的简化激活，未来可升级成 GELU。
