# Chapter 0-7 TensorBox

## 1. 组件真实用途

TensorBox 把多个 token 向量组织成 rank-3 数据结构 `float32[B,T,C]`。它是模型输入 hidden state 的最小形态。

本关重点不是“tensor 是多维数组”这句抽象定义，而是让玩家看到：

```text
两个 vector[C] 进入 T 轴
B 表示 batch
C 必须留在最后一轴
```

## 2. 前置组件

- `component.vector_rail.v1`
- `component.matrix_struct.v1`

本关不允许使用 `component.tensor_box.v1`。玩家使用 `TensorBox` primitive 构建可用组件。

## 3. 本关新增能力

- `InputTensor` 提供两个 token vector。
- `TensorBox` 把 token rows 组装为 `[B,T,C]`。
- `OutputContractGate` 检查 B/T/C 轴顺序。
- `ReferenceChecker` 检查具体布局。

## 4. 具体案例

Visible case:

```text
t0[C] = [ 1, 0, 2]
t1[C] = [-1, 3, 0.5]

tensor[B=1,T=2,C=3]
  B0,T0 = [ 1, 0, 2]
  B0,T1 = [-1, 3, 0.5]
```

输出合约：

```text
dtype = float32
axes  = [B,T,C]
dims  = [1,2,3]
```

## 5. 初始错误图

画布给出：

- `t0: InputTensor`
- `t1: InputTensor`
- `tensor_out: OutputContractGate`
- `reference: ReferenceChecker`

缺少 `tensor: TensorBox`。

## 6. 目标内部实现

```text
t0.out -> tensor.t0
t1.out -> tensor.t1
tensor.out -> tensor_out.x
tensor_out.out -> reference.x
```

其中：

- `tensor.moduleId = TensorBox`
- `tensor_out.expectedAxes = [B,T,C]`

## 7. 玩家操作

1. 拖入 `TensorBox`。
2. 将 `t0` 接入第一条 token row。
3. 将 `t1` 接入第二条 token row。
4. 接合约和 reference。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- 把两个 token 拼成 `[C=6]`：不是 rank-3 tensor。
- 输出 `[T,B,C]`：shape 数量类似，但轴语义错。
- 把 C 放到中间：后续 MatMulGate 无法消费最后一维 C。
- t0/t1 反接：shape 通过，但 reference 数值失败。

## 9. 测试设计

当前任务：

- `tensor_out` 必须是 `float32[B=1,T=2,C=3]`。
- 输出值必须与 reference 完全对应。

Hidden / certification：

- hidden 改变 C 宽度为 4，但 B/T 语义保持。
- 公开认证选择 C 宽度和 seed，系统生成两个 token vector。
- 认证只改变输入 vector/reference，不改变玩家图。

## 10. 认证后接口

```text
component.tensor_box.v1
inputs:
  t0: float32[C]
  t1: float32[C]
output:
  tensor: float32[B,T,C]
```

## 11. 后续调用

Linear、Q/K/V projection、Embedding output 都会使用 `[B,T,C]`。TensorBox 是玩家第一次把“序列位置 T”和“通道 C”区分开。
