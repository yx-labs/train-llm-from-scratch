# Chapter 0-5 MatrixStruct

## 1. 组件真实用途

MatrixStruct 把两条 `vector[C]` 组织成一个二维权重结构 `float32[C,O]`。在当前 MVP0.1 弧线里，它是 MatMulGate 的右操作数来源。

本关认知重点：

```text
vector[C] + vector[C] 不是拼成长向量
它们成为 matrix[C,O] 的两个 O 列
```

## 2. 前置组件

- `component.vector_rail.v1`
- `component.dot_product.v1`

本关不允许使用 `component.matrix_struct.v1`。玩家使用的是 `MatrixStruct` primitive，目标是构建可用的 matrix 组件。

## 3. 本关新增能力

- `InputTensor` 提供两个 `vector[C]` 输入。
- `MatrixStruct` 将输入解释为 O0/O1 两列。
- `OutputContractGate` 检查 `[C,O]`。
- `ReferenceChecker` 检查列顺序和数值布局。

## 4. 具体案例

Visible case:

```text
col0[C] = [ 0.4, 1.1, -0.7]
col1[C] = [-0.2, 0.3,  0.8]

matrix[C,O] =
  C0: [ 0.4, -0.2]
  C1: [ 1.1,  0.3]
  C2: [-0.7,  0.8]
```

输出合约：

```text
dtype = float32
axes  = [C,O]
dims  = [3,2]
```

## 5. 初始错误图

画布给出：

- `col0: InputTensor`
- `col1: InputTensor`
- `matrix_out: OutputContractGate`
- `reference: ReferenceChecker`

缺少 `matrix: MatrixStruct`，没有任何边。

## 6. 目标内部实现

```text
col0.out -> matrix.o0
col1.out -> matrix.o1
matrix.out -> matrix_out.x
matrix_out.out -> reference.x
```

其中：

- `matrix.moduleId = MatrixStruct`
- `matrix_out.expectedAxes = [C,O]`

## 7. 玩家操作

1. 拖入 `MatrixStruct`。
2. 把 `col0` 接到 `matrix.o0`。
3. 把 `col1` 接到 `matrix.o1`。
4. 连接输出合约和 reference。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- 将两个 vector 拼成 `[C=6]`：rank 和 axis 都错误。
- 把 col0/col1 接反：shape 仍正确，但 reference 数值失败。
- 把 `C` 和 `O` 轴写成 `[O,C]`：后续 MatMulGate 会消费错误内维。
- 直接接 `col0 -> matrix_out`：shape 是 `[C]`，不是 `[C,O]`。

## 9. 测试设计

当前任务：

- `matrix_out` 必须是 `float32[C=3,O=2]`。
- 输出必须 allclose 到 reference，确保列顺序正确。

Hidden / certification：

- hidden 将 C 从 3 变成 4，但 O 仍为 2。
- 公开认证选择 C 宽度和 seed，系统生成两个 vector 列。
- 认证只改变输入列和 reference，不改变玩家图。

## 10. 认证后接口

```text
component.matrix_struct.v1
inputs:
  o0: float32[C]
  o1: float32[C]
output:
  matrix: float32[C,O]
```

## 11. 后续调用

MatMulGate 会把 `hidden[B,T,C]` 与 `weight[C,O]` 相乘。MatrixStruct 必须保证 C 是第一轴、O 是第二轴，否则后续投影语义会错。
