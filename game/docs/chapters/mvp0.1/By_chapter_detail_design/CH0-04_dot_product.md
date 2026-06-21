# Chapter 0-4 DotProduct

## 1. 组件真实用途

DotProduct 是第一个必须证明“真的计算了”的数值组件。它把两个同长度 `vector[C]` 变成一个 rank-0 标量分数：

```text
score = sum_i query[i] * key[i]
```

后续 `MatMulGate`、attention score、QKScore 都可以看成许多 dot product cell 的组合。

## 2. 前置组件

- `component.scalar_cell.v1`
- `component.vector_rail.v1`

本关不允许使用 `component.dot_product.v1`。玩家要用基础 primitive 搭出它。

## 3. 本关新增能力

玩家第一次经历从“shape 合约”升级到“行为合约”：

- `ElementwiseMultiply` 负责逐 C 单元相乘。
- `SumReduce(axis=C)` 负责移除 C 轴并得到一个标量。
- `OutputContractGate` 只检查输出接口。
- `ReferenceChecker` 用数值 reference 证明行为正确。

## 4. 具体案例

Visible case:

```text
query = [0.2, -0.5, 1.0]
key   = [0.4,  0.1, -0.3]

score = 0.2*0.4 + (-0.5)*0.1 + 1.0*(-0.3)
      = -0.27
```

输出合约：

```text
float32[]
axes = []
```

## 5. 初始错误图

画布只给出：

- `query: InputTensor`
- `key: InputTensor`
- `dot_out: OutputContractGate`
- `reference: ReferenceChecker`

没有 `multiply`，没有 `sum`，也没有任何线。玩家看到的是一个缺少内部实现的组件蓝图。

## 6. 目标内部实现

目标不是拖一个 DotProduct 节点，而是搭出内部子图：

```text
query.out -> multiply.left
key.out -> multiply.right
multiply.out -> sum.x
sum.out -> dot_out.x
dot_out.out -> reference.x
```

其中：

- `multiply.moduleId = ElementwiseMultiply`
- `sum.moduleId = SumReduce`
- `sum.axis = C`

## 7. 玩家操作

1. 从组件库拖入 `ElementwiseMultiply`。
2. 从组件库拖入 `SumReduce`。
3. 将 `query` 和 `key` 接到 multiply 的左右输入。
4. 将 multiply 输出接到 sum。
5. 将 sum 输出接到 `dot_out`。
6. 将 `dot_out` 输出接到 `reference`。
7. 运行“检查当前任务”，再运行“提交认证”。

## 8. 错误路径

- 直接 `query.out -> dot_out.x`：shape 是 `[C]`，不是标量；路径断言也会失败。
- 只做 multiply 不做 sum：输出仍然带 C 轴，contract fail。
- 只取某一个 C 单元：visible 可能看起来接近，但 hidden C=5 的数值 reference 会失败。
- query/key 长度不一致：`ElementwiseMultiply` 报 shape mismatch。

## 9. Visible 测试

测试包含两类断言：

- 结构断言：必须存在 `ElementwiseMultiply` 和 `SumReduce`，且数据路径必须经过它们。
- 行为断言：`dot_out` 必须是 `float32[]`，并且 allclose 到 reference。

## 10. Hidden / Mutation 测试

Hidden case 将 C 从 3 变成 5：

```text
query = [0.4, -0.2, 1.1, 0.75, -0.6]
key   = [-0.5, 0.25, 0.9, -1.2, 0.3]
```

公开认证允许玩家选择 C 宽度和 seed。系统只改变输入 `query/key/reference`，不改变玩家图中的 `multiply`、`sum` 或 `dot_out` 参数。

## 11. Pack 后接口

认证后得到：

```text
component.dot_product.v1
inputs:
  query: float32[C]
  key: float32[C]
output:
  score: float32[]
```

## 12. 后续调用

下一阶段的 `MatrixStruct` / `MatMulGate` 要把这个思维扩展成许多 cell：

```text
output[row, col] = DotProduct(left[row, :], right[:, col])
```

因此 DotProduct 是从单个数值 cell 走向矩阵乘法的关键中间层。
