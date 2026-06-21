# Chapter 0-6 MatMulGate

## 1. 组件真实用途

MatMulGate 是第一条完整数值弧线的核心计算门。它把左侧 `hidden[B,T,C]` 和右侧 `weight[C,O]` 相乘，输出 `projected[B,T,O]`。

本关要让玩家理解 MatMul 的轴合约：

```text
left last axis C  ==  right first axis C
C 被消耗
O 成为输出通道
B/T 原样保留
```

## 2. 前置组件

- `component.tensor_box.v1`
- `component.matrix_struct.v1`

本关不允许使用 `component.matmul_gate.v1`。玩家构建的就是这个组件；可用模块是 `MatMulGate` primitive。

## 3. 本关新增能力

- `InputTensor` 提供 `hidden[B,T,C]`。
- `WeightPlate` 提供 `weight[C,O]`。
- `MatMulGate` 执行真实矩阵乘法。
- `OutputContractGate` 检查输出轴。
- `ReferenceChecker` 检查数值。

## 4. 具体案例

Visible case:

```text
hidden[B=1,T=2,C=3]
  token0 = [ 1, 0, 2]
  token1 = [-1, 3, 0.5]

weight[C=3,O=2]
  C0 = [ 0.4, -0.2]
  C1 = [ 1.1,  0.3]
  C2 = [-0.7,  0.8]
```

输出：

```text
projected[token0] = [-1.0, 1.4]
projected[token1] = [ 2.55, 1.5]
```

输出合约：

```text
float32[B,T,O] = [1,2,2]
```

## 5. 初始错误图

画布给出：

- `hidden: InputTensor`
- `weight: WeightPlate`
- `matmul_out: OutputContractGate`
- `reference: ReferenceChecker`

缺少 `matmul: MatMulGate`。

## 6. 目标内部实现

```text
hidden.out -> matmul.left
weight.out -> matmul.right
matmul.out -> matmul_out.x
matmul_out.out -> reference.x
```

其中：

- `matmul.moduleId = MatMulGate`
- `matmul_out.expectedAxes = [B,T,O]`

## 7. 玩家操作

1. 拖入 `MatMulGate`。
2. 将 hidden 接到 left。
3. 将 weight 接到 right。
4. 将 matmul 输出接到合约，再接 reference。
5. 检查当前任务并提交认证。

## 8. 错误路径

- left/right 反接：`weight[C,O] @ hidden[B,T,C]` 不符合当前合约。
- weight 轴写成 `[O,C]`：inner dim 位置错。
- 直接 `hidden -> matmul_out`：输出仍是 `[B,T,C]`，没有 O。
- 只检查 shape：可能漏掉 operand 顺序错误；必须 allclose reference。

## 9. 测试设计

当前任务：

- `matmul_out` 必须是 `float32[B=1,T=2,O=2]`。
- 输出必须 allclose 到 reference。

Hidden / certification：

- hidden 使用 `[B=2,T=3,C=4]`。
- weight 使用 `[C=4,O=3]`。
- 输出必须变成 `[B=2,T=3,O=3]`。
- 公开认证让玩家选择 B/T/C/O 和 seed；系统生成输入和 reference。

## 10. 认证后接口

```text
component.matmul_gate.v1
inputs:
  left: float32[...,C]
  right: float32[C,O] or float32[...,C,O]
output:
  out: float32[...,O]
```

## 11. 后续调用

Linear 会复用 `component.matmul_gate.v1`，QKScore 也会复用它。MatMulGate 是从基础数据结构走向可组合模型层的关键门。
