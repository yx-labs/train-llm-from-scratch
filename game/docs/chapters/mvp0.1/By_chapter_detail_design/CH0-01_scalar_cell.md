# Chapter 0-1 Scalar Cell

## 1. 组件真实用途

ScalarCell 是 MVP0.1 的第一个可用组件。它把一个原始数值 literal 包装成有 dtype 和 shape 合约的 `float32[]`。

本关的认知目标很窄：

```text
Scalar = 一个有限数字，没有行、没有列、没有长度
shape  = []
```

不要提前引入 residual、loss、scale 等后续场景；这里只证明什么是合格标量。

## 2. 前置能力

- 已完成 `Wire & Probe`，知道如何连接 source -> contract -> reference。

本关不允许使用 `component.scalar_cell.v1`。玩家要从 literal primitive 开始构造它。

## 3. 本关新增节点

- `Float32Literal`：玩家组件内部的原始数值来源。
- `OutputContractGate`：输出合约，要求 rank-0。
- `ReferenceChecker`：测试探针。

`Float32Literal` 支持玩家编辑值。输入 `0.6`、`42`、`-3.5` 都应该通过；输入文本或 NaN/Infinity 应报错。

## 4. 具体案例

Visible case 使用起始示例：

```text
value = 0.5
dtype = float32
axes  = []
dims  = []
```

`0.5` 只是示例，不是唯一答案。只要值是有限数字且输出 shape 为 `[]`，就应该通过当前任务。

## 5. 初始错误图

画布给出：

- `scalar_out: OutputContractGate`
- `reference: ReferenceChecker`

缺少 `scalar_source: Float32Literal`，也没有连接。

## 6. 目标内部实现

```text
scalar_source.out -> scalar_out.x
scalar_out.out -> reference.x
```

其中：

- `scalar_source.moduleId = Float32Literal`
- `scalar_source.value` 可由玩家编辑
- `scalar_out.expectedAxes = []`

## 7. 玩家操作

1. 从组件库拖入 `Float32Literal`。
2. 在 literal 中输入一个有限数字。
3. 连接 `scalar_source.out -> scalar_out.x`。
4. 连接 `scalar_out.out -> reference.x`。
5. 运行“检查当前任务”。
6. 在公开认证里换另一个有限数字并提交认证。

## 8. 错误路径

- 输入 `"abc"`：`Float32Literal` 报 `nan_inf`，不能把文本当 float32。
- 输出不是 rank-0：合约失败。
- 直接操作 reference：reference 只是测试设备，不是实现的一部分。
- 当前值必须等于 0.5：这是错误设计；本关检查本质，不检查示例值。

## 9. 测试设计

当前任务：

- `scalar_out` 必须输出 `float32[]`。
- 不做 allclose 到 0.5，避免把示例值误当唯一答案。

认证：

- 公开认证让玩家输入另一个有限数字，例如 `0.6`。
- 系统变体继续检查有限 rank-0 输出。
- 非数字公开值必须 blocked，而不是回退默认值。

## 10. 认证后接口

```text
component.scalar_cell.v1
output:
  scalar: float32[]
```

## 11. 后续调用

VectorRail 会把多个 `ScalarCell` 放到 C 轴上。ScalarCell 可用后，玩家第一次感受到“构造 -> 检查 -> 认证 -> 可用组件”的生命周期。
