# Chapter 0-2 Scalar Add: AddScalar

## 1. 组件真实用途

AddScalar 是玩家第一次把两个已经合格的标量组合成新标量。它的作用很小，但设计目标很关键：让玩家知道“组件可用”以后可以继续进入下一张图，而不是每次都重新输入裸数字。

```text
out = left + right
```

本关不允许拖入预制 `AddScalar`。玩家要用两个 `component.scalar_cell.v1`、一个标量加法门和输出合约，构造出可复用的 `component.add_scalar.v1`。

## 2. 前置组件

- `component.scalar_cell.v1`

ScalarCell 已经保证输入是有限 float32 rank-0 标量。本关复用它，而不是让裸数字直接参与计算。

## 3. 本关新增能力

- `component.scalar_cell.v1`：提供两个合格 rank-0 输入。
- `ScalarAddGate`：只接受两个 rank-0 float32，输出 rank-0 float32。
- `ScalarOutputContract`：检查输出仍然是标量。
- `ReferenceChecker`：检查数值等于当前案例的 `left + right`。

`ReferenceChecker` 是探针，不是目标组件的一部分。它的作用是证明玩家没有只满足 shape，而是真的完成加法。

## 4. 具体案例

Visible case 是一个很小的训练旋钮合成：

```text
base_gain = 0.50
correction = 0.10
expected = 0.60
```

玩家应该看到：0.6 不是“必须等于示例值”的答案，而是两个当前输入相加后的结果。认证时数值会变化。

## 5. 初始错误图

画布给出：

- `left_source: Float32Literal`
- `right_source: Float32Literal`
- `scalar_out: ScalarOutputContract`
- `reference: ReferenceChecker`

两个 source 默认没有被封装成 ScalarCell，也没有加法门。直接把 source 接到合约不能通过结构断言。

## 6. 目标内部实现

```text
left_source.out -> left_cell.x
right_source.out -> right_cell.x
left_cell.out -> add.left
right_cell.out -> add.right
add.out -> scalar_out.x
scalar_out.out -> reference.x
```

其中：

- `left_cell.moduleId = component.scalar_cell.v1`
- `right_cell.moduleId = component.scalar_cell.v1`
- `add.moduleId = ScalarAddGate`

## 7. 玩家操作

1. 拖入两个 `ScalarCell v1`。
2. 拖入 `ScalarAddGate`。
3. 分别把两个原始数值接进 ScalarCell。
4. 将两个 ScalarCell 输出接进加法门。
5. 将加法结果接入合约和 reference。
6. 修改任意输入值，确认当前任务重新按新值计算。
7. 点击“检查当前任务”，再提交认证。

## 8. 错误路径

- 裸数字直接接加法：结构断言失败，因为输入没有被认证成 ScalarCell。
- 裸数字直接接合约：当前案例看似 rank-0，但没有证明“可复用组件”。
- 只接 left 或 right：合约报缺失输入。
- 输入 `"0.5"` 文本：ScalarCell 报 dtype 错误。
- 硬编码输出 `0.6`：visible case 通过不了变体，hidden allclose 失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在两个 `component.scalar_cell.v1` 节点。
- 必须存在 `ScalarAddGate`。
- 加法门输出必须经过 `ScalarOutputContract`。
- 输出 dtype 为 `float32`，shape 为 `[]`。
- 输出数值 allclose 到 `left + right`。

## 10. Hidden / Mutation 测试

Hidden case A：负数和零。

```text
left = -0.25
right = 0.25
expected = 0.0
```

Hidden case B：不同有限 float32。

```text
left = 8.5
right = 12.25
expected = 20.75
```

这里没有“不能超过 10”的规则。只要是有限 float32 标量，就应该被接受。

Hidden case C：非 float 输入。

```text
left = "0.5"
right = 0.1
```

认证必须失败，并显示 dtype 错误，而不是把文本偷偷解析成数字。

## 11. 认证后接口

```text
component.add_scalar.v1
inputs:
  left: float32[]
  right: float32[]
output:
  out: float32[]
```

## 12. 后续调用

Bias、scale、loss mean 等关卡都会复用“两个同形数值相加”的概念。AddScalar 不是为了加 0.5 + 0.1，而是为了让玩家建立：组件输出可以继续成为下一个组件的输入。
