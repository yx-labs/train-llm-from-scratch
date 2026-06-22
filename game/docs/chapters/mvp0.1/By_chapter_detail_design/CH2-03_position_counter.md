# Chapter 2-3 Position Counter: PositionIds[T]

## 1. 组件真实用途

PositionCounter 根据序列长度 T 生成位置 id：

```text
T = 5
position_ids = [0,1,2,3,4]
```

它让模型知道 token 在上下文窗口里的位置。

## 2. 前置组件

- `component.token_buffer.v1`

TokenBuffer 提供 capacity T 和 valid length。本关只构造固定窗口的位置序列。

## 3. 本关新增能力

- `LengthScalar`：输入 T。
- `RangeGate`：生成 `0..T-1`。
- `IntDTypeGate`：确保输出是 int。
- `PositionAxisContract`：绑定 axis `[T]`。
- `PositionPreviewProbe`：显示首尾位置。

## 4. 具体案例

Visible case:

```text
T = 5
expected = [0,1,2,3,4]
```

输出 shape 为 `[T]`，不是 `[B,T]`。

## 5. 初始错误图

画布给出：

- `length: LengthScalar`
- `pos_out: PositionAxisContract`
- `preview: PositionPreviewProbe`
- `reference: ReferenceChecker`

缺少 RangeGate 和 dtype gate。

## 6. 目标内部实现

```text
length.out -> range.t
range.out -> dtype_gate.x
dtype_gate.out -> pos_out.x
dtype_gate.out -> preview.x
pos_out.out -> reference.x
```

## 7. 玩家操作

1. 查看 T 的当前值。
2. 拖入 `RangeGate`。
3. 拖入 `IntDTypeGate`。
4. 连接 PositionAxisContract、preview 和 reference。
5. 修改 T，确认 range 自动变化。
6. 检查当前任务并提交认证。

## 8. 错误路径

- 从 1 开始计数：reference 失败。
- 硬编码 `[0,1,2,3,4]`：T 变体失败。
- 输出 float：dtype 错。
- 加 batch 轴：PositionEmbedding lookup 期望 `[T]`。
- 输出倒序：reference 失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 RangeGate。
- 输出 dtype 为 int。
- axes 为 `[T]`。
- 数值为 `0..T-1`。

## 10. Hidden / Mutation 测试

Hidden case A：`T=1`。

```text
position_ids = [0]
```

Hidden case B：`T=8`。

必须自动生成 8 个位置。

Hidden case C：非法 T。

```text
T = 0 或 T < 0
```

必须失败并提示上下文长度非法。

## 11. 认证后接口

```text
component.position_ids.v1
inputs:
  length: int[]
output:
  position_ids: int[T]
```

## 12. 后续调用

PositionEmbedding 会用 position_ids 查表。PositionCounter 不关心 token 内容，只关心槽位索引。
