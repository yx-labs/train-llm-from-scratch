# Chapter 0-8 Typed Tensor: TypedTensor

## 1. 组件真实用途

TypedTensor 给 tensor 加上 dtype 合约。它解决的问题不是“这个数组有多大”，而是“这个数组的每个元素应该被解释成什么”。

在后续图里，以下数据不能混用：

```text
hidden: float32[B,T,C]
token_ids: int[B,T]
attention_mask: bool[B,T]
mask_bias: float32[B,T,T]
```

本关要让玩家把一个 raw literal 变成严格 typed tensor，并让错误类型被明确拒绝。

## 2. 前置组件

- `component.tensor_box.v1`

玩家已经知道 tensor 有 shape。本关在 shape 之外增加 dtype，不允许只靠数组外形过关。

## 3. 本关新增能力

- `RawLiteral`：可编辑的原始数据，可能是 number、string、bool 或数组。
- `DTypeGate`：根据目标 dtype 验证每个元素。
- `TensorBox v1`：保留 shape 信息。
- `TypeContractGate`：检查 dtype 与 shape 一致。
- `TypeProbe`：显示第一个不合格元素的位置。

`TypeProbe` 是诊断探针。它不输出可复用数据，只告诉玩家是 dtype 错还是 shape 错。

## 4. 具体案例

Visible case 是 tokenizer 后续会产出的 token id 序列：

```text
raw = [[12, 4, 7, 0]]
expected dtype = int
expected axes = [B,T]
expected shape = [1,4]
```

这里的 `12` 是整数 id，不是 float32 hidden 值，也不是字符串 `"12"`。

## 5. 初始错误图

画布给出：

- `raw_ids: RawLiteral`
- `shape_box: TensorBox v1`
- `typed_out: TypeContractGate`
- `type_probe: TypeProbe`

缺少 dtype 验证。raw literal 直接进合约会失败，因为它没有证明每个元素都是 int。

## 6. 目标内部实现

```text
raw_ids.out -> shape_box.x
shape_box.out -> dtype_gate.x
dtype_gate.out -> typed_out.x
dtype_gate.out -> type_probe.x
```

其中：

- `dtype_gate.moduleId = DTypeGate`
- `dtype_gate.expectedDType = int`
- `typed_out.expectedAxes = [B,T]`

## 7. 玩家操作

1. 查看 raw literal 的当前值。
2. 拖入 `DTypeGate`，选择 `int`。
3. 确认 raw 数据先经过 TensorBox，再经过 DTypeGate。
4. 将 DTypeGate 输出接到合约和 TypeProbe。
5. 尝试把某个元素改成 `"7"`，观察 TypeProbe 的错误。
6. 恢复为整数后检查当前任务，再提交认证。

## 8. 错误路径

- 把 dtype 设置成 `float32`：shape 可能正确，但 token id 语义错误。
- 输入文本 `"12"`：不能自动转成 int，TypeProbe 报元素类型错误。
- 输入 `12.5`：数值是 number，但不是 int。
- 只接 TensorBox：有 shape，没有 dtype。
- 只接 DTypeGate：有 dtype，没有 rank/axes 合约。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 `DTypeGate`。
- 输出 dtype 必须是 `int`。
- 输出 shape 必须是 `[1,4]`。
- 输出内容必须保持 `[[12,4,7,0]]`，不能排序、截断或 cast 成 float。
- 错误输入必须在 TypeProbe 中可定位。

## 10. Hidden / Mutation 测试

Hidden case A：不同长度。

```text
raw = [[3, 1, 4]]
expected shape = [1,3]
```

Hidden case B：非法元素。

```text
raw = [[3, "1", 4]]
```

认证必须失败并指出 `[0,1]` 是 string。

Hidden case C：bool mask 变体。

```text
raw = [[true, true, false]]
expected dtype = bool
```

公开认证允许玩家选择 `int` 或 `bool` 案例，但系统会固定当前案例的 expected dtype，防止用宽松 any 类型过关。

## 11. 认证后接口

```text
component.typed_tensor.v1
inputs:
  x: tensor
params:
  dtype: int | float32 | bool
output:
  out: typed_tensor
```

## 12. 后续调用

Tokenizer 输出必须是 int，attention mask 必须是 bool，hidden 必须是 float32。TypedTensor 是类型系统真正生效的第一关，不能退化成“shape 看起来对就放行”。
