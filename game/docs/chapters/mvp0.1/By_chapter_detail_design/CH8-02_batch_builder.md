# Chapter 8-2 Batch Builder: BatchBuilder[B,T]

## 1. 组件真实用途

BatchBuilder 把多个 `T+1` 窗口堆叠成训练 batch，并生成 input ids 与 target ids：

```text
windows[B,T+1] -> input_ids[B,T], targets[B,T]
```

## 2. 前置组件

- `component.context_window.v1`
- `component.target_shift.v1`

## 3. 本关新增能力

- `WindowStacker`：把多个窗口堆成 B 轴。
- `BatchShiftGate`：对每行做 input/target shift。
- `BatchContract`：检查 `[B,T]`。
- `BatchPreviewProbe`：显示 batch row。
- `ReferenceChecker`：检查切片。

## 4. 具体案例

Visible case:

```text
windows =
  [[12,4,7,0],
   [ 4,7,0,12]]

inputs =
  [[12,4,7],
   [ 4,7,0]]

targets =
  [[4,7,0],
   [7,0,12]]
```

## 5. 初始错误图

画布给出 windows、batch_out、preview、reference。缺少 stacker 和 shift。

## 6. 目标内部实现

```text
windows -> window_stacker
window_stacker -> batch_shift
batch_shift.inputs -> batch_out.inputs
batch_shift.targets -> batch_out.targets
batch_shift -> preview
batch_out -> reference
```

## 7. 错误路径

- 把 windows concat 成 `[B*(T+1)]`：batch 轴丢失。
- 每行 target 不右移：loss 学错。
- B 轴与 T 轴交换：shape/axis hidden 失败。
- 不检查窗口长度一致：stacker 应失败。

## 8. 测试设计

- Visible：B=2,T=3。
- Hidden A：B=1。
- Hidden B：B=3。
- Hidden C：某窗口长度不同必须失败。

## 9. 认证后接口

```text
component.batch_builder.v1
inputs:
  windows: int[B,T+1]
outputs:
  input_ids: int[B,T]
  targets: int[B,T]
```

## 10. 后续调用

ForwardRunner 使用 input_ids 运行模型，CrossEntropy 使用 targets。
