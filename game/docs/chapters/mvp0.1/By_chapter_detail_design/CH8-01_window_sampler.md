# Chapter 8-1 Window Sampler: ContextWindow

## 1. 组件真实用途

WindowSampler 从 token stream 中取长度 `T+1` 的连续窗口，用于构造 input/target：

```text
window = stream[start : start + T + 1]
```

## 2. 前置组件

- `component.dataset_stream.v1`
- `component.target_shift.v1`

## 3. 本关新增能力

- `StartIndexSource`：提供采样起点。
- `BoundsGate`：检查窗口不越界。
- `SliceWindowGate`：连续切片。
- `WindowProbe`：显示 start/end 和窗口内容。
- `ReferenceChecker`：切片参考。

## 4. 具体案例

Visible case:

```text
stream = [12,4,7,0,12,4]
T = 3
start = 1
window = [4,7,0,12]
```

## 5. 初始错误图

画布给出 stream、T、start、window_out、probe、reference。缺少 bounds 和 slice。

## 6. 目标内部实现

```text
stream + start + T -> bounds
bounds -> slice_window
slice_window -> window_out
slice_window -> window_probe
window_out -> reference
```

## 7. 错误路径

- 只取 T 个 token：target shift 缺少最后一个目标。
- 非连续采样：language modeling 语义错。
- 越界时 wrap around：必须失败。
- 硬编码 start：认证 start 变体失败。

## 8. 测试设计

- Visible：start=1,T=3。
- Hidden A：start=0。
- Hidden B：最后一个合法 start。
- Hidden C：越界 start 必须失败。

## 9. 认证后接口

```text
component.context_window.v1
inputs:
  stream: int[S]
  start: int[]
  length: int[]
output:
  window: int[T+1]
```

## 10. 后续调用

BatchBuilder 会堆叠多个 window，并把它们拆成 inputs/targets。
