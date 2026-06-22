# Chapter 9-1 Context Crop: LastTContext

## 1. 组件真实用途

ContextCrop 从不断增长的 token buffer 中保留最后 T 个 token：

```text
context = tokens[-T:]
```

生成时模型只能看到固定上下文长度。

## 2. 前置组件

- `component.prompt_encode.v1`
- `component.token_buffer.v1`

## 3. 本关新增能力

- `LengthProbe`：读取当前 token 数。
- `StartForLastTGate`：计算 `max(0, len - T)`。
- `SliceLastGate`：裁剪最后 T 个。
- `CropMaskGate`：同步裁剪 mask。
- `ReferenceChecker`：检查裁剪。

## 4. 具体案例

Visible case:

```text
tokens = [12,4,7,8,9]
T = 3
context = [7,8,9]
```

## 5. 初始错误图

画布给出 tokens、mask、T、context_out、probe、reference。缺少 start 计算和 slice。

## 6. 目标内部实现

```text
tokens -> length_probe
length_probe + T -> start_for_last_t
tokens + start -> slice_last
mask + start -> crop_mask
slice_last + crop_mask -> context_out
context_out -> reference
```

## 7. 错误路径

- 取前 T 个 token：长 prompt hidden 失败。
- token 裁剪了但 mask 未裁剪：shape/behavior 错。
- len<T 时左侧填错：应保留已有 token 并由 mask 表达。
- 硬编码 T=3：变体失败。

## 8. 测试设计

- Visible：len=5,T=3。
- Hidden A：len<T。
- Hidden B：len=T。
- Hidden C：mask 含 padding。

## 9. 认证后接口

```text
component.context_crop.v1
inputs:
  tokens: int[N]
  mask: bool[N]
  context_length: int[]
outputs:
  context_ids: int[T]
  context_mask: bool[T]
```

## 10. 后续调用

NextLogits 用裁剪后的上下文运行模型。
