# Chapter 9-4 Append Token: AppendToken

## 1. 组件真实用途

AppendToken 把新采样的 token id 加到生成序列末尾：

```text
tokens_next = tokens + [next_id]
```

如果超过上下文容量，后续 ContextCrop 会处理裁剪。本关只负责追加。

## 2. 前置组件

- `component.sample_next_token.v1`
- `component.context_crop.v1`

## 3. 本关新增能力

- `AppendGate`：把 scalar id 追加到序列尾。
- `LengthUpdateGate`：更新长度。
- `AppendProbe`：显示追加前后尾部 token。
- `ReferenceChecker`：检查序列。

## 4. 具体案例

Visible case:

```text
tokens = [12,4,7]
next_id = 0
tokens_next = [12,4,7,0]
```

## 5. 初始错误图

画布给出 tokens、next_id、append_out、probe、reference。缺少 append 和 length update。

## 6. 目标内部实现

```text
tokens + next_id -> append_gate
tokens.length + 1 -> length_update
append_gate + length_update -> append_out
append_gate -> append_probe
append_out -> reference
```

## 7. 错误路径

- prepend 到开头：生成顺序错。
- 替换最后一个 token：history 丢失。
- next_id dtype 不是 int scalar：contract 失败。
- 追加后 mask/length 不更新：后续 crop 错。

## 8. 测试设计

- Visible：追加 0。
- Hidden A：追加非 pad id。
- Hidden B：空序列追加。
- Hidden C：next_id 向量必须失败。

## 9. 认证后接口

```text
component.append_token.v1
inputs:
  tokens: int[N]
  next_id: int[]
output:
  tokens_next: int[N+1]
```

## 10. 后续调用

Decode 会把累积 tokens 转回文本；TinyChatLoop 会重复执行 append。
