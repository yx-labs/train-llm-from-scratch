# Chapter 7-3 Shift Targets: TargetShift

## 1. 组件真实用途

TargetShift 把 token 序列右移，构造 next-token training target：

```text
input_ids  = tokens[:, 0:T]
target_ids = tokens[:, 1:T+1]
```

本关只做切片对齐，不做模型计算。

## 2. 前置组件

- `component.padder.v1`
- `component.token_buffer.v1`

## 3. 本关新增能力

- `WindowPairContract`：要求输入窗口长度为 `T+1`。
- `SliceGate`：产生 input slice 与 target slice。
- `ShiftAlignmentProbe`：显示 input position i 对应 target position i。
- `ReferenceChecker`：检查切片。

## 4. 具体案例

Visible case:

```text
tokens = [[12,4,7,0]]
T = 3
input  = [[12,4,7]]
target = [[4,7,0]]
```

## 5. 初始错误图

画布给出 tokens、T、target_out、alignment_probe、reference。缺少 window contract 和 slice。

## 6. 目标内部实现

```text
tokens + T -> window_pair_contract
window_pair_contract -> slice_input(start=0, len=T)
window_pair_contract -> slice_target(start=1, len=T)
slice_target -> target_out
slice_input + slice_target -> alignment_probe
target_out -> reference
```

## 7. 错误路径

- target 不右移：模型学复制当前 token。
- 从 2 开始切：错过一个目标。
- 输出 `[B,T+1]`：长度错。
- 忽略 pad target：loss mask 后续无法对齐。

## 8. 测试设计

- Visible：`[12,4,7,0] -> [4,7,0]`。
- Hidden A：T=1。
- Hidden B：batch B=2。
- Hidden C：输入窗口不足 T+1 必须失败。

## 9. 认证后接口

```text
component.target_shift.v1
inputs:
  tokens: int[B,T+1]
output:
  targets: int[B,T]
```

## 10. 后续调用

CrossEntropy 用 targets 选择每个位置正确 token 的 logit。
