# Chapter 5-2 Head Concat: HeadConcat

## 1. 组件真实用途

HeadConcat 把每个 head 的 context 拼回 residual stream 宽度：

```text
context_heads[B,H,T,D] -> context[B,T,C]
C = H * D
```

它是 HeadSplit 的逆操作，但必须保持元素顺序。

## 2. 前置组件

- `component.parallel_heads.v1`
- `component.head_split.v1`

玩家已经知道 C 如何拆成 H 和 D。本关学习如何合回 C。

## 3. 本关新增能力

- `TransposeHeadsToTokenMajor`：把 `[B,H,T,D]` 调整为 `[B,T,H,D]`。
- `FlattenHeadChannel`：把 `[H,D]` 合成 C。
- `ConcatOrderProbe`：展示 head/channel 的拼接顺序。
- `ReferenceChecker`：检查元素映射。

## 4. 具体案例

Visible case:

```text
heads[B=1,H=3,T=2,D=2]
out[B=1,T=2,C=6]
```

映射：

```text
out[0,t,0:2] = head 0
out[0,t,2:4] = head 1
out[0,t,4:6] = head 2
```

## 5. 初始错误图

画布给出 context_heads、concat_out、order_probe、reference。缺少 transpose 和 flatten。

## 6. 目标内部实现

```text
context_heads.out -> transpose.x
transpose.out -> flatten.x
flatten.out -> concat_out.x
flatten.out -> order_probe.x
concat_out.out -> reference.x
```

其中：

- `transpose.from = [B,H,T,D]`
- `transpose.to = [B,T,H,D]`
- `flatten.axes = [H,D] -> C`

## 7. 玩家操作

1. 拖入 `TransposeHeadsToTokenMajor`。
2. 拖入 `FlattenHeadChannel`。
3. 设置 flatten 顺序 H-major then D。
4. 接入 order probe 和 reference。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- 直接 flatten `[B,H,T,D]`：T 和 H 顺序错。
- 按 D-major 拼接：reference 失败。
- sum heads 而不是 concat：shape `[B,T,D]` 错。
- 丢失 H 轴信息：无法恢复 C。
- 硬编码 C=6：H/D 变体失败。

## 9. 测试设计

- Visible：H=3,D=2，检查每个通道来源。
- Hidden A：H=2,D=4。
- Hidden B：T==H，防止轴混淆。
- Hidden C：每个 head 使用不同 sentinel，检测拼接顺序。

## 10. 认证后接口

```text
component.head_concat.v1
inputs:
  context_heads: float32[B,H,T,D]
output:
  context: float32[B,T,C]
```

## 11. 后续调用

OutputProjection 会把 concat 后的 C 维再投影回 residual stream。HeadConcat 负责结构合并，不负责混合通道。
