# Chapter 5-0 Head Split: HeadWidth and HeadSplit

## 1. 组件真实用途

HeadSplit 把单条 C 维 hidden/projection 通道拆成 H 个 attention head：

```text
C = H * D
x[B,T,C] -> x_heads[B,H,T,D]
```

本关让玩家理解 D 不是新的参数，而是由 `C/H` 推导出来的 head width。

## 2. 前置组件

- `component.axis_tensor.v1`
- `component.single_head_attention.v1`

单头 attention 已经使用 `[B,H,T,D]`。本关把 H 从 1 扩展到多个 head。

## 3. 本关新增能力

- `DivisibilityGate`：检查 C 能被 H 整除。
- `HeadWidthGate`：计算 `D=C/H`。
- `ReshapeToHeads`：把 `[B,T,C]` reshape 为 `[B,H,T,D]`。
- `HeadAxisProbe`：显示某个 channel 落入哪个 head。
- `ReferenceChecker`：检查 reshape 不改变元素顺序。

## 4. 具体案例

Visible case:

```text
x[B=1,T=2,C=6]
H = 3
D = 2
out[B=1,H=3,T=2,D=2]
```

示例映射：

```text
x[0,1,4] -> out[0,2,1,0]
x[0,1,5] -> out[0,2,1,1]
```

## 5. 初始错误图

画布给出 `x`、`head_count`、`split_out`、`head_axis_probe`、`reference`。缺少整除检查、D 计算和 reshape。

## 6. 目标内部实现

```text
x.out -> divisible.x
head_count.out -> divisible.h
divisible.out -> head_width.c
head_count.out -> head_width.h
x.out -> reshape.x
head_width.d -> reshape.d
head_count.out -> reshape.h
reshape.out -> split_out.x
reshape.out -> head_axis_probe.x
split_out.out -> reference.x
```

## 7. 玩家操作

1. 拖入 `DivisibilityGate`。
2. 拖入 `HeadWidthGate` 计算 D。
3. 拖入 `ReshapeToHeads`，设置输出 axes `[B,H,T,D]`。
4. 连接 probe 与 reference。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- 不检查整除：C=7,H=3 hidden case 应失败。
- 输出 `[B,T,H,D]`：axis 顺序错。
- 复制数据而不是 reshape：checksum/reference 失败。
- 硬编码 D=2：H/C 变体失败。
- 用单头 attention 代替 split：职责错误。

## 9. 测试设计

- Visible：`C=6,H=3,D=2`，元素映射 allclose。
- Hidden A：`C=8,H=4,D=2`。
- Hidden B：`C=7,H=3` 必须报整除错误。
- Hidden C：`T==H`，防止按长度猜 axis。

## 10. 认证后接口

```text
component.head_split.v1
inputs:
  x: float32[B,T,C]
  head_count: int[]
output:
  x_heads: float32[B,H,T,D]
```

## 11. 后续调用

ParallelHeads 会在每个 head 上运行 attention。HeadSplit 的关键是 axis 语义和元素顺序，不能只检查 shape。
