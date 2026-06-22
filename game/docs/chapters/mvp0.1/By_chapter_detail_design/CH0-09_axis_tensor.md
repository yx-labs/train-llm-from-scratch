# Chapter 0-9 Axis Tensor: AxisTensor

## 1. 组件真实用途

AxisTensor 给 tensor 的每个维度命名。它解决的是“同样长度的两个维度是否真的代表同一件事”。

```text
hidden[B,T,C]
weight[C,O]
scores[B,H,T,T]
```

这些名字不是注释。后续 MatMul、broadcast、mask 都要靠 axis name 判断能否连接。

## 2. 前置组件

- `component.tensor_box.v1`
- `component.typed_tensor.v1`

玩家已经能构造 typed tensor。本关继续增加 axis contract。

## 3. 本关新增能力

- `AxisBinder`：给每个 rank 位置绑定 axis name。
- `AxisContractGate`：检查 axis 顺序和语义。
- `AxisProbe`：显示某一维的 name、长度和样本切片。
- `ReferenceChecker`：确保 axis 标注没有改变数据顺序。

`AxisProbe` 是认知探针。它让玩家看到“第 1 维是 token 时间 T，不是 channel C”。

## 4. 具体案例

Visible case 是一小段 hidden states：

```text
hidden raw shape = [1, 2, 3]
axes = [B, T, C]

B = batch
T = token position
C = channel feature
```

样本：

```text
hidden[0,0,:] = token 0 的 3 个 channel
hidden[0,1,:] = token 1 的 3 个 channel
```

## 5. 初始错误图

画布给出：

- `hidden_source: TypedTensor`
- `axis_out: AxisContractGate`
- `axis_probe: AxisProbe`
- `reference: ReferenceChecker`

缺少 axis 绑定。source 只有 dtype 和 shape，没有说明 `[1,2,3]` 中哪个维度是 T、哪个是 C。

## 6. 目标内部实现

```text
hidden_source.out -> axis_binder.x
axis_binder.out -> axis_out.x
axis_binder.out -> axis_probe.x
axis_out.out -> reference.x
```

其中：

- `axis_binder.axes = [B,T,C]`
- `axis_out.expectedAxes = [B,T,C]`

## 7. 玩家操作

1. 观察 `hidden_source` 的 shape `[1,2,3]`。
2. 拖入 `AxisBinder`。
3. 将三个轴依次标为 `B`、`T`、`C`。
4. 连接 AxisContract、AxisProbe 和 reference。
5. 故意把 `T` 与 `C` 交换一次，观察 Probe 如何提示。
6. 恢复正确轴后检查当前任务并提交认证。

## 8. 错误路径

- 标成 `[B,C,T]`：shape 仍是 `[1,2,3]`，但后续 MatMul 会消费错误内维。
- 只写 axis names，不接 reference：可能没有证明数据顺序没有被改。
- 交换数据再标正确轴：axis 合约过，但 reference 数值失败。
- 使用动态 any axis：结构断言失败，MVP0.1 不允许跳过 axis。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 `AxisBinder`。
- Axis 顺序必须是 `[B,T,C]`。
- dtype 必须保留 `float32`。
- shape 必须保留 `[1,2,3]`。
- ReferenceChecker 确认数据值未被转置。

## 10. Hidden / Mutation 测试

Hidden case A：`T == C`。

```text
shape = [1,3,3]
axes = [B,T,C]
```

这个 case 防止玩家靠长度判断 axis。

Hidden case B：更大 batch。

```text
shape = [2,4,3]
axes = [B,T,C]
```

Hidden case C：错误标注。

```text
axes = [B,C,T]
```

认证必须失败，并提示 MatMul 内维将找不到 C。

## 11. 认证后接口

```text
component.axis_tensor.v1
inputs:
  x: typed_tensor
params:
  axes: AxisName[]
output:
  out: axis_tensor
```

## 12. 后续调用

MatMulGate 的 C 消费、Linear 的 O 输出、QKScore 的 T-to-T board 都依赖 axis name。AxisTensor 是从“数组”进入“可组合图语言”的最后一步。
