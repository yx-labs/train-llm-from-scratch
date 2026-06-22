# Chapter 6-6 Block Stack: TransformerStack

## 1. 组件真实用途

TransformerStack 把 N 个 TransformerBlock 串起来：

```text
h0 = hidden
h1 = Block0(h0)
h2 = Block1(h1)
...
hN = BlockN-1(hN-1)
```

它是从单层 block 进入小模型主体的最后一步。

## 2. 前置组件

- `component.transformer_block.v1`

本关复用同一个 block 接口，但每层有自己的参数。

## 3. 本关新增能力

- `LayerLoopGate`：按层数 N 顺序应用 block。
- `BlockWeightSelector`：为每层选择对应权重。
- `StackTraceProbe`：显示每层输出 checksum。
- `ReferenceChecker`：端到端 stack reference。

## 4. 具体案例

Visible case:

```text
N = 2
hidden[B=1,T=3,C=6]
out[B=1,T=3,C=6]
```

StackTraceProbe 显示：

```text
layer 0 checksum
layer 1 checksum
```

## 5. 初始错误图

画布给出 hidden、block_weights[2]、mask、stack_out、stack_trace、reference。缺少 loop 和 weight selector。

## 6. 目标内部实现

```text
hidden -> layer_loop.initial
block_weights -> weight_selector.weights
weight_selector.layer_weight -> layer_loop.block_weights
mask -> layer_loop.mask
layer_loop.out -> stack_out
layer_loop.trace -> stack_trace
stack_out -> reference
```

## 7. 玩家操作

1. 拖入 `LayerLoopGate`。
2. 设置层数 N 来自输入，不硬编码。
3. 拖入 `BlockWeightSelector`。
4. 将每层权重按 index 传给 block。
5. 接入 StackTraceProbe 和 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- 重复使用第 0 层权重：hidden layer-specific reference 失败。
- 所有 block 并行后相加：结构顺序错。
- 只执行一层：N=2 visible 失败。
- 硬编码 N=2：N=3 hidden 失败。
- 每层重新使用原 hidden：trace 失败。

## 9. 测试设计

- Visible：N=2。
- Hidden A：N=3。
- Hidden B：第 1 层权重设置 identity-like，检查顺序。
- Hidden C：mask 必须传入每一层。

## 10. 认证后接口

```text
component.transformer_stack.v1
inputs:
  hidden: float32[B,T,C]
  weights: BlockWeights[N]
  mask: bool[T,T]
output:
  out: float32[B,T,C]
```

## 11. 后续调用

Chapter 7 会把 stack 输出接入 final norm、LM head 和 loss。BlockStack 是 tiny model 的主干。
