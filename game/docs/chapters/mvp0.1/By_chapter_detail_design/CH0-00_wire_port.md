# Chapter 0-0 Wire & Probe

## 1. 关卡真实用途

这是 MVP0.1 的入口教学关，不构建可复用组件。它只教玩家理解 Graph OS 的三个基础事实：

- 数据只能从 output port 流向 input port。
- 合约节点是测试设备，不是玩家要封装的组件。
- reference/probe 节点提供期望值，用来证明当前数据路径是否正确。

本关必须避免和 `ScalarCell` 重复。它不解释标量、不讲 rank-0，只解释“线”和“探针”。

## 2. 预制节点

画布给出三个预制节点：

- `packet_input: InputTensor`，从测试案例读取一个 `float32[B,T]` packet。
- `port_gate: OutputContractGate`，检查 packet 是否进入输出合约。
- `reference: ReferenceChecker`，预制参考探针，提供期望 packet。

这些节点都是测试设备。本关没有组件库，也不会产出 `BuiltComponent`。

## 3. 具体案例

Visible case:

```text
packet = [[0.25, -0.75]]
dtype  = float32
axes   = [B,T]
dims   = [1,2]
```

玩家需要让这份 packet 经过合约节点，再接入 reference 探针。

## 4. 初始错误图

初始图中三个节点都在画布上，但没有任何边：

```text
packet_input    port_gate    reference
```

运行检查会因为 `port_gate.x` 没有输入而 blocked。

## 5. 目标图

目标不是新增节点，而是建立测试电路：

```text
packet_input.out -> port_gate.x
port_gate.out -> reference.x
```

`reference` 的输入 `x` 表示“被检查值已经到达 probe 入口”，reference 输出本身来自测试案例。

## 6. 玩家操作

1. 从 `packet_input.out` 拖线到 `port_gate.x`。
2. 从 `port_gate.out` 拖线到 `reference.x`。
3. 点击“检查当前任务”。
4. 通过后自动进入 ScalarCell。

## 7. 错误路径

- 只连 `packet_input -> port_gate`：packet 到达合约，但 reference 没接上，无法完成比较。
- 直接 `packet_input -> reference`：绕过了合约，不能证明输出端口方向。
- 反向拖线：Graph OS 应拒绝 input -> output。
- 误以为 reference 是玩家组件：后续关卡会再次说明 probe 不是封装对象。

## 8. 测试设计

本关测试仍然使用 shape + reference，但它的目标是验证测试电路：

- `port_gate` 输出必须保持 `float32[B,T]=[1,2]`。
- `reference` 必须连接在合约之后。
- hidden case 会把 packet 换成 `[[1.25, 0.5]]`，检查连线是否仍然有效。

## 9. 通过后状态

本关通过后不解锁组件。玩家只获得 Graph OS 的基本操作认知：端口方向、数据流、合约、探针。

下一关 `ScalarCell` 才开始构建第一个可用组件。
