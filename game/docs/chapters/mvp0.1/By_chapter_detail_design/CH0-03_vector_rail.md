# Chapter 0-3 Vector Rail

## 1. 组件真实用途

VectorRail 把多个已认证 ScalarCell 按顺序放到同一条 C 轴上，形成 `float32[C]`。

本关要让玩家理解：vector 不是“一堆数字”，而是“有顺序的一排数字”。顺序就是轴位置。

```text
c0 = 0.5
c1 = -1
c2 = 2

vector[C] = [0.5, -1, 2]
```

## 2. 前置组件

- `component.scalar_cell.v1`

本关不允许使用 `component.vector_rail.v1`。组件库提供的是三个 `ScalarCell v1` 实例和 `VectorRail` primitive。

## 3. 本关新增能力

- 复用可用组件：`component.scalar_cell.v1`
- 内部结构节点：`VectorRail`
- 输出合约：`OutputContractGate(expectedAxes=[C])`
- 行为证明：`ReferenceChecker`

这里第一次体现“之前构建的组件进入组件库，并作为后续组件的内部材料”。

## 4. 具体案例

Visible case:

```text
scalar_c0 = 0.5
scalar_c1 = -1
scalar_c2 = 2

expected vector = [0.5, -1, 2]
dtype = float32
axes  = [C]
dims  = [3]
```

## 5. 初始错误图

画布给出三个 ScalarCell 和输出合约/参考探针，但缺少把它们排成 C 轴的 `vector` 节点。

```text
scalar_c0
scalar_c1      vector missing      vector_out      reference
scalar_c2
```

## 6. 目标内部实现

```text
scalar_c0.out -> vector.c0
scalar_c1.out -> vector.c1
scalar_c2.out -> vector.c2
vector.out -> vector_out.x
vector_out.out -> reference.x
```

其中：

- `vector.moduleId = VectorRail`
- `vector_out.expectedAxes = [C]`

## 7. 玩家操作

1. 拖入 `VectorRail`。
2. 按 c0/c1/c2 顺序连接三个 scalar。
3. 将 `vector.out` 接到 `vector_out.x`。
4. 将 `vector_out.out` 接到 `reference.x`。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- c0/c1/c2 顺序接反：shape 仍是 `[C]`，但 allclose reference 失败。
- 少接一个 scalar：VectorRail blocked 或输出长度错误。
- 把三个 scalar 直接分别接到合约：合约只有一个输入，不构成 vector。
- 使用目标组件 `component.vector_rail.v1`：本关应该构建它，而不是调用它。

## 9. 测试设计

当前任务：

- shape contract：`vector_out` 是 `float32[C=3]`。
- behavior contract：输出值必须等于 `[0.5, -1, 2]`。

公开认证：

- 玩家可编辑 c0/c1/c2 三个数值。
- 系统只改变 ScalarCell 输入值，不改变 VectorRail 或合约。
- 认证重点是顺序保持，而不是固定示例值。

## 10. 认证后接口

```text
component.vector_rail.v1
inputs:
  c0: float32[]
  c1: float32[]
  c2: float32[]
output:
  vector: float32[C]
```

## 11. 后续调用

MatrixStruct 会把多个 `vector[C]` 组织成 `matrix[C,O]`。VectorRail 的核心价值是把“位置顺序”升级成明确的轴语义。
