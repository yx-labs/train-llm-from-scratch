# MVP0.1 认证变体输入设计

## 1. 设计目标

MVP0.1 的组件不是通过当前案例就立刻可用，而是要经过认证。

认证要表达的核心概念是：

```text
当前任务检查：你的图能解决眼前这个案例。
组件认证：你的图能在变体输入下仍满足组件合约。
```

因此 UI 不再使用“运行可见测试 / 运行隐藏测试”这种工程术语，而使用：

```text
检查当前任务
提交认证
```

认证通过后，系统提示：

```text
{组件名} 已经可用
```

随后自动进入下一关。

## 2. 用户流程

```text
1. 玩家进入关卡，首次自动看到任务弹窗。
2. 玩家搭图或修图。
3. 玩家点击「检查当前任务」。
4. 当前任务通过后，画布出现「认证变体」卡片。
5. 玩家用少量控件准备一个公开认证变体。
6. 玩家点击「提交认证」。
7. 系统运行：
   - 玩家准备的公开认证变体
   - 系统未公开认证变体
8. 全部通过后，组件可用，并自动进入下一关。
```

关键点：第 5 步不是让玩家手写完整输入，而是让玩家编辑认证意图。

## 3. 为什么不让玩家手写完整输入

低阶组件可以手填：

```text
Scalar: 0.6
Vector: c0=1.2, c1=0.25, c2=-2
```

但高级组件输入会迅速变成：

```text
hidden[B,T,C]
weight[C,O]
bias[O]
```

如果让玩家手写完整 tensor，会把学习目标从“理解合约和泛化”变成“填数组体力活”。

所以 MVP0.1 采用三层输入策略：

```text
当前任务案例：系统给定，用于调图。
公开认证变体：玩家编辑少量结构参数，系统生成具体输入。
系统认证变体：系统追加未公开变体，防止只适配公开变体。
```

## 4. 控件设计原则

玩家编辑的是合约相关参数，而不是完整数据。

| 组件 | 玩家编辑 | 系统生成 |
| --- | --- | --- |
| ScalarCell | 一个有限 float32 | 临时 scalar source |
| VectorRail | c0/c1/c2 三个标量 | 公开 reference vector |
| MatrixStruct | C width + seed | 两列 vector[C] 和 reference matrix[C,O] |
| TensorBox | C width + seed | 两行 token vector[C] 和 tensor[B,T,C] |
| MatMulGate | B/T/C/O + seed | hidden[B,T,C]、weight[C,O]、reference[B,T,O] |
| Linear | B/T/C/O + seed | hidden、weight、bias、reference |

`seed` 的作用是让玩家知道“数据值会变化”，但不要求手写大数组。

控件表现上不要用浏览器原生 `number` 输入把非法内容提前挡掉。玩家应该可以输入 `abc`、空值、超出
float32 范围的数等错误案例，然后由「检查当前任务」或「提交认证」明确报错：

```text
Scalar value: Enter a finite float32 number
Scalar value: Value is outside float32 range
```

这样可以把反馈落在“这个组件合约为什么不成立”上，而不是让用户困惑于输入框为什么不能输入。
普通标量值不设置 `[-10, 10]` 这种教学无关范围；只有尺寸类控件保留上限，用于避免生成过大的张量。

## 5. MVP0.1 数据结构

在 `LevelSpec` 中增加：

```ts
type LevelCertificationSpec = {
  title: string;
  narrative: string;

  publicVariantLabel: string;
  publicVariantDescription: string;
  systemVariantDescription: string;

  controls: CertificationControlSpec[];

  makePublicTests: (
    graph: GraphSpec,
    values: Record<string, CertificationControlValue>
  ) => CertificationTestSpec[];
};
```

其中 `CertificationTestSpec` 允许公开变体临时覆盖 graph：

```ts
type CertificationTestSpec = {
  testCase: TestCase;
  graph?: GraphSpec;
};
```

这是为了支持 Scalar / Vector 这类数值存在节点参数里的组件：

```text
公开认证变体可以临时把 scalar_source.value 改成 0.6
但不永久修改玩家当前图
```

## 6. 运行规则

提交认证时，系统运行：

```ts
[
  ...level.certification.makePublicTests(graph, playerValues),
  ...level.hiddenTests
]
```

每个公开变体可以有自己的临时 graph；系统隐藏变体默认使用玩家当前 graph。

认证通过条件：

```text
公开认证变体全部通过
系统未公开变体全部通过
组件依赖已满足
认证输入控件没有非法值
```

通过后：

```text
1. 记录 component available
2. 显示「xx 已经可用」
3. 自动进入下一关
```

## 7. 文案规范

| 不再使用 | 使用 |
| --- | --- |
| 运行可见测试 | 检查当前任务 |
| 运行隐藏测试 | 提交认证 |
| 可见测试 | 任务检查 |
| 隐藏测试 | 认证检查 / 系统变体 |
| Pack / 封装按钮 | 通过认证后组件自动可用 |

认证卡片推荐文案结构：

```text
当前任务已通过

你的图已经能处理当前案例。但组件要可用，还需要通过认证。

公开变体：
  玩家编辑少量参数，系统生成输入。

系统变体：
  系统追加未公开案例，检查组件是否依赖合约而不是当前样例。
```

## 8. 后续扩展

高级组件不新增手写 tensor，而扩展生成器：

```text
尺寸控件：B/T/C/O/H/V
语义控件：axis order、broadcast target、mask orientation
数据控件：seed、pattern、range、edge case
结构控件：head count、block count、vocab size
```

玩家始终编辑“结构和意图”，系统负责生成具体数据。
