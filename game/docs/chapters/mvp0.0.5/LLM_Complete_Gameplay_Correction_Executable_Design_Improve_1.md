我重新看了最新 `game` 分支的 Graph Challenge 改造。结论是：**底层方向明显进步了，但当前人测“摸不着头脑”不是普通玩家理解能力的问题，而是第一分钟体验设计出了断层。**

现在已经有了真正的 `GraphSpec / ModuleRegistry / Tiny Runtime / Test Runner` 基础，`GraphWorkbench` 也已经具备关卡列表、画布、模块库、Run Visible / Run Hidden、Trace Timeline、Inspector、参数编辑、测试结果等核心结构。

但问题是：**它现在更像把玩家直接扔进一个“无说明的可视化 IDE”，而不是一个逐步引导的游戏关卡。**

---

# 一、当前版本的主要问题判断

## 1. 普通玩家一进来就是 Graph Challenge，而且第一关就是 0-2 MatMul

`App.tsx` 里当前默认 `experienceMode` 是 `"graph"`，也就是用户打开后直接进入 Graph Challenge。

而 `graphLevels` 的第一个关卡不是 0-1 的图形基础教学，也不是更直观的 Text → Token，而是 `ch0MatMulGraph`，接着是 Transpose、Broadcast、Mask、Tokenizer。

这会造成一个认知跳跃：

```text
普通玩家刚进入游戏
↓
还没学会节点、端口、连线、Run Visible、Trace
↓
直接面对 hidden[B,T,C] @ stored_weight[O,C].transpose()
↓
不知道第一步是拖模块、连线、改参数，还是点 Run
```

对懂机器学习的人来说这只是图构建任务；对普通玩家来说，这是同时在学 UI 操作、张量概念、矩阵乘法、测试机制、调试机制。

---

## 2. Chapter 0 的前几个 Graph 关卡是空画布，缺少“坏机器”

最致命的问题是：当前 0-2、0-3、0-4、0-4F 这些关的 `initialGraph.nodes` 都是空数组。

0-2 MatMul Gate Graph 是空图。
0-3 Transpose Trap Graph 也是空图。
0-4 Broadcast Add Graph 也是空图。
0-4F Mini Mask Add 也是空图。

这会让普通玩家的第一反应变成：

```text
我要拖哪个？
拖几个？
这些模块之间是什么关系？
目标图长什么样？
我错了以后怎么知道哪里错？
```

空白画布是高级挑战，不应该是初学者第一关。对教育类游戏来说，第一阶段更应该是：

```text
给玩家一台已经搭好的坏机器
↓
让玩家运行它
↓
看到第一个明确错误
↓
修一个局部
↓
马上得到反馈
```

当前你们做成了：

```text
给玩家一片空地
↓
给一堆专业模块
↓
让玩家自己推导整个系统
```

这就是“摸不着头脑”的核心原因。

---

## 3. 模块库对开发者清晰，对普通玩家不清晰

现在画布内模块库展示的是模块 label 和 category，summary 主要作为 `title` tooltip 存在。

比如玩家看到：

```text
Input Tensor
Weight Plate
Transpose Switch
MatMul Gate
Output Contract Gate
Reference Checker
```

这些对程序员是合理的，但对普通玩家不够“可操作”。普通玩家需要的是：

```text
1. hidden[B,T,C] 输入
2. stored weight[O,C]
3. 把 weight 转成 [C,O]
4. 做 MatMul
5. 检查输出是不是 [B,T,O]
```

而不是先理解抽象模块分类。

更严重的是，有些节点 ID 和参数是通过 `getLevelNodeTemplate` 按添加顺序自动决定的。例如在 0-2 里，第一次添加 `InputTensor` 会变成 `hidden`，添加 `WeightPlate` 会变成 `weight`，添加 `TransposeSwitch` 会变成 `weight_transpose`。

这对系统实现方便，但玩家不知道“为什么我拖这个模块会变成这个角色”。这会产生隐形规则感。

---

## 4. “Run Visible” 没有承担好新手教学功能

Graph Runtime 已经是对的：先验证图结构，再按拓扑顺序执行节点，并记录 trace。

但当前如果结构验证失败，`executeGraph` 会直接返回 `trace: []`。 也就是说，在玩家最容易犯错的阶段，Trace 反而可能没有东西可看。

Graph Validation 会检查 required input 是否连接，并能生成 missing_input 错误。 但这个错误需要被包装成“下一步行动”，否则玩家看到的只是工程错误，而不是游戏提示。

现在的错误信息逻辑基本是：

```text
Missing output for node xxx
possibleCause: The graph did not execute far enough...
suggestedProbe: Inspect incoming edges...
```

这对开发者有用，但对普通玩家不够直接。玩家真正需要的是：

```text
MatMul 还缺右输入。
请把 weight_transpose.out 连到 matmul.right。
```

或者：

```text
现在 weight 是 [O,C]，MatMul 需要 [C,O]。
先在 weight 和 matmul 之间插入 Transpose Switch。
```

---

## 5. 当前缺少“本关目标图 / 任务清单 / 下一步提示”

`GraphWorkbench` 现在有左侧 Level List、中间画布、顶部 Run Bar、画布内模块库、Trace Timeline、右侧 Inspector / Test Panel。

但是缺一个新手最需要的东西：

```text
当前我在修什么？
目标管线长什么样？
现在应该先点哪里？
我离完成还差几个连接？
失败后我应该做哪一个动作？
```

也就是说，系统已经有“工具”，但还没有“教练”。

---

## 6. 参数编辑还是开发者面板，不是玩家面板

当前参数编辑是通用型的，根据值类型自动生成 input / checkbox / select。数组参数会用逗号文本输入，部分关键参数通过 `paramOptions` 提供下拉，比如 `policy`、`fallback`、`padSide`、`maskPolicy`、`orientation`、`maskOrientation`。

这对快速开发很好，但普通玩家看到的是：

```text
axisA = -2
axisB = -1
expectedAxes = B,T,O
inputKey = hidden
maskOrientation = query_key
```

这些还不是“游戏语言”。它们应该被包装成：

```text
Transpose 方式：
- 交换最后两轴 D 和 T
- 交换前两轴 B 和 H
- 反转全部轴

Mask 方向：
- 屏蔽未来 token
- 错误：屏蔽过去 token

Tokenizer 策略：
- 字符切分：安全但 token 很多
- 单词切分：token 少但容易 OOV
- 子词切分：折中方案
```

---

# 二、总体评价

我会这样给当前版本打分：

```text
底层技术方向：8/10
Graph Runtime 雏形：7.5/10
测试与 Trace 方向：7/10
普通玩家第一分钟体验：3/10
教学游戏化节奏：4/10
继续迭代潜力：8/10
```

这次改造已经把项目从“填槽答题”推进到了“可执行图系统”，这是关键进步。但现在缺的是：

```text
不是更多模块
不是更多测试
不是更多关卡
而是第一阶段的脚手架、目标引导、错误转行动
```

---

# 三、最重要的修正策略：从“自由搭图”改成“坏机器维修”

Graph Challenge 不应该一开始就是自由搭图。更合理的学习曲线是：

```text
阶段 A：修一台坏机器
给出 80% 已经搭好的图，只坏一个地方。
玩家运行、定位、修复。

阶段 B：补一段缺失管线
给出输入和输出，中间少 1-2 个模块。
玩家选择模块并连接。

阶段 C：参数调试
图已经连好，但一个参数错了。
玩家通过 Trace 和测试修改参数。

阶段 D：自由搭图
玩家从空画布构建完整系统。
```

当前版本直接从阶段 D 开始，所以普通玩家失去入口。

---

# 四、建议立即修改的具体点

## P0-1：不要让 0-2 / 0-3 / 0-4 从空图开始

你们源码里其实已经有“错误图 / 解法图”函数，只是没有作为 `initialGraph` 使用。

0-2 里已经有：

```ts
createCh0MatMulSolutionGraph()
createCh0MatMulNoTransposeGraph()
createCh0MatMulWrongOutputAxesGraph()
```

其中 `createCh0MatMulNoTransposeGraph()` 就是非常适合作为初始关卡的“坏机器”：它已经有 hidden、weight、matmul、projected、reference，只是缺少正确的 transpose。

建议把 0-2 的初始图从空图改成：

```ts
initialGraph: createCh0MatMulNoTransposeGraph()
```

对应体验会变成：

```text
玩家进入后看到一条已经搭好的管线
↓
点击 Run Visible
↓
MatMul 报 inner dimension mismatch 或 allclose fail
↓
系统提示：weight 是 [O,C]，MatMul 需要 [C,O]
↓
玩家插入 Transpose Switch
↓
通过
```

0-3 同理，用 `createCh0TransposeNoKTransposeGraph()` 作为初始坏图。它已经是“没有 K transpose”的 QK 图，非常适合让玩家修复。

0-4 可以用 `createCh0BroadcastWithoutRailGraph()` 或 `createCh0BroadcastWrongAxisGraph()`。

0-4F 可以用 `createCh0MaskWrongOrientationGraph()` 或 `createCh0MaskWithoutBroadcastGraph()`。

也就是说，第一版 Graph Challenge 不要考“从零搭建”，而是考：

```text
这台机器哪里坏了？
为什么坏？
修哪一段？
```

这会大幅降低玩家入门阻力。

---

## P0-2：第一关不要是 MatMul 空白搭建，先做一个 Graph 操作新手关

建议新增一个很短的关卡：

```ts
ch0_0_graph_controls
```

目标不是教 LLM，而是教 Graph Challenge 的基本操作。

### 关卡：0-G Graph Basics

初始图：

```text
Text Input ----x---- Tokenizer Socket ----> Token IDs Probe
```

或者更简单：

```text
Input Tensor ----x---- Output Gate
```

任务只教 4 件事：

```text
1. 点击 Run Visible
2. 看第一个失败节点
3. 从 out 端口拉线到 input 端口
4. 再次 Run Visible
```

玩家通过后再进入 0-2 MatMul Repair。

这关不应该讲太多 LLM，只讲游戏操作：

```text
节点是机器
端口是插孔
线是数据流
Run Visible 是公开测试
Trace 是运行记录
Inspector 是放大镜
```

---

## P0-3：Graph Challenge 默认入口要谨慎

现在默认直接进入 Graph Challenge。

对开发测试可以这样，但对普通玩家测试不建议。建议两种方案选一：

### 方案 A：默认 Guided Repair

```ts
const [experienceMode, setExperienceMode] = useState<"guided" | "graph">("guided");
```

玩家先通过 0-1 Shape Reader，再被引导进入 Graph Challenge。

### 方案 B：默认 Graph，但弹出 Graph 操作教学

进入后先出现：

```text
Graph Challenge 是什么？

你要修复一台 tiny LLM 机器。
每个节点是一种计算。
每条线传递 tensor / token。
先点击 Run Visible，看机器哪里坏。
然后修第一个红色节点。
```

按钮：

```text
开始：运行第一台坏机器
```

不要让玩家第一眼看到空白画布。

---

# 五、需要新增的 UI 结构

## 1. Mission Panel：任务卡片

放在画布顶部或左上角，不要只放一行 goal。

建议新增 `LevelSpec.onboarding`：

```ts
type LevelOnboarding = {
  story: string;
  startingProblem: string;
  winCondition: string;
  firstAction: string;
  targetRecipe: string[];
  allowedMistakes?: string[];
};
```

0-2 可以这样写：

```ts
onboarding: {
  story: "Linear Projection 机器坏了。hidden 已经进入 MatMul，但 weight 的方向不对。",
  startingProblem: "stored_weight 是 [O,C]，MatMul 需要 [C,O]。",
  firstAction: "先点击 Run Visible，观察 MatMul 为什么失败。",
  targetRecipe: [
    "hidden.out -> matmul.left",
    "weight.out -> transpose.x",
    "transpose.out -> matmul.right",
    "matmul.out -> projected.x",
    "projected.out -> reference.x"
  ],
  winCondition: "projected 通过 shape [B,T,O] 和 reference allclose。"
}
```

这不是剧透，而是把任务从“猜系统规则”变成“执行工程修复”。

---

## 2. Checklist：完成度清单

在右侧 Inspector 上方加一个非常明确的清单：

```text
本关需要完成：

[ ] 输入 hidden 已连接到 MatMul.left
[ ] weight 已经过 Transpose
[ ] Transpose 输出已连接到 MatMul.right
[ ] 输出轴是 [B,T,O]
[ ] allclose 通过
```

清单不需要等测试运行才出现。它是玩家的导航。

普通玩家最怕的是“不知道自己现在应该做什么”。Checklist 可以直接解决这个问题。

---

## 3. Next Step Coach：下一步教练

失败后不要只显示 diagnostic。加一个“下一步动作”区域：

```text
现在先修这里：

MatMul.right 收到的是 weight[O,C]。
但 MatMul 需要 right 的第一轴是 C。

下一步：
1. 添加 Transpose Switch
2. 连接 weight.out -> transpose.x
3. 连接 transpose.out -> matmul.right
```

这个提示可以消耗 hint，影响 Rank。当前已经有 `hintsUsed` 统计。 可以直接复用。

---

## 4. Target Ghost Graph：半透明目标骨架

对前几关，建议在画布上显示半透明虚线目标：

```text
hidden     weight
  |          |
  |       transpose
  |          |
  +------ matmul ---- projected ---- reference
```

玩家连对后，对应虚线变亮。

这比文字说明有效很多。尤其对普通玩家，“目标形状”比“目标公式”更容易理解。

---

# 六、需要修改的反馈机制

## 1. 结构错误也要生成 Trace Frame

当前 `executeGraph` 结构验证失败会返回空 trace。 建议改成：

```ts
if (!validation.ok) {
  return {
    values: {},
    trace: validation.errors.map((error, index) => ({
      step: index,
      nodeId: error.nodeId,
      moduleId: graph.nodes.find(n => n.id === error.nodeId)?.moduleId ?? "Graph",
      inputShapes: {},
      outputShapes: {},
      error
    })),
    error: validation.errors[0]
  };
}
```

这样 Trace Timeline 永远有东西可点，玩家不会遇到“我错了，但界面没告诉我在哪里”的情况。

---

## 2. GraphRunPanel 里增加“定位按钮”

现在测试结果会显示状态、message、expected、received、cause、probe。

建议每个失败项加：

```text
[定位节点]
[显示下一步]
```

点击后：

```text
选中 firstBadNodeId
切到 Shape tab
画布聚焦该节点
高亮相关端口
显示 Next Step Coach
```

---

## 3. 把 missing_input 翻译成端口级行动

现在 required input 检查能知道缺哪个端口。

建议 UI 层翻译：

```text
Required input right is not connected
```

变成：

```text
MatMul 缺少右输入。
请找一个形状为 [C,O] 的 tensor，连接到 matmul.right。
```

如果当前图里存在 `weight_transpose.out`，就直接提示：

```text
推荐连接：weight_transpose.out -> matmul.right
```

---

# 七、参数面板要改成“玩家语言”

当前参数编辑器是通用开发工具，适合内部测试，不适合普通玩家。

建议给 `ModuleDef` 加 `paramSchema`：

```ts
type ParamSchemaItem = {
  key: string;
  label: string;
  help: string;
  kind: "select" | "toggle" | "number" | "axis_pair";
  options?: Array<{
    value: unknown;
    label: string;
    consequence: string;
  }>;
  noviceHidden?: boolean;
};
```

例如 Transpose：

```ts
axisA / axisB 不直接显示

显示为：

交换方式：
- 交换最后两轴：K[B,H,T,D] -> K[B,H,D,T]
- 交换前两轴：错误，会移动 B/H
- 不交换：错误，MatMul 内维度不匹配
```

Tokenizer：

```text
切分策略：
- char：最安全，但 token 数多
- word：token 少，但容易 OOV
- subword：折中，推荐

Fallback：
- none：未知词直接失败
- char：拆成字符
- unk：变成 <unk>
```

Mask：

```text
遮罩方向：
- 屏蔽未来 token：正确
- 屏蔽过去 token：错误
```

这样玩家不是在改参数，而是在做可理解的选择。

---

# 八、关卡顺序建议

当前顺序是：

```text
0-2 MatMul
0-3 Transpose
0-4 Broadcast
0-4F Mask
1-1 Text Type Gate
1-2 Split / Merge
...
```

建议新手公开测试顺序改成：

```text
0-G Graph Basics
1-1 Text Type Gate
1-2 Split / Merge Budget
0-2 MatMul Repair
0-3 Transpose Repair
0-4 Broadcast Repair
0-4F Mask Repair
1-X Tokenizer Gauntlet
```

为什么把 1-1 / 1-2 提前？因为普通玩家更容易理解：

```text
文字不能直接进模型，要先变成 token id
```

比直接理解：

```text
hidden[B,T,C] @ stored_weight[O,C].transpose()
```

更自然。

如果必须严格按照 Chapter 0 → Chapter 1，也至少要在 0-2 前加一个纯操作关：

```text
0-G Graph 操作教学
0-1 Shape Reader Guided
0-2 MatMul Repair
```

---

# 九、建议的 7 天改造排期

## Day 1：修第一分钟

只做 4 件事：

```text
1. App 默认进入 guided，或 Graph 首次弹出操作教学
2. GraphLevels 前面加 0-G Graph Basics
3. 0-2 / 0-3 / 0-4 / 0-4F initialGraph 改成坏机器，不再空图
4. 画布顶部加 Mission Card：第一步请点击 Run Visible
```

验收标准：

```text
一个没玩过的人，30 秒内知道第一步要点 Run Visible。
```

---

## Day 2-3：修失败反馈

```text
1. validation error 也生成 trace frame
2. GraphRunPanel 加“定位节点”
3. firstBadNode 自动聚焦并高亮端口
4. missing_input / shape_mismatch / axis_semantic_error 转成玩家语言
5. 增加 Next Step Coach
```

验收标准：

```text
玩家失败后能说出“我现在要修哪个节点”。
```

---

## Day 4-5：修任务导航

```text
1. LevelSpec 增加 onboarding / checklist
2. 右侧增加关卡 checklist
3. 画布显示 target ghost graph
4. hint 按钮接入 hintsUsed 统计
```

验收标准：

```text
玩家不问开发者，也能照着 checklist 完成第一关。
```

---

## Day 6-7：修参数理解

```text
1. ModuleDef 增加 paramSchema
2. Transpose / Tokenizer / Mask 三类参数改成玩家语言
3. Tokenizer 参数变化后显示 tokens / ids / mask preview
4. axisA / axisB / expectedAxes / inputKey 对新手隐藏
```

验收标准：

```text
玩家知道自己为什么选择 subword、fallback=unk、pad-aware mask、swap last two axes。
```

---

# 十、最推荐的第一关改造示例

## 当前 0-2

```text
空画布
目标：Build hidden[B,T,C] @ stored_weight[O,C].transpose() -> projected[B,T,O]
模块库：InputTensor / WeightPlate / TransposeSwitch / MatMulGate / ...
```

## 建议 0-2A：MatMul Repair

初始画布：

```text
hidden[B,T,C] -----> MatMul.left
weight[O,C]  ------> MatMul.right
MatMul.out   ------> projected
projected    ------> reference
```

玩家第一步：

```text
点击 Run Visible
```

失败反馈：

```text
MatMul.right 的内维度不匹配。
left 最后一轴是 C=3。
right 当前第一轴是 O=5。
MatMul 需要 right 是 [C,O]。

下一步：
在 weight 和 matmul.right 之间插入 Transpose Switch。
```

玩家操作：

```text
删除 weight -> matmul.right
添加 Transpose Switch
连接 weight.out -> transpose.x
连接 transpose.out -> matmul.right
Run Visible
Run Hidden
```

完成后才解锁：

```text
0-2B MatMul Assembly
```

这时可以让玩家从较少节点开始搭，而不是全空。

---

# 十一、当前代码层面的具体修改建议

## 1. 修改 `ch0MatMulGraph`

从：

```ts
initialGraph: {
  levelId: "ch0_2_matmul_graph",
  version: 1,
  nodes: [],
  edges: [],
  outputNodes: ["projected", "reference"]
}
```

改为：

```ts
initialGraph: createCh0MatMulNoTransposeGraph()
```

0-2 源码中已有这个函数。

---

## 2. 修改 `ch0TransposeGraph`

从空图改为：

```ts
initialGraph: createCh0TransposeNoKTransposeGraph()
```

这个函数已经存在，正好表达“没有 K transpose 的坏图”。

---

## 3. 修改 `ch0BroadcastGraph`

优先使用：

```ts
initialGraph: createCh0BroadcastWithoutRailGraph()
```

或者更简单：

```ts
initialGraph: createCh0BroadcastWrongAxisGraph()
```

前者训练“缺少 BroadcastRail”，后者训练“轴对齐错误”。两个函数都已经存在。

---

## 4. 修改 `ch0MaskGraph`

使用：

```ts
initialGraph: createCh0MaskWrongOrientationGraph()
```

这比空图更适合教学，因为它让玩家理解“形状对了，但三角方向错了”。

后续再用：

```ts
createCh0MaskWithoutBroadcastGraph()
```

作为进阶修复。

---

## 5. `GraphWorkbench` 增加 GraphMissionPanel

建议放在 `graphStageHeader` 下方，canvas 上方：

```tsx
<GraphMissionPanel
  level={selectedLevel}
  graph={graph}
  runState={runState}
  onRunVisible={runVisible}
  onShowHint={showNextHint}
/>
```

内容结构：

```text
本关目标
当前机器坏在哪里
第一步
完成清单
下一步建议
```

---

## 6. `executeGraph` 对 validation error 生成 trace

当前 validation error 直接返回空 trace。

建议改为：

```ts
if (!validation.ok) {
  const trace = validation.errors.map((error, index) => ({
    step: index,
    nodeId: error.nodeId,
    moduleId: registry.maybeGet(graph.nodes.find(n => n.id === error.nodeId)?.moduleId ?? "")?.id ?? "Graph",
    inputShapes: {},
    outputShapes: {},
    error
  }));

  return {
    values: {},
    trace,
    error: validation.errors[0]
  };
}
```

这样即使是“缺线”，Trace Timeline 也能显示错误节点。

---

# 十二、人测时的验收指标

建议下一轮人测不要只问“好不好玩”，而是记录这些指标：

```text
首次有效行动时间：
玩家进入后多久完成第一次有效操作？
目标：30 秒内

首次 Run Visible 时间：
玩家多久点击第一次 Run Visible？
目标：60 秒内

第一次失败后是否知道下一步：
玩家能不能说出“我要修哪个节点/哪条线”？
目标：70% 以上

0-G 或 0-2A 首关通过率：
无开发者提示，5 分钟内通过。
目标：60% 以上

退出原因：
是看不懂目标、不会操作、看不懂错误、还是觉得无聊？
```

这几个指标能比主观反馈更快定位问题。

---

# 最终建议

当前 Graph Challenge 的技术方向是对的，但现在还不应该把“空白画布自由构建”作为新手入口。

最优先的改法是：

```text
空图挑战
↓
坏机器维修

专业参数
↓
玩家语言选择

错误报告
↓
下一步行动

模块列表
↓
目标图 + checklist

Graph 默认入口
↓
先有 0-G 操作教学或 guided 过渡
```

我建议先不要继续加新章节，也不要先做更多模块。先把 0-2 MatMul 改成一台“已搭好但少了 Transpose 的坏机器”，让普通玩家能在 3 分钟内完成一次：

```text
Run → Fail → 看 Trace → 修一条线/一个模块 → Pass
```

只要这个闭环顺了，Graph Challenge 的游戏性会明显起来。
