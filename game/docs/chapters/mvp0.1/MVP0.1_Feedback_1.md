我检查了最新 `game` 分支里 MVP0.1 的实现和 `By_chapter_detail_design` 文档。结论比较明确：

**现在的问题不是“玩家不懂”，而是后续大量关卡在设计和实现上已经退化成了同一个模板：输入节点 → 目标组件节点 → 合约节点。玩家只是在完成连线，不需要理解组件内部为什么这样构成，也不需要证明组件真的做了对应计算。**

这也解释了你们的试玩感受：**前几个关卡还可以，因为它们是手工设计的；很快变成无脑拖拽连连看，是因为后面大量关卡被自动生成成了同一种任务。**

---

# 1. 当前提交里已经有的进步

先说好的部分。MVP0.1 的方向是对的，而且这一版已经有了几个重要系统雏形：

`Mvp01App` 已经把 MVP0.1 单独作为 Component Builder 入口，组件认证通过后会写入 `availableComponents`，后续关卡根据组件 ID 判断是否解锁。

`GraphWorkbench` 已经支持 `componentFlow`，会根据 `availableComponentIds` 判断某个组件关卡是否锁定；通过认证后会调用 `onComponentAvailable`，然后自动进入下一关。

`LevelSpec` 也已经加入了 `certification`，支持公开变体、系统变体、控制参数和动态生成测试。

这些说明架构方向没有错。问题出在：**“组件构建”现在还只是表层流程，并没有真正进入组件内部构建。**

---

# 2. 文档层面的主要问题

## 问题一：By_chapter_detail_design 看似完整，实际大多是模板文档

`000_SELF_REVIEW.md` 写到已经检查了 74 个 per-level design documents，并且每个都有 component target、concrete case、player action、challenge、certification variant、pass criteria。

但实际打开具体文档会发现，大量关卡几乎只是同一个模板换了组件名。

比如 `ScalarCell` 文档写的是：

```text
观察预制输入节点和合约探针节点
添加或修复 ScalarCell 节点
将输入端口按语义连接到组件，再将组件输出连接到合约节点
```

挑战是“识别输入/输出语义，而不是只按位置连线”。

`DotProduct` 也是同样的：

```text
添加或修复 DotProduct 节点
将输入端口按语义连接到组件，再将组件输出连接到合约节点
```

挑战仍然是同一句“识别当前组件的输入/输出语义”。

`BoundarySplitter` 也是：

```text
添加或修复 BoundarySplitter 节点
将输入端口按语义连接到组件，再将组件输出连接到合约节点
```

挑战仍然是同样的“不要只按位置连线”。

到了 `QKScore` 这种本应有很强概念挑战的关卡，文档仍然是同一个结构：添加或修复 `QKScore`，连接到合约探针，检查 dtype/axis/shape。

这导致文档没有真正回答每关最重要的问题：

```text
这个组件为什么需要被建造？
它内部由哪些更小组件组成？
玩家会犯什么具体错误？
错误如何暴露知识点？
通过这一关后，玩家到底掌握了什么操作能力？
```

目前文档更像“批量生成的组件清单”，不是可执行的玩法设计。

---

# 3. 实现层面的主要问题

## 问题二：只有前 7 个左右是手工高保真关卡，后面大多走通用生成器

`mvp01GraphLevels.ts` 里只有这些关卡走了 `manualCourseLevelFactories`：

```text
0-0 Wire & Probe
0-1 Scalar Cell
0-3 Vector Rail
0-5 Matrix Struct
0-6 MatMulGate
0-7 TensorBox
3-2 Linear
```

其余大部分关卡全部走 `courseCatalogLevel(courseLevel)` 自动生成。

这正好对应试玩感受：**前几个关卡还可以，后面迅速变成模板化拖拽。**

手工关卡里，`ScalarCell` 有有限 float32 值检查，`VectorRail` 有 c0/c1/c2 顺序，`TensorBox` 有 B/T/C 轴语义，`MatMulGate` 有真实 matmul reference，`Linear` 有 MatMul + bias broadcast + Add 的组合。

但自动生成关卡只是在做：

```text
source input(s)
→ component
→ contract
```

---

## 问题三：后续关卡并没有让玩家“搭建组件”，而是直接给目标组件

自动关卡的生成逻辑里，`modulePalette` 直接就是：

```ts
modulePalette: [courseLevel.moduleId]
```

也就是说，某一关要构建 `BoundarySplitter`，系统就直接把 `BoundarySplitter` 作为可拖拽模块给玩家；要构建 `QKScore`，系统就直接把 `QKScore` 给玩家。

这和 MVP0.1 的目标相反。

目标应该是：

```text
玩家用已有组件搭出 BoundarySplitter
玩家用 MatMul + Transpose 搭出 QKScore
玩家用 Q/K/V/Mask/Softmax 搭出 AttentionHead
```

当前实际是：

```text
系统给 BoundarySplitter 节点
玩家把它拖进图里
连输入输出
通过
```

所以玩家当然不会学到组件内部构成。

---

## 问题四：通用组件节点本质上是“答案发生器”，不是计算实现

`createCourseComponentModule` 会为每个 catalog 组件自动创建一个 `ModuleDef`。这个模块接受 x/y/z 输入，但执行时并不根据输入真正计算，而是直接按参数 `outputDType / outputDims / outputAxes / seed` 生成一个符合合约的输出。

关键代码逻辑是：

```text
检查输入是否存在
读取 outputDType / outputDims / outputAxes
makeRuntimeValue(...)
直接输出对应 dtype / dims / axes 的值
```

也就是说，只要玩家把输入连到这个组件，再连到 contract，就能得到一个“看起来正确”的输出。组件内部并没有实现：

```text
DotProduct = multiply + sum
Splitter = text scanner + policy
QKScore = MatMul + K transpose
Softmax = exp / sum / divide
CrossEntropy = -log p(target)
```

这会直接造成“什么知识都没有学到”。

---

## 问题五：组件输入端口过于宽泛，类型系统失效

自动组件的输入端口 `accepts` 是：

```ts
["float32", "int", "mask", "bool", "raw_text", "string_piece", "token_piece"]
```

也就是说几乎什么都能接进去。

这等于把 MVP0.1 最应该强调的“类型系统 / 接口合约 / dtype 约束”削弱掉了。玩家不会因为把 raw_text 接进 MatMul、把 int 接进 float 运算、把 mask 接进 tokenizer 而形成清晰错误认知。

---

## 问题六：测试只检查 dtype / shape / axes，不检查行为

自动关卡的 `courseContractCase` 只生成两类断言：

```ts
{ type: "dtype", nodeId: "contract", expected: courseLevel.output.dtype }
{ type: "shape", nodeId: "contract", expectedAxes, expectedDims }
```

没有行为测试，没有数值 reference，没有 token 结果，没有 attention mask 语义，没有 softmax row sum。

这会导致：

```text
DotProduct 不需要真的点积
QKScore 不需要真的 Q @ K^T
CausalMask 不需要真的屏蔽未来 token
Softmax 不需要行和为 1
CrossEntropy 不需要看 target token
```

只要输出形状对就可以。

这不是组件认证，而是形状认证。

---

## 问题七：认证变体没有真正增加挑战

自动关卡的认证逻辑会同时改两个东西：

```ts
component: { outputDims: variantDims, seed: ... }
contract: { expectedDims: variantDims }
```

也就是说认证变体主要是在改组件输出参数和合约参数。

这不是在考组件能不能泛化，而是在让“答案发生器”跟着合约改尺寸。

真正的认证应该是：

```text
改变输入数据 / shape / case
保持组件内部实现不变
看它是否自然产生正确输出
```

现在是：

```text
改变组件输出参数
改变合约期望
组件直接吐出对应形状
```

这当然没有挑战性。

---

## 问题八：直接把输入接到 contract 在部分任务中可能也能过当前任务

自动输入节点 `input_0` 的 dtype、dims、axes 很多时候直接使用关卡输出合约。

而测试只看 `contract` 的 dtype/shape。

所以对于很多自动关卡，玩家甚至可能这样做：

```text
input_0.out -> contract.x
```

就能通过当前 visible / hidden 的 shape contract。

文档虽然把“把输出直接接到合约”列为常见错误，但实现上没有足够的测试阻止它。

这说明当前测试目标和设计目标没有对齐。

---

## 问题九：组件库只是“解锁列表”，不是玩家真实构建资产

`Mvp01App` 确实会把认证通过的组件写入 localStorage，包含 `implementationGraph` 和 certification 结果。

但这些组件并没有真正编译成由玩家图实现的动态模块。相反，`mvp01ComponentModules` 里已经提前把大量 course component module 静态注册好了。

这会造成一个结构性问题：

```text
玩家保存了 implementationGraph
但后续使用的 component module 不是从 implementationGraph 执行出来的
而是提前注册好的 static/generated module
```

也就是说，玩家“建造组件”的成果更多是 UI 状态，不是真正的可执行资产。

---

# 4. 根因总结

当前 MVP0.1 不是“组件系统太复杂”，而是**把组件系统的核心目标简化成了组件清单自动生成**。

你们想要的是：

```text
Scalar → Vector → Matrix → Tensor → MatMul → Linear → Attention...
每一步都由玩家用之前的组件搭出来
```

但实现成了：

```text
每一关给一个目标组件节点
输入接进去
输出接合约
通过 dtype/shape
```

所以玩家自然会觉得：

```text
我只是在拖一个节点，把线连上。
我不知道这个组件里面是什么。
我也不知道为什么以后能用它。
```

---

# 5. 改进方向：先停止“全量自动关卡”，回到“关键组件深度关”

## 建议一：不要继续让 74 个关卡都以同一模板可玩

`000_SELF_REVIEW.md` 里说 74 个关卡都已生成。 但当前 74 个关卡的大部分不应该进入 playable route。

建议分三类：

```text
A 类：高保真可玩关卡
真正设计玩法、错误、测试、内部构建

B 类：路线图关卡
在 Challenge Map 上可见，但暂时显示“coming soon / 设计中”

C 类：合并关卡
多个概念合并成一个更强的组件构建挑战
```

MVP0.1 下一版不要追求“0-9 全部可玩”，而是先把 8-12 个关键组件做扎实。

建议第一批高保真关卡：

```text
0-1 ScalarCell
0-3 VectorRail
0-4 DotProduct
0-5 MatrixBoard
0-6 MatMulGate
0-7 TensorBox
1-1 Splitter
1-4 VocabLookup
1-5 TokenBuffer
2-2 EmbeddingLookup
3-2 Linear
4-3 QKScore
```

其他先不要放进主路线，或者放进地图但标记为 roadmap。

---

# 6. 设计文档应该重写成“每关独有机制”

现在每份文档都只有 45 行左右，结构统一，内容抽象。比如 `DotProduct` 文档说学习目标是“相同长度向量才能点积”，但没有说明玩家如何经历这个问题。

新的设计文档每关至少要补这些字段：

```text
1. 前置组件
2. 本关新发明了什么能力
3. 真实案例数据
4. 玩家初始图
5. 玩家需要搭的内部子图
6. 禁止使用的作弊节点
7. 可预期错误路径
8. 错误如何反馈知识点
9. Visible case
10. Hidden / mutation case
11. 行为测试，而不是只有 shape 测试
12. Pack 后的组件接口
13. 后续关卡如何调用该组件
```

例如 `QKScore` 文档不能只写：

```text
QKScore 把 MatMul + K transpose 变成可复用图组件
```

而要写成：

```text
玩家已有：
MatMulGate
TransposeSwitch
AxisTensor

本关目标：
用 MatMulGate 和 TransposeSwitch 造 QKScore

初始错误：
Q[B,T,D] 直接与 K[B,T,D] 相乘，inner dim 错误

玩家任务：
对 K 交换最后两轴，得到 Kᵀ[B,D,T]
再做 Q @ Kᵀ → scores[B,T,T]

隐藏测试：
T == D 时 shape 可能看似没错，但数值 reference 会暴露错误
```

---

# 7. 实现上必须改的核心点

## 改法一：目标组件不能出现在自己的构建关 palette 里

现在自动关卡 `modulePalette: [courseLevel.moduleId]`。

这必须改。

如果本关是构建 `QKScore`，palette 不应该包含 `QKScore`。

应该是：

```text
QKScore 构建关 palette:
- InputTensor
- TransposeSwitch
- MatMulGate
- OutputContractGate
- ReferenceChecker
```

如果本关是构建 `DotProduct`，palette 不应该包含 `DotProduct`。

应该是：

```text
DotProduct 构建关 palette:
- VectorInput
- ElementwiseMultiply
- SumReduce
- OutputContractGate
- ReferenceChecker
```

---

## 改法二：删除或限制 `createCourseComponentModule`

目前 `createCourseComponentModule` 会为每个课程组件自动生成一个能直接输出合约结果的模块。

这就是“无脑连连看”的根源之一。

建议改成：

```text
未认证组件：不能作为可执行 module 使用
已认证组件：由玩家 implementationGraph 生成 DynamicModuleDef
系统预置探针：只用于测试，不算玩家组件
```

也就是说，`courseComponentModules` 不应该提前注册所有目标组件。只有玩家 `Pack/Certify` 之后，才动态加入 registry。

---

## 改法三：BuiltComponent 必须执行玩家的 implementationGraph

现在 `AvailableComponentRecord` 保存了 `implementationGraph`，但后续模块并没有真正基于这个图执行。

下一步应该实现：

```ts
function builtComponentToModuleDef(component: AvailableComponentRecord): ModuleDef {
  return {
    id: component.exportedModuleId,
    inputs: componentInterface.inputs,
    outputs: componentInterface.outputs,
    execute: (ctx) => executeInternalGraph(component.implementationGraph, ctx.inputs),
    infer: (ctx) => inferInternalGraph(component.implementationGraph, ctx.inputs)
  };
}
```

否则“玩家自己搭起来的组件”只是存档，不是系统能力。

---

## 改法四：测试必须从 contract-only 升级为 behavior + contract

现在自动测试只检查 dtype / shape。

每类组件都必须有行为测试：

```text
ScalarCell
- value finite
- rank 0
- NaN / Infinity rejected

VectorRail
- c0/c1/c2 顺序保持
- 长度变体通过

DotProduct
- 输出等于 sum_i x[i] * y[i]
- 长度不一致失败

MatrixMultiply
- 输出每个 cell 等于 dot(row, col)
- inner dim mismatch 失败
- hidden case 中 M/N/O 变化

Splitter
- "tokenizers are useful!" 输出 pieces
- char / word / subword 产生不同 token count

VocabLookup
- known piece → stable id
- unknown piece → OOV 或 fallback

CausalMask
- future positions 被 mask
- 过去位置不能被 mask

Softmax
- 每行 sum=1
- masked logits 概率接近 0

CrossEntropy
- 正确 token logit 越高 loss 越低
```

---

## 改法五：认证变体只能变输入，不能变组件输出答案

现在自动认证会改 `component.outputDims` 和 `contract.expectedDims`。

应该改成：

```text
改变 input tensors
改变 reference outputs
保持组件内部图和参数不变
测试组件是否自然输出正确结果
```

比如 MatMul certification：

```text
输入 hidden[B,T,C] 和 weight[C,O] 变化
reference = real matmul(hidden, weight)
组件图不能改
输出必须 allclose(reference)
```

Linear certification：

```text
输入 hidden / weight / bias 变化
reference = hidden @ weight + bias
组件图不能改
```

Tokenizer certification：

```text
输入文本变化
检查 pieces / ids / mask 是否合理
组件策略不能随测试临时改
```

---

# 8. 关卡体验怎么改：从“连线任务”变成“诊断 + 建造 + 证明”

每个高保真关应该有三段：

```text
A. 诊断一个具体失败
B. 搭出内部实现
C. 用隐藏变体证明组件真的成立
```

---

## 示例：重做 DotProduct 关

### 当前问题

文档说 DotProduct 学习目标是“相同长度向量才能点积”，但实现里自动组件只是输出 `float32[]`。

### 新设计

初始案例：

```text
query = [0.2, -0.5, 1.0]
key   = [0.4, 0.1, -0.3]

目标：
score = 0.2*0.4 + (-0.5)*0.1 + 1.0*(-0.3)
```

玩家不能使用 `DotProduct` 节点，只能使用：

```text
ElementwiseMultiply
SumReduce
OutputContractGate
ReferenceChecker
```

玩家搭：

```text
query, key
→ elementwise_multiply
→ sum_reduce
→ score_out
```

Visible test：

```text
shape = []
allclose(score, reference)
```

Hidden test：

```text
长度 C 从 3 改成 5
仍然必须 sum over C
```

错误路径：

```text
只取第一个元素 → visible 可能部分像，但 hidden fail
直接连 query → shape 不对
没有 sum reduce → 输出 [C]，contract fail
```

这样玩家能真正理解 dot product。

---

## 示例：重做 Splitter 关

### 当前问题

文档说 BoundarySplitter 学习目标是“文本要切成 pieces”，但实现只要求输出 `string_piece[T]`。

### 新设计

输入文本：

```text
tokenizers are useful!
```

玩家已有：

```text
TextInput
CharScanner
WhitespaceBoundary
PunctuationKeeper
PieceBuffer
```

玩家要搭：

```text
TextInput
→ scanner
→ boundary_splitter
→ piece_buffer
```

Visible output：

```text
["tokenizers", "are", "useful", "!"]
```

Hidden cases：

```text
"shape, token buffer"
"code_snake_case + punctuation"
```

错误路径：

```text
丢标点 → hidden fail
不按空格切 → pieces 超长
按字符切 → budget fail
```

这样玩家才会理解“为什么切分”。

---

## 示例：重做 QKScore 关

### 当前问题

文档说 QKScore 来自 MatMul + K transpose，但实现里可以直接拖 QKScore 节点。

### 新设计

玩家必须使用自己之前认证过的：

```text
MatMulGate
TransposeSwitch
AxisLock
```

Palette：

```text
InputTensor(Q)
InputTensor(K)
TransposeSwitch
MatMulGate
ScoreBoardContract
ReferenceChecker
CellTrace
```

玩家搭：

```text
Q[B,T,D] -> MatMul.left
K[B,T,D] -> Transpose(-2,-1) -> MatMul.right
MatMul.out -> scores[B,T,T]
```

Visible test：

```text
scores shape [B,T,T]
allclose(Q @ K^T)
```

Hidden test：

```text
T == D trap
B/T/D mutation
operand order trap
```

错误路径：

```text
不转置 K → inner dim mismatch
转错轴 → shape 可能对但 reference fail
Q/K 顺序反 → cell trace fail
```

这样 QKScore 才是一个真正的组件构建挑战。

---

# 9. 对现有代码的具体修改建议

## P0：关闭大部分自动生成关卡的可玩入口

在 `mvp01GraphLevels.ts` 中，把：

```ts
export const mvp01GraphLevels = mvp01CourseLevels.map(...)
```

改成：

```ts
const playableCodes = new Set([
  "0-0",
  "0-1",
  "0-3",
  "0-4",
  "0-5",
  "0-6",
  "0-7",
  "1-1",
  "1-4",
  "3-2",
  "4-3"
]);

export const mvp01GraphLevels = mvp01CourseLevels
  .filter(level => playableCodes.has(level.code))
  .map(...)
```

其余显示在路线图，但不可点击或标为 roadmap。

---

## P0：禁止目标组件作为自己构建关的模块

把自动生成逻辑：

```ts
modulePalette: [courseLevel.moduleId]
```

改成：

```ts
modulePalette: paletteForComponentBuild(courseLevel)
```

例如：

```ts
function paletteForComponentBuild(level) {
  switch (level.slug) {
    case "dot_product":
      return ["ElementwiseMultiply", "SumReduce", "OutputContractGate", "ReferenceChecker"];

    case "qk_score":
      return ["InputTensor", "TransposeSwitch", "component.matmul_gate.v1", "OutputContractGate", "ReferenceChecker"];

    default:
      return [];
  }
}
```

没有高保真设计的关卡不要自动开放。

---

## P0：添加路径断言，防止 input 直连 contract

新增测试断言：

```ts
{ type: "requires_node"; nodeId: "component" }
{ type: "requires_edge_path"; from: "input_0", through: "component", to: "contract" }
```

当前自动测试只看 contract 的 dtype/shape，很容易让玩家绕过组件。

---

## P1：移除自动答案组件

`createCourseComponentModule` 不应该为所有 buildable 组件提前生成答案节点。

改为：

```text
系统内置模块：只包含基础原语和探针
玩家认证组件：通过 BuiltComponent 动态注册
未认证目标：不能作为节点
```

---

## P1：重写认证逻辑

自动认证现在用：

```ts
withNodeParams(graph, {
  component: { outputDims: variantDims },
  contract: { expectedDims: variantDims }
})
```

这要改掉。

正确做法：

```text
生成新的 input_0 / input_1 / reference
保持 component graph 不变
运行 reference allclose / behavior assertions
```

---

## P1：拆分“探针节点”和“玩家组件节点”

现在玩家容易把 `ReferenceChecker`、`OutputContractGate`、`TypeContractGate` 当成普通组件。MVP0.1 应明确分层：

```text
Player-built components:
ScalarCell, VectorRail, MatMulGate...

Test equipment:
ContractGate, ReferenceChecker, Probe
```

UI 上应该不同颜色，并标明：

```text
Test equipment is not packed into your component.
```

0-0 已经在文案里解释 reference 是预制探针，不是可复用组件。 这个原则要扩展到所有关卡。

---

# 10. 设计文档改写模板

每个 By_chapter_detail_design 文档建议改成这样：

```text
# CHx-y Component Name

## 1. 组件真实用途
它未来会在哪些关卡被复用？

## 2. 前置组件
玩家必须已经认证哪些组件？

## 3. 本关新增能力
本关到底新增了什么能力？

## 4. 具体案例
给出具体数值 / 文本 / token / mask / matrix。

## 5. 初始错误图
玩家一开始看到什么坏机器？

## 6. 目标内部实现
不是目标节点，而是内部 subgraph。

## 7. 玩家操作
具体到连接、参数、调试步骤。

## 8. 错误路径
至少 3 种常见错误。

## 9. Visible 测试
公开输入和期望输出。

## 10. Hidden 测试
变体如何证明泛化？

## 11. Pack 后接口
组件封装后暴露哪些端口？

## 12. 后续调用
下一关如何使用这个组件？
```

---

# 11. 推荐下一轮修复优先级

## 第一优先级：先修体验断点

保留当前手工关卡，暂停后续自动关卡。

```text
0-0 Wire & Probe
0-1 ScalarCell
0-3 VectorRail
0-5 MatrixStruct
0-6 MatMulGate
0-7 TensorBox
3-2 Linear
```

这些可以继续打磨。

---

## 第二优先级：补两个高质量中段样例

建议补：

```text
0-4 DotProduct
1-1 BoundarySplitter
```

原因：

```text
DotProduct 能证明“组件内部搭建”不是连线。
BoundarySplitter 能证明“文本处理”不是 shape contract。
```

---

## 第三优先级：补一个高级样例

建议补：

```text
4-3 QKScore
```

这能验证玩家是否真的把 MatMul、Transpose、Axis Semantics 用到了 Attention。

---

## 第四优先级：重构组件库

从：

```text
静态注册所有组件
```

改成：

```text
只注册基础 primitive
玩家认证后动态注册 BuiltComponent
```

---

# 12. 最终判断

你们现在的 MVP0.1 提交已经搭出了“组件路线”的壳：

```text
路线图
组件解锁
认证按钮
组件库
关卡文档
```

但还没有搭出“组件构建”的肉：

```text
目标组件内部没有真正由前置组件搭成
测试没有验证行为
认证没有证明泛化
已认证组件没有真正成为可执行资产
```

所以玩家会感觉：

```text
我不是在造系统
我是在把一个现成模块从左边拖到中间，再接到右边
```

下一版最重要的改法不是继续增加关卡，而是**减少可玩关卡数量，把关键组件做深**。只要能让玩家真正经历一次：

```text
用 Scalar 造 Vector
用 Vector 造 DotProduct
用 DotProduct 造 MatMul
用 MatMul 造 Linear
```

并且后面的 Linear 真的调用玩家自己的 MatMul，这个方向就会成立。当前最大风险是继续沿着自动生成模板扩展，那会把 MVP0.1 变成一个很长但很空的“组件连线清单”。
