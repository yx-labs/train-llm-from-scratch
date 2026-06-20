# MVP0.0.5 执行计划：Graph Runtime Gameplay Correction

> 日期：2026-06-20  
> 输入文档：`LLM_Complete_Gameplay_Correction_Executable_Design.md`、`MVP0_02_IMPLEMENTATION_PLAN.md`、`TECH_DESIGN.md`、`LLM_Complete_GDD.md`、Chapter 0/1 相关详细设计  
> 当前工程：`game/app`，Vite + React + TypeScript + PixiJS  
> 基线验证：`pnpm -C game/app typecheck` 通过  
> 目标：把当前 tag-slot / guided repair 原型推进为可运行、可失败、可调试、可修复的 Graph Challenge MVP

---

## 1. 当前判断

### 1.1 现有工程状态

当前 `game/app/src` 已经完成大量 Chapter 0/1 内容：

```text
App.tsx                         大型 React 页面、阶段状态、面板、modal、评分
PixiWorkbenchCanvas.tsx          Pixi 画布、节点/边/阶段视觉
workbenchTypes.ts                Bootcamp / Repair / Tensor 类型
bootcampLevels.ts                0-1 主数据与 evaluateBootcampLevel
chapter02MatMulGateLevel.ts      0-2 Guided Repair 内容
chapter03TransposeTrapLevel.ts   0-3 Guided Repair 内容
chapter04BroadcastAddLevel.ts    0-4 Guided Repair 内容
chapter1TokenizationLevel.ts     Chapter 1 Guided Repair 内容
toyTokenizer.ts                  Toy tokenizer 运行逻辑，可复用
```

当前实现的核心判定仍然是：

```text
RepairSlot.correctTagIds
→ evaluateBootcampLevel
→ visible / behavior / reference / hidden 的文字化检查结果
```

这适合作为 Guided Repair Mode，但不足以满足 mvp0.0.5 设计文档要求的主玩法：

```text
GraphSpec
→ Module Registry
→ Tiny Runtime
→ Test Runner
→ Trace Report
→ Player Fix
→ Hidden Tests
```

### 1.2 版本方向

mvp0.0.5 不应重写现有 Bootcamp，也不应继续扩写更多知识卡。正确策略是：

```text
保留现有 Chapter 0/1 Guided Repair
新增独立 gameplayGraph 层
先用 3 个 vertical slice 证明 Build-Test-Debug 闭环
再逐步把成熟玩法回灌到已有关卡
```

### 1.3 技术原则

```text
1. Runtime 优先于视觉扩展。
2. Graph Challenge 独立实现，不把真实运行时塞进 correctTagIds。
3. 第一版允许半固定 graph 编辑，先证明执行、测试、trace。
4. TinyTensor 只支持小尺寸 deterministic arrays。
5. Tokenizer 复用 toyTokenizer，避免重新造完整 BPE。
6. 不引入后端，不接 PyTorch，不做 autograd。
7. 所有新增代码保持在 game/app/src/gameplayGraph 下，降低对 App.tsx 的冲击。
```

---

## 2. MVP0.0.5 版本目标

### 2.1 产品目标

玩家必须能完成这样的闭环：

```text
打开 Graph Challenge
→ 查看目标合同
→ 从 Module Palette 添加或启用模块
→ 连接端口 / 设置参数
→ Run Visible Tests
→ 看到 firstBadNode / expected / received / trace step
→ 修改 graph 或参数
→ 再运行
→ Visible 通过后 Run Hidden Tests
→ 获得 Rank 和 Debrief
```

### 2.2 必须交付的 vertical slices

P0 必须完成：

```text
Slice A：Chapter 1 Tokenizer Machine
Slice B：0-2 MatMul Gate Graph
Slice C：0-3 Transpose Trap Graph
```

P1 强推荐完成：

```text
Slice D：0-4 Broadcast Add Mini / Mask Add Mini
```

### 2.3 版本成功标准

版本完成时必须满足：

```text
[ ] 至少 3 个 graph_challenge level 可运行
[ ] 每个 graph level 都使用 GraphSpec / ModuleRegistry / TestRunner 判定
[ ] Visible Tests 和 Hidden Tests 输入不同
[ ] 失败报告包含 firstBadNode、expected、received、traceFrame、suggestedProbe
[ ] 至少 1 个关卡能抓住 shape pass / numeric fail
[ ] Tokenizer Machine 能处理 OOV、budget、padding、mask
[ ] MatMul Gate 能处理 [C,O] 和 stored [O,C] + transpose
[ ] Transpose Trap 能处理 T == D numeric trap
[ ] 现有 Guided Repair 入口不破坏
[ ] `pnpm -C game/app typecheck` 和 `pnpm -C game/app build` 通过
```

---

## 3. 范围冻结

### 3.1 P0 范围

#### Runtime

```text
[ ] GraphSpec / LevelSpec / ModuleDef / RuntimeValue 类型
[ ] ModuleRegistry
[ ] Graph structure validation
[ ] Shape / dtype inference
[ ] TinyTensor flat array representation
[ ] graphExecutor topological execution
[ ] trace capture
[ ] testRunner
[ ] reference implementations
[ ] scoring / rank calculation
```

#### Modules

```text
[ ] TextInput
[ ] TokenizerSocket
[ ] VocabLookup
[ ] SpecialTokenInjector
[ ] PaddingBuilder
[ ] AttentionMaskBuilder
[ ] EmbeddingReadyProbe
[ ] InputTensor
[ ] WeightPlate
[ ] TransposeSwitch
[ ] MatMulGate / BatchedMatMulGate
[ ] OutputContractGate
[ ] ReferenceChecker
```

#### UI

```text
[ ] GraphWorkbench 页面
[ ] Mode switch：Guided Repair / Graph Challenge
[ ] ModulePalette
[ ] GraphCanvasBridge，第一版可半固定布局
[ ] InspectorPanel：Summary / Shape / Values / Tests / Code
[ ] ParameterEditor
[ ] TestReportPanel
[ ] TraceTimeline
[ ] Rank / Debrief panel
```

#### Content

```text
[ ] ch1TokenizerMachine LevelSpec
[ ] ch0MatMulGraph LevelSpec
[ ] ch0TransposeGraph LevelSpec
[ ] visible / hidden tests
[ ] failure messages
[ ] debrief copy
```

#### QA

```text
[ ] Runtime unit tests
[ ] Level fixture tests
[ ] UI smoke checklist
[ ] Existing bootcamp regression checklist
```

### 3.2 P1 范围

```text
[ ] BroadcastRail
[ ] AddGate
[ ] logical broadcast plan
[ ] CellTrace for broadcast indices
[ ] CausalMask / MaskApply mini
[ ] 0-4 Broadcast Add Mini LevelSpec
[ ] ReferenceDiff mini view
```

### 3.3 P2 暂不做

```text
[ ] 完整自由画布自动布局
[ ] 完整真实 BPE / SentencePiece / tiktoken
[ ] 真实 PyTorch / autograd
[ ] 后端 FastAPI 接入
[ ] 3D tensor 重做
[ ] 完整 Attention softmax / V path
[ ] 沙盒、排行榜、多人、云存档
```

---

## 4. 新架构落点

### 4.1 新增目录

建议新增：

```text
game/app/src/gameplayGraph/
  types.ts
  moduleRegistry.ts
  levelRegistry.ts
  scoring.ts

  runtime/
    tinyTensor.ts
    shapeInference.ts
    graphValidation.ts
    graphExecutor.ts
    testRunner.ts
    trace.ts
    assertions.ts
    reference.ts

  modules/
    tokenizerModules.ts
    tensorModules.ts
    attentionModules.ts

  levels/
    ch1TokenizerMachine.ts
    ch0MatMulGraph.ts
    ch0TransposeGraph.ts
    ch0BroadcastMini.ts

  ui/
    GraphWorkbench.tsx
    ModulePalette.tsx
    GraphCanvasBridge.tsx
    InspectorPanel.tsx
    ParameterEditor.tsx
    TestReportPanel.tsx
    TraceTimeline.tsx
    TokenBufferGrid.tsx
    TensorValuePreview.tsx
```

测试目录：

```text
game/app/src/gameplayGraph/runtime/*.test.ts
game/app/src/gameplayGraph/levels/*.test.ts
```

### 4.2 最小数据流

```text
LevelSpec
→ initial GraphSpec
→ player edits graph / params
→ validateGraphStructure
→ inferPortTypes
→ inferShapes
→ executeGraph
→ capture TraceFrame[]
→ runAssertions
→ TestResult[]
→ scoreGraphRun
→ UI report
```

### 4.3 现有代码的接入方式

```text
App.tsx
  增加顶层 mode/tab 或 level mode 判断
  Guided Repair 继续走现有 bootcampLevels/evaluateBootcampLevel
  Graph Challenge 走 GraphWorkbench

toyTokenizer.ts
  迁入或包装为 tokenizerModules 的实现依赖
  不破坏现有导出

workbenchTypes.ts
  保持 Bootcamp 类型，不承载 GraphSpec

PixiWorkbenchCanvas.tsx
  保留现有 guided repair 视觉
  GraphCanvasBridge 第一版用 React/SVG/HTML 实现也可以
  若复用 Pixi，只做渲染桥，不把 runtime state 写进 Pixi 内部
```

### 4.4 图编辑器取舍

第一版不强制接 `@xyflow/react`。当前工程已经有 Pixi 画布，且 mvp0.0.5 的核心风险是 runtime，不是画布库。

推荐顺序：

```text
MVP0.0.5：半固定 graph + click-to-connect + parameter editor
MVP0.0.6：如果连接/拖拽成本继续上升，再评估 @xyflow/react
```

半固定 graph 可接受的交互：

```text
[ ] 玩家可以从 palette 添加允许模块
[ ] 玩家可以选择源端口和目标端口建立边
[ ] 玩家可以删除边
[ ] 玩家可以编辑模块参数
[ ] 节点可以固定布局或轻量拖动
```

---

## 5. 开发里程碑

### Milestone 0：范围冻结与基线保护

周期：0.5-1 天

目标：

```text
确认 Guided Repair 不重写
确认 Graph Challenge 独立接入点
建立测试脚本和空目录
```

任务：

```text
[ ] 新增 gameplayGraph 目录和 types 空骨架
[ ] 在 App 中确定 Graph Challenge 入口方案
[ ] 添加 `test` 脚本和 Vitest，或先建立 runtime smoke runner
[ ] 记录当前 `typecheck` 通过状态
[ ] 写最小 ADR：为什么本版不做后端/不做完整自由画布
```

交付：

```text
[ ] 可编译的空 graph runtime scaffold
[ ] 不影响当前页面
```

验收：

```text
pnpm -C game/app typecheck
pnpm -C game/app build
```

### Milestone 1：Graph Runtime Skeleton

周期：2-3 天

目标：

```text
一个固定 GraphSpec 能被执行，失败能返回 trace 和 TestResult。
```

任务：

```text
[ ] types.ts 定义 DType、AxisName、TensorShape、PortRef、GraphNode、GraphEdge、GraphSpec
[ ] RuntimeValue 支持 dtype、shape、data、meta
[ ] TraceFrame / RuntimeError / TestCase / TestAssertion / TestResult
[ ] ModuleDef：ports、paramsSchema、inferShape、execute、describe
[ ] ModuleRegistry 注册与查询
[ ] graphValidation：节点存在、端口存在、必需输入连接、无环
[ ] shapeInference：按拓扑顺序推导 output shapes
[ ] graphExecutor：执行小图并收集 trace
[ ] assertions：dtype、shape、axis_semantics、allclose、token_budget、mask_pad
[ ] testRunner：visible/hidden/reference 分组执行
```

P0 模块实现：

```text
[ ] ConstantTextInput
[ ] InputTensor
[ ] WeightPlate
[ ] TransposeSwitch 2D
[ ] MatMulGate 2D/3D simple
[ ] OutputContractGate
[ ] ReferenceChecker
```

验收：

```text
[ ] 能执行 hardcoded MatMul graph
[ ] 缺边时报 blocked / firstBadNode
[ ] shape mismatch 能定位到 MatMulGate
[ ] allclose fail 能返回 max_abs_error
[ ] TraceFrame 至少包含 nodeId、moduleId、inputShapes、outputShapes、samples、error
```

### Milestone 2：Graph Workbench UI Shell

周期：2-3 天

目标：

```text
玩家能在浏览器进入 Graph Challenge，查看关卡、模块、图、测试报告和 trace。
```

任务：

```text
[ ] GraphWorkbench 接收 levelId
[ ] ModulePalette 根据 level.modulePalette 显示模块
[ ] GraphCanvasBridge 显示节点、端口、边、状态
[ ] click-to-connect 建边
[ ] ParameterEditor 编辑 selected node params
[ ] InspectorPanel 展示模块合同、当前 shape、dtype、sample
[ ] TestReportPanel 展示 visible/hidden/reference 结果
[ ] TraceTimeline 点击 step 选中节点
[ ] Run Visible / Run Hidden 按钮
[ ] Visible 未通过时 Hidden disabled
```

验收：

```text
[ ] 进入 Graph Challenge 不影响 Guided Repair
[ ] 运行空图会显示结构错误
[ ] 运行正确固定图会通过 visible
[ ] 失败节点在 canvas / inspector / trace 三处联动
```

### Milestone 3：Slice A - Tokenizer Machine

周期：2-3 天

目标：

```text
复用 toyTokenizer，实现 raw text → token_ids[B,T] + attention_mask[B,T] 的可运行策略关。
```

模块：

```text
[ ] TextInput
[ ] TextInspector
[ ] TokenizerSocket
[ ] VocabLookup
[ ] FallbackSplitter
[ ] SpecialTokenInjector
[ ] PaddingBuilder
[ ] AttentionMaskBuilder
[ ] EmbeddingReadyProbe
```

参数：

```text
policy: char | word | subword
applyMerges: boolean
fallback: none | char | unk
preservePunctuation: boolean
addBos: boolean
addEos: boolean
maxLength: number
padToLength: number
padSide: left | right
maskPolicy: pad-aware | all-ones
```

Visible tests：

```text
[ ] "we train llm"
[ ] ["we train llm", "shape"]
[ ] token_ids dtype == int
[ ] token_ids axes == [B,T]
[ ] attention_mask axes == [B,T]
[ ] mask_pad matches padId
```

Hidden tests：

```text
[ ] "unknown glyph ?"
[ ] "a very very long sequence"
[ ] punctuation-heavy text
[ ] mixed Chinese / emoji falls back to unk or char
[ ] repeated run deterministic
```

验收：

```text
[ ] raw_text 直接进 EmbeddingReadyProbe 必须失败
[ ] word policy 可能触发 OOV，subword/unk/char 能修复
[ ] token budget 超限能定位 TokenizerSocket 或 Budget/Padding 节点
[ ] PAD 位置 mask=0，content 位置 mask=1
[ ] Hidden test 不与 visible 使用完全相同输入
```

### Milestone 4：Slice B - 0-2 MatMul Gate Graph

周期：2-3 天

目标：

```text
玩家构建 Linear Projection：hidden[B,T,C] @ weight[C,O] → projected[B,T,O]。
```

模块增强：

```text
[ ] TinyTensor deterministic random / seeded values
[ ] WeightPlate orientation: "C,O" | "O,C"
[ ] TransposeSwitch 2D
[ ] MatMulGate supports [...,C] @ [C,O]
[ ] OutputContractGate validates axes and dims
[ ] ReferenceChecker allclose
```

Visible tests：

```text
V1 standard:
hidden[B=2,T=4,C=3]
weight[C=3,O=5]
expected [B,T,O] = [2,4,5]

V2 stored orientation:
hidden[B=1,T=4,C=3]
stored_weight[O=5,C=3]
requires transpose
expected [1,4,5]
```

Hidden tests：

```text
H1 no size guessing:
hidden[B=4,T=3,C=2]
weight[C=2,O=16]

H2 C/O equal orientation trap:
hidden[B=2,T=3,C=4]
stored_weight[O=4,C=4]
shape may pass, numeric must fail if orientation wrong

H3 output axis order:
reject [B,O,T] / [T,B,O]
```

验收：

```text
[ ] [C,O] 直接通过
[ ] [O,C] 必须 transpose 才能数值通过
[ ] shape 正确但数值错误会 fail
[ ] firstBadNode 定位 MatMulGate / TransposeSwitch / OutputContractGate
[ ] ReferenceChecker 显示 max_abs_error
```

### Milestone 5：Slice C - 0-3 Transpose Trap Graph

周期：3-4 天

目标：

```text
玩家修复 Q[B,H,T,D] @ Kᵀ[B,H,D,T] → scores[B,H,T,T]。
```

模块增强：

```text
[ ] TransposeSwitch rank 4
[ ] negative axis index: -2 / -1
[ ] AxisLock for B/H
[ ] BatchedMatMulGate supports carry axes
[ ] ScoreBoard output axes Tq/Tk
[ ] CellTrace for scores[b,h,i,j]
[ ] ReferenceChecker QK^T
```

Visible tests：

```text
[ ] B=1,H=1,T=3,D=2 single-head QK
[ ] B=2,H=2,T=4,D=3 multi-head QK
[ ] output axes [B,H,Tq,Tk]
```

Hidden tests：

```text
H1 T != D:
B=2,H=2,T=4,D=3
no transpose shape mismatch

H2 T == D:
B=1,H=3,T=4,D=4
wrong Q @ K may shape pass but numeric fail

H3 carry axes swapped:
B=3,H=2,T=5,D=4
B/H must be preserved

H4 operand order reversed:
K_t @ Q must fail semantics or numeric
```

验收：

```text
[ ] K.transpose(-2,-1) 通过
[ ] 转置 Q 判错
[ ] 全轴反转判错
[ ] B/H 被移动判错
[ ] T == D trap 被 numeric/cell trace 抓住
[ ] CellTrace 显示 scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:])
```

### Milestone 6：Slice D - 0-4 Broadcast / Mask Mini

周期：2-3 天，P1

目标：

```text
证明 Broadcast Add 不是只看 shape，而是 axis alignment + logical broadcast + numeric reference。
```

模块：

```text
[ ] BroadcastRail
[ ] AddGate
[ ] AxisAlignmentRuler
[ ] GhostExpansionPreview
[ ] CausalMask
[ ] MaskApply 或 additive mask AddGate
[ ] CellTrace index mapping
```

MVP cases：

```text
[ ] bias[O] + projected[B,T,O]
[ ] pos_emb[T,C] + tok_emb[B,T,C]
[ ] mask[1,1,Tq,Tk] + scores[B,H,Tq,Tk]
```

Hidden traps：

```text
[ ] [B,T,C] + [T] where T != C fails
[ ] [B,T,C] + [T] where T == C shape may pass but semantic/numeric fail
[ ] mask Tq/Tk orientation swapped
[ ] materialized repeat gets warning, not hard fail
```

验收：

```text
[ ] BroadcastRail 显示 axis alignment
[ ] CellTrace 显示 out index 对应小张量 index
[ ] mask orientation 错误能被 hidden test 抓住
```

### Milestone 7：Polish、QA、回归

周期：2-3 天

目标：

```text
让版本可演示、可验证、可继续迭代。
```

任务：

```text
[ ] 错误文案按设计文档格式统一
[ ] Rank C/B/A/S 接入 Graph Challenge
[ ] Hints used / failed runs / probes used 计数
[ ] Graph progress 写入 localStorage，使用独立 key
[ ] README 更新运行与验证命令
[ ] 手动 playtest checklist
[ ] Guided Repair regression
[ ] build/typecheck/test 全通过
```

---

## 6. 任务拆分

### 6.1 Runtime 任务

```text
R-001 定义 Graph 类型
R-002 定义 ModuleDef 接口
R-003 实现 ModuleRegistry
R-004 实现 TinyTensor flat data
R-005 实现 deterministic seeded tensor generator
R-006 实现 transpose
R-007 实现 matmul 2D
R-008 实现 batched matmul
R-009 实现 allclose / max_abs_error
R-010 实现 tokenizer runtime adapter
R-011 实现 padding / mask builder
R-012 实现 graph validation
R-013 实现 shape inference
R-014 实现 graph execution
R-015 实现 trace capture
R-016 实现 assertions
R-017 实现 test runner
R-018 实现 scoring
```

### 6.2 UI 任务

```text
U-001 App 增加 Graph Challenge 入口
U-002 GraphWorkbench layout
U-003 ModulePalette
U-004 GraphCanvasBridge 节点/边渲染
U-005 端口选择与连线
U-006 边删除与节点参数编辑
U-007 Inspector Summary
U-008 Inspector Shape
U-009 Inspector Values
U-010 Inspector Tests
U-011 Inspector Code
U-012 TestReportPanel
U-013 TraceTimeline
U-014 TokenBufferGrid
U-015 TensorValuePreview
U-016 ScoreBoard mini
U-017 Rank / Debrief
```

### 6.3 Content 任务

```text
C-001 ch1TokenizerMachine LevelSpec
C-002 ch0MatMulGraph LevelSpec
C-003 ch0TransposeGraph LevelSpec
C-004 ch0BroadcastMini LevelSpec
C-005 failure message table
C-006 debrief table
C-007 hint table
C-008 visible test fixtures
C-009 hidden test fixtures
```

### 6.4 QA 任务

```text
Q-001 tinyTensor transpose tests
Q-002 tinyTensor matmul tests
Q-003 tokenizer graph tests
Q-004 matmul visible/hidden fixture tests
Q-005 transpose visible/hidden fixture tests
Q-006 shape pass / numeric fail tests
Q-007 mask_pad tests
Q-008 graph validation missing edge tests
Q-009 UI smoke: run visible fail/pass
Q-010 UI smoke: hidden locked until visible pass
Q-011 guided repair regression
Q-012 build/typecheck final gate
```

---

## 7. LevelSpec 交付细则

### 7.1 LevelSpec 最小字段

```ts
type LevelSpec = {
  id: string;
  title: string;
  mode: "graph_challenge";
  chapter: string;
  goal: string;
  modulePalette: string[];
  initialGraph: GraphSpec;
  constraints: {
    maxNodes?: number;
    maxEdges?: number;
    forbiddenModules?: string[];
  };
  visibleTests: TestCase[];
  hiddenTests: TestCase[];
  debrief: {
    completeTitle: string;
    fixedProblem: string;
    learned: string;
    nextUse: string;
  };
};
```

### 7.2 ModuleDef 最小字段

```ts
type ModuleDef = {
  id: string;
  label: string;
  category: "data" | "tokenizer" | "tensor" | "attention" | "probe" | "contract";
  inputs: PortDef[];
  outputs: PortDef[];
  defaultParams: Record<string, unknown>;
  infer: (ctx) => InferResult;
  execute: (ctx) => ExecuteResult;
  summary: string;
  pseudoCode?: string;
};
```

### 7.3 失败报告格式

UI 和 runtime 统一使用：

```text
[Error Type]
一句话说明发生了什么。

Expected:
{expected contract}

Received:
{received contract}

Why it matters:
{工程原因}

Try:
{推荐检查工具}
```

---

## 8. 测试计划

### 8.1 新增脚本

建议在 `game/app/package.json` 增加：

```json
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest"
  }
}
```

### 8.2 必跑命令

每个 milestone 完成后运行：

```powershell
pnpm -C game/app typecheck
pnpm -C game/app test
```

最终交付前运行：

```powershell
pnpm -C game/app typecheck
pnpm -C game/app build
pnpm -C game/app test
```

### 8.3 Runtime 单测矩阵

```text
Tokenizer:
[ ] raw text direct to embedding fails
[ ] subword + merges produces deterministic ids
[ ] OOV fallback resolves unknown
[ ] pad mask matches pad id
[ ] budget truncation preserves EOS when required

MatMul:
[ ] [B,T,C] @ [C,O] -> [B,T,O]
[ ] [B,T,C] @ [O,C] fails without transpose
[ ] transpose([O,C]) -> [C,O]
[ ] C/O equal orientation trap numeric fail
[ ] output axis reorder fails

Transpose:
[ ] 2D transpose value map
[ ] rank4 transpose(-2,-1)
[ ] carry axes B/H preserved
[ ] T == D wrong op numeric fail
[ ] operand order reversed fails

Broadcast P1:
[ ] [B,T,O] + [O]
[ ] [B,T,C] + [T,C]
[ ] [B,H,Tq,Tk] + [1,1,Tq,Tk]
[ ] Tq/Tk swapped fails
```

### 8.4 手动验收路径

```text
1. 打开 app。
2. 进入 Guided Repair，确认 0-1/0-2 仍可选择。
3. 切到 Graph Challenge。
4. 打开 Tokenizer Machine，故意 raw text 直连 EmbeddingReadyProbe，确认失败。
5. 修复 tokenizer pipeline，visible 通过，hidden 通过。
6. 打开 MatMul Gate，故意不转置 stored [O,C]，确认 shape/numeric 报错。
7. 插入 transpose，hidden 通过。
8. 打开 Transpose Trap，故意 Q @ K，确认错误。
9. 修复 K.transpose(-2,-1)，运行 T==D hidden trap，确认 numeric/reference 判定生效。
10. 检查 Rank / Debrief。
```

---

## 9. 代码质量约束

```text
[ ] Runtime 纯函数优先，避免依赖 React state。
[ ] GraphSpec / LevelSpec 不直接引用 UI 类型。
[ ] ModuleDef 的 execute 不修改输入 RuntimeValue。
[ ] TinyTensor 不支持大 tensor，超过限制直接 RuntimeError。
[ ] TestRunner 不吞异常，异常转 RuntimeError。
[ ] UI 不直接重新实现 runtime 逻辑。
[ ] App.tsx 只做接入，不继续扩成更大的业务核心。
[ ] 新增文案优先放 LevelSpec / message table。
```

建议限制：

```text
max tensor elements <= 4096
max graph nodes P0 <= 16
max graph edges P0 <= 24
trace frames <= executed nodes + assertion frames
```

---

## 10. 风险与控制

| 风险 | 表现 | 控制 |
|---|---|---|
| App.tsx 继续膨胀 | 新 Graph 逻辑混入现有 monolith | 新增 `gameplayGraph/ui`，App 只挂入口 |
| 图编辑器耗时过大 | 拖拽、连线、布局吃掉 runtime 工期 | 第一版半固定 graph + click-to-connect |
| Runtime 范围膨胀 | 想做完整 torch/numpy | 只做 P0 模块，小 tensor，deterministic |
| Tokenizer 复杂度膨胀 | 试图复刻真实 BPE | 复用 toyTokenizer，只扩参数和测试 |
| Hidden test 不公平 | 玩家觉得突然换规则 | visible 教规则，hidden 只换输入，不换合同 |
| Trace 过载 | 失败面板像日志墙 | 默认显示 firstBadNode，一层层展开 |
| Shape-only 假通过 | 等尺寸陷阱漏判 | 每个数值关卡必须有 allclose / cell trace |
| Guided Repair 回归 | 现有关卡无法玩 | 每次 milestone 跑手动 regression |

---

## 11. 建议排期

### 10-14 天版本节奏

```text
Day 1:
Milestone 0 + Graph types scaffold

Day 2-3:
Milestone 1 Runtime Skeleton

Day 4-5:
Milestone 2 Graph Workbench UI Shell

Day 6-7:
Milestone 3 Tokenizer Machine

Day 8-9:
Milestone 4 MatMul Gate

Day 10-12:
Milestone 5 Transpose Trap

Day 13:
Milestone 6 Broadcast Mini，若进度不足则降为 P1 文档/fixture

Day 14:
Milestone 7 QA / polish / docs
```

### 若只能做 1 周版本

```text
必须完成：
1. Runtime Skeleton
2. Tokenizer Machine
3. MatMul Gate
4. Trace / TestReport / Hidden Tests

延期：
1. Transpose Trap 完整 UI
2. Broadcast Mini
3. CellTrace 细节可视化
```

---

## 12. 最终 Done Definition

mvp0.0.5 只有同时满足以下条件才算完成：

```text
[ ] Graph Challenge 是真实 runtime 判定，不是 correctTagIds 判定
[ ] 玩家可以通过修改 graph/params 改变测试结果
[ ] Visible Tests 失败能定位 firstBadNode
[ ] Hidden Tests 输入和 visible 不相同
[ ] 至少一个 hidden test 抓住 shape pass / numeric fail
[ ] Trace Timeline 能复现失败路径
[ ] Inspector 显示 expected / received / sample values
[ ] Tokenizer、MatMul、Transpose 三个 slice 都可演示
[ ] Guided Repair 模式仍可用
[ ] typecheck/build/test 通过
[ ] README 或 docs 记录运行、验证、已知限制
```

---

## 13. 执行顺序总表

```text
1. 建立 gameplayGraph/types.ts
2. 建立 ModuleRegistry
3. 建立 TinyTensor + reference helpers
4. 建立 graph validation / execution / trace / assertions
5. 建立 test runner
6. 建立 GraphWorkbench UI shell
7. 接入 Tokenizer Machine
8. 接入 MatMul Gate
9. 接入 Transpose Trap
10. 补 Broadcast Mini 或记录为 P1
11. 接入 scoring / rank / debrief
12. 补 tests / build / docs
```

这个顺序必须保持。不要先做视觉大改，也不要先迁移现有 Guided Repair 关卡。

