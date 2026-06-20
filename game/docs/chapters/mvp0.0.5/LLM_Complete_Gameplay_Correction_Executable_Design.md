# LLM Complete 游戏性修正：可执行策划 / 设计文档

> 项目：`yx-labs/train-llm-from-scratch` / `game` 分支  
> 范围：基于当前已完成的 Chapter 0、Chapter 1，修正“游戏性弱、像填空/匹配题”的问题  
> 版本：v0.1 / 可进入开发排期版  
> 日期：2026-06-20  
> 目标交付：策划、前端、运行时/算法、QA 可直接拆任务执行

---

## 0. 结论摘要

当前 Chapter 0 / Chapter 1 的内容积累是有效的，但实际体验偏弱的根因是：

```text
玩家主要在把正确标签放到正确槽位，
而不是亲手构建一个能运行、会失败、可调试、可修好的系统。
```

本次修正不推翻已有内容，而是重新定位：

```text
现有 tag-slot / guided repair 系统
→ 降级为 Guided Repair Mode / 新手辅助层 / 知识检查层

新增 Graph Workbench / Graph Challenge Mode
→ 成为主玩法层
```

最终目标是让玩家形成一个明确体验：

```text
我搭了一个 tokenizer / matmul / attention 子系统；
它真的运行了；
它在 visible 或 hidden case 中失败了；
我通过 trace、shape、数值、cell provenance 找到原因；
我修复后再次运行，最终通过测试。
```

### 0.1 本次修正的第一优先级

不要继续扩写更多知识卡和关卡数量。第一优先级是做出一个小而真实的可执行闭环：

```text
GraphSpec
→ Module Registry
→ Tiny Runtime
→ Test Runner
→ Trace Report
→ Player Fix
→ Hidden Tests
```

第一版只需要三个 Vertical Slice：

```text
Slice A：Chapter 1 Tokenizer Machine
Slice B：0-2 MatMul Gate Graph
Slice C：0-3 QK Transpose / 0-4 Mask Add Debug
```

---

## 1. 修正目标

### 1.1 产品目标

把游戏从“互动教程”推进到“工程构建解谜”。玩家的核心动词从：

```text
读知识卡
拖标签
填槽
看文案反馈
下一关
```

改为：

```text
搭模块
连端口
调参数
运行测试
查看 trace
定位错误
修复系统
通过隐藏测试
优化评分
```

### 1.2 设计目标

| 目标 | 说明 | 验收方式 |
|---|---|---|
| 真实可运行 | 玩家提交的是 graph / 参数 / 策略，不是答案文本 | Runtime 能执行玩家图 |
| 可验证 | 每关至少有 shape、dtype、数值或行为测试 | Test Report 显示 public/hidden/reference |
| 可调试 | 失败能定位到节点、端口、shape、sample value 或 cell | Trace Timeline 可复现失败 |
| 可泛化 | 隐藏测试会改变输入文本、shape、尺寸或轴语义 | 不能靠记固定答案通关 |
| 可成长 | 完成的小模块可成为后续高级模块 | 通关后解锁模块或子图 |

### 1.3 不做什么

本阶段不优先做：

```text
- 更复杂的 3D 场景
- 更多剧情包装
- 大量新知识卡
- 完整 PyTorch / autograd
- 真实大规模 BPE
- 完整训练平台
- 排行榜 / 多人 / 商业化系统
```

本阶段要先证明：

```text
搭图 → 运行 → 失败 → trace → 修复 → hidden test
```

这个闭环是否好玩。

---

## 2. 新核心玩法循环

### 2.1 主循环

```text
观察目标 / 读输入合同
↓
拖入模块 / 连接端口 / 设置参数
↓
运行 Visible Tests
↓
失败：查看 Trace Timeline + Inspector
↓
定位错误节点 / 错误 cell / 错误 axis
↓
修复连接或参数
↓
再次运行
↓
通过 Visible Tests
↓
运行 Hidden Mutation Tests
↓
通过后获得评分与模块解锁
```

### 2.2 每次失败必须有信息量

失败不是“回答错误”，而是工程故障。失败报告必须至少包含：

```text
Error Type
First Bad Node
Expected Contract
Received Contract
Trace Step
Small Sample
Possible Cause
Suggested Probe
```

示例：

```text
Error Type: Shape Mismatch
Node: MatMulGate_02
Expected: hidden[B,T,C] @ weight[C,O]
Received: hidden[B,T,C] @ stored_weight[O,C]
First bad step: 4
Suggested probe: inspect weight orientation
```

### 2.3 Hidden Tests 的角色

Hidden Tests 不再只是“最后的通过动画”，而是玩法压力源。它们需要改变：

```text
- 输入文本
- B/T/C/O/H/D 尺寸
- 轴语义
- 权重存储方向
- OOV 字符
- padding 长度
- mask 方向
- 特殊 token 位置
```

Hidden Tests 的目的不是刁难，而是防止玩家背固定答案。

---

## 3. 游戏模式重构

### 3.1 Guided Repair Mode

保留当前 tag-slot 玩法，但改变定位。

用途：

```text
- 新手引导
- 概念复习
- 第一次见到某个模块时的安全练习
- 失败后的可选辅助修复
- 课堂演示模式
```

特点：

```text
- 可以有正确标签和正确槽位
- 可以给较强提示
- 不参与最高评分
- 不作为主线长期玩法
```

命名建议：

```text
Concept Bootcamp
Guided Repair
Training Wheels
Reference Lab
```

### 3.2 Graph Challenge Mode

成为主玩法模式。

界面结构：

```text
左侧：Module Palette / 模块库
中间：Graph Canvas / 模型搭建板
右侧：Inspector / 参数、shape、values、tests、code
底部：Trace Timeline / Run、Step、失败定位、测试报告
```

核心能力：

```text
- 拖模块
- 连端口
- 参数面板
- shape 实时推导
- dtype 检查
- Run Visible
- Run Hidden
- Trace 单步回放
- Reference Diff
```

### 3.3 Scoring / Ranking

建议第一版用轻量评分，不做复杂经济系统。

```text
Rank C：visible tests passed
Rank B：hidden tests passed
Rank A：hidden tests passed + failed runs ≤ 2
Rank S：no hint + first hidden pass + no extra modules
```

评分公式示例：

```text
base = correctness * 100
score = base
      - failedRuns * 5
      - hintsUsed * 10
      - probesUsed * 2
      - extraModules * 3
      - referenceOverlayUsed * 8
```

---

## 4. 技术架构修正

### 4.1 新增目录建议

在现有 `game/app/src` 下新增：

```text
src/gameplayGraph/
  types.ts
  moduleRegistry.ts
  levelRegistry.ts

  runtime/
    tinyTensor.ts
    graphExecutor.ts
    shapeInference.ts
    testRunner.ts
    trace.ts
    reference.ts

  levels/
    ch0ShapeOnboarding.ts
    ch0MatMulGraph.ts
    ch0TransposeGraph.ts
    ch0BroadcastGraph.ts
    ch1TokenizerGraph.ts

  ui/
    GraphWorkbench.tsx
    ModulePalette.tsx
    GraphCanvasBridge.tsx
    InspectorPanel.tsx
    TestReportPanel.tsx
    TraceTimeline.tsx
    ParameterEditor.tsx
```

### 4.2 核心数据结构

```ts
type DType =
  | "raw_text"
  | "string_piece"
  | "token_piece"
  | "int"
  | "float32"
  | "bool"
  | "mask";

type AxisName =
  | "B"   // batch
  | "T"   // token position
  | "C"   // channel / feature
  | "O"   // output channel
  | "H"   // attention head
  | "D"   // head dim
  | "Tq"  // query token axis
  | "Tk"  // key token axis
  | "V";  // vocab

type TensorShape = {
  dims: number[];
  axes: AxisName[];
  dtype: DType;
};

type PortRef = {
  nodeId: string;
  portId: string;
};

type PortDef = {
  id: string;
  label: string;
  direction: "in" | "out";
  accepts?: DType[];
  emits?: DType;
  shape?: TensorShape;
  required?: boolean;
};

type GraphNode = {
  id: string;
  moduleId: string;
  params: Record<string, unknown>;
  position: { x: number; y: number };
  locked?: boolean;
};

type GraphEdge = {
  id: string;
  from: PortRef;
  to: PortRef;
};

type GraphSpec = {
  levelId: string;
  version: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  outputNodes: string[];
};
```

### 4.3 Runtime Value

```ts
type RuntimeValue = {
  dtype: DType;
  shape?: TensorShape;
  data?: number[] | string[] | string | boolean[];
  meta?: Record<string, unknown>;
};
```

### 4.4 Trace Frame

```ts
type TraceFrame = {
  step: number;
  nodeId: string;
  moduleId: string;
  inputShapes: Record<string, TensorShape>;
  outputShapes: Record<string, TensorShape>;
  samples?: Record<string, unknown>;
  error?: RuntimeError;
};

type RuntimeError = {
  type:
    | "dtype_mismatch"
    | "shape_mismatch"
    | "axis_semantic_error"
    | "numeric_mismatch"
    | "budget_exceeded"
    | "oov_unresolved"
    | "mask_error"
    | "nan_inf";
  message: string;
  nodeId: string;
  portId?: string;
  expected?: unknown;
  received?: unknown;
  suggestedProbe?: string;
};
```

### 4.5 Test Case

```ts
type TestVisibility = "visible" | "hidden" | "reference";

type TestCase = {
  id: string;
  title: string;
  visibility: TestVisibility;
  inputSeed: string;
  inputs: Record<string, RuntimeValue>;
  assertions: TestAssertion[];
};

type TestAssertion =
  | { type: "dtype"; nodeId: string; expected: DType }
  | { type: "shape"; nodeId: string; expectedAxes: AxisName[]; expectedDims?: number[] }
  | { type: "axis_semantics"; nodeId: string; expectedAxes: AxisName[] }
  | { type: "allclose"; nodeId: string; referenceNodeId: string; atol: number }
  | { type: "token_budget"; nodeId: string; maxT: number }
  | { type: "mask_pad"; idsNodeId: string; maskNodeId: string; padId: number }
  | { type: "future_attention_zero"; nodeId: string; threshold: number }
  | { type: "row_sum"; nodeId: string; dim: AxisName; target: number; atol: number };

type TestResult = {
  id: string;
  visibility: TestVisibility;
  status: "pass" | "fail" | "blocked";
  message: string;
  firstBadNodeId?: string;
  traceFrame?: number;
  assertion?: TestAssertion;
};
```

### 4.6 Runtime 执行流程

```text
loadLevelSpec
→ buildPlayerGraph
→ validateGraphStructure
→ inferPortTypes
→ inferShapes
→ executeForwardSmallTensor
→ captureTrace
→ runVisibleAssertions
→ runHiddenAssertions
→ generateTestReport
→ updateScoreAndUnlocks
```

MVP 不需要真实大模型运行，只需要小尺寸 deterministic arrays。

---

## 5. Module Registry 第一批模块

### 5.1 Tokenizer / Text → Token 模块

| 模块 | 输入 | 输出 | 参数 | MVP |
|---|---|---|---|---|
| TextInput | none | raw_text | text | P0 |
| TextInspector | raw_text | character_stream | show invisibles | P0 |
| TokenizerSocket | raw_text | token_pieces | policy | P0 |
| BoundarySplit | raw_text | pieces | char/word/subword | P0 |
| MergeForge | pieces | pieces | mergeTable, applyMerges | P1 |
| VocabLookup | token_pieces | ids[T] | vocab | P0 |
| FallbackSplitter | OOV pieces | token_pieces | char/unk/none | P0 |
| SpecialTokenInjector | ids[T] | ids[T+2] | bos/eos | P0 |
| PaddingBuilder | ids list | token_ids[B,T] | padSide, padId | P0 |
| AttentionMaskBuilder | token_ids[B,T] | mask[B,T] | pad-aware/all-ones | P0 |
| EmbeddingReadyProbe | token_ids[B,T] | contract pass/fail | expected dtype int | P0 |

### 5.2 Tensor / Attention 基础模块

| 模块 | 输入 | 输出 | 参数 | MVP |
|---|---|---|---|---|
| InputTensor | none | tensor | shape, seed | P0 |
| WeightPlate | none | weight tensor | storage orientation | P0 |
| TransposeSwitch | tensor | tensor | swap axes | P0 |
| MatMulGate | A, B | out | batched / plain | P0 |
| OutputContractGate | tensor | tensor | expected axes | P0 |
| BroadcastRail | small tensor + target | broadcast plan | axis alignment | P1 |
| AddGate | A, B | out | plan | P1 |
| CausalMask | T | mask[T,T] | direction | P1 |
| MaskApply | scores, mask | masked_scores | fill value | P1 |
| Softmax | scores | weights | dim | P1 |
| ReferenceChecker | output, reference | report | tolerance | P0 |

---

## 6. UI / UX 修正

### 6.1 主界面布局

```text
┌─────────────────────────────────────────────────────────────┐
│ Top Bar: Chapter / Level / Mode / Run Visible / Run Hidden   │
├──────────────┬───────────────────────────────┬──────────────┤
│ Module       │ Graph Canvas                   │ Inspector    │
│ Palette      │ nodes + ports + edges          │ Summary      │
│              │ shape labels                   │ Shape        │
│              │ error highlights               │ Values       │
│              │ trace state                    │ Tests        │
│              │                               │ Code         │
├──────────────┴───────────────────────────────┴──────────────┤
│ Trace Timeline / Test Report / Failure Explanation           │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Inspector Tab

| Tab | 内容 |
|---|---|
| Summary | 模块目的、输入输出合同、当前状态 |
| Shape | shape 推导、axis semantics、dtype |
| Values | 小尺寸 sample values、局部 cell、统计 |
| Tests | 当前节点相关的失败 / 通过测试 |
| Code | 等价伪代码或简化 PyTorch 风格代码 |

### 6.3 Trace Timeline

Trace 不应该只是说明书，而是破案工具。

每一步显示：

```text
Step 4 / MatMulGate_02
Input A: float32[B=2,T=4,C=3]
Input B: float32[O=5,C=3]
Expected B: float32[C=3,O=5]
Status: FAIL
Suggested Probe: Orientation Probe
```

### 6.4 错误 UI 层级

| 层级 | 表现 |
|---|---|
| 模块级 | 节点红框，显示错误类型 |
| 端口级 | 错误端口闪烁，显示 expected/received |
| 连线级 | 数据线变红或虚线，提示 dtype/shape 不匹配 |
| 张量级 | 具体 axis 或非法区域高亮 |
| Cell 级 | score/mask/attention cell 标红 |
| 时间级 | Trace 定位到第一次出错 step |

---

## 7. Chapter 0 修正设计

## 7.1 0-1 Shape Reader：缩短为入门引导

### 新定位

0-1 不再承载长时间玩法，只承担入门认知和 UI 教学。

目标时长：

```text
5–8 分钟
```

必须学会：

```text
- tensor 是规则排列的数值容器
- rank 是轴数量
- shape 是每个轴的长度
- [2,4,8] 只是数字，必须加上 B/T/C 语义
- Inspector 在哪里
- Run Tests 在哪里
- Trace 如何定位失败
```

### 保留内容

```text
Tensor Inspector
Shape 标签
B/T/C 轴解释
hidden[B,T,C] 基础认知
简单 visible test
```

### 删除 / 后移内容

```text
过多知识卡
长流程 repair stage
大量标签匹配
复杂 hidden gauntlet
```

### 通过条件

```text
玩家能正确标注 hidden[B,T,C]
玩家能打开 Inspector 查看 dtype/shape
玩家能运行一次测试并看懂通过/失败
```

---

## 7.2 0-2 MatMul Gate：第一个真正 Graph Challenge

### 关卡一句话

玩家从空白工作区组装一个真实可执行的 Linear Projection：

```text
hidden[B,T,C] @ weight[C,O] → projected[B,T,O]
```

### 可用模块

```text
HiddenTensorSource
WeightPlate
TransposeSwitch
MatMulGate
OutputContractGate
ReferenceChecker
```

### 玩家操作

```text
1. 拖入 HiddenTensorSource
2. 拖入 WeightPlate
3. 检查 weight storage orientation
4. 必要时插入 TransposeSwitch
5. 拖入 MatMulGate
6. 连接 hidden 和 compute weight
7. 连接 ProjectedOutput
8. 标注输出合同 [B,T,O]
9. 连接 ReferenceChecker
10. 运行 Visible Tests
11. 运行 Hidden Tests
```

### Visible Tests

```text
Case V1: 标准方向
hidden[B=2,T=4,C=3]
weight[C=3,O=5]
expected output[B=2,T=4,O=5]

Case V2: 权重存储方向反了
hidden[B=1,T=4,C=3]
stored_weight[O=5,C=3]
requires transpose
expected output[B=1,T=4,O=5]
```

### Hidden Tests

```text
Case H1: 防止最大维度猜测
hidden[B=4,T=3,C=2]
weight[C=2,O=16]
expected output[B=4,T=3,O=16]

Case H2: C/O 相等但方向错
hidden[B=2,T=3,C=4]
stored_weight[O=4,C=4]
wrong orientation can produce plausible shape
numeric reference must catch it

Case H3: 输出轴顺序错误
expected [B,T,O]
reject [B,O,T] or [T,B,O]
```

### 失败反馈模板

```text
MatMul failed: inner dimension mismatch.

Left input last axis: C = {leftC}
Right input first axis: {rightAxis} = {rightSize}

MatMul requires:
activation[..., C] @ weight[C, O]
```

```text
Reference mismatch.

Your output shape is correct,
but values differ from reference.

Likely causes:
- wrong weight orientation
- unnecessary transpose
- output axes reordered
```

### 0-2 验收标准

```text
[ ] 玩家图能被 Runtime 执行
[ ] shape inference 能推导 [B,T,O]
[ ] stored_weight[O,C] 必须经过 transpose 才能数值通过
[ ] hidden tests 改变 B/T/C/O 后仍能验证
[ ] Trace 能定位到 MatMulGate 或 TransposeSwitch
[ ] ReferenceChecker 显示 max_abs_error
```

---

## 7.3 0-3 Transpose Trap：数值 Trace 破案关

### 关卡一句话

玩家修复 Attention Score Board：

```text
Q[B,H,T,D] @ Kᵀ[B,H,D,T] → scores[B,H,T,T]
```

当前错误通常是：

```text
Q @ K
transpose Q
reverse all axes
swap B/H
Kᵀ @ Q
```

### 可用模块

```text
QTensor
KTensor
TransposeSwitch
AxisLock
BatchedMatMulGate
ScoreBoard
CellTrace
ReferenceChecker
```

### 玩家操作

```text
1. Inspect Q/K contracts
2. 将 K 接入 TransposeSwitch
3. 设置 swap axes = -2, -1
4. 锁定 B/H carry axes
5. 将 Q 接 MatMul 左端
6. 将 Kᵀ 接 MatMul 右端
7. 输出到 ScoreBoard
8. 点击 score cell 检查来源
9. 运行 visible / hidden numeric tests
```

### Cell Trace 核心表现

玩家点击：

```text
scores[b,h,i,j]
```

Inspector 显示：

```text
Expected:
scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:])

Actual:
{actual source expression}

Diagnosis:
{axis T/D swapped incorrectly or K not transposed}
```

### Hidden Tests

```text
Case H1: T != D
B=2,H=2,T=4,D=3
K 未转置时 shape mismatch

Case H2: T == D
B=1,H=3,T=4,D=4
错误 Q @ K 可能 shape 看似通过，但 numeric fail

Case H3: carry axes 被交换
B=3,H=2,T=5,D=4
要求 B/H 保留，只交换 T/D

Case H4: 操作数顺序反
Kᵀ @ Q 应判错
```

### 0-3 验收标准

```text
[ ] TransposeSwitch 支持 swap(-2,-1)
[ ] AxisLock 能保护 B/H
[ ] ScoreBoard 能显示 [Tq,Tk]
[ ] CellTrace 能回溯 Q/K 来源
[ ] T == D 时 shape pass / numeric fail 能被抓住
[ ] Hidden tests 随机 B/H/T/D 能通过
```

---

## 7.4 0-4 Broadcast Add：规则推理 + 反例测试

### 关卡一句话

玩家修复 Broadcast Add Station，让小张量按正确语义轴广播到大张量。

核心场景：

```text
projected[B,T,O] + bias[O] → projected[B,T,O]
tok_emb[B,T,C] + pos_emb[T,C] → hidden[B,T,C]
scores[B,H,T,T] + mask[1,1,T,T] → masked_scores[B,H,T,T]
residual_a[B,T,C] + residual_b[B,T,C] → hidden[B,T,C]
```

### MVP 阶段

```text
0-4A Add Cell
0-4B Same-Shape Add
0-4C Broadcast Rule Lab
0-4D Bias Add
0-4E Position Add
0-4X Simplified Broadcast Gauntlet
```

0-4F Mask Add 强推荐；工期紧时可作为 0-4.5 或 Attention 前置关。

### 可用模块

```text
InputTensorA
InputTensorB
BroadcastRail
AxisAlignmentRuler
GhostExpansionPreview
AddGate
CellTrace
ReferenceChecker
SemanticWarningLens
```

### 玩家操作示例：Bias Add

```text
1. Inspect projected[B,T,O]
2. Inspect bias[O]
3. 将 bias 对齐到 BroadcastRail 的 O 槽
4. 预览 ghost expansion：bias 沿 B/T 复用
5. 运行 cell trace：out[b,t,o] = projected[b,t,o] + bias[o]
6. 运行 ReferenceChecker
```

### Hidden Tests

```text
Case H1: [B,T,C] + [T]
当 T != C，应 shape/semantic fail

Case H2: [B,T,C] + [T] 且 T == C
shape 可能过，但 semantic warning + numeric reference 捕获

Case H3: mask Tq/Tk 反转
shape 正确，但 blocked cells pattern 错误

Case H4: 真实 repeat 而非 logical broadcast
数值通过，但 efficiency 降级
```

### 0-4 验收标准

```text
[ ] BroadcastRail 能显式显示 axis alignment
[ ] GhostExpansion 能展示逻辑扩展而非真实复制
[ ] CellTrace 能显示 out index 对应的小张量 index
[ ] 语义错误不能只靠 shape 通过
[ ] mask orientation 错误可被 hidden test 抓住
```

---

## 8. Chapter 1 修正设计：Tokenizer Machine

Chapter 1 不再做大量标签槽位，而是做一台可运行的 Tokenizer Machine。

### 8.1 目标管线

```text
Raw Text
→ Tokenizer Policy
→ Token Pieces
→ Merge / Fallback
→ Vocab Lookup
→ Token IDs
→ Special Tokens
→ Padding / Truncation
→ Token Buffer[int[B,T]]
→ Attention Mask[int/bool[B,T]]
```

### 8.2 玩家可调参数

```ts
type TokenizerParams = {
  policy: "char" | "word" | "subword";
  applyMerges: boolean;
  fallback: "none" | "char" | "unk";
  preservePunctuation: boolean;
  addBos: boolean;
  addEos: boolean;
  maxLength: number;
  padToLength: number;
  padSide: "left" | "right";
  maskPolicy: "pad-aware" | "all-ones";
};
```

### 8.3 1-1 Text Type Gate

#### 核心问题

```text
为什么 raw text 不能直接接 Embedding Lookup？
```

#### 玩家目标

把错误管线：

```text
Raw Text → Embedding Lookup
```

修成：

```text
Raw Text → Tokenizer → Vocab Lookup → token_ids[T] → EmbeddingReadyProbe
```

#### Visible Test

```text
Input: "we train llm"
Expected output dtype: int
Expected shape: [T]
EmbeddingReady: true
```

#### 失败反馈

```text
Connection failed:
Embedding Lookup expects integer token IDs.
Raw text is a string object.
Insert Tokenizer and Vocab Lookup first.
```

### 8.4 1-2 Split / Merge Budget

#### 核心问题

```text
不同切分策略如何影响 T、context length 和成本？
```

#### 玩家可选策略

```text
policy = char / word / subword
applyMerges = true / false
maxLength = 8
```

#### 设计重点

不要有唯一“正确标签”。让策略产生不同后果：

```text
char split：OOV 稳定，但 token 数多，容易超预算
word split：token 数少，但 OOV 风险大
subword + merge：成本与覆盖之间更平衡
```

#### Visible Text Set

```text
we train llm.
tokenizers are useful!
shape tells truth
```

#### Hidden Text Set

```text
unknown glyph ?
a loooooong tokenized sequence
中文 mixed text!
code_snake_case + punctuation
```

#### 测试

```text
[ ] pieces 非空
[ ] token count <= maxLength 或触发合法 truncation
[ ] punctuation policy 正确
[ ] merge 后 tokens 与 reference 一致
[ ] determinism：同输入多次得到同结果
```

### 8.5 1-3 Vocab / OOV / Fallback

#### 核心问题

```text
token piece 如何稳定变成 token ID？未知 piece 怎么办？
```

#### 玩家可调

```text
fallback = none / char / unk
preservePunctuation = true / false
unkId = 3
```

#### Visible Test

```text
Input: "we train llm!"
Expected:
- known pieces light vocab rows
- punctuation preserved if policy requires
- ids stable across runs
```

#### Hidden Test

```text
Input: "unseen glyph ?"
Expected:
- no unresolved OOV
- fallback path produces valid IDs
- same piece maps to same ID
```

#### 失败反馈

```text
OOV unresolved:
This token piece has no vocab row.
Use Fallback Splitter or <unk> policy.
```

```text
Determinism failed:
Same input produced different token IDs across runs.
Tokenizer must be stable.
```

### 8.6 1-4 Special Tokens / Padding / Mask

#### 核心问题

```text
如何把不同长度的文本 batch 变成矩形 token_ids[B,T]？
```

#### 玩家可调

```text
addBos = true / false
addEos = true / false
padSide = left / right
padId = 0
maxLength = 8
maskPolicy = pad-aware / all-ones
```

#### 输出合同

```text
token_ids[B,T]
attention_mask[B,T]
tokens[B,T]  // debug display only
```

#### Visible Test

```text
Input batch:
B0: "we train llm"
B1: "shape"

Expected:
B dimension = 2
T dimension = padToLength
PAD positions use padId
attention_mask content = 1
attention_mask pad = 0
```

#### Hidden Tests

```text
Case H1: one long text exceeds maxLength
Case H2: one short text requires padding
Case H3: left padding selected but target policy requires right padding
Case H4: mask all-ones incorrectly marks PAD as valid
Case H5: EOS lost during truncation
```

#### 失败反馈

```text
Batch build failed:
Sequences have different lengths.
Use Padding Builder to create rectangular int[B,T].
```

```text
Attention mask mismatch:
PAD positions should be masked out.
Check pad_id and mask generation rule.
```

### 8.7 1-X Tokenizer Gauntlet

综合考核，不弹长知识卡，只给 Debug Checklist。

#### Gauntlet Inputs

```text
"we train llm!"
"unknown glyph ?"
"shape, token buffer"
"emoji 😀 test"
"中文 mixed text"
"a very very long sequence that exceeds budget"
```

#### 通过条件

```text
[ ] no unresolved OOV
[ ] token_ids dtype == int
[ ] token_ids shape == [B,T]
[ ] attention_mask shape == [B,T]
[ ] pad positions mask == 0
[ ] content positions mask == 1
[ ] T <= maxLength
[ ] deterministic ids
[ ] hidden text set pass
```

---

## 9. 关卡配置示例

### 9.1 MatMul Graph LevelSpec

```ts
export const ch0_2_matmul_graph: LevelSpec = {
  id: "ch0_2_matmul_graph",
  title: "0-2 MatMul Gate Graph",
  mode: "graph_challenge",
  goal: "Build hidden[B,T,C] @ weight[C,O] -> projected[B,T,O]",

  modulePalette: [
    "HiddenTensorSource",
    "WeightPlate",
    "TransposeSwitch",
    "MatMulGate",
    "OutputContractGate",
    "ReferenceChecker"
  ],

  fixedNodes: [],

  constraints: {
    maxNodes: 8,
    maxEdges: 8,
    forbiddenModules: ["AddGate", "Softmax"]
  },

  visibleTests: [
    {
      id: "standard_CO",
      inputs: {
        hidden: { shape: [2, 4, 3], axes: ["B", "T", "C"] },
        weight: { shape: [3, 5], axes: ["C", "O"], orientation: "C,O" }
      },
      assertions: [
        { type: "shape", nodeId: "projected", expectedAxes: ["B", "T", "O"], expectedDims: [2, 4, 5] },
        { type: "allclose", nodeId: "projected", referenceNodeId: "reference", atol: 1e-5 }
      ]
    }
  ],

  hiddenTests: [
    {
      id: "stored_OC_requires_transpose",
      inputs: {
        hidden: { shape: [1, 8, 4], axes: ["B", "T", "C"] },
        weight: { shape: [12, 4], axes: ["O", "C"], orientation: "O,C" }
      },
      assertions: [
        { type: "shape", nodeId: "projected", expectedAxes: ["B", "T", "O"], expectedDims: [1, 8, 12] },
        { type: "allclose", nodeId: "projected", referenceNodeId: "reference", atol: 1e-5 }
      ]
    }
  ]
};
```

### 9.2 Tokenizer Machine LevelSpec

```ts
export const ch1_tokenizer_machine: LevelSpec = {
  id: "ch1_tokenizer_machine",
  title: "Chapter 1 — Tokenizer Machine",
  mode: "graph_challenge",
  goal: "Convert raw text batch into token_ids[B,T] and attention_mask[B,T]",

  initialParams: {
    policy: "subword",
    applyMerges: true,
    fallback: "unk",
    addBos: true,
    addEos: true,
    maxLength: 8,
    padToLength: 8,
    padSide: "right",
    maskPolicy: "pad-aware"
  },

  visibleTests: [
    {
      id: "basic_batch",
      inputs: {
        texts: ["we train llm", "shape"]
      },
      assertions: [
        { type: "dtype", nodeId: "token_ids", expected: "int" },
        { type: "shape", nodeId: "token_ids", expectedAxes: ["B", "T"], expectedDims: [2, 8] },
        { type: "mask_pad", idsNodeId: "token_ids", maskNodeId: "attention_mask", padId: 0 }
      ]
    }
  ],

  hiddenTests: [
    {
      id: "unknown_and_budget",
      inputs: {
        texts: ["unknown glyph ?", "a very very long sequence"]
      },
      assertions: [
        { type: "token_budget", nodeId: "token_ids", maxT: 8 },
        { type: "mask_pad", idsNodeId: "token_ids", maskNodeId: "attention_mask", padId: 0 }
      ]
    }
  ]
};
```

---

## 10. 开发计划

### 10.1 Sprint 0：冻结范围与接入点

周期：1–2 天

任务：

```text
[ ] 标记现有 tag-slot 系统为 Guided Repair Mode
[ ] 确认当前 App 状态管理入口
[ ] 确认 Pixi canvas 与 React panels 的边界
[ ] 新增 gameplayGraph 目录
[ ] 确定 LevelSpec / GraphSpec 最小 schema
[ ] 建立 P0 / P1 / P2 backlog
```

交付：

```text
- gameplayGraph/types.ts
- levelRegistry.ts 空实现
- 一页架构说明
```

### 10.2 Sprint 1：Graph Runtime Skeleton

周期：3–5 天

任务：

```text
[ ] ModuleRegistry
[ ] GraphSpec validate
[ ] shapeInference
[ ] graphExecutor
[ ] testRunner
[ ] trace capture
[ ] TestReportPanel 初版
```

验收：

```text
- 能执行一个固定 graph
- 能返回 pass/fail
- 能返回 firstBadNode
- 能在 UI 中显示 trace step
```

### 10.3 Sprint 2：Tokenizer Machine Vertical Slice

周期：3–5 天

任务：

```text
[ ] TextInput / Tokenizer / VocabLookup / PaddingBuilder / MaskBuilder
[ ] Tokenizer parameter panel
[ ] Visible + hidden text set
[ ] OOV / budget / padding failure messages
[ ] Token Buffer 网格显示
```

验收：

```text
- raw text 不能直接通过 EmbeddingReadyProbe
- policy 改变会真实改变 token pieces / ids / mask
- hidden text 能破坏简单 hardcode 策略
- attention_mask 与 PAD 一致
```

### 10.4 Sprint 3：0-2 MatMul Graph

周期：4–6 天

任务：

```text
[ ] TinyTensor matmul
[ ] Weight orientation
[ ] TransposeSwitch 2D
[ ] OutputContractGate
[ ] ReferenceChecker
[ ] MatMul hidden tests
[ ] Orientation Probe / trace
```

验收：

```text
- [C,O] 直接通过
- [O,C] 必须 transpose
- shape 正确但数值错误会 fail
- hidden case 改 B/T/C/O 后仍可验证
```

### 10.5 Sprint 4：0-3 Transpose + 0-4 Broadcast/Mini Mask

周期：4–6 天

任务：

```text
[ ] TransposeSwitch 支持 rank 4 / -2,-1
[ ] AxisLock B/H
[ ] QK reference
[ ] ScoreBoard cell trace
[ ] BroadcastRail 初版
[ ] Bias/Position Add reference
[ ] Mask orientation hidden test，可选
```

验收：

```text
- Q @ Kᵀ 通过
- Q @ K 在 T==D 时被 numeric fail 抓住
- Broadcast bias/position 通过
- semantic warning 能提示 shape pass 但语义危险
```

### 10.6 Sprint 5：Playtest / QA / Polish

周期：3 天

任务：

```text
[ ] 10 人内测
[ ] 记录 failed runs、hint usage、time to pass
[ ] 修改错误文案
[ ] 减少无效知识卡打断
[ ] 加入 Rank C/B/A/S
[ ] 输出 playtest report
```

验收：

```text
- 至少 70% 玩家能在不看答案的情况下通过 Tokenizer Slice
- 至少 60% 玩家能解释一次 MatMul 失败原因
- 玩家平均会主动 Run Tests ≥ 3 次
- 玩家能说出“shape 通过不代表数值正确”
```

---

## 11. Backlog 拆分

### 11.1 P0 / 必须完成

| 类别 | 任务 |
|---|---|
| Runtime | GraphSpec、ModuleRegistry、ShapeInference、Executor、TestRunner、Trace |
| UI | ModulePalette、GraphCanvasBridge、Inspector、TestReport、TraceTimeline |
| Content | Tokenizer Slice、0-2 MatMul Graph、0-3 QK Transpose basic |
| QA | visible/hidden tests、failure messages、playtest checklist |

### 11.2 P1 / 强推荐

| 类别 | 任务 |
|---|---|
| Runtime | BroadcastRail、Softmax row sum、MaskApply |
| UI | CellTrace 可视化、ReferenceDiff、Rank report |
| Content | 0-4 Broadcast Add、1-X Tokenizer Gauntlet |
| UX | reference overlay 作为扣分辅助 |

### 11.3 P2 / 后续

| 类别 | 任务 |
|---|---|
| Runtime | autograd subset、optimizer step、training loop |
| UI | 3D expanded tensor view、attention heatmap stack |
| Content | Embedding、Full Attention、Transformer Block |
| Systems | Sandbox、Challenge Lab、Report export |

---

## 12. QA / 验收标准

### 12.1 全局 Done Definition

一个关卡只有同时满足以下条件，才算完成：

```text
[ ] 玩家提交的是 graph / params / strategy，不是答案标签
[ ] Runtime 能执行玩家提交
[ ] Visible Tests 通过后才开放 Hidden Tests
[ ] Hidden Tests 至少改变一个输入条件
[ ] 失败会定位 firstBadNode
[ ] Trace 能复现失败路径
[ ] Inspector 能显示 expected / received
[ ] 玩家可以通过修改图或参数修复问题
[ ] 通过后有 Debrief 和模块解锁
```

### 12.2 禁止通过方式

```text
[ ] 仅匹配 correctTagIds 就通关
[ ] 仅检查 slot assignment 就通关
[ ] 只看 shape，不做数值或行为测试
[ ] Hidden Tests 与 visible tests 使用完全相同输入
[ ] 错误只给“答案错误”而不定位原因
[ ] Hint 直接给完整解法且不扣分
```

### 12.3 体验验收

```text
[ ] 玩家会主动多次 Run Tests
[ ] 玩家会点击 Trace / Inspector 找原因
[ ] 玩家失败后知道下一步该检查哪里
[ ] 玩家能解释至少一个错误：dtype、shape、axis、numeric、mask、budget
[ ] 关卡完成后能说出一个可迁移的工程直觉
```

---

## 13. 风险与规避

| 风险 | 表现 | 规避 |
|---|---|---|
| 图编辑器范围爆炸 | 自由连线实现慢 | 第一版用半固定 graph + 参数编辑 |
| 又退回选择题 | 仍然靠 correctTagIds | 每关必须执行 Runtime 和 hidden mutation |
| 数学门槛过高 | 玩家被 shape 吓退 | 0-1 短引导 + Inspector 分层 |
| Trace 信息过载 | 玩家不知道看哪里 | 只高亮 first bad node，逐层展开 |
| Hidden Test 不公平 | 玩家觉得被偷袭 | visible 教规则，hidden 只换 case，不换规则 |
| 美术掩盖逻辑 | 科幻 UI 好看但不清楚 | shape、dtype、axis 始终作为第一视觉语言 |
| 开发周期过长 | 同时做太多模块 | 先做 Tokenizer + MatMul 两个 slice |

---

## 14. 版本成功标准

### 14.1 v0.2 修正版成功标准

```text
- 当前 Chapter0/1 不再只是标签填槽
- 至少 2 个关卡使用 GraphSpec 执行
- 至少 1 个 Tokenizer 关卡有真实策略 trade-off
- 至少 1 个 MatMul 关卡有数值 reference
- 至少 1 个 hidden test 能抓住 shape pass / numeric fail
- Trace Timeline 可定位 first bad node
- 玩家能通过修图而不是换标签来通关
```

### 14.2 可对外展示标准

```text
- 录屏中能看到玩家拖模块、连线、运行、失败、查看 trace、修复、通过
- 失败信息有工程可信度
- 关卡目标能用一句真实公式解释
- 通过后能连接到真实 LLM pipeline
```

---

## 15. 最小实现路线图

最小可玩闭环建议按这个顺序做：

```text
Week 1:
GraphSpec + Runtime Skeleton + TestRunner
Tokenizer fixed graph + parameter panel

Week 2:
MatMul Graph Challenge
Trace / ReferenceChecker / Hidden Tests

Week 3:
Transpose Trap + Broadcast Add mini
Scoring + Playtest + UI polish
```

如果只能做一个版本样片，优先顺序是：

```text
1. Tokenizer Machine
2. MatMul Gate Graph
3. QK Transpose Trap
4. Broadcast Add / Mask Add
```

理由：

```text
Tokenizer Machine 最容易让非技术玩家感到“我在调一台机器”；
MatMul Gate 最能证明核心图执行器成立；
Transpose Trap 最能体现 trace 破案；
Broadcast / Mask Add 最能连接后续 Attention。
```

---

## 16. 关键文案规范

### 16.1 错误文案格式

```text
[错误类型]
一句话说明发生了什么。

Expected:
{expected contract}

Received:
{received contract}

Why it matters:
{工程原因}

Try:
{推荐检查工具，不直接给最终答案}
```

### 16.2 Debrief 格式

```text
Stage Complete: {stage name}

你刚刚修复了：
{真实工程问题}

你现在应该知道：
{可迁移直觉}

下一个模块会用到它：
{后续连接}
```

### 16.3 Hint 分层

```text
Hint 1：指出检查区域
Hint 2：指出合同关系
Hint 3：给出局部修复方向
Reference Overlay：直接显示参考路径，扣分
```

---

## 17. 附录：错误信息库初版

### 17.1 raw text 直接进入 embedding

```text
Connection failed:
Embedding Lookup expects integer token IDs.
Raw text is a string object.
Insert Tokenizer and Vocab Lookup first.
```

### 17.2 OOV 未处理

```text
OOV unresolved:
This token piece has no vocab row.
Use Fallback Splitter or <unk> policy.
```

### 17.3 Token Budget 超限

```text
Token budget exceeded:
Current T is larger than max_T.
Apply merge rules, truncation, or reject the input.
```

### 17.4 Padding / Mask 错误

```text
Attention mask mismatch:
PAD positions should be masked out.
Check pad_id and mask generation rule.
```

### 17.5 MatMul 内维度错误

```text
MatMul failed: inner dimension mismatch.

Left input last axis: C = {leftC}
Right input first axis: {rightAxis} = {rightSize}

MatMul requires activation[..., C] @ weight[C, O].
```

### 17.6 Shape 通过但数值失败

```text
Shape passed, numeric reference failed.

The output shape is plausible,
but values differ from the reference implementation.

Use Trace to inspect orientation, axis order, and source cells.
```

### 17.7 Transpose Trap

```text
FAIL: Shape passed, but numeric reference failed.

This is the Transpose Trap.
Because T == D, the wrong operation produced the same visible shape.

But scores[i,j] should equal dot(Q[i,:], K[j,:]).
Run Cell Trace to inspect one failed cell.
```

### 17.8 Broadcast 语义错误

```text
Broadcast semantic mismatch.

The shapes can be aligned,
but the small tensor is attached to the wrong semantic axis.

Use Axis Alignment Ruler and Cell Trace.
```

### 17.9 Causal Mask 方向错误

```text
Mask orientation failed hidden test.

Shape was correct,
but query/key axes were swapped.
Blocked cells do not match causal pattern.
```

---

## 18. 交付清单

### 18.1 策划交付

```text
[ ] LevelSpec for Tokenizer Machine
[ ] LevelSpec for MatMul Graph
[ ] LevelSpec for Transpose Trap
[ ] Visible / Hidden Test 表
[ ] Failure Message 表
[ ] Concept Card 精简版
[ ] Debrief 文案
[ ] Scoring 规则
```

### 18.2 前端交付

```text
[ ] GraphWorkbench 页面
[ ] ModulePalette
[ ] ParameterEditor
[ ] InspectorPanel
[ ] TestReportPanel
[ ] TraceTimeline
[ ] TokenBufferGrid
[ ] ScoreBoard / CellTrace 初版
```

### 18.3 Runtime 交付

```text
[ ] TinyTensor 数据结构
[ ] tokenizer runtime
[ ] matmul reference
[ ] transpose reference
[ ] broadcast add reference
[ ] graph executor
[ ] shape inference
[ ] test runner
[ ] trace capture
```

### 18.4 QA 交付

```text
[ ] 每关 visible case
[ ] 每关 hidden mutation case
[ ] shape pass / numeric fail case
[ ] OOV / budget / padding edge cases
[ ] T == D trap case
[ ] B/H carry axis trap case
[ ] mask orientation case
```

---

## 19. 最终判断

本次修正的核心不是“把教程包装得更像游戏”，而是让项目真正进入：

```text
可运行
可失败
可调试
可修复
可泛化
```

当前 Chapter 0 / Chapter 1 的知识设计可以继续保留，但必须把主玩法从：

```text
找正确标签
```

升级为：

```text
构建一个能通过工程测试的 tiny LLM 子系统
```

只要第一版能让玩家在 Tokenizer、MatMul、Transpose 这三个 slice 中完成真实 Build-Test-Debug，后续 Embedding、Attention、Transformer Block、Training Loop 都会自然生长出来。
