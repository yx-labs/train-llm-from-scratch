# MVP0.1 Component System Spec

## 组件系统完整规格定义

## 0. 版本目标

MVP0.1 的核心变化是：**玩家不再只是使用系统给定节点完成图任务，而是从最基础的计算单元开始，亲手搭建、测试、认证、封装并复用自己的组件。**

上一版 Graph Challenge 的核心循环是：

```text
系统给出节点
↓
玩家连接 / 调参
↓
运行测试
↓
通过关卡
```

MVP0.1 的核心循环需要升级为：

```text
玩家用已有基础能力搭建新组件
↓
运行组件级测试
↓
通过认证
↓
封装成可复用节点
↓
进入个人组件库
↓
下一关必须使用自己造出来的组件
↓
继续升级为更高级系统
```

最终玩家不是“解完一系列题”，而是逐步拥有一套自己构建出来的 tiny LLM 系统：

```text
Scalar
→ Vector
→ Matrix
→ Tensor
→ Tokenizer
→ Embedding
→ Linear
→ Attention
→ MLP
→ Transformer Block
→ Tiny LLM Runner
```

---

# 1. 核心概念总览

MVP0.1 的组件系统由五个核心对象构成：

```text
ComponentBlueprint
BuiltComponent
Pack
Certify
Upgrade
```

它们之间的关系如下：

```text
ComponentBlueprint
组件蓝图，定义“要造什么”

        ↓ Build

ImplementationGraph
玩家搭出来的内部实现图

        ↓ Test / Certify

BuiltComponent
通过测试后的玩家组件实例

        ↓ Pack

Reusable Module
进入组件库，可作为新节点使用

        ↓ Upgrade

BuiltComponent v2 / v3 / ...
组件持续增强，支持更多能力
```

---

# 2. ComponentBlueprint 定义

## 2.1 概念说明

`ComponentBlueprint` 是一个组件的设计蓝图。它不代表玩家已经拥有这个组件，而是代表系统定义的一个“可被玩家构建的目标”。

它回答这些问题：

```text
这个组件叫什么？
它属于哪一类？
玩家需要先拥有哪些组件才能建造它？
它的输入输出接口是什么？
它内部应该由哪些已有组件组合而成？
它需要通过哪些测试？
通过后会解锁什么新节点？
后续还能升级成什么？
```

例如：

```text
Linear v1 的 ComponentBlueprint
要求玩家已经拥有：
TensorBox、ParameterMatrix、MatMulGate

玩家需要搭出：
x[B,T,C] @ weight[C,O] -> out[B,T,O]

通过测试后解锁：
LinearModule
```

---

## 2.2 TypeScript 数据结构建议

```ts
export type ComponentBlueprint = {
  id: string;
  title: string;
  shortTitle?: string;
  version: number;

  category: ComponentCategory;
  tier: ComponentTier;

  description: string;
  learningGoal: string;
  playerFantasy: string;

  requires: ComponentRequirement[];

  interface: ComponentInterface;

  buildSpec: ComponentBuildSpec;

  certification: ComponentCertificationSpec;

  packaging: ComponentPackagingSpec;

  upgrades?: ComponentUpgradeSpec[];

  unlocks?: ComponentUnlock[];

  code?: ComponentCodeSpec;

  ui?: ComponentUISpec;

  metadata?: ComponentMetadata;
};
```

---

## 2.3 字段完整说明

## `id`

组件蓝图唯一 ID。

```ts
id: "component.linear.v1"
```

命名建议：

```text
component.scalar.v1
component.vector.v1
component.matrix.v1
component.tensor.axis_aware.v1
component.tokenizer.v1
component.linear.v1
component.attention_head.v1
```

---

## `title`

展示给玩家的组件名称。

```ts
title: "Linear Projection v1"
```

中文可显示为：

```text
线性投影 v1
```

---

## `shortTitle`

图节点上显示的短名称。

```ts
shortTitle: "Linear"
```

---

## `version`

蓝图版本。

```ts
version: 1
```

注意：这是设计蓝图版本，不等于玩家实例版本。玩家实例版本由 `BuiltComponent.version` 管理。

---

## `category`

组件类别。

```ts
export type ComponentCategory =
  | "primitive"
  | "tensor"
  | "operation"
  | "data"
  | "tokenizer"
  | "model"
  | "attention"
  | "training"
  | "inference"
  | "system";
```

示例：

```text
ScalarCell         primitive
VectorRail         tensor
MatMulGate         operation
TokenizerSocket    tokenizer
Linear             model
AttentionHead      attention
CrossEntropyLoss   training
Sampler            inference
```

---

## `tier`

组件所处演进层级，用于控制解锁顺序。

```ts
export type ComponentTier =
  | "foundation"
  | "basic_tensor"
  | "data_pipeline"
  | "neural_core"
  | "attention_core"
  | "transformer_core"
  | "training_loop"
  | "inference_loop";
```

---

## `description`

面向玩家的解释。

```ts
description:
  "Linear Projection takes a hidden vector and projects it into a new channel space using a trainable weight matrix.";
```

中文：

```text
Linear Projection 会把 hidden vector 投影到新的通道空间，是 Attention、MLP、LM Head 的基础组件。
```

---

## `learningGoal`

该组件的教学目标。

```ts
learningGoal:
  "Understand that Linear is MatMul plus an optional bias, and that the C axis is consumed while O is created.";
```

---

## `playerFantasy`

这个组件在游戏中的“玩家幻想”。

```ts
playerFantasy:
  "You are building your first reusable neural network layer.";
```

中文：

```text
你正在造出第一个可以反复使用的神经网络层。
```

这个字段很重要，它负责把抽象学习目标转成游戏目标。

---

# 3. ComponentRequirement 定义

## 3.1 概念说明

`requires` 定义玩家在建造当前组件前必须已经拥有的组件或能力。

例如，建造 `Linear` 之前，玩家必须已经认证：

```text
TensorBox
MatMulGate
ParameterMatrix
```

建造 `TokenizerSocket` 之前，玩家必须已经认证：

```text
TextInput
BoundarySplitter
VocabTable
VocabLookup
TokenBuffer
AttentionMaskBuilder
```

---

## 3.2 数据结构

```ts
export type ComponentRequirement = {
  componentId: string;
  minVersion?: number;
  certificationLevel?: CertificationLevel;
  reason: string;
  optional?: boolean;
};
```

---

## 3.3 示例

```ts
requires: [
  {
    componentId: "component.tensor.axis_aware.v1",
    minVersion: 1,
    certificationLevel: "certified",
    reason: "Linear needs tensors with semantic axes so C and O are not guessed from size."
  },
  {
    componentId: "component.matmul.v1",
    minVersion: 1,
    certificationLevel: "certified",
    reason: "Linear is built on top of MatMul."
  },
  {
    componentId: "component.parameter_matrix.v1",
    minVersion: 1,
    certificationLevel: "certified",
    reason: "Linear needs a trainable weight matrix."
  }
]
```

---

# 4. ComponentInterface 定义

## 4.1 概念说明

`ComponentInterface` 定义组件对外暴露的输入输出端口。这相当于真实编程中的函数签名或类接口。

例如：

```python
def linear(x: Tensor[B,T,C]) -> Tensor[B,T,O]:
    ...
```

对应组件接口：

```text
input:
  x: float32[B,T,C]

output:
  out: float32[B,T,O]
```

---

## 4.2 数据结构

```ts
export type ComponentInterface = {
  inputs: ComponentPortSpec[];
  outputs: ComponentPortSpec[];

  params?: ComponentParamSpec[];

  contracts?: ComponentContract[];
};
```

---

## 4.3 ComponentPortSpec

```ts
export type ComponentPortSpec = {
  id: string;
  label: string;
  direction: "in" | "out";

  dtype: DType;
  shape?: ShapePattern;
  axes?: AxisPattern;

  required: boolean;

  semanticRole?: string;

  description?: string;
};
```

---

## 4.4 ShapePattern

```ts
export type ShapePattern = {
  rank?: number;
  dims?: Array<number | string>;
  allowDynamic?: boolean;
};
```

示例：

```ts
shape: {
  rank: 3,
  dims: ["B", "T", "C"],
  allowDynamic: true
}
```

---

## 4.5 AxisPattern

```ts
export type AxisPattern = {
  axes: string[];
  strictOrder: boolean;
};
```

示例：

```ts
axes: {
  axes: ["B", "T", "C"],
  strictOrder: true
}
```

---

## 4.6 ComponentParamSpec

组件参数，例如：

```text
Linear 的 outFeatures
Tokenizer 的 policy
CausalMask 的 maskedValue
Softmax 的 dim
```

```ts
export type ComponentParamSpec = {
  id: string;
  label: string;
  type: "number" | "string" | "boolean" | "enum" | "axis" | "shape";

  defaultValue: unknown;

  options?: Array<{
    value: unknown;
    label: string;
    description?: string;
  }>;

  lockedInMVP?: boolean;

  description?: string;
};
```

---

## 4.7 ComponentContract

组件必须满足的接口约束。

```ts
export type ComponentContract = {
  id: string;
  type:
    | "dtype"
    | "shape"
    | "axis_semantics"
    | "numeric"
    | "mask"
    | "no_truncation"
    | "no_oov"
    | "custom";

  targetPortId: string;

  expected: unknown;

  failureMessage: string;

  debugHint?: string;
};
```

---

## 4.8 Linear 接口示例

```ts
interface: {
  inputs: [
    {
      id: "x",
      label: "hidden",
      direction: "in",
      dtype: "float32",
      shape: { rank: 3, dims: ["B", "T", "C"], allowDynamic: true },
      axes: { axes: ["B", "T", "C"], strictOrder: true },
      required: true,
      semanticRole: "input hidden states"
    }
  ],

  outputs: [
    {
      id: "out",
      label: "projected",
      direction: "out",
      dtype: "float32",
      shape: { rank: 3, dims: ["B", "T", "O"], allowDynamic: true },
      axes: { axes: ["B", "T", "O"], strictOrder: true },
      required: true,
      semanticRole: "projected hidden states"
    }
  ],

  params: [
    {
      id: "outFeatures",
      label: "Output channels",
      type: "number",
      defaultValue: 4,
      description: "The O axis created by Linear."
    }
  ],

  contracts: [
    {
      id: "linear-output-axis",
      type: "axis_semantics",
      targetPortId: "out",
      expected: ["B", "T", "O"],
      failureMessage: "Linear must preserve B/T and create O.",
      debugHint: "Check MatMul right input orientation."
    }
  ]
}
```

---

# 5. ComponentBuildSpec 定义

## 5.1 概念说明

`ComponentBuildSpec` 定义玩家应该如何搭建这个组件。

它包含：

```text
初始图
目标图
允许使用的组件
禁止使用的组件
节点预算
边预算
是否允许自由搭建
是否允许自动修复按钮
```

---

## 5.2 数据结构

```ts
export type ComponentBuildSpec = {
  mode: "guided_repair" | "partial_build" | "free_build" | "upgrade_build";

  initialGraph: GraphSpec;
  targetGraph?: GraphSpec;

  allowedComponents: string[];
  forbiddenComponents?: string[];

  nodeBudget?: number;
  edgeBudget?: number;

  requiredRecipe?: ComponentRecipeStep[];

  buildHints?: ComponentBuildHint[];

  failureScenarios?: ComponentFailureScenario[];

  packagingTarget: PackagingTarget;
};
```

---

## 5.3 Build Mode 说明

### `guided_repair`

系统给出一台坏机器，玩家修一个局部。

适合新概念第一次出现。

```text
Linear v1 初学关：
系统给出 hidden -> matmul -> out
但 weight 方向错误
玩家插入 Transpose
```

---

### `partial_build`

系统给出输入和输出，中间缺 1-2 个模块。

适合玩家已经理解核心结构后。

```text
Tokenizer v1：
TextInput 和 TokenBuffer 已给出
玩家补 Splitter + VocabLookup
```

---

### `free_build`

玩家从已有组件库中自由搭完整内部图。

适合封装最终组件。

```text
AttentionHead v1：
玩家自由搭 Q/K/V、QK Score、Mask、Softmax、Weighted Sum
```

---

### `upgrade_build`

玩家在已有组件基础上加新能力。

```text
Tokenizer v2：
在 Tokenizer v1 内加入 MergeRules
```

---

## 5.4 ComponentRecipeStep

```ts
export type ComponentRecipeStep = {
  id: string;
  type:
    | "place_node"
    | "connect_edge"
    | "set_param"
    | "remove_edge"
    | "replace_node"
    | "pack_subgraph";

  label: string;

  target?: {
    nodeId?: string;
    moduleId?: string;
    from?: PortRef;
    to?: PortRef;
    param?: {
      nodeId: string;
      key: string;
      value: unknown;
    };
  };

  required: boolean;

  hint?: string;
};
```

---

## 5.5 示例：Linear v1 的 requiredRecipe

```ts
requiredRecipe: [
  {
    id: "place-weight",
    type: "place_node",
    label: "Place a ParameterMatrix as weight.",
    target: { nodeId: "weight", moduleId: "ParameterMatrix" },
    required: true
  },
  {
    id: "connect-x-left",
    type: "connect_edge",
    label: "Connect x.out -> matmul.left.",
    target: {
      from: { nodeId: "x", portId: "out" },
      to: { nodeId: "matmul", portId: "left" }
    },
    required: true
  },
  {
    id: "connect-weight-right",
    type: "connect_edge",
    label: "Connect weight.out -> matmul.right.",
    target: {
      from: { nodeId: "weight", portId: "out" },
      to: { nodeId: "matmul", portId: "right" }
    },
    required: true
  },
  {
    id: "connect-output",
    type: "connect_edge",
    label: "Connect matmul.out -> out.x.",
    target: {
      from: { nodeId: "matmul", portId: "out" },
      to: { nodeId: "out", portId: "x" }
    },
    required: true
  }
]
```

---

# 6. ComponentCertificationSpec 定义

## 6.1 概念说明

`Certify` 是组件系统的质量保证机制。玩家搭出来的组件不能直接进入组件库，必须通过认证。

认证不只是 visible test，而是包含多类测试：

```text
Visible Tests
Hidden Tests
Contract Tests
Mutation Tests
Reference Tests
Stress Tests
```

---

## 6.2 数据结构

```ts
export type ComponentCertificationSpec = {
  levels: CertificationLevelSpec[];

  minimumLevelToPack: CertificationLevel;

  visibleTests: TestCase[];
  hiddenTests: TestCase[];
  contractTests?: TestCase[];
  mutationTests?: TestCase[];
  stressTests?: TestCase[];

  scoring: CertificationScoringSpec;

  failurePolicy: CertificationFailurePolicy;
};
```

---

## 6.3 CertificationLevel

```ts
export type CertificationLevel =
  | "draft"
  | "visible_passed"
  | "certified"
  | "generalized"
  | "mastered";
```

---

## 6.4 认证等级说明

## Draft

组件已经开始搭建，但没有通过测试。

```text
状态：不可封装
可用于：当前关卡内部
不可用于：后续关卡
```

---

## Visible Passed

通过公开测试。

```text
状态：可预览封装
可用于：教学演示
不可用于：正式后续挑战
```

---

## Certified

通过 visible + hidden + contract tests。

```text
状态：正式认证
可用于：后续关卡
可进入：Component Library
```

---

## Generalized

通过更大范围的 shape / text / edge case mutation。

```text
状态：高质量组件
奖励：更高 Rank、可解锁 advanced upgrade
```

---

## Mastered

以低失败次数、无提示、低复杂度通过全部测试。

```text
状态：精通
奖励：组件皮肤、优化版代码、性能徽章
```

---

## 6.5 CertificationLevelSpec

```ts
export type CertificationLevelSpec = {
  level: CertificationLevel;
  requirements: CertificationRequirement[];
  unlocks?: ComponentUnlock[];
};
```

---

## 6.6 CertificationRequirement

```ts
export type CertificationRequirement =
  | {
      type: "test_group_passed";
      group: "visible" | "hidden" | "contract" | "mutation" | "stress";
    }
  | {
      type: "max_failed_runs";
      value: number;
    }
  | {
      type: "max_hints_used";
      value: number;
    }
  | {
      type: "max_extra_nodes";
      value: number;
    }
  | {
      type: "required_recipe_complete";
    };
```

---

## 6.7 CertificationScoringSpec

```ts
export type CertificationScoringSpec = {
  rankRules: CertificationRankRule[];

  metrics: CertificationMetric[];
};
```

---

## 6.8 CertificationMetric

```ts
export type CertificationMetric =
  | "visibleRuns"
  | "hiddenRuns"
  | "failedRuns"
  | "hintsUsed"
  | "extraNodes"
  | "extraEdges"
  | "runtimeSteps"
  | "componentComplexity";
```

---

## 6.9 Rank 规则示例

```ts
rankRules: [
  {
    rank: "S",
    requirements: [
      { type: "test_group_passed", group: "hidden" },
      { type: "max_failed_runs", value: 0 },
      { type: "max_hints_used", value: 0 },
      { type: "max_extra_nodes", value: 0 }
    ]
  },
  {
    rank: "A",
    requirements: [
      { type: "test_group_passed", group: "hidden" },
      { type: "max_failed_runs", value: 2 },
      { type: "max_hints_used", value: 0 }
    ]
  },
  {
    rank: "B",
    requirements: [
      { type: "test_group_passed", group: "hidden" }
    ]
  }
]
```

---

## 6.10 CertificationFailurePolicy

```ts
export type CertificationFailurePolicy = {
  allowPartialCertification: boolean;

  showFirstFailureOnly: boolean;

  unlockDebugProbesAfterFailures?: number;

  failMessages: Record<string, string>;

  remediationHints?: CertificationRemediationHint[];
};
```

---

## 6.11 Certify 运行流程

```text
1. 检查 requiredRecipe 是否完成
2. 检查接口定义是否匹配 ComponentInterface
3. 运行 visible tests
4. 运行 hidden tests
5. 运行 contract tests
6. 运行 mutation tests
7. 计算 rank
8. 写入认证结果
9. 如果达到 minimumLevelToPack，允许 Pack
```

伪代码：

```ts
function certifyComponent(
  blueprint: ComponentBlueprint,
  graph: GraphSpec,
  runStats: ComponentRunStats
): CertificationResult {
  const recipeResult = checkRequiredRecipe(blueprint.buildSpec.requiredRecipe, graph);
  if (!recipeResult.passed) return blocked("recipe_incomplete", recipeResult);

  const interfaceResult = checkInterface(blueprint.interface, graph);
  if (!interfaceResult.passed) return blocked("interface_mismatch", interfaceResult);

  const visible = runTests(graph, blueprint.certification.visibleTests);
  if (!visible.passed) return failed("visible_failed", visible);

  const hidden = runTests(graph, blueprint.certification.hiddenTests);
  if (!hidden.passed) return failed("hidden_failed", hidden);

  const contract = runTests(graph, blueprint.certification.contractTests ?? []);
  if (!contract.passed) return failed("contract_failed", contract);

  const mutation = runTests(graph, blueprint.certification.mutationTests ?? []);

  const level = computeCertificationLevel({
    visible,
    hidden,
    contract,
    mutation,
    runStats
  });

  const rank = computeRank(level, runStats, graph);

  return {
    passed: true,
    level,
    rank,
    visible,
    hidden,
    contract,
    mutation
  };
}
```

---

# 7. ComponentPackagingSpec / Pack 定义

## 7.1 概念说明

`Pack` 是把玩家搭好的内部 graph 封装成一个可复用组件节点的过程。

真实编程类比：

```python
# 玩家原来手动搭
x = matmul(hidden, weight)
x = add_bias(x, bias)

# Pack 后变成
x = linear(hidden)
```

游戏里则是：

```text
内部 graph:
Input -> MatMul -> BiasAdd -> Output

Pack 成:
Linear 节点
```

---

## 7.2 PackagingSpec 数据结构

```ts
export type ComponentPackagingSpec = {
  exportModuleId: string;

  displayName: string;

  icon?: string;

  externalInterface: ComponentInterface;

  internalGraphPolicy: InternalGraphPolicy;

  parameterExposure: ParameterExposureSpec[];

  codeExport: CodeExportSpec;

  libraryPlacement: ComponentLibraryPlacement;

  packAnimation?: PackAnimationSpec;
};
```

---

## 7.3 InternalGraphPolicy

```ts
export type InternalGraphPolicy = {
  preserveInternalGraph: boolean;

  allowExpandInside: boolean;

  allowEditAfterPack: boolean;

  editCreatesNewVersion: boolean;

  hideCertifiedInternalsByDefault: boolean;
};
```

推荐配置：

```ts
internalGraphPolicy: {
  preserveInternalGraph: true,
  allowExpandInside: true,
  allowEditAfterPack: true,
  editCreatesNewVersion: true,
  hideCertifiedInternalsByDefault: true
}
```

说明：

```text
组件封装后默认只显示一个节点。
但玩家可以点击 Expand Inside 查看内部实现。
如果修改已认证组件，不直接污染旧版本，而是创建新版本。
```

---

## 7.4 ParameterExposureSpec

定义哪些内部参数暴露到封装节点外部。

例如 `Linear` 内部有：

```text
weight
bias
outFeatures
orientation
```

但 MVP0.1 可能只暴露：

```text
outFeatures
useBias
```

```ts
export type ParameterExposureSpec = {
  internalNodeId: string;
  internalParamKey: string;

  externalParamKey: string;
  label: string;

  editable: boolean;

  defaultValue: unknown;

  description?: string;
};
```

示例：

```ts
parameterExposure: [
  {
    internalNodeId: "weight",
    internalParamKey: "outFeatures",
    externalParamKey: "outFeatures",
    label: "Output channels",
    editable: true,
    defaultValue: 4,
    description: "Controls the O axis created by Linear."
  }
]
```

---

## 7.5 CodeExportSpec

```ts
export type CodeExportSpec = {
  kind: "function" | "class" | "module";

  language: "python" | "typescript";

  name: string;

  template: string;

  imports?: string[];

  docstring?: string;
};
```

### Function 示例

```ts
codeExport: {
  kind: "function",
  language: "python",
  name: "linear",
  template: `
def linear(x, weight):
    return matmul(x, weight)
`,
  docstring: "Linear projection built from player's MatMulGate."
}
```

### Class 示例

```ts
codeExport: {
  kind: "class",
  language: "python",
  name: "Linear",
  template: `
class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Parameter([in_features, out_features])

    def __call__(self, x):
        return matmul(x, self.weight)
`
}
```

---

## 7.6 ComponentLibraryPlacement

```ts
export type ComponentLibraryPlacement = {
  category: ComponentCategory;
  shelf: string;
  tags: string[];
  sortOrder?: number;
};
```

示例：

```ts
libraryPlacement: {
  category: "model",
  shelf: "Neural Core",
  tags: ["projection", "trainable", "reusable"],
  sortOrder: 30
}
```

---

## 7.7 Pack 过程

```text
1. 玩家点击 Pack Component
2. 系统检查认证等级是否达到 minimumLevelToPack
3. 系统读取 ComponentInterface
4. 系统把内部 graph 封装成 BuiltComponent
5. 系统生成 export module
6. 系统加入 Component Library
7. 系统显示 Pack 动画
8. 下一关 modulePalette 可以引用该组件
```

伪代码：

```ts
function packComponent(
  blueprint: ComponentBlueprint,
  graph: GraphSpec,
  certification: CertificationResult
): BuiltComponent {
  if (!canPack(blueprint, certification)) {
    throw new Error("Component has not reached minimum certification level.");
  }

  return {
    id: createBuiltComponentId(blueprint.id),
    blueprintId: blueprint.id,
    title: blueprint.title,
    version: 1,
    status: "packed",
    certification,
    implementationGraph: graph,
    exportedModuleId: blueprint.packaging.exportModuleId,
    interface: blueprint.packaging.externalInterface,
    exposedParams: extractExposedParams(blueprint.packaging, graph),
    createdAt: Date.now(),
    updatedAt: Date.now(),
    lineage: {
      parentComponentId: undefined,
      sourceBlueprintId: blueprint.id
    }
  };
}
```

---

# 8. BuiltComponent 定义

## 8.1 概念说明

`BuiltComponent` 是玩家已经完成并保存的组件实例。

它不同于 `ComponentBlueprint`：

```text
ComponentBlueprint = 系统定义的目标
BuiltComponent = 玩家实际造出来的组件
```

玩家可以有多个 BuiltComponent 来自同一个 Blueprint：

```text
Linear v1 - 基础版本
Linear v2 - 增加 bias
Linear v3 - 支持 trainable params
```

也可以存在不同质量等级：

```text
TokenizerSocket v1 - Certified
TokenizerSocket v2 - Generalized
```

---

## 8.2 数据结构

```ts
export type BuiltComponent = {
  id: string;

  blueprintId: string;

  title: string;

  version: number;

  status: BuiltComponentStatus;

  certification: CertificationResult;

  implementationGraph: GraphSpec;

  exportedModuleId: string;

  interface: ComponentInterface;

  exposedParams: Record<string, unknown>;

  library: BuiltComponentLibraryInfo;

  lineage: BuiltComponentLineage;

  usageStats: BuiltComponentUsageStats;

  upgradeState: BuiltComponentUpgradeState;

  codeSnapshot?: BuiltComponentCodeSnapshot;

  createdAt: number;
  updatedAt: number;
};
```

---

## 8.3 BuiltComponentStatus

```ts
export type BuiltComponentStatus =
  | "draft"
  | "certified"
  | "packed"
  | "deprecated"
  | "upgrading"
  | "broken";
```

### Draft

玩家已经开始搭建，但还没认证。

### Certified

组件通过认证，但还没封装入库。

### Packed

组件已封装，可在后续关卡作为节点使用。

### Deprecated

旧版本被新版本替代，但仍可查看和回放。

### Upgrading

当前组件正在升级中。

### Broken

因为底层依赖升级或接口变更，组件暂时不可用。

---

## 8.4 BuiltComponentLibraryInfo

```ts
export type BuiltComponentLibraryInfo = {
  visible: boolean;

  category: ComponentCategory;

  shelf: string;

  tags: string[];

  unlockedAtLevelId: string;

  sortOrder?: number;
};
```

---

## 8.5 BuiltComponentLineage

记录组件演化关系。

```ts
export type BuiltComponentLineage = {
  sourceBlueprintId: string;

  parentComponentId?: string;

  parentVersion?: number;

  upgradeId?: string;

  childComponentIds?: string[];
};
```

例如：

```text
Linear v2 的 parent 是 Linear v1
Tokenizer v3 的 parent 是 Tokenizer v2
AttentionHead v1 依赖 Linear v2
```

---

## 8.6 BuiltComponentUsageStats

```ts
export type BuiltComponentUsageStats = {
  usedInLevels: string[];

  usedByComponents: string[];

  successfulRuns: number;

  failedRuns: number;

  lastUsedAt?: number;
};
```

这个字段用于让玩家感受到组件是“资产”：

```text
Linear v1 已被用于：
- Q Projection
- K Projection
- V Projection
- MLP Up
- LM Head
```

---

## 8.7 BuiltComponentUpgradeState

```ts
export type BuiltComponentUpgradeState = {
  availableUpgrades: string[];

  appliedUpgrades: string[];

  lockedUpgrades: Array<{
    upgradeId: string;
    reason: string;
  }>;
};
```

---

## 8.8 BuiltComponentCodeSnapshot

```ts
export type BuiltComponentCodeSnapshot = {
  language: "python" | "typescript";

  code: string;

  generatedAt: number;

  graphHash: string;
};
```

保存快照的目的是：即使后续 codegen 规则变化，也能回看当时玩家组件对应的代码。

---

# 9. Upgrade 定义

## 9.1 概念说明

`Upgrade` 是在已有组件基础上增加能力的机制。

组件不能一次性变成最终形态，而是逐步升级：

```text
Tensor v1：只有 data
Tensor v2：增加 shape
Tensor v3：增加 dtype
Tensor v4：增加 axis semantics
Tensor v5：支持 transpose
Tensor v6：支持 broadcast metadata
Tensor v7：支持 grad
```

这种演进非常适合游戏化，因为每一次 upgrade 都对应一个具体问题：

```text
没有 dtype → raw_text 被错误接进 Embedding
没有 axis semantics → [2,4,8] 无法判断 B/T/C
没有 broadcast metadata → bias[O] 不知道该扩展到哪里
```

---

## 9.2 ComponentUpgradeSpec

```ts
export type ComponentUpgradeSpec = {
  id: string;

  title: string;

  fromComponentId: string;
  fromVersion: number;

  toVersion: number;

  description: string;

  unlockCondition: UpgradeUnlockCondition;

  upgradeGoal: string;

  addedCapabilities: ComponentCapability[];

  interfaceChanges?: ComponentInterfaceChange[];

  buildSpec: ComponentBuildSpec;

  certification: ComponentCertificationSpec;

  migration?: ComponentMigrationSpec;
};
```

---

## 9.3 UpgradeUnlockCondition

```ts
export type UpgradeUnlockCondition =
  | {
      type: "component_certified";
      componentId: string;
      minVersion: number;
    }
  | {
      type: "level_completed";
      levelId: string;
    }
  | {
      type: "capability_needed";
      capability: ComponentCapability;
    }
  | {
      type: "manual_unlock";
      reason: string;
    };
```

---

## 9.4 ComponentCapability

```ts
export type ComponentCapability =
  | "stores_scalar"
  | "stores_vector"
  | "stores_matrix"
  | "has_shape"
  | "has_dtype"
  | "has_axis_semantics"
  | "supports_transpose"
  | "supports_broadcast"
  | "supports_trainable_params"
  | "supports_token_split"
  | "supports_vocab_lookup"
  | "supports_padding"
  | "supports_attention_mask"
  | "supports_softmax"
  | "supports_causal_mask"
  | "supports_gradient";
```

---

## 9.5 ComponentInterfaceChange

```ts
export type ComponentInterfaceChange = {
  type:
    | "add_input"
    | "remove_input"
    | "add_output"
    | "remove_output"
    | "change_dtype"
    | "change_shape"
    | "add_param"
    | "remove_param";

  targetId: string;

  before?: unknown;

  after?: unknown;

  migrationHint?: string;
};
```

---

## 9.6 ComponentMigrationSpec

定义升级后如何迁移旧组件。

```ts
export type ComponentMigrationSpec = {
  strategy:
    | "copy_graph"
    | "wrap_old_component"
    | "replace_internal_node"
    | "manual_rebuild";

  preserveCertification?: boolean;

  migrationSteps?: ComponentRecipeStep[];

  breakingChanges?: string[];
};
```

---

## 9.7 Upgrade 示例：Tensor v2 → Tensor v3

```ts
const tensorDTypeUpgrade: ComponentUpgradeSpec = {
  id: "upgrade.tensor.dtype.v3",
  title: "Add DType Badge",
  fromComponentId: "component.tensor.v2",
  fromVersion: 2,
  toVersion: 3,

  description:
    "Tensor now tracks whether its data is float32, int, raw_text, bool, or mask.",

  unlockCondition: {
    type: "capability_needed",
    capability: "has_dtype"
  },

  upgradeGoal:
    "Prevent raw_text from entering numeric modules and prevent int token_ids from entering float MatMul.",

  addedCapabilities: ["has_dtype"],

  interfaceChanges: [
    {
      type: "add_param",
      targetId: "dtype",
      after: "float32",
      migrationHint: "Existing numeric tensors default to float32."
    }
  ],

  buildSpec: {
    mode: "upgrade_build",
    initialGraph: "...",
    allowedComponents: ["component.tensor.v2", "component.dtype_badge.v1"],
    nodeBudget: 4,
    edgeBudget: 4,
    packagingTarget: {
      exportModuleId: "TensorBoxV3"
    }
  },

  certification: {
    minimumLevelToPack: "certified",
    levels: [],
    visibleTests: [],
    hiddenTests: [],
    scoring: {
      rankRules: [],
      metrics: ["visibleRuns", "failedRuns", "hintsUsed"]
    },
    failurePolicy: {
      allowPartialCertification: false,
      showFirstFailureOnly: true,
      failMessages: {}
    }
  }
};
```

---

# 10. Pack / Certify / Upgrade 生命周期

## 10.1 总生命周期

```text
Blueprint Available
↓
Player Starts Build
↓
Draft BuiltComponent Created
↓
Visible Tests Run
↓
Visible Passed
↓
Hidden / Contract Tests Run
↓
Certified
↓
Pack
↓
Packed Component enters Library
↓
Component reused in later Blueprints
↓
Upgrade available
↓
Upgrade Build
↓
New Version Certified
↓
Old Version deprecated or preserved
```

---

## 10.2 状态机

```ts
export type ComponentLifecycleState =
  | "blueprint_locked"
  | "blueprint_available"
  | "draft_building"
  | "visible_testing"
  | "visible_passed"
  | "certifying"
  | "certified"
  | "packing"
  | "packed"
  | "reused"
  | "upgrade_available"
  | "upgrading"
  | "upgraded"
  | "deprecated";
```

---

## 10.3 状态转移表

| 当前状态                  | 触发行为                     | 下一个状态                 |
| --------------------- | ------------------------ | --------------------- |
| `blueprint_locked`    | 满足 requires              | `blueprint_available` |
| `blueprint_available` | 玩家开始搭建                   | `draft_building`      |
| `draft_building`      | Run Visible              | `visible_testing`     |
| `visible_testing`     | Visible passed           | `visible_passed`      |
| `visible_testing`     | Visible failed           | `draft_building`      |
| `visible_passed`      | Run Certify              | `certifying`          |
| `certifying`          | Hidden + Contract passed | `certified`           |
| `certifying`          | 测试失败                     | `draft_building`      |
| `certified`           | Pack                     | `packing`             |
| `packing`             | Pack completed           | `packed`              |
| `packed`              | 后续关卡使用                   | `reused`              |
| `reused`              | 新能力需求出现                  | `upgrade_available`   |
| `upgrade_available`   | Start Upgrade            | `upgrading`           |
| `upgrading`           | 新版本通过认证                  | `upgraded`            |
| `upgraded`            | 旧版本被替代                   | `deprecated`          |

---

# 11. Component Library 规格

## 11.1 概念说明

`Component Library` 是玩家所有已认证组件的集合。

它既是工具箱，也是成长记录。

---

## 11.2 数据结构

```ts
export type ComponentLibrary = {
  ownerId: string;

  components: BuiltComponent[];

  shelves: ComponentShelf[];

  activeVersions: Record<string, string>;

  history: ComponentLibraryEvent[];
};
```

---

## 11.3 ComponentShelf

```ts
export type ComponentShelf = {
  id: string;
  title: string;
  category: ComponentCategory;
  componentIds: string[];
  collapsedByDefault?: boolean;
};
```

推荐 shelf：

```text
Primitive
Tensor Core
Math Ops
Text Pipeline
Neural Core
Attention
Transformer
Training
Inference
```

---

## 11.4 activeVersions

同一个组件可能有多个版本：

```text
Linear v1
Linear v2
Linear v3
```

`activeVersions` 定义后续关卡默认使用哪个版本。

```ts
activeVersions: {
  "component.linear": "built.linear.v2",
  "component.tokenizer": "built.tokenizer.v3"
}
```

---

## 11.5 ComponentLibraryEvent

```ts
export type ComponentLibraryEvent = {
  type:
    | "component_certified"
    | "component_packed"
    | "component_upgraded"
    | "component_deprecated"
    | "component_reused";

  componentId: string;

  levelId?: string;

  timestamp: number;

  message: string;
};
```

---

# 12. 组件依赖解析

MVP0.1 需要一个依赖解析器，用于判断：

```text
某个蓝图是否可以解锁
某个关卡是否可以开始
某个组件是否因依赖损坏而 broken
某个升级是否可用
```

---

## 12.1 DependencyResolver

```ts
export type DependencyResolver = {
  canUnlockBlueprint(
    blueprint: ComponentBlueprint,
    library: ComponentLibrary
  ): DependencyCheckResult;

  canUseComponent(
    componentId: string,
    requiredVersion: number,
    library: ComponentLibrary
  ): DependencyCheckResult;

  getMissingRequirements(
    blueprint: ComponentBlueprint,
    library: ComponentLibrary
  ): ComponentRequirement[];

  getDependents(
    componentId: string,
    library: ComponentLibrary
  ): BuiltComponent[];
};
```

---

## 12.2 DependencyCheckResult

```ts
export type DependencyCheckResult = {
  ok: boolean;

  missing: ComponentRequirement[];

  outdated: Array<{
    componentId: string;
    requiredVersion: number;
    currentVersion: number;
  }>;

  uncertified: string[];

  message: string;
};
```

---

# 13. 组件与 ModuleDef 的关系

当前 Graph Challenge 运行时依赖 `ModuleDef`。MVP0.1 不应该推翻它，而是让 `BuiltComponent` 可以导出一个动态 `ModuleDef`。

---

## 13.1 转换关系

```text
ComponentBlueprint
↓ 玩家搭建
BuiltComponent
↓ Pack
DynamicModuleDef
↓ 注册
ModuleRegistry
↓ 后续 Graph Challenge 使用
```

---

## 13.2 BuiltComponent → ModuleDef

```ts
function builtComponentToModuleDef(component: BuiltComponent): ModuleDef {
  return {
    id: component.exportedModuleId,
    label: component.title,
    category: mapComponentCategoryToModuleCategory(component.library.category),

    inputs: component.interface.inputs.map(toPortDef),
    outputs: component.interface.outputs.map(toPortDef),

    defaultParams: component.exposedParams,

    summary: `Player-built component: ${component.title}`,

    pseudoCode: component.codeSnapshot?.code,

    execute: createComponentExecutor(component),

    infer: createComponentInfer(component)
  };
}
```

---

## 13.3 执行方式

动态组件有两种执行方案。

### 方案 A：展开内部图执行

```text
Linear 节点执行时
↓
内部展开成 MatMul + BiasAdd
↓
复用现有 graphExecutor
```

优点：

```text
实现一致
方便 debug
可以 Expand Inside
```

缺点：

```text
运行成本高一点
```

---

### 方案 B：编译为原生 execute 函数

```text
Pack 时生成 execute()
Linear 直接执行 matmul
```

优点：

```text
运行快
组件像真实模块
```

缺点：

```text
debug 和 trace 更复杂
```

MVP0.1 建议使用方案 A：

```text
Pack 后外部显示成单节点
执行时内部仍然可以 trace
```

---

# 14. 用户界面规格

## 14.1 Component Builder

专门用于搭组件的界面。

布局建议：

```text
┌─────────────────────────────────────────────────────────┐
│ Component Header: Linear v1                             │
├───────────────┬────────────────────────┬────────────────┤
│ Requirements  │ Internal Graph Canvas   │ Interface/Test │
│ Component Lib │                        │ Code/Certify   │
├───────────────┴────────────────────────┴────────────────┤
│ Trace / Test Report / Pack Button                       │
└─────────────────────────────────────────────────────────┘
```

---

## 14.2 Header

显示：

```text
组件名
版本
认证状态
当前目标
Pack 状态
```

示例：

```text
Linear Projection v1
Status: Draft
Goal: Build x[B,T,C] @ W[C,O] -> out[B,T,O]
```

---

## 14.3 Requirements Panel

显示依赖组件：

```text
Required:
[x] TensorBox v4 Certified
[x] MatMulGate v1 Certified
[ ] ParameterMatrix v1 Missing
```

点击缺失组件可以跳转到对应蓝图。

---

## 14.4 Internal Graph Canvas

玩家搭组件内部实现。

和普通 Graph Challenge 的区别：

```text
这里不是解决一次性任务
而是在定义一个未来可复用节点的内部结构
```

---

## 14.5 Interface Panel

显示组件对外接口：

```text
Inputs
x: float32[B,T,C]

Outputs
out: float32[B,T,O]

Params
outFeatures: number
```

如果内部图输出不满足接口，直接报错。

---

## 14.6 Test Panel

分组显示：

```text
Visible Tests
Hidden Tests
Contract Tests
Mutation Tests
Stress Tests
```

---

## 14.7 Code Panel

显示三类代码：

```text
Interface Code
Implementation Code
Test Code
```

### Interface Code

```python
def linear(x: Tensor[B,T,C]) -> Tensor[B,T,O]:
    ...
```

### Implementation Code

```python
out = matmul(x, weight)
```

### Test Code

```python
assert out.axes == ["B", "T", "O"]
assert allclose(out, reference)
```

---

## 14.8 Pack Button

按钮状态：

```text
Pack Component       disabled: 未通过认证
Pack Component       enabled: 已达到 minimumLevelToPack
Packed               完成封装
Upgrade Available    可以升级
```

---

# 15. Pack 动画设计

为了让玩家感受到“封装”的意义，Pack 不应该只是保存按钮。

推荐动画：

```text
内部复杂 graph 缩小
↓
输入输出端口保留
↓
内部节点折叠成一个外壳
↓
外壳命名为 Linear v1
↓
组件飞入 Component Library
```

视觉变化：

```text
Before:
x -> MatMul -> BiasAdd -> out

After:
x -> [ Linear v1 ] -> out
```

文案：

```text
Linear v1 packed.
You can now reuse it in Attention, MLP, and LM Head.
```

---

# 16. Upgrade UI 设计

当组件可升级时，组件库里显示：

```text
Linear v1
Upgrade available: Add Bias
```

点击后进入 Upgrade Builder：

```text
Linear v1 内部图
+
新增目标：加入 bias[O] 并广播到 [B,T,O]
```

升级完成后：

```text
Linear v2 Certified
Linear v1 preserved
Future levels will use Linear v2 by default
```

---

# 17. 示例完整定义：Linear v1 Blueprint

```ts
export const linearV1Blueprint: ComponentBlueprint = {
  id: "component.linear.v1",
  title: "Linear Projection v1",
  shortTitle: "Linear",
  version: 1,

  category: "model",
  tier: "neural_core",

  description:
    "A reusable projection layer that maps hidden[B,T,C] into projected[B,T,O].",

  learningGoal:
    "Understand that Linear consumes C, preserves B/T, and creates a new O axis using MatMul.",

  playerFantasy:
    "Build your first reusable neural network layer.",

  requires: [
    {
      componentId: "component.tensor.axis_aware.v1",
      minVersion: 1,
      certificationLevel: "certified",
      reason: "Linear must know which axis is C."
    },
    {
      componentId: "component.matmul.v1",
      minVersion: 1,
      certificationLevel: "certified",
      reason: "Linear is built from MatMul."
    },
    {
      componentId: "component.parameter_matrix.v1",
      minVersion: 1,
      certificationLevel: "certified",
      reason: "Linear needs a trainable weight matrix."
    }
  ],

  interface: {
    inputs: [
      {
        id: "x",
        label: "hidden",
        direction: "in",
        dtype: "float32",
        shape: { rank: 3, dims: ["B", "T", "C"], allowDynamic: true },
        axes: { axes: ["B", "T", "C"], strictOrder: true },
        required: true,
        semanticRole: "input hidden states"
      }
    ],
    outputs: [
      {
        id: "out",
        label: "projected",
        direction: "out",
        dtype: "float32",
        shape: { rank: 3, dims: ["B", "T", "O"], allowDynamic: true },
        axes: { axes: ["B", "T", "O"], strictOrder: true },
        required: true,
        semanticRole: "projected hidden states"
      }
    ],
    params: [
      {
        id: "outFeatures",
        label: "Output channels",
        type: "number",
        defaultValue: 4,
        description: "The O axis created by Linear."
      }
    ],
    contracts: [
      {
        id: "linear-axis-contract",
        type: "axis_semantics",
        targetPortId: "out",
        expected: ["B", "T", "O"],
        failureMessage: "Linear must preserve B/T and create O.",
        debugHint: "Check whether weight is [C,O] before MatMul."
      }
    ]
  },

  buildSpec: {
    mode: "partial_build",
    initialGraph: {} as GraphSpec,
    targetGraph: {} as GraphSpec,
    allowedComponents: [
      "component.tensor.axis_aware.v1",
      "component.parameter_matrix.v1",
      "component.matmul.v1",
      "component.output_gate.v1"
    ],
    forbiddenComponents: [
      "component.attention_head.v1",
      "component.mlp.v1"
    ],
    nodeBudget: 6,
    edgeBudget: 6,
    requiredRecipe: [
      {
        id: "connect-x-left",
        type: "connect_edge",
        label: "Connect input hidden to MatMul left.",
        target: {
          from: { nodeId: "x", portId: "out" },
          to: { nodeId: "matmul", portId: "left" }
        },
        required: true
      },
      {
        id: "connect-weight-right",
        type: "connect_edge",
        label: "Connect weight to MatMul right.",
        target: {
          from: { nodeId: "weight", portId: "out" },
          to: { nodeId: "matmul", portId: "right" }
        },
        required: true
      },
      {
        id: "connect-output",
        type: "connect_edge",
        label: "Connect MatMul output to component output.",
        target: {
          from: { nodeId: "matmul", portId: "out" },
          to: { nodeId: "out", portId: "x" }
        },
        required: true
      }
    ],
    packagingTarget: {
      exportModuleId: "PlayerLinearV1"
    }
  },

  certification: {
    minimumLevelToPack: "certified",
    levels: [
      {
        level: "visible_passed",
        requirements: [
          { type: "test_group_passed", group: "visible" }
        ]
      },
      {
        level: "certified",
        requirements: [
          { type: "test_group_passed", group: "visible" },
          { type: "test_group_passed", group: "hidden" },
          { type: "test_group_passed", group: "contract" }
        ]
      },
      {
        level: "mastered",
        requirements: [
          { type: "test_group_passed", group: "hidden" },
          { type: "max_failed_runs", value: 0 },
          { type: "max_hints_used", value: 0 },
          { type: "max_extra_nodes", value: 0 }
        ]
      }
    ],
    visibleTests: [],
    hiddenTests: [],
    contractTests: [],
    mutationTests: [],
    scoring: {
      rankRules: [],
      metrics: ["visibleRuns", "hiddenRuns", "failedRuns", "hintsUsed", "extraNodes"]
    },
    failurePolicy: {
      allowPartialCertification: true,
      showFirstFailureOnly: true,
      failMessages: {
        shape_mismatch: "Linear output shape is wrong.",
        axis_semantic_error: "Linear output axes are wrong.",
        numeric_mismatch: "Linear shape looks right, but values differ from reference."
      }
    }
  },

  packaging: {
    exportModuleId: "PlayerLinearV1",
    displayName: "Linear v1",
    externalInterface: {} as ComponentInterface,
    internalGraphPolicy: {
      preserveInternalGraph: true,
      allowExpandInside: true,
      allowEditAfterPack: true,
      editCreatesNewVersion: true,
      hideCertifiedInternalsByDefault: true
    },
    parameterExposure: [
      {
        internalNodeId: "weight",
        internalParamKey: "outFeatures",
        externalParamKey: "outFeatures",
        label: "Output channels",
        editable: true,
        defaultValue: 4
      }
    ],
    codeExport: {
      kind: "class",
      language: "python",
      name: "Linear",
      template: `
class Linear:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):
        return matmul(x, self.weight)
`
    },
    libraryPlacement: {
      category: "model",
      shelf: "Neural Core",
      tags: ["projection", "matmul", "reusable"]
    }
  },

  upgrades: [
    {
      id: "upgrade.linear.bias.v2",
      title: "Add Bias",
      fromComponentId: "component.linear.v1",
      fromVersion: 1,
      toVersion: 2,
      description: "Add a bias[O] vector and broadcast it to [B,T,O].",
      unlockCondition: {
        type: "component_certified",
        componentId: "component.broadcast_add.v1",
        minVersion: 1
      },
      upgradeGoal: "Linear v2 supports y = xW + b.",
      addedCapabilities: ["supports_broadcast"],
      buildSpec: {} as ComponentBuildSpec,
      certification: {} as ComponentCertificationSpec
    }
  ],

  unlocks: [
    {
      type: "component_blueprint",
      id: "component.q_projection.v1",
      reason: "Q projection is a Linear layer."
    },
    {
      type: "component_blueprint",
      id: "component.mlp.v1",
      reason: "MLP is built from two Linear layers."
    }
  ],

  code: {
    language: "python",
    preferredForm: "class"
  },

  ui: {
    icon: "linear",
    colorTheme: "model",
    packAnimation: "collapse_graph_to_node"
  },

  metadata: {
    estimatedBuildMinutes: 8,
    difficulty: "medium"
  }
};
```

---

# 18. Unlock 定义

## 18.1 ComponentUnlock

```ts
export type ComponentUnlock =
  | {
      type: "component_blueprint";
      id: string;
      reason: string;
    }
  | {
      type: "module_palette_item";
      id: string;
      reason: string;
    }
  | {
      type: "upgrade";
      id: string;
      reason: string;
    }
  | {
      type: "level";
      id: string;
      reason: string;
    };
```

---

## 18.2 示例

Linear 认证后解锁：

```ts
unlocks: [
  {
    type: "component_blueprint",
    id: "component.q_projection.v1",
    reason: "Q Projection is a Linear layer applied to hidden states."
  },
  {
    type: "component_blueprint",
    id: "component.k_projection.v1",
    reason: "K Projection is a Linear layer applied to hidden states."
  },
  {
    type: "component_blueprint",
    id: "component.v_projection.v1",
    reason: "V Projection is a Linear layer applied to hidden states."
  },
  {
    type: "component_blueprint",
    id: "component.mlp.v1",
    reason: "MLP is built from Linear up and Linear down."
  }
]
```

---

# 19. MVP0.1 实现优先级

## P0：最小可用组件系统

必须实现：

```text
ComponentBlueprint
BuiltComponent
ComponentLibrary
Build → Test → Certify → Pack
BuiltComponent → ModuleDef
组件库 UI
```

先不做复杂升级，只要能让玩家：

```text
搭出 MatMulGate
Pack 成自己的 MatMulGate
下一关使用自己的 MatMulGate 搭 Linear
```

---

## P1：Upgrade 系统

实现：

```text
组件版本
UpgradeSpec
旧版本保留
新版本替代 activeVersion
升级关卡
```

示例：

```text
Tensor v1 → Tensor v2
Linear v1 → Linear v2
Tokenizer v1 → Tokenizer v2
```

---

## P2：Component Lineage / Expand Inside

实现：

```text
查看组件内部图
查看组件依赖树
查看组件被哪些高级组件使用
```

---

## P3：代码导出与高亮

实现：

```text
Component Interface Code
Implementation Code
Test Code
Graph ↔ Code line highlight
```

---

# 20. MVP0.1 验收标准

MVP0.1 的成功不以“做了多少关”为标准，而以玩家是否形成组件资产链为标准。

## 必须满足

```text
1. 玩家可以从蓝图开始搭建一个组件。
2. 玩家可以运行组件级 visible / hidden / contract tests。
3. 玩家可以把通过认证的组件 Pack 成新节点。
4. 被 Pack 的组件可以出现在后续关卡 module palette 中。
5. 后续组件可以依赖玩家之前 Pack 出来的组件。
6. 玩家可以在组件库里查看组件版本、认证状态、内部图和代码。
```

---

## 样例验收链

第一条 MVP0.1 样例链建议是：

```text
ScalarCell
→ VectorRail
→ DotProduct
→ MatrixBoard
→ MatMulGate
→ Linear
```

验收要求：

```text
玩家必须先认证 MatMulGate，才能进入 Linear 构建关。
Linear 关卡中不能直接使用系统 MatMul，必须使用玩家自己的 MatMulGate。
Linear 认证后进入组件库，并可用于 Q/K/V Projection。
```

---

# 21. 最终定义总结

## ComponentBlueprint

```text
系统给出的组件设计目标。
定义要造什么、需要什么、怎么测试、通过后解锁什么。
```

---

## BuiltComponent

```text
玩家实际造出来并保存的组件实例。
包含内部 graph、认证结果、封装节点、版本、依赖和使用记录。
```

---

## Pack

```text
把复杂内部 graph 折叠成一个可复用节点。
Pack 后组件进入玩家 Component Library。
```

---

## Certify

```text
组件质量认证流程。
通过 visible / hidden / contract / mutation tests 后，组件才可正式复用。
```

---

## Upgrade

```text
在已认证组件基础上添加新能力。
升级产生新版本，不直接破坏旧组件。
```

---

# 22. MVP0.1 的核心体验目标

MVP0.1 的最终体验应该让玩家产生一种明确的成长感：

```text
我先造出了一个数字。
我把数字排成向量。
我用向量做点积。
我把点积铺成矩阵乘法。
我把矩阵乘法封装成 Linear。
我又用 Linear 造出了 Attention。
最后，我用自己造出来的组件搭起了一个 tiny LLM。
```

这就是组件系统的核心价值：**不是教玩家记住 LLM 的结构，而是让玩家亲手把 LLM 从最小计算单元一步步搭出来。**
