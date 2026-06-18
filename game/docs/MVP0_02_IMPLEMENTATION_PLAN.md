# MVP0.02 实现计划：从关卡表单到 Model Board

**状态:** Proposed  
**日期:** 2026-06-18  
**依据:** `game/docs/LLM_Complete_GDD.md`, `game/docs/TECH_DESIGN.md`, MVP0.01 已验证实现

## 1. 结论

新的 GDD 方向成立，而且比 MVP0.01 更有可玩性。核心原因是它把游戏从“选择正确配置/观察模拟反馈”推进到了真正的 **Build-Test-Debug**：

```text
放置模块 -> 连接端口 -> 推导 shape -> 执行 tiny tensor -> 跑测试 -> 看 trace -> 修图
```

这更接近《Turing Complete》的乐趣：玩家不是回答题，而是在构造一个能运行、能被测试器验证的系统。

但完整 GDD 范围很大。下一步不应直接做完整 Transformer Block、autograd、训练、3D 展开和沙盒。MVP0.02 应只做一个可扩展的最小 Graph Workbench，证明三件事：

1. 玩家能在图上搭计算。
2. 系统能做 shape / dtype / 数值测试。
3. 失败反馈能定位到具体节点、端口、shape 或 tensor cell。

## 2. MVP0.01 现状评估

MVP0.01 已经验证了：

- Web-first 技术路线可运行。
- FastAPI + React + JSON 内容管线成立。
- mission/progress/artifact/arena 的基本 API 边界可用。
- 失败报告、训练模拟、sample-output fallback 能形成学习闭环。
- 本地 `.venv`、pnpm、测试、build 流程可用。

MVP0.01 的不足也很清楚：

- 玩法仍以表单、按钮、网格选择为主，像“交互教程”，还不像构建游戏。
- 关卡判定主要是规则判断，不是执行玩家构建的计算图。
- 没有端口、shape、dtype、trace timeline、reference 数值对比。
- “模块解锁”只是文本状态，尚未变成玩家实际可复用的组件。

所以 MVP0.02 的目标不是新增更多 SFT/DPO 内容，而是补上 GDD 中真正的核心系统。

## 3. 可玩性判断

### 3.1 最强可玩点

| 设计点 | 可玩性 | 原因 |
|---|---|---|
| Model Board | 高 | 玩家有明确构造空间，能形成试错和优化 |
| Shape/端口系统 | 高 | 错误具体、反馈快，适合谜题化 |
| Trace Mode | 高 | 失败后有侦探式诊断过程 |
| Autograder | 高 | 公开测试/隐藏测试天然形成挑战 |
| 子图封装 | 中高 | 后续能产生“我造了一个模块”的成长感 |
| 训练模拟 | 中 | 适合后期，不是当前最强核心 |
| Arena 对比 | 中 | 有教学价值，但构建感弱 |

### 3.2 当前最大可玩性风险

| 风险 | 表现 | 应对 |
|---|---|---|
| 数学前置太重 | 玩家一上来看到 `[B,H,T,D]` 卡住 | v0.02 从 concrete shape 开始，再显示符号 shape |
| 图编辑器成本膨胀 | 拖拽/连线/布局做很久但测试不强 | 先用固定节点布局 + 可连线，少做自由画布 |
| Trace 过载 | 数值表太多，界面像调试器 | 默认显示 shape + 少量 sample values，可展开详细值 |
| 关卡缺少目标感 | 玩家不知道为什么连这些模块 | 每关必须有输入、目标 output、公开测试和失败样例 |
| 过早做 3D | 美术/交互成本压过系统 | v0.02 只做 2D graph + heatmap，3D 延后 |

## 4. 可行性判断

### 4.1 技术可行性

可行，但要分层实现。

建议不要在 v0.02 做完整 PyTorch 训练，也不要做 autograd。先实现一个小型 deterministic TinyTensor executor：

- 支持小 tensor。
- 支持 shape inference。
- 支持 forward execution。
- 支持 trace capture。
- 支持 reference comparison。
- 不支持 backward。
- 不支持 optimizer。

这足够覆盖：

- MatMul Gate。
- QK Transpose。
- Causal Mask。
- Softmax row sum。
- Weighted V。

### 4.2 UI 可行性

前端建议引入 `@xyflow/react` 作为 graph canvas。理由：

- 节点拖拽、边连接、缩放、选择、mini map 都是现成能力。
- 我们可以把精力放在 LLM-specific 的 shape/trace/autograder，而不是自研画布。
- 后续支持子图封装和复杂 Model Board 更自然。

v0.02 不需要复杂 3D，也不需要 PixiJS。

### 4.3 内容可行性

GDD 中建议的 Transformer Block vertical slice 很好，但对第一轮实现仍偏大。v0.02 只做 3 个 graph levels：

1. **MatMul Gate**：理解矩阵乘法输出 shape。
2. **QK Transpose**：理解 attention score 必须是 `[B,H,T,T]`。
3. **Causal Mask Graph**：理解 mask 必须在 softmax 前，非法位置概率为 0。

这 3 关能覆盖 GDD 的核心承诺：

- module/port。
- shape inference。
- graph execution。
- numerical/reference test。
- trace/error localization。

## 5. MVP0.02 范围

### 5.1 包含

- Graph level 内容格式。
- Module registry。
- TinyTensor executor。
- Shape inference。
- Public tests。
- Trace report。
- Graph submit API。
- React Model Board 页面。
- Inspector：Summary / Shape / Values / Tests。
- 3 个 graph levels。

### 5.2 不包含

- Autograd。
- Optimizer。
- 完整 Transformer Block。
- 3D tensor inspector。
- 沙盒训练。
- 子图封装。
- 隐藏测试。
- 真实 checkpoint 推理改造。

这些内容进入 MVP0.03 或后续。

## 6. 推荐实现架构

### 6.1 后端新增目录

```text
game/
  graph_levels/
    mvp0_02.json
  modules/
    registry.json
  server/
    api/
      graph.py
    core/
      graph_runtime.py
      module_registry.py
      tiny_tensor.py
      graph_tests.py
```

### 6.2 前端新增目录

```text
game/app/src/
  graph/
    GraphWorkbench.tsx
    GraphCanvas.tsx
    ModulePalette.tsx
    Inspector.tsx
    TestReport.tsx
    TraceTimeline.tsx
    graphTypes.ts
```

### 6.3 API

```http
GET /api/graph/modules
GET /api/graph/levels
GET /api/graph/levels/{level_id}
POST /api/graph/levels/{level_id}/submit
```

`submit` 请求：

```json
{
  "nodes": [
    { "id": "q", "module_id": "InputTensor", "params": { "name": "Q" } },
    { "id": "kt", "module_id": "Transpose", "params": { "dim0": -2, "dim1": -1 } },
    { "id": "matmul", "module_id": "MatMul", "params": {} }
  ],
  "edges": [
    { "from": "q.out", "to": "matmul.left" },
    { "from": "k.out", "to": "kt.x" },
    { "from": "kt.y", "to": "matmul.right" }
  ],
  "output_node": "matmul"
}
```

`submit` 响应：

```json
{
  "passed": false,
  "tests": [
    {
      "id": "score_shape",
      "type": "shape",
      "passed": false,
      "message": "Expected [1,2,4,4], got [1,2,4,3]"
    }
  ],
  "trace": [
    {
      "node_id": "matmul",
      "module_id": "MatMul",
      "inputs": {
        "left": { "shape": [1,2,4,3], "dtype": "float" },
        "right": { "shape": [1,2,4,3], "dtype": "float" }
      },
      "outputs": {
        "out": { "shape": null, "dtype": null }
      },
      "error": "MatMul axis mismatch"
    }
  ],
  "incident_report": {
    "title": "K needs transpose(-2, -1)",
    "node_id": "matmul",
    "port": "right",
    "hints": ["Insert Transpose before MatMul."]
  }
}
```

## 7. Module Registry v0.02

最小模块集：

| Module | Inputs | Outputs | Params | Tests enabled |
|---|---|---|---|---|
| InputTensor | - | out | fixture_name | shape/dtype |
| Transpose | x | y | dim0, dim1 | shape/numeric |
| MatMul | left, right | out | - | shape/numeric |
| Scale | x | y | factor | numeric |
| CausalMask | - | mask | T, direction | mask |
| MaskApply | scores, mask | masked | fill_value | mask/numeric |
| Softmax | x | y | dim | row_sum/numeric |

暂不实现：

- Linear。
- Embedding。
- LayerNorm。
- MLP。
- Backward。
- Optimizer。

## 8. Graph Level 设计

### 8.1 Level 1：MatMul Gate

目标：

```text
A[2,3] @ B[3,4] -> out[2,4]
```

玩家任务：

- 连接 A 和 B 到 MatMul。
- 选择 MatMul 输出作为提交目标。

失败反馈：

- 未连接输入。
- 连接顺序反了。
- shape mismatch。

通过测试：

- output shape `[2,4]`。
- output numeric close to reference。

### 8.2 Level 2：QK Transpose

目标：

```text
Q[1,2,4,3] @ K.transpose(-2,-1)[1,2,3,4] -> scores[1,2,4,4]
```

玩家任务：

- 插入 Transpose。
- 设置 dim0=-2, dim1=-1。
- 将结果接入 MatMul right。

失败反馈：

- 直接 Q @ K：axis mismatch。
- transpose 维度错：输出 shape 不对。

通过测试：

- scores shape `[1,2,4,4]`。
- numeric close。

### 8.3 Level 3：Causal Mask Graph

目标：

```text
scores -> MaskApply(causal lower triangle, fill=-1e9) -> Softmax(dim=-1)
```

玩家任务：

- 生成 causal mask。
- 在 softmax 前应用 mask。
- softmax 行归一。

失败反馈：

- mask 方向反。
- mask 放在 softmax 后。
- fill value 不够小，future weight 非 0。

通过测试：

- future attention weight <= 1e-6。
- every row sum close to 1。
- numeric close to reference。

## 9. 前端交互要求

### 9.1 Model Board

v0.02 只需要：

- 左侧 Module Palette。
- 中央 Graph Canvas。
- 右侧 Inspector。
- 底部 Test Report / Trace。

节点状态：

- idle。
- selected。
- shape-ok。
- warning。
- error。
- passed。

边状态：

- valid type/shape。
- unknown。
- invalid。

### 9.2 Inspector

先做三个 tab：

| Tab | v0.02 内容 |
|---|---|
| Summary | 模块说明、输入输出端口 |
| Shape | 当前输入输出 shape/dtype |
| Tests | 相关测试结果和错误定位 |

Values tab 可先作为 trace table，不做复杂 tensor viewer。

### 9.3 Trace

v0.02 Trace 显示：

- node id。
- module id。
- input shapes。
- output shapes。
- sample values 前 2-4 个。
- node error。

不做逐步动画，先做可点击列表。

## 10. 后端实现步骤

1. 定义 `registry.json`。
2. 定义 `graph_levels/mvp0_02.json`。
3. 实现 `TinyTensor`：
   - dtype。
   - shape。
   - values nested/list 或 flat list。
   - small helper: transpose, matmul, scale, mask_apply, softmax。
4. 实现 graph topological execution。
5. 实现 tests：
   - shape。
   - numeric_close。
   - row_sum。
   - illegal_attention。
6. 实现 trace。
7. 实现 API。
8. 加 pytest。

## 11. 前端实现步骤

1. 添加 `@xyflow/react`。
2. 增加 Graph route/view。
3. Module Palette 从 API 读取 registry。
4. Graph Canvas 加载 level 初始图。
5. 支持添加模块和连线。
6. 节点可编辑 params。
7. Submit 后渲染 test report。
8. Inspector 显示 selected node trace。
9. Pipeline 增加入口：`Graph Slice`。

## 12. 验收标准

MVP0.02 通过标准：

- 玩家可以在浏览器里完成 MatMul Gate。
- 玩家可以在 QK Transpose 关卡里因为未转置 K 失败，并看到具体 shape 错误。
- 玩家可以在 Causal Mask Graph 关卡里因为 mask 放错位置或方向错误失败，并看到非法 attention cell。
- 后端测试覆盖 3 个关卡的成功和失败路径。
- 前端 build/typecheck 通过。
- 不破坏 MVP0.01 的 8 个原有关卡和 Arena。

## 13. 风险控制

### 13.1 图编辑器复杂度

风险：React Flow 集成、节点 UI、连线、参数编辑一起做容易超范围。

控制：

- 第一版允许固定初始节点位置。
- 只支持必要模块。
- 不做自动布局。
- 不做节点分组。

### 13.2 TinyTensor 范围膨胀

风险：很快想做 embedding、linear、layernorm、backward。

控制：

- v0.02 只支持 attention 前半段必要 ops。
- autograd 明确不做。
- Tensor size 只允许小 fixture。

### 13.3 UI 信息过载

风险：shape、values、tests、trace 同时出现，玩家看不懂。

控制：

- 默认只显示 selected node 的 shape。
- Test report 优先显示第一条失败。
- Values 默认折叠。

## 14. 推荐开发顺序

优先顺序：

1. 后端 TinyTensor + graph tests。
2. API smoke。
3. React Flow 最小画布。
4. MatMul Gate 前端闭环。
5. QK Transpose。
6. Causal Mask Graph。
7. Inspector/Trace 打磨。
8. README 和测试补齐。

不要先做视觉大改或 3D 展示。先把“搭图能过测试”的手感跑通。

## 15. 与当前 MVP0.01 的关系

MVP0.01 保留为 `Campaign / Legacy Missions` 或 `Foundations`，不要删除。它提供：

- 数据、token、SFT、Arena 的教学内容。
- Progress/artifact/arena 的现成基础设施。

MVP0.02 新增 `Graph Slice`，作为真正核心玩法的第一版。后续可以逐步把 MVP0.01 中的关卡迁移为 graph-based levels。

