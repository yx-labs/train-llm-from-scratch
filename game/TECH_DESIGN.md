# LLM 学习游戏技术设计

**Status:** Draft  
**Date:** 2026-06-18  
**Scope:** `game/` 目录内的游戏化学习产品  
**Related:** `game/GAME_DESIGN.md`

## 1. 设计结论

MVP 推荐采用 **Web-first 技术路线**，不使用 Unity/Godot/Unreal 作为第一阶段主引擎。

推荐栈：

```text
Frontend: Vite + React + TypeScript
Rendering: SVG + Canvas, 后续按需引入 PixiJS
Backend: FastAPI
Simulation: Python rule engine
Storage: SQLite + local JSON manifests
Inference: Adapter 调用现有 PyTorch 代码
Packaging: 本地 Web 服务，后续可用 Tauri/Electron 打包
```

核心理由：

- 本项目的核心是教学交互、图表、调试面板、模型对比和训练模拟，不是物理、3D、角色动作或大地图探索。
- Web 技术更适合做高密度信息 UI、可视化、表单、图表、日志、Arena 对比和本地工具集成。
- FastAPI/Python 能直接复用当前仓库训练、评测、推理代码，减少跨语言桥接成本。
- SVG/Canvas 足够覆盖 token、attention、mask、loss curve、reward/KL 仪表等关键表现。
- 传统游戏引擎会增加资产管线、构建、Python 集成和 UI 复杂度，MVP 收益不明显。

传统游戏引擎不是永久否定。若后续产品走 Steam/独立游戏发行、强剧情探索、复杂动画或跨平台离线包，再重新评估 Godot 或 Unity。

## 2. 需求与约束

### 2.1 功能需求

游戏系统需要支持：

- Pipeline 地图和关卡解锁。
- 关卡实验台，包括可操作画布、配置面板、实时反馈。
- 训练模拟，包括 loss、reward、KL、accuracy、显存、时间等曲线。
- 失败诊断，包括事故报告、提示分级、失败样例。
- 模型 Arena，包括 base/SFT/DPO/PPO/GRPO 等 checkpoint 对比。
- 成果物管理，包括 checkpoint、tokenizer、config、eval report、sample outputs。
- 与现有训练/推理代码集成。

### 2.2 非功能需求

- MVP 本地可运行，不依赖云服务。
- 基础关卡不需要 GPU。
- 推理体验可以依赖本地 CPU/GPU，但必须能优雅降级。
- 关卡内容数据化，避免写死在 UI 中。
- 所有游戏相关代码和文档位于 `game/`。
- 不破坏现有训练项目结构。

### 2.3 团队与研发假设

当前更像小团队/个人研发模式，技术选择应优先考虑：

- 开发速度。
- 可维护性。
- 与 Python 训练代码集成成本。
- UI 表现力。
- 后续扩展到 Web/桌面分发的可能性。

## 3. 引擎选择

这里的“引擎”拆成四类：

1. **表现引擎**：画面、交互、动画、UI。
2. **任务引擎**：关卡加载、判定、解锁、进度。
3. **模拟引擎**：训练曲线、失败规则、资源消耗。
4. **模型引擎**：真实 checkpoint 推理、评测、reward 打分。

不要把所有问题都交给 Unity/Godot 这类游戏引擎解决。LLM 学习游戏的复杂度主要在任务系统、数据可视化和 Python 模型集成。

## 4. 表现引擎方案比较

### 4.1 方案 A：继续基于 Streamlit

| 维度 | 评估 |
|---|---|
| 开发速度 | 高 |
| 复杂交互 | 低 |
| 动画表现 | 低 |
| 数据/训练面板 | 高 |
| 游戏感 | 低 |
| 现有项目集成 | 高 |

优点：

- 当前仓库已有 Streamlit 控制台。
- 很适合训练配置、日志、指标、admin/debug 页面。
- Python 直连成本低。

缺点：

- 不适合图灵完备式的拖拽、连线、格子、动画反馈。
- 页面结构和交互状态控制受限。
- 游戏感弱，容易变成“带教程的控制台”。

结论：

- 不作为主游戏 UI。
- 可保留为开发者控制台或训练资产管理后台。

### 4.2 方案 B：Web App，React + SVG/Canvas

| 维度 | 评估 |
|---|---|
| 开发速度 | 高 |
| 复杂交互 | 高 |
| 动画表现 | 中高 |
| 数据/训练面板 | 高 |
| 游戏感 | 中高 |
| 现有项目集成 | 高 |

优点：

- 高密度 UI、图表、表单、日志、对比面板都很成熟。
- SVG 适合 token、mask、attention heatmap、模块连线等教学可视化。
- Canvas 适合动态流线、粒子、训练过程动画。
- 和 FastAPI/Python 集成自然。
- 可以先 Web 本地运行，后续用 Tauri/Electron 打包桌面。

缺点：

- 需要自己设计任务系统和渲染抽象。
- 游戏引擎内置的场景管理、动画编辑器、资源管理较少。
- 复杂动画多了以后需要引入 PixiJS 或自建渲染层约束。

结论：

- **MVP 推荐方案。**

### 4.3 方案 C：Web App + PixiJS

| 维度 | 评估 |
|---|---|
| 开发速度 | 中 |
| 复杂交互 | 高 |
| 动画表现 | 高 |
| 数据/训练面板 | 高 |
| 游戏感 | 高 |
| 现有项目集成 | 高 |

优点：

- 比纯 Canvas 更适合 2D 游戏式场景、动画、精灵、粒子。
- 仍然可以和 React UI 共存。
- 适合后续做“实验室地图”“模块装配”“token 流动画”。

缺点：

- 比 SVG/Canvas 多一层学习和架构成本。
- 对 MVP 的前几关不是必须。
- React 状态和 Pixi 场景状态需要清楚分层。

结论：

- MVP 初期不强制引入。
- 当关卡画布动画明显复杂时再引入。

### 4.4 方案 D：Godot

| 维度 | 评估 |
|---|---|
| 开发速度 | 中 |
| 复杂交互 | 高 |
| 动画表现 | 高 |
| 数据/训练面板 | 中 |
| 游戏感 | 高 |
| 现有项目集成 | 中低 |

优点：

- 开源、轻量、适合 2D。
- 游戏场景、动画、资源管理比 Web 原生完整。
- 更像传统独立游戏。

缺点：

- 与 Python/PyTorch 推理服务仍需 HTTP/WebSocket 桥接。
- 高密度数据 UI 和图表不是强项。
- 团队需要维护游戏工程和 Python 服务两套生态。

结论：

- 如果未来目标转向独立游戏发行，可以重新评估。
- MVP 不建议选。

### 4.5 方案 E：Unity

| 维度 | 评估 |
|---|---|
| 开发速度 | 中低 |
| 复杂交互 | 高 |
| 动画表现 | 高 |
| 数据/训练面板 | 中 |
| 游戏感 | 高 |
| 现有项目集成 | 低 |

优点：

- 生态成熟。
- 发行、动画、资产工具强。
- 如果后续做大型商业游戏，有人才和工具优势。

缺点：

- 对当前教育工具类 MVP 过重。
- C#/Python/PyTorch 跨栈成本高。
- UI 和数据可视化开发效率不如 Web。
- 构建和资产流程增加维护成本。

结论：

- 不适合作为第一阶段主引擎。

### 4.6 方案 F：Unreal

| 维度 | 评估 |
|---|---|
| 开发速度 | 低 |
| 复杂交互 | 高 |
| 动画表现 | 很高 |
| 数据/训练面板 | 中低 |
| 游戏感 | 很高 |
| 现有项目集成 | 低 |

结论：

- 当前需求明显不匹配。
- 不建议使用。

## 5. 最终推荐

### 5.1 MVP 推荐架构

```text
Browser
  React App
    Pipeline Map
    Mission Workbench
    Simulation Console
    Model Arena
    Notebook
    SVG/Canvas Renderers
      Tokenizer Renderer
      Attention Renderer
      Mask Grid Renderer
      Curve Renderer
      Reward/KL Renderer

Local API
  FastAPI
    Mission API
    Simulation API
    Artifact API
    Inference API
    Progress API

Python Core
  Mission Runtime
  Rule-based Simulator
  Artifact Registry
  Inference Adapter
  Existing repo scripts/src

Storage
  game/missions/*.json
  game/artifacts/manifest.json
  game/progress/local.sqlite
  checkpoints/tokenizers/eval reports
```

### 5.2 技术分工

| 层 | 技术 | 职责 |
|---|---|---|
| UI Shell | React + TypeScript | 页面、状态、路由、交互 |
| 可视化 | SVG/Canvas | token、attention、mask、曲线 |
| 动画增强 | PixiJS，按需 | token 流、粒子、场景过渡 |
| API | FastAPI | 前后端边界 |
| 任务运行 | Python | 关卡判定、解锁、进度 |
| 模拟 | Python rules | 训练曲线和失败报告 |
| 推理 | PyTorch adapter | 调用真实 checkpoint |
| 存储 | SQLite + JSON | 本地进度和内容定义 |

### 5.3 为什么不是传统游戏引擎

本产品主要交互对象是：

- 表格。
- 图表。
- 代码/配置。
- 可视化调试器。
- 多模型输出对比。
- 训练指标。
- 失败报告。

这些都是 Web 的强项。传统游戏引擎的强项是：

- 物理。
- 摄像机。
- 角色。
- 大场景。
- 动画编辑。
- 资源打包。

当前 MVP 不需要这些能力。过早引入会显著增加技术债。

## 6. 前端设计

### 6.1 页面结构

```text
/                         Pipeline Map
/mission/:missionId       Mission Workbench
/arena                    Model Arena
/artifacts                Artifact Gallery
/notebook                 Concept Notebook
/settings                 Runtime Settings
```

### 6.2 组件结构

```text
GameApp
  AppShell
    TopStatusBar
    PipelineSidebar
    RouteContent

MissionWorkbench
  MissionHeader
  ObjectivePanel
  InteractiveCanvas
  ControlPanel
  MetricsPanel
  HintPanel
  IncidentReportModal

Renderers
  TokenStreamRenderer
  TokenMergeBoard
  AttentionHeatmap
  MaskGrid
  TransformerBlockDiagram
  TrainingCurveChart
  RewardScale
  GroupResponseBoard

Arena
  PromptEditor
  ModelSelector
  GenerationControls
  OutputComparisonGrid
  EvalSummaryPanel
```

### 6.3 状态管理

MVP 不需要复杂全局状态库。推荐：

- React Query：API 数据、缓存、加载状态。
- Zustand：本地 UI 状态、当前关卡操作草稿。
- URL params：当前关卡、当前 arena 配置。

状态分类：

| 状态 | 存放位置 |
|---|---|
| 关卡定义 | API + React Query |
| 玩家进度 | API + SQLite |
| 当前画布编辑 | Zustand/local component |
| 模拟运行结果 | API + React Query |
| Arena prompt 草稿 | Zustand/local storage |
| 推理任务状态 | API polling 或 WebSocket |

### 6.4 渲染策略

| 内容 | 推荐渲染 |
|---|---|
| token tile、mask grid | SVG |
| attention heatmap | SVG 或 Canvas |
| token 流动画 | Canvas，后续 PixiJS |
| loss/reward 曲线 | SVG chart |
| pipeline map | SVG |
| 大量粒子/流动效果 | PixiJS |
| 文档说明 | Markdown renderer |

原则：

- 需要可点击、可解释、可无障碍标注的内容优先 SVG。
- 需要大量动态对象的内容优先 Canvas/PixiJS。
- 不在 MVP 中使用 WebGL，除非要做 3D embedding 空间。

## 7. 后端设计

### 7.1 FastAPI 服务边界

后端只做本地服务，不在 MVP 中引入多用户认证。

主要模块：

```text
game/server/
  main.py
  api/
    missions.py
    progress.py
    simulate.py
    artifacts.py
    arena.py
  core/
    mission_runtime.py
    simulator.py
    artifact_registry.py
    inference_adapter.py
    progress_store.py
  schemas/
    mission.py
    simulation.py
    artifact.py
```

### 7.2 API 草案

#### Missions

```http
GET /api/missions
GET /api/missions/{id}
POST /api/missions/{id}/submit
```

`submit` 输入玩家方案，输出：

- pass/fail。
- unlocked modules。
- unlocked artifacts。
- metrics。
- incident report。
- next mission。

#### Simulation

```http
POST /api/simulate/train
POST /api/simulate/failure-preview
```

用于训练模拟和失败报告。

#### Artifacts

```http
GET /api/artifacts
GET /api/artifacts/{id}
GET /api/artifacts/{id}/eval
```

用于读取 checkpoint manifest、评测报告、样例输出。

#### Arena

```http
POST /api/arena/generate
POST /api/arena/compare
POST /api/arena/reward-score
```

`compare` 用于同一 prompt 对比多个模型。

#### Progress

```http
GET /api/progress
POST /api/progress/reset
POST /api/progress/unlock
```

MVP 只支持本地单玩家。

### 7.3 推理任务处理

推理可能慢于普通 API 请求。MVP 可先同步执行，小模型足够快时问题不大。

后续升级：

```text
POST /api/arena/generate -> job_id
GET /api/jobs/{job_id}
GET /api/jobs/{job_id}/stream
```

如果需要 token streaming，可用 Server-Sent Events 或 WebSocket。

## 8. 模拟引擎设计

### 8.1 模拟目标

模拟引擎不是为了数值精确，而是为了真实地表现工程规律：

- 学习率过大导致 loss 爆炸。
- 学习率过小导致预算内学不动。
- batch 太小导致曲线抖动。
- causal mask 泄漏导致假成功。
- SFT mask 错导致模型复读 prompt。
- KL 太弱导致 reward hacking。
- KL 太强导致 RL 学不动。

### 8.2 输入

```json
{
  "mission_id": "07_01_pretrain_sim",
  "stage": "pretrain",
  "config": {
    "learning_rate": 0.0003,
    "batch_size": 16,
    "context_length": 256,
    "causal_mask": true,
    "data_quality": 0.82
  },
  "player_state": {
    "unlocked_modules": ["tokenizer.basic", "attention.causal_mask"]
  }
}
```

### 8.3 输出

```json
{
  "status": "failed",
  "metrics": {
    "train_loss": [4.8, 3.7, 2.1, 8.4, 12.0],
    "val_loss": [4.9, 3.9, 2.5, 9.1, 13.2],
    "gpu_memory_mb": [1200, 1350, 1350, 1350, 1350]
  },
  "events": [
    {
      "type": "loss_explosion",
      "step": 320,
      "severity": "error"
    }
  ],
  "incident_report": {
    "title": "学习率过高导致训练发散",
    "symptoms": [
      "loss 在短暂下降后快速上升",
      "梯度仪表进入红区"
    ],
    "suspected_modules": ["optimizer.learning_rate"],
    "hints": [
      "学习率控制每次参数更新的步长。",
      "过大的步长可能越过最优区域并导致发散。"
    ]
  }
}
```

### 8.4 规则层

推荐用 JSON 定义失败规则，用 Python 执行：

```json
{
  "id": "learning_rate_too_high",
  "stage": ["pretrain", "sft"],
  "when": {
    "learning_rate_gt": 0.001
  },
  "effects": {
    "curve_template": "loss_explosion",
    "incident": "learning_rate_too_high"
  }
}
```

规则优先级：

1. 硬错误：shape 错、OOM、缺失必要模块。
2. 假成功：数据泄漏、mask 泄漏。
3. 训练不稳：lr、batch、grad clipping。
4. 质量不足：数据质量、模型容量、训练步数。

### 8.5 曲线模板

MVP 准备这些曲线模板：

| 模板 | 用途 |
|---|---|
| `healthy_decay` | 正常收敛 |
| `slow_decay` | 学习率低或容量不足 |
| `noisy_decay` | batch 太小 |
| `loss_explosion` | 学习率过大 |
| `flatline` | 没学到信号 |
| `overfit` | train 降、val 升 |
| `suspicious_fast_drop` | 泄漏假成功 |
| `reward_hacking` | reward 升、human score 降 |
| `kl_explosion` | PPO 偏离参考模型 |

曲线可以由参数化函数生成，不需要存大量静态数据。

## 9. 任务引擎设计

### 9.1 Mission schema 核心字段

```json
{
  "id": "04_02_causal_mask",
  "chapter": "Attention 实验室",
  "title": "禁止偷看未来",
  "type": "interactive_grid",
  "learning_objectives": [],
  "prerequisites": [],
  "unlocks": [],
  "inputs": {},
  "controls": {},
  "success_conditions": [],
  "failure_rules": [],
  "hints": [],
  "post_success": {}
}
```

### 9.2 判定模型

关卡判定应支持三类条件：

| 类型 | 示例 |
|---|---|
| 结构判定 | mask 是否为下三角 |
| 数值判定 | val loss 是否低于目标 |
| 行为判定 | 输出是否包含 `<answer>` 且答案正确 |

### 9.3 解锁模型

```text
Mission complete
  -> unlock module
  -> unlock artifact
  -> append notebook concept
  -> recommend next mission
```

### 9.4 本地进度

SQLite 表：

```sql
players(id, created_at, updated_at)
mission_progress(player_id, mission_id, status, attempts, best_score, updated_at)
unlocked_modules(player_id, module_id, unlocked_at)
unlocked_artifacts(player_id, artifact_id, unlocked_at)
notebook_entries(player_id, concept_id, status, unlocked_at)
```

## 10. 成果物系统

### 10.1 成果物类型

| 类型 | 示例 |
|---|---|
| checkpoint | `pretrain_good.pt`, `sft.pt` |
| tokenizer | tokenizer files |
| config | training/inference config |
| eval_report | GSM8K 或自定义 eval |
| sample_outputs | 固定 prompt 输出 |
| training_log | jsonl metrics |
| failure_case | bad model sample |

### 10.2 Artifact manifest

```json
{
  "id": "sft",
  "stage": "sft",
  "kind": "checkpoint",
  "display_name": "SFT 指令模型",
  "paths": {
    "checkpoint": "artifacts/checkpoints/sft.pt",
    "config": "artifacts/configs/sft.json",
    "tokenizer": "artifacts/tokenizers/base",
    "eval": "artifacts/eval/sft.json"
  },
  "capabilities": ["chat_template", "instruction_following"],
  "unlocked_by": "09_02_sft_mask",
  "fallback": {
    "mode": "sample_outputs",
    "path": "artifacts/samples/sft_outputs.jsonl"
  }
}
```

### 10.3 缺少 checkpoint 时的降级

必须支持没有真实模型文件的情况：

1. 优先使用真实 checkpoint 推理。
2. 如果 checkpoint 不存在，使用 sample outputs。
3. 如果 sample outputs 不存在，显示资产缺失但不阻塞关卡。

这样可以先开发游戏流程，再逐步补齐模型资产。

## 11. 美术方案

### 11.1 美术定位

推荐方向：**抽象化 LLM 实验室 + 工业控制台 + 数据流可视化**。

不要走：

- 拟人化 AI 助手。
- 科幻大场景堆料。
- 过度写实服务器机房。
- 大量装饰性插画。

原因：

- 本游戏的主要对象是机制和反馈。
- 画面应服务于理解 token、attention、loss、reward 等抽象概念。
- 太强的世界观和装饰会降低信息密度。

### 11.2 视觉关键词

- 精密。
- 可调试。
- 清晰。
- 模块化。
- 工程感。
- 抽象实验。
- 数据流动。

### 11.3 视觉隐喻

| LLM 概念 | 美术隐喻 |
|---|---|
| raw text | 原料矿石/文本碎片 |
| token | 标准化零件/芯片块 |
| context window | 传送带/固定长度轨道 |
| embedding | 向量插槽/坐标舱 |
| attention | 光束连线/权重热力网 |
| causal mask | 单向闸门/遮罩墙 |
| transformer block | 可插拔处理模块 |
| loss | 压力表/误差仪 |
| optimizer | 校准旋钮 |
| checkpoint | 冷冻样本/模型胶囊 |
| SFT mask | 高亮的训练责任区 |
| reward model | 天平/评分仪 |
| KL | 安全绳/偏移警戒线 |
| GRPO group | 多候选竞技台 |
| evaluation | 检测台/质检报告 |

### 11.4 色彩系统

延续现有文档的语义色彩，但做 UI 级规范：

| 语义 | 颜色建议 | 用途 |
|---|---|---|
| Data | green | 数据、语料、token 输入 |
| Preprocess | blue | 清洗、转换、packing |
| Model | amber/yellow | Transformer、checkpoint |
| Loss/Error | red | loss、错误、事故报告 |
| Alignment/RL | orange | reward、PPO、GRPO |
| Evaluation | purple | eval、benchmark、arena |
| Neutral | gray | disabled、checkpoint、背景线 |

注意：

- 不要做单一蓝紫渐变主题。
- 背景应克制，避免影响热力图和指标判断。
- 错误红只用于真正异常，避免视觉疲劳。
- 所有颜色需要配合形状/图标，不只依赖颜色区分。

### 11.5 UI 密度

这是学习工具，不是营销落地页。界面应保持：

- 高信息密度。
- 清楚分区。
- 控件稳定。
- 动画不抢操作焦点。
- 文本短而具体。

每个关卡画面优先回答：

1. 当前目标是什么。
2. 我能操作什么。
3. 系统现在发生了什么。
4. 为什么成功或失败。

### 11.6 核心场景美术

#### Pipeline 地图

表现：

- 横向或纵向模块链路。
- 每个 stage 是一个大型节点。
- 节点内显示关卡完成度、解锁资产和当前状态。

避免：

- 复杂开放世界地图。
- 无意义路径装饰。

#### 关卡实验台

表现：

- 中央是可操作画布。
- 左侧是目标和模块库。
- 右侧是指标。
- 底部是运行和事故报告。

#### 训练模拟控制台

表现：

- 曲线图、仪表、事件时间线。
- 训练事件以小标签落在曲线上。
- 异常点可点击展开。

#### 模型 Arena

表现：

- 多模型列对比。
- 同 prompt 输出分栏。
- 差异高亮。
- 指标摘要紧贴输出。

### 11.7 动效原则

动效只服务三件事：

- 表示数据流动。
- 表示因果关系。
- 表示异常发生。

推荐动效：

- token 沿传送带进入 context window。
- attention 连线逐渐亮起。
- mask 生效时未来区域暗下去。
- loss 曲线随训练推进绘制。
- reward hacking 时 reward 曲线和 human score 分叉。

避免：

- 循环装饰动画。
- 大面积发光背景。
- 无功能粒子。
- 影响读数的抖动。

### 11.8 字体与图标

推荐：

- UI 字体使用系统 sans-serif。
- 数字和代码使用 monospace。
- 图标使用统一线性图标库，Web 端可用 lucide-react。

图标用途：

- 运行、暂停、重试、提示、查看报告、解锁、警告、checkpoint、评测。

不要用图标替代关键概念名。专业概念第一次出现时必须有文字。

### 11.9 资产生产方式

MVP 尽量使用程序化和矢量资产：

- token tile：CSS/SVG。
- attention heatmap：SVG/Canvas。
- pipeline node：SVG/HTML。
- icon：lucide。
- 背景网格：CSS。
- checkpoint capsule：SVG component。

暂不需要大量手绘图。如果需要封面图或宣传图，再单独用插画或生成图。

### 11.10 美术验收标准

每个界面验收：

- 玩家一眼能识别当前可操作对象。
- 指标和文字不被装饰遮挡。
- 成功、警告、失败状态有明显区别。
- token、attention、loss、reward 的视觉语言一致。
- 截图静止状态也能表达玩法。

## 12. 交互方案

### 12.1 Tokenizer 关卡

交互：

- 文本被切成字符/子词方块。
- 玩家点击相邻 token 合并。
- 右侧显示 vocab size、token count、compression ratio。

失败：

- vocab 太小：context 被占满。
- vocab 太大：泛化提示下降。

### 12.2 Causal Mask 关卡

交互：

- 网格表示 query position x key position。
- 玩家涂抹可见区域。
- 运行后 token 尝试预测下一个 token。

成功：

- mask 为下三角。

失败：

- 上三角开放触发 future leak 假成功报告。

### 12.3 Pretrain 模拟关卡

交互：

- 玩家调整 learning rate、batch size、context length、data quality。
- 点击 run 后显示训练曲线。

成功：

- val loss 达标，且未触发泄漏/过拟合。

失败：

- lr 爆炸、过拟合、OOM、训练预算耗尽。

### 12.4 SFT Mask 关卡

交互：

- 展示 chat sample。
- 玩家标记 assistant tokens 参与 loss。
- user/system tokens 默认灰色。

成功：

- 只有 assistant response 区域参与 loss。

失败：

- 全文参与 loss 后模型复读 prompt。

### 12.5 Arena 对比

交互：

- 选择两个或多个模型。
- 输入 prompt。
- 调 temperature/top-p。
- 并排生成结果。

核心：

- 让玩家感受阶段差异。
- 不是聊天产品，而是教学对比工具。

## 13. 运行与部署

### 13.1 本地开发

建议命令：

```bash
# backend
python -m game.server

# frontend
cd game/app
npm install
npm run dev
```

后续可做根目录脚本：

```bash
python -m game.dev
```

同时启动前后端。

### 13.2 本地资产

默认路径：

```text
game/artifacts/
  manifest.json
  checkpoints/
  tokenizers/
  configs/
  eval/
  samples/
```

大型 checkpoint 不建议直接提交 git。manifest 可以记录：

- local path。
- download URL。
- checksum。
- fallback sample outputs。

### 13.3 桌面打包

如果 MVP 成功，桌面化优先考虑：

1. Tauri：轻量，适合本地 Web UI + Python sidecar。
2. Electron：生态成熟，但体积更大。
3. Godot/Unity：仅当产品形态转为传统游戏时重新评估。

## 14. 质量与测试

### 14.1 前端测试

- Mission renderer smoke tests。
- 关键交互组件单测。
- Playwright 跑主线流程。
- 截图检查关键画布非空、无重叠。

### 14.2 后端测试

- Mission schema validation。
- Failure rule trigger tests。
- Simulator deterministic tests。
- Artifact manifest validation。
- Inference adapter fallback tests。

### 14.3 内容测试

- 每关至少有一个成功路径。
- 每个失败规则可触发。
- 每个提示分级内容准确。
- 每个解锁物可在后续使用。

## 15. 风险与取舍

### 15.1 Web 路线风险

风险：

- 游戏感不如传统引擎。
- 复杂动画需要额外工程纪律。

应对：

- 用强交互和清晰反馈建立游戏感。
- 动画复杂后引入 PixiJS。
- 不在 MVP 追求大场景。

### 15.2 Python 推理风险

风险：

- checkpoint 加载慢。
- CPU 推理体验差。
- 不同机器依赖复杂。

应对：

- 支持 sample output fallback。
- tiny/small checkpoint 优先。
- 推理服务懒加载模型。
- 明确显示当前使用真实模型还是样例输出。

### 15.3 美术范围风险

风险：

- 过早追求精美导致核心玩法延期。

应对：

- MVP 用程序化视觉资产。
- 优先打磨 token、attention、loss、SFT mask 四个核心视觉。
- 宣传美术晚于可玩纵切片。

## 16. ADR

### ADR-001: MVP 使用 Web-first 表现引擎

**Status:** Proposed

#### Context

游戏需要大量教学 UI、图表、可视化调试器和模型对比，同时需要复用 Python/PyTorch 训练资产。

#### Decision

MVP 使用 Vite + React + TypeScript 作为主 UI，SVG/Canvas 作为主要可视化层，FastAPI 作为本地后端。

#### Options Considered

| 选项 | 复杂度 | 表现力 | Python 集成 | 结论 |
|---|---|---|---|---|
| Streamlit | 低 | 低 | 高 | 仅用于 admin/debug |
| React + SVG/Canvas | 中 | 中高 | 高 | MVP 推荐 |
| React + PixiJS | 中高 | 高 | 高 | 动画复杂后引入 |
| Godot | 中高 | 高 | 中低 | 后续可评估 |
| Unity | 高 | 高 | 低 | 暂不采用 |
| Unreal | 很高 | 很高 | 低 | 不采用 |

#### Consequences

变容易：

- 快速做出可交互教学 UI。
- 接入现有 Python 代码。
- 做模型 Arena、日志、图表和事故报告。

变困难：

- 需要自建任务/渲染约束。
- 传统游戏式大场景不是强项。

需要复盘：

- 当关卡动画复杂到 SVG/Canvas 难维护时，引入 PixiJS。
- 当产品目标转为独立游戏发行时，重新评估 Godot。

### ADR-002: 训练过程默认模拟，成果物真实体验

**Status:** Proposed

#### Context

真实训练时间长、设备差异大、失败成本高，会破坏游戏节奏。但完全虚构训练会失去教学可信度。

#### Decision

训练过程由规则模拟引擎驱动；阶段通过后解锁真实 checkpoint、eval report 和 sample outputs。

#### Consequences

变容易：

- 控制节奏。
- 明确设计失败体验。
- 支持无 GPU 玩家。

变困难：

- 模拟规则需要足够可信。
- 需要维护真实成果物和样例输出。

### ADR-003: 美术采用抽象实验室风格

**Status:** Proposed

#### Context

游戏的主要学习对象是抽象机制，不是角色或场景探索。

#### Decision

采用抽象 LLM 实验室 + 工业控制台 + 数据流可视化风格。优先程序化矢量资产。

#### Consequences

变容易：

- 资产成本低。
- 概念表达清楚。
- UI 和教学反馈一致。

变困难：

- 需要通过交互和动效建立游戏感。
- 宣传视觉冲击力可能弱于传统插画路线。

## 17. MVP 技术任务拆分

### 阶段 1：技术骨架

- 建立 `game/app` React 项目。
- 建立 `game/server` FastAPI 项目。
- 定义 mission/artifact/progress schema。
- 实现 `/api/missions` 和 `/api/progress`。

### 阶段 2：关卡运行

- 实现 mission runtime。
- 实现 causal mask 关卡。
- 实现 tokenizer 合并关卡。
- 实现 submit 判定。

### 阶段 3：模拟引擎

- 实现 curve templates。
- 实现 failure rules。
- 实现 incident report。
- 接入 pretrain simulation UI。

### 阶段 4：Arena

- 实现 artifact registry。
- 实现 sample outputs fallback。
- 接入真实 inference adapter。
- 实现 base vs SFT 对比。

### 阶段 5：美术和体验打磨

- 固化颜色、token tile、heatmap、curve 样式。
- 增加关键动效。
- 做主线 8 关串联。
- 用 Playwright 验证主要界面。

## 18. 下一步

建议立即落这些文件：

```text
game/README.md
game/schemas/mission.schema.json
game/schemas/artifact.schema.json
game/missions/mvp.json
game/artifacts/manifest.json
game/simulator/rules/*.json
```

然后先实现一个不依赖真实 checkpoint 的最小闭环：

```text
Pipeline Map -> Causal Mask Mission -> Submit -> Failure/Success -> Unlock -> Arena sample output
```

这个闭环跑通后，再接真实模型推理。

