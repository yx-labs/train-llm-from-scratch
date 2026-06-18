# LLM 学习游戏策划与落地方案

## 1. 项目定位

本项目目标是在现有 `train-llm-from-scratch` 教学与训练代码基础上，开发一个“真实机制、抽象表现、关卡驱动”的 LLM 学习游戏。玩家不只是阅读教程或运行脚本，而是在一系列可交互挑战中，从 token、张量、attention、loss、optimizer 开始，逐步理解并搭建一条完整的 LLM 训练与对齐流程。

产品参考方向是《图灵完备》：不把复杂技术简化成纯科普动画，而是保留真实规则，让玩家通过构造、调试、失败和优化建立工程直觉。

核心体验不是“等待训练完成”，而是：

```text
理解机制 -> 完成挑战 -> 运行训练模拟 -> 诊断失败/改进方案 -> 解锁真实阶段成果物 -> 体验模型能力变化
```

## 2. 设计目标

### 2.1 教学目标

玩家完成主线后应能理解：

- 文本如何变成 token id，以及 token 粒度如何影响上下文与训练效率。
- decoder-only Transformer 的基本数据流：token ids -> embeddings -> blocks -> logits -> loss。
- causal mask、multi-head attention、residual、LayerNorm、MLP 的作用。
- next-token prediction 为什么能产生语言能力。
- pretraining、SFT、Reward Model、DPO、PPO、GRPO 的区别。
- 为什么训练指标可能“看起来变好但模型变坏”。
- 训练 LLM 时常见工程取舍：显存、batch、context、学习率、数据质量、评测泛化。

### 2.2 产品目标

- 让非专家玩家能通过操作理解 LLM 基础机制。
- 让有工程背景的玩家觉得机制足够真实，不是伪科普。
- 让训练过程可玩、可失败、可诊断，而不是单纯进度条。
- 利用现有项目的真实阶段资产，让玩家体验不同阶段模型的真实差异。
- 先落地一个可玩的 MVP 纵切片，再逐步扩展到完整 pipeline。

### 2.3 边界

第一阶段不追求：

- 真实大模型训练规模。
- 从零开发完整 3D 游戏。
- 浏览器内真实训练大模型。
- 面向生产的训练平台。

第一阶段追求：

- 真实概念映射。
- 清楚的关卡反馈。
- 可对比的阶段成果物体验。
- 能复用本仓库的训练、评测和推理代码。

## 3. 核心设计原则

### 3.1 真实机制，抽象表现

游戏中的每个模块都应该对应真实 LLM 训练机制：

- token 方块对应 token id。
- 向量槽对应 embedding。
- 注意力连线和热力图对应 attention weights。
- 遮罩墙对应 causal mask。
- loss 仪表对应 cross entropy 或偏好损失。
- 训练资源对应显存、时间、batch、context length。
- 奖励计分器对应 reward model、verifier 或 preference objective。

表现可以抽象，但规则必须可解释、可回到真实代码。

### 3.2 先手工体验，再封装自动化

每个核心机制都先让玩家手动解决一个小问题，再解锁自动模块：

- 手动合并字符 -> 解锁 tokenizer。
- 手动放置 mask -> 解锁 causal attention。
- 手动比较 logits -> 解锁 cross entropy 面板。
- 手动挑选 chosen/rejected -> 解锁 reward dataset。
- 手动设 verifier -> 解锁 GRPO 训练模拟。

这样玩家会理解模块为什么存在，而不是只记住名词。

### 3.3 失败是主要教学手段

失败不能只是“挑战失败”。它要提供可诊断症状：

- 曲线异常。
- 模型输出异常。
- attention 图异常。
- 训练资源异常。
- 评测和体验结果不一致。

玩家应通过观察症状反推原因，并通过具体修改完成修复。

### 3.4 成果物真实化

训练过程可以模拟，但阶段完成后应解锁真实资产：

- checkpoint。
- tokenizer。
- config。
- training log。
- eval report。
- sample prompts。
- failure examples。

玩家用真实模型体验每个阶段完成后的差异。

### 3.5 以 pipeline 为世界地图

现有项目 pipeline 天然适合作为主线地图：

```text
Data -> Pretraining -> SFT -> Reward Model -> DPO -> PPO -> GRPO -> Evaluation -> Inference
```

每个 stage 是一个区域，每个区域内有多个挑战关卡。

## 4. 目标玩家

### 4.1 主要玩家

- 想系统学习 LLM 原理的程序员。
- 读过一些 AI 文章，但缺少训练直觉的学习者。
- 有 Python/PyTorch 基础，想从“会调用 API”进阶到理解训练过程的人。

### 4.2 次要玩家

- 对 AI 感兴趣的高级玩家。
- 机器学习课程学生。
- 想做内部培训的团队。

### 4.3 玩家能力假设

MVP 不要求玩家会写 PyTorch，但默认玩家能理解：

- 基本变量和数组。
- 简单数学图表。
- 配置参数会影响系统行为。
- 失败可以通过观察和调参修复。

## 5. 总体游戏结构

### 5.1 主线章节

| 章节 | 名称 | 对应现有项目内容 | 玩家核心收获 |
|---|---|---|---|
| 0 | 实验室启动 | `README.md`, `docs/README.md` | 理解训练 pipeline 全貌 |
| 1 | 文本矿场 | `docs/01_data_pipeline.md` | 数据质量决定模型上限 |
| 2 | Tokenizer 车间 | `docs/foundations/tokenization.md` | 文本如何变成 token |
| 3 | 向量机房 | `docs/foundations/transformer.md` | token id 如何变成向量 |
| 4 | Attention 实验室 | `docs/foundations/attention.md` | causal attention 和多头机制 |
| 5 | Transformer 装配线 | `docs/foundations/transformer.md` | block、residual、LayerNorm、MLP |
| 6 | Loss 与优化控制台 | `docs/foundations/objectives.md`, `optimization.md` | next-token loss、AdamW、LR、batch |
| 7 | 预训练引擎 | `docs/02_pretraining.md` | base model 学语言分布 |
| 8 | 生成舱 | `docs/foundations/generation.md`, `docs/09_inference.md` | decoding 和采样参数 |
| 9 | SFT 指令工坊 | `docs/03_sft.md` | chat template 与 assistant loss mask |
| 10 | 偏好竞技场 | `docs/04_reward_model.md`, `docs/05_dpo.md` | 偏好数据、RM、DPO |
| 11 | RL 控制室 | `docs/06_ppo.md` | PPO、reward hacking、KL |
| 12 | 推理训练场 | `docs/07_grpo.md` | verifier reward、GRPO、数学推理 |
| 13 | 评测中心 | `docs/08_evaluation.md` | 指标、泛化、失败样例 |
| 14 | 模型展厅 | `docs/09_inference.md` | 对比所有阶段模型 |

### 5.2 单关卡循环

每个关卡遵循同一套体验：

```text
目标说明
-> 可操作系统
-> 玩家提交方案
-> 模拟运行
-> 指标/视觉/输出反馈
-> 成功解锁或失败诊断
-> 笔记归档
```

关卡通过以下方式判定：

- 配置是否满足约束。
- 模拟曲线是否达到阈值。
- 关卡内小任务是否通过。
- 模型输出是否满足格式或正确性。
- 诊断题是否定位到真实原因。

### 5.3 成长与解锁

玩家进度不建议设计成传统等级数值，而是以能力模块解锁：

- Tokenizer。
- Data packer。
- Causal attention。
- Multi-head attention。
- Transformer block。
- Trainer。
- Generation sampler。
- SFT formatter。
- Reward scorer。
- DPO optimizer。
- PPO controller。
- GRPO verifier。
- Eval suite。

每个模块既是学习成果，也是后续关卡中的可复用工具。

## 6. 训练模拟与真实成果物

### 6.1 为什么训练模拟

真实训练存在明显产品问题：

- 时长不可控。
- 设备差异大。
- 失败成本高。
- 初学者很难分辨等待、训练、失败的区别。

因此游戏内训练默认使用模拟，但模拟必须基于真实训练规律。

### 6.2 模拟层职责

训练模拟负责：

- 根据玩家配置生成 loss/reward/KL/accuracy 曲线。
- 根据典型错误生成失败症状。
- 根据资源配置计算显存、时间、稳定性。
- 产出“训练事故报告”。
- 在达成目标后解锁真实阶段成果物。

模拟不是随机动画。它应由规则、曲线模板和少量真实日志驱动。

### 6.3 真实成果物职责

真实成果物用于体验阶段完成后的结果：

- 玩家在 arena 里和真实 checkpoint 对话。
- 同一 prompt 可对比不同阶段输出。
- 评测中心可展示真实 eval report。
- Reward Model 阶段可以真实打分 chosen/rejected。
- Inference 阶段可真实调整 temperature/top-p。

### 6.4 推荐资产时间线

| 资产 | 用途 |
|---|---|
| `random_init.pt` | 展示未训练模型的乱码/无意义输出 |
| `pretrain_early.pt` | 展示刚开始学到局部模式但不稳定 |
| `pretrain_good.pt` | 展示 base model 能续写但不听指令 |
| `sft.pt` | 展示 instruction following 与格式遵循 |
| `reward_model.pt` | 展示偏好打分能力 |
| `dpo.pt` | 展示偏好优化后回答更稳 |
| `ppo.pt` | 展示 RLHF 后奖励优化结果 |
| `grpo.pt` | 展示 verifiable reasoning 任务提升 |
| `bad_no_mask.pt` | 展示 causal mask 泄漏的假成功 |
| `bad_sft_no_loss_mask.pt` | 展示 SFT mask 错误导致复读 prompt |
| `bad_reward_hacked.pt` | 展示 reward hacking |

MVP 不需要全部资产一次到位。第一阶段优先准备：

- `pretrain_good.pt`
- `sft.pt`
- `bad_sft_no_loss_mask.pt` 或通过样例输出模拟
- 对应 tokenizer、config、eval report、sample outputs

### 6.5 同 prompt 阶段对比

游戏中应固定一组“标尺 prompt”，贯穿所有阶段：

```text
请用 <think> 和 <answer> 回答：23 + 48 = ?
```

期望体验：

| 阶段 | 典型表现 |
|---|---|
| random | 乱码或无结构 token |
| pretrain | 可能续写文本，但不稳定、不一定回答 |
| SFT | 遵循格式，能按指令回答 |
| DPO | 更偏向人类喜欢的表达 |
| PPO | 如果 reward 合理，格式和答案更稳 |
| GRPO | 算术或可验证任务准确率提升 |

## 7. 失败体验设计

### 7.1 失败体验目标

失败体验要让玩家形成工程诊断能力：

```text
现象 -> 假设 -> 检查 -> 修复 -> 验证
```

失败不是惩罚，不应让玩家单纯重来。每次失败都要留下可学习信息。

### 7.2 三层失败反馈

#### 视觉症状

- loss 曲线爆炸、震荡或平台期。
- attention 热力图看到未来 token。
- token 流被切得过碎。
- 梯度仪表变红。
- 奖励很高但输出质量变差。

#### 指标症状

- train loss 下降但 val loss 上升。
- val loss 异常低，怀疑泄漏。
- reward 上升但 human score 下降。
- KL 飙升。
- accuracy 不变但 format score 上升。
- GPU memory 超预算。

#### 成果物后果

- 模型输出乱码。
- 模型复读 user prompt。
- 模型只输出固定模板。
- 模型钻 verifier 漏洞。
- 模型在训练题上对，新题上错。

### 7.3 必须设计的“假成功”

LLM 工程中最有教育价值的是假成功：

| 假成功 | 表面现象 | 真实问题 |
|---|---|---|
| causal mask 泄漏 | loss 极低 | 训练时偷看答案，生成时不能用 |
| 数据泄漏 | eval 分数很高 | 评测题在训练集中 |
| SFT 全文 loss | loss 降低 | 模型学会复读用户输入 |
| reward hacking | reward 很高 | 人类评分变差 |
| 格式奖励过强 | 格式完美 | 内容不正确 |
| 过拟合小数据 | 样例表现好 | 泛化很差 |

### 7.4 失败提示分级

每个失败提供三档提示：

| 档位 | 说明 | 示例 |
|---|---|---|
| 症状提示 | 描述异常，不给答案 | “验证 loss 好得不正常，但生成质量没有提升。” |
| 方向提示 | 指向模块 | “检查 attention 是否看到了未来 token。” |
| 直接提示 | 给出修复动作 | “使用下三角 causal mask。” |

### 7.5 训练事故报告模板

每次失败后展示：

```text
训练事故报告

关卡：Causal Mask 调试
结果：未通过

主要症状：
- train loss 快速下降到异常低
- generation arena 中回答不稳定
- attention 图中 token 关注了未来位置

可能原因：
模型训练时发生未来信息泄漏。

建议检查：
- attention mask 是否为下三角
- target shift 是否正确
- eval 是否使用 autoregressive generation

下一步：
修正 mask 后重新运行模拟。
```

### 7.6 失败规则示例

```json
{
  "id": "causal_mask_leak",
  "trigger": {
    "causal_mask": false
  },
  "symptoms": {
    "train_loss": "fast_drop",
    "val_loss": "suspiciously_low",
    "generation": "unstable",
    "attention": "future_tokens_visible"
  },
  "hints": [
    "指标好得不正常。",
    "检查模型是否看到了答案。",
    "使用下三角 causal mask。"
  ],
  "fix": {
    "causal_mask": true
  }
}
```

## 8. 关卡设计

### 8.1 MVP 纵切片关卡

MVP 应先做 8 个关卡，覆盖从 token 到 SFT 体验：

| 序号 | 关卡 | 玩家操作 | 成功条件 | 解锁 |
|---|---|---|---|---|
| 1 | 文本清洗 | 删除坏样本、重复样本 | 数据质量分达到阈值 | clean corpus |
| 2 | Token 合并 | 手动合并高频片段 | token 数下降且不过度合并 | tokenizer module |
| 3 | 数据打包 | 把 token stream 切成 block | input/target shift 正确 | packed dataset |
| 4 | Causal Mask | 放置 attention mask | 未来 token 不可见 | attention module |
| 5 | Tiny Transformer | 组装 embedding/block/lm head | forward shape 正确 | trainer module |
| 6 | 预训练模拟 | 调 lr/batch/context | val loss 达到目标且无泄漏 | `pretrain_good.pt` |
| 7 | Generation Arena | 调 temperature/top-p | 输出稳定且不重复 | sampler module |
| 8 | SFT Mask | 标出 assistant loss 区域 | loss mask 正确 | `sft.pt` 和对比 arena |

### 8.2 第一批完整主线关卡

| 编号 | 名称 | 机制 | 失败点 |
|---|---|---|---|
| 0-1 | 点亮实验室 | pipeline 总览 | 不涉及 |
| 1-1 | 脏文本筛选 | 数据清洗 | 保留 HTML/乱码 |
| 1-2 | 重复样本陷阱 | 去重 | eval 泄漏 |
| 1-3 | 训练/验证切分 | split | train/val 混淆 |
| 2-1 | 字符太碎 | tokenizer | vocab 太小 |
| 2-2 | 合并过头 | tokenizer | vocab 太大，泛化差 |
| 2-3 | 特殊 token | tokenizer | stop token 丢失 |
| 3-1 | Token 到向量 | embedding | shape 错误 |
| 3-2 | 位置感 | position embedding | 顺序任务失败 |
| 4-1 | Q/K/V 连线 | attention | 注意力分数错误 |
| 4-2 | 禁止偷看未来 | causal mask | 泄漏假成功 |
| 4-3 | 多头分工 | multi-head | 单头容量不足 |
| 5-1 | 残差桥 | residual | 深层训练不稳 |
| 5-2 | 归一化阀门 | LayerNorm | loss 震荡 |
| 5-3 | MLP 扩展器 | feed-forward | 容量不足 |
| 6-1 | 交叉熵仪表 | CE loss | target shift 错 |
| 6-2 | 学习率旋钮 | optimizer | lr 过大/过小 |
| 6-3 | 显存预算 | batch/context | OOM |
| 7-1 | Base Model 启动 | pretraining | 数据量不足 |
| 7-2 | 泛化检查 | eval | 过拟合 |
| 8-1 | 贪心与采样 | generation | 输出单调 |
| 8-2 | 温度实验 | temperature | 发散或保守 |
| 9-1 | 指令格式 | chat template | 格式错 |
| 9-2 | Assistant Mask | SFT loss mask | 复读 prompt |
| 9-3 | SFT 对比舱 | model comparison | 指令不稳 |
| 10-1 | 偏好样本 | chosen/rejected | 标注噪声 |
| 10-2 | 奖励标尺 | reward model | 打分反转 |
| 10-3 | DPO 平衡 | beta | 风格过偏 |
| 11-1 | Rollout | PPO | 采样质量差 |
| 11-2 | KL 护栏 | PPO KL | reward hacking |
| 11-3 | Advantage | GAE | 更新不稳 |
| 12-1 | Verifier | RLVR | 奖励漏洞 |
| 12-2 | 组内比较 | GRPO | 组大小不合理 |
| 12-3 | 数学训练场 | GRPO eval | 格式对但答案错 |
| 13-1 | 指标不是答案 | evaluation | 只看平均分 |
| 13-2 | 失败样例分析 | evaluation | 忽略边界 case |
| 14-1 | 模型展厅 | inference | 采样配置误用 |

## 9. 系统设计

### 9.1 模块分层

```text
Game UI
  -> Mission Runtime
  -> Simulation Engine
  -> Artifact Registry
  -> Evaluation/Inference Adapter
  -> Existing Training Code
```

#### Game UI

负责：

- 关卡地图。
- 可交互挑战。
- 训练模拟可视化。
- 失败报告。
- 模型对比 arena。
- 玩家笔记与进度。

#### Mission Runtime

负责：

- 读取关卡定义。
- 管理玩家提交。
- 判定成功/失败。
- 发放解锁模块。
- 记录进度。

#### Simulation Engine

负责：

- 生成指标曲线。
- 应用失败规则。
- 模拟资源消耗。
- 生成诊断事件。

#### Artifact Registry

负责：

- 管理 checkpoint、tokenizer、config、logs、eval reports。
- 提供阶段资产元数据。
- 控制解锁状态。

#### Evaluation/Inference Adapter

负责：

- 调用现有推理脚本或封装后的 Python API。
- 对比不同 checkpoint 输出。
- 加载 reward model 打分。
- 运行轻量 eval。

### 9.2 建议目录结构

所有游戏相关内容放在 `game/` 下：

```text
game/
  GAME_DESIGN.md
  README.md
  missions/
    00_lab_start.json
    01_data.json
    02_tokenizer.json
    03_attention.json
    04_pretrain.json
    05_sft.json
  schemas/
    mission.schema.json
    artifact.schema.json
    simulation.schema.json
  artifacts/
    manifest.json
    samples/
      base_vs_sft_prompts.jsonl
      failure_cases.jsonl
  simulator/
    README.md
    rules/
      optimizer_failures.json
      attention_failures.json
      sft_failures.json
  app/
    README.md
```

MVP 可以先只落：

- `game/GAME_DESIGN.md`
- `game/missions/*.json`
- `game/artifacts/manifest.json`
- 一个最小 Web 或 Streamlit 原型

### 9.3 关卡数据模型

```json
{
  "id": "04_02_causal_mask",
  "title": "禁止偷看未来",
  "chapter": "Attention 实验室",
  "learning_objectives": [
    "理解 causal mask 的作用",
    "识别未来信息泄漏导致的假成功"
  ],
  "unlocks": ["attention.causal_mask"],
  "inputs": {
    "sequence": ["I", "love", "LLM", "."],
    "allowed_controls": ["mask_grid"]
  },
  "success_conditions": [
    {
      "type": "mask_is_lower_triangular"
    },
    {
      "type": "no_future_attention"
    }
  ],
  "failure_rules": ["causal_mask_leak"],
  "hints": [
    "每个位置只能看到自己和左侧 token。",
    "如果模型看到未来 token，loss 会变得不真实。"
  ],
  "post_success": {
    "show_concept": "causal_self_attention",
    "unlock_artifact": null
  }
}
```

### 9.4 成果物 manifest

```json
{
  "artifacts": [
    {
      "id": "pretrain_good",
      "stage": "pretrain",
      "type": "checkpoint",
      "path": "artifacts/checkpoints/pretrain_good.pt",
      "config": "artifacts/configs/pretrain_good.json",
      "tokenizer": "artifacts/tokenizers/base",
      "eval_report": "artifacts/eval/pretrain_good.json",
      "unlocked_by": "07_01_pretrain_sim",
      "description": "Base model after successful pretraining simulation."
    }
  ]
}
```

### 9.5 进度状态

```json
{
  "player_id": "local",
  "completed_missions": ["01_01_clean_text", "02_01_token_merge"],
  "unlocked_modules": ["tokenizer.basic", "data.packer"],
  "unlocked_artifacts": ["pretrain_good"],
  "notebook": [
    {
      "concept": "causal_mask",
      "status": "learned"
    }
  ]
}
```

## 10. UI/UX 设计

### 10.1 主要界面

#### Pipeline 地图

以现有训练流程为地图：

```text
Data -> Tokenizer -> Transformer -> Pretrain -> SFT -> Preference -> RL -> Eval -> Chat
```

每个节点展示：

- 已完成关卡数。
- 已解锁模块。
- 已解锁成果物。
- 当前推荐下一关。

#### 关卡实验台

关卡内主要区域：

- 左侧：目标和可用模块。
- 中央：可操作画布。
- 右侧：实时指标和约束。
- 底部：运行、提交、提示、事故报告。

#### 训练模拟控制台

展示：

- loss 曲线。
- val loss 曲线。
- perplexity。
- learning rate。
- GPU memory。
- tokens/sec。
- warning events。

#### 模型 Arena

对比两个或多个 checkpoint：

- 同一 prompt 同时发给多个模型。
- 展示输出差异。
- 展示指标差异。
- 展示采样参数。
- 可保存失败样例。

#### 训练事故报告

失败时展示：

- 主要异常。
- 可疑模块。
- 失败样例。
- 可展开提示。
- 可直接跳回需要修复的控件。

### 10.2 图形风格

建议风格：

- 抽象实验室。
- 高对比但克制。
- 主要元素用网格、流线、热力图、模块板。
- 不做拟真 GPU/服务器。
- 不用大量装饰性插画。

信息优先级：

1. 玩家当前要操作什么。
2. 操作对指标产生什么影响。
3. 为什么失败或成功。
4. 这个机制对应真实 LLM 哪一部分。

### 10.3 关键可视化

| 机制 | 可视化 |
|---|---|
| Tokenization | 文本切分成不同长度 token 方块 |
| Context window | 固定长度传送带 |
| Embedding | token 落入向量槽 |
| Attention | Q/K/V 连线和热力图 |
| Causal mask | 上三角区域被遮挡 |
| Cross entropy | 正确 token 概率槽 |
| Optimization | loss 曲线和梯度仪表 |
| SFT mask | user token 灰色，assistant token 高亮 |
| Reward model | 两个回答进入评分秤 |
| DPO | chosen/rejected log-prob 差值 |
| PPO | reward、KL、policy update 三联仪表 |
| GRPO | 同 prompt 多个答案的组内排名 |

## 11. 与现有项目的集成

### 11.1 可复用内容

| 现有内容 | 用途 |
|---|---|
| `docs/foundations/*.md` | 关卡背后的概念说明 |
| `docs/01_data_pipeline.md` | 数据章节内容 |
| `docs/02_pretraining.md` | 预训练章节内容 |
| `docs/03_sft.md` | SFT 章节内容 |
| `docs/04_reward_model.md` | Reward Model 章节内容 |
| `docs/05_dpo.md` | DPO 章节内容 |
| `docs/06_ppo.md` | PPO 章节内容 |
| `docs/07_grpo.md` | GRPO 章节内容 |
| `docs/08_evaluation.md` | 评测章节内容 |
| `docs/09_inference.md` | Chat/Inference 体验 |
| `configs/smoke/*.json` | 快速真实运行或校验 |
| `scripts/train_*.py` | 真实训练资产生成 |
| `scripts/eval_post_training.py` | 真实评测 |
| `src/post_training/inference.py` | 模型体验 |
| `ui/stages.py` | 现有 stage 元数据来源 |

### 11.2 不建议直接复用的内容

现有 Streamlit UI 适合作为控制台，但不适合长期承载图灵完备式游戏交互。建议：

- 短期：可用 Streamlit 做 debug/admin 面板。
- MVP：用 Web 前端实现核心交互体验。
- 后续：FastAPI 封装训练、推理、资产接口。

### 11.3 推荐技术方案

```text
Frontend: Vite + React + TypeScript
Rendering: SVG/Canvas, 后期可加 PixiJS
Backend: FastAPI
State: SQLite
Simulation: Python rule engine
Inference: 复用现有 PyTorch 代码
Artifacts: 本地 manifest + 文件目录
```

MVP 也可以先做单机模式：

```text
React app -> local FastAPI -> repo scripts/src
```

### 11.4 API 草案

| API | 说明 |
|---|---|
| `GET /missions` | 获取关卡列表 |
| `GET /missions/{id}` | 获取关卡详情 |
| `POST /missions/{id}/submit` | 提交玩家方案 |
| `POST /simulate/train` | 运行训练模拟 |
| `GET /progress` | 获取玩家进度 |
| `POST /arena/generate` | 指定 checkpoint 生成文本 |
| `POST /arena/compare` | 多 checkpoint 对比生成 |
| `POST /reward/score` | Reward Model 打分 |
| `GET /artifacts` | 获取可用成果物 |
| `GET /eval/{artifact_id}` | 获取评测报告 |

## 12. MVP 范围

### 12.1 MVP 目标

做出一条完整、可玩的纵切片：

```text
Tokenizer -> Causal Mask -> Tiny Pretrain Simulation -> Base Model Arena -> SFT Mask -> SFT Arena
```

玩家完成后能明确感受到：

- tokenization 会影响训练输入。
- causal mask 错误会造成假成功。
- base model 和 SFT model 的行为不同。
- loss mask 是 SFT 的关键机制。
- 训练可以失败，失败有可诊断原因。

### 12.2 MVP 包含内容

- Pipeline 地图。
- 8 个关卡。
- 训练模拟控制台。
- 至少 2 个真实成果物体验：`pretrain_good` 和 `sft`。
- Base vs SFT prompt arena。
- 至少 3 类失败报告：
  - causal mask leak。
  - learning rate too high。
  - SFT loss mask wrong。

### 12.3 MVP 不包含内容

- PPO/GRPO 完整玩法。
- 多用户账号。
- 在线排行榜。
- 复杂剧情。
- 大模型训练。
- 付费或部署系统。

### 12.4 MVP 验收标准

功能标准：

- 玩家能从第一关推进到 SFT arena。
- 每关有明确成功/失败反馈。
- 至少一个假成功案例可触发。
- 至少两个真实 checkpoint 可被 arena 调用。
- 关卡定义数据化，不硬编码在 UI 中。

学习标准：

- 玩家能解释 causal mask 为什么必要。
- 玩家能解释 base model 和 SFT model 区别。
- 玩家能识别 loss 好看但模型坏掉的至少一种原因。

技术标准：

- 所有游戏相关文件位于 `game/`。
- 不破坏现有训练脚本。
- 能在本地启动。
- 能离线运行基础关卡模拟。

## 13. 排期建议

### 第 1 周：策划固化与资产清点

- 完成主策划文档。
- 定义 MVP 关卡列表。
- 定义 mission schema。
- 清点可用 checkpoint 和 eval 输出。
- 确认第一批真实成果物来源。

产出：

- `game/GAME_DESIGN.md`
- `game/schemas/mission.schema.json`
- `game/artifacts/manifest.json`
- MVP mission 草案

### 第 2 周：模拟引擎原型

- 实现规则驱动的训练模拟。
- 实现 loss 曲线模板。
- 实现 failure rule 触发。
- 生成训练事故报告。

产出：

- `game/simulator/`
- 3 个 failure rules
- 预训练/SFT 模拟曲线

### 第 3 周：前端纵切片

- Pipeline 地图。
- 关卡实验台。
- Causal mask 交互。
- 训练模拟控制台。

产出：

- 可本地运行的 Web 原型。
- 完成前 5 个关卡。

### 第 4 周：真实成果物 Arena

- 接入 checkpoint manifest。
- 接入推理接口。
- 实现 base vs SFT 对比。
- 加入采样参数面板。

产出：

- Prompt arena。
- Base/SFT 对比体验。
- 固定 prompt 集。

### 第 5 周：失败体验打磨

- 增加假成功案例。
- 增加事故报告。
- 增加提示分级。
- 增加失败样例库。

产出：

- `causal_mask_leak` 完整体验。
- `sft_no_loss_mask` 完整体验。
- `lr_explosion` 完整体验。

### 第 6 周：MVP 收口

- 跑完整用户流程。
- 修正关卡难度。
- 完善文档和启动方式。
- 确定下一阶段 Reward/DPO 扩展范围。

产出：

- MVP 版本。
- README。
- 下一阶段 backlog。

## 14. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 训练成果物太大 | 分发困难 | 使用 tiny/small checkpoint，完整资产可选下载 |
| 模拟太假 | 失去工程真实感 | 使用真实日志模板和真实失败规则 |
| 数学门槛太高 | 玩家流失 | 先体验症状，再展开公式 |
| 前端交互范围膨胀 | MVP 延期 | 第一版只做 SVG/Canvas 必需交互 |
| 现有训练代码接口不稳定 | 接入成本高 | 用 adapter 包一层，不直接改核心训练代码 |
| checkpoint 效果不明显 | 阶段差异弱 | 设计固定 prompt 和小任务，让差异清晰 |
| 玩家不知道下一步 | 学习中断 | Pipeline 地图始终给推荐下一关 |

## 15. 成功指标

### 15.1 产品指标

- 完成 MVP 主线的玩家比例。
- 每个失败关卡平均重试次数。
- 玩家是否主动查看提示。
- Arena 中模型对比的使用次数。
- 玩家完成后能否正确回答关键概念题。

### 15.2 学习指标

通过内置小测或关卡表现衡量：

- 能否识别 causal mask 泄漏。
- 能否解释 SFT loss mask。
- 能否区分 pretraining 和 SFT。
- 能否识别 reward hacking。
- 能否理解 eval 指标和真实体验不一致。

### 15.3 技术指标

- 本地启动时间。
- 单关卡模拟响应时间。
- 推理接口平均延迟。
- 资产加载失败率。
- mission schema 校验通过率。

## 16. 后续扩展

### 16.1 Reward/DPO 扩展

新增玩法：

- 玩家标注 chosen/rejected。
- 训练一个 reward scorer。
- 观察偏好数据噪声。
- 调 DPO beta。
- 比较 SFT vs DPO 输出风格。

核心失败：

- 偏好标签反转。
- beta 太大导致模型过偏。
- 数据分布太窄导致泛化差。

### 16.2 PPO 扩展

新增玩法：

- rollout 采样。
- reward + KL 联合控制。
- advantage 估计。
- policy update 稳定性。

核心失败：

- reward hacking。
- KL 爆炸。
- KL 太强学不动。
- reward 高但 human score 下降。

### 16.3 GRPO/RLVR 扩展

新增玩法：

- 设计 verifier。
- 同 prompt 多答案组内比较。
- curriculum 难度提升。
- 数学题准确率曲线。

核心失败：

- verifier 漏洞。
- 格式奖励压过正确性。
- 组大小太小导致信号不稳。
- prompt 分布太简单导致迁移失败。

### 16.4 创作/沙盒模式

完成主线后提供沙盒：

- 自定义小数据集。
- 自定义 reward rule。
- 自定义 eval prompt。
- 对比不同训练路线。
- 导出训练事故报告。

## 17. 近期行动项

建议下一步按顺序执行：

1. 在 `game/` 下补 `README.md`，说明游戏工作区边界和启动方式。
2. 定义 `mission.schema.json`。
3. 写 MVP 8 个 mission JSON 草案。
4. 写 `artifacts/manifest.json` 草案。
5. 选定或生成 `pretrain_good` 与 `sft` 两个可体验成果物。
6. 做一个最小 mission runner，先不做完整前端。
7. 做 causal mask 关卡和 base vs SFT arena 作为第一条可演示链路。

## 18. 决策记录

| 决策 | 结论 | 原因 |
|---|---|---|
| 游戏相关文件放置 | 全部放在 `game/` | 保持和现有训练项目隔离 |
| 分支 | 使用 `game` 分支 | 避免影响主线训练项目 |
| 训练过程 | 默认模拟 | 保证节奏和设备兼容 |
| 阶段结果 | 使用真实成果物 | 保留真实体验 |
| 第一版范围 | Tokenizer 到 SFT | 最短路径证明核心玩法 |
| UI 技术方向 | Web 前端优先 | 复杂交互比 Streamlit 更合适 |
| 后端集成 | Adapter 包裹现有代码 | 减少对训练代码的侵入 |

