# LLM Complete — 第 1 章 Text → Token 可执行详细设计文档（更新版）

> 版本：v0.2 / 4 大关 + 微挑战结构  
> 章节定位：Chapter 1 / Text → Token  
> 前置章节：Chapter 0 / Tensor Bootcamp  
> 本章目标：让玩家亲手把 `raw text` 修复成模型可以消费的 `token_ids[B,T]`，并理解 tokenizer 在 LLM 输入管线中的真实工程作用。

---

## 0. 本次更新要点

本版将原先较散的 Text → Token 内容，重构为与 Chapter 0 一致的 **“大关卡 + 阶梯挑战 + 概念胶囊 + 可验证工程任务”** 结构。

新的章节结构为：

```text
第 1 章：Text → Token
├─ 1-1 Text Cannot Flow
│  ├─ 1-1A Raw Text Object
│  ├─ 1-1B Type Gate Failure
│  ├─ 1-1C Tokenizer Socket
│  └─ 1-1D Token ID Contract
│
├─ 1-2 Token Split
│  ├─ 1-2A Boundary Cutter
│  ├─ 1-2B Split Comparison
│  ├─ 1-2C Token Count Meter
│  ├─ 1-2D Merge Forge
│  └─ 1-2E Preserve Symbols
│
├─ 1-3 Vocab Address
│  ├─ 1-3A Vocab Lookup
│  ├─ 1-3B Address Lighting
│  ├─ 1-3C Stable ID Test
│  └─ 1-3D Token Buffer Build
│
├─ 1-4 Unknown & Buffer
│  ├─ 1-4A OOV Failure
│  ├─ 1-4B Fallback Splitter
│  ├─ 1-4C Special Token Injector
│  ├─ 1-4D Padding Builder
│  └─ 1-4E Token Budget Gate
│
└─ 1-X Tokenizer Gauntlet
```

本章不再按“教程页”推进，而是让玩家经历：

```text
接错 raw text
→ 发现类型合同失败
→ 插入 tokenizer
→ 观察 token split
→ 控制 token 数
→ 查 vocab 得到 ID
→ 处理 unknown
→ 注入 special token
→ padding 成 batch
→ 通过 hidden tests
```

最终玩家应形成一句工程直觉：

> LLM 的输入不是文字，而是 tokenizer 生成的整数 token ID 张量。  
> Text → Token 的目标是把不规则文本修复成稳定、可查表、可批处理的 `int[B,T]` 输入合同。

---

## 1. 章节总览

### 1.1 一句话定义

第 1 章是一组 **Tokenizer Pipeline 修复任务**。玩家需要搭建并调试：

```text
Raw Text
→ Tokenizer
→ Token Pieces
→ Vocab Lookup
→ Token IDs
→ Special Tokens
→ Padding / Truncation
→ Token Buffer[int[B,T]]
```

最终输出：

```text
token_ids[B,T]      int tensor
attention_mask[B,T] int/bool tensor
tokens[B,T]         debug display only
```

---

### 1.2 与 Chapter 0 的衔接

Chapter 0 中，玩家已经理解：

```text
Tensor / Rank / Shape / Axis Semantics
hidden[B,T,C]
token_ids[B,T] → Embedding Lookup → hidden[B,T,C]
```

但在 0-1 里，`Tokenizer` 是一个黑盒：

```text
Text Batch → Tokenizer → token_ids[B,T]
```

第 1 章要拆开这个黑盒，让玩家知道：

```text
Tokenizer 不是把“单词”简单编号。
它是一个确定性转换系统：
文本规范化、边界切分、子词合并、词表查找、fallback、特殊 token、padding 都属于输入合同的一部分。
```

---

### 1.3 本章学习成果

玩家通关后，应能理解并操作：

```text
1. raw text 是 string，不是 tensor，不能直接进入 Embedding Lookup。
2. Tokenizer 的职责是把 raw text 转成 token pieces，再转成 token IDs。
3. Token 不等于单词，也不等于字符，它由 tokenizer policy 决定。
4. 不同切分策略会产生不同 token 数，从而改变 T、上下文占用和成本。
5. Vocab table 把 token piece 映射到 token ID。
6. token ID 是 embedding table 的行地址。
7. 同一个 tokenizer 下，同一个 token piece 必须得到稳定 ID。
8. 未登录 token / OOV 需要 fallback，而不是让管线崩溃。
9. `<bos>`、`<eos>`、`<pad>`、`<unk>` 等特殊 token 也是词表中的 ID。
10. batch 中不同长度文本需要 padding 到统一 T。
11. padding 通常需要 attention mask / loss mask 参与后续章节。
12. 最终模型入口合同是 `token_ids[B,T]`，而不是自然语言文本。
```

---

### 1.4 本章设计原则

延续 Chapter 0 的核心组织：

```text
Concept Capsule / 最小概念卡
→ Broken Board / 故障画布
→ Repair Operation / 玩家修复
→ Probe & Trace / 探针观察
→ Autograder / 可验证测试
→ Debrief / 阶段小结
→ Tool Unlock / 工具解锁
```

每个微挑战只引入一个核心概念，且必须立刻服务于操作任务。

错误反馈不使用“答错了”，而使用工程解释：

```text
Connection failed:
Embedding Lookup expects int token IDs,
but received raw text:string.
```

---

### 1.5 MVP 技术边界

MVP 不要求复刻 GPT-2 / SentencePiece / tiktoken 的完整真实行为，建议实现一个 **ToyByteBPE Tokenizer**：

```text
1. 保留大小写，除非某个挑战显式开启 Normalization。
2. 空格显示为可见 marker `▁`，用于教学。
3. 标点独立成候选片段。
4. 英文可先拆成字符或短子词，再通过 merge table 合并。
5. 中文 / 日文 / emoji / 未知 Unicode 使用 char fallback 或 byte fallback。
6. vocab 规模 80–300 个 token 即可。
7. merge table 使用 20–50 条 ranked merges。
8. 特殊 token 固定：`<pad>=0`、`<bos>=1`、`<eos>=2`、`<unk>=3`。
```

目标是教学机制真实，规模可控。

---

## 2. 章节开场设计

### 2.1 开场标题

```text
Chapter 1 — Text → Token
Tokenizer Engineering Bay
```

### 2.2 开场文案

```text
在上一章中，你已经让 token_ids[B,T] 进入了 Embedding Lookup。

但 token_ids 从哪里来？

本章你将打开 Tokenizer 黑盒，修复一条从 raw text 到 int[B,T] 的输入管线。

最终目标：
让任意文本 batch 都能被转换成合法的 token_ids[B,T]，
并通过 Embedding Lookup 的输入合同检查。
```

### 2.3 开场画面

```text
[Raw Text] --red X--> [Embedding Lookup]

System Error:
Expected int[B,T]
Received string
```

随后出现本章目标管线：

```text
Raw Text
→ Tokenizer
→ Token Pieces
→ Vocab Lookup
→ Token IDs
→ Batch Token Buffer[int[B,T]]
```

按钮：

```text
Start Tokenizer Bootcamp
```

---

## 3. 章节关卡地图

### 3.1 4 大关 + 1 综合考核

| 大关 | 核心问题 | 玩家最终产物 | 教学点 |
|---|---|---|---|
| 1-1 Text Cannot Flow | 为什么文字不能直接进模型 | 插入 Tokenizer 的最小输入管线 | raw text 与 int tensor 类型不同 |
| 1-2 Token Split | 文本如何被切成 token | 可控 token boundary 与 token budget | token 数影响 T、上下文、成本 |
| 1-3 Vocab Address | token 如何变成 ID | token piece → vocab ID → token buffer | token ID 是 embedding table 地址 |
| 1-4 Unknown & Buffer | 真实输入如何补全 | fallback + special token + padding + mask | tokenizer 策略影响输入合法性 |
| 1-X Tokenizer Gauntlet | 综合迁移考核 | 完整 Text → Token pipeline | 泛化到未见文本和隐藏测试 |

### 3.2 推荐节奏

每个大关 8–15 分钟，整个第 1 章主线 45–70 分钟。  
可选支线和 S/A 排名挑战可以追加 30–60 分钟。

---

## 4. 1-1 Text Cannot Flow

### 4.1 大关定位

**核心问题：** 为什么 `raw text` 不能直接接入模型数值模块？  
**最终目标：** 让玩家修复一个 `raw text → Embedding Lookup` 的类型合同错误，插入 Tokenizer，并生成最小 token ID 序列。  
**本关输出：**

```text
token_ids[T]
dtype = int
ready_for_embedding = true
```

### 4.2 本关节点

```text
Text Input
Text Inspector
Type Gate
Tokenizer Socket
Token ID Emitter
Embedding Lookup Probe
```

---

### 4.3 1-1A Raw Text Object

#### Concept Card

**标题：** Raw Text 不是 Tensor  

**玩家可见文案：**

```text
你输入的是文字，
但模型内部大部分模块处理的是数字 tensor。

Raw text 的类型是 string。
它可以被查看、切分、编码，
但不能直接进入 Embedding Lookup。
```

**图示：**

```text
"we train llm" : string

不是：
int tensor
float tensor
```

**本阶段任务：**

```text
把一段 raw text 放入 Text Inspector，查看它的类型和字符结构。
```

#### 画布状态

```text
[Raw Text Object] → [Text Inspector]
```

#### 玩家操作

玩家将文本卡拖入 Inspector：

```text
"we train llm"
```

Inspector 显示：

```text
object_type: raw_text
storage_type: string
length_chars: 12
contains_spaces: true
contains_punctuation: false
model_input_ready: false
```

#### 验证规则

```text
✓ raw text inspected
✓ type recognized as string
✓ not marked as tensor
```

#### 失败反馈

如果玩家尝试给 raw text 贴 tensor tag：

```text
Type mismatch:
Raw text is a string object.
It has characters, but no tensor shape yet.
Use a tokenizer to turn text into token IDs.
```

#### Debrief

```text
Raw text 是输入材料，不是模型数值输入。
下一步你会尝试把它接入 Embedding Lookup，并看到为什么会失败。
```

#### 解锁

```text
Type Gate
Embedding Lookup Probe
```

---

### 4.4 1-1B Type Gate Failure

#### Concept Card

**标题：** 模型节点有输入合同  

**玩家可见文案：**

```text
每个模型节点都声明自己能接收什么类型的数据。

Embedding Lookup 需要：
int token IDs

它不能读取原始 string。
```

**图示：**

```text
Embedding Lookup
expected: int[T] or int[B,T]
received: string
```

**本阶段任务：**

```text
尝试连接 Raw Text 到 Embedding Lookup，观察类型合同失败。
```

#### 画布状态

```text
[Text Input] --?--> [Embedding Lookup Probe]
```

#### 玩家操作

玩家把 Text Input 输出线拖到 Embedding Lookup。连接线变红，Type Gate 弹出错误。

#### 验证规则

```text
✓ player attempted invalid connection
✓ Type Gate detected expected int / received string
✓ error trace shown
```

#### 失败反馈 / 教学反馈

```text
Connection failed:
Embedding Lookup expects token IDs.

Expected:
int[T] or int[B,T]

Received:
raw_text:string

Insert a Tokenizer between text and numeric model nodes.
```

#### Debrief

```text
这次失败是正确的。
它说明 raw text 还没有进入模型输入合同。
下一步：插入 Tokenizer Socket。
```

#### 解锁

```text
Tokenizer Socket
Token ID Emitter
```

---

### 4.5 1-1C Tokenizer Socket

#### Concept Card

**标题：** Tokenizer 是文本入口适配器  

**玩家可见文案：**

```text
Tokenizer 的最小职责是：
把 raw text 转换成模型可以处理的 token ID 序列。

它位于文字世界和数值模型之间。
```

**图示：**

```text
Raw Text → Tokenizer → Token IDs
string       adapter       int tensor
```

**本阶段任务：**

```text
把 Tokenizer Socket 插入 Text Input 和 Embedding Lookup 之间。
```

#### 画布状态

```text
[Text Input]      [Embedding Lookup Probe]
        
工具栏新增：[Tokenizer Socket]
```

#### 玩家操作

玩家拖入 Tokenizer Socket 并连接：

```text
Text Input.output:string
→ Tokenizer.input:string
→ Tokenizer.output:token_stream
→ Token ID Emitter.input
```

初始 Tokenizer 可以是最简单的 whitespace tokenizer。

#### Trace 显示

```text
raw: "we train llm"
pre_tokens: ["we", "train", "llm"]
```

#### 验证规则

```text
✓ Tokenizer inserted
✓ raw_text no longer connects directly to embedding
✓ tokenizer emits token pieces
```

#### 失败反馈

如果玩家绕过 Tokenizer：

```text
Bypass detected:
Text still flows directly into numeric module.
All model numeric nodes require token IDs, not raw strings.
```

#### Debrief

```text
现在文字已经被切成 token pieces。
但 token pieces 仍然是字符串片段，还不是 token IDs。
```

#### 解锁

```text
Token ID Contract viewer
Vocab Table preview locked
```

---

### 4.6 1-1D Token ID Contract

#### Concept Card

**标题：** 模型入口需要 token ID  

**玩家可见文案：**

```text
token piece 仍然是字符串。
模型最终需要的是整数 token ID。

在后续章节中，这些 ID 会作为 Embedding Table 的行地址。
```

**图示：**

```text
["we", "train", "llm"]
        ↓
[502, 2841, 9172]
        ↓
int[T]
```

**本阶段任务：**

```text
让 Tokenizer 输出 int token ID 序列，并通过 Embedding Lookup 的输入检查。
```

#### 画布状态

```text
[Text Input]
   ↓
[Tokenizer Socket]
   ↓ token pieces
[Token ID Emitter]
   ↓ int[T]
[Embedding Lookup Probe]
```

#### 玩家操作

玩家启用 Token ID Emitter，使用 toy vocab 将 token piece 转成 ID。

可见输出：

```text
tokens:    ["we", "train", "llm"]
token_ids: [502, 2841, 9172]
dtype:     int
shape:     [T=3]
```

#### 验证规则

```text
✓ token pieces produced
✓ token ids produced
✓ token_ids dtype = int
✓ shape rank = 1 for single sequence
✓ Embedding Lookup accepts token_ids[T]
```

#### 失败反馈

如果输出仍是 token string：

```text
Contract incomplete:
Tokenizer emitted token strings, but Embedding Lookup requires integer IDs.
Use Vocab Lookup / Token ID Emitter.
```

#### Debrief

```text
你已经修复了最小输入管线：
raw text → token pieces → token IDs。

下一关会研究：这些 token pieces 到底应该怎么切。
```

#### 解锁

```text
1-2 Token Split
Boundary Cutter
Token Count Meter
```

---

## 5. 1-2 Token Split

### 5.1 大关定位

**核心问题：** 文本如何被切成 token？  
**最终目标：** 玩家理解 tokenization policy 会改变 token 数、T、上下文预算和成本。  
**本关输出：**

```text
token_pieces[T]
token_count = T
budget_status = pass/fail
```

### 5.2 本关节点

```text
Text Input
Boundary Cutter
Split Policy Switch
Token Count Meter
Merge Forge
Symbol Keeper
Token Budget Gate
```

---

### 5.3 1-2A Boundary Cutter

#### Concept Card

**标题：** Token Boundary：哪里切开文本  

**玩家可见文案：**

```text
Tokenizer 首先需要决定文本边界。

最简单的策略是按空格和标点切分，
但这只是许多 tokenizer 策略中的一种。
```

**图示：**

```text
"we train llm!"
→ [we] [train] [llm] [!]
```

**本阶段任务：**

```text
使用 Boundary Cutter 把一句文本切成可见 token pieces。
```

#### 画布状态

```text
[Text Input] → [Boundary Cutter] → [Token Piece Strip]
```

#### 玩家操作

玩家拖动切割规则开关：

```text
split_on_space = true
split_punctuation = true
keep_space_marker = false
```

输出：

```text
raw: "we train llm!"
pieces: ["we", "train", "llm", "!"]
T = 4
```

#### 验证规则

```text
✓ spaces handled
✓ punctuation split
✓ token pieces emitted in original order
```

#### 失败反馈

如果玩家不处理标点：

```text
Boundary issue:
"llm!" remains a combined piece.
This may be valid in some tokenizers, but the current vocab has separate entries for "llm" and "!".
Split punctuation for this board.
```

#### Debrief

```text
Token boundary 是 tokenizer 的第一层规则。
同一句文本，不同边界规则会得到不同 token pieces。
```

#### 解锁

```text
Split Policy Switch
```

---

### 5.4 1-2B Split Comparison

#### Concept Card

**标题：** Token 不一定是单词  

**玩家可见文案：**

```text
同一句文本可以被不同策略切分。

字符级：token 多，但覆盖性强。
词级：token 少，但容易遇到未知词。
子词级：介于两者之间，是很多 LLM tokenizer 的核心直觉。
```

**图示：**

```text
"training"
char: [t,r,a,i,n,i,n,g]
word: [training]
subword: [train, ing]
```

**本阶段任务：**

```text
对同一句文本比较三种切分策略，观察 token 数变化。
```

#### 画布状态

```text
[Text Input]
   ↓
[Split Policy Switch]
   ├─ char
   ├─ word
   └─ subword
   ↓
[Comparison Table]
```

#### 玩家操作

输入文本：

```text
"training tokenizers"
```

玩家切换策略：

```text
char     → T = 19
word     → T = 2, but OOV risk
subword  → T = 4 or 5
```

#### 验证规则

```text
✓ all three strategies evaluated
✓ token counts displayed
✓ player selects policy that passes vocab + budget constraints
```

#### 失败反馈

如果玩家只选择词级导致 OOV：

```text
Word-level split is compact,
but "tokenizers" is not in the current vocab.
Use subword split or fallback.
```

#### Debrief

```text
Tokenizer policy 是一种取舍：
更少 token、更小词表、更强覆盖性之间需要平衡。
```

#### 解锁

```text
Token Count Meter
```

---

### 5.5 1-2C Token Count Meter

#### Concept Card

**标题：** Token 数决定 T  

**玩家可见文案：**

```text
模型看到的是 token 序列。

一段文本被切成多少 token，
就会占用多少个 token position，也就是 T。

T 会影响上下文长度、显存和计算成本。
```

**图示：**

```text
text → 6 tokens → T = 6
text → 18 tokens → T = 18
```

**本阶段任务：**

```text
在 context budget 限制下，调整切分策略，让 token 数不超过上限。
```

#### 画布状态

```text
[Split Output] → [Token Count Meter] → [Token Budget Gate]
```

当前任务：

```text
context_budget = 8
```

#### 玩家操作

玩家尝试不同策略，Meter 实时显示：

```text
char split:    T = 29  FAIL
word split:    T = 4   OOV warning
subword split: T = 6   PASS
```

#### 验证规则

```text
✓ final T <= budget
✓ no unresolved OOV
✓ order preserved
```

#### 失败反馈

```text
Budget exceeded:
Current T = 29
Budget T = 8

Token count affects context usage.
Try using merge rules or subword pieces.
```

#### Debrief

```text
Tokenization 会直接改变 T。
这就是为什么同样长度的文字，在不同 tokenizer 下成本可能不同。
```

#### 解锁

```text
Merge Forge
```

---

### 5.6 1-2D Merge Forge

#### Concept Card

**标题：** Merge：从小片段合成常见 token  

**玩家可见文案：**

```text
很多 tokenizer 从小片段开始，
再根据 merge rules 合并常见组合。

例如：
[train] + [ing] → [training]

这能减少 token 数，
但必须保证合并后的 token 存在于 vocab 中。
```

**图示：**

```text
[tok] [en] [izer]
   merge
[token] [izer]
```

**本阶段任务：**

```text
选择合适的 merge rules，把 token 数降低到 budget 内。
```

#### 画布状态

```text
[Initial Pieces] → [Merge Forge] → [Merged Tokens]
```

Merge Table：

```text
("train", "ing") → "training"
("token", "izer") → "tokenizer"
("tokenizer", "s") → "tokenizers"
```

#### 玩家操作

玩家将 merge rules 放到 Merge Forge 的 priority slots。Forge 按 rank 应用。

#### 验证规则

```text
✓ selected merges exist in merge table
✓ merged tokens exist in vocab
✓ final T <= budget
✓ final sequence decodes back to original text under current policy
```

#### 失败反馈

如果玩家合并出词表不存在的 token：

```text
Invalid merge:
The piece "trainllm" is not in vocab.
Merge rules must produce known token pieces.
```

#### Debrief

```text
Merge rules 让 tokenizer 不必在“字符级”和“词级”之间二选一。
这就是子词 tokenizer 的核心游戏规则。
```

#### 解锁

```text
Symbol Keeper
Preserve Symbols challenge
```

---

### 5.7 1-2E Preserve Symbols

#### Concept Card

**标题：** 空格和标点也是信息  

**玩家可见文案：**

```text
Tokenizer 必须决定如何处理空格、换行、标点和大小写。

这些符号会影响 token 边界，
也会影响文本是否能被正确还原。
```

**图示：**

```text
"hello world"   vs   "helloworld"
"A"             vs   "a"
"line\nbreak"    vs   "line break"
```

**本阶段任务：**

```text
配置 Symbol Keeper，让 encode → decode 后能保持目标文本结构。
```

#### 画布状态

```text
[Raw Text] → [Symbol Keeper] → [Boundary Cutter] → [Decode Check]
```

#### 玩家操作

可切换策略：

```text
space_marker: keep_as_▁ / drop / split_only
case_policy: preserve / lowercase
newline_policy: keep / normalize_to_space
punct_policy: split / attach_left / attach_right
```

#### 验证规则

```text
✓ required spaces preserved
✓ punctuation not lost
✓ decode_check passes for visible text
✓ no illegal empty tokens
```

#### 失败反馈

```text
Decode mismatch:
Original: "hello, world"
Decoded:  "hello world"

The comma was lost during tokenization.
Preserve punctuation or add a punctuation token.
```

#### Debrief

```text
Tokenizer 不只是切词。
它定义了文字如何被稳定编码和还原。
```

#### 解锁

```text
1-3 Vocab Address
Vocab Table
```

---

## 6. 1-3 Vocab Address

### 6.1 大关定位

**核心问题：** token piece 如何变成模型可用的整数 ID？  
**最终目标：** 玩家理解 token ID 是 vocab / embedding table 的地址，并构建 `int[T]` / `int[B,T]` token buffer。  
**本关输出：**

```text
token_pieces[T] → token_ids[T]
```

### 6.2 本关节点

```text
Token Piece Strip
Vocab Table
Vocab Lookup Gate
Address Lighting
ID Stability Tester
Token Buffer Writer
Embedding Row Probe
```

---

### 6.3 1-3A Vocab Lookup

#### Concept Card

**标题：** Vocab Table：token 到 ID 的地图  

**玩家可见文案：**

```text
token piece 还是字符串片段。

Vocab Table 会给每个已知 token 一个稳定整数 ID。
这个 ID 才能进入模型数值模块。
```

**图示：**

```text
"train" → 1352
"ing"   → 278
"!"     → 9
```

**本阶段任务：**

```text
把 token pieces 接入 Vocab Table，查出对应 token IDs。
```

#### 画布状态

```text
[Token Pieces] → [Vocab Table] → [ID Stream]
```

#### 玩家操作

玩家把 token pieces 逐个拖到 Vocab Table：

```text
"we"     → 502
"train"  → 2841
"llm"    → 9172
```

#### 验证规则

```text
✓ every token piece looked up
✓ all pieces found or marked for fallback
✓ output IDs are integers
```

#### 失败反馈

```text
Lookup failed:
"tokenizers" is not in the current vocab.
Use fallback split or merge policy before vocab lookup.
```

#### Debrief

```text
Vocab Lookup 是从文本片段到整数空间的入口。
下一步你会看到这些 ID 如何点亮 embedding table 的行。
```

#### 解锁

```text
Address Lighting
Embedding Row Probe
```

---

### 6.4 1-3B Address Lighting

#### Concept Card

**标题：** Token ID 是 Embedding Table 的行地址  

**玩家可见文案：**

```text
Embedding Table 是一个大矩阵。

每个 token ID 指向其中一行。
查到 ID 后，模型就能取出这个 token 的向量。
```

**图示：**

```text
token_id = 502
      ↓
embedding_table[502]
      ↓
float vector[C]
```

**本阶段任务：**

```text
用 token ID 点亮 Embedding Table 中对应的行。
```

#### 画布状态

```text
[ID Stream] → [Address Lighting] → [Embedding Row Probe]
```

#### 玩家操作

玩家点击 token ID，Embedding Table 对应行发光：

```text
ID 502 → row 502 highlighted
ID 2841 → row 2841 highlighted
```

#### 验证规则

```text
✓ each ID maps to exactly one vocab row
✓ row index equals token_id
✓ selected row outputs float[C]
```

#### 失败反馈

如果玩家把 token order 当 row index：

```text
Address error:
Position T=1 is not the same as token ID.

T is where the token appears in sequence.
ID is which vocab row to read.
```

#### Debrief

```text
T 是序列位置，token ID 是词表地址。
这两个数字经常同时出现，但含义完全不同。
```

#### 解锁

```text
Stable ID Test
```

---

### 6.5 1-3C Stable ID Test

#### Concept Card

**标题：** 同一个 tokenizer 下，ID 必须稳定  

**玩家可见文案：**

```text
如果同一个 token 每次得到不同 ID，
Embedding Lookup 就会读取不同的行，模型输入会不稳定。

Tokenizer 必须是确定性的。
```

**图示：**

```text
"train" → 2841
"train" → 2841
"train" → 2841
```

**本阶段任务：**

```text
用同一组文本重复运行 tokenizer，验证 token ID 是否稳定。
```

#### 画布状态

```text
[Tokenizer Pipeline] → [Run x3] → [ID Stability Tester]
```

#### 玩家操作

玩家运行 3 次 trace。系统显示：

```text
run 1: [502, 2841, 9172]
run 2: [502, 2841, 9172]
run 3: [502, 2841, 9172]
```

#### 验证规则

```text
✓ same raw text produces same token pieces
✓ same token pieces produce same token IDs
✓ no randomized vocab lookup
```

#### 失败反馈

```text
Non-deterministic tokenizer detected:
The same token "train" produced IDs 2841 and 7902.

Tokenizer output must be stable for model training and inference.
```

#### Debrief

```text
Tokenizer 是模型合同的一部分。
它必须在训练和推理时保持一致。
```

#### 解锁

```text
Token Buffer Writer
```

---

### 6.6 1-3D Token Buffer Build

#### Concept Card

**标题：** Token IDs 要写入 Buffer  

**玩家可见文案：**

```text
一个 token ID 序列可以写成 int[T]。
多条样本组成 batch 后，会写成 int[B,T]。

这个 token buffer 是后续 Embedding Lookup 的直接输入。
```

**图示：**

```text
single sequence:
[502, 2841, 9172] → int[T]

batch:
[
  [502, 2841, 9172],
  [1042, 7191, 3910]
] → int[B,T]
```

**本阶段任务：**

```text
把 token IDs 按顺序写入 Token Buffer，并通过 int tensor 合同检查。
```

#### 画布状态

```text
[ID Stream] → [Token Buffer Writer] → [Token Buffer]
```

#### 玩家操作

玩家按顺序写入 ID，并设置轴标签：

```text
Axis 0: T for single sequence
or
Axis 0: B, Axis 1: T for batch mode
```

#### 验证规则

```text
✓ IDs preserve token order
✓ dtype = int
✓ shape = [T] in single mode
✓ shape = [B,T] in batch mode
✓ Embedding Lookup accepts buffer
```

#### 失败反馈

```text
Order mismatch:
Token IDs were written out of sequence.

Tokenizer output order defines T positions.
Do not sort IDs by numeric value.
```

#### Debrief

```text
你已经从 token pieces 得到了真正的模型入口：token_ids。
下一关要处理真实输入中的未知词、特殊 token 和 batch 长度问题。
```

#### 解锁

```text
1-4 Unknown & Buffer
Fallback Splitter
Special Token Injector
Padding Builder
```

---

## 7. 1-4 Unknown & Buffer

### 7.1 大关定位

**核心问题：** 真实输入不会总是刚好适配 vocab 和固定长度。  
**最终目标：** 玩家实现 fallback、special tokens、padding、attention mask、token budget，完成可批处理输入合同。  
**本关输出：**

```text
token_ids[B,T]
attention_mask[B,T]
special_token_positions
budget_status
```

### 7.2 本关节点

```text
OOV Detector
Fallback Splitter
Special Token Injector
Padding Builder
Attention Mask Builder
Truncation Gate
Token Budget Gate
Batch Token Buffer
```

---

### 7.3 1-4A OOV Failure

#### Concept Card

**标题：** OOV：词表中不存在的片段  

**玩家可见文案：**

```text
Vocab 只包含有限 token。

当一个 token piece 不在 vocab 中，
它不能直接变成 ID。

这叫 OOV / out-of-vocabulary。
```

**图示：**

```text
"we" → 502
"train" → 2841
"neurotokenizer" → ?
```

**本阶段任务：**

```text
让 OOV Detector 找出无法查表的 token pieces。
```

#### 画布状态

```text
[Token Pieces] → [Vocab Lookup] → [OOV Detector]
```

#### 玩家操作

玩家运行 lookup，OOV token 被标红：

```text
"neurotokenizer" not found
```

#### 验证规则

```text
✓ OOV pieces detected
✓ pipeline blocks before invalid ID emission
✓ player marks OOV for fallback
```

#### 失败反馈

如果玩家强行输出 ID：

```text
Invalid ID emission:
No vocab row exists for "neurotokenizer".
Use fallback split or <unk> policy.
```

#### Debrief

```text
真实输入总会遇到词表外片段。
一个可用 tokenizer 必须有 fallback 策略。
```

#### 解锁

```text
Fallback Splitter
<unk> token preview
```

---

### 7.4 1-4B Fallback Splitter

#### Concept Card

**标题：** Fallback：把未知片段拆到可编码  

**玩家可见文案：**

```text
当一个片段不在 vocab 中，
tokenizer 可以把它继续拆小。

常见 fallback 包括：
字符 fallback、子词 fallback、字节 fallback，
或者退回到 <unk>。
```

**图示：**

```text
"neurotokenizer"
→ [neuro] [token] [izer]
→ IDs found
```

**本阶段任务：**

```text
选择 fallback 策略，让所有片段都能映射到 ID。
```

#### 画布状态

```text
[OOV Piece] → [Fallback Splitter] → [Vocab Lookup]
```

#### 玩家操作

玩家选择策略：

```text
try_subword = true
try_char = true
use_unk_if_needed = false/true
```

输出：

```text
"neurotokenizer" → ["neuro", "token", "izer"]
```

#### 验证规则

```text
✓ all fallback pieces exist in vocab
✓ order preserved
✓ final ID stream contains no unresolved OOV
✓ if <unk> used, policy allows it
```

#### 失败反馈

如果 fallback 改变文本顺序：

```text
Fallback order error:
Subpieces must preserve original text order.
Tokenizer cannot reorder unknown text to fit vocab.
```

如果过度使用 `<unk>`：

```text
Coverage warning:
<unk> resolves the pipeline, but loses text information.
Try subword fallback first for this challenge.
```

#### Debrief

```text
Fallback 的目标不是猜意思，而是保证输入可以被稳定编码。
子词和字节级 fallback 能显著提升覆盖性。
```

#### 解锁

```text
Special Token Injector
```

---

### 7.5 1-4C Special Token Injector

#### Concept Card

**标题：** Special Tokens 也是 ID  

**玩家可见文案：**

```text
有些 token 不来自原始文本，
而是用于控制序列结构。

例如：
<bos> 表示开始
<eos> 表示结束
<pad> 表示补齐
<unk> 表示未知

它们也在 vocab 中，也有整数 ID。
```

**图示：**

```text
raw tokens: [we, train, llm]
with special:
[<bos>, we, train, llm, <eos>]
```

**本阶段任务：**

```text
在正确位置插入 BOS / EOS，并保持 token IDs 合同。
```

#### 画布状态

```text
[ID Stream] → [Special Token Injector] → [Specialized ID Stream]
```

Special Vocab：

```text
<pad> = 0
<bos> = 1
<eos> = 2
<unk> = 3
```

#### 玩家操作

玩家放置特殊 token：

```text
prepend <bos>
append <eos>
```

输出：

```text
tokens:    [<bos>, we, train, llm, <eos>]
token_ids: [1, 502, 2841, 9172, 2]
```

#### 验证规则

```text
✓ BOS at sequence start if required
✓ EOS at sequence end if required
✓ special tokens map to valid vocab IDs
✓ no duplicate EOS unless policy allows
```

#### 失败反馈

如果 `<eos>` 插在中间：

```text
Sequence control error:
<eos> marks end of sequence.
Tokens after <eos> may be ignored by downstream modules.
Move <eos> to the end.
```

如果特殊 token 当普通字符串查表失败：

```text
Special token handling error:
<bos> is not raw text.
It must be injected by Special Token Injector and looked up in special vocab.
```

#### Debrief

```text
Special tokens 是序列控制合同。
它们帮助模型知道开始、结束、补齐和未知位置。
```

#### 解锁

```text
Padding Builder
Attention Mask Builder
```

---

### 7.6 1-4D Padding Builder

#### Concept Card

**标题：** Batch 需要统一 T  

**玩家可见文案：**

```text
一个 batch 里可能有多条不同长度的文本。

为了组成 token_ids[B,T]，
它们必须被补齐到同一个 T。

补齐的位置使用 <pad>，
并用 attention_mask 标记哪些位置是真实 token。
```

**图示：**

```text
sample 0: [1, 502, 2841, 2]
sample 1: [1, 8910, 5521, 3401, 2]

pad to T=5:
[
  [1, 502, 2841, 2, 0],
  [1, 8910, 5521, 3401, 2]
]
```

**本阶段任务：**

```text
将不同长度序列 padding 成 token_ids[B,T]，并生成 attention_mask[B,T]。
```

#### 画布状态

```text
[Sequence ID Streams] → [Padding Builder] → [Batch Token Buffer]
                                    └→ [Attention Mask Builder]
```

#### 玩家操作

玩家设置：

```text
padding_side = right
pad_id = 0
max_T = max sequence length in batch
```

输出：

```text
token_ids[B,T]
attention_mask[B,T]
```

#### 验证规则

```text
✓ all batch rows same length T
✓ pad positions use pad_id
✓ attention_mask = 1 for real tokens
✓ attention_mask = 0 for pad tokens
✓ axis labels B/T correct
```

#### 失败反馈

如果没 padding：

```text
Batch assembly failed:
Rows have different lengths.
A tensor int[B,T] requires the same T for every sample.
Use <pad> to fill shorter sequences.
```

如果 mask 错：

```text
Mask mismatch:
Pad token at B=0,T=4 has attention_mask=1.
Pad positions should usually be masked out.
```

#### Debrief

```text
Padding 是把不规则文本变成规则 batch tensor 的关键步骤。
attention_mask 让后续 Attention / Loss 知道哪些位置是真实 token。
```

#### 解锁

```text
Token Budget Gate
Truncation Gate
```

---

### 7.7 1-4E Token Budget Gate

#### Concept Card

**标题：** 上下文窗口是 token 预算  

**玩家可见文案：**

```text
模型的上下文限制通常按 token 数计算。

如果一条序列超过最大 T，
它不能直接进入固定长度输入管线。

你需要选择：
减少切分 token、截断、或调整输入策略。
```

**图示：**

```text
max_T = 8
current T = 11
→ FAIL
```

**本阶段任务：**

```text
让 batch 中所有样本在插入 special tokens 后仍然不超过 max_T。
```

#### 画布状态

```text
[Specialized ID Streams] → [Token Budget Gate] → [Padding Builder]
```

#### 玩家操作

玩家可选策略：

```text
use_merge_to_reduce_T
truncate_right
truncate_middle
reject_input
increase_budget (locked in main challenge)
```

MVP 主线建议使用：

```text
merge first, then truncate if allowed
```

#### 验证规则

```text
✓ every sequence length <= max_T
✓ BOS/EOS preserved if required
✓ truncation policy explicit
✓ no invalid token IDs introduced
```

#### 失败反馈

```text
Budget failed:
Sequence length after special tokens = 11
max_T = 8

Remember: <bos> and <eos> also consume token positions.
```

如果截断掉 EOS：

```text
Truncation contract failed:
EOS was removed but this board requires EOS at sequence end.
Reserve one token slot for EOS.
```

#### Debrief

```text
Tokenizer 决定 T，而 T 决定上下文占用。
从现在开始，token budget 会成为训练和推理成本的一部分。
```

#### 解锁

```text
1-X Tokenizer Gauntlet
Full Pipeline Board
```

---

## 8. 1-X Tokenizer Gauntlet

### 8.1 综合考核定位

**核心目标：** 玩家独立搭建完整 Text → Token Pipeline，并通过未见输入的隐藏测试。

最终管线：

```text
Text Input
→ Symbol Keeper / Normalization
→ Boundary Cutter
→ Split Policy
→ Merge Forge
→ Vocab Lookup
→ Fallback Splitter
→ Special Token Injector
→ Token Budget Gate
→ Padding Builder
→ Token Buffer[int[B,T]]
→ Embedding Lookup Probe
```

### 8.2 Concept Card

**标题：** 从不规则文本到规则 token tensor  

**玩家可见文案：**

```text
真实输入是混乱的：
有空格、标点、大小写、未知词、多语言、emoji 和不同长度。

Tokenizer Pipeline 的目标是把它们稳定转换成：

token_ids[B,T]
attention_mask[B,T]

并让后续 Embedding Lookup 可以读取。
```

**本阶段任务：**

```text
修复完整 tokenizer 管线，通过 visible tests 和 hidden tests。
```

---

### 8.3 Visible Test Cases

```text
case 1:
"we train llm"

case 2:
"training tokenizers is useful!"

case 3:
"hello, world"

case 4:
["short", "a longer tokenization test"]
```

### 8.4 Hidden Test Cases

隐藏测试不展示完整答案，只展示失败 trace。

```text
hidden A: punctuation-heavy text
hidden B: mixed Chinese / English
hidden C: rare compound word
hidden D: emoji / unknown unicode
hidden E: sequence requiring padding
hidden F: sequence requiring budget handling
```

### 8.5 玩家操作

玩家需要完成：

```text
1. Raw text 不直接进入 Embedding。
2. Boundary / symbol rules 保证文本可切分。
3. Merge / fallback 保证所有 token piece 可查表。
4. Special tokens 被正确注入。
5. 超预算序列被明确处理。
6. 不同长度样本被 padding 到统一 T。
7. attention_mask 与 pad positions 对齐。
8. token_ids[B,T] 被 Embedding Lookup Probe 接受。
```

### 8.6 Autograder Checks

```text
✓ type_contract: raw text never enters numeric nodes directly
✓ token_piece_contract: no empty illegal token pieces
✓ vocab_contract: all final pieces have IDs
✓ fallback_contract: no unresolved OOV
✓ special_contract: BOS/EOS/PAD rules satisfied
✓ budget_contract: all sequences length <= max_T
✓ batch_contract: token_ids rank = 2 and axes = [B,T]
✓ dtype_contract: token_ids dtype = int
✓ mask_contract: attention_mask shape = [B,T]
✓ embedding_contract: Embedding Lookup accepts token_ids[B,T]
✓ determinism_contract: repeated run yields same IDs
```

### 8.7 失败反馈示例

#### raw text bypass

```text
Pipeline bypass detected:
Raw text is still connected to Embedding Lookup.
Insert tokenizer and vocab lookup before numeric model nodes.
```

#### unresolved OOV

```text
OOV unresolved:
"nanobotizer" has no vocab ID.
Fallback Splitter did not produce known subpieces.
```

#### budget failed

```text
Token budget exceeded:
B=1 sequence length = 13
max_T = 10

Try applying merge rules before truncation.
```

#### padding / mask mismatch

```text
Mask contract failed:
PAD token found at [B=0,T=6], but attention_mask is 1.
Pad positions should be masked out.
```

### 8.8 通关反馈

```text
Chapter 1 Complete: Text → Token

You built a working tokenizer input pipeline:
raw text → token pieces → token IDs → int[B,T]

Unlocked:
Chapter 2 — Embedding Lookup & Position Encoding
Tools: Token Buffer Viewer, Attention Mask Viewer, Embedding Row Probe
```

评分：

```text
Rank C: visible tests passed
Rank B: hidden tests passed
Rank A: hidden tests + no contract bypass
Rank S: hidden tests + no hints + deterministic trace + token budget optimized
```

---

## 9. 可选支线挑战

### 9.1 Bonus A — Decode Check

**目标：** 让玩家理解 encode/decode 可逆性不是默认保证。

```text
raw text → tokenize → token IDs → decode → text'
```

验证：

```text
✓ text' matches target policy
✓ spaces/punctuation preserved if required
```

失败：

```text
Decode mismatch:
Space marker was dropped.
Current tokenizer cannot reconstruct original spacing.
```

---

### 9.2 Bonus B — Token Cost Lab

**目标：** 让玩家观察不同文本类型下 token 数变化。

输入组：

```text
plain English
code snippet
Chinese sentence
emoji-heavy text
punctuation-heavy text
```

输出：

```text
token_count
compression_ratio
budget_status
```

---

### 9.3 Bonus C — Vocab Surgery

**目标：** 让玩家体验新增 vocab token 与 merge rule 的取舍。

玩法：

```text
Add token to vocab → token count reduced → vocab size increases
Add merge rule → token count reduced only when rule applies
```

约束：

```text
vocab slots <= limited budget
```

---

### 9.4 Bonus D — Multilingual Fallback Lab

**目标：** 让玩家处理混合语言和未知 unicode。

输入：

```text
"AI模型 test🙂"
```

挑战：

```text
CJK char fallback
ASCII subword merge
emoji byte fallback
```

---

## 10. 节点与工具设计

### 10.1 Text Input Node

```text
inputs: none
outputs: raw_text:string or text_batch[B]
inspector:
  char_count
  byte_count
  has_space
  has_punctuation
  has_unknown_unicode
```

视觉：文本卡片 / 文本纸带。

---

### 10.2 Text Inspector

作用：显示 raw text 的类型和字符结构。  
不输出 token，只输出诊断信息。

```text
inspect(raw_text) → metadata
```

---

### 10.3 Type Gate

作用：阻止 string 直接连接到 numeric modules。

```text
expected: int[T] / int[B,T]
received: string
```

视觉：合同闸门，非法连接红灯。

---

### 10.4 Tokenizer Socket

最小 tokenizer adapter，负责：

```text
raw_text:string → token_pieces[T]:string[]
```

MVP 初始策略：whitespace + punctuation split。

---

### 10.5 Boundary Cutter

负责初始切分：

```text
split_on_space
split_punctuation
keep_space_marker
newline_policy
case_policy
```

---

### 10.6 Split Policy Switch

切换策略：

```text
char
word
subword
byte fallback
```

每种策略输出 token count 和 OOV risk。

---

### 10.7 Token Count Meter

显示：

```text
T
budget
cost indicator
```

可附加：

```text
attention_cost_preview ≈ T²
```

此处只做预告，不深入 attention。

---

### 10.8 Merge Forge

使用 ranked merge table。

```text
input: pieces[T]
merge_rules: [(a,b)->ab]
output: merged_pieces[T']
```

检查：合并结果是否在 vocab 中。

---

### 10.9 Vocab Table

```text
token_string → token_id:int
```

显示字段：

```text
token
id
is_special
frequency_hint optional
```

---

### 10.10 Vocab Lookup Gate

检查每个 token piece 是否存在于 vocab。

```text
found → id
not found → OOV
```

---

### 10.11 Fallback Splitter

策略：

```text
subword fallback
char fallback
byte fallback
unk fallback
```

主线建议默认：subword → char → unk。

---

### 10.12 Special Token Injector

插入：

```text
<bos>
<eos>
<pad>
<unk>
```

注意 `<pad>` 通常由 Padding Builder 插入，而不是直接加入每条原始序列。

---

### 10.13 Padding Builder

```text
input: list[int[T_i]]
output: int[B,T_max]
```

配置：

```text
padding_side = left/right
pad_id = 0
max_T
```

---

### 10.14 Attention Mask Builder

```text
attention_mask[b,t] = 1 if token_ids[b,t] != pad_id else 0
```

输出：

```text
int[B,T] or bool[B,T]
```

---

### 10.15 Token Buffer

最终输出节点。

显示：

```text
dtype: int
shape: [B,T]
axis_labels: B/T
tokens debug view
token_ids numeric view
mask overlay
```

---

### 10.16 Embedding Lookup Probe

只做输入合同验证，不在本章展开 embedding 细节。

```text
accepts: int[B,T]
rejects: string / token_piece[] / ragged list
```

---

## 11. 数据与配置建议

### 11.1 Toy Vocab 示例

```json
{
  "<pad>": 0,
  "<bos>": 1,
  "<eos>": 2,
  "<unk>": 3,
  "▁": 4,
  "!": 5,
  ",": 6,
  ".": 7,
  "we": 502,
  "train": 2841,
  "llm": 9172,
  "shape": 1042,
  "token": 8910,
  "izer": 4412,
  "tokenizer": 5521,
  "tokenizers": 5522,
  "ing": 278,
  "training": 2842,
  "use": 3301,
  "ful": 3302,
  "useful": 3303
}
```

### 11.2 Merge Table 示例

```json
[
  ["train", "ing", "training"],
  ["token", "izer", "tokenizer"],
  ["tokenizer", "s", "tokenizers"],
  ["use", "ful", "useful"]
]
```

### 11.3 Visible Text Set

```text
"we train llm"
"training tokenizers is useful!"
"hello, world"
"shape tells truth"
```

### 11.4 Hidden Text Set

```text
"tokenization changes cost"
"AI模型 test🙂"
"line\nbreak test"
"unknownwordizer"
"shape→token→id"
```

---

## 12. Autograder 规则库

### 12.1 Type Contract

```ts
function checkNoRawTextIntoNumeric(board) {
  return !board.edges.some(e =>
    e.source.type === "raw_text" && e.target.acceptsNumeric === true
  );
}
```

### 12.2 Token Piece Contract

```ts
function checkTokenPieces(pieces) {
  return pieces.length > 0 && pieces.every(p => typeof p === "string" && p.length > 0);
}
```

### 12.3 Vocab Contract

```ts
function checkAllPiecesResolvable(pieces, vocab, fallbackPolicy) {
  return pieces.every(p => vocab[p] !== undefined || fallbackPolicy.canResolve(p));
}
```

### 12.4 Determinism Contract

```ts
function checkDeterminism(tokenizer, text) {
  const a = tokenizer.encode(text);
  const b = tokenizer.encode(text);
  return JSON.stringify(a.ids) === JSON.stringify(b.ids);
}
```

### 12.5 Batch Contract

```ts
function checkBatchTensor(tokenIds, attentionMask) {
  const B = tokenIds.length;
  const T = tokenIds[0].length;
  return tokenIds.every(row => row.length === T)
      && attentionMask.length === B
      && attentionMask.every(row => row.length === T)
      && tokenIds.flat().every(Number.isInteger);
}
```

### 12.6 Mask Contract

```ts
function checkMaskMatchesPad(tokenIds, mask, padId = 0) {
  for (let b = 0; b < tokenIds.length; b++) {
    for (let t = 0; t < tokenIds[b].length; t++) {
      const expected = tokenIds[b][t] === padId ? 0 : 1;
      if (mask[b][t] !== expected) return false;
    }
  }
  return true;
}
```

---

## 13. 关卡状态机

```ts
type Chapter1Stage =
  | "chapter_intro"
  | "1-1A_raw_text_object"
  | "1-1B_type_gate_failure"
  | "1-1C_tokenizer_socket"
  | "1-1D_token_id_contract"
  | "1-2A_boundary_cutter"
  | "1-2B_split_comparison"
  | "1-2C_token_count_meter"
  | "1-2D_merge_forge"
  | "1-2E_preserve_symbols"
  | "1-3A_vocab_lookup"
  | "1-3B_address_lighting"
  | "1-3C_stable_id_test"
  | "1-3D_token_buffer_build"
  | "1-4A_oov_failure"
  | "1-4B_fallback_splitter"
  | "1-4C_special_token_injector"
  | "1-4D_padding_builder"
  | "1-4E_token_budget_gate"
  | "1-X_tokenizer_gauntlet"
  | "chapter_complete";
```

阶段进入规则：

```text
每个 stage 进入时显示 Concept Card。
第一次失败后显示 Failure Lesson。
通过 visible checks 后显示 Debrief。
通过大关最后阶段后解锁下一大关。
```

---

## 14. 关卡配置示例

```ts
const level_1_1 = {
  id: "1-1",
  title: "Text Cannot Flow",
  goal: "Repair raw text into token IDs before Embedding Lookup.",
  stages: [
    {
      id: "1-1A",
      title: "Raw Text Object",
      concept: {
        title: "Raw Text 不是 Tensor",
        body: "Raw text 的类型是 string。它可以被查看和切分，但不能直接进入 Embedding Lookup。"
      },
      nodes: ["TextInput", "TextInspector"],
      checks: ["raw_text_inspected", "type_is_string", "not_tensor"]
    },
    {
      id: "1-1B",
      title: "Type Gate Failure",
      concept: {
        title: "模型节点有输入合同",
        body: "Embedding Lookup 需要 int token IDs，而不是 raw text。"
      },
      nodes: ["TextInput", "EmbeddingLookupProbe", "TypeGate"],
      requiredFailure: "raw_text_to_embedding",
      checks: ["type_gate_triggered", "error_trace_shown"]
    },
    {
      id: "1-1C",
      title: "Tokenizer Socket",
      nodes: ["TextInput", "TokenizerSocket", "TokenIdEmitter"],
      checks: ["tokenizer_inserted", "token_pieces_emitted"]
    },
    {
      id: "1-1D",
      title: "Token ID Contract",
      nodes: ["TokenIdEmitter", "EmbeddingLookupProbe"],
      checks: ["ids_are_int", "embedding_accepts_ids"]
    }
  ],
  rewards: ["Boundary Cutter", "Token Count Meter", "Chapter 1-2"]
};
```

---

## 15. UI / 视觉设计补充

### 15.1 Token Piece 表现

Token piece 使用可拖拽 chip：

```text
[we] [train] [llm] [!]
```

每个 chip 有三种视图：

```text
string view: "train"
id view: 2841
status view: found / OOV / special / fallback
```

### 15.2 Vocab Table 表现

Vocab Table 像一个可滚动地址表：

```text
ID       TOKEN       TYPE
0        <pad>       special
1        <bos>       special
502      we          normal
2841     train       normal
```

查表时，token chip 发出连线点亮 row。

### 15.3 Token Buffer 表现

Batch Token Buffer 是一个二维网格：

```text
          T0      T1      T2      T3      T4
B0       <bos>   we      train   llm     <eos>
B1       <bos>   shape   tells   truth   <eos>
```

可以切换显示：

```text
token text / token id / attention mask / pad overlay
```

### 15.4 数据线颜色

```text
白色：raw text / string
蓝色：token pieces / string token stream
青色：token IDs / int tensor
紫色：special token / control symbol
绿色：validated output
橙色：warning / fallback path
红色：contract failure
```

---

## 16. MVP 实现建议

### 16.1 第一版必做

```text
1. Text Input / Text Inspector
2. Type Gate failure
3. Tokenizer Socket with simple split
4. Token piece chips
5. Toy Vocab Lookup
6. OOV detection
7. Fallback split to chars or <unk>
8. Special token insertion BOS/EOS/PAD
9. Padding Builder
10. Token Buffer[int[B,T]]
11. Attention mask display
12. Visible + hidden tests
```

### 16.2 第二版增强

```text
1. Merge Forge with ranked merge table
2. Decode Check
3. Multilingual fallback
4. Token Cost Lab
5. Vocab Surgery
6. Tokenizer determinism test visualization
7. S/A rank scoring
```

### 16.3 暂不建议主线实现

```text
完整真实 BPE 训练
SentencePiece Unigram LM
真实 GPT-2 byte-level BPE 的所有空格规则
大规模 vocab 管理
复杂 Unicode normalization 全规则
```

这些可以作为高级 DLC / Appendix / Sandbox。

---

## 17. 失败信息库

### 17.1 raw text 直接进模型

```text
Connection failed:
Embedding Lookup expects integer token IDs.
Raw text is a string object.
Insert Tokenizer and Vocab Lookup first.
```

### 17.2 token piece 未查表

```text
Contract incomplete:
Token pieces are still strings.
Use Vocab Table to emit token IDs.
```

### 17.3 OOV 未处理

```text
OOV unresolved:
This token piece has no vocab row.
Use Fallback Splitter or <unk> policy.
```

### 17.4 token 数超预算

```text
Token budget exceeded:
Current T is larger than max_T.
Apply merge rules, truncation, or reject the input.
```

### 17.5 padding 缺失

```text
Batch build failed:
Sequences have different lengths.
Use Padding Builder to create rectangular int[B,T].
```

### 17.6 mask 错误

```text
Attention mask mismatch:
PAD positions should be masked out.
Check pad_id and mask generation rule.
```

### 17.7 ID 不稳定

```text
Determinism failed:
Same input produced different token IDs across runs.
Tokenizer must be stable.
```

---

## 18. 与后续章节的衔接

第 1 章完成后，玩家得到：

```text
token_ids[B,T]
attention_mask[B,T]
```

第 2 章可以自然进入：

```text
Token IDs → Embedding Lookup
```

衔接点：

```text
1. token_id 点亮 embedding table 行。
2. token_ids[B,T] 通过查表变成 hidden[B,T,C]。
3. attention_mask[B,T] 会在 Attention 章节中参与 mask。
4. padding / EOS / token budget 会在 Training Data Pipeline 中影响 loss 计算。
```

推荐下一章名称：

```text
Chapter 2 — Embedding Lookup & Position Encoding
```

开场任务：

```text
Use token_ids[B,T] to read embedding_table[V,C],
producing hidden[B,T,C].
```

---

## 19. 制作任务拆分

### 19.1 设计 / 关卡

```text
- 制作 1-1 到 1-X 的 stage 配置
- 编写 Concept Card 文案
- 编写 visible / hidden test cases
- 编写失败信息库
- 设计评分规则
```

### 19.2 前端 / UI

```text
- Token chip 拖拽与状态显示
- Text Inspector
- Vocab Table 地址点亮
- Token Buffer 网格
- Attention Mask overlay
- Concept Card / Debrief / Failure Lesson UI
```

### 19.3 引擎 / 逻辑

```text
- Toy tokenizer implementation
- Merge table application
- Vocab lookup
- Fallback splitter
- Special token injection
- Padding builder
- Autograder rules
- Determinism test runner
```

### 19.4 可视化

```text
- Raw text string flow
- Token split animation
- Merge animation
- Vocab row lighting
- Token ID stream writing
- Padding / mask overlay
```

---

## 20. 总结

第 1 章不应该只让玩家知道“tokenizer 会把文本变成 token”。  
它应该让玩家亲手修复一条完整输入管线，并在每个阶段看到工程原因：

```text
raw text 不能直接进模型
→ tokenizer 负责切分
→ token piece 需要查 vocab
→ ID 是 embedding 地址
→ OOV 需要 fallback
→ special token 管理序列结构
→ batch 需要 padding
→ 最终输出 int[B,T]
```

这套结构保持了 Chapter 0 的硬核可验证体验，同时把节奏从“概念教学”推进到“真实模型数据入口构建”。

玩家通关后的核心掌握不是一句定义，而是一条可以运行、可以调试、可以验证的工程链路：

```text
Text Batch → Tokenizer Pipeline → token_ids[B,T] → Embedding Ready
```
