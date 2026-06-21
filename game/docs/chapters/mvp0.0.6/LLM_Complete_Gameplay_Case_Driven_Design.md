再次检查后，我的判断是：**这一轮改进方向是对的，已经明显从“无说明的可视化 IDE”往“维修式关卡”推进了；但试玩仍觉得抽象，是因为现在的核心表达仍然是“公式 / shape / contract 驱动”，还没有充分变成“具体输入数据 → 玩家观察结果 → 修图 → 代码对应”的体验。**

你的判断很准确：**应该引入具体数据和案例驱动，而且应该同步显示 graph 对应的真实代码。** 这两个改动会非常关键。

---

# 1. 当前提交已经解决了不少上轮问题

最新版本已经有明显进步：

第一，Graph Challenge 现在不是直接从 0-2 MatMul 开始，而是加了 `0-G Graph Basics`，关卡顺序也改成了 `Graph Basics → Text Type Gate → Split / Merge Budget → MatMul...`，比之前合理很多。

第二，默认入口已经改回 `guided`，Graph Challenge 变成可切换模式，同时也加了中英文切换。

第三，`LevelSpec` 已经加入了 `onboarding` 和 `targetGraph`，这说明现在关卡可以表达任务故事、初始问题、第一步行动、目标步骤、胜利条件和目标图。

第四，Graph UI 里已经加入 Mission Modal、Target Ghost Graph、Checklist、Next Step Coach、定位失败节点、可调整布局等，这比上一版强很多。

第五，结构错误现在也会生成 trace frame，不再是失败后 trace 为空，这个改动非常重要。

所以这轮不是白改，基础可玩性已经比上一版强。但现在的问题变成了更深一层：**玩家知道要点按钮了，但还不知道“这些节点到底在处理什么东西”。**

---

# 2. 为什么现在还是抽象

当前 0-2 MatMul 的目标仍然是：

```text
Build hidden[B,T,C] @ stored_weight[O,C].transpose() -> projected[B,T,O]
```

这在代码里就是关卡 goal。

Mission 里虽然比之前具体了一点，但核心表达还是：

```text
stored_weight is [O,C]. MatMul needs its right input to behave like [C,O].
```

这对有机器学习背景的人是清楚的，对普通玩家仍然像抽象规则。

Tokenizer 这边其实已经有很好的具体文本数据，比如：

```text
"we train llm."
"tokenizers are useful!"
"shape tells truth"
"unknown glyph ?"
"中文 mixed text!"
"code_snake_case + punctuation"
```

这些都已经写在关卡数据里。

而且 `TokenizerSocket` 运行时已经能输出 `tokens / ids / mask` samples。 右侧 Values tab 也已经能展示 trace frame 的 samples。

问题是：**这些具体数据现在藏在测试输入和 trace 里，玩家一开始看不到。**
所以玩家仍然先看到“policy、fallback、maskPolicy、shape、B/T/C”，而不是先看到：

```text
输入文本：
tokenizers are useful!

当前切分：
t o k e n i z e r s ...  太长了

目标：
在 T <= 8 的预算内，把它变成更稳定的 token ids
```

也就是说，当前是：

```text
公式 / contract → graph → 测试 → trace data
```

应该改成：

```text
具体案例 → 观察失败结果 → graph 修复 → 代码对应 → 抽象公式总结
```

---

# 3. 建议改成“Case-first”关卡结构

我建议给 Graph Challenge 新增一层概念：**Case Study / 当前样本案例**。

每一关不要先展示公式，而是先展示一个具体机器问题。

## 新的关卡进入顺序

```text
1. 这关正在处理什么具体数据？
2. 当前机器输出了什么错误结果？
3. 玩家要修哪一个 graph 局部？
4. 修完后数据如何变化？
5. 对应真实代码是什么？
6. 最后再总结 shape / formula / contract
```

这会显著降低抽象感。

---

# 4. Tokenizer 关卡最适合先改

你提到的“切分 token 时给一段真正文本”非常对。Chapter 1 应该成为最先落地 Case-first 的章节，因为它天然有可读文本，普通玩家最容易理解。

## 以 1-2 Split / Merge Budget 为例

当前关卡目标是：

```text
Tune tokenizer policy and merges so useful pieces stay intact while T stays within budget.
```

并且 visible tests 已经用真实文本集合。

建议改成这种界面：

---

## 1-2 关卡 Mission 改写

### 当前案例

```text
模型收到三条训练文本：

1. we train llm.
2. tokenizers are useful!
3. shape tells truth

模型最多只能看 T = 8 个 token。
现在 tokenizer 使用 char split，虽然不会 OOV，但 token 数太多。
```

### 当前失败

```text
文本：tokenizers are useful!

char split 结果：
t o k e n i z e r s   a r e   u s e f u l !
token count = 23
预算 T = 8
失败：token 太多
```

### 玩家任务

```text
把切分策略从 Character 改成 Subword
打开 Apply merges
保持 fallback = <unk>
再次运行 Visible
```

### 通过后显示

```text
subword + merges 结果：
tokenizers / are / useful / !
token count = 4
通过 T <= 8
```

---

## UI 上新增一个 Case Panel

建议在 Mission Modal 或画布上方常驻一个 **Case Panel**：

```text
Case: "tokenizers are useful!"

Raw Text
tokenizers are useful!

Current Pieces
["t", "o", "k", "e", "n", "i", "z", "e", "r", "s", ...]

Token IDs
[17, 21, 14, 8, 13, ...]

Attention Mask
[1, 1, 1, 1, 1, 1, 1, 1]

Budget
23 / 8  ❌
```

玩家修改 tokenizer 参数后，这个 panel 立即更新：

```text
Current Pieces
["tokenizers", "are", "useful", "!"]

Token IDs
[31, 8, 42, 5, 0, 0, 0, 0]

Attention Mask
[1, 1, 1, 1, 0, 0, 0, 0]

Budget
4 / 8  ✅
```

这比让玩家读 `token_budget` assertion 更直观。

---

# 5. 把公式降级为“总结”，不要作为入口

不是说公式不要。公式很重要，但应该在玩家已经看过具体数据后出现。

例如 0-2 MatMul，不要一开始就说：

```text
hidden[B,T,C] @ stored_weight[O,C].transpose() -> projected[B,T,O]
```

可以改成：

```text
你正在修一个“每个 token 打 5 个分数”的小机器。

输入：
2 条样本
每条 4 个 token
每个 token 有 3 个特征

所以 hidden 是：
2 × 4 × 3

权重板里有：
5 个输出分数
每个分数需要看 3 个特征

所以 weight 当前是：
5 × 3

但 MatMul 需要：
3 × 5

现在 weight 板横着插进去了，需要先转一下方向。
```

然后在最后 debrief 再给公式：

```text
抽象成公式就是：
hidden[B,T,C] @ W[C,O] -> projected[B,T,O]
```

这会让玩家先建立感性图像，再接受符号。

---

# 6. 每一类关卡都可以改成具体案例

## 0-G Graph Basics

这个现在已经不错，但还可以更具象：

```text
输入是一盒数字积木：
shape = [1,2,3]

现在这盒积木没有送到检查门。
把 input.out 接到 shape_gate.x。
```

这个不需要再讲 formula。

---

## 1-1 Text Type Gate

改成：

```text
文本：
"we train llm"

错误机器：
Text Input 直接接 Embedding Ready Probe

失败原因：
Embedding 不认识英文字符串。
它只认识整数 ID。

修复：
中间插入 Tokenizer Socket。
```

对应数据：

```text
raw text:
"we train llm"

token pieces:
["we", "train", "llm"]

token ids:
[12, 44, 91, 0]
```

---

## 1-2 Split / Merge Budget

如上，用真实文本驱动：

```text
"tokenizers are useful!"
```

核心体验是比较三种策略：

```text
Character：太长
Word：短，但 hidden case 容易 OOV
Subword + merges：平衡
```

---

## 1-3 OOV / Fallback

用这个案例：

```text
visible:
"we train llm!"

hidden:
"unseen glyph ?"
```

界面显示：

```text
unseen → OOV
glyph → OOV
? → known punctuation

fallback = none:
失败

fallback = <unk>:
["<unk>", "<unk>", "?"]
通过
```

---

## 1-4 Padding / Mask

用两条长度不同的文本：

```text
sample 0: "we train llm"
sample 1: "shape"
```

界面显示二维表：

```text
token_ids[B,T]

B0  <bos> we train llm <eos> <pad> <pad> <pad>
B1  <bos> shape <eos> <pad> <pad> <pad> <pad> <pad>

attention_mask[B,T]

B0  1 1 1 1 1 0 0 0
B1  1 1 1 0 0 0 0 0
```

当前 `TokenizerSocket` 已经能输出 `ids` 和 `mask`，所以这部分不是从零做，只是把 samples 可视化。

---

## 0-2 MatMul

用真实小矩阵，不要先讲抽象 shape：

```text
一个 token 的 hidden vector：
[0.20, -0.10, 0.50]

一条输出权重：
[0.30, 0.70, -0.20]

Dot cell：
0.20*0.30 + (-0.10)*0.70 + 0.50*(-0.20)
```

然后再扩展：

```text
一共有 2 条样本 × 4 个 token。
每个 token 都做同样的 5 个 dot cells。
```

---

## 0-3 Transpose

用 Q/K 的具体 token 案例：

```text
Query token i = "train"
Key token j = "llm"

score[i,j] 应该是：
dot(Q["train"], K["llm"])
```

当前 `CellTrace` 已经能展示 QK cell 的公式、qVector、kVector、expectedDot、actualScore、delta。 这很适合做成玩家看到的主画面，而不是只藏在 Values tab。

---

## 0-4 Mask

不要先讲 `scores[B,H,T,T] + causal_mask[T,T]`。

先显示一张 token-to-token 表：

```text
tokens:
0 <bos>
1 we
2 train
3 llm

query \ key
        <bos>   we    train   llm
<bos>     OK     X       X      X
we        OK     OK      X      X
train     OK     OK      OK     X
llm       OK     OK      OK     OK
```

玩家一眼能理解：**每个位置只能看自己和过去，不能偷看未来。**

然后再让他调 `maskOrientation`。

当前 `CausalMask` 有 `maskOrientation` 参数，且参数选项已经写了 `query_key` 和 `key_query` 的后果。 这部分很好，只是需要把它图形化成表格。

---

# 7. 是否应该同步显示 graph 对应的真实代码？

**应该，而且这会非常有价值。**

当前已有 Code tab，但它显示的是单个 module 的 `pseudoCode`：

```tsx
<pre className="graphCodeBlock">
  {module.pseudoCode ?? "No pseudo code registered for this module."}
</pre>
```

也就是说，现在的 Code tab 是“节点级伪代码”，不是“当前 graph 的真实代码”。

`ModuleDef` 里也只是有可选的 `pseudoCode` 字段。 例如 MatMul Gate 当前 pseudoCode 是：

```text
out[..., o] = sum_c left[..., c] * right[c, o]
qk[..., i, j] = sum_d Q[..., i, d] * Kt[..., d, j]
```

这仍然偏公式化。

建议新增一个 **Graph Code Panel**，展示整个 graph 对应的真实代码，并且随当前 graph、参数和测试 case 同步变化。

---

# 8. Graph Code Panel 应该显示什么

建议不要一开始就显示复杂 TypeScript runtime 代码，而是显示“学习者能理解的 PyTorch-like 代码”。

例如 1-2 Split / Merge Budget：

```python
texts = [
    "we train llm.",
    "tokenizers are useful!",
    "shape tells truth",
]

tokenizer = ToyTokenizer(
    policy="char",
    apply_merges=False,
    fallback="<unk>",
    max_length=8,
    pad_to_length=8,
    pad_side="right",
    mask_policy="pad-aware",
)

token_ids, attention_mask = tokenizer.encode_batch(texts)

assert token_ids.shape == [3, 8]
assert attention_mask.shape == [3, 8]
assert token_count <= 8
```

当玩家把 `policy` 改成 `subword`、打开 `applyMerges` 后，代码同步变成：

```python
tokenizer = ToyTokenizer(
    policy="subword",
    apply_merges=True,
    fallback="<unk>",
    max_length=8,
    pad_to_length=8,
    pad_side="right",
    mask_policy="pad-aware",
)
```

这样玩家会立即理解：**graph 的操作不是抽象 UI，它对应真实代码。**

---

# 9. 0-2 MatMul 的同步代码示例

当前 graph：

```text
hidden.out -> matmul.left
weight.out -> matmul.right
```

对应代码可以显示：

```python
hidden = Tensor("hidden", shape=[2, 4, 3], axes=["B", "T", "C"])
stored_weight = Tensor("weight", shape=[5, 3], axes=["O", "C"])

# 当前错误连接
projected = matmul(hidden, stored_weight)

# 失败：
# hidden last dim = 3
# stored_weight first dim = 5
```

玩家插入 Transpose 后：

```python
hidden = Tensor("hidden", shape=[2, 4, 3], axes=["B", "T", "C"])
stored_weight = Tensor("weight", shape=[5, 3], axes=["O", "C"])

compute_weight = stored_weight.transpose(0, 1)  # [O,C] -> [C,O]
projected = matmul(hidden, compute_weight)      # [B,T,C] @ [C,O]

assert projected.shape == [2, 4, 5]
assert allclose(projected, reference)
```

这比单独显示 `out[..., o] = sum_c ...` 更有教学价值。

---

# 10. 建议新增的数据结构

可以在 `types.ts` 里扩展：

```ts
export type LevelCaseStudy = {
  title: string;
  narrative: string;
  visibleInputFocus?: string;
  dataPanels: CaseDataPanel[];
  playerQuestion: string;
  successObservation: string;
};

export type CaseDataPanel =
  | {
      type: "text_batch";
      title: string;
      inputKey: string;
      focusText?: string;
    }
  | {
      type: "tokenizer_preview";
      title: string;
      tokenizerNodeId: string;
      textInputKey: string;
    }
  | {
      type: "tensor_preview";
      title: string;
      inputKey: string;
      maxRows?: number;
      maxCols?: number;
    }
  | {
      type: "attention_table";
      title: string;
      tokensInputKey?: string;
      scoresNodeId: string;
    };

export type LevelSpec = {
  ...
  caseStudy?: LevelCaseStudy;
};
```

然后 `GraphMissionModal` 不再只显示 onboarding 文案，而是增加：

```tsx
<GraphCasePanel
  level={selectedLevel}
  graph={graph}
  runState={runState}
  activeVisibleCase={selectedLevel.visibleTests[0]}
/>
```

---

# 11. Graph Code 的数据结构

建议给 module 增加代码生成能力，而不是只保留 `pseudoCode`：

```ts
export type CodeEmitContext = {
  node: GraphNode;
  module: ModuleDef;
  inputVars: Record<string, string>;
  outputVars: Record<string, string>;
  testInputs: Record<string, RuntimeValue>;
  language: "python" | "typescript";
};

export type CodeEmitResult = {
  lines: CodeLine[];
};

export type CodeLine = {
  id: string;
  nodeId?: string;
  edgeId?: string;
  text: string;
  highlightWhenSelected?: boolean;
};

export type ModuleDef = {
  ...
  pseudoCode?: string;
  emitCode?: (context: CodeEmitContext) => CodeEmitResult;
};
```

再新增：

```ts
generateGraphCode(graph, registry, testCase, language)
```

生成顺序直接复用 runtime 的拓扑顺序。当前 `executeGraph` 已经会按 topological order 执行节点。 代码生成也应当走同一个顺序，这样 Trace Step 和 Code Line 可以一一对应。

---

# 12. Code Panel 的交互方式

建议分三层：

## A. Case Code

显示当前 visible case 的输入：

```python
texts = ["tokenizers are useful!"]
```

或：

```python
hidden.shape = [2, 4, 3]
weight.shape = [5, 3]
```

## B. Graph Code

显示 graph 当前连接生成的代码：

```python
pieces = tokenizer.split(texts)
token_ids, mask = tokenizer.encode_batch(texts)
embedding_ready(token_ids)
```

或：

```python
weight_T = transpose(weight, 0, 1)
projected = matmul(hidden, weight_T)
```

## C. Test Code

显示当前测试为什么通过/失败：

```python
assert token_ids.shape == [B, T]
assert no_oov(tokens)
assert attention_mask[pad_positions] == 0
```

这样玩家能同时理解：

```text
输入是什么
graph 做了什么
测试检查什么
```

---

# 13. Trace 与 Code 应该互相高亮

这是提升“懂了”的关键。

当玩家点击 Trace Step：

```text
step 2 tokenizer
```

右侧 Code Panel 高亮：

```python
token_ids, attention_mask = tokenizer.encode_batch(texts)
```

当玩家点击代码行：

```python
compute_weight = stored_weight.transpose(0, 1)
```

画布高亮：

```text
weight -> weight_transpose -> matmul.right
```

这会把 graph、数据、代码三者绑定起来。

---

# 14. 当前 Code Tab 建议改造

不要删当前 Code tab，但要分成：

```text
Node Code
Graph Code
Test Code
```

当前的 `module.pseudoCode` 可以放到 `Node Code` 里。

新增 `Graph Code`：

```text
根据当前 graph 自动生成
```

新增 `Test Code`：

```text
根据当前 visible / hidden assertions 自动生成
```

这样既保留已有功能，又能解决“同步显示真正代码”的需求。

---

# 15. 推荐优先实现顺序

## P0：Case-first Tokenizer 改造

先只改 1-2 Split / Merge Budget。

因为这关数据最直观，而且已有真实文本和 tokenizer samples。

新增：

```text
Case Panel
Token pieces preview
Token IDs preview
Mask preview
Budget meter
```

验收标准：

```text
普通玩家能说出：
“char 切得太碎，所以超预算；subword merge 后 token 少了。”
```

---

## P1：Graph Code Panel

先支持这几个模块：

```text
TextInput
TokenizerSocket
EmbeddingReadyProbe
InputTensor
WeightPlate
TransposeSwitch
MatMulGate
OutputContractGate
ReferenceChecker
```

不用一口气覆盖所有模块。

验收标准：

```text
玩家改 tokenizer policy 或插入 transpose 后，代码同步变化。
```

---

## P2：0-2 MatMul Case Panel

给 0-2 增加具体 tensor preview：

```text
hidden: 2 samples × 4 tokens × 3 features
weight: 5 output channels × 3 features
```

显示一个 token vector 和一条 weight vector 的 dot product。

验收标准：

```text
普通玩家能说出：
“weight 要转一下，是因为 MatMul 右边要先对齐 3 个特征。”
```

---

## P3：Attention Table / Mask Table

给 0-4F 增加 token-to-token mask 表：

```text
query token 只能看过去 key token
```

验收标准：

```text
普通玩家能肉眼看出 key_query 是反了。
```

---

# 16. 一个具体落地方案：先改 1-2

### 新增 caseStudy

```ts
caseStudy: {
  title: "Make this sentence fit into 8 token slots",
  narrative: "The model can only read T=8 token slots. Character split is safe but too long.",
  visibleInputFocus: "tokenizers are useful!",
  dataPanels: [
    {
      type: "text_batch",
      title: "Training texts",
      inputKey: "texts",
      focusText: "tokenizers are useful!"
    },
    {
      type: "tokenizer_preview",
      title: "Current tokenizer output",
      tokenizerNodeId: "tokenizer",
      textInputKey: "texts"
    }
  ],
  playerQuestion: "Which split strategy keeps useful words while staying under T=8?",
  successObservation: "Subword + merges keeps tokenizers/useful intact and stays within budget."
}
```

### 新增 Case Panel 展示

```text
Text:
tokenizers are useful!

Current strategy:
Character split

Pieces:
t | o | k | e | n | i | z | e | r | s | ...

Budget:
23 / 8 ❌

Try:
Split strategy = Subword
Apply merges = On
```

### 右侧 Code Panel

```python
texts = [
    "we train llm.",
    "tokenizers are useful!",
    "shape tells truth",
]

tokenizer = ToyTokenizer(
    policy="char",
    apply_merges=False,
    fallback="<unk>",
    max_length=8,
    pad_to_length=8,
)

token_ids, attention_mask = tokenizer.encode_batch(texts)

assert token_ids.shape == [3, 8]
assert no_oov(token_ids)
assert token_count <= 8
```

玩家改完参数后自动变：

```python
tokenizer = ToyTokenizer(
    policy="subword",
    apply_merges=True,
    fallback="<unk>",
    max_length=8,
    pad_to_length=8,
)
```

---

# 17. 设计原则总结

接下来不要再继续强化公式表达，而是按这个优先级组织信息：

```text
第一层：具体案例
玩家看得懂的文本、数字、表格、token、mask。

第二层：graph 操作
我需要连哪条线、改哪个参数、插哪个模块。

第三层：运行结果
这条文本怎么被切分，哪个 token 超预算，哪个矩阵维度不匹配。

第四层：代码
当前 graph 等价于哪几行 PyTorch-like 代码。

第五层：抽象公式
hidden[B,T,C] @ W[C,O] -> out[B,T,O]
```

现在的版本大体是从第五层开始讲，所以普通玩家会觉得抽象。把入口换成第一层，问题会改善很多。

---

# 最终建议

你的两个方向我都赞成：

**第一，必须用具体数据和案例驱动。**
尤其是 tokenizer 关卡，应该把真实文本、切分结果、token id、attention mask 和 budget meter 常驻显示出来，而不是藏在 trace samples 里。

**第二，应该同步显示 graph 对应代码。**
但不要只显示 module pseudoCode。当前 Code tab 只是节点级 `pseudoCode`，还不是真正的 graph 代码。建议新增 Graph Code Panel：根据当前 GraphSpec、节点参数、连线和当前 test case 自动生成 PyTorch-like 代码，并与 Trace / 画布互相高亮。

优先落地顺序建议是：

```text
1. 先把 1-2 Split / Merge Budget 改成 Case-first
2. 增加 Tokenizer Preview：text → pieces → token_ids → mask → budget
3. 增加 Graph Code Panel，先支持 tokenizer 三节点
4. 再把 0-2 MatMul 改成具体小矩阵案例
5. 最后扩展到 QK / Broadcast / Mask
```

只要 1-2 这一关能让玩家产生“我改了切分策略，文本真的变成了不同 token，代码也同步变了”的体验，Graph Challenge 的可理解性会明显上一个台阶。
