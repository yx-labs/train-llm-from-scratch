我建议把 MVP0.1 的核心命题定义成：

**玩家不是在“完成一关图任务”，而是在亲手发明、测试、封装、升级一套 tiny LLM 工具链。**

也就是从：

```text
玩家使用系统给定节点解题
```

升级为：

```text
玩家先造出节点
再把自己造出的节点用于下一关
最后这些节点组合成一个可运行 tiny LLM
```

这和主体项目的学习路径是匹配的。主体文档里的基础链路本身就是：

```text
raw text → tokenizer → token ids → decoder-only Transformer → next-token cross-entropy → checkpoint → inference/chat
```

并且强调同一个 backbone 会在 base、SFT、reward、DPO/PPO/GRPO 等阶段复用。

当前游戏代码已经具备继续做这个方向的基础：`LevelSpec` 里已经有 `caseStudy`，`ModuleDef` 也已经预留了 `emitCode`，说明“案例驱动”和“graph → code”都已经进入数据结构层。

---

# 一、MVP0.1 的核心设计原则

## 1. 一切节点都应该有“出生过程”

之前的 Graph Challenge 是：

```text
我给你 MatMulGate / TokenizerSocket / EmbeddingReadyProbe
你拿它们解题
```

MVP0.1 应该变成：

```text
你先造 Scalar
Scalar 组合成 Vector
Vector 组合成 Matrix
Matrix 扩展成 Tensor
Tensor + Shape + Axis + DType 变成 Typed Tensor
Typed Tensor 才能进入 MatMul / Embedding / Attention
```

也就是说，后面用到的每个高级节点，都应该是前面某个关卡的产物。

---

## 2. 组件不是一次性答案，而是玩家资产

玩家完成一个组件后，它应该进入自己的 **Component Library**：

```text
My Components
- ScalarCell v1
- VectorRail v1
- MatrixBoard v1
- TensorBox v2
- TokenizerSocket v1
- MatMulGate v2
- LinearModule v1
```

后续关卡不能直接给高级节点，而是要求玩家从自己的库里拖出来。

---

## 3. 每个组件必须有五件东西

一个组件不是一个图标，而是一套完整工程对象：

```text
Interface：输入输出端口、dtype、shape、axis contract
Implementation：内部 graph / recipe
Tests：公开测试、隐藏测试、边界测试
Code：对应真实代码
Upgrade Path：后续可扩展属性
```

这对应真实编程思想里的：

```text
函数签名 / 类接口
内部实现
单元测试
源码
版本迭代
```

---

## 4. 抽象必须从具体案例里长出来

比如不要直接教：

```text
Tensor = data + shape + dtype + axes
```

而是先让玩家遇到问题：

```text
一个数字可以传过去
一排数字也可以传过去
一张表也可以传过去
但下游模块不知道它是几维、每个方向代表什么
所以我们要给它加 shape / dtype / axis
```

每一个抽象属性，都应该来自一个具体失败案例。

---

# 二、完整组件演进总图谱

下面是从标量到可运行 LLM 的完整演进链。

```text
0. Graph OS / 编程工作台
   Wire / Port / Input / Output / Test / Trace / Component Packager

1. Numeric Foundations / 数字基础
   Scalar
   → Vector
   → Matrix
   → Tensor
   → Typed Tensor
   → Axis-aware Tensor
   → Parameter Tensor

2. Tensor Operations / 张量操作
   Scalar Add / Mul
   → Vector Dot
   → Matrix Multiply
   → Transpose
   → Broadcast
   → Reduce / Sum
   → Softmax

3. Data Pipeline / 文本输入管线
   Raw Text
   → TextInput
   → Splitter
   → Merge Rules
   → Vocab Table
   → Vocab Lookup
   → Token IDs
   → Token Buffer [B,T]
   → Padding
   → Attention Mask
   → Target Shift

4. Representation / 表征层
   Token IDs
   → Embedding Table
   → Token Embedding [B,T,C]
   → Position IDs
   → Position Embedding [T,C]
   → Hidden Tensor [B,T,C]

5. Neural Building Blocks / 神经网络基础模块
   Parameter Tensor
   → Linear
   → Bias Add
   → Activation
   → MLP
   → LayerNorm
   → Residual Add

6. Attention / 注意力组件
   Hidden Tensor
   → Q / K / V Linear
   → Head Split
   → QKᵀ Scores
   → Scale
   → Causal Mask
   → Softmax
   → Weighted Sum V
   → Head Concat
   → Output Projection
   → Multi-Head Attention

7. Transformer Block / Transformer 块
   LayerNorm
   → Attention
   → Residual
   → LayerNorm
   → MLP
   → Residual
   → Transformer Block
   → Block Stack

8. LM Head / 语言模型输出
   Final Hidden [B,T,C]
   → Final LayerNorm
   → LM Head Linear C→V
   → Logits [B,T,V]
   → Softmax Distribution
   → Next Token Prediction

9. Training Loop / 训练系统
   Dataset Window
   → Batch Builder
   → Target Shift
   → Cross Entropy
   → Loss Meter
   → Backward Trace
   → Optimizer
   → Checkpoint

10. Generation / 推理生成
   Prompt Encode
   → Context Window
   → Forward Pass
   → Sample Next Token
   → Append Token
   → Decode Text
   → Stop Condition
```

这个总图谱和主体项目的真实逻辑一致。主体文档中说明 Transformer 实际接收的是 `token ids (B,T)`，经过 token embedding、position embedding、Transformer blocks、final LayerNorm、LM head，输出 `logits (B,T,V)`。 主体文档也明确了 `B/T/C/H/D/V` 这些 shape 符号的意义。

---

# 三、组件系统的数据模型设计

建议新增一个和 `ModuleDef` 平行但更高层的概念：`ComponentBlueprint`。

当前 `ModuleDef` 更像“系统内置模块定义”。MVP0.1 需要的是“玩家可构建、可封装、可升级、可复用的组件定义”。

```ts
type ComponentBlueprint = {
  id: string;
  title: string;
  category:
    | "primitive"
    | "tensor"
    | "operation"
    | "data"
    | "model"
    | "training"
    | "inference";

  version: number;

  // 组件接口
  inputs: PortSpec[];
  outputs: PortSpec[];

  // 玩家需要先拥有哪些组件
  requires: string[];

  // 组件由哪些子组件搭建出来
  buildRecipe: ComponentRecipeStep[];

  // 初始坏图 / 目标图
  initialGraph: GraphSpec;
  targetGraph: GraphSpec;

  // 测试与认证
  visibleTests: TestCase[];
  hiddenTests: TestCase[];

  // 代码映射
  codeTemplate: CodeTemplate;

  // 封装后会解锁的新节点
  exportsModuleId: string;

  // 后续升级点
  upgradeSlots: ComponentUpgradeSlot[];
};
```

玩家完成并通过隐藏测试后，生成：

```ts
type BuiltComponent = {
  blueprintId: string;
  version: number;
  certified: boolean;
  implementationGraph: GraphSpec;
  exportedModuleId: string;
  testStats: {
    visibleRuns: number;
    hiddenRuns: number;
    failedRuns: number;
    hintsUsed: number;
  };
  unlockedAtLevelId: string;
};
```

后续关卡的 `modulePalette` 不再只来自系统内置模块，而是：

```ts
systemModules + playerBuiltComponents
```

---

# 四、组件演进详细设计

## Chapter 0：Graph OS 与数字基础

这一章的目标不是讲 LLM，而是让玩家先发明一套“能表达计算”的最小系统。

| 关卡                  | 玩家构建的组件                           | 从什么构成                        | 学到什么                     | 后续用途                   |
| ------------------- | --------------------------------- | ---------------------------- | ------------------------ | ---------------------- |
| 0-0 Wire & Port     | `Wire`, `InputPort`, `OutputPort` | 空画布、输入、输出                    | 数据沿边流动，端口有方向             | 所有关卡                   |
| 0-1 Scalar Cell     | `ScalarCell`                      | Number literal + dtype badge | 一个数字也有类型                 | loss、mask value、scale  |
| 0-2 Scalar Add      | `AddScalar`                       | 两个 ScalarCell                | 运算节点、输入检查                | residual add 的起点       |
| 0-3 Vector Rail     | `Vector`                          | 多个 ScalarCell + length       | 一排数字，长度是第一种 shape        | token vector           |
| 0-4 Dot Product     | `DotProduct`                      | Vector + multiply + sum      | 相同长度向量才能点积               | attention score、linear |
| 0-5 Matrix Board    | `Matrix`                          | 多条 Vector                    | 二维结构、row/col             | weight、embedding table |
| 0-6 Matrix Multiply | `MatMul2D`                        | DotProduct 网格                | 矩阵乘法是很多 dot cells        | Linear / QK            |
| 0-7 Tensor Box      | `Tensor`                          | data buffer + shape          | rank、shape、索引            | 所有模型数据                 |
| 0-8 Typed Tensor    | `TypedTensor`                     | Tensor + dtype               | float/int/bool/mask 不可乱接 | token ids vs hidden    |
| 0-9 Axis Tensor     | `AxisTensor`                      | Tensor + axis names          | `[2,4,8]` 不等于 `[B,T,C]`  | Broadcast / Attention  |

### 关键设计

不要一上来讲 Tensor。让玩家先亲手搭：

```text
Scalar → Vector → Matrix → Tensor
```

这样 `shape` 和 `rank` 就不是定义，而是玩家自己发现的复杂度。

---

## Chapter 1：Text → Token Pipeline

主体项目文档中明确说，Transformer 不看字符或单词，而是看整数 token ids；tokenizer 是语言和 tensor 之间的边界。 所以这一章要让玩家亲手造出这条边界。

| 关卡                         | 玩家构建的组件              | 从什么构成                                    | 学到什么                     | 后续用途               |
| -------------------------- | -------------------- | ---------------------------------------- | ------------------------ | ------------------ |
| 1-0 Text Input             | `TextInput`          | string source + output port              | 原始文本是 raw_text，不是 tensor | tokenizer 输入       |
| 1-1 Splitter               | `BoundarySplitter`   | text scanner + policy                    | 文本要切成 pieces             | tokenizer          |
| 1-2 Merge Forge            | `MergeRules`         | adjacent pair merge                      | subword 的压缩思想            | BPE-like tokenizer |
| 1-3 Vocab Table            | `VocabTable`         | token string → id map                    | token 是词表地址              | embedding lookup   |
| 1-4 Vocab Lookup           | `VocabLookup`        | pieces + vocab                           | pieces 变成 integer ids    | token_ids          |
| 1-5 Token Buffer           | `TokenBuffer[B,T]`   | token ids + T slots                      | T 是上下文窗口长度               | Transformer 输入     |
| 1-6 Padder                 | `Padder`             | token buffer + pad id                    | batch 需要矩形               | attention mask     |
| 1-7 Attention Mask Builder | `AttentionMask[B,T]` | token_ids + pad id                       | pad 不该参与 attention       | training/inference |
| 1-8 Tokenizer Component    | `TokenizerSocket`    | Splitter + Merge + Vocab + Buffer + Mask | 封装成可复用 tokenizer         | 所有文本任务             |

这里要把 `T` 明确作为玩家亲手造出的 **TokenBuffer 容量**，而不是凭空出现的公式变量。主体文档也说明训练 loader 会切 `context_length + 1` 的 token window，前 `context_length` 是输入，后 `context_length` 是 shift target。

---

## Chapter 2：Embedding 与 Hidden Tensor

这一章从 `token_ids[B,T]` 进入模型内部。

| 关卡                     | 玩家构建的组件                  | 从什么构成                   | 学到什么                           | 后续用途               |
| ---------------------- | ------------------------ | ----------------------- | ------------------------------ | ------------------ |
| 2-0 Parameter Matrix   | `ParameterMatrix`        | Matrix + trainable flag | 权重是可学习 tensor                  | embedding / linear |
| 2-1 Embedding Table    | `EmbeddingTable[V,C]`    | ParameterMatrix         | id 查表得到 vector                 | token embedding    |
| 2-2 Embedding Lookup   | `EmbeddingLookup`        | token_ids + table       | int ids → float vectors        | hidden[B,T,C]      |
| 2-3 Position Counter   | `PositionIds[T]`         | range builder           | 模型需要位置                         | position embedding |
| 2-4 Position Embedding | `PositionEmbedding[T,C]` | position ids + table    | 每个位置也有向量                       | hidden init        |
| 2-5 Hidden Init        | `Token + Position Add`   | broadcast add           | token vector + position vector | block 输入           |

主体项目中 token embedding 和 position embedding 是两张 `nn.Embedding` 表，并且两者相加后进入 Transformer block。

### 玩家最终封装

```text
InputEmbedder
inputs:
  token_ids[B,T]

outputs:
  hidden[B,T,C]

internal:
  token_embedding
  position_ids
  position_embedding
  add
```

后续所有 Transformer 关卡必须使用玩家自己造的 `InputEmbedder`。

---

## Chapter 3：Linear、Activation、MLP

这一章让玩家把基础矩阵操作封装成神经网络模块。

| 关卡                      | 玩家构建的组件       | 从什么构成                   | 学到什么                | 后续用途              |
| ----------------------- | ------------- | ----------------------- | ------------------- | ----------------- |
| 3-0 Linear Core         | `LinearCore`  | MatMul + Weight         | C → O 投影            | Q/K/V、MLP、LM head |
| 3-1 Bias Add            | `BiasAdd`     | Broadcast + Add         | bias[O] 广播到 [B,T,O] | Linear            |
| 3-2 Linear Module       | `Linear`      | MatMul + Bias           | 封装常用投影层             | 所有模型层             |
| 3-3 Activation          | `ReLU`        | scalar max / vector map | 非线性变换               | MLP               |
| 3-4 MLP Up Projection   | `Linear C→4C` | Linear                  | 扩宽通道                | MLP               |
| 3-5 MLP Down Projection | `Linear 4C→C` | Linear                  | 回到 hidden width     | MLP               |
| 3-6 MLP Module          | `MLP`         | Linear + ReLU + Linear  | 每个 token 独立计算       | Transformer block |

主体项目里的 MLP 是 `Linear(n_embed, 4*n_embed) → ReLU → Linear(4*n_embed, n_embed)`。

### 玩家最终封装

```text
MLP
inputs:
  hidden[B,T,C]

outputs:
  hidden[B,T,C]

internal:
  linear_up C→4C
  relu
  linear_down 4C→C
```

---

## Chapter 4：Attention Head

这一章是核心挑战。主体文档里 attention 的真实结构是：

```text
Q = X WQ
K = X WK
V = X WV
Attention = softmax(QKᵀ / sqrt(D) + M) V
```

其中 causal mask 让每个位置只能看自己和过去。

| 关卡                 | 玩家构建的组件           | 从什么构成                              | 学到什么                 | 后续用途             |
| ------------------ | ----------------- | ---------------------------------- | -------------------- | ---------------- |
| 4-0 Q Projection   | `QLinear`         | Linear C→D                         | query 是“我想找什么”       | attention        |
| 4-1 K Projection   | `KLinear`         | Linear C→D                         | key 是“我有什么信息”        | attention        |
| 4-2 V Projection   | `VLinear`         | Linear C→D                         | value 是“我传什么内容”      | attention        |
| 4-3 QK Score       | `QKScore`         | MatMul + K transpose               | token-to-token 分数    | attention        |
| 4-4 Scale          | `ScaleBySqrtD`    | Scalar + multiply                  | 稳定 softmax           | attention        |
| 4-5 Causal Mask    | `CausalMask[T,T]` | lower triangle                     | 不能看未来                | decoder-only     |
| 4-6 Mask Apply     | `MaskAdd`         | Broadcast + Add                    | future logits → -inf | attention        |
| 4-7 Softmax        | `SoftmaxLastDim`  | exp/sum/div                        | 分数变权重                | attention        |
| 4-8 Weighted Sum   | `AttnWeights @ V` | MatMul                             | 汇聚 value             | attention output |
| 4-9 Attention Head | `AttentionHead`   | Q/K/V + Score + Mask + Softmax + V | 封装单头注意力              | multi-head       |

### 玩家最终封装

```text
AttentionHead
inputs:
  hidden[B,T,C]

outputs:
  head_out[B,T,D]

internal:
  q = Linear(hidden)
  k = Linear(hidden)
  v = Linear(hidden)
  scores = q @ k.transpose(-2,-1)
  scores = scores / sqrt(D)
  scores = scores + causal_mask
  weights = softmax(scores)
  out = weights @ v
```

---

## Chapter 5：Multi-Head Attention

主体项目中 multi-head 是创建多个 Head，然后 concat，再经过 output projection。

| 关卡                       | 玩家构建的组件              | 从什么构成                       | 学到什么       | 后续用途              |
| ------------------------ | -------------------- | --------------------------- | ---------- | ----------------- |
| 5-0 Head Split           | `HeadWidth D=C/H`    | shape arithmetic            | C 被分成 H 个头 | MHA               |
| 5-1 Parallel Heads       | `HeadArray`          | 多个 AttentionHead            | 多种关系并行看    | MHA               |
| 5-2 Head Concat          | `ConcatHeads`        | concatenate                 | H×D 回到 C   | MHA               |
| 5-3 Output Projection    | `AttentionOutLinear` | Linear C→C                  | 混合 heads   | MHA               |
| 5-4 Multi-Head Attention | `MHA`                | HeadArray + Concat + Linear | 完整注意力层     | Transformer block |

### 玩家最终封装

```text
MultiHeadAttention
inputs:
  hidden[B,T,C]

outputs:
  hidden[B,T,C]

internal:
  heads = [AttentionHead_i(hidden) for i in H]
  concat = concat(heads, dim=C)
  out = linear(concat)
```

---

## Chapter 6：LayerNorm、Residual、Transformer Block

主体项目使用 pre-norm residual 结构：

```python
x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
```

这在文档中有明确说明。

| 关卡                     | 玩家构建的组件               | 从什么构成                             | 学到什么              | 后续用途        |
| ---------------------- | --------------------- | --------------------------------- | ----------------- | ----------- |
| 6-0 Mean / Variance    | `MeanVar`             | reduce over C                     | LayerNorm 的基础     | LayerNorm   |
| 6-1 LayerNorm          | `LayerNorm`           | normalize + gamma/beta            | 稳定每个 token vector | block       |
| 6-2 Residual Add       | `ResidualAdd`         | Add same shape                    | 学习 update，不替换信息   | block       |
| 6-3 Attention Sublayer | `LN → MHA → Residual` | LayerNorm + MHA + Add             | tokens 通信         | block       |
| 6-4 MLP Sublayer       | `LN → MLP → Residual` | LayerNorm + MLP + Add             | token 内计算         | block       |
| 6-5 Transformer Block  | `Block`               | Attention Sublayer + MLP Sublayer | 最小 Transformer 单元 | block stack |
| 6-6 Block Stack        | `TransformerStack`    | N 个 Block                         | 深层模型              | full model  |

---

## Chapter 7：LM Head 与 Loss

| 关卡                     | 玩家构建的组件         | 从什么构成              | 学到什么                 | 后续用途            |
| ---------------------- | --------------- | ------------------ | -------------------- | --------------- |
| 7-0 Final LayerNorm    | `FinalNorm`     | LayerNorm          | 输出前归一化               | logits          |
| 7-1 LM Head            | `LMHead`        | Linear C→V         | 每个 token 输出 vocab 分数 | next token      |
| 7-2 Logits Board       | `Logits[B,T,V]` | hidden + LMHead    | 每个位置有 V 个候选          | loss/generation |
| 7-3 Shift Targets      | `TargetShift`   | token_ids[:,1:]    | 预测下一个 token          | CE              |
| 7-4 Cross Entropy Cell | `CrossEntropy`  | logits + target id | 正确 token 概率越高越好      | training        |
| 7-5 Loss Reducer       | `LossMean`      | reduce over B/T    | batch loss           | optimizer       |

主体项目文档明确说 base LLM 是条件概率模型，学习的是 next token distribution。

---

## Chapter 8：Training Loop

| 关卡                 | 玩家构建的组件             | 从什么构成            | 学到什么         | 后续用途          |
| ------------------ | ------------------- | ---------------- | ------------ | ------------- |
| 8-0 Dataset Stream | `TokenStream`       | flat ids         | 数据是一长串 token | batch         |
| 8-1 Window Sampler | `ContextWindow`     | T+1 slice        | 输入和目标错一位     | pretraining   |
| 8-2 Batch Builder  | `BatchBuilder[B,T]` | 多个窗口             | batch 并行     | training      |
| 8-3 Forward Runner | `ForwardPass`       | model + batch    | logits       | loss          |
| 8-4 Backward Trace | `BackwardPass`      | loss → gradients | 参数如何收到更新信号   | optimizer     |
| 8-5 Optimizer      | `AdamW` 简化版         | params + grads   | 更新权重         | training loop |
| 8-6 Checkpoint     | `CheckpointSaver`   | params snapshot  | 保存模型         | generation    |

---

## Chapter 9：Generation

| 关卡                 | 玩家构建的组件            | 从什么构成                               | 学到什么         | 后续用途       |
| ------------------ | ------------------ | ----------------------------------- | ------------ | ---------- |
| 9-0 Prompt Encoder | `PromptEncode`     | Tokenizer                           | prompt 变 ids | inference  |
| 9-1 Context Crop   | `ContextWindow`    | last T tokens                       | 不能超过 context | generation |
| 9-2 Next Logits    | `ForwardLastToken` | model forward                       | 取最后位置 logits | sampler    |
| 9-3 Sampler        | `SampleNextToken`  | softmax + temperature/top-k         | 从概率选 token   | generation |
| 9-4 Append Token   | `AppendToken`      | buffer update                       | 自回归循环        | generation |
| 9-5 Decode         | `TokenDecode`      | ids → text                          | 还原文字         | chat       |
| 9-6 Tiny Chat Loop | `GenerateLoop`     | encoder + model + sampler + decoder | 可运行 tiny LLM | 最终演示       |

---

# 五、关键组件版本演进

MVP0.1 很重要的一点是：**组件不是一次性完成，而是逐步升级。**

## Tensor 的版本演进

```text
Tensor v0：只是 data buffer
Tensor v1：增加 shape
Tensor v2：增加 rank
Tensor v3：增加 dtype
Tensor v4：增加 axis semantics
Tensor v5：支持 view / transpose
Tensor v6：支持 broadcast metadata
Tensor v7：支持 requires_grad / grad
```

后续关卡用到 Tensor 时，会要求对应版本：

```text
MatMul 需要 Tensor v4
Broadcast 需要 Tensor v6
Training 需要 ParameterTensor v7
```

---

## Tokenizer 的版本演进

```text
Tokenizer v0：TextInput，只能输出 raw_text
Tokenizer v1：Splitter，把文本切成 pieces
Tokenizer v2：VocabLookup，把 pieces 变 ids
Tokenizer v3：TokenBuffer，限制 T
Tokenizer v4：Padder，补齐 batch
Tokenizer v5：AttentionMask，输出 mask
Tokenizer v6：MergeRules，支持 subword merges
Tokenizer v7：ChatTemplate，支持 user/assistant 格式
```

主体文档中 tokenization 是文本变成整数 id 的边界，也解释了 subword 的必要性：word-level 难处理罕见词，char-level 会让序列变长，subword 是折中。

---

## Linear 的版本演进

```text
Linear v0：只做 vector dot
Linear v1：支持 matrix weight
Linear v2：支持 batched [B,T,C]
Linear v3：支持 weight orientation [O,C] → [C,O]
Linear v4：支持 bias broadcast
Linear v5：变成 trainable ParameterModule
```

---

## Attention Head 的版本演进

```text
AttentionHead v0：QK dot scores
AttentionHead v1：K transpose
AttentionHead v2：scale by sqrt(D)
AttentionHead v3：causal mask
AttentionHead v4：softmax
AttentionHead v5：weights @ V
AttentionHead v6：封装为 reusable component
```

---

# 六、MVP0.1 推荐落地范围

“完整图谱”可以先设计到 runnable LLM，但 MVP0.1 实现不要一次铺太宽。我建议 MVP0.1 的可玩闭环做到：

```text
玩家能从 Scalar / Vector / Matrix / Tensor
一路构建到一个 Tiny Decoder Forward Pass
并能对一段 prompt 生成下一个 token。
```

也就是 MVP0.1 的最终演示是：

```text
Raw Text
→ Player-built Tokenizer
→ Player-built Embedding
→ Player-built One-Head Attention
→ Player-built MLP
→ Player-built LM Head
→ Logits
→ Sample Next Token
```

不强求完整训练 loop，但需要能跑 forward 和 generation。训练 loop 可以作为 MVP0.2。

---

# 七、MVP0.1 章节建议

## MVP0.1-A：Build the Number System

目标：让玩家拥有基础数值组件。

```text
0-1 Scalar Forge
0-2 Vector Rail
0-3 Dot Product
0-4 Matrix Board
0-5 Matrix Multiply
0-6 Tensor Box
0-7 Axis / DType Upgrade
0-8 Pack Tensor Component
```

最终产物：

```text
TensorCore v1
MatMulGate v1
TransposeSwitch v1
BroadcastAdd v1
```

---

## MVP0.1-B：Build the Text Frontend

目标：让玩家拥有 tokenizer。

```text
1-1 Text Input
1-2 Splitter
1-3 Vocab Lookup
1-4 Token Buffer T=8
1-5 Padding
1-6 Attention Mask
1-7 Merge Rules
1-8 Pack TokenizerSocket
```

最终产物：

```text
TokenizerSocket v1
TokenBuffer v1
AttentionMaskBuilder v1
```

---

## MVP0.1-C：Build the First Model Layer

目标：让 token ids 进入 hidden space。

```text
2-1 Parameter Matrix
2-2 Embedding Lookup
2-3 Position Embedding
2-4 Token + Position Add
2-5 Pack InputEmbedder
```

最终产物：

```text
InputEmbedder v1
hidden[B,T,C]
```

---

## MVP0.1-D：Build a Tiny Decoder Block

目标：构建最小 Transformer block。

```text
3-1 Linear
3-2 Q/K/V Projection
3-3 QK Score
3-4 Causal Mask
3-5 Softmax
3-6 Weighted Sum V
3-7 MLP
3-8 Residual + LayerNorm
3-9 Pack TinyBlock
```

最终产物：

```text
TinyAttentionHead v1
TinyMLP v1
TinyBlock v1
```

---

## MVP0.1-E：Build Tiny LLM Runner

目标：把所有组件组合成可运行模型。

```text
4-1 Stack: Tokenizer + Embedder + TinyBlock
4-2 LM Head
4-3 Logits Viewer
4-4 Next Token Sampler
4-5 Decode
4-6 Tiny Generate Loop
```

最终产物：

```text
TinyLLM v1
```

---

# 八、组件封装玩法

每次玩家完成一个组件，不是直接过关，而是进入一个 “Pack & Certify” 流程。

## 1. Build

玩家用已有组件搭内部 graph：

```text
VectorDot = Mul + Sum
Linear = MatMul + BiasAdd
Tokenizer = Splitter + VocabLookup + Padder + MaskBuilder
```

## 2. Test

运行三类测试：

```text
Visible Test：公开样例
Mutation Test：shape / text / batch 变化
Contract Test：端口类型和输出语义
```

## 3. Certify

通过后获得认证：

```text
Certified Component: Linear v1
```

## 4. Pack

组件被压缩成一个新节点：

```text
内部图隐藏
对外只显示输入 / 输出 / 参数
```

## 5. Reuse

后续关卡必须使用玩家自己打包的组件。

---

# 九、组件依赖关系示例

## TokenizerSocket

```text
TokenizerSocket
requires:
  TextInput
  BoundarySplitter
  MergeRules
  VocabTable
  VocabLookup
  TokenBuffer
  Padder
  AttentionMaskBuilder

exports:
  token_ids[B,T]
  attention_mask[B,T]
  token_pieces
```

## Linear

```text
Linear
requires:
  TensorCore
  ParameterMatrix
  MatMulGate
  BroadcastAdd

exports:
  out[B,T,O]
```

## AttentionHead

```text
AttentionHead
requires:
  Linear
  TransposeSwitch
  MatMulGate
  Scale
  CausalMask
  Softmax
  WeightedSum

exports:
  head_out[B,T,D]
```

## TransformerBlock

```text
TransformerBlock
requires:
  LayerNorm
  MultiHeadAttention
  MLP
  ResidualAdd

exports:
  hidden[B,T,C]
```

## TinyLLM

```text
TinyLLM
requires:
  TokenizerSocket
  InputEmbedder
  TransformerBlock
  FinalNorm
  LMHead
  Sampler
  Decoder

exports:
  generated_text
```

---

# 十、真实编程思想如何融入玩法

## 1. 函数

最早的组件像函数：

```python
def add(a, b):
    return a + b
```

对应游戏：

```text
AddScalar
inputs: a, b
output: sum
```

---

## 2. 类型系统

端口就是类型检查：

```python
def matmul(left: Tensor[float32], right: Tensor[float32]) -> Tensor[float32]:
```

对应游戏：

```text
float32 才能进 MatMul
int token_ids 不能直接进 MatMul
raw_text 不能进 Embedding
```

---

## 3. 类与封装

玩家把内部 graph 打包：

```python
class Linear:
    def __init__(self):
        self.weight = Parameter(...)
        self.bias = Parameter(...)

    def forward(self, x):
        return x @ self.weight + self.bias
```

对应游戏：

```text
MatMul + BiasAdd + Parameter
→ Pack as Linear
```

---

## 4. 单元测试

每个组件都必须有测试：

```python
assert linear(x).shape == [B,T,O]
assert allclose(linear(x), reference)
```

对应游戏：

```text
Visible Test
Hidden Shape Mutation
Reference Diff
```

---

## 5. 重构

玩家从重复搭图到封装组件：

```text
每次都手搭 MatMul + Bias 太麻烦
→ Pack Linear
→ 后续直接复用
```

这是真实编程里的 refactor / abstraction。

---

## 6. 版本升级

```text
Linear v1：无 bias
Linear v2：加 bias
Linear v3：支持 batch
Linear v4：支持 trainable params
```

对应真实项目中的逐步工程演进。

---

# 十一、样例：从 Scalar 到 MatMulGate 的演进

## 0-1 Scalar Forge

玩家任务：

```text
造一个能保存数字的 ScalarCell。
```

输入：

```text
value = 3.0
dtype = float32
```

测试：

```text
输出值必须是 3.0
dtype 必须是 float32
```

代码：

```python
x = Scalar(3.0, dtype="float32")
```

---

## 0-2 Vector Rail

玩家任务：

```text
把多个 ScalarCell 组合成一条 Vector。
```

输入：

```text
[0.2, -0.7, 1.4]
```

测试：

```text
length = 3
axis = C
```

代码：

```python
v = Vector([0.2, -0.7, 1.4], axis="C")
```

---

## 0-3 Dot Product

玩家任务：

```text
用 Mul + Sum 造 DotProduct。
```

输入：

```text
x = [0.2, -0.7, 1.4]
w = [0.5, 0.1, -0.3]
```

输出：

```text
sum_i x[i] * w[i]
```

代码：

```python
score = dot(x, w)
```

---

## 0-4 Matrix Board

玩家任务：

```text
把多条 Vector 排成 Matrix。
```

输入：

```text
W = [
  [0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6],
]
```

测试：

```text
shape = [2,3]
axes = [O,C]
```

代码：

```python
W = Matrix(rows=2, cols=3, axes=["O", "C"])
```

---

## 0-5 MatMulGate

玩家任务：

```text
用一组 DotProduct 造 MatMul。
```

从：

```text
x[C] dot w_o[C] → scalar
```

扩展到：

```text
x[C] @ W[C,O] → y[O]
```

再扩展到：

```text
hidden[B,T,C] @ W[C,O] → out[B,T,O]
```

代码：

```python
out = matmul(hidden, W)
```

这个 MatMulGate 后面会成为 Linear、QK Score、Attention Weighted Sum 的基础。

---

# 十二、样例：从 TextInput 到 TokenizerSocket 的演进

## 1-1 TextInput

```text
raw text:
"we train llm"
```

代码：

```python
text = "we train llm"
```

---

## 1-2 Splitter

```text
pieces:
["we", "train", "llm"]
```

代码：

```python
pieces = split(text)
```

---

## 1-3 VocabLookup

```text
we    -> 8
train -> 9
llm   -> 10
```

代码：

```python
ids = [vocab[p] for p in pieces]
```

---

## 1-4 TokenBuffer

```text
T = 8

[8][9][10][pad][pad][pad][pad][pad]
```

代码：

```python
token_ids = pad(ids, length=8)
```

---

## 1-5 MaskBuilder

```text
attention_mask:
[1][1][1][0][0][0][0][0]
```

代码：

```python
mask = [1 if id != PAD else 0 for id in token_ids]
```

---

## 1-6 Pack TokenizerSocket

```text
TextInput + Splitter + VocabLookup + Buffer + Mask
→ TokenizerSocket
```

代码：

```python
token_ids, attention_mask = tokenizer.encode_batch(texts)
```

---

# 十三、MVP0.1 需要的系统功能

## 1. Component Library

显示玩家已经拥有的组件：

```text
Primitive
- ScalarCell
- VectorRail
- MatrixBoard

Tensor
- TensorBox
- MatMulGate
- TransposeSwitch

Data
- TextInput
- TokenizerSocket
- TokenBuffer

Model
- EmbeddingLookup
- Linear
- AttentionHead
```

每个组件显示：

```text
版本
认证状态
通过测试数量
内部图复杂度
可升级项
被哪些组件依赖
```

---

## 2. Component Builder

专门用于造组件的界面，不是普通 Graph Challenge。

```text
左侧：已有组件库
中间：组件内部 graph
右侧：接口定义 / 测试 / 代码
底部：Pack / Certify / Upgrade
```

---

## 3. Component Contract Editor

玩家要定义组件输入输出：

```text
Input:
  x: float32[B,T,C]

Output:
  y: float32[B,T,O]
```

如果定义错误，后续封装失败。

---

## 4. Component Certification

组件不是通过当前案例就立刻可用，而是要经过认证。MVP0.1 的交互文案统一为：

```text
检查当前任务：验证当前展示案例，帮助玩家调图。
提交认证：运行公开认证变体 + 系统未公开变体，验证组件是否能泛化到当前案例之外。
```

认证通过后不再要求玩家手动点击 Pack / 封装按钮，而是直接提示：

```text
{组件名} 已经可用
```

然后自动进入下一关。

认证输入不要求玩家手写完整 tensor。玩家编辑少量结构参数，系统生成具体输入：

```text
Scalar: 直接输入一个有限 float32
Vector: 编辑 c0/c1/c2
Matrix / Tensor: 编辑 C width + seed
MatMul / Linear: 编辑 B/T/C/O + seed
```

详细方案见：

```text
MVP0.1_Certification_Variant_Design.md
```

---

## 5. Component Upgrade

后续关卡不是重新造，而是升级：

```text
TokenizerSocket v1 → v2
新增 merge rules

TensorBox v3 → v4
新增 axis semantics

Linear v2 → v3
新增 bias
```

---

# 十四、MVP0.1 第一批建议实现组件

不要一口气做完整 LLM。第一版先实现 30 个核心组件：

## Primitive

```text
ScalarCell
VectorRail
MatrixBoard
TensorBox
ShapeBadge
DTypeBadge
AxisBadge
ParameterTensor
```

## Ops

```text
Add
Multiply
Sum
DotProduct
MatMul
Transpose
Broadcast
Softmax
```

## Data

```text
TextInput
BoundarySplitter
MergeRules
VocabTable
VocabLookup
TokenBuffer
Padder
AttentionMaskBuilder
TokenizerSocket
```

## Model

```text
EmbeddingTable
EmbeddingLookup
PositionIds
PositionEmbedding
Linear
ReLU
LayerNorm
ResidualAdd
AttentionHead
MLP
LMHead
```

## Runner

```text
ForwardRunner
LogitsViewer
Sampler
Decoder
TinyLLMRunner
```

---

# 十五、MVP0.1 最终演示关卡

最终关卡可以叫：

```text
Build Your First Tiny LLM
```

玩家只能使用自己之前造出的组件：

```text
TokenizerSocket
InputEmbedder
AttentionHead
MLP
LMHead
Sampler
Decoder
```

目标：

```text
输入：
"we train"

输出：
模型预测下一个 token 的概率分布，并采样一个 token。
```

Graph：

```text
Raw Text
→ TokenizerSocket
→ InputEmbedder
→ AttentionHead
→ MLP
→ LMHead
→ Sampler
→ Decode
```

代码同步：

```python
texts = ["we train"]

token_ids, attention_mask = tokenizer.encode_batch(texts)
hidden = embedder(token_ids)
hidden = attention_head(hidden, attention_mask)
hidden = mlp(hidden)
logits = lm_head(hidden)
next_id = sample(logits[:, -1, :])
text = decode(next_id)
```

完成后给玩家一个明确复盘：

```text
你不是使用了一个 LLM。
你刚刚搭出了一条最小 LLM forward path。
```

---

# 十六、和当前版本的衔接方式

当前版本已经有：

```text
GraphSpec
ModuleDef
LevelSpec
caseStudy
codegen
visible / hidden tests
GraphCodePanel
GraphCasePanel
```

下一步不需要推翻，而是新增一层：

```text
ComponentBlueprint
BuiltComponent
ComponentLibrary
ComponentBuilder
Pack / Certify
```

当前的 `ModuleDef` 可以继续作为运行时节点；新系统只是允许玩家通过 `ComponentBlueprint` 生成自己的 `ModuleDef`。

关系是：

```text
玩家搭 GraphSpec
↓
通过测试
↓
Pack 成 BuiltComponent
↓
注册为 ModuleDef
↓
出现在后续 modulePalette
```

---

# 十七、最重要的设计风险

## 风险 1：从 Scalar 开始太无聊

解决方式：

```text
每个基础组件都必须服务一个可见目标。
不要让玩家只是学习定义。
```

例如 Scalar 不要单独讲“什么是数字”，而是：

```text
修一个 loss meter 的数值显示。
```

Vector 不要单独讲“一排数字”，而是：

```text
让一个 token 拥有 3 个特征。
```

Matrix 不要单独讲“二维数组”，而是：

```text
造一个可以把 token vector 投影成分数的 weight board。
```

---

## 风险 2：组件太多，玩家记不住

解决方式：

```text
组件库按层级折叠
每个组件有用途标签
后续关卡只显示当前需要的组件
```

标签示例：

```text
Data
Shape
Math
Neural
Attention
Training
Inference
```

---

## 风险 3：封装后看不见内部，失去学习价值

解决方式：

```text
每个封装组件都可以 Expand Inside
```

玩家拖出 `Linear`，可以展开看到：

```text
MatMul + BiasAdd + Weight
```

这像真实 IDE 里的 “Go to Definition”。

---

## 风险 4：真实 LLM 太复杂，MVP0.1 做不完

解决方式：

```text
MVP0.1 只做 Tiny Decoder Forward
训练 loop 延后
```

最小 runnable LLM 可以是：

```text
V = 40
T = 8
C = 4
H = 1
N_BLOCKS = 1
```

它不需要真的训练得很好，但必须能完整 forward、输出 logits、采样 token。

---

# 十八、建议的下一步产出

我建议下一步直接做三份文档 / 设计稿：

```text
1. MVP0.1 Component System Spec
   定义 ComponentBlueprint / BuiltComponent / Pack / Certify / Upgrade。

2. Component Evolution Roadmap
   把上面所有组件拆成关卡、依赖、测试、解锁关系。

3. Sample Arc Detailed Design
   选 Scalar → Vector → Matrix → Tensor → MatMulGate 作为第一个完整样例弧。
```

如果要先做一个最能证明方向的 playable slice，我建议不是从完整 LLM 开始，而是做：

```text
Scalar → Vector → Matrix → MatMulGate → Linear
```

因为这个链条短、可视化强、后续复用价值极高。只要玩家真的先造出了 `MatMulGate`，再用它造 `Linear`，就能立刻感受到 MVP0.1 的核心变化：

```text
我不是在用别人给我的节点。
我是在把自己造出来的东西继续升级成更大的系统。
```
