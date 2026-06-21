# Chapter 1-1 Splitter: BoundarySplitter

## 1. 组件真实用途

BoundarySplitter 是文本流水线的第一个可构建组件。它不负责 vocab、不负责 token id，也不做 embedding；它只把一个 `raw_text` 句子切成有顺序的 `string_piece[T]`。

本关要让玩家建立一个清晰认知：

```text
"tokenizers are useful!" -> ["tokenizers", "are", "useful", "!"]
```

标点不是噪声。在后续 tokenizer 中，标点可能对应独立 token，因此本关必须保留标点边界。

## 2. 前置组件

- `component.tensor_box.v1`

本关暂不依赖 `component.text_input.v1`，因为 TextInput 仍是 roadmap 节点；画布直接提供 `TextInput` 作为数据源。玩家构建的是 `component.splitter.v1`，因此本关禁止使用 `component.splitter.v1` 本身。

## 3. 本关新增能力

- `TextInput`：预制数据源，从测试案例读取 `raw_text`。
- `BoundarySplitter`：真正的玩家实现节点，按 word / punctuation 边界输出 pieces。
- `PieceBuffer`：领域探针，检查输出是有限、有序的 `string_piece[T]` buffer。
- `TypeContractGate`：输出合约，检查 dtype / shape / axis。

`PieceBuffer` 不是答案节点。它的作用是让玩家看到“切出来的 pieces 进入了一个 T 轴缓冲”，避免把 raw text 直接接进合约。

## 4. 具体案例

Visible case:

```text
raw_text = "tokenizers are useful!"

expected pieces:
T0 = "tokenizers"
T1 = "are"
T2 = "useful"
T3 = "!"
```

输出合约：

```text
dtype = string_piece
axes  = [T]
dims  = [4]
```

## 5. 初始错误图

画布只给出：

- `text: TextInput`
- `pieces_out: TypeContractGate`

没有 splitter，没有 piece buffer，也没有连线。玩家面对的是一个缺少内部实现的组件蓝图。

## 6. 目标内部实现

目标子图：

```text
text.out -> splitter.text
splitter.pieces -> piece_buffer.pieces
piece_buffer.out -> pieces_out.x
```

其中：

- `splitter.moduleId = BoundarySplitter`
- `splitter.policy = word`
- `splitter.preservePunctuation = true`
- `piece_buffer.moduleId = PieceBuffer`
- `pieces_out.expectedDType = string_piece`
- `pieces_out.expectedAxes = [T]`

## 7. 玩家操作

1. 从组件库拖入 `BoundarySplitter`。
2. 从组件库拖入 `PieceBuffer`。
3. 连接 `text.out -> splitter.text`。
4. 连接 `splitter.pieces -> piece_buffer.pieces`。
5. 连接 `piece_buffer.out -> pieces_out.x`。
6. 运行“检查当前任务”。
7. 选择一个公开文本变体并“提交认证”。

## 8. 错误路径

- `text.out -> pieces_out.x`：raw_text 不是 string_piece，结构断言也会提示缺少 splitter。
- 跳过 `PieceBuffer`：即使 pieces dtype 看起来正确，也无法证明输出经过 T 轴缓冲。
- `preservePunctuation = false`：visible case 会丢掉 `!`，`pieces_equal` 失败。
- 改成 char split：dtype/axis 可能仍对，但 pieces 序列不等于期望。

## 9. Visible 测试

测试分三层：

- 结构断言：必须存在 `BoundarySplitter` 和 `PieceBuffer`，路径必须经过它们。
- 合约断言：`pieces_out` 必须是 `string_piece[T=4]`。
- 行为断言：pieces 必须精确等于 `["tokenizers", "are", "useful", "!"]`。

## 10. Hidden / Mutation 测试

Hidden case:

```text
raw_text = "shape, token buffer"
expected = ["shape", ",", "token", "buffer"]
```

公开认证让玩家选择小型文本 fixture：

- `shape, token buffer`
- `we train llm.`
- `tokenizers are useful!`

系统提供 raw text 和 expected pieces，不要求玩家手写数组。认证只改变输入案例，不改变玩家的实现图。

## 11. 认证后接口

认证后得到：

```text
component.splitter.v1
input:
  text: raw_text
output:
  pieces: string_piece[T]
```

## 12. 后续调用

后续 MergeRules / TokenBuffer / TokenizerComponent 会继续消费 `string_piece[T]`。如果本关只检查 dtype 和 shape，玩家会误以为“任何长度为 T 的字符串数组都可以”；因此本关必须检查实际 pieces 行为。
