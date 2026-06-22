# Chapter 1-0 Text Input: TextInput

## 1. 组件真实用途

TextInput 是文本管线的入口。它把玩家可编辑的 prompt 明确封装成 `raw_text[]`，并保留原始字符内容，供 Splitter 后续处理。

本关不拆词、不查词表，只解决一个问题：图里的文本输入必须是 string scalar，而不是随便接入的数组、数字或已经切好的 token。

## 2. 前置组件

- `component.scalar_cell.v1`

ScalarCell 让玩家理解 rank-0 输出。本关把同样的“单个值”概念迁移到文本 dtype。

## 3. 本关新增能力

- `PromptLiteral`：可编辑字符串。
- `Utf8TextGate`：验证输入是合法文本。
- `RawTextContract`：要求 dtype 为 `raw_text`、shape 为 `[]`。
- `TextPreviewProbe`：显示字符长度、空格和换行，不参与组件输出。
- `ReferenceChecker`：检查文本内容被原样保留。

## 4. 具体案例

Visible case:

```text
prompt = "we train llm"
```

期望输出：

```text
raw_text[] = "we train llm"
```

空格是有效内容，不能被 trim 掉；大小写也不能被自动改写。

## 5. 初始错误图

画布给出：

- `prompt_source: PromptLiteral`
- `text_out: RawTextContract`
- `preview: TextPreviewProbe`
- `reference: ReferenceChecker`

缺少 `Utf8TextGate`。source 直接接合约会被视为未验证文本。

## 6. 目标内部实现

```text
prompt_source.out -> text_gate.x
text_gate.out -> text_out.x
text_gate.out -> preview.x
text_out.out -> reference.x
```

其中：

- `text_gate.moduleId = Utf8TextGate`
- `text_out.expectedDType = raw_text`

## 7. 玩家操作

1. 修改 prompt，确认文本框支持连续输入和 backspace。
2. 拖入 `Utf8TextGate`。
3. 将 PromptLiteral 连接到 text gate。
4. 将 text gate 输出连接到合约、preview 和 reference。
5. 用数字或数组试错，观察 dtype 报错。
6. 恢复合法文本后检查当前任务并提交认证。

## 8. 错误路径

- 输入 `123`：不是 raw text，必须报 dtype 错误。
- 自动 trim：reference 失败，因为空格被删除。
- 自动 lowercase：reference 失败，因为 TextInput 不负责规范化。
- 预先 split 成数组：本关输出必须是 rank-0 raw_text。
- 跳过 TextPreviewProbe：结构断言失败，玩家无法看到空格保留情况。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 `Utf8TextGate`。
- 输出 dtype 为 `raw_text`，shape 为 `[]`。
- 输出内容等于 `"we train llm"`。
- TextPreviewProbe 显示 length 和空格位置。

## 10. Hidden / Mutation 测试

Hidden case A：带标点。

```text
prompt = "hi, llm!"
```

Hidden case B：前后空格。

```text
prompt = " train "
```

认证必须保留空格。

Hidden case C：非法类型。

```text
prompt = [119,101]
```

认证必须失败，而不是把数组解码成文本。

## 11. 认证后接口

```text
component.text_input.v1
inputs:
  prompt: string
output:
  text: raw_text[]
```

## 12. 后续调用

Splitter 会消费 raw_text。TextInput 的价值在于把“用户输入框”变成图上可认证的数据节点，而不是提前把 tokenizer 的责任混进来。
