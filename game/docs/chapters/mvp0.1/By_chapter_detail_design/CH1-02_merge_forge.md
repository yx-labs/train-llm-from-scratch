# Chapter 1-2 Merge Forge

## 1. 组件真实用途

MergeForge 把 BoundarySplitter 产出的细粒度 `string_piece[T]` 合并成更稳定的 `token_piece[T]`。它是 BPE-like tokenizer 的核心压缩步骤。

本关要让玩家看到：

```text
["token", "izer", "s", "are", "use", "ful", "!"]
-> ["tokenizers", "are", "useful", "!"]
```

合并不是随便拼接字符串，而是按照规则表对相邻 pieces 做有序 merge。

## 2. 前置组件

- `component.splitter.v1`

本关不允许使用 `component.merge_forge.v1`。玩家要搭出 pair scan -> rule lookup -> apply merge -> output buffer 的内部结构。

## 3. 建议内部节点

- `PieceCaseSource`：提供 `string_piece[T]`。
- `AdjacentPairScanner`：扫描相邻 pair。
- `MergeRuleTable`：保存允许合并的 pair，例如 `token + izer -> tokenizer`。
- `MergeApplyGate`：按规则应用合并。
- `PieceBuffer`：输出 `token_piece[T]`。
- `TypeContractGate`：检查 dtype/axis。

## 4. 具体案例

Visible case:

```text
input pieces:
["token", "izer", "s", "are", "use", "ful", "!"]

merge rules:
token + izer -> tokenizer
tokenizer + s -> tokenizers
use + ful -> useful

expected:
["tokenizers", "are", "useful", "!"]
```

输出合约：

```text
dtype = token_piece
axes  = [T]
dims  = [4]
```

## 5. 初始错误图

只给出 pieces 输入、合约和 reference。缺少 pair scanner、rule table 和 merge apply。

玩家必须搭出合并过程，而不是拖一个 `MergeRules` 答案组件。

## 6. 目标内部实现

```text
pieces.out -> pair_scanner.pieces
pair_scanner.pairs -> rule_table.pairs
rule_table.matches -> merge_apply.matches
pieces.out -> merge_apply.pieces
merge_apply.out -> token_buffer.pieces
token_buffer.out -> merge_out.x
merge_out.out -> reference.x
```

## 7. 错误路径

- 只做 word split 不做 merge：会得到 7 个 pieces，token budget 变差。
- 只应用一轮 merge：`token + izer` 后没有继续合并 `tokenizer + s`。
- 把所有字母片段强行拼成一个 token：`are` 和 `useful` 边界丢失。
- 丢掉标点：`!` 必须保留为独立 token_piece。

## 8. 测试设计

当前任务：

- 结构断言：必须经过 pair scanner、rule table、merge apply。
- 合约断言：输出是 `token_piece[T=4]`。
- 行为断言：pieces 精确等于 `["tokenizers", "are", "useful", "!"]`。

Hidden cases：

- `["train", "ing", "token", "izer", "s"] -> ["training", "tokenizers"]`
- 没有规则的 piece 必须原样保留。

## 9. 认证变体

公开认证不让玩家手写数组。玩家选择规则集：

- tiny tokenizer rules
- training suffix rules
- no-merge baseline

系统生成 input pieces 和 expected output。认证只改变输入和规则表数据，不改变玩家内部图。

## 10. 认证后接口

```text
component.merge_forge.v1
input:
  pieces: string_piece[T]
output:
  pieces: token_piece[T']
```

## 11. 后续调用

VocabLookup 会把 `token_piece[T]` 映射到 integer IDs。MergeForge 如果只检查 dtype/shape，后续 vocab 会因为 pieces 表面不稳定而失败。
