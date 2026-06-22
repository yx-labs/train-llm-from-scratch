# Chapter 1-3 Vocab Table: VocabTable

## 1. 组件真实用途

VocabTable 保存 token piece 到 token id 的映射。它不是 lookup 本身，而是后续 VocabLookup 需要查询的表资产。

```text
"we" -> 12
" train" -> 4
" llm" -> 7
"<pad>" -> 0
"<unk>" -> 1
```

本关让玩家构造一张可验证的词表，而不是拖一个已经装好答案的 VocabTable。

## 2. 前置组件

- `component.merge_forge.v1`

玩家已经知道 token piece 是字符串片段。本关给这些 piece 分配稳定 id。

## 3. 本关新增能力

- `VocabRow`：一行 `{piece, id}`。
- `RowStacker`：把多行合成表。
- `UniqueKeyGate`：检查 piece 不重复。
- `UniqueIdGate`：检查 id 不重复，保留 `<pad>=0`。
- `VocabPreviewProbe`：展示按 id 排序后的表。
- `ReferenceChecker`：检查表内容与案例词表一致。

## 4. 具体案例

Visible case 构建 5 行词表：

```text
id 0: "<pad>"
id 1: "<unk>"
id 4: " train"
id 7: " llm"
id 12: "we"
```

注意 `"train"` 和 `" train"` 不同。前导空格是 piece 的一部分。

## 5. 初始错误图

画布给出：

- `row_pad: VocabRow`
- `row_unk: VocabRow`
- `row_we: VocabRow`
- `row_train: VocabRow`
- `row_llm: VocabRow`
- `vocab_out: VocabContract`
- `preview: VocabPreviewProbe`
- `reference: ReferenceChecker`

缺少 RowStacker 和唯一性检查。

## 6. 目标内部实现

```text
row_* -> stacker.rows
stacker.out -> unique_keys.x
unique_keys.out -> unique_ids.x
unique_ids.out -> vocab_out.x
unique_ids.out -> preview.x
vocab_out.out -> reference.x
```

其中：

- `unique_keys.keyField = piece`
- `unique_ids.keyField = id`
- `vocab_out.requiredSpecials = ["<pad>", "<unk>"]`

## 7. 玩家操作

1. 检查每个 VocabRow 的 piece 和 id。
2. 拖入 `RowStacker`，接入所有行。
3. 依次接入 UniqueKeyGate 和 UniqueIdGate。
4. 连接 VocabContract、preview 和 reference。
5. 试着把 `" train"` 改成 `"train"`，观察 reference 失败。
6. 恢复后提交认证。

## 8. 错误路径

- piece 重复：UniqueKeyGate 失败。
- id 重复：UniqueIdGate 失败。
- 缺少 `<unk>`：后续 unknown fallback 无法工作。
- 把 `<pad>` id 改成 9：padder/attention mask 后续失败。
- 自动 trim piece：reference 失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 RowStacker、UniqueKeyGate、UniqueIdGate。
- `<pad>` id 必须是 0。
- `<unk>` 必须存在。
- `" train"` 的前导空格必须保留。
- 表内容 allclose/equal 到 reference map。

## 10. Hidden / Mutation 测试

Hidden case A：重复 piece。

```text
"we" -> 12
"we" -> 13
```

必须失败。

Hidden case B：稀疏 id。

```text
ids = [0,1,4,7,12]
```

稀疏 id 合法，不要求连续。

Hidden case C：未知词保留。

删除 `<unk>` 必须失败。

## 11. 认证后接口

```text
component.vocab_table.v1
inputs:
  rows: VocabRow[]
output:
  vocab: vocab_table
```

## 12. 后续调用

VocabLookup 会读取这张表，把 token pieces 转成 int ids。VocabTable 本身只负责表的完整性和稳定身份。
