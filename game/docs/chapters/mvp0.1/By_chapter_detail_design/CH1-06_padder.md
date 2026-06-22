# Chapter 1-6 Padder: Padder

## 1. 组件真实用途

Padder 把 TokenBuffer 的 EMPTY 槽位填成 pad id，生成模型能接收的固定长度 token ids。

```text
buffer = [12,4,7,EMPTY,EMPTY]
valid  = [1,1,1,0,0]
pad_id = 0
out    = [12,4,7,0,0]
```

本关必须使用 valid mask，不能简单把所有 0 当成 pad。

## 2. 前置组件

- `component.token_buffer.v1`
- `component.vocab_table.v1`

VocabTable 提供 `<pad>` 的 id，TokenBuffer 提供有效槽位。

## 3. 本关新增能力

- `PadIdLookup`：从 vocab 中读取 `<pad>` id。
- `FillEmptyGate`：按 valid mask 填充 EMPTY。
- `PadContract`：检查输出 `int[B,T]`。
- `PadProbe`：显示哪些位置来自原 token，哪些位置来自 pad。
- `ReferenceChecker`：检查结果。

## 4. 具体案例

Visible case:

```text
buffer = [12,4,7,EMPTY,EMPTY]
valid = [true,true,true,false,false]
pad_id = 0
```

期望：

```text
padded = [[12,4,7,0,0]]
axes = [B,T]
```

## 5. 初始错误图

画布给出：

- `buffer: TokenBufferOutput`
- `vocab: VocabTable`
- `pad_out: PadContract`
- `pad_probe: PadProbe`
- `reference: ReferenceChecker`

缺少 pad id lookup 和 fill gate。

## 6. 目标内部实现

```text
vocab.out -> pad_lookup.vocab
buffer.buffer -> fill.buffer
buffer.valid -> fill.valid
pad_lookup.pad_id -> fill.pad_id
fill.out -> pad_out.x
fill.out -> pad_probe.x
pad_out.out -> reference.x
```

## 7. 玩家操作

1. 拖入 `PadIdLookup`。
2. 从 vocab 中读取 `<pad>` id。
3. 拖入 `FillEmptyGate`。
4. 接入 buffer、valid mask 和 pad id。
5. 连接 PadProbe 与 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- 硬编码 pad id 0：visible 过，变体 vocab 失败。
- 用 `id == 0` 判断 pad：真实 token id 0 hidden case 失败。
- 丢掉 batch 轴：输出 `[T]` 而不是 `[B,T]`。
- 把所有空槽填 `<unk>`：reference 失败。
- 使用预制 Padder：shortcut，结构断言失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 `PadIdLookup` 和 `FillEmptyGate`。
- 输出 dtype 为 int。
- 输出 axes 为 `[B,T]`。
- pad 只出现在 valid=false 的槽位。
- PadProbe 能说明每个 pad 的来源。

## 10. Hidden / Mutation 测试

Hidden case A：pad id 不是 0。

```text
<pad> -> 99
```

Hidden case B：真实 token id 0。

```text
buffer = [12,0,7,EMPTY]
valid = [1,1,1,0]
```

Hidden case C：无 padding。

```text
valid = [1,1,1]
```

输出不得新增 pad。

## 11. 认证后接口

```text
component.padder.v1
inputs:
  buffer: int_or_empty[T]
  valid: bool[T]
  vocab: vocab_table
output:
  token_ids: int[B,T]
```

## 12. 后续调用

AttentionMaskBuilder 会根据 valid mask 生成 padding mask。Padder 负责“填值”，不负责“告诉模型哪些 token 可见”。
