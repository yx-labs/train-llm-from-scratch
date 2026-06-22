# Chapter 1-7 Attention Mask Builder: AttentionMask[B,T]

## 1. 组件真实用途

AttentionMaskBuilder 根据 TokenBuffer 的 valid mask 生成 padding attention mask：

```text
valid = [1,1,1,0,0]
attention_mask = [[1,1,1,0,0]]
```

这张 mask 告诉模型哪些 token 是真实输入，哪些只是 padding。

## 2. 前置组件

- `component.token_buffer.v1`
- `component.padder.v1`

本关不通过 token id 判断 mask，而是使用 TokenBuffer 的 valid 信息。

## 3. 本关新增能力

- `ValidToMaskGate`：把 valid slots 转成 bool/int mask。
- `BatchLift`：添加 B 轴。
- `MaskContract`：检查 dtype 为 bool 或 mask，axes 为 `[B,T]`。
- `MaskProbe`：展示 token 与 mask 的对应关系。
- `ReferenceChecker`：检查 mask 数值。

## 4. 具体案例

Visible case:

```text
padded_ids = [[12,4,7,0,0]]
valid = [true,true,true,false,false]
```

期望：

```text
attention_mask = [[true,true,true,false,false]]
```

## 5. 初始错误图

画布给出：

- `padded_ids: PadderOutput`
- `valid: ValidMask`
- `mask_out: MaskContract`
- `mask_probe: MaskProbe`
- `reference: ReferenceChecker`

缺少 ValidToMaskGate 和 BatchLift。

## 6. 目标内部实现

```text
valid.out -> valid_to_mask.x
valid_to_mask.out -> batch_lift.x
batch_lift.out -> mask_out.x
batch_lift.out -> mask_probe.mask
padded_ids.out -> mask_probe.ids
mask_out.out -> reference.x
```

## 7. 玩家操作

1. 拖入 `ValidToMaskGate`。
2. 将 valid mask 转成 attention mask。
3. 拖入 `BatchLift`，输出 `[B,T]`。
4. 连接 MaskProbe 和 reference。
5. 用含 token id 0 的输入检查 mask 是否仍正确。
6. 检查当前任务并提交认证。

## 8. 错误路径

- 根据 `token_id != pad_id` 生成 mask：真实 id 0 会被误判。
- 输出 `[T]`：缺少 batch 轴。
- 反转 mask：padding 被当成真实 token。
- 使用 causal mask：本关是 padding mask，不是未来遮挡。
- 不接 MaskProbe：玩家看不到 token/mask 对齐。

## 9. Visible 测试

Visible 测试要求：

- 必须使用 valid mask。
- 输出 axes 为 `[B,T]`。
- 前三个位置为 true，后两个为 false。
- MaskProbe 显示 padded id 与 mask 的对应。

## 10. Hidden / Mutation 测试

Hidden case A：真实 token id 0。

```text
padded_ids = [[12,0,7,0]]
valid = [1,1,1,0]
mask = [1,1,1,0]
```

Hidden case B：全部有效。

```text
valid = [1,1,1]
```

Hidden case C：空输入保护。

```text
valid = [0,0,0]
```

认证通过但标记为 warning，后续训练样本应过滤。

## 11. 认证后接口

```text
component.attention_mask.v1
inputs:
  valid: bool[T]
output:
  mask: bool[B,T]
```

## 12. 后续调用

Tokenizer 组件会同时输出 token ids 和 attention mask。Chapter 4 的 causal mask 是另一种 mask，不能和本关的 padding mask 混为一谈。
