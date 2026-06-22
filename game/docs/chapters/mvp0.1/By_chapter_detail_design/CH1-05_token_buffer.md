# Chapter 1-5 Token Buffer: TokenBuffer

## 1. 组件真实用途

TokenBuffer 把一串变长 token ids 放入固定容量 T 的槽位，并保留哪些槽位是真实 token。

```text
ids = [12,4,7]
capacity T = 5
buffer = [12,4,7,EMPTY,EMPTY]
valid = [true,true,true,false,false]
```

Padder 会在下一关把 EMPTY 填成 pad id。本关不能提前 padding。

## 2. 前置组件

- `component.vocab_lookup.v1`

玩家已经能得到 token ids。本关处理固定上下文窗口。

## 3. 本关新增能力

- `CapacityScalar`：上下文长度 T。
- `SlotWriter`：按顺序写入 ids。
- `BoundsGate`：检查 token 数不能超过 T。
- `ValidityMaskProbe`：显示每个槽位是否有效。
- `BufferContract`：输出 buffer 和 valid mask。

## 4. 具体案例

Visible case:

```text
token_ids = [12,4,7]
T = 5
```

期望：

```text
buffer = [12,4,7,EMPTY,EMPTY]
valid  = [1,1,1,0,0]
```

## 5. 初始错误图

画布给出：

- `token_ids: TokenIdSource`
- `capacity: CapacityScalar`
- `buffer_out: BufferContract`
- `valid_probe: ValidityMaskProbe`

缺少 SlotWriter 和 BoundsGate。

## 6. 目标内部实现

```text
token_ids.out -> bounds.ids
capacity.out -> bounds.capacity
bounds.out -> slot_writer.ids
capacity.out -> slot_writer.capacity
slot_writer.buffer -> buffer_out.buffer
slot_writer.valid -> buffer_out.valid
slot_writer.valid -> valid_probe.x
```

## 7. 玩家操作

1. 查看 token ids 和容量 T。
2. 拖入 `BoundsGate`，防止溢出。
3. 拖入 `SlotWriter`，按顺序写槽。
4. 同时连接 buffer 和 valid mask。
5. 检查 ValidityMaskProbe。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- 直接输出 `[12,4,7,0,0]`：提前 padding，valid 信息丢失。
- 不检查 bounds：长输入 hidden case 溢出。
- 反向写槽：buffer 数值顺序失败。
- 丢掉 valid mask：后续无法区分真实 id 0 和 pad 0。
- 把 T 写死为 5：认证 T=3/6 失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 BoundsGate 和 SlotWriter。
- buffer 前三位为 `[12,4,7]`。
- valid 为 `[true,true,true,false,false]`。
- EMPTY 不能被误认为 pad id。

## 10. Hidden / Mutation 测试

Hidden case A：刚好填满。

```text
ids = [2,3,4]
T = 3
valid = [1,1,1]
```

Hidden case B：溢出。

```text
ids length = 6
T = 5
```

必须失败并提示超出上下文容量。

Hidden case C：真实 id 0。

```text
ids = [12,0,7]
valid = [1,1,1,0]
```

认证必须保留 valid mask，不能靠 `id != 0` 判断。

## 11. 认证后接口

```text
component.token_buffer.v1
inputs:
  ids: int[N]
  capacity: int[]
outputs:
  buffer: int_or_empty[T]
  valid: bool[T]
```

## 12. 后续调用

Padder 使用 buffer 和 valid mask 产生 `[B,T]` token ids。TokenBuffer 的关键认知是“容量”和“有效长度”分离。
