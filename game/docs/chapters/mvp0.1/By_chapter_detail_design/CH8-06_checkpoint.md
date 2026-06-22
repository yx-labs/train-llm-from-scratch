# Chapter 8-6 Checkpoint: CheckpointSaver

## 1. 组件真实用途

CheckpointSaver 把当前训练状态保存成可恢复快照：

```text
checkpoint = {params, optimizer_state, step, config_hash}
```

本关不训练模型，只验证保存内容完整、可恢复。

## 2. 前置组件

- `component.adamw_step.v1`
- `component.parameter_matrix.v1`

## 3. 本关新增能力

- `StatePackGate`：打包 params、optimizer state、step。
- `ConfigHashGate`：记录模型配置 hash。
- `CheckpointHashGate`：生成快照 hash。
- `RestoreProbe`：从 checkpoint 读取一个参数并比对。
- `ReferenceChecker`：检查 snapshot 完整性。

## 4. 具体案例

Visible case:

```text
step = 10
params = 3 tensors
optimizer_state = m/v for each tensor
```

Checkpoint 必须包含每个参数名和 shape。

## 5. 初始错误图

画布给出 params、state、step、checkpoint_out、restore_probe、reference。缺少 pack/hash/restore。

## 6. 目标内部实现

```text
params + state + step -> state_pack
model_config -> config_hash
state_pack + config_hash -> checkpoint_hash
checkpoint_hash -> checkpoint_out
checkpoint_hash -> restore_probe
checkpoint_out -> reference
```

## 7. 错误路径

- 只保存 params，不保存 optimizer state：恢复训练不一致。
- 参数名丢失：无法匹配更新。
- config hash 缺失：结构变化时误加载。
- hash 不随参数变化：integrity 失败。

## 8. 测试设计

- Visible：3 个参数。
- Hidden A：参数值改变，checkpoint hash 应改变。
- Hidden B：optimizer state 缺项，失败。
- Hidden C：restore probe 读取指定参数并 allclose。

## 9. 认证后接口

```text
component.checkpoint_saver.v1
inputs:
  params: Parameter[P]
  optimizer_state: OptimizerState[P]
  step: int[]
output:
  checkpoint: Checkpoint
```

## 10. 后续调用

Generation 可以加载 checkpoint 参数运行模型。
