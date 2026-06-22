# Chapter 2-0 Parameter Matrix: ParameterMatrix

## 1. 组件真实用途

ParameterMatrix 把普通矩阵标记成可训练参数。它仍然是数值 tensor，但多了参数身份、初始化 seed 和 trainable flag。

```text
param.name = "tok_embed.weight"
param.value: float32[V,C]
param.trainable = true
```

本关让玩家区分“输入数据”和“模型参数”。

## 2. 前置组件

- `component.matrix_board.v1`
- `component.typed_tensor.v1`
- `component.axis_tensor.v1`

玩家已经能描述矩阵 shape 和 axes。本关增加参数元数据。

## 3. 本关新增能力

- `InitMatrixSource`：按 seed 生成 float32 矩阵。
- `TrainableTag`：标记参数是否参与优化。
- `ParameterNameGate`：绑定稳定名称。
- `ParameterContract`：检查 axes、dtype、trainable。
- `ParamPreviewProbe`：显示 shape、name、checksum。

## 4. 具体案例

Visible case:

```text
name = "tok_embed.weight"
shape = [V=5,C=3]
axes = [V,C]
trainable = true
seed = 17
```

输出是一张参数矩阵，不是输入 batch。

## 5. 初始错误图

画布给出：

- `init_matrix: InitMatrixSource`
- `param_out: ParameterContract`
- `preview: ParamPreviewProbe`
- `reference: ReferenceChecker`

缺少 TrainableTag 和 ParameterNameGate。

## 6. 目标内部实现

```text
init_matrix.out -> trainable_tag.x
trainable_tag.out -> name_gate.x
name_gate.out -> param_out.x
name_gate.out -> preview.x
param_out.out -> reference.x
```

其中：

- `trainable_tag.trainable = true`
- `name_gate.name = "tok_embed.weight"`
- `param_out.expectedAxes = [V,C]`

## 7. 玩家操作

1. 查看初始化矩阵 shape。
2. 拖入 `TrainableTag`，设置 true。
3. 拖入 `ParameterNameGate`，填写参数名。
4. 接入 ParameterContract、preview 和 reference。
5. 修改 seed，确认 checksum 改变但 contract 不变。
6. 检查当前任务并提交认证。

## 8. 错误路径

- 把参数当输入 batch：axes 错。
- trainable=false：优化器不会更新，contract 失败。
- 参数名为空：checkpoint 无法保存。
- 用 int 矩阵：dtype 错。
- 硬编码 visible seed 的 reference：seed 变体失败。

## 9. Visible 测试

Visible 测试要求：

- 必须存在 TrainableTag 和 ParameterNameGate。
- 输出 dtype 为 float32。
- axes 为 `[V,C]`。
- trainable 为 true。
- preview 显示 name 和 checksum。

## 10. Hidden / Mutation 测试

Hidden case A：不同 seed。

矩阵值变化，但 shape/axes/trainable 不变。

Hidden case B：不同 V/C。

```text
shape = [V=7,C=4]
```

Hidden case C：非 trainable。

必须失败，除非关卡明确是 frozen 参数。

## 11. 认证后接口

```text
component.parameter_matrix.v1
inputs:
  init: float32[R,C]
params:
  name: string
  trainable: bool
output:
  param: parameter_matrix[R,C]
```

## 12. 后续调用

EmbeddingTable、Linear weight、LM head 都是 ParameterMatrix 的具体用途。这个组件负责参数身份，不负责查表或矩阵乘法。
