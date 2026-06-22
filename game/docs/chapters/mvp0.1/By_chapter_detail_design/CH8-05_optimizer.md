# Chapter 8-5 Optimizer: AdamW Step

## 1. 组件真实用途

Optimizer 根据参数和梯度产生更新后的参数。MVP0.1 使用简化 AdamW：

```text
m = beta1*m + (1-beta1)*grad
v = beta2*v + (1-beta2)*grad^2
param_next = param - lr * (m_hat/(sqrt(v_hat)+eps) + weight_decay*param)
```

## 2. 前置组件

- `component.backward_pass.v1`
- `component.parameter_matrix.v1`

## 3. 本关新增能力

- `OptimizerStateGate`：读取 m/v/step。
- `AdamMomentUpdate`：更新一阶和二阶矩。
- `WeightDecayGate`：添加 decoupled weight decay。
- `ParamUpdateGate`：生成新参数。
- `StepProbe`：显示 lr、grad norm、delta norm。

## 4. 具体案例

Visible case 使用一个小参数：

```text
param = [1.0, -2.0]
grad = [0.1, -0.2]
lr = 0.01
step = 1
```

输出 `param_next` 必须与 AdamW reference 一致。

## 5. 初始错误图

画布给出 params、grads、optimizer_state、updated_params_out、step_probe、reference。缺少 moment/update gates。

## 6. 目标内部实现

```text
params + grads + state -> adam_moment_update
params + adam_moment_update + weight_decay -> weight_decay_gate
weight_decay_gate + lr -> param_update
param_update -> updated_params_out
param_update -> step_probe
updated_params_out -> reference
```

## 7. 错误路径

- 普通 SGD 代替 AdamW：visible 可能接近，hidden step 失败。
- weight decay 乘到 grad 上而非 decoupled：reference 失败。
- 更新 frozen 参数：contract 失败。
- 忽略 optimizer state：step>1 hidden 失败。

## 8. 测试设计

- Visible：step=1。
- Hidden A：step=2，已有 m/v。
- Hidden B：weight_decay=0。
- Hidden C：frozen 参数不更新。

## 9. 认证后接口

```text
component.adamw_step.v1
inputs:
  params: Parameter[P]
  grads: Gradient[P]
  state: OptimizerState[P]
output:
  next_params: Parameter[P]
  next_state: OptimizerState[P]
```

## 10. 后续调用

Checkpoint 会保存更新后的参数和 optimizer state。
