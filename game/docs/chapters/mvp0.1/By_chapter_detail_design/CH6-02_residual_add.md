# Chapter 6-2 Residual Add: ResidualAdd

## 1. 组件真实用途

ResidualAdd 把子层输出加回原 hidden：

```text
out = residual + branch
```

它要求两侧都是同一 shape 和 axes 的 `[B,T,C]`。

## 2. 前置组件

- `component.bias_add.v1`
- `component.layernorm.v1`

玩家已经学过 broadcast add。本关强调 residual add 不允许 broadcast。

## 3. 本关新增能力

- `SameShapeGate`：检查 residual 与 branch 完全同形。
- `AddGate`：逐元素相加。
- `ResidualTraceProbe`：展示 output cell 的两个来源。
- `ReferenceChecker`：检查数值。

## 4. 具体案例

Visible case:

```text
residual[0,1,2] = 0.4
branch[0,1,2] = -0.1
out[0,1,2] = 0.3
```

## 5. 初始错误图

画布给出 residual、branch、residual_out、trace、reference。缺少 SameShapeGate 和 AddGate。

## 6. 目标内部实现

```text
residual -> same_shape.left
branch -> same_shape.right
same_shape.left_out -> add.left
same_shape.right_out -> add.right
add -> residual_out
add -> trace
residual_out -> reference
```

## 7. 玩家操作

1. 拖入 `SameShapeGate`。
2. 将 residual 和 branch 接入检查。
3. 拖入 `AddGate`。
4. 接入 trace 和 reference。
5. 检查当前任务，再提交认证。

## 8. 错误路径

- branch `[B,T,4C]` 直接相加：shape 失败。
- 使用 broadcast：ResidualAdd 不允许隐式扩展。
- residual/branch 接反不影响加法，但 trace 必须标明来源。
- concat 而不是 add：shape 错。
- 漏接 branch：数值错。

## 9. 测试设计

- Visible：数值逐元素检查。
- Hidden A：branch 全 0，out 应等于 residual。
- Hidden B：shape mismatch 必须失败。
- Hidden C：T==C，仍要同轴同名。

## 10. 认证后接口

```text
component.residual_add.v1
inputs:
  residual: float32[B,T,C]
  branch: float32[B,T,C]
output:
  out: float32[B,T,C]
```

## 11. 后续调用

Attention sublayer 和 MLP sublayer 都以 ResidualAdd 收尾。
