# Chapter 7-0 Final LayerNorm: FinalNorm

## 1. 组件真实用途

FinalNorm 在 TransformerStack 输出后再做一次 LayerNorm，稳定进入 LM head 的 hidden：

```text
normed = LayerNorm(stack_out, gamma_final, beta_final)
```

它复用 LayerNorm，但参数名和位置不同。

## 2. 前置组件

- `component.transformer_stack.v1`
- `component.layernorm.v1`

## 3. 本关新增能力

- `FinalNormParamContract`：检查 gamma/beta 属于 final norm。
- `component.layernorm.v1`：执行标准化。
- `FinalNormProbe`：展示 stack_out 与 normed 的分布变化。
- `ReferenceChecker`：数值参考。

## 4. 具体案例

Visible case:

```text
stack_out[B=1,T=3,C=4]
gamma_final[C=4]
beta_final[C=4]
normed[B=1,T=3,C=4]
```

## 5. 初始错误图

画布给出 stack_out、gamma、beta、final_norm_out、probe、reference。缺少参数合约和 LayerNorm。

## 6. 目标内部实现

```text
gamma/beta -> final_param_contract
stack_out -> layernorm.x
final_param_contract.gamma -> layernorm.gamma
final_param_contract.beta -> layernorm.beta
layernorm.y -> final_norm_out
layernorm.y -> probe
final_norm_out -> reference
```

## 7. 错误路径

- 跳过 final norm：shape 对但 logits 分布错误。
- 使用 block 内部 LN 参数：参数身份错。
- 只减 mean 不除 std：reference 失败。
- gamma/beta 对齐 T：axis 错。

## 8. 测试设计

- Visible：allclose 到 LayerNorm reference。
- Hidden A：非默认 gamma/beta。
- Hidden B：常量 token，无 NaN。
- Hidden C：参数名错必须失败。

## 9. 认证后接口

```text
component.final_layernorm.v1
inputs:
  hidden: float32[B,T,C]
  gamma: float32[C]
  beta: float32[C]
output:
  normed: float32[B,T,C]
```

## 10. 后续调用

LMHead 消费 final norm 输出，生成每个 token 的 vocabulary logits。
