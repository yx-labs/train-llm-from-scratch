# Chapter 6-1 LayerNorm: Normalize + Scale + Shift

## 1. 组件真实用途

LayerNorm 对每个 token 的 C 维做标准化，再乘 gamma 加 beta：

```text
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

它稳定 residual stream 的数值范围。

## 2. 前置组件

- `component.channel_stats.v1`
- `component.bias_add.v1`

本关复用统计量和广播思维。

## 3. 本关新增能力

- `EpsilonAdd`：给 variance 加 eps。
- `RsqrtGate`：计算 `1/sqrt(var+eps)`。
- `NormalizeGate`：中心化并缩放。
- `AffineGate`：应用 gamma/beta。
- `NormProbe`：显示标准化后 mean 近似 0、var 近似 1。

## 4. 具体案例

Visible case:

```text
x[B=1,T=2,C=3]
gamma[C] = [1,1,1]
beta[C] = [0,0,0]
eps = 1e-5
```

期望：每个 token 的输出在 C 维上均值约 0，方差约 1。

## 5. 初始错误图

画布给出 x、gamma、beta、ln_out、norm_probe、reference。缺少 stats、eps、rsqrt、affine。

## 6. 目标内部实现

```text
x -> channel_stats
channel_stats.var -> epsilon_add -> rsqrt
x + channel_stats.mean + rsqrt -> normalize
normalize + gamma + beta -> affine
affine -> ln_out
affine -> norm_probe
ln_out -> reference
```

## 7. 玩家操作

1. 拖入 `ChannelStats v1`。
2. 加 eps 并计算 rsqrt。
3. 用 NormalizeGate 标准化 x。
4. 将 gamma/beta broadcast 到 `[B,T,C]` 并 affine。
5. 接入 NormProbe 和 reference。
6. 检查当前任务，再提交认证。

## 8. 错误路径

- 沿 B/T 归一化：NormProbe 失败。
- 忘记 eps：常量 hidden 产生 NaN。
- 只标准化不 affine：非默认 gamma/beta hidden 失败。
- gamma/beta 对齐 T：axis 错。
- 使用预制 LayerNorm：shortcut。

## 9. 测试设计

- Visible：gamma=1,beta=0。
- Hidden A：非默认 gamma/beta。
- Hidden B：常量 token，必须无 NaN。
- Hidden C：T==C，防止 axis 错。

## 10. 认证后接口

```text
component.layernorm.v1
inputs:
  x: float32[B,T,C]
  gamma: float32[C]
  beta: float32[C]
output:
  y: float32[B,T,C]
```

## 11. 后续调用

Attention sublayer 和 MLP sublayer 都采用 pre-norm 结构：先 LayerNorm，再子层，再 ResidualAdd。
