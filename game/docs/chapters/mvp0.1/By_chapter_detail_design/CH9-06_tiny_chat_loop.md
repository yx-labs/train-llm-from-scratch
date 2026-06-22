# Chapter 9-6 Tiny Chat Loop: GenerateLoop

## 1. 组件真实用途

TinyChatLoop 把生成所需组件串成可重复执行的循环：

```text
prompt -> encode -> crop -> next_logits -> sample -> append -> decode
repeat N times
```

这是 MVP0.1 的闭环：训练得到的 tiny model 可以用来生成文本。

## 2. 前置组件

- `component.prompt_encode.v1`
- `component.context_crop.v1`
- `component.forward_last_token.v1`
- `component.sample_next_token.v1`
- `component.append_token.v1`
- `component.token_decode.v1`
- `component.checkpoint_saver.v1`

## 3. 本关新增能力

- `GenerationLoopGate`：按 max_new_tokens 重复执行。
- `LoopStateProbe`：显示每步 tokens、next_id、text。
- `StopTokenGate`：遇到 EOS 时提前停止。
- `ReferenceChecker`：用固定 seed 检查生成序列。

## 4. 具体案例

Visible case:

```text
prompt = "we"
max_new_tokens = 3
seed = 42
```

LoopStateProbe 展示每一步：

```text
step 0: prompt ids
step 1: append next_id
step 2: append next_id
step 3: decode text
```

## 5. 初始错误图

画布给出 prompt、tokenizer、model checkpoint、sampling config、chat_out、loop_probe、reference。缺少循环和停止条件。

## 6. 目标内部实现

```text
prompt + tokenizer -> prompt_encode
prompt_encode -> generation_loop.initial_tokens
checkpoint.weights -> generation_loop.model
sampling_config -> generation_loop.sampler_config
generation_loop uses:
  context_crop -> forward_last_token -> sample_next_token -> append_token -> stop_token
generation_loop.tokens -> token_decode
token_decode.text -> chat_out
generation_loop.trace -> loop_probe
chat_out -> reference
```

## 7. 错误路径

- 只生成一步：max_new_tokens 变体失败。
- 不 crop context：长 prompt 超出模型窗口。
- 不传 seed：认证不可重复。
- 遇 EOS 不停：stop policy 失败。
- 每步都重新从 prompt 开始：tokens 不累积。

## 8. 测试设计

- Visible：固定 seed 生成 3 步。
- Hidden A：max_new_tokens=1。
- Hidden B：模型第一步输出 EOS，必须提前停止。
- Hidden C：prompt 长于 context，必须 crop。

## 9. 认证后接口

```text
component.generate_loop.v1
inputs:
  prompt: raw_text[]
  checkpoint: Checkpoint
  tokenizer: component.tokenizer.v1
  config: GenerationConfig
output:
  text: raw_text[]
```

## 10. 课程闭环

本关证明玩家构建的组件链可以从字符输入走到 token、hidden、transformer、logits、采样，再回到文本输出。它是 MVP0.1 全路线的最终验收关。
