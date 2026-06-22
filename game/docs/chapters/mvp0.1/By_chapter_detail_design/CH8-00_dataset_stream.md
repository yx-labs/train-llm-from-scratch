# Chapter 8-0 Dataset Stream: TokenStream

## 1. 组件真实用途

DatasetStream 把训练语料编码后的 token ids 组织成一条长流：

```text
stream[S] = [12,4,7,12,4,0,...]
```

它不是 batch，也不是单个 prompt，而是可采样的训练数据源。

## 2. 前置组件

- `component.tokenizer.v1`

## 3. 本关新增能力

- `CorpusShardSource`：提供一段文本或 token id shard。
- `TokenizerReuseGate`：用已认证 tokenizer 编码文本。
- `StreamConcatGate`：拼接多个 shard。
- `StreamContract`：检查 dtype int、axis `[S]`。
- `StreamProbe`：显示长度、前几个 ids 和 EOS 分隔。

## 4. 具体案例

Visible case:

```text
texts = ["we train llm", "we test"]
stream = tokenizer(text0) + [eos] + tokenizer(text1) + [eos]
```

## 5. 初始错误图

画布给出 corpus shards、tokenizer、stream_out、probe、reference。缺少 reuse 和 concat。

## 6. 目标内部实现

```text
shards -> tokenizer_reuse
tokenizer_reuse.ids -> stream_concat
stream_concat.out -> stream_out
stream_concat.out -> stream_probe
stream_out -> reference
```

## 7. 错误路径

- 每个样本单独输出 `[B,T]`：这不是 stream。
- 忘记 EOS：window sampler 会跨样本错位。
- 手写 ids 绕过 tokenizer：结构失败。
- dtype float：contract 失败。

## 8. 测试设计

- Visible：两个 shard 拼接。
- Hidden A：空 shard 被跳过但不报错。
- Hidden B：不同文本顺序 checksum 改变。
- Hidden C：没有 EOS 必须失败。

## 9. 认证后接口

```text
component.dataset_stream.v1
inputs:
  shards: raw_text[]
  tokenizer: component.tokenizer.v1
output:
  stream: int[S]
```

## 10. 后续调用

WindowSampler 从 stream 中切出训练窗口。
