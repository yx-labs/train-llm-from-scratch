import type { TensorEdge, TensorNode, TraceStep, WorkbenchMode } from "./workbenchTypes";

export const sceneSize = {
  width: 1240,
  height: 640
};

export const modeLabels: Record<WorkbenchMode, { label: string; title: string; description: string }> = {
  build: {
    label: "Build",
    title: "Build Mode / 搭建模式",
    description: "从模块库放置计算块，连接端口，并持续显示 dtype 与 shape 合约。"
  },
  trace: {
    label: "Trace",
    title: "Trace Mode / 单步追踪",
    description: "让数据沿模型流动，暂停查看每个中间 tensor、mask、row_sum 与错误定位。"
  },
  train: {
    label: "Train",
    title: "Train Mode / 训练模拟",
    description: "显示 loss、gradient、optimizer update 与 tokens/sec，但仍保留可检查的真实张量路径。"
  }
};

export const toolboxModules = [
  "Dataset",
  "Tokenizer",
  "Embedding",
  "Linear",
  "QKV Split",
  "QK^T",
  "Causal Mask",
  "Softmax",
  "Attention x V",
  "Residual Add"
];

export const tensorNodes: TensorNode[] = [
  {
    id: "dataset",
    title: "Dataset",
    subtitle: "jsonl shard",
    kind: "source",
    semanticName: "text_batch",
    dtype: "utf8",
    shape: "[B]",
    source: "tiny corpus sampler",
    consumer: "tokenizer.encode",
    stats: { min: "-", max: "-", mean: "-" },
    sample: ["The model learns tensors", "Masks block future tokens"],
    checks: [
      { label: "clean text", state: "pass", detail: "control chars removed" },
      { label: "batch size", state: "pass", detail: "B=2 in this trace" }
    ],
    x: 58,
    y: 154,
    w: 132,
    h: 70,
    color: 0x1a4164
  },
  {
    id: "token_ids",
    title: "Token IDs",
    subtitle: "tokenizer output",
    kind: "matrix",
    semanticName: "token_ids",
    dtype: "int32",
    shape: "[B,T]",
    source: "tokenizer.encode(text_batch)",
    consumer: "embedding_lookup",
    stats: { min: "0", max: "1023", mean: "318.5" },
    sample: ["[154, 420, 91, 77]", "[154, 502, 33, 91]"],
    checks: [
      { label: "dtype", state: "pass", detail: "int32 accepted by embedding table" },
      { label: "sequence length", state: "pass", detail: "T=4 fits context window" }
    ],
    x: 236,
    y: 154,
    w: 142,
    h: 70,
    color: 0x1f4b72
  },
  {
    id: "hidden",
    title: "Embedding Tensor",
    subtitle: "activation block",
    kind: "tensor",
    semanticName: "hidden",
    dtype: "float32",
    shape: "[B,T,C]",
    source: "embedding_table[token_ids]",
    consumer: "qkv_linear",
    stats: { min: "-0.31", max: "0.42", mean: "0.008" },
    sample: ["h[0,0,:] = [-0.12, 0.03, 0.08, ...]", "C=64 channels"],
    checks: [
      { label: "shape", state: "pass", detail: "[B,T,C] matches QKV input" },
      { label: "finite", state: "pass", detail: "no NaN or Inf detected" }
    ],
    x: 448,
    y: 136,
    w: 174,
    h: 104,
    color: 0x24608a
  },
  {
    id: "qkv",
    title: "QKV Projection",
    subtitle: "parameter board",
    kind: "parameter",
    semanticName: "qkv_linear",
    dtype: "float32",
    shape: "[C,3C]",
    source: "hidden @ Wqkv + bqkv",
    consumer: "split_heads",
    stats: { min: "-0.18", max: "0.21", mean: "0.0004" },
    sample: ["Wqkv: 64 x 192", "bias: 192"],
    checks: [
      { label: "params", state: "pass", detail: "12,480 trainable values" },
      { label: "split", state: "pass", detail: "3 tensors -> Q, K, V" }
    ],
    x: 692,
    y: 154,
    w: 150,
    h: 70,
    color: 0x5a4c8f
  },
  {
    id: "attn_scores",
    title: "Attention Scores",
    subtitle: "QK^T / sqrt(D)",
    kind: "attention",
    semanticName: "scores",
    dtype: "float32",
    shape: "[B,H,T,T]",
    source: "Q @ K^T / sqrt(D)",
    consumer: "causal_mask",
    stats: { min: "-2.13", max: "1.74", mean: "-0.06" },
    sample: ["head 0 row 3: [0.18, 0.72, -0.41, 1.10]", "H=4 heatmap layers"],
    checks: [
      { label: "transpose", state: "pass", detail: "K is [B,H,D,T]" },
      { label: "score shape", state: "pass", detail: "[B,H,T,T] produced" }
    ],
    x: 910,
    y: 122,
    w: 224,
    h: 124,
    color: 0x1b6f78
  },
  {
    id: "mask",
    title: "Causal Mask",
    subtitle: "blocked region",
    kind: "mask",
    semanticName: "mask",
    dtype: "bool",
    shape: "[T,T]",
    source: "tril(ones(T,T))",
    consumer: "masked_scores",
    stats: { min: "0", max: "1", mean: "0.625" },
    sample: ["allow j <= i", "block j > i"],
    checks: [
      { label: "direction", state: "pass", detail: "upper triangle is blocked" },
      { label: "debug cell", state: "warn", detail: "cell [2,3] is highlighted for repair flow" }
    ],
    x: 532,
    y: 360,
    w: 156,
    h: 92,
    color: 0x344358
  },
  {
    id: "attn_probs",
    title: "Softmax Probabilities",
    subtitle: "attention map stack",
    kind: "attention",
    semanticName: "attn_probs",
    dtype: "float32",
    shape: "[B,H,T,T]",
    source: "softmax(masked_scores)",
    consumer: "attn_probs @ V",
    stats: { min: "0.0000", max: "0.7631", mean: "0.2500", rowSum: "1.0000 +/- 1e-6" },
    sample: ["row 0: [1.000, 0.000, 0.000, 0.000]", "row 3: [0.21, 0.18, 0.37, 0.24]"],
    checks: [
      { label: "row sum", state: "pass", detail: "all rows sum to 1.0000" },
      { label: "masked cells", state: "pass", detail: "future cells remain 0" }
    ],
    x: 734,
    y: 348,
    w: 204,
    h: 110,
    color: 0x168c96
  },
  {
    id: "block_out",
    title: "Block Output",
    subtitle: "residual stream",
    kind: "tensor",
    semanticName: "block_out",
    dtype: "float32",
    shape: "[B,T,C]",
    source: "merge_heads(attn_probs @ V)",
    consumer: "lm_head",
    stats: { min: "-0.77", max: "0.92", mean: "0.015" },
    sample: ["residual path preserved", "next module: LayerNorm + MLP"],
    checks: [
      { label: "residual shape", state: "pass", detail: "input and output both [B,T,C]" },
      { label: "finite", state: "pass", detail: "activation values in range" }
    ],
    x: 992,
    y: 366,
    w: 154,
    h: 88,
    color: 0x236998
  },
  {
    id: "loss",
    title: "Loss",
    subtitle: "cross entropy",
    kind: "scalar",
    semanticName: "loss",
    dtype: "float32",
    shape: "scalar",
    source: "cross_entropy(logits, targets)",
    consumer: "backward",
    stats: { min: "2.91", max: "3.34", mean: "3.08" },
    sample: ["loss = 3.08", "perplexity = 21.8"],
    checks: [
      { label: "finite", state: "pass", detail: "loss is not NaN" },
      { label: "trend", state: "warn", detail: "needs more batches to prove descent" }
    ],
    x: 992,
    y: 510,
    w: 154,
    h: 58,
    color: 0x74491a
  }
];

export const tensorEdges: TensorEdge[] = [
  { id: "e_dataset_tokens", from: "dataset", to: "token_ids", label: "utf8 text", color: 0x7dd3fc, flow: "forward" },
  { id: "e_tokens_hidden", from: "token_ids", to: "hidden", label: "int[B,T] token_ids", color: 0x7dd3fc, flow: "forward" },
  { id: "e_hidden_qkv", from: "hidden", to: "qkv", label: "float[B,T,C] hidden", color: 0x7dd3fc, flow: "forward" },
  { id: "e_qkv_scores", from: "qkv", to: "attn_scores", label: "float[B,H,T,D] Q/K/V", color: 0x7dd3fc, flow: "forward" },
  { id: "e_scores_mask", from: "attn_scores", to: "mask", label: "float[B,H,T,T] scores", color: 0x7dd3fc, flow: "forward", route: "down" },
  { id: "e_mask_probs", from: "mask", to: "attn_probs", label: "masked scores", color: 0x94a3b8, flow: "mask" },
  { id: "e_probs_out", from: "attn_probs", to: "block_out", label: "float[B,H,T,T] probs @ V", color: 0x7dd3fc, flow: "forward" },
  { id: "e_out_loss", from: "block_out", to: "loss", label: "float[B,T,V] logits", color: 0x7dd3fc, flow: "forward", route: "down" },
  { id: "e_grad_loop", from: "loss", to: "qkv", label: "grad dLoss/dWqkv", color: 0xf59e0b, flow: "gradient", route: "loop" }
];

export const traceSteps: TraceStep[] = [
  { id: "step_qkt", title: "QK^T", state: "pass", detail: "[B,H,T,T]", selectNodeId: "attn_scores" },
  { id: "step_scale", title: "Scale", state: "pass", detail: "divide by sqrt(D)", selectNodeId: "attn_scores" },
  { id: "step_mask", title: "Mask", state: "warn", detail: "inspect cell [2,3]", selectNodeId: "mask" },
  { id: "step_softmax", title: "Softmax", state: "pass", detail: "row sum = 1", selectNodeId: "attn_probs" },
  { id: "step_vsum", title: "V sum", state: "pass", detail: "[B,H,T,D]", selectNodeId: "block_out" },
  { id: "step_loss", title: "Loss", state: "warn", detail: "watch descent over batches", selectNodeId: "loss" }
];

export const nodeById = new Map(tensorNodes.map((node) => [node.id, node]));
