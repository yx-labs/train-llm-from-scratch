import type { BootcampLevel, CheckState, TensorNode } from "./workbenchTypes";

type NodeSpec = Omit<TensorNode, "stats" | "sample" | "checks"> & {
  stats?: TensorNode["stats"];
  sample?: string[];
  checks?: TensorNode["checks"];
};

const defaultStats = { min: "-", max: "-", mean: "-" };

function makeNode(spec: NodeSpec): TensorNode {
  return {
    ...spec,
    stats: spec.stats ?? defaultStats,
    sample: spec.sample ?? [],
    checks: spec.checks ?? [{ label: "inspectable", state: "pass", detail: "click to inspect shape, dtype, and sample values" }]
  };
}

function check(label: string, state: CheckState, detail: string) {
  return { label, state, detail };
}

export const matMulGateLevel: BootcampLevel = {
  id: "0-2",
  title: "MatMul Gate",
  subtitle: "Build Linear Projection from dot cells",
  objective: "按 0-2A 到 0-2X 的阶梯挑战，从点积单元搭建 Linear Projection，并证明 hidden[B,T,C] @ weight[C,O] -> out[B,T,O] 能泛化。",
  sceneTitle: "Chapter 0-2 - MatMul Gate",
  sceneSubtitle: "Challenge ladder: one dot cell, one token projection, sequence reuse, batch reuse, orientation trap, Linear assembly, hidden tests.",
  defaultSelectedNodeId: "matmul_gate",
  briefing: [
    "上一关恢复了 hidden[B,T,C]，本关只消费最后一维 C。",
    "Linear Projection 用权重板把每个 C 维 feature vector 投影到 O 维新特征空间。",
    "目标不是记住一个形状，而是掌握 MatMul Contract：被消费轴必须对齐，携带轴必须保留。"
  ],
  knowledgeCards: [
    {
      title: "0-2 MatMul Gate",
      body: "hidden[B,T,C] 里的每个 token 都带着一条 C 维 feature vector。\n\nLinear Projection 会把这条向量送过权重板，生成新的 O 维 feature vector。",
      visual: ["hidden[B,T,C]", "take one C vector", "@ weight[C,O]", "output O vector"]
    },
    {
      title: "Linear Projection",
      body: "矩阵乘法不是视觉装饰。\n\n输入的最后一维 C 必须和权重的第一维 C 对齐；权重的第二维 O 会成为输出 feature 宽度。",
      visual: ["[C] @ [C,O] -> [O]", "[T,C] @ [C,O] -> [T,O]", "[B,T,C] @ [C,O] -> [B,T,O]"]
    },
    {
      title: "本关要修复什么",
      body: "你会先构造一个 Dot Cell，再逐步扩展到 token、sequence、batch，最后处理权重存储方向 [O,C] 和计算方向 [C,O] 的差异。",
      visual: ["Dot Cell", "Token Projection", "Sequence Projection", "Batch Projection", "Orientation Trap", "Linear Module", "Gauntlet"]
    }
  ],
  knowledgeTransition: {
    title: "MatMul Debugger Bootcamp",
    body: "进入画布后，从 0-2A 开始。\n\n每个阶段只引入一个新合同：先看一个输出值怎么来，再扩展到完整 Linear。",
    buttonLabel: "Start MatMul Gate"
  },
  mission: {
    title: "Repair Mission: MatMul Gate Ladder",
    body: "当前训练板会按 0-2A 到 0-2X 逐步打开。你需要把 C 轴消费、O 轴生成、B/T 携带、权重方向修复和 reference allclose 串成一个完整 Linear Projection。",
    success: [
      "0-2A 到 0-2F 每个阶段都在画布上完成槽位修复。",
      "权重方向陷阱按 [O,C] 存储、[C,O] 计算来处理。",
      "最终隐藏测试覆盖标准方向、存储方向反转和尺寸猜测陷阱。"
    ]
  },
  unlocks: ["MatMul Gate", "Weight Plate", "Transpose Switch", "Linear Projection"],
  contracts: [
    "dot: [C] dot [C] -> scalar",
    "token: [C] @ [C,O] -> [O]",
    "sequence: [T,C] @ [C,O] -> [T,O]",
    "batch: [B,T,C] @ [C,O] -> [B,T,O]",
    "storage trap: weight stored [O,C] may need T(W)"
  ],
  nodes: [
    makeNode({
      id: "feature_vector",
      title: "Feature Vector",
      subtitle: "one hidden token",
      kind: "tensor",
      semanticName: "x",
      dtype: "float32",
      shape: "[C]",
      source: "hidden[b,t,:]",
      consumer: "dot_cell.left",
      stats: { min: "-0.8", max: "1.1", mean: "0.06" },
      sample: ["x = [0.2, -0.5, 0.7, 1.0]", "length C must match weight vector"],
      checks: [check("feature axis", "warn", "identify this as the consumed C vector")],
      x: 74,
      y: 168,
      w: 198,
      h: 106,
      color: 0x24608a
    }),
    makeNode({
      id: "weight_vector",
      title: "Weight Vector",
      subtitle: "one output column",
      kind: "parameter",
      semanticName: "w_o",
      dtype: "float32",
      shape: "[C]",
      source: "weight[:,o]",
      consumer: "dot_cell.right",
      stats: { min: "-0.2", max: "0.2", mean: "0.01" },
      sample: ["w_o = [0.1, 0.3, -0.2, 0.4]", "same C length as feature vector"],
      checks: [check("inner length", "warn", "dot product requires two equal-length vectors")],
      x: 340,
      y: 168,
      w: 198,
      h: 106,
      color: 0x5a4c8f
    }),
    makeNode({
      id: "dot_cell",
      title: "Dot Cell",
      subtitle: "multiply and sum",
      kind: "operation",
      semanticName: "dot",
      dtype: "op",
      shape: "[C] dot [C]",
      source: "feature vector, weight vector",
      consumer: "single output scalar",
      sample: ["sum_i x[i] * w[i]", "one scalar fills one output cell"],
      checks: [check("dot contract", "warn", "both input vectors must be C-length")],
      x: 622,
      y: 176,
      w: 184,
      h: 88,
      color: 0x304b6a
    }),
    makeNode({
      id: "dot_scalar",
      title: "Dot Output",
      subtitle: "one output value",
      kind: "scalar",
      semanticName: "y_o",
      dtype: "float32",
      shape: "[]",
      source: "dot_cell",
      consumer: "output vector cell",
      stats: { min: "0.41", max: "0.41", mean: "0.41" },
      sample: ["one scalar", "one cell inside the O vector"],
      checks: [check("scalar", "warn", "the result of one dot cell is rank 0")],
      x: 884,
      y: 178,
      w: 176,
      h: 84,
      color: 0x74491a
    }),
    makeNode({
      id: "token_vector",
      title: "Token Vector",
      subtitle: "single token hidden",
      kind: "tensor",
      semanticName: "x_token",
      dtype: "float32",
      shape: "[C]",
      source: "hidden[b,t,:]",
      consumer: "matmul_gate.left",
      stats: { min: "-0.9", max: "1.2", mean: "0.04" },
      sample: ["visible C=4", "[C] enters the gate"],
      checks: [check("left vector", "warn", "input width is C")],
      x: 78,
      y: 160,
      w: 198,
      h: 108,
      color: 0x24608a
    }),
    makeNode({
      id: "weight_plate",
      title: "Weight Plate",
      subtitle: "C by O projection plate",
      kind: "parameter",
      semanticName: "W",
      dtype: "float32",
      shape: "[C,O]",
      source: "trainable parameter",
      consumer: "matmul_gate.right",
      stats: { min: "-0.12", max: "0.13", mean: "0.0008" },
      sample: ["compute view: [C,O]", "storage trap may show [O,C]"],
      checks: [check("orientation", "warn", "first axis must match consumed C at compute time")],
      x: 348,
      y: 160,
      w: 202,
      h: 112,
      color: 0x5a4c8f
    }),
    makeNode({
      id: "matmul_gate",
      title: "MatMul Gate",
      subtitle: "inner-dim latch",
      kind: "operation",
      semanticName: "matmul",
      dtype: "op",
      shape: "[C]@[C,O]",
      source: "input, weight",
      consumer: "projection output",
      sample: ["consume C", "generate O"],
      checks: [check("inner dims", "warn", "C must meet C before output can form")],
      x: 642,
      y: 170,
      w: 194,
      h: 96,
      color: 0x304b6a
    }),
    makeNode({
      id: "token_output",
      title: "Token Output",
      subtitle: "projected feature vector",
      kind: "tensor",
      semanticName: "y_token",
      dtype: "float32",
      shape: "[O]",
      source: "matmul_gate",
      consumer: "next block",
      stats: { min: "-1.3", max: "1.4", mean: "0.03" },
      sample: ["O output cells", "C is consumed, not preserved"],
      checks: [check("output width", "warn", "last axis is O")],
      x: 930,
      y: 160,
      w: 206,
      h: 108,
      color: 0x24608a
    }),
    makeNode({
      id: "sequence_tensor",
      title: "Sequence Tensor",
      subtitle: "row of token vectors",
      kind: "tensor",
      semanticName: "x_seq",
      dtype: "float32",
      shape: "[T,C]",
      source: "hidden[b,:,:]",
      consumer: "matmul_gate.left",
      stats: { min: "-1.0", max: "1.1", mean: "0.02" },
      sample: ["T token positions", "each row has C features"],
      checks: [check("carrier axis", "warn", "T is carried, C is consumed")],
      x: 76,
      y: 150,
      w: 210,
      h: 124,
      color: 0x24608a
    }),
    makeNode({
      id: "sequence_output",
      title: "Sequence Output",
      subtitle: "projected token row",
      kind: "tensor",
      semanticName: "y_seq",
      dtype: "float32",
      shape: "[T,O]",
      source: "matmul_gate",
      consumer: "next block",
      stats: { min: "-1.6", max: "1.5", mean: "0.01" },
      sample: ["same T positions", "each row now has O features"],
      checks: [check("sequence contract", "warn", "T should remain outside the dot product")],
      x: 930,
      y: 150,
      w: 210,
      h: 124,
      color: 0x24608a
    }),
    makeNode({
      id: "linear_input",
      title: "Hidden Input",
      subtitle: "activation tensor",
      kind: "tensor",
      semanticName: "hidden",
      dtype: "float32",
      shape: "[B,T,C]",
      source: "0-1 hidden tensor",
      consumer: "matmul_gate.left",
      stats: { min: "-0.9", max: "1.2", mean: "0.04" },
      sample: ["visible: [2,4,8]", "B/T are carried axes"],
      checks: [check("left contract", "pass", "last dim is C")],
      x: 76,
      y: 164,
      w: 210,
      h: 124,
      color: 0x24608a
    }),
    makeNode({
      id: "transpose_switch",
      title: "Transpose Switch",
      subtitle: "storage -> compute view",
      kind: "operation",
      semanticName: "transpose",
      dtype: "op",
      shape: "[O,C] -> [C,O]",
      source: "stored weight",
      consumer: "matmul_gate.right",
      sample: ["swap last two axes", "values unchanged, axis order changes"],
      checks: [check("orientation fix", "warn", "use only when stored plate is [O,C]")],
      x: 386,
      y: 328,
      w: 210,
      h: 92,
      color: 0x304b6a
    }),
    makeNode({
      id: "linear_module",
      title: "Linear Module",
      subtitle: "MatMul + locked bias",
      kind: "operation",
      semanticName: "linear",
      dtype: "op",
      shape: "bias disabled",
      source: "matmul_gate",
      consumer: "reference_checker",
      sample: ["Linear(x)=x@W", "bias intentionally locked in 0-2"],
      checks: [check("assembly", "warn", "wrap MatMul as a reusable Linear Projection")],
      x: 642,
      y: 328,
      w: 206,
      h: 92,
      color: 0x304b6a
    }),
    makeNode({
      id: "linear_out",
      title: "Linear Output",
      subtitle: "projected hidden",
      kind: "tensor",
      semanticName: "out",
      dtype: "float32",
      shape: "[B,T,O]",
      source: "linear(input, W)",
      consumer: "reference_checker",
      stats: { min: "-1.8", max: "1.7", mean: "0.02" },
      sample: ["output keeps B and T", "last axis is O"],
      checks: [check("target", "warn", "label output feature axis O")],
      x: 936,
      y: 164,
      w: 216,
      h: 124,
      color: 0x24608a
    }),
    makeNode({
      id: "reference_checker",
      title: "Reference Checker",
      subtitle: "allclose gate",
      kind: "scalar",
      semanticName: "max_error",
      dtype: "float32",
      shape: "<= 1e-5",
      source: "linear_out, referenceLinear",
      consumer: "gauntlet",
      stats: { min: "0", max: "1e-5", mean: "0" },
      sample: ["compare against referenceLinear", "shape correct is not enough"],
      checks: [check("numeric check", "warn", "run allclose after assembly")],
      x: 936,
      y: 342,
      w: 216,
      h: 100,
      color: 0x74491a
    }),
    makeNode({
      id: "matmul_tests",
      title: "MatMul Gauntlet",
      subtitle: "hidden cases",
      kind: "scalar",
      semanticName: "hidden_tests",
      dtype: "bool",
      shape: "4 cases",
      source: "reference checker",
      consumer: "chapter unlock",
      sample: ["Case A: standard", "Case B: stored [O,C]", "Case C: no size guess", "Case D: optional large"],
      checks: [check("hidden cases", "warn", "prove the contract generalizes")],
      x: 642,
      y: 470,
      w: 220,
      h: 96,
      color: 0x74491a
    })
  ],
  edges: [
    { id: "e_02_feature_dot", from: "feature_vector", to: "dot_cell", label: "x[C]", color: 0x7dd3fc, flow: "forward" },
    { id: "e_02_weight_dot", from: "weight_vector", to: "dot_cell", label: "w_o[C]", color: 0xfbbf24, flow: "parameter" },
    { id: "e_02_dot_scalar", from: "dot_cell", to: "dot_scalar", label: "scalar", color: 0x22c55e, flow: "check" },
    { id: "e_02_token_gate", from: "token_vector", to: "matmul_gate", label: "x[C]", color: 0x7dd3fc, flow: "forward" },
    { id: "e_02_weight_gate", from: "weight_plate", to: "matmul_gate", label: "W[C,O]", color: 0xfbbf24, flow: "parameter" },
    { id: "e_02_gate_token_out", from: "matmul_gate", to: "token_output", label: "y[O]", color: 0x22c55e, flow: "forward" },
    { id: "e_02_sequence_gate", from: "sequence_tensor", to: "matmul_gate", label: "x[T,C]", color: 0x7dd3fc, flow: "forward" },
    { id: "e_02_gate_sequence_out", from: "matmul_gate", to: "sequence_output", label: "y[T,O]", color: 0x22c55e, flow: "forward" },
    { id: "e_02_batch_gate", from: "linear_input", to: "matmul_gate", label: "hidden[B,T,C]", color: 0x7dd3fc, flow: "forward" },
    { id: "e_02_gate_linear_out", from: "matmul_gate", to: "linear_out", label: "out[B,T,O]", color: 0x22c55e, flow: "forward" },
    { id: "e_02_weight_transpose", from: "weight_plate", to: "transpose_switch", label: "stored [O,C]", color: 0xfbbf24, flow: "parameter", route: "down" },
    { id: "e_02_transpose_gate", from: "transpose_switch", to: "matmul_gate", label: "compute [C,O]", color: 0xfbbf24, flow: "parameter", route: "down" },
    { id: "e_02_gate_module", from: "matmul_gate", to: "linear_module", label: "core op", color: 0x60a5fa, flow: "check", route: "down" },
    { id: "e_02_module_out", from: "linear_module", to: "linear_out", label: "Linear output", color: 0x22c55e, flow: "forward", route: "down" },
    { id: "e_02_out_reference", from: "linear_out", to: "reference_checker", label: "allclose", color: 0x22c55e, flow: "check", route: "down" },
    { id: "e_02_reference_tests", from: "reference_checker", to: "matmul_tests", label: "hidden cases", color: 0x60a5fa, flow: "check", route: "down" }
  ],
  repair: {
    kind: "matmul_gate",
    targetContract: "hidden[B,T,C] @ weight[C,O] -> out[B,T,O]",
    brokenMessage: "MatMul Contract is incomplete: C must be consumed, B/T must be carried, O must be generated, and stored [O,C] must be converted to compute [C,O].",
    budget: { probes: 8, referenceRuns: 4 },
    tags: [
      { id: "vec_c", label: "C-length feature vector", shortLabel: "C vec", detail: "the vector length consumed by dot / matmul", category: "axis" },
      { id: "vec_o", label: "O-length output vector", shortLabel: "O vec", detail: "wrong for the input side of Dot Cell", category: "axis" },
      { id: "dot_weight_c", label: "C-length weight vector", shortLabel: "C weight", detail: "one output column has C weights", category: "axis" },
      { id: "dot_weight_o", label: "O-length weight vector", shortLabel: "O weight", detail: "wrong: a single dot cell does not use O weights", category: "axis" },
      { id: "out_scalar", label: "Scalar dot result", shortLabel: "scalar", detail: "one dot cell produces one rank-0 output value", category: "contract" },
      { id: "out_vector", label: "Vector dot result", shortLabel: "vector", detail: "wrong: many dot cells together make a vector", category: "contract" },
      { id: "token_c", label: "Input token axis C", shortLabel: "[C]", detail: "single token feature width", category: "axis" },
      { id: "token_t", label: "Token position axis T", shortLabel: "[T]", detail: "wrong for a single token vector", category: "axis" },
      { id: "weight_co", label: "Compute weight [C,O]", shortLabel: "[C,O]", detail: "first axis matches consumed C, second axis generates O", category: "contract" },
      { id: "weight_oc", label: "Stored weight [O,C]", shortLabel: "[O,C]", detail: "common storage orientation, wrong for direct compute view", category: "contract" },
      { id: "axis_o", label: "Output feature axis O", shortLabel: "O", detail: "projected feature width", category: "axis" },
      { id: "axis_c", label: "Input feature axis C", shortLabel: "C", detail: "wrong on the output side: C was consumed", category: "axis" },
      { id: "preserve_t", label: "Carry T unchanged", shortLabel: "keep T", detail: "sequence positions are carried outside the dot product", category: "axis" },
      { id: "consume_t", label: "Consume T", shortLabel: "eat T", detail: "wrong: MatMul consumes C here, not T", category: "axis" },
      { id: "shape_to", label: "Sequence output [T,O]", shortLabel: "[T,O]", detail: "T is preserved and O is generated", category: "contract" },
      { id: "shape_tc", label: "Sequence output [T,C]", shortLabel: "[T,C]", detail: "wrong: C should not remain the output feature axis", category: "contract" },
      { id: "preserve_bt", label: "Carry B and T unchanged", shortLabel: "keep B/T", detail: "batch and sequence are parallel carrier axes", category: "axis" },
      { id: "consume_b", label: "Consume B", shortLabel: "eat B", detail: "wrong: batch is not part of the dot product", category: "axis" },
      { id: "shape_bto", label: "Batch output [B,T,O]", shortLabel: "[B,T,O]", detail: "full Linear output contract", category: "contract" },
      { id: "shape_tbo", label: "Batch output [T,B,O]", shortLabel: "[T,B,O]", detail: "wrong: batch and sequence order swapped", category: "contract" },
      { id: "shared_weight", label: "Reuse one weight plate", shortLabel: "reuse W", detail: "the same W is applied to every token and sample", category: "operation" },
      { id: "stored_oc", label: "Storage orientation [O,C]", shortLabel: "stored [O,C]", detail: "PyTorch-style Linear weight storage", category: "contract" },
      { id: "stored_co", label: "Storage orientation [C,O]", shortLabel: "stored [C,O]", detail: "wrong for the trap case in this stage", category: "contract" },
      { id: "transpose_weight", label: "Insert Transpose Switch", shortLabel: "T(W)", detail: "turn stored [O,C] into compute [C,O]", category: "operation" },
      { id: "rotate_to_co", label: "Rotate plate to [C,O]", shortLabel: "rotate", detail: "manual orientation fix accepted by this bootcamp", category: "operation" },
      { id: "compute_co", label: "Compute view [C,O]", shortLabel: "compute [C,O]", detail: "the view that can enter MatMul", category: "contract" },
      { id: "compute_oc", label: "Compute view [O,C]", shortLabel: "compute [O,C]", detail: "wrong: inner dimension remains mismatched", category: "contract" },
      { id: "matmul_core", label: "MatMul core", shortLabel: "MatMul", detail: "Linear's core operation when bias is locked", category: "operation" },
      { id: "linear_bias_locked", label: "Bias locked off", shortLabel: "bias off", detail: "0-2 only verifies x @ W", category: "operation" },
      { id: "reference_allclose", label: "Reference allclose", shortLabel: "allclose", detail: "numeric output matches referenceLinear", category: "contract" },
      { id: "gauntlet_case_a", label: "Case A standard orientation", shortLabel: "A", detail: "hidden[B,T,C] with compute weight [C,O]", category: "contract" },
      { id: "gauntlet_case_b", label: "Case B storage orientation", shortLabel: "B", detail: "stored [O,C] repaired before compute", category: "contract" },
      { id: "gauntlet_no_size_guess", label: "No size-based guess", shortLabel: "no guess", detail: "passes cases where C and O change order/size", category: "contract" }
    ],
    slots: [
      {
        id: "dot_left_vector",
        label: "Left Vector",
        nodeId: "feature_vector",
        focusNodeId: "feature_vector",
        emptyLabel: "?",
        correctTagIds: ["vec_c"],
        expected: "left vector length is C",
        successDetail: "Dot Cell consumes a C-length feature vector.",
        failureDetail: "The left side of this dot product should be the C feature vector, not an output axis."
      },
      {
        id: "dot_weight_vector",
        label: "Weight Vector",
        nodeId: "weight_vector",
        focusNodeId: "weight_vector",
        emptyLabel: "?",
        correctTagIds: ["dot_weight_c"],
        expected: "weight vector length is C",
        successDetail: "The weight vector has the same C length as the feature vector.",
        failureDetail: "Dot product requires equal inner lengths; this weight vector must be C-length."
      },
      {
        id: "dot_output_scalar",
        label: "Dot Output",
        nodeId: "dot_scalar",
        focusNodeId: "dot_scalar",
        emptyLabel: "?",
        correctTagIds: ["out_scalar"],
        expected: "one dot cell produces one scalar",
        successDetail: "One output value comes from one multiply-and-sum operation.",
        failureDetail: "A single Dot Cell creates one scalar; the O vector is many dot cells together."
      },
      {
        id: "token_input_c",
        label: "Token Input",
        nodeId: "token_vector",
        focusNodeId: "token_vector",
        emptyLabel: "?",
        correctTagIds: ["token_c"],
        expected: "single token vector is [C]",
        successDetail: "The token input is one C-dimensional feature vector.",
        failureDetail: "This stage uses a single token vector, so T is not an input axis here."
      },
      {
        id: "token_weight_co",
        label: "Weight Plate",
        nodeId: "weight_plate",
        focusNodeId: "weight_plate",
        emptyLabel: "?",
        correctTagIds: ["weight_co"],
        expected: "weight compute view is [C,O]",
        successDetail: "The first weight axis consumes C and the second axis generates O.",
        failureDetail: "Direct token projection needs compute weight [C,O]."
      },
      {
        id: "token_output_o",
        label: "Token Output",
        nodeId: "token_output",
        focusNodeId: "token_output",
        emptyLabel: "?",
        correctTagIds: ["axis_o"],
        expected: "output axis is O",
        successDetail: "C is consumed and the token now has O projected features.",
        failureDetail: "The output is not [C]; C was consumed by the projection."
      },
      {
        id: "sequence_carrier_t",
        label: "Sequence Axis",
        nodeId: "sequence_tensor",
        focusNodeId: "sequence_tensor",
        emptyLabel: "?",
        correctTagIds: ["preserve_t"],
        expected: "T is carried outside the dot product",
        successDetail: "T stays as a carrier axis while each token row consumes C.",
        failureDetail: "T should not be consumed by this MatMul; only C is consumed."
      },
      {
        id: "sequence_weight_shared",
        label: "Shared Weight",
        nodeId: "weight_plate",
        focusNodeId: "weight_plate",
        emptyLabel: "?",
        correctTagIds: ["shared_weight"],
        expected: "same W is reused for every token",
        successDetail: "The same weight plate applies to each T position.",
        failureDetail: "Sequence projection reuses one W; it does not create a different plate per token."
      },
      {
        id: "sequence_output_to",
        label: "Sequence Output",
        nodeId: "sequence_output",
        focusNodeId: "sequence_output",
        emptyLabel: "?",
        correctTagIds: ["shape_to"],
        expected: "output shape is [T,O]",
        successDetail: "T is preserved and C is replaced by O.",
        failureDetail: "Sequence output should be [T,O], not [T,C] or a collapsed vector."
      },
      {
        id: "batch_carrier_bt",
        label: "Batch/Time Axes",
        nodeId: "linear_input",
        focusNodeId: "linear_input",
        emptyLabel: "?",
        correctTagIds: ["preserve_bt"],
        expected: "B and T are carried axes",
        successDetail: "B and T are parallel carrier axes outside the dot product.",
        failureDetail: "Batch and time should not be consumed by Linear Projection."
      },
      {
        id: "batch_weight_shared",
        label: "Shared Weight",
        nodeId: "weight_plate",
        focusNodeId: "weight_plate",
        emptyLabel: "?",
        correctTagIds: ["shared_weight"],
        expected: "one W is reused for every B,T vector",
        successDetail: "Every sample and token position uses the same projection plate.",
        failureDetail: "Linear Projection shares W across B and T."
      },
      {
        id: "batch_output_bto",
        label: "Batch Output",
        nodeId: "linear_out",
        focusNodeId: "linear_out",
        emptyLabel: "?",
        correctTagIds: ["shape_bto"],
        expected: "output shape is [B,T,O]",
        successDetail: "The full batch output keeps B/T and generates O.",
        failureDetail: "The full output contract is [B,T,O]."
      },
      {
        id: "stored_weight_orientation",
        label: "Stored Plate",
        nodeId: "weight_plate",
        focusNodeId: "weight_plate",
        emptyLabel: "?",
        correctTagIds: ["stored_oc"],
        expected: "stored weight is [O,C]",
        successDetail: "You identified the storage orientation as [O,C].",
        failureDetail: "This trap case intentionally stores the weight as [O,C]."
      },
      {
        id: "orientation_fix",
        label: "Orientation Fix",
        nodeId: "transpose_switch",
        focusNodeId: "transpose_switch",
        emptyLabel: "?",
        correctTagIds: ["transpose_weight", "rotate_to_co"],
        expected: "repair stored [O,C] into compute [C,O]",
        successDetail: "The compute path now presents [C,O] to MatMul.",
        failureDetail: "Orientation must be fixed before the weight enters the gate."
      },
      {
        id: "compute_weight_co",
        label: "Compute View",
        nodeId: "matmul_gate",
        focusNodeId: "matmul_gate",
        emptyLabel: "?",
        correctTagIds: ["compute_co"],
        expected: "compute weight is [C,O]",
        successDetail: "MatMul receives weight in the correct compute orientation.",
        failureDetail: "The gate still needs compute view [C,O]."
      },
      {
        id: "linear_core",
        label: "Linear Core",
        nodeId: "linear_module",
        focusNodeId: "linear_module",
        emptyLabel: "?",
        correctTagIds: ["matmul_core"],
        expected: "Linear core is MatMul",
        successDetail: "Linear Projection is assembled around the MatMul core.",
        failureDetail: "For this stage, Linear's core operation should be MatMul."
      },
      {
        id: "bias_locked",
        label: "Bias Policy",
        nodeId: "linear_module",
        focusNodeId: "linear_module",
        emptyLabel: "?",
        correctTagIds: ["linear_bias_locked"],
        expected: "bias is locked off",
        successDetail: "Bias is intentionally excluded from the 0-2 reference.",
        failureDetail: "0-2 verifies projection first; bias stays locked."
      },
      {
        id: "reference_checker",
        label: "Reference Check",
        nodeId: "reference_checker",
        focusNodeId: "reference_checker",
        emptyLabel: "?",
        correctTagIds: ["reference_allclose"],
        expected: "numeric output allclose to reference",
        successDetail: "Shape and numeric output both match the reference implementation.",
        failureDetail: "Shape passing is not enough; the numeric output must match referenceLinear."
      },
      {
        id: "gauntlet_standard",
        label: "Case A",
        nodeId: "matmul_tests",
        focusNodeId: "matmul_tests",
        emptyLabel: "?",
        correctTagIds: ["gauntlet_case_a"],
        expected: "standard orientation passes",
        successDetail: "Standard compute weight [C,O] passes.",
        failureDetail: "Case A should use the normal [B,T,C] @ [C,O] contract."
      },
      {
        id: "gauntlet_storage",
        label: "Case B",
        nodeId: "matmul_tests",
        focusNodeId: "matmul_tests",
        emptyLabel: "?",
        correctTagIds: ["gauntlet_case_b"],
        expected: "stored [O,C] repair passes",
        successDetail: "Stored [O,C] is correctly converted before compute.",
        failureDetail: "Case B must test storage orientation repair, not direct [O,C] compute."
      },
      {
        id: "gauntlet_size",
        label: "Case C",
        nodeId: "matmul_tests",
        focusNodeId: "matmul_tests",
        emptyLabel: "?",
        correctTagIds: ["gauntlet_no_size_guess"],
        expected: "size-based guesses fail",
        successDetail: "The solution follows axis roles instead of guessing from dimension sizes.",
        failureDetail: "Hidden cases are designed to catch max-dimension or fixed-size guesses."
      }
    ],
    probes: [
      {
        id: "shape_probe",
        label: "Shape Probe",
        detail: "shows which axes are consumed and carried",
        budgetCost: 1,
        observations: {
          dot_left_vector: {
            id: "dot_left_shape",
            title: "Dot Cell left side is C-length",
            detail: "The left input is one token feature vector.",
            evidence: ["x shape: [C]", "no B/T axis in this stage"],
            possibleSemantic: "C vector",
            confidence: "high"
          },
          sequence_carrier_t: {
            id: "sequence_t_shape",
            title: "T is repeated rows",
            detail: "Each T row gets the same projection.",
            evidence: ["input: [T,C]", "dot happens along C for every T"],
            possibleSemantic: "keep T",
            confidence: "high"
          },
          batch_carrier_bt: {
            id: "batch_bt_shape",
            title: "B and T are carrier axes",
            detail: "Linear is applied independently at each B,T position.",
            evidence: ["input: [B,T,C]", "weight: [C,O]", "output: [B,T,O]"],
            possibleSemantic: "keep B/T",
            confidence: "high"
          }
        }
      },
      {
        id: "orientation_probe",
        label: "Orientation Probe",
        detail: "checks storage orientation against compute orientation",
        budgetCost: 1,
        observations: {
          stored_weight_orientation: {
            id: "stored_weight_probe",
            title: "Stored plate is [O,C]",
            detail: "The row count is output features, but MatMul needs C first.",
            evidence: ["stored W: [O,C]", "gate needs: [C,O]"],
            possibleSemantic: "stored [O,C]",
            confidence: "high"
          },
          orientation_fix: {
            id: "orientation_fix_probe",
            title: "Transpose repairs axis order",
            detail: "The switch changes axis order without changing values.",
            evidence: ["before: [O,C]", "after: [C,O]"],
            possibleSemantic: "T(W)",
            confidence: "high"
          },
          compute_weight_co: {
            id: "compute_weight_probe",
            title: "Gate accepts [C,O]",
            detail: "The input last dimension and weight first dimension now both mean C.",
            evidence: ["hidden last axis: C", "weight first axis: C"],
            possibleSemantic: "compute [C,O]",
            confidence: "high"
          }
        }
      },
      {
        id: "reference_probe",
        label: "Reference Probe",
        detail: "shows the referenceLinear contract",
        budgetCost: 1,
        observations: {
          reference_checker: {
            id: "reference_allclose_probe",
            title: "Reference compares numbers, not just shape",
            detail: "A transposed or swapped contract may have plausible shape but wrong values.",
            evidence: ["reference: x @ W_compute", "tolerance: 1e-5"],
            possibleSemantic: "allclose",
            confidence: "high"
          },
          gauntlet_size: {
            id: "size_guess_probe",
            title: "Hidden cases change C and O",
            detail: "The largest dimension is not a reliable semantic rule.",
            evidence: ["case: B=1,T=5,C=3,O=7", "case: B=3,T=2,C=9,O=4"],
            possibleSemantic: "no size guess",
            confidence: "medium"
          }
        }
      }
    ],
    checks: [
      {
        id: "dot_cell_contract",
        title: "0-2A Dot Cell produces a scalar",
        group: "visible",
        slotIds: ["dot_left_vector", "dot_weight_vector", "dot_output_scalar"],
        expected: "[C] dot [C] -> scalar",
        passDetail: "One output value now comes from one valid dot product.",
        failDetail: "Dot Cell contract is incomplete.",
        focusNodeId: "dot_cell"
      },
      {
        id: "token_projection_contract",
        title: "0-2B Single token projection",
        group: "visible",
        slotIds: ["token_input_c", "token_weight_co", "token_output_o"],
        expected: "[C] @ [C,O] -> [O]",
        passDetail: "Single token projection consumes C and generates O.",
        failDetail: "Token projection contract is wrong.",
        focusNodeId: "matmul_gate"
      },
      {
        id: "sequence_projection_contract",
        title: "0-2C Sequence projection",
        group: "visible",
        slotIds: ["sequence_carrier_t", "sequence_weight_shared", "sequence_output_to"],
        expected: "[T,C] @ [C,O] -> [T,O]",
        passDetail: "T is carried and the same W is reused across token positions.",
        failDetail: "Sequence projection contract is wrong.",
        focusNodeId: "sequence_tensor"
      },
      {
        id: "batch_projection_contract",
        title: "0-2D Batch projection",
        group: "behavior",
        slotIds: ["batch_carrier_bt", "batch_weight_shared", "batch_output_bto"],
        expected: "[B,T,C] @ [C,O] -> [B,T,O]",
        passDetail: "B/T are carrier axes and C is projected into O.",
        failDetail: "Batch projection contract is wrong.",
        blockedDetail: "Batch projection is blocked until the dot, token, and sequence contracts pass.",
        focusNodeId: "linear_input"
      },
      {
        id: "weight_orientation_contract",
        title: "0-2E Weight orientation trap",
        group: "behavior",
        slotIds: ["stored_weight_orientation", "orientation_fix", "compute_weight_co"],
        expected: "stored [O,C] -> compute [C,O]",
        passDetail: "Stored [O,C] is converted to compute [C,O] before MatMul.",
        failDetail: "Weight orientation remains wrong.",
        blockedDetail: "Orientation trap is blocked until the projection contract is established.",
        focusNodeId: "transpose_switch"
      },
      {
        id: "linear_assembly_contract",
        title: "0-2F Linear module assembly",
        group: "behavior",
        slotIds: ["linear_core", "bias_locked", "reference_checker"],
        expected: "Linear = MatMul core, bias locked, reference allclose",
        passDetail: "Linear Projection is assembled and numerically checked.",
        failDetail: "Linear module assembly is incomplete.",
        blockedDetail: "Linear assembly is blocked until orientation repair passes.",
        focusNodeId: "linear_module"
      },
      {
        id: "linear_reference_allclose",
        title: "reference linear allclose",
        group: "reference",
        slotIds: ["batch_output_bto", "orientation_fix", "compute_weight_co", "reference_checker"],
        expected: "max_error <= 1e-5",
        passDetail: "The repaired Linear matches reference output within tolerance.",
        failDetail: "Reference trace diverged because the Linear contract is wrong.",
        blockedDetail: "Reference trace blocked until visible and behavior contracts pass.",
        focusNodeId: "reference_checker"
      },
      {
        id: "matmul_gauntlet_cases",
        title: "0-2X hidden MatMul cases pass",
        group: "hidden",
        slotIds: ["gauntlet_standard", "gauntlet_storage", "gauntlet_size"],
        expected: "standard, storage-orientation, and size-trap cases pass",
        passDetail: "The MatMul Contract generalizes across hidden cases.",
        failDetail: "Hidden MatMul Gauntlet failed.",
        blockedDetail: "Hidden tests are blocked until reference Linear passes.",
        focusNodeId: "matmul_tests"
      }
    ],
    hiddenCases: [
      "Case A: hidden[2,4,8] @ W[8,16]",
      "Case B: hidden[1,5,6] @ stored_W[10,6] with T(W)",
      "Case C: hidden[3,2,9] @ W[9,4] catches size guessing",
      "Case D optional: hidden[2,7,5] @ W[5,11]"
    ],
    successSummary: "MatMul Gate repaired: Dot Cell, token projection, sequence/batch carrier axes, weight orientation, Linear assembly, and hidden gauntlet all pass."
  },
  traceSteps: [
    { id: "02_step_dot", title: "Dot Cell", state: "warn", detail: "[C] dot [C] -> scalar", selectNodeId: "dot_cell" },
    { id: "02_step_token", title: "Token", state: "warn", detail: "[C] @ [C,O] -> [O]", selectNodeId: "matmul_gate" },
    { id: "02_step_sequence", title: "Sequence", state: "warn", detail: "carry T, reuse W", selectNodeId: "sequence_tensor" },
    { id: "02_step_batch", title: "Batch", state: "warn", detail: "carry B/T, consume C", selectNodeId: "linear_input" },
    { id: "02_step_orientation", title: "Orientation", state: "warn", detail: "stored [O,C] -> [C,O]", selectNodeId: "transpose_switch" },
    { id: "02_step_linear", title: "Linear", state: "warn", detail: "MatMul + reference", selectNodeId: "linear_module" },
    { id: "02_step_gauntlet", title: "Gauntlet", state: "warn", detail: "hidden cases", selectNodeId: "matmul_tests" }
  ]
};
