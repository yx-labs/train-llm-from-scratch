import type { BootcampAnswerMap, BootcampLevel, BootcampResult, CheckState, TensorNode } from "./workbenchTypes";

type NodeSpec = Omit<TensorNode, "stats" | "sample" | "checks"> & {
  stats?: TensorNode["stats"];
  sample?: string[];
  checks?: TensorNode["checks"];
};

type EvaluateMetrics = {
  probeUses: number;
  referenceRuns: number;
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

export const bootcampLevels: BootcampLevel[] = [
  {
    id: "0-1",
    title: "Shape Reader",
    subtitle: "Repair the Hidden Tensor contract",
    objective: "按 0-1A 到 0-1X 的阶梯挑战，完成 tensor 对象识别、rank/shape 读取、token grid、embedding 展开、hidden[B,T,C] 修复和下游验证。",
    sceneTitle: "Chapter 0-1 - Shape Reader",
    sceneSubtitle: "Challenge ladder: build tensor intuition first, then repair hidden[B,T,C] and prove it with consumers plus hidden shape tests.",
    defaultSelectedNodeId: "hidden_tensor",
    briefing: [
      "你即将修复一条 LLM 内部数据链路。",
      "知识会按 0-1A 到 0-1X 逐步解锁，每一步都必须通过画布上的工程验证。",
      "先从 tensor 对象开始，不提前背答案。"
    ],
    knowledgeCards: [
      {
        title: "0-1 Shape Reader",
        body: "LLM 的内部不是直接流动文字，而是流动一组组 tensor。\n\n在这个训练关里，你会从最基础的数据对象开始，一步步修复一条模型数据链路。\n\n每个概念只会在需要操作它时解锁。",
        visual: ["Text", "Token IDs", "Embedding", "Hidden Tensor", "Shape Contract", "Downstream Tests"]
      }
    ],
    knowledgeTransition: {
      title: "Tensor Debugger Bootcamp",
      body: "进入画布后，从 0-1A 开始修机器。\n\n不要先背最终合同；先观察对象、操作工具，再让测试告诉你哪里还缺信息。",
      buttonLabel: "Start Bootcamp"
    },
    mission: {
      title: "Repair Mission: Shape Reader Ladder",
      body: "当前训练板会按阶段打开。每个阶段都会在画布上给出一个新工具、一个新对象和一个可验证目标。\n\n先完成 0-1A：把不同数字对象送进 Tensor Inspector。",
      success: [
        "所有操作都在画布上完成：拖组件、连端口、填槽位、运行验证。",
        "每一小阶段只解锁一个新概念。",
        "最终让数据链路和下游测试一起证明合同正确。"
      ]
    },
    unlocks: ["Shape Inspector", "Axis Tags", "Tensor Probe", "Consumer Validation"],
    contracts: [
      "target: hidden[B,T,C]",
      "visible: hidden[2,4,8]",
      "behavior: B separates samples / T preserves order / C feeds Linear",
      "hidden: [1,8,16], [4,3,32], [2,12,6]"
    ],
    nodes: [
      makeNode({
        id: "raw_objects",
        title: "Raw Number Objects",
        subtitle: "tensor candidates",
        kind: "source",
        semanticName: "objects",
        dtype: "float32",
        shape: "0D / 1D / 2D / 3D",
        source: "challenge tray",
        consumer: "tensor inspector",
        sample: ["Scalar: 3.14", "Vector: [0.2,-0.7,1.4]", "Matrix: [[1,2,3],[4,5,6]]", "Tensor Block: stack of matrices"],
        checks: [check("objects", "warn", "inspect all four tensor-like objects")],
        x: 70,
        y: 152,
        w: 174,
        h: 98,
        color: 0x1a4164
      }),
      makeNode({
        id: "tensor_inspector",
        title: "Tensor Inspector",
        subtitle: "accepts tensor-like objects",
        kind: "operation",
        semanticName: "inspector",
        dtype: "tool",
        shape: "dtype/value",
        source: "raw objects",
        consumer: "type check",
        sample: ["dtype: float32", "sample values visible", "rank/shape locked"],
        checks: [check("dtype", "warn", "send scalar, vector, matrix, and 3D tensor into the inspector")],
        x: 306,
        y: 152,
        w: 190,
        h: 98,
        color: 0x304b6a
      }),
      makeNode({
        id: "type_check",
        title: "Type Check",
        subtitle: "waiting for objects",
        kind: "scalar",
        semanticName: "tensor_object_gate",
        dtype: "bool",
        shape: "4 objects",
        source: "tensor inspector",
        consumer: "rank scanner",
        sample: ["scalar accepted", "vector accepted", "matrix accepted", "3D tensor accepted"],
        checks: [check("accepted", "warn", "all four objects must be inspected")],
        x: 560,
        y: 164,
        w: 156,
        h: 78,
        color: 0x74491a
      }),
      makeNode({
        id: "rank_scanner",
        title: "Rank Scanner",
        subtitle: "axis detector",
        kind: "operation",
        semanticName: "rank_scanner",
        dtype: "tool",
        shape: "rank stamps",
        source: "tensor objects",
        consumer: "rank gate",
        sample: ["rank 0: scalar", "rank 1: vector", "rank 2: matrix", "rank 3: tensor block"],
        checks: [check("rank", "warn", "stamp each object with detected rank")],
        x: 292,
        y: 150,
        w: 210,
        h: 100,
        color: 0x304b6a
      }),
      makeNode({
        id: "rank_gate",
        title: "Rank Gate",
        subtitle: "axis count check",
        kind: "scalar",
        semanticName: "rank_gate",
        dtype: "bool",
        shape: "rank == stamp",
        source: "rank scanner",
        consumer: "shape caliper",
        sample: ["vector: one independent axis", "matrix: two independent axes"],
        checks: [check("rank stamps", "warn", "rank stamps must match detected axes")],
        x: 560,
        y: 162,
        w: 156,
        h: 78,
        color: 0x74491a
      }),
      makeNode({
        id: "shape_caliper",
        title: "Shape Caliper",
        subtitle: "measure axis lengths",
        kind: "operation",
        semanticName: "caliper",
        dtype: "tool",
        shape: "axis lengths",
        source: "unknown tensor",
        consumer: "shape gate",
        sample: ["Axis 0 length = 2", "Axis 1 length = 4", "Axis 2 length = 8"],
        checks: [check("measure", "warn", "place measured lengths into shape slots")],
        x: 374,
        y: 156,
        w: 196,
        h: 92,
        color: 0x304b6a
      }),
      makeNode({
        id: "shape_gate",
        title: "Shape Gate",
        subtitle: "needs [2,4,8]",
        kind: "scalar",
        semanticName: "shape_gate",
        dtype: "bool",
        shape: "[?,?,?]",
        source: "shape caliper",
        consumer: "semantic inspector",
        sample: ["shape order follows axis order", "rank 3 implies three slots"],
        checks: [check("shape tag", "warn", "slots must read [2,4,8]")],
        x: 634,
        y: 164,
        w: 156,
        h: 78,
        color: 0x74491a
      }),
      makeNode({
        id: "semantic_inspector",
        title: "Semantic Inspector",
        subtitle: "shape known / semantics missing",
        kind: "operation",
        semanticName: "semantic_contract",
        dtype: "rule",
        shape: "axis[?,?,?]",
        source: "float32[2,4,8]",
        consumer: "axis tags",
        sample: ["Axis 0: unknown", "Axis 1: unknown", "Axis 2: unknown", "downstream modules reject unresolved axes"],
        checks: [check("semantics", "warn", "mark unresolved semantic contract")],
        x: 474,
        y: 158,
        w: 214,
        h: 96,
        color: 0x304b6a
      }),
      makeNode({
        id: "text_batch",
        title: "Text Batch",
        subtitle: "raw examples",
        kind: "source",
        semanticName: "text_batch",
        dtype: "utf8",
        shape: "[B]",
        source: "dataset sampler",
        consumer: "tokenizer",
        sample: ["sample 0: tensor games", "sample 1: masks teach order"],
        checks: [check("batch source", "pass", "independent examples enter together")],
        x: 70,
        y: 166,
        w: 130,
        h: 72,
        color: 0x1a4164
      }),
      makeNode({
        id: "tokenizer",
        title: "Tokenizer",
        subtitle: "input port open",
        kind: "operation",
        semanticName: "tokenizer",
        dtype: "op",
        shape: "utf8[B] -> int[B,T]",
        source: "text_batch",
        consumer: "embedding lookup",
        stats: { min: "-", max: "-", mean: "-" },
        sample: ["encode('tensor games')", "-> [18, 204, 77, 5]"],
        checks: [check("input line", "warn", "connect Text Batch to Tokenizer")],
        x: 230,
        y: 160,
        w: 154,
        h: 84,
        color: 0x304b6a
      }),
      makeNode({
        id: "token_grid",
        title: "Token Grid",
        subtitle: "empty table",
        kind: "tensor",
        semanticName: "token_ids",
        dtype: "int32",
        shape: "[?,?]",
        source: "tokenizer",
        consumer: "embedding lookup",
        stats: { min: "0", max: "9172", mean: "3102" },
        sample: ["          T0    T1    T2    T3", "B0       502  2841  9172     0", "B1      1042  7191  3910     0"],
        checks: [check("grid axes", "warn", "mark rows as B and columns as T")],
        x: 456,
        y: 142,
        w: 214,
        h: 132,
        color: 0x24608a
      }),
      makeNode({
        id: "embedding_table",
        title: "Embedding Table",
        subtitle: "lookup rows",
        kind: "parameter",
        semanticName: "embedding_table",
        dtype: "float32",
        shape: "[V,C]",
        source: "trainable table",
        consumer: "embedding lookup",
        stats: { min: "-0.13", max: "0.14", mean: "0.001" },
        sample: ["row 9172 -> C-wide vector", "V indexes vocabulary rows", "C is vector width"],
        checks: [check("table", "pass", "embedding table provides C-dimensional rows")],
        x: 392,
        y: 332,
        w: 190,
        h: 86,
        color: 0x5a4c8f
      }),
      makeNode({
        id: "embedding_lookup",
        title: "Embedding Lookup",
        subtitle: "table output open",
        kind: "parameter",
        semanticName: "embedding_table",
        dtype: "float32",
        shape: "int[B,T] -> float[B,T,C]",
        source: "tokenizer output",
        consumer: "hidden tensor",
        stats: { min: "3", max: "811", mean: "236.0" },
        sample: ["token ids [B,T] index rows", "each id returns one C-wide vector"],
        checks: [check("lookup line", "warn", "connect Tokenizer output to Embedding Lookup")],
        x: 420,
        y: 152,
        w: 184,
        h: 98,
        color: 0x5a4c8f
      }),
      makeNode({
        id: "hidden_tensor",
        title: "Hidden Tensor",
        subtitle: "not generated",
        kind: "tensor",
        semanticName: "hidden",
        dtype: "float32",
        shape: "[?,?,?]",
        source: "embedding_table[token_ids]",
        consumer: "Batch Viewer / Causal Mask / Linear Probe",
        stats: { min: "-0.41", max: "0.38", mean: "0.006" },
        sample: ["hidden[0,2,:] = [0.04, -0.11, ...]", "visible case: [2,4,8]"],
        checks: [
          check("data input open", "warn", "connect Embedding Lookup before axis labels can be trusted"),
          check("dtype", "pass", "float32 activations can enter Linear")
        ],
        x: 635,
        y: 146,
        w: 212,
        h: 132,
        color: 0x24608a
      }),
      makeNode({
        id: "shape_tests",
        title: "Shape Tests",
        subtitle: "autograder",
        kind: "scalar",
        semanticName: "axis_tests",
        dtype: "bool",
        shape: "visible + hidden",
        source: "consumer validation",
        consumer: "chapter unlock",
        sample: ["visible: hidden[2,4,8]", "hidden: [1,8,16], [4,3,32]"],
        checks: [check("ready", "warn", "contract tests are blocked until slots are filled")],
        x: 870,
        y: 330,
        w: 156,
        h: 76,
        color: 0x74491a
      }),
      makeNode({
        id: "batch_viewer",
        title: "Batch Viewer",
        subtitle: "needs B",
        kind: "operation",
        semanticName: "batch_viewer",
        dtype: "consumer",
        shape: "input: B",
        source: "hidden.B",
        consumer: "sample display",
        sample: ["sample 0", "sample 1"],
        checks: [check("B consumer", "warn", "connect Hidden.B to Batch Viewer")],
        x: 876,
        y: 88,
        w: 174,
        h: 78,
        color: 0x304b6a
      }),
      makeNode({
        id: "causal_mask",
        title: "Causal Mask",
        subtitle: "needs T",
        kind: "mask",
        semanticName: "causal_mask",
        dtype: "bool",
        shape: "[T,T]",
        source: "hidden.T",
        consumer: "attention",
        sample: ["lower triangular", "future tokens blocked"],
        checks: [check("T consumer", "warn", "connect Hidden.T to Causal Mask")],
        x: 876,
        y: 220,
        w: 174,
        h: 78,
        color: 0x304b6a
      }),
      makeNode({
        id: "linear_probe",
        title: "Linear Probe",
        subtitle: "needs C",
        kind: "operation",
        semanticName: "linear_probe",
        dtype: "consumer",
        shape: "[C,O]",
        source: "hidden.C",
        consumer: "projection",
        sample: ["input feature dimension = C", "output hidden[B,T,O]"],
        checks: [check("C consumer", "warn", "connect Hidden.C to Linear Probe")],
        x: 876,
        y: 352,
        w: 174,
        h: 78,
        color: 0x304b6a
      })
    ],
    edges: [
      { id: "e_01_objects_inspector", from: "raw_objects", to: "tensor_inspector", label: "inspect", color: 0x7dd3fc, flow: "forward" },
      { id: "e_01_inspector_type", from: "tensor_inspector", to: "type_check", label: "dtype/value", color: 0xfbbf24, flow: "check" },
      { id: "e_01_objects_rank", from: "raw_objects", to: "rank_scanner", label: "scan axes", color: 0x7dd3fc, flow: "forward" },
      { id: "e_01_rank_gate", from: "rank_scanner", to: "rank_gate", label: "rank stamps", color: 0xfbbf24, flow: "check" },
      { id: "e_01_hidden_caliper", from: "hidden_tensor", to: "shape_caliper", label: "measure axes", color: 0x7dd3fc, flow: "forward" },
      { id: "e_01_caliper_shape", from: "shape_caliper", to: "shape_gate", label: "shape slots", color: 0xfbbf24, flow: "check" },
      { id: "e_01_shape_semantic", from: "shape_gate", to: "semantic_inspector", label: "shape != semantics", color: 0xfbbf24, flow: "check" },
      { id: "e_01_text_tokenizer", from: "text_batch", to: "tokenizer", label: "repair data line", color: 0x7dd3fc, flow: "forward" },
      { id: "e_01_tokenizer_grid", from: "tokenizer", to: "token_grid", label: "token ids", color: 0x7dd3fc, flow: "forward" },
      { id: "e_01_grid_embedding", from: "token_grid", to: "embedding_lookup", label: "ids[B,T]", color: 0x7dd3fc, flow: "forward" },
      { id: "e_01_table_embedding", from: "embedding_table", to: "embedding_lookup", label: "table[V,C]", color: 0x9f7aea, flow: "parameter" },
      { id: "e_01_embedding_hidden", from: "embedding_lookup", to: "hidden_tensor", label: "repair hidden line", color: 0x7dd3fc, flow: "forward" },
      { id: "e_01_hidden_batch", from: "hidden_tensor", to: "batch_viewer", label: "B axis", color: 0xfbbf24, flow: "check" },
      { id: "e_01_hidden_mask", from: "hidden_tensor", to: "causal_mask", label: "T axis", color: 0xfbbf24, flow: "check" },
      { id: "e_01_hidden_linear", from: "hidden_tensor", to: "linear_probe", label: "C axis", color: 0xfbbf24, flow: "check" },
      { id: "e_01_consumers_tests", from: "causal_mask", to: "shape_tests", label: "hidden tests", color: 0xfbbf24, flow: "check", route: "down" }
    ],
    repair: {
      kind: "axis_labels",
      targetContract: "hidden[B,T,C]",
      brokenMessage: "Shape Reader ladder incomplete: tensor object, rank, shape, token grid, embedding expansion, hidden contract, and consumer validation must all pass.",
      budget: { probes: 5, referenceRuns: 3 },
      tags: [
        { id: "object_scalar", label: "Inspect Scalar", shortLabel: "scalar", detail: "send scalar 3.14 into Tensor Inspector", category: "object" },
        { id: "object_vector", label: "Inspect Vector", shortLabel: "vector", detail: "send a 1D vector into Tensor Inspector", category: "object" },
        { id: "object_matrix", label: "Inspect Matrix", shortLabel: "matrix", detail: "send a 2D matrix into Tensor Inspector", category: "object" },
        { id: "object_block", label: "Inspect Tensor Block", shortLabel: "3D", detail: "send a stack of matrices into Tensor Inspector", category: "object" },
        { id: "rank_0", label: "Rank 0", shortLabel: "0", detail: "no independent axes", category: "rank" },
        { id: "rank_1", label: "Rank 1", shortLabel: "1", detail: "one independent axis", category: "rank" },
        { id: "rank_2", label: "Rank 2", shortLabel: "2", detail: "two independent axes", category: "rank" },
        { id: "rank_3", label: "Rank 3", shortLabel: "3", detail: "three independent axes", category: "rank" },
        { id: "len_2", label: "Length 2", shortLabel: "2", detail: "axis length measured as 2", category: "shape" },
        { id: "len_4", label: "Length 4", shortLabel: "4", detail: "axis length measured as 4", category: "shape" },
        { id: "len_8", label: "Length 8", shortLabel: "8", detail: "axis length measured as 8", category: "shape" },
        { id: "semantic_unresolved", label: "Mark Unresolved", shortLabel: "?,?,?", detail: "shape is known, but axis semantics are still missing", category: "semantic" },
        { id: "wire_text_tokenizer", label: "Text -> Tokenizer", shortLabel: "utf8", detail: "connect raw examples into the tokenizer", category: "data" },
        { id: "wire_tokenizer_grid", label: "Tokenizer -> Token Grid", shortLabel: "ids", detail: "connect token ids into the 2D token table", category: "data" },
        { id: "wire_grid_embedding", label: "Token Grid -> Embedding", shortLabel: "ids[B,T]", detail: "connect token_ids[B,T] into embedding lookup", category: "data" },
        { id: "wire_table_embedding", label: "Embedding Table -> Lookup", shortLabel: "V,C", detail: "connect embedding table rows into lookup", category: "data" },
        { id: "wire_embedding_hidden", label: "Embedding -> Hidden", shortLabel: "vec", detail: "connect embedding vectors into hidden tensor", category: "data" },
        { id: "token_axis_b", label: "Token Grid Rows = B", shortLabel: "B", detail: "rows separate independent samples", category: "token" },
        { id: "token_axis_t", label: "Token Grid Columns = T", shortLabel: "T", detail: "columns preserve token positions", category: "token" },
        { id: "token_sample_1", label: "Select Sample 1", shortLabel: "B1", detail: "select the second independent sample row", category: "token" },
        { id: "token_t2", label: "Select T2 Column", shortLabel: "T2", detail: "select token position 2 across all samples", category: "token" },
        { id: "embedding_one", label: "Inspect token_ids[0,2]", shortLabel: "9172", detail: "highlight embedding table row 9172", category: "embedding" },
        { id: "embedding_autofill", label: "Auto Fill Embeddings", shortLabel: "fill", detail: "fill every token cell with its C-wide embedding vector", category: "embedding" },
        { id: "tag_b", label: "[B] Batch", shortLabel: "B", detail: "independent samples", category: "axis" },
        { id: "tag_t", label: "[T] Token Position", shortLabel: "T", detail: "ordered token positions", category: "axis" },
        { id: "tag_c", label: "[C] Channel", shortLabel: "C", detail: "per-token feature channels", category: "axis" },
        { id: "consumer_b", label: "Hidden.B -> Batch Viewer", shortLabel: "B", detail: "Batch Viewer consumes the sample axis", category: "consumer" },
        { id: "consumer_t", label: "Hidden.T -> Causal Mask", shortLabel: "T", detail: "Causal Mask consumes token position axis", category: "consumer" },
        { id: "consumer_c", label: "Hidden.C -> Linear Probe", shortLabel: "C", detail: "Linear Probe consumes feature channel axis", category: "consumer" }
      ],
      slots: [
        {
          id: "object_scalar",
          label: "Scalar",
          nodeId: "tensor_inspector",
          focusNodeId: "tensor_inspector",
          emptyLabel: "open",
          correctTagIds: ["object_scalar"],
          expected: "Scalar object is accepted by Tensor Inspector",
          successDetail: "Scalar is accepted as a tensor-like object.",
          failureDetail: "Tensor Inspector still has not inspected the scalar object."
        },
        {
          id: "object_vector",
          label: "Vector",
          nodeId: "tensor_inspector",
          focusNodeId: "tensor_inspector",
          emptyLabel: "open",
          correctTagIds: ["object_vector"],
          expected: "Vector object is accepted by Tensor Inspector",
          successDetail: "Vector is accepted as a tensor-like object.",
          failureDetail: "Tensor Inspector still has not inspected the vector object."
        },
        {
          id: "object_matrix",
          label: "Matrix",
          nodeId: "tensor_inspector",
          focusNodeId: "tensor_inspector",
          emptyLabel: "open",
          correctTagIds: ["object_matrix"],
          expected: "Matrix object is accepted by Tensor Inspector",
          successDetail: "Matrix is accepted as a tensor-like object.",
          failureDetail: "Tensor Inspector still has not inspected the matrix object."
        },
        {
          id: "object_block",
          label: "3D Tensor",
          nodeId: "tensor_inspector",
          focusNodeId: "tensor_inspector",
          emptyLabel: "open",
          correctTagIds: ["object_block"],
          expected: "3D tensor block is accepted by Tensor Inspector",
          successDetail: "Tensor block is accepted as a tensor-like object.",
          failureDetail: "Tensor Inspector still has not inspected the 3D tensor block."
        },
        {
          id: "rank_scalar",
          label: "Scalar Rank",
          nodeId: "rank_scanner",
          focusNodeId: "rank_scanner",
          emptyLabel: "?",
          correctTagIds: ["rank_0"],
          expected: "Scalar rank = 0",
          successDetail: "Scalar has no independent axes.",
          failureDetail: "Rank mismatch: scalar should be rank 0."
        },
        {
          id: "rank_vector",
          label: "Vector Rank",
          nodeId: "rank_scanner",
          focusNodeId: "rank_scanner",
          emptyLabel: "?",
          correctTagIds: ["rank_1"],
          expected: "Vector rank = 1",
          successDetail: "Vector has one independent axis.",
          failureDetail: "Rank mismatch: vector should be rank 1."
        },
        {
          id: "rank_matrix",
          label: "Matrix Rank",
          nodeId: "rank_scanner",
          focusNodeId: "rank_scanner",
          emptyLabel: "?",
          correctTagIds: ["rank_2"],
          expected: "Matrix rank = 2",
          successDetail: "Matrix has two independent axes.",
          failureDetail: "Rank mismatch: matrix should be rank 2."
        },
        {
          id: "rank_block",
          label: "Tensor Rank",
          nodeId: "rank_scanner",
          focusNodeId: "rank_scanner",
          emptyLabel: "?",
          correctTagIds: ["rank_3"],
          expected: "3D tensor block rank = 3",
          successDetail: "Tensor block has three independent axes.",
          failureDetail: "Rank mismatch: tensor block should be rank 3."
        },
        {
          id: "shape_axis_0",
          label: "Shape Slot 0",
          nodeId: "shape_gate",
          focusNodeId: "shape_gate",
          emptyLabel: "?",
          correctTagIds: ["len_2"],
          expected: "Axis 0 length = 2",
          successDetail: "Axis 0 length is placed in slot 0.",
          failureDetail: "Shape order mismatch: slot 0 should receive length 2."
        },
        {
          id: "shape_axis_1",
          label: "Shape Slot 1",
          nodeId: "shape_gate",
          focusNodeId: "shape_gate",
          emptyLabel: "?",
          correctTagIds: ["len_4"],
          expected: "Axis 1 length = 4",
          successDetail: "Axis 1 length is placed in slot 1.",
          failureDetail: "Shape order mismatch: slot 1 should receive length 4."
        },
        {
          id: "shape_axis_2",
          label: "Shape Slot 2",
          nodeId: "shape_gate",
          focusNodeId: "shape_gate",
          emptyLabel: "?",
          correctTagIds: ["len_8"],
          expected: "Axis 2 length = 8",
          successDetail: "Axis 2 length is placed in slot 2.",
          failureDetail: "Shape order mismatch: slot 2 should receive length 8."
        },
        {
          id: "semantic_unresolved",
          label: "Semantic Contract",
          nodeId: "semantic_inspector",
          focusNodeId: "semantic_inspector",
          emptyLabel: "missing",
          correctTagIds: ["semantic_unresolved"],
          expected: "Shape known, axis semantics unresolved",
          successDetail: "Semantic Inspector now records unresolved axis contract.",
          failureDetail: "Mark the tensor as shape-known but semantics-unresolved before assigning B/T/C."
        },
        {
          id: "flow_text_tokenizer",
          label: "Text -> Tokenizer",
          nodeId: "tokenizer",
          focusNodeId: "tokenizer",
          emptyLabel: "open",
          correctTagIds: ["wire_text_tokenizer"],
          expected: "Text Batch output feeds Tokenizer input",
          successDetail: "Raw examples now reach the Tokenizer.",
          failureDetail: "Tokenizer input still does not receive raw text examples."
        },
        {
          id: "flow_tokenizer_grid",
          label: "Tokenizer -> Grid",
          nodeId: "token_grid",
          focusNodeId: "token_grid",
          emptyLabel: "open",
          correctTagIds: ["wire_tokenizer_grid"],
          expected: "Tokenizer token ids fill Token Grid",
          successDetail: "Token Grid now receives token_ids[B,T].",
          failureDetail: "Token Grid needs integer token ids from the Tokenizer."
        },
        {
          id: "token_grid_b",
          label: "Rows",
          nodeId: "token_grid",
          focusNodeId: "token_grid",
          emptyLabel: "?",
          correctTagIds: ["token_axis_b"],
          expected: "Token Grid rows = B",
          successDetail: "Rows now separate independent samples.",
          failureDetail: "Rows should be B because each row is an independent sample."
        },
        {
          id: "token_grid_t",
          label: "Columns",
          nodeId: "token_grid",
          focusNodeId: "token_grid",
          emptyLabel: "?",
          correctTagIds: ["token_axis_t"],
          expected: "Token Grid columns = T",
          successDetail: "Columns now preserve token positions.",
          failureDetail: "Columns should be T because they preserve token order."
        },
        {
          id: "token_task_sample1",
          label: "Select B1",
          nodeId: "token_grid",
          focusNodeId: "token_grid",
          emptyLabel: "todo",
          correctTagIds: ["token_sample_1"],
          expected: "Select sample 1 row",
          successDetail: "You selected one independent sample row.",
          failureDetail: "Select the B1 row, not a token-position column."
        },
        {
          id: "token_task_t2",
          label: "Select T2",
          nodeId: "token_grid",
          focusNodeId: "token_grid",
          emptyLabel: "todo",
          correctTagIds: ["token_t2"],
          expected: "Select token position 2 across batch",
          successDetail: "You selected T2 across all samples.",
          failureDetail: "Select the T2 column, not a sample row."
        },
        {
          id: "flow_grid_embedding",
          label: "Grid -> Lookup",
          nodeId: "embedding_lookup",
          focusNodeId: "embedding_lookup",
          emptyLabel: "open",
          correctTagIds: ["wire_grid_embedding"],
          expected: "Token Grid token_ids[B,T] feed Embedding Lookup",
          successDetail: "Embedding Lookup now receives token_ids[B,T].",
          failureDetail: "Embedding Lookup requires token_ids[B,T]."
        },
        {
          id: "flow_table_embedding",
          label: "Table -> Lookup",
          nodeId: "embedding_lookup",
          focusNodeId: "embedding_lookup",
          emptyLabel: "open",
          correctTagIds: ["wire_table_embedding"],
          expected: "Embedding Table[V,C] feeds Embedding Lookup",
          successDetail: "Embedding Lookup can now read C-wide rows from the table.",
          failureDetail: "Embedding Lookup needs the embedding table before it can expand tokens."
        },
        {
          id: "embedding_probe",
          label: "Row 9172",
          nodeId: "embedding_lookup",
          focusNodeId: "embedding_lookup",
          emptyLabel: "todo",
          correctTagIds: ["embedding_one"],
          expected: "Inspect token_ids[0,2] through table row 9172",
          successDetail: "The selected token id maps to one C-wide embedding vector.",
          failureDetail: "Inspect token_ids[0,2] before auto filling all embeddings."
        },
        {
          id: "embedding_autofill",
          label: "Auto Fill",
          nodeId: "hidden_tensor",
          focusNodeId: "hidden_tensor",
          emptyLabel: "todo",
          correctTagIds: ["embedding_autofill"],
          expected: "All token cells expand into C-wide vectors",
          successDetail: "Every token cell is expanded into feature channels.",
          failureDetail: "Auto Fill must complete before Hidden Tensor can be trusted."
        },
        {
          id: "flow_embedding_hidden",
          label: "Embedding -> Hidden",
          nodeId: "hidden_tensor",
          focusNodeId: "hidden_tensor",
          emptyLabel: "open",
          correctTagIds: ["wire_embedding_hidden"],
          expected: "Embedding vectors generate Hidden Tensor",
          successDetail: "Hidden Tensor can now be generated from embedding vectors.",
          failureDetail: "Hidden Tensor must be produced by embedding vectors before it can expose axes."
        },
        {
          id: "axis_0",
          label: "Axis 0",
          nodeId: "hidden_tensor",
          focusNodeId: "hidden_tensor",
          emptyLabel: "?",
          correctTagIds: ["tag_b"],
          expected: "Axis 0 = B / batch",
          successDetail: "Axis 0 separates independent examples.",
          failureDetail: "Batch Loader expected the B axis here, but the assigned tag does not separate samples."
        },
        {
          id: "axis_1",
          label: "Axis 1",
          nodeId: "hidden_tensor",
          focusNodeId: "tokenizer",
          emptyLabel: "?",
          correctTagIds: ["tag_t"],
          expected: "Axis 1 = T / token position",
          successDetail: "Axis 1 is ordered and matches token positions.",
          failureDetail: "Causal attention requires token-position axis T here."
        },
        {
          id: "axis_2",
          label: "Axis 2",
          nodeId: "hidden_tensor",
          focusNodeId: "hidden_tensor",
          emptyLabel: "?",
          correctTagIds: ["tag_c"],
          expected: "Axis 2 = C / feature channel",
          successDetail: "Axis 2 is consumed by Linear as feature dimension C.",
          failureDetail: "Linear expected feature axis C, but received a non-channel semantic."
        },
        {
          id: "consumer_b",
          label: "Batch Viewer",
          nodeId: "batch_viewer",
          focusNodeId: "batch_viewer",
          emptyLabel: "open",
          correctTagIds: ["consumer_b"],
          expected: "Hidden.B feeds Batch Viewer",
          successDetail: "Batch Viewer shows independent samples.",
          failureDetail: "Batch Viewer must receive B, not T or C."
        },
        {
          id: "consumer_t",
          label: "Causal Mask",
          nodeId: "causal_mask",
          focusNodeId: "causal_mask",
          emptyLabel: "open",
          correctTagIds: ["consumer_t"],
          expected: "Hidden.T feeds Causal Mask",
          successDetail: "Causal Mask builds a [T,T] lower-triangular board.",
          failureDetail: "Causal Mask must receive T. Attention should not connect samples as time."
        },
        {
          id: "consumer_c",
          label: "Linear Probe",
          nodeId: "linear_probe",
          focusNodeId: "linear_probe",
          emptyLabel: "open",
          correctTagIds: ["consumer_c"],
          expected: "Hidden.C feeds Linear Probe",
          successDetail: "Linear Probe consumes the feature channel dimension.",
          failureDetail: "Linear Probe must receive C as its input feature width."
        }
      ],
      probes: [
        {
          id: "batch_probe",
          label: "Batch Probe",
          detail: "reveals independent sample slices",
          budgetCost: 1,
          observations: {
            axis_0: {
              id: "axis0_batch",
              title: "Axis 0 slices are independent samples",
              detail: "Each slice has its own token sequence.",
              evidence: ["sample 0: tensor games", "sample 1: masks teach order", "sample 2: a small model"],
              possibleSemantic: "B",
              confidence: "high"
            },
            axis_1: {
              id: "axis1_batch",
              title: "Axis 1 does not split independent examples",
              detail: "Slices advance through one sample instead of changing samples.",
              evidence: ["pos 0 -> pos 1 -> pos 2", "ordered within one sentence"],
              possibleSemantic: "T",
              confidence: "medium"
            },
            axis_2: {
              id: "axis2_batch",
              title: "Axis 2 is dense numeric features",
              detail: "Slices are feature values, not separate examples.",
              evidence: ["[-0.04, 0.11, -0.38, ...]", "Linear consumes this dimension"],
              possibleSemantic: "C",
              confidence: "medium"
            }
          }
        },
        {
          id: "time_probe",
          label: "Time Probe",
          detail: "animates token positions",
          budgetCost: 1,
          observations: {
            axis_0: {
              id: "axis0_time",
              title: "Axis 0 jumps between samples",
              detail: "It is not ordered token time.",
              evidence: ["sample 0 text", "sample 1 text"],
              possibleSemantic: "B",
              confidence: "medium"
            },
            axis_1: {
              id: "axis1_time",
              title: "Axis 1 is ordered token position",
              detail: "Causal mask will be built along this dimension.",
              evidence: ["pos 0 -> pos 1 -> pos 2 -> pos 3", "future positions are blocked"],
              possibleSemantic: "T",
              confidence: "high"
            },
            axis_2: {
              id: "axis2_time",
              title: "Axis 2 is not token order",
              detail: "It exposes feature bars for a single token.",
              evidence: ["feature 0", "feature 1", "feature 2"],
              possibleSemantic: "C",
              confidence: "medium"
            }
          }
        },
        {
          id: "channel_probe",
          label: "Channel Probe",
          detail: "shows per-token feature bars",
          budgetCost: 1,
          observations: {
            axis_0: {
              id: "axis0_channel",
              title: "Axis 0 changes the example",
              detail: "This axis is too coarse for feature channels.",
              evidence: ["one full sentence per slice"],
              possibleSemantic: "B",
              confidence: "medium"
            },
            axis_1: {
              id: "axis1_channel",
              title: "Axis 1 changes token position",
              detail: "It controls order, not feature channels.",
              evidence: ["token 0", "token 1", "token 2"],
              possibleSemantic: "T",
              confidence: "medium"
            },
            axis_2: {
              id: "axis2_channel",
              title: "Axis 2 is continuous feature channels",
              detail: "Linear / MLP / QKV projections consume this dimension.",
              evidence: ["[-0.04, 0.11, -0.38, 0.27, ...]", "projection input width = C"],
              possibleSemantic: "C",
              confidence: "high"
            }
          }
        }
      ],
      checks: [
        {
          id: "tensor_objects",
          title: "0-1A tensor objects are accepted",
          group: "visible",
          slotIds: ["object_scalar", "object_vector", "object_matrix", "object_block"],
          expected: "scalar, vector, matrix, and 3D block are tensor-like objects",
          passDetail: "Tensor Inspector accepts all four numeric containers.",
          failDetail: "Tensor Object challenge is incomplete.",
          focusNodeId: "tensor_inspector"
        },
        {
          id: "rank_scanner",
          title: "0-1B rank stamps are correct",
          group: "visible",
          slotIds: ["rank_scalar", "rank_vector", "rank_matrix", "rank_block"],
          expected: "rank == 0, 1, 2, 3",
          passDetail: "Rank Scanner reads the number of axes for each object.",
          failDetail: "Rank Scanner has at least one incorrect stamp.",
          focusNodeId: "rank_scanner"
        },
        {
          id: "shape_caliper",
          title: "0-1C shape lengths are ordered",
          group: "visible",
          slotIds: ["shape_axis_0", "shape_axis_1", "shape_axis_2"],
          expected: "shape == [2,4,8]",
          passDetail: "Shape Caliper records [2,4,8] in axis order.",
          failDetail: "Shape Caliper has a length in the wrong slot.",
          focusNodeId: "shape_gate"
        },
        {
          id: "semantic_gap",
          title: "0-1D shape is not semantics",
          group: "visible",
          slotIds: ["semantic_unresolved"],
          expected: "shape is known while B/T/C semantics are unresolved",
          passDetail: "Semantic Inspector marks the shape-known tensor as unresolved.",
          failDetail: "Shape [2,4,8] was treated as if it already meant [B,T,C].",
          focusNodeId: "semantic_inspector"
        },
        {
          id: "token_grid_builder",
          title: "0-1E token_ids[B,T] is built",
          group: "visible",
          slotIds: ["flow_text_tokenizer", "flow_tokenizer_grid", "token_grid_b", "token_grid_t", "token_task_sample1", "token_task_t2"],
          expected: "Text Batch -> Tokenizer -> token_ids[B,T]",
          passDetail: "Token Grid separates samples by rows and token positions by columns.",
          failDetail: "Token Grid is not a valid token_ids[B,T] board yet.",
          focusNodeId: "token_grid"
        },
        {
          id: "embedding_expansion",
          title: "0-1F token ids expand into embeddings",
          group: "visible",
          slotIds: ["flow_grid_embedding", "flow_table_embedding", "embedding_probe", "embedding_autofill", "flow_embedding_hidden"],
          expected: "token_ids[B,T] + embedding_table[V,C] -> hidden[B,T,C]",
          passDetail: "Every token id expands into one C-wide vector.",
          failDetail: "Embedding expansion is incomplete.",
          focusNodeId: "embedding_lookup"
        },
        {
          id: "hidden_contract",
          title: "0-1G hidden axes are labeled [B,T,C]",
          group: "behavior",
          slotIds: ["axis_0", "axis_1", "axis_2"],
          expected: "axis labels == [B,T,C]",
          passDetail: "Hidden Tensor contract is repaired as hidden[B,T,C].",
          failDetail: "Hidden Tensor axis labels do not match [B,T,C].",
          blockedDetail: "Hidden contract repair is blocked until the tensor object, rank, shape, token grid, and embedding expansion checks pass.",
          focusNodeId: "hidden_tensor"
        },
        {
          id: "consumer_validation",
          title: "0-1H downstream consumers accept B/T/C",
          group: "behavior",
          slotIds: ["consumer_b", "consumer_t", "consumer_c"],
          expected: "B feeds Batch Viewer, T feeds Causal Mask, C feeds Linear Probe",
          passDetail: "Downstream modules consume the repaired contract correctly.",
          failDetail: "At least one downstream consumer receives the wrong semantic axis.",
          blockedDetail: "Consumer validation is blocked until hidden[B,T,C] is repaired.",
          focusNodeId: "shape_tests"
        },
        {
          id: "hidden_cases",
          title: "0-1X hidden shape gauntlet passes",
          group: "hidden",
          slotIds: [
            "object_scalar",
            "object_vector",
            "object_matrix",
            "object_block",
            "rank_scalar",
            "rank_vector",
            "rank_matrix",
            "rank_block",
            "shape_axis_0",
            "shape_axis_1",
            "shape_axis_2",
            "semantic_unresolved",
            "flow_text_tokenizer",
            "flow_tokenizer_grid",
            "token_grid_b",
            "token_grid_t",
            "token_task_sample1",
            "token_task_t2",
            "flow_grid_embedding",
            "flow_table_embedding",
            "embedding_probe",
            "embedding_autofill",
            "flow_embedding_hidden",
            "axis_0",
            "axis_1",
            "axis_2",
            "consumer_b",
            "consumer_t",
            "consumer_c"
          ],
          expected: "hidden[1,8,16], hidden[4,3,32], hidden[2,12,6]",
          passDetail: "The semantic contract generalizes to hidden test shapes.",
          failDetail: "Hidden tests failed because the repair learned numeric positions without preserving semantics.",
          blockedDetail: "Hidden tests are blocked until all visible and behavior challenges pass.",
          focusNodeId: "shape_tests"
        }
      ],
      hiddenCases: ["hidden[1,8,16]", "hidden[4,3,32]", "hidden[2,12,6]"],
      successSummary: "Contract restored: the full 0-1 challenge ladder builds token_ids[B,T], expands hidden[B,T,C], validates consumers, and passes hidden shape cases."
    },
    traceSteps: [
      { id: "01_step_objects", title: "Objects", state: "warn", detail: "inspect tensor-like number containers", selectNodeId: "tensor_inspector" },
      { id: "01_step_shape", title: "Rank / Shape", state: "warn", detail: "read axes and lengths before semantics", selectNodeId: "shape_gate" },
      { id: "01_step_tokens", title: "Token Grid", state: "warn", detail: "build token_ids[B,T]", selectNodeId: "token_grid" },
      { id: "01_step_hidden", title: "Hidden", state: "warn", detail: "expand embeddings and label B/T/C", selectNodeId: "hidden_tensor" },
      { id: "01_step_tests", title: "Gauntlet", state: "warn", detail: "consumer validation + hidden tests", selectNodeId: "shape_tests" }
    ]
  },
  {
    id: "0-2",
    title: "MatMul Gate",
    subtitle: "Repair Linear inner dimension",
    objective: "修复 Linear Gate 的权重方向，让 input[B,T,C] @ weight[C,O] 输出 [B,T,O]。",
    sceneTitle: "Chapter 0-2 - MatMul Gate",
    sceneSubtitle: "Broken board: weight plate is mounted as [O,C], so the inner dimension cannot lock.",
    defaultSelectedNodeId: "linear_gate",
    briefing: [
      "Linear 层要求输入最后一维 C 和权重第一维 C 对齐。",
      "真实规则：input[B,T,C] @ weight[C,O] -> output[B,T,O]。"
    ],
    unlocks: ["Linear Gate", "Weight Plate", "Transpose Switch Preview"],
    contracts: ["input[B,T,C]", "weight[C,O]", "output[B,T,O]", "reference allclose <= 1e-5"],
    nodes: [
      makeNode({
        id: "linear_input",
        title: "Input",
        subtitle: "activation",
        kind: "tensor",
        semanticName: "input",
        dtype: "float32",
        shape: "[B,T,C]",
        source: "hidden tensor",
        consumer: "linear_gate.left",
        stats: { min: "-0.9", max: "1.2", mean: "0.04" },
        sample: ["visible: [2,4,8]", "C=8 enters the Linear gate"],
        checks: [check("left contract", "pass", "last dim is C")],
        x: 76,
        y: 164,
        w: 204,
        h: 118,
        color: 0x24608a
      }),
      makeNode({
        id: "weight_plate",
        title: "Weight Plate",
        subtitle: "mounted wrong",
        kind: "parameter",
        semanticName: "W",
        dtype: "float32",
        shape: "[O,C]",
        source: "trainable parameter",
        consumer: "linear_gate.right",
        stats: { min: "-0.12", max: "0.13", mean: "0.0008" },
        sample: ["current: [O,C]", "required: [C,O] or transpose switch"],
        checks: [check("orientation", "warn", "rotate the plate or insert a transpose switch")],
        x: 350,
        y: 172,
        w: 192,
        h: 102,
        color: 0x5a4c8f
      }),
      makeNode({
        id: "linear_gate",
        title: "Linear Gate",
        subtitle: "inner-dim latch",
        kind: "operation",
        semanticName: "linear",
        dtype: "op",
        shape: "[B,T,C] @ [?,?]",
        source: "input, weight",
        consumer: "linear_out",
        sample: ["inner latch accepts C == C", "red light until weight is repaired"],
        checks: [check("inner dims", "warn", "pending weight repair")],
        x: 640,
        y: 178,
        w: 188,
        h: 92,
        color: 0x304b6a
      }),
      makeNode({
        id: "linear_out",
        title: "Output",
        subtitle: "needs O axis",
        kind: "tensor",
        semanticName: "output",
        dtype: "float32",
        shape: "[B,T,?]",
        source: "linear(input, W)",
        consumer: "reference tests",
        stats: { min: "-1.8", max: "1.7", mean: "0.02" },
        sample: ["output keeps B and T", "last axis must be O"],
        checks: [check("target", "warn", "label output feature axis O")],
        x: 936,
        y: 164,
        w: 210,
        h: 118,
        color: 0x24608a
      })
    ],
    edges: [
      { id: "e_02_input_gate", from: "linear_input", to: "linear_gate", label: "float[B,T,C]", color: 0x7dd3fc, flow: "forward" },
      { id: "e_02_weight_gate", from: "weight_plate", to: "linear_gate", label: "blocked [O,C]", color: 0xfbbf24, flow: "parameter" },
      { id: "e_02_gate_out", from: "linear_gate", to: "linear_out", label: "target [B,T,O]", color: 0x7dd3fc, flow: "forward" }
    ],
    repair: {
      kind: "matmul_gate",
      targetContract: "input[B,T,C] @ weight[C,O] -> output[B,T,O]",
      brokenMessage: "Inner dimension mismatch: Linear expected weight[C,O], but received weight[O,C].",
      budget: { probes: 4, referenceRuns: 3 },
      tags: [
        { id: "weight_oc", label: "Keep [O,C]", shortLabel: "[O,C]", detail: "wrong orientation" },
        { id: "weight_co", label: "Rotate to [C,O]", shortLabel: "[C,O]", detail: "correct weight plate orientation" },
        { id: "transpose_weight", label: "Insert Transpose Switch", shortLabel: "T(W)", detail: "turns [O,C] into [C,O]" },
        { id: "axis_o", label: "Output axis O", shortLabel: "O", detail: "projected feature axis" },
        { id: "axis_c", label: "Output axis C", shortLabel: "C", detail: "wrong: output is no longer input C" }
      ],
      slots: [
        {
          id: "weight_fix",
          label: "Weight Plate",
          nodeId: "weight_plate",
          focusNodeId: "weight_plate",
          emptyLabel: "[O,C]",
          correctTagIds: ["weight_co", "transpose_weight"],
          expected: "weight is [C,O]",
          successDetail: "Weight first dim now matches input C.",
          failureDetail: "Linear Gate expected weight first dim C, but received a non-matching orientation."
        },
        {
          id: "output_axis",
          label: "Output Axis",
          nodeId: "linear_out",
          focusNodeId: "linear_out",
          emptyLabel: "?",
          correctTagIds: ["axis_o"],
          expected: "output last axis = O",
          successDetail: "Output keeps B/T and projects C into O.",
          failureDetail: "Output axis should be O; C is consumed by the projection."
        }
      ],
      probes: [
        {
          id: "shape_probe",
          label: "Shape Probe",
          detail: "checks input and weight port shapes",
          budgetCost: 1,
          observations: {
            weight_fix: {
              id: "weight_shape",
              title: "Weight Plate is mounted [O,C]",
              detail: "The gate needs the first weight axis to be C.",
              evidence: ["input last axis: C=8", "current weight first axis: O=16", "required weight first axis: C=8"],
              possibleSemantic: "[C,O] or T(W)",
              confidence: "high"
            },
            output_axis: {
              id: "output_shape",
              title: "Output axis is produced by the weight columns",
              detail: "MatMul keeps B/T and uses the second weight axis as output width.",
              evidence: ["[B,T,C] @ [C,O] -> [B,T,O]"],
              possibleSemantic: "O",
              confidence: "high"
            }
          }
        },
        {
          id: "reference_probe",
          label: "Reference Probe",
          detail: "shows tiny Linear reference trace",
          budgetCost: 1,
          observations: {
            weight_fix: {
              id: "linear_ref_weight",
              title: "Reference trace transposes the broken plate",
              detail: "The reference path uses W.T before the gate.",
              evidence: ["broken W: [O,C]", "after switch: [C,O]"],
              possibleSemantic: "Transpose Switch",
              confidence: "medium"
            },
            output_axis: {
              id: "linear_ref_output",
              title: "Reference output has projected feature width",
              detail: "The last axis changes from C to O.",
              evidence: ["input: [2,4,8]", "reference output: [2,4,16]"],
              possibleSemantic: "O",
              confidence: "high"
            }
          }
        }
      ],
      checks: [
        {
          id: "inner_dim",
          title: "Linear inner dimension locks",
          group: "visible",
          slotIds: ["weight_fix"],
          expected: "input last dim C == weight first dim C",
          passDetail: "The Linear Gate accepts the repaired weight path.",
          failDetail: "Inner dimension mismatch remains at the weight port.",
          focusNodeId: "linear_gate"
        },
        {
          id: "output_contract",
          title: "output shape is [B,T,O]",
          group: "visible",
          slotIds: ["weight_fix", "output_axis"],
          expected: "output == [B,T,O]",
          passDetail: "Output contract matches [B,T,O].",
          failDetail: "Output axis contract is incomplete or mislabeled.",
          focusNodeId: "linear_out"
        },
        {
          id: "linear_allclose",
          title: "reference matmul allclose",
          group: "reference",
          slotIds: ["weight_fix", "output_axis"],
          expected: "max_error <= 1e-5",
          passDetail: "Your repaired Linear matches reference within tolerance.",
          failDetail: "Reference trace diverged because the Linear contract is wrong.",
          blockedDetail: "Reference trace blocked until visible Linear contract passes.",
          focusNodeId: "linear_out"
        },
        {
          id: "linear_hidden_shapes",
          title: "hidden Linear shapes pass",
          group: "hidden",
          slotIds: ["weight_fix", "output_axis"],
          expected: "[1,8,C] and [4,3,C] cases produce [B,T,O]",
          passDetail: "The repair generalizes to hidden batch/sequence sizes.",
          failDetail: "Hidden shapes failed the Linear contract.",
          blockedDetail: "Hidden tests are blocked until visible Linear contract passes.",
          focusNodeId: "linear_gate"
        }
      ],
      hiddenCases: ["input[1,8,C] @ W[C,O]", "input[4,3,C] @ W[C,O]"],
      successSummary: "Linear Gate repaired: input[B,T,C] @ weight[C,O] -> output[B,T,O]."
    },
    traceSteps: [
      { id: "02_step_fault", title: "Fault", state: "warn", detail: "weight [O,C]", selectNodeId: "weight_plate" },
      { id: "02_step_gate", title: "Gate", state: "warn", detail: "inner dim latch", selectNodeId: "linear_gate" },
      { id: "02_step_out", title: "Output", state: "warn", detail: "label O", selectNodeId: "linear_out" }
    ]
  },
  {
    id: "0-3",
    title: "Transpose Trap",
    subtitle: "Repair QK^T",
    objective: "在 K 路径上插入 Transpose Switch 并交换最后两轴，生成 scores[B,H,T,T]。",
    sceneTitle: "Chapter 0-3 - Transpose Trap",
    sceneSubtitle: "Broken board: Q @ K is wired directly, but attention needs Q @ K.transpose(-2,-1).",
    defaultSelectedNodeId: "transpose_k",
    briefing: [
      "Attention score 不是 Q @ K，而是 Q @ K.transpose(-2,-1)。",
      "如果 Q/K 是 [B,H,T,D]，K^T 必须是 [B,H,D,T]，scores 才是 [B,H,T,T]。"
    ],
    unlocks: ["Transpose Switch", "Attention Score Board"],
    contracts: ["Q[B,H,T,D]", "K^T[B,H,D,T]", "scores[B,H,T,T]", "square over token positions"],
    nodes: [
      makeNode({
        id: "q_tensor",
        title: "Q Tensor",
        subtitle: "query heads",
        kind: "tensor",
        semanticName: "Q",
        dtype: "float32",
        shape: "[B,H,T,D]",
        source: "q projection",
        consumer: "qk_matmul.left",
        stats: { min: "-1.6", max: "1.4", mean: "-0.02" },
        sample: ["visible: [1,2,4,3]", "query axis is T"],
        checks: [check("left shape", "pass", "[B,H,T,D]")],
        x: 76,
        y: 156,
        w: 196,
        h: 116,
        color: 0x24608a
      }),
      makeNode({
        id: "k_tensor",
        title: "K Tensor",
        subtitle: "key heads",
        kind: "tensor",
        semanticName: "K",
        dtype: "float32",
        shape: "[B,H,T,D]",
        source: "k projection",
        consumer: "transpose_k",
        stats: { min: "-1.5", max: "1.7", mean: "0.01" },
        sample: ["raw K matches Q shape", "right operand must become [B,H,D,T]"],
        checks: [check("trap", "warn", "raw K has last dims T,D")],
        x: 76,
        y: 360,
        w: 196,
        h: 116,
        color: 0x24608a
      }),
      makeNode({
        id: "transpose_k",
        title: "Transpose Switch",
        subtitle: "not configured",
        kind: "operation",
        semanticName: "transpose_switch",
        dtype: "op",
        shape: "off",
        source: "K[B,H,T,D]",
        consumer: "qk_matmul.right",
        sample: ["before: K[B,H,T,D]", "target: K^T[B,H,D,T]"],
        checks: [check("required", "warn", "insert switch and set T <-> D")],
        x: 390,
        y: 374,
        w: 200,
        h: 88,
        color: 0x304b6a
      }),
      makeNode({
        id: "qk_matmul",
        title: "QK MatMul",
        subtitle: "blocked",
        kind: "operation",
        semanticName: "qk_scores",
        dtype: "op",
        shape: "[T,D]@[?,?]",
        source: "Q, K path",
        consumer: "scores",
        sample: ["needs [T,D] @ [D,T]", "raw K gives [T,D] @ [T,D]"],
        checks: [check("inner dims", "warn", "D must align with D")],
        x: 650,
        y: 250,
        w: 188,
        h: 92,
        color: 0x304b6a
      }),
      makeNode({
        id: "scores_tensor",
        title: "Scores",
        subtitle: "attention board",
        kind: "attention",
        semanticName: "scores",
        dtype: "float32",
        shape: "[B,H,?,?]",
        source: "Q @ K path",
        consumer: "mask + softmax",
        stats: { min: "-2.4", max: "2.1", mean: "-0.03" },
        sample: ["target: square T x T board", "visible: [1,2,4,4]"],
        checks: [check("target", "warn", "must match [B,H,T,T]")],
        x: 928,
        y: 220,
        w: 226,
        h: 136,
        color: 0x1b6f78
      })
    ],
    edges: [
      { id: "e_03_q_matmul", from: "q_tensor", to: "qk_matmul", label: "Q[B,H,T,D]", color: 0x7dd3fc, flow: "forward" },
      { id: "e_03_k_transpose", from: "k_tensor", to: "transpose_k", label: "K[B,H,T,D]", color: 0x7dd3fc, flow: "forward" },
      { id: "e_03_transpose_matmul", from: "transpose_k", to: "qk_matmul", label: "blocked K path", color: 0xfbbf24, flow: "check" },
      { id: "e_03_matmul_scores", from: "qk_matmul", to: "scores_tensor", label: "target scores", color: 0x7dd3fc, flow: "forward" }
    ],
    repair: {
      kind: "transpose_switch",
      targetContract: "Q[B,H,T,D] @ K^T[B,H,D,T] -> scores[B,H,T,T]",
      brokenMessage: "Q @ K is wired directly. Right operand must swap the last two axes.",
      budget: { probes: 4, referenceRuns: 3 },
      tags: [
        { id: "switch_none", label: "No Switch", shortLabel: "off", detail: "wrong: leaves K as [B,H,T,D]" },
        { id: "switch_transpose", label: "Insert Transpose Switch", shortLabel: "T", detail: "enables axis swap on K path" },
        { id: "swap_td", label: "Swap T <-> D", shortLabel: "T<->D", detail: "correct last-two-axis transpose" },
        { id: "swap_ht", label: "Swap H <-> T", shortLabel: "H<->T", detail: "wrong: breaks heads and sequence" },
        { id: "scores_tt", label: "Scores [B,H,T,T]", shortLabel: "T,T", detail: "square over token positions" },
        { id: "scores_dd", label: "Scores [B,H,D,D]", shortLabel: "D,D", detail: "wrong: D is consumed as inner dim" }
      ],
      slots: [
        {
          id: "k_switch",
          label: "K Path",
          nodeId: "transpose_k",
          focusNodeId: "transpose_k",
          emptyLabel: "off",
          correctTagIds: ["switch_transpose"],
          expected: "Transpose Switch inserted on K path",
          successDetail: "K path now has an axis swap module.",
          failureDetail: "QK MatMul still receives raw K without transpose."
        },
        {
          id: "swap_axes",
          label: "Switch Setting",
          nodeId: "transpose_k",
          focusNodeId: "transpose_k",
          emptyLabel: "?",
          correctTagIds: ["swap_td"],
          expected: "swap last two axes T <-> D",
          successDetail: "K[B,H,T,D] becomes K^T[B,H,D,T].",
          failureDetail: "The switch is swapping the wrong axes."
        },
        {
          id: "score_board",
          label: "Score Board",
          nodeId: "scores_tensor",
          focusNodeId: "scores_tensor",
          emptyLabel: "?",
          correctTagIds: ["scores_tt"],
          expected: "scores shape = [B,H,T,T]",
          successDetail: "Attention score board is square over token positions.",
          failureDetail: "Score board must be T x T, not D x D or reordered axes."
        }
      ],
      probes: [
        {
          id: "matmul_probe",
          label: "MatMul Probe",
          detail: "compares inner dimensions before and after transpose",
          budgetCost: 1,
          observations: {
            k_switch: {
              id: "qk_probe_switch",
              title: "Raw K cannot be the right operand",
              detail: "Q last axis is D, so right operand's second-last axis must also be D.",
              evidence: ["Q: [B,H,T,D]", "raw K: [B,H,T,D]", "needed: [B,H,D,T]"],
              possibleSemantic: "Transpose Switch",
              confidence: "high"
            },
            swap_axes: {
              id: "qk_probe_swap",
              title: "Only the last two axes should swap",
              detail: "B and H are batch-like carrier dimensions and must stay fixed.",
              evidence: ["before: [B,H,T,D]", "after: [B,H,D,T]"],
              possibleSemantic: "T<->D",
              confidence: "high"
            },
            score_board: {
              id: "qk_probe_scores",
              title: "Scores are query token by key token",
              detail: "Each head receives a T x T attention board.",
              evidence: ["[T,D] @ [D,T] -> [T,T]", "visible: [1,2,4,4]"],
              possibleSemantic: "T,T",
              confidence: "high"
            }
          }
        }
      ],
      checks: [
        {
          id: "k_transposed",
          title: "K last two axes swapped",
          group: "visible",
          slotIds: ["k_switch", "swap_axes"],
          expected: "K^T == [B,H,D,T]",
          passDetail: "K path now produces [B,H,D,T].",
          failDetail: "K path has no valid last-two-axis transpose.",
          focusNodeId: "transpose_k"
        },
        {
          id: "scores_shape",
          title: "scores shape is [B,H,T,T]",
          group: "visible",
          slotIds: ["k_switch", "swap_axes", "score_board"],
          expected: "scores == [B,H,T,T]",
          passDetail: "Score board is square over token positions.",
          failDetail: "Score board contract is still wrong.",
          focusNodeId: "scores_tensor"
        },
        {
          id: "qk_reference",
          title: "QK reference allclose",
          group: "reference",
          slotIds: ["k_switch", "swap_axes", "score_board"],
          expected: "max_error <= 1e-5",
          passDetail: "Your QK path matches reference scores.",
          failDetail: "Reference scores diverged.",
          blockedDetail: "Reference trace blocked until QK contract passes.",
          focusNodeId: "scores_tensor"
        },
        {
          id: "qk_hidden_heads",
          title: "hidden head/depth cases pass",
          group: "hidden",
          slotIds: ["k_switch", "swap_axes", "score_board"],
          expected: "[B,1,T,D] and [B,4,T,D] produce [B,H,T,T]",
          passDetail: "Transpose repair generalizes across heads and depth.",
          failDetail: "Hidden attention shapes failed.",
          blockedDetail: "Hidden tests are blocked until visible QK contract passes.",
          focusNodeId: "qk_matmul"
        }
      ],
      hiddenCases: ["Q[2,1,8,16] @ K^T[2,1,16,8]", "Q[1,4,3,32] @ K^T[1,4,32,3]"],
      successSummary: "QK^T repaired: scores[B,H,T,T] is now valid."
    },
    traceSteps: [
      { id: "03_step_fault", title: "Fault", state: "warn", detail: "Q @ K", selectNodeId: "qk_matmul" },
      { id: "03_step_switch", title: "Switch", state: "warn", detail: "insert on K", selectNodeId: "transpose_k" },
      { id: "03_step_scores", title: "Scores", state: "warn", detail: "target T x T", selectNodeId: "scores_tensor" }
    ]
  },
  {
    id: "0-4",
    title: "Broadcast Add",
    subtitle: "Repair broadcast rail",
    objective: "把 pos_emb[T,C] 和 bias[C] 对齐到 hidden[B,T,C] 的正确语义轴，修复 Broadcast Add。",
    sceneTitle: "Chapter 0-4 - Broadcast Add",
    sceneSubtitle: "Broken board: small tensors exist, but their axes are not aligned to the broadcast rail.",
    defaultSelectedNodeId: "broadcast_rule",
    briefing: [
      "Broadcast 会从右侧尾部维度对齐，但玩家还要保证语义轴没有错位。",
      "hidden[B,T,C] + pos_emb[T,C] + bias[C] -> [B,T,C]。"
    ],
    unlocks: ["Broadcast Rail", "Bias Add", "Position Add"],
    contracts: ["hidden[B,T,C]", "pos_emb aligns to [T,C]", "bias aligns to [C]", "result[B,T,C]"],
    nodes: [
      makeNode({
        id: "activation_x",
        title: "Hidden",
        subtitle: "base tensor",
        kind: "tensor",
        semanticName: "hidden",
        dtype: "float32",
        shape: "[B,T,C]",
        source: "previous module",
        consumer: "add_bias.left",
        stats: { min: "-0.9", max: "1.1", mean: "0.03" },
        sample: ["visible: [2,4,8]", "target rail: [B][T][C]"],
        checks: [check("target rank", "pass", "3D activation")],
        x: 82,
        y: 182,
        w: 224,
        h: 128,
        color: 0x24608a
      }),
      makeNode({
        id: "pos_emb",
        title: "Position Emb",
        subtitle: "needs T/C rail",
        kind: "parameter",
        semanticName: "pos_emb",
        dtype: "float32",
        shape: "[T,C]",
        source: "position table",
        consumer: "broadcast_rule",
        stats: { min: "-0.08", max: "0.09", mean: "0.001" },
        sample: ["one vector per position", "align under T and C"],
        checks: [check("rail", "warn", "align to [T,C]")],
        x: 118,
        y: 394,
        w: 180,
        h: 78,
        color: 0x5a4c8f
      }),
      makeNode({
        id: "bias_vector",
        title: "Bias",
        subtitle: "needs C rail",
        kind: "parameter",
        semanticName: "bias",
        dtype: "float32",
        shape: "[C]",
        source: "trainable parameter",
        consumer: "broadcast_rule",
        stats: { min: "-0.04", max: "0.05", mean: "0.002" },
        sample: ["one scalar per channel", "align under C only"],
        checks: [check("rail", "warn", "align to [C]")],
        x: 356,
        y: 394,
        w: 168,
        h: 78,
        color: 0x5a4c8f
      }),
      makeNode({
        id: "broadcast_rule",
        title: "Broadcast Rail",
        subtitle: "not aligned",
        kind: "operation",
        semanticName: "broadcast_add",
        dtype: "rule",
        shape: "[B,T,C] + ?",
        source: "hidden, pos_emb, bias",
        consumer: "add_out",
        sample: ["hidden:  [B][T][C]", "pos_emb:    [T][C]", "bias:          [C]"],
        checks: [check("rule", "warn", "align small tensors to semantic axes")],
        x: 606,
        y: 294,
        w: 220,
        h: 92,
        color: 0x304b6a
      }),
      makeNode({
        id: "biased_out",
        title: "Output",
        subtitle: "broadcast sum",
        kind: "tensor",
        semanticName: "out",
        dtype: "float32",
        shape: "[B,T,C]",
        source: "hidden + pos_emb + bias",
        consumer: "next module",
        stats: { min: "-0.94", max: "1.14", mean: "0.032" },
        sample: ["out[b,t,c] = h[b,t,c] + p[t,c] + b[c]", "no illegal semantic expansion"],
        checks: [check("target", "warn", "broadcast output must preserve hidden shape")],
        x: 936,
        y: 218,
        w: 220,
        h: 126,
        color: 0x24608a
      })
    ],
    edges: [
      { id: "e_04_x_add", from: "activation_x", to: "broadcast_rule", label: "hidden[B,T,C]", color: 0x7dd3fc, flow: "forward" },
      { id: "e_04_pos_broadcast", from: "pos_emb", to: "broadcast_rule", label: "pos_emb[T,C]", color: 0xc4b5fd, flow: "parameter" },
      { id: "e_04_bias_broadcast", from: "bias_vector", to: "broadcast_rule", label: "bias[C]", color: 0xc4b5fd, flow: "parameter" },
      { id: "e_04_add_out", from: "broadcast_rule", to: "biased_out", label: "out[B,T,C]", color: 0x7dd3fc, flow: "forward" }
    ],
    repair: {
      kind: "broadcast_rail",
      targetContract: "hidden[B,T,C] + pos_emb[T,C] + bias[C] -> out[B,T,C]",
      brokenMessage: "Cannot align axes: pos_emb and bias are not mounted on the broadcast rail.",
      budget: { probes: 5, referenceRuns: 3 },
      tags: [
        { id: "align_tc", label: "Align [T,C]", shortLabel: "[T,C]", detail: "position embedding rail" },
        { id: "align_bt", label: "Align [B,T]", shortLabel: "[B,T]", detail: "wrong for pos_emb" },
        { id: "align_c", label: "Align [C]", shortLabel: "[C]", detail: "channel bias rail" },
        { id: "align_b", label: "Align [B]", shortLabel: "[B]", detail: "wrong for channel bias" },
        { id: "right_align", label: "Right-align rule", shortLabel: "right", detail: "broadcast from trailing dimensions" },
        { id: "left_align", label: "Left-align rule", shortLabel: "left", detail: "wrong broadcast rule" }
      ],
      slots: [
        {
          id: "pos_rail",
          label: "pos_emb Rail",
          nodeId: "pos_emb",
          focusNodeId: "pos_emb",
          emptyLabel: "?",
          correctTagIds: ["align_tc"],
          expected: "pos_emb aligns to [T,C]",
          successDetail: "Position embedding lines up with token position and channel axes.",
          failureDetail: "pos_emb must not expand on B or lose the C axis."
        },
        {
          id: "bias_rail",
          label: "bias Rail",
          nodeId: "bias_vector",
          focusNodeId: "bias_vector",
          emptyLabel: "?",
          correctTagIds: ["align_c"],
          expected: "bias aligns to [C]",
          successDetail: "Bias supplies one scalar per channel.",
          failureDetail: "Bias must align only to C; B/T alignment changes the semantics."
        },
        {
          id: "broadcast_rule_slot",
          label: "Rule",
          nodeId: "broadcast_rule",
          focusNodeId: "broadcast_rule",
          emptyLabel: "?",
          correctTagIds: ["right_align"],
          expected: "broadcast aligns trailing dimensions",
          successDetail: "Missing leading dims are treated as 1 and expanded over B/T.",
          failureDetail: "Broadcast must align from the right, not from the left or by element count."
        }
      ],
      probes: [
        {
          id: "rail_probe",
          label: "Rail Probe",
          detail: "shows how each small tensor sits under [B,T,C]",
          budgetCost: 1,
          observations: {
            pos_rail: {
              id: "pos_probe",
              title: "pos_emb owns token position and channel",
              detail: "It has one vector for each token position.",
              evidence: ["hidden:  [B][T][C]", "pos_emb:    [T][C]"],
              possibleSemantic: "[T,C]",
              confidence: "high"
            },
            bias_rail: {
              id: "bias_probe",
              title: "bias owns channel only",
              detail: "It is reused across batch and time.",
              evidence: ["hidden:  [B][T][C]", "bias:          [C]"],
              possibleSemantic: "[C]",
              confidence: "high"
            },
            broadcast_rule_slot: {
              id: "rule_probe",
              title: "Broadcast compares trailing dimensions",
              detail: "The small tensor is padded with leading 1s before expansion.",
              evidence: ["[C] -> [1,1,C]", "[T,C] -> [1,T,C]"],
              possibleSemantic: "right-align",
              confidence: "high"
            }
          }
        }
      ],
      checks: [
        {
          id: "pos_alignment",
          title: "pos_emb aligns to T/C",
          group: "visible",
          slotIds: ["pos_rail"],
          expected: "pos_emb[T,C] sits under hidden[T,C]",
          passDetail: "Position embedding follows token position and channel.",
          failDetail: "Position embedding is mounted on the wrong semantic rail.",
          focusNodeId: "pos_emb"
        },
        {
          id: "bias_alignment",
          title: "bias aligns to C",
          group: "visible",
          slotIds: ["bias_rail"],
          expected: "bias[C] sits under hidden[C]",
          passDetail: "Bias broadcasts across B and T.",
          failDetail: "Bias rail is semantically invalid.",
          focusNodeId: "bias_vector"
        },
        {
          id: "broadcast_rule",
          title: "broadcast rule is right-aligned",
          group: "visible",
          slotIds: ["broadcast_rule_slot"],
          expected: "trailing dimensions align",
          passDetail: "Broadcast uses trailing dimension alignment.",
          failDetail: "Broadcast rule is not the real tensor rule.",
          focusNodeId: "broadcast_rule"
        },
        {
          id: "broadcast_reference",
          title: "reference add allclose",
          group: "reference",
          slotIds: ["pos_rail", "bias_rail", "broadcast_rule_slot"],
          expected: "max_error <= 1e-5",
          passDetail: "Your broadcast sum matches reference.",
          failDetail: "Reference sum diverged.",
          blockedDetail: "Reference trace blocked until visible broadcast contract passes.",
          focusNodeId: "biased_out"
        },
        {
          id: "broadcast_hidden",
          title: "hidden broadcast shapes pass",
          group: "hidden",
          slotIds: ["pos_rail", "bias_rail", "broadcast_rule_slot"],
          expected: "[1,8,16] and [4,3,32] variants pass",
          passDetail: "Broadcast repair generalizes across batch, time, and channel sizes.",
          failDetail: "Hidden broadcast shapes failed.",
          blockedDetail: "Hidden tests are blocked until visible broadcast contract passes.",
          focusNodeId: "broadcast_rule"
        }
      ],
      hiddenCases: ["hidden[1,8,16] + pos[8,16] + bias[16]", "hidden[4,3,32] + pos[3,32] + bias[32]"],
      successSummary: "Broadcast Rail repaired: hidden[B,T,C] + pos_emb[T,C] + bias[C] -> out[B,T,C]."
    },
    traceSteps: [
      { id: "04_step_fault", title: "Fault", state: "warn", detail: "cannot align axes", selectNodeId: "broadcast_rule" },
      { id: "04_step_pos", title: "pos_emb", state: "warn", detail: "mount [T,C]", selectNodeId: "pos_emb" },
      { id: "04_step_bias", title: "bias", state: "warn", detail: "mount [C]", selectNodeId: "bias_vector" },
      { id: "04_step_out", title: "Output", state: "warn", detail: "[B,T,C]", selectNodeId: "biased_out" }
    ]
  }
];

export function evaluateBootcampLevel(level: BootcampLevel, assignments: BootcampAnswerMap, metrics: EvaluateMetrics): BootcampResult {
  const tagsById = new Map(level.repair.tags.map((tag) => [tag.id, tag]));
  const slotsById = new Map(level.repair.slots.map((slot) => [slot.id, slot]));

  const slotPasses = (slotIds: string[]) =>
    slotIds.every((slotId) => {
      const slot = slotsById.get(slotId);
      if (!slot) return true;
      return slot.correctTagIds.includes(assignments[slotId]);
    });

  const definitionPasses = (definition: BootcampLevel["repair"]["checks"][number]) => (definition.slotIds.length === 0 ? true : slotPasses(definition.slotIds));
  const visiblePassed = level.repair.checks.filter((item) => item.group === "visible").every(definitionPasses);
  const behaviorPassed = visiblePassed && level.repair.checks.filter((item) => item.group === "behavior").every(definitionPasses);

  const checks = level.repair.checks.map((definition) => {
    const blocked =
      (definition.group === "behavior" && !visiblePassed) ||
      ((definition.group === "hidden" || definition.group === "reference") && !behaviorPassed);
    const passed = definition.slotIds.length === 0 ? true : slotPasses(definition.slotIds);
    const state: CheckState = blocked ? "warn" : passed ? "pass" : "fail";
    const failDetail = explainRepairFailure(level, definition, assignments, slotsById);
    const received = definition.slotIds.length
      ? definition.slotIds
          .map((slotId) => {
            const slot = slotsById.get(slotId);
            const tag = tagsById.get(assignments[slotId]);
            return `${slot?.label ?? slotId}: ${tag?.shortLabel ?? "empty"}`;
          })
          .join("; ")
      : "rank observed: 3";

    return {
      id: definition.id,
      title: definition.title,
      group: definition.group,
      state,
      detail: blocked ? definition.blockedDetail ?? "Blocked until previous checks pass." : passed ? definition.passDetail : failDetail,
      expected: definition.expected,
      received,
      focusNodeId: definition.focusNodeId
    };
  });

  const failed = checks.find((item) => item.state === "fail");
  const passed = !failed && checks.every((item) => item.state === "pass");
  const probeEfficiency = Math.max(0, Math.round((1 - metrics.probeUses / Math.max(level.repair.budget.probes, 1)) * 100));
  const referenceOk = metrics.referenceRuns <= level.repair.budget.referenceRuns;
  const rank: "A" | "B" | "C" = passed && probeEfficiency >= 60 && referenceOk ? "A" : passed ? "B" : "C";

  return {
    passed,
    summary: passed ? level.repair.successSummary : failed ? `${failed.title}: ${failed.detail}` : "Visible and behavior checks must pass before hidden tests can run.",
    errorType: passed && level.id === "0-1" ? "Contract Restored" : passed ? "Contract Passed" : "Contract Failure",
    focusNodeId: failed?.focusNodeId ?? level.defaultSelectedNodeId,
    score: {
      rank,
      probeEfficiency: `${probeEfficiency}%`,
      probeUses: metrics.probeUses,
      probeBudget: level.repair.budget.probes,
      referenceRuns: metrics.referenceRuns,
      referenceBudget: level.repair.budget.referenceRuns
    },
    checks
  };
}

function explainRepairFailure(
  level: BootcampLevel,
  definition: BootcampLevel["repair"]["checks"][number],
  assignments: BootcampAnswerMap,
  slotsById: Map<string, BootcampLevel["repair"]["slots"][number]>
) {
  const firstFailedSlot = definition.slotIds.map((slotId) => slotsById.get(slotId)).find((slot) => slot && !slot.correctTagIds.includes(assignments[slot.id]));

  if (level.id !== "0-1") {
    return firstFailedSlot?.failureDetail ?? definition.failDetail;
  }

  if (
    definition.id === "tensor_objects" ||
    definition.id === "rank_scanner" ||
    definition.id === "shape_caliper" ||
    definition.id === "semantic_gap" ||
    definition.id === "token_grid_builder" ||
    definition.id === "embedding_expansion" ||
    definition.id === "consumer_validation"
  ) {
    return firstFailedSlot?.failureDetail ?? definition.failDetail;
  }

  const axisPattern = [assignments.axis_0, assignments.axis_1, assignments.axis_2].join(",");
  if (definition.id === "hidden_contract" || definition.id === "hidden_cases") {
    if (axisPattern === "tag_t,tag_b,tag_c") {
      return "B 和 T 标反：批次探针看到 Axis 0 在沿 token 顺序前进，而 Time Probe 看到 Axis 1 在切换样本。Causal Mask 会把样本当成时间轴。";
    }
    if (axisPattern === "tag_b,tag_c,tag_t") {
      return "T 和 C 标反：Axis 1 表现为连续特征条，不是 token 时间线。Attention 会失去正确的序列轴。";
    }
    if (axisPattern === "tag_c,tag_t,tag_b") {
      return "C 和 B 标反：Axis 0 是特征通道却被当成 batch，Linear 会把样本维度当作输入宽度。";
    }
    return firstFailedSlot?.failureDetail ?? definition.failDetail;
  }

  return firstFailedSlot?.failureDetail ?? definition.failDetail;
}
