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
    objective: "恢复数据流，观察 Hidden Tensor 的三个轴，标注 B/T/C，并把合同线接入 Axis Decoder。",
    sceneTitle: "Chapter 0-1 - Shape Reader",
    sceneSubtitle: "Broken board: data flow is open, hidden tensor semantics are missing, and the Axis Decoder contract is disconnected.",
    defaultSelectedNodeId: "hidden_tensor",
    briefing: [
      "LLM 不直接读取文字；文字先变成 token，再查表得到 hidden tensor。",
      "shape 只说明大小，[2,4,8] 本身不说明哪个轴是样本、位置或特征。",
      "hidden[B,T,C] 中，B 是样本批次，T 是 token 位置，C 是每个 token 的特征通道。"
    ],
    knowledgeCards: [
      {
        title: "LLM 不直接读取文字",
        body: "你输入的是文字，\n但模型内部处理的是数字。\n\n文字会先被切成 token，\n每个 token 再被转换成整数 ID。",
        visual: ["we train llm", "[we] [train] [llm]", "[we: 502] [train: 2841] [llm: 9172]", "token_ids = [502, 2841, 9172]"]
      },
      {
        title: "Tensor 是模型里的数字容器",
        body: "Tensor 可以理解为一组排列整齐的数字。\n\n一个数字是一种 tensor。\n一排数字是一种 tensor。\n一张数字表也是 tensor。\n一叠数字表仍然是 tensor。",
        visual: ["Scalar: 3.14", "Vector: [0.2, -0.7, 1.4, 0.5]", "Matrix: [[0.1,0.2,0.3],[0.4,0.5,0.6]]", "3D Tensor: 一叠矩阵"]
      },
      {
        title: "Rank 表示有几个轴",
        body: "一个数字没有轴，rank = 0。\n一排数字有 1 个轴，rank = 1。\n一张表有 2 个轴，rank = 2。\n一叠表有 3 个轴，rank = 3。",
        visual: ["rank 0: scalar / shape []", "rank 1: vector / shape [4]", "rank 2: matrix / shape [3,4]", "rank 3: tensor / shape [2,3,4]"]
      },
      {
        title: "Shape 表示每个轴的长度",
        body: "shape = [2,4,8]\n\n这表示：\n第 0 轴长度是 2，\n第 1 轴长度是 4，\n第 2 轴长度是 8。\n\nshape 说明 tensor 的结构大小。",
        visual: ["float32[2,4,8]", "Axis 0 length = 2", "Axis 1 length = 4", "Axis 2 length = 8"]
      },
      {
        title: "Shape 不等于语义",
        body: "[2,4,8] 只告诉你每个轴有多长，\n但没有告诉你每个轴代表什么。\n\n在模型里，轴的语义非常重要。\n下游模块必须知道哪个轴是 batch，\n哪个轴是 token position，\n哪个轴是 channel。",
        visual: ["float32[2,4,8]", "Axis 0: ?", "Axis 1: ?", "Axis 2: ?", "possible: [B,T,C] / [T,B,C] / [B,C,T]"]
      },
      {
        title: "Token IDs 通常是 [B,T]",
        body: "B 是 batch，表示一次送进模型的多条样本。\nT 是 token position，表示每条样本里的 token 位置。\n\n例如：\n2 条样本，每条 4 个 token，\ntoken_ids.shape = [2,4]\n也可以写成 token_ids[B,T]。",
        visual: ["          T0    T1    T2    T3", "B0       502  2841  9172     0", "B1      1042  7191  3910     0", "B = 2, T = 4"]
      },
      {
        title: "Embedding 把 token ID 变成向量",
        body: "token id 只是一个整数。\n\nEmbedding Lookup 会用 token id 查表，\n把每个 token 变成一条 C 维向量。\n\nC 是 channel，\n也可以理解为每个 token 的特征维度。",
        visual: ["token_id = 502", "embedding_table row 502", "embedding[502] = [0.12, -0.08, 0.31, 0.44, ...]", "C = channel / embedding dimension"]
      },
      {
        title: "Hidden Tensor 通常是 [B,T,C]",
        body: "当 B 条样本中的每个 token，\n都被转换成 C 维向量后，\n我们就得到了 hidden tensor。\n\nhidden[B,T,C] 表示：\n\nB：有几条样本\nT：每条样本有几个 token 位置\nC：每个 token 有多少个特征值",
        visual: ["B = 2, T = 4, C = 8", "hidden.shape = [2,4,8]", "hidden[0,2,:] = sample 0 / token 2 / full C vector", "hidden[0,2,5] = channel 5 value"]
      },
      {
        title: "为什么叫 Hidden Tensor？",
        body: "Hidden 指模型内部的中间表示。\n\n它不是原始文字，\n也不是最后输出的答案，\n而是模型在每一层中持续更新的工作状态。\n\nTransformer Block 会不断读取和改写 hidden tensor，\n但它通常仍然保持 [B,T,C] 的形状。",
        visual: ["token_ids[B,T]", "↓ Embedding", "hidden[B,T,C]", "↓ Transformer Block", "hidden[B,T,C]", "↓ LM Head", "logits[B,T,V]"]
      }
    ],
    knowledgeTransition: {
      title: "Briefing Complete",
      body: "你已经知道：\n\nTensor 是模型内部流动的数字结构。\nShape 描述 tensor 的轴长度。\nHidden Tensor 是 Transformer 中常见的中间表示。\n\n现在，一台训练板上的 Hidden Tensor 丢失了轴标签。\n进入 Workbench，恢复它的 shape contract。\n\n目标合同：\nhidden[B,T,C]",
      buttonLabel: "Enter Tensor Workbench"
    },
    mission: {
      title: "Repair Mission: Hidden Tensor Contract",
      body: "当前模型板已经生成了一个 Hidden Tensor：\n\nfloat32[2,4,8]\n\n但它的三个轴标签丢失了。\n\n你需要通过探针观察它的结构，判断哪个轴是 B、哪个轴是 T、哪个轴是 C，并将它修复为：\n\nhidden[B,T,C]",
      success: [
        "连接 Text Batch -> Tokenizer -> Embedding Lookup -> Hidden Tensor。",
        "使用 Probe 观察 Hidden Tensor 的三个轴。",
        "将 B / T / C 标签拖到正确轴槽。",
        "将 Hidden Tensor 连接到 Axis Decoder。",
        "运行 Shape Tests，并通过 hidden tests。"
      ]
    },
    unlocks: ["Shape Inspector", "Axis Tags", "Tensor Probe", "Axis Decoder"],
    contracts: [
      "target: hidden[B,T,C]",
      "visible: hidden[2,4,8]",
      "behavior: B separates samples / T preserves order / C feeds Linear",
      "hidden: [1,8,16], [4,3,32], [2,12,6]"
    ],
    nodes: [
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
        consumer: "Axis Decoder / Linear / Attention",
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
        id: "axis_decoder",
        title: "Axis Decoder",
        subtitle: "contract ports open",
        kind: "operation",
        semanticName: "axis_decoder",
        dtype: "rule",
        shape: "requires [B,T,C]",
        source: "player axis tags",
        consumer: "shape tests",
        sample: ["blocked: hidden axis semantics unknown", "required: hidden[B,T,C]"],
        checks: [check("repairable", "warn", "run probes, assign tags, then test")],
        x: 870,
        y: 164,
        w: 188,
        h: 94,
        color: 0x304b6a
      }),
      makeNode({
        id: "shape_tests",
        title: "Shape Tests",
        subtitle: "autograder",
        kind: "scalar",
        semanticName: "axis_tests",
        dtype: "bool",
        shape: "visible + hidden",
        source: "axis_decoder",
        consumer: "chapter unlock",
        sample: ["visible: hidden[2,4,8]", "hidden: [1,8,16], [4,3,32]"],
        checks: [check("ready", "warn", "contract tests are blocked until slots are filled")],
        x: 870,
        y: 330,
        w: 156,
        h: 76,
        color: 0x74491a
      })
    ],
    edges: [
      { id: "e_01_text_tokenizer", from: "text_batch", to: "tokenizer", label: "repair data line", color: 0x7dd3fc, flow: "forward" },
      { id: "e_01_tokenizer_embedding", from: "tokenizer", to: "embedding_lookup", label: "repair token line", color: 0x7dd3fc, flow: "forward" },
      { id: "e_01_embedding_hidden", from: "embedding_lookup", to: "hidden_tensor", label: "repair hidden line", color: 0x7dd3fc, flow: "forward" },
      { id: "e_01_hidden_axes", from: "hidden_tensor", to: "axis_decoder", label: "contract ports", color: 0xfbbf24, flow: "check" },
      { id: "e_01_axes_tests", from: "axis_decoder", to: "shape_tests", label: "visible / behavior / hidden", color: 0xfbbf24, flow: "check", route: "down" }
    ],
    repair: {
      kind: "axis_labels",
      targetContract: "hidden[B,T,C]",
      brokenMessage: "Shape contract incomplete: data flow is disconnected, hidden tensor axes are unlabeled, and Axis Decoder cannot verify B/T/C.",
      budget: { probes: 5, referenceRuns: 3 },
      tags: [
        { id: "wire_text_tokenizer", label: "Text -> Tokenizer", shortLabel: "utf8", detail: "connect raw examples into the tokenizer", category: "data" },
        { id: "wire_tokenizer_embedding", label: "Tokenizer -> Embedding", shortLabel: "ids", detail: "connect token ids into embedding lookup", category: "data" },
        { id: "wire_embedding_hidden", label: "Embedding -> Hidden", shortLabel: "vec", detail: "connect embedding vectors into hidden tensor", category: "data" },
        { id: "tag_b", label: "[B] Batch", shortLabel: "B", detail: "independent samples", category: "axis" },
        { id: "tag_t", label: "[T] Token Position", shortLabel: "T", detail: "ordered token positions", category: "axis" },
        { id: "tag_c", label: "[C] Channel", shortLabel: "C", detail: "per-token feature channels", category: "axis" },
        { id: "contract_b", label: "Decoder Port B", shortLabel: "B ->", detail: "wire batch semantic into Axis Decoder port B", category: "contract" },
        { id: "contract_t", label: "Decoder Port T", shortLabel: "T ->", detail: "wire token-position semantic into Axis Decoder port T", category: "contract" },
        { id: "contract_c", label: "Decoder Port C", shortLabel: "C ->", detail: "wire channel semantic into Axis Decoder port C", category: "contract" }
      ],
      slots: [
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
          id: "flow_tokenizer_embedding",
          label: "Tokenizer -> Embedding",
          nodeId: "embedding_lookup",
          focusNodeId: "embedding_lookup",
          emptyLabel: "open",
          correctTagIds: ["wire_tokenizer_embedding"],
          expected: "Tokenizer token ids feed Embedding Lookup",
          successDetail: "Token ids now index the embedding table.",
          failureDetail: "Embedding Lookup requires integer token ids, not raw text or hidden vectors."
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
          id: "contract_b",
          label: "Decoder B",
          nodeId: "axis_decoder",
          focusNodeId: "axis_decoder",
          emptyLabel: "open",
          correctTagIds: ["contract_b"],
          expected: "Axis Decoder port B receives batch semantic",
          successDetail: "Axis Decoder B port is wired.",
          failureDetail: "Axis Decoder B port is still missing the batch semantic."
        },
        {
          id: "contract_t",
          label: "Decoder T",
          nodeId: "axis_decoder",
          focusNodeId: "axis_decoder",
          emptyLabel: "open",
          correctTagIds: ["contract_t"],
          expected: "Axis Decoder port T receives token-position semantic",
          successDetail: "Axis Decoder T port is wired.",
          failureDetail: "Axis Decoder T port is still missing the token-position semantic."
        },
        {
          id: "contract_c",
          label: "Decoder C",
          nodeId: "axis_decoder",
          focusNodeId: "axis_decoder",
          emptyLabel: "open",
          correctTagIds: ["contract_c"],
          expected: "Axis Decoder port C receives channel semantic",
          successDetail: "Axis Decoder C port is wired.",
          failureDetail: "Axis Decoder C port is still missing the channel semantic."
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
          id: "data_flow_connected",
          title: "data flow is restored",
          group: "visible",
          slotIds: ["flow_text_tokenizer", "flow_tokenizer_embedding", "flow_embedding_hidden"],
          expected: "Text Batch -> Tokenizer -> Embedding Lookup -> Hidden Tensor",
          passDetail: "The forward data path can now generate hidden activations.",
          failDetail: "Data flow repair is incomplete; the hidden tensor cannot be trusted yet.",
          focusNodeId: "hidden_tensor"
        },
        {
          id: "rank_check",
          title: "hidden rank is 3",
          group: "visible",
          slotIds: [],
          expected: "rank(hidden) == 3",
          passDetail: "Visible tensor has three axes.",
          failDetail: "Hidden tensor rank mismatch.",
          focusNodeId: "hidden_tensor"
        },
        {
          id: "axis_contract",
          title: "axis labels match [B,T,C]",
          group: "visible",
          slotIds: ["axis_0", "axis_1", "axis_2"],
          expected: "axis labels == [B,T,C]",
          passDetail: "All axis labels satisfy hidden[B,T,C].",
          failDetail: "Axis semantics do not satisfy hidden[B,T,C].",
          focusNodeId: "axis_decoder"
        },
        {
          id: "decoder_contract",
          title: "Axis Decoder ports are connected",
          group: "visible",
          slotIds: ["contract_b", "contract_t", "contract_c"],
          expected: "Decoder receives B, T, and C semantic lines",
          passDetail: "Axis Decoder can read the repaired shape contract.",
          failDetail: "Axis Decoder still has at least one open contract port.",
          focusNodeId: "axis_decoder"
        },
        {
          id: "behavior_b",
          title: "B separates independent samples",
          group: "behavior",
          slotIds: ["axis_0", "contract_b"],
          expected: "axis 0 changes sample identity",
          passDetail: "Changing B switches to a different text sample.",
          failDetail: "Batch behavior failed: the marked B axis does not separate samples.",
          blockedDetail: "Behavior checks are blocked until visible data and contract repairs pass.",
          focusNodeId: "hidden_tensor"
        },
        {
          id: "behavior_t",
          title: "T preserves token order",
          group: "behavior",
          slotIds: ["axis_1", "contract_t"],
          expected: "axis 1 walks token position 0 -> 1 -> 2",
          passDetail: "Changing T moves through token positions in order.",
          failDetail: "Time behavior failed: the marked T axis does not preserve token order.",
          blockedDetail: "Behavior checks are blocked until visible data and contract repairs pass.",
          focusNodeId: "hidden_tensor"
        },
        {
          id: "behavior_c",
          title: "C exposes feature channels",
          group: "behavior",
          slotIds: ["axis_2", "contract_c"],
          expected: "axis 2 is a dense feature vector consumed by Linear",
          passDetail: "Changing C selects feature channels for the current token.",
          failDetail: "Channel behavior failed: the marked C axis is not a feature channel.",
          blockedDetail: "Behavior checks are blocked until visible data and contract repairs pass.",
          focusNodeId: "hidden_tensor"
        },
        {
          id: "hidden_cases",
          title: "hidden shape generalizes",
          group: "hidden",
          slotIds: [
            "flow_text_tokenizer",
            "flow_tokenizer_embedding",
            "flow_embedding_hidden",
            "axis_0",
            "axis_1",
            "axis_2",
            "contract_b",
            "contract_t",
            "contract_c"
          ],
          expected: "hidden[1,8,16], hidden[4,3,32], hidden[2,12,6]",
          passDetail: "Axis semantics generalize across hidden shapes.",
          failDetail: "Hidden tests failed because the contract is semantic, not numeric.",
          blockedDetail: "Hidden tests are blocked until visible and behavior checks pass.",
          focusNodeId: "shape_tests"
        }
      ],
      hiddenCases: ["hidden[1,8,16]", "hidden[4,3,32]", "hidden[2,12,6]"],
      successSummary: "Contract restored: Hidden Tensor is now float32[B,T,C], data flow is live, and Axis Decoder accepts the shape contract."
    },
    traceSteps: [
      { id: "01_step_flow", title: "Data Flow", state: "warn", detail: "connect the open forward lines", selectNodeId: "tokenizer" },
      { id: "01_step_probe", title: "Probe", state: "warn", detail: "inspect axis patterns", selectNodeId: "hidden_tensor" },
      { id: "01_step_axis", title: "Axis Tags", state: "warn", detail: "assign B/T/C to Axis 0/1/2", selectNodeId: "hidden_tensor" },
      { id: "01_step_contract", title: "Contract", state: "warn", detail: "wire B/T/C into Axis Decoder", selectNodeId: "axis_decoder" },
      { id: "01_step_tests", title: "Tests", state: "warn", detail: "visible + behavior + hidden", selectNodeId: "shape_tests" }
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

  if (definition.id === "data_flow_connected") {
    return firstFailedSlot?.failureDetail ?? "Forward data flow still has an open port.";
  }

  if (definition.id === "decoder_contract") {
    return firstFailedSlot?.failureDetail ?? "Axis Decoder still has an open B/T/C contract port.";
  }

  const axisPattern = [assignments.axis_0, assignments.axis_1, assignments.axis_2].join(",");
  if (definition.id === "axis_contract" || definition.id.startsWith("behavior_")) {
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
