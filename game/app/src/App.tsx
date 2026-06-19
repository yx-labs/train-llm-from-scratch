import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type DragEvent as ReactDragEvent } from "react";
import { AlertTriangle, BookOpen, Boxes, CheckCircle2, Circle, Cpu, Eye, Maximize2, Minimize2, Pause, Play, RotateCcw, Search, Wrench } from "lucide-react";
import { bootcampLevels, evaluateBootcampLevel } from "./bootcampLevels";
import {
  PixiWorkbenchCanvas,
  type CanvasActionHint,
  type CanvasConnectionOverlay,
  type CanvasContextTarget,
  type CanvasStageKnowledge,
  type RepairSlotOverlay
} from "./PixiWorkbenchCanvas";
import { modeLabels } from "./sceneData";
import type {
  BootcampAnswerMap,
  BootcampLevel,
  BootcampResult,
  CheckState,
  ProbeObservation,
  RepairSlot,
  RepairTag,
  TensorNode,
  WorkbenchMode
} from "./workbenchTypes";

type ObservationLogItem = ProbeObservation & {
  probeId: string;
  probeLabel: string;
  slotId: string;
  slotLabel: string;
};

type LevelRepairState = {
  assignments: BootcampAnswerMap;
  activeTagId?: string;
  activeProbeId?: string;
  selectedSlotId?: string;
  probeUses: number;
  referenceRuns: number;
  observations: ObservationLogItem[];
};

type NodePositionMap = Record<string, { x: number; y: number }>;
type StageKnowledgePositionMap = Record<string, { x: number; y: number }>;

type CanvasMenuAction = {
  id: string;
  label: string;
  detail: string;
  disabled?: boolean;
  onSelect: () => void;
};

type InspectorTaskItem = {
  id: string;
  label: string;
  value: string;
  state: "pass" | "fail" | "warn";
  active: boolean;
};

type TensorObjectShowcase = {
  id: string;
  kind: "scalar" | "vector" | "matrix" | "block";
  title: string;
  shape: string;
  dtype: string;
  code: string;
  value: TensorObjectValue;
};

type TensorObjectValue = number | TensorObjectValue[];

type TensorObjectDetailState = {
  slotId: string;
  modalOpen: boolean;
};

type LevelPhase =
  | "knowledge_intro"
  | "mission_modal"
  | "tensor_object"
  | "rank_scanner"
  | "shape_caliper"
  | "semantic_gap"
  | "token_grid_builder"
  | "embedding_expansion"
  | "hidden_contract_repair"
  | "consumer_validation"
  | "hidden_test_gauntlet"
  | "matmul_dot_cell"
  | "matmul_token_projection"
  | "matmul_sequence_projection"
  | "matmul_batch_projection"
  | "matmul_orientation_trap"
  | "matmul_linear_assembly"
  | "matmul_gauntlet"
  | "data_flow_repair"
  | "tensor_generated"
  | "axis_probe"
  | "axis_tagging"
  | "contract_wiring"
  | "visible_testing"
  | "hidden_testing"
  | "completed";

const repairTagDragMime = "application/x-llm-complete-repair-tag";

const modeIcons: Record<WorkbenchMode, JSX.Element> = {
  build: <Wrench size={17} />,
  trace: <Search size={17} />,
  train: <Cpu size={17} />
};

type Chapter01ChallengeDef = {
  phase: LevelPhase;
  code: string;
  title: string;
  brief: string;
  slotIds: string[];
  nodeIds: string[];
  edgeIds: string[];
  tagIds?: string[];
  tagCategories: NonNullable<RepairTag["category"]>[];
};

type Chapter01StageKnowledge = CanvasStageKnowledge & {
  inspectorNotes: string[];
  failureLesson: string[];
  debrief: string;
};

type Chapter01StageIntro = {
  code: string;
  title: string;
  body: string;
  visualLines: string[];
  taskPrompt: string;
};

const chapter01Challenges: Chapter01ChallengeDef[] = [
  {
    phase: "tensor_object",
    code: "0-1A",
    title: "Tensor Object",
    brief: "把 scalar / vector / matrix / 3D block 送进 Tensor Inspector。",
    slotIds: ["object_scalar", "object_vector", "object_matrix", "object_block"],
    nodeIds: ["tensor_inspector", "type_check"],
    edgeIds: ["e_01_inspector_type"],
    tagCategories: ["object"]
  },
  {
    phase: "rank_scanner",
    code: "0-1B",
    title: "Rank Scanner",
    brief: "给四种对象盖上 rank 0 / 1 / 2 / 3 的轴数量印章。",
    slotIds: ["rank_scalar", "rank_vector", "rank_matrix", "rank_block"],
    nodeIds: ["raw_objects", "rank_scanner", "rank_gate"],
    edgeIds: ["e_01_objects_rank", "e_01_rank_gate"],
    tagCategories: ["rank"]
  },
  {
    phase: "shape_caliper",
    code: "0-1C",
    title: "Shape Caliper",
    brief: "读取 hidden 的三个轴长度，按顺序修成 [2,4,8]。",
    slotIds: ["shape_axis_0", "shape_axis_1", "shape_axis_2"],
    nodeIds: ["hidden_tensor", "shape_caliper", "shape_gate"],
    edgeIds: ["e_01_hidden_caliper", "e_01_caliper_shape"],
    tagCategories: ["shape"]
  },
  {
    phase: "semantic_gap",
    code: "0-1D",
    title: "Shape != Semantics",
    brief: "确认 [2,4,8] 只是长度，B/T/C 语义仍然未知。",
    slotIds: ["semantic_unresolved"],
    nodeIds: ["shape_gate", "semantic_inspector"],
    edgeIds: ["e_01_shape_semantic"],
    tagCategories: ["semantic"]
  },
  {
    phase: "token_grid_builder",
    code: "0-1E",
    title: "Token Grid Builder",
    brief: "修复 token_ids[B,T]，并完成 sample row / token column 微挑战。",
    slotIds: ["flow_text_tokenizer", "flow_tokenizer_grid", "token_grid_b", "token_grid_t", "token_task_sample1", "token_task_t2"],
    nodeIds: ["text_batch", "tokenizer", "token_grid"],
    edgeIds: ["e_01_text_tokenizer", "e_01_tokenizer_grid"],
    tagCategories: ["data", "token"]
  },
  {
    phase: "embedding_expansion",
    code: "0-1F",
    title: "Embedding Expansion",
    brief: "用 token_ids[B,T] 查表，把每个 token 展开成 C 维向量。",
    slotIds: ["flow_grid_embedding", "flow_table_embedding", "embedding_probe", "embedding_autofill", "flow_embedding_hidden"],
    nodeIds: ["token_grid", "embedding_table", "embedding_lookup", "hidden_tensor"],
    edgeIds: ["e_01_grid_embedding", "e_01_table_embedding", "e_01_embedding_hidden"],
    tagCategories: ["data", "embedding"]
  },
  {
    phase: "hidden_contract_repair",
    code: "0-1G",
    title: "Hidden Contract Repair",
    brief: "用 Probe 观察三个轴，把 hidden 修复为 [B,T,C]。",
    slotIds: ["axis_0", "axis_1", "axis_2"],
    nodeIds: ["hidden_tensor"],
    edgeIds: [],
    tagCategories: ["axis"]
  },
  {
    phase: "consumer_validation",
    code: "0-1H",
    title: "Consumer Validation",
    brief: "把 B/T/C 分别接给 Batch Viewer、Causal Mask、Linear Probe。",
    slotIds: ["consumer_b", "consumer_t", "consumer_c"],
    nodeIds: ["hidden_tensor", "batch_viewer", "causal_mask", "linear_probe", "shape_tests"],
    edgeIds: ["e_01_hidden_batch", "e_01_hidden_mask", "e_01_hidden_linear", "e_01_consumers_tests"],
    tagCategories: ["consumer"]
  },
  {
    phase: "hidden_test_gauntlet",
    code: "0-1X",
    title: "Hidden Test Gauntlet",
    brief: "运行隐藏 shape 变体，验证修复不是只记住 [2,4,8]。",
    slotIds: [],
    nodeIds: ["hidden_tensor", "batch_viewer", "causal_mask", "linear_probe", "shape_tests"],
    edgeIds: ["e_01_hidden_batch", "e_01_hidden_mask", "e_01_hidden_linear", "e_01_consumers_tests"],
    tagCategories: []
  }
];

const chapter01ChallengeByPhase = new Map(chapter01Challenges.map((challenge) => [challenge.phase, challenge]));

const chapter02Challenges: Chapter01ChallengeDef[] = [
  {
    phase: "matmul_dot_cell",
    code: "0-2A",
    title: "Dot Cell",
    brief: "确认一个输出值来自一次 [C] dot [C] 点积，结果是 scalar。",
    slotIds: ["dot_left_vector", "dot_weight_vector", "dot_output_scalar"],
    nodeIds: ["feature_vector", "weight_vector", "dot_cell", "dot_scalar"],
    edgeIds: ["e_02_feature_dot", "e_02_weight_dot", "e_02_dot_scalar"],
    tagIds: ["vec_c", "vec_o", "dot_weight_c", "dot_weight_o", "out_scalar", "out_vector"],
    tagCategories: ["axis", "contract"]
  },
  {
    phase: "matmul_token_projection",
    code: "0-2B",
    title: "Token Projection",
    brief: "把单个 token 的 [C] 向量投影成 [O]。",
    slotIds: ["token_input_c", "token_weight_co", "token_output_o"],
    nodeIds: ["token_vector", "weight_plate", "matmul_gate", "token_output"],
    edgeIds: ["e_02_token_gate", "e_02_weight_gate", "e_02_gate_token_out"],
    tagIds: ["token_c", "token_t", "weight_co", "weight_oc", "axis_o", "axis_c"],
    tagCategories: ["axis", "contract"]
  },
  {
    phase: "matmul_sequence_projection",
    code: "0-2C",
    title: "Sequence Projection",
    brief: "确认 T 轴是携带轴，同一块 W 被复用，输出 [T,O]。",
    slotIds: ["sequence_carrier_t", "sequence_weight_shared", "sequence_output_to"],
    nodeIds: ["sequence_tensor", "weight_plate", "matmul_gate", "sequence_output"],
    edgeIds: ["e_02_sequence_gate", "e_02_weight_gate", "e_02_gate_sequence_out"],
    tagIds: ["preserve_t", "consume_t", "shared_weight", "shape_to", "shape_tc"],
    tagCategories: ["axis", "operation", "contract"]
  },
  {
    phase: "matmul_batch_projection",
    code: "0-2D",
    title: "Batch Projection",
    brief: "扩展到 hidden[B,T,C]，B/T 并行携带，C 被投影为 O。",
    slotIds: ["batch_carrier_bt", "batch_weight_shared", "batch_output_bto"],
    nodeIds: ["linear_input", "weight_plate", "matmul_gate", "linear_out"],
    edgeIds: ["e_02_batch_gate", "e_02_weight_gate", "e_02_gate_linear_out"],
    tagIds: ["preserve_bt", "consume_b", "shared_weight", "shape_bto", "shape_tbo"],
    tagCategories: ["axis", "operation", "contract"]
  },
  {
    phase: "matmul_orientation_trap",
    code: "0-2E",
    title: "Weight Orientation Trap",
    brief: "区分存储方向 [O,C] 和计算方向 [C,O]，插入转置或旋转权重板。",
    slotIds: ["stored_weight_orientation", "orientation_fix", "compute_weight_co"],
    nodeIds: ["linear_input", "weight_plate", "transpose_switch", "matmul_gate", "linear_out"],
    edgeIds: ["e_02_batch_gate", "e_02_weight_transpose", "e_02_transpose_gate", "e_02_gate_linear_out"],
    tagIds: ["stored_oc", "stored_co", "transpose_weight", "rotate_to_co", "compute_co", "compute_oc"],
    tagCategories: ["operation", "contract"]
  },
  {
    phase: "matmul_linear_assembly",
    code: "0-2F",
    title: "Linear Module Assembly",
    brief: "把 MatMul Gate 封装为 Linear Projection，锁定 bias，并通过 reference allclose。",
    slotIds: ["linear_core", "bias_locked", "reference_checker"],
    nodeIds: ["linear_input", "weight_plate", "transpose_switch", "matmul_gate", "linear_module", "linear_out", "reference_checker"],
    edgeIds: ["e_02_batch_gate", "e_02_weight_transpose", "e_02_transpose_gate", "e_02_gate_module", "e_02_module_out", "e_02_out_reference"],
    tagIds: ["matmul_core", "linear_bias_locked", "reference_allclose", "compute_co", "transpose_weight"],
    tagCategories: ["operation", "contract"]
  },
  {
    phase: "matmul_gauntlet",
    code: "0-2X",
    title: "MatMul Gauntlet",
    brief: "运行标准方向、存储方向和尺寸陷阱隐藏测试，证明合同可泛化。",
    slotIds: ["gauntlet_standard", "gauntlet_storage", "gauntlet_size"],
    nodeIds: ["linear_input", "weight_plate", "transpose_switch", "matmul_gate", "linear_out", "reference_checker", "matmul_tests"],
    edgeIds: ["e_02_batch_gate", "e_02_weight_transpose", "e_02_transpose_gate", "e_02_gate_linear_out", "e_02_out_reference", "e_02_reference_tests"],
    tagIds: ["gauntlet_case_a", "gauntlet_case_b", "gauntlet_no_size_guess", "reference_allclose"],
    tagCategories: ["contract"]
  }
];

const chapter02ChallengeByPhase = new Map(chapter02Challenges.map((challenge) => [challenge.phase, challenge]));

const chapter01StageIntros: Partial<Record<LevelPhase, Chapter01StageIntro>> = {
  tensor_object: {
    code: "0-1A",
    title: "Tensor：模型里的数字对象",
    body: "Tensor 可以理解为一组排列好的数字。\n\n它可以是：\n一个数字，\n一排数字，\n一张数字表，\n或者一叠数字表。\n\n在模型内部，几乎所有数据都会以 tensor 的形式流动。",
    visualLines: ["3.14", "-> [0.2, -0.7, 1.4]", "-> 2D number grid", "-> stacked number grids"],
    taskPrompt: "把不同的数据对象拖入 Tensor Inspector，\n确认它们都可以被系统识别为 tensor。"
  },
  rank_scanner: {
    code: "0-1B",
    title: "Rank：tensor 有几个轴",
    body: "Rank 表示 tensor 有几个可以索引的方向。\n\n一个数字没有轴，rank = 0。\n一排数字有 1 个轴，rank = 1。\n一张表有 2 个轴，rank = 2。\n一叠表有 3 个轴，rank = 3。",
    visualLines: ["rank 0: 点", "rank 1: 线", "rank 2: 面", "rank 3: 体"],
    taskPrompt: "使用 Rank Scanner 扫描不同 tensor，\n给它们贴上正确的 rank 标记。"
  },
  shape_caliper: {
    code: "0-1C",
    title: "Shape：tensor 的结构尺寸",
    body: "Shape 告诉我们每个轴有多长。\n\n例如：\n\nshape = [2,4,8]\n\n表示这个 tensor 有 3 个轴：\nAxis 0 长度是 2，\nAxis 1 长度是 4，\nAxis 2 长度是 8。",
    visualLines: ["Axis 0 length = 2", "Axis 1 length = 4", "Axis 2 length = 8", "shape = [2,4,8]"],
    taskPrompt: "使用 Shape Caliper 测量三条轴，\n拼出这个 tensor 的 shape。"
  },
  semantic_gap: {
    code: "0-1D",
    title: "Shape 不等于语义",
    body: "float32[2,4,8] 只告诉你这个 tensor 的大小。\n\n它没有告诉你：\n哪个轴是样本？\n哪个轴是 token 位置？\n哪个轴是特征通道？\n\n在模型里，下游模块需要知道轴的意义。\n这叫 axis semantics。",
    visualLines: ["float32[2,4,8]", "Axis 0: ?", "Axis 1: ?", "Axis 2: ?", "Batch Viewer / Causal Mask / Linear need semantics"],
    taskPrompt: "尝试把只有 shape、没有轴语义的 tensor 连接到下游模块，\n观察为什么它会失败。"
  },
  token_grid_builder: {
    code: "0-1E",
    title: "Token IDs：从文字到二维表格",
    body: "一句文字会先被切成 token，\n每个 token 会变成一个整数 ID。\n\n当多条文本一起送进模型时，\ntoken ids 会组成一个二维 tensor：\n\ntoken_ids[B,T]\n\nB 表示 batch 中的样本。\nT 表示每条样本里的 token 位置。",
    visualLines: ['sample 0: "we train llm"', 'sample 1: "shape tells truth"', "↓", "token_ids[B,T]", "B0: 502 2841 9172 0", "B1: 1042 7191 3910 0"],
    taskPrompt: "连接 Text Batch -> Tokenizer -> Token Grid，\n生成 token_ids[B,T]。\n然后给二维表格标出 B 轴和 T 轴。"
  },
  embedding_expansion: {
    code: "0-1F",
    title: "Embedding：给每个 token 增加特征通道",
    body: "token id 只是一个整数。\n\nEmbedding Lookup 会用 token id 去查表，\n把每个 token 转换成一条向量。\n\n这条向量的长度叫 C，\n也就是 channel / embedding dimension。",
    visualLines: ["token_id = 9172", "↓ lookup", "embedding[9172] = [0.12, -0.08, 0.31, 0.44, ...]", "token_ids[B,T] -> hidden[B,T,C]"],
    taskPrompt: "连接 Token Grid 和 Embedding Lookup，\n观察每个 token 如何扩展成 C 维 feature vector。"
  },
  hidden_contract_repair: {
    code: "0-1G",
    title: "Hidden Tensor：模型内部的工作状态",
    body: "Hidden Tensor 不是最终输出，\n也不是原始 token id。\n\n它是模型内部正在处理的中间表示。\n\n在 Transformer 中，常见的 hidden tensor 是：\n\nhidden[B,T,C]\n\n每个 token 在每一层里，\n都会带着一条 C 维向量继续向前流动。",
    visualLines: ["token_ids[B,T]", "↓ Embedding", "hidden[B,T,C]", "↓ Transformer Block", "hidden[B,T,C]", "↓ LM Head", "logits[B,T,V]"],
    taskPrompt: "使用 Probe 观察 Hidden Tensor 的三个轴，\n把 B / T / C 标签贴到正确位置。"
  },
  consumer_validation: {
    code: "0-1H",
    title: "Axis Semantics 是模块合同",
    body: "下游模块读取 tensor 时，\n不只关心 shape 的数字大小，\n还关心每个轴代表什么。\n\nBatch Viewer 需要 B。\nCausal Mask 需要 T。\nLinear Projection 需要 C。\n\n如果轴语义错误，模块可能仍然收到数据，\n但它会在错误的方向上计算。",
    visualLines: ["hidden[B,T,C]", "B -> Batch Viewer", "T -> Causal Mask", "C -> Linear Projection"],
    taskPrompt: "把 Hidden Tensor 的 B/T/C 语义端口\n连接到正确的下游模块。"
  },
  hidden_test_gauntlet: {
    code: "0-1X",
    title: "B/T/C 不是固定数字",
    body: "在一个 tiny model 里，\nC 可能只有 4。\n\n在一个长上下文样本里，\nT 可能是 16、128、甚至更长。\n\n所以不能靠“哪个数字最大”判断 C，\n也不能靠固定位置死记 B/T/C。\n\n你需要观察：\n哪个轴分离样本，\n哪个轴保持 token 顺序，\n哪个轴存储 feature values。",
    visualLines: ["hidden[2,4,8]", "hidden[1,16,4]", "hidden[4,3,32]", "same semantics, changing lengths"],
    taskPrompt: "连续修复多个未知 hidden tensor。\n使用 Probe 判断轴语义，\n通过所有隐藏测试。"
  }
};

const chapter02StageIntros: Partial<Record<LevelPhase, Chapter01StageIntro>> = {
  matmul_dot_cell: {
    code: "0-2A",
    title: "Dot Cell：一个输出值来自一次点积",
    body: "Linear Projection 最小的计算单元是一格输出。\n\n这一格不是凭空生成的：它来自输入 C 维 feature vector 和一条 C 维权重向量的点积。\n\n两个向量长度必须相同，结果是一个 scalar。",
    visualLines: ["x[C]", "w_o[C]", "sum_i x[i] * w_o[i]", "scalar"],
    taskPrompt: "标出左侧 C 向量、右侧 C 权重向量，确认 Dot Cell 的输出是一个 scalar。"
  },
  matmul_token_projection: {
    code: "0-2B",
    title: "Token Projection：单 token 投影",
    body: "一个 token 的 hidden state 是一条 C 维向量。\n\n如果要生成 O 个输出值，就需要 O 条权重向量。\n\n把这些权重向量排成权重板，就是 compute view [C,O]。",
    visualLines: ["token[C]", "@ W[C,O]", "O dot cells", "token_out[O]"],
    taskPrompt: "把单 token 输入、权重板方向和输出 O 轴修成 [C] @ [C,O] -> [O]。"
  },
  matmul_sequence_projection: {
    code: "0-2C",
    title: "Sequence Projection：T 轴不参与点积",
    body: "一句话是一排 token vector。\n\nMatMul 对每个 token 的 C 维向量做同样的投影；T 轴只负责携带 token 顺序。\n\n同一块权重板会在所有 T 位置复用。",
    visualLines: ["x[T,C]", "same W[C,O]", "for each T row", "y[T,O]"],
    taskPrompt: "确认 T 被保留、W 被复用，并把输出合同修为 [T,O]。"
  },
  matmul_batch_projection: {
    code: "0-2D",
    title: "Batch Projection：B/T 是并行携带轴",
    body: "Batch 只是多条样本并行。\n\nLinear Projection 不应该把 B 或 T 放进点积里；它只消费最后一维 C。\n\n完整公式是 hidden[B,T,C] @ W[C,O] -> out[B,T,O]。",
    visualLines: ["hidden[B,T,C]", "consume C", "carry B/T", "out[B,T,O]"],
    taskPrompt: "标出 B/T 携带、共享权重和完整输出 [B,T,O]。"
  },
  matmul_orientation_trap: {
    code: "0-2E",
    title: "Weight Orientation Trap：存储方向不是计算方向",
    body: "有些框架会把 Linear weight 存成 [O,C]。\n\n但本关的 MatMul Gate 需要 compute view [C,O]。\n\n转置修复的是轴顺序，不是数值魔法。",
    visualLines: ["stored W[O,C]", "Transpose Switch", "compute W[C,O]", "hidden[B,T,C] @ W[C,O]"],
    taskPrompt: "识别 stored [O,C]，插入转置或旋转权重板，再让 gate 收到 [C,O]。"
  },
  matmul_linear_assembly: {
    code: "0-2F",
    title: "Linear Module Assembly：组装可复用模块",
    body: "Linear 是 Transformer 里的常用构件。\n\n在 0-2，我们先锁定 bias，只验证核心投影 x @ W。\n\nShape 正确不等于数值正确，所以最终还要通过 reference allclose。",
    visualLines: ["Linear(x)", "MatMul core", "bias locked", "reference allclose"],
    taskPrompt: "把 MatMul core、bias 策略和 reference checker 组装成 Linear Projection。"
  },
  matmul_gauntlet: {
    code: "0-2X",
    title: "MatMul Gauntlet：合同必须能泛化",
    body: "最后测试不会只检查一个可见 shape。\n\n它会覆盖标准方向、存储方向反转，以及防止靠最大维度猜 C/O 的隐藏 case。\n\n正确策略是按轴角色推理：消费 C，生成 O，保留 B/T。",
    visualLines: ["Case A: standard", "Case B: stored [O,C]", "Case C: no size guess", "all hidden cases pass"],
    taskPrompt: "完成三个隐藏测试槽位，然后运行最终验证。"
  }
};

const chapter01StageKnowledge: Partial<Record<LevelPhase, Chapter01StageKnowledge>> = {
  tensor_object: {
    code: "0-1A",
    title: "Tensor Object Intake",
    concept: "object intake / dtype visible / type accepted",
    tool: "Tensor Inspector",
    mission: "target: scalar / vector / matrix / 3D block",
    visual: "tensor_objects",
    inspectorNotes: ["一个数字也可以是 tensor。", "一排数、一张表、一叠表只是排列方式不同。", "本阶段不讨论轴和长度。"],
    failureLesson: ["Tensor Object 还没全部送检。", "不要按文字含义判断；先确认它是不是结构化数字。", "把缺失对象拖入 Inspector 槽位。"],
    debrief: "Tensor 是承载数字数据的统一对象。"
  },
  rank_scanner: {
    code: "0-1B",
    title: "Rank：tensor 有几个轴",
    concept: "Rank = tensor 的轴数量。",
    tool: "Rank Scanner",
    mission: "扫描对象，贴 rank 0 / 1 / 2 / 3。",
    visual: "rank_axes",
    inspectorNotes: ["scalar 没有轴，rank = 0。", "vector 是 1 条轴，matrix 是 2 条轴。", "一叠矩阵是 3 条轴。"],
    failureLesson: ["Rank mismatch。", "Rank 只表示轴的数量，不表示数值大小。", "重新观察对象有几个独立方向。"],
    debrief: "Rank 描述 tensor 有几个轴。下一步测量每个轴有多长。"
  },
  shape_caliper: {
    code: "0-1C",
    title: "Shape：tensor 的结构尺寸",
    concept: "Shape = 每个轴的长度，按轴顺序排列。",
    tool: "Shape Caliper",
    mission: "测量 Axis 0 / 1 / 2，修成 [2,4,8]。",
    visual: "shape_caliper",
    inspectorNotes: ["Axis 0 的长度放在第 1 个位置。", "Axis 1 的长度放在第 2 个位置。", "Axis 2 的长度放在第 3 个位置。"],
    failureLesson: ["Shape order mismatch。", "Shape 的数字顺序必须跟 axis 顺序一致。", "把量尺结果按 Axis 0、1、2 依次放入槽位。"],
    debrief: "Rank 表示有几个轴，Shape 表示每个轴有多长。下一步判断这些轴分别代表什么。"
  },
  semantic_gap: {
    code: "0-1D",
    title: "Shape 不等于语义",
    concept: "float32[2,4,8] 只说明大小，不说明每个轴的用途。",
    tool: "Semantic Inspector",
    mission: "标记语义缺失，观察下游阻塞。",
    visual: "semantic_gap",
    inspectorNotes: ["shape known 不代表 contract ready。", "下游模块需要知道哪个轴可被消费。", "这是合同缺失，不是数值错误。"],
    failureLesson: ["Connection blocked。", "下游模块不只需要 rank 和长度，还需要 axis semantics。", "先把语义状态标为 unresolved。"],
    debrief: "Shape 说明大小，Axis Semantics 说明每个轴的用途。下一步从 token_ids[B,T] 建立 B 和 T。"
  },
  token_grid_builder: {
    code: "0-1E",
    title: "Token IDs：从文字到二维表格",
    concept: "多条文本的 token ids 通常组织成 token_ids[B,T]。",
    tool: "Tokenizer + Token Grid",
    mission: "连 Text -> Tokenizer -> Grid，标 row=B、column=T。",
    visual: "token_grid",
    inspectorNotes: ["一行是一条独立样本。", "一列是所有样本的同一个 token 位置。", "B 保存样本分离，T 保存顺序。"],
    failureLesson: ["Axis semantic error。", "Batch axis 分离样本；Token axis 保持样本内部顺序。", "不要把横轴标成 batch。"],
    debrief: "token_ids[B,T] 建立了 B 和 T：B 让模型一次处理多条样本，T 保存每条样本内部的 token 顺序。"
  },
  embedding_expansion: {
    code: "0-1F",
    title: "Embedding：给每个 token 增加特征通道",
    concept: "Embedding Lookup 把每个 token id 变成一条 C 维向量。",
    tool: "Embedding Lookup",
    mission: "连接 ids/table，展开 hidden[B,T,C]。",
    visual: "embedding_expansion",
    inspectorNotes: ["token id 是查表用的整数。", "表中的每一行是一条 C 维向量。", "C 是 token 内部的 feature channel。"],
    failureLesson: ["Expansion error。", "C 不是新的时间步，T 的位置没有增加。", "C 是每个 token 内部的向量长度。"],
    debrief: "Embedding 把 token_ids[B,T] 扩展为 hidden[B,T,C]：B 是样本，T 是 token 位置，C 是特征通道。"
  },
  hidden_contract_repair: {
    code: "0-1G",
    title: "Hidden Tensor：模型内部的工作状态",
    concept: "Hidden Tensor 是中间表示，常见合同是 hidden[B,T,C]。",
    tool: "Batch / Time / Channel Probe",
    mission: "Probe 三个轴，再贴 B/T/C。",
    visual: "hidden_contract",
    inspectorNotes: ["B 轴分离不同样本。", "T 轴沿 token 顺序变化。", "C 轴包含连续 feature values。"],
    failureLesson: ["Contract failed。", "你标错的轴表现出了另一个语义。", "先看 Probe 证据，再贴 B/T/C。"],
    debrief: "你已经恢复 hidden[B,T,C]。B/T/C 不是装饰，下一步要证明它们能被下游模块消费。"
  },
  consumer_validation: {
    code: "0-1H",
    title: "Axis Semantics 是模块合同",
    concept: "B/T/C 不是装饰标签，而是下游模块的连接合同。",
    tool: "Consumer Ports",
    mission: "手工连接 B/T/C 到三个消费者。",
    visual: "consumer_contract",
    inspectorNotes: ["Batch Viewer 消费 B。", "Causal Mask 消费 T，生成 [T,T] mask。", "Linear Projection 消费 C。"],
    failureLesson: ["Consumer contract failed。", "模块收到错误轴会在错误方向上计算。", "把 B/T/C 接给对应消费者。"],
    debrief: "B 被样本查看器使用，T 被 attention / causal mask 使用，C 被 linear / projection 使用。"
  },
  hidden_test_gauntlet: {
    code: "0-1X",
    title: "B/T/C 不是固定数字",
    concept: "轴语义靠行为和合同判断，不靠哪个长度最大。",
    tool: "Hidden Tests",
    mission: "运行 shape 变体，验证不是记数字。",
    visual: "hidden_tests",
    inspectorNotes: ["同样是 hidden[B,T,C]，尺寸可以变化。", "长轴不一定是 C。", "用 Probe 行为和下游合同判断语义。"],
    failureLesson: ["Size-based guess failed。", "Axis length alone does not define meaning。", "回到行为证据，而不是按数字大小猜。"],
    debrief: "你已经掌握 Tensor、Rank、Shape、Axis Semantics、token_ids[B,T]、Embedding Expansion 和 hidden[B,T,C]。"
  }
};

const chapter02StageKnowledge: Partial<Record<LevelPhase, Chapter01StageKnowledge>> = {
  matmul_dot_cell: {
    code: "0-2A",
    title: "Dot Cell / 点积单元",
    concept: "一个输出值 = 一次 [C] dot [C] multiply-and-sum。",
    tool: "Dot Cell",
    mission: "修复 C 向量、C 权重向量、scalar 输出。",
    visual: "dot_cell",
    inspectorNotes: ["点积要求两个向量长度相同。", "O 不是单个 Dot Cell 的输入长度。", "一个 Dot Cell 只生成一个 scalar。"],
    failureLesson: ["Inner length mismatch。", "先确认两个输入都是 C-length。", "不要把完整 O vector 当成单个点积结果。"],
    debrief: "你已经把 Linear 的最小计算单元拆成了一个 Dot Cell：两个 C-length 向量生成一个 scalar。"
  },
  matmul_token_projection: {
    code: "0-2B",
    title: "Token Projection / 单 token 投影",
    concept: "O 个输出值来自 O 条 C-length 权重向量。",
    tool: "Weight Plate + MatMul Gate",
    mission: "修复 [C] @ [C,O] -> [O]。",
    visual: "token_projection",
    inspectorNotes: ["输入 token 是 [C]。", "compute weight 必须是 [C,O]。", "输出轴 O 来自权重板的列。"],
    failureLesson: ["Output axis error。", "C 被消费，不应该继续当作输出 feature 轴。", "权重板方向决定 O 从哪里生成。"],
    debrief: "单 token 投影已经成立：[C] @ [C,O] -> [O]。下一步把它复制到整条序列。"
  },
  matmul_sequence_projection: {
    code: "0-2C",
    title: "Sequence Projection / 序列投影",
    concept: "T 轴是携带轴，不参与点积；同一块 W 被复用。",
    tool: "Sequence MatMul",
    mission: "修复 [T,C] @ [C,O] -> [T,O]。",
    visual: "sequence_projection",
    inspectorNotes: ["每个 T 位置都有一条 C 向量。", "W 不随 token 位置变化。", "输出保留 T，把 C 替换成 O。"],
    failureLesson: ["Carrier axis consumed。", "T 轴负责顺序，不是 MatMul 的 inner dimension。", "把点积限制在最后一维 C。"],
    debrief: "序列投影已经成立：T 被保留，C 被消费，O 被生成。下一步扩展到 batch。"
  },
  matmul_batch_projection: {
    code: "0-2D",
    title: "Batch Projection / 批量投影",
    concept: "B/T 都是并行携带轴，Linear 只消费最后一维 C。",
    tool: "Batch MatMul",
    mission: "修复 [B,T,C] @ [C,O] -> [B,T,O]。",
    visual: "batch_projection",
    inspectorNotes: ["B 分离样本。", "T 保持 token 顺序。", "C 是 Linear 的输入宽度。"],
    failureLesson: ["Consumed wrong axis。", "B 和 T 不该参与点积。", "完整输出合同必须是 [B,T,O]。"],
    debrief: "完整 batch 投影已经成立：hidden[B,T,C] @ W[C,O] -> out[B,T,O]。下一步处理权重方向陷阱。"
  },
  matmul_orientation_trap: {
    code: "0-2E",
    title: "Weight Orientation Trap / 权重方向陷阱",
    concept: "stored [O,C] 必须转成 compute [C,O] 才能进入 gate。",
    tool: "Transpose Switch",
    mission: "识别存储方向，修复计算方向。",
    visual: "weight_orientation",
    inspectorNotes: ["存储方向可能是 [O,C]。", "MatMul compute view 需要 [C,O]。", "Transpose 改轴顺序，不改变数值本身。"],
    failureLesson: ["Weight orientation error。", "不要把存储 shape 直接送进 MatMul。", "先把 [O,C] 修成 [C,O]。"],
    debrief: "你已经区分存储方向和计算方向。下一步把修好的 MatMul 封装成 Linear Projection。"
  },
  matmul_linear_assembly: {
    code: "0-2F",
    title: "Linear Module Assembly / 组装 Linear",
    concept: "Linear Projection = MatMul core；本关 bias 锁定；reference 检查数值。",
    tool: "Linear Module + Reference Checker",
    mission: "组装 MatMul core、bias policy、allclose。",
    visual: "linear_assembly",
    inspectorNotes: ["Linear 的核心是 x @ W。", "0-2 暂时锁定 bias，避免混入广播加法。", "reference allclose 防止 shape 对但数值错。"],
    failureLesson: ["Numeric mismatch。", "Shape 正确不代表计算方向正确。", "用 reference checker 检查最终数值。"],
    debrief: "Linear Projection 已组装完成。最后用隐藏 case 证明这个合同不是只适配一个可见形状。"
  },
  matmul_gauntlet: {
    code: "0-2X",
    title: "MatMul Gauntlet / 最终隐藏测试",
    concept: "MatMul Contract 必须跨方向、尺寸和隐藏 case 泛化。",
    tool: "Hidden Test Gauntlet",
    mission: "通过标准方向、存储方向、尺寸陷阱三类测试。",
    visual: "matmul_gauntlet",
    inspectorNotes: ["Case A 检查标准 [C,O]。", "Case B 检查 stored [O,C] 修复。", "Case C 防止按大小猜轴。"],
    failureLesson: ["Hidden case failed。", "回到轴角色：消费 C，生成 O，保留 B/T。", "不要依赖固定数字或最大维度。"],
    debrief: "0-2 完成：你已经从 Dot Cell 推导到完整 Linear Projection，并通过隐藏 MatMul Contract 测试。"
  }
};

const chapter01PhaseNodePositions: Partial<Record<LevelPhase, Record<string, { x: number; y: number }>>> = {
  tensor_object: {
    tensor_inspector: { x: 548, y: 278 },
    type_check: { x: 1052, y: 316 }
  },
  rank_scanner: {
    raw_objects: { x: 360, y: 296 },
    rank_scanner: { x: 628, y: 250 },
    rank_gate: { x: 1030, y: 296 }
  },
  shape_caliper: {
    hidden_tensor: { x: 330, y: 338 },
    shape_caliper: { x: 660, y: 278 },
    shape_gate: { x: 1060, y: 336 }
  },
  semantic_gap: {
    shape_gate: { x: 374, y: 318 },
    semantic_inspector: { x: 740, y: 278 }
  },
  token_grid_builder: {
    text_batch: { x: 340, y: 328 },
    tokenizer: { x: 604, y: 320 },
    token_grid: { x: 910, y: 260 }
  },
  embedding_expansion: {
    token_grid: { x: 338, y: 214 },
    embedding_table: { x: 338, y: 474 },
    embedding_lookup: { x: 710, y: 304 },
    hidden_tensor: { x: 1110, y: 304 }
  },
  hidden_contract_repair: {
    hidden_tensor: { x: 690, y: 338 }
  },
  consumer_validation: {
    hidden_tensor: { x: 324, y: 332 },
    batch_viewer: { x: 800, y: 146 },
    causal_mask: { x: 800, y: 332 },
    linear_probe: { x: 800, y: 520 },
    shape_tests: { x: 1136, y: 332 }
  },
  hidden_test_gauntlet: {
    hidden_tensor: { x: 324, y: 332 },
    batch_viewer: { x: 784, y: 146 },
    causal_mask: { x: 784, y: 332 },
    linear_probe: { x: 784, y: 520 },
    shape_tests: { x: 1124, y: 332 }
  }
};

const chapter02PhaseNodePositions: Partial<Record<LevelPhase, Record<string, { x: number; y: number }>>> = {
  matmul_dot_cell: {
    feature_vector: { x: 260, y: 246 },
    weight_vector: { x: 504, y: 246 },
    dot_cell: { x: 758, y: 256 },
    dot_scalar: { x: 994, y: 258 }
  },
  matmul_token_projection: {
    token_vector: { x: 270, y: 250 },
    weight_plate: { x: 520, y: 244 },
    matmul_gate: { x: 780, y: 254 },
    token_output: { x: 1030, y: 250 }
  },
  matmul_sequence_projection: {
    sequence_tensor: { x: 250, y: 244 },
    weight_plate: { x: 520, y: 250 },
    matmul_gate: { x: 790, y: 258 },
    sequence_output: { x: 1038, y: 244 }
  },
  matmul_batch_projection: {
    linear_input: { x: 238, y: 244 },
    weight_plate: { x: 516, y: 250 },
    matmul_gate: { x: 790, y: 258 },
    linear_out: { x: 1040, y: 244 }
  },
  matmul_orientation_trap: {
    linear_input: { x: 226, y: 220 },
    weight_plate: { x: 500, y: 190 },
    transpose_switch: { x: 520, y: 378 },
    matmul_gate: { x: 786, y: 256 },
    linear_out: { x: 1040, y: 220 }
  },
  matmul_linear_assembly: {
    linear_input: { x: 210, y: 218 },
    weight_plate: { x: 484, y: 186 },
    transpose_switch: { x: 492, y: 386 },
    matmul_gate: { x: 750, y: 230 },
    linear_module: { x: 754, y: 386 },
    linear_out: { x: 1006, y: 218 },
    reference_checker: { x: 1010, y: 386 }
  },
  matmul_gauntlet: {
    linear_input: { x: 180, y: 218 },
    weight_plate: { x: 430, y: 174 },
    transpose_switch: { x: 430, y: 362 },
    matmul_gate: { x: 676, y: 238 },
    linear_out: { x: 916, y: 218 },
    reference_checker: { x: 916, y: 376 },
    matmul_tests: { x: 1158, y: 302 }
  }
};

export function App() {
  const [mode, setMode] = useState<WorkbenchMode>("build");
  const [selectedLevelId, setSelectedLevelId] = useState(bootcampLevels[0].id);
  const [selectedId, setSelectedId] = useState(bootcampLevels[0].defaultSelectedNodeId);
  const [playing, setPlaying] = useState(true);
  const [repairStates, setRepairStates] = useState<Record<string, LevelRepairState>>({});
  const [results, setResults] = useState<Record<string, BootcampResult>>({});
  const [introSeen, setIntroSeen] = useState<Record<string, boolean>>({});
  const [stageIntroSeen, setStageIntroSeen] = useState<Record<string, Record<string, boolean>>>({});
  const [missionStarted, setMissionStarted] = useState<Record<string, boolean>>({});
  const [completionDismissed, setCompletionDismissed] = useState<Record<string, boolean>>({});
  const [nodePositions, setNodePositions] = useState<Record<string, NodePositionMap>>({});
  const [stageKnowledgePositions, setStageKnowledgePositions] = useState<Record<string, StageKnowledgePositionMap>>({});
  const [pendingStageDebrief, setPendingStageDebrief] = useState<Record<string, LevelPhase | undefined>>({});
  const [stageDebriefOpen, setStageDebriefOpen] = useState<Record<string, boolean>>({});
  const [stageIntroRecallOpen, setStageIntroRecallOpen] = useState(false);
  const [tensorObjectDetail, setTensorObjectDetail] = useState<TensorObjectDetailState | null>(null);
  const [tensorRunOutputs, setTensorRunOutputs] = useState<Record<string, string>>({});
  const [canvasMenu, setCanvasMenu] = useState<CanvasContextTarget | null>(null);

  const activeLevel = useMemo(
    () => bootcampLevels.find((level) => level.id === selectedLevelId) ?? bootcampLevels[0],
    [selectedLevelId]
  );
  const activeRepairState = repairStates[activeLevel.id] ?? createInitialRepairState();
  const levelResult = results[activeLevel.id];
  const showKnowledgeIntro = Boolean(activeLevel.knowledgeCards?.length && !introSeen[activeLevel.id]);
  const showMissionModal = Boolean(activeLevel.mission && !showKnowledgeIntro && !missionStarted[activeLevel.id]);
  const derivedLevelPhase = deriveLevelPhase(activeLevel, activeRepairState, levelResult, showKnowledgeIntro, showMissionModal);
  const pendingDebriefPhase = pendingStageDebrief[activeLevel.id];
  const levelPhase = pendingDebriefPhase ?? derivedLevelPhase;
  const baseDisplayNodes = useMemo(() => buildDisplayNodes(activeLevel, activeRepairState.assignments), [activeLevel, activeRepairState.assignments]);
  const phaseDisplayNodes = useMemo(() => filterDisplayNodesForPhase(activeLevel, baseDisplayNodes, levelPhase), [activeLevel, baseDisplayNodes, levelPhase]);
  const displayNodes = useMemo(() => applyNodePositions(phaseDisplayNodes, nodePositions[activeLevel.id]), [activeLevel.id, phaseDisplayNodes, nodePositions]);
  const displayEdges = useMemo(
    () => filterDisplayEdgesForPhase(activeLevel, buildDisplayEdges(activeLevel, activeRepairState.assignments), displayNodes, levelPhase),
    [activeLevel, activeRepairState.assignments, displayNodes, levelPhase]
  );
  const selectedNode = useMemo(
    () => displayNodes.find((node) => node.id === selectedId) ?? displayNodes[0] ?? baseDisplayNodes[0],
    [baseDisplayNodes, displayNodes, selectedId]
  );
  const modeCopy = modeLabels[mode];
  const passedCount = bootcampLevels.filter((level) => results[level.id]?.passed).length;
  const latestObservation = activeRepairState.observations[0];
  const stageKnowledge = useMemo(() => buildCanvasStageKnowledge(activeLevel, levelPhase), [activeLevel, levelPhase]);
  const stageKnowledgePosition = stageKnowledge ? stageKnowledgePositions[activeLevel.id]?.[stageKnowledge.code] : undefined;
  const activeLevelIndex = useMemo(() => bootcampLevels.findIndex((level) => level.id === activeLevel.id), [activeLevel.id]);
  const nextLevel = activeLevelIndex >= 0 ? bootcampLevels[activeLevelIndex + 1] : undefined;
  const showCompletionModal = Boolean(levelResult?.passed && !completionDismissed[activeLevel.id]);
  const activeStageIntro = !pendingDebriefPhase ? stageIntroForLevelPhase(activeLevel, levelPhase) : undefined;
  const pendingDebriefKnowledge = pendingDebriefPhase ? stageKnowledgeForLevelPhase(activeLevel, pendingDebriefPhase) : undefined;
  const pendingDebriefNextLabel = nextStageLabel(activeLevel, pendingDebriefPhase);
  const showStageIntro = Boolean(
    activeStageIntro &&
      !showKnowledgeIntro &&
      !showMissionModal &&
      !levelResult?.passed &&
      !stageIntroSeen[activeLevel.id]?.[levelPhase]
  );
  const slotOverlays = useMemo(
    () => buildSlotOverlays(activeLevel, activeRepairState, levelPhase),
    [activeLevel, activeRepairState.assignments, activeRepairState.selectedSlotId, levelPhase]
  );
  const connectionOverlays = useMemo(
    () => buildCanvasConnections(activeLevel, activeRepairState, levelPhase),
    [activeLevel, activeRepairState.assignments, levelPhase]
  );
  const actionHints = useMemo(
    () => buildCanvasActionHints(activeLevel, activeRepairState, levelPhase),
    [activeLevel, activeRepairState.assignments, activeRepairState.observations.length, levelPhase]
  );
  const inspectorTasks = useMemo(
    () => buildInspectorTasks(activeLevel, activeRepairState, levelPhase),
    [activeLevel, activeRepairState.assignments, activeRepairState.observations, activeRepairState.selectedSlotId, levelPhase]
  );
  const activeTensorObjectShowcase = tensorObjectDetail ? tensorObjectShowcases[tensorObjectDetail.slotId] : undefined;

  useEffect(() => {
    if (!displayNodes.some((node) => node.id === selectedId)) {
      setSelectedId(displayNodes[0]?.id ?? activeLevel.defaultSelectedNodeId);
    }
  }, [activeLevel.defaultSelectedNodeId, displayNodes, selectedId]);

  useEffect(() => {
    setCanvasMenu(null);
    setStageIntroRecallOpen(false);
  }, [activeLevel.id, levelPhase]);

  useEffect(() => {
    if (activeLevel.id === "0-1" && levelPhase === "tensor_object") return;
    setTensorObjectDetail(null);
  }, [activeLevel.id, levelPhase]);

  useEffect(() => {
    if (!supportsStageDebrief(activeLevel) || pendingDebriefPhase || levelResult?.passed) return;
    if (!stageReadyForDebrief(activeLevel, levelPhase, activeRepairState.assignments, activeRepairState.observations)) return;
    setPendingStageDebrief((current) => ({ ...current, [activeLevel.id]: levelPhase }));
    setStageDebriefOpen((current) => ({ ...current, [activeLevel.id]: false }));
  }, [activeLevel, activeRepairState.assignments, activeRepairState.observations, levelPhase, levelResult?.passed, pendingDebriefPhase]);

  function selectLevel(level: BootcampLevel) {
    setSelectedLevelId(level.id);
    setSelectedId(level.defaultSelectedNodeId);
    setPlaying(true);
  }

  function continueAfterCompletion() {
    setCompletionDismissed((current) => ({ ...current, [activeLevel.id]: true }));
    if (nextLevel) {
      selectLevel(nextLevel);
    }
  }

  function updateActiveRepairState(updater: (state: LevelRepairState) => LevelRepairState) {
    setRepairStates((current) => {
      const previous = current[activeLevel.id] ?? createInitialRepairState();
      return { ...current, [activeLevel.id]: updater(previous) };
    });
  }

  function clearActiveResult() {
    setResults((current) => {
      if (!current[activeLevel.id]) return current;
      const next = { ...current };
      delete next[activeLevel.id];
      return next;
    });
  }

  function selectTag(tagId: string) {
    updateActiveRepairState((state) => ({
      ...state,
      activeTagId: state.activeTagId === tagId ? undefined : tagId,
      activeProbeId: undefined
    }));
  }

  function selectProbe(probeId: string) {
    updateActiveRepairState((state) => ({
      ...state,
      activeProbeId: state.activeProbeId === probeId ? undefined : probeId,
      activeTagId: undefined
    }));
  }

  function assignTagToSlot(slot: RepairSlot, tagId: string) {
    const nextAssignments = { ...activeRepairState.assignments };
    for (const [slotId, assignedTagId] of Object.entries(nextAssignments)) {
      if (assignedTagId === tagId) delete nextAssignments[slotId];
    }
    nextAssignments[slot.id] = tagId;
    const assignedCorrectly = slot.correctTagIds.includes(tagId);
    const shouldOpenTensorObjectDetail = activeLevel.id === "0-1" && levelPhase === "tensor_object" && assignedCorrectly && Boolean(tensorObjectShowcases[slot.id]);
    const shouldShowStageDebrief =
      supportsStageDebrief(activeLevel) &&
      !pendingDebriefPhase &&
      stageReadyForDebrief(activeLevel, levelPhase, nextAssignments, activeRepairState.observations);

    updateActiveRepairState((state) => {
      const assignments = { ...state.assignments };
      for (const [slotId, assignedTagId] of Object.entries(assignments)) {
        if (assignedTagId === tagId) delete assignments[slotId];
      }
      assignments[slot.id] = tagId;
      return { ...state, assignments, selectedSlotId: slot.id, activeTagId: undefined };
    });
    if (shouldShowStageDebrief) {
      setPendingStageDebrief((current) => ({ ...current, [activeLevel.id]: levelPhase }));
      setStageDebriefOpen((current) => ({ ...current, [activeLevel.id]: false }));
    }
    if (shouldOpenTensorObjectDetail) {
      setTensorObjectDetail({ slotId: slot.id, modalOpen: true });
    }
    clearActiveResult();
    setSelectedId(tensorObjectGeneratedNodeId(slot.id) ?? slot.focusNodeId);
    setPlaying(true);
  }

  function openTensorObjectDetail(slotId: string) {
    if (!tensorObjectShowcases[slotId]) return;
    setTensorObjectDetail({ slotId, modalOpen: true });
  }

  function minimizeTensorObjectDetail(slotId: string) {
    setTensorObjectDetail({ slotId, modalOpen: false });
  }

  function runTensorObjectShowcase(showcase: TensorObjectShowcase) {
    setTensorRunOutputs((current) => ({
      ...current,
      [showcase.id]: runTensorObjectProgram(showcase)
    }));
  }

  function connectCanvasSlot(slotId: string, tagId: string) {
    const slot = activeLevel.repair.slots.find((item) => item.id === slotId);
    if (!slot) return;
    assignTagToSlot(slot, tagId);
  }

  function assignCanvasSlot(slotId: string, tagId: string) {
    const slot = activeLevel.repair.slots.find((item) => item.id === slotId);
    if (!slot) return;
    assignTagToSlot(slot, tagId);
    setCanvasMenu(null);
  }

  function probeCanvasSlot(slotId: string, probeId: string) {
    const slot = activeLevel.repair.slots.find((item) => item.id === slotId);
    if (!slot) return;
    runProbeOnSlot(slot, probeId);
    setCanvasMenu(null);
  }

  function runCanvasTests() {
    runTests();
    setCanvasMenu(null);
  }

  function moveCanvasNode(nodeId: string, x: number, y: number) {
    setNodePositions((current) => ({
      ...current,
      [activeLevel.id]: {
        ...(current[activeLevel.id] ?? {}),
        [nodeId]: { x, y }
      }
    }));
    setSelectedId(nodeId);
  }

  function moveStageKnowledge(x: number, y: number) {
    if (!stageKnowledge) return;
    setStageKnowledgePositions((current) => ({
      ...current,
      [activeLevel.id]: {
        ...(current[activeLevel.id] ?? {}),
        [stageKnowledge.code]: { x, y }
      }
    }));
  }

  function runProbeOnSlot(slot: RepairSlot, probeId: string) {
    const probe = activeLevel.repair.probes.find((item) => item.id === probeId);
    if (!probe) return;

    const fallback: ProbeObservation = {
      id: `${probe.id}_${slot.id}`,
      title: "No strong signal",
      detail: "This probe does not reveal much about the selected slot.",
      evidence: ["Try a different probe or inspect another slot."],
      possibleSemantic: "unknown",
      confidence: "low"
    };
    const observation = probe.observations[slot.id] ?? fallback;
    const nextObservations = [
      {
        ...observation,
        probeId: probe.id,
        probeLabel: probe.label,
        slotId: slot.id,
        slotLabel: slot.label
      },
      ...activeRepairState.observations
    ].slice(0, 6);

    updateActiveRepairState((state) => ({
      ...state,
      selectedSlotId: slot.id,
      probeUses: state.probeUses + probe.budgetCost,
      observations: nextObservations
    }));
    if (
      supportsStageDebrief(activeLevel) &&
      !pendingDebriefPhase &&
      stageReadyForDebrief(activeLevel, levelPhase, activeRepairState.assignments, nextObservations)
    ) {
      setPendingStageDebrief((current) => ({ ...current, [activeLevel.id]: levelPhase }));
      setStageDebriefOpen((current) => ({ ...current, [activeLevel.id]: false }));
    }
    setSelectedId(slot.focusNodeId);
    setPlaying(true);
  }

  function activateSlot(slotId: string) {
    const slot = activeLevel.repair.slots.find((item) => item.id === slotId);
    if (!slot) return;

    if (activeRepairState.activeProbeId) {
      runProbeOnSlot(slot, activeRepairState.activeProbeId);
      return;
    }

    if (activeRepairState.activeTagId) {
      assignTagToSlot(slot, activeRepairState.activeTagId);
      return;
    }

    const defaultProbeId = defaultProbeForSlot(activeLevel, levelPhase, slot);
    if (defaultProbeId) {
      runProbeOnSlot(slot, defaultProbeId);
      return;
    }

    updateActiveRepairState((state) => ({ ...state, selectedSlotId: slot.id }));
    setSelectedId(slot.focusNodeId);
  }

  function runTests() {
    const nextReferenceRuns = activeRepairState.referenceRuns + 1;
    const result = evaluateBootcampLevel(activeLevel, activeRepairState.assignments, {
      probeUses: activeRepairState.probeUses,
      referenceRuns: nextReferenceRuns
    });
    updateActiveRepairState((state) => ({ ...state, referenceRuns: nextReferenceRuns }));
    setResults((current) => ({ ...current, [activeLevel.id]: result }));
    setSelectedId(result.focusNodeId);
    setPlaying(true);
  }

  function getCanvasMenuActions(target: CanvasContextTarget | null): CanvasMenuAction[] {
    if (!target || activeLevel.id !== "0-1") return [];

    const actions: CanvasMenuAction[] = [];
    const targetSlot = target.kind === "slot" ? activeLevel.repair.slots.find((slot) => slot.id === target.id) : undefined;

    if (targetSlot) {
      const slotCategory = activeLevel.repair.tags.find((tag) => tag.id === targetSlot.correctTagIds[0])?.category;
      const axisSlot = slotCategory === "axis";

      if (axisSlot) {
        visibleProbesForPhase(activeLevel, levelPhase).forEach((probe) => {
          actions.push({
            id: `probe_${probe.id}`,
            label: probe.label,
            detail: probe.detail,
            onSelect: () => probeCanvasSlot(targetSlot.id, probe.id)
          });
        });
      }
    }

    if ((target.kind === "node" && target.id === "shape_tests") || target.kind === "canvas") {
      if (levelPhase === "hidden_test_gauntlet") {
        actions.push({
          id: "run_tests",
          label: "Run Hidden Test Gauntlet",
          detail: "Execute visible, behavior, and hidden shape checks.",
          onSelect: runCanvasTests
        });
      }
    }

    return actions;
  }

  function resetCurrentLevel() {
    setRepairStates((current) => {
      const next = { ...current };
      delete next[activeLevel.id];
      return next;
    });
    setResults((current) => {
      const next = { ...current };
      delete next[activeLevel.id];
      return next;
    });
    setNodePositions((current) => {
      if (!current[activeLevel.id]) return current;
      const next = { ...current };
      delete next[activeLevel.id];
      return next;
    });
    setPendingStageDebrief((current) => {
      if (!current[activeLevel.id]) return current;
      const next = { ...current };
      delete next[activeLevel.id];
      return next;
    });
    setStageDebriefOpen((current) => {
      if (!current[activeLevel.id]) return current;
      const next = { ...current };
      delete next[activeLevel.id];
      return next;
    });
    setStageIntroSeen((current) => {
      if (!current[activeLevel.id]) return current;
      const next = { ...current };
      delete next[activeLevel.id];
      return next;
    });
    setTensorObjectDetail(null);
    setTensorRunOutputs({});
    setSelectedId(activeLevel.defaultSelectedNodeId);
    setPlaying(true);
  }

  return (
    <main className="appShell">
      <header className="topBar">
        <div className="brand">
          <div className="brandMark">
            <Boxes size={28} />
          </div>
          <div>
            <p className="eyebrow">LLM Complete / MVP 0.0.3-dev</p>
            <h1>Tensor Bootcamp</h1>
          </div>
        </div>
        <div className="topControls">
          <div className="modeSwitch" aria-label="workbench mode">
            {(Object.keys(modeLabels) as WorkbenchMode[]).map((modeKey) => (
              <button key={modeKey} className={mode === modeKey ? "modeButton active" : "modeButton"} onClick={() => setMode(modeKey)}>
                {modeIcons[modeKey]}
                {modeLabels[modeKey].label}
              </button>
            ))}
          </div>
          <button className="iconButton" title={playing ? "Pause flow animation" : "Play flow animation"} onClick={() => setPlaying((value) => !value)}>
            {playing ? <Pause size={17} /> : <Play size={17} />}
          </button>
          <button
            className="iconButton"
            title="Reset current level"
            onClick={() => {
              setMode("build");
              resetCurrentLevel();
            }}
          >
            <RotateCcw size={17} />
          </button>
        </div>
      </header>

      <section className="statusStrip">
        <StatusPill label="Chapter" value={`${activeLevel.id} / ${activeLevel.title}`} />
        <StatusPill label="Mode" value={modeCopy.title} />
        <StatusPill label="Selected" value={`${selectedNode.semanticName} ${selectedNode.shape}`} />
        <StatusPill label="Tool" value={activeToolLabel(activeLevel, activeRepairState)} />
        <StatusPill label="Objective" value={phaseLabel(levelPhase)} />
        <StatusPill label="Progress" value={`${passedCount}/${bootcampLevels.length} passed`} />
      </section>

      <section className="workbenchGrid">
        <aside className="panel moduleShelf">
          <div className="panelHeader">
            <Wrench size={18} />
            <h2>Chapter 0</h2>
          </div>
          <p className="panelNote">玩家产物现在是可验证的 shape repair，而不是选择题答案。</p>

          <div className="levelList">
            {bootcampLevels.map((level) => {
              const result = results[level.id];
              const statusClass = result ? (result.passed ? "pass" : "fail") : "pending";
              const showNestedChallenges = supportsStageRail(level) && selectedLevelId === level.id;
              return (
                <div key={level.id} className={`levelGroup ${showNestedChallenges ? "expanded" : ""}`}>
                  <button className={`levelItem ${selectedLevelId === level.id ? "active" : ""} ${statusClass}`} onClick={() => selectLevel(level)}>
                    <span className="levelId">{level.id}</span>
                    <span>
                      <b>{level.title}</b>
                      <small>{level.subtitle}</small>
                      <small>Tool: {repairKindLabel(level.repair.kind)}</small>
                      <small>Reward: {level.unlocks[0]}</small>
                    </span>
                    <StateIcon state={result ? (result.passed ? "pass" : "fail") : "warn"} />
                  </button>
                  {showNestedChallenges ? (
                    <ChapterChallengeRail level={level} repairState={activeRepairState} phase={levelPhase} result={levelResult} />
                  ) : null}
                </div>
              );
            })}
          </div>

          <div className="contractBox">
            <b>Shape Contract</b>
            <code>{activeLevel.repair.targetContract}</code>
            {activeLevel.contracts.map((contract) => (
              <code key={contract}>{contract}</code>
            ))}
          </div>

          <div className="unlockBox">
            <b>Unlocks</b>
            <div>
              {activeLevel.unlocks.map((unlock) => (
                <span key={unlock}>{unlock}</span>
              ))}
            </div>
          </div>
        </aside>

        <section className="stagePanel panel">
          <PixiWorkbenchCanvas
            selectedId={selectedId}
            mode={mode}
            playing={playing}
            nodes={displayNodes}
            edges={displayEdges}
            slotOverlays={slotOverlays}
            connectionOverlays={connectionOverlays}
            actionHints={actionHints}
            stageKnowledge={stageKnowledge}
            stageKnowledgePosition={stageKnowledgePosition}
            onSlotSelect={activateSlot}
            onCanvasConnect={connectCanvasSlot}
            onCanvasDropTag={assignCanvasSlot}
            onCanvasContextMenu={setCanvasMenu}
            onNodeMove={moveCanvasNode}
            onStageKnowledgeMove={moveStageKnowledge}
            onSelect={setSelectedId}
          />
          {activeStageIntro && !showKnowledgeIntro && !showMissionModal && !pendingDebriefPhase && !levelResult?.passed ? (
            <button
              className="canvasKnowledgeButton"
              type="button"
              title="打开当前阶段知识卡片"
              aria-label="打开当前阶段知识卡片"
              onClick={() => setStageIntroRecallOpen(true)}
            >
              <BookOpen size={18} />
              <span>知识卡片</span>
            </button>
          ) : null}
          {activeLevel.id === "0-1" && levelPhase !== "tensor_object" ? (
            <div className="canvasPaletteLayer">
              <BlueprintPalette level={activeLevel} repairState={activeRepairState} phase={levelPhase} onSelectTag={selectTag} onSelectProbe={selectProbe} />
            </div>
          ) : null}
          <CanvasContextMenu target={canvasMenu} actions={getCanvasMenuActions(canvasMenu)} onClose={() => setCanvasMenu(null)} />
          {pendingDebriefPhase && !stageDebriefOpen[activeLevel.id] ? (
            <StageDebriefPrompt
              stageKnowledge={pendingDebriefKnowledge}
              nextLabel={pendingDebriefNextLabel}
              onOpen={() => setStageDebriefOpen((current) => ({ ...current, [activeLevel.id]: true }))}
            />
          ) : null}
          {pendingDebriefPhase && stageDebriefOpen[activeLevel.id] ? (
            <StageDebriefOverlay
              stageKnowledge={pendingDebriefKnowledge}
              nextLabel={pendingDebriefNextLabel}
              onCancel={() => setStageDebriefOpen((current) => ({ ...current, [activeLevel.id]: false }))}
              onContinue={() => {
                setPendingStageDebrief((current) => {
                  const next = { ...current };
                  delete next[activeLevel.id];
                  return next;
                });
                setStageDebriefOpen((current) => {
                  const next = { ...current };
                  delete next[activeLevel.id];
                  return next;
                });
              }}
            />
          ) : null}
          {levelResult?.passed ? (
            <LevelCompletePrompt
              level={activeLevel}
              result={levelResult}
              nextLevel={nextLevel}
              onContinue={continueAfterCompletion}
            />
          ) : null}
        </section>

        <aside className="rightStack">
          <TensorInspector
            node={selectedNode}
            observation={latestObservation}
            stageKnowledge={stageKnowledge}
            selectedSlotId={activeRepairState.selectedSlotId}
            tensorObjectDetailSlotId={tensorObjectDetail?.slotId}
            tensorRunOutputs={tensorRunOutputs}
            taskItems={inspectorTasks}
            result={levelResult}
            onOpenTensorObjectDetail={openTensorObjectDetail}
            onRunTensorObjectShowcase={runTensorObjectShowcase}
          />
          {activeLevel.id !== "0-1" ? (
            <section className="panel levelControlPanel">
              <div className="panelHeader">
                <Wrench size={18} />
                <h2>Repair Console</h2>
              </div>
              <RepairConsole
                level={activeLevel}
                repairState={activeRepairState}
                phase={levelPhase}
                onSelectTag={selectTag}
                onSelectProbe={selectProbe}
                onActivateSlot={activateSlot}
                onAssignTag={assignTagToSlot}
              />
              <div className="consoleActions">
                <button className="runButton" onClick={runTests}>
                  <Play size={16} />
                  Run Contract Tests
                </button>
              </div>
              <ResultPanel result={levelResult} activeLevel={activeLevel} />
            </section>
          ) : null}
        </aside>
      </section>

      {showKnowledgeIntro ? (
        <KnowledgeIntroOverlay
          level={activeLevel}
          onContinue={() => {
            setIntroSeen((current) => ({ ...current, [activeLevel.id]: true }));
            setPlaying(false);
          }}
        />
      ) : null}

      {showMissionModal ? (
        <MissionOverlay
          level={activeLevel}
          onStart={() => {
            setMissionStarted((current) => ({ ...current, [activeLevel.id]: true }));
            setPlaying(true);
          }}
        />
      ) : null}

      {(showStageIntro || stageIntroRecallOpen) && activeStageIntro ? (
        <StageIntroOverlay
          intro={activeStageIntro}
          actionLabel={stageIntroRecallOpen && !showStageIntro ? "返回画布" : "进入画布"}
          onStart={() => {
            setStageIntroSeen((current) => ({
              ...current,
              [activeLevel.id]: {
                ...(current[activeLevel.id] ?? {}),
                [levelPhase]: true
              }
            }));
            setStageIntroRecallOpen(false);
            setPlaying(true);
          }}
        />
      ) : null}

      {showCompletionModal ? (
        <CompletionOverlay
          level={activeLevel}
          result={levelResult}
          nextLevel={nextLevel}
          onContinue={continueAfterCompletion}
        />
      ) : null}

      {activeTensorObjectShowcase && tensorObjectDetail?.modalOpen ? (
        <TensorObjectShowcaseOverlay
          showcase={activeTensorObjectShowcase}
          output={tensorRunOutputs[activeTensorObjectShowcase.id]}
          onRun={() => runTensorObjectShowcase(activeTensorObjectShowcase)}
          onMinimize={() => minimizeTensorObjectDetail(activeTensorObjectShowcase.id)}
        />
      ) : null}
    </main>
  );
}

function createInitialRepairState(): LevelRepairState {
  return {
    assignments: {},
    probeUses: 0,
    referenceRuns: 0,
    observations: []
  };
}

function deriveLevelPhase(
  level: BootcampLevel,
  repairState: LevelRepairState,
  result: BootcampResult | undefined,
  showKnowledgeIntro: boolean,
  showMissionModal: boolean
): LevelPhase {
  if (showKnowledgeIntro) return "knowledge_intro";
  if (showMissionModal) return "mission_modal";
  if (result?.passed) return "completed";

  if (level.id !== "0-1") {
    if (level.id === "0-2") {
      if (!areSlotsCorrect(level, repairState, chapter02SlotIds("matmul_dot_cell"))) return "matmul_dot_cell";
      if (!areSlotsCorrect(level, repairState, chapter02SlotIds("matmul_token_projection"))) return "matmul_token_projection";
      if (!areSlotsCorrect(level, repairState, chapter02SlotIds("matmul_sequence_projection"))) return "matmul_sequence_projection";
      if (!areSlotsCorrect(level, repairState, chapter02SlotIds("matmul_batch_projection"))) return "matmul_batch_projection";
      if (!areSlotsCorrect(level, repairState, chapter02SlotIds("matmul_orientation_trap"))) return "matmul_orientation_trap";
      if (!areSlotsCorrect(level, repairState, chapter02SlotIds("matmul_linear_assembly"))) return "matmul_linear_assembly";
      return "matmul_gauntlet";
    }
    return result ? "visible_testing" : "axis_tagging";
  }

  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("tensor_object"))) return "tensor_object";
  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("rank_scanner"))) return "rank_scanner";
  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("shape_caliper"))) return "shape_caliper";
  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("semantic_gap"))) return "semantic_gap";
  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("token_grid_builder"))) return "token_grid_builder";
  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("embedding_expansion"))) return "embedding_expansion";

  const axesReady = areSlotsCorrect(level, repairState, chapter01SlotIds("hidden_contract_repair"));
  const probeEvidenceReady = chapter01ProbeEvidenceReady(repairState.observations);
  if (!axesReady || !probeEvidenceReady) return "hidden_contract_repair";

  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("consumer_validation"))) return "consumer_validation";
  return "hidden_test_gauntlet";
}

function areSlotsCorrect(level: BootcampLevel, repairState: LevelRepairState, slotIds: string[]) {
  return areAssignmentsCorrect(level, repairState.assignments, slotIds);
}

function areAssignmentsCorrect(level: BootcampLevel, assignments: BootcampAnswerMap, slotIds: string[]) {
  return slotIds.every((slotId) => {
    const slot = level.repair.slots.find((item) => item.id === slotId);
    return slot ? slot.correctTagIds.includes(assignments[slotId]) : true;
  });
}

function supportsStageRail(level: BootcampLevel) {
  return level.id === "0-1" || level.id === "0-2";
}

function supportsStageDebrief(level: BootcampLevel) {
  return supportsStageRail(level);
}

function challengesForLevel(level: BootcampLevel) {
  if (level.id === "0-1") return chapter01Challenges;
  if (level.id === "0-2") return chapter02Challenges;
  return [];
}

function challengeForLevelPhase(level: BootcampLevel, phase: LevelPhase) {
  if (level.id === "0-1") return chapter01ChallengeForPhase(phase);
  if (level.id === "0-2") return chapter02ChallengeForPhase(phase);
  return undefined;
}

function stageIntroForLevelPhase(level: BootcampLevel, phase: LevelPhase) {
  if (level.id === "0-1") return chapter01StageIntros[phase];
  if (level.id === "0-2") return chapter02StageIntros[phase];
  return undefined;
}

function stageKnowledgeForLevelPhase(level: BootcampLevel, phase: LevelPhase) {
  if (level.id === "0-1") return chapter01StageKnowledge[phase];
  if (level.id === "0-2") return chapter02StageKnowledge[phase];
  return undefined;
}

function stageReadyForDebrief(level: BootcampLevel, phase: LevelPhase, assignments: BootcampAnswerMap, observations: ObservationLogItem[]) {
  if (level.id === "0-1") return chapter01StageReadyForDebrief(level, phase, assignments, observations);
  if (level.id === "0-2") {
    const challenge = chapter02ChallengeForPhase(phase);
    return Boolean(challenge?.slotIds.length && areAssignmentsCorrect(level, assignments, challenge.slotIds));
  }
  return false;
}

function chapter01StageReadyForDebrief(level: BootcampLevel, phase: LevelPhase, assignments: BootcampAnswerMap, observations: ObservationLogItem[]) {
  const challenge = chapter01ChallengeForPhase(phase);
  if (!challenge?.slotIds.length) return false;
  if (!areAssignmentsCorrect(level, assignments, challenge.slotIds)) return false;
  if (challenge.phase === "hidden_contract_repair") return chapter01ProbeEvidenceReady(observations);
  return true;
}

function chapter01ProbeEvidenceReady(observations: ObservationLogItem[]) {
  const observedAxisSlots = new Set(observations.map((observation) => observation.slotId));
  return ["axis_0", "axis_1", "axis_2"].every((slotId) => observedAxisSlots.has(slotId));
}

function chapter01ProbeEvidenceCount(observations: ObservationLogItem[]) {
  const observedAxisSlots = new Set(observations.map((observation) => observation.slotId));
  return ["axis_0", "axis_1", "axis_2"].filter((slotId) => observedAxisSlots.has(slotId)).length;
}

function chapter01SlotIds(phase: LevelPhase) {
  return chapter01ChallengeByPhase.get(phase)?.slotIds ?? [];
}

function chapter02SlotIds(phase: LevelPhase) {
  return chapter02ChallengeByPhase.get(phase)?.slotIds ?? [];
}

function chapter01ChallengeForPhase(phase: LevelPhase) {
  if (phase === "knowledge_intro" || phase === "mission_modal") return chapter01Challenges[0];
  if (phase === "completed") return chapter01ChallengeByPhase.get("hidden_test_gauntlet");
  return chapter01ChallengeByPhase.get(phase);
}

function chapter02ChallengeForPhase(phase: LevelPhase) {
  if (phase === "knowledge_intro" || phase === "mission_modal") return chapter02Challenges[0];
  if (phase === "completed") return chapter02ChallengeByPhase.get("matmul_gauntlet");
  return chapter02ChallengeByPhase.get(phase);
}

function buildCanvasStageKnowledge(level: BootcampLevel, phase: LevelPhase): Chapter01StageKnowledge | undefined {
  if (level.id !== "0-1" && level.id !== "0-2") return undefined;
  const finalPhase = level.id === "0-1" ? "hidden_test_gauntlet" : "matmul_gauntlet";
  const normalizedPhase = phase === "completed" ? finalPhase : phase;
  const knowledge = stageKnowledgeForLevelPhase(level, normalizedPhase);
  if (!knowledge) return undefined;

  const challenges = challengesForLevel(level);
  const index = challenges.findIndex((challenge) => challenge.phase === normalizedPhase);
  const previous = index > 0 ? stageKnowledgeForLevelPhase(level, challenges[index - 1].phase) : undefined;
  return previous ? { ...knowledge, carryForward: `已掌握：${previous.debrief}` } : knowledge;
}

function phaseLabel(phase: LevelPhase) {
  const labels: Record<LevelPhase, string> = {
    knowledge_intro: "Knowledge Intro",
    mission_modal: "Mission Brief",
    tensor_object: "0-1A Tensor Object",
    rank_scanner: "0-1B Rank Scanner",
    shape_caliper: "0-1C Shape Caliper",
    semantic_gap: "0-1D Shape != Semantics",
    token_grid_builder: "0-1E Token Grid",
    embedding_expansion: "0-1F Embedding Expansion",
    hidden_contract_repair: "0-1G Hidden Contract",
    consumer_validation: "0-1H Consumer Validation",
    hidden_test_gauntlet: "0-1X Hidden Tests",
    matmul_dot_cell: "0-2A Dot Cell",
    matmul_token_projection: "0-2B Token Projection",
    matmul_sequence_projection: "0-2C Sequence Projection",
    matmul_batch_projection: "0-2D Batch Projection",
    matmul_orientation_trap: "0-2E Weight Orientation",
    matmul_linear_assembly: "0-2F Linear Assembly",
    matmul_gauntlet: "0-2X MatMul Gauntlet",
    data_flow_repair: "Restore Data Flow",
    tensor_generated: "Tensor Generated",
    axis_probe: "Probe Axes",
    axis_tagging: "Tag B/T/C",
    contract_wiring: "Wire Contract",
    visible_testing: "Run Visible Tests",
    hidden_testing: "Run Hidden Tests",
    completed: "Completed"
  };
  return labels[phase];
}

function visibleTagsForPhase(level: BootcampLevel, phase: LevelPhase): RepairTag[] {
  if (level.id === "0-2") {
    const challenge = chapter02ChallengeForPhase(phase);
    if (!challenge) return level.repair.tags;
    if (challenge.tagIds?.length) {
      const tagIds = new Set(challenge.tagIds);
      return level.repair.tags.filter((tag) => tagIds.has(tag.id));
    }
    const categories = new Set(challenge.tagCategories);
    return level.repair.tags.filter((tag) => tag.category && categories.has(tag.category));
  }
  if (level.id !== "0-1") return level.repair.tags;
  const challenge = chapter01ChallengeForPhase(phase);
  if (!challenge) return [];
  const categories = new Set(challenge.tagCategories);
  return level.repair.tags.filter((tag) => tag.category && categories.has(tag.category));
}

function visiblePaletteTagsForPhase(level: BootcampLevel, phase: LevelPhase): RepairTag[] {
  return visibleTagsForPhase(level, phase).filter((tag) => tag.category !== "data" && tag.category !== "consumer");
}

function visibleSlotsForPhase(level: BootcampLevel, phase: LevelPhase): RepairSlot[] {
  if (level.id === "0-2") {
    const challenge = chapter02ChallengeForPhase(phase);
    if (!challenge) return [];
    const slotIds = new Set(challenge.slotIds);
    return level.repair.slots.filter((slot) => slotIds.has(slot.id));
  }
  if (level.id !== "0-1") return level.repair.slots;
  const challenge = chapter01ChallengeForPhase(phase);
  if (!challenge) return [];
  const slotIds = new Set(challenge.slotIds);
  return level.repair.slots.filter((slot) => slotIds.has(slot.id));
}

function visibleProbesForPhase(level: BootcampLevel, phase: LevelPhase) {
  if (level.id === "0-2") {
    const probeIds =
      phase === "matmul_orientation_trap"
        ? ["orientation_probe"]
        : phase === "matmul_linear_assembly" || phase === "matmul_gauntlet"
          ? ["reference_probe"]
          : ["shape_probe"];
    return level.repair.probes.filter((probe) => probeIds.includes(probe.id));
  }
  if (level.id !== "0-1") return level.repair.probes;
  return phase === "hidden_contract_repair" ? level.repair.probes : [];
}

function defaultProbeForSlot(level: BootcampLevel, phase: LevelPhase, slot: RepairSlot) {
  if (level.id === "0-2") {
    if (phase === "matmul_orientation_trap") return "orientation_probe";
    if (phase === "matmul_linear_assembly" || phase === "matmul_gauntlet") return "reference_probe";
    return "shape_probe";
  }
  if (level.id !== "0-1" || phase !== "hidden_contract_repair") return undefined;
  const probeBySlot: Record<string, string> = {
    axis_0: "batch_probe",
    axis_1: "time_probe",
    axis_2: "channel_probe"
  };
  return probeBySlot[slot.id];
}

function visibleCanvasSlotsForPhase(level: BootcampLevel, phase: LevelPhase): RepairSlot[] {
  return visibleSlotsForPhase(level, phase);
}

function applyNodePositions(nodes: TensorNode[], positions?: NodePositionMap): TensorNode[] {
  if (!positions) return nodes;
  return nodes.map((node) => {
    const position = positions[node.id];
    return position ? { ...node, x: position.x, y: position.y } : node;
  });
}

function filterDisplayNodesForPhase(level: BootcampLevel, nodes: TensorNode[], phase: LevelPhase): TensorNode[] {
  if (level.id === "0-2") {
    const challenge = chapter02ChallengeForPhase(phase);
    if (!challenge) return nodes;
    const visibleIds = new Set(challenge.nodeIds);
    const phasePositions = chapter02PhaseNodePositions[phase] ?? {};
    return nodes
      .filter((node) => visibleIds.has(node.id))
      .map((node) => sizeChapter02PhaseNode(phasePositions[node.id] ? { ...node, ...phasePositions[node.id] } : node, phase));
  }
  if (level.id !== "0-1") return nodes;
  const challenge = chapter01ChallengeForPhase(phase);
  if (!challenge) return nodes;
  const visibleIds = new Set(challenge.nodeIds);
  const phasePositions = chapter01PhaseNodePositions[phase] ?? {};
  return nodes.filter((node) => visibleIds.has(node.id) || (phase === "tensor_object" && Boolean(tensorObjectSlotIdFromNodeId(node.id)))).map((node) => {
    const positioned = phasePositions[node.id] ? { ...node, ...phasePositions[node.id] } : node;
    if (phase !== "tensor_object") return sizeChapter01PhaseNode(positioned, phase);

    if (positioned.id === "tensor_inspector") {
      return {
        ...positioned,
        w: 330,
        h: 184,
        subtitle: "single object inspection port",
        shape: "one tensor candidate"
      };
    }

    if (positioned.id === "type_check") {
      return {
        ...positioned,
        w: 182,
        h: 92,
        subtitle: "bool4 independent checks"
      };
    }

    return positioned;
  });
}

function sizeChapter02PhaseNode(node: TensorNode, phase: LevelPhase): TensorNode {
  if (phase === "matmul_dot_cell") {
    if (node.id === "feature_vector" || node.id === "weight_vector") return { ...node, w: 204, h: 118 };
    if (node.id === "dot_cell") return { ...node, w: 188, h: 94 };
    if (node.id === "dot_scalar") return { ...node, w: 176, h: 86 };
  }

  if (phase === "matmul_token_projection") {
    if (node.id === "token_vector" || node.id === "token_output") return { ...node, w: 206, h: 116 };
    if (node.id === "weight_plate") return { ...node, w: 210, h: 122 };
    if (node.id === "matmul_gate") return { ...node, w: 194, h: 98 };
  }

  if (phase === "matmul_sequence_projection") {
    if (node.id === "sequence_tensor" || node.id === "sequence_output") return { ...node, w: 220, h: 132 };
    if (node.id === "weight_plate") return { ...node, w: 206, h: 116 };
  }

  if (phase === "matmul_batch_projection") {
    if (node.id === "linear_input" || node.id === "linear_out") return { ...node, w: 224, h: 134 };
    if (node.id === "weight_plate") return { ...node, w: 206, h: 116 };
  }

  if (phase === "matmul_orientation_trap") {
    if (node.id === "weight_plate") return { ...node, w: 218, h: 124, subtitle: "stored [O,C] trap", shape: "[O,C]" };
    if (node.id === "transpose_switch") return { ...node, w: 218, h: 98 };
    if (node.id === "matmul_gate") return { ...node, w: 202, h: 100 };
  }

  if (phase === "matmul_linear_assembly" || phase === "matmul_gauntlet") {
    if (node.id === "linear_input" || node.id === "linear_out") return { ...node, w: 210, h: 126 };
    if (node.id === "weight_plate" || node.id === "transpose_switch" || node.id === "linear_module" || node.id === "reference_checker") return { ...node, w: 204, h: 96 };
    if (node.id === "matmul_tests") return { ...node, w: 210, h: 104 };
  }

  return node;
}

function sizeChapter01PhaseNode(node: TensorNode, phase: LevelPhase): TensorNode {
  if (phase === "rank_scanner") {
    if (node.id === "raw_objects") return { ...node, w: 178, h: 106 };
    if (node.id === "rank_scanner") return { ...node, w: 258, h: 154, subtitle: "scan axis count only" };
    if (node.id === "rank_gate") return { ...node, w: 176, h: 92 };
  }

  if (phase === "shape_caliper") {
    if (node.id === "hidden_tensor") return { ...node, w: 226, h: 142, subtitle: "rank 3 / axes unnamed" };
    if (node.id === "shape_caliper") return { ...node, w: 270, h: 190, subtitle: "axis length reader" };
    if (node.id === "shape_gate") return { ...node, w: 184, h: 96 };
  }

  if (phase === "semantic_gap") {
    if (node.id === "shape_gate") return { ...node, w: 196, h: 100 };
    if (node.id === "semantic_inspector") return { ...node, w: 338, h: 190, subtitle: "contract missing" };
  }

  if (phase === "token_grid_builder") {
    if (node.id === "text_batch") return { ...node, w: 156, h: 86 };
    if (node.id === "tokenizer") return { ...node, w: 184, h: 96 };
    if (node.id === "token_grid") return { ...node, w: 258, h: 172, subtitle: "build token_ids[B,T]" };
  }

  if (phase === "embedding_expansion") {
    if (node.id === "token_grid") return { ...node, w: 246, h: 150 };
    if (node.id === "embedding_table") return { ...node, w: 226, h: 112 };
    if (node.id === "embedding_lookup") return { ...node, w: 238, h: 156, subtitle: "lookup id -> C vector" };
    if (node.id === "hidden_tensor") return { ...node, w: 236, h: 150 };
  }

  if (phase === "hidden_contract_repair") {
    if (node.id === "hidden_tensor") return { ...node, w: 286, h: 184, subtitle: "probe axes / repair contract" };
  }

  if (phase === "consumer_validation" || phase === "hidden_test_gauntlet") {
    if (node.id === "hidden_tensor") return { ...node, w: 260, h: 164 };
    if (node.id === "batch_viewer" || node.id === "causal_mask" || node.id === "linear_probe") return { ...node, w: 202, h: 100 };
    if (node.id === "shape_tests") return { ...node, w: 190, h: 118 };
  }

  return node;
}

const manualConnectionEdgeIds = new Set([
  "e_01_text_tokenizer",
  "e_01_tokenizer_grid",
  "e_01_grid_embedding",
  "e_01_table_embedding",
  "e_01_embedding_hidden",
  "e_01_hidden_batch",
  "e_01_hidden_mask",
  "e_01_hidden_linear"
]);

function filterDisplayEdgesForPhase(level: BootcampLevel, edges: BootcampLevel["edges"], nodes: TensorNode[], phase: LevelPhase) {
  if (level.id === "0-2") {
    const challenge = chapter02ChallengeForPhase(phase);
    if (!challenge) return edges;
    const visibleNodeIds = new Set(nodes.map((node) => node.id));
    const visibleEdgeIds = new Set(challenge.edgeIds);
    return edges.filter((edge) => visibleEdgeIds.has(edge.id) && visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to));
  }
  if (level.id !== "0-1") return edges;
  const challenge = chapter01ChallengeForPhase(phase);
  if (!challenge) return edges;
  const visibleNodeIds = new Set(nodes.map((node) => node.id));
  const visibleEdgeIds = new Set(challenge.edgeIds);
  if (phase === "tensor_object") {
    const generatedEdges = edges.filter((edge) => edge.id.startsWith("e_01_object_tensor_") && visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to));
    const baseEdges = edges.filter(
      (edge) =>
        visibleEdgeIds.has(edge.id) &&
        edge.id !== "e_01_inspector_type" &&
        !manualConnectionEdgeIds.has(edge.id) &&
        visibleNodeIds.has(edge.from) &&
        visibleNodeIds.has(edge.to)
    );
    return [...baseEdges, ...generatedEdges];
  }
  return edges.filter((edge) => visibleEdgeIds.has(edge.id) && !manualConnectionEdgeIds.has(edge.id) && visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to));
}

function buildDisplayEdges(level: BootcampLevel, assignments: BootcampAnswerMap) {
  if (level.id !== "0-1") return level.edges;

  const slotDone = (slotId: string) => {
    const slot = level.repair.slots.find((item) => item.id === slotId);
    return slot ? slot.correctTagIds.includes(assignments[slotId]) : false;
  };

  const dataEdgeSlots: Record<string, string> = {
    e_01_text_tokenizer: "flow_text_tokenizer",
    e_01_tokenizer_grid: "flow_tokenizer_grid",
    e_01_grid_embedding: "flow_grid_embedding",
    e_01_table_embedding: "flow_table_embedding",
    e_01_embedding_hidden: "flow_embedding_hidden",
    e_01_hidden_batch: "consumer_b",
    e_01_hidden_mask: "consumer_t",
    e_01_hidden_linear: "consumer_c"
  };

  const edges = level.edges.map((edge) => {
    const dataSlotId = dataEdgeSlots[edge.id];
    if (dataSlotId) {
      const done = slotDone(dataSlotId);
      const contractEdge = dataSlotId.startsWith("consumer_");
      return {
        ...edge,
        label: done ? "connected" : "open port",
        color: done ? 0x22c55e : contractEdge ? 0xfbbf24 : 0x38bdf8
      };
    }
    return edge;
  });

  tensorObjectGeneratedSlotIds(assignments).forEach((slotId) => {
    const nodeId = tensorObjectGeneratedNodeId(slotId);
    if (!nodeId) return;
    const showcase = tensorObjectShowcases[slotId];
    edges.push({
      id: `e_01_object_tensor_${slotId}`,
      from: nodeId,
      to: "type_check",
      label: showcase ? `${showcase.kind} accepted` : "accepted",
      color: 0x22c55e,
      flow: "check",
      route: "direct"
    });
  });

  return edges;
}

function buildCanvasConnections(level: BootcampLevel, repairState: LevelRepairState, phase: LevelPhase): CanvasConnectionOverlay[] {
  if (level.id !== "0-1") return [];
  const assignment = (slotId: string) => repairState.assignments[slotId];
  const connection = (
    id: string,
    slotId: string,
    tagId: string,
    fromNodeId: string,
    toNodeId: string,
    fromLabel: string,
    toLabel: string,
    label: string,
    kind: CanvasConnectionOverlay["kind"],
    enabled: boolean
  ): CanvasConnectionOverlay => ({
    id,
    slotId,
    tagId,
    fromNodeId,
    toNodeId,
    fromLabel,
    toLabel,
    label,
    kind,
    state: assignment(slotId) === tagId ? "connected" : "open",
    enabled
  });

  const tokenGridEnabled = phase === "token_grid_builder";
  const embeddingEnabled = phase === "embedding_expansion";
  const consumerEnabled = phase === "consumer_validation";
  const tokenGridConnections = [
    connection("canvas_flow_text_tokenizer", "flow_text_tokenizer", "wire_text_tokenizer", "text_batch", "tokenizer", "utf8", "in", "utf8[B]", "data", tokenGridEnabled),
    connection("canvas_flow_tokenizer_grid", "flow_tokenizer_grid", "wire_tokenizer_grid", "tokenizer", "token_grid", "ids", "in", "int[B,T]", "data", tokenGridEnabled)
  ];
  const embeddingConnections = [
    connection("canvas_flow_grid_embedding", "flow_grid_embedding", "wire_grid_embedding", "token_grid", "embedding_lookup", "ids", "in", "token_ids[B,T]", "data", embeddingEnabled),
    connection("canvas_flow_table_embedding", "flow_table_embedding", "wire_table_embedding", "embedding_table", "embedding_lookup", "V,C", "table", "embedding_table[V,C]", "data", embeddingEnabled),
    connection("canvas_flow_embedding_hidden", "flow_embedding_hidden", "wire_embedding_hidden", "embedding_lookup", "hidden_tensor", "vec", "in", "float[B,T,C]", "data", embeddingEnabled)
  ];
  const consumerConnections = [
    connection("canvas_consumer_b", "consumer_b", "consumer_b", "hidden_tensor", "batch_viewer", "B", "in", "B -> samples", "contract", consumerEnabled),
    connection("canvas_consumer_t", "consumer_t", "consumer_t", "hidden_tensor", "causal_mask", "T", "in", "T -> mask", "contract", consumerEnabled),
    connection("canvas_consumer_c", "consumer_c", "consumer_c", "hidden_tensor", "linear_probe", "C", "in", "C -> linear", "contract", consumerEnabled)
  ];

  const showConnections = (enabled: boolean, connections: CanvasConnectionOverlay[]) => enabled || connections.some((item) => item.state === "connected");
  return [
    ...(showConnections(tokenGridEnabled, tokenGridConnections) ? tokenGridConnections : []),
    ...(showConnections(embeddingEnabled, embeddingConnections) ? embeddingConnections : []),
    ...(showConnections(consumerEnabled || phase === "hidden_test_gauntlet" || phase === "completed", consumerConnections) ? consumerConnections : [])
  ];
}

function buildCanvasActionHints(level: BootcampLevel, repairState: LevelRepairState, phase: LevelPhase): CanvasActionHint[] {
  if (level.id === "0-2") {
    const hints: CanvasActionHint[] =
      phase === "matmul_gauntlet"
        ? [
            {
              kind: "node",
              id: "matmul_tests",
              icon: "run",
              tooltip: "Use the repair console to run the final MatMul Gauntlet",
              pulse: true
            }
          ]
        : [];
    visibleCanvasSlotsForPhase(level, phase).forEach((slot) => {
      const filled = Boolean(repairState.assignments[slot.id]);
      hints.push({
        kind: "slot",
        id: slot.id,
        icon: "menu",
        tooltip: filled ? "Repair slot filled. Click to inspect or replace it from the console." : "Click this repair slot, then choose a matching tag.",
        pulse: !filled
      });
    });
    return hints;
  }

  if (level.id !== "0-1") return [];

  if (phase === "hidden_test_gauntlet") {
    return [
      {
        kind: "node",
        id: "shape_tests",
        icon: "run",
        tooltip: "Click or right-click: run contract tests",
        pulse: true
      }
    ];
  }

  const hints: CanvasActionHint[] = [];
  visibleCanvasSlotsForPhase(level, phase).forEach((slot) => {
    const category = level.repair.tags.find((tag) => tag.id === slot.correctTagIds[0])?.category;
    if (category === "data" || category === "consumer") return;
    const filled = Boolean(repairState.assignments[slot.id]);
    const observed = repairState.observations.some((observation) => observation.slotId === slot.id);
    const icon: CanvasActionHint["icon"] = phase === "hidden_contract_repair" ? "probe" : "menu";
    const tooltip =
      phase === "hidden_contract_repair"
        ? observed
          ? "Probe evidence recorded. Click again to review this axis."
          : filled
            ? "Click: collect probe evidence for this labeled axis"
            : "Click: probe this axis; drag B/T/C from the palette into this slot"
        : "Drag a palette variable into this slot";
    hints.push({
      kind: "slot",
      id: slot.id,
      icon,
      tooltip,
      pulse: phase === "hidden_contract_repair" ? !observed : !filled
    });
  });
  return hints;
}

function check(label: string, state: CheckState, detail: string) {
  return { label, state, detail };
}

function buildDisplayNodes(level: BootcampLevel, assignments: BootcampAnswerMap): TensorNode[] {
  const tagLabel = (slotId: string) => {
    const tag = level.repair.tags.find((item) => item.id === assignments[slotId]);
    return tag?.shortLabel ?? "?";
  };
  const nodes = level.nodes.map((node) => ({ ...node, checks: [...node.checks], sample: [...node.sample], stats: { ...node.stats } }));
  const patchNode = (id: string, patch: Partial<TensorNode>) => {
    const node = nodes.find((item) => item.id === id);
    if (node) Object.assign(node, patch);
  };

  if (level.id === "0-1") {
    const slotCorrect = (slotId: string) => {
      const slot = level.repair.slots.find((item) => item.id === slotId);
      return slot ? slot.correctTagIds.includes(assignments[slotId]) : false;
    };
    const slotsCorrect = (slotIds: string[]) => slotIds.every(slotCorrect);
    const objectsReady = slotsCorrect(chapter01SlotIds("tensor_object"));
    const ranksReady = slotsCorrect(chapter01SlotIds("rank_scanner"));
    const shapeReady = slotsCorrect(chapter01SlotIds("shape_caliper"));
    const semanticReady = slotsCorrect(chapter01SlotIds("semantic_gap"));
    const tokenReady = slotsCorrect(chapter01SlotIds("token_grid_builder"));
    const embeddingReady = slotsCorrect(chapter01SlotIds("embedding_expansion"));
    const axisReady = slotsCorrect(chapter01SlotIds("hidden_contract_repair"));
    const consumerReady = slotsCorrect(chapter01SlotIds("consumer_validation"));
    const semanticShape = `[${tagLabel("axis_0")},${tagLabel("axis_1")},${tagLabel("axis_2")}]`;
    const numericShape = shapeReady ? "[2,4,8]" : "[?,?,?]";
    const complete = objectsReady && ranksReady && shapeReady && semanticReady && tokenReady && embeddingReady && axisReady && consumerReady;
    patchNode("raw_objects", {
      subtitle: objectsReady ? "all inspected" : "tensor candidates",
      checks: [check("objects", objectsReady ? "pass" : "warn", objectsReady ? "scalar, vector, matrix, and 3D block are inspectable" : "inspect all four numeric containers")]
    });
    patchNode("tensor_inspector", {
      subtitle: objectsReady ? "4/4 accepted" : "accepts tensor-like objects",
      checks: [check("type acceptance", objectsReady ? "pass" : "warn", objectsReady ? "all numeric containers are tensor-like" : "send all four objects into the inspector")]
    });
    patchNode("type_check", {
      subtitle: objectsReady ? "passed" : "waiting for objects",
      checks: [check("accepted", objectsReady ? "pass" : "warn", objectsReady ? "object gate unlocked" : "all four objects must be inspected")]
    });
    nodes.push(...buildTensorObjectGeneratedNodes(assignments));
    patchNode("rank_scanner", {
      subtitle: ranksReady ? "rank stamps locked" : "axis detector",
      checks: [check("rank stamps", ranksReady ? "pass" : "warn", ranksReady ? "rank 0/1/2/3 match object axes" : "stamp scalar/vector/matrix/block with rank")]
    });
    patchNode("rank_gate", {
      subtitle: ranksReady ? "passed" : "waiting for rank",
      checks: [check("rank contract", ranksReady ? "pass" : "warn", ranksReady ? "rank is the number of axes" : "rank stamps are still incomplete")]
    });
    patchNode("shape_caliper", {
      subtitle: shapeReady ? "measured [2,4,8]" : "measure axis lengths",
      checks: [check("axis lengths", shapeReady ? "pass" : "warn", shapeReady ? "axis 0/1/2 lengths are 2/4/8" : "place length tags in axis order")]
    });
    patchNode("shape_gate", {
      shape: numericShape,
      subtitle: shapeReady ? "shape locked" : "shape slots open",
      checks: [check("shape order", shapeReady ? "pass" : "warn", shapeReady ? "shape is [2,4,8]" : "shape describes axis lengths only")]
    });
    patchNode("semantic_inspector", {
      subtitle: semanticReady ? "semantics marked missing" : "shape known / meaning missing",
      checks: [
        check(
          "semantic gap",
          semanticReady ? "pass" : "warn",
          semanticReady ? "shape [2,4,8] is not treated as [B,T,C]" : "mark axis semantics unresolved before assigning B/T/C"
        )
      ]
    });
    patchNode("tokenizer", {
      subtitle: assignments.flow_text_tokenizer === "wire_text_tokenizer" ? "text connected" : "input port open",
      checks: [
        check(
          "input line",
          assignments.flow_text_tokenizer === "wire_text_tokenizer" ? "pass" : "warn",
          assignments.flow_text_tokenizer === "wire_text_tokenizer" ? "utf8[B] reaches the tokenizer" : "connect Text Batch to Tokenizer"
        )
      ]
    });
    patchNode("token_grid", {
      shape: tokenReady ? "[B,T]" : "[?,?]",
      subtitle: tokenReady ? "token_ids[B,T]" : "build 2D token table",
      checks: [
        check("token ids", assignments.flow_tokenizer_grid === "wire_tokenizer_grid" ? "pass" : "warn", assignments.flow_tokenizer_grid === "wire_tokenizer_grid" ? "Tokenizer output fills the grid" : "connect Tokenizer to Token Grid"),
        check("B/T axes", slotCorrect("token_grid_b") && slotCorrect("token_grid_t") ? "pass" : "warn", slotCorrect("token_grid_b") && slotCorrect("token_grid_t") ? "rows are B, columns are T" : "label rows as B and columns as T"),
        check("slice drills", slotCorrect("token_task_sample1") && slotCorrect("token_task_t2") ? "pass" : "warn", slotCorrect("token_task_sample1") && slotCorrect("token_task_t2") ? "sample row and T2 column selections pass" : "complete B1 row and T2 column micro challenges")
      ]
    });
    patchNode("embedding_table", {
      subtitle: "trainable table[V,C]",
      checks: [check("table", "pass", "each row stores one C-wide token vector")]
    });
    patchNode("embedding_lookup", {
      subtitle: embeddingReady ? "expanded to C vectors" : "lookup inputs open",
      checks: [
        check("token ids", slotCorrect("flow_grid_embedding") ? "pass" : "warn", slotCorrect("flow_grid_embedding") ? "token_ids[B,T] indexes the table" : "connect Token Grid to Embedding Lookup"),
        check("embedding table", slotCorrect("flow_table_embedding") ? "pass" : "warn", slotCorrect("flow_table_embedding") ? "embedding_table[V,C] is available" : "connect Embedding Table to Lookup"),
        check("row probe", slotCorrect("embedding_probe") ? "pass" : "warn", slotCorrect("embedding_probe") ? "token_ids[0,2] maps to row 9172" : "inspect one token id through the table"),
        check("auto fill", slotCorrect("embedding_autofill") ? "pass" : "warn", slotCorrect("embedding_autofill") ? "every token cell expands into a vector" : "auto fill all token embeddings")
      ]
    });
    patchNode("hidden_tensor", {
      shape: embeddingReady ? (axisReady ? semanticShape : numericShape) : numericShape,
      subtitle: !embeddingReady ? "not fully generated" : axisReady ? "hidden[B,T,C]" : "generated / axes unknown",
      checks: [
        check("data input", embeddingReady ? "pass" : "warn", embeddingReady ? "embedding vectors generate hidden[2,4,8]" : "finish Embedding Expansion before axis labels can be trusted"),
        check(axisReady ? "axis labels assigned" : "contract incomplete", axisReady ? "pass" : "warn", axisReady ? `current contract hidden${semanticShape}` : "probe axes and assign B/T/C tags"),
        check("dtype", "pass", "float32 activations can enter Linear")
      ]
    });
    patchNode("batch_viewer", {
      subtitle: slotCorrect("consumer_b") ? "B connected" : "needs B",
      checks: [check("B consumer", slotCorrect("consumer_b") ? "pass" : "warn", slotCorrect("consumer_b") ? "independent samples are visible" : "connect Hidden.B to Batch Viewer")]
    });
    patchNode("causal_mask", {
      subtitle: slotCorrect("consumer_t") ? "T connected" : "needs T",
      checks: [check("T consumer", slotCorrect("consumer_t") ? "pass" : "warn", slotCorrect("consumer_t") ? "causal mask uses token order" : "connect Hidden.T to Causal Mask")]
    });
    patchNode("linear_probe", {
      subtitle: slotCorrect("consumer_c") ? "C connected" : "needs C",
      checks: [check("C consumer", slotCorrect("consumer_c") ? "pass" : "warn", slotCorrect("consumer_c") ? "linear input width consumes feature channels" : "connect Hidden.C to Linear Probe")]
    });
    patchNode("shape_tests", {
      subtitle: complete ? "ready to pass" : "blocked",
      checks: [check("gauntlet", complete ? "pass" : "warn", complete ? "visible, behavior, and hidden tests can run" : "finish all 0-1A through 0-1H repairs before hidden tests")]
    });
  }

  if (level.id === "0-2") {
    const slotCorrect = (slotId: string) => {
      const slot = level.repair.slots.find((item) => item.id === slotId);
      return slot ? slot.correctTagIds.includes(assignments[slotId]) : false;
    };
    const slotsCorrect = (slotIds: string[]) => slotIds.every(slotCorrect);
    const dotReady = slotsCorrect(chapter02SlotIds("matmul_dot_cell"));
    const tokenReady = slotsCorrect(chapter02SlotIds("matmul_token_projection"));
    const sequenceReady = slotsCorrect(chapter02SlotIds("matmul_sequence_projection"));
    const batchReady = slotsCorrect(chapter02SlotIds("matmul_batch_projection"));
    const orientationReady = slotsCorrect(chapter02SlotIds("matmul_orientation_trap"));
    const linearReady = slotsCorrect(chapter02SlotIds("matmul_linear_assembly"));
    const gauntletReady = slotsCorrect(chapter02SlotIds("matmul_gauntlet"));
    const orientationFix = assignments.orientation_fix;
    const computeWeight = tagLabel("compute_weight_co");
    const tokenOutput = tagLabel("token_output_o");
    const sequenceOutput = tagLabel("sequence_output_to");
    const batchOutput = tagLabel("batch_output_bto");

    patchNode("dot_cell", {
      subtitle: dotReady ? "dot contract locked" : "multiply and sum",
      checks: [check("dot contract", dotReady ? "pass" : "warn", dotReady ? "[C] dot [C] produces one scalar" : "match both vectors on C and confirm scalar output")]
    });
    patchNode("dot_scalar", {
      shape: slotCorrect("dot_output_scalar") ? "[]" : "?",
      subtitle: slotCorrect("dot_output_scalar") ? "scalar output" : "waiting for output role",
      checks: [check("rank 0", slotCorrect("dot_output_scalar") ? "pass" : "warn", slotCorrect("dot_output_scalar") ? "one cell is scalar" : "a single Dot Cell does not output an O vector")]
    });
    patchNode("weight_plate", {
      shape: orientationReady ? "[C,O] via T(W)" : assignments.stored_weight_orientation === "stored_oc" ? "[O,C]" : tokenReady || sequenceReady || batchReady ? "[C,O]" : "[?,?]",
      subtitle: orientationReady ? "compute view repaired" : assignments.stored_weight_orientation === "stored_oc" ? "stored orientation trap" : tokenReady || sequenceReady || batchReady ? "compute plate ready" : "C by O projection plate",
      checks: [
        check(
          "orientation",
          orientationReady || tokenReady || sequenceReady || batchReady ? "pass" : "warn",
          orientationReady ? "stored [O,C] is converted to compute [C,O]" : tokenReady || sequenceReady || batchReady ? "compute view is [C,O]" : "first axis must match consumed C"
        )
      ]
    });
    patchNode("matmul_gate", {
      shape: orientationReady
        ? `[B,T,C]@${computeWeight}`
        : batchReady
          ? "[B,T,C]@[C,O]"
          : sequenceReady
            ? "[T,C]@[C,O]"
            : tokenReady
              ? "[C]@[C,O]"
              : dotReady
                ? "[C] dot [C]"
                : "[?]@[?]",
      subtitle: orientationReady || batchReady || sequenceReady || tokenReady ? "inner dim locked" : "inner-dim latch",
      checks: [
        check(
          "inner dims",
          orientationReady || batchReady || sequenceReady || tokenReady ? "pass" : "warn",
          orientationReady ? "compute weight arrives as [C,O]" : batchReady ? "B/T carried and C consumed" : sequenceReady ? "T carried and C consumed" : tokenReady ? "single token projection is valid" : "C must meet C"
        )
      ]
    });
    patchNode("token_output", {
      shape: slotCorrect("token_output_o") ? "[O]" : `[${tokenOutput}]`,
      subtitle: tokenReady ? "token projection ready" : "projected feature vector"
    });
    patchNode("sequence_output", {
      shape: slotCorrect("sequence_output_to") ? "[T,O]" : sequenceOutput === "?" ? "[T,?]" : sequenceOutput,
      subtitle: sequenceReady ? "sequence projection ready" : "projected token row"
    });
    patchNode("linear_input", {
      subtitle: batchReady ? "B/T carriers confirmed" : "activation tensor",
      checks: [check("carrier axes", batchReady ? "pass" : "warn", batchReady ? "B and T are preserved while C is projected" : "identify B/T as carrier axes")]
    });
    patchNode("transpose_switch", {
      subtitle: orientationFix === "transpose_weight" ? "T(W) inserted" : orientationFix === "rotate_to_co" ? "manual rotation accepted" : "storage -> compute view",
      checks: [check("orientation fix", orientationReady ? "pass" : "warn", orientationReady ? "compute path exposes [C,O]" : "repair stored [O,C] before MatMul")]
    });
    patchNode("linear_module", {
      subtitle: linearReady ? "Linear assembled" : "MatMul + locked bias",
      checks: [
        check("core", slotCorrect("linear_core") ? "pass" : "warn", slotCorrect("linear_core") ? "MatMul core selected" : "select MatMul as Linear core"),
        check("bias", slotCorrect("bias_locked") ? "pass" : "warn", slotCorrect("bias_locked") ? "bias locked for 0-2" : "lock bias off for this stage"),
        check("reference", slotCorrect("reference_checker") ? "pass" : "warn", slotCorrect("reference_checker") ? "reference allclose enabled" : "attach reference checker")
      ]
    });
    patchNode("linear_out", {
      shape: slotCorrect("batch_output_bto") ? "[B,T,O]" : batchOutput === "?" ? "[B,T,?]" : batchOutput,
      subtitle: batchReady || linearReady ? "contract repaired" : "projected hidden"
    });
    patchNode("reference_checker", {
      subtitle: linearReady ? "allclose ready" : "waiting for Linear assembly",
      checks: [check("numeric check", linearReady ? "pass" : "warn", linearReady ? "referenceLinear allclose can run" : "shape correct is not enough")]
    });
    patchNode("matmul_tests", {
      subtitle: gauntletReady ? "ready to run" : "hidden cases",
      checks: [check("gauntlet", gauntletReady ? "pass" : "warn", gauntletReady ? "standard, storage, and size-trap cases armed" : "complete Case A/B/C hidden test setup")]
    });
  }

  if (level.id === "0-3") {
    const switchOn = assignments.k_switch === "switch_transpose";
    const swapOk = assignments.swap_axes === "swap_td";
    const scores = tagLabel("score_board");
    patchNode("transpose_k", {
      shape: switchOn && swapOk ? "[B,H,D,T]" : switchOn ? "wrong swap" : "off",
      subtitle: switchOn && swapOk ? "configured" : "not configured"
    });
    patchNode("qk_matmul", { shape: switchOn && swapOk ? "[T,D]@[D,T]" : "[T,D]@[?,?]", subtitle: switchOn && swapOk ? "inner dims locked" : "blocked" });
    patchNode("scores_tensor", { shape: scores === "T,T" ? "[B,H,T,T]" : "[B,H,?,?]", subtitle: scores === "T,T" ? "score board repaired" : "attention board" });
  }

  if (level.id === "0-4") {
    const pos = tagLabel("pos_rail");
    const bias = tagLabel("bias_rail");
    const rule = tagLabel("broadcast_rule_slot");
    const repaired = pos === "[T,C]" && bias === "[C]" && rule === "right";
    patchNode("broadcast_rule", {
      shape: repaired ? "[B,T,C]+[T,C]+[C]" : `[B,T,C]+${pos}+${bias}`,
      subtitle: repaired ? "rail aligned" : "not aligned"
    });
    patchNode("biased_out", { subtitle: repaired ? "broadcast repaired" : "broadcast sum" });
  }

  return nodes;
}

const tensorObjectNodePositions: Record<string, { x: number; y: number }> = {
  object_scalar: { x: 374, y: 596 },
  object_vector: { x: 616, y: 596 },
  object_matrix: { x: 374, y: 738 },
  object_block: { x: 616, y: 738 }
};

function tensorObjectGeneratedSlotIds(assignments: BootcampAnswerMap) {
  return ["object_scalar", "object_vector", "object_matrix", "object_block"].filter((slotId) => assignments[slotId] === slotId);
}

function tensorObjectGeneratedNodeId(slotId: string) {
  return tensorObjectShowcases[slotId] ? `generated_${slotId}` : undefined;
}

function tensorObjectSlotIdFromNodeId(nodeId: string) {
  if (!nodeId.startsWith("generated_")) return undefined;
  const slotId = nodeId.replace("generated_", "");
  return tensorObjectShowcases[slotId] ? slotId : undefined;
}

function buildTensorObjectGeneratedNodes(assignments: BootcampAnswerMap): TensorNode[] {
  const nodes: TensorNode[] = [];
  tensorObjectGeneratedSlotIds(assignments).forEach((slotId) => {
    const showcase = tensorObjectShowcases[slotId];
    const nodeId = tensorObjectGeneratedNodeId(slotId);
    const position = tensorObjectNodePositions[slotId];
    if (!showcase || !nodeId || !position) return;
    const nodeKind = showcase.kind === "scalar" ? "scalar" : showcase.kind === "matrix" ? "matrix" : "tensor";
    nodes.push({
      id: nodeId,
      title: showcase.title,
      subtitle: `${showcase.kind} instance`,
      kind: nodeKind,
      semanticName: showcase.kind,
      dtype: showcase.dtype.replace("torch.", ""),
      shape: tensorObjectShapeLabel(showcase.shape),
      source: "Tensor Inspector",
      consumer: "Type Check",
      stats: tensorObjectStats(showcase),
      sample: tensorObjectSample(showcase),
      checks: [check("type check", "pass", `${showcase.title} is accepted as a tensor object`)],
      x: position.x,
      y: position.y,
      w: showcase.kind === "block" ? 156 : 146,
      h: 82,
      color: showcase.kind === "scalar" ? 0x74491a : showcase.kind === "matrix" ? 0x24608a : 0x1f4b72
    });
  });
  return nodes;
}

function flattenTensorObjectValue(value: TensorObjectValue): number[] {
  if (typeof value === "number") return [value];
  return value.flatMap((item) => flattenTensorObjectValue(item));
}

function tensorObjectStats(showcase: TensorObjectShowcase): TensorNode["stats"] {
  const values = flattenTensorObjectValue(showcase.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    min: Number.isInteger(min) ? String(min) : min.toFixed(2),
    max: Number.isInteger(max) ? String(max) : max.toFixed(2),
    mean: Number.isInteger(mean) ? String(mean) : mean.toFixed(3)
  };
}

function tensorObjectShapeLabel(torchShape: string) {
  const match = torchShape.match(/^torch\.Size\((.*)\)$/);
  return match?.[1] ?? torchShape;
}

function tensorObjectSample(showcase: TensorObjectShowcase) {
  if (showcase.kind === "scalar") return ["x = 3.14", "rank 0 / no axis"];
  if (showcase.kind === "vector") return ["x = [0.2, -0.7, 1.4, 0.0]", "one axis / 4 values"];
  if (showcase.kind === "matrix") return ["x[0] = [1.0, 2.0, 3.0]", "x[1] = [4.0, 5.0, 6.0]"];
  return ["x[0,:,:] = [[1,2],[3,4]]", "x[1,:,:] = [[5,6],[7,8]]"];
}

function buildSlotOverlays(level: BootcampLevel, state: LevelRepairState, phase: LevelPhase): RepairSlotOverlay[] {
  return visibleCanvasSlotsForPhase(level, phase).map((slot) => {
    const tag = level.repair.tags.find((item) => item.id === state.assignments[slot.id]);
    return {
      slotId: slot.id,
      nodeId: slot.nodeId,
      label: slot.label,
      value: tag?.shortLabel ?? slot.emptyLabel,
      state: state.selectedSlotId === slot.id ? "active" : tag ? "filled" : "empty"
    };
  });
}

function buildInspectorTasks(level: BootcampLevel, state: LevelRepairState, phase: LevelPhase): InspectorTaskItem[] {
  const tasks: InspectorTaskItem[] = visibleSlotsForPhase(level, phase).map((slot) => {
    const assignedTagId = state.assignments[slot.id];
    const tag = level.repair.tags.find((item) => item.id === assignedTagId);
    const passed = Boolean(assignedTagId && slot.correctTagIds.includes(assignedTagId));
    const failed = Boolean(assignedTagId && !passed);
    return {
      id: slot.id,
      label: slot.label,
      value: passed ? "done" : tag?.shortLabel ?? slot.emptyLabel,
      state: passed ? "pass" : failed ? "fail" : "warn",
      active: state.selectedSlotId === slot.id
    };
  });

  if (level.id === "0-1" && phase === "hidden_contract_repair") {
    const probeCount = chapter01ProbeEvidenceCount(state.observations);
    tasks.push({
      id: "probe_evidence",
      label: "Probe Evidence",
      value: probeCount >= 3 ? "all axes observed" : `${probeCount}/3 axes observed`,
      state: probeCount >= 3 ? "pass" : "warn",
      active: false
    });
  }

  return tasks;
}

function activeToolLabel(level: BootcampLevel, state: LevelRepairState) {
  const tag = level.repair.tags.find((item) => item.id === state.activeTagId);
  if (tag) return `tag ${tag.shortLabel}`;
  const probe = level.repair.probes.find((item) => item.id === state.activeProbeId);
  if (probe) return probe.label;
  return "canvas context";
}

function repairKindLabel(kind: BootcampLevel["repair"]["kind"]) {
  if (kind === "axis_labels") return "Axis Tags";
  if (kind === "matmul_gate") return "Weight Plate";
  if (kind === "transpose_switch") return "Transpose Switch";
  return "Broadcast Rail";
}

function StatusPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="statusPill">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function ConceptBrief({
  level,
  repairState,
  phase,
  result
}: {
  level: BootcampLevel;
  repairState: LevelRepairState;
  phase: LevelPhase;
  result?: BootcampResult;
}) {
  return (
    <section className="briefingBox conceptBrief">
      <b>Concept Brief</b>
      {level.briefing.map((item) => (
        <p key={item}>{item}</p>
      ))}
      <div className="faultBox">
        <AlertTriangle size={16} />
        <span>{level.repair.brokenMessage}</span>
      </div>
      <ObjectiveTracker level={level} repairState={repairState} phase={phase} result={result} />
      <div className="budgetGrid">
        <span>Probe used</span>
        <b>
          {repairState.probeUses}/{level.repair.budget.probes}
        </b>
        <span>Reference runs</span>
        <b>
          {repairState.referenceRuns}/{level.repair.budget.referenceRuns}
        </b>
        <span>Hidden tests</span>
        <b>{level.repair.hiddenCases.length} cases</b>
      </div>
    </section>
  );
}

function ObjectiveTracker({
  level,
  repairState,
  phase,
  result
}: {
  level: BootcampLevel;
  repairState: LevelRepairState;
  phase: LevelPhase;
  result?: BootcampResult;
}) {
  const tested = Boolean(result);
  const objectives =
    supportsStageRail(level)
      ? challengesForLevel(level).map((challenge) => ({
          id: challenge.code,
          label: `${challenge.code} ${challenge.title}`,
          done:
            challenge.phase === "hidden_contract_repair"
              ? areSlotsCorrect(level, repairState, challenge.slotIds) && repairState.observations.length >= 3
              : challenge.phase === "hidden_test_gauntlet" || challenge.phase === "matmul_gauntlet"
                ? Boolean(result?.passed)
                : areSlotsCorrect(level, repairState, challenge.slotIds),
          active: phase === challenge.phase
        }))
      : [
          {
            id: "repair",
            label: level.repair.targetContract,
            done: Boolean(result?.passed),
            active: !result?.passed
          }
        ];

  return (
    <div className="objectiveBox">
      <div>
        <b>Current Objective</b>
        <small>{phaseLabel(phase)}</small>
      </div>
      <div className="objectiveList">
        {objectives.map((objective) => (
          <div key={objective.id} className={`objectiveItem ${objective.done ? "done" : ""} ${objective.active ? "active" : ""}`}>
            <StateIcon state={objective.done ? "pass" : objective.active || !tested ? "warn" : "fail"} />
            <span>{objective.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function BlueprintPalette({
  level,
  repairState,
  phase,
  onSelectTag,
  onSelectProbe
}: {
  level: BootcampLevel;
  repairState: LevelRepairState;
  phase: LevelPhase;
  onSelectTag: (tagId: string) => void;
  onSelectProbe: (probeId: string) => void;
}) {
  const tags = visiblePaletteTagsForPhase(level, phase);
  const probes = visibleProbesForPhase(level, phase);
  const lineSlots = visibleSlotsForPhase(level, phase).filter((slot) => {
    const category = level.repair.tags.find((tag) => tag.id === slot.correctTagIds[0])?.category;
    return category === "data" || category === "consumer";
  });

  function handleDragStart(event: ReactDragEvent<HTMLButtonElement>, tag: RepairTag) {
    event.dataTransfer.effectAllowed = "copy";
    event.dataTransfer.setData(repairTagDragMime, tag.id);
    event.dataTransfer.setData("text/plain", tag.id);
    onSelectTag(tag.id);
  }

  return (
    <div className="blueprintPalette">
      <div className="paletteHeader">
        <b>Blueprint Palette</b>
        <small>{phaseLabel(phase)}</small>
      </div>
      {probes.length ? (
        <div className="paletteToolGroup">
          <b>Probe Tools</b>
          <div className="paletteItems">
            {probes.map((probe) => (
              <button
                key={probe.id}
                className={`paletteItem probeTool ${repairState.activeProbeId === probe.id ? "active" : ""}`}
                title={probe.detail}
                onClick={() => onSelectProbe(probe.id)}
              >
                <span>Probe</span>
                <b>{probe.label.replace(" Probe", "")}</b>
                <small>{probe.detail}</small>
              </button>
            ))}
          </div>
        </div>
      ) : null}
      {tags.length ? (
        <div className="paletteToolGroup">
          <b>{probes.length ? "Variables" : "Variables / Components"}</b>
          <div className="paletteItems">
            {tags.map((tag) => {
              const used = Object.values(repairState.assignments).includes(tag.id);
              return (
                <button
                  key={tag.id}
                  className={`paletteItem ${tag.category ?? "operation"} ${repairState.activeTagId === tag.id ? "active" : ""} ${used ? "used" : ""}`}
                  draggable
                  title={tag.detail}
                  onClick={() => onSelectTag(tag.id)}
                  onDragStart={(event) => handleDragStart(event, tag)}
                >
                  <span>{paletteKindLabel(tag)}</span>
                  <b>{tag.shortLabel}</b>
                  <small>{tag.label}</small>
                </button>
              );
            })}
          </div>
        </div>
      ) : (
        <p className="paletteEmpty">当前阶段没有可拖变量；从发光输出端口拖线到目标输入端口。</p>
      )}
      {lineSlots.length ? (
        <div className="wireHintBox">
          <b>Port Links</b>
          {lineSlots.map((slot) => (
            <small key={slot.id}>{slot.label}</small>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function paletteKindLabel(tag: RepairTag) {
  if (tag.category === "object" || tag.category === "rank" || tag.category === "shape" || tag.category === "token" || tag.category === "axis") {
    return "Variable";
  }
  if (tag.category === "embedding") return "Component";
  return "Item";
}

function ChapterChallengeRail({
  level,
  repairState,
  phase,
  result
}: {
  level: BootcampLevel;
  repairState: LevelRepairState;
  phase: LevelPhase;
  result?: BootcampResult;
}) {
  const challenges = challengesForLevel(level);
  return (
    <div className="challengeRail">
      {challenges.map((challenge) => {
        const active = phase === challenge.phase;
        const done =
          challenge.phase === "hidden_contract_repair"
            ? areSlotsCorrect(level, repairState, challenge.slotIds) && chapter01ProbeEvidenceReady(repairState.observations)
            : challenge.phase === "hidden_test_gauntlet" || challenge.phase === "matmul_gauntlet"
              ? Boolean(result?.passed)
              : areSlotsCorrect(level, repairState, challenge.slotIds);
        return (
          <div key={challenge.code} className={`challengeStep ${active ? "active" : ""} ${done ? "done" : ""}`}>
            <StateIcon state={done ? "pass" : active ? "warn" : "warn"} />
            <span>
              <b>
                {challenge.code} {challenge.title}
              </b>
              <small>{challenge.brief}</small>
            </span>
          </div>
        );
      })}
    </div>
  );
}

function nextStageLabel(level: BootcampLevel, phase: LevelPhase | undefined) {
  const challenges = challengesForLevel(level);
  const index = challenges.findIndex((challenge) => challenge.phase === phase);
  const next = index >= 0 ? challenges[index + 1] : undefined;
  return next ? `下一步：${next.code} ${next.title}` : "下一步：运行最终验证。";
}

function StageDebriefPrompt({
  stageKnowledge,
  nextLabel,
  onOpen
}: {
  stageKnowledge?: Chapter01StageKnowledge;
  nextLabel: string;
  onOpen: () => void;
}) {
  if (!stageKnowledge) return null;

  return (
    <div className="stageDebriefPromptLayer" aria-live="polite">
      <button className="stageDebriefPrompt" onClick={onOpen}>
        <CheckCircle2 size={18} />
        <span>
          <b>{stageKnowledge.code} 完成</b>
          <small>{nextLabel}</small>
        </span>
        <code>查看小结</code>
      </button>
    </div>
  );
}

function StageDebriefOverlay({
  stageKnowledge,
  nextLabel,
  onCancel,
  onContinue
}: {
  stageKnowledge?: Chapter01StageKnowledge;
  nextLabel: string;
  onCancel: () => void;
  onContinue: () => void;
}) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onCancel();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  if (!stageKnowledge) return null;

  return (
    <div className="stageDebriefOverlay" role="dialog" aria-modal="true" aria-label="stage complete">
      <section className="stageDebriefCard">
        <CheckCircle2 size={24} />
        <p className="eyebrow">Stage Complete</p>
        <h2>{stageKnowledge.code} {stageKnowledge.title}</h2>
        <p>{stageKnowledge.debrief}</p>
        <small>{nextLabel}</small>
        <div className="stageDebriefActions">
          <button className="ghostButton" onClick={onCancel} aria-label="返回当前关卡，快捷键 Escape">
            返回当前关卡
            <small>Esc</small>
          </button>
          <button className="runButton modalAction" onClick={onContinue}>
            <Play size={16} />
            Continue
          </button>
        </div>
      </section>
    </div>
  );
}

function LevelCompletePrompt({
  level,
  result,
  nextLevel,
  onContinue
}: {
  level: BootcampLevel;
  result: BootcampResult;
  nextLevel?: BootcampLevel;
  onContinue: () => void;
}) {
  return (
    <div className="stageDebriefPromptLayer" aria-live="polite">
      <button className="stageDebriefPrompt levelCompletePrompt" onClick={onContinue}>
        <CheckCircle2 size={18} />
        <span>
          <b>{level.id} {level.title} 完成</b>
          <small>{nextLevel ? `进入 ${nextLevel.id} ${nextLevel.title}` : result.summary}</small>
        </span>
        <code>{nextLevel ? "下一关" : "完成"}</code>
      </button>
    </div>
  );
}

function StageIntroOverlay({ intro, actionLabel = "进入画布", onStart }: { intro: Chapter01StageIntro; actionLabel?: string; onStart: () => void }) {
  return (
    <div className="modalBackdrop stageIntroBackdrop" role="dialog" aria-modal="true" aria-label={`${intro.code} stage briefing`}>
      <section className="stageIntroPanel">
        <div className="stageIntroTop">
          <div>
            <p className="eyebrow">Stage Briefing</p>
            <h2>
              <span>{intro.code}</span>
              {intro.title}
            </h2>
          </div>
          <code>{intro.code}</code>
        </div>

        <div className="stageIntroGrid">
          <section className="stageIntroVisual" aria-label="concept visual">
            <div className="visualStage stageIntroSequence">
              {intro.visualLines.map((line, index) => (
                <code
                  key={`${intro.code}_${line}_${index}`}
                  className={index === intro.visualLines.length - 1 ? "active" : ""}
                  style={{ "--reveal-index": index } as CSSProperties}
                >
                  {line}
                </code>
              ))}
            </div>
          </section>

          <article className="stageIntroCopy">
            <p>{intro.body}</p>
            <div className="stageIntroTask">
              <span>Task</span>
              <b>{intro.taskPrompt}</b>
            </div>
          </article>
        </div>

        <div className="stageIntroActions">
          <button className="runButton modalAction" onClick={onStart}>
            <Play size={16} />
            {actionLabel}
          </button>
        </div>
      </section>
    </div>
  );
}

function CanvasContextMenu({
  target,
  actions,
  onClose
}: {
  target: CanvasContextTarget | null;
  actions: CanvasMenuAction[];
  onClose: () => void;
}) {
  const menuRef = useRef<HTMLDivElement | null>(null);
  const [position, setPosition] = useState({ left: 12, top: 12, maxHeight: 360 });
  const actionKey = actions.map((action) => action.id).join("|");

  useLayoutEffect(() => {
    if (!target) return;

    const margin = 12;
    const offset = 8;
    const menu = menuRef.current;
    const menuWidth = menu?.offsetWidth ?? Math.min(280, window.innerWidth - margin * 2);
    const menuHeight = menu?.offsetHeight ?? 360;
    let left = target.clientX + offset;
    let top = target.clientY + offset;

    if (left + menuWidth > window.innerWidth - margin) {
      left = Math.max(margin, target.clientX - menuWidth - offset);
    }
    if (top + menuHeight > window.innerHeight - margin) {
      top = Math.max(margin, target.clientY - menuHeight - offset);
    }

    setPosition({
      left,
      top,
      maxHeight: Math.max(180, window.innerHeight - top - margin)
    });
  }, [target?.clientX, target?.clientY, target?.kind, target?.id, actionKey]);

  if (!target || actions.length === 0) return null;

  const title = target.kind === "slot" ? "Repair Slot" : target.kind === "node" ? "Node" : "Canvas";
  return (
    <div ref={menuRef} className="canvasContextMenu" style={position} onContextMenu={(event) => event.preventDefault()}>
      <div className="canvasContextHeader">
        <b>{title}</b>
        <button type="button" aria-label="Close context menu" onClick={onClose}>
          x
        </button>
      </div>
      <div className="canvasContextActions">
        {actions.length ? (
          actions.map((action) => (
            <button key={action.id} type="button" disabled={action.disabled} onClick={action.onSelect}>
              <b>{action.label}</b>
              <span>{action.detail}</span>
            </button>
          ))
        ) : (
          <p>No canvas action available here.</p>
        )}
      </div>
    </div>
  );
}

function CanvasStatePanel({
  level,
  repairState,
  phase
}: {
  level: BootcampLevel;
  repairState: LevelRepairState;
  phase: LevelPhase;
}) {
  const slots = visibleSlotsForPhase(level, phase);
  return (
    <section className="canvasStatePanel">
      <div className="canvasStateSection">
        <b>Canvas Repairs</b>
        <div className="readonlySlotGrid">
          {slots.map((slot) => {
            const tag = level.repair.tags.find((item) => item.id === repairState.assignments[slot.id]);
            return (
              <div key={slot.id} className={tag ? "readonlySlot filled" : "readonlySlot"}>
                <span>
                  <b>{slot.label}</b>
                  <small>{slot.nodeId}</small>
                </span>
                <code>{tag?.shortLabel ?? slot.emptyLabel}</code>
              </div>
            );
          })}
        </div>
      </div>

      <div className="canvasStateSection">
        <b>Probe Evidence</b>
        {repairState.observations.length ? (
          <div className="probeLog readonlyProbeLog">
            {repairState.observations.slice(0, 4).map((observation) => (
              <div key={`${observation.id}_${observation.slotId}_${observation.probeId}`} className="probeLogItem">
                <span>
                  {observation.probeLabel} / {observation.slotLabel}
                </span>
                <b>{observation.title}</b>
                <small>
                  possible: <code>{observation.possibleSemantic}</code> confidence: {observation.confidence}
                </small>
              </div>
            ))}
          </div>
        ) : (
          <p className="canvasStateEmpty">No probe evidence recorded.</p>
        )}
      </div>
    </section>
  );
}

function TraceOverview({ steps }: { steps: BootcampLevel["traceSteps"] }) {
  return (
    <div className="traceSteps traceOverview">
      {steps.map((step) => (
        <div key={step.id} className={`traceStep ${step.state}`}>
          <StateIcon state={step.state} />
          <span>
            <b>{step.title}</b>
            <small>{step.detail}</small>
          </span>
        </div>
      ))}
    </div>
  );
}

function RepairConsole({
  level,
  repairState,
  phase,
  onSelectTag,
  onSelectProbe,
  onActivateSlot,
  onAssignTag
}: {
  level: BootcampLevel;
  repairState: LevelRepairState;
  phase: LevelPhase;
  onSelectTag: (tagId: string) => void;
  onSelectProbe: (probeId: string) => void;
  onActivateSlot: (slotId: string) => void;
  onAssignTag: (slot: RepairSlot, tagId: string) => void;
}) {
  const tags = visibleTagsForPhase(level, phase);
  const slots = visibleSlotsForPhase(level, phase);
  const probes = visibleProbesForPhase(level, phase);

  return (
    <section className="repairConsole">
      <div className="consoleSection">
        <div className="consoleTitle">
          <b>Available Repair Tags</b>
          <small>{phaseLabel(phase)}: click a tag, then click a slot; dragging works on slots below</small>
        </div>
        <div className="tagTray">
          {tags.map((tag) => {
            const used = Object.values(repairState.assignments).includes(tag.id);
            return (
              <button
                key={tag.id}
                draggable
                className={`repairTag ${tag.category ?? "operation"} ${repairState.activeTagId === tag.id ? "active" : ""} ${used ? "used" : ""}`}
                title={tag.detail}
                onClick={() => onSelectTag(tag.id)}
                onDragStart={(event) => event.dataTransfer.setData("text/plain", tag.id)}
              >
                <b>{tag.shortLabel}</b>
                <span>{tag.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="consoleSection">
        <div className="consoleTitle">
          <b>Probe Tools</b>
          <small>select a probe, then click a slot to collect evidence</small>
        </div>
        <div className="probeTray">
          {probes.length ? (
            probes.map((probe) => (
              <button
                key={probe.id}
                className={repairState.activeProbeId === probe.id ? "probeButton active" : "probeButton"}
                title={probe.detail}
                onClick={() => onSelectProbe(probe.id)}
              >
                <Search size={15} />
                <span>
                  <b>{probe.label}</b>
                  <small>{probe.detail}</small>
                </span>
              </button>
            ))
          ) : (
            <p className="consoleEmpty">先恢复数据流；Hidden Tensor 生成后探针才有观察对象。</p>
          )}
        </div>
      </div>

      <div className="consoleSection">
        <div className="consoleTitle">
          <b>Repair Slots</b>
          <small>target contract: {level.repair.targetContract}</small>
        </div>
        <div className="slotGrid">
          {slots.map((slot) => {
            const assignedTag = level.repair.tags.find((tag) => tag.id === repairState.assignments[slot.id]);
            return (
              <button
                key={slot.id}
                className={`repairSlot ${repairState.selectedSlotId === slot.id ? "active" : ""} ${assignedTag ? "filled" : ""}`}
                onClick={() => onActivateSlot(slot.id)}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  const tagId = event.dataTransfer.getData("text/plain");
                  if (tagId) onAssignTag(slot, tagId);
                }}
              >
                <span>
                  <b>{slot.label}</b>
                  <small>{slot.nodeId}</small>
                </span>
                <code>{assignedTag?.shortLabel ?? slot.emptyLabel}</code>
              </button>
            );
          })}
        </div>
      </div>

      <div className="probeLog">
        <b>Probe Evidence</b>
        {repairState.observations.length ? (
          repairState.observations.slice(0, 3).map((observation) => (
            <div key={`${observation.id}_${observation.slotId}_${observation.probeId}`} className="probeLogItem">
              <span>
                {observation.probeLabel} / {observation.slotLabel}
              </span>
              <b>{observation.title}</b>
              <small>
                possible: <code>{observation.possibleSemantic}</code> confidence: {observation.confidence}
              </small>
            </div>
          ))
        ) : (
          <p>选择 Probe 后点击槽位，Inspector 会显示观察证据。</p>
        )}
      </div>
    </section>
  );
}

function ResultPanel({ result, activeLevel }: { result?: BootcampResult; activeLevel: BootcampLevel }) {
  if (!result) {
    return (
      <section className="resultPanel pending">
        <AlertTriangle size={18} />
        <div>
          <b>Contract tests not run</b>
          <p>修复槽位后运行测试。系统会执行 visible checks、behavior checks 和 hidden tests。</p>
          <code>{activeLevel.repair.checks.length} checks pending</code>
        </div>
      </section>
    );
  }

  return (
    <section className={`resultPanel ${result.passed ? "pass" : "fail"}`}>
      <StateIcon state={result.passed ? "pass" : "fail"} />
      <div>
        <b>{result.errorType}</b>
        <p>{result.summary}</p>
        <div className="scoreStrip">
          <code>Rank {result.score.rank}</code>
          <code>Probe efficiency {result.score.probeEfficiency}</code>
          <code>
            Reference {result.score.referenceRuns}/{result.score.referenceBudget}
          </code>
        </div>
        <div className="resultChecks">
          {result.checks.map((item) => (
            <div key={item.id} className={`resultCheck ${item.state}`}>
              <StateIcon state={item.state} />
              <span>
                <b>
                  {item.group}: {item.title}
                </b>
                <small>{item.detail}</small>
                <small>
                  expected: <code>{item.expected}</code>
                </small>
                <small>
                  received: <code>{item.received}</code>
                </small>
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function KnowledgeIntroOverlay({ level, onContinue }: { level: BootcampLevel; onContinue: () => void }) {
  const [cardIndex, setCardIndex] = useState(0);
  const cards = level.knowledgeCards ?? [];
  const showingTransition = cardIndex >= cards.length;
  const card = cards[Math.min(cardIndex, Math.max(cards.length - 1, 0))];
  const progress = cards.length ? Math.min(cardIndex + 1, cards.length) : 0;
  const transition = level.knowledgeTransition;
  const visualLines = showingTransition ? [] : card.visual ?? [];

  if (!cards.length) return null;

  return (
    <div className="modalBackdrop" role="dialog" aria-modal="true" aria-label="knowledge intro">
      <section className="knowledgePanel">
        <div className="knowledgeTop">
          <div>
            <p className="eyebrow">Bootcamp Briefing</p>
            <h2>{showingTransition ? transition?.title ?? "Briefing Complete" : card.title}</h2>
          </div>
          <code>
            {showingTransition ? "ready" : `${progress}/${cards.length}`}
          </code>
        </div>

        <div className="knowledgeWorkbench">
          <section className={`knowledgeVisual ${showingTransition ? "complete" : ""}`}>
            {showingTransition ? (
              <>
                <b>Target Contract</b>
                <code>hidden[B,T,C]</code>
                <div className="conceptTable">
                  <span>Tensor</span>
                  <small>structured numbers</small>
                  <span>Rank</span>
                  <small>axis count</small>
                  <span>Shape</span>
                  <small>axis lengths</small>
                  <span>Contract</span>
                  <small>shape + semantics</small>
                </div>
              </>
            ) : (
              <>
                <div className="visualStage">
                  {visualLines.map((line, index) => (
                    <code
                      key={`${line}_${index}`}
                      className={index === visualLines.length - 1 ? "active" : ""}
                      style={{ "--reveal-index": index } as CSSProperties}
                    >
                      {line}
                    </code>
                  ))}
                </div>
              </>
            )}
          </section>

          <article className="knowledgeCard active">
            <span>{showingTransition ? "OK" : String(progress).padStart(2, "0")}</span>
            <p>{showingTransition ? transition?.body : card.body}</p>
          </article>
        </div>

        <div className="knowledgeActions">
          <span />
          <div className="knowledgeDots" aria-label="briefing progress">
            {cards.map((item, index) => (
              <button
                key={item.title}
                className={index === cardIndex ? "active" : ""}
                aria-label={`Open card ${index + 1}`}
                onClick={() => setCardIndex(index)}
              />
            ))}
          </div>
          {showingTransition ? (
            <button className="runButton modalAction" onClick={onContinue}>
              <Play size={16} />
              {transition?.buttonLabel ?? "Enter Workbench"}
            </button>
          ) : (
            <button className="runButton modalAction" onClick={() => setCardIndex((value) => value + 1)}>
              <Play size={16} />
              {cardIndex === cards.length - 1 ? "Finish Briefing" : "Next Card"}
            </button>
          )}
        </div>
      </section>
    </div>
  );
}

function MissionOverlay({ level, onStart }: { level: BootcampLevel; onStart: () => void }) {
  if (!level.mission) return null;

  return (
    <div className="modalBackdrop" role="dialog" aria-modal="true" aria-label="mission brief">
      <section className="missionModal">
        <p className="eyebrow">Repair Ticket</p>
        <h2>{level.mission.title}</h2>
        <p>{level.mission.body}</p>
        <div className="missionSuccess">
          <b>Success Conditions</b>
          {level.mission.success.map((item) => (
            <div key={item}>
              <CheckCircle2 size={15} />
              <span>{item}</span>
            </div>
          ))}
        </div>
        <button className="runButton modalAction" onClick={onStart}>
          <Wrench size={16} />
          Start Repair
        </button>
      </section>
    </div>
  );
}

function CompletionOverlay({
  level,
  result,
  nextLevel,
  onContinue
}: {
  level: BootcampLevel;
  result?: BootcampResult;
  nextLevel?: BootcampLevel;
  onContinue: () => void;
}) {
  if (!result) return null;

  return (
    <div className="modalBackdrop" role="dialog" aria-modal="true" aria-label="contract restored">
      <section className="completionModal">
        <CheckCircle2 size={28} />
        <p className="eyebrow">Contract Restored</p>
        <h2>{level.id} {level.title} accepted</h2>
        <p>{result.summary}</p>
        <div className="scoreStrip">
          <code>Rank {result.score.rank}</code>
          <code>Probe efficiency {result.score.probeEfficiency}</code>
          <code>
            Test run {result.score.referenceRuns}/{result.score.referenceBudget}
          </code>
        </div>
        <div className="unlockBox modalUnlocks">
          <b>Unlocked</b>
          <div>
            {level.unlocks.map((unlock) => (
              <span key={unlock}>{unlock}</span>
            ))}
          </div>
        </div>
        <button className="runButton modalAction" onClick={onContinue}>
          {nextLevel ? `进入 ${nextLevel.id}` : "Continue"}
        </button>
      </section>
    </div>
  );
}

function TensorInspector({
  node,
  observation,
  stageKnowledge,
  selectedSlotId,
  tensorObjectDetailSlotId,
  tensorRunOutputs,
  taskItems,
  result,
  onOpenTensorObjectDetail,
  onRunTensorObjectShowcase
}: {
  node: TensorNode;
  observation?: ObservationLogItem;
  stageKnowledge?: Chapter01StageKnowledge;
  selectedSlotId?: string;
  tensorObjectDetailSlotId?: string;
  tensorRunOutputs: Record<string, string>;
  taskItems: InspectorTaskItem[];
  result?: BootcampResult;
  onOpenTensorObjectDetail: (slotId: string) => void;
  onRunTensorObjectShowcase: (showcase: TensorObjectShowcase) => void;
}) {
  const failedCheck = result?.passed ? undefined : result?.checks.find((item) => item.state === "fail");
  const taskDoneCount = taskItems.filter((item) => item.state === "pass").length;
  const selectedObjectSlotId = tensorObjectSlotIdFromNodeId(node.id) ?? selectedSlotId ?? tensorObjectDetailSlotId;
  const tensorObjectShowcase = getTensorObjectShowcase(stageKnowledge, taskItems, selectedObjectSlotId);
  return (
    <aside className="panel inspector">
      <div className="panelHeader">
        <Eye size={18} />
        <h2>Tensor Inspector</h2>
      </div>
      <div className="inspectorTitle">
        <span className={`kindBadge ${node.kind}`}>{node.kind}</span>
        <h3>{node.title}</h3>
        <small>{node.subtitle}</small>
      </div>

      {taskItems.length ? (
        <section className="taskChecklist" aria-label="current stage tasks">
          <div className="taskChecklistHeader">
            <h3>Task List</h3>
            <code>
              {taskDoneCount}/{taskItems.length}
            </code>
          </div>
          <div className="taskRows">
            {taskItems.map((item) => (
              <label key={item.id} className={`taskRow ${item.state} ${item.active ? "active" : ""}`}>
                <input type="checkbox" checked={item.state === "pass"} readOnly />
                <span>
                  <b>{item.label}</b>
                  <small>{item.value}</small>
                </span>
              </label>
            ))}
          </div>
        </section>
      ) : null}

      {tensorObjectShowcase ? (
        <TensorObjectShowcasePanel
          showcase={tensorObjectShowcase}
          output={tensorRunOutputs[tensorObjectShowcase.id]}
          variant="sidebar"
          onRun={() => onRunTensorObjectShowcase(tensorObjectShowcase)}
          onOpen={() => onOpenTensorObjectDetail(tensorObjectShowcase.id)}
        />
      ) : null}

      <dl className="statGrid">
        <StatRow label="name" value={node.semanticName} />
        <StatRow label="shape" value={node.shape} />
        <StatRow label="dtype" value={node.dtype} />
        <StatRow label="min / max" value={`${node.stats.min} / ${node.stats.max}`} />
        <StatRow label="mean" value={node.stats.mean} />
        {node.stats.rowSum ? <StatRow label="row_sum" value={node.stats.rowSum} /> : null}
        <StatRow label="source" value={node.source} />
        <StatRow label="consumer" value={node.consumer} />
      </dl>

      {observation ? (
        <section className="evidenceBox">
          <h3>Probe Evidence</h3>
          <b>
            {observation.probeLabel} / {observation.slotLabel}
          </b>
          <p>{observation.detail}</p>
          <code>possible semantic: {observation.possibleSemantic}</code>
          <code>confidence: {observation.confidence}</code>
          {observation.evidence.map((line) => (
            <small key={line}>{line}</small>
          ))}
        </section>
      ) : null}

      {failedCheck && stageKnowledge ? (
        <section className="failureLesson">
          <h3>Failure Lesson</h3>
          <b>{failedCheck.title}</b>
          <p>{failedCheck.detail}</p>
          {stageKnowledge.failureLesson.map((line) => (
            <small key={line}>{line}</small>
          ))}
        </section>
      ) : null}

      <section className="sampleBox">
        <h3>Sample values</h3>
        {node.sample.map((line) => (
          <code key={line}>{line}</code>
        ))}
      </section>

      <section className="checkList">
        {node.checks.map((checkItem) => (
          <div key={checkItem.label} className={`checkRow ${checkItem.state}`}>
            <StateIcon state={checkItem.state} />
            <span>
              <b>{checkItem.label}</b>
              <small>{checkItem.detail}</small>
            </span>
          </div>
        ))}
      </section>
    </aside>
  );
}

function getTensorObjectShowcase(
  stageKnowledge: Chapter01StageKnowledge | undefined,
  taskItems: InspectorTaskItem[],
  selectedSlotId: string | undefined
): TensorObjectShowcase | undefined {
  if (stageKnowledge?.visual !== "tensor_objects") return undefined;
  const selectedTask = selectedSlotId ? taskItems.find((item) => item.id === selectedSlotId && item.state === "pass") : undefined;
  const fallbackTask = [...taskItems].reverse().find((item) => item.state === "pass");
  const task = selectedTask ?? fallbackTask;
  if (!task) return undefined;

  return tensorObjectShowcases[task.id];
}

const tensorObjectShowcases: Record<string, TensorObjectShowcase> = {
  object_scalar: {
    id: "object_scalar",
    kind: "scalar",
    title: "Scalar Tensor",
    shape: "torch.Size([])",
    dtype: "torch.float32",
    value: 3.14,
    code: `import torch

x = torch.tensor(3.14, dtype=torch.float32)

print(x)
print(x.shape)
print(x.dtype)`
  },
  object_vector: {
    id: "object_vector",
    kind: "vector",
    title: "Vector Tensor",
    shape: "torch.Size([4])",
    dtype: "torch.float32",
    value: [0.2, -0.7, 1.4, 0.0],
    code: `import torch

x = torch.tensor([0.2, -0.7, 1.4, 0.0], dtype=torch.float32)

print(x)
print(x.shape)
print(x.dtype)`
  },
  object_matrix: {
    id: "object_matrix",
    kind: "matrix",
    title: "Matrix Tensor",
    shape: "torch.Size([2, 3])",
    dtype: "torch.float32",
    value: [
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0]
    ],
    code: `import torch

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
], dtype=torch.float32)

print(x)
print(x.shape)
print(x.dtype)`
  },
  object_block: {
    id: "object_block",
    kind: "block",
    title: "Block Tensor",
    shape: "torch.Size([2, 2, 2])",
    dtype: "torch.float32",
    value: [
      [
        [1.0, 2.0],
        [3.0, 4.0]
      ],
      [
        [5.0, 6.0],
        [7.0, 8.0]
      ]
    ],
    code: `import torch

x = torch.tensor([
    [[1.0, 2.0], [3.0, 4.0]],
    [[5.0, 6.0], [7.0, 8.0]],
], dtype=torch.float32)

print(x)
print(x.shape)
print(x.dtype)`
  }
};

function TensorObjectShowcaseOverlay({
  showcase,
  output,
  onRun,
  onMinimize
}: {
  showcase: TensorObjectShowcase;
  output?: string;
  onRun: () => void;
  onMinimize: () => void;
}) {
  const [closing, setClosing] = useState(false);
  const closeTimerRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    return () => {
      if (closeTimerRef.current) {
        window.clearTimeout(closeTimerRef.current);
      }
    };
  }, []);

  const handleMinimize = () => {
    if (closing) return;
    setClosing(true);
    closeTimerRef.current = window.setTimeout(onMinimize, 220);
  };

  return (
    <div className={`modalBackdrop tensorRuntimeBackdrop ${closing ? "closing" : ""}`} role="dialog" aria-modal="true" aria-label={`${showcase.title} runtime`}>
      <section className="tensorRuntimeModal">
        <TensorObjectShowcasePanel showcase={showcase} output={output} variant="modal" onRun={onRun} onMinimize={handleMinimize} />
      </section>
    </div>
  );
}

function TensorObjectShowcasePanel({
  showcase,
  output,
  variant = "sidebar",
  onRun,
  onOpen,
  onMinimize
}: {
  showcase: TensorObjectShowcase;
  output?: string;
  variant?: "sidebar" | "modal";
  onRun?: () => void;
  onOpen?: () => void;
  onMinimize?: () => void;
}) {
  return (
    <section className={`tensorObjectShowcase ${variant}`}>
      <div className="tensorObjectShowcaseHeader">
        <div>
          <h3>{showcase.title}</h3>
          <code>{showcase.shape}</code>
        </div>
        <div className="tensorObjectShowcaseActions">
          {onOpen ? (
            <button className="showcaseIconButton" title="Open tensor detail" onClick={onOpen}>
              <Maximize2 size={14} />
            </button>
          ) : null}
          {onMinimize ? (
            <button className="showcaseIconButton" title="Minimize to inspector" onClick={onMinimize}>
              <Minimize2 size={14} />
            </button>
          ) : null}
        </div>
      </div>
      <TensorObjectVisual kind={showcase.kind} />
      <div className="tensorObjectMeta">
        <code>{showcase.dtype}</code>
      </div>
      <section className="tensorCodePanel">
        <div className="tensorCodeHeader">
          <b>Code</b>
          {onRun && variant === "modal" ? (
            <button className="showcaseRunButton" onClick={onRun}>
              <Play size={14} />
              Run
            </button>
          ) : null}
        </div>
        <pre className="tensorCodeBlock">
          <code>{showcase.code}</code>
        </pre>
      </section>
      <section className={`tensorOutputPanel ${output ? "ready" : ""}`}>
        <div>
          <b>Output</b>
          <small>{output ? "runtime result" : "press Run to execute"}</small>
        </div>
        <pre>{output ?? "waiting for run..."}</pre>
      </section>
    </section>
  );
}

function TensorObjectVisual({ kind }: { kind: TensorObjectShowcase["kind"] }) {
  if (kind === "scalar") {
    return (
      <div className="tensorObjectVisual scalar">
        <div className="tensorScalarOrb">3.14</div>
      </div>
    );
  }

  if (kind === "vector") {
    return (
      <div className="tensorObjectVisual vector">
        {[0.2, -0.7, 1.4, 0.0].map((value, index) => (
          <span key={`${value}_${index}`}>{value}</span>
        ))}
      </div>
    );
  }

  if (kind === "matrix") {
    return (
      <div className="tensorObjectVisual matrix">
        {[1, 2, 3, 4, 5, 6].map((value) => (
          <span key={value}>{value}</span>
        ))}
      </div>
    );
  }

  return (
    <div className="tensorObjectVisual block">
      {[0, 1, 2].map((plane) => (
        <div key={plane} className="tensorBlockPlane">
          {[0, 1, 2, 3].map((cell) => (
            <span key={cell} />
          ))}
        </div>
      ))}
    </div>
  );
}

function runTensorObjectProgram(showcase: TensorObjectShowcase) {
  const shape = inferTensorShape(showcase.value);
  return [formatTensorValue(showcase.value), `torch.Size(${formatShape(shape)})`, showcase.dtype].join("\n");
}

function inferTensorShape(value: TensorObjectValue): number[] {
  if (!Array.isArray(value)) return [];
  if (!value.length) return [0];
  return [value.length, ...inferTensorShape(value[0])];
}

function formatShape(shape: number[]) {
  return `[${shape.join(", ")}]`;
}

function formatTensorValue(value: TensorObjectValue, depth = 0): string {
  if (!Array.isArray(value)) return `tensor(${formatNumber(value)})`;
  return `tensor(${formatNestedTensorValue(value, depth)})`;
}

function formatNestedTensorValue(value: TensorObjectValue[], depth: number): string {
  if (!value.length) return "[]";
  if (value.every((item) => !Array.isArray(item))) {
    return `[${value.map((item) => formatNumber(item as number)).join(", ")}]`;
  }

  const indent = " ".repeat(depth * 8);
  const nextIndent = " ".repeat((depth + 1) * 8);
  return `[\n${nextIndent}${value.map((item) => (Array.isArray(item) ? formatNestedTensorValue(item, depth + 1) : formatNumber(item))).join(`,\n${nextIndent}`)}\n${indent}]`;
}

function formatNumber(value: number) {
  if (Object.is(value, -0)) return "0.0000";
  return value.toFixed(4);
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function StateIcon({ state }: { state: CheckState }) {
  if (state === "pass") return <CheckCircle2 size={16} />;
  if (state === "warn") return <AlertTriangle size={16} />;
  if (state === "fail") return <AlertTriangle size={16} />;
  return <Circle size={16} />;
}
