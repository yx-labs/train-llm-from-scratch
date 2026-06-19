import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type DragEvent as ReactDragEvent } from "react";
import { AlertTriangle, BookOpen, Boxes, CheckCircle2, ChevronDown, Circle, Cpu, Eye, Maximize2, Minimize2, Pause, Play, RotateCcw, Search, Wrench } from "lucide-react";
import { bootcampLevels, evaluateBootcampLevel } from "./bootcampLevels";
import {
  PixiWorkbenchCanvas,
  type CanvasActionHint,
  type CanvasConnectionOverlay,
  type CanvasContextTarget,
  type CanvasStageKnowledge,
  type CanvasTaskHint,
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

type BootcampProgressSave = {
  version: 1;
  selectedLevelId?: string;
  repairStates?: Record<string, LevelRepairState>;
  results?: Record<string, BootcampResult>;
  introSeen?: Record<string, boolean>;
  stageIntroSeen?: Record<string, Record<string, boolean>>;
  missionStarted?: Record<string, boolean>;
  completionDismissed?: Record<string, boolean>;
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
  | "transpose_matrix_flip"
  | "transpose_inner_dim_repair"
  | "transpose_higher_rank_axis_swap"
  | "transpose_single_head_qk"
  | "transpose_multi_head_trap"
  | "transpose_trap_debugger"
  | "transpose_gauntlet"
  | "broadcast_add_cell"
  | "broadcast_same_shape_add"
  | "broadcast_rule_lab"
  | "broadcast_bias_add"
  | "broadcast_position_add"
  | "broadcast_mask_add"
  | "broadcast_trap_debugger"
  | "broadcast_bonus_unsqueeze"
  | "broadcast_bonus_no_copy"
  | "broadcast_bonus_residual"
  | "broadcast_bonus_mask_value"
  | "broadcast_gauntlet"
  | "token_raw_text_object"
  | "token_type_gate_failure"
  | "token_tokenizer_socket"
  | "token_token_id_contract"
  | "token_boundary_cutter"
  | "token_split_comparison"
  | "token_count_meter"
  | "token_merge_forge"
  | "token_preserve_symbols"
  | "token_vocab_lookup"
  | "token_address_lighting"
  | "token_stable_id_test"
  | "token_buffer_build"
  | "token_oov_failure"
  | "token_fallback_splitter"
  | "token_special_token_injector"
  | "token_padding_builder"
  | "token_budget_gate"
  | "tokenizer_gauntlet"
  | "data_flow_repair"
  | "tensor_generated"
  | "axis_probe"
  | "axis_tagging"
  | "contract_wiring"
  | "visible_testing"
  | "hidden_testing"
  | "completed";

const repairTagDragMime = "application/x-llm-complete-repair-tag";
const progressStorageKey = "llm-complete:bootcamp-progress:v1";
const bootcampLevelTransitions: Partial<Record<string, string>> = {
  "0-1": "0-2",
  "0-2": "0-3",
  "0-3": "0-4",
  "0-4": "1-1"
};

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

const chapter03Challenges: Chapter01ChallengeDef[] = [
  {
    phase: "transpose_matrix_flip",
    code: "0-3A",
    title: "Matrix Flip",
    brief: "用 Transpose Switch 把 matrix[R,C] 变成 matrix^T[C,R]，并验证 A[i,j]=A^T[j,i]。",
    slotIds: ["matrix_flip_switch", "matrix_flip_shape", "matrix_flip_index"],
    nodeIds: ["matrix_plate", "transpose_2d", "matrix_t_plate", "index_checker"],
    edgeIds: ["e_03a_matrix_switch", "e_03a_switch_out", "e_03a_out_index"],
    tagIds: ["op_transpose_2d", "op_relabel_only", "shape_cr", "shape_rc", "index_mirror", "index_shuffle"],
    tagCategories: ["operation", "contract"]
  },
  {
    phase: "transpose_inner_dim_repair",
    code: "0-3B",
    title: "Inner-Dim Repair",
    brief: "识别 B[P,N] 的方向错误，转置 B 得到 [N,P]，让 A[M,N] @ B^T[N,P] 通过 reference。",
    slotIds: ["inner_dim_fault", "inner_dim_fix", "inner_dim_output", "inner_dim_numeric"],
    nodeIds: ["matmul_a", "matmul_b_bad", "transpose_b_repair", "matmul_repair_gate", "matmul_repair_out", "reference_checker_03"],
    edgeIds: ["e_03b_a_matmul", "e_03b_b_transpose", "e_03b_transpose_matmul", "e_03b_matmul_out", "e_03b_out_ref"],
    tagIds: ["b_orientation_fault", "a_orientation_fault", "transpose_b", "transpose_a", "shape_mp", "shape_np", "reference_allclose_03b"],
    tagCategories: ["operation", "contract"]
  },
  {
    phase: "transpose_higher_rank_axis_swap",
    code: "0-3C",
    title: "Higher-Rank Axis Swap",
    brief: "把 swap(-2,-1) 扩展到 [B,T,D] 和 [B,H,T,D]，证明 B/H 是携带轴。",
    slotIds: ["rank3_swap_axes", "rank3_carry_axis", "rank4_carry_axes", "no_full_reverse"],
    nodeIds: ["rank3_tensor", "rank4_tensor", "axis_swap_switch", "carry_axis_lock", "higher_rank_checker"],
    edgeIds: ["e_03c_rank3_switch", "e_03c_rank4_switch", "e_03c_switch_lock", "e_03c_lock_checker"],
    tagIds: ["swap_last_two", "reverse_all_axes", "preserve_b_03c", "preserve_bh_03c", "swap_ht_wrong", "not_full_reverse"],
    tagCategories: ["operation", "axis", "contract"]
  },
  {
    phase: "transpose_single_head_qk",
    code: "0-3D",
    title: "Single-Head QK Score",
    brief: "在单头 Q[T,D] 与 K[T,D] 之间插入 K^T，生成 token-to-token scores[T,T]。",
    slotIds: ["single_k_transpose", "single_score_shape", "single_score_semantics"],
    nodeIds: ["q_single", "k_single", "k_single_transpose", "single_qk_matmul", "single_scores"],
    edgeIds: ["e_03d_q_matmul", "e_03d_k_transpose", "e_03d_transpose_matmul", "e_03d_matmul_scores"],
    tagIds: ["transpose_k_single", "transpose_q_single", "scores_tt_single", "scores_dd_single", "query_key_axes", "feature_feature_axes"],
    tagCategories: ["operation", "axis", "contract"]
  },
  {
    phase: "transpose_multi_head_trap",
    code: "0-3E",
    title: "Multi-Head Transpose Trap",
    brief: "对 K[B,H,T,D] 只交换最后两轴，不移动 B/H，得到 scores[B,H,T,T]。",
    slotIds: ["multi_k_transpose", "multi_carry_axes", "multi_score_shape"],
    nodeIds: ["q_heads", "k_heads", "k_heads_transpose", "multi_qk_matmul", "multi_scores"],
    edgeIds: ["e_03e_q_matmul", "e_03e_k_transpose", "e_03e_transpose_matmul", "e_03e_matmul_scores"],
    tagIds: ["transpose_k_multi", "transpose_q_multi", "preserve_bh_03e", "move_bh_wrong", "scores_bhtt", "scores_bhdd"],
    tagCategories: ["operation", "axis", "contract"]
  },
  {
    phase: "transpose_trap_debugger",
    code: "0-3F",
    title: "Trap Debugger",
    brief: "调试操作数顺序、T==D 数值陷阱、轴语义错误和不完整 patch。",
    slotIds: ["debug_operand_order", "debug_numeric_trap", "debug_axis_contract", "debug_patch"],
    nodeIds: ["faulty_board", "trace_inspector", "cell_source_checker", "trap_patch", "trap_reference"],
    edgeIds: ["e_03f_fault_trace", "e_03f_trace_cell", "e_03f_cell_patch", "e_03f_patch_ref"],
    tagIds: ["debug_operand_order", "debug_shape_only", "debug_numeric_cell", "debug_axis_contract", "debug_patch_rules"],
    tagCategories: ["operation", "axis", "contract"]
  },
  {
    phase: "transpose_gauntlet",
    code: "0-3X",
    title: "Transpose Gauntlet",
    brief: "运行标准、携带轴、T==D 和操作数顺序隐藏测试，证明 QK^T 合同可泛化。",
    slotIds: ["gauntlet_standard", "gauntlet_carry", "gauntlet_td_equal", "gauntlet_operand_order"],
    nodeIds: ["gauntlet_cases", "qk_contract_terminal", "hidden_reference", "gauntlet_result"],
    edgeIds: ["e_03x_cases_contract", "e_03x_contract_ref", "e_03x_ref_result"],
    tagIds: ["case_standard", "case_carry_axes", "case_td_equal", "case_operand_order"],
    tagCategories: ["contract"]
  }
];

const chapter03ChallengeByPhase = new Map(chapter03Challenges.map((challenge) => [challenge.phase, challenge]));

const chapter04Challenges: Chapter01ChallengeDef[] = [
  {
    phase: "broadcast_add_cell",
    code: "0-4A",
    title: "Add Cell",
    brief: "连接两个 scalar cell，并用 Cell Trace 证明输出是 A + B。",
    slotIds: ["cell_left_input", "cell_right_input", "cell_trace_exact"],
    nodeIds: ["scalar_a", "scalar_b", "add_gate_cell", "output_cell", "cell_trace"],
    edgeIds: ["e_04a_a_gate", "e_04a_b_gate", "e_04a_gate_out", "e_04a_out_trace"],
    tagIds: ["wire_scalar_a", "wire_scalar_b", "manual_output", "cell_trace_exact"],
    tagCategories: ["data", "operation"]
  },
  {
    phase: "broadcast_same_shape_add",
    code: "0-4B",
    title: "Same-Shape Add",
    brief: "修复 [T,C] + [T,C]，确认输出 shape 不变且 cell 来源同坐标。",
    slotIds: ["same_left_shape", "same_right_shape", "same_output_shape", "same_trace_cell"],
    nodeIds: ["same_a", "same_b", "same_add_gate", "same_out", "same_trace"],
    edgeIds: ["e_04b_a_gate", "e_04b_b_gate", "e_04b_gate_out", "e_04b_out_trace"],
    tagIds: ["same_shape_tc", "same_shape_ct", "same_output_tc", "same_trace_cell"],
    tagCategories: ["contract", "operation"]
  },
  {
    phase: "broadcast_rule_lab",
    code: "0-4C",
    title: "Broadcast Rule Lab",
    brief: "完成 [C]、[T,C]、[1,T,1] 三类 broadcast plan 与 Ghost Expansion。",
    slotIds: ["rule_vector_c", "rule_matrix_tc", "rule_singleton", "rule_logical_view"],
    nodeIds: ["target_btc", "vector_c", "matrix_tc", "singleton_scale", "broadcast_rail_lab", "ghost_expansion", "plan_checker"],
    edgeIds: [
      "e_04c_target_rail",
      "e_04c_vector_rail",
      "e_04c_matrix_rail",
      "e_04c_singleton_rail",
      "e_04c_rail_ghost",
      "e_04c_rail_plan"
    ],
    tagIds: ["align_c_right", "align_c_to_t", "insert_missing_b", "insert_missing_c", "expand_singleton_bt", "logical_view", "materialized_repeat"],
    tagCategories: ["rail", "rank", "operation"]
  },
  {
    phase: "broadcast_bias_add",
    code: "0-4D",
    title: "Bias Add",
    brief: "把 bias[O] 对齐到 projected[B,T,O] 的 O 轴，并沿 B/T 广播。",
    slotIds: ["bias_axis_o", "bias_expand_bt", "bias_trace"],
    nodeIds: ["projected_bto", "bias_o", "bias_rail", "bias_add_gate", "bias_out", "bias_reference"],
    edgeIds: ["e_04d_projected_rail", "e_04d_bias_rail", "e_04d_rail_gate", "e_04d_gate_out", "e_04d_out_ref"],
    tagIds: ["axis_o", "axis_t", "expand_bt", "bias_cell_trace"],
    tagCategories: ["axis", "rank", "operation"]
  },
  {
    phase: "broadcast_position_add",
    code: "0-4E",
    title: "Position Add",
    brief: "把 pos_emb[T,C] 对齐 T/C，沿 B 广播，不能退化成 bias[C]。",
    slotIds: ["position_align_tc", "position_broadcast_b", "position_keep_t"],
    nodeIds: ["tok_emb", "pos_sheet", "position_rail", "position_add_gate", "hidden_out", "position_trace"],
    edgeIds: ["e_04e_tok_rail", "e_04e_pos_rail", "e_04e_rail_gate", "e_04e_gate_hidden", "e_04e_hidden_trace"],
    tagIds: ["align_tc", "align_c_only", "broadcast_b", "preserve_position_t"],
    tagCategories: ["axis", "rank", "semantic"]
  },
  {
    phase: "broadcast_mask_add",
    code: "0-4F",
    title: "Mask Add",
    brief: "把 mask[1,1,Tq,Tk] 加到 scores[B,H,Tq,Tk]，对齐 Tq/Tk 并沿 B/H 广播。",
    slotIds: ["mask_tqtk_axes", "mask_expand_bh", "mask_additive", "mask_trace"],
    nodeIds: ["scores_tensor", "mask_plate", "mask_rail", "mask_add_gate", "masked_scores", "illegal_cell_checker"],
    edgeIds: ["e_04f_scores_rail", "e_04f_mask_rail", "e_04f_rail_gate", "e_04f_gate_masked", "e_04f_masked_check"],
    tagIds: ["align_tqtk", "swap_tqtk", "expand_bh", "add_negative_mask", "multiply_mask", "mask_cell_trace"],
    tagCategories: ["axis", "rank", "operation"]
  },
  {
    phase: "broadcast_trap_debugger",
    code: "0-4G",
    title: "Broadcast Trap Debugger",
    brief: "调试右对齐、B/T 等尺寸陷阱、pos-as-bias 和 materialized repeat。",
    slotIds: ["trap_right_align", "trap_bt_swap", "trap_pos_bias", "trap_no_repeat"],
    nodeIds: ["trap_cases", "trap_inspector", "semantic_warning", "trace_probe_04", "trap_patch", "trap_reference_04"],
    edgeIds: ["e_04g_cases_inspector", "e_04g_inspector_warning", "e_04g_warning_trace", "e_04g_trace_patch", "e_04g_patch_ref"],
    tagIds: ["debug_right_align", "debug_bt_swap", "debug_pos_bias", "debug_no_repeat", "materialized_repeat"],
    tagCategories: ["operation", "semantic"]
  },
  {
    phase: "broadcast_bonus_unsqueeze",
    code: "Bonus A",
    title: "Unsqueeze Lab",
    brief: "插入长度为 1 的缺失轴，证明 [T,C] -> [1,T,C] 不改变值。",
    slotIds: ["bonus_unsqueeze_axis", "bonus_unsqueeze_values"],
    nodeIds: ["target_btc", "matrix_tc", "broadcast_rail_lab", "ghost_expansion", "plan_checker"],
    edgeIds: ["e_04c_target_rail", "e_04c_matrix_rail", "e_04c_rail_ghost", "e_04c_rail_plan"],
    tagIds: ["unsqueeze_leading_b", "unsqueeze_wrong_tail", "unsqueeze_keep_values"],
    tagCategories: ["rank", "semantic"]
  },
  {
    phase: "broadcast_bonus_no_copy",
    code: "Bonus B",
    title: "No-Copy Broadcast",
    brief: "用 logical view 和 source reuse 证明 broadcast 不是 materialized repeat。",
    slotIds: ["bonus_no_copy_view", "bonus_no_copy_stride"],
    nodeIds: ["target_btc", "matrix_tc", "broadcast_rail_lab", "ghost_expansion", "plan_checker"],
    edgeIds: ["e_04c_target_rail", "e_04c_matrix_rail", "e_04c_rail_ghost", "e_04c_rail_plan"],
    tagIds: ["view_no_copy", "repeat_full_copy", "stride_zero_view"],
    tagCategories: ["operation", "semantic"]
  },
  {
    phase: "broadcast_bonus_residual",
    code: "Bonus C",
    title: "Residual Add Preview",
    brief: "预览 Transformer Block residual add，确认 residual 分支应为同 shape 相加。",
    slotIds: ["bonus_residual_same", "bonus_residual_trace"],
    nodeIds: ["same_a", "same_b", "same_add_gate", "same_out", "same_trace"],
    edgeIds: ["e_04b_a_gate", "e_04b_b_gate", "e_04b_gate_out", "e_04b_out_trace"],
    tagIds: ["residual_same_shape", "residual_broadcast_wrong", "residual_preview_trace"],
    tagCategories: ["contract", "operation"]
  },
  {
    phase: "broadcast_bonus_mask_value",
    code: "Bonus D",
    title: "Mask Value Experiment",
    brief: "验证为什么 additive mask 要用很大的负数，而不是 0。",
    slotIds: ["bonus_mask_value", "bonus_mask_softmax"],
    nodeIds: ["scores_tensor", "mask_plate", "mask_rail", "mask_add_gate", "masked_scores", "illegal_cell_checker"],
    edgeIds: ["e_04f_scores_rail", "e_04f_mask_rail", "e_04f_rail_gate", "e_04f_gate_masked", "e_04f_masked_check"],
    tagIds: ["mask_value_large_negative", "mask_value_zero_wrong", "mask_softmax_probe"],
    tagCategories: ["operation", "semantic"]
  },
  {
    phase: "broadcast_gauntlet",
    code: "0-4X",
    title: "Broadcast Gauntlet",
    brief: "运行 bias、position、singleton、mask、equal-dim 和 semantic trace 隐藏测试。",
    slotIds: ["gauntlet_bias", "gauntlet_position", "gauntlet_singleton", "gauntlet_mask", "gauntlet_equal_dim", "gauntlet_semantic_trace"],
    nodeIds: ["broadcast_gauntlet_cases", "broadcast_contract_terminal", "hidden_broadcast_reference", "broadcast_gauntlet_result"],
    edgeIds: ["e_04x_cases_contract", "e_04x_contract_ref", "e_04x_ref_result"],
    tagIds: ["case_bias", "case_position", "case_singleton", "case_mask", "case_equal_dim", "case_semantic_trace"],
    tagCategories: ["contract"]
  }
];

const chapter04ChallengeByPhase = new Map(chapter04Challenges.map((challenge) => [challenge.phase, challenge]));

const chapter1TokenizationChallenges: Chapter01ChallengeDef[] = [
  {
    phase: "token_raw_text_object",
    code: "1-1A",
    title: "Raw Text Object",
    brief: "观察 raw text 是 utf8 文本对象，但不是 numeric tensor。",
    slotIds: ["c1_raw_text_object", "c1_text_inspector_open", "c1_not_numeric_tensor"],
    nodeIds: ["raw_text_input", "text_inspector", "type_gate"],
    edgeIds: ["e_c1_text_inspector", "e_c1_raw_type_gate"],
    tagIds: ["raw_text_is_utf8_object", "inspect_text_metadata", "raw_text_not_numeric_tensor", "raw_text_is_tensor_float"],
    tagCategories: ["object", "operation", "contract"]
  },
  {
    phase: "token_type_gate_failure",
    code: "1-1B",
    title: "Type Gate Failure",
    brief: "让 raw text 在模型数值端口前失败，明确 Embedding 需要 int ids。",
    slotIds: ["c1_type_gate_input", "c1_type_gate_failure", "c1_embedding_requires_ids"],
    nodeIds: ["raw_text_input", "type_gate", "raw_embedding_probe"],
    edgeIds: ["e_c1_raw_type_gate", "e_c1_type_embedding_reject"],
    tagIds: ["connect_raw_text_to_type_gate", "reject_string_for_embedding", "embedding_requires_int_ids", "send_raw_text_to_embedding"],
    tagCategories: ["data", "contract", "embedding"]
  },
  {
    phase: "token_tokenizer_socket",
    code: "1-1C",
    title: "Tokenizer Socket",
    brief: "插入 Tokenizer Socket，把 utf8 转成 ordered token pieces，并阻止 raw bypass。",
    slotIds: ["c1_tokenizer_socket", "c1_tokenizer_input_contract", "c1_raw_bypass_blocked"],
    nodeIds: ["raw_text_input", "tokenizer_socket", "token_piece_stream"],
    edgeIds: ["e_c1_text_tokenizer", "e_c1_tokenizer_pieces"],
    tagIds: ["insert_tokenizer_socket", "utf8_to_token_pieces", "block_raw_bypass", "token_pieces_are_ids"],
    tagCategories: ["operation", "token", "contract"]
  },
  {
    phase: "token_token_id_contract",
    code: "1-1D",
    title: "Token ID Contract",
    brief: "把 token pieces 继续变成 integer token ids，证明它们能 index embedding rows。",
    slotIds: ["c1_token_id_output", "c1_token_id_rank", "c1_embedding_id_contract"],
    nodeIds: ["token_piece_stream", "token_id_emitter", "embedding_lookup_probe"],
    edgeIds: ["e_c1_pieces_ids", "e_c1_ids_embedding"],
    tagIds: ["emit_int_token_ids", "token_ids_rank_1", "ids_can_index_embedding", "token_pieces_are_ids"],
    tagCategories: ["token", "rank", "embedding"]
  },
  {
    phase: "token_boundary_cutter",
    code: "1-2A",
    title: "Boundary Cutter",
    brief: "用空格和标点切出 pieces，同时保持源文本顺序。",
    slotIds: ["c1_boundary_spaces", "c1_boundary_punctuation", "c1_boundary_order"],
    nodeIds: ["raw_text_input", "boundary_cutter", "token_piece_stream"],
    edgeIds: ["e_c1_text_tokenizer", "e_c1_boundary_pieces"],
    tagIds: ["split_on_spaces", "split_punctuation", "preserve_piece_order", "drop_punctuation"],
    tagCategories: ["operation", "token", "semantic"]
  },
  {
    phase: "token_split_comparison",
    code: "1-2B",
    title: "Split Comparison",
    brief: "比较 char / word / subword 的 token cost 与 OOV 风险，选择 subword 主线。",
    slotIds: ["c1_compare_char_policy", "c1_compare_word_policy", "c1_compare_subword_policy"],
    nodeIds: ["boundary_cutter", "split_policy_switch", "token_count_meter"],
    edgeIds: ["e_c1_policy_meter"],
    tagIds: ["char_policy_many_tokens", "word_policy_oov_risk", "subword_policy_balanced", "choose_char_always"],
    tagCategories: ["token"]
  },
  {
    phase: "token_count_meter",
    code: "1-2C",
    title: "Token Count Meter",
    brief: "在 split / merge 后计算 T，并在超过可见预算 T<=8 时阻塞。",
    slotIds: ["c1_count_meter_input", "c1_count_budget", "c1_count_fail_fast"],
    nodeIds: ["split_policy_switch", "token_count_meter", "budget_gate"],
    edgeIds: ["e_c1_policy_meter", "e_c1_budget_buffer"],
    tagIds: ["count_tokens_after_split", "budget_t_le_8", "meter_blocks_overflow", "ignore_budget"],
    tagCategories: ["operation", "shape", "contract"]
  },
  {
    phase: "token_merge_forge",
    code: "1-2D",
    title: "Merge Forge",
    brief: "实现确定性 merge rules，例如 train+ing、tokenizer+s，并同步降低 token count。",
    slotIds: ["c1_merge_train_ing", "c1_merge_tokenizer_s", "c1_merge_count"],
    nodeIds: ["token_piece_stream", "merge_forge", "token_count_meter"],
    edgeIds: ["e_c1_merge_meter"],
    tagIds: ["merge_train_ing", "merge_tokenizer_s", "merge_reduces_count", "ignore_budget"],
    tagCategories: ["token", "operation"]
  },
  {
    phase: "token_preserve_symbols",
    code: "1-2E",
    title: "Preserve Symbols",
    brief: "保留标点、空格/词首调试信息，让 decode check 能抓出丢符号问题。",
    slotIds: ["c1_preserve_space_marker", "c1_preserve_punctuation", "c1_preserve_reversible_debug"],
    nodeIds: ["token_piece_stream", "symbol_keeper", "vocab_lookup_gate"],
    edgeIds: ["e_c1_symbols_lookup"],
    tagIds: ["preserve_space_marker", "preserve_symbols", "decode_debug_possible", "drop_punctuation"],
    tagCategories: ["token", "semantic"]
  },
  {
    phase: "token_vocab_lookup",
    code: "1-3A",
    title: "Vocab Lookup",
    brief: "让每个 piece 查 Toy Vocab，已知 piece 输出 row id，未知 piece 标为 OOV。",
    slotIds: ["c1_lookup_vocab_rows", "c1_lookup_unknown_flag", "c1_lookup_emit_ids"],
    nodeIds: ["token_piece_stream", "vocab_table", "vocab_lookup_gate", "token_id_emitter"],
    edgeIds: ["e_c1_vocab_lookup", "e_c1_lookup_ids"],
    tagIds: ["lookup_all_pieces", "flag_missing_vocab", "emit_vocab_ids", "skip_vocab_lookup"],
    tagCategories: ["operation", "contract", "token"]
  },
  {
    phase: "token_address_lighting",
    code: "1-3B",
    title: "Address Lighting",
    brief: "把 piece、vocab row 和 embedding row 连起来，明确 token id 就是行地址。",
    slotIds: ["c1_address_piece_row", "c1_address_id_equals_row", "c1_address_embedding_probe"],
    nodeIds: ["vocab_table", "vocab_lookup_gate", "address_lighting", "embedding_lookup_probe"],
    edgeIds: ["e_c1_lookup_address", "e_c1_address_embedding"],
    tagIds: ["piece_lights_row", "id_is_row_address", "embedding_row_probe", "random_id_allocator"],
    tagCategories: ["semantic", "contract", "embedding"]
  },
  {
    phase: "token_stable_id_test",
    code: "1-3C",
    title: "Stable ID Test",
    brief: "验证同一 piece 跨运行、跨 batch 始终映射到同一个 id，不允许运行时随机扩表。",
    slotIds: ["c1_stable_same_piece", "c1_stable_repeat_run", "c1_stable_no_random_vocab"],
    nodeIds: ["vocab_lookup_gate", "stable_id_checker", "token_buffer"],
    edgeIds: ["e_c1_stable_buffer"],
    tagIds: ["same_piece_same_id", "deterministic_run", "no_random_vocab", "random_id_allocator"],
    tagCategories: ["contract", "operation"]
  },
  {
    phase: "token_buffer_build",
    code: "1-3D",
    title: "Token Buffer Build",
    brief: "把多条文本的 token ids 整理为 rectangular int[B,T]，行是 B，列是 T。",
    slotIds: ["c1_buffer_axis_b", "c1_buffer_axis_t", "c1_buffer_rectangular"],
    nodeIds: ["token_id_emitter", "stable_id_checker", "token_buffer"],
    edgeIds: ["e_c1_stable_buffer"],
    tagIds: ["buffer_axis_b", "buffer_axis_t", "rectangular_token_buffer", "mask_all_ones"],
    tagCategories: ["axis", "shape"]
  },
  {
    phase: "token_oov_failure",
    code: "1-4A",
    title: "OOV Failure",
    brief: "让未知 piece 先以可诊断方式失败：检测、阻塞并报告具体 piece。",
    slotIds: ["c1_oov_detector", "c1_oov_block", "c1_oov_report"],
    nodeIds: ["vocab_lookup_gate", "oov_detector", "fallback_splitter"],
    edgeIds: ["e_c1_buffer_oov", "e_c1_oov_fallback"],
    tagIds: ["detect_oov", "block_unresolved_oov", "oov_report_piece", "skip_vocab_lookup"],
    tagCategories: ["operation", "contract", "semantic"]
  },
  {
    phase: "token_fallback_splitter",
    code: "1-4B",
    title: "Fallback Splitter",
    brief: "按 subword -> char -> <unk> 顺序把 OOV 变成确定性 ids。",
    slotIds: ["c1_fallback_subword", "c1_fallback_char", "c1_fallback_unk"],
    nodeIds: ["oov_detector", "fallback_splitter", "special_token_injector"],
    edgeIds: ["e_c1_oov_fallback", "e_c1_fallback_special"],
    tagIds: ["fallback_subword", "fallback_char", "fallback_unk", "random_id_allocator"],
    tagCategories: ["operation", "token"]
  },
  {
    phase: "token_special_token_injector",
    code: "1-4C",
    title: "Special Token Injector",
    brief: "注入 <bos>/<eos>，并保留 <pad>/<unk> 的稳定特殊 id。",
    slotIds: ["c1_special_bos", "c1_special_eos", "c1_special_reserved"],
    nodeIds: ["fallback_splitter", "special_token_injector", "padding_builder"],
    edgeIds: ["e_c1_fallback_special", "e_c1_special_padding"],
    tagIds: ["inject_bos", "inject_eos", "reserve_special_ids", "random_id_allocator"],
    tagCategories: ["token", "contract"]
  },
  {
    phase: "token_padding_builder",
    code: "1-4D",
    title: "Padding Builder",
    brief: "在 EOS 后右侧 padding，使用 pad id 0，并构建与 padding 对齐的 attention_mask。",
    slotIds: ["c1_padding_side", "c1_padding_id", "c1_mask_matches_pad"],
    nodeIds: ["special_token_injector", "padding_builder", "attention_mask_builder", "token_buffer"],
    edgeIds: ["e_c1_special_padding", "e_c1_padding_mask"],
    tagIds: ["right_padding", "pad_id_zero", "attention_mask_matches_pad", "left_padding", "mask_all_ones"],
    tagCategories: ["operation", "token", "mask"]
  },
  {
    phase: "token_budget_gate",
    code: "1-4E",
    title: "Token Budget Gate",
    brief: "统一处理超长文本：应用 max T，截断或拒绝，并输出可读 budget trace。",
    slotIds: ["c1_budget_policy", "c1_budget_truncate", "c1_budget_trace"],
    nodeIds: ["token_count_meter", "budget_gate", "token_buffer", "tokenizer_contract_terminal"],
    edgeIds: ["e_c1_budget_buffer", "e_c1_budget_gauntlet"],
    tagIds: ["apply_token_budget", "truncate_or_reject", "budget_trace", "ignore_budget"],
    tagCategories: ["contract", "operation", "semantic"]
  },
  {
    phase: "tokenizer_gauntlet",
    code: "1-X",
    title: "Tokenizer Gauntlet",
    brief: "运行 visible/hidden 文本集，验证 type、pieces、vocab、fallback、special、batch、mask、budget、determinism。",
    slotIds: [
      "c1_gauntlet_type_contract",
      "c1_gauntlet_piece_contract",
      "c1_gauntlet_vocab_contract",
      "c1_gauntlet_fallback_contract",
      "c1_gauntlet_special_contract",
      "c1_gauntlet_batch_contract",
      "c1_gauntlet_mask_contract",
      "c1_gauntlet_determinism_contract",
      "c1_gauntlet_budget_contract",
      "c1_gauntlet_embedding_contract"
    ],
    nodeIds: ["gauntlet_cases", "tokenizer_contract_terminal", "tokenizer_gauntlet_result", "embedding_lookup_probe"],
    edgeIds: ["e_c1_cases_terminal", "e_c1_gauntlet_result", "e_c1_ids_embedding"],
    tagIds: [
      "gauntlet_type_contract",
      "gauntlet_piece_contract",
      "gauntlet_vocab_contract",
      "gauntlet_fallback_contract",
      "gauntlet_special_contract",
      "gauntlet_batch_contract",
      "gauntlet_mask_contract",
      "gauntlet_determinism_contract",
      "gauntlet_budget_contract",
      "gauntlet_embedding_ready",
      "ignore_budget",
      "mask_all_ones"
    ],
    tagCategories: ["contract", "token", "operation", "shape", "mask", "embedding"]
  }
];

const chapter1TokenizationChallengeByPhase = new Map(chapter1TokenizationChallenges.map((challenge) => [challenge.phase, challenge]));

const chapter1TokenizationChallengeGroups: Array<{
  code: string;
  title: string;
  brief: string;
  phases: LevelPhase[];
}> = [
  {
    code: "1-1",
    title: "Text Cannot Flow",
    brief: "raw text 不能直接进入模型，先建立 tokenizer 边界。",
    phases: ["token_raw_text_object", "token_type_gate_failure", "token_tokenizer_socket", "token_token_id_contract"]
  },
  {
    code: "1-2",
    title: "Token Split",
    brief: "处理切分策略、token count、merge 与符号保留。",
    phases: ["token_boundary_cutter", "token_split_comparison", "token_count_meter", "token_merge_forge", "token_preserve_symbols"]
  },
  {
    code: "1-3",
    title: "Vocab Address",
    brief: "把 piece 解析成稳定 vocab row id，并构建 token buffer。",
    phases: ["token_vocab_lookup", "token_address_lighting", "token_stable_id_test", "token_buffer_build"]
  },
  {
    code: "1-4",
    title: "Unknown & Buffer",
    brief: "补齐 OOV、fallback、special token、padding、mask 和 budget。",
    phases: ["token_oov_failure", "token_fallback_splitter", "token_special_token_injector", "token_padding_builder", "token_budget_gate"]
  },
  {
    code: "1-X",
    title: "Tokenizer Gauntlet",
    brief: "visible/hidden tokenizer contract 综合考核。",
    phases: ["tokenizer_gauntlet"]
  }
];

type CampaignChapterDef = {
  id: "chapter0" | "chapter1";
  code: string;
  title: string;
  subtitle: string;
  levelIds: string[];
};

const campaignChapterDefs: CampaignChapterDef[] = [
  {
    id: "chapter0",
    code: "Chapter 0",
    title: "Tensor Bootcamp",
    subtitle: "Tensor、Shape、Linear、Transpose、Broadcast 基础。",
    levelIds: ["0-1", "0-2", "0-3", "0-4"]
  },
  {
    id: "chapter1",
    code: "Chapter 1",
    title: "Text -> Token",
    subtitle: "Split、Vocab、OOV、Padding、Mask 与 Token Buffer。",
    levelIds: ["1-1"]
  }
];

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

const chapter03StageIntros: Partial<Record<LevelPhase, Chapter01StageIntro>> = {
  transpose_matrix_flip: {
    code: "0-3A",
    title: "Matrix Flip：二维转置先看索引",
    body: "Transpose 不是把 shape 标签改一下。\n\n对二维矩阵来说，它会交换行轴和列轴，并且每个值都要移动到镜像坐标。\n\n如果 A[i,j] 没有出现在 A^T[j,i]，那就不是有效转置。",
    visualLines: ["A[R,C]", "Transpose", "A^T[C,R]", "A[i,j] = A^T[j,i]"],
    taskPrompt: "配置 2D Transpose Switch，修复输出 shape，并通过 index mapping 检查。"
  },
  transpose_inner_dim_repair: {
    code: "0-3B",
    title: "Inner-Dim Repair：转置可以修复 MatMul 方向",
    body: "0-2 告诉我们 MatMul 的内维度必须对齐。\n\n如果 A 是 [M,N]，但 B 以 [P,N] 存储，直接相乘会让内维度不匹配。\n\n把 B 转置成 [N,P] 后，N 才能和 N 相遇。",
    visualLines: ["A[M,N]", "B[P,N]", "T(B) -> [N,P]", "A @ B^T -> [M,P]"],
    taskPrompt: "识别 B 的方向错误，转置 B，并让输出通过 shape 和 reference allclose。"
  },
  transpose_higher_rank_axis_swap: {
    code: "0-3C",
    title: "Higher-Rank Axis Swap：只交换最后两轴",
    body: "高阶 tensor 里有一些轴只是携带维度。\n\n在 QK^T 相关场景里，B 和 H 是并行层，真正要交换的是最后两轴 T 和 D。\n\nswap(-2,-1) 不是 reverse all axes。",
    visualLines: ["K[B,T,D] -> K[B,D,T]", "K[B,H,T,D] -> K[B,H,D,T]", "keep B/H", "swap T/D"],
    taskPrompt: "配置 last-two-axis swap，保护 B/H 携带轴，并明确拒绝全轴反转。"
  },
  transpose_single_head_qk: {
    code: "0-3D",
    title: "Single-Head QK Score：为什么输出是 T x T",
    body: "Q 和 K 都是一张 token vector 表。\n\nQ 的每一行是一个 query token，K 转置后每一列是一个 key token。\n\n所以 Q[T,D] @ K^T[D,T] 会生成 scores[T,T]。",
    visualLines: ["Q[T,D]", "K[T,D] -> K^T[D,T]", "scores[T,T]", "rows=query / cols=key"],
    taskPrompt: "把 K 转置后接入 QK MatMul，并标出 score board 的行列语义。"
  },
  transpose_multi_head_trap: {
    code: "0-3E",
    title: "Multi-Head Transpose Trap：真实 attention 的形状",
    body: "多头 attention 不是把所有轴都混在一起。\n\n每个 B/H 层里单独执行 [T,D] @ [D,T]。\n\n因此 K[B,H,T,D] 只交换最后两轴，输出 scores[B,H,T,T]。",
    visualLines: ["Q[B,H,T,D]", "K[B,H,T,D]", "K^T[B,H,D,T]", "scores[B,H,T,T]"],
    taskPrompt: "只转置 K 的 T/D 轴，保留 B/H，并修复多头 score stack。"
  },
  transpose_trap_debugger: {
    code: "0-3F",
    title: "Trap Debugger：shape 通过不代表正确",
    body: "不是所有 transpose bug 都会报 shape 错。\n\n当 T 和 D 相等时，错误线路也可能输出一个看起来正确的方阵。\n\n调试时先看轴合同，再检查一个 score cell 的来源。",
    visualLines: ["check operand order", "check axes", "trace scores[tq,tk]", "then allclose"],
    taskPrompt: "诊断操作数顺序、T==D 数值陷阱、轴语义错误，并记录完整 patch。"
  },
  transpose_gauntlet: {
    code: "0-3X",
    title: "Transpose Gauntlet：合同必须能泛化",
    body: "最终测试不会只复用可见形状。\n\n它会改变 B/H/T/D，包含 T==D 数值陷阱，并检查操作数顺序。\n\n正确策略是固定合同：Q[...,T,D] @ K[...,D,T] -> scores[...,T,T]。",
    visualLines: ["Case A: standard", "Case B: carry axes", "Case C: T==D", "Case D: operand order"],
    taskPrompt: "完成四个隐藏 case 槽位，然后运行最终 Transpose Gauntlet。"
  }
};

const chapter04StageIntros: Partial<Record<LevelPhase, Chapter01StageIntro>> = {
  broadcast_add_cell: {
    code: "0-4A",
    title: "Add Cell：加法的最小单位",
    body: "Add 的最小单位是一个 cell。\n\n两个 numeric cell 进入 Add Gate，输出 cell 必须由它们计算得到，而不是手动填写。",
    visualLines: ["A = 2.0", "B = 5.0", "A + B -> 7.0", "trace source cells"],
    taskPrompt: "连接两个 scalar 输入，运行 Cell Trace，证明输出来源。"
  },
  broadcast_same_shape_add: {
    code: "0-4B",
    title: "Same-Shape Add：同坐标相加",
    body: "如果两个 tensor shape 完全相同，Add Gate 会逐位置相加。\n\n输出 shape 不会改变，每个 out[t,c] 都来自同坐标的 A[t,c] 和 B[t,c]。",
    visualLines: ["A[T,C]", "+ B[T,C]", "-> Out[T,C]", "out[1,2] = A[1,2] + B[1,2]"],
    taskPrompt: "修复同形状加法，并追踪一个输出 cell。"
  },
  broadcast_rule_lab: {
    code: "0-4C",
    title: "Broadcast Rule Lab：缺失轴和 1 轴",
    body: "Broadcast 会把小张量对齐到目标 shape。\n\n缺失轴可看作长度 1；长度为 1 的轴可以逻辑扩展，但语义轴仍必须正确。",
    visualLines: ["[C] -> [1,1,C]", "[T,C] -> [1,T,C]", "[1,T,1] -> [B,T,C]", "logical view, no copy"],
    taskPrompt: "完成三个 broadcast plan，并使用 Ghost Expansion 表示逻辑视图。"
  },
  broadcast_bias_add: {
    code: "0-4D",
    title: "Bias Add：一条偏置加到每个 token",
    body: "Linear bias 属于输出 feature 轴。\n\nprojected[B,T,O] + bias[O] 会把 bias 逻辑扩展成 [1,1,O]，沿 B 和 T 复用。",
    visualLines: ["projected[B,T,O]", "+ bias[O]", "view [1,1,O]", "out[b,t,o] += bias[o]"],
    taskPrompt: "把 Bias Strip 挂到 O 轴，并证明它沿 B/T 广播。"
  },
  broadcast_position_add: {
    code: "0-4E",
    title: "Position Add：位置表不是 bias",
    body: "pos_emb[T,C] 同时拥有 token 位置轴和 channel 轴。\n\n它只缺 B 轴，所以沿 batch 广播；如果丢掉 T，它就变成了错误的 bias[C]。",
    visualLines: ["tok_emb[B,T,C]", "+ pos_emb[T,C]", "view [1,T,C]", "keep T variation"],
    taskPrompt: "对齐 pos_emb 的 T/C，并用 trace 证明位置信息没有丢失。"
  },
  broadcast_mask_add: {
    code: "0-4F",
    title: "Mask Add：把未来位置变成不可能",
    body: "Attention scores 有 query token 和 key token 两个 T 轴。\n\nmask[1,1,Tq,Tk] 沿 B/H 广播，并把 future key cell 加上很大的负数。",
    visualLines: ["scores[B,H,Tq,Tk]", "+ mask[1,1,Tq,Tk]", "Tq/Tk aligned", "future cell -> -1e9"],
    taskPrompt: "对齐 mask 的 Tq/Tk，沿 B/H 广播，并选择 additive mask。"
  },
  broadcast_trap_debugger: {
    code: "0-4G",
    title: "Broadcast Trap Debugger：shape 通过也可能错",
    body: "可 broadcast 不等于语义正确。\n\n等尺寸轴、pos-as-bias 和真实 repeat 都会让表面测试变得危险。Cell Trace 是最后判断依据。",
    visualLines: ["rank", "axis alignment", "semantic warning", "cell trace", "no repeat"],
    taskPrompt: "按调试顺序修复三类 broadcast trap。"
  },
  broadcast_bonus_unsqueeze: {
    code: "Bonus A",
    title: "Unsqueeze Lab：插入长度为 1 的轴",
    body: "Unsqueeze 不改变 tensor 的值。\n\n它只是插入一个长度为 1 的轴，让 [T,C] 可以作为 [1,T,C] 参与后续 broadcast。",
    visualLines: ["pos[T,C]", "unsqueeze B", "view[1,T,C]", "values unchanged"],
    taskPrompt: "选择正确的 singleton 轴插入位置，并证明值没有被改写。"
  },
  broadcast_bonus_no_copy: {
    code: "Bonus B",
    title: "No-Copy Broadcast：逻辑扩展",
    body: "Broadcast view 会在访问时复用源值。\n\n它不应该把小 tensor 真实复制成完整目标 shape；否则数值可能对，但内存模型错。",
    visualLines: ["small storage", "logical view", "source reuse", "no repeat buffer"],
    taskPrompt: "标记 no-copy broadcast view，并证明扩展 cell 复用同一个来源。"
  },
  broadcast_bonus_residual: {
    code: "Bonus C",
    title: "Residual Add Preview：残差预览",
    body: "Transformer Block 里的 residual add 通常不依赖 broadcast。\n\n两条分支应当已经是相同 hidden shape，然后逐坐标相加。",
    visualLines: ["branch A[B,T,C]", "+ branch B[B,T,C]", "same-coordinate", "hidden[B,T,C]"],
    taskPrompt: "确认 residual 分支同 shape，并 trace 一个 residual 输出 cell。"
  },
  broadcast_bonus_mask_value: {
    code: "Bonus D",
    title: "Mask Value Experiment：为什么是很大的负数",
    body: "Causal mask 在 softmax 前加入 scores。\n\n非法位置加 0 不会被压制；加一个很大的负数，softmax 后概率才会接近 0。",
    visualLines: ["future score", "+ -1e9", "softmax", "prob ~= 0"],
    taskPrompt: "选择 large negative additive mask，并验证 softmax 后的非法位置概率。"
  },
  broadcast_gauntlet: {
    code: "0-4X",
    title: "Broadcast Gauntlet：泛化测试",
    body: "最终测试会更换 B/T/C/O/H 的尺寸，并加入 equal-dimension trap。\n\n你需要证明自己掌握的是 broadcast contract，而不是某个固定形状。",
    visualLines: ["bias", "position", "singleton", "mask", "equal dims", "semantic trace"],
    taskPrompt: "配置全部隐藏测试 case，运行最终 Broadcast Gauntlet。"
  }
};

const chapter1TokenizationStageIntros: Partial<Record<LevelPhase, Chapter01StageIntro>> = {
  token_raw_text_object: {
    code: "1-1A",
    title: "Raw Text Object：文字先只是对象",
    body: "Chapter 0 已经证明模型内部流动的是 tensor。\n\n这里先不要急着 tokenize，先确认 raw text 是 utf8 文本对象，而不是 numeric tensor。",
    visualLines: ["raw text", "dtype=utf8", "not numeric", "inspect first"],
    taskPrompt: "把 raw text、Text Inspector 和非数值合同放到正确槽位。"
  },
  token_type_gate_failure: {
    code: "1-1B",
    title: "Type Gate Failure：需要一次有信息量的失败",
    body: "Raw text 直接接 Embedding 不应该悄悄通过。\n\nType Gate 要明确报出：模型数值端口需要 integer token ids。",
    visualLines: ["raw utf8", "Type Gate", "expected int ids", "reject string"],
    taskPrompt: "让 Type Gate 暴露 raw text 失败原因，并标出 Embedding 的输入合同。"
  },
  token_tokenizer_socket: {
    code: "1-1C",
    title: "Tokenizer Socket：插入黑盒入口",
    body: "Tokenizer Socket 是 text side 和 model side 的边界。\n\n它把 utf8 text 转成 ordered token pieces，同时阻止 raw text 绕过。",
    visualLines: ["text", "Tokenizer Socket", "token pieces", "no bypass"],
    taskPrompt: "插入 tokenizer，声明 pieces 输出，并关闭 raw bypass。"
  },
  token_token_id_contract: {
    code: "1-1D",
    title: "Token ID Contract：pieces 还不是数字",
    body: "Token piece 仍然是字符串片段。\n\n只有经过 vocab lookup 后生成 integer token ids，Embedding 才能用它们查行。",
    visualLines: ["piece", "vocab row", "int id", "embedding row"],
    taskPrompt: "把 pieces 到 integer ids 的合同补完整。"
  },
  token_boundary_cutter: {
    code: "1-2A",
    title: "Boundary Cutter：切边界但不丢顺序",
    body: "切分不是随便拆字符串。\n\n空格、标点和原始顺序都必须能被 trace，否则后续 T 轴会失真。",
    visualLines: ["spaces", "punctuation", "ordered pieces", "T order"],
    taskPrompt: "配置空格、标点和顺序保留。"
  },
  token_split_comparison: {
    code: "1-2B",
    title: "Split Comparison：三种切法的代价",
    body: "Char split 覆盖强但 token 多；word split 紧凑但容易 OOV。\n\n本章主线采用 subword，因为它在预算和覆盖之间折中。",
    visualLines: ["char: high T", "word: OOV", "subword: balanced"],
    taskPrompt: "完成 char / word / subword 三个 policy 判断。"
  },
  token_count_meter: {
    code: "1-2C",
    title: "Token Count Meter：T 是资源约束",
    body: "每个 token 都占用 T 轴上的一个位置。\n\n所以 token count 必须在构建 buffer 前被测量和限制。",
    visualLines: ["pieces", "count T", "max T=8", "block overflow"],
    taskPrompt: "把计数来源、预算和超限阻塞接好。"
  },
  token_merge_forge: {
    code: "1-2D",
    title: "Merge Forge：频繁片段合并",
    body: "Merge rule 会把相邻 pieces 合成更常见的 token。\n\n它必须是确定性的，并且要同步改变 token count。",
    visualLines: ["train + ing", "tokenizer + s", "lower T", "deterministic"],
    taskPrompt: "配置两个 merge rule，并让 Token Count Meter 读取合并后的 T。"
  },
  token_preserve_symbols: {
    code: "1-2E",
    title: "Preserve Symbols：标点也是输入",
    body: "标点、空格和词首信息不能随手丢弃。\n\nDebug decode 需要它们来发现 tokenizer 是否改变了文本。",
    visualLines: ["punctuation", "space marker", "decode check", "no silent drop"],
    taskPrompt: "保留符号和可调试信息。"
  },
  token_vocab_lookup: {
    code: "1-3A",
    title: "Vocab Lookup：piece 找地址",
    body: "Vocab 是固定表：piece -> row id。\n\n找不到 row 的 piece 必须被标为 OOV，而不是随便造 id。",
    visualLines: ["piece", "vocab table", "row id", "OOV flag"],
    taskPrompt: "完成全量 lookup、missing flag 和 id 输出。"
  },
  token_address_lighting: {
    code: "1-3B",
    title: "Address Lighting：id 就是行地址",
    body: "Token id 的意义是“第几行”。\n\n同一个 id 会点亮 vocab row 和 embedding row，这就是后续 Embedding Lookup 的基础。",
    visualLines: ["token", "row id", "embedding row", "probe"],
    taskPrompt: "把 piece、row id 和 embedding row 连接起来。"
  },
  token_stable_id_test: {
    code: "1-3C",
    title: "Stable ID Test：不能运行时随机扩表",
    body: "同一个 piece 在任何运行中都必须得到同一个 id。\n\n如果 tokenizer 运行时造新 id，训练和推理会失去可复现性。",
    visualLines: ["run A", "run B", "same ids", "fixed vocab"],
    taskPrompt: "修复 same piece、repeat run 和 no random vocab 三个合同。"
  },
  token_buffer_build: {
    code: "1-3D",
    title: "Token Buffer Build：从序列到 batch tensor",
    body: "单条文本是 token_ids[T]；多条文本组成 token_ids[B,T]。\n\n行是样本 B，列是 token 位置 T。",
    visualLines: ["row=B", "col=T", "int[B,T]", "rectangular"],
    taskPrompt: "把 buffer 的 B/T 轴和 rectangular int tensor 合同补完整。"
  },
  token_oov_failure: {
    code: "1-4A",
    title: "OOV Failure：先让未知词正确失败",
    body: "OOV 不应该被吞掉。\n\n好的失败会检测未知 piece、阻止它进入 ids，并报告具体哪个 piece 失败。",
    visualLines: ["unknown piece", "detect", "block", "report"],
    taskPrompt: "配置 OOV detector 的检测、阻塞和报告。"
  },
  token_fallback_splitter: {
    code: "1-4B",
    title: "Fallback Splitter：把失败变成稳定输出",
    body: "OOV 之后先尝试更小 subword，再尝试 char，最后才落到 reserved <unk>。\n\n每一步都必须确定性。",
    visualLines: ["subword", "char", "<unk>", "stable id"],
    taskPrompt: "接好 subword / char / <unk> fallback 链。"
  },
  token_special_token_injector: {
    code: "1-4C",
    title: "Special Token Injector：序列边界也是 token",
    body: "<bos> 和 <eos> 显式标记序列边界。\n\n<pad> 和 <unk> 需要稳定保留 id，避免和普通词冲突。",
    visualLines: ["<bos>", "content", "<eos>", "<pad>/<unk> reserved"],
    taskPrompt: "注入 BOS/EOS，并保留特殊 token id。"
  },
  token_padding_builder: {
    code: "1-4D",
    title: "Padding Builder：batch 需要矩形",
    body: "不同文本长度不同，但 batch tensor 必须是矩形。\n\nPadding 填空位，attention_mask 告诉模型哪些位置是真内容。",
    visualLines: ["right pad", "pad id 0", "mask 1/0", "int[B,T]"],
    taskPrompt: "配置右 padding、pad id 和 attention mask。"
  },
  token_budget_gate: {
    code: "1-4E",
    title: "Token Budget Gate：超长必须处理",
    body: "超出 max T 的文本不能静默溢出。\n\n可以截断或拒绝，但必须输出 trace 告诉玩家预算消耗在哪里。",
    visualLines: ["max T", "truncate/reject", "trace pieces", "no overflow"],
    taskPrompt: "修复 token budget policy、overflow action 和 trace。"
  },
  tokenizer_gauntlet: {
    code: "1-X",
    title: "Tokenizer Gauntlet：完整合同考核",
    body: "最终考核会混合标点、OOV、padding、budget 和 determinism。\n\n目标不是记住 visible strings，而是证明 tokenizer contract 可以泛化。",
    visualLines: ["visible set", "hidden set", "reference encode", "determinism"],
    taskPrompt: "填满 10 个 gauntlet 合同槽位，然后运行最终测试。"
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

const chapter03StageKnowledge: Partial<Record<LevelPhase, Chapter01StageKnowledge>> = {
  transpose_matrix_flip: {
    code: "0-3A",
    title: "Matrix Flip",
    concept: "Transpose = swap axes + preserve mirrored values.",
    tool: "2D Transpose Switch",
    mission: "repair [R,C] -> [C,R] and A[i,j] -> A^T[j,i].",
    visual: "transpose_matrix_flip",
    inspectorNotes: ["二维 transpose 交换行轴和列轴。", "shape 通过后还要验证 index mapping。", "只改标签会被数值检查抓住。"],
    failureLesson: ["Matrix flip failed。", "检查输出 shape 是否是 [C,R]。", "确认 A[i,j] 的值移动到了 A^T[j,i]。"],
    debrief: "0-3A 完成：你已经证明 transpose 不是改标签，而是带有坐标映射的轴交换。"
  },
  transpose_inner_dim_repair: {
    code: "0-3B",
    title: "Inner-Dim Repair",
    concept: "A[M,N] needs right operand [N,P]; T(B) repairs stored [P,N].",
    tool: "Transpose B + MatMul Gate",
    mission: "repair B orientation, output [M,P], then allclose reference.",
    visual: "transpose_inner_dim",
    inspectorNotes: ["A 的最后一轴 N 已经在正确位置。", "B 存成 [P,N] 时需要转成 [N,P]。", "输出 [M,P] 还必须数值 allclose。"],
    failureLesson: ["Inner dim repair failed。", "故障在 B 的方向，不在 A。", "N 被 MatMul 消费，不能留在输出里。"],
    debrief: "0-3B 完成：你已经把 0-2 的内维度规则和 0-3A 的转置规则连接起来。"
  },
  transpose_higher_rank_axis_swap: {
    code: "0-3C",
    title: "Higher-Rank Axis Swap",
    concept: "swap(-2,-1) touches only T/D; prefix axes are carried.",
    tool: "Axis Swap Switch + Axis Lock",
    mission: "repair rank-3 and rank-4 tensors without moving B/H.",
    visual: "transpose_higher_rank",
    inspectorNotes: ["-1 是最后一轴 D。", "-2 是倒数第二轴 T。", "B/H 只是并行携带轴，不参与 swap。"],
    failureLesson: ["Higher-rank swap failed。", "不要 reverse all axes。", "检查 B/H 是否仍然在前缀位置。"],
    debrief: "0-3C 完成：你已经掌握高阶 tensor 中只交换最后两轴，同时保护携带轴。"
  },
  transpose_single_head_qk: {
    code: "0-3D",
    title: "Single-Head QK Score",
    concept: "Q[T,D] @ K^T[D,T] creates token-to-token scores[T,T].",
    tool: "K^T Switch + Score Board",
    mission: "transpose K, build [T,T], and label query/key axes.",
    visual: "transpose_single_qk",
    inspectorNotes: ["Q 的 token 轴成为 score 行。", "K 转置后的 token 轴成为 score 列。", "输出不是 feature-to-feature 的 D x D。"],
    failureLesson: ["Single-head QK failed。", "不要转置 Q。", "score board 的两个 T 分别是 query token 和 key token。"],
    debrief: "0-3D 完成：你已经从单头 Q/K 看清 QK^T 为什么输出 T x T。"
  },
  transpose_multi_head_trap: {
    code: "0-3E",
    title: "Multi-Head Transpose Trap",
    concept: "Inside every B/H layer: [T,D] @ [D,T] -> [T,T].",
    tool: "Multi-Head K^T Switch",
    mission: "only swap K last axes, preserve B/H, produce scores[B,H,T,T].",
    visual: "transpose_multi_qk",
    inspectorNotes: ["B 是 batch carry axis。", "H 是 head carry axis。", "每个 B/H 层里都有一个 T x T score board。"],
    failureLesson: ["Multi-head QK failed。", "检查是否移动了 B/H。", "K 应为 [B,H,D,T]，scores 应为 [B,H,T,T]。"],
    debrief: "0-3E 完成：你已经修复真实多头 QK^T 的方向陷阱。"
  },
  transpose_trap_debugger: {
    code: "0-3F",
    title: "Trap Debugger",
    concept: "Debug order: operand order -> axis contract -> cell source -> allclose.",
    tool: "Trace Inspector + Cell Source Checker",
    mission: "diagnose order, T==D numeric trap, axis contract, and patch rules.",
    visual: "transpose_debugger",
    inspectorNotes: ["Q 必须在左边。", "T==D 时 shape 会欺骗你。", "scores[tq,tk] 必须来自 q_tq dot k_tk。"],
    failureLesson: ["Trap debugger failed。", "不要只看输出 shape。", "按操作数、轴合同、单格来源、reference 的顺序排查。"],
    debrief: "0-3F 完成：你已经能调试 shape 通过但数值或语义错误的 transpose 陷阱。"
  },
  transpose_gauntlet: {
    code: "0-3X",
    title: "Transpose Gauntlet",
    concept: "QK^T contract generalizes across sizes, carry axes, and hidden traps.",
    tool: "Hidden Transpose Tests",
    mission: "pass standard, carry-axis, T==D, and operand-order cases.",
    visual: "transpose_gauntlet",
    inspectorNotes: ["Case A 检查标准形状。", "Case B 检查携带轴。", "Case C 检查 T==D 数值陷阱。", "Case D 检查操作数顺序。"],
    failureLesson: ["Hidden transpose case failed。", "回到 Q[...,T,D] @ K[...,D,T] -> scores[...,T,T]。", "不要用固定尺寸或输出 shape 猜答案。"],
    debrief: "0-3 完成：你已经掌握 QK^T 中 K 的 T/D 轴交换、B/H 携带轴和 T==D 数值陷阱。"
  }
};

const chapter04StageKnowledge: Partial<Record<LevelPhase, Chapter01StageKnowledge>> = {
  broadcast_add_cell: {
    code: "0-4A",
    title: "Add Cell",
    concept: "Elementwise Add 的最小证明是一个输出 cell 的来源。",
    tool: "Add Gate + Cell Trace",
    mission: "连接 A/B 两个 scalar，并 trace 出 7.0。",
    visual: "broadcast_add_cell",
    inspectorNotes: ["Add Gate 需要两个 numeric 输入。", "输出 cell 必须由输入 cell 计算得到。", "手填结果会被 reference 检查拒绝。"],
    failureLesson: ["Add Cell failed。", "检查两个输入是否都接入 Add Gate。", "用 Cell Trace 证明 out[] = A[] + B[]。"],
    debrief: "0-4A 完成：你已经把 Add 的本质落实到单个 cell 的来源证明。"
  },
  broadcast_same_shape_add: {
    code: "0-4B",
    title: "Same-Shape Add",
    concept: "同 shape 张量逐坐标相加，输出 shape 不改变。",
    tool: "Tensor Add Grid",
    mission: "修复 [T,C] + [T,C] -> [T,C]。",
    visual: "broadcast_same_shape",
    inspectorNotes: ["同 shape 输入可以直接逐元素对齐。", "T/C 轴顺序不能交换。", "Residual Add 通常就是 same-shape add。"],
    failureLesson: ["Same-shape add failed。", "检查两个输入是否都是 [T,C]。", "输出不应改变 shape 或交换坐标。"],
    debrief: "0-4B 完成：你已经从单个 cell 扩展到整张同 shape tensor 的逐元素加法。"
  },
  broadcast_rule_lab: {
    code: "0-4C",
    title: "Broadcast Rule Lab",
    concept: "缺失轴和 singleton 轴可以形成 logical broadcast view。",
    tool: "Broadcast Rail + Ghost Expansion",
    mission: "配置 [C]、[T,C]、[1,T,1] 三种 broadcast plan。",
    visual: "broadcast_rule",
    inspectorNotes: ["[C] 右对齐到 C。", "[T,C] 缺少的是 B。", "[1,T,1] 可沿 B/C 扩展。", "Ghost Expansion 表示逻辑视图，不是真复制。"],
    failureLesson: ["Broadcast plan failed。", "先补 rank，再检查轴长度，再看语义。", "不要把 C 对齐到 T，也不要真实 repeat。"],
    debrief: "0-4C 完成：你已经掌握缺失轴、长度 1 轴和 logical broadcast view。"
  },
  broadcast_bias_add: {
    code: "0-4D",
    title: "Bias Add",
    concept: "bias[O] 沿 B/T 广播，给每个输出通道加偏移。",
    tool: "Bias Strip + Broadcast Add Gate",
    mission: "把 bias[O] 对齐到 projected[B,T,O] 的 O 轴。",
    visual: "broadcast_bias",
    inspectorNotes: ["Bias 属于 output feature 轴 O。", "B 和 T 是广播轴。", "cell trace 应该读取 bias[o]。"],
    failureLesson: ["Bias alignment failed。", "Bias 不应对齐到 T 或 B。", "检查 out[b,t,o] 是否使用 bias[o]。"],
    debrief: "0-4D 完成：你已经把 0-2 中锁定的 Linear bias 加法补上了。"
  },
  broadcast_position_add: {
    code: "0-4E",
    title: "Position Add",
    concept: "pos_emb[T,C] 沿 B 广播，但必须保留 T 位置信息。",
    tool: "Position Sheet + Cell Trace",
    mission: "对齐 T/C，沿 B 广播，并拒绝 pos-as-bias。",
    visual: "broadcast_position",
    inspectorNotes: ["pos_emb 有一行对应每个 token 位置。", "缺失轴是 B，不是 T。", "如果只剩 [C]，位置差异就丢失了。"],
    failureLesson: ["Position add failed。", "检查 pos_emb 的 T 轴是否仍在。", "用 trace 确认 hidden[b,t,c] 使用 pos_emb[t,c]。"],
    debrief: "0-4E 完成：你已经理解 token embedding 如何通过 position embedding 获得位置信息。"
  },
  broadcast_mask_add: {
    code: "0-4F",
    title: "Mask Add",
    concept: "causal mask 通过加法把 future-token score 变成极小值。",
    tool: "Mask Plate + Illegal Cell Checker",
    mission: "对齐 Tq/Tk，沿 B/H 广播，并使用 additive negative mask。",
    visual: "broadcast_mask",
    inspectorNotes: ["scores 有 query token 和 key token 两个 T 轴。", "mask 的 B/H 是 singleton。", "乘法 mask 不是本阶段目标。"],
    failureLesson: ["Mask add failed。", "检查 Tq/Tk 是否交换。", "未来 cell 应该接收很大的负数，而不是被乘 0。"],
    debrief: "0-4F 完成：你已经把 0-3 的 QK score board 接到了 attention mask 前置步骤。"
  },
  broadcast_trap_debugger: {
    code: "0-4G",
    title: "Broadcast Trap Debugger",
    concept: "shape-valid broadcast 仍可能 semantic-invalid。",
    tool: "Semantic Warning Lens + Cell Trace Probe",
    mission: "识别 right-align、B/T swap、pos-as-bias 和 materialized repeat。",
    visual: "broadcast_debugger",
    inspectorNotes: ["先看 raw broadcast 是否可行。", "再看 axis semantics 是否匹配。", "最后用 cell trace 和 reference 判断。"],
    failureLesson: ["Broadcast trap missed。", "不要只看 shape。", "等尺寸轴必须靠语义和 cell source 判断。"],
    debrief: "0-4G 完成：你已经能调试可广播但语义错误的陷阱。"
  },
  broadcast_bonus_unsqueeze: {
    code: "Bonus A",
    title: "Unsqueeze Lab",
    concept: "Unsqueeze inserts a singleton axis; it does not edit values.",
    tool: "Axis Alignment Ruler",
    mission: "turn [T,C] into [1,T,C] and preserve source values.",
    visual: "broadcast_rule",
    inspectorNotes: ["缺失 B 轴应插到 T/C 前面。", "值不应被复制、重排或改写。", "view[0,t,c] 对应原 tensor[t,c]。"],
    failureLesson: ["Unsqueeze failed。", "重新确认 singleton 轴的位置。", "不要把 unsqueeze 当成 repeat 或 transpose。"],
    debrief: "Bonus A 完成：你已经掌握插入长度为 1 的轴来准备 broadcast。"
  },
  broadcast_bonus_no_copy: {
    code: "Bonus B",
    title: "No-Copy Broadcast",
    concept: "Broadcast view presents a larger shape while reusing source storage.",
    tool: "Ghost Expansion",
    mission: "prove logical broadcast view and source reuse instead of materialized repeat.",
    visual: "broadcast_rule",
    inspectorNotes: ["真实存储保持小 tensor。", "扩展 cell 通过 source mapping 复用来源。", "repeat 是效率警告。"],
    failureLesson: ["No-copy broadcast failed。", "把 materialized repeat 替换为 logical view。", "用 source reuse 证明没有全量复制。"],
    debrief: "Bonus B 完成：你已经理解 broadcast view 与真实 repeat 的区别。"
  },
  broadcast_bonus_residual: {
    code: "Bonus C",
    title: "Residual Add Preview",
    concept: "Residual add is normally same-shape elementwise addition.",
    tool: "Residual Add Preview",
    mission: "verify residual branches share [B,T,C] and trace one output cell.",
    visual: "broadcast_same_shape",
    inspectorNotes: ["Residual 分支应该同 shape。", "不要用 broadcast 掩盖分支 shape 错误。", "out[b,t,c] 同时来自两条分支。"],
    failureLesson: ["Residual preview failed。", "回到 same-shape add 的 cell contract。", "不要把 residual branch 当 bias strip。"],
    debrief: "Bonus C 完成：你已经把 same-shape add 连接到了 Transformer residual 场景。"
  },
  broadcast_bonus_mask_value: {
    code: "Bonus D",
    title: "Mask Value Experiment",
    concept: "A large negative additive mask makes illegal logits vanish after softmax.",
    tool: "Mask Value Probe",
    mission: "choose a large negative mask value and verify the softmax effect.",
    visual: "broadcast_mask",
    inspectorNotes: ["0 不会阻止 future key。", "-1e9 这类大负数会让概率接近 0。", "mask value 实验解释 0-4F 的数值选择。"],
    failureLesson: ["Mask value experiment failed。", "确认 mask 是 additive 且在 softmax 前。", "观察非法 cell 的 softmax 概率。"],
    debrief: "Bonus D 完成：你已经理解为什么 attention mask 使用很大的负数。"
  },
  broadcast_gauntlet: {
    code: "0-4X",
    title: "Broadcast Gauntlet",
    concept: "Broadcast contract 必须跨尺寸、场景和语义陷阱泛化。",
    tool: "Hidden Broadcast Tests",
    mission: "通过 bias、position、singleton、mask、equal-dim 和 semantic trace case。",
    visual: "broadcast_gauntlet",
    inspectorNotes: ["隐藏测试会改变 B/T/C/O/H。", "equal-dim trap 会让 shape 证据失效。", "semantic trace 是最终证明。"],
    failureLesson: ["Broadcast gauntlet failed。", "回到 rank -> alignment -> singleton -> semantics -> cell trace。", "不要依赖固定维度数字。"],
    debrief: "0-4 完成：你已经掌握 Broadcast Add 在 LLM 中的 bias、position、mask 和 semantic-debug 用法。"
  }
};

const chapter1TokenizationStageKnowledge: Partial<Record<LevelPhase, Chapter01StageKnowledge>> = {
  token_raw_text_object: {
    code: "1-1A",
    title: "Raw Text Object",
    concept: "Raw text 是 utf8 对象，不是 numeric tensor。",
    tool: "Text Inspector",
    mission: "确认 text 可被检查，但不能作为模型数值输入。",
    visual: "token_raw_text",
    inspectorNotes: ["Text 有长度、字符和字节。", "Text 没有 numeric shape。", "本阶段只建立对象与类型边界。"],
    failureLesson: ["Raw text object contract missing。", "不要把文字当成 float tensor。", "先用 Text Inspector 暴露 dtype。"],
    debrief: "Raw text 可以被检查，但它还不是模型能消费的数字。"
  },
  token_type_gate_failure: {
    code: "1-1B",
    title: "Type Gate Failure",
    concept: "失败也要有信息量：Embedding 需要 integer token ids。",
    tool: "Type Gate",
    mission: "让 raw text 在数值端口前失败，并标出 expected/received。",
    visual: "token_type_gate",
    inspectorNotes: ["expected: int64 token ids。", "received: utf8 string。", "这是正确的教学失败。"],
    failureLesson: ["Type Gate reason mismatch。", "如果 raw text 直接进 Embedding，就是绕过了 tokenizer。", "把失败原因标成 string rejected at numeric port。"],
    debrief: "你已经证明 raw text 不能直接进入模型侧数值端口。"
  },
  token_tokenizer_socket: {
    code: "1-1C",
    title: "Tokenizer Socket",
    concept: "Tokenizer 是 text side 与 model side 的边界。",
    tool: "Tokenizer Socket",
    mission: "插入 tokenizer，输出 ordered token pieces，并关闭 raw bypass。",
    visual: "token_split",
    inspectorNotes: ["Tokenizer 输出 pieces，不是最终 ids。", "所有 raw text 路径都必须经过 socket。", "pieces 保留文本顺序。"],
    failureLesson: ["Tokenizer Socket missing。", "raw bypass 会让模型侧收到字符串。", "把 tokenizer 放在 Text 和 Piece Stream 之间。"],
    debrief: "Tokenizer Socket 已成为文字进入模型前的唯一入口。"
  },
  token_token_id_contract: {
    code: "1-1D",
    title: "Token ID Contract",
    concept: "Pieces 经过 vocab lookup 后才成为 integer token ids。",
    tool: "Token ID Emitter",
    mission: "声明 token_ids[T] 并证明 ids 能 index embedding row。",
    visual: "token_vocab",
    inspectorNotes: ["Piece 是字符串片段。", "Token id 是整数 row address。", "Embedding Lookup 只接受 ids。"],
    failureLesson: ["Token id contract incomplete。", "不要把 pieces 直接当 ids。", "需要 Vocab Lookup 输出 integer row ids。"],
    debrief: "从单条文本到 token_ids[T] 的模型侧数字合同已经成立。"
  },
  token_boundary_cutter: {
    code: "1-2A",
    title: "Boundary Cutter",
    concept: "切分要保留边界、标点和顺序。",
    tool: "Boundary Cutter",
    mission: "按空格和标点切 pieces，并保持 T 顺序。",
    visual: "token_split",
    inspectorNotes: ["空格是候选边界。", "标点是可见输入，不能静默丢弃。", "pieces 的顺序就是之后 T 轴顺序。"],
    failureLesson: ["Boundary split failed。", "检查标点是否被丢弃。", "确认 pieces 没有重排。"],
    debrief: "你已经让文本变成有序 pieces，而不是一串不可控字符。"
  },
  token_split_comparison: {
    code: "1-2B",
    title: "Split Comparison",
    concept: "Char、word、subword 是成本与覆盖的取舍。",
    tool: "Split Policy Switch",
    mission: "比较三种策略，并选择 subword 作为主线策略。",
    visual: "token_split",
    inspectorNotes: ["Char 覆盖好但 T 很长。", "Word 紧凑但 OOV 多。", "Subword 是本章主线折中。"],
    failureLesson: ["Policy comparison wrong。", "不要因为 char 最稳就忽略预算。", "不要因为 word 最短就忽略 OOV。"],
    debrief: "你已经用工程约束选择了 subword，而不是凭直觉切词。"
  },
  token_count_meter: {
    code: "1-2C",
    title: "Token Count Meter",
    concept: "Token count 决定 T 轴长度，是资源约束。",
    tool: "Token Count Meter",
    mission: "在 split/merge 后计算 T，并阻塞超过 T<=8 的输入。",
    visual: "token_buffer",
    inspectorNotes: ["预算检查要发生在当前 pieces 上。", "T 超限会破坏后续 buffer。", "失败要早于 Embedding。"],
    failureLesson: ["Token budget missing。", "不要等 batch buffer 才发现溢出。", "把 max T 和 fail-fast 接到 meter。"],
    debrief: "你已经把 token 数变成可测试资源，而不是隐形副作用。"
  },
  token_merge_forge: {
    code: "1-2D",
    title: "Merge Forge",
    concept: "Merge rules 把常见相邻 pieces 合成稳定 token。",
    tool: "Merge Forge",
    mission: "配置 train+ing、tokenizer+s，并让 count 使用合并结果。",
    visual: "token_split",
    inspectorNotes: ["Merge rule 必须按固定顺序执行。", "Merge 后 T 会改变。", "Merge 不应打乱原始顺序。"],
    failureLesson: ["Merge rule failed。", "确认规则输入是相邻 pieces。", "确认 Token Count Meter 读取合并后列表。"],
    debrief: "你已经实现确定性 merge，并看到它如何降低 token budget 压力。"
  },
  token_preserve_symbols: {
    code: "1-2E",
    title: "Preserve Symbols",
    concept: "标点和空格信息是 tokenizer 可调试合同的一部分。",
    tool: "Symbol Keeper",
    mission: "保留符号、space marker 和 decode debug 所需信息。",
    visual: "token_split",
    inspectorNotes: ["标点可能影响语义和训练目标。", "space marker 能帮助定位词边界。", "decode check 可以发现静默丢符号。"],
    failureLesson: ["Symbols dropped。", "不要把标点当噪声直接删除。", "保留足够调试信息以便 decode check。"],
    debrief: "Token pieces 现在既能供模型使用，也能被人调试。"
  },
  token_vocab_lookup: {
    code: "1-3A",
    title: "Vocab Lookup",
    concept: "Vocab 是 piece 到稳定 row id 的固定映射。",
    tool: "Vocab Lookup Gate",
    mission: "每个 piece 都必须 lookup，missing piece 必须标 OOV。",
    visual: "token_vocab",
    inspectorNotes: ["已知 piece 输出 row id。", "未知 piece 不能随便造 id。", "OOV 是需要处理的中间状态。"],
    failureLesson: ["Vocab lookup incomplete。", "检查是否有 piece 绕过 lookup。", "找不到 row 时先 flag OOV。"],
    debrief: "Pieces 已经接上 Toy Vocab，模型侧开始获得稳定整数地址。"
  },
  token_address_lighting: {
    code: "1-3B",
    title: "Address Lighting",
    concept: "Token id 的意义是 vocab/embedding 表中的行地址。",
    tool: "Address Lighting",
    mission: "让 piece 点亮 vocab row，并用 probe 读 embedding row。",
    visual: "token_vocab",
    inspectorNotes: ["id 不是分数。", "id 是查表地址。", "Embedding row 与 vocab row 共享这个地址。"],
    failureLesson: ["Address contract failed。", "不要把 id 当随机编号。", "用 row lighting 证明 id=row。"],
    debrief: "你已经把 token id 和 Embedding Lookup 的查表行为连接起来。"
  },
  token_stable_id_test: {
    code: "1-3C",
    title: "Stable ID Test",
    concept: "同一 piece 必须跨运行映射到同一 id。",
    tool: "Stable ID Checker",
    mission: "验证 same piece、repeat run 和 no random vocab。",
    visual: "token_vocab",
    inspectorNotes: ["Vocab 在 encode 时固定。", "推理和训练必须复现同一 ids。", "运行时扩表会破坏 checkpoint。"],
    failureLesson: ["Stable id failed。", "检查是否运行时新建 id。", "同一 piece 必须查到同一 row。"],
    debrief: "Tokenizer 输出现在可复现，后续训练/推理可以共享同一 vocab。"
  },
  token_buffer_build: {
    code: "1-3D",
    title: "Token Buffer Build",
    concept: "多条文本需要 rectangular token_ids[B,T]。",
    tool: "Token Buffer",
    mission: "标出 B rows、T columns 和 int[B,T] rectangular buffer。",
    visual: "token_buffer",
    inspectorNotes: ["B 轴分离样本。", "T 轴保存每条样本内的 token 位置。", "Batch tensor 不能是 ragged list。"],
    failureLesson: ["Token Buffer shape failed。", "确认行是 B、列是 T。", "输出必须是整数矩形 buffer。"],
    debrief: "单条 token_ids[T] 已经升级为可批处理的 token_ids[B,T]。"
  },
  token_oov_failure: {
    code: "1-4A",
    title: "OOV Failure",
    concept: "未知 piece 要先明确失败，不能静默进入模型。",
    tool: "OOV Detector",
    mission: "检测 OOV，阻塞 unresolved piece，并报告具体 piece。",
    visual: "token_vocab",
    inspectorNotes: ["OOV 是 vocab lookup 的失败状态。", "未解决 OOV 不能输出普通 id。", "好的错误信息会指向具体 piece。"],
    failureLesson: ["OOV failure uninformative。", "检查 detector 是否拿到了 missing piece。", "报告中必须包含 unresolved token。"],
    debrief: "OOV 已经变成可定位、可处理的失败，而不是随机坏数据。"
  },
  token_fallback_splitter: {
    code: "1-4B",
    title: "Fallback Splitter",
    concept: "Fallback 把 OOV 转成确定性 token ids。",
    tool: "Fallback Splitter",
    mission: "按 subword -> char -> <unk> 处理 OOV。",
    visual: "token_vocab",
    inspectorNotes: ["Subword fallback 尽量保留信息。", "Char fallback 覆盖更强但更贵。", "<unk> 是最后的稳定保底。"],
    failureLesson: ["Fallback chain failed。", "不要随机分配新 id。", "用 reserved <unk> 结束无法解析的路径。"],
    debrief: "未知文本现在也能以确定性方式进入 token id 流。"
  },
  token_special_token_injector: {
    code: "1-4C",
    title: "Special Token Injector",
    concept: "BOS/EOS/PAD/UNK 是序列合同的一部分。",
    tool: "Special Token Injector",
    mission: "注入 BOS/EOS，并保留 PAD/UNK 的固定 id。",
    visual: "token_buffer",
    inspectorNotes: ["BOS 标记序列开始。", "EOS 标记内容结束。", "PAD/UNK 的 id 必须稳定保留。"],
    failureLesson: ["Special token contract failed。", "检查 BOS/EOS 的位置。", "不要让普通 token 占用 reserved ids。"],
    debrief: "序列边界和特殊状态现在都显式写进 token 流。"
  },
  token_padding_builder: {
    code: "1-4D",
    title: "Padding Builder",
    concept: "Padding 让 batch 成为矩形，attention_mask 区分内容和空位。",
    tool: "Padding Builder + Attention Mask",
    mission: "右侧 padding、pad id 0，并构建 mask 1/0。",
    visual: "token_mask",
    inspectorNotes: ["Pad 应在内容之后。", "pad id 在 Toy Vocab 中是 0。", "mask=0 的位置不应参与 attention 内容。"],
    failureLesson: ["Padding / mask mismatch。", "检查 pad id 是否为 0。", "mask 必须对 pad 位置置 0。"],
    debrief: "token_ids[B,T] 与 attention_mask[B,T] 已经成对成立。"
  },
  token_budget_gate: {
    code: "1-4E",
    title: "Token Budget Gate",
    concept: "超长文本必须被截断或拒绝，并留下 trace。",
    tool: "Token Budget Gate",
    mission: "应用 max T，处理 overflow，并显示预算消耗。",
    visual: "token_buffer",
    inspectorNotes: ["Budget gate 保护后续固定 shape。", "截断或拒绝都不能静默发生。", "trace 要指出哪些 pieces 占用了 T。"],
    failureLesson: ["Budget gate failed。", "检查是否忽略了 max T。", "超长文本需要明确策略和 trace。"],
    debrief: "长文本现在不会破坏 batch tensor，而是通过显式 budget policy 处理。"
  },
  tokenizer_gauntlet: {
    code: "1-X",
    title: "Tokenizer Gauntlet",
    concept: "完整 tokenizer contract 必须跨 visible/hidden 文本泛化。",
    tool: "Tokenizer Autograder",
    mission: "通过 type、pieces、vocab、fallback、special、batch、mask、determinism、budget、embedding 十项合同。",
    visual: "tokenizer_gauntlet",
    inspectorNotes: ["Hidden cases 会包含标点、OOV、padding 和 budget。", "同一输入会重复 encode 检查 determinism。", "最终输出必须能接 Chapter 2 Embedding。"],
    failureLesson: ["Tokenizer Gauntlet failed。", "按失败项回到对应小节。", "不要只适配 visible strings。"],
    debrief: "Chapter 1 完成：你已经把 raw text 变成可批处理、可查表、可测试的 token_ids[B,T]。"
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

const chapter04PhaseNodePositions: Partial<Record<LevelPhase, Record<string, { x: number; y: number }>>> = {
  broadcast_add_cell: {
    scalar_a: { x: 210, y: 220 },
    scalar_b: { x: 210, y: 414 },
    add_gate_cell: { x: 520, y: 318 },
    output_cell: { x: 820, y: 320 },
    cell_trace: { x: 1084, y: 320 }
  },
  broadcast_same_shape_add: {
    same_a: { x: 190, y: 210 },
    same_b: { x: 190, y: 420 },
    same_add_gate: { x: 536, y: 316 },
    same_out: { x: 844, y: 292 },
    same_trace: { x: 1142, y: 316 }
  },
  broadcast_rule_lab: {
    target_btc: { x: 132, y: 196 },
    vector_c: { x: 126, y: 434 },
    matrix_tc: { x: 374, y: 434 },
    singleton_scale: { x: 626, y: 434 },
    broadcast_rail_lab: { x: 812, y: 304 },
    ghost_expansion: { x: 1126, y: 206 },
    plan_checker: { x: 1126, y: 388 }
  },
  broadcast_bias_add: {
    projected_bto: { x: 158, y: 276 },
    bias_o: { x: 454, y: 446 },
    bias_rail: { x: 704, y: 320 },
    bias_add_gate: { x: 984, y: 320 },
    bias_out: { x: 1250, y: 286 },
    bias_reference: { x: 1250, y: 462 }
  },
  broadcast_position_add: {
    tok_emb: { x: 144, y: 282 },
    pos_sheet: { x: 456, y: 440 },
    position_rail: { x: 742, y: 322 },
    position_add_gate: { x: 1022, y: 322 },
    hidden_out: { x: 1288, y: 286 },
    position_trace: { x: 1288, y: 462 }
  },
  broadcast_mask_add: {
    scores_tensor: { x: 144, y: 244 },
    mask_plate: { x: 468, y: 454 },
    mask_rail: { x: 742, y: 326 },
    mask_add_gate: { x: 1018, y: 326 },
    masked_scores: { x: 1282, y: 286 },
    illegal_cell_checker: { x: 1282, y: 468 }
  },
  broadcast_trap_debugger: {
    trap_cases: { x: 126, y: 284 },
    trap_inspector: { x: 398, y: 264 },
    semantic_warning: { x: 666, y: 264 },
    trace_probe_04: { x: 934, y: 264 },
    trap_patch: { x: 680, y: 456 },
    trap_reference_04: { x: 1138, y: 368 }
  },
  broadcast_bonus_unsqueeze: {
    target_btc: { x: 210, y: 212 },
    matrix_tc: { x: 210, y: 436 },
    broadcast_rail_lab: { x: 572, y: 316 },
    ghost_expansion: { x: 904, y: 220 },
    plan_checker: { x: 904, y: 408 }
  },
  broadcast_bonus_no_copy: {
    target_btc: { x: 210, y: 212 },
    matrix_tc: { x: 210, y: 436 },
    broadcast_rail_lab: { x: 572, y: 316 },
    ghost_expansion: { x: 904, y: 220 },
    plan_checker: { x: 904, y: 408 }
  },
  broadcast_bonus_residual: {
    same_a: { x: 190, y: 210 },
    same_b: { x: 190, y: 420 },
    same_add_gate: { x: 536, y: 316 },
    same_out: { x: 844, y: 292 },
    same_trace: { x: 1142, y: 316 }
  },
  broadcast_bonus_mask_value: {
    scores_tensor: { x: 144, y: 244 },
    mask_plate: { x: 468, y: 454 },
    mask_rail: { x: 742, y: 326 },
    mask_add_gate: { x: 1018, y: 326 },
    masked_scores: { x: 1282, y: 286 },
    illegal_cell_checker: { x: 1282, y: 468 }
  },
  broadcast_gauntlet: {
    broadcast_gauntlet_cases: { x: 220, y: 292 },
    broadcast_contract_terminal: { x: 566, y: 292 },
    hidden_broadcast_reference: { x: 916, y: 292 },
    broadcast_gauntlet_result: { x: 1242, y: 292 }
  }
};

const chapter1TokenizationPhaseNodePositions: Partial<Record<LevelPhase, Record<string, { x: number; y: number }>>> = {
  token_raw_text_object: {
    raw_text_input: { x: 246, y: 300 },
    text_inspector: { x: 590, y: 286 },
    type_gate: { x: 936, y: 302 }
  },
  token_type_gate_failure: {
    raw_text_input: { x: 230, y: 304 },
    type_gate: { x: 584, y: 298 },
    raw_embedding_probe: { x: 934, y: 298 }
  },
  token_tokenizer_socket: {
    raw_text_input: { x: 226, y: 306 },
    tokenizer_socket: { x: 562, y: 298 },
    token_piece_stream: { x: 914, y: 292 }
  },
  token_token_id_contract: {
    token_piece_stream: { x: 218, y: 292 },
    token_id_emitter: { x: 578, y: 292 },
    embedding_lookup_probe: { x: 952, y: 300 }
  },
  token_boundary_cutter: {
    raw_text_input: { x: 164, y: 292 },
    boundary_cutter: { x: 492, y: 292 },
    token_piece_stream: { x: 834, y: 288 }
  },
  token_split_comparison: {
    boundary_cutter: { x: 210, y: 292 },
    split_policy_switch: { x: 564, y: 286 },
    token_count_meter: { x: 940, y: 296 }
  },
  token_count_meter: {
    split_policy_switch: { x: 202, y: 286 },
    token_count_meter: { x: 568, y: 294 },
    budget_gate: { x: 934, y: 296 }
  },
  token_merge_forge: {
    token_piece_stream: { x: 196, y: 252 },
    merge_forge: { x: 566, y: 294 },
    token_count_meter: { x: 944, y: 296 }
  },
  token_preserve_symbols: {
    token_piece_stream: { x: 198, y: 286 },
    symbol_keeper: { x: 560, y: 292 },
    vocab_lookup_gate: { x: 932, y: 294 }
  },
  token_vocab_lookup: {
    token_piece_stream: { x: 120, y: 286 },
    vocab_table: { x: 402, y: 252 },
    vocab_lookup_gate: { x: 736, y: 292 },
    token_id_emitter: { x: 1082, y: 292 }
  },
  token_address_lighting: {
    vocab_table: { x: 170, y: 252 },
    vocab_lookup_gate: { x: 508, y: 292 },
    address_lighting: { x: 834, y: 292 },
    embedding_lookup_probe: { x: 1166, y: 296 }
  },
  token_stable_id_test: {
    vocab_lookup_gate: { x: 224, y: 292 },
    stable_id_checker: { x: 590, y: 292 },
    token_buffer: { x: 966, y: 284 }
  },
  token_buffer_build: {
    token_id_emitter: { x: 214, y: 292 },
    stable_id_checker: { x: 562, y: 292 },
    token_buffer: { x: 936, y: 282 }
  },
  token_oov_failure: {
    vocab_lookup_gate: { x: 202, y: 292 },
    oov_detector: { x: 558, y: 292 },
    fallback_splitter: { x: 916, y: 292 }
  },
  token_fallback_splitter: {
    oov_detector: { x: 204, y: 292 },
    fallback_splitter: { x: 562, y: 292 },
    special_token_injector: { x: 936, y: 292 }
  },
  token_special_token_injector: {
    fallback_splitter: { x: 204, y: 292 },
    special_token_injector: { x: 570, y: 292 },
    padding_builder: { x: 960, y: 292 }
  },
  token_padding_builder: {
    special_token_injector: { x: 160, y: 262 },
    padding_builder: { x: 520, y: 262 },
    attention_mask_builder: { x: 880, y: 228 },
    token_buffer: { x: 880, y: 424 }
  },
  token_budget_gate: {
    token_count_meter: { x: 184, y: 282 },
    budget_gate: { x: 540, y: 290 },
    token_buffer: { x: 890, y: 216 },
    tokenizer_contract_terminal: { x: 890, y: 420 }
  },
  tokenizer_gauntlet: {
    gauntlet_cases: { x: 178, y: 292 },
    tokenizer_contract_terminal: { x: 546, y: 272 },
    tokenizer_gauntlet_result: { x: 934, y: 298 },
    embedding_lookup_probe: { x: 1240, y: 302 }
  }
};

export function App() {
  const [initialProgress] = useState<BootcampProgressSave>(() => readBootcampProgressSave());
  const initialSelectedLevelId = resolveSavedLevelId(initialProgress.selectedLevelId);
  const initialSelectedLevel = bootcampLevels.find((level) => level.id === initialSelectedLevelId) ?? bootcampLevels[0];
  const [mode, setMode] = useState<WorkbenchMode>("build");
  const [selectedLevelId, setSelectedLevelId] = useState(initialSelectedLevel.id);
  const [selectedId, setSelectedId] = useState(initialSelectedLevel.defaultSelectedNodeId);
  const [playing, setPlaying] = useState(true);
  const [repairStates, setRepairStates] = useState<Record<string, LevelRepairState>>(() => filterBootcampRecord(initialProgress.repairStates));
  const [results, setResults] = useState<Record<string, BootcampResult>>(() => filterBootcampRecord(initialProgress.results));
  const [introSeen, setIntroSeen] = useState<Record<string, boolean>>(() => filterBootcampRecord(initialProgress.introSeen));
  const [stageIntroSeen, setStageIntroSeen] = useState<Record<string, Record<string, boolean>>>(() => filterBootcampRecord(initialProgress.stageIntroSeen));
  const [missionStarted, setMissionStarted] = useState<Record<string, boolean>>(() => filterBootcampRecord(initialProgress.missionStarted));
  const [completionDismissed, setCompletionDismissed] = useState<Record<string, boolean>>(() => filterBootcampRecord(initialProgress.completionDismissed));
  const [expandedLevelIds, setExpandedLevelIds] = useState<Record<string, boolean>>(() => ({
    [initialSelectedLevel.id]: supportsStageRail(initialSelectedLevel) && !Boolean(initialProgress.results?.[initialSelectedLevel.id]?.passed)
  }));
  const [expandedChapterIds, setExpandedChapterIds] = useState<Record<string, boolean>>(() => ({
    [chapterIdForLevel(initialSelectedLevel)]: true
  }));
  const [expandedChapter1GroupIds, setExpandedChapter1GroupIds] = useState<Record<string, boolean>>({});
  const [nodePositions, setNodePositions] = useState<Record<string, NodePositionMap>>({});
  const [stageKnowledgePositions, setStageKnowledgePositions] = useState<Record<string, StageKnowledgePositionMap>>({});
  const [pendingStageDebrief, setPendingStageDebrief] = useState<Record<string, LevelPhase | undefined>>({});
  const [stageDebriefOpen, setStageDebriefOpen] = useState<Record<string, boolean>>({});
  const [stageIntroRecallOpen, setStageIntroRecallOpen] = useState(false);
  const [tensorObjectDetail, setTensorObjectDetail] = useState<TensorObjectDetailState | null>(null);
  const [tensorRunOutputs, setTensorRunOutputs] = useState<Record<string, string>>({});
  const [canvasMenu, setCanvasMenu] = useState<CanvasContextTarget | null>(null);
  const [canvasTaskHint, setCanvasTaskHint] = useState<CanvasTaskHint | undefined>();

  const activeLevel = useMemo(
    () => bootcampLevels.find((level) => level.id === selectedLevelId) ?? bootcampLevels[0],
    [selectedLevelId]
  );
  const activeRepairState = repairStates[activeLevel.id] ?? createInitialRepairState();
  const levelResult = results[activeLevel.id];
  const levelCompleted = Boolean(levelResult?.passed);
  const showKnowledgeIntro = Boolean(!levelCompleted && activeLevel.knowledgeCards?.length && !introSeen[activeLevel.id]);
  const showMissionModal = Boolean(!levelCompleted && activeLevel.mission && !showKnowledgeIntro && !missionStarted[activeLevel.id]);
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
  const nextLevel = useMemo(() => nextBootcampLevelAfter(activeLevel), [activeLevel]);
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
  const showCanvasPalette = shouldShowCanvasPalette(activeLevel, levelPhase) && !showKnowledgeIntro && !showMissionModal && !levelResult?.passed;
  const showCanvasRunButton = shouldShowCanvasRunButton(activeLevel, levelPhase);
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
    setCanvasTaskHint(undefined);
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

  useEffect(() => {
    writeBootcampProgressSave({
      version: 1,
      selectedLevelId,
      repairStates,
      results,
      introSeen,
      stageIntroSeen,
      missionStarted,
      completionDismissed
    });
  }, [selectedLevelId, repairStates, results, introSeen, stageIntroSeen, missionStarted, completionDismissed]);

  function selectLevel(level: BootcampLevel) {
    const alreadySelected = selectedLevelId === level.id;
    setSelectedLevelId(level.id);
    setSelectedId(level.defaultSelectedNodeId);
    setExpandedChapterIds((current) => ({ ...current, [chapterIdForLevel(level)]: true }));
    if (supportsStageRail(level)) {
      setExpandedLevelIds((current) => ({
        ...current,
        [level.id]: alreadySelected ? !(current[level.id] ?? !results[level.id]?.passed) : !results[level.id]?.passed
      }));
    }
    if (results[level.id]?.passed) {
      setCompletionDismissed((current) => ({ ...current, [level.id]: true }));
    }
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
    nextAssignments[slot.id] = tagId;
    const assignedCorrectly = slot.correctTagIds.includes(tagId);
    const shouldOpenTensorObjectDetail = activeLevel.id === "0-1" && levelPhase === "tensor_object" && assignedCorrectly && Boolean(tensorObjectShowcases[slot.id]);
    const shouldShowStageDebrief =
      supportsStageDebrief(activeLevel) &&
      !pendingDebriefPhase &&
      stageReadyForDebrief(activeLevel, levelPhase, nextAssignments, activeRepairState.observations);

    updateActiveRepairState((state) => {
      const assignments = { ...state.assignments };
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

  function requestTaskHint(taskId: string) {
    const slot = taskHintSlotForTask(activeLevel, levelPhase, activeRepairState, taskId);
    if (!slot) return;

    const operation = buildTaskHintOperation(activeLevel, levelPhase, activeRepairState, slot);
    updateActiveRepairState((state) => ({
      ...state,
      selectedSlotId: slot.id,
      activeTagId: operation.activeTagId,
      activeProbeId: operation.activeProbeId
    }));
    setSelectedId(slot.focusNodeId);
    setPlaying(true);
    setCanvasTaskHint({
      id: Date.now(),
      taskId,
      slotId: slot.id,
      nodeId: slot.focusNodeId,
      title: slot.label,
      detail: operation.detail
    });
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
    if (!target || (activeLevel.id !== "0-1" && activeLevel.id !== "0-2" && activeLevel.id !== "0-3" && activeLevel.id !== "0-4" && activeLevel.id !== "1-1")) return [];

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

    if (activeLevel.id === "0-2" && ((target.kind === "node" && target.id === "matmul_tests") || target.kind === "canvas")) {
      if (levelPhase === "matmul_gauntlet") {
        actions.push({
          id: "run_matmul_tests",
          label: "Run MatMul Gauntlet",
          detail: "Execute reference and hidden MatMul cases.",
          onSelect: runCanvasTests
        });
      }
    }

    if (activeLevel.id === "0-3" && ((target.kind === "node" && target.id === "gauntlet_result") || target.kind === "canvas")) {
      if (levelPhase === "transpose_gauntlet") {
        actions.push({
          id: "run_transpose_tests",
          label: "Run Transpose Gauntlet",
          detail: "Execute reference and hidden QK^T cases.",
          onSelect: runCanvasTests
        });
      }
    }

    if (activeLevel.id === "0-4" && ((target.kind === "node" && target.id === "broadcast_gauntlet_result") || target.kind === "canvas")) {
      if (levelPhase === "broadcast_gauntlet") {
        actions.push({
          id: "run_broadcast_tests",
          label: "Run Broadcast Gauntlet",
          detail: "Execute reference and hidden Broadcast Add cases.",
          onSelect: runCanvasTests
        });
      }
    }

    if (activeLevel.id === "1-1" && ((target.kind === "node" && target.id === "tokenizer_gauntlet_result") || target.kind === "canvas")) {
      if (levelPhase === "tokenizer_gauntlet") {
        actions.push({
          id: "run_tokenizer_tests",
          label: "Run Tokenizer Gauntlet",
          detail: "Execute visible and hidden tokenizer contract cases.",
          onSelect: runCanvasTests
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
    if (supportsStageRail(activeLevel)) {
      setExpandedLevelIds((current) => ({ ...current, [activeLevel.id]: true }));
    }
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
            <h1>Campaign Workbench</h1>
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
        <StatusPill label="Chapter" value={levelDisplayTitle(activeLevel)} />
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
            <h2>Campaign</h2>
          </div>
          <p className="panelNote">玩家产物是可验证的模型构建 repair，而不是选择题答案。</p>

          <div className="levelList">
            {campaignChapterDefs.map((chapter) => {
              const chapterLevels = levelsForCampaignChapter(chapter);
              const chapterSelected = chapterLevels.some((level) => level.id === activeLevel.id);
              const chapterExpanded = expandedChapterIds[chapter.id] ?? chapterSelected;
              const chapterPassedCount = chapterLevels.filter((level) => results[level.id]?.passed).length;
              const chapterHasFail = chapterLevels.some((level) => results[level.id] && !results[level.id]?.passed);
              const chapterState: CheckState = chapterPassedCount === chapterLevels.length ? "pass" : chapterHasFail ? "fail" : "warn";
              const useChapter1Groups = chapter.id === "chapter1" && chapterLevels.length === 1;
              return (
                <section key={chapter.id} className={`chapterNavBlock ${chapterExpanded ? "expanded" : ""} ${chapterSelected ? "active" : ""}`}>
                  <button
                    className={`chapterNavHeader ${chapterSelected ? "active" : ""} ${chapterState}`}
                    type="button"
                    aria-expanded={chapterExpanded}
                    title={`${chapter.code}: ${chapterExpanded ? "收起章节" : "展开章节"}`}
                    onClick={() => {
                      if (useChapter1Groups && !chapterSelected) {
                        selectLevel(chapterLevels[0]);
                        return;
                      }
                      setExpandedChapterIds((current) => ({ ...current, [chapter.id]: !(current[chapter.id] ?? chapterSelected) }));
                    }}
                  >
                    <span className="chapterCode">{chapter.code.replace("Chapter ", "Ch")}</span>
                    <span>
                      <b>{chapter.title}</b>
                      <small>{chapter.subtitle}</small>
                    </span>
                    <code>{chapterPassedCount}/{chapterLevels.length}</code>
                    <span className={`levelExpandIcon ${chapterExpanded ? "open" : ""}`} aria-hidden="true">
                      <ChevronDown size={16} />
                    </span>
                    <StateIcon state={chapterState} />
                  </button>

                  {chapterExpanded ? useChapter1Groups ? (
                    <div className="chapterLevelList chapter1LevelList">
                      {chapterLevels.map((level) => {
                        const result = results[level.id];
                        const selected = selectedLevelId === level.id;
                        const levelRepairState = selected ? activeRepairState : repairStates[level.id] ?? createInitialRepairState();
                        const displayPhase = selected ? levelPhase : deriveLevelPhase(level, levelRepairState, result, false, false);
                        return (
                          <Chapter1GroupLevelList
                            key={level.id}
                            level={level}
                            repairState={levelRepairState}
                            phase={displayPhase}
                            result={result}
                            selected={selected}
                            expandedGroups={expandedChapter1GroupIds}
                            onSelect={() => selectLevel(level)}
                            onToggleGroup={(groupCode, currentlyExpanded) =>
                              setExpandedChapter1GroupIds((current) => ({ ...current, [groupCode]: !currentlyExpanded }))
                            }
                          />
                        );
                      })}
                    </div>
                  ) : (
                    <div className="chapterLevelList">
                      {chapterLevels.map((level) => {
                        const result = results[level.id];
                        const statusClass = result ? (result.passed ? "pass" : "fail") : "pending";
                        const hasNestedChallenges = supportsStageRail(level);
                        const selected = selectedLevelId === level.id;
                        const showNestedChallenges = hasNestedChallenges && selected && Boolean(expandedLevelIds[level.id]);
                        const levelRepairState = selected ? activeRepairState : repairStates[level.id] ?? createInitialRepairState();
                        return (
                          <div key={level.id} className={`levelGroup ${showNestedChallenges ? "expanded" : ""}`}>
                            <button
                              className={`levelItem ${selected ? "active" : ""} ${statusClass} ${hasNestedChallenges ? "collapsible" : ""}`}
                              aria-expanded={hasNestedChallenges ? showNestedChallenges : undefined}
                              title={hasNestedChallenges ? `${level.title}: ${showNestedChallenges ? "收起阶段列表" : "展开阶段列表"}` : level.title}
                              onClick={() => selectLevel(level)}
                            >
                              <span className="levelId">{levelDisplayBadge(level)}</span>
                              <span>
                                <b>{level.title}</b>
                                <small>{level.subtitle}</small>
                                <small>Tool: {repairKindLabel(level.repair.kind)}</small>
                                <small>Reward: {level.unlocks[0]}</small>
                              </span>
                              {hasNestedChallenges ? (
                                <span className={`levelExpandIcon ${showNestedChallenges ? "open" : ""}`} aria-hidden="true">
                                  <ChevronDown size={16} />
                                </span>
                              ) : (
                                <span className="levelExpandIcon placeholder" aria-hidden="true" />
                              )}
                              <StateIcon state={result ? (result.passed ? "pass" : "fail") : "warn"} />
                            </button>
                            {showNestedChallenges ? (
                              <ChapterChallengeRail level={level} repairState={levelRepairState} phase={levelPhase} result={result} />
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                  ) : null}
                </section>
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
            taskHint={canvasTaskHint}
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
          {showCanvasPalette ? (
            <div className="canvasPaletteLayer">
              <BlueprintPalette
                level={activeLevel}
                repairState={activeRepairState}
                phase={levelPhase}
                runLabel={canvasRunTestLabel(activeLevel, levelPhase)}
                onSelectTag={selectTag}
                onSelectProbe={selectProbe}
                onRunTests={showCanvasRunButton ? runTests : undefined}
              />
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
            hintedTaskId={canvasTaskHint?.taskId}
            result={levelResult}
            onTaskHint={requestTaskHint}
            onOpenTensorObjectDetail={openTensorObjectDetail}
            onRunTensorObjectShowcase={runTensorObjectShowcase}
          />
          {levelResult ? (
            <ResultPanel result={levelResult} activeLevel={activeLevel} />
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

function readBootcampProgressSave(): BootcampProgressSave {
  if (typeof window === "undefined") return { version: 1 };

  try {
    const raw = window.localStorage.getItem(progressStorageKey);
    if (!raw) return { version: 1 };
    const parsed = JSON.parse(raw) as Partial<BootcampProgressSave>;
    if (parsed.version !== 1) return { version: 1 };
    const repairStates = parsed.repairStates ? { ...parsed.repairStates } : undefined;
    const results = parsed.results ? { ...parsed.results } : undefined;
    const introSeen = parsed.introSeen ? { ...parsed.introSeen } : undefined;
    const stageIntroSeen = parsed.stageIntroSeen ? { ...parsed.stageIntroSeen } : undefined;
    const missionStarted = parsed.missionStarted ? { ...parsed.missionStarted } : undefined;
    const completionDismissed = parsed.completionDismissed ? { ...parsed.completionDismissed } : undefined;
    const savedTransposeResult = results?.["0-3"];
    const hasCurrentTransposeGauntlet =
      savedTransposeResult?.checks.some((check) => check.id === "transpose_gauntlet_cases") ?? false;
    if (savedTransposeResult && !hasCurrentTransposeGauntlet) {
      delete results?.["0-3"];
      delete repairStates?.["0-3"];
      delete introSeen?.["0-3"];
      delete stageIntroSeen?.["0-3"];
      delete missionStarted?.["0-3"];
      delete completionDismissed?.["0-3"];
    }
    return {
      version: 1,
      selectedLevelId: parsed.selectedLevelId,
      repairStates,
      results,
      introSeen,
      stageIntroSeen,
      missionStarted,
      completionDismissed
    };
  } catch {
    return { version: 1 };
  }
}

function writeBootcampProgressSave(progress: BootcampProgressSave) {
  if (typeof window === "undefined") return;

  try {
    window.localStorage.setItem(
      progressStorageKey,
      JSON.stringify({
        ...progress,
        repairStates: filterBootcampRecord(progress.repairStates),
        results: filterBootcampRecord(progress.results),
        introSeen: filterBootcampRecord(progress.introSeen),
        stageIntroSeen: filterBootcampRecord(progress.stageIntroSeen),
        missionStarted: filterBootcampRecord(progress.missionStarted),
        completionDismissed: filterBootcampRecord(progress.completionDismissed)
      })
    );
  } catch {
    // Progress persistence should never block gameplay.
  }
}

function resolveSavedLevelId(levelId: string | undefined) {
  return bootcampLevels.some((level) => level.id === levelId) ? levelId : bootcampLevels[0].id;
}

function nextBootcampLevelAfter(level: BootcampLevel) {
  const explicitNextId = bootcampLevelTransitions[level.id];
  if (explicitNextId) {
    const explicitNext = bootcampLevels.find((candidate) => candidate.id === explicitNextId);
    if (explicitNext) return explicitNext;
  }

  const levelIndex = bootcampLevels.findIndex((candidate) => candidate.id === level.id);
  return levelIndex >= 0 ? bootcampLevels[levelIndex + 1] : undefined;
}

function filterBootcampRecord<T>(record: Record<string, T> | undefined): Record<string, T> {
  if (!record || typeof record !== "object") return {};
  const validIds = new Set(bootcampLevels.map((level) => level.id));
  return Object.fromEntries(Object.entries(record).filter(([levelId]) => validIds.has(levelId)));
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

  if (level.id === "1-1") {
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_raw_text_object"))) return "token_raw_text_object";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_type_gate_failure"))) return "token_type_gate_failure";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_tokenizer_socket"))) return "token_tokenizer_socket";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_token_id_contract"))) return "token_token_id_contract";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_boundary_cutter"))) return "token_boundary_cutter";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_split_comparison"))) return "token_split_comparison";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_count_meter"))) return "token_count_meter";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_merge_forge"))) return "token_merge_forge";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_preserve_symbols"))) return "token_preserve_symbols";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_vocab_lookup"))) return "token_vocab_lookup";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_address_lighting"))) return "token_address_lighting";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_stable_id_test"))) return "token_stable_id_test";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_buffer_build"))) return "token_buffer_build";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_oov_failure"))) return "token_oov_failure";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_fallback_splitter"))) return "token_fallback_splitter";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_special_token_injector"))) return "token_special_token_injector";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_padding_builder"))) return "token_padding_builder";
    if (!areSlotsCorrect(level, repairState, chapter1TokenizationSlotIds("token_budget_gate"))) return "token_budget_gate";
    return "tokenizer_gauntlet";
  }

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
    if (level.id === "0-3") {
      if (!areSlotsCorrect(level, repairState, chapter03SlotIds("transpose_matrix_flip"))) return "transpose_matrix_flip";
      if (!areSlotsCorrect(level, repairState, chapter03SlotIds("transpose_inner_dim_repair"))) return "transpose_inner_dim_repair";
      if (!areSlotsCorrect(level, repairState, chapter03SlotIds("transpose_higher_rank_axis_swap"))) return "transpose_higher_rank_axis_swap";
      if (!areSlotsCorrect(level, repairState, chapter03SlotIds("transpose_single_head_qk"))) return "transpose_single_head_qk";
      if (!areSlotsCorrect(level, repairState, chapter03SlotIds("transpose_multi_head_trap"))) return "transpose_multi_head_trap";
      if (!areSlotsCorrect(level, repairState, chapter03SlotIds("transpose_trap_debugger"))) return "transpose_trap_debugger";
      return "transpose_gauntlet";
    }
    if (level.id === "0-4") {
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_add_cell"))) return "broadcast_add_cell";
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_same_shape_add"))) return "broadcast_same_shape_add";
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_rule_lab"))) return "broadcast_rule_lab";
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_bias_add"))) return "broadcast_bias_add";
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_position_add"))) return "broadcast_position_add";
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_mask_add"))) return "broadcast_mask_add";
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_trap_debugger"))) return "broadcast_trap_debugger";
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_bonus_unsqueeze"))) return "broadcast_bonus_unsqueeze";
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_bonus_no_copy"))) return "broadcast_bonus_no_copy";
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_bonus_residual"))) return "broadcast_bonus_residual";
      if (!areSlotsCorrect(level, repairState, chapter04SlotIds("broadcast_bonus_mask_value"))) return "broadcast_bonus_mask_value";
      return "broadcast_gauntlet";
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
  return level.id === "0-1" || level.id === "0-2" || level.id === "0-3" || level.id === "0-4" || level.id === "1-1";
}

function shouldShowCanvasPalette(level: BootcampLevel, phase: LevelPhase) {
  if (level.id === "0-1") return phase !== "tensor_object" && Boolean(chapter01ChallengeByPhase.get(phase));
  if (level.id === "0-2") return Boolean(chapter02ChallengeByPhase.get(phase));
  if (level.id === "0-3") return Boolean(chapter03ChallengeByPhase.get(phase));
  if (level.id === "0-4") return Boolean(chapter04ChallengeByPhase.get(phase));
  if (level.id === "1-1") return Boolean(chapter1TokenizationChallengeByPhase.get(phase));
  return visibleSlotsForPhase(level, phase).length > 0 && visibleTagsForPhase(level, phase).length > 0;
}

function shouldShowCanvasRunButton(level: BootcampLevel, phase: LevelPhase) {
  if (level.id === "0-1") return phase === "hidden_test_gauntlet";
  if (level.id === "0-2") return phase === "matmul_gauntlet";
  if (level.id === "0-3") return phase === "transpose_gauntlet";
  if (level.id === "0-4") return phase === "broadcast_gauntlet";
  if (level.id === "1-1") return phase === "tokenizer_gauntlet";
  return true;
}

function canvasRunTestLabel(level: BootcampLevel, phase: LevelPhase) {
  if (level.id === "0-1" && phase === "hidden_test_gauntlet") return "Run Hidden Test Gauntlet";
  if (level.id === "0-2" && phase === "matmul_gauntlet") return "Run MatMul Gauntlet";
  if (level.id === "0-3" && phase === "transpose_gauntlet") return "Run Transpose Gauntlet";
  if (level.id === "0-4" && phase === "broadcast_gauntlet") return "Run Broadcast Gauntlet";
  if (level.id === "1-1" && phase === "tokenizer_gauntlet") return "Run Tokenizer Gauntlet";
  return "Run Contract Tests";
}

function supportsStageDebrief(level: BootcampLevel) {
  return supportsStageRail(level);
}

function challengesForLevel(level: BootcampLevel) {
  if (level.id === "0-1") return chapter01Challenges;
  if (level.id === "0-2") return chapter02Challenges;
  if (level.id === "0-3") return chapter03Challenges;
  if (level.id === "0-4") return chapter04Challenges;
  if (level.id === "1-1") return chapter1TokenizationChallenges;
  return [];
}

function challengeForLevelPhase(level: BootcampLevel, phase: LevelPhase) {
  if (level.id === "0-1") return chapter01ChallengeForPhase(phase);
  if (level.id === "0-2") return chapter02ChallengeForPhase(phase);
  if (level.id === "0-3") return chapter03ChallengeForPhase(phase);
  if (level.id === "0-4") return chapter04ChallengeForPhase(phase);
  if (level.id === "1-1") return chapter1TokenizationChallengeForPhase(phase);
  return undefined;
}

function stageIntroForLevelPhase(level: BootcampLevel, phase: LevelPhase) {
  if (level.id === "0-1") return chapter01StageIntros[phase];
  if (level.id === "0-2") return chapter02StageIntros[phase];
  if (level.id === "0-3") return chapter03StageIntros[phase];
  if (level.id === "0-4") return chapter04StageIntros[phase];
  if (level.id === "1-1") return chapter1TokenizationStageIntros[phase];
  return undefined;
}

function stageKnowledgeForLevelPhase(level: BootcampLevel, phase: LevelPhase) {
  if (level.id === "0-1") return chapter01StageKnowledge[phase];
  if (level.id === "0-2") return chapter02StageKnowledge[phase];
  if (level.id === "0-3") return chapter03StageKnowledge[phase];
  if (level.id === "0-4") return chapter04StageKnowledge[phase];
  if (level.id === "1-1") return chapter1TokenizationStageKnowledge[phase];
  return undefined;
}

function stageReadyForDebrief(level: BootcampLevel, phase: LevelPhase, assignments: BootcampAnswerMap, observations: ObservationLogItem[]) {
  if (level.id === "0-1") return chapter01StageReadyForDebrief(level, phase, assignments, observations);
  if (level.id === "0-2") {
    const challenge = chapter02ChallengeForPhase(phase);
    return Boolean(challenge?.slotIds.length && areAssignmentsCorrect(level, assignments, challenge.slotIds));
  }
  if (level.id === "0-3") {
    const challenge = chapter03ChallengeForPhase(phase);
    return Boolean(challenge?.slotIds.length && areAssignmentsCorrect(level, assignments, challenge.slotIds));
  }
  if (level.id === "0-4") {
    const challenge = chapter04ChallengeForPhase(phase);
    return Boolean(challenge?.slotIds.length && areAssignmentsCorrect(level, assignments, challenge.slotIds));
  }
  if (level.id === "1-1") {
    const challenge = chapter1TokenizationChallengeForPhase(phase);
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

function chapter03SlotIds(phase: LevelPhase) {
  return chapter03ChallengeByPhase.get(phase)?.slotIds ?? [];
}

function chapter04SlotIds(phase: LevelPhase) {
  return chapter04ChallengeByPhase.get(phase)?.slotIds ?? [];
}

function chapter1TokenizationSlotIds(phase: LevelPhase) {
  return chapter1TokenizationChallengeByPhase.get(phase)?.slotIds ?? [];
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

function chapter03ChallengeForPhase(phase: LevelPhase) {
  if (phase === "knowledge_intro" || phase === "mission_modal") return chapter03Challenges[0];
  if (phase === "completed") return chapter03ChallengeByPhase.get("transpose_gauntlet");
  return chapter03ChallengeByPhase.get(phase);
}

function chapter04ChallengeForPhase(phase: LevelPhase) {
  if (phase === "knowledge_intro" || phase === "mission_modal") return chapter04Challenges[0];
  if (phase === "completed") return chapter04ChallengeByPhase.get("broadcast_gauntlet");
  return chapter04ChallengeByPhase.get(phase);
}

function chapter1TokenizationChallengeForPhase(phase: LevelPhase) {
  if (phase === "knowledge_intro" || phase === "mission_modal") return chapter1TokenizationChallenges[0];
  if (phase === "completed") return chapter1TokenizationChallengeByPhase.get("tokenizer_gauntlet");
  return chapter1TokenizationChallengeByPhase.get(phase);
}

function buildCanvasStageKnowledge(level: BootcampLevel, phase: LevelPhase): Chapter01StageKnowledge | undefined {
  const finalPhases: Partial<Record<string, LevelPhase>> = {
    "0-1": "hidden_test_gauntlet",
    "0-2": "matmul_gauntlet",
    "0-3": "transpose_gauntlet",
    "0-4": "broadcast_gauntlet",
    "1-1": "tokenizer_gauntlet"
  };
  const finalPhase = finalPhases[level.id];
  if (!finalPhase) return undefined;
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
    transpose_matrix_flip: "0-3A Matrix Flip",
    transpose_inner_dim_repair: "0-3B Inner-Dim Repair",
    transpose_higher_rank_axis_swap: "0-3C Higher-Rank Axis Swap",
    transpose_single_head_qk: "0-3D Single-Head QK",
    transpose_multi_head_trap: "0-3E Multi-Head Trap",
    transpose_trap_debugger: "0-3F Trap Debugger",
    transpose_gauntlet: "0-3X Transpose Gauntlet",
    broadcast_add_cell: "0-4A Add Cell",
    broadcast_same_shape_add: "0-4B Same-Shape Add",
    broadcast_rule_lab: "0-4C Broadcast Rule Lab",
    broadcast_bias_add: "0-4D Bias Add",
    broadcast_position_add: "0-4E Position Add",
    broadcast_mask_add: "0-4F Mask Add",
    broadcast_trap_debugger: "0-4G Broadcast Trap Debugger",
    broadcast_bonus_unsqueeze: "Bonus A Unsqueeze Lab",
    broadcast_bonus_no_copy: "Bonus B No-Copy Broadcast",
    broadcast_bonus_residual: "Bonus C Residual Add",
    broadcast_bonus_mask_value: "Bonus D Mask Value",
    broadcast_gauntlet: "0-4X Broadcast Gauntlet",
    token_raw_text_object: "1-1A Raw Text Object",
    token_type_gate_failure: "1-1B Type Gate Failure",
    token_tokenizer_socket: "1-1C Tokenizer Socket",
    token_token_id_contract: "1-1D Token ID Contract",
    token_boundary_cutter: "1-2A Boundary Cutter",
    token_split_comparison: "1-2B Split Comparison",
    token_count_meter: "1-2C Token Count Meter",
    token_merge_forge: "1-2D Merge Forge",
    token_preserve_symbols: "1-2E Preserve Symbols",
    token_vocab_lookup: "1-3A Vocab Lookup",
    token_address_lighting: "1-3B Address Lighting",
    token_stable_id_test: "1-3C Stable ID Test",
    token_buffer_build: "1-3D Token Buffer Build",
    token_oov_failure: "1-4A OOV Failure",
    token_fallback_splitter: "1-4B Fallback Splitter",
    token_special_token_injector: "1-4C Special Token Injector",
    token_padding_builder: "1-4D Padding Builder",
    token_budget_gate: "1-4E Token Budget Gate",
    tokenizer_gauntlet: "1-X Tokenizer Gauntlet",
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
  if (level.id === "0-2" || level.id === "0-3" || level.id === "0-4" || level.id === "1-1") {
    const challenge =
      level.id === "0-2"
        ? chapter02ChallengeForPhase(phase)
        : level.id === "0-3"
          ? chapter03ChallengeForPhase(phase)
          : level.id === "0-4"
            ? chapter04ChallengeForPhase(phase)
            : chapter1TokenizationChallengeForPhase(phase);
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
  if (level.id === "0-4" || level.id === "1-1") return visibleTagsForPhase(level, phase);
  return visibleTagsForPhase(level, phase).filter((tag) => tag.category !== "data" && tag.category !== "consumer");
}

function visibleSlotsForPhase(level: BootcampLevel, phase: LevelPhase): RepairSlot[] {
  if (level.id === "0-2" || level.id === "0-3" || level.id === "0-4" || level.id === "1-1") {
    const challenge =
      level.id === "0-2"
        ? chapter02ChallengeForPhase(phase)
        : level.id === "0-3"
          ? chapter03ChallengeForPhase(phase)
          : level.id === "0-4"
            ? chapter04ChallengeForPhase(phase)
            : chapter1TokenizationChallengeForPhase(phase);
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
  if (level.id === "0-3") {
    const probeIds =
      phase === "transpose_matrix_flip" || phase === "transpose_higher_rank_axis_swap"
        ? ["transpose_probe"]
        : phase === "transpose_inner_dim_repair"
          ? ["matmul_probe"]
          : phase === "transpose_single_head_qk" || phase === "transpose_multi_head_trap"
            ? ["qk_probe"]
            : ["debug_probe"];
    return level.repair.probes.filter((probe) => probeIds.includes(probe.id));
  }
  if (level.id === "0-4") {
    const probeIds =
      phase === "broadcast_add_cell" || phase === "broadcast_same_shape_add" || phase === "broadcast_bonus_residual"
        ? ["add_probe"]
        : phase === "broadcast_rule_lab" || phase === "broadcast_bonus_unsqueeze"
          ? ["broadcast_probe"]
        : phase === "broadcast_bias_add" || phase === "broadcast_position_add"
          ? ["llm_broadcast_probe"]
          : phase === "broadcast_mask_add" || phase === "broadcast_bonus_mask_value"
            ? ["mask_probe"]
            : ["trap_probe"];
    return level.repair.probes.filter((probe) => probeIds.includes(probe.id));
  }
  if (level.id === "1-1") {
    const probeIds =
      phase === "token_raw_text_object" || phase === "token_type_gate_failure" || phase === "token_tokenizer_socket" || phase === "token_token_id_contract"
        ? ["text_probe"]
        : phase === "token_boundary_cutter" ||
            phase === "token_split_comparison" ||
            phase === "token_count_meter" ||
            phase === "token_merge_forge" ||
            phase === "token_preserve_symbols"
          ? ["split_probe"]
          : phase === "token_vocab_lookup" ||
              phase === "token_address_lighting" ||
              phase === "token_stable_id_test" ||
              phase === "token_buffer_build" ||
              phase === "token_oov_failure" ||
              phase === "token_fallback_splitter"
            ? ["vocab_probe"]
            : ["buffer_probe"];
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
  if (level.id === "0-3") {
    return visibleProbesForPhase(level, phase)[0]?.id;
  }
  if (level.id === "0-4") {
    return visibleProbesForPhase(level, phase)[0]?.id;
  }
  if (level.id === "1-1") {
    return visibleProbesForPhase(level, phase)[0]?.id;
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
  if (level.id === "1-1") {
    const challenge = chapter1TokenizationChallengeForPhase(phase);
    if (!challenge) return nodes;
    const visibleIds = new Set(challenge.nodeIds);
    const phasePositions = chapter1TokenizationPhaseNodePositions[phase] ?? {};
    return nodes
      .filter((node) => visibleIds.has(node.id))
      .map((node) => {
        const positioned = phasePositions[node.id] ? { ...node, ...phasePositions[node.id] } : node;
        return sizeChapter1TokenizationPhaseNode(positioned, phase);
      });
  }
  if (level.id === "0-2" || level.id === "0-3" || level.id === "0-4") {
    const challenge =
      level.id === "0-2" ? chapter02ChallengeForPhase(phase) : level.id === "0-3" ? chapter03ChallengeForPhase(phase) : chapter04ChallengeForPhase(phase);
    if (!challenge) return nodes;
    const visibleIds = new Set(challenge.nodeIds);
    const phasePositions = level.id === "0-2" ? chapter02PhaseNodePositions[phase] ?? {} : level.id === "0-4" ? chapter04PhaseNodePositions[phase] ?? {} : {};
    return nodes
      .filter((node) => visibleIds.has(node.id))
      .map((node) => {
        const positioned = phasePositions[node.id] ? { ...node, ...phasePositions[node.id] } : node;
        return level.id === "0-2" ? sizeChapter02PhaseNode(positioned, phase) : level.id === "0-3" ? sizeChapter03PhaseNode(positioned, phase) : sizeChapter04PhaseNode(positioned, phase);
      });
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

function sizeChapter1TokenizationPhaseNode(node: TensorNode, phase: LevelPhase): TensorNode {
  if (phase === "tokenizer_gauntlet") {
    if (node.id === "tokenizer_contract_terminal") return { ...node, w: 294, h: 160 };
    if (node.id === "gauntlet_cases") return { ...node, w: 250, h: 142 };
  }

  if (node.id === "token_piece_stream" || node.id === "token_buffer") return { ...node, w: Math.max(node.w, 236), h: Math.max(node.h, 122) };
  if (node.id === "vocab_table") return { ...node, w: Math.max(node.w, 238), h: Math.max(node.h, 136) };
  if (node.id === "attention_mask_builder") return { ...node, w: Math.max(node.w, 236), h: Math.max(node.h, 120) };
  return node;
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

function sizeChapter03PhaseNode(node: TensorNode, phase: LevelPhase): TensorNode {
  if (phase === "transpose_matrix_flip") {
    if (node.id === "matrix_plate" || node.id === "matrix_t_plate") return { ...node, w: 232, h: 142 };
    if (node.id === "transpose_2d") return { ...node, w: 224, h: 102 };
  }

  if (phase === "transpose_inner_dim_repair") {
    if (node.id === "matmul_a" || node.id === "matmul_b_bad") return { ...node, w: 210, h: 124 };
    if (node.id === "transpose_b_repair" || node.id === "matmul_repair_gate") return { ...node, w: 220, h: 104 };
    if (node.id === "matmul_repair_out" || node.id === "reference_checker_03") return { ...node, w: 232, h: 112 };
  }

  if (phase === "transpose_higher_rank_axis_swap") {
    if (node.id === "rank3_tensor" || node.id === "rank4_tensor") return { ...node, w: 234, h: 126 };
    if (node.id === "axis_swap_switch" || node.id === "carry_axis_lock") return { ...node, w: 238, h: 112 };
  }

  if (phase === "transpose_single_head_qk" || phase === "transpose_multi_head_trap") {
    if (node.kind === "tensor" || node.kind === "attention") return { ...node, w: 228, h: 128 };
    if (node.kind === "operation") return { ...node, w: 222, h: 104 };
  }

  if (phase === "transpose_trap_debugger") {
    if (node.id === "faulty_board") return { ...node, w: 244, h: 146 };
    if (node.kind === "operation") return { ...node, w: 238, h: 114 };
  }

  if (phase === "transpose_gauntlet") {
    if (node.id === "qk_contract_terminal") return { ...node, w: 274, h: 120 };
    return { ...node, w: 238, h: 122 };
  }

  return node;
}

function sizeChapter04PhaseNode(node: TensorNode, phase: LevelPhase): TensorNode {
  if (phase === "broadcast_add_cell") {
    if (node.id === "scalar_a" || node.id === "scalar_b") return { ...node, w: 164, h: 84 };
    if (node.id === "add_gate_cell" || node.id === "cell_trace") return { ...node, w: 206, h: 104 };
    if (node.id === "output_cell") return { ...node, w: 190, h: 100 };
  }

  if (phase === "broadcast_same_shape_add" || phase === "broadcast_bonus_residual") {
    if (node.id === "same_a" || node.id === "same_b") return { ...node, w: 232, h: 132 };
    if (node.id === "same_add_gate") return { ...node, w: 226, h: 106 };
    if (node.id === "same_out") return { ...node, w: 244, h: 140 };
    if (node.id === "same_trace") return { ...node, w: 204, h: 108 };
  }

  if (phase === "broadcast_rule_lab" || phase === "broadcast_bonus_unsqueeze" || phase === "broadcast_bonus_no_copy") {
    if (node.id === "target_btc") return { ...node, w: 236, h: 136 };
    if (node.id === "vector_c" || node.id === "matrix_tc" || node.id === "singleton_scale") return { ...node, w: 196, h: 92 };
    if (node.id === "broadcast_rail_lab") return { ...node, w: 250, h: 124 };
    if (node.id === "ghost_expansion" || node.id === "plan_checker") return { ...node, w: 224, h: 104 };
  }

  if (phase === "broadcast_bias_add" || phase === "broadcast_position_add" || phase === "broadcast_mask_add" || phase === "broadcast_bonus_mask_value") {
    if (node.kind === "tensor" || node.kind === "attention") return { ...node, w: 238, h: 136 };
    if (node.kind === "parameter" || node.kind === "mask") return { ...node, w: 210, h: 94 };
    if (node.kind === "operation") return { ...node, w: 238, h: 112 };
    if (node.kind === "scalar") return { ...node, w: 224, h: 100 };
  }

  if (phase === "broadcast_trap_debugger") {
    if (node.id === "trap_cases") return { ...node, w: 234, h: 132 };
    if (node.kind === "operation") return { ...node, w: 236, h: 110 };
    if (node.kind === "scalar") return { ...node, w: 226, h: 104 };
  }

  if (phase === "broadcast_gauntlet") {
    if (node.id === "broadcast_contract_terminal") return { ...node, w: 286, h: 128 };
    return { ...node, w: 246, h: 124 };
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
  if (level.id === "1-1") {
    const challenge = chapter1TokenizationChallengeForPhase(phase);
    if (!challenge) return edges;
    const visibleNodeIds = new Set(nodes.map((node) => node.id));
    const visibleEdgeIds = new Set(challenge.edgeIds);
    return edges.filter((edge) => visibleEdgeIds.has(edge.id) && visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to));
  }
  if (level.id === "0-2" || level.id === "0-3" || level.id === "0-4") {
    const challenge =
      level.id === "0-2" ? chapter02ChallengeForPhase(phase) : level.id === "0-3" ? chapter03ChallengeForPhase(phase) : chapter04ChallengeForPhase(phase);
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
  if (level.id === "0-2" || level.id === "0-3" || level.id === "0-4" || level.id === "1-1") {
    const finalGauntletPhase =
      level.id === "0-2" ? "matmul_gauntlet" : level.id === "0-3" ? "transpose_gauntlet" : level.id === "0-4" ? "broadcast_gauntlet" : "tokenizer_gauntlet";
    const finalGauntletNodeId =
      level.id === "0-2" ? "matmul_tests" : level.id === "0-3" ? "gauntlet_result" : level.id === "0-4" ? "broadcast_gauntlet_result" : "tokenizer_gauntlet_result";
    const gauntletLabel = level.id === "0-2" ? "MatMul" : level.id === "0-3" ? "Transpose" : level.id === "0-4" ? "Broadcast" : "Tokenizer";
    const hints: CanvasActionHint[] =
      phase === finalGauntletPhase
        ? [
            {
              kind: "node",
              id: finalGauntletNodeId,
              icon: "run",
              tooltip: `Click or right-click: run the final ${gauntletLabel} Gauntlet`,
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

  if (level.id === "1-1") {
    const slotCorrect = (slotId: string) => {
      const slot = level.repair.slots.find((item) => item.id === slotId);
      return slot ? slot.correctTagIds.includes(assignments[slotId]) : false;
    };
    const stageReady = (phase: LevelPhase) => chapter1TokenizationSlotIds(phase).every(slotCorrect);
    const textReady = stageReady("token_raw_text_object");
    const typeReady = stageReady("token_type_gate_failure");
    const tokenizerReady = stageReady("token_tokenizer_socket");
    const idsReady = stageReady("token_token_id_contract");
    const splitReady = stageReady("token_boundary_cutter") && stageReady("token_split_comparison");
    const mergeReady = stageReady("token_merge_forge");
    const vocabReady = stageReady("token_vocab_lookup") && stageReady("token_address_lighting");
    const stableReady = stageReady("token_stable_id_test");
    const bufferReady = stageReady("token_buffer_build");
    const oovReady = stageReady("token_oov_failure") && stageReady("token_fallback_splitter");
    const specialReady = stageReady("token_special_token_injector");
    const padReady = stageReady("token_padding_builder");
    const budgetReady = stageReady("token_budget_gate");

    patchNode("raw_text_input", {
      subtitle: textReady ? "utf8 object inspected" : "raw utf8 string",
      checks: [check("text object", textReady ? "pass" : "warn", textReady ? "raw text is identified as non-numeric utf8" : "inspect raw text before model-side ports")]
    });
    patchNode("type_gate", {
      subtitle: typeReady ? "failure explained" : "numeric port guard",
      checks: [check("type contract", typeReady ? "pass" : "warn", typeReady ? "string input is rejected; int ids required" : "make the raw text failure explicit")]
    });
    patchNode("tokenizer_socket", {
      subtitle: tokenizerReady ? "socket installed" : "utf8 -> pieces",
      checks: [check("tokenizer boundary", tokenizerReady ? "pass" : "warn", tokenizerReady ? "raw text must pass through tokenizer" : "insert tokenizer and block bypass")]
    });
    patchNode("token_piece_stream", {
      subtitle: splitReady ? "ordered pieces" : tokenizerReady ? "piece stream open" : "waiting for tokenizer",
      checks: [
        check("pieces", tokenizerReady ? "pass" : "warn", tokenizerReady ? "tokenizer emits pieces" : "install tokenizer first"),
        check("boundaries", splitReady ? "pass" : "warn", splitReady ? "space, punctuation, and policy checks pass" : "finish split policy and boundary checks"),
        check("merge/symbols", mergeReady && stageReady("token_preserve_symbols") ? "pass" : "warn", mergeReady ? "merge rules are active; verify symbols remain visible" : "configure merges and symbol preservation")
      ]
    });
    patchNode("token_count_meter", {
      subtitle: stageReady("token_count_meter") ? "budget enforced" : "budget T <= 8",
      checks: [check("T budget", stageReady("token_count_meter") ? "pass" : "warn", stageReady("token_count_meter") ? "token count blocks overflow" : "count current pieces and enforce max T")]
    });
    patchNode("vocab_lookup_gate", {
      subtitle: vocabReady ? "pieces resolved" : "resolve pieces",
      checks: [check("vocab ids", vocabReady ? "pass" : "warn", vocabReady ? "pieces resolve to stable row ids or OOV" : "lookup every piece and flag missing rows")]
    });
    patchNode("token_id_emitter", {
      subtitle: idsReady || vocabReady ? "integer ids emitted" : "pieces -> int ids",
      checks: [check("id contract", idsReady || vocabReady ? "pass" : "warn", idsReady || vocabReady ? "integer token ids can index embedding rows" : "emit integer ids before embedding")]
    });
    patchNode("stable_id_checker", {
      subtitle: stableReady ? "deterministic" : "same input -> same ids",
      checks: [check("stable ids", stableReady ? "pass" : "warn", stableReady ? "same piece maps to same id across runs" : "prove repeated encode is stable")]
    });
    patchNode("token_buffer", {
      subtitle: padReady ? "token_ids[B,T] + mask ready" : bufferReady ? "token_ids[B,T]" : "batch buffer pending",
      checks: [
        check("B/T buffer", bufferReady ? "pass" : "warn", bufferReady ? "rectangular int[B,T] buffer built" : "build rows=B and columns=T"),
        check("padding", padReady ? "pass" : "warn", padReady ? "padding and attention_mask agree" : "finish padding and mask builder"),
        check("budget", budgetReady ? "pass" : "warn", budgetReady ? "max T is enforced before buffer output" : "apply token budget gate")
      ]
    });
    patchNode("oov_detector", {
      subtitle: stageReady("token_oov_failure") ? "OOV reported" : "find missing pieces",
      checks: [check("OOV", stageReady("token_oov_failure") ? "pass" : "warn", stageReady("token_oov_failure") ? "missing pieces are detected, blocked, and named" : "make OOV failure informative")]
    });
    patchNode("fallback_splitter", {
      subtitle: oovReady ? "fallback chain locked" : "subword / char / unk",
      checks: [check("fallback", oovReady ? "pass" : "warn", oovReady ? "OOV resolves deterministically through fallback chain" : "configure subword, char, and <unk> fallback")]
    });
    patchNode("special_token_injector", {
      subtitle: specialReady ? "special ids reserved" : "BOS / EOS / PAD / UNK",
      checks: [check("special tokens", specialReady ? "pass" : "warn", specialReady ? "BOS/EOS/PAD/UNK contract is explicit" : "inject BOS/EOS and reserve special ids")]
    });
    patchNode("attention_mask_builder", {
      subtitle: padReady ? "mask matches pad" : "1 content / 0 pad",
      checks: [check("mask", padReady ? "pass" : "warn", padReady ? "attention_mask mirrors pad positions" : "build mask from padded token ids")]
    });
    patchNode("budget_gate", {
      subtitle: budgetReady ? "max T guarded" : "max T enforcement",
      checks: [check("budget", budgetReady ? "pass" : "warn", budgetReady ? "over-budget texts are truncated or rejected with trace" : "choose overflow policy and trace")]
    });
    patchNode("tokenizer_contract_terminal", {
      subtitle: budgetReady ? "ready for gauntlet" : "autograder checks",
      checks: [check("pipeline", budgetReady ? "pass" : "warn", budgetReady ? "visible stages ready; run tokenizer gauntlet" : "complete visible stages before final tests")]
    });
    return nodes;
  }

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
    const slotCorrect = (slotId: string) => {
      const slot = level.repair.slots.find((item) => item.id === slotId);
      return slot ? slot.correctTagIds.includes(assignments[slotId]) : false;
    };
    const slotsCorrect = (slotIds: string[]) => slotIds.every(slotCorrect);
    const matrixReady = slotsCorrect(chapter03SlotIds("transpose_matrix_flip"));
    const innerReady = slotsCorrect(chapter03SlotIds("transpose_inner_dim_repair"));
    const higherRankReady = slotsCorrect(chapter03SlotIds("transpose_higher_rank_axis_swap"));
    const singleReady = slotsCorrect(chapter03SlotIds("transpose_single_head_qk"));
    const multiReady = slotsCorrect(chapter03SlotIds("transpose_multi_head_trap"));
    const debugReady = slotsCorrect(chapter03SlotIds("transpose_trap_debugger"));
    const gauntletReady = slotsCorrect(chapter03SlotIds("transpose_gauntlet"));

    patchNode("transpose_2d", {
      shape: slotCorrect("matrix_flip_switch") ? "[R,C] -> [C,R]" : "[R,C] -> ?",
      subtitle: slotCorrect("matrix_flip_switch") ? "2D transpose configured" : "2D switch off",
      checks: [check("matrix flip", matrixReady ? "pass" : "warn", matrixReady ? "shape and mirrored index mapping pass" : "configure real transpose, output shape, and index mapping")]
    });
    patchNode("matrix_t_plate", {
      shape: slotCorrect("matrix_flip_shape") ? "[C,R]" : "[?,?]",
      subtitle: matrixReady ? "values mirrored" : "waiting for values",
      checks: [
        check("shape", slotCorrect("matrix_flip_shape") ? "pass" : "warn", slotCorrect("matrix_flip_shape") ? "flipped shape is [C,R]" : "set output shape to [C,R]"),
        check("index", slotCorrect("matrix_flip_index") ? "pass" : "warn", slotCorrect("matrix_flip_index") ? "A[i,j] maps to A^T[j,i]" : "prove value movement, not relabeling")
      ]
    });
    patchNode("index_checker", {
      subtitle: matrixReady ? "passed" : "A[i,j] -> A^T[j,i]",
      checks: [check("numeric mapping", matrixReady ? "pass" : "warn", matrixReady ? "all visible cells mirror correctly" : "index mirror still incomplete")]
    });

    patchNode("matmul_b_bad", {
      subtitle: slotCorrect("inner_dim_fault") ? "fault identified" : "stored wrong way",
      checks: [check("orientation", slotCorrect("inner_dim_fault") ? "pass" : "warn", slotCorrect("inner_dim_fault") ? "B is stored as [P,N]" : "identify whether A or B faces the wrong way")]
    });
    patchNode("transpose_b_repair", {
      shape: slotCorrect("inner_dim_fix") ? "[P,N] -> [N,P]" : "[P,N] -> ?",
      subtitle: slotCorrect("inner_dim_fix") ? "T(B) inserted" : "repair inner dim",
      checks: [check("repair", slotCorrect("inner_dim_fix") ? "pass" : "warn", slotCorrect("inner_dim_fix") ? "B^T exposes N for the inner dimension" : "transpose B, not A")]
    });
    patchNode("matmul_repair_gate", {
      shape: slotCorrect("inner_dim_fix") ? "[M,N]@[N,P]" : "[M,N]@[?,?]",
      subtitle: innerReady ? "inner dims locked" : "inner dim blocked",
      checks: [check("inner dims", innerReady ? "pass" : "warn", innerReady ? "N aligns with N and output can form" : "repair right operand orientation")]
    });
    patchNode("matmul_repair_out", {
      shape: slotCorrect("inner_dim_output") ? "[M,P]" : "[?,?]",
      subtitle: innerReady ? "reference ready" : "waiting for reference",
      checks: [check("output", slotCorrect("inner_dim_output") ? "pass" : "warn", slotCorrect("inner_dim_output") ? "N is consumed; P is generated" : "set output contract to [M,P]")]
    });
    patchNode("reference_checker_03", {
      subtitle: innerReady ? "allclose armed" : "allclose gate",
      checks: [check("numeric", slotCorrect("inner_dim_numeric") ? "pass" : "warn", slotCorrect("inner_dim_numeric") ? "A @ B.T matches reference" : "shape is not enough; attach allclose")]
    });

    patchNode("axis_swap_switch", {
      shape: slotCorrect("rank3_swap_axes") ? "swap(-2,-1)" : "last-two axes",
      subtitle: higherRankReady ? "T/D swap locked" : "swap(-2,-1) not set",
      checks: [check("axis config", higherRankReady ? "pass" : "warn", higherRankReady ? "last two axes swap while prefix axes stay fixed" : "configure swap(-2,-1) and reject reverse all axes")]
    });
    patchNode("carry_axis_lock", {
      subtitle: slotCorrect("rank4_carry_axes") ? "B/H protected" : "carry axes protected",
      checks: [
        check("B axis", slotCorrect("rank3_carry_axis") ? "pass" : "warn", slotCorrect("rank3_carry_axis") ? "B remains fixed in rank-3 case" : "preserve B in [B,T,D]"),
        check("B/H axes", slotCorrect("rank4_carry_axes") ? "pass" : "warn", slotCorrect("rank4_carry_axes") ? "B/H remain fixed in rank-4 case" : "preserve B/H in [B,H,T,D]")
      ]
    });
    patchNode("higher_rank_checker", {
      subtitle: higherRankReady ? "rank cases pass" : "rank 3 + rank 4",
      checks: [check("not reverse", slotCorrect("no_full_reverse") ? "pass" : "warn", slotCorrect("no_full_reverse") ? "full axis reversal rejected" : "mark transpose trap as last-two-axis swap only")]
    });

    patchNode("k_single_transpose", {
      shape: slotCorrect("single_k_transpose") ? "[T,D] -> [D,T]" : "[T,D] -> ?",
      subtitle: slotCorrect("single_k_transpose") ? "K^T configured" : "not configured",
      checks: [check("K transpose", slotCorrect("single_k_transpose") ? "pass" : "warn", slotCorrect("single_k_transpose") ? "K token axis becomes columns" : "transpose K, not Q")]
    });
    patchNode("single_qk_matmul", {
      shape: slotCorrect("single_k_transpose") ? "[T,D]@[D,T]" : "[T,D]@[?,?]",
      subtitle: singleReady ? "inner dims locked" : "blocked",
      checks: [check("single QK", singleReady ? "pass" : "warn", singleReady ? "single-head score contract passes" : "build [T,D] @ [D,T]")]
    });
    patchNode("single_scores", {
      shape: slotCorrect("single_score_shape") ? "[T,T]" : "[?,?]",
      subtitle: singleReady ? "query x key board" : "token x token",
      checks: [
        check("shape", slotCorrect("single_score_shape") ? "pass" : "warn", slotCorrect("single_score_shape") ? "scores shape is [T,T]" : "target score board is [T,T]"),
        check("semantics", slotCorrect("single_score_semantics") ? "pass" : "warn", slotCorrect("single_score_semantics") ? "rows=query tokens, cols=key tokens" : "label the two T axes")
      ]
    });

    patchNode("k_heads_transpose", {
      shape: slotCorrect("multi_k_transpose") ? "[B,H,D,T]" : "off",
      subtitle: slotCorrect("multi_k_transpose") ? "K last axes swapped" : "not configured",
      checks: [
        check("K transpose", slotCorrect("multi_k_transpose") ? "pass" : "warn", slotCorrect("multi_k_transpose") ? "K becomes [B,H,D,T]" : "transpose K's last two axes"),
        check("carry axes", slotCorrect("multi_carry_axes") ? "pass" : "warn", slotCorrect("multi_carry_axes") ? "B/H stay fixed" : "do not move B/H")
      ]
    });
    patchNode("multi_qk_matmul", {
      shape: slotCorrect("multi_k_transpose") && slotCorrect("multi_carry_axes") ? "[T,D]@[D,T]" : "[T,D]@[?,?]",
      subtitle: multiReady ? "per-head inner dims locked" : "blocked",
      checks: [check("multi QK", multiReady ? "pass" : "warn", multiReady ? "every B/H layer produces a T x T board" : "preserve B/H and transpose K")]
    });
    patchNode("multi_scores", {
      shape: slotCorrect("multi_score_shape") ? "[B,H,T,T]" : "[B,H,?,?]",
      subtitle: multiReady ? "score stack repaired" : "attention board stack",
      checks: [check("score stack", slotCorrect("multi_score_shape") ? "pass" : "warn", slotCorrect("multi_score_shape") ? "scores[B,H,T,T]" : "target is [B,H,T,T]")]
    });

    patchNode("trace_inspector", {
      subtitle: slotCorrect("debug_operand_order") && slotCorrect("debug_axis_contract") ? "axis contract checked" : "axis contract first",
      checks: [
        check("operand order", slotCorrect("debug_operand_order") ? "pass" : "warn", slotCorrect("debug_operand_order") ? "Q stays left, K^T stays right" : "check operand order first"),
        check("axis contract", slotCorrect("debug_axis_contract") ? "pass" : "warn", slotCorrect("debug_axis_contract") ? "semantic axes inspected before sizes" : "inspect axis semantics")
      ]
    });
    patchNode("cell_source_checker", {
      subtitle: slotCorrect("debug_numeric_trap") ? "cell source traced" : "scores[tq,tk]",
      checks: [check("numeric trap", slotCorrect("debug_numeric_trap") ? "pass" : "warn", slotCorrect("debug_numeric_trap") ? "T==D trap handled by provenance" : "trace one score cell")]
    });
    patchNode("trap_patch", {
      subtitle: slotCorrect("debug_patch") ? "patch complete" : "waiting",
      checks: [check("patch rules", slotCorrect("debug_patch") ? "pass" : "warn", slotCorrect("debug_patch") ? "order, K^T, and carry axes are fixed" : "record all required fixes")]
    });
    patchNode("trap_reference", {
      subtitle: debugReady ? "debug allclose ready" : "debug allclose",
      checks: [check("allclose", debugReady ? "pass" : "warn", debugReady ? "debug cases can run reference checks" : "finish debugger patch")]
    });

    patchNode("gauntlet_cases", {
      subtitle: gauntletReady ? "4 hidden cases armed" : "hidden QK boards",
      checks: [
        check("A", slotCorrect("gauntlet_standard") ? "pass" : "warn", slotCorrect("gauntlet_standard") ? "standard case armed" : "arm standard QK^T case"),
        check("B", slotCorrect("gauntlet_carry") ? "pass" : "warn", slotCorrect("gauntlet_carry") ? "carry-axis case armed" : "arm carry-axis variant")
      ]
    });
    patchNode("hidden_reference", {
      subtitle: slotCorrect("gauntlet_td_equal") ? "T==D trap covered" : "allclose + shape",
      checks: [check("T==D", slotCorrect("gauntlet_td_equal") ? "pass" : "warn", slotCorrect("gauntlet_td_equal") ? "equal-size numeric trap covered" : "arm T==D hidden test")]
    });
    patchNode("gauntlet_result", {
      subtitle: gauntletReady ? "ready to run" : "run final tests",
      checks: [check("operand order", slotCorrect("gauntlet_operand_order") ? "pass" : "warn", slotCorrect("gauntlet_operand_order") ? "operand-order trap covered" : "arm operand-order hidden test")]
    });
  }

  if (level.id === "0-4") {
    const slotCorrect = (slotId: string) => {
      const slot = level.repair.slots.find((item) => item.id === slotId);
      return slot ? slot.correctTagIds.includes(assignments[slotId]) : false;
    };
    const slotState = (slotId: string): CheckState => (slotCorrect(slotId) ? "pass" : assignments[slotId] ? "fail" : "warn");
    const slotCheck = (slotId: string, label: string, passDetail: string, pendingDetail: string) =>
      check(label, slotState(slotId), slotCorrect(slotId) ? passDetail : pendingDetail);
    const slotsCorrect = (slotIds: string[]) => slotIds.every(slotCorrect);
    const addReady = slotsCorrect(chapter04SlotIds("broadcast_add_cell"));
    const sameReady = slotsCorrect(chapter04SlotIds("broadcast_same_shape_add"));
    const ruleReady = slotsCorrect(chapter04SlotIds("broadcast_rule_lab"));
    const biasReady = slotsCorrect(chapter04SlotIds("broadcast_bias_add"));
    const positionReady = slotsCorrect(chapter04SlotIds("broadcast_position_add"));
    const maskReady = slotsCorrect(chapter04SlotIds("broadcast_mask_add"));
    const trapReady = slotsCorrect(chapter04SlotIds("broadcast_trap_debugger"));
    const gauntletReady = slotsCorrect(chapter04SlotIds("broadcast_gauntlet"));

    patchNode("scalar_a", {
      subtitle: slotCorrect("cell_left_input") ? "connected to Add left" : "left cell",
      checks: [slotCheck("cell_left_input", "A input", "Scalar A feeds the Add Gate.", "connect Scalar A into Add Gate")]
    });
    patchNode("scalar_b", {
      subtitle: slotCorrect("cell_right_input") ? "connected to Add right" : "right cell",
      checks: [slotCheck("cell_right_input", "B input", "Scalar B feeds the Add Gate.", "connect Scalar B into Add Gate")]
    });
    patchNode("add_gate_cell", {
      subtitle: addReady ? "cell add locked" : "cell inputs open",
      checks: [
        slotCheck("cell_left_input", "left", "A[] connected", "left input missing"),
        slotCheck("cell_right_input", "right", "B[] connected", "right input missing")
      ]
    });
    patchNode("output_cell", {
      subtitle: addReady ? "7.0 computed" : "waiting for trace",
      sample: addReady ? ["out[] = 2.0 + 5.0", "computed by Add Gate"] : ["expected: 7.0", "must be computed, not typed manually"],
      checks: [slotCheck("cell_trace_exact", "trace", "output source cells verified", "trace the output source cells")]
    });
    patchNode("cell_trace", {
      subtitle: slotCorrect("cell_trace_exact") ? "source proof captured" : "source proof",
      checks: [slotCheck("cell_trace_exact", "cell trace", "out[] <- A[] + B[]", "prove output source cells")]
    });

    patchNode("same_add_gate", {
      subtitle: sameReady ? "same-shape add locked" : "same-shape gate",
      shape: slotCorrect("same_left_shape") && slotCorrect("same_right_shape") ? "[T,C]+[T,C]" : "[?,?]+[?,?]",
      checks: [
        slotCheck("same_left_shape", "left shape", "left input is [T,C]", "set left operand to [T,C]"),
        slotCheck("same_right_shape", "right shape", "right input is [T,C]", "set right operand to [T,C]")
      ]
    });
    patchNode("same_out", {
      subtitle: slotCorrect("same_output_shape") ? "shape preserved" : "waiting",
      shape: slotCorrect("same_output_shape") ? "[T,C]" : "[?,?]",
      checks: [slotCheck("same_output_shape", "output", "elementwise add preserves [T,C]", "output shape must remain [T,C]")]
    });
    patchNode("same_trace", {
      subtitle: slotCorrect("same_trace_cell") ? "matching coordinates traced" : "cell provenance",
      checks: [slotCheck("same_trace_cell", "grid trace", "out[t,c] uses matching source cells", "trace one output cell")]
    });

    patchNode("broadcast_rail_lab", {
      subtitle: ruleReady ? "broadcast plans locked" : "plan incomplete",
      shape: ruleReady ? "[B,T,C] compatible" : "[B][T][C]",
      checks: [
        slotCheck("rule_vector_c", "[C]", "[C] right-aligns under C", "right-align [C] to channel axis"),
        slotCheck("rule_matrix_tc", "[T,C]", "missing B axis inserted", "insert leading singleton B"),
        slotCheck("rule_singleton", "[1,T,1]", "singleton axes expand", "expand only length-1 axes")
      ]
    });
    patchNode("ghost_expansion", {
      subtitle: slotCorrect("rule_logical_view") ? "logical view, no copy" : "logical view",
      checks: [slotCheck("rule_logical_view", "memory", "broadcast stays logical", "use Ghost Expansion instead of repeat")]
    });
    patchNode("plan_checker", {
      subtitle: ruleReady ? "3 cases pass" : "3 cases",
      checks: [check("broadcast lab", ruleReady ? "pass" : "warn", ruleReady ? "all broadcast rule lab cases pass" : "finish [C], [T,C], singleton, and logical view")]
    });

    patchNode("bias_o", {
      subtitle: slotCorrect("bias_axis_o") ? "mounted on O axis" : "channel offsets",
      checks: [slotCheck("bias axis", "O axis", "bias belongs to O", "align bias to output feature axis O")]
    });
    patchNode("bias_rail", {
      subtitle: biasReady ? "B/T broadcast locked" : "O axis not mounted",
      shape: slotCorrect("bias_expand_bt") ? "[1,1,O] -> [B,T,O]" : "[B,T,O] + [O]",
      checks: [
        slotCheck("bias_axis_o", "axis", "bias aligned to O", "mount bias on O axis"),
        slotCheck("bias_expand_bt", "expand", "bias expands over B/T", "broadcast over B and T")
      ]
    });
    patchNode("bias_add_gate", {
      subtitle: biasReady ? "additive bias ready" : "bias add",
      checks: [check("operation", biasReady ? "pass" : "warn", biasReady ? "projected[b,t,o] + bias[o]" : "finish bias axis, expansion, and trace")]
    });
    patchNode("bias_out", {
      subtitle: biasReady ? "reference-ready output" : "waiting",
      checks: [check("output", biasReady ? "pass" : "warn", biasReady ? "output stays [B,T,O]" : "bias broadcast contract incomplete")]
    });
    patchNode("bias_reference", {
      subtitle: slotCorrect("bias_trace") ? "allclose trace armed" : "allclose",
      checks: [slotCheck("bias_trace", "cell trace", "out[b,t,o] uses bias[o]", "trace the output bias source")]
    });

    patchNode("pos_sheet", {
      subtitle: slotCorrect("position_align_tc") ? "T/C axes locked" : "T by C",
      checks: [slotCheck("position_align_tc", "axes", "pos_emb owns T and C", "align position sheet to T and C")]
    });
    patchNode("position_rail", {
      subtitle: positionReady ? "B broadcast locked" : "T/C not locked",
      shape: slotCorrect("position_broadcast_b") ? "[1,T,C] -> [B,T,C]" : "[B,T,C] + [T,C]",
      checks: [
        slotCheck("position_align_tc", "T/C", "position axes preserved", "preserve T/C axes"),
        slotCheck("position_broadcast_b", "B", "B is the only missing axis", "broadcast over B only")
      ]
    });
    patchNode("position_add_gate", {
      subtitle: positionReady ? "position add ready" : "position add",
      checks: [check("operation", positionReady ? "pass" : "warn", positionReady ? "hidden[b,t,c] uses pos_emb[t,c]" : "finish position axes, B broadcast, and trace")]
    });
    patchNode("hidden_out", {
      subtitle: positionReady ? "T variation preserved" : "waiting",
      checks: [check("output", positionReady ? "pass" : "warn", positionReady ? "hidden stays [B,T,C]" : "position broadcast incomplete")]
    });
    patchNode("position_trace", {
      subtitle: slotCorrect("position_keep_t") ? "not bias[C]" : "not bias",
      checks: [slotCheck("position_keep_t", "semantic trace", "position-specific values retained", "prove pos_emb did not collapse into bias")]
    });

    patchNode("mask_rail", {
      subtitle: maskReady ? "Tq/Tk + B/H locked" : "query/key alignment",
      shape: slotCorrect("mask_expand_bh") ? "[1,1,Tq,Tk] -> [B,H,Tq,Tk]" : "[B,H,Tq,Tk] + mask",
      checks: [
        slotCheck("mask_tqtk_axes", "Tq/Tk", "mask axes match score axes", "align query/key token axes"),
        slotCheck("mask_expand_bh", "B/H", "mask broadcasts over B/H", "broadcast singleton B/H axes")
      ]
    });
    patchNode("mask_add_gate", {
      subtitle: slotCorrect("mask_additive") ? "additive pre-softmax" : "mask add",
      checks: [slotCheck("mask_additive", "operation", "scores + negative mask", "use additive negative mask, not multiplication")]
    });
    patchNode("masked_scores", {
      subtitle: maskReady ? "future cells blocked" : "waiting",
      checks: [check("masked scores", maskReady ? "pass" : "warn", maskReady ? "illegal future cells receive -1e9" : "mask broadcast contract incomplete")]
    });
    patchNode("illegal_cell_checker", {
      subtitle: slotCorrect("mask_trace") ? "future cell traced" : "future cell",
      checks: [slotCheck("mask_trace", "illegal cell", "future attention cell is blocked", "trace q,k orientation on one future cell")]
    });

    patchNode("trap_inspector", {
      subtitle: slotCorrect("trap_right_align") ? "right alignment checked" : "debug order",
      checks: [slotCheck("trap_right_align", "right align", "raw broadcast alignment checked", "start from rank and trailing-axis alignment")]
    });
    patchNode("semantic_warning", {
      subtitle: slotCorrect("trap_bt_swap") ? "B/T trap caught" : "semantic trap",
      checks: [slotCheck("trap_bt_swap", "B/T swap", "equal sizes no longer hide semantics", "catch B/T semantic swaps")]
    });
    patchNode("trace_probe_04", {
      subtitle: slotCorrect("trap_pos_bias") ? "pos-as-bias caught" : "cell source",
      checks: [slotCheck("trap_pos_bias", "pos bias", "position sheet did not collapse", "trace pos_emb[T,C] versus bias[C]")]
    });
    patchNode("trap_patch", {
      subtitle: slotCorrect("trap_no_repeat") ? "logical patch" : "repair rules",
      checks: [slotCheck("trap_no_repeat", "no repeat", "logical broadcast view preserved", "avoid materialized repeat")]
    });
    patchNode("trap_reference_04", {
      subtitle: trapReady ? "debug reference ready" : "reference",
      checks: [check("trap debugger", trapReady ? "pass" : "warn", trapReady ? "right-align, semantic traps, trace, and no-repeat checks pass" : "finish every trap debugger repair")]
    });

    patchNode("broadcast_gauntlet_cases", {
      subtitle: gauntletReady ? "6 hidden cases armed" : "hidden cases",
      checks: [
        slotCheck("gauntlet_bias", "A", "bias hidden case armed", "arm bias hidden case"),
        slotCheck("gauntlet_position", "B", "position hidden case armed", "arm position hidden case"),
        slotCheck("gauntlet_singleton", "C", "singleton hidden case armed", "arm singleton hidden case")
      ]
    });
    patchNode("broadcast_contract_terminal", {
      subtitle: gauntletReady ? "contract generalized" : "contract terminal",
      checks: [
        slotCheck("gauntlet_mask", "D", "mask hidden case armed", "arm mask hidden case"),
        slotCheck("gauntlet_equal_dim", "E", "equal-dim trap covered", "arm equal-dimension trap")
      ]
    });
    patchNode("hidden_broadcast_reference", {
      subtitle: gauntletReady ? "reference allclose armed" : "hidden reference",
      checks: [slotCheck("gauntlet_semantic_trace", "F", "semantic trace armed", "arm semantic trace case")]
    });
    patchNode("broadcast_gauntlet_result", {
      subtitle: gauntletReady ? "ready to run" : "run final tests",
      checks: [check("hidden tests", gauntletReady ? "pass" : "warn", gauntletReady ? "Broadcast Gauntlet can run" : "complete all six hidden case slots")]
    });
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

function taskHintSlotForTask(level: BootcampLevel, phase: LevelPhase, state: LevelRepairState, taskId: string) {
  const slots = visibleSlotsForPhase(level, phase);
  const directSlot = slots.find((slot) => slot.id === taskId);
  if (directSlot) return directSlot;

  if (taskId === "probe_evidence") {
    return (
      slots.find((slot) => !state.observations.some((observation) => observation.slotId === slot.id)) ??
      slots.find((slot) => !areAssignmentsCorrect(level, state.assignments, [slot.id])) ??
      slots[0]
    );
  }

  return undefined;
}

function buildTaskHintOperation(level: BootcampLevel, phase: LevelPhase, state: LevelRepairState, slot: RepairSlot) {
  const correctTags = slot.correctTagIds.map((tagId) => level.repair.tags.find((tag) => tag.id === tagId)).filter(Boolean) as RepairTag[];
  const firstCorrectTag = correctTags[0];
  const tagSummary = correctTags.map((tag) => tag.shortLabel).join(" / ");
  const assignedTagId = state.assignments[slot.id];
  const passed = Boolean(assignedTagId && slot.correctTagIds.includes(assignedTagId));
  const observed = state.observations.some((observation) => observation.slotId === slot.id);
  const probeId = level.id === "0-1" && phase === "hidden_contract_repair" && !observed ? defaultProbeForSlot(level, phase, slot) : undefined;
  const probe = probeId ? level.repair.probes.find((item) => item.id === probeId) : undefined;

  if (probe) {
    return {
      activeProbeId: probe.id,
      activeTagId: undefined,
      detail: `Probe here with ${probe.label}; then choose ${tagSummary}.`
    };
  }

  if (passed) {
    return {
      activeProbeId: undefined,
      activeTagId: undefined,
      detail: `${tagSummary || slot.expected} is already placed here.`
    };
  }

  if (level.id === "0-1" && phase === "tensor_object" && firstCorrectTag) {
    return {
      activeProbeId: undefined,
      activeTagId: firstCorrectTag.id,
      detail: `Drag ${firstCorrectTag.shortLabel} into Tensor Inspector.`
    };
  }

  if (firstCorrectTag?.category === "data" || firstCorrectTag?.category === "consumer") {
    return {
      activeProbeId: undefined,
      activeTagId: firstCorrectTag.id,
      detail: `Connect ${tagSummary} to this slot.`
    };
  }

  return {
    activeProbeId: undefined,
    activeTagId: firstCorrectTag?.id,
    detail: firstCorrectTag ? `Pick ${tagSummary}; click this slot.` : `Place ${slot.expected} in this slot.`
  };
}

function activeToolLabel(level: BootcampLevel, state: LevelRepairState) {
  const tag = level.repair.tags.find((item) => item.id === state.activeTagId);
  if (tag) return `tag ${tag.shortLabel}`;
  const probe = level.repair.probes.find((item) => item.id === state.activeProbeId);
  if (probe) return probe.label;
  return "canvas context";
}

function levelsForCampaignChapter(chapter: CampaignChapterDef) {
  const levelIds = new Set(chapter.levelIds);
  return bootcampLevels.filter((level) => levelIds.has(level.id));
}

function campaignChapterForLevel(level: BootcampLevel) {
  return campaignChapterDefs.find((chapter) => chapter.levelIds.includes(level.id)) ?? campaignChapterDefs[0];
}

function chapterIdForLevel(level: BootcampLevel) {
  return campaignChapterForLevel(level).id;
}

function levelDisplayBadge(level: BootcampLevel) {
  if (level.id === "1-1") return "1";
  return level.id;
}

function levelDisplayTitle(level: BootcampLevel) {
  const chapter = campaignChapterForLevel(level);
  return `${chapter.code.replace("Chapter ", "")} / ${level.title}`;
}

function repairKindLabel(kind: BootcampLevel["repair"]["kind"]) {
  if (kind === "axis_labels") return "Axis Tags";
  if (kind === "matmul_gate") return "Weight Plate";
  if (kind === "transpose_switch") return "Transpose Switch";
  if (kind === "tokenizer_pipeline") return "Tokenizer Pipeline";
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
                : challenge.phase === "hidden_test_gauntlet" ||
                  challenge.phase === "matmul_gauntlet" ||
                  challenge.phase === "transpose_gauntlet" ||
                  challenge.phase === "broadcast_gauntlet" ||
                  challenge.phase === "tokenizer_gauntlet"
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
  runLabel,
  onSelectTag,
  onSelectProbe,
  onRunTests
}: {
  level: BootcampLevel;
  repairState: LevelRepairState;
  phase: LevelPhase;
  runLabel: string;
  onSelectTag: (tagId: string) => void;
  onSelectProbe: (probeId: string) => void;
  onRunTests?: () => void;
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
      {onRunTests ? (
        <div className="paletteActions">
          <button className="runButton paletteRunButton" type="button" onClick={onRunTests}>
            <Play size={16} />
            {runLabel}
          </button>
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

function Chapter1GroupLevelList({
  level,
  repairState,
  phase,
  result,
  selected,
  expandedGroups,
  onSelect,
  onToggleGroup
}: {
  level: BootcampLevel;
  repairState: LevelRepairState;
  phase: LevelPhase;
  result?: BootcampResult;
  selected: boolean;
  expandedGroups: Record<string, boolean>;
  onSelect: () => void;
  onToggleGroup: (groupCode: string, currentlyExpanded: boolean) => void;
}) {
  const challenges = challengesForLevel(level);
  const challengeByPhase = new Map(challenges.map((challenge) => [challenge.phase, challenge]));
  const challengeDone = (challenge: Chapter01ChallengeDef) => isChallengeDone(level, repairState, challenge, result);

  return (
    <>
      {chapter1TokenizationChallengeGroups.map((group) => {
        const groupChallenges = group.phases.map((groupPhase) => challengeByPhase.get(groupPhase)).filter(Boolean) as Chapter01ChallengeDef[];
        const doneCount = groupChallenges.filter(challengeDone).length;
        const groupDone = doneCount === groupChallenges.length;
        const groupActive = selected && (groupChallenges.some((challenge) => challenge.phase === phase) || (phase === "completed" && group.code === "1-X"));
        const expanded = expandedGroups[group.code] ?? groupActive;
        return (
          <div key={group.code} className={`levelGroup chapter1LevelGroup ${expanded ? "expanded" : ""}`}>
            <button
              className={`levelItem chapter1GroupItem ${groupActive ? "active" : ""} ${groupDone ? "pass" : "pending"} collapsible`}
              aria-expanded={expanded}
              title={`${group.code} ${group.title}: ${expanded ? "收起阶段列表" : "展开阶段列表"}`}
              onClick={() => {
                onSelect();
                onToggleGroup(group.code, expanded);
              }}
            >
              <span className="levelId">{group.code}</span>
              <span>
                <b>{group.title}</b>
                <small>{group.brief}</small>
              </span>
              <span className={`levelExpandIcon ${expanded ? "open" : ""}`} aria-hidden="true">
                <ChevronDown size={16} />
              </span>
              <StateIcon state={groupDone ? "pass" : groupActive ? "warn" : "warn"} />
            </button>
            {expanded ? (
              <div className="challengeRail chapter1StageRail">
                {groupChallenges.map((challenge) => {
                  const active = phase === challenge.phase;
                  const done = challengeDone(challenge);
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
            ) : null}
          </div>
        );
      })}
    </>
  );
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
  const challengeDone = (challenge: Chapter01ChallengeDef) => isChallengeDone(level, repairState, challenge, result);

  if (level.id === "1-1") {
    const challengeByPhase = new Map(challenges.map((challenge) => [challenge.phase, challenge]));
    return (
      <div className="challengeRail grouped">
        {chapter1TokenizationChallengeGroups.map((group) => {
          const groupChallenges = group.phases.map((groupPhase) => challengeByPhase.get(groupPhase)).filter(Boolean) as Chapter01ChallengeDef[];
          const doneCount = groupChallenges.filter(challengeDone).length;
          const groupDone = doneCount === groupChallenges.length;
          const groupActive = groupChallenges.some((challenge) => challenge.phase === phase);
          const expanded = groupActive || (phase === "completed" && group.code === "1-X");
          return (
            <section key={group.code} className={`challengeGroupBlock ${groupActive ? "active" : ""} ${groupDone ? "done" : ""}`}>
              <div className="challengeGroupHeader">
                <StateIcon state={groupDone ? "pass" : groupActive ? "warn" : "warn"} />
                <span className="challengeGroupCode">{group.code}</span>
                <span className="challengeGroupTitle">
                  <b>{group.title}</b>
                  <small>{group.brief}</small>
                </span>
                <code>{doneCount}/{groupChallenges.length}</code>
              </div>
              {expanded ? (
                <div className="challengeGroupSteps">
                  {groupChallenges.map((challenge) => {
                    const active = phase === challenge.phase;
                    const done = challengeDone(challenge);
                    return (
                      <div key={challenge.code} className={`challengeStep compact ${active ? "active" : ""} ${done ? "done" : ""}`}>
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
              ) : null}
            </section>
          );
        })}
      </div>
    );
  }

  return (
    <div className="challengeRail">
      {challenges.map((challenge) => {
        const active = phase === challenge.phase;
        const done = challengeDone(challenge);
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

function isChallengeDone(level: BootcampLevel, repairState: LevelRepairState, challenge: Chapter01ChallengeDef, result?: BootcampResult) {
  if (challenge.phase === "hidden_contract_repair") {
    return areSlotsCorrect(level, repairState, challenge.slotIds) && chapter01ProbeEvidenceReady(repairState.observations);
  }
  if (
    challenge.phase === "hidden_test_gauntlet" ||
    challenge.phase === "matmul_gauntlet" ||
    challenge.phase === "transpose_gauntlet" ||
    challenge.phase === "broadcast_gauntlet" ||
    challenge.phase === "tokenizer_gauntlet"
  ) {
    return Boolean(result?.passed);
  }
  return areSlotsCorrect(level, repairState, challenge.slotIds);
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
  hintedTaskId,
  result,
  onTaskHint,
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
  hintedTaskId?: string;
  result?: BootcampResult;
  onTaskHint: (taskId: string) => void;
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
              <div
                key={item.id}
                className={`taskRow ${item.state} ${item.active ? "active" : ""} ${hintedTaskId === item.id ? "hintTarget" : ""}`}
                role="button"
                tabIndex={0}
                title="Double-click to highlight the matching canvas operation"
                onDoubleClick={() => onTaskHint(item.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onTaskHint(item.id);
                  }
                }}
              >
                <input type="checkbox" checked={item.state === "pass"} readOnly tabIndex={-1} aria-hidden="true" />
                <span>
                  <b>{item.label}</b>
                  <small>{item.value}</small>
                </span>
              </div>
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
