import { useEffect, useLayoutEffect, useMemo, useRef, useState, type DragEvent as ReactDragEvent } from "react";
import { AlertTriangle, Boxes, CheckCircle2, Circle, Cpu, Eye, Maximize2, Minimize2, Pause, Play, RotateCcw, Search, Wrench } from "lucide-react";
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
  tagCategories: NonNullable<RepairTag["category"]>[];
};

type Chapter01StageKnowledge = CanvasStageKnowledge & {
  inspectorNotes: string[];
  failureLesson: string[];
  debrief: string;
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
    concept: "Rank 表示 tensor 有几个可以索引的方向。",
    tool: "Rank Scanner",
    mission: "给 scalar / vector / matrix / block 贴上 rank 0 / 1 / 2 / 3。",
    visual: "rank_axes",
    inspectorNotes: ["scalar 没有轴，rank = 0。", "vector 是 1 条轴，matrix 是 2 条轴。", "一叠矩阵是 3 条轴。"],
    failureLesson: ["Rank mismatch。", "Rank 只表示轴的数量，不表示数值大小。", "重新观察对象有几个独立方向。"],
    debrief: "Rank 说明 tensor 有几个轴。"
  },
  shape_caliper: {
    code: "0-1C",
    title: "Shape：tensor 的结构尺寸",
    concept: "Shape 是每个轴的长度，按 axis order 排列。",
    tool: "Shape Caliper",
    mission: "测量 Axis 0 / 1 / 2，把 shape 修成 [2,4,8]。",
    visual: "shape_caliper",
    inspectorNotes: ["Axis 0 的长度放在第 1 个位置。", "Axis 1 的长度放在第 2 个位置。", "Axis 2 的长度放在第 3 个位置。"],
    failureLesson: ["Shape order mismatch。", "Shape 的数字顺序必须跟 axis 顺序一致。", "把量尺结果按 Axis 0、1、2 依次放入槽位。"],
    debrief: "Shape 记录每条轴的长度。"
  },
  semantic_gap: {
    code: "0-1D",
    title: "Shape 不等于语义",
    concept: "float32[2,4,8] 只说明大小，不说明每个轴的用途。",
    tool: "Semantic Inspector",
    mission: "标记 shape 已知但轴语义缺失，观察下游为什么不能运行。",
    visual: "semantic_gap",
    inspectorNotes: ["shape known 不代表 contract ready。", "下游模块需要知道哪个轴可被消费。", "这是合同缺失，不是数值错误。"],
    failureLesson: ["Connection blocked。", "下游模块不只需要 rank 和长度，还需要 axis semantics。", "先把语义状态标为 unresolved。"],
    debrief: "Shape 说明大小；语义说明用途。"
  },
  token_grid_builder: {
    code: "0-1E",
    title: "Token IDs：从文字到二维表格",
    concept: "多条文本的 token ids 通常组织成 token_ids[B,T]。",
    tool: "Tokenizer + Token Grid",
    mission: "手工连 Text -> Tokenizer -> Grid，再标出 row=B、column=T。",
    visual: "token_grid",
    inspectorNotes: ["一行是一条独立样本。", "一列是所有样本的同一个 token 位置。", "B 保存样本分离，T 保存顺序。"],
    failureLesson: ["Axis semantic error。", "Batch axis 分离样本；Token axis 保持样本内部顺序。", "不要把横轴标成 batch。"],
    debrief: "token_ids[B,T] 建立了 B 和 T 的语义。"
  },
  embedding_expansion: {
    code: "0-1F",
    title: "Embedding：给每个 token 增加特征通道",
    concept: "Embedding Lookup 把每个 token id 变成一条 C 维向量。",
    tool: "Embedding Lookup",
    mission: "连接 token grid 和 embedding table，把 [B,T] 展开成 [B,T,C]。",
    visual: "embedding_expansion",
    inspectorNotes: ["token id 是查表用的整数。", "表中的每一行是一条 C 维向量。", "C 是 token 内部的 feature channel。"],
    failureLesson: ["Expansion error。", "C 不是新的时间步，T 的位置没有增加。", "C 是每个 token 内部的向量长度。"],
    debrief: "Embedding 把 token_ids[B,T] 扩展为 hidden[B,T,C]。"
  },
  hidden_contract_repair: {
    code: "0-1G",
    title: "Hidden Tensor：模型内部的工作状态",
    concept: "Hidden Tensor 是中间表示，常见合同是 hidden[B,T,C]。",
    tool: "Batch / Time / Channel Probe",
    mission: "用 Probe 观察三个轴，再把 B/T/C 标签贴到正确位置。",
    visual: "hidden_contract",
    inspectorNotes: ["B 轴分离不同样本。", "T 轴沿 token 顺序变化。", "C 轴包含连续 feature values。"],
    failureLesson: ["Contract failed。", "你标错的轴表现出了另一个语义。", "先看 Probe 证据，再贴 B/T/C。"],
    debrief: "hidden[B,T,C] 把大小和轴语义组合成合同。"
  },
  consumer_validation: {
    code: "0-1H",
    title: "Axis Semantics 是模块合同",
    concept: "B/T/C 不是装饰标签，而是下游模块的连接合同。",
    tool: "Consumer Ports",
    mission: "把 Hidden 的 B/T/C 端口手工连到 Batch Viewer、Causal Mask、Linear Probe。",
    visual: "consumer_contract",
    inspectorNotes: ["Batch Viewer 消费 B。", "Causal Mask 消费 T，生成 [T,T] mask。", "Linear Projection 消费 C。"],
    failureLesson: ["Consumer contract failed。", "模块收到错误轴会在错误方向上计算。", "把 B/T/C 接给对应消费者。"],
    debrief: "Axis semantics 决定 tensor 可以进入哪些模块。"
  },
  hidden_test_gauntlet: {
    code: "0-1X",
    title: "B/T/C 不是固定数字",
    concept: "轴语义靠行为和合同判断，不靠哪个长度最大。",
    tool: "Hidden Tests",
    mission: "运行不同 shape 变体，验证修复不是只记住 [2,4,8]。",
    visual: "hidden_tests",
    inspectorNotes: ["同样是 hidden[B,T,C]，尺寸可以变化。", "长轴不一定是 C。", "用 Probe 行为和下游合同判断语义。"],
    failureLesson: ["Size-based guess failed。", "Axis length alone does not define meaning。", "回到行为证据，而不是按数字大小猜。"],
    debrief: "语义可迁移，shape 数字会变化。"
  }
};

const chapter01PhaseNodePositions: Partial<Record<LevelPhase, Record<string, { x: number; y: number }>>> = {
  tensor_object: {
    tensor_inspector: { x: 500, y: 236 },
    type_check: { x: 940, y: 342 }
  },
  token_grid_builder: {
    text_batch: { x: 54, y: 170 },
    tokenizer: { x: 330, y: 164 },
    token_grid: { x: 640, y: 138 }
  },
  embedding_expansion: {
    token_grid: { x: 62, y: 144 },
    embedding_table: { x: 82, y: 348 },
    embedding_lookup: { x: 394, y: 172 },
    hidden_tensor: { x: 704, y: 146 }
  },
  consumer_validation: {
    hidden_tensor: { x: 88, y: 194 },
    batch_viewer: { x: 464, y: 86 },
    causal_mask: { x: 464, y: 220 },
    linear_probe: { x: 464, y: 354 },
    shape_tests: { x: 760, y: 220 }
  },
  hidden_test_gauntlet: {
    hidden_tensor: { x: 88, y: 194 },
    batch_viewer: { x: 464, y: 86 },
    causal_mask: { x: 464, y: 220 },
    linear_probe: { x: 464, y: 354 },
    shape_tests: { x: 760, y: 220 }
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
  const [missionStarted, setMissionStarted] = useState<Record<string, boolean>>({});
  const [completionDismissed, setCompletionDismissed] = useState<Record<string, boolean>>({});
  const [nodePositions, setNodePositions] = useState<Record<string, NodePositionMap>>({});
  const [stageKnowledgePositions, setStageKnowledgePositions] = useState<Record<string, StageKnowledgePositionMap>>({});
  const [pendingStageDebrief, setPendingStageDebrief] = useState<Record<string, LevelPhase | undefined>>({});
  const [stageDebriefOpen, setStageDebriefOpen] = useState<Record<string, boolean>>({});
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
  const showCompletionModal = Boolean(levelResult?.passed && activeLevel.id === "0-1" && !completionDismissed[activeLevel.id]);
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
    [activeLevel, activeRepairState.assignments, activeRepairState.selectedSlotId, levelPhase]
  );
  const activeTensorObjectShowcase = tensorObjectDetail ? tensorObjectShowcases[tensorObjectDetail.slotId] : undefined;

  useEffect(() => {
    if (!displayNodes.some((node) => node.id === selectedId)) {
      setSelectedId(displayNodes[0]?.id ?? activeLevel.defaultSelectedNodeId);
    }
  }, [activeLevel.defaultSelectedNodeId, displayNodes, selectedId]);

  useEffect(() => {
    setCanvasMenu(null);
  }, [activeLevel.id, levelPhase]);

  useEffect(() => {
    if (activeLevel.id === "0-1" && levelPhase === "tensor_object") return;
    setTensorObjectDetail(null);
  }, [activeLevel.id, levelPhase]);

  function selectLevel(level: BootcampLevel) {
    setSelectedLevelId(level.id);
    setSelectedId(level.defaultSelectedNodeId);
    setPlaying(true);
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
    const currentChallenge = activeLevel.id === "0-1" ? chapter01ChallengeForPhase(levelPhase) : undefined;
    const shouldShowStageDebrief =
      Boolean(currentChallenge?.slotIds.length) &&
      !pendingDebriefPhase &&
      areAssignmentsCorrect(activeLevel, nextAssignments, currentChallenge?.slotIds ?? []);

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
    setSelectedId(slot.focusNodeId);
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

    updateActiveRepairState((state) => ({
      ...state,
      selectedSlotId: slot.id,
      probeUses: state.probeUses + probe.budgetCost,
      observations: [
        {
          ...observation,
          probeId: probe.id,
          probeLabel: probe.label,
          slotId: slot.id,
          slotLabel: slot.label
        },
        ...state.observations
      ].slice(0, 6)
    }));
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
              const showNestedChallenges = level.id === "0-1" && selectedLevelId === level.id;
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
          {activeLevel.id === "0-1" && levelPhase !== "tensor_object" ? (
            <div className="canvasPaletteLayer">
              <BlueprintPalette level={activeLevel} repairState={activeRepairState} phase={levelPhase} onSelectTag={selectTag} />
            </div>
          ) : null}
          <CanvasContextMenu target={canvasMenu} actions={getCanvasMenuActions(canvasMenu)} onClose={() => setCanvasMenu(null)} />
          {pendingDebriefPhase && !stageDebriefOpen[activeLevel.id] ? (
            <StageDebriefPrompt
              stageKnowledge={chapter01StageKnowledge[pendingDebriefPhase]}
              nextLabel={nextChapter01StageLabel(pendingDebriefPhase)}
              onOpen={() => setStageDebriefOpen((current) => ({ ...current, [activeLevel.id]: true }))}
            />
          ) : null}
          {pendingDebriefPhase && stageDebriefOpen[activeLevel.id] ? (
            <StageDebriefOverlay
              stageKnowledge={chapter01StageKnowledge[pendingDebriefPhase]}
              nextLabel={nextChapter01StageLabel(pendingDebriefPhase)}
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
        </section>

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

      {showCompletionModal ? (
        <CompletionOverlay
          level={activeLevel}
          result={levelResult}
          onClose={() => setCompletionDismissed((current) => ({ ...current, [activeLevel.id]: true }))}
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
    return result ? "visible_testing" : "axis_tagging";
  }

  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("tensor_object"))) return "tensor_object";
  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("rank_scanner"))) return "rank_scanner";
  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("shape_caliper"))) return "shape_caliper";
  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("semantic_gap"))) return "semantic_gap";
  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("token_grid_builder"))) return "token_grid_builder";
  if (!areSlotsCorrect(level, repairState, chapter01SlotIds("embedding_expansion"))) return "embedding_expansion";

  const axesReady = areSlotsCorrect(level, repairState, chapter01SlotIds("hidden_contract_repair"));
  const probeEvidenceReady = repairState.observations.length >= 3;
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

function chapter01SlotIds(phase: LevelPhase) {
  return chapter01ChallengeByPhase.get(phase)?.slotIds ?? [];
}

function chapter01ChallengeForPhase(phase: LevelPhase) {
  if (phase === "knowledge_intro" || phase === "mission_modal") return chapter01Challenges[0];
  if (phase === "completed") return chapter01ChallengeByPhase.get("hidden_test_gauntlet");
  return chapter01ChallengeByPhase.get(phase);
}

function buildCanvasStageKnowledge(level: BootcampLevel, phase: LevelPhase): Chapter01StageKnowledge | undefined {
  if (level.id !== "0-1") return undefined;
  const normalizedPhase = phase === "completed" ? "hidden_test_gauntlet" : phase;
  const knowledge = chapter01StageKnowledge[normalizedPhase];
  if (!knowledge) return undefined;

  const index = chapter01Challenges.findIndex((challenge) => challenge.phase === normalizedPhase);
  const previous = index > 0 ? chapter01StageKnowledge[chapter01Challenges[index - 1].phase] : undefined;
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
  if (level.id !== "0-1") return level.repair.slots;
  const challenge = chapter01ChallengeForPhase(phase);
  if (!challenge) return [];
  const slotIds = new Set(challenge.slotIds);
  return level.repair.slots.filter((slot) => slotIds.has(slot.id));
}

function visibleProbesForPhase(level: BootcampLevel, phase: LevelPhase) {
  if (level.id !== "0-1") return level.repair.probes;
  return phase === "hidden_contract_repair" ? level.repair.probes : [];
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
  if (level.id !== "0-1") return nodes;
  const challenge = chapter01ChallengeForPhase(phase);
  if (!challenge) return nodes;
  const visibleIds = new Set(challenge.nodeIds);
  const phasePositions = chapter01PhaseNodePositions[phase] ?? {};
  return nodes.filter((node) => visibleIds.has(node.id)).map((node) => {
    const positioned = phasePositions[node.id] ? { ...node, ...phasePositions[node.id] } : node;
    if (phase !== "tensor_object") return positioned;

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

function filterDisplayEdgesForPhase(level: BootcampLevel, edges: BootcampLevel["edges"], nodes: TensorNode[], phase: LevelPhase) {
  if (level.id !== "0-1") return edges;
  const challenge = chapter01ChallengeForPhase(phase);
  if (!challenge) return edges;
  const visibleNodeIds = new Set(nodes.map((node) => node.id));
  const visibleEdgeIds = new Set(challenge.edgeIds);
  return edges.filter((edge) => visibleEdgeIds.has(edge.id) && visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to));
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

  return level.edges.map((edge) => {
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
    const icon: CanvasActionHint["icon"] = phase === "hidden_contract_repair" && !filled ? "probe" : "menu";
    const tooltip =
      phase === "hidden_contract_repair"
        ? filled
          ? "Right-click: collect probe evidence; drag a palette variable here to replace the tag"
          : "Right-click to probe; drag B/T/C from the palette into this slot"
        : "Drag a palette variable into this slot";
    hints.push({
      kind: "slot",
      id: slot.id,
      icon,
      tooltip,
      pulse: !filled
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
    const weightTag = assignments.weight_fix;
    const outputAxis = tagLabel("output_axis");
    const repairedWeight = weightTag === "weight_co" || weightTag === "transpose_weight";
    patchNode("weight_plate", {
      shape: weightTag === "transpose_weight" ? "[C,O] via T(W)" : weightTag === "weight_co" ? "[C,O]" : "[O,C]",
      subtitle: repairedWeight ? "repaired" : "mounted wrong"
    });
    patchNode("linear_gate", { shape: repairedWeight ? "[B,T,C]@[C,O]" : "[B,T,C]@[O,C]", subtitle: repairedWeight ? "inner dim locked" : "inner-dim latch" });
    patchNode("linear_out", { shape: `[B,T,${outputAxis}]`, subtitle: outputAxis === "O" ? "contract repaired" : "needs O axis" });
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
  return visibleSlotsForPhase(level, phase).map((slot) => {
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
    level.id === "0-1"
      ? chapter01Challenges.map((challenge) => ({
          id: challenge.code,
          label: `${challenge.code} ${challenge.title}`,
          done:
            challenge.phase === "hidden_contract_repair"
              ? areSlotsCorrect(level, repairState, challenge.slotIds) && repairState.observations.length >= 3
              : challenge.phase === "hidden_test_gauntlet"
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
  onSelectTag
}: {
  level: BootcampLevel;
  repairState: LevelRepairState;
  phase: LevelPhase;
  onSelectTag: (tagId: string) => void;
}) {
  const tags = visiblePaletteTagsForPhase(level, phase);
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
      {tags.length ? (
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
  return (
    <div className="challengeRail">
      {chapter01Challenges.map((challenge) => {
        const active = phase === challenge.phase;
        const done =
          challenge.phase === "hidden_contract_repair"
            ? areSlotsCorrect(level, repairState, challenge.slotIds) && repairState.observations.length >= 3
            : challenge.phase === "hidden_test_gauntlet"
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

function nextChapter01StageLabel(phase: LevelPhase | undefined) {
  const index = chapter01Challenges.findIndex((challenge) => challenge.phase === phase);
  const next = index >= 0 ? chapter01Challenges[index + 1] : undefined;
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

type KnowledgeCardSpec = NonNullable<BootcampLevel["knowledgeCards"]>[number];

function getVisibleKnowledgeVisual(card: KnowledgeCardSpec, visualStep: number) {
  const lines = card.visual ?? [];
  if (!lines.length) return [];
  return lines.slice(0, Math.min(lines.length, Math.max(visualStep, 0) + 1));
}

function KnowledgeIntroOverlay({ level, onContinue }: { level: BootcampLevel; onContinue: () => void }) {
  const [cardIndex, setCardIndex] = useState(0);
  const [visualStep, setVisualStep] = useState(0);
  const cards = level.knowledgeCards ?? [];
  const showingTransition = cardIndex >= cards.length;
  const card = cards[Math.min(cardIndex, Math.max(cards.length - 1, 0))];
  const progress = cards.length ? Math.min(cardIndex + 1, cards.length) : 0;
  const transition = level.knowledgeTransition;
  const visualLines = showingTransition ? [] : getVisibleKnowledgeVisual(card, visualStep);
  const maxVisualStep = Math.max((card.visual?.length ?? 1) - 1, 0);
  const showClickGuide = !showingTransition && visualStep < maxVisualStep;
  const advanceVisual = () => setVisualStep((value) => Math.min(value + 1, maxVisualStep));

  useEffect(() => {
    setVisualStep(0);
  }, [cardIndex]);

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
          <section
            className={`knowledgeVisual ${showingTransition ? "complete" : ""} ${!showingTransition ? "interactive" : ""}`}
            role={showingTransition ? undefined : "button"}
            tabIndex={showingTransition ? undefined : 0}
            onClick={showingTransition ? undefined : advanceVisual}
            onKeyDown={
              showingTransition
                ? undefined
                : (event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      advanceVisual();
                    }
                  }
            }
          >
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
                    <code key={`${line}_${index}`} className={index === visualLines.length - 1 ? "active" : ""}>
                      {line}
                      {index === visualLines.length - 1 && showClickGuide ? (
                        <span className="lineClickGuide" aria-hidden="true">
                          <span className="lineClickGuideBubble">
                            <span />
                            点击这里
                          </span>
                        </span>
                      ) : null}
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
          {!showingTransition ? (
            <button className="ghostButton" onClick={onContinue}>
              Skip Briefing
            </button>
          ) : (
            <span />
          )}
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
  onClose
}: {
  level: BootcampLevel;
  result?: BootcampResult;
  onClose: () => void;
}) {
  if (!result) return null;

  return (
    <div className="modalBackdrop" role="dialog" aria-modal="true" aria-label="contract restored">
      <section className="completionModal">
        <CheckCircle2 size={28} />
        <p className="eyebrow">Contract Restored</p>
        <h2>hidden[B,T,C] accepted</h2>
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
        <button className="runButton modalAction" onClick={onClose}>
          Continue
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
  const tensorObjectShowcase = getTensorObjectShowcase(stageKnowledge, taskItems, tensorObjectDetailSlotId ?? selectedSlotId);
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

      {stageKnowledge && stageKnowledge.visual !== "tensor_objects" ? (
        <section className="inspectorNote">
          <h3>Inspector Note</h3>
          <b>{stageKnowledge.title}</b>
          {stageKnowledge.inspectorNotes.map((line) => (
            <small key={line}>{line}</small>
          ))}
        </section>
      ) : null}

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
