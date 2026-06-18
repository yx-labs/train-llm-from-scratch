import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Boxes, CheckCircle2, Circle, Cpu, Eye, Pause, Play, RotateCcw, Search, Wrench } from "lucide-react";
import { bootcampLevels, evaluateBootcampLevel } from "./bootcampLevels";
import { PixiWorkbenchCanvas, type CanvasActionHint, type CanvasConnectionOverlay, type CanvasContextTarget, type RepairSlotOverlay } from "./PixiWorkbenchCanvas";
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

type CanvasMenuAction = {
  id: string;
  label: string;
  detail: string;
  disabled?: boolean;
  onSelect: () => void;
};

type LevelPhase =
  | "knowledge_intro"
  | "mission_modal"
  | "data_flow_repair"
  | "tensor_generated"
  | "axis_probe"
  | "axis_tagging"
  | "contract_wiring"
  | "visible_testing"
  | "hidden_testing"
  | "completed";

const modeIcons: Record<WorkbenchMode, JSX.Element> = {
  build: <Wrench size={17} />,
  trace: <Search size={17} />,
  train: <Cpu size={17} />
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
  const [canvasMenu, setCanvasMenu] = useState<CanvasContextTarget | null>(null);

  const activeLevel = useMemo(
    () => bootcampLevels.find((level) => level.id === selectedLevelId) ?? bootcampLevels[0],
    [selectedLevelId]
  );
  const activeRepairState = repairStates[activeLevel.id] ?? createInitialRepairState();
  const baseDisplayNodes = useMemo(() => buildDisplayNodes(activeLevel, activeRepairState.assignments), [activeLevel, activeRepairState.assignments]);
  const displayNodes = useMemo(() => applyNodePositions(baseDisplayNodes, nodePositions[activeLevel.id]), [activeLevel.id, baseDisplayNodes, nodePositions]);
  const displayEdges = useMemo(() => buildDisplayEdges(activeLevel, activeRepairState.assignments), [activeLevel, activeRepairState.assignments]);
  const selectedNode = useMemo(
    () => displayNodes.find((node) => node.id === selectedId) ?? displayNodes[0],
    [displayNodes, selectedId]
  );
  const modeCopy = modeLabels[mode];
  const levelResult = results[activeLevel.id];
  const passedCount = bootcampLevels.filter((level) => results[level.id]?.passed).length;
  const latestObservation = activeRepairState.observations[0];
  const showKnowledgeIntro = Boolean(activeLevel.knowledgeCards?.length && !introSeen[activeLevel.id]);
  const showMissionModal = Boolean(activeLevel.mission && !showKnowledgeIntro && !missionStarted[activeLevel.id]);
  const levelPhase = deriveLevelPhase(activeLevel, activeRepairState, levelResult, showKnowledgeIntro, showMissionModal);
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

  useEffect(() => {
    if (!displayNodes.some((node) => node.id === selectedId)) {
      setSelectedId(activeLevel.defaultSelectedNodeId);
    }
  }, [activeLevel.defaultSelectedNodeId, displayNodes, selectedId]);

  useEffect(() => {
    setCanvasMenu(null);
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
    updateActiveRepairState((state) => {
      const assignments = { ...state.assignments };
      for (const [slotId, assignedTagId] of Object.entries(assignments)) {
        if (assignedTagId === tagId) delete assignments[slotId];
      }
      assignments[slot.id] = tagId;
      return { ...state, assignments, selectedSlotId: slot.id, activeTagId: undefined };
    });
    clearActiveResult();
    setSelectedId(slot.focusNodeId);
    setPlaying(true);
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
    const axisTags = activeLevel.repair.tags.filter((tag) => tag.category === "axis");
    const axisSlotIds = new Set(["axis_0", "axis_1", "axis_2"]);

    if (targetSlot && axisSlotIds.has(targetSlot.id) && (levelPhase === "axis_probe" || levelPhase === "axis_tagging")) {
      visibleProbesForPhase(activeLevel, levelPhase).forEach((probe) => {
        actions.push({
          id: `probe_${probe.id}`,
          label: probe.label,
          detail: probe.detail,
          onSelect: () => probeCanvasSlot(targetSlot.id, probe.id)
        });
      });

      axisTags.forEach((tag) => {
        actions.push({
          id: `tag_${tag.id}`,
          label: `Assign ${tag.shortLabel}`,
          detail: tag.detail,
          disabled: levelPhase === "axis_probe" && activeRepairState.observations.length === 0,
          onSelect: () => assignCanvasSlot(targetSlot.id, tag.id)
        });
      });
    }

    if (target.kind === "node" && target.id === "axis_decoder" && levelPhase === "contract_wiring") {
      [
        { slotId: "contract_b", tagId: "contract_b", label: "Wire Decoder B", detail: "Axis Decoder B port receives batch semantic." },
        { slotId: "contract_t", tagId: "contract_t", label: "Wire Decoder T", detail: "Axis Decoder T port receives token-position semantic." },
        { slotId: "contract_c", tagId: "contract_c", label: "Wire Decoder C", detail: "Axis Decoder C port receives channel semantic." }
      ].forEach((item) => {
        const alreadyConnected = activeRepairState.assignments[item.slotId] === item.tagId;
        actions.push({
          id: item.slotId,
          label: item.label,
          detail: item.detail,
          disabled: alreadyConnected,
          onSelect: () => assignCanvasSlot(item.slotId, item.tagId)
        });
      });
    }

    if ((target.kind === "node" && target.id === "shape_tests") || target.kind === "canvas") {
      if (levelPhase === "visible_testing" || levelPhase === "hidden_testing") {
        actions.push({
          id: "run_tests",
          label: "Run Contract Tests",
          detail: "Execute visible, behavior, reference, and hidden checks.",
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
            <p className="eyebrow">LLM Complete / MVP 0.0.2</p>
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
              return (
                <button
                  key={level.id}
                  className={`levelItem ${selectedLevelId === level.id ? "active" : ""} ${statusClass}`}
                  onClick={() => selectLevel(level)}
                >
                  <span className="levelId">{level.id}</span>
                  <span>
                    <b>{level.title}</b>
                    <small>{level.subtitle}</small>
                    <small>Tool: {repairKindLabel(level.repair.kind)}</small>
                    <small>Reward: {level.unlocks[0]}</small>
                  </span>
                  <StateIcon state={result ? (result.passed ? "pass" : "fail") : "warn"} />
                </button>
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
            onSlotSelect={activateSlot}
            onCanvasConnect={connectCanvasSlot}
            onCanvasContextMenu={setCanvasMenu}
            onNodeMove={moveCanvasNode}
            onSelect={setSelectedId}
          />
          <CanvasContextMenu target={canvasMenu} actions={getCanvasMenuActions(canvasMenu)} onClose={() => setCanvasMenu(null)} />
        </section>

        <TensorInspector node={selectedNode} observation={latestObservation} />
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

  const dataReady = areSlotsCorrect(level, repairState, ["flow_text_tokenizer", "flow_tokenizer_embedding", "flow_embedding_hidden"]);
  if (!dataReady) return "data_flow_repair";

  const axesFilled = areSlotsFilled(repairState, ["axis_0", "axis_1", "axis_2"]);
  if (!axesFilled && repairState.observations.length === 0) return "axis_probe";
  if (!axesFilled) return "axis_tagging";

  const contractFilled = areSlotsFilled(repairState, ["contract_b", "contract_t", "contract_c"]);
  if (!contractFilled) return "contract_wiring";

  if (!result) return "visible_testing";
  return "hidden_testing";
}

function areSlotsCorrect(level: BootcampLevel, repairState: LevelRepairState, slotIds: string[]) {
  return slotIds.every((slotId) => {
    const slot = level.repair.slots.find((item) => item.id === slotId);
    return slot ? slot.correctTagIds.includes(repairState.assignments[slotId]) : true;
  });
}

function areSlotsFilled(repairState: LevelRepairState, slotIds: string[]) {
  return slotIds.every((slotId) => Boolean(repairState.assignments[slotId]));
}

function phaseLabel(phase: LevelPhase) {
  const labels: Record<LevelPhase, string> = {
    knowledge_intro: "Knowledge Intro",
    mission_modal: "Mission Brief",
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
  if (phase === "data_flow_repair" || phase === "knowledge_intro" || phase === "mission_modal") {
    return level.repair.tags.filter((tag) => tag.category === "data");
  }
  if (phase === "axis_probe" || phase === "axis_tagging") {
    return level.repair.tags.filter((tag) => tag.category === "axis");
  }
  if (phase === "contract_wiring") {
    return level.repair.tags.filter((tag) => tag.category === "axis" || tag.category === "contract");
  }
  return level.repair.tags;
}

function visibleSlotsForPhase(level: BootcampLevel, phase: LevelPhase): RepairSlot[] {
  if (level.id !== "0-1") return level.repair.slots;
  const dataSlots = new Set(["flow_text_tokenizer", "flow_tokenizer_embedding", "flow_embedding_hidden"]);
  const axisSlots = new Set(["axis_0", "axis_1", "axis_2"]);
  const contractSlots = new Set(["contract_b", "contract_t", "contract_c"]);

  if (phase === "data_flow_repair" || phase === "knowledge_intro" || phase === "mission_modal") {
    return level.repair.slots.filter((slot) => dataSlots.has(slot.id));
  }
  if (phase === "axis_probe" || phase === "axis_tagging") {
    return level.repair.slots.filter((slot) => axisSlots.has(slot.id));
  }
  if (phase === "contract_wiring") {
    return level.repair.slots.filter((slot) => axisSlots.has(slot.id) || contractSlots.has(slot.id));
  }
  return level.repair.slots;
}

function visibleProbesForPhase(level: BootcampLevel, phase: LevelPhase) {
  if (level.id !== "0-1") return level.repair.probes;
  if (phase === "data_flow_repair" || phase === "knowledge_intro" || phase === "mission_modal") return [];
  return level.repair.probes;
}

function visibleCanvasSlotsForPhase(level: BootcampLevel, phase: LevelPhase): RepairSlot[] {
  if (level.id !== "0-1") return visibleSlotsForPhase(level, phase);
  const axisSlots = new Set(["axis_0", "axis_1", "axis_2"]);
  if (phase === "axis_probe" || phase === "axis_tagging" || phase === "contract_wiring" || phase === "visible_testing" || phase === "hidden_testing" || phase === "completed") {
    return level.repair.slots.filter((slot) => axisSlots.has(slot.id));
  }
  return [];
}

function applyNodePositions(nodes: TensorNode[], positions?: NodePositionMap): TensorNode[] {
  if (!positions) return nodes;
  return nodes.map((node) => {
    const position = positions[node.id];
    return position ? { ...node, x: position.x, y: position.y } : node;
  });
}

function buildDisplayEdges(level: BootcampLevel, assignments: BootcampAnswerMap) {
  if (level.id !== "0-1") return level.edges;

  const slotDone = (slotId: string) => {
    const slot = level.repair.slots.find((item) => item.id === slotId);
    return slot ? slot.correctTagIds.includes(assignments[slotId]) : false;
  };

  const dataEdgeSlots: Record<string, string> = {
    e_01_text_tokenizer: "flow_text_tokenizer",
    e_01_tokenizer_embedding: "flow_tokenizer_embedding",
    e_01_embedding_hidden: "flow_embedding_hidden"
  };
  const contractReady = ["contract_b", "contract_t", "contract_c"].every(slotDone);

  return level.edges.map((edge) => {
    const dataSlotId = dataEdgeSlots[edge.id];
    if (dataSlotId) {
      const done = slotDone(dataSlotId);
      return {
        ...edge,
        label: done ? "connected" : "open port",
        color: done ? 0x22c55e : 0x38bdf8
      };
    }
    if (edge.id === "e_01_hidden_axes") {
      return {
        ...edge,
        label: contractReady ? "contract connected" : "contract ports open",
        color: contractReady ? 0x22c55e : 0xfbbf24
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

  const dataEnabled = phase === "data_flow_repair";
  const contractEnabled = phase === "contract_wiring";
  const dataConnections = [
    connection("canvas_flow_text_tokenizer", "flow_text_tokenizer", "wire_text_tokenizer", "text_batch", "tokenizer", "utf8", "in", "utf8[B]", "data", dataEnabled),
    connection("canvas_flow_tokenizer_embedding", "flow_tokenizer_embedding", "wire_tokenizer_embedding", "tokenizer", "embedding_lookup", "ids", "in", "int[B,T]", "data", dataEnabled),
    connection("canvas_flow_embedding_hidden", "flow_embedding_hidden", "wire_embedding_hidden", "embedding_lookup", "hidden_tensor", "vec", "in", "float[B,T,C]", "data", dataEnabled)
  ];
  const showData = dataEnabled || dataConnections.some((item) => item.state === "connected");

  const contractConnections = [
    connection("canvas_contract_b", "contract_b", "contract_b", "hidden_tensor", "axis_decoder", "B", "B", "B axis", "contract", contractEnabled),
    connection("canvas_contract_t", "contract_t", "contract_t", "hidden_tensor", "axis_decoder", "T", "T", "T axis", "contract", contractEnabled),
    connection("canvas_contract_c", "contract_c", "contract_c", "hidden_tensor", "axis_decoder", "C", "C", "C axis", "contract", contractEnabled)
  ];
  const showContract = contractEnabled || ["visible_testing", "hidden_testing", "completed"].includes(phase) || contractConnections.some((item) => item.state === "connected");

  return [...(showData ? dataConnections : []), ...(showContract ? contractConnections : [])];
}

function buildCanvasActionHints(level: BootcampLevel, repairState: LevelRepairState, phase: LevelPhase): CanvasActionHint[] {
  if (level.id !== "0-1") return [];

  if (phase === "axis_probe") {
    return ["axis_0", "axis_1", "axis_2"].map((slotId) => ({
      kind: "slot",
      id: slotId,
      icon: "probe",
      tooltip: "Click or right-click: probe this axis",
      pulse: true
    }));
  }

  if (phase === "axis_tagging") {
    return ["axis_0", "axis_1", "axis_2"].map((slotId) => ({
      kind: "slot",
      id: slotId,
      icon: repairState.assignments[slotId] ? "menu" : "probe",
      tooltip: repairState.assignments[slotId] ? "Click or right-click: change axis tag" : "Click or right-click: assign B/T/C",
      pulse: !repairState.assignments[slotId]
    }));
  }

  if (phase === "contract_wiring") {
    return [
      {
        kind: "node",
        id: "axis_decoder",
        icon: "wire",
        tooltip: "Click or right-click: wire Decoder B/T/C",
        pulse: true
      }
    ];
  }

  if (phase === "visible_testing" || phase === "hidden_testing") {
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

  return [];
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
    const dataReady = ["flow_text_tokenizer", "flow_tokenizer_embedding", "flow_embedding_hidden"].every((slotId) => {
      const slot = level.repair.slots.find((item) => item.id === slotId);
      return slot ? slot.correctTagIds.includes(assignments[slotId]) : true;
    });
    const axisReady = ["axis_0", "axis_1", "axis_2"].every((slotId) => {
      const slot = level.repair.slots.find((item) => item.id === slotId);
      return slot ? slot.correctTagIds.includes(assignments[slotId]) : true;
    });
    const contractReady = ["contract_b", "contract_t", "contract_c"].every((slotId) => {
      const slot = level.repair.slots.find((item) => item.id === slotId);
      return slot ? slot.correctTagIds.includes(assignments[slotId]) : true;
    });
    const shape = `[${tagLabel("axis_0")},${tagLabel("axis_1")},${tagLabel("axis_2")}]`;
    const complete = dataReady && axisReady && contractReady && shape === "[B,T,C]";
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
    patchNode("embedding_lookup", {
      subtitle: assignments.flow_tokenizer_embedding === "wire_tokenizer_embedding" ? "token ids connected" : "table output open",
      checks: [
        check(
          "lookup line",
          assignments.flow_tokenizer_embedding === "wire_tokenizer_embedding" ? "pass" : "warn",
          assignments.flow_tokenizer_embedding === "wire_tokenizer_embedding" ? "int[B,T] indexes the table" : "connect Tokenizer output to Embedding Lookup"
        )
      ]
    });
    patchNode("hidden_tensor", {
      shape: dataReady ? shape : "[?,?,?]",
      subtitle: !dataReady ? "not generated" : axisReady ? "axis labels assigned" : "generated / axes unknown",
      checks: [
        check("data input", dataReady ? "pass" : "warn", dataReady ? "embedding vectors generate hidden[2,4,8]" : "connect Embedding Lookup before axis labels can be trusted"),
        check(axisReady ? "axis labels assigned" : "contract incomplete", axisReady ? "pass" : "warn", axisReady ? `current contract hidden${shape}` : "assign B/T/C tags to axis slots"),
        check("dtype", "pass", "float32 activations can enter Linear")
      ]
    });
    patchNode("axis_decoder", {
      subtitle: contractReady ? "contract wired" : "contract ports open",
      shape: complete ? "accepts [B,T,C]" : "requires [B,T,C]",
      checks: [
        check("decoder ports", contractReady ? "pass" : "warn", contractReady ? "B/T/C semantic lines are connected" : "wire Decoder B, T, and C ports"),
        check("semantic order", shape === "[B,T,C]" ? "pass" : "warn", shape === "[B,T,C]" ? "axis order matches target" : "axis order must be [B,T,C]")
      ]
    });
    patchNode("shape_tests", {
      subtitle: complete ? "ready to pass" : "blocked",
      checks: [check("ready", complete ? "pass" : "warn", complete ? "visible, behavior, and hidden tests can run" : "contract tests are blocked until slots are filled")]
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
  const dataReady = areSlotsCorrect(level, repairState, ["flow_text_tokenizer", "flow_tokenizer_embedding", "flow_embedding_hidden"]);
  const axesReady = areSlotsCorrect(level, repairState, ["axis_0", "axis_1", "axis_2"]);
  const contractReady = areSlotsCorrect(level, repairState, ["contract_b", "contract_t", "contract_c"]);
  const tested = Boolean(result);
  const objectives =
    level.id === "0-1"
      ? [
          { id: "flow", label: "Restore data flow", done: dataReady, active: phase === "data_flow_repair" },
          { id: "probe", label: "Probe axis behavior", done: repairState.observations.length > 0, active: phase === "axis_probe" },
          { id: "axes", label: "Tag Axis 0/1/2 as B/T/C", done: axesReady, active: phase === "axis_tagging" },
          { id: "decoder", label: "Wire Axis Decoder B/T/C", done: contractReady, active: phase === "contract_wiring" },
          { id: "tests", label: "Run visible, behavior, hidden tests", done: Boolean(result?.passed), active: phase === "visible_testing" || phase === "hidden_testing" }
        ]
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

  if (!target) return null;

  const title = target.kind === "slot" ? "Axis Slot" : target.kind === "node" ? "Node" : "Canvas";
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

function TensorInspector({ node, observation }: { node: TensorNode; observation?: ObservationLogItem }) {
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
