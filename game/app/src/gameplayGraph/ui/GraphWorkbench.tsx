import {
  createContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useContext,
  type DragEvent as ReactDragEvent,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent
} from "react";
import { AlertTriangle, BookOpenText, CheckCircle2, GitBranchPlus, Minus, MousePointer2, Move, Play, Plus, RotateCcw, Trash2, Wrench, X } from "lucide-react";
import { graphLevels } from "../levelRegistry";
import { createGameplayRegistry } from "../modules";
import type {
  CertificationControlSpec,
  CertificationControlValue,
  CertificationTestSpec,
  DType,
  GraphEdge,
  GraphNode,
  GraphSpec,
  LevelCertificationSpec,
  LevelSpec,
  ModuleDef,
  PortDef,
  PortRef,
  RuntimeValue,
  TestAssertion,
  TestCase,
  TestResult,
  TraceFrame
} from "../types";
import { runTestCaseDetailed, runTests, type RunTestsResult, type TestCaseRunResult } from "../runtime/testRunner";
import { graphStatusText, graphText, type GraphLanguage } from "../i18n";
import { createTokenizerPreview, generateGraphCodeSections, getCaseTexts, type TokenizerCasePreview } from "../codegen";

type RunState = {
  visible?: RunTestsResult;
  hidden?: RunTestsResult;
  stats: LevelRunStats;
};

type LevelRunStats = {
  visibleRuns: number;
  hiddenRuns: number;
  failedRuns: number;
  hintsUsed: number;
};

type GraphSelection = { type: "node"; id: string } | { type: "edge"; id: string };
type InspectorTab = "summary" | "shape" | "values" | "tests" | "code";
type TraceRunKey = "visible" | "hidden";
type TraceSelection = {
  runKey: TraceRunKey;
  caseId: string;
  step: number;
};

type CompletionNotice = {
  componentTitle?: string;
  nextLevelTitle?: string;
  routeComplete: boolean;
};

type CertificationRunPreview = {
  id: string;
  levelId: string;
  title: string;
  graph: GraphSpec;
  controls: CertificationControlSpec[];
  values: Record<string, CertificationControlValue>;
  testCase: TestCase;
  publicCase: TestCaseRunResult;
  result: RunTestsResult;
  codeLines: CertificationAnimationLine[];
  outputLines: string[];
};

type CertificationAnimationLine = {
  id: string;
  text: string;
  section?: boolean;
};

type CertificationValuesByLevel = Record<string, Record<string, CertificationControlValue>>;

type WireSource = PortRef & {
  dtype?: DType;
};

type NodeDragState = {
  nodeId: string;
  offsetX: number;
  offsetY: number;
};

type CanvasPoint = {
  x: number;
  y: number;
};

type CanvasViewport = {
  x: number;
  y: number;
  scale: number;
};

type PortAnchorMap = Record<string, CanvasPoint>;

type PanDragState = {
  startClientX: number;
  startClientY: number;
  startViewport: CanvasViewport;
};

type WorkbenchLayout = {
  sidebarWidth: number;
  inspectorWidth: number;
  traceHeight: number;
};

type ResizeRegion = "sidebar" | "inspector" | "trace";

type ParamOption = {
  value: string;
  label: string;
  consequence: string;
};

type ParamCopy = {
  label: string;
  help?: string;
};

export type ComponentFlowLevelSpec = {
  componentId: string;
  title: string;
  version: number;
  exportModuleId: string;
  requires: string[];
  unlocks: string[];
  shelf: string;
};

export type ComponentAvailabilityPayload = {
  level: LevelSpec;
  component: ComponentFlowLevelSpec;
  graph: GraphSpec;
  visible?: RunTestsResult;
  hidden?: RunTestsResult;
};

export type ComponentFlowConfig = {
  specs: Record<string, ComponentFlowLevelSpec>;
  availableComponentIds: string[];
  onComponentAvailable: (payload: ComponentAvailabilityPayload) => void;
};

const moduleDragMime = "application/x-llm-complete-graph-module";
const graphNodeWidth = 240;
const graphNodeMinHeight = 156;
const graphWorldWidth = 2400;
const graphWorldHeight = 1600;
const graphWorldDragMargin = 1200;
const graphWorldMinX = -graphWorldDragMargin;
const graphWorldMinY = -graphWorldDragMargin;
const graphWorldMaxX = graphWorldWidth + graphWorldDragMargin;
const graphWorldMaxY = graphWorldHeight + graphWorldDragMargin;
const graphWorldLayerWidth = graphWorldMaxX - graphWorldMinX;
const graphWorldLayerHeight = graphWorldMaxY - graphWorldMinY;
const minCanvasScale = 0.45;
const maxCanvasScale = 1.8;
const defaultWorkbenchLayout: WorkbenchLayout = {
  sidebarWidth: 285,
  inspectorWidth: 365,
  traceHeight: 178
};
const resizeHandleSize = 8;
const minSidebarWidth = 210;
const minInspectorWidth = 280;
const minCenterWidth = 480;
const minTraceHeight = 112;
const GraphLanguageContext = createContext<GraphLanguage>("en");

function useGraphT() {
  const language = useContext(GraphLanguageContext);
  return {
    language,
    t: (text: string) => graphText(language, text),
    status: (text: string) => graphStatusText(language, text)
  };
}

function isLevelPlayable(level: LevelSpec) {
  return level.routeStatus === "playable";
}

function firstPlayableLevel(levels: LevelSpec[]) {
  return levels.find(isLevelPlayable) ?? levels[0];
}

export function GraphWorkbench({
  language = "en",
  levels = graphLevels,
  componentFlow
}: {
  language?: GraphLanguage;
  levels?: LevelSpec[];
  componentFlow?: ComponentFlowConfig;
}) {
  const registry = useMemo(() => createGameplayRegistry(), []);
  const modules = useMemo(() => registry.list(), [registry]);
  const availableComponentIds = useMemo(() => new Set(componentFlow?.availableComponentIds ?? []), [componentFlow?.availableComponentIds]);
  const [selectedLevelId, setSelectedLevelId] = useState(() => firstPlayableLevel(levels).id);
  const selectedLevel = levels.find((level) => level.id === selectedLevelId) ?? firstPlayableLevel(levels);
  const selectedLevelPlayable = isLevelPlayable(selectedLevel);
  const selectedComponent = componentFlow?.specs[selectedLevel.id];
  const [graphs, setGraphs] = useState<Record<string, GraphSpec>>(() => initialGraphsByLevel(levels));
  const [selection, setSelection] = useState<GraphSelection | undefined>(() => {
    const firstNodeId = firstPlayableLevel(levels).initialGraph.nodes[0]?.id;
    return firstNodeId ? { type: "node", id: firstNodeId } : undefined;
  });
  const [wireSource, setWireSource] = useState<WireSource>();
  const [wirePointer, setWirePointer] = useState<CanvasPoint>();
  const [canvasNotice, setCanvasNotice] = useState("ready");
  const [dragOverCanvas, setDragOverCanvas] = useState(false);
  const [viewport, setViewport] = useState<CanvasViewport>({ x: 0, y: 0, scale: 1 });
  const [runs, setRuns] = useState<Record<string, RunState>>({});
  const [certificationValuesByLevel, setCertificationValuesByLevel] = useState<CertificationValuesByLevel>(() => initialCertificationValuesByLevel(levels));
  const [traceSelection, setTraceSelection] = useState<TraceSelection>();
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("summary");
  const [missionOpen, setMissionOpen] = useState(false);
  const [challengeMapOpen, setChallengeMapOpen] = useState(false);
  const [completionNotice, setCompletionNotice] = useState<CompletionNotice>();
  const [certificationRun, setCertificationRun] = useState<CertificationRunPreview>();
  const [layout, setLayout] = useState<WorkbenchLayout>(defaultWorkbenchLayout);
  const [portAnchors, setPortAnchors] = useState<PortAnchorMap>({});
  const workbenchRef = useRef<HTMLElement>(null);
  const stageRef = useRef<HTMLElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const seenMissionLevelIdsRef = useRef<Set<string>>(new Set());
  const autoAdvanceTimerRef = useRef<number>();
  const nodeDragRef = useRef<NodeDragState>();
  const panDragRef = useRef<PanDragState>();
  const wireSourceRef = useRef<WireSource>();
  const suppressNextPortClickRef = useRef(false);
  const graph = graphs[selectedLevel.id] ?? selectedLevel.initialGraph;
  const runState = runs[selectedLevel.id] ?? createEmptyRunState();
  const certificationValues = certificationValuesForLevel(selectedLevel, certificationValuesByLevel[selectedLevel.id]);
  const certificationErrors = certificationControlErrors(selectedLevel.certification, certificationValues);
  const missingComponentRequirements = selectedComponent?.requires.filter((componentId) => !availableComponentIds.has(componentId)) ?? [];
  const componentLevelLocked = !selectedLevelPlayable || missingComponentRequirements.length > 0;
  const componentAvailable = selectedComponent ? availableComponentIds.has(selectedComponent.componentId) : false;
  const selectedNode = selection?.type === "node" ? graph.nodes.find((node) => node.id === selection.id) : undefined;
  const selectedEdge = selection?.type === "edge" ? graph.edges.find((edge) => edge.id === selection.id) : undefined;
  const selectedModule = selectedNode ? registry.maybeGet(selectedNode.moduleId) : undefined;
  const wireSourcePort = wireSource ? getOutputPort(graph, modules, wireSource) : undefined;
  const activeTraceCase = resolveTraceCase(runState, traceSelection);
  const activeTraceFrame = activeTraceCase && traceSelection?.caseId === activeTraceCase.id ? activeTraceCase.execution.trace[traceSelection.step] : undefined;
  const certificationLocked = runState.visible?.status !== "pass";
  const canSubmitCertification = !certificationLocked && !componentLevelLocked && certificationErrors.length === 0 && !certificationRun;
  const workbenchStyle = {
    "--graph-sidebar-width": `${layout.sidebarWidth}px`,
    "--graph-inspector-width": `${layout.inspectorWidth}px`
  } as CSSProperties;
  const stageStyle = {
    "--graph-trace-height": `${layout.traceHeight}px`
  } as CSSProperties;

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (event.key === "Escape" && wireSourceRef.current) {
        event.preventDefault();
        cancelWire();
        return;
      }
      if (event.key === "Escape" && missionOpen) {
        event.preventDefault();
        setMissionOpen(false);
        return;
      }
      if (target?.closest("input, textarea, select")) return;
      if (event.key !== "Delete" && event.key !== "Backspace") return;
      if (!selection) return;
      event.preventDefault();
      deleteSelection();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  useEffect(() => {
    if (seenMissionLevelIdsRef.current.has(selectedLevel.id)) {
      setMissionOpen(false);
      return;
    }
    seenMissionLevelIdsRef.current.add(selectedLevel.id);
    setMissionOpen(true);
  }, [selectedLevel.id]);

  useEffect(() => {
    return () => {
      if (autoAdvanceTimerRef.current) window.clearTimeout(autoAdvanceTimerRef.current);
    };
  }, []);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const canvasRect = canvas.getBoundingClientRect();
    const nextAnchors: PortAnchorMap = {};
    canvas.querySelectorAll<HTMLElement>("[data-graph-port-dot='true']").forEach((element) => {
      const nodeId = element.dataset.nodeId;
      const portId = element.dataset.portId;
      const direction = element.dataset.portDirection as "in" | "out" | undefined;
      if (!nodeId || !portId || !direction) return;
      const rect = element.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2 - canvasRect.left;
      const centerY = rect.top + rect.height / 2 - canvasRect.top;
      nextAnchors[portAnchorKey(nodeId, portId, direction)] = viewportToWorld(centerX, centerY);
    });
    setPortAnchors(nextAnchors);
  }, [graph, selectedLevel.id, viewport.x, viewport.y, viewport.scale, language]);

  function selectLevel(level: LevelSpec) {
    if (!isLevelPlayable(level)) {
      setCanvasNotice("design_ready level is not playable yet");
      return;
    }
    if (autoAdvanceTimerRef.current) {
      window.clearTimeout(autoAdvanceTimerRef.current);
      autoAdvanceTimerRef.current = undefined;
    }
    const levelGraph = graphs[level.id] ?? level.initialGraph;
    setSelectedLevelId(level.id);
    setSelection(levelGraph.nodes[0] ? { type: "node", id: levelGraph.nodes[0].id } : undefined);
    clearWireSource();
    setWirePointer(undefined);
    setTraceSelection(undefined);
    setInspectorTab("summary");
    setCompletionNotice(undefined);
    setCertificationRun(undefined);
    setCanvasNotice("ready");
  }

  function resetLevel() {
    setGraphs((current) => ({ ...current, [selectedLevel.id]: selectedLevel.initialGraph }));
    setRuns((current) => ({ ...current, [selectedLevel.id]: createEmptyRunState() }));
    setCertificationValuesByLevel((current) => ({ ...current, [selectedLevel.id]: defaultCertificationValues(selectedLevel) }));
    setSelection(selectedLevel.initialGraph.nodes[0] ? { type: "node", id: selectedLevel.initialGraph.nodes[0].id } : undefined);
    clearWireSource();
    setWirePointer(undefined);
    setTraceSelection(undefined);
    setInspectorTab("summary");
    setCompletionNotice(undefined);
    setCertificationRun(undefined);
    setViewport({ x: 0, y: 0, scale: 1 });
    setCanvasNotice("reset");
  }

  function runVisible() {
    if (componentLevelLocked) {
      setCanvasNotice(`missing requirements: ${missingComponentRequirements.join(", ")}`);
      return;
    }
    const result = runTests(graph, registry, selectedLevel.visibleTests);
    setRuns((current) => {
      const previous = current[selectedLevel.id] ?? createEmptyRunState();
      return {
        ...current,
        [selectedLevel.id]: {
          ...previous,
          visible: result,
          hidden: undefined,
          stats: {
            ...previous.stats,
            visibleRuns: previous.stats.visibleRuns + 1,
            failedRuns: previous.stats.failedRuns + (result.status === "pass" ? 0 : 1)
          }
        }
      };
    });
    setTraceSelection(defaultTraceSelection(result, "visible"));
    focusFirstBadNode(result);
  }

  function runHidden() {
    if (componentLevelLocked) {
      setCanvasNotice(`missing requirements: ${missingComponentRequirements.join(", ")}`);
      return;
    }
    if (certificationErrors.length > 0) {
      setCanvasNotice("validation variant needs valid inputs");
      return;
    }
    const certificationTests = selectedLevel.certification
      ? [...selectedLevel.certification.makePublicTests(graph, certificationValues), ...selectedLevel.hiddenTests.map((testCase) => ({ testCase }))]
      : selectedLevel.hiddenTests.map((testCase) => ({ testCase }));
    const result = runCertificationTests(graph, certificationTests);
    const preview = buildCertificationRunPreview({
      level: selectedLevel,
      graph,
      modules,
      certificationTests,
      result,
      values: certificationValues
    });
    if (preview) {
      setMissionOpen(false);
      setCompletionNotice(undefined);
      setCertificationRun(preview);
      setCanvasNotice("validation running");
      return;
    }
    finishCertificationRun(result);
  }

  function finishCertificationRun(result: RunTestsResult) {
    setCertificationRun(undefined);
    setRuns((current) => {
      const previous = current[selectedLevel.id] ?? createEmptyRunState();
      return {
        ...current,
        [selectedLevel.id]: {
          ...previous,
          hidden: result,
          stats: {
            ...previous.stats,
            hiddenRuns: previous.stats.hiddenRuns + 1,
            failedRuns: previous.stats.failedRuns + (result.status === "pass" ? 0 : 1)
          }
        }
      };
    });
    setTraceSelection(defaultTraceSelection(result, "hidden"));
    if (result.status === "pass") {
      completeSelectedLevel(result);
    } else {
      focusFirstBadNode(result);
    }
  }

  function completeSelectedLevel(hiddenResult: RunTestsResult) {
    const nextLevel = nextLevelAfter(selectedLevel.id);
    if (selectedComponent && componentFlow && !componentAvailable) {
      componentFlow.onComponentAvailable({
        level: selectedLevel,
        component: selectedComponent,
        graph,
        visible: runState.visible,
        hidden: hiddenResult
      });
    }

    setMissionOpen(false);
    setCompletionNotice({
      componentTitle: selectedComponent?.title,
      nextLevelTitle: nextLevel?.title,
      routeComplete: !nextLevel
    });
    setCanvasNotice(selectedComponent ? `${selectedComponent.title} available` : "level complete");

    if (!nextLevel) return;
    if (autoAdvanceTimerRef.current) window.clearTimeout(autoAdvanceTimerRef.current);
    autoAdvanceTimerRef.current = window.setTimeout(() => {
      autoAdvanceTimerRef.current = undefined;
      selectLevel(nextLevel);
    }, 1400);
  }

  function nextLevelAfter(levelId: string) {
    const currentIndex = levels.findIndex((level) => level.id === levelId);
    if (currentIndex < 0) return undefined;
    return levels.slice(currentIndex + 1).find(isLevelPlayable);
  }

  function updateCertificationValue(controlId: string, value: CertificationControlValue) {
    setCertificationValuesByLevel((current) => ({
      ...current,
      [selectedLevel.id]: {
        ...certificationValuesForLevel(selectedLevel, current[selectedLevel.id]),
        [controlId]: value
      }
    }));
    setRuns((current) => {
      const previous = current[selectedLevel.id] ?? createEmptyRunState();
      return { ...current, [selectedLevel.id]: { ...previous, hidden: undefined } };
    });
    setCompletionNotice(undefined);
  }

  function runCertificationTests(defaultGraph: GraphSpec, certificationTests: CertificationTestSpec[]): RunTestsResult {
    const cases = certificationTests.map((item) => runTestCaseDetailed(item.graph ?? defaultGraph, registry, item.testCase));
    const results = cases.flatMap((testCase) => testCase.results);
    return { status: statusFromResults(results), results, cases };
  }

  function buildCertificationRunPreview({
    level,
    graph: baseGraph,
    modules: graphModules,
    certificationTests,
    result,
    values
  }: {
    level: LevelSpec;
    graph: GraphSpec;
    modules: ModuleDef[];
    certificationTests: CertificationTestSpec[];
    result: RunTestsResult;
    values: Record<string, CertificationControlValue>;
  }): CertificationRunPreview | undefined {
    const publicSpec = certificationTests[0];
    const publicCase = result.cases[0];
    if (!publicSpec || !publicCase) return undefined;
    const runGraph = publicSpec.graph ?? baseGraph;
    return {
      id: `${level.id}:${publicSpec.testCase.id}:${Date.now()}`,
      levelId: level.id,
      title: level.certification?.title ?? "Validation",
      graph: runGraph,
      controls: level.certification?.controls ?? [],
      values,
      testCase: publicSpec.testCase,
      publicCase,
      result,
      codeLines: buildCertificationAnimationLines(level, runGraph, graphModules, publicSpec.testCase, level.certification?.controls ?? [], values),
      outputLines: buildCertificationOutputLines(publicSpec.testCase, publicCase, result)
    };
  }

  function focusFirstBadNode(result: RunTestsResult) {
    const firstBadNodeId = result.results.find((item) => item.firstBadNodeId)?.firstBadNodeId;
    if (firstBadNodeId) setSelection({ type: "node", id: firstBadNodeId.split(".")[0] });
  }

  function insertTransposeRepair() {
    const nextGraph = selectedLevel.id === "ch0_3_transpose_graph" ? withKTranspose(graph) : withWeightTranspose(graph);
    setGraphs((current) => ({ ...current, [selectedLevel.id]: nextGraph }));
    setRuns((current) => {
      const previous = current[selectedLevel.id] ?? createEmptyRunState();
      return {
        ...current,
        [selectedLevel.id]: {
          ...previous,
          visible: undefined,
          hidden: undefined,
          stats: { ...previous.stats, hintsUsed: previous.stats.hintsUsed + 1 }
        }
      };
    });
    setSelection({ type: "node", id: selectedLevel.id === "ch0_3_transpose_graph" ? "k_transpose" : "weight_transpose" });
    clearWireSource();
    setWirePointer(undefined);
    setCanvasNotice("patched");
  }

  const canInsertTranspose =
    (selectedLevel.id === "ch0_2_matmul_graph" &&
      graph.nodes.some((node) => node.id === "weight") &&
      graph.nodes.some((node) => node.id === "matmul") &&
      !graph.nodes.some((node) => node.id === "weight_transpose")) ||
    (selectedLevel.id === "ch0_3_transpose_graph" &&
      graph.nodes.some((node) => node.id === "k") &&
      graph.nodes.some((node) => node.id === "qk_matmul") &&
      !graph.nodes.some((node) => node.id === "k_transpose"));
  const transposeLabel = selectedLevel.id === "ch0_3_transpose_graph" ? "Auto K Transpose" : "Auto Transpose";

  function setGraphForSelectedLevel(updater: (current: GraphSpec) => GraphSpec, invalidateRuns = true) {
    setGraphs((current) => {
      const currentGraph = current[selectedLevel.id] ?? selectedLevel.initialGraph;
      return { ...current, [selectedLevel.id]: updater(currentGraph) };
    });
    if (invalidateRuns) {
      setRuns((current) => {
        const previous = current[selectedLevel.id] ?? createEmptyRunState();
        return { ...current, [selectedLevel.id]: { ...previous, visible: undefined, hidden: undefined } };
      });
    }
  }

  function moveNode(nodeId: string, x: number, y: number) {
    setGraphForSelectedLevel(
      (current) => ({
        ...current,
        nodes: current.nodes.map((node) => (node.id === nodeId ? { ...node, position: { x, y } } : node))
      }),
      false
    );
  }

  function beginNodeDrag(event: ReactPointerEvent<HTMLDivElement>, node: GraphNode) {
    if ((event.target as HTMLElement).closest("[data-graph-port='true'], input, select, textarea, button")) return;
    const point = clientToWorld(event.clientX, event.clientY);
    if (!point) return;
    nodeDragRef.current = {
      nodeId: node.id,
      offsetX: point.x - node.position.x,
      offsetY: point.y - node.position.y
    };
    setSelection({ type: "node", id: node.id });
    setCanvasNotice("moving");
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handleCanvasPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    updateWirePointer(event);
    const panState = panDragRef.current;
    if (panState) {
      setViewport({
        ...panState.startViewport,
        x: panState.startViewport.x + event.clientX - panState.startClientX,
        y: panState.startViewport.y + event.clientY - panState.startClientY
      });
      return;
    }

    const dragState = nodeDragRef.current;
    if (!dragState) return;
    const point = clientToWorld(event.clientX, event.clientY);
    if (!point) return;
    const position = clampNodePosition(point.x - dragState.offsetX, point.y - dragState.offsetY);
    moveNode(dragState.nodeId, position.x, position.y);
  }

  function beginCanvasPan(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    const target = event.target as HTMLElement;
    if (target.closest(".graphNode, .graphCanvasToolbar, .graphCanvasHud, .graphComponentLibrary, button, input, select, textarea")) return;
    if (wireSourceRef.current) return;
    panDragRef.current = {
      startClientX: event.clientX,
      startClientY: event.clientY,
      startViewport: viewport
    };
    setSelection(undefined);
    setCanvasNotice("panning");
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function endCanvasInteraction() {
    const wasMoving = Boolean(nodeDragRef.current || panDragRef.current);
    nodeDragRef.current = undefined;
    panDragRef.current = undefined;
    if (wasMoving) setCanvasNotice(wireSourceRef.current ? "select input port" : "ready");
  }

  function handleCanvasDragOver(event: ReactDragEvent<HTMLDivElement>) {
    if (!event.dataTransfer.types.includes(moduleDragMime)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setDragOverCanvas(true);
  }

  function handleCanvasDrop(event: ReactDragEvent<HTMLDivElement>) {
    const moduleId = event.dataTransfer.getData(moduleDragMime);
    if (!moduleId) return;
    event.preventDefault();
    setDragOverCanvas(false);
    const point = clientToWorld(event.clientX, event.clientY);
    if (!point) return;
    addModuleNode(moduleId, point.x - graphNodeWidth / 2, point.y - graphNodeMinHeight / 2);
  }

  function addModuleNode(moduleId: string, x = 320, y = 220) {
    if (componentLevelLocked) {
      setCanvasNotice(`blueprint locked: ${missingComponentRequirements.join(", ")}`);
      return;
    }
    const module = registry.get(moduleId);
    const forbidden = selectedLevel.constraints?.forbiddenModules?.includes(moduleId);
    if (forbidden) {
      setCanvasNotice(`${module.label} is locked out`);
      return;
    }
    if (selectedLevel.constraints?.maxNodes && graph.nodes.length >= selectedLevel.constraints.maxNodes) {
      setCanvasNotice(`node budget ${selectedLevel.constraints.maxNodes}/${selectedLevel.constraints.maxNodes}`);
      return;
    }

    const template = getLevelNodeTemplate(selectedLevel.id, graph, moduleId);
    const nodeId = template.id ?? uniqueNodeId(graph, moduleId);
    const position = clampNodePosition(x, y);
    const nextNode: GraphNode = {
      id: nodeId,
      moduleId,
      params: { ...cloneParams(module.defaultParams), ...cloneParams(template.params ?? {}) },
      position
    };
    setGraphForSelectedLevel((current) => ({ ...current, nodes: [...current.nodes, nextNode] }));
    setSelection({ type: "node", id: nodeId });
    setCanvasNotice(`${module.label} added`);
  }

  function handlePortClick(node: GraphNode, module: ModuleDef, port: PortDef) {
    if (suppressNextPortClickRef.current) {
      suppressNextPortClickRef.current = false;
      return;
    }

    if (port.direction === "out") {
      armWireSource(node, port);
      return;
    }

    completeWireToInput(node, port);
  }

  function handlePortPointerDown(event: ReactPointerEvent<HTMLButtonElement>, node: GraphNode, port: PortDef) {
    event.stopPropagation();
    if (port.direction !== "out") return;
    event.preventDefault();
    armWireSource(node, port);
    updateWirePointer(event);
  }

  function handlePortPointerUp(event: ReactPointerEvent<HTMLButtonElement>, node: GraphNode, port: PortDef) {
    if (port.direction !== "in") return;
    if (!wireSourceRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    completeWireToInput(node, port);
    suppressNextPortClickRef.current = true;
  }

  function armWireSource(node: GraphNode, port: PortDef) {
    const source = { nodeId: node.id, portId: port.id, dtype: port.emits };
    wireSourceRef.current = source;
    setWireSource(source);
    setWirePointer(undefined);
    setSelection({ type: "node", id: node.id });
    setCanvasNotice(`${node.id}.${port.id} armed`);
  }

  function completeWireToInput(node: GraphNode, port: PortDef) {
    if (componentLevelLocked) {
      setCanvasNotice(`blueprint locked: ${missingComponentRequirements.join(", ")}`);
      return;
    }
    const activeWire = wireSourceRef.current;
    if (!activeWire) {
      setSelection({ type: "node", id: node.id });
      setCanvasNotice("select output port");
      return;
    }

    if (activeWire.nodeId === node.id) {
      setCanvasNotice("self edge rejected");
      return;
    }

    const fromNode = graph.nodes.find((item) => item.id === activeWire.nodeId);
    const fromModule = fromNode ? registry.maybeGet(fromNode.moduleId) : undefined;
    const fromPort = fromModule?.outputs.find((item) => item.id === activeWire.portId);
    if (!fromNode || !fromPort) {
      clearWireSource();
      setCanvasNotice("source port missing");
      return;
    }

    if (!portsCompatible(fromPort, port)) {
      setCanvasNotice(`${fromPort.emits ?? "value"} cannot enter ${port.accepts?.join("/") ?? "input"}`);
      return;
    }

    if (selectedLevel.constraints?.maxEdges && graph.edges.length >= selectedLevel.constraints.maxEdges && !graph.edges.some((edge) => edge.to.nodeId === node.id && edge.to.portId === port.id)) {
      setCanvasNotice(`edge budget ${selectedLevel.constraints.maxEdges}/${selectedLevel.constraints.maxEdges}`);
      return;
    }

    const nextEdge: GraphEdge = {
      id: uniqueEdgeId(graph, activeWire, { nodeId: node.id, portId: port.id }),
      from: { nodeId: activeWire.nodeId, portId: activeWire.portId },
      to: { nodeId: node.id, portId: port.id }
    };
    setGraphForSelectedLevel((current) => ({
      ...current,
      edges: [...current.edges.filter((edge) => !(edge.to.nodeId === node.id && edge.to.portId === port.id)), nextEdge]
    }));
    clearWireSource();
    setWirePointer(undefined);
    setSelection({ type: "edge", id: nextEdge.id });
    setCanvasNotice(`${nextEdge.from.nodeId}.${nextEdge.from.portId} -> ${nextEdge.to.nodeId}.${nextEdge.to.portId}`);
  }

  function clearWireSource() {
    wireSourceRef.current = undefined;
    setWireSource(undefined);
  }

  function deleteSelection() {
    if (!selection) return;
    if (selection.type === "edge") {
      setGraphForSelectedLevel((current) => ({ ...current, edges: current.edges.filter((edge) => edge.id !== selection.id) }));
      setSelection(undefined);
      clearWireSource();
      setWirePointer(undefined);
      setCanvasNotice("edge deleted");
      return;
    }

    const remainingNodes = graph.nodes.filter((node) => node.id !== selection.id);
    if (!remainingNodes.length) {
      setCanvasNotice("last node kept");
      return;
    }
    setGraphForSelectedLevel((current) => ({
      ...current,
      nodes: current.nodes.filter((node) => node.id !== selection.id),
      edges: current.edges.filter((edge) => edge.from.nodeId !== selection.id && edge.to.nodeId !== selection.id),
      outputNodes: current.outputNodes.filter((nodeId) => nodeId !== selection.id)
    }));
    setSelection({ type: "node", id: remainingNodes[0].id });
    clearWireSource();
    setWirePointer(undefined);
    setCanvasNotice("node deleted");
  }

  function rewireSelectedEdge() {
    if (!selectedEdge) return;
    const sourcePort = getOutputPort(graph, registry.list(), selectedEdge.from);
    setGraphForSelectedLevel((current) => ({ ...current, edges: current.edges.filter((edge) => edge.id !== selectedEdge.id) }));
    const source = { ...selectedEdge.from, dtype: sourcePort?.emits };
    wireSourceRef.current = source;
    setWireSource(source);
    setWirePointer(undefined);
    setSelection({ type: "node", id: selectedEdge.from.nodeId });
    setCanvasNotice("select new input");
  }

  function updateNodeParam(nodeId: string, key: string, value: unknown) {
    setGraphForSelectedLevel((current) => ({
      ...current,
      nodes: current.nodes.map((node) =>
        node.id === nodeId
          ? {
              ...node,
              params: {
                ...node.params,
                [key]: value
              }
            }
          : node
      )
    }));
    setCanvasNotice(`${key} updated`);
  }

  function autoLayoutGraph() {
    const nextGraph = layoutGraph(graph);
    setGraphForSelectedLevel(() => nextGraph, false);
    setCanvasNotice("layout applied");
  }

  function updateWirePointer(event: ReactPointerEvent<HTMLElement>) {
    if (!wireSourceRef.current) return;
    const point = clientToWorld(event.clientX, event.clientY);
    if (point) setWirePointer(point);
  }

  function cancelWire() {
    clearWireSource();
    setWirePointer(undefined);
    nodeDragRef.current = undefined;
    panDragRef.current = undefined;
    setCanvasNotice("wire cancelled");
  }

  function handleCanvasContextMenu(event: ReactMouseEvent<HTMLDivElement>) {
    if (!wireSourceRef.current) return;
    event.preventDefault();
    cancelWire();
  }

  function clientToWorld(clientX: number, clientY: number): CanvasPoint | undefined {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return undefined;
    return viewportToWorld(clientX - rect.left, clientY - rect.top);
  }

  function viewportToWorld(x: number, y: number): CanvasPoint {
    return {
      x: (x - viewport.x) / viewport.scale,
      y: (y - viewport.y) / viewport.scale
    };
  }

  function zoomCanvas(factor: number, anchor?: CanvasPoint) {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const anchorPoint = anchor ?? { x: rect.width / 2, y: rect.height / 2 };
    const worldAnchor = viewportToWorld(anchorPoint.x, anchorPoint.y);
    const nextScale = clamp(viewport.scale * factor, minCanvasScale, maxCanvasScale);
    setViewport({
      scale: nextScale,
      x: anchorPoint.x - worldAnchor.x * nextScale,
      y: anchorPoint.y - worldAnchor.y * nextScale
    });
    setCanvasNotice(`zoom ${Math.round(nextScale * 100)}%`);
  }

  function handleCanvasWheel(event: ReactWheelEvent<HTMLDivElement>) {
    event.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    zoomCanvas(event.deltaY < 0 ? 1.1 : 0.9, {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    });
  }

  function resetView() {
    setViewport({ x: 0, y: 0, scale: 1 });
    setCanvasNotice("view reset");
  }

  function focusNode(nodeId: string, tab: InspectorTab = "shape") {
    const realNodeId = nodeId.split(".")[0];
    const node = graph.nodes.find((item) => item.id === realNodeId);
    if (!node) return;
    setSelection({ type: "node", id: realNodeId });
    setInspectorTab(tab);
    const rect = canvasRef.current?.getBoundingClientRect();
    if (rect) {
      setViewport((current) => ({
        ...current,
        x: rect.width / 2 - (node.position.x + graphNodeWidth / 2) * current.scale,
        y: rect.height / 2 - (node.position.y + graphNodeMinHeight / 2) * current.scale
      }));
    }
    setCanvasNotice(`focused ${realNodeId}`);
  }

  function showNextHint() {
    const nextNodeId = firstBadNodeIdFromRunState(runState);
    setRuns((current) => {
      const previous = current[selectedLevel.id] ?? createEmptyRunState();
      return {
        ...current,
        [selectedLevel.id]: {
          ...previous,
          stats: { ...previous.stats, hintsUsed: previous.stats.hintsUsed + 1 }
        }
      };
    });
    if (nextNodeId) focusNode(nextNodeId);
    setCanvasNotice("next step shown");
  }

  function beginWorkbenchResize(event: ReactPointerEvent<HTMLDivElement>, region: ResizeRegion) {
    event.preventDefault();
    event.stopPropagation();
    const startX = event.clientX;
    const startY = event.clientY;
    const startLayout = layout;
    const workbenchRect = workbenchRef.current?.getBoundingClientRect();
    const stageRect = stageRef.current?.getBoundingClientRect();
    const previousCursor = document.body.style.cursor;
    const previousUserSelect = document.body.style.userSelect;
    document.body.style.cursor = region === "trace" ? "row-resize" : "col-resize";
    document.body.style.userSelect = "none";

    function handleMove(moveEvent: PointerEvent) {
      const dx = moveEvent.clientX - startX;
      const dy = moveEvent.clientY - startY;
      setLayout((current) => {
        if (region === "sidebar") {
          const totalWidth = workbenchRect?.width ?? 1280;
          const maxSidebarWidth = Math.max(minSidebarWidth, totalWidth - startLayout.inspectorWidth - minCenterWidth - resizeHandleSize * 2);
          return {
            ...current,
            sidebarWidth: clamp(startLayout.sidebarWidth + dx, minSidebarWidth, Math.min(460, maxSidebarWidth))
          };
        }
        if (region === "inspector") {
          const totalWidth = workbenchRect?.width ?? 1280;
          const maxInspectorWidth = Math.max(minInspectorWidth, totalWidth - startLayout.sidebarWidth - minCenterWidth - resizeHandleSize * 2);
          return {
            ...current,
            inspectorWidth: clamp(startLayout.inspectorWidth - dx, minInspectorWidth, Math.min(540, maxInspectorWidth))
          };
        }
        const stageHeight = stageRect?.height ?? 720;
        const maxTraceHeight = Math.max(minTraceHeight, Math.min(420, stageHeight - 230));
        return {
          ...current,
          traceHeight: clamp(startLayout.traceHeight - dy, minTraceHeight, maxTraceHeight)
        };
      });
    }

    function handleEnd() {
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousUserSelect;
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleEnd);
      window.removeEventListener("pointercancel", handleEnd);
    }

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleEnd);
    window.addEventListener("pointercancel", handleEnd);
  }

  function resetResizeRegion(region: ResizeRegion) {
    setLayout((current) => {
      if (region === "sidebar") return { ...current, sidebarWidth: defaultWorkbenchLayout.sidebarWidth };
      if (region === "inspector") return { ...current, inspectorWidth: defaultWorkbenchLayout.inspectorWidth };
      return { ...current, traceHeight: defaultWorkbenchLayout.traceHeight };
    });
  }

  const wirePreview = wireSource && wirePointer ? getWirePreviewPath(graph, modules, wireSource, wirePointer, portAnchors) : undefined;
  const t = (text: string) => graphText(language, text);

  return (
    <GraphLanguageContext.Provider value={language}>
    <section ref={workbenchRef} className="graphWorkbench" style={workbenchStyle}>
      <aside className="graphSidebar panel">
        <div className="panelHeader">
          <Wrench size={18} />
          <h2>{t("Component Library")}</h2>
        </div>
        <GraphComponentLibrary level={selectedLevel} registry={registry} />
        <section className="graphChallengeMapLauncher">
          <button className="graphChallengeMapToggle" type="button" onClick={() => setChallengeMapOpen((current) => !current)}>
            <BookOpenText size={16} />
            <span>
              <b>{t("Challenge Map")}</b>
              <small>{challengeMapOpen ? t("Hide level route") : t("Open level route")}: {t(selectedLevel.title)}</small>
            </span>
            <code>{levels.length}</code>
          </button>
        </section>
      </aside>

      <div
        className="graphResizeHandle vertical"
        role="separator"
        aria-label="Resize graph levels"
        aria-orientation="vertical"
        title={t("Drag to resize Graph Levels")}
        onPointerDown={(event) => beginWorkbenchResize(event, "sidebar")}
        onDoubleClick={() => resetResizeRegion("sidebar")}
      />

      <section ref={stageRef} className="graphStage panel" style={stageStyle}>
        <div className="graphStageHeader">
          <div>
            <p className="eyebrow">{t(componentFlow ? "Component Builder" : "Graph Challenge")}</p>
            <h2>{t(selectedLevel.title)}</h2>
            <small>{t(selectedLevel.goal)}</small>
            {selectedComponent ? (
              <ComponentLifecycleHeader
                component={selectedComponent}
                available={componentAvailable}
                visiblePassed={runState.visible?.status === "pass"}
                missingRequirements={missingComponentRequirements}
                availableComponentIds={availableComponentIds}
              />
            ) : null}
          </div>
          <div className="graphRunBar">
            <button className="ghostButton" onClick={() => setMissionOpen(true)}>
              <BookOpenText size={15} />
              {t("Mission")}
            </button>
            {canInsertTranspose ? (
              <button className="ghostButton" onClick={insertTransposeRepair}>
                <Wrench size={15} />
                {t(transposeLabel)}
              </button>
            ) : null}
            <button className="ghostButton" onClick={resetLevel}>
              <RotateCcw size={15} />
              {t("Reset")}
            </button>
            <button className="ghostButton" onClick={autoLayoutGraph}>
              <GitBranchPlus size={15} />
              {t("Auto Layout")}
            </button>
            <button className="runButton" disabled={componentLevelLocked} onClick={runVisible}>
              <Play size={15} />
              {t("Check Current Task")}
            </button>
            <button className="runButton" disabled={!canSubmitCertification} onClick={runHidden}>
              <Play size={15} />
              {t("Submit Validation")}
            </button>
            <button className="ghostButton" disabled={!selection} onClick={deleteSelection}>
              <Trash2 size={15} />
              {t("Delete")}
            </button>
          </div>
        </div>

        {missionOpen ? (
          <GraphMissionModal
            level={selectedLevel}
            graph={graph}
            modules={modules}
            runState={runState}
            onRunVisible={runVisible}
            onShowHint={showNextHint}
            onClose={() => setMissionOpen(false)}
          />
        ) : null}

        {completionNotice ? <GraphCompletionNotice notice={completionNotice} /> : null}
        {challengeMapOpen ? (
          <GraphChallengeMapOverlay
            levels={levels}
            selectedLevelId={selectedLevel.id}
            componentFlow={componentFlow}
            availableComponentIds={availableComponentIds}
            onSelect={(level) => {
              selectLevel(level);
              setChallengeMapOpen(false);
            }}
            onClose={() => setChallengeMapOpen(false)}
          />
        ) : null}
        {certificationRun ? (
          <GraphCertificationRunModal
            run={certificationRun}
            onComplete={() => finishCertificationRun(certificationRun.result)}
          />
        ) : null}

        <div
          ref={canvasRef}
          className={`graphCanvas ${dragOverCanvas ? "dragOver" : ""} ${wireSource ? "wiring" : ""}`}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragOverCanvas(false);
          }}
          onDragOver={handleCanvasDragOver}
          onDrop={handleCanvasDrop}
          onPointerDown={beginCanvasPan}
          onPointerMove={handleCanvasPointerMove}
          onPointerUp={endCanvasInteraction}
          onPointerCancel={endCanvasInteraction}
          onWheel={handleCanvasWheel}
          onContextMenu={handleCanvasContextMenu}
          onDoubleClick={() => {
            if (wireSource) cancelWire();
          }}
        >
          <div className="graphCanvasHud">
            <span>
              <MousePointer2 size={14} />
              {canvasNotice}
            </span>
            {wireSource ? <code>{`${wireSource.nodeId}.${wireSource.portId} ->`}</code> : <code>{graph.nodes.length} nodes / {graph.edges.length} edges</code>}
            <code>{Math.round(viewport.scale * 100)}%</code>
          </div>
          {componentLevelLocked ? (
            <div className="graphComponentLockOverlay">
              <AlertTriangle size={18} />
              <b>{t("Blueprint locked")}</b>
              <span>{t("Unlock required components first")}: {missingComponentRequirements.join(", ")}</span>
            </div>
          ) : null}
          {selectedLevel.certification && runState.visible?.status === "pass" ? (
            <GraphCertificationPanel
              certification={selectedLevel.certification}
              values={certificationValues}
              errors={certificationErrors}
              taskPassed={runState.visible?.status === "pass"}
              certificationResult={runState.hidden}
              onChange={updateCertificationValue}
              onSubmit={runHidden}
            />
          ) : null}
          <div className="graphCanvasToolbar">
            <button className="iconButton" title="Zoom out" onClick={() => zoomCanvas(0.9)}>
              <Minus size={15} />
            </button>
            <button className="iconButton" title="Reset view" onClick={resetView}>
              <Move size={15} />
            </button>
            <button className="iconButton" title="Zoom in" onClick={() => zoomCanvas(1.1)}>
              <Plus size={15} />
            </button>
          </div>
          <div
            className="graphWorld"
            style={{
              width: graphWorldWidth,
              height: graphWorldHeight,
              transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.scale})`
            }}
          >
            <TargetGhostGraph targetGraph={selectedLevel.targetGraph} currentGraph={graph} />
            <svg
              className="graphEdgeLayer"
              width={graphWorldLayerWidth}
              height={graphWorldLayerHeight}
              viewBox={`${graphWorldMinX} ${graphWorldMinY} ${graphWorldLayerWidth} ${graphWorldLayerHeight}`}
              style={{ left: graphWorldMinX, top: graphWorldMinY, width: graphWorldLayerWidth, height: graphWorldLayerHeight }}
              role="img"
              aria-label="graph edges"
            >
              {wirePreview ? <path className="graphWirePreview" d={wirePreview} /> : null}
              {graph.edges.map((edge) => {
                const fromNode = graph.nodes.find((node) => node.id === edge.from.nodeId);
                const toNode = graph.nodes.find((node) => node.id === edge.to.nodeId);
                if (!fromNode || !toNode) return null;
                const fromModule = registry.get(fromNode.moduleId);
                const toModule = registry.get(toNode.moduleId);
                const from = getPortAnchor(fromNode, fromModule, edge.from.portId, "out", portAnchors);
                const to = getPortAnchor(toNode, toModule, edge.to.portId, "in", portAnchors);
                const selected = selectedEdge?.id === edge.id;
                const edgeCaseLabel = getEdgeCaseLabel(selectedLevel, graph, modules, edge);
                const pathD = `M ${from.x} ${from.y} C ${from.x + 88} ${from.y}, ${to.x - 88} ${to.y}, ${to.x} ${to.y}`;
                return (
                  <g
                    key={edge.id}
                    className={`graphEdge ${selected ? "selected" : ""}`}
                    onPointerDown={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                      setSelection({ type: "edge", id: edge.id });
                      setCanvasNotice("edge selected");
                    }}
                  >
                    <path className="graphEdgeHit" d={pathD} />
                    <path className="graphEdgeLine" d={pathD} />
                    <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 8}>
                      {`${edge.from.portId} -> ${edge.to.portId}`}
                    </text>
                    {edgeCaseLabel ? (
                      <text className="graphEdgeCaseText" x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 + 10}>
                        {edgeCaseLabel}
                      </text>
                    ) : null}
                  </g>
                );
              })}
            </svg>
            {graph.nodes.map((node) => {
              const module = registry.get(node.moduleId);
              const selected = selectedNode?.id === node.id;
              const failed = firstBadNodeIds(runState).has(node.id);
              return (
                <div
                  key={node.id}
                  className={`graphNode ${selected ? "selected" : ""} ${failed ? "failed" : ""}`}
                  style={{ left: node.position.x, top: node.position.y }}
                  onPointerDown={(event) => beginNodeDrag(event, node)}
                  onClick={() => setSelection({ type: "node", id: node.id })}
                  role="button"
                  tabIndex={0}
                >
                  <div className="graphNodeHeader">
                    <span>{t(module.category)}</span>
                    <b>{node.id}</b>
                    <small>{t(module.label)}</small>
                  </div>
                  <GraphNodeCaseChips level={selectedLevel} graph={graph} modules={modules} node={node} module={module} onParamChange={updateNodeParam} />
                  <div className="graphPorts">
                    <GraphPortColumn
                      node={node}
                      module={module}
                      direction="in"
                      wireSource={wireSource}
                      wireSourcePort={wireSourcePort}
                      onPortClick={handlePortClick}
                      onPortPointerDown={handlePortPointerDown}
                      onPortPointerUp={handlePortPointerUp}
                    />
                    <GraphPortColumn
                      node={node}
                      module={module}
                      direction="out"
                      wireSource={wireSource}
                      wireSourcePort={wireSourcePort}
                      onPortClick={handlePortClick}
                      onPortPointerDown={handlePortPointerDown}
                      onPortPointerUp={handlePortPointerUp}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <div
          className="graphResizeHandle horizontal"
          role="separator"
        aria-label="Resize trace timeline"
        aria-orientation="horizontal"
        title={t("Drag to resize Trace Timeline")}
          onPointerDown={(event) => beginWorkbenchResize(event, "trace")}
          onDoubleClick={() => resetResizeRegion("trace")}
        />
        <GraphTraceTimeline
          visible={runState.visible}
          hidden={runState.hidden}
          selection={traceSelection}
          onSelect={(nextSelection, frame) => {
            setTraceSelection(nextSelection);
            focusNode(frame.nodeId);
          }}
        />
      </section>

      <div
        className="graphResizeHandle vertical"
        role="separator"
        aria-label="Resize inspector"
        aria-orientation="vertical"
        title={t("Drag to resize Inspector")}
        onPointerDown={(event) => beginWorkbenchResize(event, "inspector")}
        onDoubleClick={() => resetResizeRegion("inspector")}
      />

      <aside className="graphInspector panel">
        <div className="panelHeader">
          <CheckCircle2 size={18} />
          <h2>{t("Inspector")}</h2>
        </div>
        {selectedNode && selectedModule ? (
          <GraphInspectorNodePanel
            node={selectedNode}
            module={selectedModule}
            graph={graph}
            runState={runState}
            activeTraceCase={activeTraceCase}
            activeTraceFrame={activeTraceFrame}
            tab={inspectorTab}
            onTabChange={setInspectorTab}
            onParamChange={updateNodeParam}
            onLocateNode={focusNode}
            onShowNextStep={showNextHint}
          />
        ) : null}
        {selectedEdge ? (
          <section className="graphInspectorBlock">
            <p className="eyebrow">{t("Selected Edge")}</p>
            <h3>{selectedEdge.id}</h3>
            <code>{`${selectedEdge.from.nodeId}.${selectedEdge.from.portId} -> ${selectedEdge.to.nodeId}.${selectedEdge.to.portId}`}</code>
            <div className="graphInspectorActions">
              <button className="ghostButton" onClick={rewireSelectedEdge}>
                <GitBranchPlus size={14} />
                {t("Rewire Target")}
              </button>
              <button className="ghostButton" onClick={deleteSelection}>
                <Trash2 size={14} />
                {t("Delete Edge")}
              </button>
            </div>
          </section>
        ) : null}

        <GraphCodePanel level={selectedLevel} graph={graph} modules={modules} />
        <GraphRankPanel level={selectedLevel} graph={graph} runState={runState} />
        <GraphRunPanel title="Task Check" result={runState.visible} onLocateNode={focusNode} onShowNextStep={showNextHint} />
        <GraphRunPanel title="Validation Check" result={runState.hidden} locked={certificationLocked} onLocateNode={focusNode} onShowNextStep={showNextHint} />
      </aside>
    </section>
    </GraphLanguageContext.Provider>
  );
}

function GraphMissionModal({
  level,
  graph,
  modules,
  runState,
  onRunVisible,
  onShowHint,
  onClose
}: {
  level: LevelSpec;
  graph: GraphSpec;
  modules: ModuleDef[];
  runState: RunState;
  onRunVisible: () => void;
  onShowHint: () => void;
  onClose: () => void;
}) {
  const { language, t } = useGraphT();
  const onboarding = level.onboarding;
  const caseStudy = level.caseStudy;
  const checklist = getChecklistItems(level, graph);
  const coach = getNextStepCoach(level, graph, runState, language);
  return (
    <div className="graphMissionOverlay" role="dialog" aria-modal="true" aria-labelledby="graphMissionTitle">
      <section className="graphMissionPanel">
        <div className="graphMissionHeader">
          <div>
            <p className="eyebrow">{t("Mission")}</p>
            <h3 id="graphMissionTitle">{t(level.title)}</h3>
          </div>
          <button className="iconButton" type="button" title={t("Collapse mission")} onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="graphMissionCopy">
          <b>{t(caseStudy?.title ?? onboarding?.story ?? level.goal)}</b>
          <small>{t(caseStudy?.narrative ?? onboarding?.startingProblem ?? "Check the current task to reveal the first failing node, then repair the graph.")}</small>
        </div>

        {caseStudy ? <GraphTaskDataPanels level={level} graph={graph} modules={modules} /> : null}

        <div className="graphMissionSteps">
          <div>
            <b>{t(caseStudy ? "Task Check" : "First action")}</b>
            <small>{t(caseStudy?.playerQuestion ?? onboarding?.firstAction ?? "Click Check Current Task.")}</small>
          </div>
          <div>
            <b>{t(caseStudy ? "Pass Condition" : "Win condition")}</b>
            <small>{t(caseStudy?.successObservation ?? onboarding?.winCondition ?? "Task check and validation pass.")}</small>
          </div>
        </div>

        {checklist.length ? (
          <div className="graphMissionChecklist">
            {checklist.map((item) => (
              <span key={item.label} className={item.done ? "done" : ""}>
                {item.done ? "[x]" : "[ ]"} {item.label}
              </span>
            ))}
          </div>
        ) : null}

        <div className="graphNextCoach">
          <b>{t("Next step")}</b>
          <small>{coach}</small>
          <div className="graphMissionActions">
            <button className="ghostButton" onClick={onRunVisible}>
              <Play size={14} />
              {t("Check Current Task")}
            </button>
            <button className="ghostButton" onClick={onShowHint}>
              <MousePointer2 size={14} />
              {t("Show Hint")}
            </button>
            <button className="ghostButton" onClick={onClose}>
              {t("Collapse")}
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

function TargetGhostGraph({
  targetGraph,
  currentGraph
}: {
  targetGraph?: GraphSpec;
  currentGraph: GraphSpec;
}) {
  if (!targetGraph) return null;
  const currentNodeSizes = estimateCurrentNodeSizes(currentGraph);
  return (
    <svg
      className="targetGhostLayer"
      width={graphWorldLayerWidth}
      height={graphWorldLayerHeight}
      viewBox={`${graphWorldMinX} ${graphWorldMinY} ${graphWorldLayerWidth} ${graphWorldLayerHeight}`}
      style={{ left: graphWorldMinX, top: graphWorldMinY, width: graphWorldLayerWidth, height: graphWorldLayerHeight }}
      aria-hidden="true"
    >
      {targetGraph.nodes.map((node) => {
        const matched = currentGraph.nodes.some((item) => item.id === node.id && item.moduleId === node.moduleId);
        if (matched) return null;
        const size = currentNodeSizes[node.moduleId] ?? { width: graphNodeWidth, height: graphNodeMinHeight };
        return (
          <g key={node.id} className="targetGhostNode" transform={`translate(${node.position.x} ${node.position.y})`}>
            <rect width={size.width} height={size.height} rx={8} />
            <text x={14} y={24}>{node.id}</text>
          </g>
        );
      })}
    </svg>
  );
}

type CaseChip = {
  label: string;
  value: string;
  tone?: "ok" | "warn" | "bad" | "muted";
  help?: string;
  error?: string;
  editable?: {
    paramKey: string;
    kind: "finite_float32";
  };
};

function GraphNodeCaseChips({
  level,
  graph,
  modules,
  node,
  module,
  onParamChange
}: {
  level: LevelSpec;
  graph: GraphSpec;
  modules: ModuleDef[];
  node: GraphNode;
  module: ModuleDef;
  onParamChange: (nodeId: string, key: string, value: unknown) => void;
}) {
  const { t } = useGraphT();
  const chips = getNodeCaseChips(level, graph, modules, node, module);
  if (!chips.length) return null;
  return (
    <div className="graphNodeCase">
      <span>{t("Data Flow")}</span>
      <div>
        {chips.map((chip) => {
          const editable = chip.editable;
          const tooltip = chip.error ? t(chip.error) : chip.help ? t(chip.help) : undefined;
          return (
          <div
            key={`${chip.label}:${editable?.paramKey ?? chip.value}`}
            className="graphNodeCaseSlot"
            title={tooltip}
            aria-label={tooltip}
            data-tooltip={tooltip}
          >
            {editable ? (
              <label className={`graphNodeCaseEditable ${chip.tone ?? "muted"}`}>
                <b>{t(chip.label)}</b>
                <input
                  value={chip.value}
                  aria-invalid={Boolean(chip.error)}
                  title={tooltip}
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={(event) => event.stopPropagation()}
                  onKeyDown={(event) => event.stopPropagation()}
                  onKeyUp={(event) => event.stopPropagation()}
                  onChange={(event) => {
                    const parsed = parseEditableCaseValue(editable, event.currentTarget.value);
                    onParamChange(node.id, editable.paramKey, parsed.ok ? parsed.value : event.currentTarget.value);
                  }}
                />
              </label>
            ) : (
              <code className={chip.tone ?? "muted"}>
                <b>{t(chip.label)}</b>
                {t(chip.value)}
              </code>
            )}
            {chip.error ? <small className="graphNodeCaseError">{t(chip.error)}</small> : null}
          </div>
          );
        })}
      </div>
    </div>
  );
}

function GraphPortColumn({
  node,
  module,
  direction,
  wireSource,
  wireSourcePort,
  onPortClick,
  onPortPointerDown,
  onPortPointerUp
}: {
  node: GraphNode;
  module: ModuleDef;
  direction: "in" | "out";
  wireSource?: WireSource;
  wireSourcePort?: PortDef;
  onPortClick: (node: GraphNode, module: ModuleDef, port: PortDef) => void;
  onPortPointerDown: (event: ReactPointerEvent<HTMLButtonElement>, node: GraphNode, port: PortDef) => void;
  onPortPointerUp: (event: ReactPointerEvent<HTMLButtonElement>, node: GraphNode, port: PortDef) => void;
}) {
  const { t } = useGraphT();
  const ports = direction === "in" ? module.inputs : module.outputs;
  return (
    <div className={`graphPortColumn ${direction}`}>
      {ports.map((port) => {
        const armed = direction === "out" && wireSource?.nodeId === node.id && wireSource.portId === port.id;
        const connectable = direction === "in" && Boolean(wireSourcePort) && wireSource?.nodeId !== node.id && portsCompatible(wireSourcePort as PortDef, port);
        const incompatible = direction === "in" && Boolean(wireSourcePort) && !connectable;
        return (
          <button
            key={port.id}
            className={`graphPort ${direction} ${armed ? "armed" : ""} ${connectable ? "connectable" : ""} ${incompatible ? "incompatible" : ""}`}
            data-graph-port="true"
            title={`${node.id}.${port.id}`}
            onPointerDown={(event) => onPortPointerDown(event, node, port)}
            onPointerUp={(event) => onPortPointerUp(event, node, port)}
            onClick={(event) => {
              event.stopPropagation();
              onPortClick(node, module, port);
            }}
          >
            <span data-graph-port-dot="true" data-node-id={node.id} data-port-id={port.id} data-port-direction={direction} />
            <b>{t(port.label)}</b>
          </button>
        );
      })}
    </div>
  );
}

function GraphComponentLibrary({
  level,
  registry
}: {
  level: LevelSpec;
  registry: ReturnType<typeof createGameplayRegistry>;
}) {
  const { t } = useGraphT();
  return (
    <section className="graphComponentLibrary">
      <div className="graphComponentLibraryHeader">
        <b>{t("Available Modules")}</b>
        <code>{level.modulePalette.length}</code>
      </div>
      <small className="graphComponentLibraryHint">{t("Drag a module onto the canvas to add it.")}</small>
      <div className="graphComponentLibraryList">
        {level.modulePalette.map((moduleId) => {
          const module = registry.get(moduleId);
          return (
            <button
              key={module.id}
              className="graphModuleItem library"
              type="button"
              draggable
              title={t(module.summary)}
              onDragStart={(event) => {
                event.dataTransfer.setData(moduleDragMime, module.id);
                event.dataTransfer.effectAllowed = "copy";
              }}
            >
              <b>{t(module.label)}</b>
              <small>{t(module.category)}</small>
            </button>
          );
        })}
      </div>
    </section>
  );
}

type ChallengeMapNode = {
  level: LevelSpec;
  x: number;
  y: number;
  cx: number;
  cy: number;
  status: "active" | "available" | "locked" | "draft" | "design_ready";
  lifecycle?: string;
};

type ChallengeMapRegion = {
  id: string;
  title: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

type ChallengeMapPoint = {
  x: number;
  y: number;
};

type ChallengeMapView = {
  x: number;
  y: number;
  scale: number;
};

type ChallengeMapPanState = {
  pointerId: number;
  startX: number;
  startY: number;
  originX: number;
  originY: number;
};

type ChallengeMapChapterGroup = {
  id: string;
  title: string;
  levels: LevelSpec[];
};

const challengeMapNodeWidth = 164;
const challengeMapNodeHeight = 70;
const challengeMapMinScale = 0.42;
const challengeMapMaxScale = 1.25;
const challengeMapDefaultScale = 0.72;
const challengeMapJitterPattern = [
  { x: 0, y: 0 },
  { x: 18, y: -10 },
  { x: -14, y: 13 },
  { x: 24, y: 8 },
  { x: -22, y: -8 },
  { x: 10, y: 16 },
  { x: -8, y: -12 },
  { x: 22, y: 10 },
  { x: -18, y: 4 },
  { x: 12, y: -14 },
  { x: -26, y: 12 },
  { x: 16, y: 2 }
];

function GraphChallengeMapOverlay({
  levels,
  selectedLevelId,
  componentFlow,
  availableComponentIds,
  onSelect,
  onClose
}: {
  levels: LevelSpec[];
  selectedLevelId: string;
  componentFlow?: ComponentFlowConfig;
  availableComponentIds: Set<string>;
  onSelect: (level: LevelSpec) => void;
  onClose: () => void;
}) {
  const { t } = useGraphT();
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const panStateRef = useRef<ChallengeMapPanState | undefined>(undefined);
  const [mapView, setMapView] = useState<ChallengeMapView>({ x: 18, y: 18, scale: challengeMapDefaultScale });
  const [panning, setPanning] = useState(false);
  const layout = useMemo(
    () => buildChallengeMapLayout(levels, selectedLevelId, componentFlow, availableComponentIds),
    [availableComponentIds, componentFlow, levels, selectedLevelId]
  );

  useLayoutEffect(() => {
    centerChallengeMapOnActiveNode();
    // Recenter when a different challenge opens; user pan/zoom remains local to the open map session otherwise.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout.width, layout.height, selectedLevelId]);

  function centerChallengeMapOnActiveNode(scale = challengeMapDefaultScale) {
    const rect = canvasRef.current?.getBoundingClientRect();
    const activeNode = layout.nodes.find((node) => node.level.id === selectedLevelId) ?? layout.nodes[0];
    if (!rect || !activeNode) {
      setMapView({ x: 18, y: 18, scale });
      return;
    }
    setMapView({
      scale,
      x: rect.width * 0.46 - activeNode.cx * scale,
      y: rect.height * 0.38 - activeNode.cy * scale
    });
  }

  function zoomChallengeMap(factor: number, anchor?: ChallengeMapPoint) {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    setMapView((current) => {
      const nextScale = clamp(current.scale * factor, challengeMapMinScale, challengeMapMaxScale);
      const anchorPoint = anchor ?? { x: rect.width / 2, y: rect.height / 2 };
      const worldX = (anchorPoint.x - current.x) / current.scale;
      const worldY = (anchorPoint.y - current.y) / current.scale;
      return {
        scale: nextScale,
        x: anchorPoint.x - worldX * nextScale,
        y: anchorPoint.y - worldY * nextScale
      };
    });
  }

  function handleMapWheel(event: ReactWheelEvent<HTMLDivElement>) {
    event.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    zoomChallengeMap(event.deltaY < 0 ? 1.1 : 0.9, {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    });
  }

  function handleMapPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    const target = event.target as HTMLElement;
    if (target.closest(".graphChallengeMapNode, .graphChallengeMapControls")) return;
    event.preventDefault();
    panStateRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: mapView.x,
      originY: mapView.y
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    setPanning(true);
  }

  function handleMapPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const state = panStateRef.current;
    if (!state || state.pointerId !== event.pointerId) return;
    setMapView((current) => ({
      ...current,
      x: state.originX + event.clientX - state.startX,
      y: state.originY + event.clientY - state.startY
    }));
  }

  function endMapPan(event: ReactPointerEvent<HTMLDivElement>) {
    const state = panStateRef.current;
    if (state?.pointerId === event.pointerId) {
      panStateRef.current = undefined;
      event.currentTarget.releasePointerCapture(event.pointerId);
      setPanning(false);
    }
  }

  return (
    <div className="graphChallengeMapOverlay" role="dialog" aria-modal="true" aria-label={t("Challenge Map")} onClick={onClose}>
      <section className="graphChallengeMapPanel" onClick={(event) => event.stopPropagation()}>
        <header className="graphChallengeMapHeader">
          <span>
            <p className="eyebrow">{t("Challenge Map")}</p>
            <h3>{t("Chapter route")}</h3>
          </span>
          <button className="iconButton" type="button" title={t("Collapse map")} onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <div
          ref={canvasRef}
          className={`graphChallengeMapCanvas${panning ? " panning" : ""}`}
          onWheel={handleMapWheel}
          onPointerDown={handleMapPointerDown}
          onPointerMove={handleMapPointerMove}
          onPointerUp={endMapPan}
          onPointerCancel={endMapPan}
        >
          <div className="graphChallengeMapControls" onPointerDown={(event) => event.stopPropagation()}>
            <button className="iconButton" type="button" title={t("Zoom out")} onClick={() => zoomChallengeMap(0.9)}>
              <Minus size={15} />
            </button>
            <code>{Math.round(mapView.scale * 100)}%</code>
            <button className="iconButton" type="button" title={t("Zoom in")} onClick={() => zoomChallengeMap(1.1)}>
              <Plus size={15} />
            </button>
            <button className="iconButton" type="button" title={t("Center active challenge")} onClick={() => centerChallengeMapOnActiveNode(mapView.scale)}>
              <Move size={15} />
            </button>
            <button className="iconButton" type="button" title={t("Reset map view")} onClick={() => centerChallengeMapOnActiveNode()}>
              <RotateCcw size={15} />
            </button>
          </div>
          <div
            className="graphChallengeMapWorld"
            style={{
              width: layout.width,
              height: layout.height,
              transform: `translate3d(${mapView.x}px, ${mapView.y}px, 0) scale(${mapView.scale})`
            }}
          >
            {layout.regions.map((region) => (
              <section
                key={region.id}
                className="graphChallengeMapRegion"
                style={{ left: region.x, top: region.y, width: region.width, height: region.height }}
                aria-hidden="true"
              >
                <span>{t(region.title)}</span>
              </section>
            ))}
            <svg className="graphChallengeRouteLayer" width={layout.width} height={layout.height} viewBox={`0 0 ${layout.width} ${layout.height}`}>
              <path className="graphChallengeRoutePath shadow" d={layout.path} />
              <path className="graphChallengeRoutePath" d={layout.path} />
            </svg>
            {layout.nodes.map((node) => (
              <button
                key={node.level.id}
                type="button"
                className={`graphChallengeMapNode ${node.status}`}
                style={{ left: node.x, top: node.y }}
                disabled={!isLevelPlayable(node.level)}
                onClick={() => {
                  if (isLevelPlayable(node.level)) onSelect(node.level);
                }}
              >
                <span className="graphChallengeMapPort in" aria-hidden="true" />
                <span className="graphChallengeMapPort out" aria-hidden="true" />
                <b>{t(node.level.title)}</b>
                <small>{t(node.level.chapter)}</small>
                {node.lifecycle ? <code>{t(node.lifecycle)}</code> : null}
              </button>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function buildChallengeMapLayout(
  levels: LevelSpec[],
  selectedLevelId: string,
  componentFlow: ComponentFlowConfig | undefined,
  availableComponentIds: Set<string>
) {
  if (!levels.length) {
    return { nodes: [], regions: [], path: "", width: 640, height: 360 };
  }

  const nodeWidth = challengeMapNodeWidth;
  const nodeHeight = challengeMapNodeHeight;
  const mapColumns = 3;
  const regionCellWidth = 980;
  const regionCellHeight = 760;
  const margin = 64;
  const regionPadX = 46;
  const regionPadTop = 84;
  const regionPadBottom = 44;
  const localGapX = 116;
  const localGapY = 90;
  const regionDriftPattern = [
    { x: 0, y: 0 },
    { x: 34, y: 78 },
    { x: -18, y: 26 },
    { x: 42, y: -12 },
    { x: -32, y: 64 },
    { x: 12, y: 118 }
  ];
  const groups = groupChallengeMapLevels(levels);
  const regions: ChallengeMapRegion[] = [];
  const nodes: ChallengeMapNode[] = [];
  let routeIndex = 0;

  groups.forEach((group, groupIndex) => {
    const count = group.levels.length;
    const localColumns = count <= 3 ? count : count <= 6 ? 2 : 3;
    const localRows = Math.ceil(count / localColumns);
    const regionWidth = regionPadX * 2 + localColumns * nodeWidth + (localColumns - 1) * localGapX;
    const regionHeight = regionPadTop + localRows * nodeHeight + (localRows - 1) * localGapY + regionPadBottom;
    const mapRow = Math.floor(groupIndex / mapColumns);
    const colInRow = groupIndex % mapColumns;
    const mapCol = mapRow % 2 === 0 ? colInRow : mapColumns - 1 - colInRow;
    const drift = regionDriftPattern[groupIndex % regionDriftPattern.length];
    const region: ChallengeMapRegion = {
      id: group.id,
      title: group.title,
      x: margin + mapCol * regionCellWidth + drift.x,
      y: margin + mapRow * regionCellHeight + drift.y,
      width: regionWidth,
      height: regionHeight
    };
    regions.push(region);

    group.levels.forEach((level, localIndex) => {
      const localRow = Math.floor(localIndex / localColumns);
      const colInLocalRow = localIndex % localColumns;
      const localCol = localRow % 2 === 0 ? colInLocalRow : localColumns - 1 - colInLocalRow;
      const jitter = challengeMapJitterPattern[(routeIndex + groupIndex) % challengeMapJitterPattern.length];
      const x = region.x + regionPadX + localCol * (nodeWidth + localGapX) + jitter.x;
      const y = region.y + regionPadTop + localRow * (nodeHeight + localGapY) + jitter.y;
      const component = componentFlow?.specs[level.id];
      const missingCount = component?.requires.filter((componentId) => !availableComponentIds.has(componentId)).length ?? 0;
      const available = component ? availableComponentIds.has(component.componentId) : false;
      const designReady = !isLevelPlayable(level);
      const lifecycle = designReady ? "design_ready" : component ? lifecycleLabel({ available, missingCount }) : undefined;
      const status = level.id === selectedLevelId ? "active" : designReady ? "design_ready" : missingCount > 0 ? "locked" : available ? "available" : "draft";
      nodes.push({ level, x, y, cx: x + nodeWidth / 2, cy: y + nodeHeight / 2, status, lifecycle });
      routeIndex += 1;
    });
  });
  const maxRegionX = Math.max(...regions.map((region) => region.x + region.width));
  const maxRegionY = Math.max(...regions.map((region) => region.y + region.height));
  const maxNodeX = Math.max(...nodes.map((node) => node.x + nodeWidth));
  const maxNodeY = Math.max(...nodes.map((node) => node.y + nodeHeight));
  return {
    nodes,
    regions,
    path: challengeMapRoutePath(nodes),
    width: Math.ceil(Math.max(maxRegionX, maxNodeX) + margin),
    height: Math.ceil(Math.max(maxRegionY, maxNodeY) + margin)
  };
}

function groupChallengeMapLevels(levels: LevelSpec[]) {
  const groups: ChallengeMapChapterGroup[] = [];
  const byTitle = new Map<string, ChallengeMapChapterGroup>();
  levels.forEach((level) => {
    const title = level.chapter;
    const id = title.toLowerCase().replace(/[^a-z0-9]+/g, "-") || `chapter-${groups.length}`;
    let group = byTitle.get(title);
    if (!group) {
      group = { id, title, levels: [] };
      byTitle.set(title, group);
      groups.push(group);
    }
    group.levels.push(level);
  });
  return groups;
}

function challengeMapRoutePath(nodes: ChallengeMapNode[]) {
  if (nodes.length < 2) return "";
  const parts: string[] = [];
  for (let index = 1; index < nodes.length; index += 1) {
    const from = nodes[index - 1];
    const to = nodes[index];
    parts.push(challengeMapRouteSegment(from, to, index - 1));
  }
  return parts.join(" ");
}

function challengeMapRouteSegment(from: ChallengeMapNode, to: ChallengeMapNode, index: number) {
  const start = challengeMapOutPoint(from);
  const end = challengeMapInPoint(to);
  return challengeMapInOutCurve(start, end, index);
}

function challengeMapInPoint(node: ChallengeMapNode): ChallengeMapPoint {
  return { x: node.x, y: node.cy };
}

function challengeMapOutPoint(node: ChallengeMapNode): ChallengeMapPoint {
  return { x: node.x + challengeMapNodeWidth, y: node.cy };
}

function challengeMapInOutCurve(start: ChallengeMapPoint, end: ChallengeMapPoint, index: number) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const waveSize = Math.min(112, 28 + Math.abs(dy) * 0.08 + Math.abs(dx) * 0.035);
  const wave = (index % 2 === 0 ? 1 : -1) * waveSize;
  if (dx >= 60) {
    const lead = Math.min(180, Math.max(58, dx * 0.34));
    return [
      `M ${start.x} ${start.y}`,
      `C ${start.x + lead} ${start.y + wave}, ${end.x - lead} ${end.y - wave}, ${end.x} ${end.y}`
    ].join(" ");
  }
  const loop = Math.min(220, 104 + Math.abs(dx) * 0.12 + Math.abs(dy) * 0.06);
  return [
    `M ${start.x} ${start.y}`,
    `C ${start.x + loop} ${start.y + wave}, ${start.x + loop} ${start.y + dy * 0.34}, ${(start.x + end.x) / 2} ${start.y + dy * 0.5}`,
    `C ${end.x - loop} ${end.y - dy * 0.34}, ${end.x - loop} ${end.y - wave}, ${end.x} ${end.y}`
  ].join(" ");
}

function GraphTaskDataPanels({
  level,
  graph,
  modules
}: {
  level: LevelSpec;
  graph: GraphSpec;
  modules: ModuleDef[];
}) {
  const { t } = useGraphT();
  const caseStudy = level.caseStudy;
  const visibleCase = level.visibleTests[0];
  if (!caseStudy) return null;
  const panels = level.id.startsWith("mvp01_")
    ? caseStudy.dataPanels.filter((panel) => panel.type !== "text_batch" || panel.inputKey !== "case")
    : caseStudy.dataPanels;
  if (!panels.length) return null;

  return (
    <div className="graphCaseDataGrid">
      {panels.map((panel) => {
        if (panel.type === "text_batch") {
          const texts = getCaseTexts(visibleCase, panel.inputKey);
          const focusText = panel.focusText ?? caseStudy.visibleInputFocus;
          return (
            <section key={`${panel.type}:${panel.inputKey}`} className="graphCaseDataCard">
              <b>{t(panel.title)}</b>
              <div className="graphCaseTextList">
                {texts.map((text) => (
                  <code key={text} className={text === focusText ? "focus" : ""}>
                    {t(text)}
                  </code>
                ))}
              </div>
            </section>
          );
        }

        if (panel.type === "tokenizer_preview") {
          const preview = createTokenizerPreview(level, graph, modules, visibleCase, panel.tokenizerNodeId, panel.textInputKey);
          return (
            <section key={`${panel.type}:${panel.tokenizerNodeId}`} className="graphCaseDataCard">
              <TokenizerPreviewCard title={panel.title} preview={preview} />
            </section>
          );
        }

        if (panel.type === "tensor_preview") {
          return (
            <section key={`${panel.type}:${panel.inputKey}`} className="graphCaseDataCard">
              <TensorPreviewCard
                title={panel.title}
                value={visibleCase.inputs[panel.inputKey]}
                maxRows={panel.maxRows}
                maxCols={panel.maxCols}
              />
            </section>
          );
        }

        return (
          <section key={`${panel.type}:${panel.title}`} className="graphCaseDataCard">
            <b>{t(panel.title)}</b>
            <p className="graphTraceEmpty">{t("Task preview is coming in the next slice.")}</p>
          </section>
        );
      })}
    </div>
  );
}

function TensorPreviewCard({
  title,
  value,
  maxRows = 4,
  maxCols = 6
}: {
  title: string;
  value?: RuntimeValue;
  maxRows?: number;
  maxCols?: number;
}) {
  const { t } = useGraphT();
  if (!value?.shape || !Array.isArray(value.data)) {
    return (
      <>
        <b>{t(title)}</b>
        <p className="graphTraceEmpty">{t("No shaped values.")}</p>
      </>
    );
  }

  const rows = tensorPreviewRows(value, maxRows, maxCols);
  return (
    <>
      <div className="graphCasePreviewHeader">
        <b>{t(title)}</b>
        <code>{`${value.dtype}[${value.shape.axes.join(",") || "scalar"}]=[${value.shape.dims.join(",")}]`}</code>
      </div>
      <div className="graphTensorPreviewGrid">
        {rows.map((row) => (
          <div key={row.label} className="graphTensorPreviewRow">
            <small>{row.label}</small>
            <span>
              {row.values.map((item, index) => (
                <code key={`${row.label}:${index}`}>{item}</code>
              ))}
            </span>
          </div>
        ))}
      </div>
    </>
  );
}

function TokenizerPreviewCard({ title, preview }: { title: string; preview?: TokenizerCasePreview }) {
  const { t } = useGraphT();
  if (!preview) {
    return (
      <>
        <b>{t(title)}</b>
        <p className="graphTraceEmpty">{t("Tokenizer node is missing from this graph.")}</p>
      </>
    );
  }

  const budgetClass = preview.withinBudget && !preview.truncated ? "ok" : "over";
  return (
    <>
      <div className="graphCasePreviewHeader">
        <b>{t(title)}</b>
        <code className={budgetClass}>
          {preview.rawTokenCount} / {preview.maxBudget}
        </code>
      </div>
      <GraphKeyValue label={t("Raw text")} value={preview.focusText} />
      <div className="graphCasePreviewMeta">
        <code>{t("Split strategy")}: {t(formatPolicy(preview.policy))}</code>
        <code>{t("Apply merges")}: {preview.applyMerges ? t("On") : t("Off")}</code>
        <code>{t("Unknown fallback")}: {preview.fallback}</code>
      </div>
      <TokenRow label={t("Pieces")} values={preview.pieces} />
      <TokenRow label={t("Token IDs")} values={preview.ids.map(String)} compact />
      <TokenRow label={t("Attention Mask")} values={preview.mask.map(String)} compact />
      <p className={`graphCaseBudget ${budgetClass}`}>
        {preview.withinBudget && !preview.truncated ? t("Within budget.") : t("Over budget. Try Subword plus Apply merges.")}
      </p>
      {preview.unresolved.length ? <p className="graphCaseBudget over">{t("Unresolved")}: {preview.unresolved.join(", ")}</p> : null}
    </>
  );
}

function TokenRow({ label, values, compact = false }: { label: string; values: string[]; compact?: boolean }) {
  const displayed = values.slice(0, compact ? 12 : 18);
  return (
    <div className="graphTokenRow">
      <span>{label}</span>
      <div>
        {displayed.map((value, index) => (
          <code key={`${value}:${index}`}>{value}</code>
        ))}
        {values.length > displayed.length ? <code>+{values.length - displayed.length}</code> : null}
      </div>
    </div>
  );
}

function GraphCodePanel({ level, graph, modules }: { level: LevelSpec; graph: GraphSpec; modules: ModuleDef[] }) {
  const { t } = useGraphT();
  const [collapsed, setCollapsed] = useState(false);
  const sections = useMemo(() => generateGraphCodeSections(level, graph, modules, level.visibleTests[0]), [graph, level, modules]);

  return (
    <section className={`graphCodePanel ${collapsed ? "collapsed" : ""}`}>
      <div className="graphRunPanelHeader">
        <h3>{t("Code")}</h3>
        <button className="iconButton" type="button" title={t(collapsed ? "Open code" : "Collapse code")} onClick={() => setCollapsed((current) => !current)}>
          {collapsed ? <BookOpenText size={14} /> : <X size={14} />}
        </button>
      </div>
      {!collapsed ? (
        <div className="graphCodeSections">
          <CodeSection title="Task Input" lines={sections.caseCode} />
          <CodeSection title="Graph Code" lines={sections.graphCode} />
          <CodeSection title="Test Code" lines={sections.testCode} />
        </div>
      ) : null}
    </section>
  );
}

function CodeSection({ title, lines }: { title: string; lines: Array<{ id: string; text: string }> }) {
  const { t } = useGraphT();
  return (
    <section className="graphCodeSection">
      <b>{t(title)}</b>
      <pre className="graphCodeBlock">{lines.map((line) => line.text).join("\n")}</pre>
    </section>
  );
}

function GraphTraceTimeline({
  visible,
  hidden,
  selection,
  onSelect
}: {
  visible?: RunTestsResult;
  hidden?: RunTestsResult;
  selection?: TraceSelection;
  onSelect: (selection: TraceSelection, frame: TraceFrame) => void;
}) {
  const { t, status } = useGraphT();
  const runs: Array<{ key: TraceRunKey; label: string; result?: RunTestsResult }> = [
    { key: "visible", label: "Task Check", result: visible },
    { key: "hidden", label: "Validation", result: hidden }
  ];
  const hasTrace = runs.some((run) => run.result?.cases.length);

  return (
    <section className="graphTracePanel">
      <div className="graphTraceHeader">
        <div>
          <p className="eyebrow">{t("Trace Timeline")}</p>
          <h3>{t("Run path / first failure / node state")}</h3>
        </div>
        <code>{hasTrace ? t("click a step") : t("idle")}</code>
      </div>
      {hasTrace ? (
        <div className="graphTraceRuns">
          {runs.map((run) =>
            run.result?.cases.map((testCase) => (
              <section key={`${run.key}:${testCase.id}`} className={`graphTraceCase ${testCase.status}`}>
                <div className="graphTraceCaseHeader">
                  <span>{t(run.label)}</span>
                  <b>{t(testCase.title)}</b>
                  <code>{status(testCase.status)}</code>
                </div>
                <div className="graphTraceSteps">
                  {testCase.execution.trace.map((frame) => {
                    const selected = selection?.runKey === run.key && selection.caseId === testCase.id && selection.step === frame.step;
                    return (
                      <button
                        key={`${testCase.id}:${frame.step}`}
                        className={`graphTraceStep ${frame.error ? "fail" : "pass"} ${selected ? "selected" : ""}`}
                        onClick={() => onSelect({ runKey: run.key, caseId: testCase.id, step: frame.step }, frame)}
                      >
                        <span>{frame.step}</span>
                        <b>{frame.nodeId}</b>
                        <small>{frame.error?.type ?? frame.moduleId}</small>
                      </button>
                    );
                  })}
                </div>
              </section>
            ))
          )}
        </div>
      ) : (
        <p className="graphTraceEmpty">{t("Check current task to capture a trace.")}</p>
      )}
    </section>
  );
}

function GraphInspectorNodePanel({
  node,
  module,
  graph,
  runState,
  activeTraceCase,
  activeTraceFrame,
  tab,
  onTabChange,
  onParamChange,
  onLocateNode,
  onShowNextStep
}: {
  node: GraphNode;
  module: ModuleDef;
  graph: GraphSpec;
  runState: RunState;
  activeTraceCase?: TestCaseRunResult;
  activeTraceFrame?: TraceFrame;
  tab: InspectorTab;
  onTabChange: (tab: InspectorTab) => void;
  onParamChange: (nodeId: string, key: string, value: unknown) => void;
  onLocateNode: (nodeId: string) => void;
  onShowNextStep: () => void;
}) {
  const { t } = useGraphT();
  const nodeFrame = activeTraceFrame?.nodeId === node.id ? activeTraceFrame : findLatestNodeFrame(activeTraceCase, node.id);
  const relatedResults = getNodeTestResults(runState, node.id);
  const failingResult = relatedResults.find((result) => result.status !== "pass");
  const incoming = graph.edges.filter((edge) => edge.to.nodeId === node.id);
  const outgoing = graph.edges.filter((edge) => edge.from.nodeId === node.id);
  const tabs: Array<{ id: InspectorTab; label: string }> = [
    { id: "summary", label: t("Summary") },
    { id: "shape", label: t("Shape") },
    { id: "values", label: t("Values") },
    { id: "tests", label: t("Tests") },
    { id: "code", label: t("Code") }
  ];

  return (
    <section className="graphInspectorBlock">
      <p className="eyebrow">{t("Selected Node")}</p>
      <h3>{node.id}</h3>
      <code>{node.moduleId}</code>
      <div className="graphInspectorTabs">
        {tabs.map((item) => (
          <button key={item.id} className={tab === item.id ? "active" : ""} onClick={() => onTabChange(item.id)}>
            {item.label}
          </button>
        ))}
      </div>

      {tab === "summary" ? (
        <div className="graphInspectorTabBody">
          <GraphKeyValue label={t("category")} value={t(module.category)} />
          <GraphKeyValue label={t("purpose")} value={t(module.summary)} />
          <GraphKeyValue label={t("incoming")} value={incoming.map((edge) => `${edge.from.nodeId}.${edge.from.portId} -> ${edge.to.portId}`).join("\n") || t("none")} />
          <GraphKeyValue label={t("outgoing")} value={outgoing.map((edge) => `${edge.from.portId} -> ${edge.to.nodeId}.${edge.to.portId}`).join("\n") || t("none")} />
          {failingResult ? <GraphFailureReport result={failingResult} /> : null}
          <GraphParamEditor node={node} module={module} onChange={onParamChange} />
        </div>
      ) : null}

      {tab === "shape" ? (
        <div className="graphInspectorTabBody">
          {nodeFrame ? (
            <>
              <ShapeRecord title={t("Inputs")} shapes={nodeFrame.inputShapes} />
              <ShapeRecord title={t("Outputs")} shapes={nodeFrame.outputShapes} />
              {nodeFrame.error ? <GraphRuntimeErrorBox frame={nodeFrame} /> : null}
            </>
          ) : (
            <>
              <PortContractList title={t("Input Contracts")} ports={module.inputs} />
              <PortContractList title={t("Output Contracts")} ports={module.outputs} />
            </>
          )}
        </div>
      ) : null}

      {tab === "values" ? (
        <div className="graphInspectorTabBody">
          {nodeFrame?.samples ? (
            <pre className="graphValueBlock">{formatUnknown(nodeFrame.samples)}</pre>
          ) : (
            <p className="graphTraceEmpty">{t("No sample values captured for this node yet.")}</p>
          )}
          {failingResult?.diagnostic?.sample ? (
            <>
              <h4>{t("Failure sample")}</h4>
              <pre className="graphValueBlock">{formatUnknown(failingResult.diagnostic.sample)}</pre>
            </>
          ) : null}
        </div>
      ) : null}

      {tab === "tests" ? (
        <div className="graphInspectorTabBody">
          {relatedResults.length ? (
            <div className="graphResultList">
              {relatedResults.map((result) => (
                <GraphResultItem key={result.id} item={result} detailed onLocateNode={onLocateNode} onShowNextStep={onShowNextStep} />
              ))}
            </div>
          ) : (
            <p className="graphTraceEmpty">{t("No captured test result targets this node yet.")}</p>
          )}
        </div>
      ) : null}

      {tab === "code" ? (
        <div className="graphInspectorTabBody">
          <pre className="graphCodeBlock">{module.pseudoCode ?? t("No pseudo code registered for this module.")}</pre>
        </div>
      ) : null}
    </section>
  );
}

function GraphKeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="graphKeyValue">
      <span>{label}</span>
      <code>{value}</code>
    </div>
  );
}

function ShapeRecord({ title, shapes }: { title: string; shapes: TraceFrame["inputShapes"] }) {
  const { t } = useGraphT();
  const entries = Object.entries(shapes);
  return (
    <section className="graphShapeRecord">
      <h4>{title}</h4>
      {entries.length ? (
        entries.map(([portId, shape]) => (
          <div key={portId}>
            <span>{portId}</span>
            <code>{formatShape(shape)}</code>
          </div>
        ))
      ) : (
        <p className="graphTraceEmpty">{t("No shaped values.")}</p>
      )}
    </section>
  );
}

function PortContractList({ title, ports }: { title: string; ports: PortDef[] }) {
  const { t } = useGraphT();
  return (
    <section className="graphShapeRecord">
      <h4>{title}</h4>
      {ports.length ? (
        ports.map((port) => (
          <div key={port.id}>
            <span>{port.id}</span>
            <code>{port.direction === "in" ? port.accepts?.join(" | ") || t("any") : port.emits ?? t("value")}</code>
          </div>
        ))
      ) : (
        <p className="graphTraceEmpty">{t("No ports.")}</p>
      )}
    </section>
  );
}

function GraphRuntimeErrorBox({ frame }: { frame: TraceFrame }) {
  const { t } = useGraphT();
  if (!frame.error) return null;
  return (
    <section className="graphFailureReport">
      <b>{frame.error.type}</b>
      <p>{frame.error.message}</p>
      <GraphKeyValue label={t("expected")} value={formatUnknown(frame.error.expected)} />
      <GraphKeyValue label={t("received")} value={formatUnknown(frame.error.received)} />
    </section>
  );
}

function GraphFailureReport({ result }: { result: TestResult }) {
  const { t } = useGraphT();
  const diagnostic = result.diagnostic;
  if (!diagnostic) return null;
  return (
    <section className="graphFailureReport">
      <b>{diagnostic.errorType ?? result.status}</b>
      <p>{result.message}</p>
      <GraphKeyValue label={t("expected")} value={formatUnknown(diagnostic.expected)} />
      <GraphKeyValue label={t("received")} value={formatUnknown(diagnostic.received)} />
      <GraphKeyValue label={t("cause")} value={diagnostic.possibleCause ?? "unknown"} />
      <GraphKeyValue label={t("probe hint")} value={diagnostic.suggestedProbe ?? t("step through trace")} />
    </section>
  );
}

function GraphParamEditor({
  node,
  module,
  onChange
}: {
  node: GraphNode;
  module: ModuleDef;
  onChange: (nodeId: string, key: string, value: unknown) => void;
}) {
  const { t } = useGraphT();
  const keys = [...new Set([...Object.keys(module.defaultParams), ...Object.keys(node.params)])].filter((key) => !hiddenParamKeys.has(key));
  if (!keys.length) {
    return (
      <section className="graphParamEditor">
        <div className="graphParamHeader">
          <b>{t("Parameters")}</b>
          <code>{t("none")}</code>
        </div>
      </section>
    );
  }

  return (
    <section className="graphParamEditor">
      <div className="graphParamHeader">
        <b>{t("Parameters")}</b>
        <code>{keys.length}</code>
      </div>
      <div className="graphParamRows">
        {keys.map((key) => {
          const defaultValue = module.defaultParams[key];
          const value = node.params[key] ?? defaultValue;
          return <GraphParamRow key={key} nodeId={node.id} paramKey={key} value={value} defaultValue={defaultValue} onChange={onChange} />;
        })}
      </div>
    </section>
  );
}

function GraphParamRow({
  nodeId,
  paramKey,
  value,
  defaultValue,
  onChange
}: {
  nodeId: string;
  paramKey: string;
  value: unknown;
  defaultValue: unknown;
  onChange: (nodeId: string, key: string, value: unknown) => void;
}) {
  const { t } = useGraphT();
  const options = paramOptions[paramKey];
  const meta = paramCopy[paramKey] ?? { label: paramKey };
  const valueType = typeof value;

  if (Array.isArray(value) || Array.isArray(defaultValue)) {
    const arrayValue = (Array.isArray(value) ? value : defaultValue) as unknown[];
    return (
      <label className="graphParamRow">
        <span>{t(meta.label)}</span>
        <input
          type="text"
          value={arrayValue.join(",")}
          onChange={(event) =>
            onChange(
              nodeId,
              paramKey,
              event.currentTarget.value
                .split(",")
                .map((item) => item.trim())
                .filter(Boolean)
            )
          }
        />
        {meta.help ? <small>{t(meta.help)}</small> : null}
      </label>
    );
  }

  if (typeof value === "boolean" || typeof defaultValue === "boolean") {
    return (
      <label className="graphParamRow boolean">
        <span>{t(meta.label)}</span>
        <input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(nodeId, paramKey, event.currentTarget.checked)} />
        {meta.help ? <small>{t(meta.help)}</small> : null}
      </label>
    );
  }

  if (typeof value === "number" || typeof defaultValue === "number") {
    const isFloat32ValueParam = paramKey === "value";
    const error = isFloat32ValueParam ? finiteFloat32InputError(value) : undefined;
    return (
      <label className="graphParamRow">
        <span>{t(meta.label)}</span>
        <input
          type="text"
          inputMode="decimal"
          value={String(value ?? "")}
          aria-invalid={Boolean(error)}
          title={error ? t(error) : undefined}
          onChange={(event) => {
            const text = event.currentTarget.value;
            if (isFloat32ValueParam) {
              const parsed = parseFiniteFloat32Input(text);
              onChange(nodeId, paramKey, parsed.ok ? parsed.value : text);
              return;
            }

            const parsed = Number(text);
            onChange(nodeId, paramKey, text.trim() !== "" && Number.isFinite(parsed) ? parsed : text);
          }}
        />
        {error ? <small className="graphParamError">{t(error)}</small> : meta.help ? <small>{t(meta.help)}</small> : null}
      </label>
    );
  }

  if (typeof value === "string" || typeof defaultValue === "string") {
    if (options?.length) {
      return (
        <label className="graphParamRow">
          <span>{t(meta.label)}</span>
          <select value={String(value ?? "")} onChange={(event) => onChange(nodeId, paramKey, event.currentTarget.value)}>
            {options.map((option) => (
              <option key={option.value} value={option.value}>
                {t(option.label)}
              </option>
            ))}
          </select>
          <small>{t(options.find((option) => option.value === String(value ?? ""))?.consequence ?? meta.help ?? "")}</small>
        </label>
      );
    }

    return (
      <label className="graphParamRow">
        <span>{t(meta.label)}</span>
        <input type="text" value={String(value ?? "")} onChange={(event) => onChange(nodeId, paramKey, event.currentTarget.value)} />
        {meta.help ? <small>{t(meta.help)}</small> : null}
      </label>
    );
  }

  return (
    <div className="graphParamRow readonly">
      <span>{t(meta.label)}</span>
      <code>{JSON.stringify(value)}</code>
      <small>{valueType === "object" ? t("structured value") : valueType}</small>
    </div>
  );
}

const hiddenParamKeys = new Set(["inputKey", "shape", "axes", "referenceKey", "expectedAxes", "expectedPrefixAxes"]);

const paramCopy: Record<string, ParamCopy> = {
  policy: { label: "Split strategy", help: "Choose how raw text becomes token pieces." },
  applyMerges: { label: "Apply merges", help: "Use learned subword merges so common pieces stay intact." },
  fallback: { label: "Unknown fallback", help: "What the tokenizer does when a piece is missing from vocab." },
  preservePunctuation: { label: "Keep punctuation", help: "Keeps punctuation as meaningful token pieces." },
  addBos: { label: "Add <bos>", help: "Marks the start of each sequence." },
  addEos: { label: "Add <eos>", help: "Marks the end of each sequence and should survive truncation." },
  maxLength: { label: "Token budget", help: "Maximum length T before truncation." },
  padToLength: { label: "Pad to length", help: "Pads every row to a stable [B,T] tensor." },
  padSide: { label: "Pad side", help: "Where <pad> tokens are inserted." },
  maskPolicy: { label: "Mask rule", help: "PAD-aware masks hide padding positions." },
  orientation: { label: "Weight storage", help: "How the weight plate is stored before projection." },
  axisA: { label: "Swap axis A", help: "Use -2 with axis B -1 to swap the final two axes." },
  axisB: { label: "Swap axis B", help: "Use -1 with axis A -2 to swap the final two axes." },
  alignAxes: { label: "Broadcast slots", help: "Semantic axes where the smaller tensor plugs into the target." },
  maskOrientation: { label: "Mask direction", help: "query_key blocks future keys for each query row." },
  maskedValue: { label: "Masked value", help: "Large negative value added to blocked future cells." },
  b: { label: "Batch index", help: "Which batch row the cell trace should inspect." },
  h: { label: "Head index", help: "Which attention head the cell trace should inspect." },
  t: { label: "Token index", help: "Which token position the cell trace should inspect." },
  o: { label: "Channel index", help: "Which output channel the cell trace should inspect." },
  i: { label: "Query index", help: "Which query row the cell trace should inspect." },
  j: { label: "Key index", help: "Which key column the cell trace should inspect." }
};

const paramOptions: Record<string, ParamOption[]> = {
  policy: [
    { value: "subword", label: "Subword", consequence: "Balanced pieces; this is the expected tokenizer repair." },
    { value: "word", label: "Word", consequence: "Simple, but cannot split unknown compound pieces." },
    { value: "char", label: "Character", consequence: "Always covers text but usually blows the token budget." }
  ],
  fallback: [
    { value: "unk", label: "<unk>", consequence: "Stable fallback for unseen pieces." },
    { value: "char", label: "Character fallback", consequence: "Recovers unknown words at the cost of longer sequences." },
    { value: "none", label: "No fallback", consequence: "Blocks when hidden text contains unseen pieces." }
  ],
  padSide: [
    { value: "right", label: "Right", consequence: "Expected for the current training batches." },
    { value: "left", label: "Left", consequence: "Can shift EOS and mask positions." }
  ],
  maskPolicy: [
    { value: "pad-aware", label: "PAD-aware", consequence: "Correct: padding tokens are masked out." },
    { value: "all-ones", label: "All ones", consequence: "Wrong for padded batches; PAD positions look valid." }
  ],
  orientation: [
    { value: "C,O", label: "C,O", consequence: "Already projection-ready for hidden [B,T,C]." },
    { value: "O,C", label: "O,C", consequence: "Stored backward; add a transpose before MatMul." }
  ],
  maskOrientation: [
    { value: "query_key", label: "Query x Key", consequence: "Correct triangle: each query row blocks future keys." },
    { value: "key_query", label: "Key x Query", consequence: "Reversed triangle; future cells leak through." }
  ]
};

function GraphRankPanel({ level, graph, runState }: { level: LevelSpec; graph: GraphSpec; runState: RunState }) {
  const { language, t } = useGraphT();
  const rank = computeRank(level, graph, runState, language);
  return (
    <section className={`graphRankPanel ${rank.rank === "-" ? "unranked" : rank.rank.toLowerCase()}`}>
      <div className="graphRunPanelHeader">
        <h3>{t("Rank / Debrief")}</h3>
        <code>{rank.rank}</code>
      </div>
      <p>{rank.message}</p>
      <div className="graphRankStats">
        <code>{t("visible")} {runState.stats.visibleRuns}</code>
        <code>{t("hidden")} {runState.stats.hiddenRuns}</code>
        <code>{t("failed")} {runState.stats.failedRuns}</code>
        <code>{t("hints")} {runState.stats.hintsUsed}</code>
        <code>{t("extra")} {rank.extraModules}</code>
      </div>
      {runState.hidden?.status === "pass" ? (
        <div className="graphDebriefBox">
          <b>{t(level.debrief.completeTitle)}</b>
          <small>{t(level.debrief.fixedProblem)}</small>
          <small>{t(level.debrief.learned)}</small>
          <small>{t(level.debrief.nextUse)}</small>
        </div>
      ) : null}
    </section>
  );
}

function ComponentLifecycleHeader({
  component,
  available,
  visiblePassed,
  missingRequirements,
  availableComponentIds
}: {
  component: ComponentFlowLevelSpec;
  available: boolean;
  visiblePassed: boolean;
  missingRequirements: string[];
  availableComponentIds: Set<string>;
}) {
  const { t } = useGraphT();
  const lifecycle = available ? "Available" : visiblePassed ? "Visible Passed" : missingRequirements.length ? "Blueprint Locked" : "Draft";
  return (
    <div className="graphLifecycleHeader">
      <code>{t(lifecycle)}</code>
      <code>{component.title}</code>
      <code>{`v${component.version}`}</code>
      {component.requires.map((componentId) => {
        const met = availableComponentIds.has(componentId);
        return (
          <span key={componentId} className={met ? "met" : "missing"}>
            {met ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
            {componentId}
          </span>
        );
      })}
    </div>
  );
}

function GraphCompletionNotice({ notice }: { notice: CompletionNotice }) {
  const { t } = useGraphT();
  const title = notice.componentTitle ? `${t(notice.componentTitle)} ${t("is now available")}` : t("Level complete");
  const next = notice.nextLevelTitle ? `${t("Entering next challenge")}: ${t(notice.nextLevelTitle)}` : t("Route complete");
  return (
    <div className="graphCompletionNotice" role="status" aria-live="polite">
      <CheckCircle2 size={18} />
      <span>
        <b>{title}</b>
        <small>{next}</small>
      </span>
    </div>
  );
}

function buildCertificationAnimationLines(
  level: LevelSpec,
  graph: GraphSpec,
  modules: ModuleDef[],
  testCase: TestCase,
  controls: CertificationControlSpec[],
  values: Record<string, CertificationControlValue>
): CertificationAnimationLine[] {
  const sections = generateGraphCodeSections(level, graph, modules, testCase);
  return [
    { id: "cert-section-values", text: "# Validation values", section: true },
    ...certificationValueLines(controls, values),
    { id: "cert-section-input", text: "# Generated task input", section: true },
    ...sections.caseCode.map((line) => ({ id: `cert-case-${line.id}`, text: line.text })),
    { id: "cert-section-graph", text: "# Execute implementation graph", section: true },
    ...sections.graphCode.map((line) => ({ id: `cert-graph-${line.id}`, text: line.text })),
    { id: "cert-section-test", text: "# Validation assertions", section: true },
    ...sections.testCode.map((line) => ({ id: `cert-test-${line.id}`, text: line.text }))
  ];
}

function certificationValueLines(controls: CertificationControlSpec[], values: Record<string, CertificationControlValue>): CertificationAnimationLine[] {
  if (!controls.length) return [{ id: "cert-value-system", text: "variant = system_validation_case()" }];
  return controls.map((control) => ({
    id: `cert-value-${control.id}`,
    text: `${control.id} = ${formatCertificationLiteral(values[control.id] ?? control.defaultValue)}`
  }));
}

function formatCertificationLiteral(value: CertificationControlValue) {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function buildCertificationOutputLines(testCase: TestCase, publicCase: TestCaseRunResult, result: RunTestsResult) {
  const outputNodeId = firstAssertionOutputNodeId(testCase.assertions);
  const outputValue = outputNodeId ? runtimeValueForNode(publicCase.execution.values, outputNodeId) : undefined;
  const lines = [
    `public_variant = ${publicCase.status}`,
    `all_variants = ${result.status}`
  ];
  if (outputNodeId) {
    lines.push(`${outputNodeId} -> ${formatRuntimeValueSummary(outputValue)}`);
  }
  const firstFailure = result.results.find((item) => item.status !== "pass");
  if (firstFailure) {
    lines.push(`first_failure = ${firstFailure.message}`);
  }
  return lines;
}

function firstAssertionOutputNodeId(assertions: TestAssertion[]) {
  for (const assertion of assertions) {
    const nodeId = assertionOutputNodeId(assertion);
    if (nodeId) return nodeId;
  }
  return undefined;
}

function assertionOutputNodeId(assertion: TestAssertion) {
  switch (assertion.type) {
    case "dtype":
    case "shape":
    case "axis_semantics":
    case "allclose":
    case "pieces_non_empty":
    case "pieces_equal":
    case "no_oov":
    case "tokens_include":
    case "eos_preserved":
    case "token_budget":
    case "future_attention_zero":
    case "row_sum":
      return assertion.nodeId;
    case "mask_pad":
      return assertion.maskNodeId;
    default:
      return undefined;
  }
}

function runtimeValueForNode(values: Record<string, RuntimeValue>, nodeId: string) {
  if (values[nodeId]) return values[nodeId];
  if (nodeId.includes(".")) return values[nodeId];
  return values[`${nodeId}.out`] ?? Object.entries(values).find(([key]) => key.startsWith(`${nodeId}.`))?.[1];
}

function formatRuntimeValueSummary(value: RuntimeValue | undefined) {
  if (!value) return "no runtime value";
  const shape = value.shape ? formatShape(value.shape) : value.dtype;
  if (Array.isArray(value.data)) return `${shape} sample=${formatSmallSample(value.data)}`;
  if (value.data !== undefined) return `${shape} value=${formatUnknown(value.data)}`;
  return shape;
}

function GraphCertificationRunModal({
  run,
  onComplete
}: {
  run: CertificationRunPreview;
  onComplete: () => void;
}) {
  const { t, status } = useGraphT();
  const [activeLine, setActiveLine] = useState(0);
  const [phase, setPhase] = useState<"running" | "result">("running");
  const completedRef = useRef(false);
  const visibleCodeLines = phase === "result" ? run.codeLines : run.codeLines.slice(0, activeLine + 1);

  function completeOnce() {
    if (completedRef.current) return;
    completedRef.current = true;
    onComplete();
  }

  useEffect(() => {
    let lineIndex = 0;
    completedRef.current = false;
    setActiveLine(0);
    setPhase("running");

    const interval = window.setInterval(() => {
      lineIndex += 1;
      if (lineIndex >= run.codeLines.length) {
        window.clearInterval(interval);
        setPhase("result");
        return;
      }
      setActiveLine(lineIndex);
    }, 115);

    return () => {
      window.clearInterval(interval);
    };
  }, [onComplete, run.id, run.codeLines.length]);

  const resultStatus = run.result.status;
  return (
    <div className="graphCertificationRunOverlay" role="dialog" aria-modal="true" aria-labelledby="graphCertificationRunTitle">
      <section className={`graphCertificationRunPanel ${phase} ${resultStatus}`}>
        <div className="graphCertificationRunHeader">
          <span>
            <p className="eyebrow">{t("Validation Run")}</p>
            <h3 id="graphCertificationRunTitle">{t(run.title)}</h3>
            <small>{t("Running generated validation code with your filled values.")}</small>
          </span>
          <code>{phase === "running" ? t("running") : status(resultStatus)}</code>
        </div>

        <div className="graphCertificationRunValues">
          <b>{t("Filled values")}</b>
          <div>
            {run.controls.length ? run.controls.map((control) => (
              <code key={control.id}>{t(control.label)} = {formatUnknown(run.values[control.id] ?? control.defaultValue)}</code>
            )) : <code>{t("System validation input")}</code>}
          </div>
        </div>

        <div className="graphCertificationRunCode" aria-live="polite">
          {visibleCodeLines.map((line, index) => (
            <div
              key={line.id}
              className={`graphCertificationRunLine ${line.section ? "section" : ""} ${index === activeLine && phase === "running" ? "active" : ""} ${index < activeLine || phase === "result" ? "done" : ""}`}
            >
              <span>{line.section ? ">" : String(index + 1).padStart(2, "0")}</span>
              <code>{renderHighlightedCode(line.text)}</code>
            </div>
          ))}
        </div>

        <div className={`graphCertificationRunResult ${phase === "result" ? "shown" : ""}`}>
          <b>{t("Result")}: {status(resultStatus)}</b>
          <small>{t(run.publicCase.title)}</small>
          {run.outputLines.map((line) => (
            <code key={line}>{line}</code>
          ))}
        </div>

        <div className="graphCertificationRunFooter">
          <span>{phase === "running" ? t("Executing trace...") : t("Validation result captured.")}</span>
          <button className="ghostButton" type="button" onClick={completeOnce}>
            {phase === "running" ? t("Skip animation") : t("Continue")}
          </button>
        </div>
      </section>
    </div>
  );
}

function renderHighlightedCode(text: string) {
  return tokenizeCodeLine(text).map((token, index) => (
    <span key={`${index}:${token.text}`} className={token.className}>
      {token.text}
    </span>
  ));
}

function tokenizeCodeLine(text: string): Array<{ text: string; className?: string }> {
  const tokens: Array<{ text: string; className?: string }> = [];
  let index = 0;
  while (index < text.length) {
    const char = text[index];

    if (/\s/.test(char)) {
      const start = index;
      while (index < text.length && /\s/.test(text[index])) index += 1;
      tokens.push({ text: text.slice(start, index) });
      continue;
    }

    if (char === "#") {
      tokens.push({ text: text.slice(index), className: "syntaxComment" });
      break;
    }

    if (char === "\"" || char === "'") {
      const quote = char;
      const start = index;
      index += 1;
      while (index < text.length) {
        if (text[index] === "\\") {
          index += 2;
          continue;
        }
        if (text[index] === quote) {
          index += 1;
          break;
        }
        index += 1;
      }
      tokens.push({ text: text.slice(start, index), className: "syntaxString" });
      continue;
    }

    const numberMatch = text.slice(index).match(/^-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/i);
    if (numberMatch) {
      tokens.push({ text: numberMatch[0], className: "syntaxNumber" });
      index += numberMatch[0].length;
      continue;
    }

    const identifierMatch = text.slice(index).match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (identifierMatch) {
      const word = identifierMatch[0];
      const nextChar = text.slice(index + word.length).trimStart()[0];
      const previousChar = previousNonSpace(text, index);
      tokens.push({
        text: word,
        className: codeKeywordSet.has(word)
          ? "syntaxKeyword"
          : nextChar === "(" || previousChar === "."
            ? "syntaxFunction"
            : undefined
      });
      index += word.length;
      continue;
    }

    tokens.push({
      text: char,
      className: "=+-*/@<>!~:,[](){}.".includes(char) ? "syntaxOperator" : undefined
    });
    index += 1;
  }
  return tokens;
}

function previousNonSpace(text: string, index: number) {
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    if (!/\s/.test(text[cursor])) return text[cursor];
  }
  return undefined;
}

const codeKeywordSet = new Set([
  "assert",
  "True",
  "False",
  "None",
  "and",
  "or",
  "not",
  "in",
  "is",
  "like",
  "module"
]);

function GraphCertificationPanel({
  certification,
  values,
  errors,
  taskPassed,
  certificationResult,
  onChange,
  onSubmit
}: {
  certification: LevelCertificationSpec;
  values: Record<string, CertificationControlValue>;
  errors: string[];
  taskPassed: boolean;
  certificationResult?: RunTestsResult;
  onChange: (controlId: string, value: CertificationControlValue) => void;
  onSubmit: () => void;
}) {
  const { t, status } = useGraphT();
  const canSubmit = taskPassed && errors.length === 0;
  return (
    <section className={`graphCertificationPanel ${taskPassed ? "ready" : "locked"} ${certificationResult?.status ?? ""}`}>
      <div className="graphCertificationHeader">
        <span>
          <p className="eyebrow">{t("Validation")}</p>
          <h3>{t(certification.title)}</h3>
        </span>
        <code>{certificationResult ? status(certificationResult.status) : taskPassed ? t("ready") : t("locked")}</code>
      </div>
      <p>{t(taskPassed ? certification.narrative : "The current task must pass before validation can start.")}</p>
      <div className="graphCertificationVariant">
        <b>{t(certification.publicVariantLabel)}</b>
        <small>{t(certification.publicVariantDescription)}</small>
        <div className="graphCertificationControls">
          {certification.controls.map((control) => (
            <CertificationControl key={control.id} control={control} value={values[control.id] ?? control.defaultValue} disabled={!taskPassed} onChange={onChange} />
          ))}
        </div>
      </div>
      <div className="graphCertificationSystem">
        <b>{t("System variants")}</b>
        <small>{t(certification.systemVariantDescription)}</small>
      </div>
      {errors.length ? (
        <div className="graphCertificationErrors">
          {errors.map((error) => (
            <small key={error}>{t(error)}</small>
          ))}
        </div>
      ) : null}
      <button className="runButton" type="button" disabled={!canSubmit} onClick={onSubmit}>
        <Play size={14} />
        {t("Submit Validation")}
      </button>
    </section>
  );
}

function CertificationControl({
  control,
  value,
  disabled,
  onChange
}: {
  control: CertificationControlSpec;
  value: CertificationControlValue;
  disabled: boolean;
  onChange: (controlId: string, value: CertificationControlValue) => void;
}) {
  const { t } = useGraphT();
  if (control.kind === "select") {
    return (
      <label className="graphCertificationControl">
        <span>{t(control.label)}</span>
        <select disabled={disabled} value={String(value)} onChange={(event) => onChange(control.id, event.currentTarget.value)}>
          {(control.options ?? []).map((option) => (
            <option key={option.value} value={option.value}>
              {t(option.label)}
            </option>
          ))}
        </select>
        {control.help ? <small>{t(control.help)}</small> : null}
      </label>
    );
  }

  return (
    <label className="graphCertificationControl">
      <span>{t(control.label)}</span>
      <input
        disabled={disabled}
        type="text"
        inputMode={control.kind === "integer" ? "numeric" : "decimal"}
        value={String(value)}
        onChange={(event) => onChange(control.id, event.currentTarget.value)}
      />
      {control.help ? <small>{t(control.help)}</small> : null}
    </label>
  );
}

function GraphRunPanel({
  title,
  result,
  locked = false,
  onLocateNode,
  onShowNextStep
}: {
  title: string;
  result?: RunTestsResult;
  locked?: boolean;
  onLocateNode: (nodeId: string) => void;
  onShowNextStep: () => void;
}) {
  const { t, status } = useGraphT();
  return (
    <section className={`graphRunPanel ${result?.status ?? (locked ? "blocked" : "idle")}`}>
      <div className="graphRunPanelHeader">
        <h3>{t(title)}</h3>
        <code>{locked ? t("locked") : status(result?.status ?? "idle")}</code>
      </div>
      {locked ? <p>{t("Task check must pass before validation.")}</p> : null}
      {result ? (
        <div className="graphResultList">
          {result.results.map((item) => (
            <GraphResultItem key={item.id} item={item} detailed={item.status !== "pass"} onLocateNode={onLocateNode} onShowNextStep={onShowNextStep} />
          ))}
        </div>
      ) : null}
    </section>
  );
}

function GraphResultItem({
  item,
  detailed = false,
  onLocateNode,
  onShowNextStep
}: {
  item: TestResult;
  detailed?: boolean;
  onLocateNode: (nodeId: string) => void;
  onShowNextStep: () => void;
}) {
  const { t, status } = useGraphT();
  const Icon = item.status === "pass" ? CheckCircle2 : AlertTriangle;
  return (
    <div className={`graphResultItem ${item.status}`}>
      <Icon size={15} />
      <span>
        <b>{status(item.status)}</b>
        <small>{item.message}</small>
        {detailed && item.diagnostic ? (
          <>
            <small>{t("expected")}: {formatUnknown(item.diagnostic.expected)}</small>
            <small>{t("received")}: {formatUnknown(item.diagnostic.received)}</small>
            <small>{t("cause")}: {item.diagnostic.possibleCause ?? "unknown"}</small>
            <small>{t("probe hint")}: {item.diagnostic.suggestedProbe ?? t("step through trace")}</small>
          </>
        ) : null}
        {detailed ? (
          <span className="graphResultActions">
            {item.firstBadNodeId ? (
              <button type="button" onClick={() => onLocateNode(item.firstBadNodeId ?? "")}>
                {t("Locate Node")}
              </button>
            ) : null}
            <button type="button" onClick={onShowNextStep}>
              {t("Next step")}
            </button>
          </span>
        ) : null}
      </span>
    </div>
  );
}

function initialGraphsByLevel(levels: LevelSpec[]) {
  return Object.fromEntries(levels.map((level) => [level.id, level.initialGraph]));
}

function initialCertificationValuesByLevel(levels: LevelSpec[]): CertificationValuesByLevel {
  return Object.fromEntries(levels.map((level) => [level.id, defaultCertificationValues(level)]));
}

function defaultCertificationValues(level: LevelSpec) {
  return Object.fromEntries((level.certification?.controls ?? []).map((control) => [control.id, control.defaultValue]));
}

function certificationValuesForLevel(level: LevelSpec, values: Record<string, CertificationControlValue> | undefined) {
  return { ...defaultCertificationValues(level), ...(values ?? {}) };
}

function certificationControlErrors(certification: LevelCertificationSpec | undefined, values: Record<string, CertificationControlValue>) {
  if (!certification) return [];
  return certification.controls.flatMap((control) => {
    const value = values[control.id] ?? control.defaultValue;
    if (control.kind === "select") {
      const valid = (control.options ?? []).some((option) => option.value === String(value));
      return valid ? [] : ["Invalid validation option"];
    }

    const parsed = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(parsed)) return [control.kind === "number" ? "Enter a finite float32 number" : "Enter a finite number"];
    if (control.kind === "number" && !Number.isFinite(Math.fround(parsed))) {
      return ["Value is outside float32 range"];
    }
    if (control.kind === "integer" && !Number.isInteger(parsed)) return ["Enter an integer"];
    if (control.min !== undefined && parsed < control.min) return [`Minimum is ${control.min}`];
    if (control.max !== undefined && parsed > control.max) return [`Maximum is ${control.max}`];
    return [];
  });
}

function statusFromResults(results: TestResult[]) {
  return results.some((result) => result.status === "fail")
    ? "fail"
    : results.some((result) => result.status === "blocked")
      ? "blocked"
      : "pass";
}

function lifecycleLabel({ available, missingCount }: { available: boolean; missingCount: number }) {
  if (available) return "Available";
  if (missingCount > 0) return "Locked";
  return "Draft";
}

function tensorPreviewRows(value: RuntimeValue, maxRows: number, maxCols: number) {
  const data = Array.isArray(value.data) ? value.data : [];
  const dims = value.shape?.dims ?? [];
  const axes = value.shape?.axes ?? [];
  if (!dims.length) {
    return [{ label: "scalar", values: [formatTensorCell(data[0])] }];
  }
  if (dims.length === 1) {
    return [{ label: axes[0] ?? "dim0", values: data.slice(0, maxCols).map(formatTensorCell) }];
  }

  const rowCount = Math.min(maxRows, dims.slice(0, -1).reduce((total, dim) => total * dim, 1));
  const colCount = Math.min(maxCols, dims[dims.length - 1]);
  const rows: Array<{ label: string; values: string[] }> = [];
  for (let row = 0; row < rowCount; row += 1) {
    const prefixIndex = unravelPreviewIndex(row, dims.slice(0, -1));
    const label = prefixIndex.map((slot, index) => `${axes[index] ?? `d${index}`}=${slot}`).join(" ");
    const offset = row * dims[dims.length - 1];
    rows.push({
      label,
      values: data.slice(offset, offset + colCount).map(formatTensorCell)
    });
  }
  return rows;
}

function unravelPreviewIndex(flatIndex: number, dims: number[]) {
  const index = Array.from({ length: dims.length }, () => 0);
  let remainder = flatIndex;
  for (let dimIndex = dims.length - 1; dimIndex >= 0; dimIndex -= 1) {
    index[dimIndex] = dims[dimIndex] ? remainder % dims[dimIndex] : 0;
    remainder = dims[dimIndex] ? Math.floor(remainder / dims[dimIndex]) : 0;
  }
  return index;
}

function formatTensorCell(value: unknown) {
  if (typeof value === "number") return Number(value.toFixed(4)).toString();
  if (typeof value === "string") return value;
  return String(value ?? "");
}

function createEmptyRunState(): RunState {
  return {
    stats: {
      visibleRuns: 0,
      hiddenRuns: 0,
      failedRuns: 0,
      hintsUsed: 0
    }
  };
}

function firstBadNodeIdFromRunState(runState: RunState) {
  const firstBad =
    runState.visible?.results.find((result) => result.status !== "pass" && result.firstBadNodeId) ??
    runState.hidden?.results.find((result) => result.status !== "pass" && result.firstBadNodeId);
  return firstBad?.firstBadNodeId?.split(".")[0];
}

function getChecklistItems(level: LevelSpec, graph: GraphSpec) {
  const recipe = level.onboarding?.targetRecipe ?? [];
  return recipe.map((label) => ({
    label,
    done: recipeItemDone(label, graph)
  }));
}

function recipeItemDone(label: string, graph: GraphSpec) {
  const edge = parseRecipeEdge(label);
  if (edge) return graphHasEdge(graph, edge.from, edge.to);
  const param = parseRecipeParam(label);
  if (param) {
    const node = graph.nodes.find((item) => item.id === param.nodeId);
    return String(node?.params[param.key]) === param.value;
  }
  return false;
}

function parseRecipeEdge(label: string): { from: PortRef; to: PortRef } | undefined {
  const match = label.match(/([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)/);
  if (!match) return undefined;
  return {
    from: { nodeId: match[1], portId: match[2] },
    to: { nodeId: match[3], portId: match[4] }
  };
}

function parseRecipeParam(label: string): { nodeId: string; key: string; value: string } | undefined {
  const match = label.match(/([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*=\s*(-?[A-Za-z0-9_.,]+)/);
  if (!match) return undefined;
  return { nodeId: match[1], key: match[2], value: match[3] };
}

function graphHasEdge(graph: GraphSpec, from: PortRef, to: PortRef) {
  return graph.edges.some((edge) => edge.from.nodeId === from.nodeId && edge.from.portId === from.portId && edge.to.nodeId === to.nodeId && edge.to.portId === to.portId);
}

function estimateCurrentNodeSizes(graph: GraphSpec) {
  const sizes: Record<string, { width: number; height: number }> = {};
  graph.nodes.forEach((node) => {
    const chipCount = node.moduleId === "TokenizerSocket" || node.moduleId === "MatMulGate" || node.moduleId === "InputTensor" || node.moduleId === "WeightPlate" ? 3 : 2;
    sizes[node.moduleId] = {
      width: graphNodeWidth,
      height: graphNodeMinHeight + Math.ceil(chipCount / 2) * 32 + 18
    };
  });
  return sizes;
}

function getNodeCaseChips(level: LevelSpec, graph: GraphSpec, modules: ModuleDef[], node: GraphNode, module: ModuleDef): CaseChip[] {
  const visibleCase = level.visibleTests[0];
  if (!visibleCase) return [];
  const mvp01Chips = getMvp01NodeCaseChips(level, graph, node, module, visibleCase);
  if (mvp01Chips.length) return mvp01Chips;

  if (node.moduleId === "TextInput") {
    const inputKey = String(node.params.inputKey ?? "texts");
    const texts = getCaseTexts(visibleCase, inputKey);
    const focusText = getFocusText(level, texts);
    return [
      ...(focusText ? [{ label: "text", value: quoteShort(focusText), tone: "ok" as const }] : []),
      ...(texts.length > 1 ? [{ label: "batch", value: `${texts.length} texts`, tone: "muted" as const }] : [])
    ];
  }

  if (node.moduleId === "TokenizerSocket") {
    const preview = tokenizerPreviewForNode(level, graph, modules, node);
    if (!preview) return [];
    return [
      { label: "strategy", value: `${formatPolicy(preview.policy)}${preview.applyMerges ? " + merges" : ""}`, tone: preview.withinBudget ? "ok" : "warn" },
      { label: "budget", value: `${preview.rawTokenCount}/${preview.maxBudget}`, tone: preview.withinBudget && !preview.truncated ? "ok" : "bad" },
      { label: "pieces", value: preview.pieces.slice(0, 4).join(" | ") + (preview.pieces.length > 4 ? " ..." : ""), tone: preview.withinBudget ? "ok" : "warn" }
    ];
  }

  if (node.moduleId === "EmbeddingReadyProbe") {
    const tokenizerNode = incomingNode(graph, node, "ids");
    const preview = tokenizerNode?.moduleId === "TokenizerSocket" ? tokenizerPreviewForNode(level, graph, modules, tokenizerNode) : undefined;
    if (preview) {
      return [
        { label: "ids", value: `int[1,${preview.ids.length}]`, tone: "ok" },
        { label: "mask", value: preview.mask.slice(0, 8).join(" "), tone: preview.mask.includes(0) ? "ok" : "muted" }
      ];
    }
    return [{ label: "expects", value: "int token IDs", tone: "muted" }];
  }

  if (isMatMulModule(node.moduleId)) {
    const left = estimateOutputShape(level, graph, node, "left");
    const right = estimateOutputShape(level, graph, node, "right");
    const out = estimateNodeOutputShape(level, graph, node);
    return [
      ...(left ? [{ label: "left", value: formatShape(left), tone: "muted" as const }] : []),
      ...(right ? [{ label: "right", value: formatShape(right), tone: left && right && left.dims[left.dims.length - 1] === right.dims[0] ? "ok" as const : "warn" as const }] : []),
      ...(out ? [{ label: "out", value: formatShape(out), tone: "ok" as const }] : [])
    ];
  }

  if (node.moduleId === "InputTensor" || node.moduleId === "WeightPlate") {
    const inputKey = String(node.params.inputKey ?? node.id);
    const value = visibleCase.inputs[inputKey];
    return [
      { label: "input", value: inputKey, tone: "muted" },
      ...(value?.shape ? [{ label: "shape", value: formatShape(value.shape), tone: "ok" as const }] : []),
      ...(Array.isArray(value?.data) ? [{ label: "sample", value: formatSmallSample(value.data), tone: "muted" as const }] : [])
    ];
  }

  if (node.moduleId === "TransposeSwitch") {
    const inputShape = estimateOutputShape(level, graph, node, "x");
    return [
      ...(inputShape ? [{ label: "in", value: formatShape(inputShape), tone: "muted" as const }] : []),
      { label: "swap", value: `${node.params.axisA ?? -2} <-> ${node.params.axisB ?? -1}`, tone: "ok" }
    ];
  }

  if (node.moduleId === "OutputContractGate" || node.moduleId === "ScoreBoard" || node.moduleId === "AxisAlignmentRuler" || node.moduleId === "AxisLock") {
    const expectedAxes = Array.isArray(node.params.expectedAxes)
      ? node.params.expectedAxes
      : Array.isArray(node.params.expectedPrefixAxes)
        ? node.params.expectedPrefixAxes
        : [];
    const inputPort = module.inputs[0]?.id ?? "x";
    const inputShape = estimateOutputShape(level, graph, node, inputPort);
    return [
      ...(inputShape ? [{ label: "task", value: formatShape(inputShape), tone: "muted" as const }] : []),
      ...(expectedAxes.length ? [{ label: "expects", value: `[${expectedAxes.join(",")}]`, tone: "ok" as const }] : [])
    ];
  }

  if (node.moduleId === "ReferenceChecker") {
    const referenceKey = String(node.params.referenceKey ?? "reference");
    const value = visibleCase.inputs[referenceKey];
    return [
      { label: "Role", value: "prebuilt probe", tone: "ok" },
      { label: "INPUT", value: "compare x", tone: "muted" },
      ...(value?.shape ? [{ label: "reference", value: formatShape(value.shape), tone: "muted" as const }] : [])
    ];
  }

  return [];
}

function getMvp01NodeCaseChips(level: LevelSpec, graph: GraphSpec, node: GraphNode, module: ModuleDef, visibleCase: TestCase): CaseChip[] {
  if (!level.id.startsWith("mvp01_")) return [];
  const caseHeadline = getCaseHeadline(visibleCase);
  const outputShape = estimateNodeOutputShape(level, graph, node);

  if (node.moduleId === "Float32Literal") {
    const value = node.params.value ?? 0.5;
    const error = finiteFloat32InputError(value);
    return [
      { label: "Source", value: "scalar value", tone: "muted", help: "A raw finite number primitive used to build the ScalarCell implementation." },
      { label: "VALUE", value: String(value), tone: error ? "bad" : "ok", help: "Edit this literal; any finite float32 value is valid for the scalar contract.", error, editable: { paramKey: "value", kind: "finite_float32" } },
      { label: "SHAPE", value: error ? "blocked" : "float32[]", tone: error ? "bad" : "ok", help: "A scalar has rank 0: no row, column, or axis dimensions." }
    ];
  }

  if (node.moduleId === "component.scalar_cell.v1") {
    const value = node.params.value ?? 0.5;
    const error = finiteFloat32InputError(value);
    return [
      { label: "Component", value: "ScalarCell", tone: "ok", help: "Validated ScalarCell can be reused as a small component in later graphs." },
      { label: "VALUE", value: String(value), tone: error ? "bad" : "ok", help: "Edit this scalar instance; it must remain a finite float32 value.", error, editable: { paramKey: "value", kind: "finite_float32" } },
      { label: "Output", value: error ? "blocked" : "float32[]", tone: error ? "bad" : "muted", help: "The reusable component still exposes a rank-0 float32 output." }
    ];
  }

  if (node.moduleId === "InputTensor" || node.moduleId === "WeightPlate") {
    const inputKey = String(node.params.inputKey ?? node.id);
    const value = visibleCase.inputs[inputKey];
    return [
      ...(caseHeadline ? [{ label: "Source", value: "current case", tone: "muted" as const, help: "This node reads the current task case. Validation can replace it with variant inputs." }] : []),
      { label: "INPUT", value: inputKey, tone: "muted", help: "The case field consumed by this source node." },
      ...(value?.shape ? [{ label: "SHAPE", value: formatShape(value.shape), tone: "ok" as const, help: "Shape and axis labels carried by the current case value." }] : []),
      ...(Array.isArray(value?.data) ? [{ label: "Sample", value: formatSmallSample(value.data), tone: "muted" as const, help: "A small preview of the current case value; do not hard-code only this sample." }] : [])
    ];
  }

  if (node.moduleId === "TextInput") {
    const inputKey = String(node.params.inputKey ?? "texts");
    const texts = getCaseTexts(visibleCase, inputKey);
    const focusText = getFocusText(level, texts);
    return [
      { label: "Source", value: "current case", tone: "muted", help: "This node reads the current text case. Validation can swap in different text." },
      { label: "INPUT", value: inputKey, tone: "muted", help: "The text field consumed by this source node." },
      ...(focusText ? [{ label: "Sample", value: quoteShort(focusText), tone: "muted" as const, help: "A preview of the sentence being split in the current task." }] : [])
    ];
  }

  if (node.moduleId === "VectorRail") {
    return [
      { label: "Role", value: "make vector[C]", tone: "muted", help: "Combines scalar cells into one ordered C-axis feature vector." },
      { label: "Combine", value: "inputs -> C", tone: "ok", help: "Input source order becomes C-axis position." },
      { label: "Output", value: outputShape ? formatShape(outputShape) : "float32[C=3]", tone: "ok", help: "The vector carries one semantic C axis." }
    ];
  }

  if (node.moduleId === "MatrixStruct") {
    return [
      { label: "Role", value: "build matrix[C,O]", tone: "muted", help: "Stacks vector columns while preserving C as the input feature axis." },
      { label: "Structure", value: "O0/O1 columns", tone: "ok", help: "Each input vector becomes one O-axis column." },
      { label: "Output", value: outputShape ? formatShape(outputShape) : "float32[C=3,O=2]", tone: "ok", help: "The matrix has C rows and O output columns." }
    ];
  }

  if (node.moduleId === "TensorBox") {
    return [
      { label: "Role", value: "make tensor[B,T,C]", tone: "muted", help: "Wraps token vectors into batch and time axes without losing the C feature rail." },
      { label: "Stack", value: "t0,t1 -> T", tone: "ok", help: "Each vector becomes one token row along T." },
      { label: "Output", value: outputShape ? formatShape(outputShape) : "float32[B=1,T=2,C=3]", tone: "ok", help: "The tensor carries batch B, token T, and channel C axes." }
    ];
  }

  if (node.moduleId === "ElementwiseMultiply") {
    const left = estimateOutputShape(level, graph, node, "left");
    const right = estimateOutputShape(level, graph, node, "right");
    const compatible = Boolean(left && right && left.dims.join(",") === right.dims.join(",") && left.axes.join(",") === right.axes.join(","));
    return [
      { label: "Role", value: "cellwise multiply", tone: "muted", help: "Multiplies matching cells from the two inputs and keeps the same shape and axes." },
      ...(left ? [{ label: "LEFT", value: formatShape(left), tone: "muted" as const, help: "Left tensor entering the cellwise multiply." }] : []),
      ...(right ? [{ label: "RIGHT", value: formatShape(right), tone: compatible ? "ok" as const : "warn" as const, help: "Right tensor must match the left shape and axes cell by cell." }] : [])
    ];
  }

  if (node.moduleId === "SumReduce") {
    const input = estimateOutputShape(level, graph, node, "x");
    const axis = String(node.params.axis ?? "C");
    return [
      { label: "Role", value: "sum over C", tone: "muted", help: "Adds all values along the selected axis and removes that axis from the output." },
      ...(input ? [{ label: "INPUT", value: formatShape(input), tone: "muted" as const, help: "Tensor being reduced before the output contract." }] : []),
      { label: "Axis", value: axis, tone: input?.axes.some((item) => item === axis) ? "ok" : "warn", help: "The semantic axis to reduce. DotProduct reduces the shared C axis." }
    ];
  }

  if (isMatMulModule(node.moduleId)) {
    const left = estimateOutputShape(level, graph, node, "left");
    const right = estimateOutputShape(level, graph, node, "right");
    const compatible = Boolean(left && right && left.dims[left.dims.length - 1] === right.dims[0]);
    return [
      { label: "Role", value: "C axis match", tone: "muted", help: "Consumes the left input's final C axis against the right input's leading C axis." },
      ...(left ? [{ label: "LEFT", value: formatShape(left), tone: "muted" as const, help: "Left activation tensor entering MatMul." }] : []),
      ...(right ? [{ label: "RIGHT", value: formatShape(right), tone: compatible ? "ok" as const : "warn" as const, help: "Right weight tensor must expose a matching C axis." }] : []),
      { label: "Output", value: outputShape ? formatShape(outputShape) : "[B,T,O]", tone: compatible || outputShape ? "ok" : "warn", help: "MatMul keeps carrier axes and emits the O output axis." }
    ];
  }

  if (node.moduleId === "BroadcastRail") {
    const target = estimateOutputShape(level, graph, node, "target");
    const small = estimateOutputShape(level, graph, node, "small");
    return [
      { label: "Role", value: "broadcast bias", tone: "muted", help: "Expands a small tensor logically across the target carrier axes." },
      ...(small ? [{ label: "BIAS", value: formatShape(small), tone: "muted" as const, help: "Small tensor being aligned before addition." }] : []),
      { label: "Align", value: `O -> ${formatAxisList(target?.axes ?? ["B", "T", "O"])}`, tone: "ok", help: "Bias should attach to O, then expand across B and T." },
      ...(target ? [{ label: "Output", value: formatShape(target), tone: "ok" as const, help: "Broadcast output matches the target tensor shape." }] : [])
    ];
  }

  if (node.moduleId === "AddGate") {
    const left = estimateOutputShape(level, graph, node, "left");
    const right = estimateOutputShape(level, graph, node, "right");
    return [
      { label: "Role", value: "add aligned tensors", tone: "muted", help: "Adds two tensors only after their shape and semantic axes match." },
      ...(left ? [{ label: "SCORE", value: formatShape(left), tone: "muted" as const, help: "Main tensor entering the add gate." }] : []),
      ...(right ? [{ label: "BIAS", value: formatShape(right), tone: "muted" as const, help: "Aligned bias tensor entering the add gate." }] : []),
      { label: "Merge", value: "score + bias", tone: "ok", help: "The output is the elementwise sum of both aligned inputs." }
    ];
  }

  if (node.moduleId === "BoundarySplitter") {
    const keepPunctuation = node.params.preservePunctuation !== false;
    return [
      { label: "Role", value: "split boundaries", tone: "muted", help: "Cuts raw text into ordered pieces before token IDs exist." },
      { label: "Policy", value: String(node.params.policy ?? "word"), tone: "ok", help: "Controls how raw text boundaries are chosen." },
      { label: "Punctuation", value: keepPunctuation ? "keep" : "drop", tone: keepPunctuation ? "ok" : "warn", help: "This challenge needs punctuation to survive as its own piece." }
    ];
  }

  if (node.moduleId === "PieceBuffer") {
    return [
      { label: "Role", value: "hold string_piece[T]", tone: "muted", help: "Keeps splitter output as an ordered T-axis piece buffer for the contract." },
      { label: "Capacity", value: String(node.params.maxPieces ?? 12), tone: "ok", help: "Maximum number of pieces allowed before the buffer rejects the output." }
    ];
  }

  if (node.moduleId === "TypeContractGate") {
    return [
      { label: "Check", value: "string_piece[T]", tone: "ok", help: "Checks that the implementation returns ordered string pieces, not raw text." },
      { label: "Role", value: "output contract", tone: "muted", help: "A contract gate validates the reusable component boundary before validation." }
    ];
  }

  if (node.moduleId === "TransposeSwitch") {
    const inputShape = estimateOutputShape(level, graph, node, "x");
    return [
      { label: "Role", value: "swap last axes", tone: "muted", help: "Swaps only the selected axes while preserving all carrier axes." },
      ...(inputShape ? [{ label: "INPUT", value: formatShape(inputShape), tone: "muted" as const, help: "Tensor entering the transpose switch." }] : []),
      { label: "Swap", value: `${node.params.axisA ?? -2} <-> ${node.params.axisB ?? -1}`, tone: "ok", help: "QKScore needs K's final two axes transposed before MatMul." }
    ];
  }

  if (node.moduleId === "ScoreBoard") {
    return [
      { label: "Check", value: "scores[B,H,T,T]", tone: "ok", help: "Validates the attention score board shape after Q @ K^T." },
      { label: "Role", value: "attention scores", tone: "muted", help: "This is a contract/probe node for score-board semantics, not the MatMul implementation itself." }
    ];
  }

  if (node.moduleId === "CellTrace") {
    return [
      { label: "Probe", value: "one score cell", tone: "ok", help: "Shows how one output score is produced from a query/key dot product." },
      { label: "Role", value: "diagnostic probe", tone: "muted", help: "Use this probe to debug numeric or axis-orientation mistakes." }
    ];
  }

  if (node.moduleId === "OutputContractGate") {
    const expectedAxes = Array.isArray(node.params.expectedAxes) ? node.params.expectedAxes.map(String) : [];
    const inputShape = estimateOutputShape(level, graph, node, "x");
    return [
      ...(inputShape ? [{ label: "Received", value: formatShape(inputShape), tone: "muted" as const, help: "The value currently entering the component output boundary." }] : []),
      { label: "Expected", value: expectedAxes.length ? formatAxisList(expectedAxes) : "rank-0 []", tone: "ok", help: "The output contract accepts only this axis/rank structure." },
      { label: "Check", value: "output contract", tone: "ok", help: "This gate checks the component boundary before the reference probe compares behavior." }
    ];
  }

  if (node.moduleId === "ReferenceChecker") {
    const referenceKey = String(node.params.referenceKey ?? "reference");
    const value = visibleCase.inputs[referenceKey];
    return [
      { label: "Probe", value: "reference answer", tone: "ok", help: "Prebuilt testing equipment: compares your output with the expected answer for this case." },
      { label: "INPUT", value: "compare x", tone: "muted", help: "Connect the component output here so the probe can compare behavior." },
      { label: "REFERENCE", value: referenceKey, tone: "muted", help: "The expected answer comes from the test case, not from the reusable component." },
      ...(value?.shape ? [{ label: "SHAPE", value: formatShape(value.shape), tone: "ok" as const, help: "Shape of the reference answer for the current case." }] : []),
      ...(Array.isArray(value?.data) ? [{ label: "Sample", value: formatSmallSample(value.data), tone: "muted" as const, help: "A small preview of the expected answer for this case." }] : [])
    ];
  }

  return [
    ...(caseHeadline ? [{ label: "Role", value: module.label, tone: "muted" as const, help: "This node participates in the current component build graph." }] : []),
    ...(outputShape ? [{ label: "Output", value: formatShape(outputShape), tone: "ok" as const, help: "Estimated output shape for the current graph wiring." }] : [{ label: "Node", value: module.label, tone: "muted" as const, help: "Module placed in the current component build graph." }])
  ];
}

function getEdgeCaseLabel(level: LevelSpec, graph: GraphSpec, modules: ModuleDef[], edge: GraphEdge) {
  const sourceNode = graph.nodes.find((node) => node.id === edge.from.nodeId);
  if (!sourceNode) return undefined;

  if (sourceNode.moduleId === "TextInput") {
    const inputKey = String(sourceNode.params.inputKey ?? "texts");
    const texts = getCaseTexts(level.visibleTests[0], inputKey);
    const focusText = getFocusText(level, texts);
    return focusText ? quoteShort(focusText, 26) : undefined;
  }

  if (sourceNode.moduleId === "TokenizerSocket") {
    const preview = tokenizerPreviewForNode(level, graph, modules, sourceNode);
    if (!preview) return undefined;
    if (edge.from.portId === "pieces") return `pieces ${preview.rawTokenCount}/${preview.maxBudget}`;
    if (edge.from.portId === "mask") return `mask[${preview.mask.length}]`;
    return `ids[1,${preview.ids.length}]`;
  }

  const shape = estimateNodeOutputShape(level, graph, sourceNode, edge.from.portId);
  return shape ? formatShape(shape) : undefined;
}

function tokenizerPreviewForNode(level: LevelSpec, graph: GraphSpec, modules: ModuleDef[], node: GraphNode) {
  const textInputKey = tokenizerTextInputKey(graph, node) ?? "texts";
  return createTokenizerPreview(level, graph, modules, level.visibleTests[0], node.id, textInputKey);
}

function tokenizerTextInputKey(graph: GraphSpec, tokenizerNode: GraphNode) {
  const textNode = incomingNode(graph, tokenizerNode, "text");
  return textNode ? String(textNode.params.inputKey ?? "texts") : undefined;
}

function incomingNode(graph: GraphSpec, node: GraphNode, portId: string) {
  const edge = graph.edges.find((item) => item.to.nodeId === node.id && item.to.portId === portId);
  return edge ? graph.nodes.find((item) => item.id === edge.from.nodeId) : undefined;
}

function estimateOutputShape(level: LevelSpec, graph: GraphSpec, node: GraphNode, inputPortId: string) {
  const edge = graph.edges.find((item) => item.to.nodeId === node.id && item.to.portId === inputPortId);
  if (!edge) return undefined;
  const sourceNode = graph.nodes.find((item) => item.id === edge.from.nodeId);
  return sourceNode ? estimateNodeOutputShape(level, graph, sourceNode, edge.from.portId) : undefined;
}

function estimateNodeOutputShape(level: LevelSpec, graph: GraphSpec, node: GraphNode, portId = "out"): RuntimeValue["shape"] | undefined {
  const visibleCase = level.visibleTests[0];
  if (!visibleCase) return undefined;

  if (node.moduleId === "Float32Literal" || node.moduleId === "component.scalar_cell.v1") {
    return { dtype: "float32", dims: [], axes: [] };
  }

  if (node.moduleId === "InputTensor" || node.moduleId === "WeightPlate" || node.moduleId === "ReferenceChecker") {
    const inputKey = node.moduleId === "ReferenceChecker" ? String(node.params.referenceKey ?? "reference") : String(node.params.inputKey ?? node.id);
    return visibleCase.inputs[inputKey]?.shape;
  }

  if (node.moduleId === "VectorRail" || node.moduleId === "component.vector_rail.v1") {
    return { dtype: "float32", dims: [3], axes: ["C"] };
  }

  if (node.moduleId === "MatrixStruct" || node.moduleId === "component.matrix_struct.v1") {
    const left = estimateOutputShape(level, graph, node, "o0");
    const c = left?.dims[0] ?? 3;
    return { dtype: "float32", dims: [c, 2], axes: ["C", "O"] };
  }

  if (node.moduleId === "TensorBox" || node.moduleId === "component.tensor_box.v1") {
    const firstToken = estimateOutputShape(level, graph, node, "t0");
    const c = firstToken?.dims[0] ?? 3;
    return { dtype: "float32", dims: [1, 2, c], axes: ["B", "T", "C"] };
  }

  if (node.moduleId === "TokenizerSocket") {
    const preview = tokenizerPreviewForNode(level, graph, [], node);
    if (!preview) return undefined;
    if (portId === "mask") return { dtype: "mask", dims: [1, preview.mask.length], axes: ["B", "T"] };
    if (portId === "pieces") return { dtype: "token_piece", dims: [preview.rawTokenCount], axes: ["T"] };
    return { dtype: "int", dims: [1, preview.ids.length], axes: ["B", "T"] };
  }

  if (node.moduleId === "TransposeSwitch") {
    const input = estimateOutputShape(level, graph, node, "x");
    if (!input) return undefined;
    const rank = input.dims.length;
    const axisA = normalizePreviewAxis(Number(node.params.axisA ?? -2), rank);
    const axisB = normalizePreviewAxis(Number(node.params.axisB ?? -1), rank);
    const dims = [...input.dims];
    const axes = [...input.axes];
    [dims[axisA], dims[axisB]] = [dims[axisB], dims[axisA]];
    [axes[axisA], axes[axisB]] = [axes[axisB], axes[axisA]];
    return { ...input, dims, axes };
  }

  if (isMatMulModule(node.moduleId)) {
    const left = estimateOutputShape(level, graph, node, "left");
    const right = estimateOutputShape(level, graph, node, "right");
    if (!left || !right || left.dims.length < 1 || right.dims.length < 2) return undefined;
    const outDims = [...left.dims.slice(0, -1), right.dims[right.dims.length - 1]];
    const outAxes = [...left.axes.slice(0, -1), right.axes[right.axes.length - 1]];
    return { dtype: "float32", dims: outDims, axes: outAxes };
  }

  if (
    node.moduleId === "OutputContractGate" ||
    node.moduleId === "ScoreBoard" ||
    node.moduleId === "AxisAlignmentRuler" ||
    node.moduleId === "AxisLock" ||
    node.moduleId === "EmbeddingReadyProbe" ||
    node.moduleId === "GhostExpansionPreview" ||
    node.moduleId === "SemanticWarningLens"
  ) {
    const inputPort = node.moduleId === "ScoreBoard" ? "scores" : node.moduleId === "EmbeddingReadyProbe" ? "ids" : node.moduleId === "AxisAlignmentRuler" ? "x" : "x";
    return estimateOutputShape(level, graph, node, inputPort);
  }

  return undefined;
}

function getFocusText(level: LevelSpec, texts: string[]) {
  return level.caseStudy?.visibleInputFocus ?? texts[0];
}

function getCaseHeadline(testCase: TestCase) {
  const raw = testCase.inputs.case;
  if (raw?.dtype !== "raw_text") return undefined;
  if (Array.isArray(raw.data)) return raw.data.find((item): item is string => typeof item === "string");
  return typeof raw.data === "string" ? raw.data : undefined;
}

function parseEditableCaseValue(editable: CaseChip["editable"], rawValue: string): { ok: true; value: unknown } | { ok: false } {
  if (!editable) return { ok: false };
  if (editable.kind === "finite_float32") {
    return { ok: true, value: rawValue };
  }
  return { ok: false };
}

function finiteFloat32InputError(value: unknown) {
  return parseFiniteFloat32Input(value).error;
}

function parseFiniteFloat32Input(value: unknown): { ok: true; value: number; error?: undefined } | { ok: false; error: string } {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return { ok: false, error: "Enter a finite float32 number" };
    if (!Number.isFinite(Math.fround(value))) return { ok: false, error: "Value is outside float32 range" };
    return { ok: true, value };
  }

  const text = String(value ?? "");
  const trimmed = text.trim();
  if (!trimmed) return { ok: false, error: "VALUE cannot be empty" };
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed)) {
    return { ok: false, error: "Enter a finite float32 number" };
  }

  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) return { ok: false, error: "Enter a finite float32 number" };
  if (!Number.isFinite(Math.fround(parsed))) return { ok: false, error: "Value is outside float32 range" };
  return { ok: true, value: parsed };
}

function quoteShort(value: string, maxLength = 32) {
  return `"${shortText(value, maxLength)}"`;
}

function shortText(value: string, maxLength = 32) {
  return value.length > maxLength ? `${value.slice(0, Math.max(0, maxLength - 1))}...` : value;
}

function formatSmallSample(data: RuntimeValue["data"]) {
  if (!Array.isArray(data)) return "";
  return `[${data.slice(0, 3).map((item) => (typeof item === "number" ? Number(item).toFixed(2) : String(item))).join(", ")}${data.length > 3 ? ", ..." : ""}]`;
}

function formatAxisList(axes: string[]) {
  return `[${axes.join(",")}]`;
}

function isMatMulModule(moduleId: string) {
  return moduleId === "MatMulGate" || moduleId === "component.matmul_gate.v1";
}

function normalizePreviewAxis(axis: number, rank: number) {
  const normalized = axis < 0 ? rank + axis : axis;
  return clamp(normalized, 0, Math.max(0, rank - 1));
}

function getNextStepCoach(level: LevelSpec, graph: GraphSpec, runState: RunState, language: GraphLanguage) {
  const firstFail =
    runState.visible?.results.find((result) => result.status !== "pass") ??
    runState.hidden?.results.find((result) => result.status !== "pass");
  if (!firstFail) {
    if (runState.visible?.status === "pass" && runState.hidden?.status !== "pass") return graphText(language, "Current task passed. Prepare a validation variant, then submit validation.");
    return graphText(language, level.onboarding?.firstAction ?? "Click Check Current Task to start the repair loop.");
  }
  if (firstFail.status === "blocked" && firstFail.diagnostic?.errorType === "missing_input") {
    const nodeId = firstFail.firstBadNodeId?.split(".")[0] ?? "node";
    const node = graph.nodes.find((item) => item.id === nodeId);
    const portText = node ? missingPortText(firstFail.message, node.id, language) : firstFail.message;
    return `${portText} ${firstFail.diagnostic.suggestedProbe ?? graphText(language, "Connect the missing input, then check the current task again.")}`;
  }
  return firstFail.diagnostic?.suggestedProbe ?? firstFail.message ?? graphText(language, "Inspect the first red node and compare its input/output shapes.");
}

function missingPortText(message: string, nodeId: string, language: GraphLanguage) {
  const match = message.match(/Required input ([A-Za-z0-9_]+) is not connected/);
  if (!match) return message;
  return language === "zh" ? `${nodeId}.${match[1]} 没有接线。` : `${nodeId}.${match[1]} is unplugged.`;
}

function computeRank(level: LevelSpec, graph: GraphSpec, runState: RunState, language: GraphLanguage) {
  const extraModules = countExtraModules(level.id, graph);
  if (runState.hidden?.status === "pass") {
    if (runState.stats.hiddenRuns === 1 && runState.stats.failedRuns === 0 && runState.stats.hintsUsed === 0 && extraModules === 0) {
      return { rank: "S", message: graphText(language, "Validation passed on the first clean attempt."), extraModules };
    }
    if (runState.stats.failedRuns <= 2 && runState.stats.hintsUsed === 0) {
      return { rank: "A", message: graphText(language, "Validation passed with a low failure count."), extraModules };
    }
    return { rank: "B", message: graphText(language, "Validation passed. The graph handles variants beyond the current task."), extraModules };
  }
  if (runState.visible?.status === "pass") {
    return { rank: "C", message: graphText(language, "Current task passed. Submit validation with a variant to prove generalization."), extraModules };
  }
  return { rank: "-", message: graphText(language, "Check the current task to start the challenge loop."), extraModules };
}

function countExtraModules(levelId: string, graph: GraphSpec) {
  const expected = expectedNodeIdsByLevel[levelId];
  if (!expected) return 0;
  return graph.nodes.filter((node) => !expected.has(node.id)).length;
}

const expectedNodeIdsByLevel: Record<string, Set<string>> = {
  ch0_0_graph_basics: new Set(["input", "shape_gate"]),
  ch0_2_matmul_graph: new Set(["hidden", "weight", "weight_transpose", "matmul", "projected", "reference"]),
  ch0_3_transpose_graph: new Set(["q", "k", "k_transpose", "axis_lock", "qk_matmul", "score_board", "cell_trace", "reference"]),
  ch0_4_broadcast_add: new Set(["projected", "bias", "axis_ruler", "broadcast", "semantic_lens", "ghost", "biased", "cell_trace", "reference"]),
  ch0_4f_mask_add: new Set(["scores", "score_board", "mask", "mask_ruler", "broadcast", "semantic_lens", "ghost", "masked_scores", "cell_trace", "reference"]),
  ch1_1_text_type_gate: new Set(["text", "tokenizer", "token_ids"]),
  ch1_2_split_merge_budget: new Set(["text", "tokenizer", "token_ids"]),
  ch1_3_oov_fallback: new Set(["text", "tokenizer", "token_ids"]),
  ch1_4_padding_mask: new Set(["text", "tokenizer", "token_ids"]),
  ch1_tokenizer_machine: new Set(["text", "tokenizer", "token_ids"])
};

function firstBadNodeIds(runState: RunState) {
  return new Set([...(runState.visible?.results ?? []), ...(runState.hidden?.results ?? [])].map((item) => item.firstBadNodeId?.split(".")[0]).filter(Boolean) as string[]);
}

function defaultTraceSelection(result: RunTestsResult, runKey: TraceRunKey): TraceSelection | undefined {
  const testCase = result.cases.find((item) => item.status !== "pass") ?? result.cases[0];
  if (!testCase) return undefined;
  const firstBadResult = testCase.results.find((item) => item.status !== "pass");
  const errorFrame = testCase.execution.trace.find((frame) => frame.error);
  return {
    runKey,
    caseId: testCase.id,
    step: firstBadResult?.traceFrame ?? errorFrame?.step ?? Math.max(0, testCase.execution.trace.length - 1)
  };
}

function resolveTraceCase(runState: RunState, selection: TraceSelection | undefined) {
  if (selection) {
    return runState[selection.runKey]?.cases.find((testCase) => testCase.id === selection.caseId);
  }
  return (
    runState.hidden?.cases.find((testCase) => testCase.status !== "pass") ??
    runState.visible?.cases.find((testCase) => testCase.status !== "pass") ??
    runState.hidden?.cases[0] ??
    runState.visible?.cases[0]
  );
}

function findLatestNodeFrame(testCase: TestCaseRunResult | undefined, nodeId: string) {
  if (!testCase) return undefined;
  return [...testCase.execution.trace].reverse().find((frame) => frame.nodeId === nodeId);
}

function getNodeTestResults(runState: RunState, nodeId: string) {
  return [...(runState.visible?.results ?? []), ...(runState.hidden?.results ?? [])].filter((result) => {
    const firstBadNodeId = result.firstBadNodeId?.split(".")[0];
    return firstBadNodeId === nodeId || assertionTouchesNode(result.assertion, nodeId);
  });
}

function assertionTouchesNode(assertion: TestAssertion | undefined, nodeId: string) {
  if (!assertion) return false;
  switch (assertion.type) {
    case "dtype":
    case "shape":
    case "axis_semantics":
    case "pieces_non_empty":
    case "pieces_equal":
    case "no_oov":
    case "tokens_include":
    case "eos_preserved":
    case "token_budget":
    case "future_attention_zero":
    case "row_sum":
      return assertion.nodeId.split(".")[0] === nodeId;
    case "allclose":
      return assertion.nodeId.split(".")[0] === nodeId || assertion.referenceNodeId.split(".")[0] === nodeId;
    case "mask_pad":
      return assertion.idsNodeId.split(".")[0] === nodeId || assertion.maskNodeId.split(".")[0] === nodeId;
    case "requires_node":
      return assertion.nodeId === nodeId;
    case "requires_edge_path":
      return assertion.from === nodeId || assertion.through === nodeId || assertion.to === nodeId;
    default:
      return false;
  }
}

function formatShape(shape: { dtype: DType; dims: number[]; axes: string[] }) {
  const axes = shape.axes.map((axis, index) => `${axis}=${shape.dims[index] ?? "?"}`).join(",");
  return `${shape.dtype}[${axes}]`;
}

function formatPolicy(policy: string) {
  if (policy === "char") return "Character";
  if (policy === "word") return "Word";
  if (policy === "subword") return "Subword";
  return policy;
}

function formatUnknown(value: unknown) {
  if (value === undefined) return "n/a";
  if (value === null) return "null";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function getOutputPort(graph: GraphSpec, modules: ModuleDef[], source: PortRef) {
  const node = graph.nodes.find((item) => item.id === source.nodeId);
  const module = node ? modules.find((item) => item.id === node.moduleId) : undefined;
  return module?.outputs.find((port) => port.id === source.portId);
}

function getWirePreviewPath(graph: GraphSpec, modules: ModuleDef[], source: PortRef, pointer: CanvasPoint, anchors?: PortAnchorMap) {
  const sourceNode = graph.nodes.find((node) => node.id === source.nodeId);
  const sourceModule = sourceNode ? modules.find((module) => module.id === sourceNode.moduleId) : undefined;
  if (!sourceNode || !sourceModule) return undefined;
  const from = getPortAnchor(sourceNode, sourceModule, source.portId, "out", anchors);
  return `M ${from.x} ${from.y} C ${from.x + 88} ${from.y}, ${pointer.x - 88} ${pointer.y}, ${pointer.x} ${pointer.y}`;
}

function layoutGraph(graph: GraphSpec): GraphSpec {
  const layers = new Map(graph.nodes.map((node) => [node.id, 0]));
  for (let pass = 0; pass < graph.nodes.length; pass += 1) {
    let changed = false;
    for (const edge of graph.edges) {
      const fromLayer = layers.get(edge.from.nodeId);
      const toLayer = layers.get(edge.to.nodeId);
      if (fromLayer === undefined || toLayer === undefined) continue;
      const nextLayer = Math.max(toLayer, fromLayer + 1);
      if (nextLayer !== toLayer) {
        layers.set(edge.to.nodeId, nextLayer);
        changed = true;
      }
    }
    if (!changed) break;
  }

  const maxLayer = Math.max(0, ...layers.values());
  const xGap = Math.min(300, Math.max(230, (graphWorldWidth - graphNodeWidth - 180) / Math.max(maxLayer + 1, 3)));
  const yGap = 150;
  const rowCounts = new Map<number, number>();

  return {
    ...graph,
    nodes: graph.nodes.map((node) => {
      const layer = layers.get(node.id) ?? 0;
      const row = rowCounts.get(layer) ?? 0;
      rowCounts.set(layer, row + 1);
      const position = { x: 70 + layer * xGap, y: 110 + row * yGap };
      return {
        ...node,
        position: clampNodePosition(position.x, position.y)
      };
    })
  };
}

function cloneParams(params: Record<string, unknown>) {
  return JSON.parse(JSON.stringify(params)) as Record<string, unknown>;
}

function getLevelNodeTemplate(levelId: string, graph: GraphSpec, moduleId: string): { id?: string; params?: Record<string, unknown> } {
  if (levelId === "ch0_0_graph_basics") {
    const taken = new Set(graph.nodes.map((node) => node.id));
    if (moduleId === "InputTensor" && !taken.has("input")) return { id: "input", params: { inputKey: "input" } };
    if (moduleId === "OutputContractGate" && !taken.has("shape_gate")) return { id: "shape_gate", params: { expectedAxes: ["B", "T", "C"] } };
  }

  if (levelId === "ch0_2_matmul_graph") {
    const taken = new Set(graph.nodes.map((node) => node.id));
    if (moduleId === "InputTensor" && !taken.has("hidden")) return { id: "hidden", params: { inputKey: "hidden" } };
    if (moduleId === "WeightPlate" && !taken.has("weight")) return { id: "weight", params: { inputKey: "weight", orientation: "O,C" } };
    if (moduleId === "TransposeSwitch" && !taken.has("weight_transpose")) return { id: "weight_transpose", params: { axisA: 0, axisB: 1 } };
    if (moduleId === "MatMulGate" && !taken.has("matmul")) return { id: "matmul" };
    if (moduleId === "OutputContractGate" && !taken.has("projected")) return { id: "projected", params: { expectedAxes: ["B", "T", "O"] } };
    if (moduleId === "ReferenceChecker" && !taken.has("reference")) return { id: "reference", params: { referenceKey: "reference" } };
  }

  if (levelId === "ch0_3_transpose_graph") {
    const taken = new Set(graph.nodes.map((node) => node.id));
    if (moduleId === "InputTensor" && !taken.has("q")) return { id: "q", params: { inputKey: "q" } };
    if (moduleId === "InputTensor" && !taken.has("k")) return { id: "k", params: { inputKey: "k" } };
    if (moduleId === "TransposeSwitch" && !taken.has("k_transpose")) return { id: "k_transpose", params: { axisA: -2, axisB: -1 } };
    if (moduleId === "AxisLock" && !taken.has("axis_lock")) return { id: "axis_lock", params: { expectedPrefixAxes: ["B", "H"] } };
    if (moduleId === "MatMulGate" && !taken.has("qk_matmul")) return { id: "qk_matmul" };
    if (moduleId === "ScoreBoard" && !taken.has("score_board")) return { id: "score_board", params: { expectedAxes: ["B", "H", "T", "T"] } };
    if (moduleId === "CellTrace" && !taken.has("cell_trace")) return { id: "cell_trace", params: { b: 0, h: 0, i: 0, j: 1 } };
    if (moduleId === "ReferenceChecker" && !taken.has("reference")) return { id: "reference", params: { referenceKey: "reference" } };
  }

  if (levelId === "ch0_4_broadcast_add") {
    const taken = new Set(graph.nodes.map((node) => node.id));
    if (moduleId === "InputTensor" && !taken.has("projected")) return { id: "projected", params: { inputKey: "projected" } };
    if (moduleId === "InputTensor" && !taken.has("bias")) return { id: "bias", params: { inputKey: "bias", shape: [4], axes: ["O"] } };
    if (moduleId === "AxisAlignmentRuler" && !taken.has("axis_ruler")) return { id: "axis_ruler", params: { expectedAxes: ["O"] } };
    if (moduleId === "BroadcastRail" && !taken.has("broadcast")) return { id: "broadcast", params: { alignAxes: ["O"] } };
    if (moduleId === "SemanticWarningLens" && !taken.has("semantic_lens")) return { id: "semantic_lens" };
    if (moduleId === "GhostExpansionPreview" && !taken.has("ghost")) return { id: "ghost" };
    if (moduleId === "AddGate" && !taken.has("biased")) return { id: "biased" };
    if (moduleId === "CellTrace" && !taken.has("cell_trace")) return { id: "cell_trace", params: { b: 0, t: 0, o: 1 } };
    if (moduleId === "ReferenceChecker" && !taken.has("reference")) return { id: "reference", params: { referenceKey: "reference" } };
  }

  if (levelId === "ch0_4f_mask_add") {
    const taken = new Set(graph.nodes.map((node) => node.id));
    if (moduleId === "InputTensor" && !taken.has("scores")) return { id: "scores", params: { inputKey: "scores" } };
    if (moduleId === "ScoreBoard" && !taken.has("score_board")) return { id: "score_board", params: { expectedAxes: ["B", "H", "T", "T"] } };
    if (moduleId === "CausalMask" && !taken.has("mask")) return { id: "mask", params: { maskOrientation: "query_key", maskedValue: -10000 } };
    if (moduleId === "AxisAlignmentRuler" && !taken.has("mask_ruler")) return { id: "mask_ruler", params: { expectedAxes: ["T", "T"] } };
    if (moduleId === "BroadcastRail" && !taken.has("broadcast")) return { id: "broadcast", params: { alignAxes: ["T", "T"] } };
    if (moduleId === "SemanticWarningLens" && !taken.has("semantic_lens")) return { id: "semantic_lens" };
    if (moduleId === "GhostExpansionPreview" && !taken.has("ghost")) return { id: "ghost" };
    if (moduleId === "AddGate" && !taken.has("masked_scores")) return { id: "masked_scores" };
    if (moduleId === "CellTrace" && !taken.has("cell_trace")) return { id: "cell_trace", params: { b: 0, h: 0, i: 0, j: 1 } };
    if (moduleId === "ReferenceChecker" && !taken.has("reference")) return { id: "reference", params: { referenceKey: "reference" } };
  }

  if (levelId.startsWith("mvp01_")) {
    const taken = new Set(graph.nodes.map((node) => node.id));
    if (levelId.startsWith("mvp01_ch") && moduleId.startsWith("component.") && !taken.has("component")) return { id: "component" };
    if (moduleId === "ReferenceChecker" && !taken.has("reference")) return { id: "reference", params: { referenceKey: "reference" } };

    if (levelId === "mvp01_1_scalar_cell") {
      if (moduleId === "Float32Literal" && !taken.has("scalar_source")) return { id: "scalar_source", params: { value: 0.5 } };
      if (moduleId === "OutputContractGate" && !taken.has("scalar_out")) return { id: "scalar_out", params: { expectedAxes: [] } };
    }

    if (levelId === "mvp01_2_vector_rail") {
      const scalarTemplate = [
        { id: "scalar_c0", value: 0.5 },
        { id: "scalar_c1", value: -1 },
        { id: "scalar_c2", value: 2 }
      ].find((item) => !taken.has(item.id));
      if (moduleId === "component.scalar_cell.v1" && scalarTemplate) {
        return { id: scalarTemplate.id, params: { value: scalarTemplate.value } };
      }
      if (moduleId === "VectorRail" && !taken.has("vector")) return { id: "vector" };
      if (moduleId === "OutputContractGate" && !taken.has("vector_out")) return { id: "vector_out", params: { expectedAxes: ["C"] } };
    }

    if (levelId === "mvp01_3_matrix_struct") {
      if (moduleId === "InputTensor" && !taken.has("col0")) return { id: "col0", params: { inputKey: "col0", shape: [3], axes: ["C"] } };
      if (moduleId === "InputTensor" && !taken.has("col1")) return { id: "col1", params: { inputKey: "col1", shape: [3], axes: ["C"] } };
      if (moduleId === "MatrixStruct" && !taken.has("matrix")) return { id: "matrix" };
      if (moduleId === "OutputContractGate" && !taken.has("matrix_out")) return { id: "matrix_out", params: { expectedAxes: ["C", "O"] } };
    }

    if (levelId === "mvp01_4_tensor_box") {
      if (moduleId === "InputTensor" && !taken.has("t0")) return { id: "t0", params: { inputKey: "t0", shape: [3], axes: ["C"] } };
      if (moduleId === "InputTensor" && !taken.has("t1")) return { id: "t1", params: { inputKey: "t1", shape: [3], axes: ["C"] } };
      if (moduleId === "TensorBox" && !taken.has("tensor")) return { id: "tensor" };
      if (moduleId === "OutputContractGate" && !taken.has("tensor_out")) return { id: "tensor_out", params: { expectedAxes: ["B", "T", "C"] } };
    }

    if (levelId === "mvp01_ch1_01_splitter") {
      if (moduleId === "TextInput" && !taken.has("text")) return { id: "text", params: { inputKey: "texts" } };
      if (moduleId === "BoundarySplitter" && !taken.has("splitter")) return { id: "splitter", params: { policy: "word", preservePunctuation: true, expectedT: 4 } };
      if (moduleId === "PieceBuffer" && !taken.has("piece_buffer")) return { id: "piece_buffer", params: { maxPieces: 12 } };
      if (moduleId === "TypeContractGate" && !taken.has("pieces_out")) return { id: "pieces_out", params: { expectedDType: "string_piece", expectedDims: [4], expectedAxes: ["T"] } };
    }

    if (levelId === "mvp01_5_matmul_gate") {
      if (moduleId === "InputTensor" && !taken.has("hidden")) return { id: "hidden", params: { inputKey: "hidden", shape: [1, 2, 3], axes: ["B", "T", "C"] } };
      if (moduleId === "WeightPlate" && !taken.has("weight")) return { id: "weight", params: { inputKey: "weight", shape: [3, 2], axes: ["C", "O"], orientation: "C,O" } };
      if (moduleId === "MatMulGate" && !taken.has("matmul")) return { id: "matmul" };
      if (moduleId === "OutputContractGate" && !taken.has("matmul_out")) return { id: "matmul_out", params: { expectedAxes: ["B", "T", "O"] } };
    }

    if (levelId === "mvp01_6_linear") {
      if (moduleId === "InputTensor" && !taken.has("hidden")) return { id: "hidden", params: { inputKey: "hidden", shape: [1, 2, 3], axes: ["B", "T", "C"] } };
      if (moduleId === "InputTensor" && !taken.has("bias")) return { id: "bias", params: { inputKey: "bias", shape: [2], axes: ["O"] } };
      if (moduleId === "WeightPlate" && !taken.has("weight")) return { id: "weight", params: { inputKey: "weight", shape: [3, 2], axes: ["C", "O"], orientation: "C,O" } };
      if ((moduleId === "MatMulGate" || moduleId === "component.matmul_gate.v1") && !taken.has("matmul")) return { id: "matmul" };
      if (moduleId === "BroadcastRail" && !taken.has("bias_broadcast")) return { id: "bias_broadcast", params: { alignAxes: ["O"] } };
      if (moduleId === "AddGate" && !taken.has("linear_add")) return { id: "linear_add" };
      if (moduleId === "OutputContractGate" && !taken.has("linear_out")) return { id: "linear_out", params: { expectedAxes: ["B", "T", "O"] } };
    }

    if (levelId === "mvp01_ch4_03_qk_score") {
      if (moduleId === "InputTensor" && !taken.has("q")) return { id: "q", params: { inputKey: "q", shape: [1, 1, 3, 2], axes: ["B", "H", "T", "D"] } };
      if (moduleId === "InputTensor" && !taken.has("k")) return { id: "k", params: { inputKey: "k", shape: [1, 1, 3, 2], axes: ["B", "H", "T", "D"] } };
      if (moduleId === "TransposeSwitch" && !taken.has("k_transpose")) return { id: "k_transpose", params: { axisA: -2, axisB: -1 } };
      if ((moduleId === "MatMulGate" || moduleId === "component.matmul_gate.v1") && !taken.has("qk_matmul")) return { id: "qk_matmul" };
      if (moduleId === "ScoreBoard" && !taken.has("score_board")) return { id: "score_board", params: { expectedAxes: ["B", "H", "T", "T"] } };
      if (moduleId === "CellTrace" && !taken.has("cell_trace")) return { id: "cell_trace", params: { b: 0, h: 0, i: 0, j: 1 } };
      if (moduleId === "ReferenceChecker" && !taken.has("reference")) return { id: "reference", params: { referenceKey: "reference" } };
    }
  }

  if (levelId.startsWith("ch1_")) {
    const taken = new Set(graph.nodes.map((node) => node.id));
    if (moduleId === "TextInput" && !taken.has("text")) return { id: "text", params: { inputKey: "texts" } };
    if (moduleId === "TokenizerSocket" && !taken.has("tokenizer")) return { id: "tokenizer", params: getTokenizerTemplateParams(levelId) };
    if (moduleId === "EmbeddingReadyProbe" && !taken.has("token_ids")) return { id: "token_ids" };
  }

  return {};
}

function getTokenizerTemplateParams(levelId: string) {
  const base = {
    policy: "subword",
    applyMerges: true,
    fallback: "unk",
    preservePunctuation: true,
    addBos: true,
    addEos: true,
    maxLength: 8,
    padToLength: 8,
    padSide: "right",
    maskPolicy: "pad-aware"
  };
  if (levelId === "ch1_1_text_type_gate") return { ...base, policy: "word", addBos: false, addEos: false, maxLength: 4, padToLength: 4 };
  if (levelId === "ch1_2_split_merge_budget") return { ...base, policy: "char", applyMerges: false, addBos: false, addEos: false };
  if (levelId === "ch1_3_oov_fallback") return { ...base, fallback: "none", addBos: false, addEos: false };
  if (levelId === "ch1_4_padding_mask") return { ...base, maskPolicy: "all-ones" };
  return base;
}

function uniqueNodeId(graph: GraphSpec, moduleId: string) {
  const base = moduleId.replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase();
  const taken = new Set(graph.nodes.map((node) => node.id));
  let index = 1;
  while (taken.has(`${base}_${index}`)) index += 1;
  return `${base}_${index}`;
}

function uniqueEdgeId(graph: GraphSpec, from: PortRef, to: PortRef) {
  const base = `e_${from.nodeId}_${from.portId}_${to.nodeId}_${to.portId}`.replace(/[^a-zA-Z0-9_]/g, "_");
  const taken = new Set(graph.edges.map((edge) => edge.id));
  if (!taken.has(base)) return base;
  let index = 2;
  while (taken.has(`${base}_${index}`)) index += 1;
  return `${base}_${index}`;
}

function portsCompatible(from: PortDef, to: PortDef) {
  if (from.direction !== "out" || to.direction !== "in") return false;
  if (!from.emits || !to.accepts?.length) return true;
  return to.accepts.includes(from.emits);
}

function clampNodePosition(x: number, y: number) {
  const maxX = Math.max(graphWorldMinX, graphWorldMaxX - graphNodeWidth);
  const maxY = Math.max(graphWorldMinY, graphWorldMaxY - graphNodeMinHeight);
  return {
    x: Math.min(Math.max(graphWorldMinX, x), maxX),
    y: Math.min(Math.max(graphWorldMinY, y), maxY)
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function portAnchorKey(nodeId: string, portId: string, direction: "in" | "out") {
  return `${nodeId}:${portId}:${direction}`;
}

function getPortAnchor(node: GraphNode, module: ModuleDef, portId: string, direction: "in" | "out", anchors?: PortAnchorMap) {
  const measured = anchors?.[portAnchorKey(node.id, portId, direction)];
  if (measured) return measured;
  const ports = direction === "in" ? module.inputs : module.outputs;
  const portIndex = Math.max(0, ports.findIndex((port) => port.id === portId));
  const rowY = node.position.y + 76 + portIndex * 24;
  return {
    x: node.position.x + (direction === "out" ? graphNodeWidth : 0),
    y: rowY
  };
}

function withWeightTranspose(graph: GraphSpec): GraphSpec {
  if (graph.nodes.some((node) => node.id === "weight_transpose")) return graph;
  return {
    ...graph,
    nodes: [
      ...graph.nodes,
      { id: "weight_transpose", moduleId: "TransposeSwitch", params: { axisA: 0, axisB: 1 }, position: { x: 240, y: 260 } }
    ],
    edges: [
      ...graph.edges.filter((edge) => edge.id !== "e_weight_matmul"),
      { id: "e_weight_transpose", from: { nodeId: "weight", portId: "out" }, to: { nodeId: "weight_transpose", portId: "x" } },
      { id: "e_transpose_matmul", from: { nodeId: "weight_transpose", portId: "out" }, to: { nodeId: "matmul", portId: "right" } }
    ]
  };
}

function withKTranspose(graph: GraphSpec): GraphSpec {
  if (graph.nodes.some((node) => node.id === "k_transpose")) return graph;
  const needsAxisLock = !graph.nodes.some((node) => node.id === "axis_lock");
  return {
    ...graph,
    nodes: [
      ...graph.nodes,
      { id: "k_transpose", moduleId: "TransposeSwitch", params: { axisA: -2, axisB: -1 }, position: { x: 240, y: 300 } },
      ...(needsAxisLock
        ? [{ id: "axis_lock", moduleId: "AxisLock", params: { expectedPrefixAxes: ["B", "H"] }, position: { x: 470, y: 300 } }]
        : [])
    ],
    edges: [
      ...graph.edges.filter(
        (edge) =>
          !(edge.to.nodeId === "qk_matmul" && edge.to.portId === "right") &&
          !["e_k_transpose", "e_transpose_axis_lock", "e_axis_lock_matmul"].includes(edge.id)
      ),
      { id: "e_k_transpose", from: { nodeId: "k", portId: "out" }, to: { nodeId: "k_transpose", portId: "x" } },
      { id: "e_transpose_axis_lock", from: { nodeId: "k_transpose", portId: "out" }, to: { nodeId: "axis_lock", portId: "x" } },
      { id: "e_axis_lock_matmul", from: { nodeId: "axis_lock", portId: "out" }, to: { nodeId: "qk_matmul", portId: "right" } }
    ]
  };
}
