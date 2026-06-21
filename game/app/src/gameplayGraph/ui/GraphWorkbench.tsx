import {
  createContext,
  useEffect,
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
import type { DType, GraphEdge, GraphNode, GraphSpec, LevelSpec, ModuleDef, PortDef, PortRef, TestAssertion, TestResult, TraceFrame } from "../types";
import { runTests, type RunTestsResult, type TestCaseRunResult } from "../runtime/testRunner";
import { graphStatusText, graphText, type GraphLanguage } from "../i18n";
import { createTokenizerPreview, generateCaseCodeLines, getCaseTexts, type TokenizerCasePreview } from "../codegen";

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

const moduleDragMime = "application/x-llm-complete-graph-module";
const graphNodeWidth = 190;
const graphNodeMinHeight = 118;
const graphWorldWidth = 2400;
const graphWorldHeight = 1600;
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

export function GraphWorkbench({ language = "en" }: { language?: GraphLanguage }) {
  const registry = useMemo(() => createGameplayRegistry(), []);
  const modules = useMemo(() => registry.list(), [registry]);
  const [selectedLevelId, setSelectedLevelId] = useState(graphLevels[0].id);
  const selectedLevel = graphLevels.find((level) => level.id === selectedLevelId) ?? graphLevels[0];
  const [graphs, setGraphs] = useState<Record<string, GraphSpec>>(() => initialGraphsByLevel());
  const [selection, setSelection] = useState<GraphSelection | undefined>(() => {
    const firstNodeId = graphLevels[0].initialGraph.nodes[0]?.id;
    return firstNodeId ? { type: "node", id: firstNodeId } : undefined;
  });
  const [wireSource, setWireSource] = useState<WireSource>();
  const [wirePointer, setWirePointer] = useState<CanvasPoint>();
  const [canvasNotice, setCanvasNotice] = useState("ready");
  const [dragOverCanvas, setDragOverCanvas] = useState(false);
  const [viewport, setViewport] = useState<CanvasViewport>({ x: 0, y: 0, scale: 1 });
  const [runs, setRuns] = useState<Record<string, RunState>>({});
  const [traceSelection, setTraceSelection] = useState<TraceSelection>();
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("summary");
  const [missionOpen, setMissionOpen] = useState(true);
  const [layout, setLayout] = useState<WorkbenchLayout>(defaultWorkbenchLayout);
  const workbenchRef = useRef<HTMLElement>(null);
  const stageRef = useRef<HTMLElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const nodeDragRef = useRef<NodeDragState>();
  const panDragRef = useRef<PanDragState>();
  const wireSourceRef = useRef<WireSource>();
  const suppressNextPortClickRef = useRef(false);
  const graph = graphs[selectedLevel.id] ?? selectedLevel.initialGraph;
  const runState = runs[selectedLevel.id] ?? createEmptyRunState();
  const selectedNode = selection?.type === "node" ? graph.nodes.find((node) => node.id === selection.id) : undefined;
  const selectedEdge = selection?.type === "edge" ? graph.edges.find((edge) => edge.id === selection.id) : undefined;
  const selectedModule = selectedNode ? registry.maybeGet(selectedNode.moduleId) : undefined;
  const wireSourcePort = wireSource ? getOutputPort(graph, modules, wireSource) : undefined;
  const activeTraceCase = resolveTraceCase(runState, traceSelection);
  const activeTraceFrame = activeTraceCase && traceSelection?.caseId === activeTraceCase.id ? activeTraceCase.execution.trace[traceSelection.step] : undefined;
  const hiddenLocked = runState.visible?.status !== "pass";
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
    setMissionOpen(true);
  }, [selectedLevel.id]);

  function selectLevel(level: LevelSpec) {
    const levelGraph = graphs[level.id] ?? level.initialGraph;
    setSelectedLevelId(level.id);
    setSelection(levelGraph.nodes[0] ? { type: "node", id: levelGraph.nodes[0].id } : undefined);
    clearWireSource();
    setWirePointer(undefined);
    setTraceSelection(undefined);
    setInspectorTab("summary");
    setCanvasNotice("ready");
  }

  function resetLevel() {
    setGraphs((current) => ({ ...current, [selectedLevel.id]: selectedLevel.initialGraph }));
    setRuns((current) => ({ ...current, [selectedLevel.id]: createEmptyRunState() }));
    setSelection(selectedLevel.initialGraph.nodes[0] ? { type: "node", id: selectedLevel.initialGraph.nodes[0].id } : undefined);
    clearWireSource();
    setWirePointer(undefined);
    setTraceSelection(undefined);
    setInspectorTab("summary");
    setViewport({ x: 0, y: 0, scale: 1 });
    setCanvasNotice("reset");
  }

  function runVisible() {
    const result = runTests(graph, registry, selectedLevel.visibleTests);
    setRuns((current) => {
      const previous = current[selectedLevel.id] ?? createEmptyRunState();
      return {
        ...current,
        [selectedLevel.id]: {
          ...previous,
          visible: result,
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
    const result = runTests(graph, registry, selectedLevel.hiddenTests);
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
    focusFirstBadNode(result);
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
    if ((event.target as HTMLElement).closest("[data-graph-port='true']")) return;
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
    if (target.closest(".graphNode, .graphCanvasToolbar, .graphCanvasHud, .graphCanvasModulePalette, .graphCasePanel, button, input, select, textarea")) return;
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
      setSelection(graph.nodes[0] ? { type: "node", id: graph.nodes[0].id } : undefined);
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

  function addModuleAtCenter(moduleId: string) {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) {
      addModuleNode(moduleId);
      return;
    }
    const center = viewportToWorld(rect.width / 2, rect.height / 2);
    addModuleNode(moduleId, center.x - graphNodeWidth / 2, center.y - graphNodeMinHeight / 2);
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

  const wirePreview = wireSource && wirePointer ? getWirePreviewPath(graph, modules, wireSource, wirePointer) : undefined;
  const t = (text: string) => graphText(language, text);

  return (
    <GraphLanguageContext.Provider value={language}>
    <section ref={workbenchRef} className="graphWorkbench" style={workbenchStyle}>
      <aside className="graphSidebar panel">
        <div className="panelHeader">
          <Wrench size={18} />
          <h2>{t("Graph Levels")}</h2>
        </div>
        <div className="graphLevelList">
          {graphLevels.map((level) => (
            <button key={level.id} className={`graphLevelButton ${level.id === selectedLevel.id ? "active" : ""}`} onClick={() => selectLevel(level)}>
              <span>
                <b>{t(level.title)}</b>
                <small>{t(level.goal)}</small>
              </span>
            </button>
          ))}
        </div>

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
            <p className="eyebrow">{t("Graph Challenge")}</p>
            <h2>{t(selectedLevel.title)}</h2>
            <small>{t(selectedLevel.goal)}</small>
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
            <button className="runButton" onClick={runVisible}>
              <Play size={15} />
              {t("Run Visible")}
            </button>
            <button className="runButton" disabled={hiddenLocked} onClick={runHidden}>
              <Play size={15} />
              {t("Run Hidden")}
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
            runState={runState}
            onRunVisible={runVisible}
            onShowHint={showNextHint}
            onClose={() => setMissionOpen(false)}
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
          <GraphCasePanel key={selectedLevel.id} level={selectedLevel} graph={graph} modules={modules} />
          <GraphCanvasModulePalette
            level={selectedLevel}
            registry={registry}
            onAddModule={addModuleAtCenter}
          />
          <div
            className="graphWorld"
            style={{
              width: graphWorldWidth,
              height: graphWorldHeight,
              transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.scale})`
            }}
          >
            <TargetGhostGraph targetGraph={selectedLevel.targetGraph} currentGraph={graph} modules={modules} />
            <svg className="graphEdgeLayer" width={graphWorldWidth} height={graphWorldHeight} viewBox={`0 0 ${graphWorldWidth} ${graphWorldHeight}`} role="img" aria-label="graph edges">
              {wirePreview ? <path className="graphWirePreview" d={wirePreview} /> : null}
              {graph.edges.map((edge) => {
                const fromNode = graph.nodes.find((node) => node.id === edge.from.nodeId);
                const toNode = graph.nodes.find((node) => node.id === edge.to.nodeId);
                if (!fromNode || !toNode) return null;
                const fromModule = registry.get(fromNode.moduleId);
                const toModule = registry.get(toNode.moduleId);
                const from = getPortAnchor(fromNode, fromModule, edge.from.portId, "out");
                const to = getPortAnchor(toNode, toModule, edge.to.portId, "in");
                const selected = selectedEdge?.id === edge.id;
                return (
                  <g key={edge.id} className={`graphEdge ${selected ? "selected" : ""}`} onClick={() => setSelection({ type: "edge", id: edge.id })}>
                    <path d={`M ${from.x} ${from.y} C ${from.x + 88} ${from.y}, ${to.x - 88} ${to.y}, ${to.x} ${to.y}`} />
                    <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 8}>
                      {`${edge.from.portId} -> ${edge.to.portId}`}
                    </text>
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

        <GraphCodePanel level={selectedLevel} />
        <GraphRankPanel level={selectedLevel} graph={graph} runState={runState} />
        <GraphRunPanel title="Visible Tests" result={runState.visible} onLocateNode={focusNode} onShowNextStep={showNextHint} />
        <GraphRunPanel title="Hidden Tests" result={runState.hidden} locked={hiddenLocked} onLocateNode={focusNode} onShowNextStep={showNextHint} />
      </aside>
    </section>
    </GraphLanguageContext.Provider>
  );
}

function GraphMissionModal({
  level,
  graph,
  runState,
  onRunVisible,
  onShowHint,
  onClose
}: {
  level: LevelSpec;
  graph: GraphSpec;
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
          <small>{t(caseStudy?.narrative ?? onboarding?.startingProblem ?? "Run Visible to reveal the first failing node, then repair the graph.")}</small>
        </div>

        <div className="graphMissionSteps">
          <div>
            <b>{t("First action")}</b>
            <small>{t(caseStudy?.playerQuestion ?? onboarding?.firstAction ?? "Click Run Visible.")}</small>
          </div>
          <div>
            <b>{t("Win condition")}</b>
            <small>{t(caseStudy?.successObservation ?? onboarding?.winCondition ?? "Visible and hidden tests pass.")}</small>
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
              {t("Run Visible")}
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
  currentGraph,
  modules
}: {
  targetGraph?: GraphSpec;
  currentGraph: GraphSpec;
  modules: ModuleDef[];
}) {
  if (!targetGraph) return null;
  return (
    <svg className="targetGhostLayer" width={graphWorldWidth} height={graphWorldHeight} viewBox={`0 0 ${graphWorldWidth} ${graphWorldHeight}`} aria-hidden="true">
      {targetGraph.edges.map((edge) => {
        const fromNode = targetGraph.nodes.find((node) => node.id === edge.from.nodeId);
        const toNode = targetGraph.nodes.find((node) => node.id === edge.to.nodeId);
        if (!fromNode || !toNode) return null;
        const fromModule = modules.find((module) => module.id === fromNode.moduleId);
        const toModule = modules.find((module) => module.id === toNode.moduleId);
        if (!fromModule || !toModule) return null;
        const from = getPortAnchor(fromNode, fromModule, edge.from.portId, "out");
        const to = getPortAnchor(toNode, toModule, edge.to.portId, "in");
        const matched = graphHasEdge(currentGraph, edge.from, edge.to);
        return (
          <g key={edge.id} className={`targetGhostEdge ${matched ? "matched" : ""}`}>
            <path d={`M ${from.x} ${from.y} C ${from.x + 88} ${from.y}, ${to.x - 88} ${to.y}, ${to.x} ${to.y}`} />
          </g>
        );
      })}
      {targetGraph.nodes.map((node) => {
        const matched = currentGraph.nodes.some((item) => item.id === node.id && item.moduleId === node.moduleId);
        return (
          <g key={node.id} className={`targetGhostNode ${matched ? "matched" : ""}`} transform={`translate(${node.position.x} ${node.position.y})`}>
            <rect width={graphNodeWidth} height={graphNodeMinHeight} rx={8} />
            <text x={14} y={24}>{node.id}</text>
          </g>
        );
      })}
    </svg>
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
            <span />
            <b>{t(port.label)}</b>
          </button>
        );
      })}
    </div>
  );
}

function GraphCanvasModulePalette({
  level,
  registry,
  onAddModule
}: {
  level: LevelSpec;
  registry: ReturnType<typeof createGameplayRegistry>;
  onAddModule: (moduleId: string) => void;
}) {
  const { t } = useGraphT();
  return (
    <section className="graphCanvasModulePalette">
      <div className="graphCanvasModulePaletteHeader">
        <b>{t("Available Modules")}</b>
        <code>{level.modulePalette.length}</code>
      </div>
      <div className="graphCanvasModuleList">
        {level.modulePalette.map((moduleId) => {
          const module = registry.get(moduleId);
          return (
            <button
              key={module.id}
              className="graphModuleItem canvas"
              draggable
              title={t(module.summary)}
              onDoubleClick={() => onAddModule(module.id)}
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

function GraphCasePanel({
  level,
  graph,
  modules
}: {
  level: LevelSpec;
  graph: GraphSpec;
  modules: ModuleDef[];
}) {
  const { t } = useGraphT();
  const [collapsed, setCollapsed] = useState(false);
  const caseStudy = level.caseStudy;
  const visibleCase = level.visibleTests[0];
  if (!caseStudy) return null;

  return (
    <section className={`graphCasePanel ${collapsed ? "collapsed" : ""}`}>
      <div className="graphCasePanelHeader">
        <div>
          <p className="eyebrow">{t("Current Case")}</p>
          <h3>{t(caseStudy.title)}</h3>
        </div>
        <button className="iconButton" type="button" title={t(collapsed ? "Open case" : "Collapse case")} onClick={() => setCollapsed((current) => !current)}>
          {collapsed ? <BookOpenText size={15} /> : <X size={15} />}
        </button>
      </div>
      {!collapsed ? (
        <>
          <p className="graphCaseNarrative">{t(caseStudy.narrative)}</p>
          <div className="graphCaseDataGrid">
            {caseStudy.dataPanels.map((panel) => {
              if (panel.type === "text_batch") {
                const texts = getCaseTexts(visibleCase, panel.inputKey);
                const focusText = panel.focusText ?? caseStudy.visibleInputFocus;
                return (
                  <section key={`${panel.type}:${panel.inputKey}`} className="graphCaseDataCard">
                    <b>{t(panel.title)}</b>
                    <div className="graphCaseTextList">
                      {texts.map((text) => (
                        <code key={text} className={text === focusText ? "focus" : ""}>
                          {text}
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

              return (
                <section key={`${panel.type}:${panel.title}`} className="graphCaseDataCard">
                  <b>{t(panel.title)}</b>
                  <p className="graphTraceEmpty">{t("Case preview is coming in the next slice.")}</p>
                </section>
              );
            })}
          </div>
          <div className="graphCaseQuestion">
            <b>{t("Player question")}</b>
            <small>{t(caseStudy.playerQuestion)}</small>
          </div>
        </>
      ) : null}
    </section>
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

function GraphCodePanel({ level }: { level: LevelSpec }) {
  const { t } = useGraphT();
  const [collapsed, setCollapsed] = useState(false);
  const lines = useMemo(() => generateCaseCodeLines(level, level.visibleTests[0]), [level]);

  return (
    <section className={`graphCodePanel ${collapsed ? "collapsed" : ""}`}>
      <div className="graphRunPanelHeader">
        <h3>{t("Case Code")}</h3>
        <button className="iconButton" type="button" title={t(collapsed ? "Open code" : "Collapse code")} onClick={() => setCollapsed((current) => !current)}>
          {collapsed ? <BookOpenText size={14} /> : <X size={14} />}
        </button>
      </div>
      {!collapsed ? <pre className="graphCodeBlock">{lines.map((line) => line.text).join("\n")}</pre> : null}
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
    { key: "visible", label: "Visible", result: visible },
    { key: "hidden", label: "Hidden", result: hidden }
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
        <p className="graphTraceEmpty">{t("Run visible tests to capture a trace.")}</p>
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
    return (
      <label className="graphParamRow">
        <span>{t(meta.label)}</span>
        <input type="number" value={Number(value ?? 0)} step={1} onChange={(event) => onChange(nodeId, paramKey, Number(event.currentTarget.value))} />
        {meta.help ? <small>{t(meta.help)}</small> : null}
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
      {locked ? <p>{t("Visible tests must pass before hidden tests run.")}</p> : null}
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

function initialGraphsByLevel() {
  return Object.fromEntries(graphLevels.map((level) => [level.id, level.initialGraph]));
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

function getNextStepCoach(level: LevelSpec, graph: GraphSpec, runState: RunState, language: GraphLanguage) {
  const firstFail =
    runState.visible?.results.find((result) => result.status !== "pass") ??
    runState.hidden?.results.find((result) => result.status !== "pass");
  if (!firstFail) {
    if (runState.visible?.status === "pass" && runState.hidden?.status !== "pass") return graphText(language, "Visible is green. Run Hidden to check generalization.");
    return graphText(language, level.onboarding?.firstAction ?? "Click Run Visible to start the repair loop.");
  }
  if (firstFail.status === "blocked" && firstFail.diagnostic?.errorType === "missing_input") {
    const nodeId = firstFail.firstBadNodeId?.split(".")[0] ?? "node";
    const node = graph.nodes.find((item) => item.id === nodeId);
    const portText = node ? missingPortText(firstFail.message, node.id, language) : firstFail.message;
    return `${portText} ${firstFail.diagnostic.suggestedProbe ?? graphText(language, "Connect the missing input, then run Visible again.")}`;
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
      return { rank: "S", message: graphText(language, "Hidden tests passed on the first clean attempt."), extraModules };
    }
    if (runState.stats.failedRuns <= 2 && runState.stats.hintsUsed === 0) {
      return { rank: "A", message: graphText(language, "Hidden tests passed with a low failure count."), extraModules };
    }
    return { rank: "B", message: graphText(language, "Hidden tests passed. The graph generalizes beyond visible inputs."), extraModules };
  }
  if (runState.visible?.status === "pass") {
    return { rank: "C", message: graphText(language, "Visible tests passed. Run hidden mutation tests to prove generalization."), extraModules };
  }
  return { rank: "-", message: graphText(language, "Run visible tests to start the challenge loop."), extraModules };
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

function getWirePreviewPath(graph: GraphSpec, modules: ModuleDef[], source: PortRef, pointer: CanvasPoint) {
  const sourceNode = graph.nodes.find((node) => node.id === source.nodeId);
  const sourceModule = sourceNode ? modules.find((module) => module.id === sourceNode.moduleId) : undefined;
  if (!sourceNode || !sourceModule) return undefined;
  const from = getPortAnchor(sourceNode, sourceModule, source.portId, "out");
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
  const maxX = Math.max(12, graphWorldWidth - graphNodeWidth - 12);
  const maxY = Math.max(12, graphWorldHeight - graphNodeMinHeight - 12);
  return {
    x: Math.min(Math.max(12, x), maxX),
    y: Math.min(Math.max(48, y), maxY)
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function getPortAnchor(node: GraphNode, module: ModuleDef, portId: string, direction: "in" | "out") {
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
