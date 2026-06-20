import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent as ReactDragEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent
} from "react";
import { AlertTriangle, CheckCircle2, GitBranchPlus, Minus, MousePointer2, Move, Play, Plus, RotateCcw, Trash2, Wrench } from "lucide-react";
import { graphLevels } from "../levelRegistry";
import { createGameplayRegistry } from "../modules";
import type { DType, GraphEdge, GraphNode, GraphSpec, LevelSpec, ModuleDef, PortDef, PortRef, TestAssertion, TestResult, TraceFrame } from "../types";
import { runTests, type RunTestsResult, type TestCaseRunResult } from "../runtime/testRunner";

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

const moduleDragMime = "application/x-llm-complete-graph-module";
const graphNodeWidth = 190;
const graphNodeMinHeight = 118;
const graphWorldWidth = 2400;
const graphWorldHeight = 1600;
const minCanvasScale = 0.45;
const maxCanvasScale = 1.8;

export function GraphWorkbench() {
  const registry = useMemo(() => createGameplayRegistry(), []);
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
  const wireSourcePort = wireSource ? getOutputPort(graph, registry.list(), wireSource) : undefined;
  const activeTraceCase = resolveTraceCase(runState, traceSelection);
  const activeTraceFrame = activeTraceCase && traceSelection?.caseId === activeTraceCase.id ? activeTraceCase.execution.trace[traceSelection.step] : undefined;
  const hiddenLocked = runState.visible?.status !== "pass";

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (event.key === "Escape" && wireSourceRef.current) {
        event.preventDefault();
        cancelWire();
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
    if (target.closest(".graphNode, .graphCanvasToolbar, .graphCanvasHud, .graphCanvasModulePalette, button, input, select, textarea")) return;
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

  const wirePreview = wireSource && wirePointer ? getWirePreviewPath(graph, registry.list(), wireSource, wirePointer) : undefined;

  return (
    <section className="graphWorkbench">
      <aside className="graphSidebar panel">
        <div className="panelHeader">
          <Wrench size={18} />
          <h2>Graph Levels</h2>
        </div>
        <div className="graphLevelList">
          {graphLevels.map((level) => (
            <button key={level.id} className={`graphLevelButton ${level.id === selectedLevel.id ? "active" : ""}`} onClick={() => selectLevel(level)}>
              <span>
                <b>{level.title}</b>
                <small>{level.goal}</small>
              </span>
            </button>
          ))}
        </div>

      </aside>

      <section className="graphStage panel">
        <div className="graphStageHeader">
          <div>
            <p className="eyebrow">Graph Challenge</p>
            <h2>{selectedLevel.title}</h2>
            <small>{selectedLevel.goal}</small>
          </div>
          <div className="graphRunBar">
            {canInsertTranspose ? (
              <button className="ghostButton" onClick={insertTransposeRepair}>
                <Wrench size={15} />
                {transposeLabel}
              </button>
            ) : null}
            <button className="ghostButton" onClick={resetLevel}>
              <RotateCcw size={15} />
              Reset
            </button>
            <button className="ghostButton" onClick={autoLayoutGraph}>
              <GitBranchPlus size={15} />
              Auto Layout
            </button>
            <button className="runButton" onClick={runVisible}>
              <Play size={15} />
              Run Visible
            </button>
            <button className="runButton" disabled={hiddenLocked} onClick={runHidden}>
              <Play size={15} />
              Run Hidden
            </button>
            <button className="ghostButton" disabled={!selection} onClick={deleteSelection}>
              <Trash2 size={15} />
              Delete
            </button>
          </div>
        </div>

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
                    <span>{module.category}</span>
                    <b>{node.id}</b>
                    <small>{module.label}</small>
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
        <GraphTraceTimeline
          visible={runState.visible}
          hidden={runState.hidden}
          selection={traceSelection}
          onSelect={(nextSelection, frame) => {
            setTraceSelection(nextSelection);
            setInspectorTab("shape");
            setSelection({ type: "node", id: frame.nodeId });
          }}
        />
      </section>

      <aside className="graphInspector panel">
        <div className="panelHeader">
          <CheckCircle2 size={18} />
          <h2>Inspector</h2>
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
          />
        ) : null}
        {selectedEdge ? (
          <section className="graphInspectorBlock">
            <p className="eyebrow">Selected Edge</p>
            <h3>{selectedEdge.id}</h3>
            <code>{`${selectedEdge.from.nodeId}.${selectedEdge.from.portId} -> ${selectedEdge.to.nodeId}.${selectedEdge.to.portId}`}</code>
            <div className="graphInspectorActions">
              <button className="ghostButton" onClick={rewireSelectedEdge}>
                <GitBranchPlus size={14} />
                Rewire Target
              </button>
              <button className="ghostButton" onClick={deleteSelection}>
                <Trash2 size={14} />
                Delete Edge
              </button>
            </div>
          </section>
        ) : null}

        <GraphRankPanel level={selectedLevel} graph={graph} runState={runState} />
        <GraphRunPanel title="Visible Tests" result={runState.visible} />
        <GraphRunPanel title="Hidden Tests" result={runState.hidden} locked={hiddenLocked} />
      </aside>
    </section>
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
            <b>{port.label}</b>
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
  return (
    <section className="graphCanvasModulePalette">
      <div className="graphCanvasModulePaletteHeader">
        <b>Available Modules</b>
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
              title={module.summary}
              onDoubleClick={() => onAddModule(module.id)}
              onDragStart={(event) => {
                event.dataTransfer.setData(moduleDragMime, module.id);
                event.dataTransfer.effectAllowed = "copy";
              }}
            >
              <b>{module.label}</b>
              <small>{module.category}</small>
            </button>
          );
        })}
      </div>
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
  const runs: Array<{ key: TraceRunKey; label: string; result?: RunTestsResult }> = [
    { key: "visible", label: "Visible", result: visible },
    { key: "hidden", label: "Hidden", result: hidden }
  ];
  const hasTrace = runs.some((run) => run.result?.cases.length);

  return (
    <section className="graphTracePanel">
      <div className="graphTraceHeader">
        <div>
          <p className="eyebrow">Trace Timeline</p>
          <h3>Run path / first failure / node state</h3>
        </div>
        <code>{hasTrace ? "click a step" : "idle"}</code>
      </div>
      {hasTrace ? (
        <div className="graphTraceRuns">
          {runs.map((run) =>
            run.result?.cases.map((testCase) => (
              <section key={`${run.key}:${testCase.id}`} className={`graphTraceCase ${testCase.status}`}>
                <div className="graphTraceCaseHeader">
                  <span>{run.label}</span>
                  <b>{testCase.title}</b>
                  <code>{testCase.status}</code>
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
        <p className="graphTraceEmpty">Run visible tests to capture a trace.</p>
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
  onParamChange
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
}) {
  const nodeFrame = activeTraceFrame?.nodeId === node.id ? activeTraceFrame : findLatestNodeFrame(activeTraceCase, node.id);
  const relatedResults = getNodeTestResults(runState, node.id);
  const failingResult = relatedResults.find((result) => result.status !== "pass");
  const incoming = graph.edges.filter((edge) => edge.to.nodeId === node.id);
  const outgoing = graph.edges.filter((edge) => edge.from.nodeId === node.id);
  const tabs: Array<{ id: InspectorTab; label: string }> = [
    { id: "summary", label: "Summary" },
    { id: "shape", label: "Shape" },
    { id: "values", label: "Values" },
    { id: "tests", label: "Tests" },
    { id: "code", label: "Code" }
  ];

  return (
    <section className="graphInspectorBlock">
      <p className="eyebrow">Selected Node</p>
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
          <GraphKeyValue label="category" value={module.category} />
          <GraphKeyValue label="purpose" value={module.summary} />
          <GraphKeyValue label="incoming" value={incoming.map((edge) => `${edge.from.nodeId}.${edge.from.portId} -> ${edge.to.portId}`).join("\n") || "none"} />
          <GraphKeyValue label="outgoing" value={outgoing.map((edge) => `${edge.from.portId} -> ${edge.to.nodeId}.${edge.to.portId}`).join("\n") || "none"} />
          {failingResult ? <GraphFailureReport result={failingResult} /> : null}
          <GraphParamEditor node={node} module={module} onChange={onParamChange} />
        </div>
      ) : null}

      {tab === "shape" ? (
        <div className="graphInspectorTabBody">
          {nodeFrame ? (
            <>
              <ShapeRecord title="Inputs" shapes={nodeFrame.inputShapes} />
              <ShapeRecord title="Outputs" shapes={nodeFrame.outputShapes} />
              {nodeFrame.error ? <GraphRuntimeErrorBox frame={nodeFrame} /> : null}
            </>
          ) : (
            <>
              <PortContractList title="Input Contracts" ports={module.inputs} />
              <PortContractList title="Output Contracts" ports={module.outputs} />
            </>
          )}
        </div>
      ) : null}

      {tab === "values" ? (
        <div className="graphInspectorTabBody">
          {nodeFrame?.samples ? (
            <pre className="graphValueBlock">{formatUnknown(nodeFrame.samples)}</pre>
          ) : (
            <p className="graphTraceEmpty">No sample values captured for this node yet.</p>
          )}
          {failingResult?.diagnostic?.sample ? (
            <>
              <h4>Failure sample</h4>
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
                <GraphResultItem key={result.id} item={result} detailed />
              ))}
            </div>
          ) : (
            <p className="graphTraceEmpty">No captured test result targets this node yet.</p>
          )}
        </div>
      ) : null}

      {tab === "code" ? (
        <div className="graphInspectorTabBody">
          <pre className="graphCodeBlock">{module.pseudoCode ?? "No pseudo code registered for this module."}</pre>
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
        <p className="graphTraceEmpty">No shaped values.</p>
      )}
    </section>
  );
}

function PortContractList({ title, ports }: { title: string; ports: PortDef[] }) {
  return (
    <section className="graphShapeRecord">
      <h4>{title}</h4>
      {ports.length ? (
        ports.map((port) => (
          <div key={port.id}>
            <span>{port.id}</span>
            <code>{port.direction === "in" ? port.accepts?.join(" | ") || "any" : port.emits ?? "value"}</code>
          </div>
        ))
      ) : (
        <p className="graphTraceEmpty">No ports.</p>
      )}
    </section>
  );
}

function GraphRuntimeErrorBox({ frame }: { frame: TraceFrame }) {
  if (!frame.error) return null;
  return (
    <section className="graphFailureReport">
      <b>{frame.error.type}</b>
      <p>{frame.error.message}</p>
      <GraphKeyValue label="expected" value={formatUnknown(frame.error.expected)} />
      <GraphKeyValue label="received" value={formatUnknown(frame.error.received)} />
    </section>
  );
}

function GraphFailureReport({ result }: { result: TestResult }) {
  const diagnostic = result.diagnostic;
  if (!diagnostic) return null;
  return (
    <section className="graphFailureReport">
      <b>{diagnostic.errorType ?? result.status}</b>
      <p>{result.message}</p>
      <GraphKeyValue label="expected" value={formatUnknown(diagnostic.expected)} />
      <GraphKeyValue label="received" value={formatUnknown(diagnostic.received)} />
      <GraphKeyValue label="cause" value={diagnostic.possibleCause ?? "unknown"} />
      <GraphKeyValue label="probe" value={diagnostic.suggestedProbe ?? "step through trace"} />
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
  const keys = [...new Set([...Object.keys(module.defaultParams), ...Object.keys(node.params)])];
  if (!keys.length) {
    return (
      <section className="graphParamEditor">
        <div className="graphParamHeader">
          <b>Parameters</b>
          <code>none</code>
        </div>
      </section>
    );
  }

  return (
    <section className="graphParamEditor">
      <div className="graphParamHeader">
        <b>Parameters</b>
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
  const options = paramOptions[paramKey];
  const valueType = typeof value;

  if (Array.isArray(value) || Array.isArray(defaultValue)) {
    const arrayValue = (Array.isArray(value) ? value : defaultValue) as unknown[];
    return (
      <label className="graphParamRow">
        <span>{paramKey}</span>
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
      </label>
    );
  }

  if (typeof value === "boolean" || typeof defaultValue === "boolean") {
    return (
      <label className="graphParamRow boolean">
        <span>{paramKey}</span>
        <input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(nodeId, paramKey, event.currentTarget.checked)} />
      </label>
    );
  }

  if (typeof value === "number" || typeof defaultValue === "number") {
    return (
      <label className="graphParamRow">
        <span>{paramKey}</span>
        <input type="number" value={Number(value ?? 0)} step={1} onChange={(event) => onChange(nodeId, paramKey, Number(event.currentTarget.value))} />
      </label>
    );
  }

  if (typeof value === "string" || typeof defaultValue === "string") {
    if (options?.length) {
      return (
        <label className="graphParamRow">
          <span>{paramKey}</span>
          <select value={String(value ?? "")} onChange={(event) => onChange(nodeId, paramKey, event.currentTarget.value)}>
            {options.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      );
    }

    return (
      <label className="graphParamRow">
        <span>{paramKey}</span>
        <input type="text" value={String(value ?? "")} onChange={(event) => onChange(nodeId, paramKey, event.currentTarget.value)} />
      </label>
    );
  }

  return (
    <div className="graphParamRow readonly">
      <span>{paramKey}</span>
      <code>{JSON.stringify(value)}</code>
      <small>{valueType === "object" ? "structured value" : valueType}</small>
    </div>
  );
}

const paramOptions: Record<string, string[]> = {
  policy: ["subword", "word", "char"],
  fallback: ["unk", "char", "none"],
  padSide: ["right", "left"],
  maskPolicy: ["pad-aware", "all-ones"],
  orientation: ["C,O", "O,C"],
  maskOrientation: ["query_key", "key_query"]
};

function GraphRankPanel({ level, graph, runState }: { level: LevelSpec; graph: GraphSpec; runState: RunState }) {
  const rank = computeRank(level, graph, runState);
  return (
    <section className={`graphRankPanel ${rank.rank === "-" ? "unranked" : rank.rank.toLowerCase()}`}>
      <div className="graphRunPanelHeader">
        <h3>Rank / Debrief</h3>
        <code>{rank.rank}</code>
      </div>
      <p>{rank.message}</p>
      <div className="graphRankStats">
        <code>visible {runState.stats.visibleRuns}</code>
        <code>hidden {runState.stats.hiddenRuns}</code>
        <code>failed {runState.stats.failedRuns}</code>
        <code>hints {runState.stats.hintsUsed}</code>
        <code>extra {rank.extraModules}</code>
      </div>
      {runState.hidden?.status === "pass" ? (
        <div className="graphDebriefBox">
          <b>{level.debrief.completeTitle}</b>
          <small>{level.debrief.fixedProblem}</small>
          <small>{level.debrief.learned}</small>
          <small>{level.debrief.nextUse}</small>
        </div>
      ) : null}
    </section>
  );
}

function GraphRunPanel({ title, result, locked = false }: { title: string; result?: RunTestsResult; locked?: boolean }) {
  return (
    <section className={`graphRunPanel ${result?.status ?? (locked ? "blocked" : "idle")}`}>
      <div className="graphRunPanelHeader">
        <h3>{title}</h3>
        <code>{locked ? "locked" : result?.status ?? "idle"}</code>
      </div>
      {locked ? <p>Visible tests must pass before hidden tests run.</p> : null}
      {result ? (
        <div className="graphResultList">
          {result.results.map((item) => (
            <GraphResultItem key={item.id} item={item} detailed={item.status !== "pass"} />
          ))}
        </div>
      ) : null}
    </section>
  );
}

function GraphResultItem({ item, detailed = false }: { item: TestResult; detailed?: boolean }) {
  const Icon = item.status === "pass" ? CheckCircle2 : AlertTriangle;
  return (
    <div className={`graphResultItem ${item.status}`}>
      <Icon size={15} />
      <span>
        <b>{item.status}</b>
        <small>{item.message}</small>
        {detailed && item.diagnostic ? (
          <>
            <small>expected: {formatUnknown(item.diagnostic.expected)}</small>
            <small>received: {formatUnknown(item.diagnostic.received)}</small>
            <small>cause: {item.diagnostic.possibleCause ?? "unknown"}</small>
            <small>probe: {item.diagnostic.suggestedProbe ?? "step through trace"}</small>
          </>
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

function computeRank(level: LevelSpec, graph: GraphSpec, runState: RunState) {
  const extraModules = countExtraModules(level.id, graph);
  if (runState.hidden?.status === "pass") {
    if (runState.stats.hiddenRuns === 1 && runState.stats.failedRuns === 0 && runState.stats.hintsUsed === 0 && extraModules === 0) {
      return { rank: "S", message: "Hidden tests passed on the first clean attempt.", extraModules };
    }
    if (runState.stats.failedRuns <= 2 && runState.stats.hintsUsed === 0) {
      return { rank: "A", message: "Hidden tests passed with a low failure count.", extraModules };
    }
    return { rank: "B", message: "Hidden tests passed. The graph generalizes beyond visible inputs.", extraModules };
  }
  if (runState.visible?.status === "pass") {
    return { rank: "C", message: "Visible tests passed. Run hidden mutation tests to prove generalization.", extraModules };
  }
  return { rank: "-", message: "Run visible tests to start the challenge loop.", extraModules };
}

function countExtraModules(levelId: string, graph: GraphSpec) {
  const expected = expectedNodeIdsByLevel[levelId];
  if (!expected) return 0;
  return graph.nodes.filter((node) => !expected.has(node.id)).length;
}

const expectedNodeIdsByLevel: Record<string, Set<string>> = {
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
