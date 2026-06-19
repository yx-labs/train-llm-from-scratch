import { useEffect, useRef, type MutableRefObject } from "react";
import { Application, Container, Graphics, Rectangle, Text, type FederatedPointerEvent, type Ticker } from "pixi.js";
import { sceneSize } from "./sceneData";
import type { TensorEdge, TensorNode, WorkbenchMode } from "./workbenchTypes";

const connectionSnapRadius = 44;
const minCanvasScale = 0.42;
const maxCanvasScale = 1.7;
const worldBounds = new Rectangle(-2400, -1800, 5600, 4200);
const repairTagDragMime = "application/x-llm-complete-repair-tag";
const initialViewCenter = { x: 720, y: 390 };

export type RepairSlotOverlay = {
  slotId: string;
  nodeId: string;
  label: string;
  value: string;
  state: "empty" | "filled" | "active";
};

export type CanvasConnectionOverlay = {
  id: string;
  slotId: string;
  tagId: string;
  fromNodeId: string;
  toNodeId: string;
  fromLabel: string;
  toLabel: string;
  label: string;
  kind: "data" | "contract";
  state: "open" | "connected";
  enabled: boolean;
};

export type CanvasContextTarget = {
  kind: "canvas" | "node" | "slot";
  id?: string;
  clientX: number;
  clientY: number;
  worldX: number;
  worldY: number;
};

export type CanvasActionHint = {
  kind: "node" | "slot";
  id: string;
  icon: "menu" | "probe" | "run";
  tooltip: string;
  pulse?: boolean;
};

export type CanvasTaskHint = {
  id: number;
  taskId: string;
  slotId?: string;
  nodeId?: string;
  title: string;
  detail: string;
};

export type CanvasStageKnowledge = {
  code: string;
  title: string;
  concept: string;
  tool: string;
  mission: string;
  visual:
    | "tensor_objects"
    | "rank_axes"
    | "shape_caliper"
    | "semantic_gap"
    | "token_grid"
    | "embedding_expansion"
    | "hidden_contract"
    | "consumer_contract"
    | "hidden_tests"
    | "dot_cell"
    | "token_projection"
    | "sequence_projection"
    | "batch_projection"
    | "weight_orientation"
    | "linear_assembly"
    | "matmul_gauntlet"
    | "transpose_matrix_flip"
    | "transpose_inner_dim"
    | "transpose_higher_rank"
    | "transpose_single_qk"
    | "transpose_multi_qk"
    | "transpose_debugger"
    | "transpose_gauntlet"
    | "broadcast_add_cell"
    | "broadcast_same_shape"
    | "broadcast_rule"
    | "broadcast_bias"
    | "broadcast_position"
    | "broadcast_mask"
    | "broadcast_debugger"
    | "broadcast_gauntlet";
  carryForward?: string;
};

type PixiWorkbenchCanvasProps = {
  selectedId: string;
  mode: WorkbenchMode;
  playing: boolean;
  nodes: TensorNode[];
  edges: TensorEdge[];
  slotOverlays?: RepairSlotOverlay[];
  connectionOverlays?: CanvasConnectionOverlay[];
  actionHints?: CanvasActionHint[];
  taskHint?: CanvasTaskHint;
  stageKnowledge?: CanvasStageKnowledge;
  stageKnowledgePosition?: Point;
  onSlotSelect?: (slotId: string) => void;
  onCanvasConnect?: (slotId: string, tagId: string) => void;
  onCanvasDropTag?: (slotId: string, tagId: string) => void;
  onCanvasContextMenu?: (target: CanvasContextTarget) => void;
  onNodeMove?: (nodeId: string, x: number, y: number) => void;
  onStageKnowledgeMove?: (x: number, y: number) => void;
  onSelect: (id: string) => void;
};

type Point = {
  x: number;
  y: number;
};

type FontWeight = "400" | "500" | "600" | "700" | "800" | "900" | "bold";

type ViewState = {
  x: number;
  y: number;
  scale: number;
  initialized: boolean;
};

export function PixiWorkbenchCanvas({
  selectedId,
  mode,
  playing,
  nodes,
  edges,
  slotOverlays = [],
  connectionOverlays = [],
  actionHints = [],
  taskHint,
  stageKnowledge,
  stageKnowledgePosition,
  onSlotSelect,
  onCanvasConnect,
  onCanvasDropTag,
  onCanvasContextMenu,
  onNodeMove,
  onStageKnowledgeMove,
  onSelect
}: PixiWorkbenchCanvasProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<Application | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const viewRef = useRef<ViewState>({ x: 0, y: 0, scale: 1, initialized: false });
  const latestRef = useRef({
    selectedId,
    mode,
    playing,
    nodes,
    edges,
    slotOverlays,
    connectionOverlays,
    actionHints,
    taskHint,
    stageKnowledge,
    stageKnowledgePosition,
    onSlotSelect,
    onCanvasConnect,
    onCanvasDropTag,
    onCanvasContextMenu,
    onNodeMove,
    onStageKnowledgeMove,
    onSelect
  });

  useEffect(() => {
    latestRef.current = {
      selectedId,
      mode,
      playing,
      nodes,
      edges,
      slotOverlays,
      connectionOverlays,
      actionHints,
      taskHint,
      stageKnowledge,
      stageKnowledgePosition,
      onSlotSelect,
      onCanvasConnect,
      onCanvasDropTag,
      onCanvasContextMenu,
      onNodeMove,
      onStageKnowledgeMove,
      onSelect
    };
    if (!appRef.current) return;

    cleanupRef.current?.();
    cleanupRef.current = drawScene(appRef.current, latestRef.current, viewRef);
  }, [
    selectedId,
    mode,
    playing,
    nodes,
    edges,
    slotOverlays,
    connectionOverlays,
    actionHints,
    taskHint,
    stageKnowledge,
    stageKnowledgePosition,
    onSlotSelect,
    onCanvasConnect,
    onCanvasDropTag,
    onCanvasContextMenu,
    onNodeMove,
    onStageKnowledgeMove,
    onSelect
  ]);

  useEffect(() => {
    let disposed = false;
    let observer: ResizeObserver | null = null;
    const host = hostRef.current;
    const app = new Application();

    async function mount() {
      if (!host) return;
      const size = measureHost(host);

      await app.init({
        width: size.width,
        height: size.height,
        backgroundAlpha: 0,
        antialias: true,
        autoDensity: true,
        resolution: Math.min(window.devicePixelRatio || 1, 3),
        preference: "webgl"
      });

      if (disposed) {
        app.destroy({ removeView: true, releaseGlobalResources: true }, { children: true, texture: true, textureSource: true });
        return;
      }

      appRef.current = app;
      app.canvas.className = "workbenchCanvasElement";
      app.stage.eventMode = "static";
      app.stage.hitArea = new Rectangle(0, 0, app.screen.width, app.screen.height);
      host.appendChild(app.canvas);
      app.ticker.maxFPS = 60;
      cleanupRef.current = drawScene(app, latestRef.current, viewRef);

      observer = new ResizeObserver((entries) => {
        const entry = entries[0];
        const width = Math.max(320, Math.floor(entry.contentRect.width));
        const height = Math.max(420, Math.floor(entry.contentRect.height));
        app.renderer.resize(width, height);
        app.stage.hitArea = new Rectangle(0, 0, app.screen.width, app.screen.height);
        cleanupRef.current?.();
        cleanupRef.current = drawScene(app, latestRef.current, viewRef);
      });
      observer.observe(host);
    }

    mount();

    return () => {
      disposed = true;
      observer?.disconnect();
      if (appRef.current === app) {
        cleanupRef.current?.();
        cleanupRef.current = null;
        app.destroy({ removeView: true, releaseGlobalResources: true }, { children: true, texture: true, textureSource: true });
        appRef.current = null;
      }
    };
  }, []);

  return <div ref={hostRef} className="pixiHost" aria-label="PixiJS tensor workbench canvas" />;
}

function measureHost(host: HTMLElement) {
  const width = Math.floor(host.clientWidth || host.getBoundingClientRect().width || 760);
  const height = Math.floor(host.clientHeight || host.getBoundingClientRect().height || 560);
  return {
    width: Math.max(320, width),
    height: Math.max(420, height)
  };
}

function drawScene(
  app: Application,
  state: {
    selectedId: string;
    mode: WorkbenchMode;
    playing: boolean;
    nodes: TensorNode[];
    edges: TensorEdge[];
    slotOverlays: RepairSlotOverlay[];
    connectionOverlays: CanvasConnectionOverlay[];
    actionHints: CanvasActionHint[];
    taskHint?: CanvasTaskHint;
    stageKnowledge?: CanvasStageKnowledge;
    stageKnowledgePosition?: Point;
    onSlotSelect?: (slotId: string) => void;
    onCanvasConnect?: (slotId: string, tagId: string) => void;
    onCanvasDropTag?: (slotId: string, tagId: string) => void;
    onCanvasContextMenu?: (target: CanvasContextTarget) => void;
    onNodeMove?: (nodeId: string, x: number, y: number) => void;
    onStageKnowledgeMove?: (x: number, y: number) => void;
    onSelect: (id: string) => void;
  },
  viewRef: MutableRefObject<ViewState>
) {
  for (const child of app.stage.removeChildren()) {
    child.destroy({ children: true });
  }

  const screenWidth = app.screen.width;
  const screenHeight = app.screen.height;
  const background = new Graphics();
  background.roundRect(0, 0, screenWidth, screenHeight, 10).fill({ color: 0x07111f, alpha: 1 });
  app.stage.addChild(background);

  const view = viewRef.current;
  if (!view.initialized) {
    const scale = Math.min((screenWidth - 28) / sceneSize.width, (screenHeight - 20) / sceneSize.height);
    view.scale = clamp(Math.max(scale, 0.74), minCanvasScale, maxCanvasScale);
    view.x = snapCanvasPixel(screenWidth / 2 - initialViewCenter.x * view.scale);
    view.y = snapCanvasPixel(screenHeight / 2 - initialViewCenter.y * view.scale);
    view.initialized = true;
  }

  const scene = new Container({ label: "tensor-workbench-scene", sortableChildren: true });
  scene.scale.set(view.scale);
  scene.x = view.x;
  scene.y = view.y;
  app.stage.addChild(scene);

  const gridLayer = new Graphics({ label: "infinite-grid" });
  const knowledgeLayer = new Container({ label: "stage-knowledge" });
  const edgeLayer = new Container({ label: "typed-data-lines" });
  const connectionLayer = new Container({ label: "repair-connections" });
  const nodeLayer = new Container({ label: "tensor-nodes" });
  const objectLayer = new Container({ label: "tensor-object-drag-layer", sortableChildren: true });
  const overlayLayer = new Container({ label: "labels-and-overlays" });
  const previewLayer = new Graphics({ label: "connection-preview" });
  const pulseLayer = new Graphics({ label: "flow-pulses" });
  const tooltipLayer = new Container({ label: "hover-tooltips" });
  gridLayer.eventMode = "none";
  knowledgeLayer.eventMode = "static";
  edgeLayer.eventMode = "none";
  connectionLayer.eventMode = "none";
  nodeLayer.eventMode = "static";
  objectLayer.eventMode = "static";
  overlayLayer.eventMode = "static";
  previewLayer.eventMode = "none";
  pulseLayer.eventMode = "none";
  tooltipLayer.eventMode = "none";
  scene.addChild(gridLayer, knowledgeLayer, edgeLayer, connectionLayer, nodeLayer, objectLayer, overlayLayer, previewLayer, pulseLayer, tooltipLayer);
  scene.eventMode = "static";
  scene.hitArea = worldBounds;

  drawWorldGrid(gridLayer, worldBounds);
  let knowledgeDrag: {
    group: Container;
    startPointer: Point;
    startPosition: Point;
    moved: boolean;
  } | null = null;

  let tensorObjectDrag: {
    group: Container;
    item: TensorObjectDragItem;
    startPointer: Point;
    startPosition: Point;
    moved: boolean;
  } | null = null;

  if (state.stageKnowledge) {
    drawStageKnowledge(
      knowledgeLayer,
      state.stageKnowledge,
      state.stageKnowledgePosition,
      (event, group, position) => {
        event.stopPropagation();
        const pointer = event.getLocalPosition(scene);
        knowledgeDrag = {
          group,
          startPointer: pointer,
          startPosition: position,
          moved: false
        };
        group.cursor = "grabbing";
      }
    );
  }

  const nodeLookup = new Map(state.nodes.map((node) => [node.id, node]));
  const nodeHints = new Map(state.actionHints.filter((hint) => hint.kind === "node").map((hint) => [hint.id, hint]));
  const slotHints = new Map(state.actionHints.filter((hint) => hint.kind === "slot").map((hint) => [hint.id, hint]));
  const visibleEdges = state.edges.filter((edge) => edge.flow !== "gradient" || state.mode === "train");
  let currentNodeLookup = nodeLookup;
  let currentConnectionPorts = buildConnectionPorts(state.connectionOverlays, currentNodeLookup);
  let currentSlotBounds = buildRepairSlotBounds(state.slotOverlays, currentNodeLookup);

  let connectionDrag: {
    connection: CanvasConnectionOverlay;
    from: Point;
  } | null = null;

  let nodeDrag: {
    group: Container;
    node: TensorNode;
    startPointer: Point;
    moved: boolean;
  } | null = null;

  let canvasDrag: {
    startGlobal: Point;
    startView: Point;
    moved: boolean;
  } | null = null;
  let suppressNextTap = false;

  const syncSceneView = () => {
    viewRef.current.x = snapCanvasPixel(viewRef.current.x);
    viewRef.current.y = snapCanvasPixel(viewRef.current.y);
    scene.x = viewRef.current.x;
    scene.y = viewRef.current.y;
    scene.scale.set(viewRef.current.scale);
  };

  const openContextMenu = (event: FederatedPointerEvent, target: Pick<CanvasContextTarget, "kind" | "id">) => {
    event.preventDefault();
    event.stopPropagation();
    const client = getClientPoint(event, app);
    const world = event.getLocalPosition(scene);
    state.onCanvasContextMenu?.({
      ...target,
      clientX: client.x,
      clientY: client.y,
      worldX: world.x,
      worldY: world.y
    });
  };

  const clearConnectionPreview = () => {
    previewLayer.clear();
  };

  const showTooltip = (text: string, point: Point) => {
    clearLayer(tooltipLayer);
    drawTooltip(tooltipLayer, text, point.x, point.y);
  };

  const hideTooltip = () => {
    clearLayer(tooltipLayer);
  };

  const drawConnectionPreview = (from: Point, to: Point, color: number, valid: boolean) => {
    const previewColor = valid ? 0x22c55e : color;
    previewLayer.clear();
    previewLayer.moveTo(from.x, from.y);
    const midX = Math.max(from.x + 50, (from.x + to.x) / 2);
    previewLayer.bezierCurveTo(midX, from.y, midX, to.y, to.x, to.y).stroke({
      width: valid ? 3.4 : 2.4,
      color: previewColor,
      alpha: valid ? 0.92 : 0.58,
      cap: "round"
    });
    previewLayer.circle(to.x, to.y, valid ? 6 : 4.5).fill({ color: previewColor, alpha: valid ? 0.96 : 0.74 });
  };

  const connectionPortHandlers = {
    onStart: (connection: CanvasConnectionOverlay, point: Point, event: FederatedPointerEvent) => {
      event.stopPropagation();
      if (!connection.enabled || connection.state === "connected") return;
      connectionDrag = { connection, from: point };
      drawConnectionPreview(point, point, connectionColor(connection), true);
    },
    onComplete: (connection: CanvasConnectionOverlay, event: FederatedPointerEvent) => {
      event.stopPropagation();
      if (connectionDrag?.connection.id === connection.id) {
        state.onCanvasConnect?.(connection.slotId, connection.tagId);
      }
      connectionDrag = null;
      clearConnectionPreview();
    }
  };

  const draggedNodeLookup = (nodeId: string, x: number, y: number) =>
    new Map(
      state.nodes.map((node) => [
        node.id,
        node.id === nodeId
          ? {
              ...node,
              x,
              y
            }
          : node
      ])
    );

  const renderDynamicLayers = (lookup: Map<string, TensorNode>) => {
    currentNodeLookup = lookup;
    currentConnectionPorts = buildConnectionPorts(state.connectionOverlays, currentNodeLookup);
    currentSlotBounds = buildRepairSlotBounds(state.slotOverlays, currentNodeLookup);
    clearLayer(edgeLayer);
    clearLayer(connectionLayer);
    clearLayer(overlayLayer);
    visibleEdges.forEach((edge) => drawEdge(edgeLayer, overlayLayer, edge, state.mode, currentNodeLookup));
    drawCanvasConnectionLines(connectionLayer, overlayLayer, state.connectionOverlays, currentConnectionPorts);
    drawRepairSlots(
      overlayLayer,
      state.stageKnowledge?.visual === "tensor_objects" ? state.slotOverlays.filter((slot) => !slot.slotId.startsWith("object_")) : state.slotOverlays,
      currentNodeLookup,
      slotHints,
      state.onSlotSelect,
      (event, slotId) => openContextMenu(event, { kind: "slot", id: slotId }),
      showTooltip,
      hideTooltip
    );
    drawCanvasConnectionPorts(overlayLayer, state.connectionOverlays, currentConnectionPorts, connectionPortHandlers, showTooltip, hideTooltip);
    if (state.taskHint) {
      drawCanvasTaskHint(overlayLayer, state.taskHint, currentNodeLookup, currentSlotBounds, state.stageKnowledge);
    }
    if (state.mode === "train") {
      drawTrainOverlay(overlayLayer);
    }
  };

  renderDynamicLayers(currentNodeLookup);

  state.nodes.forEach((node) => {
    drawTensorNode(
      nodeLayer,
      node,
      state.selectedId === node.id,
      state.onSelect,
      state.mode,
      (event, group) => {
        const pointer = event.getLocalPosition(scene);
        nodeDrag = {
          group,
          node,
          startPointer: pointer,
          moved: false
        };
        group.zIndex = 20;
      },
      (event) => openContextMenu(event, { kind: "node", id: node.id }),
      nodeHints.get(node.id),
      (event) => openContextMenu(event, { kind: "node", id: node.id }),
      showTooltip,
      hideTooltip,
      {
        knowledge: state.stageKnowledge,
        slots: new Map(state.slotOverlays.map((slot) => [slot.slotId, slot])),
        connections: new Map(state.connectionOverlays.map((connection) => [connection.id, connection]))
      }
    );
  });

  if (state.stageKnowledge?.visual === "tensor_objects") {
    drawTensorObjectDragChallenge(
      objectLayer,
      state.slotOverlays,
      currentNodeLookup,
      (event, group, item, position) => {
        event.stopPropagation();
        const pointer = event.getLocalPosition(scene);
        tensorObjectDrag = {
          group,
          item,
          startPointer: pointer,
          startPosition: position,
          moved: false
        };
        group.cursor = "grabbing";
        group.zIndex = 50;
      }
    );
  }

  let elapsed = 0;
  const tick = (ticker: Ticker) => {
    elapsed += ticker.deltaMS;
    drawPulses(pulseLayer, visibleEdges, elapsed, state.playing || state.mode === "train", currentNodeLookup);
  };
  app.ticker.add(tick);

  const handleGlobalMove = (event: FederatedPointerEvent) => {
    if (connectionDrag) {
      const pointer = event.getLocalPosition(scene);
      const target = currentConnectionPorts.get(connectionDrag.connection.id)?.target;
      const validTarget = Boolean(target && distance(pointer, target.point) <= connectionSnapRadius);
      drawConnectionPreview(connectionDrag.from, pointer, connectionColor(connectionDrag.connection), validTarget);
    }

    if (knowledgeDrag) {
      const pointer = event.getLocalPosition(scene);
      const dx = pointer.x - knowledgeDrag.startPointer.x;
      const dy = pointer.y - knowledgeDrag.startPointer.y;
      knowledgeDrag.group.x = knowledgeDrag.startPosition.x + dx;
      knowledgeDrag.group.y = knowledgeDrag.startPosition.y + dy;
      knowledgeDrag.moved = Math.abs(dx) > 2 || Math.abs(dy) > 2;
    }

    if (tensorObjectDrag) {
      const pointer = event.getLocalPosition(scene);
      const dx = pointer.x - tensorObjectDrag.startPointer.x;
      const dy = pointer.y - tensorObjectDrag.startPointer.y;
      tensorObjectDrag.group.x = tensorObjectDrag.startPosition.x + dx;
      tensorObjectDrag.group.y = tensorObjectDrag.startPosition.y + dy;
      tensorObjectDrag.moved = Math.abs(dx) > 2 || Math.abs(dy) > 2;
    }

    if (nodeDrag) {
      const pointer = event.getLocalPosition(scene);
      const dx = pointer.x - nodeDrag.startPointer.x;
      const dy = pointer.y - nodeDrag.startPointer.y;
      nodeDrag.group.x = dx;
      nodeDrag.group.y = dy;
      nodeDrag.moved = Math.abs(dx) > 2 || Math.abs(dy) > 2;
      renderDynamicLayers(draggedNodeLookup(nodeDrag.node.id, nodeDrag.node.x + dx, nodeDrag.node.y + dy));
    }

    if (canvasDrag) {
      const global = { x: event.global.x, y: event.global.y };
      const dx = global.x - canvasDrag.startGlobal.x;
      const dy = global.y - canvasDrag.startGlobal.y;
      viewRef.current.x = snapCanvasPixel(canvasDrag.startView.x + dx);
      viewRef.current.y = snapCanvasPixel(canvasDrag.startView.y + dy);
      canvasDrag.moved = Math.abs(dx) > 2 || Math.abs(dy) > 2;
      syncSceneView();
    }
  };

  const finishGestures = (event?: FederatedPointerEvent) => {
    if (knowledgeDrag) {
      const pointer = event?.getLocalPosition(scene);
      const dx = pointer ? pointer.x - knowledgeDrag.startPointer.x : knowledgeDrag.group.x - knowledgeDrag.startPosition.x;
      const dy = pointer ? pointer.y - knowledgeDrag.startPointer.y : knowledgeDrag.group.y - knowledgeDrag.startPosition.y;
      const nextX = knowledgeDrag.startPosition.x + dx;
      const nextY = knowledgeDrag.startPosition.y + dy;
      knowledgeDrag.group.cursor = "grab";
      if (knowledgeDrag.moved) {
        state.onStageKnowledgeMove?.(snapCanvasPixel(nextX), snapCanvasPixel(nextY));
      }
      knowledgeDrag = null;
    }

    if (tensorObjectDrag) {
      const pointer = event?.getLocalPosition(scene);
      const dx = pointer ? pointer.x - tensorObjectDrag.startPointer.x : tensorObjectDrag.group.x - tensorObjectDrag.startPosition.x;
      const dy = pointer ? pointer.y - tensorObjectDrag.startPointer.y : tensorObjectDrag.group.y - tensorObjectDrag.startPosition.y;
      const nextPosition = {
        x: tensorObjectDrag.startPosition.x + dx,
        y: tensorObjectDrag.startPosition.y + dy
      };
      const dropPoint = {
        x: nextPosition.x + tensorObjectDrag.item.width / 2,
        y: nextPosition.y + tensorObjectDrag.item.height / 2
      };
      const inspector = currentNodeLookup.get("tensor_inspector");
      const accepted = Boolean(inspector && tensorInspectorDropRect(inspector).contains(dropPoint.x, dropPoint.y));
      tensorObjectDrag.group.cursor = "grab";

      if (accepted) {
        state.onCanvasDropTag?.(tensorObjectDrag.item.slotId, tensorObjectDrag.item.tagId);
      } else {
        tensorObjectDrag.group.x = tensorObjectDrag.startPosition.x;
        tensorObjectDrag.group.y = tensorObjectDrag.startPosition.y;
      }
      tensorObjectDrag = null;
    }

    if (connectionDrag) {
      const pointer = event?.getLocalPosition(scene);
      const target = currentConnectionPorts.get(connectionDrag.connection.id)?.target;
      if (pointer && target && distance(pointer, target.point) <= connectionSnapRadius) {
        state.onCanvasConnect?.(connectionDrag.connection.slotId, connectionDrag.connection.tagId);
      }
      connectionDrag = null;
      clearConnectionPreview();
    }

    if (nodeDrag) {
      const pointer = event?.getLocalPosition(scene);
      const dx = pointer ? pointer.x - nodeDrag.startPointer.x : nodeDrag.group.x;
      const dy = pointer ? pointer.y - nodeDrag.startPointer.y : nodeDrag.group.y;
      if (nodeDrag.moved) {
        const nextX = nodeDrag.node.x + dx;
        const nextY = nodeDrag.node.y + dy;
        const snappedX = snapCanvasPixel(nextX);
        const snappedY = snapCanvasPixel(nextY);
        renderDynamicLayers(draggedNodeLookup(nodeDrag.node.id, snappedX, snappedY));
        state.onNodeMove?.(nodeDrag.node.id, snappedX, snappedY);
      } else {
        nodeDrag.group.x = 0;
        nodeDrag.group.y = 0;
        renderDynamicLayers(nodeLookup);
      }
      nodeDrag = null;
    }

    if (canvasDrag) {
      suppressNextTap = canvasDrag.moved;
      canvasDrag = null;
    }
  };

  const handleScenePointerDown = (event: FederatedPointerEvent) => {
    if (event.button === 2) return;
    const pointer = event.getLocalPosition(scene);
    const connectionStart = findNearestConnectableConnectionStart(pointer, state.connectionOverlays, currentConnectionPorts);
    if (connectionStart) {
      connectionDrag = connectionStart;
      drawConnectionPreview(connectionStart.from, pointer, connectionColor(connectionStart.connection), true);
      return;
    }
    canvasDrag = {
      startGlobal: { x: event.global.x, y: event.global.y },
      startView: { x: viewRef.current.x, y: viewRef.current.y },
      moved: false
    };
  };

  const handleSceneTap = (event: FederatedPointerEvent) => {
    if (suppressNextTap) {
      suppressNextTap = false;
      return;
    }
    if (nodeDrag || connectionDrag || knowledgeDrag || tensorObjectDrag) return;
    event.stopPropagation();
  };

  const handleSceneRightClick = (event: FederatedPointerEvent) => {
    openContextMenu(event, { kind: "canvas" });
  };

  const handleWheel = (event: WheelEvent) => {
    event.preventDefault();
    const rect = app.canvas.getBoundingClientRect();
    const screenPoint = clientToScreenPoint(event.clientX, event.clientY, rect, app);
    const worldPoint = screenToWorld(screenPoint, viewRef.current);
    const nextScale = clamp(viewRef.current.scale * Math.exp(-event.deltaY * 0.0012), minCanvasScale, maxCanvasScale);
    viewRef.current.scale = nextScale;
    viewRef.current.x = snapCanvasPixel(screenPoint.x - worldPoint.x * nextScale);
    viewRef.current.y = snapCanvasPixel(screenPoint.y - worldPoint.y * nextScale);
    syncSceneView();
  };

  const draggedRepairTag = (event: DragEvent) => event.dataTransfer?.getData(repairTagDragMime) || event.dataTransfer?.getData("text/plain") || "";

  const dropSlotForEvent = (event: DragEvent) => {
    const rect = app.canvas.getBoundingClientRect();
    const screenPoint = clientToScreenPoint(event.clientX, event.clientY, rect, app);
    return findRepairSlotAtPoint(screenToWorld(screenPoint, viewRef.current), currentSlotBounds);
  };

  const handleCanvasDragOver = (event: DragEvent) => {
    if (!dropSlotForEvent(event)) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  };

  const handleCanvasDrop = (event: DragEvent) => {
    const tagId = draggedRepairTag(event);
    const slotId = dropSlotForEvent(event);
    if (!tagId || !slotId) return;
    event.preventDefault();
    state.onCanvasDropTag?.(slotId, tagId);
  };

  const preventBrowserContextMenu = (event: MouseEvent) => {
    event.preventDefault();
  };

  scene.on("pointerdown", handleScenePointerDown);
  scene.on("globalpointermove", handleGlobalMove);
  scene.on("pointertap", handleSceneTap);
  scene.on("rightdown", handleSceneRightClick);
  scene.on("pointerup", finishGestures);
  scene.on("pointerupoutside", finishGestures);
  app.canvas.addEventListener("wheel", handleWheel, { passive: false });
  app.canvas.addEventListener("dragover", handleCanvasDragOver);
  app.canvas.addEventListener("drop", handleCanvasDrop);
  app.canvas.addEventListener("contextmenu", preventBrowserContextMenu);

  return () => {
    app.ticker.remove(tick);
    scene.off("pointerdown", handleScenePointerDown);
    scene.off("globalpointermove", handleGlobalMove);
    scene.off("pointertap", handleSceneTap);
    scene.off("rightdown", handleSceneRightClick);
    scene.off("pointerup", finishGestures);
    scene.off("pointerupoutside", finishGestures);
    app.canvas.removeEventListener("wheel", handleWheel);
    app.canvas.removeEventListener("dragover", handleCanvasDragOver);
    app.canvas.removeEventListener("drop", handleCanvasDrop);
    app.canvas.removeEventListener("contextmenu", preventBrowserContextMenu);
  };
}

function drawWorldGrid(graphics: Graphics, bounds: Rectangle) {
  const maxX = bounds.x + bounds.width;
  const maxY = bounds.y + bounds.height;
  for (let x = bounds.x; x <= maxX; x += 32) {
    graphics.moveTo(x, bounds.y).lineTo(x, maxY).stroke({ width: 1, color: 0x5f7aa0, alpha: x % 128 === 0 ? 0.22 : 0.08 });
  }
  for (let y = bounds.y; y <= maxY; y += 32) {
    graphics.moveTo(bounds.x, y).lineTo(maxX, y).stroke({ width: 1, color: 0x5f7aa0, alpha: y % 128 === 0 ? 0.22 : 0.08 });
  }
}

function drawStageKnowledge(
  layer: Container,
  knowledge: CanvasStageKnowledge,
  position: Point | undefined,
  onDragStart?: (event: FederatedPointerEvent, group: Container, position: Point) => void
) {
  const group = new Container({ label: `stage-knowledge-${knowledge.code}` });
  group.x = position?.x ?? 420;
  group.y = position?.y ?? 24;

  const x = 0;
  const y = 0;
  const w = 560;
  const h = knowledge.carryForward ? 126 : 110;
  group.eventMode = "static";
  group.cursor = "grab";
  group.hitArea = new Rectangle(x, y, w, h);
  group.on("pointerdown", (event) => {
    if (event.button === 2) return;
    onDragStart?.(event, group, { x: group.x, y: group.y });
  });

  const shell = new Graphics();
  shell
    .roundRect(x, y, w, h, 10)
    .fill({ color: 0x081524, alpha: 0.9 })
    .stroke({ width: 1.4, color: 0x315f94, alpha: 0.78 });
  shell.rect(x + 1, y + 1, 7, h - 2).fill({ color: 0x38bdf8, alpha: 0.52 });
  group.addChild(shell);

  const code = new Graphics();
  code.roundRect(x + 18, y + 15, 50, 25, 999).fill({ color: 0x0f2740, alpha: 0.95 }).stroke({ width: 1, color: 0x60a5fa, alpha: 0.72 });
  group.addChild(code);
  drawDragGrip(group, x + w - 34, y + 8);
  addText(group, knowledge.code, x + 43, y + 28, 12, 0xbfe5ff, "900", 0.5);
  addText(group, knowledge.title, x + 80, y + 28, 15, 0xe8f2ff, "900", 0);
  addText(group, knowledge.concept, x + 20, y + 57, 12, 0xc8d8eb, "700", 0);

  drawStageVisual(group, knowledge.visual, x + 324, y + 19, 214, h - 38);

  const toolLabel = makeText(`Tool: ${knowledge.tool}`, 10, 0x93c5fd, "800");
  toolLabel.x = x + 20;
  toolLabel.y = y + 76;
  group.addChild(toolLabel);
  addText(group, knowledge.mission, x + 20, y + 96, 10, 0x9db2ca, "700", 0);

  if (knowledge.carryForward) {
    const carry = new Graphics();
    carry.roundRect(x + 18, y + h - 24, 282, 18, 5).fill({ color: 0x08251f, alpha: 0.82 }).stroke({ width: 1, color: 0x22c55e, alpha: 0.38 });
    group.addChild(carry);
    addText(group, knowledge.carryForward, x + 28, y + h - 14, 9, 0xbbf7d0, "800", 0);
  }

  layer.addChild(group);
}

function drawDragGrip(layer: Container, x: number, y: number) {
  const grip = new Graphics();
  for (let row = 0; row < 2; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      grip.circle(x + col * 7, y + row * 7, 2).fill({ color: 0x93c5fd, alpha: 0.76 });
    }
  }
  layer.addChild(grip);
}

function drawStageVisual(layer: Container, visual: CanvasStageKnowledge["visual"], x: number, y: number, w: number, h: number) {
  const panel = new Graphics();
  panel.roundRect(x, y, w, h, 8).fill({ color: 0x06111f, alpha: 0.82 }).stroke({ width: 1, color: 0x263b55, alpha: 0.95 });
  layer.addChild(panel);

  switch (visual) {
    case "tensor_objects":
      drawTensorObjectSequence(layer, x + 12, y + 17);
      break;
    case "rank_axes":
      drawRankAxisSequence(layer, x + 14, y + 17);
      break;
    case "shape_caliper":
      drawShapeCaliperVisual(layer, x + 20, y + 12);
      break;
    case "semantic_gap":
      drawSemanticGapVisual(layer, x + 16, y + 14);
      break;
    case "token_grid":
      drawTokenGridVisual(layer, x + 20, y + 13);
      break;
    case "embedding_expansion":
      drawEmbeddingExpansionVisual(layer, x + 14, y + 15);
      break;
    case "hidden_contract":
      drawHiddenContractVisual(layer, x + 22, y + 11);
      break;
    case "consumer_contract":
      drawConsumerContractVisual(layer, x + 18, y + 13);
      break;
    case "hidden_tests":
      drawHiddenTestsVisual(layer, x + 14, y + 15);
      break;
    case "dot_cell":
      drawMatMulStageVisual(layer, x + 14, y + 15, "[C]", "dot", "scalar");
      break;
    case "token_projection":
      drawMatMulStageVisual(layer, x + 14, y + 15, "[C]", "@ [C,O]", "[O]");
      break;
    case "sequence_projection":
      drawMatMulStageVisual(layer, x + 14, y + 15, "[T,C]", "@ [C,O]", "[T,O]");
      break;
    case "batch_projection":
      drawMatMulStageVisual(layer, x + 14, y + 15, "[B,T,C]", "@ [C,O]", "[B,T,O]");
      break;
    case "weight_orientation":
      drawMatMulStageVisual(layer, x + 14, y + 15, "stored [O,C]", "T(W)", "compute [C,O]");
      break;
    case "linear_assembly":
      drawMatMulStageVisual(layer, x + 14, y + 15, "MatMul", "+ bias off", "Linear");
      break;
    case "matmul_gauntlet":
      drawMatMulStageVisual(layer, x + 14, y + 15, "A/B/C", "hidden", "pass");
      break;
    case "transpose_matrix_flip":
      drawMatMulStageVisual(layer, x + 14, y + 15, "A[R,C]", "T", "A^T[C,R]");
      break;
    case "transpose_inner_dim":
      drawMatMulStageVisual(layer, x + 14, y + 15, "A[M,N]", "T(B)", "[M,P]");
      break;
    case "transpose_higher_rank":
      drawMatMulStageVisual(layer, x + 14, y + 15, "[B,H,T,D]", "-2<->-1", "[B,H,D,T]");
      break;
    case "transpose_single_qk":
      drawMatMulStageVisual(layer, x + 14, y + 15, "Q[T,D]", "K^T", "scores[T,T]");
      break;
    case "transpose_multi_qk":
      drawMatMulStageVisual(layer, x + 14, y + 15, "Q[B,H,T,D]", "K^T", "[B,H,T,T]");
      break;
    case "transpose_debugger":
      drawMatMulStageVisual(layer, x + 14, y + 15, "axis", "cell trace", "allclose");
      break;
    case "transpose_gauntlet":
      drawMatMulStageVisual(layer, x + 14, y + 15, "T==D", "hidden", "pass");
      break;
    case "broadcast_add_cell":
      drawBroadcastStageVisual(layer, x + 14, y + 15, "A[]", "+ B[]", "out[]", "one cell, two sources");
      break;
    case "broadcast_same_shape":
      drawBroadcastStageVisual(layer, x + 14, y + 15, "[T,C]", "+ [T,C]", "[T,C]", "same coordinates");
      break;
    case "broadcast_rule":
      drawBroadcastStageVisual(layer, x + 14, y + 15, "[C]", "view", "[B,T,C]", "right-align + singleton axes");
      break;
    case "broadcast_bias":
      drawBroadcastStageVisual(layer, x + 14, y + 15, "bias[O]", "*B/*T", "[B,T,O]", "feature-axis offset");
      break;
    case "broadcast_position":
      drawBroadcastStageVisual(layer, x + 14, y + 15, "pos[T,C]", "*B", "[B,T,C]", "keep token position");
      break;
    case "broadcast_mask":
      drawBroadcastStageVisual(layer, x + 14, y + 15, "mask", "+ -1e9", "scores", "Tq/Tk aligned");
      break;
    case "broadcast_debugger":
      drawBroadcastStageVisual(layer, x + 14, y + 15, "shape", "trace", "semantic", "shape-valid traps");
      break;
    case "broadcast_gauntlet":
      drawBroadcastStageVisual(layer, x + 14, y + 15, "A-F", "hidden", "pass", "generalize contract");
      break;
  }
}

function drawTensorObjectSequence(layer: Container, x: number, y: number) {
  const baseY = y + 20;
  const labelY = y + 55;
  drawScalarGlyph(layer, x, baseY, "3.14");
  drawSmallArrow(layer, { x: x + 30, y: baseY }, { x: x + 42, y: baseY }, 0x60a5fa);
  drawVectorGlyph(layer, x + 51, baseY - 11, 4, 0x7dd3fc);
  drawSmallArrow(layer, { x: x + 98, y: baseY }, { x: x + 109, y: baseY }, 0x60a5fa);
  drawMatrixGlyph(layer, x + 117, baseY - 15, 2, 3, 7, 0x38bdf8);
  drawSmallArrow(layer, { x: x + 149, y: baseY }, { x: x + 158, y: baseY }, 0x60a5fa);
  drawStackedMatrixGlyph(layer, x + 156, baseY - 12, 0x38bdf8);
  drawAlignedStageLabel(layer, "number", x + 14, labelY);
  drawAlignedStageLabel(layer, "vector", x + 73, labelY);
  drawAlignedStageLabel(layer, "grid", x + 132, labelY);
  drawAlignedStageLabel(layer, "stack", x + 174, labelY);
}

function drawAlignedStageLabel(layer: Container, label: string, x: number, y: number) {
  const marker = new Graphics();
  marker.circle(x, y - 10, 1.8).fill({ color: 0x60a5fa, alpha: 0.62 });
  marker.moveTo(x, y - 7).lineTo(x, y - 2).stroke({ width: 1, color: 0x60a5fa, alpha: 0.34 });
  layer.addChild(marker);
  addText(layer, label, x, y + 4, 8, 0x9db2ca, "800", 0.5);
}

function drawRankAxisSequence(layer: Container, x: number, y: number) {
  drawScalarGlyph(layer, x, y + 28, "r0");
  drawAxisLineGlyph(layer, x + 52, y + 28, 36, "r1", 0x7dd3fc);
  drawPlaneGlyph(layer, x + 105, y + 10, "r2", 0x38bdf8);
  drawStackedMatrixGlyph(layer, x + 172, y + 10, 0xfbbf24);
  addText(layer, "0 / 1 / 2 / 3 axes", x + 105, y + 62, 10, 0xdbeafe, "900", 0.5);
}

function drawShapeCaliperVisual(layer: Container, x: number, y: number) {
  drawMiniCuboid(layer, x + 42, y + 14, 112, 50, 18, 0x24608a, 0.82);
  drawDimensionLine(layer, { x: x + 42, y: y + 75 }, { x: x + 154, y: y + 75 }, "Axis 1 = 4", 0x7dd3fc);
  drawDimensionLine(layer, { x: x + 27, y: y + 64 }, { x: x + 27, y: y + 22 }, "Axis 0 = 2", 0xfbbf24);
  drawDimensionLine(layer, { x: x + 160, y: y + 12 }, { x: x + 179, y: y - 5 }, "Axis 2 = 8", 0x22c55e);
  addText(layer, "shape = [2,4,8]", x + 100, y + 88, 10, 0xe8f2ff, "900", 0.5);
}

function drawSemanticGapVisual(layer: Container, x: number, y: number) {
  drawMiniCuboid(layer, x + 7, y + 15, 92, 46, 16, 0x24608a, 0.78);
  addText(layer, "float32[2,4,8]", x + 54, y + 75, 9, 0xe8f2ff, "900", 0.5);
  addText(layer, "Axis 0 ?", x + 122, y + 17, 9, 0xfbbf24, "900", 0);
  addText(layer, "Axis 1 ?", x + 122, y + 39, 9, 0x7dd3fc, "900", 0);
  addText(layer, "Axis 2 ?", x + 122, y + 61, 9, 0x22c55e, "900", 0);
  drawBlockedPlug(layer, x + 187, y + 17, "Batch");
  drawBlockedPlug(layer, x + 187, y + 39, "Mask");
  drawBlockedPlug(layer, x + 187, y + 61, "Linear");
}

function drawTokenGridVisual(layer: Container, x: number, y: number) {
  const values = [
    ["502", "2841", "9172", "0"],
    ["1042", "7191", "3910", "0"]
  ];
  addText(layer, "T0     T1     T2     T3", x + 58, y + 6, 9, 0x7dd3fc, "900", 0);
  for (let row = 0; row < 2; row += 1) {
    addText(layer, `B${row}`, x, y + 26 + row * 24, 10, 0xfbbf24, "900", 0);
    for (let col = 0; col < 4; col += 1) {
      const cell = new Graphics();
      cell.roundRect(x + 28 + col * 42, y + 15 + row * 24, 36, 18, 4).fill({ color: 0x0f2740, alpha: 0.92 }).stroke({ width: 1, color: 0x315f94, alpha: 0.8 });
      layer.addChild(cell);
      addText(layer, values[row][col], x + 46 + col * 42, y + 25 + row * 24, 8, 0xe8f2ff, "800", 0.5);
    }
  }
  drawDimensionLine(layer, { x: x + 20, y: y + 11 }, { x: x + 20, y: y + 61 }, "B", 0xfbbf24);
  drawDimensionLine(layer, { x: x + 28, y: y + 67 }, { x: x + 194, y: y + 67 }, "T", 0x7dd3fc);
}

function drawEmbeddingExpansionVisual(layer: Container, x: number, y: number) {
  const token = new Graphics();
  token.roundRect(x, y + 22, 45, 24, 5).fill({ color: 0x0f2740, alpha: 0.94 }).stroke({ width: 1, color: 0x7dd3fc, alpha: 0.9 });
  layer.addChild(token);
  addText(layer, "9172", x + 22, y + 35, 10, 0xe8f2ff, "900", 0.5);
  drawSmallArrow(layer, { x: x + 52, y: y + 34 }, { x: x + 82, y: y + 34 }, 0x60a5fa);
  drawMatrixGlyph(layer, x + 88, y + 12, 2, 6, 7, 0xa78bfa);
  addText(layer, "row", x + 106, y + 64, 9, 0xc4b5fd, "900", 0.5);
  drawSmallArrow(layer, { x: x + 139, y: y + 34 }, { x: x + 167, y: y + 34 }, 0x60a5fa);
  drawVectorGlyph(layer, x + 174, y + 25, 5, 0x22c55e);
  addText(layer, "C vector", x + 196, y + 58, 9, 0xbbf7d0, "900", 0.5);
}

function drawHiddenContractVisual(layer: Container, x: number, y: number) {
  drawMiniCuboid(layer, x + 32, y + 12, 116, 54, 20, 0x24608a, 0.84);
  drawDimensionLine(layer, { x: x + 25, y: y + 66 }, { x: x + 25, y: y + 26 }, "B?", 0xfbbf24);
  drawDimensionLine(layer, { x: x + 36, y: y + 77 }, { x: x + 150, y: y + 77 }, "T?", 0x7dd3fc);
  drawDimensionLine(layer, { x: x + 154, y: y + 13 }, { x: x + 175, y: y - 5 }, "C?", 0x22c55e);
  addText(layer, "Probe -> evidence -> B/T/C", x + 104, y + 92, 9, 0xe8f2ff, "900", 0.5);
}

function drawConsumerContractVisual(layer: Container, x: number, y: number) {
  drawMiniCuboid(layer, x, y + 18, 78, 42, 14, 0x24608a, 0.84);
  addText(layer, "hidden", x + 38, y + 72, 9, 0xe8f2ff, "900", 0.5);
  const consumers = [
    { label: "B -> Batch", y: y + 8, color: 0xfbbf24 },
    { label: "T -> Mask", y: y + 35, color: 0x7dd3fc },
    { label: "C -> Linear", y: y + 62, color: 0x22c55e }
  ];
  consumers.forEach((item) => {
    drawSmallArrow(layer, { x: x + 92, y: item.y + 9 }, { x: x + 125, y: item.y + 9 }, item.color);
    const box = new Graphics();
    box.roundRect(x + 130, item.y, 74, 18, 4).fill({ color: 0x081524, alpha: 0.96 }).stroke({ width: 1, color: item.color, alpha: 0.76 });
    layer.addChild(box);
    addText(layer, item.label, x + 167, item.y + 10, 8, 0xe8f2ff, "800", 0.5);
  });
}

function drawHiddenTestsVisual(layer: Container, x: number, y: number) {
  const cases = [
    { label: "[2,4,8]", x },
    { label: "[1,16,4]", x: x + 74 },
    { label: "[4,3,32]", x: x + 152 }
  ];
  cases.forEach((item, index) => {
    drawMiniCuboid(layer, item.x + 8, y + 16, 48, 28, 9, index === 1 ? 0x1f4b72 : 0x24608a, 0.82);
    addText(layer, item.label, item.x + 32, y + 61, 8, 0xe8f2ff, "900", 0.5);
  });
  addText(layer, "same semantics, changing lengths", x + 105, y + 82, 8, 0x9db2ca, "800", 0.5);
}

function drawMatMulStageVisual(layer: Container, x: number, y: number, leftLabel: string, opLabel: string, rightLabel: string) {
  const nodes = [
    { label: leftLabel, x, color: 0x24608a },
    { label: opLabel, x: x + 82, color: 0x304b6a },
    { label: rightLabel, x: x + 164, color: 0x1f6f54 }
  ];

  nodes.forEach((item, index) => {
    const box = new Graphics();
    box.roundRect(item.x, y + 28, 58, 30, 6).fill({ color: item.color, alpha: 0.82 }).stroke({ width: 1, color: 0x7dd3fc, alpha: index === 1 ? 0.45 : 0.3 });
    layer.addChild(box);
    addText(layer, item.label, item.x + 29, y + 44, 8, 0xe8f2ff, "900", 0.5);
  });

  drawSmallArrow(layer, { x: x + 61, y: y + 43 }, { x: x + 78, y: y + 43 }, 0x60a5fa);
  drawSmallArrow(layer, { x: x + 143, y: y + 43 }, { x: x + 160, y: y + 43 }, 0x60a5fa);
  drawVectorGlyph(layer, x + 5, y + 3, 4, 0x7dd3fc);
  drawMatrixGlyph(layer, x + 92, y, 3, 4, 6, 0xa78bfa);
  drawVectorGlyph(layer, x + 174, y + 3, 4, 0x22c55e);
  addText(layer, "consume C -> generate O", x + 105, y + 78, 9, 0x9db2ca, "800", 0.5);
}

function drawBroadcastStageVisual(layer: Container, x: number, y: number, leftLabel: string, opLabel: string, rightLabel: string, footer: string) {
  const nodes = [
    { label: leftLabel, x, color: 0x24608a },
    { label: opLabel, x: x + 82, color: 0x304b6a },
    { label: rightLabel, x: x + 164, color: 0x1f6f54 }
  ];

  nodes.forEach((item, index) => {
    const box = new Graphics();
    box.roundRect(item.x, y + 28, 58, 30, 6).fill({ color: item.color, alpha: 0.82 }).stroke({ width: 1, color: 0x7dd3fc, alpha: index === 1 ? 0.5 : 0.34 });
    layer.addChild(box);
    addText(layer, item.label, item.x + 29, y + 44, 8, 0xe8f2ff, "900", 0.5);
  });

  drawSmallArrow(layer, { x: x + 61, y: y + 43 }, { x: x + 78, y: y + 43 }, 0x60a5fa);
  drawSmallArrow(layer, { x: x + 143, y: y + 43 }, { x: x + 160, y: y + 43 }, 0x60a5fa);
  drawVectorGlyph(layer, x + 6, y + 4, 4, 0x7dd3fc);
  drawBroadcastGhostGlyph(layer, x + 89, y + 4);
  drawMatrixGlyph(layer, x + 174, y + 2, 3, 4, 6, 0x22c55e);
  addText(layer, footer, x + 105, y + 78, 9, 0x9db2ca, "800", 0.5);
}

function drawBroadcastGhostGlyph(layer: Container, x: number, y: number) {
  const ghost = new Graphics();
  for (let i = 0; i < 3; i += 1) {
    ghost.roundRect(x + i * 10, y + i * 3, 38, 22, 5).stroke({ width: 1, color: 0xa78bfa, alpha: 0.32 + i * 0.18 });
  }
  ghost.circle(x + 18, y + 12, 3).fill({ color: 0xa78bfa, alpha: 0.8 });
  layer.addChild(ghost);
}

function drawScalarGlyph(layer: Container, x: number, y: number, label: string) {
  const dot = new Graphics();
  dot.circle(x + 14, y, 12).fill({ color: 0x422b16, alpha: 0.96 }).stroke({ width: 1.4, color: 0xfbbf24, alpha: 0.9 });
  layer.addChild(dot);
  addText(layer, label, x + 14, y, 7, 0xffedd5, "900", 0.5);
}

function drawVectorGlyph(layer: Container, x: number, y: number, count: number, color: number) {
  const cells = new Graphics();
  for (let i = 0; i < count; i += 1) {
    cells.roundRect(x + i * 12, y, 9, 22, 3).fill({ color, alpha: 0.34 + i * 0.08 }).stroke({ width: 1, color, alpha: 0.8 });
  }
  layer.addChild(cells);
}

function drawMatrixGlyph(layer: Container, x: number, y: number, rows: number, columns: number, size: number, color: number) {
  const grid = new Graphics();
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      grid.roundRect(x + col * (size + 3), y + row * (size + 3), size, size, 2).fill({ color, alpha: 0.22 + ((row + col) % 3) * 0.14 });
    }
  }
  grid.roundRect(x - 4, y - 4, columns * (size + 3) + 3, rows * (size + 3) + 3, 5).stroke({ width: 1, color, alpha: 0.68 });
  layer.addChild(grid);
}

function drawStackedMatrixGlyph(layer: Container, x: number, y: number, color: number) {
  for (let i = 2; i >= 0; i -= 1) {
    drawMatrixGlyph(layer, x + i * 7, y - i * 5, 3, 3, 7, color);
  }
}

function drawAxisLineGlyph(layer: Container, x: number, y: number, length: number, label: string, color: number) {
  const axis = new Graphics();
  axis.moveTo(x, y).lineTo(x + length, y).stroke({ width: 2.3, color, alpha: 0.9, cap: "round" });
  axis.circle(x, y, 4).fill({ color, alpha: 0.86 });
  axis.circle(x + length, y, 4).fill({ color, alpha: 0.86 });
  layer.addChild(axis);
  addText(layer, label, x + length / 2, y + 18, 8, 0xe8f2ff, "900", 0.5);
}

function drawPlaneGlyph(layer: Container, x: number, y: number, label: string, color: number) {
  drawMatrixGlyph(layer, x, y, 3, 4, 8, color);
  addText(layer, label, x + 20, y + 50, 8, 0xe8f2ff, "900", 0.5);
}

function drawMiniCuboid(layer: Container, x: number, y: number, w: number, h: number, depth: number, color: number, alpha: number) {
  const top = new Graphics();
  top
    .poly([x, y, x + depth, y - depth, x + w + depth, y - depth, x + w, y], true)
    .fill({ color: tint(color, 1.24), alpha })
    .stroke({ width: 1, color: 0x7dd3fc, alpha: 0.78 });
  const side = new Graphics();
  side
    .poly([x + w, y, x + w + depth, y - depth, x + w + depth, y + h - depth, x + w, y + h], true)
    .fill({ color: tint(color, 0.56), alpha })
    .stroke({ width: 1, color: 0x7dd3fc, alpha: 0.72 });
  const front = new Graphics();
  front.roundRect(x, y, w, h, 5).fill({ color, alpha }).stroke({ width: 1.2, color: 0x7dd3fc, alpha: 0.86 });
  layer.addChild(top, side, front);
}

function drawSmallArrow(layer: Container, from: Point, to: Point, color: number) {
  const line = new Graphics();
  line.moveTo(from.x, from.y).lineTo(to.x, to.y).stroke({ width: 1.6, color, alpha: 0.82, cap: "round" });
  layer.addChild(line);
  drawArrow(layer, to, from, color);
}

function drawDimensionLine(layer: Container, from: Point, to: Point, label: string, color: number) {
  const line = new Graphics();
  line.moveTo(from.x, from.y).lineTo(to.x, to.y).stroke({ width: 1.2, color, alpha: 0.88, cap: "round" });
  line.circle(from.x, from.y, 3).fill({ color, alpha: 0.9 });
  line.circle(to.x, to.y, 3).fill({ color, alpha: 0.9 });
  layer.addChild(line);
  addText(layer, label, (from.x + to.x) / 2 + 3, (from.y + to.y) / 2 - 8, 8, color, "900", 0.5);
}

function drawBlockedPlug(layer: Container, x: number, y: number, label: string) {
  const plug = new Graphics();
  plug.roundRect(x, y - 8, 36, 16, 4).fill({ color: 0x1c1917, alpha: 0.88 }).stroke({ width: 1, color: 0xf59e0b, alpha: 0.7 });
  plug.moveTo(x + 5, y - 4).lineTo(x + 13, y + 4).moveTo(x + 13, y - 4).lineTo(x + 5, y + 4).stroke({ width: 1.4, color: 0xef4444, alpha: 0.92 });
  layer.addChild(plug);
  addText(layer, label, x + 44, y + 1, 8, 0xfde68a, "800", 0);
}

function drawTensorNode(
  layer: Container,
  node: TensorNode,
  selected: boolean,
  onSelect: (id: string) => void,
  mode: WorkbenchMode,
  onDragStart?: (event: FederatedPointerEvent, group: Container) => void,
  onContextMenu?: (event: FederatedPointerEvent) => void,
  actionHint?: CanvasActionHint,
  onActionHint?: (event: FederatedPointerEvent) => void,
  showTooltip?: (text: string, point: Point) => void,
  hideTooltip?: () => void,
  stageContext?: StageNodeContext
) {
  const group = new Container({ label: `node-${node.id}` });
  group.eventMode = "static";
  group.cursor = "grab";
  group.hitArea = new Rectangle(node.x - 26, node.y - 38, node.w + 74, node.h + 70);
  group.on("pointerdown", (event) => {
    if (event.button === 2) return;
    event.stopPropagation();
    group.cursor = "grabbing";
    onDragStart?.(event, group);
  });
  group.on("pointertap", (event) => {
    event.stopPropagation();
    if (actionHint) {
      onActionHint?.(event);
      return;
    }
    onSelect(node.id);
  });
  group.on("rightdown", (event) => {
    onContextMenu?.(event);
  });

  if (node.id === "generated_object_vector") {
    drawVectorTensorShell(group, node, selected);
  } else if (node.kind === "tensor" || node.kind === "attention") {
    drawCuboid(group, node, selected);
  } else {
    const shell = new Graphics();
    shell
      .roundRect(node.x, node.y, node.w, node.h, 8)
      .fill({ color: node.color, alpha: node.kind === "mask" ? 0.78 : 0.93 })
      .stroke({ width: selected ? 3 : 1.4, color: selected ? 0xfbbf24 : 0x7dd3fc, alpha: selected ? 1 : 0.82 });
    group.addChild(shell);
  }

  drawNodeBodyVisual(group, node, stageContext);

  drawPort(group, node.x - 9, node.y + node.h / 2, selected, "in");
  drawPort(group, node.x + node.w + 9, node.y + node.h / 2, selected, "out");
  if (mode === "build") {
    drawPortLabels(group, node);
  }

  const usesEmbeddedLabel = node.id === "tensor_inspector";
  if (!usesEmbeddedLabel) {
    addText(group, node.title, node.x + node.w / 2, node.y + 24, 15, 0xe8f2ff, "800", 0.5);
    if (!stageNodeHasReadout(node, stageContext)) {
      addText(group, `${node.dtype}${node.shape}`, node.x + node.w / 2, node.y + node.h - 18, 12, 0xb7c7dc, "700", 0.5);
    }
  }

  if (node.kind === "matrix") {
    drawMiniCells(group, node.x + 18, node.y + 38, 7, 3, 8, 0x7dd3fc);
  }
  if (node.kind === "attention") {
    drawMiniHeatmap(group, node.x + node.w - 86, node.y + 42, node.id === "attn_scores");
  }
  if (node.kind === "mask") {
    drawMaskGlyph(group, node.x + node.w - 58, node.y + 23);
  }
  if (node.kind === "scalar") {
    drawGauge(group, node.x + 18, node.y + 30);
  }

  if (actionHint) {
    drawActionBadge(
      group,
      node.x + node.w - 12,
      node.y - 11,
      actionHint,
      (event) => {
        event.stopPropagation();
        onActionHint?.(event);
      },
      showTooltip,
      hideTooltip
    );
  }

  layer.addChild(group);
}

function drawNodeBodyVisual(layer: Container, node: TensorNode, stageContext?: StageNodeContext) {
  if (node.id === "generated_object_vector") {
    drawVectorGlyph(layer, node.x + 40, node.y + 42, 4, 0x7dd3fc);
    return;
  }

  if (node.id === "raw_objects") {
    const y = node.y + 50;
    drawScalarGlyph(layer, node.x + 17, y + 5, "3.14");
    drawVectorGlyph(layer, node.x + 52, y - 6, 4, 0x7dd3fc);
    drawMatrixGlyph(layer, node.x + 104, y - 12, 2, 3, 7, 0x38bdf8);
    drawStackedMatrixGlyph(layer, node.x + 144, y - 11, 0x38bdf8);
    drawStageSpecificNodeReadout(layer, node, stageContext);
    return;
  }

  if (node.id === "rank_scanner") {
    drawRankAxisSequence(layer, node.x + 20, node.y + 35);
    drawStageSpecificNodeReadout(layer, node, stageContext);
    return;
  }

  if (node.id === "shape_caliper") {
    drawShapeCaliperVisual(layer, node.x + 20, node.y + 35);
    drawStageSpecificNodeReadout(layer, node, stageContext);
    return;
  }

  if (node.id === "token_grid") {
    drawTokenGridVisual(layer, node.x + 14, node.y + 34);
    drawStageSpecificNodeReadout(layer, node, stageContext);
    return;
  }

  if (node.id === "embedding_lookup") {
    drawEmbeddingExpansionVisual(layer, node.x + 17, node.y + 34);
    drawStageSpecificNodeReadout(layer, node, stageContext);
    return;
  }

  if (node.id === "hidden_tensor") {
    const semanticKnown = node.shape.includes("B") || node.shape.includes("T") || node.shape.includes("C");
    drawDimensionLine(layer, { x: node.x - 12, y: node.y + node.h + 8 }, { x: node.x - 12, y: node.y + 12 }, semanticKnown ? "B" : "A0", 0xfbbf24);
    drawDimensionLine(layer, { x: node.x + 12, y: node.y + node.h + 16 }, { x: node.x + node.w - 6, y: node.y + node.h + 16 }, semanticKnown ? "T" : "A1", 0x7dd3fc);
    drawDimensionLine(layer, { x: node.x + node.w + 8, y: node.y + 12 }, { x: node.x + node.w + 30, y: node.y - 8 }, semanticKnown ? "C" : "A2", 0x22c55e);
    drawStageSpecificNodeReadout(layer, node, stageContext);
    return;
  }

  drawStageSpecificNodeReadout(layer, node, stageContext);
}

function stageNodeHasReadout(node: TensorNode, context?: StageNodeContext) {
  const visual = context?.knowledge?.visual;
  if (!visual) return false;
  if (visual === "rank_axes") return node.id === "rank_scanner";
  if (visual === "shape_caliper") return node.id === "shape_caliper" || node.id === "shape_gate";
  if (visual === "semantic_gap") return node.id === "semantic_inspector";
  if (visual === "token_grid") return node.id === "token_grid" || node.id === "tokenizer";
  if (visual === "embedding_expansion") return node.id === "embedding_lookup" || node.id === "hidden_tensor";
  if (visual === "hidden_contract") return node.id === "hidden_tensor";
  if (visual === "consumer_contract") return node.id === "batch_viewer" || node.id === "causal_mask" || node.id === "linear_probe" || node.id === "shape_tests";
  if (visual === "hidden_tests") return node.id === "shape_tests";
  return false;
}

function drawStageSpecificNodeReadout(layer: Container, node: TensorNode, context?: StageNodeContext) {
  if (!context?.knowledge || !stageNodeHasReadout(node, context)) return;
  const visual = context.knowledge.visual;

  if (visual === "rank_axes" && node.id === "rank_scanner") {
    drawReadoutPanel(layer, node.x + 14, node.y + 92, node.w - 28, 70, "Axis Count", [
      readoutRow("scalar", slotText(context, "rank_scalar", "drop r0"), 0xfbbf24),
      readoutRow("vector", slotText(context, "rank_vector", "drop r1"), 0x7dd3fc),
      readoutRow("matrix", slotText(context, "rank_matrix", "drop r2"), 0x38bdf8),
      readoutRow("3D block", slotText(context, "rank_block", "drop r3"), 0x22c55e)
    ]);
    return;
  }

  if (visual === "shape_caliper" && node.id === "shape_caliper") {
    drawReadoutPanel(layer, node.x + 14, node.y + 118, node.w - 28, 58, "Caliper Readout", [
      readoutRow("Axis 0", "length 2", 0xfbbf24),
      readoutRow("Axis 1", "length 4", 0x7dd3fc),
      readoutRow("Axis 2", "length 8", 0x22c55e)
    ]);
    return;
  }

  if (visual === "shape_caliper" && node.id === "shape_gate") {
    const shape = `[${slotText(context, "shape_axis_0", "?")},${slotText(context, "shape_axis_1", "?")},${slotText(context, "shape_axis_2", "?")}]`;
    drawReadoutPanel(layer, node.x + 12, node.y + 46, node.w - 24, 38, "Shape Contract", [readoutRow("axis order", shape, shape === "[2,4,8]" ? 0x22c55e : 0xfbbf24)]);
    return;
  }

  if (visual === "semantic_gap" && node.id === "semantic_inspector") {
    const marked = slotIsFilled(context, "semantic_unresolved");
    drawReadoutPanel(layer, node.x + 16, node.y + 56, node.w - 32, 106, "Contract State", [
      readoutRow("float32[2,4,8]", "size known", 0x7dd3fc),
      readoutRow("Axis 0 / 1 / 2", marked ? "semantics unresolved" : "meaning unknown", marked ? 0x22c55e : 0xfbbf24),
      readoutRow("Batch Viewer", "blocked without B", 0xef4444),
      readoutRow("Causal Mask", "blocked without T", 0xef4444),
      readoutRow("Linear", "blocked without C", 0xef4444)
    ]);
    return;
  }

  if (visual === "token_grid" && node.id === "tokenizer") {
    drawReadoutPanel(layer, node.x + 12, node.y + 50, node.w - 24, 34, "Tokenizer Port", [
      readoutRow("input", connectionIsConnected(context, "canvas_flow_text_tokenizer") ? "utf8[B] connected" : "drag line from Text", connectionIsConnected(context, "canvas_flow_text_tokenizer") ? 0x22c55e : 0xfbbf24)
    ]);
    return;
  }

  if (visual === "token_grid" && node.id === "token_grid") {
    drawReadoutPanel(layer, node.x + 14, node.y + 118, node.w - 28, 48, "Grid Semantics", [
      readoutRow("rows", slotText(context, "token_grid_b", "B?"), slotIsFilled(context, "token_grid_b") ? 0x22c55e : 0xfbbf24),
      readoutRow("columns", slotText(context, "token_grid_t", "T?"), slotIsFilled(context, "token_grid_t") ? 0x22c55e : 0xfbbf24),
      readoutRow("micro drill", `${slotText(context, "token_task_sample1", "B1?")} / ${slotText(context, "token_task_t2", "T2?")}`, slotIsFilled(context, "token_task_sample1") && slotIsFilled(context, "token_task_t2") ? 0x22c55e : 0x93c5fd)
    ]);
    return;
  }

  if (visual === "embedding_expansion" && node.id === "embedding_lookup") {
    drawReadoutPanel(layer, node.x + 14, node.y + 100, node.w - 28, 48, "Lookup Trace", [
      readoutRow("token_ids[0,2]", slotIsFilled(context, "embedding_probe") ? "9172 -> row 9172" : "inspect row", slotIsFilled(context, "embedding_probe") ? 0x22c55e : 0xfbbf24),
      readoutRow("row vector", slotIsFilled(context, "embedding_autofill") ? "float32[C]" : "needs fill", slotIsFilled(context, "embedding_autofill") ? 0x22c55e : 0x93c5fd),
      readoutRow("C", "feature channel", 0x22c55e)
    ]);
    return;
  }

  if (visual === "embedding_expansion" && node.id === "hidden_tensor") {
    drawReadoutPanel(layer, node.x + 16, node.y + 76, node.w - 32, 54, "Expansion Output", [
      readoutRow("[B,T]", connectionIsConnected(context, "canvas_flow_grid_embedding") ? "token positions kept" : "open", 0x7dd3fc),
      readoutRow("+ C", slotIsFilled(context, "embedding_autofill") ? "vector per token" : "waiting", slotIsFilled(context, "embedding_autofill") ? 0x22c55e : 0xfbbf24),
      readoutRow("hidden", connectionIsConnected(context, "canvas_flow_embedding_hidden") ? "[B,T,C]" : "[?,?,?]", connectionIsConnected(context, "canvas_flow_embedding_hidden") ? 0x22c55e : 0x9db2ca)
    ]);
    return;
  }

  if (visual === "hidden_contract" && node.id === "hidden_tensor") {
    drawReadoutPanel(layer, node.x + 18, node.y + 82, node.w - 36, 78, "Probe Evidence -> Contract", [
      readoutRow("Axis 0", `${slotText(context, "axis_0", "?")} / sample slices`, slotIsFilled(context, "axis_0") ? 0x22c55e : 0xfbbf24),
      readoutRow("Axis 1", `${slotText(context, "axis_1", "?")} / token order`, slotIsFilled(context, "axis_1") ? 0x22c55e : 0x7dd3fc),
      readoutRow("Axis 2", `${slotText(context, "axis_2", "?")} / feature values`, slotIsFilled(context, "axis_2") ? 0x22c55e : 0x22c55e),
      readoutRow("contract", `hidden[${slotText(context, "axis_0", "?")},${slotText(context, "axis_1", "?")},${slotText(context, "axis_2", "?")}]`, 0xe8f2ff)
    ]);
    return;
  }

  if (visual === "consumer_contract") {
    if (node.id === "batch_viewer") {
      drawReadoutPanel(layer, node.x + 12, node.y + 54, node.w - 24, 34, "Consumer", [
        readoutRow("expects", "B / samples", connectionIsConnected(context, "canvas_consumer_b") ? 0x22c55e : 0xfbbf24)
      ]);
      return;
    }
    if (node.id === "causal_mask") {
      drawReadoutPanel(layer, node.x + 12, node.y + 54, node.w - 24, 34, "Consumer", [
        readoutRow("expects", "T -> [T,T] mask", connectionIsConnected(context, "canvas_consumer_t") ? 0x22c55e : 0xfbbf24)
      ]);
      return;
    }
    if (node.id === "linear_probe") {
      drawReadoutPanel(layer, node.x + 12, node.y + 54, node.w - 24, 34, "Consumer", [
        readoutRow("expects", "C -> [C,O]", connectionIsConnected(context, "canvas_consumer_c") ? 0x22c55e : 0xfbbf24)
      ]);
      return;
    }
    if (node.id === "shape_tests") {
      const ready = connectionIsConnected(context, "canvas_consumer_b") && connectionIsConnected(context, "canvas_consumer_t") && connectionIsConnected(context, "canvas_consumer_c");
      drawReadoutPanel(layer, node.x + 12, node.y + 48, node.w - 24, 52, "Consumer Test", [
        readoutRow("B/T/C ports", ready ? "accepted" : "waiting", ready ? 0x22c55e : 0xfbbf24),
        readoutRow("next", "hidden tests", 0x93c5fd)
      ]);
      return;
    }
  }

  if (visual === "hidden_tests" && node.id === "shape_tests") {
    drawReadoutPanel(layer, node.x + 12, node.y + 48, node.w - 24, 54, "Hidden Cases", [
      readoutRow("visible", "[2,4,8]", 0x7dd3fc),
      readoutRow("variant A", "[1,16,4]", 0xfbbf24),
      readoutRow("variant B", "[4,3,32]", 0x22c55e)
    ]);
  }
}

function readoutRow(label: string, value: string, color: number) {
  return { label, value, color };
}

function drawReadoutPanel(
  layer: Container,
  x: number,
  y: number,
  width: number,
  height: number,
  title: string,
  rows: Array<{ label: string; value: string; color: number }>
) {
  const shell = new Graphics();
  shell
    .roundRect(x, y, width, height, 7)
    .fill({ color: 0x06111f, alpha: 0.9 })
    .stroke({ width: 1.1, color: 0x315f94, alpha: 0.7 });
  shell.rect(x, y, 4, height).fill({ color: 0x38bdf8, alpha: 0.36 });
  layer.addChild(shell);
  addText(layer, title, x + 12, y + 12, 9, 0x93c5fd, "900", 0);
  rows.forEach((row, index) => {
    const rowY = y + 28 + index * 11;
    addText(layer, row.label, x + 12, rowY, 8, 0x9db2ca, "800", 0);
    addText(layer, row.value, x + width - 10, rowY, 8, row.color, "900", 1);
  });
}

function slotText(context: StageNodeContext, slotId: string, fallback: string) {
  const slot = context.slots.get(slotId);
  if (!slot || slot.state === "empty") return fallback;
  return slot.value;
}

function slotIsFilled(context: StageNodeContext, slotId: string) {
  const slot = context.slots.get(slotId);
  return Boolean(slot && slot.state !== "empty");
}

function connectionIsConnected(context: StageNodeContext, connectionId: string) {
  return context.connections.get(connectionId)?.state === "connected";
}

type ConnectionPort = {
  connectionId: string;
  role: "source" | "target";
  point: Point;
  label: string;
};

type StageNodeContext = {
  knowledge?: CanvasStageKnowledge;
  slots: Map<string, RepairSlotOverlay>;
  connections: Map<string, CanvasConnectionOverlay>;
};

type ConnectionPortMap = Map<string, { source: ConnectionPort; target: ConnectionPort }>;

function buildConnectionPorts(connections: CanvasConnectionOverlay[], nodeLookup: Map<string, TensorNode>): ConnectionPortMap {
  const sourceGroups = new Map<string, CanvasConnectionOverlay[]>();
  const targetGroups = new Map<string, CanvasConnectionOverlay[]>();

  connections.forEach((connection) => {
    sourceGroups.set(connection.fromNodeId, [...(sourceGroups.get(connection.fromNodeId) ?? []), connection]);
    targetGroups.set(connection.toNodeId, [...(targetGroups.get(connection.toNodeId) ?? []), connection]);
  });

  const result: ConnectionPortMap = new Map();
  const pointFor = (node: TensorNode, connectionsForNode: CanvasConnectionOverlay[], connection: CanvasConnectionOverlay, role: "source" | "target") => {
    const index = Math.max(0, connectionsForNode.findIndex((item) => item.id === connection.id));
    const gap = node.h / (connectionsForNode.length + 1);
    return {
      x: role === "source" ? node.x + node.w + 10 : node.x - 10,
      y: node.y + gap * (index + 1)
    };
  };

  connections.forEach((connection) => {
    const fromNode = nodeLookup.get(connection.fromNodeId);
    const toNode = nodeLookup.get(connection.toNodeId);
    if (!fromNode || !toNode) return;
    result.set(connection.id, {
      source: {
        connectionId: connection.id,
        role: "source",
        point: pointFor(fromNode, sourceGroups.get(connection.fromNodeId) ?? [], connection, "source"),
        label: connection.fromLabel
      },
      target: {
        connectionId: connection.id,
        role: "target",
        point: pointFor(toNode, targetGroups.get(connection.toNodeId) ?? [], connection, "target"),
        label: connection.toLabel
      }
    });
  });

  return result;
}

function drawCanvasConnectionLines(
  edgeLayer: Container,
  labelLayer: Container,
  connections: CanvasConnectionOverlay[],
  ports: ConnectionPortMap
) {
  connections.forEach((connection) => {
    if (connection.state !== "connected") return;
    const endpoints = ports.get(connection.id);
    if (!endpoints) return;
    const color = connectionColor(connection);
    const alpha = 0.9;
    const width = 3.2;
    drawCurvedConnection(edgeLayer, endpoints.source.point, endpoints.target.point, color, alpha, width);

    const midpoint = curvedMidpoint(endpoints.source.point, endpoints.target.point);
    drawDataLabel(labelLayer, connection.label, midpoint.x, midpoint.y - 18, color);
  });
}

function drawCanvasConnectionPorts(
  layer: Container,
  connections: CanvasConnectionOverlay[],
  ports: ConnectionPortMap,
  handlers: {
    onStart: (connection: CanvasConnectionOverlay, point: Point, event: FederatedPointerEvent) => void;
    onComplete: (connection: CanvasConnectionOverlay, event: FederatedPointerEvent) => void;
  },
  showTooltip?: (text: string, point: Point) => void,
  hideTooltip?: () => void
) {
  connections.forEach((connection) => {
    const endpoints = ports.get(connection.id);
    if (!endpoints) return;
    drawConnectionPort(layer, connection, endpoints.source, handlers, showTooltip, hideTooltip);
    drawConnectionPort(layer, connection, endpoints.target, handlers, showTooltip, hideTooltip);
  });
}

function drawConnectionPort(
  layer: Container,
  connection: CanvasConnectionOverlay,
  port: ConnectionPort,
  handlers: {
    onStart: (connection: CanvasConnectionOverlay, point: Point, event: FederatedPointerEvent) => void;
    onComplete: (connection: CanvasConnectionOverlay, event: FederatedPointerEvent) => void;
  },
  showTooltip?: (text: string, point: Point) => void,
  hideTooltip?: () => void
) {
  const group = new Container({ label: `connection-port-${connection.id}-${port.role}` });
  const color = connectionColor(connection);
  const active = connection.enabled && connection.state === "open";
  const connected = connection.state === "connected";
  const cursor = active && port.role === "source" ? "crosshair" : active && port.role === "target" ? "copy" : "default";
  group.eventMode = "passive";
  group.cursor = cursor;
  group.hitArea = new Rectangle(port.point.x - 34, port.point.y - 28, 68, 56);

  const hitPad = new Graphics();
  hitPad
    .roundRect(port.point.x - 34, port.point.y - 28, 68, 56, 12)
    .fill({ color: 0xffffff, alpha: active ? 0.002 : 0 });
  hitPad.eventMode = active ? "static" : "none";
  hitPad.cursor = cursor;
  if (port.role === "source") {
    hitPad.on("pointerdown", (event) => handlers.onStart(connection, port.point, event));
  } else {
    hitPad.on("pointerup", (event) => handlers.onComplete(connection, event));
  }
  if (active) {
    hitPad.on("pointerover", () => {
      showTooltip?.(
        port.role === "source" ? `Drag ${port.label} output` : `Release on ${port.label} input`,
        { x: port.point.x, y: port.point.y - 34 }
      );
    });
    hitPad.on("pointerout", () => hideTooltip?.());
  }
  group.addChild(hitPad);

  if (active) {
    const focus = new Graphics();
    focus.circle(port.point.x, port.point.y, 21).stroke({ width: 2, color, alpha: 0.24 });
    focus.circle(port.point.x, port.point.y, 15).stroke({ width: 1.4, color, alpha: 0.46 });
    group.addChild(focus);
  }

  const shell = new Graphics();
  shell
    .circle(port.point.x, port.point.y, connected ? 8 : active ? 10 : 5)
    .fill({ color: connected ? 0x0f2f2d : 0x061629, alpha: 1 })
    .stroke({ width: connected ? 2.4 : active ? 2.6 : 1.3, color, alpha: active || connected ? 1 : 0.42 });
  group.addChild(shell);

  if (active) {
    const halo = new Graphics();
    halo.circle(port.point.x, port.point.y, 27).stroke({ width: 1.2, color, alpha: 0.28 });
    group.addChild(halo);
  }

  const labelX = port.role === "source" ? port.point.x + 16 : port.point.x - 16;
  const label = makeText(port.label, 10, connected ? 0xd9f99d : active ? 0xe0f2fe : 0x8aa6bf, "900");
  label.anchor.set(port.role === "source" ? 0 : 1, 0.5);
  label.x = labelX;
  label.y = port.point.y;
  group.addChild(label);
  layer.addChild(group);
}

function drawCurvedConnection(layer: Container, from: Point, to: Point, color: number, alpha: number, width: number) {
  const midX = Math.max(from.x + 46, (from.x + to.x) / 2);
  const path = new Graphics();
  path.moveTo(from.x, from.y);
  path.bezierCurveTo(midX, from.y, midX, to.y, to.x, to.y).stroke({
    width,
    color,
    alpha,
    cap: "round",
    join: "round"
  });
  layer.addChild(path);
}

function curvedMidpoint(from: Point, to: Point): Point {
  const midX = Math.max(from.x + 46, (from.x + to.x) / 2);
  const t = 0.5;
  const oneMinus = 1 - t;
  return {
    x: oneMinus ** 3 * from.x + 3 * oneMinus ** 2 * t * midX + 3 * oneMinus * t ** 2 * midX + t ** 3 * to.x,
    y: oneMinus ** 3 * from.y + 3 * oneMinus ** 2 * t * from.y + 3 * oneMinus * t ** 2 * to.y + t ** 3 * to.y
  };
}

function connectionColor(connection: CanvasConnectionOverlay) {
  if (connection.state === "connected") return 0x22c55e;
  return connection.kind === "contract" ? 0xfbbf24 : 0x38bdf8;
}

function buildRepairSlotBounds(overlays: RepairSlotOverlay[], nodeLookup: Map<string, TensorNode>) {
  const bounds = new Map<string, Rectangle>();
  const grouped = new Map<string, RepairSlotOverlay[]>();
  overlays.forEach((overlay) => {
    grouped.set(overlay.nodeId, [...(grouped.get(overlay.nodeId) ?? []), overlay]);
  });

  grouped.forEach((slots, nodeId) => {
    const node = nodeLookup.get(nodeId);
    if (!node) return;
    const slotWidth = Math.max(82, Math.min(118, (node.w + 54) / Math.max(slots.length, 1)));
    const startX = node.x + node.w / 2 - (slots.length * slotWidth + (slots.length - 1) * 8) / 2;
    const y = node.y + node.h + 18;
    slots.forEach((slot, index) => {
      bounds.set(slot.slotId, new Rectangle(startX + index * (slotWidth + 8), y, slotWidth, 45));
    });
  });

  return bounds;
}

function findRepairSlotAtPoint(point: Point, bounds: Map<string, Rectangle>) {
  for (const [slotId, rect] of bounds.entries()) {
    if (rect.contains(point.x, point.y)) return slotId;
  }
  return undefined;
}

function drawCanvasTaskHint(
  layer: Container,
  hint: CanvasTaskHint,
  nodeLookup: Map<string, TensorNode>,
  slotBounds: Map<string, Rectangle>,
  stageKnowledge?: CanvasStageKnowledge
) {
  const rect = resolveTaskHintRect(hint, nodeLookup, slotBounds, stageKnowledge);
  if (!rect) return;

  const color = 0xfbbf24;
  const outer = new Graphics();
  outer
    .roundRect(rect.x - 12, rect.y - 12, rect.width + 24, rect.height + 24, 14)
    .fill({ color, alpha: 0.08 })
    .stroke({ width: 3, color, alpha: 0.94 });
  outer
    .roundRect(rect.x - 20, rect.y - 20, rect.width + 40, rect.height + 40, 18)
    .stroke({ width: 1.4, color, alpha: 0.38 });
  layer.addChild(outer);

  const calloutWidth = 292;
  const calloutHeight = 76;
  const placeLeft = rect.x + rect.width + calloutWidth + 44 > worldBounds.x + worldBounds.width;
  const calloutX = placeLeft ? rect.x - calloutWidth - 24 : rect.x + rect.width + 24;
  const calloutY = Math.max(worldBounds.y + 24, rect.y - 16);
  const pointerStart = {
    x: placeLeft ? rect.x - 6 : rect.x + rect.width + 6,
    y: rect.y + rect.height / 2
  };
  const pointerEnd = {
    x: placeLeft ? calloutX + calloutWidth : calloutX,
    y: calloutY + calloutHeight / 2
  };

  drawSmallArrow(layer, pointerStart, pointerEnd, color);

  const card = new Graphics();
  card
    .roundRect(calloutX, calloutY, calloutWidth, calloutHeight, 10)
    .fill({ color: 0x07111f, alpha: 0.96 })
    .stroke({ width: 1.6, color, alpha: 0.86 });
  layer.addChild(card);
  addText(layer, "Operation Hint", calloutX + 14, calloutY + 17, 10, 0xfde68a, "900", 0);
  addText(layer, compactCanvasText(hint.title, 31), calloutX + 14, calloutY + 38, 12, 0xe8f2ff, "900", 0);
  addText(layer, compactCanvasText(hint.detail, 48), calloutX + 14, calloutY + 59, 10, 0xb7c7dc, "800", 0);
}

function resolveTaskHintRect(
  hint: CanvasTaskHint,
  nodeLookup: Map<string, TensorNode>,
  slotBounds: Map<string, Rectangle>,
  stageKnowledge?: CanvasStageKnowledge
) {
  if (hint.slotId && stageKnowledge?.visual === "tensor_objects") {
    const item = tensorObjectDragItems.find((candidate) => candidate.slotId === hint.slotId);
    if (item) return new Rectangle(item.x, item.y, item.width, item.height);
  }

  if (hint.slotId) {
    const slotRect = slotBounds.get(hint.slotId);
    if (slotRect) return slotRect;
  }

  if (hint.nodeId) {
    const node = nodeLookup.get(hint.nodeId);
    if (node) return new Rectangle(node.x - 8, node.y - 8, node.w + 16, node.h + 16);
  }

  return undefined;
}

function compactCanvasText(text: string, maxLength: number) {
  return text.length <= maxLength ? text : `${text.slice(0, Math.max(0, maxLength - 1))}…`;
}

function findNearestConnectableConnectionStart(
  point: Point,
  connections: CanvasConnectionOverlay[],
  ports: ConnectionPortMap
): { connection: CanvasConnectionOverlay; from: Point } | null {
  let nearest: { connection: CanvasConnectionOverlay; from: Point; distance: number } | null = null;

  for (const connection of connections) {
    if (!connection.enabled || connection.state !== "open") continue;
    const endpoints = ports.get(connection.id);
    if (!endpoints) continue;
    const endpointDistance = Math.min(distance(point, endpoints.source.point), distance(point, endpoints.target.point));
    if (endpointDistance > connectionSnapRadius + 18) continue;
    if (!nearest || endpointDistance < nearest.distance) {
      nearest = {
        connection,
        from: endpoints.source.point,
        distance: endpointDistance
      };
    }
  }

  return nearest ? { connection: nearest.connection, from: nearest.from } : null;
}

function drawRepairSlots(
  layer: Container,
  overlays: RepairSlotOverlay[],
  nodeLookup: Map<string, TensorNode>,
  slotHints: Map<string, CanvasActionHint>,
  onSlotSelect?: (slotId: string) => void,
  onSlotContextMenu?: (event: FederatedPointerEvent, slotId: string) => void,
  showTooltip?: (text: string, point: Point) => void,
  hideTooltip?: () => void
) {
  const grouped = new Map<string, RepairSlotOverlay[]>();
  overlays.forEach((overlay) => {
    grouped.set(overlay.nodeId, [...(grouped.get(overlay.nodeId) ?? []), overlay]);
  });

  grouped.forEach((slots, nodeId) => {
    const node = nodeLookup.get(nodeId);
    if (!node) return;

    const slotWidth = Math.max(82, Math.min(118, (node.w + 54) / Math.max(slots.length, 1)));
    const startX = node.x + node.w / 2 - (slots.length * slotWidth + (slots.length - 1) * 8) / 2;
    const y = node.y + node.h + 18;

    slots.forEach((slot, index) => {
      const x = startX + index * (slotWidth + 8);
      const group = new Container({ label: `repair-slot-${slot.slotId}` });
      group.eventMode = "static";
      group.cursor = "pointer";
      group.hitArea = new Rectangle(x, y, slotWidth, 45);
      group.on("pointertap", (event) => {
        event.stopPropagation();
        onSlotSelect?.(slot.slotId);
      });
      group.on("rightdown", (event) => {
        onSlotContextMenu?.(event, slot.slotId);
      });

      const active = slot.state === "active";
      const filled = slot.state === "filled";
      const shell = new Graphics();
      shell
        .roundRect(x, y, slotWidth, 45, 7)
        .fill({ color: active ? 0x123052 : filled ? 0x0f2f2d : 0x07111f, alpha: 0.96 })
        .stroke({ width: active ? 2 : 1.2, color: active ? 0xfbbf24 : filled ? 0x22c55e : 0x60a5fa, alpha: active ? 1 : 0.82 });
      group.addChild(shell);

      addText(group, slot.label, x + slotWidth / 2, y + 14, 10, 0xa8c7e8, "800", 0.5);
      addText(group, slot.value, x + slotWidth / 2, y + 32, 13, filled || active ? 0xe8f2ff : 0x94a3b8, "900", 0.5);
      const hint = slotHints.get(slot.slotId);
      if (hint) {
        drawActionBadge(
          group,
          x + slotWidth - 10,
          y - 8,
          hint,
          (event) => {
            event.stopPropagation();
            onSlotSelect?.(slot.slotId);
          },
          showTooltip,
          hideTooltip
        );
      }
      layer.addChild(group);
    });
  });
}

function drawActionBadge(
  layer: Container,
  x: number,
  y: number,
  hint: CanvasActionHint,
  onOpen: (event: FederatedPointerEvent) => void,
  showTooltip?: (text: string, point: Point) => void,
  hideTooltip?: () => void
) {
  const badge = new Container({ label: `action-${hint.kind}-${hint.id}` });
  const color = hint.icon === "run" ? 0x22c55e : hint.icon === "probe" ? 0x38bdf8 : 0x93c5fd;
  const label = hint.icon === "run" ? "run" : hint.icon === "probe" ? "probe" : "tag";
  const width = Math.max(34, label.length * 8 + 18);

  badge.eventMode = "static";
  badge.cursor = "pointer";
  badge.hitArea = new Rectangle(x - width / 2 - 8, y - 16, width + 16, 32);
  badge.on("pointertap", onOpen);
  badge.on("rightdown", onOpen);
  badge.on("pointerover", () => showTooltip?.(hint.tooltip, { x, y: y - 30 }));
  badge.on("pointerout", () => hideTooltip?.());

  if (hint.pulse) {
    const pulse = new Graphics();
    pulse.roundRect(x - width / 2 - 5, y - 13, width + 10, 26, 999).stroke({ width: 2, color, alpha: 0.24 });
    badge.addChild(pulse);
  }

  const shell = new Graphics();
  shell
    .roundRect(x - width / 2, y - 11, width, 22, 999)
    .fill({ color: 0x07111f, alpha: 0.96 })
    .stroke({ width: 1.6, color, alpha: 0.9 });
  badge.addChild(shell);

  const text = makeText(label, label.length > 4 ? 8 : 9, 0xe8f2ff, "900");
  text.anchor.set(0.5, 0.5);
  text.x = x;
  text.y = y - 1;
  badge.addChild(text);
  layer.addChild(badge);
}

function drawTooltip(layer: Container, text: string, x: number, y: number) {
  const label = makeText(text, 10, 0xe8f2ff, "800");
  label.anchor.set(0, 0.5);
  const width = Math.min(230, Math.max(86, label.width + 18));
  const height = 28;
  const left = x + 12;
  const top = y - height / 2;
  const shell = new Graphics();
  shell
    .roundRect(left, top, width, height, 7)
    .fill({ color: 0x07111f, alpha: 0.96 })
    .stroke({ width: 1, color: 0x60a5fa, alpha: 0.72 });
  label.x = left + 9;
  label.y = y;
  layer.addChild(shell, label);
}

function clearLayer(layer: Container) {
  for (const child of layer.removeChildren()) {
    child.destroy({ children: true });
  }
}

type TensorObjectDragItem = {
  slotId: string;
  tagId: string;
  title: string;
  caption: string;
  kind: "scalar" | "vector" | "matrix" | "block";
  x: number;
  y: number;
  width: number;
  height: number;
};

const tensorObjectDragItems: TensorObjectDragItem[] = [
  { slotId: "object_scalar", tagId: "object_scalar", title: "Scalar", caption: "one number", kind: "scalar", x: 82, y: 235, width: 130, height: 82 },
  { slotId: "object_vector", tagId: "object_vector", title: "Vector", caption: "one row", kind: "vector", x: 226, y: 235, width: 130, height: 82 },
  { slotId: "object_matrix", tagId: "object_matrix", title: "Matrix", caption: "number grid", kind: "matrix", x: 82, y: 334, width: 130, height: 82 },
  { slotId: "object_block", tagId: "object_block", title: "3D Block", caption: "stacked grids", kind: "block", x: 226, y: 334, width: 130, height: 82 }
];

function drawTensorObjectDragChallenge(
  layer: Container,
  slots: RepairSlotOverlay[],
  nodeLookup: Map<string, TensorNode>,
  onDragStart: (event: FederatedPointerEvent, group: Container, item: TensorObjectDragItem, position: Point) => void
) {
  const inspector = nodeLookup.get("tensor_inspector");
  if (!inspector) return;

  const slotById = new Map(slots.map((slot) => [slot.slotId, slot]));
  const acceptedItems = tensorObjectDragItems.filter((item) => isTensorObjectAccepted(slotById.get(item.slotId)));
  const dropRect = tensorInspectorDropRect(inspector);
  const target = new Graphics();
  target
    .roundRect(dropRect.x, dropRect.y, dropRect.width, dropRect.height, 12)
    .fill({ color: 0x0f2740, alpha: 0.14 })
    .stroke({ width: 2, color: 0x7dd3fc, alpha: 0.45 });
  target.roundRect(inspector.x + 16, inspector.y + 16, inspector.w - 32, 30, 8).fill({ color: 0x07111f, alpha: 0.78 }).stroke({ width: 1, color: 0x315f94, alpha: 0.7 });
  layer.addChild(target);
  addText(layer, "Single Tensor Inspector", inspector.x + inspector.w / 2, inspector.y + 31, 11, 0xbfdbfe, "900", 0.5);

  drawObjectBench(layer);
  drawTensorObjectInspectorNote(layer, inspector, acceptedItems.length);
  drawRecognitionLog(layer, inspector, acceptedItems);

  tensorObjectDragItems.forEach((item) => {
    if (isTensorObjectAccepted(slotById.get(item.slotId))) return;
    const card = drawTensorObjectCard(item, false);
    card.eventMode = "static";
    card.cursor = "grab";
    card.hitArea = new Rectangle(0, 0, item.width, item.height);
    card.on("pointerdown", (event) => {
      if (event.button === 2) return;
      onDragStart(event, card, item, { x: card.x, y: card.y });
    });
    layer.addChild(card);
  });
}

function tensorInspectorDropRect(node: TensorNode) {
  return new Rectangle(node.x - 26, node.y - 26, node.w + 52, node.h + 174);
}

function isTensorObjectAccepted(slot?: RepairSlotOverlay) {
  return Boolean(slot && slot.value !== "open" && (slot.state === "filled" || slot.state === "active"));
}

function drawObjectBench(layer: Container) {
  const x = 60;
  const y = 194;
  const width = 326;
  const height = 248;
  const shell = new Graphics();
  shell
    .roundRect(x, y, width, height, 10)
    .fill({ color: 0x06111f, alpha: 0.58 })
    .stroke({ width: 1.2, color: 0x315f94, alpha: 0.42 });
  layer.addChild(shell);
  addText(layer, "Object Bench", x + 18, y + 18, 13, 0xe8f2ff, "900", 0);
  addText(layer, "candidates", x + 18, y + 38, 10, 0x9db2ca, "800", 0);
}

function drawTensorObjectInspectorNote(layer: Container, inspector: TensorNode, acceptedCount: number) {
  const x = inspector.x + 18;
  const y = inspector.y + 58;
  const width = inspector.w - 36;
  const height = 94;
  const rows = acceptedCount
    ? ["port=object intake", "output=tensor nodes", "status=accepted"]
    : ["port=idle", "input=drop object", "output=tensor node"];

  const shell = new Graphics();
  shell
    .roundRect(x, y, width, height, 9)
    .fill({ color: 0x06111f, alpha: 0.92 })
    .stroke({ width: 1.1, color: acceptedCount ? 0x22c55e : 0x315f94, alpha: acceptedCount ? 0.68 : 0.58 });
  layer.addChild(shell);
  addText(layer, "Port State", x + 12, y + 16, 10, 0x93c5fd, "900", 0);
  rows.forEach((row, index) => {
    addText(layer, row, x + 12, y + 38 + index * 16, 11, index === 2 && acceptedCount ? 0xbbf7d0 : 0xdbeafe, "800", 0);
  });
  addText(layer, `checked ${acceptedCount}/4`, x + 12, y + height - 13, 10, acceptedCount === 4 ? 0xbbf7d0 : 0x9db2ca, "900", 0);
}

function drawRecognitionLog(layer: Container, inspector: TensorNode, acceptedItems: TensorObjectDragItem[]) {
  const x = inspector.x + 12;
  const y = inspector.y + inspector.h + 24;
  const width = inspector.w - 24;
  const height = 98;
  const shell = new Graphics();
  shell
    .roundRect(x, y, width, height, 8)
    .fill({ color: 0x07111f, alpha: 0.9 })
    .stroke({ width: 1.2, color: acceptedItems.length ? 0x22c55e : 0x315f94, alpha: acceptedItems.length ? 0.7 : 0.5 });
  layer.addChild(shell);
  addText(layer, "Recognition Log", x + 12, y + 15, 9, 0x93c5fd, "900", 0);
  addText(layer, `${acceptedItems.length}/4`, x + width - 12, y + 15, 8, 0x8aa6bf, "800", 1);

  if (!acceptedItems.length) {
    addText(layer, "empty", x + 12, y + 42, 10, 0x8aa6bf, "800", 0);
    return;
  }

  acceptedItems.forEach((item, index) => {
    const rowY = y + 38 + index * 14;
    addText(layer, "✓", x + 13, rowY, 10, 0x22c55e, "900", 0);
    addText(layer, item.title, x + 32, rowY, 9, 0xe8f2ff, "900", 0);
    addText(layer, "accepted", x + 122, rowY, 8, 0x9db2ca, "800", 0);
  });
}

function drawTensorObjectCard(item: TensorObjectDragItem, accepted: boolean) {
  const group = new Container({ label: `tensor-object-${item.slotId}` });
  group.x = item.x;
  group.y = item.y;
  const shell = new Graphics();
  shell
    .roundRect(0, 0, item.width, item.height, 9)
    .fill({ color: accepted ? 0x08251f : 0x0a1524, alpha: 0.96 })
    .stroke({ width: accepted ? 1.6 : 1.3, color: accepted ? 0x22c55e : 0x60a5fa, alpha: accepted ? 0.88 : 0.68 });
  group.addChild(shell);
  addText(group, item.title, 12, 16, 12, 0xe8f2ff, "900", 0);
  addText(group, item.caption, 12, 34, 9, 0x9db2ca, "800", 0);
  drawTensorObjectCardGlyph(group, item.kind, 72, 52);
  if (accepted) {
    const stamp = new Graphics();
    stamp.circle(item.width - 18, 18, 10).fill({ color: 0x08251f, alpha: 0.96 }).stroke({ width: 1.4, color: 0x22c55e, alpha: 1 });
    stamp.moveTo(item.width - 23, 18).lineTo(item.width - 19, 22).lineTo(item.width - 13, 14).stroke({ width: 1.8, color: 0x22c55e, alpha: 1, cap: "round", join: "round" });
    group.addChild(stamp);
  }
  return group;
}

function drawTensorObjectCardGlyph(layer: Container, kind: TensorObjectDragItem["kind"], x: number, y: number) {
  if (kind === "scalar") {
    drawScalarGlyph(layer, x - 10, y, "3.14");
    return;
  }
  if (kind === "vector") {
    drawVectorGlyph(layer, x - 28, y - 10, 5, 0x7dd3fc);
    return;
  }
  if (kind === "matrix") {
    drawMatrixGlyph(layer, x - 23, y - 21, 3, 4, 8, 0x38bdf8);
    return;
  }
  drawStackedMatrixGlyph(layer, x - 27, y - 17, 0x38bdf8);
}

function drawCuboid(layer: Container, node: TensorNode, selected: boolean) {
  const depth = node.kind === "attention" ? 30 : 24;
  const top = new Graphics();
  top
    .poly([node.x, node.y, node.x + depth, node.y - depth, node.x + node.w + depth, node.y - depth, node.x + node.w, node.y], true)
    .fill({ color: tint(node.color, 1.28), alpha: 0.9 })
    .stroke({ width: selected ? 2.5 : 1.2, color: selected ? 0xfbbf24 : 0x7dd3fc, alpha: 0.92 });

  const side = new Graphics();
  side
    .poly(
      [
        node.x + node.w,
        node.y,
        node.x + node.w + depth,
        node.y - depth,
        node.x + node.w + depth,
        node.y + node.h - depth,
        node.x + node.w,
        node.y + node.h
      ],
      true
    )
    .fill({ color: tint(node.color, 0.56), alpha: 0.92 })
    .stroke({ width: selected ? 2.5 : 1.2, color: selected ? 0xfbbf24 : 0x7dd3fc, alpha: 0.92 });

  const front = new Graphics();
  front
    .roundRect(node.x, node.y, node.w, node.h, 8)
    .fill({ color: node.color, alpha: 0.9 })
    .stroke({ width: selected ? 3 : 1.4, color: selected ? 0xfbbf24 : 0x7dd3fc, alpha: 0.92 });

  layer.addChild(top, side, front);
}

function drawVectorTensorShell(layer: Container, node: TensorNode, selected: boolean) {
  const shell = new Graphics();
  shell
    .roundRect(node.x, node.y, node.w, node.h, 8)
    .fill({ color: node.color, alpha: 0.9 })
    .stroke({ width: selected ? 3 : 1.4, color: selected ? 0xfbbf24 : 0x7dd3fc, alpha: selected ? 1 : 0.88 });
  layer.addChild(shell);

  const axis = new Graphics();
  const y = node.y + node.h - 24;
  axis
    .moveTo(node.x + 26, y)
    .lineTo(node.x + node.w - 26, y)
    .stroke({ width: 2, color: 0x7dd3fc, alpha: 0.42, cap: "round" });
  axis.circle(node.x + 26, y, 3).fill({ color: 0x7dd3fc, alpha: 0.76 });
  axis.circle(node.x + node.w - 26, y, 3).fill({ color: 0x7dd3fc, alpha: 0.76 });
  layer.addChild(axis);
}

function drawPort(layer: Container, x: number, y: number, selected: boolean, direction: "in" | "out") {
  const port = new Graphics();
  port
    .circle(x, y, selected ? 6 : 5)
    .fill({ color: 0x061629, alpha: 1 })
    .stroke({ width: 1.6, color: selected ? 0xfbbf24 : direction === "in" ? 0x60a5fa : 0x22c55e, alpha: 1 });
  layer.addChild(port);
}

function drawPortLabels(layer: Container, node: TensorNode) {
  addText(layer, "in", node.x - 10, node.y + node.h / 2 - 18, 10, 0x93c5fd, "700", 0.5);
  addText(layer, "out", node.x + node.w + 10, node.y + node.h / 2 - 18, 10, 0x86efac, "700", 0.5);
}

function drawEdge(edgeLayer: Container, labelLayer: Container, edge: TensorEdge, mode: WorkbenchMode, nodeLookup: Map<string, TensorNode>) {
  const points = getEdgePoints(edge, nodeLookup);
  if (points.length < 2) return;

  const path = new Graphics();
  path.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) {
    path.lineTo(point.x, point.y);
  }
  path.stroke({
    width: edge.flow === "gradient" ? 2.2 : 2.6,
    color: edge.color,
    alpha: edge.flow === "gradient" ? 0.72 : 0.88,
    cap: "round",
    join: "round"
  });
  edgeLayer.addChild(path);

  const end = points[points.length - 1];
  const beforeEnd = points[points.length - 2];
  drawArrow(labelLayer, end, beforeEnd, edge.color);

  if (mode !== "build" && edge.flow !== "gradient") {
    const midpoint = pointAtPath(points, 0.5);
    drawDataLabel(labelLayer, edge.label, midpoint.x, midpoint.y - 20, edge.color);
  }
}

function getEdgePoints(edge: TensorEdge, nodeLookup: Map<string, TensorNode>): Point[] {
  const from = nodeLookup.get(edge.from);
  const to = nodeLookup.get(edge.to);
  if (!from || !to) return [];

  const start = { x: from.x + from.w + 9, y: from.y + from.h / 2 };
  const end = { x: to.x - 9, y: to.y + to.h / 2 };

  if (edge.route === "loop") {
    return [
      start,
      { x: start.x + 48, y: start.y },
      { x: start.x + 48, y: 596 },
      { x: to.x + to.w / 2, y: 596 },
      { x: to.x + to.w / 2, y: to.y + to.h + 14 }
    ];
  }

  if (edge.route === "down") {
    const downStart = { x: from.x + from.w / 2, y: from.y + from.h + 9 };
    const downEnd = { x: to.x + to.w / 2, y: to.y - 9 };
    const bendY = (downStart.y + downEnd.y) / 2;
    return [downStart, { x: downStart.x, y: bendY }, { x: downEnd.x, y: bendY }, downEnd];
  }

  if (Math.abs(start.y - end.y) > 24) {
    const midX = (start.x + end.x) / 2;
    return [start, { x: midX, y: start.y }, { x: midX, y: end.y }, end];
  }

  return [start, end];
}

function drawDataLabel(layer: Container, label: string, x: number, y: number, color: number) {
  const text = makeText(label, 11, 0xdbeafe, "700");
  const width = Math.max(92, text.width + 16);
  const pill = new Graphics();
  pill
    .roundRect(x - width / 2, y - 13, width, 25, 5)
    .fill({ color: 0x0b1728, alpha: 0.92 })
    .stroke({ width: 1, color, alpha: 0.45 });
  text.x = x - text.width / 2;
  text.y = y - 8;
  layer.addChild(pill, text);
}

function drawArrow(layer: Container, end: Point, beforeEnd: Point, color: number) {
  const angle = Math.atan2(end.y - beforeEnd.y, end.x - beforeEnd.x);
  const size = 10;
  const p1 = {
    x: end.x - Math.cos(angle - Math.PI / 6) * size,
    y: end.y - Math.sin(angle - Math.PI / 6) * size
  };
  const p2 = {
    x: end.x - Math.cos(angle + Math.PI / 6) * size,
    y: end.y - Math.sin(angle + Math.PI / 6) * size
  };
  const arrow = new Graphics();
  arrow.poly([end.x, end.y, p1.x, p1.y, p2.x, p2.y], true).fill({ color, alpha: 0.95 });
  layer.addChild(arrow);
}

function drawMiniCells(layer: Container, x: number, y: number, columns: number, rows: number, size: number, color: number) {
  const cells = new Graphics();
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < columns; col += 1) {
      const alpha = 0.24 + ((row + col) % 3) * 0.18;
      cells.roundRect(x + col * (size + 3), y + row * (size + 3), size, size, 2).fill({ color, alpha });
    }
  }
  layer.addChild(cells);
}

function drawMiniHeatmap(layer: Container, x: number, y: number, emphasizeLeak: boolean) {
  const colors = [0x0f2740, 0x145374, 0x1d9bf0, 0xfbbf24, 0xfb7185];
  for (let head = 0; head < 3; head += 1) {
    const board = new Graphics();
    const offset = head * 7;
    board.roundRect(x + offset, y - offset, 60, 60, 4).fill({ color: 0x0f172a, alpha: 0.96 }).stroke({ width: 1, color: 0x334155, alpha: 1 });
    for (let row = 0; row < 4; row += 1) {
      for (let col = 0; col < 4; col += 1) {
        const blocked = col > row;
        const leak = emphasizeLeak && row === 2 && col === 3;
        const valueColor = leak ? 0xef4444 : blocked ? 0x475569 : colors[(row + col + head) % colors.length];
        board.roundRect(x + offset + 7 + col * 11, y - offset + 7 + row * 11, 9, 9, 2).fill({ color: valueColor, alpha: blocked && !leak ? 0.32 : 0.88 });
      }
    }
    layer.addChild(board);
  }
}

function drawMaskGlyph(layer: Container, x: number, y: number) {
  const glyph = new Graphics();
  glyph.roundRect(x, y, 42, 42, 4).fill({ color: 0x0f172a, alpha: 0.86 }).stroke({ width: 1, color: 0x64748b, alpha: 1 });
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      const debug = row === 2 && col === 3;
      const blocked = col > row;
      glyph.roundRect(x + 5 + col * 8, y + 5 + row * 8, 6, 6, 1).fill({
        color: debug ? 0xef4444 : blocked ? 0x64748b : 0x22c55e,
        alpha: debug ? 0.95 : blocked ? 0.34 : 0.82
      });
    }
  }
  layer.addChild(glyph);
}

function drawGauge(layer: Container, x: number, y: number) {
  const gauge = new Graphics();
  gauge.circle(x, y, 12).fill({ color: 0x0f172a, alpha: 0.9 }).stroke({ width: 1.6, color: 0xfbbf24, alpha: 0.9 });
  gauge.moveTo(x, y).lineTo(x + 7, y - 6).stroke({ width: 1.6, color: 0xfbbf24, alpha: 1, cap: "round" });
  layer.addChild(gauge);
}

function drawTrainOverlay(layer: Container) {
  const panel = new Graphics();
  panel
    .roundRect(60, 484, 388, 70, 8)
    .fill({ color: 0x1c1917, alpha: 0.86 })
    .stroke({ width: 1.2, color: 0xf59e0b, alpha: 0.8 });
  layer.addChild(panel);
  addText(layer, "Optimizer update arm", 82, 512, 15, 0xffedd5, "800", 0);
  addText(layer, "grad_norm=0.82  lr=3e-4  tokens/sec=18.4k", 82, 539, 12, 0xfbbf24, "700", 0);
}

function drawPulses(layer: Graphics, edges: TensorEdge[], elapsedMS: number, active: boolean, nodeLookup: Map<string, TensorNode>) {
  layer.clear();
  if (!active) return;

  edges.forEach((edge, index) => {
    const points = getEdgePoints(edge, nodeLookup);
    if (points.length < 2) return;
    const speed = edge.flow === "gradient" ? 0.00034 : 0.00048;
    const phase = (elapsedMS * speed + index * 0.16) % 1;
    const p = pointAtPath(points, phase);
    layer.circle(p.x, p.y, edge.flow === "gradient" ? 4.2 : 5).fill({ color: edge.color, alpha: 0.96 });
    layer.circle(p.x, p.y, edge.flow === "gradient" ? 8 : 9).stroke({ width: 1.1, color: edge.color, alpha: 0.36 });
  });
}

function pointAtPath(points: Point[], t: number): Point {
  const segments = [];
  let total = 0;
  for (let i = 0; i < points.length - 1; i += 1) {
    const a = points[i];
    const b = points[i + 1];
    const length = Math.hypot(b.x - a.x, b.y - a.y);
    segments.push({ a, b, length });
    total += length;
  }

  let distance = total * t;
  for (const segment of segments) {
    if (distance <= segment.length) {
      const local = segment.length === 0 ? 0 : distance / segment.length;
      return {
        x: segment.a.x + (segment.b.x - segment.a.x) * local,
        y: segment.a.y + (segment.b.y - segment.a.y) * local
      };
    }
    distance -= segment.length;
  }

  return points[points.length - 1];
}

function addText(layer: Container, text: string, x: number, y: number, fontSize: number, fill: number, fontWeight: FontWeight, anchorX: number) {
  const item = makeText(text, fontSize, fill, fontWeight);
  item.x = x;
  item.y = y;
  item.anchor.set(anchorX, 0.5);
  layer.addChild(item);
  return item;
}

function makeText(text: string, fontSize: number, fill: number, fontWeight: FontWeight) {
  return new Text({
    text,
    resolution: textTextureResolution(),
    roundPixels: true,
    autoGenerateMipmaps: true,
    style: {
      fill,
      fontFamily: 'Fira Code, Consolas, "Microsoft YaHei", "Segoe UI", ui-monospace, monospace',
      fontSize,
      fontWeight,
      letterSpacing: 0
    }
  });
}

function textTextureResolution() {
  if (typeof window === "undefined") return 2;
  return Math.min(Math.max(window.devicePixelRatio || 1, 2.5), 4);
}

function snapCanvasPixel(value: number) {
  return Math.round(value);
}

function tint(color: number, factor: number) {
  const r = Math.max(0, Math.min(255, Math.round(((color >> 16) & 255) * factor)));
  const g = Math.max(0, Math.min(255, Math.round(((color >> 8) & 255) * factor)));
  const b = Math.max(0, Math.min(255, Math.round((color & 255) * factor)));
  return (r << 16) + (g << 8) + b;
}

function getClientPoint(event: FederatedPointerEvent, app: Application): Point {
  const native = event.nativeEvent;
  if (native && "clientX" in native && "clientY" in native) {
    return {
      x: Number(native.clientX),
      y: Number(native.clientY)
    };
  }
  const rect = app.canvas.getBoundingClientRect();
  return {
    x: rect.left + event.global.x,
    y: rect.top + event.global.y
  };
}

function clientToScreenPoint(clientX: number, clientY: number, rect: DOMRect, app: Application): Point {
  return {
    x: ((clientX - rect.left) / Math.max(rect.width, 1)) * app.screen.width,
    y: ((clientY - rect.top) / Math.max(rect.height, 1)) * app.screen.height
  };
}

function screenToWorld(point: Point, view: ViewState): Point {
  return {
    x: (point.x - view.x) / view.scale,
    y: (point.y - view.y) / view.scale
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
