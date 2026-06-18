import { useEffect, useRef, type MutableRefObject } from "react";
import { Application, Container, Graphics, Rectangle, Text, type FederatedPointerEvent, type Ticker } from "pixi.js";
import { sceneSize } from "./sceneData";
import type { TensorEdge, TensorNode, WorkbenchMode } from "./workbenchTypes";

const connectionSnapRadius = 44;
const minCanvasScale = 0.42;
const maxCanvasScale = 1.7;
const worldBounds = new Rectangle(-2400, -1800, 5600, 4200);

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
  icon: "menu" | "probe" | "run" | "wire";
  tooltip: string;
  pulse?: boolean;
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
  onSlotSelect?: (slotId: string) => void;
  onCanvasConnect?: (slotId: string, tagId: string) => void;
  onCanvasContextMenu?: (target: CanvasContextTarget) => void;
  onNodeMove?: (nodeId: string, x: number, y: number) => void;
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
  onSlotSelect,
  onCanvasConnect,
  onCanvasContextMenu,
  onNodeMove,
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
    onSlotSelect,
    onCanvasConnect,
    onCanvasContextMenu,
    onNodeMove,
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
      onSlotSelect,
      onCanvasConnect,
      onCanvasContextMenu,
      onNodeMove,
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
    onSlotSelect,
    onCanvasConnect,
    onCanvasContextMenu,
    onNodeMove,
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
        resolution: Math.min(window.devicePixelRatio || 1, 2),
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
    onSlotSelect?: (slotId: string) => void;
    onCanvasConnect?: (slotId: string, tagId: string) => void;
    onCanvasContextMenu?: (target: CanvasContextTarget) => void;
    onNodeMove?: (nodeId: string, x: number, y: number) => void;
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
    view.x = screenWidth / 2 - 555 * view.scale;
    view.y = screenHeight / 2 - 270 * view.scale;
    view.initialized = true;
  }

  const scene = new Container({ label: "tensor-workbench-scene", sortableChildren: true });
  scene.scale.set(view.scale);
  scene.x = view.x;
  scene.y = view.y;
  app.stage.addChild(scene);

  const gridLayer = new Graphics({ label: "infinite-grid" });
  const edgeLayer = new Container({ label: "typed-data-lines" });
  const connectionLayer = new Container({ label: "repair-connections" });
  const nodeLayer = new Container({ label: "tensor-nodes" });
  const overlayLayer = new Container({ label: "labels-and-overlays" });
  const previewLayer = new Graphics({ label: "connection-preview" });
  const pulseLayer = new Graphics({ label: "flow-pulses" });
  const tooltipLayer = new Container({ label: "hover-tooltips" });
  gridLayer.eventMode = "none";
  edgeLayer.eventMode = "none";
  connectionLayer.eventMode = "none";
  nodeLayer.eventMode = "static";
  overlayLayer.eventMode = "static";
  previewLayer.eventMode = "none";
  pulseLayer.eventMode = "none";
  tooltipLayer.eventMode = "none";
  scene.addChild(gridLayer, edgeLayer, connectionLayer, nodeLayer, overlayLayer, previewLayer, pulseLayer, tooltipLayer);
  scene.eventMode = "static";
  scene.hitArea = worldBounds;

  drawWorldGrid(gridLayer, worldBounds);

  const nodeLookup = new Map(state.nodes.map((node) => [node.id, node]));
  const nodeHints = new Map(state.actionHints.filter((hint) => hint.kind === "node").map((hint) => [hint.id, hint]));
  const slotHints = new Map(state.actionHints.filter((hint) => hint.kind === "slot").map((hint) => [hint.id, hint]));
  const visibleEdges = state.edges.filter((edge) => edge.flow !== "gradient" || state.mode === "train");
  visibleEdges.forEach((edge) => drawEdge(edgeLayer, overlayLayer, edge, state.mode, nodeLookup));

  const connectionPorts = buildConnectionPorts(state.connectionOverlays, nodeLookup);

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

  const connectCanvasConnection = (connection: CanvasConnectionOverlay) => {
    if (!connection.enabled || connection.state !== "open") return;
    state.onCanvasConnect?.(connection.slotId, connection.tagId);
  };

  drawCanvasConnectionLines(connectionLayer, overlayLayer, state.connectionOverlays, connectionPorts, connectCanvasConnection, showTooltip, hideTooltip);

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
        state.onSelect(node.id);
      },
      (event) => openContextMenu(event, { kind: "node", id: node.id }),
      nodeHints.get(node.id),
      (event) => openContextMenu(event, { kind: "node", id: node.id }),
      showTooltip,
      hideTooltip
    );
  });

  drawRepairSlots(
    overlayLayer,
    state.slotOverlays,
    nodeLookup,
    slotHints,
    state.onSlotSelect,
    (event, slotId) => openContextMenu(event, { kind: "slot", id: slotId }),
    showTooltip,
    hideTooltip
  );
  drawCanvasConnectionPorts(overlayLayer, state.connectionOverlays, connectionPorts, {
    onStart: (connection, point, event) => {
      event.stopPropagation();
      if (!connection.enabled || connection.state === "connected") return;
      connectionDrag = { connection, from: point };
      drawConnectionPreview(point, point, connectionColor(connection), true);
    },
    onComplete: (connection, event) => {
      event.stopPropagation();
      if (connectionDrag?.connection.id === connection.id) {
        state.onCanvasConnect?.(connection.slotId, connection.tagId);
      }
      connectionDrag = null;
      clearConnectionPreview();
    },
    onQuickConnect: (connection, event) => {
      event.stopPropagation();
      if (connection.enabled && connection.state === "open") {
        state.onCanvasConnect?.(connection.slotId, connection.tagId);
      }
    }
  }, showTooltip, hideTooltip);
  if (state.mode === "train") {
    drawTrainOverlay(overlayLayer);
  }

  let elapsed = 0;
  const tick = (ticker: Ticker) => {
    elapsed += ticker.deltaMS;
    drawPulses(pulseLayer, visibleEdges, elapsed, state.playing || state.mode === "train", nodeLookup);
  };
  app.ticker.add(tick);

  const handleGlobalMove = (event: FederatedPointerEvent) => {
    if (connectionDrag) {
      const pointer = event.getLocalPosition(scene);
      const target = connectionPorts.get(connectionDrag.connection.id)?.target;
      const validTarget = Boolean(target && distance(pointer, target.point) <= connectionSnapRadius);
      drawConnectionPreview(connectionDrag.from, pointer, connectionColor(connectionDrag.connection), validTarget);
    }

    if (nodeDrag) {
      const pointer = event.getLocalPosition(scene);
      const dx = pointer.x - nodeDrag.startPointer.x;
      const dy = pointer.y - nodeDrag.startPointer.y;
      nodeDrag.group.x = dx;
      nodeDrag.group.y = dy;
      nodeDrag.moved = Math.abs(dx) > 2 || Math.abs(dy) > 2;
    }

    if (canvasDrag) {
      const global = { x: event.global.x, y: event.global.y };
      const dx = global.x - canvasDrag.startGlobal.x;
      const dy = global.y - canvasDrag.startGlobal.y;
      viewRef.current.x = canvasDrag.startView.x + dx;
      viewRef.current.y = canvasDrag.startView.y + dy;
      canvasDrag.moved = Math.abs(dx) > 2 || Math.abs(dy) > 2;
      syncSceneView();
    }
  };

  const finishGestures = (event?: FederatedPointerEvent) => {
    if (connectionDrag) {
      const pointer = event?.getLocalPosition(scene);
      const target = connectionPorts.get(connectionDrag.connection.id)?.target;
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
        state.onNodeMove?.(nodeDrag.node.id, nextX, nextY);
      }
      nodeDrag.group.x = 0;
      nodeDrag.group.y = 0;
      nodeDrag = null;
    }

    if (canvasDrag) {
      suppressNextTap = canvasDrag.moved;
      canvasDrag = null;
    }
  };

  const handleScenePointerDown = (event: FederatedPointerEvent) => {
    if (event.button === 2) return;
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
    if (nodeDrag || connectionDrag) return;
    const pointer = event.getLocalPosition(scene);
    const connection = findNearestConnectableConnection(pointer, state.connectionOverlays, connectionPorts);
    if (connection) {
      state.onCanvasConnect?.(connection.slotId, connection.tagId);
    }
  };

  const handleSceneRightClick = (event: FederatedPointerEvent) => {
    openContextMenu(event, { kind: "canvas" });
  };

  const handleWheel = (event: WheelEvent) => {
    event.preventDefault();
    const rect = app.canvas.getBoundingClientRect();
    const screenPoint = {
      x: ((event.clientX - rect.left) / Math.max(rect.width, 1)) * app.screen.width,
      y: ((event.clientY - rect.top) / Math.max(rect.height, 1)) * app.screen.height
    };
    const worldPoint = screenToWorld(screenPoint, viewRef.current);
    const nextScale = clamp(viewRef.current.scale * Math.exp(-event.deltaY * 0.0012), minCanvasScale, maxCanvasScale);
    viewRef.current.scale = nextScale;
    viewRef.current.x = screenPoint.x - worldPoint.x * nextScale;
    viewRef.current.y = screenPoint.y - worldPoint.y * nextScale;
    syncSceneView();
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
  hideTooltip?: () => void
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

  if (node.kind === "tensor" || node.kind === "attention") {
    drawCuboid(group, node, selected);
  } else {
    const shell = new Graphics();
    shell
      .roundRect(node.x, node.y, node.w, node.h, 8)
      .fill({ color: node.color, alpha: node.kind === "mask" ? 0.78 : 0.93 })
      .stroke({ width: selected ? 3 : 1.4, color: selected ? 0xfbbf24 : 0x7dd3fc, alpha: selected ? 1 : 0.82 });
    group.addChild(shell);
  }

  drawPort(group, node.x - 9, node.y + node.h / 2, selected, "in");
  drawPort(group, node.x + node.w + 9, node.y + node.h / 2, selected, "out");
  if (mode === "build") {
    drawPortLabels(group, node);
  }

  addText(group, node.title, node.x + node.w / 2, node.y + 24, 15, 0xe8f2ff, "800", 0.5);
  addText(group, `${node.dtype}${node.shape}`, node.x + node.w / 2, node.y + node.h - 18, 12, 0xb7c7dc, "700", 0.5);

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

type ConnectionPort = {
  connectionId: string;
  role: "source" | "target";
  point: Point;
  label: string;
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
  ports: ConnectionPortMap,
  onQuickConnect?: (connection: CanvasConnectionOverlay) => void,
  showTooltip?: (text: string, point: Point) => void,
  hideTooltip?: () => void
) {
  connections.forEach((connection) => {
    const endpoints = ports.get(connection.id);
    if (!endpoints) return;
    const color = connectionColor(connection);
    const alpha = connection.state === "connected" ? 0.9 : connection.enabled ? 0.48 : 0.24;
    const width = connection.state === "connected" ? 3.2 : 1.8;
    drawCurvedConnection(edgeLayer, endpoints.source.point, endpoints.target.point, color, alpha, width);

    if (connection.state === "connected") {
      const midpoint = curvedMidpoint(endpoints.source.point, endpoints.target.point);
      drawDataLabel(labelLayer, connection.label, midpoint.x, midpoint.y - 18, color);
    } else if (connection.enabled) {
      const midpoint = curvedMidpoint(endpoints.source.point, endpoints.target.point);
      const chipY = connection.kind === "data" ? Math.min(endpoints.source.point.y, endpoints.target.point.y) - 48 : midpoint.y - 22;
      drawConnectionActionChip(labelLayer, connection, midpoint.x, chipY, onQuickConnect, showTooltip, hideTooltip);
    }
  });
}

function drawConnectionActionChip(
  layer: Container,
  connection: CanvasConnectionOverlay,
  x: number,
  y: number,
  onQuickConnect?: (connection: CanvasConnectionOverlay) => void,
  showTooltip?: (text: string, point: Point) => void,
  hideTooltip?: () => void
) {
  const label = connection.kind === "contract" ? "WIRE" : "CONNECT";
  const color = connectionColor(connection);
  const width = label === "CONNECT" ? 86 : 58;
  const chip = new Container({ label: `connection-action-${connection.id}` });
  chip.eventMode = "static";
  chip.cursor = "pointer";
  chip.hitArea = new Rectangle(x - width / 2 - 8, y - 16, width + 16, 32);
  chip.on("pointertap", (event) => {
    event.stopPropagation();
    onQuickConnect?.(connection);
  });
  chip.on("pointerover", () => showTooltip?.(`Click to repair ${connection.label}`, { x, y: y - 30 }));
  chip.on("pointerout", () => hideTooltip?.());

  const halo = new Graphics();
  halo.roundRect(x - width / 2 - 6, y - 16, width + 12, 32, 999).stroke({ width: 2.4, color, alpha: 0.26 });
  chip.addChild(halo);

  const shell = new Graphics();
  shell
    .roundRect(x - width / 2, y - 13, width, 26, 999)
    .fill({ color: connection.kind === "contract" ? 0x211705 : 0x061629, alpha: 0.96 })
    .stroke({ width: 1.6, color, alpha: 0.95 });
  chip.addChild(shell);

  const text = makeText(label, 10, 0xe8f2ff, "900");
  text.anchor.set(0.5, 0.5);
  text.x = x;
  text.y = y;
  chip.addChild(text);

  layer.addChild(chip);
}

function drawCanvasConnectionPorts(
  layer: Container,
  connections: CanvasConnectionOverlay[],
  ports: ConnectionPortMap,
  handlers: {
    onStart: (connection: CanvasConnectionOverlay, point: Point, event: FederatedPointerEvent) => void;
    onComplete: (connection: CanvasConnectionOverlay, event: FederatedPointerEvent) => void;
    onQuickConnect: (connection: CanvasConnectionOverlay, event: FederatedPointerEvent) => void;
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
    onQuickConnect: (connection: CanvasConnectionOverlay, event: FederatedPointerEvent) => void;
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
  hitPad.on("pointertap", (event) => handlers.onQuickConnect(connection, event));
  if (active) {
    hitPad.on("pointerover", () => {
      showTooltip?.(
        port.role === "source" ? `Drag or click ${port.label} output` : `Release or click ${port.label} input`,
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

function findNearestConnectableConnection(
  point: Point,
  connections: CanvasConnectionOverlay[],
  ports: ConnectionPortMap
) {
  let nearestConnection: CanvasConnectionOverlay | undefined;
  let nearestDistance = Number.POSITIVE_INFINITY;

  connections.forEach((connection) => {
    if (!connection.enabled || connection.state !== "open") return;
    const endpoints = ports.get(connection.id);
    if (!endpoints) return;
    const endpointDistance = Math.min(distance(point, endpoints.source.point), distance(point, endpoints.target.point));
    if (endpointDistance > connectionSnapRadius) return;
    if (endpointDistance < nearestDistance) {
      nearestConnection = connection;
      nearestDistance = endpointDistance;
    }
  });

  return nearestConnection;
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
        if (onSlotContextMenu) {
          onSlotContextMenu(event, slot.slotId);
          return;
        }
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
            onSlotContextMenu?.(event, slot.slotId);
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
  const color = hint.icon === "run" ? 0x22c55e : hint.icon === "wire" ? 0xfbbf24 : hint.icon === "probe" ? 0x38bdf8 : 0x93c5fd;
  const label = hint.icon === "run" ? "run" : hint.icon === "wire" ? "wire" : hint.icon === "probe" ? "probe" : "tag";
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
    style: {
      fill,
      fontFamily: "Fira Code, Consolas, ui-monospace, monospace",
      fontSize,
      fontWeight,
      letterSpacing: 0
    }
  });
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
