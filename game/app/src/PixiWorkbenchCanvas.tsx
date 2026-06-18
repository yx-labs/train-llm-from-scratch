import { useEffect, useRef } from "react";
import { Application, Container, Graphics, Rectangle, Text, type Ticker } from "pixi.js";
import { nodeById, sceneSize, tensorEdges, tensorNodes } from "./sceneData";
import type { TensorEdge, TensorNode, WorkbenchMode } from "./workbenchTypes";

type PixiWorkbenchCanvasProps = {
  selectedId: string;
  mode: WorkbenchMode;
  playing: boolean;
  onSelect: (id: string) => void;
};

type Point = {
  x: number;
  y: number;
};

type FontWeight = "400" | "500" | "600" | "700" | "800" | "900" | "bold";

export function PixiWorkbenchCanvas({ selectedId, mode, playing, onSelect }: PixiWorkbenchCanvasProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const appRef = useRef<Application | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const latestRef = useRef({ selectedId, mode, playing, onSelect });

  useEffect(() => {
    latestRef.current = { selectedId, mode, playing, onSelect };
    if (!appRef.current) return;

    cleanupRef.current?.();
    cleanupRef.current = drawScene(appRef.current, latestRef.current);
  }, [selectedId, mode, playing, onSelect]);

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
      host.appendChild(app.canvas);
      app.ticker.maxFPS = 60;
      cleanupRef.current = drawScene(app, latestRef.current);

      observer = new ResizeObserver((entries) => {
        const entry = entries[0];
        const width = Math.max(760, Math.floor(entry.contentRect.width));
        const height = Math.max(520, Math.floor(entry.contentRect.height));
        app.renderer.resize(width, height);
        cleanupRef.current?.();
        cleanupRef.current = drawScene(app, latestRef.current);
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
  return {
    width: Math.max(760, Math.floor(host.clientWidth || 960)),
    height: Math.max(520, Math.floor(host.clientHeight || 560))
  };
}

function drawScene(
  app: Application,
  state: { selectedId: string; mode: WorkbenchMode; playing: boolean; onSelect: (id: string) => void }
) {
  for (const child of app.stage.removeChildren()) {
    child.destroy({ children: true });
  }

  const screenWidth = app.screen.width;
  const screenHeight = app.screen.height;
  const background = new Graphics();
  background.roundRect(0, 0, screenWidth, screenHeight, 10).fill({ color: 0x07111f, alpha: 1 });
  drawGrid(background, screenWidth, screenHeight);
  app.stage.addChild(background);

  const scene = new Container({ label: "tensor-workbench-scene", sortableChildren: true });
  const scale = Math.min((screenWidth - 28) / sceneSize.width, (screenHeight - 20) / sceneSize.height);
  scene.scale.set(scale);
  scene.x = (screenWidth - sceneSize.width * scale) / 2;
  scene.y = (screenHeight - sceneSize.height * scale) / 2;
  app.stage.addChild(scene);

  const edgeLayer = new Container({ label: "typed-data-lines" });
  const nodeLayer = new Container({ label: "tensor-nodes" });
  const overlayLayer = new Container({ label: "labels-and-overlays" });
  const pulseLayer = new Graphics({ label: "flow-pulses" });
  scene.addChild(edgeLayer, nodeLayer, overlayLayer, pulseLayer);

  drawSceneHeader(overlayLayer, state.mode);

  const visibleEdges = tensorEdges.filter((edge) => edge.flow !== "gradient" || state.mode === "train");
  visibleEdges.forEach((edge) => drawEdge(edgeLayer, overlayLayer, edge, state.mode));

  tensorNodes.forEach((node) => {
    drawTensorNode(nodeLayer, node, state.selectedId === node.id, state.onSelect, state.mode);
  });

  drawAttentionLegend(overlayLayer);
  if (state.mode === "train") {
    drawTrainOverlay(overlayLayer);
  }

  let elapsed = 0;
  const tick = (ticker: Ticker) => {
    elapsed += ticker.deltaMS;
    drawPulses(pulseLayer, visibleEdges, elapsed, state.playing || state.mode === "train");
  };
  app.ticker.add(tick);
  return () => {
    app.ticker.remove(tick);
  };
}

function drawGrid(graphics: Graphics, width: number, height: number) {
  for (let x = 0; x <= width; x += 32) {
    graphics.moveTo(x, 0).lineTo(x, height).stroke({ width: 1, color: 0x5f7aa0, alpha: x % 128 === 0 ? 0.22 : 0.09 });
  }
  for (let y = 0; y <= height; y += 32) {
    graphics.moveTo(0, y).lineTo(width, y).stroke({ width: 1, color: 0x5f7aa0, alpha: y % 128 === 0 ? 0.22 : 0.09 });
  }
}

function drawSceneHeader(layer: Container, mode: WorkbenchMode) {
  const panel = new Graphics();
  panel
    .roundRect(44, 34, 1150, 74, 10)
    .fill({ color: 0x0b1728, alpha: 0.82 })
    .stroke({ width: 1.2, color: 0x315f94, alpha: 0.95 });
  layer.addChild(panel);
  addText(layer, "LLM Complete v0.02 - 3D Tensor Workbench", 70, 62, 21, 0xe8f2ff, "800", 0);
  addText(
    layer,
    mode === "build"
      ? "Build focus: module shells, typed ports, and shape labels."
      : mode === "trace"
        ? "Trace focus: forward activation, attention mask, softmax row checks, and repairable diagnostics."
        : "Train focus: backward gradient path and optimizer update are visible instead of hidden behind a training spinner.",
    70,
    91,
    13,
    0xa8c7e8,
    "600",
    0
  );
}

function drawTensorNode(layer: Container, node: TensorNode, selected: boolean, onSelect: (id: string) => void, mode: WorkbenchMode) {
  const group = new Container({ label: `node-${node.id}` });
  group.eventMode = "static";
  group.cursor = "pointer";
  group.hitArea = new Rectangle(node.x - 26, node.y - 38, node.w + 74, node.h + 70);
  group.on("pointertap", () => onSelect(node.id));

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
    drawGauge(group, node.x + 24, node.y + 30);
  }

  layer.addChild(group);
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

function drawEdge(edgeLayer: Container, labelLayer: Container, edge: TensorEdge, mode: WorkbenchMode) {
  const points = getEdgePoints(edge);
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

  if (mode !== "build" || edge.flow !== "gradient") {
    const midpoint = pointAtPath(points, 0.5);
    drawDataLabel(labelLayer, edge.label, midpoint.x, midpoint.y - 20, edge.color);
  }
}

function getEdgePoints(edge: TensorEdge): Point[] {
  const from = nodeById.get(edge.from);
  const to = nodeById.get(edge.to);
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
    const bendX = Math.max(start.x + 36, end.x - 40);
    return [start, { x: bendX, y: start.y }, { x: bendX, y: end.y }, end];
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
  gauge.circle(x, y, 17).fill({ color: 0x0f172a, alpha: 0.9 }).stroke({ width: 2, color: 0xfbbf24, alpha: 0.9 });
  gauge.moveTo(x, y).lineTo(x + 10, y - 8).stroke({ width: 2, color: 0xfbbf24, alpha: 1, cap: "round" });
  layer.addChild(gauge);
}

function drawAttentionLegend(layer: Container) {
  const panel = new Graphics();
  panel
    .roundRect(44, 585, 1150, 38, 8)
    .fill({ color: 0x0f1f32, alpha: 0.88 })
    .stroke({ width: 1, color: 0x334d67, alpha: 1 });
  layer.addChild(panel);
  addText(
    layer,
    "Legend: cyan = forward activation, gray = mask/frozen path, orange = backward gradient. Shape tags are gameplay information, not decoration.",
    68,
    609,
    13,
    0xcfe7ff,
    "700",
    0
  );
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

function drawPulses(layer: Graphics, edges: TensorEdge[], elapsedMS: number, active: boolean) {
  layer.clear();
  if (!active) return;

  edges.forEach((edge, index) => {
    const points = getEdgePoints(edge);
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
