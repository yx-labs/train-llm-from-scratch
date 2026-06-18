import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Boxes,
  CheckCircle2,
  Circle,
  Cpu,
  Eye,
  Pause,
  Play,
  RotateCcw,
  Search,
  Wrench
} from "lucide-react";
import { modeLabels, tensorNodes, toolboxModules, traceSteps } from "./sceneData";
import { PixiWorkbenchCanvas } from "./PixiWorkbenchCanvas";
import type { CheckState, TensorNode, WorkbenchMode } from "./workbenchTypes";

const modeIcons: Record<WorkbenchMode, JSX.Element> = {
  build: <Wrench size={17} />,
  trace: <Search size={17} />,
  train: <Cpu size={17} />
};

export function App() {
  const [mode, setMode] = useState<WorkbenchMode>("trace");
  const [selectedId, setSelectedId] = useState("attn_probs");
  const [playing, setPlaying] = useState(true);

  const selectedNode = useMemo(() => tensorNodes.find((node) => node.id === selectedId) ?? tensorNodes[0], [selectedId]);
  const modeCopy = modeLabels[mode];

  return (
    <main className="appShell">
      <header className="topBar">
        <div className="brand">
          <div className="brandMark">
            <Boxes size={28} />
          </div>
          <div>
            <p className="eyebrow">LLM Complete / MVP 0.02</p>
            <h1>3D Tensor Workbench</h1>
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
          <button className="iconButton" title={playing ? "Pause trace animation" : "Play trace animation"} onClick={() => setPlaying((value) => !value)}>
            {playing ? <Pause size={17} /> : <Play size={17} />}
          </button>
          <button
            className="iconButton"
            title="Reset view"
            onClick={() => {
              setMode("trace");
              setSelectedId("attn_probs");
              setPlaying(true);
            }}
          >
            <RotateCcw size={17} />
          </button>
        </div>
      </header>

      <section className="statusStrip">
        <StatusPill label="Mode" value={modeCopy.title} />
        <StatusPill label="Selected" value={`${selectedNode.semanticName} ${selectedNode.shape}`} />
        <StatusPill label="Renderer" value="PixiJS v8 / WebGL preferred" />
      </section>

      <section className="workbenchGrid">
        <aside className="panel moduleShelf">
          <div className="panelHeader">
            <Wrench size={18} />
            <h2>Module Shelf</h2>
          </div>
          <p className="panelNote">v0.02 先固定模块，后续把这些条目升级为可拖拽组件。</p>
          <div className="moduleList">
            {toolboxModules.map((moduleName) => (
              <button key={moduleName} className="moduleItem">
                <span />
                {moduleName}
              </button>
            ))}
          </div>
          <div className="contractBox">
            <b>Shape contracts</b>
            <code>int[B,T]</code>
            <code>float[B,T,C]</code>
            <code>float[B,H,T,T]</code>
            <code>scalar loss</code>
          </div>
        </aside>

        <section className="stagePanel panel">
          <div className="stageHeader">
            <div>
              <p className="eyebrow">Canvas / PixiJS foundation</p>
              <h2>{modeCopy.title}</h2>
            </div>
            <p>{modeCopy.description}</p>
          </div>
          <PixiWorkbenchCanvas selectedId={selectedId} mode={mode} playing={playing} onSelect={setSelectedId} />
        </section>

        <TensorInspector node={selectedNode} />

        <section className="testBench panel">
          <div className="testHeader">
            <div>
              <p className="eyebrow">Test Runner</p>
              <h2>attention_mask_reference</h2>
            </div>
            <button className="runButton" onClick={() => setPlaying(true)}>
              <Play size={16} />
              Trace
            </button>
          </div>
          <div className="traceSteps">
            {traceSteps.map((step) => (
              <button key={step.id} className={`traceStep ${step.state}`} onClick={() => setSelectedId(step.selectNodeId)}>
                <StateIcon state={step.state} />
                <span>
                  <b>{step.title}</b>
                  <small>{step.detail}</small>
                </span>
              </button>
            ))}
          </div>
          <div className="errorPanel">
            <AlertTriangle size={18} />
            <div>
              <b>Repairable failure sample</b>
              <p>
                Causal mask debug cell <code>[2,3]</code> is marked in red. The player should verify it is blocked before softmax, then re-run row-sum checks.
              </p>
            </div>
          </div>
        </section>
      </section>
    </main>
  );
}

function StatusPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="statusPill">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function TensorInspector({ node }: { node: TensorNode }) {
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

      <section className="sampleBox">
        <h3>Sample values</h3>
        {node.sample.map((line) => (
          <code key={line}>{line}</code>
        ))}
      </section>

      <section className="checkList">
        {node.checks.map((check) => (
          <div key={check.label} className={`checkRow ${check.state}`}>
            <StateIcon state={check.state} />
            <span>
              <b>{check.label}</b>
              <small>{check.detail}</small>
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
  return <Circle size={16} />;
}
