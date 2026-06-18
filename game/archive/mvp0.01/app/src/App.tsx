import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Boxes,
  CheckCircle2,
  Circle,
  FlaskConical,
  Gauge,
  GitBranch,
  Lock,
  MessageSquareText,
  Play,
  RotateCcw,
  Sparkles
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import { compareArena, fetchArtifacts, fetchMissions, fetchProgress, resetProgress, submitMission } from "./api";
import type { ArenaCompareResponse, Artifact, IncidentReport, MetricPoint, Mission, Progress, SubmitResponse } from "./types";

type View = "missions" | "arena";
type Submission = Record<string, unknown>;

const stageLabels: Record<string, string> = {
  data: "Data",
  tokenizer: "Tokenizer",
  attention: "Attention",
  model: "Model",
  pretrain: "Pretrain",
  inference: "Generation",
  sft: "SFT"
};

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function cloneSubmission(mission: Mission): Submission {
  return JSON.parse(JSON.stringify(mission.default_submission)) as Submission;
}

export function App() {
  const queryClient = useQueryClient();
  const missionsQuery = useQuery({ queryKey: ["missions"], queryFn: fetchMissions });
  const progressQuery = useQuery({ queryKey: ["progress"], queryFn: fetchProgress });
  const [view, setView] = useState<View>("missions");
  const [selectedMissionId, setSelectedMissionId] = useState<string | null>(null);

  const missions = missionsQuery.data ?? [];
  const selectedMission = useMemo(() => {
    if (missions.length === 0) return null;
    if (selectedMissionId) {
      return missions.find((mission) => mission.id === selectedMissionId) ?? missions[0];
    }
    return missions.find((mission) => mission.status === "unlocked") ?? missions[0];
  }, [missions, selectedMissionId]);

  useEffect(() => {
    if (!selectedMissionId && selectedMission) {
      setSelectedMissionId(selectedMission.id);
    }
  }, [selectedMission, selectedMissionId]);

  const resetMutation = useMutation({
    mutationFn: resetProgress,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["missions"] });
      queryClient.invalidateQueries({ queryKey: ["progress"] });
      queryClient.invalidateQueries({ queryKey: ["artifacts"] });
      setSelectedMissionId("01_clean_text");
    }
  });

  return (
    <main className="appShell">
      <header className="topBar">
        <div className="brand">
          <div className="brandMark">
            <FlaskConical size={28} />
          </div>
          <div>
            <p className="eyebrow">LLM Learning Game</p>
            <h1>训练实验室 MVP</h1>
          </div>
        </div>
        <nav className="topActions">
          <button className={view === "missions" ? "navButton active" : "navButton"} onClick={() => setView("missions")}>
            <GitBranch size={17} />
            Pipeline
          </button>
          <button className={view === "arena" ? "navButton active" : "navButton"} onClick={() => setView("arena")}>
            <MessageSquareText size={17} />
            Arena
          </button>
          <button className="iconButton" onClick={() => resetMutation.mutate()} title="Reset local progress">
            <RotateCcw size={17} />
          </button>
        </nav>
      </header>

      <section className="statusStrip">
        <StatusPill label="Missions" value={`${progressQuery.data?.completed_missions.length ?? 0}/${missions.length}`} />
        <StatusPill label="Modules" value={`${progressQuery.data?.unlocked_modules.length ?? 0}`} />
        <StatusPill label="Artifacts" value={`${progressQuery.data?.unlocked_artifacts.length ?? 0}`} />
      </section>

      {view === "missions" ? (
        <div className="missionLayout">
          <PipelineMap
            missions={missions}
            selectedMissionId={selectedMission?.id ?? null}
            onSelect={(missionId) => {
              setSelectedMissionId(missionId);
              setView("missions");
            }}
          />
          {selectedMission ? (
            <MissionWorkbench
              mission={selectedMission}
              progress={progressQuery.data ?? null}
              onJump={(missionId) => setSelectedMissionId(missionId)}
            />
          ) : (
            <div className="panel">Loading missions...</div>
          )}
        </div>
      ) : (
        <ArenaView />
      )}
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

function PipelineMap({
  missions,
  selectedMissionId,
  onSelect
}: {
  missions: Mission[];
  selectedMissionId: string | null;
  onSelect: (missionId: string) => void;
}) {
  return (
    <aside className="pipeline panel">
      <div className="panelHeader">
        <Boxes size={18} />
        <h2>主线关卡</h2>
      </div>
      <div className="missionList">
        {missions.map((mission) => (
          <button
            key={mission.id}
            className={`missionNode ${mission.status} ${selectedMissionId === mission.id ? "selected" : ""}`}
            onClick={() => onSelect(mission.id)}
          >
            <span className="missionIcon">
              {mission.status === "completed" ? <CheckCircle2 size={17} /> : mission.status === "locked" ? <Lock size={17} /> : <Circle size={17} />}
            </span>
            <span>
              <b>{mission.title}</b>
              <small>{stageLabels[mission.stage] ?? mission.stage}</small>
            </span>
          </button>
        ))}
      </div>
    </aside>
  );
}

function MissionWorkbench({ mission, progress, onJump }: { mission: Mission; progress: Progress | null; onJump: (missionId: string) => void }) {
  const queryClient = useQueryClient();
  const [submission, setSubmission] = useState<Submission>(() => cloneSubmission(mission));
  const [result, setResult] = useState<SubmitResponse | null>(null);

  useEffect(() => {
    setSubmission(cloneSubmission(mission));
    setResult(null);
  }, [mission.id]);

  const submitMutation = useMutation({
    mutationFn: () => submitMission(mission.id, submission),
    onSuccess: (data) => {
      setResult(data);
      queryClient.invalidateQueries({ queryKey: ["missions"] });
      queryClient.invalidateQueries({ queryKey: ["progress"] });
      queryClient.invalidateQueries({ queryKey: ["artifacts"] });
    }
  });

  const locked = mission.status === "locked";

  return (
    <section className="workbench">
      <div className="panel missionBrief">
        <div className="missionTitleRow">
          <div>
            <p className="eyebrow">{mission.chapter}</p>
            <h2>{mission.title}</h2>
          </div>
          <span className={`stateBadge ${mission.status}`}>{mission.status}</span>
        </div>
        <p className="summary">{mission.summary}</p>
        <div className="objectiveGrid">
          {mission.learning_objectives.map((objective) => (
            <div key={objective} className="objective">
              <Sparkles size={15} />
              {objective}
            </div>
          ))}
        </div>
        {locked ? (
          <div className="notice">
            <Lock size={16} />
            先完成前置关卡：{mission.prerequisites.join(", ")}
          </div>
        ) : null}
      </div>

      <div className="panel labPanel">
        <MissionInteraction mission={mission} submission={submission} setSubmission={setSubmission} />
      </div>

      <div className="sidePanel">
        <div className="panel">
          <div className="panelHeader">
            <Gauge size={18} />
            <h2>提交与反馈</h2>
          </div>
          <button
            className="primaryButton"
            disabled={locked || submitMutation.isPending}
            onClick={() => submitMutation.mutate()}
          >
            <Play size={17} />
            {submitMutation.isPending ? "Running..." : "Run Check"}
          </button>
          <button className="secondaryButton" onClick={() => setSubmission(cloneSubmission(mission))}>
            Reset controls
          </button>
          <HintBox hints={mission.hints} />
        </div>

        {result ? (
          <ResultPanel result={result} onJump={onJump} />
        ) : (
          <div className="panel subtle">
            <p>运行检查后，这里会显示通过结果、训练曲线、事故报告和解锁内容。</p>
          </div>
        )}

        {progress ? (
          <div className="panel subtle">
            <h3>已解锁模块</h3>
            <div className="chipRow">
              {progress.unlocked_modules.length ? progress.unlocked_modules.map((item) => <span className="chip" key={item}>{item}</span>) : <span className="muted">None yet</span>}
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function MissionInteraction({
  mission,
  submission,
  setSubmission
}: {
  mission: Mission;
  submission: Submission;
  setSubmission: (next: Submission) => void;
}) {
  if (mission.type === "sample_filter") {
    const samples = (mission.inputs.samples as Array<{ id: string; text: string; label: string }>) ?? [];
    const kept = new Set(asStringArray(submission.kept_sample_ids));
    return (
      <div>
        <h3>选择要保留的训练样本</h3>
        <div className="sampleList">
          {samples.map((sample) => (
            <label key={sample.id} className={`sampleCard ${kept.has(sample.id) ? "selected" : ""}`}>
              <input
                type="checkbox"
                checked={kept.has(sample.id)}
                onChange={() => {
                  const next = new Set(kept);
                  if (next.has(sample.id)) next.delete(sample.id);
                  else next.add(sample.id);
                  setSubmission({ ...submission, kept_sample_ids: Array.from(next) });
                }}
              />
              <span>{sample.text}</span>
              <small>{sample.label}</small>
            </label>
          ))}
        </div>
      </div>
    );
  }

  if (mission.type === "token_merge") {
    const tokens = asStringArray(submission.tokens);
    const merge = (index: number) => {
      const next = [...tokens];
      next.splice(index, 2, `${next[index]}${next[index + 1]}`);
      setSubmission({ ...submission, tokens: next });
    };
    return (
      <div>
        <h3>点击间隔按钮合并相邻 token</h3>
        <div className="tokenMergeBoard">
          {tokens.map((token, index) => (
            <span className="tokenMergeGroup" key={`${token}-${index}`}>
              <span className="tokenTile">{token === " " ? "space" : token}</span>
              {index < tokens.length - 1 ? <button className="mergeButton" onClick={() => merge(index)}>+</button> : null}
            </span>
          ))}
        </div>
        <MetricSummary items={[["Token count", String(tokens.length)], ["Target", "LLM / learns / tokens"]]} />
      </div>
    );
  }

  if (mission.type === "shift_pairs") {
    const tokens = asStringArray((mission.inputs as { tokens?: unknown }).tokens);
    const options = ((mission.inputs as { offset_options?: number[] }).offset_options ?? []) as number[];
    const offset = asNumber(submission.target_offset, 0);
    return (
      <div>
        <h3>选择 target 相对 input 的偏移</h3>
        <div className="controlLine">
          <label>Target offset</label>
          <select value={offset} onChange={(event) => setSubmission({ ...submission, target_offset: Number(event.target.value) })}>
            {options.map((option) => (
              <option key={option} value={option}>{option > 0 ? `+${option}` : option}</option>
            ))}
          </select>
        </div>
        <div className="pairGrid">
          {tokens.slice(0, -1).map((token, index) => (
            <div className="pairCard" key={`${token}-${index}`}>
              <small>input</small>
              <b>{token}</b>
              <small>target</small>
              <b>{tokens[index + offset] ?? "out of range"}</b>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (mission.type === "mask_grid") {
    const sequence = asStringArray((mission.inputs as { sequence?: unknown }).sequence);
    const mask = (submission.mask as boolean[][]) ?? [];
    const setCell = (row: number, col: number) => {
      const next = mask.map((line) => [...line]);
      next[row][col] = !next[row][col];
      setSubmission({ ...submission, mask: next });
    };
    const setCausal = () => {
      const size = sequence.length;
      setSubmission({
        ...submission,
        mask: Array.from({ length: size }, (_, row) => Array.from({ length: size }, (_, col) => col <= row))
      });
    };
    return (
      <div>
        <div className="interactionHeader">
          <h3>Attention 可见矩阵</h3>
          <button className="secondaryButton compact" onClick={setCausal}>Apply causal mask</button>
        </div>
        <div className="maskGrid" style={{ gridTemplateColumns: `repeat(${sequence.length + 1}, minmax(52px, 1fr))` }}>
          <div />
          {sequence.map((token) => <div className="axisLabel" key={`col-${token}`}>{token}</div>)}
          {sequence.map((rowToken, row) => (
            <>
              <div className="axisLabel" key={`row-label-${rowToken}`}>{rowToken}</div>
              {sequence.map((_, col) => (
                <button
                  key={`${row}-${col}`}
                  className={`maskCell ${mask[row]?.[col] ? "on" : "off"} ${col > row ? "future" : ""}`}
                  onClick={() => setCell(row, col)}
                >
                  {mask[row]?.[col] ? "see" : "mask"}
                </button>
              ))}
            </>
          ))}
        </div>
      </div>
    );
  }

  if (mission.type === "module_assembly") {
    const modules = (mission.inputs.modules as Array<{ id: string; label: string }>) ?? [];
    const selected = new Set(asStringArray(submission.selected_modules));
    return (
      <div>
        <h3>选择最小 Decoder-only Transformer 所需模块</h3>
        <div className="moduleGrid">
          {modules.map((module) => (
            <label key={module.id} className={`moduleCard ${selected.has(module.id) ? "selected" : ""}`}>
              <input
                type="checkbox"
                checked={selected.has(module.id)}
                onChange={() => {
                  const next = new Set(selected);
                  if (next.has(module.id)) next.delete(module.id);
                  else next.add(module.id);
                  setSubmission({ ...submission, selected_modules: Array.from(next) });
                }}
              />
              <span>{module.label}</span>
            </label>
          ))}
        </div>
      </div>
    );
  }

  if (mission.type === "train_simulation") {
    return (
      <div>
        <h3>训练模拟控制台</h3>
        <SliderControl label="Learning rate" value={asNumber(submission.learning_rate, 0.0003)} min={0.00005} max={0.002} step={0.00005} onChange={(value) => setSubmission({ ...submission, learning_rate: value })} />
        <SliderControl label="Batch size" value={asNumber(submission.batch_size, 16)} min={4} max={64} step={4} onChange={(value) => setSubmission({ ...submission, batch_size: value })} />
        <SliderControl label="Context length" value={asNumber(submission.context_length, 256)} min={64} max={512} step={64} onChange={(value) => setSubmission({ ...submission, context_length: value })} />
        <SliderControl label="Data quality" value={asNumber(submission.data_quality, 0.8)} min={0.4} max={1} step={0.05} onChange={(value) => setSubmission({ ...submission, data_quality: value })} />
      </div>
    );
  }

  if (mission.type === "generation_controls") {
    return (
      <div>
        <h3>采样参数</h3>
        <p className="promptBox">{String(mission.inputs.prompt ?? "")}</p>
        <SliderControl label="Temperature" value={asNumber(submission.temperature, 0.7)} min={0.1} max={1.5} step={0.1} onChange={(value) => setSubmission({ ...submission, temperature: value })} />
        <SliderControl label="Top-p" value={asNumber(submission.top_p, 0.9)} min={0.3} max={1} step={0.05} onChange={(value) => setSubmission({ ...submission, top_p: value })} />
        <SliderControl label="Repetition penalty" value={asNumber(submission.repetition_penalty, 1.05)} min={1} max={1.3} step={0.05} onChange={(value) => setSubmission({ ...submission, repetition_penalty: value })} />
      </div>
    );
  }

  if (mission.type === "sft_mask") {
    const tokens = (mission.inputs.tokens as Array<{ text: string; role: string; expected_loss: boolean }>) ?? [];
    const mask = (submission.loss_mask as boolean[]) ?? [];
    return (
      <div>
        <h3>点亮参与 loss 的 token</h3>
        <div className="sftTokenBoard">
          {tokens.map((token, index) => (
            <button
              key={`${token.text}-${index}`}
              className={`sftToken ${mask[index] ? "active" : ""} role-${token.role}`}
              onClick={() => {
                const next = [...mask];
                next[index] = !next[index];
                setSubmission({ ...submission, loss_mask: next });
              }}
            >
              <small>{token.role}</small>
              {token.text}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return <pre>{JSON.stringify(submission, null, 2)}</pre>;
}

function SliderControl({
  label,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="sliderControl">
      <span>
        {label}
        <b>{value}</b>
      </span>
      <input type="range" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function MetricSummary({ items }: { items: Array<[string, string]> }) {
  return (
    <div className="metricSummary">
      {items.map(([label, value]) => (
        <div key={label}>
          <small>{label}</small>
          <b>{value}</b>
        </div>
      ))}
    </div>
  );
}

function HintBox({ hints }: { hints: string[] }) {
  const [level, setLevel] = useState(1);
  return (
    <div className="hintBox">
      <div className="hintHeader">
        <b>Hints</b>
        <button className="textButton" onClick={() => setLevel(Math.min(hints.length, level + 1))}>更多提示</button>
      </div>
      {hints.slice(0, level).map((hint) => (
        <p key={hint}>{hint}</p>
      ))}
    </div>
  );
}

function ResultPanel({ result, onJump }: { result: SubmitResponse; onJump: (missionId: string) => void }) {
  return (
    <div className={`panel resultPanel ${result.passed ? "passed" : "failed"}`}>
      <div className="resultHeader">
        {result.passed ? <CheckCircle2 size={19} /> : <AlertTriangle size={19} />}
        <h2>{result.passed ? "Challenge passed" : "Needs diagnosis"}</h2>
      </div>
      {result.preview ? <p className="previewBox">{result.preview}</p> : null}
      {result.metrics ? <TrainingChart metrics={result.metrics} /> : null}
      {result.incident_report ? <Incident report={result.incident_report} /> : null}
      {result.unlocked_modules.length || result.unlocked_artifacts.length ? (
        <div className="unlockBox">
          <b>Unlocked</b>
          <div className="chipRow">
            {result.unlocked_modules.map((item) => <span className="chip" key={item}>{item}</span>)}
            {result.unlocked_artifacts.map((item) => <span className="chip artifact" key={item}>{item}</span>)}
          </div>
        </div>
      ) : null}
      {result.passed && result.next_mission_id ? (
        <button className="primaryButton" onClick={() => onJump(result.next_mission_id!)}>Next mission</button>
      ) : null}
    </div>
  );
}

function Incident({ report }: { report: IncidentReport }) {
  return (
    <div className="incident">
      <b>{report.title}</b>
      {report.symptoms?.length ? (
        <ul>
          {report.symptoms.map((item) => <li key={item}>{item}</li>)}
        </ul>
      ) : null}
      {report.hints?.length ? (
        <div className="hintList">
          {report.hints.map((item) => <p key={item}>{item}</p>)}
        </div>
      ) : null}
    </div>
  );
}

function TrainingChart({ metrics }: { metrics: MetricPoint[] }) {
  return (
    <div className="chartBox">
      <ResponsiveContainer width="100%" height={210}>
        <LineChart data={metrics} margin={{ left: 4, right: 12, top: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step" />
          <YAxis domain={["auto", "auto"]} />
          <Tooltip />
          <Line type="monotone" dataKey="train_loss" stroke="#1f77b4" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="val_loss" stroke="#d97706" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ArenaView() {
  const artifactsQuery = useQuery({ queryKey: ["artifacts"], queryFn: fetchArtifacts });
  const artifacts = artifactsQuery.data ?? [];
  const [prompt, setPrompt] = useState("请用 <think> 和 <answer> 回答：23 + 48 = ?");
  const [selected, setSelected] = useState<string[]>(["pretrain_good", "sft"]);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [result, setResult] = useState<ArenaCompareResponse | null>(null);

  const compareMutation = useMutation({
    mutationFn: () => compareArena({ prompt, artifact_ids: selected, temperature, top_p: topP }),
    onSuccess: setResult
  });

  return (
    <section className="arenaLayout">
      <div className="panel arenaControls">
        <div className="panelHeader">
          <MessageSquareText size={18} />
          <h2>模型 Arena</h2>
        </div>
        <label className="fieldLabel">
          Prompt
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
        </label>
        <div className="artifactSelect">
          {artifacts.map((artifact) => (
            <ArtifactCheckbox
              key={artifact.id}
              artifact={artifact}
              checked={selected.includes(artifact.id)}
              onChange={() => {
                setSelected((current) => current.includes(artifact.id) ? current.filter((item) => item !== artifact.id) : [...current, artifact.id]);
              }}
            />
          ))}
        </div>
        <SliderControl label="Temperature" value={temperature} min={0.1} max={1.5} step={0.1} onChange={setTemperature} />
        <SliderControl label="Top-p" value={topP} min={0.3} max={1} step={0.05} onChange={setTopP} />
        <button className="primaryButton" disabled={compareMutation.isPending || selected.length === 0} onClick={() => compareMutation.mutate()}>
          Compare models
        </button>
      </div>
      <div className="arenaResults">
        {(result?.results ?? []).map((row) => (
          <article className="panel outputCard" key={row.artifact_id}>
            <div className="outputHeader">
              <h3>{row.display_name}</h3>
              <span className={row.unlocked ? "stateBadge completed" : "stateBadge locked"}>{row.unlocked ? "unlocked" : "locked fallback"}</span>
            </div>
            <pre>{row.output}</pre>
            <p>{row.notes}</p>
            <small>source: {row.source}</small>
          </article>
        ))}
        {!result ? (
          <div className="panel subtle">
            <p>选择模型并运行对比。MVP 使用 sample output fallback，后续可接真实 checkpoint。</p>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function ArtifactCheckbox({ artifact, checked, onChange }: { artifact: Artifact; checked: boolean; onChange: () => void }) {
  return (
    <label className={`artifactCard ${checked ? "selected" : ""}`}>
      <input type="checkbox" checked={checked} onChange={onChange} />
      <span>
        <b>{artifact.display_name}</b>
        <small>{artifact.unlocked ? "unlocked" : "locked, fallback available"}</small>
      </span>
    </label>
  );
}
