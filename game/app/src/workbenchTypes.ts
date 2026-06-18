export type WorkbenchMode = "build" | "trace" | "train";

export type TensorKind = "source" | "tensor" | "matrix" | "attention" | "mask" | "parameter" | "scalar" | "operation";

export type CheckState = "pass" | "fail" | "warn";

export type TensorNode = {
  id: string;
  title: string;
  subtitle: string;
  kind: TensorKind;
  semanticName: string;
  dtype: string;
  shape: string;
  source: string;
  consumer: string;
  stats: {
    min: string;
    max: string;
    mean: string;
    rowSum?: string;
  };
  sample: string[];
  checks: Array<{
    label: string;
    state: CheckState;
    detail: string;
  }>;
  x: number;
  y: number;
  w: number;
  h: number;
  color: number;
};

export type TensorEdge = {
  id: string;
  from: string;
  to: string;
  label: string;
  color: number;
  flow: "forward" | "mask" | "gradient" | "parameter" | "check";
  route?: "direct" | "down" | "loop";
};

export type TraceStep = {
  id: string;
  title: string;
  state: CheckState;
  detail: string;
  selectNodeId: string;
};

export type RepairTag = {
  id: string;
  label: string;
  shortLabel: string;
  detail: string;
  category?: "data" | "axis" | "contract" | "operation" | "rail";
};

export type ProbeObservation = {
  id: string;
  title: string;
  detail: string;
  evidence: string[];
  possibleSemantic: string;
  confidence: "low" | "medium" | "high";
};

export type ProbeTool = {
  id: string;
  label: string;
  detail: string;
  budgetCost: number;
  observations: Record<string, ProbeObservation>;
};

export type RepairSlot = {
  id: string;
  label: string;
  nodeId: string;
  focusNodeId: string;
  emptyLabel: string;
  correctTagIds: string[];
  expected: string;
  successDetail: string;
  failureDetail: string;
};

export type RepairCheckDefinition = {
  id: string;
  title: string;
  group: "visible" | "behavior" | "hidden" | "reference";
  slotIds: string[];
  expected: string;
  passDetail: string;
  failDetail: string;
  blockedDetail?: string;
  focusNodeId: string;
};

export type RepairSpec = {
  kind: "axis_labels" | "matmul_gate" | "transpose_switch" | "broadcast_rail";
  targetContract: string;
  brokenMessage: string;
  budget: {
    probes: number;
    referenceRuns: number;
  };
  tags: RepairTag[];
  slots: RepairSlot[];
  probes: ProbeTool[];
  checks: RepairCheckDefinition[];
  hiddenCases: string[];
  successSummary: string;
};

export type BootcampLevel = {
  id: string;
  title: string;
  subtitle: string;
  objective: string;
  sceneTitle: string;
  sceneSubtitle: string;
  defaultSelectedNodeId: string;
  briefing: string[];
  knowledgeCards?: Array<{
    title: string;
    body: string;
    visual?: string[];
  }>;
  knowledgeTransition?: {
    title: string;
    body: string;
    buttonLabel: string;
  };
  mission?: {
    title: string;
    body: string;
    success: string[];
  };
  unlocks: string[];
  contracts: string[];
  nodes: TensorNode[];
  edges: TensorEdge[];
  repair: RepairSpec;
  traceSteps: TraceStep[];
};

export type BootcampAnswerMap = Record<string, string>;

export type BootcampCheck = {
  id: string;
  title: string;
  group: "visible" | "behavior" | "hidden" | "reference";
  state: CheckState;
  detail: string;
  expected: string;
  received: string;
  focusNodeId: string;
};

export type BootcampResult = {
  passed: boolean;
  summary: string;
  errorType: string;
  focusNodeId: string;
  score: {
    rank: "A" | "B" | "C";
    probeEfficiency: string;
    probeUses: number;
    probeBudget: number;
    referenceRuns: number;
    referenceBudget: number;
  };
  checks: BootcampCheck[];
};
