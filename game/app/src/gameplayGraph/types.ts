export type DType =
  | "raw_text"
  | "string_piece"
  | "token_piece"
  | "int"
  | "float32"
  | "bool"
  | "mask";

export type AxisName =
  | "B"
  | "T"
  | "C"
  | "O"
  | "H"
  | "D"
  | "Tq"
  | "Tk"
  | "V"
  | "R"
  | "M"
  | "N"
  | "P";

export type TensorShape = {
  dims: number[];
  axes: AxisName[];
  dtype: DType;
};

export type PortRef = {
  nodeId: string;
  portId: string;
};

export type PortDef = {
  id: string;
  label: string;
  direction: "in" | "out";
  accepts?: DType[];
  emits?: DType;
  shape?: TensorShape;
  required?: boolean;
};

export type GraphNode = {
  id: string;
  moduleId: string;
  params: Record<string, unknown>;
  position: { x: number; y: number };
  locked?: boolean;
};

export type GraphEdge = {
  id: string;
  from: PortRef;
  to: PortRef;
};

export type GraphSpec = {
  levelId: string;
  version: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  outputNodes: string[];
};

export type RuntimeValue = {
  dtype: DType;
  shape?: TensorShape;
  data?: number[] | string[] | string | boolean[];
  meta?: Record<string, unknown>;
};

export type RuntimeErrorType =
  | "dtype_mismatch"
  | "shape_mismatch"
  | "axis_semantic_error"
  | "numeric_mismatch"
  | "budget_exceeded"
  | "oov_unresolved"
  | "mask_error"
  | "nan_inf"
  | "graph_structure"
  | "missing_input";

export type RuntimeError = {
  type: RuntimeErrorType;
  message: string;
  nodeId: string;
  portId?: string;
  expected?: unknown;
  received?: unknown;
  suggestedProbe?: string;
};

export type TraceFrame = {
  step: number;
  nodeId: string;
  moduleId: string;
  inputShapes: Record<string, TensorShape>;
  outputShapes: Record<string, TensorShape>;
  samples?: Record<string, unknown>;
  error?: RuntimeError;
};

export type TestVisibility = "visible" | "hidden" | "reference";

export type TestCase = {
  id: string;
  title: string;
  visibility: TestVisibility;
  inputSeed: string;
  inputs: Record<string, RuntimeValue>;
  assertions: TestAssertion[];
};

export type TestAssertion =
  | { type: "dtype"; nodeId: string; expected: DType }
  | { type: "shape"; nodeId: string; expectedAxes: AxisName[]; expectedDims?: number[] }
  | { type: "axis_semantics"; nodeId: string; expectedAxes: AxisName[] }
  | { type: "allclose"; nodeId: string; referenceNodeId: string; atol: number }
  | { type: "pieces_non_empty"; nodeId: string }
  | { type: "no_oov"; nodeId: string }
  | { type: "tokens_include"; nodeId: string; token: string }
  | { type: "eos_preserved"; nodeId: string; eosToken?: string; padToken?: string }
  | { type: "token_budget"; nodeId: string; maxT: number }
  | { type: "mask_pad"; idsNodeId: string; maskNodeId: string; padId: number }
  | { type: "future_attention_zero"; nodeId: string; threshold: number }
  | { type: "row_sum"; nodeId: string; dim: AxisName; target: number; atol: number };

export type TestResult = {
  id: string;
  visibility: TestVisibility;
  status: "pass" | "fail" | "blocked";
  message: string;
  firstBadNodeId?: string;
  traceFrame?: number;
  assertion?: TestAssertion;
  diagnostic?: {
    errorType?: RuntimeErrorType | "assertion_failed";
    expected?: unknown;
    received?: unknown;
    possibleCause?: string;
    suggestedProbe?: string;
    sample?: unknown;
  };
};

export type ModuleCategory = "data" | "tokenizer" | "tensor" | "attention" | "probe" | "contract";

export type ModuleContext = {
  node: GraphNode;
  inputs: Record<string, RuntimeValue>;
  testInputs: Record<string, RuntimeValue>;
};

export type CodeEmitContext = {
  node: GraphNode;
  module: ModuleDef;
  inputVars: Record<string, string>;
  outputVars: Record<string, string>;
  testInputs: Record<string, RuntimeValue>;
  language: "python" | "typescript";
};

export type CodeLine = {
  id: string;
  nodeId?: string;
  edgeId?: string;
  text: string;
  highlightWhenSelected?: boolean;
};

export type CodeEmitResult = {
  lines: CodeLine[];
};

export type InferResult = {
  outputs: Partial<Record<string, TensorShape>>;
  error?: RuntimeError;
};

export type ExecuteResult = {
  outputs: Partial<Record<string, RuntimeValue>>;
  samples?: Record<string, unknown>;
  error?: RuntimeError;
};

export type ModuleDef = {
  id: string;
  label: string;
  category: ModuleCategory;
  inputs: PortDef[];
  outputs: PortDef[];
  defaultParams: Record<string, unknown>;
  summary: string;
  pseudoCode?: string;
  emitCode?: (context: CodeEmitContext) => CodeEmitResult;
  infer: (context: ModuleContext) => InferResult;
  execute: (context: ModuleContext) => ExecuteResult;
};

export type LevelOnboarding = {
  story: string;
  startingProblem: string;
  firstAction: string;
  targetRecipe: string[];
  winCondition: string;
  allowedMistakes?: string[];
};

export type CaseDataPanel =
  | {
      type: "text_batch";
      title: string;
      inputKey: string;
      focusText?: string;
    }
  | {
      type: "tokenizer_preview";
      title: string;
      tokenizerNodeId: string;
      textInputKey: string;
    }
  | {
      type: "tensor_preview";
      title: string;
      inputKey: string;
      maxRows?: number;
      maxCols?: number;
    }
  | {
      type: "attention_table";
      title: string;
      tokensInputKey?: string;
      scoresNodeId: string;
    };

export type LevelCaseStudy = {
  title: string;
  narrative: string;
  visibleInputFocus?: string;
  dataPanels: CaseDataPanel[];
  playerQuestion: string;
  successObservation: string;
};

export type CertificationControlValue = string | number;

export type CertificationControlSpec = {
  id: string;
  label: string;
  kind: "number" | "integer" | "select";
  defaultValue: CertificationControlValue;
  min?: number;
  max?: number;
  step?: number;
  help?: string;
  options?: Array<{ value: string; label: string }>;
};

export type CertificationTestSpec = {
  testCase: TestCase;
  graph?: GraphSpec;
};

export type LevelCertificationSpec = {
  title: string;
  narrative: string;
  publicVariantLabel: string;
  publicVariantDescription: string;
  systemVariantDescription: string;
  controls: CertificationControlSpec[];
  makePublicTests: (graph: GraphSpec, values: Record<string, CertificationControlValue>) => CertificationTestSpec[];
};

export type LevelSpec = {
  id: string;
  title: string;
  mode: "graph_challenge";
  chapter: string;
  goal: string;
  modulePalette: string[];
  initialGraph: GraphSpec;
  constraints?: {
    maxNodes?: number;
    maxEdges?: number;
    forbiddenModules?: string[];
  };
  visibleTests: TestCase[];
  hiddenTests: TestCase[];
  onboarding?: LevelOnboarding;
  caseStudy?: LevelCaseStudy;
  certification?: LevelCertificationSpec;
  targetGraph?: GraphSpec;
  debrief: {
    completeTitle: string;
    fixedProblem: string;
    learned: string;
    nextUse: string;
  };
};

export type GraphExecutionResult = {
  values: Record<string, RuntimeValue>;
  trace: TraceFrame[];
  error?: RuntimeError;
};

export function valueKey(nodeId: string, portId = "out") {
  return `${nodeId}.${portId}`;
}
