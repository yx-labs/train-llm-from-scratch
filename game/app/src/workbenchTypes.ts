export type WorkbenchMode = "build" | "trace" | "train";

export type TensorKind = "source" | "tensor" | "matrix" | "attention" | "mask" | "parameter" | "scalar";

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
  flow: "forward" | "mask" | "gradient" | "parameter";
  route?: "direct" | "down" | "loop";
};

export type TraceStep = {
  id: string;
  title: string;
  state: CheckState;
  detail: string;
  selectNodeId: string;
};
