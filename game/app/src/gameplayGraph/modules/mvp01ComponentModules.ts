import type { ModuleDef, RuntimeError, RuntimeValue, TensorShape } from "../types";
import { makeTensor, sampleValues, shapeOf, type TinyTensor } from "../runtime/tinyTensor";
import { matMulGateModule } from "./tensorModules";

function tensorValue(tensor: TinyTensor, meta?: Record<string, unknown>): RuntimeValue {
  return {
    dtype: tensor.dtype,
    shape: shapeOf(tensor),
    data: tensor.data,
    meta
  };
}

function tensorFromValue(value: RuntimeValue | undefined): TinyTensor | undefined {
  if (!value?.shape || !Array.isArray(value.data)) return undefined;
  if (value.dtype !== "float32") return undefined;
  return makeTensor("float32", value.shape.dims, value.shape.axes, value.data as number[]);
}

function moduleError(type: RuntimeError["type"], nodeId: string, message: string, expected?: unknown, received?: unknown, portId?: string): RuntimeError {
  return { type, nodeId, portId, message, expected, received };
}

export const float32LiteralModule: ModuleDef = {
  id: "Float32Literal",
  label: "Float32 Literal",
  category: "data",
  inputs: [],
  outputs: [{ id: "out", label: "scalar", direction: "out", emits: "float32" }],
  defaultParams: { value: 0.5 },
  summary: "Raw construction primitive that emits one finite rank-0 float32 value.",
  pseudoCode: "scalar = float32(value)",
  execute: ({ node }) => {
    const value = parseFiniteFloat32Param(node.params.value);
    if (value === undefined) {
      return { outputs: {}, error: moduleError("nan_inf", node.id, "Float32Literal value must be finite", "finite number", node.params.value) };
    }
    const tensor = makeTensor("float32", [], [], [value]);
    return { outputs: { out: tensorValue(tensor, { source: "Float32Literal" }) }, samples: { out: [value] } };
  },
  infer: () => ({ outputs: { out: { dtype: "float32", dims: [], axes: [] } } })
};

export const vectorRailModule: ModuleDef = {
  id: "VectorRail",
  label: "Vector Rail",
  category: "tensor",
  inputs: [
    { id: "c0", label: "c0", direction: "in", accepts: ["float32"], required: true },
    { id: "c1", label: "c1", direction: "in", accepts: ["float32"], required: true },
    { id: "c2", label: "c2", direction: "in", accepts: ["float32"], required: true }
  ],
  outputs: [{ id: "out", label: "vector[C]", direction: "out", emits: "float32" }],
  defaultParams: {},
  summary: "Combines three ScalarCell outputs into a feature vector with axis C.",
  pseudoCode: "vector = stack([c0, c1, c2], axis='C')",
  execute: ({ node, inputs }) => {
    const scalars = ["c0", "c1", "c2"].map((portId) => scalarNumber(inputs[portId]));
    if (scalars.some((value) => value === undefined)) {
      return { outputs: {}, error: moduleError("shape_mismatch", node.id, "VectorRail requires three rank-0 scalar inputs", "float32[]", inputs) };
    }
    const tensor = makeTensor("float32", [3], ["C"], scalars as number[]);
    return { outputs: { out: tensorValue(tensor, { combinedFrom: ["c0", "c1", "c2"] }) }, samples: { out: sampleValues(tensor) } };
  },
  infer: () => ({ outputs: { out: { dtype: "float32", dims: [3], axes: ["C"] } } })
};

export const matrixStructModule: ModuleDef = {
  id: "MatrixStruct",
  label: "Matrix Struct",
  category: "tensor",
  inputs: [
    { id: "o0", label: "O0 column", direction: "in", accepts: ["float32"], required: true },
    { id: "o1", label: "O1 column", direction: "in", accepts: ["float32"], required: true }
  ],
  outputs: [{ id: "out", label: "matrix[C,O]", direction: "out", emits: "float32" }],
  defaultParams: {},
  summary: "Stacks two feature vectors as a matrix plus rank/dim/axis/stride metadata.",
  pseudoCode: "matrix = stack([o0, o1], axis='O')  # shape [C,O]",
  execute: ({ node, inputs }) => {
    const left = tensorFromValue(inputs.o0);
    const right = tensorFromValue(inputs.o1);
    if (!left || !right || left.dims.length !== 1 || right.dims.length !== 1 || left.dims[0] !== right.dims[0]) {
      return { outputs: {}, error: moduleError("shape_mismatch", node.id, "MatrixStruct requires two same-length vector[C] inputs", "float32[C], float32[C]", { o0: inputs.o0?.shape, o1: inputs.o1?.shape }) };
    }
    const data: number[] = [];
    for (let c = 0; c < left.dims[0]; c += 1) {
      data.push(left.data[c], right.data[c]);
    }
    const tensor = makeTensor("float32", [left.dims[0], 2], ["C", "O"], data);
    return {
      outputs: { out: tensorValue(tensor, { struct: { rank: 2, dims: tensor.dims, axes: tensor.axes, strides: [2, 1] } }) },
      samples: { out: sampleValues(tensor), struct: { rank: 2, strides: [2, 1] } }
    };
  },
  infer: ({ inputs }) => {
    const c = inputs.o0?.shape?.dims[0] ?? inputs.o1?.shape?.dims[0] ?? 3;
    return { outputs: { out: { dtype: "float32", dims: [c, 2], axes: ["C", "O"] } } };
  }
};

export const tensorBoxModule: ModuleDef = {
  id: "TensorBox",
  label: "Tensor Box",
  category: "tensor",
  inputs: [
    { id: "t0", label: "token 0", direction: "in", accepts: ["float32"], required: true },
    { id: "t1", label: "token 1", direction: "in", accepts: ["float32"], required: true }
  ],
  outputs: [{ id: "out", label: "tensor[B,T,C]", direction: "out", emits: "float32" }],
  defaultParams: {},
  summary: "Stacks two vector[C] token rows into a rank-3 tensor [B,T,C].",
  pseudoCode: "tensor = stack([[t0, t1]], axes=['B','T','C'])",
  execute: ({ node, inputs }) => {
    const t0 = tensorFromValue(inputs.t0);
    const t1 = tensorFromValue(inputs.t1);
    if (!t0 || !t1 || t0.dims.length !== 1 || t1.dims.length !== 1 || t0.dims[0] !== t1.dims[0]) {
      return { outputs: {}, error: moduleError("shape_mismatch", node.id, "TensorBox requires two same-length vector[C] token inputs", "float32[C], float32[C]", { t0: inputs.t0?.shape, t1: inputs.t1?.shape }) };
    }
    const tensor = makeTensor("float32", [1, 2, t0.dims[0]], ["B", "T", "C"], [...t0.data, ...t1.data]);
    return { outputs: { out: tensorValue(tensor, { fromTokens: 2 }) }, samples: { out: sampleValues(tensor) } };
  },
  infer: ({ inputs }) => {
    const c = inputs.t0?.shape?.dims[0] ?? inputs.t1?.shape?.dims[0] ?? 3;
    return { outputs: { out: { dtype: "float32", dims: [1, 2, c], axes: ["B", "T", "C"] } } };
  }
};

const packedScalarCellModule = asPackedComponent(float32LiteralModule, "component.scalar_cell.v1", "ScalarCell v1");
const packedVectorRailModule = asPackedComponent(vectorRailModule, "component.vector_rail.v1", "VectorRail v1");
const packedMatrixStructModule = asPackedComponent(matrixStructModule, "component.matrix_struct.v1", "MatrixStruct v1");
const packedTensorBoxModule = asPackedComponent(tensorBoxModule, "component.tensor_box.v1", "TensorBox v1");
const packedMatMulGateModule = asPackedComponent(matMulGateModule, "component.matmul_gate.v1", "MatMulGate v1");

export const mvp01ComponentModules = [
  float32LiteralModule,
  vectorRailModule,
  matrixStructModule,
  tensorBoxModule,
  packedScalarCellModule,
  packedVectorRailModule,
  packedMatrixStructModule,
  packedTensorBoxModule,
  packedMatMulGateModule
];

function scalarNumber(value: RuntimeValue | undefined) {
  if (!value?.shape || !Array.isArray(value.data)) return undefined;
  const shape = value.shape as TensorShape;
  if (value.dtype !== "float32" || shape.dims.length !== 0 || value.data.length !== 1) return undefined;
  const scalar = Number(value.data[0]);
  return Number.isFinite(scalar) ? scalar : undefined;
}

function parseFiniteFloat32Param(value: unknown) {
  if (typeof value === "number") {
    return Number.isFinite(value) && Number.isFinite(Math.fround(value)) ? value : undefined;
  }

  const text = String(value ?? "").trim();
  if (!text) return undefined;
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(text)) return undefined;
  const parsed = Number(text);
  return Number.isFinite(parsed) && Number.isFinite(Math.fround(parsed)) ? parsed : undefined;
}

function asPackedComponent(module: ModuleDef, id: string, label: string): ModuleDef {
  return {
    ...module,
    id,
    label,
    summary: `Available MVP0.1 component: ${module.summary}`
  };
}
