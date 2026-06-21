import type { AxisName, DType, ModuleDef, RuntimeError, RuntimeValue, TensorShape } from "../types";
import { makeTensor, sampleValues, shapeOf, type TinyTensor } from "../runtime/tinyTensor";
import { mvp01CourseInputCount, type Mvp01CourseLevelDef } from "../../mvp01/mvp01CourseCatalog";
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

export const courseCaseSourceModule: ModuleDef = {
  id: "CourseCaseSource",
  label: "Course Case Source",
  category: "data",
  inputs: [],
  outputs: [{ id: "out", label: "case", direction: "out", emits: "float32" }],
  defaultParams: { inputKey: "case_input", dtype: "float32", dims: [1], axes: ["C"], seed: "course-source" },
  summary: "Provides the active course case value. The runtime dtype is controlled by the test case.",
  execute: ({ node, testInputs }) => {
    const inputKey = String(node.params.inputKey ?? "case_input");
    const testValue = testInputs[inputKey];
    if (testValue) return { outputs: { out: testValue }, samples: { out: previewRuntimeValue(testValue) } };
    const value = makeRuntimeValue(
      String(node.params.dtype ?? "float32") as DType,
      numericArrayParam(node.params.dims, [1]),
      stringArrayParam(node.params.axes, ["C"]) as AxisName[],
      String(node.params.seed ?? inputKey)
    );
    return { outputs: { out: value }, samples: { out: previewRuntimeValue(value) } };
  },
  infer: ({ node, testInputs }) => {
    const inputKey = String(node.params.inputKey ?? "case_input");
    const testValue = testInputs[inputKey];
    if (testValue?.shape) return { outputs: { out: testValue.shape } };
    return {
      outputs: {
        out: {
          dtype: String(node.params.dtype ?? "float32") as DType,
          dims: numericArrayParam(node.params.dims, [1]),
          axes: stringArrayParam(node.params.axes, ["C"]) as AxisName[]
        }
      }
    };
  }
};

export const typeContractGateModule: ModuleDef = {
  id: "TypeContractGate",
  label: "Type Contract Gate",
  category: "contract",
  inputs: [{ id: "x", label: "x", direction: "in", accepts: ["float32", "int", "mask", "bool", "raw_text", "string_piece", "token_piece"], required: true }],
  outputs: [{ id: "out", label: "out", direction: "out", emits: "float32" }],
  defaultParams: { expectedDType: "float32", expectedDims: [1], expectedAxes: ["C"] },
  summary: "Checks dtype, rank, dims, and axis names for course component outputs.",
  execute: ({ node, inputs }) => {
    const input = inputs.x;
    if (!input) return { outputs: {}, error: moduleError("missing_input", node.id, "TypeContractGate requires an input value", "RuntimeValue", input, "x") };
    const expectedDType = String(node.params.expectedDType ?? "float32") as DType;
    const expectedDims = numericArrayParam(node.params.expectedDims, []);
    const expectedAxes = stringArrayParam(node.params.expectedAxes, []) as AxisName[];
    const contractError = validateRuntimeContract(node.id, input, expectedDType, expectedDims, expectedAxes);
    if (contractError) return { outputs: {}, error: contractError };
    return { outputs: { out: input }, samples: { dtype: input.dtype, shape: input.shape, data: previewRuntimeValue(input) } };
  },
  infer: ({ node, inputs }) => {
    const input = inputs.x;
    if (!input?.shape) {
      return {
        outputs: {
          out: {
            dtype: String(node.params.expectedDType ?? "float32") as DType,
            dims: numericArrayParam(node.params.expectedDims, []),
            axes: stringArrayParam(node.params.expectedAxes, []) as AxisName[]
          }
        }
      };
    }
    return { outputs: { out: input.shape } };
  }
};

const courseComponentModules: ModuleDef[] = [];

export const mvp01ComponentModules = [
  float32LiteralModule,
  vectorRailModule,
  matrixStructModule,
  tensorBoxModule,
  courseCaseSourceModule,
  typeContractGateModule,
  packedScalarCellModule,
  packedVectorRailModule,
  packedMatrixStructModule,
  packedTensorBoxModule,
  packedMatMulGateModule,
  ...courseComponentModules
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

function createCourseComponentModule(level: Mvp01CourseLevelDef): ModuleDef {
  const inputCount = mvp01CourseInputCount(level);
  return {
    id: level.moduleId,
    label: `${level.component} v1`,
    category: courseModuleCategory(level.output.dtype),
    inputs: Array.from({ length: inputCount }, (_, index) => ({
      id: inputPortId(index),
      label: inputPortId(index),
      direction: "in" as const,
      accepts: ["float32", "int", "mask", "bool", "raw_text", "string_piece", "token_piece"],
      required: true
    })),
    outputs: [{ id: "out", label: level.output.axes.length ? `${level.output.dtype}[${level.output.axes.join(",")}]` : `${level.output.dtype}[]`, direction: "out", emits: level.output.dtype }],
    defaultParams: {
      outputDType: level.output.dtype,
      outputDims: level.output.dims,
      outputAxes: level.output.axes,
      seed: level.slug
    },
    summary: `${level.component} course component: ${level.learn}.`,
    pseudoCode: `${level.component}(${Array.from({ length: inputCount }, (_, index) => inputPortId(index)).join(", ")}) -> ${level.output.dtype}[${level.output.axes.join(",")}]`,
    execute: ({ node, inputs }) => {
      for (let index = 0; index < inputCount; index += 1) {
        const portId = inputPortId(index);
        if (!inputs[portId]) {
          return { outputs: {}, error: moduleError("missing_input", node.id, `${level.component} requires input ${portId}`, portId, undefined, portId) };
        }
      }
      const outputDType = String(node.params.outputDType ?? level.output.dtype) as DType;
      const outputDims = numericArrayParam(node.params.outputDims, level.output.dims);
      const outputAxes = stringArrayParam(node.params.outputAxes, level.output.axes) as AxisName[];
      const value = makeRuntimeValue(outputDType, outputDims, outputAxes, String(node.params.seed ?? level.slug), {
        component: level.component,
        courseLevel: level.code,
        inputs: Object.fromEntries(Object.entries(inputs).map(([key, value]) => [key, value.shape ?? value.dtype]))
      });
      return { outputs: { out: value }, samples: { out: previewRuntimeValue(value), contract: value.shape } };
    },
    infer: ({ node }) => ({
      outputs: {
        out: {
          dtype: String(node.params.outputDType ?? level.output.dtype) as DType,
          dims: numericArrayParam(node.params.outputDims, level.output.dims),
          axes: stringArrayParam(node.params.outputAxes, level.output.axes) as AxisName[]
        }
      }
    })
  };
}

function inputPortId(index: number) {
  return index === 0 ? "x" : index === 1 ? "y" : "z";
}

function courseModuleCategory(dtype: DType): ModuleDef["category"] {
  if (dtype === "raw_text" || dtype === "string_piece" || dtype === "token_piece") return "tokenizer";
  if (dtype === "mask") return "attention";
  return "tensor";
}

function makeRuntimeValue(dtype: DType, dims: number[], axes: AxisName[], seed: string, meta?: Record<string, unknown>): RuntimeValue {
  if (dtype === "raw_text") {
    const text = `case ${seed}: tiny model data flow`;
    return { dtype, data: text, meta: { ...meta, batch: [text] } };
  }

  const count = Math.max(1, elementCount(dims));
  if (dtype === "string_piece" || dtype === "token_piece") {
    return {
      dtype,
      shape: { dtype, dims, axes },
      data: Array.from({ length: count }, (_, index) => `${seed}_${index}`),
      meta
    };
  }

  const numericDType = dtype === "bool" || dtype === "mask" || dtype === "int" ? dtype : "float32";
  const data = deterministicNumbers(count, seed, numericDType);
  return {
    dtype,
    shape: { dtype, dims, axes },
    data,
    meta
  };
}

function validateRuntimeContract(nodeId: string, input: RuntimeValue, expectedDType: DType, expectedDims: number[], expectedAxes: AxisName[]) {
  if (input.dtype !== expectedDType) {
    return moduleError("dtype_mismatch", nodeId, "Output dtype does not match contract", expectedDType, input.dtype, "x");
  }
  if (expectedDType === "raw_text") return undefined;
  if (!input.shape) {
    return moduleError("shape_mismatch", nodeId, "Output must be shaped", { expectedDims, expectedAxes }, input, "x");
  }
  if (input.shape.axes.join(",") !== expectedAxes.join(",")) {
    return moduleError("axis_semantic_error", nodeId, "Output axes do not match contract", expectedAxes, input.shape.axes, "x");
  }
  if (input.shape.dims.join(",") !== expectedDims.join(",")) {
    return moduleError("shape_mismatch", nodeId, "Output dims do not match contract", expectedDims, input.shape.dims, "x");
  }
  return undefined;
}

function deterministicNumbers(count: number, seed: string, dtype: "float32" | "int" | "bool" | "mask") {
  let state = hashSeed(seed);
  return Array.from({ length: count }, () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    const normalized = state / 0xffffffff;
    if (dtype === "bool" || dtype === "mask") return normalized > 0.35 ? 1 : 0;
    if (dtype === "int") return Math.floor(normalized * 17);
    return Number((normalized * 2 - 1).toFixed(4));
  });
}

function hashSeed(seed: string) {
  let hash = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function elementCount(dims: number[]) {
  return dims.reduce((total, dim) => total * dim, 1);
}

function numericArrayParam(value: unknown, fallback: number[]) {
  return Array.isArray(value) && value.every((item) => Number.isFinite(Number(item))) ? value.map((item) => Number(item)) : fallback;
}

function stringArrayParam(value: unknown, fallback: string[]) {
  return Array.isArray(value) && value.every((item) => typeof item === "string") ? value : fallback;
}

function previewRuntimeValue(value: RuntimeValue) {
  if (Array.isArray(value.data)) return value.data.slice(0, 6);
  return value.data;
}
