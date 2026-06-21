import type { AxisName, ModuleDef, RuntimeError, RuntimeValue, TensorShape } from "../types";
import { valueKey } from "../types";
import {
  addTensors,
  broadcastTo,
  elementCount,
  getValue,
  makeTensor,
  matmul,
  sampleValues,
  seededTensor,
  shapeOf,
  transpose,
  type TinyTensor
} from "../runtime/tinyTensor";

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
  if (value.dtype !== "float32" && value.dtype !== "int" && value.dtype !== "bool" && value.dtype !== "mask") return undefined;
  return makeTensor(value.dtype, value.shape.dims, value.shape.axes, value.data as number[]);
}

function error(type: RuntimeError["type"], nodeId: string, message: string, expected?: unknown, received?: unknown, portId?: string): RuntimeError {
  return {
    type,
    nodeId,
    portId,
    message,
    expected,
    received
  };
}

export const inputTensorModule: ModuleDef = {
  id: "InputTensor",
  label: "Input Tensor",
  category: "data",
  inputs: [],
  outputs: [{ id: "out", label: "out", direction: "out", emits: "float32" }],
  defaultParams: { inputKey: "tensor", shape: [2, 3], axes: ["B", "C"], seed: "input" },
  summary: "Creates a deterministic small float32 tensor from test input or params.",
  execute: ({ node, testInputs }) => {
    const inputKey = String(node.params.inputKey ?? "tensor");
    const testValue = testInputs[inputKey];
    if (testValue) return { outputs: { out: testValue }, samples: { out: testValue.data?.slice?.(0, 6) } };

    const dims = node.params.shape as number[];
    const axes = node.params.axes as TensorShape["axes"];
    const seed = String(node.params.seed ?? node.id);
    const tensor = seededTensor(dims, axes, seed);
    return { outputs: { out: tensorValue(tensor) }, samples: { out: sampleValues(tensor) } };
  },
  infer: ({ node, testInputs }) => {
    const inputKey = String(node.params.inputKey ?? "tensor");
    const testValue = testInputs[inputKey];
    if (testValue?.shape) return { outputs: { out: testValue.shape } };
    return {
      outputs: {
        out: {
          dtype: "float32",
          dims: node.params.shape as number[],
          axes: node.params.axes as TensorShape["axes"]
        }
      }
    };
  }
};

export const weightPlateModule: ModuleDef = {
  id: "WeightPlate",
  label: "Weight Plate",
  category: "tensor",
  inputs: [],
  outputs: [{ id: "out", label: "weight", direction: "out", emits: "float32" }],
  defaultParams: { inputKey: "weight", shape: [3, 5], axes: ["C", "O"], seed: "weight", orientation: "C,O" },
  summary: "Creates or reads a deterministic weight tensor with an explicit storage orientation.",
  execute: ({ node, testInputs }) => {
    const inputKey = String(node.params.inputKey ?? "weight");
    const testValue = testInputs[inputKey];
    if (testValue) return { outputs: { out: testValue }, samples: { orientation: node.params.orientation, out: testValue.data?.slice?.(0, 6) } };
    const dims = node.params.shape as number[];
    const axes = node.params.axes as TensorShape["axes"];
    const seed = String(node.params.seed ?? node.id);
    const tensor = seededTensor(dims, axes, seed);
    return { outputs: { out: tensorValue(tensor) }, samples: { orientation: node.params.orientation, out: sampleValues(tensor) } };
  },
  infer: ({ node, testInputs }) => {
    const inputKey = String(node.params.inputKey ?? "weight");
    const testValue = testInputs[inputKey];
    if (testValue?.shape) return { outputs: { out: testValue.shape } };
    return {
      outputs: {
        out: {
          dtype: "float32",
          dims: node.params.shape as number[],
          axes: node.params.axes as TensorShape["axes"]
        }
      }
    };
  }
};

export const transposeSwitchModule: ModuleDef = {
  id: "TransposeSwitch",
  label: "Transpose Switch",
  category: "tensor",
  inputs: [{ id: "x", label: "x", direction: "in", accepts: ["float32"], required: true }],
  outputs: [{ id: "out", label: "xT", direction: "out", emits: "float32" }],
  defaultParams: { axisA: -2, axisB: -1 },
  summary: "Swaps two tensor axes and preserves all other axes.",
  execute: ({ node, inputs }) => {
    const input = tensorFromValue(inputs.x);
    if (!input) {
      return { outputs: {}, error: error("missing_input", node.id, "TransposeSwitch requires input tensor", "float32 tensor", inputs.x, "x") };
    }
    try {
      const out = transpose(input, Number(node.params.axisA ?? -2), Number(node.params.axisB ?? -1));
      return { outputs: { out: tensorValue(out) }, samples: { out: sampleValues(out) } };
    } catch (caught) {
      return { outputs: {}, error: error("axis_semantic_error", node.id, caught instanceof Error ? caught.message : "Transpose failed") };
    }
  },
  infer: ({ node, inputs }) => {
    const shape = inputs.x?.shape;
    if (!shape) return { outputs: {} };
    const rank = shape.dims.length;
    const axisA = normalizeAxis(Number(node.params.axisA ?? -2), rank);
    const axisB = normalizeAxis(Number(node.params.axisB ?? -1), rank);
    const dims = [...shape.dims];
    const axes = [...shape.axes];
    [dims[axisA], dims[axisB]] = [dims[axisB], dims[axisA]];
    [axes[axisA], axes[axisB]] = [axes[axisB], axes[axisA]];
    return { outputs: { out: { dtype: shape.dtype, dims, axes } } };
  }
};

export const matMulGateModule: ModuleDef = {
  id: "MatMulGate",
  label: "MatMul Gate",
  category: "tensor",
  inputs: [
    { id: "left", label: "left", direction: "in", accepts: ["float32"], required: true },
    { id: "right", label: "right", direction: "in", accepts: ["float32"], required: true }
  ],
  outputs: [{ id: "out", label: "out", direction: "out", emits: "float32" }],
  defaultParams: {},
  summary: "Runs [..., C] @ [C, O] or batched [..., M, C] @ [..., C, O].",
  pseudoCode: "out[..., o] = sum_c left[..., c] * right[c, o]\nqk[..., i, j] = sum_d Q[..., i, d] * Kt[..., d, j]",
  execute: ({ node, inputs }) => {
    const left = tensorFromValue(inputs.left);
    const right = tensorFromValue(inputs.right);
    if (!left || !right) {
      return { outputs: {}, error: error("missing_input", node.id, "MatMulGate requires left and right float32 inputs") };
    }
    try {
      const out = matmul(left, right);
      return { outputs: { out: tensorValue(out) }, samples: { out: sampleValues(out) } };
    } catch (caught) {
      return {
        outputs: {},
        error: error("shape_mismatch", node.id, caught instanceof Error ? caught.message : "MatMul failed", "[..., C] @ [C, O] or [..., M, C] @ [..., C, O]", {
          left: shapeOf(left),
          right: shapeOf(right)
        })
      };
    }
  },
  infer: ({ node, inputs }) => {
    const left = inputs.left?.shape;
    const right = inputs.right?.shape;
    if (!left || !right) return { outputs: {} };
    const rightIsMatrix = right.dims.length === 2;
    const leftC = left.dims[left.dims.length - 1];
    const rightC = rightIsMatrix ? right.dims[0] : right.dims[right.dims.length - 2];
    const leftCarrierDims = rightIsMatrix ? left.dims.slice(0, -1) : left.dims.slice(0, -2);
    const rightCarrierDims = rightIsMatrix ? [] : right.dims.slice(0, -2);
    if (!rightIsMatrix && leftCarrierDims.join(",") !== rightCarrierDims.join(",")) {
      return {
        outputs: {},
        error: error("shape_mismatch", node.id, "MatMul carrier dimensions mismatch", leftCarrierDims, rightCarrierDims, "right")
      };
    }
    if (leftC !== rightC) {
      return {
        outputs: {},
        error: error("shape_mismatch", node.id, "MatMul inner dimension mismatch", leftC, rightC, "right")
      };
    }
    const outDims = rightIsMatrix
      ? [...left.dims.slice(0, -1), right.dims[1]]
      : [...left.dims.slice(0, -2), left.dims[left.dims.length - 2], right.dims[right.dims.length - 1]];
    const outAxes = rightIsMatrix
      ? [...left.axes.slice(0, -1), right.axes[1] ?? "O"]
      : [...left.axes.slice(0, -2), left.axes[left.axes.length - 2], right.axes[right.axes.length - 1]];
    return {
      outputs: {
        out: {
          dtype: "float32",
          dims: outDims,
          axes: outAxes
        }
      }
    };
  }
};

export const elementwiseMultiplyModule: ModuleDef = {
  id: "ElementwiseMultiply",
  label: "Elementwise Multiply",
  category: "tensor",
  inputs: [
    { id: "left", label: "left", direction: "in", accepts: ["float32"], required: true },
    { id: "right", label: "right", direction: "in", accepts: ["float32"], required: true }
  ],
  outputs: [{ id: "out", label: "left*right", direction: "out", emits: "float32" }],
  defaultParams: {},
  summary: "Multiplies two same-shape float32 tensors cell by cell.",
  pseudoCode: "out[i] = left[i] * right[i]",
  execute: ({ node, inputs }) => {
    const left = tensorFromValue(inputs.left);
    const right = tensorFromValue(inputs.right);
    if (!left || !right) {
      return { outputs: {}, error: error("missing_input", node.id, "ElementwiseMultiply requires left and right float32 tensors") };
    }
    if (left.dims.join(",") !== right.dims.join(",") || left.axes.join(",") !== right.axes.join(",")) {
      return {
        outputs: {},
        error: error("shape_mismatch", node.id, "ElementwiseMultiply requires identical shape and axes", shapeOf(left), shapeOf(right))
      };
    }
    const out = makeTensor("float32", left.dims, left.axes, left.data.map((value, index) => Number((value * right.data[index]).toFixed(6))));
    return { outputs: { out: tensorValue(out) }, samples: { out: sampleValues(out) } };
  },
  infer: ({ inputs }) => {
    const left = inputs.left?.shape;
    const right = inputs.right?.shape;
    if (!left || !right) return { outputs: {} };
    if (left.dims.join(",") !== right.dims.join(",") || left.axes.join(",") !== right.axes.join(",")) {
      return { outputs: {}, error: error("shape_mismatch", "ElementwiseMultiply", "ElementwiseMultiply requires identical shape and axes", left, right) };
    }
    return { outputs: { out: left } };
  }
};

export const sumReduceModule: ModuleDef = {
  id: "SumReduce",
  label: "Sum Reduce",
  category: "tensor",
  inputs: [{ id: "x", label: "x", direction: "in", accepts: ["float32"], required: true }],
  outputs: [{ id: "out", label: "sum", direction: "out", emits: "float32" }],
  defaultParams: { axis: "C" },
  summary: "Sums a tensor along one named axis, removing that axis from the output.",
  pseudoCode: "out = x.sum(axis='C')",
  execute: ({ node, inputs }) => {
    const input = tensorFromValue(inputs.x);
    if (!input) return { outputs: {}, error: error("missing_input", node.id, "SumReduce requires a float32 tensor", "float32 tensor", inputs.x, "x") };
    const axis = String(node.params.axis ?? "C") as AxisName;
    const axisIndex = input.axes.indexOf(axis);
    if (axisIndex < 0) {
      return { outputs: {}, error: error("axis_semantic_error", node.id, `SumReduce axis ${axis} is missing`, axis, input.axes, "x") };
    }
    const out = reduceSum(input, axisIndex);
    return { outputs: { out: tensorValue(out, { reducedAxis: axis }) }, samples: { out: sampleValues(out), reducedAxis: axis } };
  },
  infer: ({ node, inputs }) => {
    const shape = inputs.x?.shape;
    if (!shape) return { outputs: {} };
    const axis = String(node.params.axis ?? "C") as AxisName;
    const axisIndex = shape.axes.indexOf(axis);
    if (axisIndex < 0) {
      return { outputs: {}, error: error("axis_semantic_error", "SumReduce", `SumReduce axis ${axis} is missing`, axis, shape.axes, "x") };
    }
    return {
      outputs: {
        out: {
          dtype: "float32",
          dims: shape.dims.filter((_, index) => index !== axisIndex),
          axes: shape.axes.filter((_, index) => index !== axisIndex)
        }
      }
    };
  }
};

export const outputContractGateModule: ModuleDef = {
  id: "OutputContractGate",
  label: "Output Contract Gate",
  category: "contract",
  inputs: [{ id: "x", label: "x", direction: "in", accepts: ["float32", "int", "mask", "bool"], required: true }],
  outputs: [{ id: "out", label: "out", direction: "out" }],
  defaultParams: { expectedAxes: [] },
  summary: "Passes a value through only if its axis contract matches.",
  execute: ({ node, inputs }) => {
    const input = inputs.x;
    if (!input?.shape) {
      return { outputs: {}, error: error("missing_input", node.id, "OutputContractGate requires a shaped value", "shaped RuntimeValue", input, "x") };
    }
    const expectedAxes = node.params.expectedAxes as string[] | undefined;
    if (expectedAxes?.length && expectedAxes.join(",") !== input.shape.axes.join(",")) {
      return {
        outputs: {},
        error: error("axis_semantic_error", node.id, "Output axes do not match expected contract", expectedAxes, input.shape.axes)
      };
    }
    return { outputs: { out: input }, samples: { contract: input.shape.axes.join(",") } };
  },
  infer: ({ inputs }) => (inputs.x?.shape ? { outputs: { out: inputs.x.shape } } : { outputs: {} })
};

export const axisLockModule: ModuleDef = {
  id: "AxisLock",
  label: "Axis Lock",
  category: "contract",
  inputs: [{ id: "x", label: "x", direction: "in", accepts: ["float32"], required: true }],
  outputs: [{ id: "out", label: "locked", direction: "out", emits: "float32" }],
  defaultParams: { expectedPrefixAxes: ["B", "H"] },
  summary: "Passes a tensor through only when protected carrier axes keep their order.",
  execute: ({ node, inputs }) => {
    const input = inputs.x;
    if (!input?.shape) {
      return { outputs: {}, error: error("missing_input", node.id, "AxisLock requires a shaped tensor", "float32 tensor", input, "x") };
    }
    const expectedPrefixAxes = (node.params.expectedPrefixAxes as AxisName[] | undefined) ?? [];
    const receivedPrefixAxes = input.shape.axes.slice(0, expectedPrefixAxes.length);
    if (expectedPrefixAxes.length && expectedPrefixAxes.join(",") !== receivedPrefixAxes.join(",")) {
      return {
        outputs: {},
        error: error("axis_semantic_error", node.id, "Protected carrier axes were changed", expectedPrefixAxes, receivedPrefixAxes, "x")
      };
    }
    return {
      outputs: { out: input },
      samples: {
        lockedAxes: receivedPrefixAxes,
        shape: input.shape
      }
    };
  },
  infer: ({ inputs }) => (inputs.x?.shape ? { outputs: { out: inputs.x.shape } } : { outputs: {} })
};

export const axisAlignmentRulerModule: ModuleDef = {
  id: "AxisAlignmentRuler",
  label: "Axis Alignment Ruler",
  category: "contract",
  inputs: [{ id: "x", label: "small", direction: "in", accepts: ["float32"], required: true }],
  outputs: [{ id: "out", label: "small", direction: "out", emits: "float32" }],
  defaultParams: { expectedAxes: ["O"] },
  summary: "Checks that a small tensor is attached to the intended semantic axes before broadcasting.",
  execute: ({ node, inputs }) => {
    const input = inputs.x;
    if (!input?.shape) {
      return { outputs: {}, error: error("missing_input", node.id, "AxisAlignmentRuler requires a shaped small tensor", "float32 tensor", input, "x") };
    }
    const expectedAxes = (node.params.expectedAxes as AxisName[] | undefined) ?? [];
    if (expectedAxes.length && expectedAxes.join(",") !== input.shape.axes.join(",")) {
      return {
        outputs: {},
        error: error("axis_semantic_error", node.id, "Small tensor axes do not match the selected semantic slot", expectedAxes, input.shape.axes, "x")
      };
    }
    return { outputs: { out: input }, samples: { expectedAxes, receivedAxes: input.shape.axes } };
  },
  infer: ({ node, inputs }) => {
    const shape = inputs.x?.shape;
    if (!shape) return { outputs: {} };
    const expectedAxes = (node.params.expectedAxes as AxisName[] | undefined) ?? [];
    if (expectedAxes.length && expectedAxes.join(",") !== shape.axes.join(",")) {
      return {
        outputs: {},
        error: error("axis_semantic_error", node.id, "Small tensor axes do not match the selected semantic slot", expectedAxes, shape.axes, "x")
      };
    }
    return { outputs: { out: shape } };
  }
};

export const broadcastRailModule: ModuleDef = {
  id: "BroadcastRail",
  label: "Broadcast Rail",
  category: "tensor",
  inputs: [
    { id: "small", label: "small", direction: "in", accepts: ["float32"], required: true },
    { id: "target", label: "target", direction: "in", accepts: ["float32"], required: true }
  ],
  outputs: [{ id: "out", label: "expanded", direction: "out", emits: "float32" }],
  defaultParams: { alignAxes: ["O"] },
  summary: "Logically expands a small tensor along the unselected target axes.",
  execute: ({ node, inputs }) => {
    const small = tensorFromValue(inputs.small);
    const target = tensorFromValue(inputs.target);
    if (!small || !target) {
      return { outputs: {}, error: error("missing_input", node.id, "BroadcastRail requires small and target tensors") };
    }
    const alignAxes = (node.params.alignAxes as AxisName[] | undefined) ?? [];
    try {
      const expanded = broadcastTo(small, target.dims, target.axes, alignAxes);
      const semanticWarning = alignAxes.join(",") !== small.axes.join(",");
      return {
        outputs: {
          out: tensorValue(expanded, {
            broadcast: {
              logical: true,
              sourceAxes: small.axes,
              sourceDims: small.dims,
              alignAxes,
              targetAxes: target.axes,
              targetDims: target.dims,
              semanticWarning
            }
          })
        },
        samples: {
          logical: true,
          source: shapeOf(small),
          target: shapeOf(target),
          alignAxes,
          semanticWarning,
          ghostIndex: previewBroadcastIndex(small, target, alignAxes)
        }
      };
    } catch (caught) {
      return {
        outputs: {},
        error: error("shape_mismatch", node.id, caught instanceof Error ? caught.message : "Broadcast failed", alignAxes, {
          small: shapeOf(small),
          target: shapeOf(target)
        })
      };
    }
  },
  infer: ({ node, inputs }) => {
    const targetShape = inputs.target?.shape;
    const smallShape = inputs.small?.shape;
    if (!targetShape || !smallShape) return { outputs: {} };
    const alignAxes = (node.params.alignAxes as AxisName[] | undefined) ?? [];
    if (alignAxes.length !== smallShape.dims.length) {
      return {
        outputs: {},
        error: error("shape_mismatch", node.id, "Broadcast axis rank mismatch", alignAxes, smallShape.axes, "small")
      };
    }
    return { outputs: { out: targetShape } };
  }
};

export const ghostExpansionPreviewModule: ModuleDef = {
  id: "GhostExpansionPreview",
  label: "Ghost Expansion",
  category: "probe",
  inputs: [{ id: "x", label: "expanded", direction: "in", accepts: ["float32"], required: true }],
  outputs: [{ id: "out", label: "expanded", direction: "out", emits: "float32" }],
  defaultParams: {},
  summary: "Shows that broadcast is logical expansion instead of a new repeated parameter tensor.",
  execute: ({ node, inputs }) => {
    const input = inputs.x;
    if (!input?.shape) {
      return { outputs: {}, error: error("missing_input", node.id, "GhostExpansionPreview requires a shaped broadcast value", "float32 tensor", input, "x") };
    }
    return {
      outputs: { out: input },
      samples: {
        broadcast: input.meta?.broadcast ?? "no broadcast metadata",
        previewValues: Array.isArray(input.data) ? input.data.slice(0, 8) : []
      }
    };
  },
  infer: ({ inputs }) => (inputs.x?.shape ? { outputs: { out: inputs.x.shape } } : { outputs: {} })
};

export const semanticWarningLensModule: ModuleDef = {
  id: "SemanticWarningLens",
  label: "Semantic Warning Lens",
  category: "probe",
  inputs: [{ id: "x", label: "value", direction: "in", accepts: ["float32"], required: true }],
  outputs: [{ id: "out", label: "value", direction: "out", emits: "float32" }],
  defaultParams: {},
  summary: "Surfaces cases where shape can broadcast but semantic axis labels disagree.",
  execute: ({ node, inputs }) => {
    const input = inputs.x;
    if (!input?.shape) {
      return { outputs: {}, error: error("missing_input", node.id, "SemanticWarningLens requires a shaped value", "float32 tensor", input, "x") };
    }
    const broadcast = input.meta?.broadcast as { semanticWarning?: boolean; sourceAxes?: AxisName[]; alignAxes?: AxisName[] } | undefined;
    return {
      outputs: { out: input },
      samples: {
        warning: broadcast?.semanticWarning ? "shape can align, but source axes differ from selected semantic slot" : "none",
        sourceAxes: broadcast?.sourceAxes ?? input.shape.axes,
        alignAxes: broadcast?.alignAxes ?? input.shape.axes
      }
    };
  },
  infer: ({ inputs }) => (inputs.x?.shape ? { outputs: { out: inputs.x.shape } } : { outputs: {} })
};

export const addGateModule: ModuleDef = {
  id: "AddGate",
  label: "Add Gate",
  category: "tensor",
  inputs: [
    { id: "left", label: "left", direction: "in", accepts: ["float32"], required: true },
    { id: "right", label: "right", direction: "in", accepts: ["float32"], required: true }
  ],
  outputs: [{ id: "out", label: "out", direction: "out", emits: "float32" }],
  defaultParams: {},
  summary: "Adds two already aligned tensors with identical shape and axis semantics.",
  pseudoCode: "out[b,t,o] = left[b,t,o] + right[b,t,o]",
  execute: ({ node, inputs }) => {
    const left = tensorFromValue(inputs.left);
    const right = tensorFromValue(inputs.right);
    if (!left || !right) {
      return { outputs: {}, error: error("missing_input", node.id, "AddGate requires left and right float32 tensors") };
    }
    try {
      const out = addTensors(left, right);
      return { outputs: { out: tensorValue(out) }, samples: { out: sampleValues(out) } };
    } catch (caught) {
      return {
        outputs: {},
        error: error("shape_mismatch", node.id, caught instanceof Error ? caught.message : "AddGate failed", shapeOf(left), shapeOf(right))
      };
    }
  },
  infer: ({ node, inputs }) => {
    const left = inputs.left?.shape;
    const right = inputs.right?.shape;
    if (!left || !right) return { outputs: {} };
    if (left.axes.join(",") !== right.axes.join(",") || left.dims.join(",") !== right.dims.join(",")) {
      return {
        outputs: {},
        error: error("shape_mismatch", node.id, "AddGate requires identical shapes after broadcast", left, right)
      };
    }
    return { outputs: { out: left } };
  }
};

export const causalMaskModule: ModuleDef = {
  id: "CausalMask",
  label: "Causal Mask",
  category: "attention",
  inputs: [{ id: "target", label: "scores", direction: "in", accepts: ["float32"], required: true }],
  outputs: [{ id: "out", label: "mask", direction: "out", emits: "float32" }],
  defaultParams: { maskOrientation: "query_key", maskedValue: -10000 },
  summary: "Creates an additive causal mask where future key positions receive a large negative value.",
  pseudoCode: "mask[i,j] = key_j > query_i ? -10000 : 0",
  execute: ({ node, inputs }) => {
    const target = tensorFromValue(inputs.target);
    if (!target) {
      return { outputs: {}, error: error("missing_input", node.id, "CausalMask requires score board target tensor", "float32 tensor", inputs.target, "target") };
    }
    if (target.dims.length < 2) {
      return { outputs: {}, error: error("shape_mismatch", node.id, "CausalMask target must have query/key axes", "rank >= 2", shapeOf(target)) };
    }
    const querySize = target.dims[target.dims.length - 2];
    const keySize = target.dims[target.dims.length - 1];
    if (querySize !== keySize) {
      return { outputs: {}, error: error("shape_mismatch", node.id, "CausalMask MVP expects square T/T score board", querySize, keySize) };
    }
    const mask = createCausalMaskTensor(querySize, String(node.params.maskOrientation ?? "query_key"), Number(node.params.maskedValue ?? -10000));
    return {
      outputs: { out: tensorValue(mask, { maskOrientation: node.params.maskOrientation ?? "query_key" }) },
      samples: {
        orientation: node.params.maskOrientation ?? "query_key",
        preview: mask.data.slice(0, Math.min(mask.data.length, 12))
      }
    };
  },
  infer: ({ node, inputs }) => {
    const target = inputs.target?.shape;
    if (!target) return { outputs: {} };
    if (target.dims.length < 2) {
      return {
        outputs: {},
        error: error("shape_mismatch", node.id, "CausalMask target must have query/key axes", "rank >= 2", target)
      };
    }
    const querySize = target.dims[target.dims.length - 2];
    const keySize = target.dims[target.dims.length - 1];
    if (querySize !== keySize) {
      return {
        outputs: {},
        error: error("shape_mismatch", node.id, "CausalMask MVP expects square T/T score board", querySize, keySize)
      };
    }
    return { outputs: { out: { dtype: "float32", dims: [querySize, keySize], axes: ["T", "T"] } } };
  }
};

export const scoreBoardModule: ModuleDef = {
  id: "ScoreBoard",
  label: "Score Board",
  category: "attention",
  inputs: [{ id: "scores", label: "scores", direction: "in", accepts: ["float32"], required: true }],
  outputs: [{ id: "out", label: "scores", direction: "out", emits: "float32" }],
  defaultParams: { expectedAxes: ["B", "H", "T", "T"] },
  summary: "Validates the attention score board contract after Q @ K.transpose(-2,-1).",
  execute: ({ node, inputs }) => {
    const scores = inputs.scores;
    if (!scores?.shape) {
      return { outputs: {}, error: error("missing_input", node.id, "ScoreBoard requires QK score tensor", "float32 tensor", scores, "scores") };
    }
    const expectedAxes = (node.params.expectedAxes as AxisName[] | undefined) ?? [];
    if (expectedAxes.length && expectedAxes.join(",") !== scores.shape.axes.join(",")) {
      return {
        outputs: {},
        error: error("axis_semantic_error", node.id, "ScoreBoard axes must be [B,H,T,T]", expectedAxes, scores.shape.axes, "scores")
      };
    }
    return {
      outputs: { out: scores },
      samples: {
        shape: scores.shape,
        values: Array.isArray(scores.data) ? scores.data.slice(0, 8) : []
      }
    };
  },
  infer: ({ node, inputs }) => {
    const shape = inputs.scores?.shape;
    if (!shape) return { outputs: {} };
    const expectedAxes = (node.params.expectedAxes as AxisName[] | undefined) ?? [];
    if (expectedAxes.length && expectedAxes.join(",") !== shape.axes.join(",")) {
      return {
        outputs: {},
        error: error("axis_semantic_error", node.id, "ScoreBoard axes must be [B,H,T,T]", expectedAxes, shape.axes, "scores")
      };
    }
    return { outputs: { out: shape } };
  }
};

export const cellTraceModule: ModuleDef = {
  id: "CellTrace",
  label: "Cell Trace",
  category: "probe",
  inputs: [
    { id: "scores", label: "value", direction: "in", accepts: ["float32"], required: true },
    { id: "q", label: "Q", direction: "in", accepts: ["float32"] },
    { id: "k", label: "K", direction: "in", accepts: ["float32"] },
    { id: "left", label: "left", direction: "in", accepts: ["float32"] },
    { id: "small", label: "small", direction: "in", accepts: ["float32"] }
  ],
  outputs: [{ id: "out", label: "value", direction: "out", emits: "float32" }],
  defaultParams: { b: 0, h: 0, i: 0, j: 1, t: 0, o: 1 },
  summary: "Shows one output cell as a QK dot product or a broadcast-add source lookup.",
  pseudoCode: "scores[b,h,i,j] = sum_d Q[b,h,i,d] * K[b,h,j,d]\nout[b,t,o] = left[b,t,o] + small[o]",
  execute: ({ node, inputs }) => {
    const scores = tensorFromValue(inputs.scores);
    const q = tensorFromValue(inputs.q);
    const k = tensorFromValue(inputs.k);
    const left = tensorFromValue(inputs.left);
    const small = tensorFromValue(inputs.small);
    if (!scores) {
      return { outputs: {}, error: error("missing_input", node.id, "CellTrace requires an output value") };
    }
    const sample =
      q && k
        ? sampleQkCell(q, k, scores, {
            b: Number(node.params.b ?? 0),
            h: Number(node.params.h ?? 0),
            i: Number(node.params.i ?? 0),
            j: Number(node.params.j ?? 1)
          })
        : left && small
          ? sampleBroadcastAddCell(left, small, scores, {
              b: Number(node.params.b ?? 0),
              h: Number(node.params.h ?? 0),
              t: Number(node.params.t ?? 0),
              i: Number(node.params.i ?? 0),
              j: Number(node.params.j ?? 1),
              o: Number(node.params.o ?? 1)
            })
          : { note: "Connect Q/K for attention trace or left/small for broadcast-add trace.", outputShape: shapeOf(scores) };
    return { outputs: { out: inputs.scores }, samples: sample };
  },
  infer: ({ inputs }) => (inputs.scores?.shape ? { outputs: { out: inputs.scores.shape } } : { outputs: {} })
};

export const referenceCheckerModule: ModuleDef = {
  id: "ReferenceChecker",
  label: "Reference Checker",
  category: "probe",
  inputs: [{ id: "x", label: "x", direction: "in", accepts: ["float32"], required: true }],
  outputs: [{ id: "out", label: "reference", direction: "out", emits: "float32" }],
  defaultParams: { referenceKey: "reference" },
  summary: "Prebuilt probe: x marks the value under test, and reference comes from the active test case.",
  pseudoCode: "reference = test_case[referenceKey]\n# probe only; not part of the reusable component",
  execute: ({ node, testInputs }) => {
    const referenceKey = String(node.params.referenceKey ?? "reference");
    const reference = testInputs[referenceKey];
    if (!reference) {
      return { outputs: {}, error: error("missing_input", node.id, `Missing reference input ${referenceKey}`, referenceKey) };
    }
    return { outputs: { out: reference }, samples: { out: reference.data?.slice?.(0, 6) } };
  },
  infer: ({ node, testInputs }) => {
    const referenceKey = String(node.params.referenceKey ?? "reference");
    const reference = testInputs[referenceKey];
    return reference?.shape ? { outputs: { out: reference.shape } } : { outputs: {} };
  }
};

export const tensorModules = [
  inputTensorModule,
  weightPlateModule,
  transposeSwitchModule,
  matMulGateModule,
  elementwiseMultiplyModule,
  sumReduceModule,
  outputContractGateModule,
  axisLockModule,
  axisAlignmentRulerModule,
  broadcastRailModule,
  ghostExpansionPreviewModule,
  semanticWarningLensModule,
  addGateModule,
  causalMaskModule,
  scoreBoardModule,
  cellTraceModule,
  referenceCheckerModule
];

function normalizeAxis(axis: number, rank: number) {
  const normalized = axis < 0 ? rank + axis : axis;
  if (normalized < 0 || normalized >= rank) throw new Error(`Axis ${axis} is out of range for rank ${rank}`);
  return normalized;
}

function reduceSum(tensor: TinyTensor, axisIndex: number) {
  const outDims = tensor.dims.filter((_, index) => index !== axisIndex);
  const outAxes = tensor.axes.filter((_, index) => index !== axisIndex);
  const outCount = Math.max(1, elementCount(outDims));
  const data = Array.from({ length: outCount }, () => 0);
  tensor.data.forEach((value, flatIndex) => {
    const inputIndex = unravelTensorIndex(flatIndex, tensor.dims);
    const outputIndex = inputIndex.filter((_, index) => index !== axisIndex);
    const outputFlat = ravelTensorIndex(outputIndex, outDims);
    data[outputFlat] = Number((data[outputFlat] + value).toFixed(6));
  });
  return makeTensor("float32", outDims, outAxes, data);
}

function ravelTensorIndex(index: number[], dims: number[]) {
  if (!dims.length) return 0;
  let flat = 0;
  for (let dimIndex = 0; dimIndex < dims.length; dimIndex += 1) {
    flat = flat * dims[dimIndex] + index[dimIndex];
  }
  return flat;
}

function unravelTensorIndex(flatIndex: number, dims: number[]) {
  if (!dims.length) return [];
  const index = Array.from({ length: dims.length }, () => 0);
  let remainder = flatIndex;
  for (let dimIndex = dims.length - 1; dimIndex >= 0; dimIndex -= 1) {
    index[dimIndex] = remainder % dims[dimIndex];
    remainder = Math.floor(remainder / dims[dimIndex]);
  }
  return index;
}

export function getNodeOutput(values: Record<string, RuntimeValue>, nodeId: string) {
  return values[valueKey(nodeId, "out")];
}

function sampleQkCell(q: TinyTensor, k: TinyTensor, scores: TinyTensor, requested: { b: number; h: number; i: number; j: number }) {
  if (q.dims.length !== 4 || k.dims.length !== 4 || scores.dims.length !== 4) {
    return {
      note: "CellTrace expects rank-4 Q, K, and scores tensors.",
      qShape: shapeOf(q),
      kShape: shapeOf(k),
      scoresShape: shapeOf(scores)
    };
  }

  const b = clampIndex(requested.b, q.dims[0]);
  const h = clampIndex(requested.h, q.dims[1]);
  const i = clampIndex(requested.i, q.dims[2]);
  const j = clampIndex(requested.j, k.dims[2]);
  const dSize = q.dims[3];
  const qVector = Array.from({ length: dSize }, (_, d) => getValue(q, [b, h, i, d]));
  const kVector = Array.from({ length: dSize }, (_, d) => getValue(k, [b, h, j, d]));
  const dot = Number(qVector.reduce((sum, qValue, d) => sum + qValue * kVector[d], 0).toFixed(6));
  const actual = getValue(scores, [b, h, i, j]);
  return {
    cell: { b, h, queryToken: i, keyToken: j },
    formula: "dot(Q[b,h,i,:], K[b,h,j,:])",
    qVector,
    kVector,
    expectedDot: dot,
    actualScore: actual,
    delta: Number(Math.abs(dot - actual).toFixed(6))
  };
}

function sampleBroadcastAddCell(
  left: TinyTensor,
  small: TinyTensor,
  out: TinyTensor,
  requested: { b: number; h?: number; t: number; i?: number; j?: number; o: number }
) {
  if (left.dims.length !== out.dims.length || out.dims.length < 1) {
    return {
      note: "Broadcast-add trace expects left and output tensors with the same rank.",
      leftShape: shapeOf(left),
      smallShape: shapeOf(small),
      outShape: shapeOf(out)
    };
  }

  const outIndex = out.axes.map((axis, axisIndex) => {
    const occurrence = axisOccurrence(out.axes, axis, axisIndex);
    if (axis === "B") return clampIndex(requested.b, out.dims[axisIndex]);
    if (axis === "H") return clampIndex(requested.h ?? 0, out.dims[axisIndex]);
    if (axis === "T" && countAxis(out.axes, "T") > 1) {
      return occurrence === 0
        ? clampIndex(requested.i ?? requested.t, out.dims[axisIndex])
        : clampIndex(requested.j ?? requested.t, out.dims[axisIndex]);
    }
    if (axis === "T") return clampIndex(requested.t, out.dims[axisIndex]);
    if (axis === "O" || axis === "C") return clampIndex(requested.o, out.dims[axisIndex]);
    return 0;
  });
  const smallIndex = small.axes.map((axis, axisIndex) => {
    const outAxisIndex = targetSlotForAxisOccurrence(out.axes, axis, axisOccurrence(small.axes, axis, axisIndex));
    return outAxisIndex >= 0 ? outIndex[outAxisIndex] : 0;
  });
  const leftValue = getValue(left, outIndex);
  const smallValue = getValue(small, smallIndex);
  const actual = getValue(out, outIndex);
  return {
    cell: Object.fromEntries(out.axes.map((axis, index) => [axis, outIndex[index]])),
    formula: "left[cell] + small[aligned semantic index]",
    leftIndex: outIndex,
    smallAxes: small.axes,
    smallIndex,
    leftValue,
    smallValue,
    expectedSum: Number((leftValue + smallValue).toFixed(6)),
    actualValue: actual,
    delta: Number(Math.abs(leftValue + smallValue - actual).toFixed(6))
  };
}

function previewBroadcastIndex(small: TinyTensor, target: TinyTensor, alignAxes: AxisName[]) {
  const targetIndex = target.dims.map((dim) => (dim > 1 ? 1 : 0));
  const smallIndex = alignAxes.map((axis, axisIndex) => {
    const targetAxisIndex = targetSlotForAxisOccurrence(target.axes, axis, axisOccurrence(alignAxes, axis, axisIndex));
    return targetAxisIndex >= 0 ? targetIndex[targetAxisIndex] : 0;
  });
  return {
    targetCell: Object.fromEntries(target.axes.map((axis, index) => [axis, targetIndex[index]])),
    smallCell: Object.fromEntries(small.axes.map((axis, index) => [axis, smallIndex[index] ?? 0]))
  };
}

function axisOccurrence(axes: AxisName[], axis: AxisName, axisIndex: number) {
  return axes.slice(0, axisIndex + 1).filter((item) => item === axis).length - 1;
}

function countAxis(axes: AxisName[], axis: AxisName) {
  return axes.filter((item) => item === axis).length;
}

function targetSlotForAxisOccurrence(axes: AxisName[], axis: AxisName, occurrence: number) {
  let seen = 0;
  for (let index = 0; index < axes.length; index += 1) {
    if (axes[index] !== axis) continue;
    if (seen === occurrence) return index;
    seen += 1;
  }
  return -1;
}

function createCausalMaskTensor(size: number, orientation: string, maskedValue: number) {
  const data: number[] = [];
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const queryIndex = orientation === "key_query" ? col : row;
      const keyIndex = orientation === "key_query" ? row : col;
      data.push(keyIndex > queryIndex ? maskedValue : 0);
    }
  }
  return makeTensor("float32", [size, size], ["T", "T"], data);
}

function clampIndex(index: number, size: number) {
  if (!Number.isFinite(index)) return 0;
  return Math.min(Math.max(0, Math.trunc(index)), Math.max(0, size - 1));
}
