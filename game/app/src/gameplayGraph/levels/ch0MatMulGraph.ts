import type { GraphSpec, LevelSpec, RuntimeValue } from "../types";
import { referenceLinear } from "../runtime/reference";
import { seededTensor, type TinyTensor } from "../runtime/tinyTensor";

const visibleHidden = seededTensor([2, 4, 3], ["B", "T", "C"], "matmul-visible-hidden");
const visibleWeight = seededTensor([5, 3], ["O", "C"], "matmul-visible-weight");
const visibleReference = referenceLinear(visibleHidden, visibleWeight, "O,C");

const hiddenMutationHidden = seededTensor([4, 3, 2], ["B", "T", "C"], "matmul-hidden-mutation-hidden");
const hiddenMutationWeight = seededTensor([16, 2], ["O", "C"], "matmul-hidden-mutation-weight");
const hiddenMutationReference = referenceLinear(hiddenMutationHidden, hiddenMutationWeight, "O,C");

const hiddenEqualHidden = seededTensor([2, 3, 4], ["B", "T", "C"], "matmul-hidden-equal-hidden");
const hiddenEqualWeight = seededTensor([4, 4], ["O", "C"], "matmul-hidden-equal-weight");
const hiddenEqualReference = referenceLinear(hiddenEqualHidden, hiddenEqualWeight, "O,C");

const hiddenAxisHidden = seededTensor([1, 5, 3], ["B", "T", "C"], "matmul-hidden-axis-hidden");
const hiddenAxisWeight = seededTensor([7, 3], ["O", "C"], "matmul-hidden-axis-weight");
const hiddenAxisReference = referenceLinear(hiddenAxisHidden, hiddenAxisWeight, "O,C");

export const ch0MatMulGraph: LevelSpec = {
  id: "ch0_2_matmul_graph",
  title: "0-2 MatMul Gate Graph",
  mode: "graph_challenge",
  chapter: "Chapter 0",
  goal: "Build hidden[B,T,C] @ stored_weight[O,C].transpose() -> projected[B,T,O]",
  modulePalette: ["InputTensor", "WeightPlate", "TransposeSwitch", "MatMulGate", "OutputContractGate", "ReferenceChecker"],
  initialGraph: {
    levelId: "ch0_2_matmul_graph",
    version: 1,
    nodes: [],
    edges: [],
    outputNodes: ["projected", "reference"]
  },
  constraints: {
    maxNodes: 8,
    maxEdges: 8,
    forbiddenModules: ["AddGate", "Softmax"]
  },
  visibleTests: [
    {
      id: "stored_weight_visible",
      title: "stored weight requires transpose",
      visibility: "visible",
      inputSeed: "matmul-visible",
      inputs: {
        hidden: runtimeValue(visibleHidden),
        weight: runtimeValue(visibleWeight),
        reference: runtimeValue(visibleReference)
      },
      assertions: [
        { type: "shape", nodeId: "projected", expectedAxes: ["B", "T", "O"], expectedDims: [2, 4, 5] },
        { type: "allclose", nodeId: "projected", referenceNodeId: "reference", atol: 1e-5 }
      ]
    }
  ],
  hiddenTests: [
    {
      id: "mutation_btco",
      title: "B/T/C/O mutation",
      visibility: "hidden",
      inputSeed: "matmul-hidden-mutation",
      inputs: {
        hidden: runtimeValue(hiddenMutationHidden),
        weight: runtimeValue(hiddenMutationWeight),
        reference: runtimeValue(hiddenMutationReference)
      },
      assertions: [
        { type: "shape", nodeId: "projected", expectedAxes: ["B", "T", "O"], expectedDims: [4, 3, 16] },
        { type: "allclose", nodeId: "projected", referenceNodeId: "reference", atol: 1e-5 }
      ]
    },
    {
      id: "co_equal_orientation_trap",
      title: "C/O equal orientation trap",
      visibility: "hidden",
      inputSeed: "matmul-hidden-equal",
      inputs: {
        hidden: runtimeValue(hiddenEqualHidden),
        weight: runtimeValue(hiddenEqualWeight),
        reference: runtimeValue(hiddenEqualReference)
      },
      assertions: [
        { type: "shape", nodeId: "projected", expectedAxes: ["B", "T", "O"], expectedDims: [2, 3, 4] },
        { type: "allclose", nodeId: "projected", referenceNodeId: "reference", atol: 1e-5 }
      ]
    },
    {
      id: "output_axis_contract",
      title: "output axis contract stays [B,T,O]",
      visibility: "hidden",
      inputSeed: "matmul-hidden-axis",
      inputs: {
        hidden: runtimeValue(hiddenAxisHidden),
        weight: runtimeValue(hiddenAxisWeight),
        reference: runtimeValue(hiddenAxisReference)
      },
      assertions: [
        { type: "shape", nodeId: "projected", expectedAxes: ["B", "T", "O"], expectedDims: [1, 5, 7] },
        { type: "allclose", nodeId: "projected", referenceNodeId: "reference", atol: 1e-5 }
      ]
    }
  ],
  debrief: {
    completeTitle: "MatMul Gate Restored",
    fixedProblem: "stored_weight[O,C] is now transposed into compute_weight[C,O] before MatMul.",
    learned: "MatMul consumes C, preserves B/T, and creates O. Shape can look plausible when C and O match, so reference diff matters.",
    nextUse: "QK transpose uses the same inner-dimension contract."
  }
};

export function createCh0MatMulSolutionGraph(): GraphSpec {
  return {
    levelId: "ch0_2_matmul_graph",
    version: 1,
    nodes: [
      { id: "hidden", moduleId: "InputTensor", params: { inputKey: "hidden" }, position: { x: 80, y: 120 } },
      { id: "weight", moduleId: "WeightPlate", params: { inputKey: "weight", orientation: "O,C" }, position: { x: 80, y: 280 } },
      { id: "weight_transpose", moduleId: "TransposeSwitch", params: { axisA: 0, axisB: 1 }, position: { x: 330, y: 280 } },
      { id: "matmul", moduleId: "MatMulGate", params: {}, position: { x: 580, y: 190 } },
      { id: "projected", moduleId: "OutputContractGate", params: { expectedAxes: ["B", "T", "O"] }, position: { x: 830, y: 190 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 830, y: 360 } }
    ],
    edges: [
      { id: "e_hidden_matmul", from: { nodeId: "hidden", portId: "out" }, to: { nodeId: "matmul", portId: "left" } },
      { id: "e_weight_transpose", from: { nodeId: "weight", portId: "out" }, to: { nodeId: "weight_transpose", portId: "x" } },
      { id: "e_transpose_matmul", from: { nodeId: "weight_transpose", portId: "out" }, to: { nodeId: "matmul", portId: "right" } },
      { id: "e_matmul_projected", from: { nodeId: "matmul", portId: "out" }, to: { nodeId: "projected", portId: "x" } },
      { id: "e_projected_ref", from: { nodeId: "projected", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["projected", "reference"]
  };
}

export function createCh0MatMulNoTransposeGraph(): GraphSpec {
  const graph = createCh0MatMulSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => node.id !== "weight_transpose"),
    edges: [
      { id: "e_hidden_matmul", from: { nodeId: "hidden", portId: "out" }, to: { nodeId: "matmul", portId: "left" } },
      { id: "e_weight_matmul", from: { nodeId: "weight", portId: "out" }, to: { nodeId: "matmul", portId: "right" } },
      { id: "e_matmul_projected", from: { nodeId: "matmul", portId: "out" }, to: { nodeId: "projected", portId: "x" } },
      { id: "e_projected_ref", from: { nodeId: "projected", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ]
  };
}

export function createCh0MatMulWrongOutputAxesGraph(): GraphSpec {
  const graph = createCh0MatMulSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.map((node) =>
      node.id === "projected" ? { ...node, params: { expectedAxes: ["B", "O", "T"] } } : node
    )
  };
}

function runtimeValue(tensor: TinyTensor): RuntimeValue {
  return {
    dtype: tensor.dtype,
    shape: { dtype: tensor.dtype, dims: tensor.dims, axes: tensor.axes },
    data: tensor.data
  };
}
