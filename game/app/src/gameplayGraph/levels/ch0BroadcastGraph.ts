import type { GraphSpec, LevelSpec, RuntimeValue } from "../types";
import { referenceBroadcastAdd } from "../runtime/reference";
import { seededTensor, type TinyTensor } from "../runtime/tinyTensor";

const visibleProjected = seededTensor([2, 3, 4], ["B", "T", "O"], "broadcast-visible-projected");
const visibleBias = seededTensor([4], ["O"], "broadcast-visible-bias");
const visibleReference = referenceBroadcastAdd(visibleProjected, visibleBias, ["O"]);

const hiddenMutationProjected = seededTensor([4, 2, 5], ["B", "T", "O"], "broadcast-hidden-mutation-projected");
const hiddenMutationBias = seededTensor([5], ["O"], "broadcast-hidden-mutation-bias");
const hiddenMutationReference = referenceBroadcastAdd(hiddenMutationProjected, hiddenMutationBias, ["O"]);

const hiddenEqualProjected = seededTensor([2, 4, 4], ["B", "T", "O"], "broadcast-hidden-equal-projected");
const hiddenEqualBias = seededTensor([4], ["O"], "broadcast-hidden-equal-bias");
const hiddenEqualReference = referenceBroadcastAdd(hiddenEqualProjected, hiddenEqualBias, ["O"]);

const hiddenBatchProjected = seededTensor([3, 5, 2], ["B", "T", "O"], "broadcast-hidden-batch-projected");
const hiddenBatchBias = seededTensor([2], ["O"], "broadcast-hidden-batch-bias");
const hiddenBatchReference = referenceBroadcastAdd(hiddenBatchProjected, hiddenBatchBias, ["O"]);

export const ch0BroadcastGraph: LevelSpec = {
  id: "ch0_4_broadcast_add",
  title: "0-4 Broadcast Add Graph",
  mode: "graph_challenge",
  chapter: "Chapter 0",
  goal: "Build projected[B,T,O] + bias[O] -> biased[B,T,O] with semantic-axis broadcasting.",
  modulePalette: [
    "InputTensor",
    "AxisAlignmentRuler",
    "BroadcastRail",
    "GhostExpansionPreview",
    "SemanticWarningLens",
    "AddGate",
    "CellTrace",
    "ReferenceChecker"
  ],
  initialGraph: {
    levelId: "ch0_4_broadcast_add",
    version: 1,
    nodes: [],
    edges: [],
    outputNodes: ["biased", "cell_trace", "reference"]
  },
  constraints: {
    maxNodes: 10,
    maxEdges: 12,
    forbiddenModules: ["Softmax"]
  },
  visibleTests: [
    {
      id: "bias_add_visible",
      title: "bias broadcasts along B/T",
      visibility: "visible",
      inputSeed: "broadcast-visible",
      inputs: {
        projected: runtimeValue(visibleProjected),
        bias: runtimeValue(visibleBias),
        reference: runtimeValue(visibleReference)
      },
      assertions: [
        { type: "shape", nodeId: "axis_ruler", expectedAxes: ["O"], expectedDims: [4] },
        { type: "shape", nodeId: "broadcast", expectedAxes: ["B", "T", "O"], expectedDims: [2, 3, 4] },
        { type: "shape", nodeId: "ghost", expectedAxes: ["B", "T", "O"], expectedDims: [2, 3, 4] },
        { type: "shape", nodeId: "biased", expectedAxes: ["B", "T", "O"], expectedDims: [2, 3, 4] },
        { type: "shape", nodeId: "cell_trace", expectedAxes: ["B", "T", "O"], expectedDims: [2, 3, 4] },
        { type: "allclose", nodeId: "biased", referenceNodeId: "reference", atol: 1e-5 }
      ]
    }
  ],
  hiddenTests: [
    {
      id: "bt_o_mutation",
      title: "B/T/O mutation",
      visibility: "hidden",
      inputSeed: "broadcast-hidden-mutation",
      inputs: {
        projected: runtimeValue(hiddenMutationProjected),
        bias: runtimeValue(hiddenMutationBias),
        reference: runtimeValue(hiddenMutationReference)
      },
      assertions: [
        { type: "shape", nodeId: "broadcast", expectedAxes: ["B", "T", "O"], expectedDims: [4, 2, 5] },
        { type: "shape", nodeId: "biased", expectedAxes: ["B", "T", "O"], expectedDims: [4, 2, 5] },
        { type: "allclose", nodeId: "biased", referenceNodeId: "reference", atol: 1e-5 }
      ]
    },
    {
      id: "t_equals_o_semantic_trap",
      title: "T == O semantic trap",
      visibility: "hidden",
      inputSeed: "broadcast-hidden-equal",
      inputs: {
        projected: runtimeValue(hiddenEqualProjected),
        bias: runtimeValue(hiddenEqualBias),
        reference: runtimeValue(hiddenEqualReference)
      },
      assertions: [
        { type: "shape", nodeId: "broadcast", expectedAxes: ["B", "T", "O"], expectedDims: [2, 4, 4] },
        { type: "shape", nodeId: "biased", expectedAxes: ["B", "T", "O"], expectedDims: [2, 4, 4] },
        { type: "allclose", nodeId: "biased", referenceNodeId: "reference", atol: 1e-5 }
      ]
    },
    {
      id: "batch_and_token_mutation",
      title: "B/T mutation with small O",
      visibility: "hidden",
      inputSeed: "broadcast-hidden-batch",
      inputs: {
        projected: runtimeValue(hiddenBatchProjected),
        bias: runtimeValue(hiddenBatchBias),
        reference: runtimeValue(hiddenBatchReference)
      },
      assertions: [
        { type: "shape", nodeId: "broadcast", expectedAxes: ["B", "T", "O"], expectedDims: [3, 5, 2] },
        { type: "shape", nodeId: "biased", expectedAxes: ["B", "T", "O"], expectedDims: [3, 5, 2] },
        { type: "allclose", nodeId: "biased", referenceNodeId: "reference", atol: 1e-5 }
      ]
    }
  ],
  debrief: {
    completeTitle: "Broadcast Add Restored",
    fixedProblem: "bias[O] is aligned to the O rail, then logically expanded across B and T before AddGate.",
    learned: "Broadcast is not just matching lengths. The small tensor must attach to the correct semantic axis, or T==O cases can hide a wrong solution until numeric reference checks run.",
    nextUse: "Position embeddings and attention masks use the same semantic-axis broadcast rule."
  }
};

export function createCh0BroadcastSolutionGraph(): GraphSpec {
  return {
    levelId: "ch0_4_broadcast_add",
    version: 1,
    nodes: [
      { id: "projected", moduleId: "InputTensor", params: { inputKey: "projected" }, position: { x: 80, y: 110 } },
      { id: "bias", moduleId: "InputTensor", params: { inputKey: "bias", shape: [4], axes: ["O"] }, position: { x: 80, y: 310 } },
      { id: "axis_ruler", moduleId: "AxisAlignmentRuler", params: { expectedAxes: ["O"] }, position: { x: 320, y: 310 } },
      { id: "broadcast", moduleId: "BroadcastRail", params: { alignAxes: ["O"] }, position: { x: 560, y: 250 } },
      { id: "semantic_lens", moduleId: "SemanticWarningLens", params: {}, position: { x: 790, y: 250 } },
      { id: "ghost", moduleId: "GhostExpansionPreview", params: {}, position: { x: 1020, y: 250 } },
      { id: "biased", moduleId: "AddGate", params: {}, position: { x: 1020, y: 90 } },
      { id: "cell_trace", moduleId: "CellTrace", params: { b: 0, t: 0, o: 1 }, position: { x: 1260, y: 90 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 1260, y: 310 } }
    ],
    edges: [
      { id: "e_projected_broadcast_target", from: { nodeId: "projected", portId: "out" }, to: { nodeId: "broadcast", portId: "target" } },
      { id: "e_bias_axis_ruler", from: { nodeId: "bias", portId: "out" }, to: { nodeId: "axis_ruler", portId: "x" } },
      { id: "e_axis_ruler_broadcast", from: { nodeId: "axis_ruler", portId: "out" }, to: { nodeId: "broadcast", portId: "small" } },
      { id: "e_broadcast_semantic_lens", from: { nodeId: "broadcast", portId: "out" }, to: { nodeId: "semantic_lens", portId: "x" } },
      { id: "e_semantic_lens_ghost", from: { nodeId: "semantic_lens", portId: "out" }, to: { nodeId: "ghost", portId: "x" } },
      { id: "e_projected_add", from: { nodeId: "projected", portId: "out" }, to: { nodeId: "biased", portId: "left" } },
      { id: "e_ghost_add", from: { nodeId: "ghost", portId: "out" }, to: { nodeId: "biased", portId: "right" } },
      { id: "e_biased_cell_trace", from: { nodeId: "biased", portId: "out" }, to: { nodeId: "cell_trace", portId: "scores" } },
      { id: "e_projected_cell_trace", from: { nodeId: "projected", portId: "out" }, to: { nodeId: "cell_trace", portId: "left" } },
      { id: "e_bias_cell_trace", from: { nodeId: "bias", portId: "out" }, to: { nodeId: "cell_trace", portId: "small" } },
      { id: "e_biased_ref", from: { nodeId: "biased", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["biased", "cell_trace", "reference"]
  };
}

export function createCh0BroadcastWrongAxisGraph(): GraphSpec {
  const graph = createCh0BroadcastSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (node.id === "broadcast" ? { ...node, params: { alignAxes: ["T"] } } : node))
  };
}

export function createCh0BroadcastWithoutRailGraph(): GraphSpec {
  const graph = createCh0BroadcastSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => !["axis_ruler", "broadcast", "semantic_lens", "ghost"].includes(node.id)),
    edges: [
      { id: "e_projected_add", from: { nodeId: "projected", portId: "out" }, to: { nodeId: "biased", portId: "left" } },
      { id: "e_bias_add", from: { nodeId: "bias", portId: "out" }, to: { nodeId: "biased", portId: "right" } },
      { id: "e_biased_cell_trace", from: { nodeId: "biased", portId: "out" }, to: { nodeId: "cell_trace", portId: "scores" } },
      { id: "e_projected_cell_trace", from: { nodeId: "projected", portId: "out" }, to: { nodeId: "cell_trace", portId: "left" } },
      { id: "e_bias_cell_trace", from: { nodeId: "bias", portId: "out" }, to: { nodeId: "cell_trace", portId: "small" } },
      { id: "e_biased_ref", from: { nodeId: "biased", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ]
  };
}

function runtimeValue(tensor: TinyTensor): RuntimeValue {
  return {
    dtype: tensor.dtype,
    shape: { dtype: tensor.dtype, dims: tensor.dims, axes: tensor.axes },
    data: tensor.data
  };
}
