import type { GraphSpec, LevelSpec, RuntimeValue } from "../types";
import { causalMask, referenceBroadcastAdd } from "../runtime/reference";
import { seededTensor, type TinyTensor } from "../runtime/tinyTensor";

const visibleScores = seededTensor([1, 2, 4, 4], ["B", "H", "T", "T"], "mask-visible-scores");
const visibleReference = referenceBroadcastAdd(visibleScores, causalMask(4), ["T", "T"]);

const hiddenWideScores = seededTensor([2, 3, 5, 5], ["B", "H", "T", "T"], "mask-hidden-wide-scores");
const hiddenWideReference = referenceBroadcastAdd(hiddenWideScores, causalMask(5), ["T", "T"]);

const hiddenSmallScores = seededTensor([3, 1, 3, 3], ["B", "H", "T", "T"], "mask-hidden-small-scores");
const hiddenSmallReference = referenceBroadcastAdd(hiddenSmallScores, causalMask(3), ["T", "T"]);

export const ch0MaskGraph: LevelSpec = {
  id: "ch0_4f_mask_add",
  title: "0-4F Mini Mask Add",
  mode: "graph_challenge",
  chapter: "Chapter 0",
  goal: "Build scores[B,H,T,T] + causal_mask[T,T] -> masked_scores[B,H,T,T] with the correct query/key mask direction.",
  modulePalette: [
    "InputTensor",
    "ScoreBoard",
    "CausalMask",
    "AxisAlignmentRuler",
    "BroadcastRail",
    "GhostExpansionPreview",
    "SemanticWarningLens",
    "AddGate",
    "CellTrace",
    "ReferenceChecker"
  ],
  initialGraph: createCh0MaskWrongOrientationGraph(),
  targetGraph: createCh0MaskSolutionGraph(),
  constraints: {
    maxNodes: 11,
    maxEdges: 13,
    forbiddenModules: ["Softmax"]
  },
  visibleTests: [
    {
      id: "causal_mask_visible",
      title: "causal mask blocks future keys",
      visibility: "visible",
      inputSeed: "mask-visible",
      inputs: {
        scores: runtimeValue(visibleScores),
        reference: runtimeValue(visibleReference)
      },
      assertions: [
        { type: "shape", nodeId: "score_board", expectedAxes: ["B", "H", "T", "T"], expectedDims: [1, 2, 4, 4] },
        { type: "shape", nodeId: "mask", expectedAxes: ["T", "T"], expectedDims: [4, 4] },
        { type: "shape", nodeId: "broadcast", expectedAxes: ["B", "H", "T", "T"], expectedDims: [1, 2, 4, 4] },
        { type: "shape", nodeId: "masked_scores", expectedAxes: ["B", "H", "T", "T"], expectedDims: [1, 2, 4, 4] },
        { type: "shape", nodeId: "cell_trace", expectedAxes: ["B", "H", "T", "T"], expectedDims: [1, 2, 4, 4] },
        { type: "future_attention_zero", nodeId: "masked_scores", threshold: -999 },
        { type: "allclose", nodeId: "masked_scores", referenceNodeId: "reference", atol: 1e-5 }
      ]
    }
  ],
  hiddenTests: [
    {
      id: "mask_orientation_wide",
      title: "mask orientation with wider T",
      visibility: "hidden",
      inputSeed: "mask-hidden-wide",
      inputs: {
        scores: runtimeValue(hiddenWideScores),
        reference: runtimeValue(hiddenWideReference)
      },
      assertions: [
        { type: "shape", nodeId: "broadcast", expectedAxes: ["B", "H", "T", "T"], expectedDims: [2, 3, 5, 5] },
        { type: "shape", nodeId: "masked_scores", expectedAxes: ["B", "H", "T", "T"], expectedDims: [2, 3, 5, 5] },
        { type: "future_attention_zero", nodeId: "masked_scores", threshold: -999 },
        { type: "allclose", nodeId: "masked_scores", referenceNodeId: "reference", atol: 1e-5 }
      ]
    },
    {
      id: "mask_orientation_batch_mutation",
      title: "mask orientation with B/H mutation",
      visibility: "hidden",
      inputSeed: "mask-hidden-small",
      inputs: {
        scores: runtimeValue(hiddenSmallScores),
        reference: runtimeValue(hiddenSmallReference)
      },
      assertions: [
        { type: "shape", nodeId: "broadcast", expectedAxes: ["B", "H", "T", "T"], expectedDims: [3, 1, 3, 3] },
        { type: "shape", nodeId: "masked_scores", expectedAxes: ["B", "H", "T", "T"], expectedDims: [3, 1, 3, 3] },
        { type: "future_attention_zero", nodeId: "masked_scores", threshold: -999 },
        { type: "allclose", nodeId: "masked_scores", referenceNodeId: "reference", atol: 1e-5 }
      ]
    }
  ],
  onboarding: {
    story: "Mini Mask machine is wired and the shapes look right, but it blocks the wrong triangle.",
    startingProblem: "CausalMask is set to key_query, so future-key cells are not masked in query/key order.",
    firstAction: "Run Visible and inspect the future_attention_zero failure on masked_scores.",
    targetRecipe: [
      "scores.out -> score_board.scores",
      "score_board.out -> mask.target",
      "mask maskOrientation = query_key",
      "mask.out -> mask_ruler.x",
      "mask_ruler.out -> broadcast.small",
      "broadcast.out -> ghost.x",
      "score_board.out -> masked_scores.left",
      "ghost.out -> masked_scores.right"
    ],
    winCondition: "future key cells are <= -999 and masked_scores matches the reference."
  },
  debrief: {
    completeTitle: "Mini Mask Add Restored",
    fixedProblem: "The causal mask is generated in query/key order and then broadcast logically across B/H before AddGate.",
    learned: "A causal mask can have the right [T,T] shape while blocking the wrong triangle. Numeric reference and Cell Trace catch the query/key inversion.",
    nextUse: "Softmax and attention probabilities depend on this masked score board."
  }
};

export function createCh0MaskSolutionGraph(): GraphSpec {
  return {
    levelId: "ch0_4f_mask_add",
    version: 1,
    nodes: [
      { id: "scores", moduleId: "InputTensor", params: { inputKey: "scores" }, position: { x: 80, y: 120 } },
      { id: "score_board", moduleId: "ScoreBoard", params: { expectedAxes: ["B", "H", "T", "T"] }, position: { x: 320, y: 120 } },
      { id: "mask", moduleId: "CausalMask", params: { maskOrientation: "query_key", maskedValue: -10000 }, position: { x: 560, y: 300 } },
      { id: "mask_ruler", moduleId: "AxisAlignmentRuler", params: { expectedAxes: ["T", "T"] }, position: { x: 790, y: 300 } },
      { id: "broadcast", moduleId: "BroadcastRail", params: { alignAxes: ["T", "T"] }, position: { x: 790, y: 120 } },
      { id: "semantic_lens", moduleId: "SemanticWarningLens", params: {}, position: { x: 1020, y: 120 } },
      { id: "ghost", moduleId: "GhostExpansionPreview", params: {}, position: { x: 1250, y: 120 } },
      { id: "masked_scores", moduleId: "AddGate", params: {}, position: { x: 1250, y: 300 } },
      { id: "cell_trace", moduleId: "CellTrace", params: { b: 0, h: 0, i: 0, j: 1 }, position: { x: 1490, y: 300 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 1490, y: 500 } }
    ],
    edges: [
      { id: "e_scores_score_board", from: { nodeId: "scores", portId: "out" }, to: { nodeId: "score_board", portId: "scores" } },
      { id: "e_score_board_mask", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "mask", portId: "target" } },
      { id: "e_mask_ruler", from: { nodeId: "mask", portId: "out" }, to: { nodeId: "mask_ruler", portId: "x" } },
      { id: "e_mask_broadcast", from: { nodeId: "mask_ruler", portId: "out" }, to: { nodeId: "broadcast", portId: "small" } },
      { id: "e_score_board_broadcast", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "broadcast", portId: "target" } },
      { id: "e_broadcast_semantic_lens", from: { nodeId: "broadcast", portId: "out" }, to: { nodeId: "semantic_lens", portId: "x" } },
      { id: "e_semantic_lens_ghost", from: { nodeId: "semantic_lens", portId: "out" }, to: { nodeId: "ghost", portId: "x" } },
      { id: "e_score_board_add", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "masked_scores", portId: "left" } },
      { id: "e_ghost_add", from: { nodeId: "ghost", portId: "out" }, to: { nodeId: "masked_scores", portId: "right" } },
      { id: "e_masked_cell_trace", from: { nodeId: "masked_scores", portId: "out" }, to: { nodeId: "cell_trace", portId: "scores" } },
      { id: "e_score_board_cell_trace", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "cell_trace", portId: "left" } },
      { id: "e_mask_cell_trace", from: { nodeId: "mask", portId: "out" }, to: { nodeId: "cell_trace", portId: "small" } },
      { id: "e_masked_ref", from: { nodeId: "masked_scores", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["masked_scores", "cell_trace", "reference"]
  };
}

export function createCh0MaskWrongOrientationGraph(): GraphSpec {
  const graph = createCh0MaskSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (node.id === "mask" ? { ...node, params: { maskOrientation: "key_query", maskedValue: -10000 } } : node))
  };
}

export function createCh0MaskWithoutBroadcastGraph(): GraphSpec {
  const graph = createCh0MaskSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => !["mask_ruler", "broadcast", "semantic_lens", "ghost"].includes(node.id)),
    edges: [
      { id: "e_scores_score_board", from: { nodeId: "scores", portId: "out" }, to: { nodeId: "score_board", portId: "scores" } },
      { id: "e_score_board_mask", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "mask", portId: "target" } },
      { id: "e_score_board_add", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "masked_scores", portId: "left" } },
      { id: "e_mask_add", from: { nodeId: "mask", portId: "out" }, to: { nodeId: "masked_scores", portId: "right" } },
      { id: "e_masked_cell_trace", from: { nodeId: "masked_scores", portId: "out" }, to: { nodeId: "cell_trace", portId: "scores" } },
      { id: "e_score_board_cell_trace", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "cell_trace", portId: "left" } },
      { id: "e_mask_cell_trace", from: { nodeId: "mask", portId: "out" }, to: { nodeId: "cell_trace", portId: "small" } },
      { id: "e_masked_ref", from: { nodeId: "masked_scores", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
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
