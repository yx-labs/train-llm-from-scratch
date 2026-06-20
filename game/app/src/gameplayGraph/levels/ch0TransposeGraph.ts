import type { GraphSpec, LevelSpec, RuntimeValue } from "../types";
import { referenceQKScores } from "../runtime/reference";
import { seededTensor, type TinyTensor } from "../runtime/tinyTensor";

const visibleQ = seededTensor([2, 2, 4, 3], ["B", "H", "T", "D"], "qk-visible-q");
const visibleK = seededTensor([2, 2, 4, 3], ["B", "H", "T", "D"], "qk-visible-k");
const visibleReference = referenceQKScores(visibleQ, visibleK);

const hiddenEqualQ = seededTensor([1, 3, 4, 4], ["B", "H", "T", "D"], "qk-hidden-equal-q");
const hiddenEqualK = seededTensor([1, 3, 4, 4], ["B", "H", "T", "D"], "qk-hidden-equal-k");
const hiddenEqualReference = referenceQKScores(hiddenEqualQ, hiddenEqualK);

const hiddenCarryQ = seededTensor([3, 2, 5, 4], ["B", "H", "T", "D"], "qk-hidden-carry-q");
const hiddenCarryK = seededTensor([3, 2, 5, 4], ["B", "H", "T", "D"], "qk-hidden-carry-k");
const hiddenCarryReference = referenceQKScores(hiddenCarryQ, hiddenCarryK);

const hiddenSingleHeadQ = seededTensor([2, 1, 3, 5], ["B", "H", "T", "D"], "qk-hidden-single-head-q");
const hiddenSingleHeadK = seededTensor([2, 1, 3, 5], ["B", "H", "T", "D"], "qk-hidden-single-head-k");
const hiddenSingleHeadReference = referenceQKScores(hiddenSingleHeadQ, hiddenSingleHeadK);

export const ch0TransposeGraph: LevelSpec = {
  id: "ch0_3_transpose_graph",
  title: "0-3 Transpose Trap Graph",
  mode: "graph_challenge",
  chapter: "Chapter 0",
  goal: "Build scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:]) by wiring Q[B,H,T,D] @ K.transpose(-2,-1)[B,H,D,T].",
  modulePalette: ["InputTensor", "TransposeSwitch", "AxisLock", "MatMulGate", "ScoreBoard", "CellTrace", "ReferenceChecker"],
  initialGraph: createCh0TransposeNoKTransposeGraph(),
  targetGraph: createCh0TransposeSolutionGraph(),
  constraints: {
    maxNodes: 9,
    maxEdges: 10,
    forbiddenModules: ["AddGate", "Softmax"]
  },
  visibleTests: [
    {
      id: "multi_head_qk",
      title: "multi-head QK transpose",
      visibility: "visible",
      inputSeed: "qk-visible",
      inputs: {
        q: runtimeValue(visibleQ),
        k: runtimeValue(visibleK),
        reference: runtimeValue(visibleReference)
      },
      assertions: [
        { type: "shape", nodeId: "score_board", expectedAxes: ["B", "H", "T", "T"], expectedDims: [2, 2, 4, 4] },
        { type: "shape", nodeId: "cell_trace", expectedAxes: ["B", "H", "T", "T"], expectedDims: [2, 2, 4, 4] },
        { type: "allclose", nodeId: "score_board", referenceNodeId: "reference", atol: 1e-5 }
      ]
    }
  ],
  hiddenTests: [
    {
      id: "t_equals_d_numeric_trap",
      title: "T == D numeric trap",
      visibility: "hidden",
      inputSeed: "qk-hidden-equal",
      inputs: {
        q: runtimeValue(hiddenEqualQ),
        k: runtimeValue(hiddenEqualK),
        reference: runtimeValue(hiddenEqualReference)
      },
      assertions: [
        { type: "shape", nodeId: "score_board", expectedAxes: ["B", "H", "T", "T"], expectedDims: [1, 3, 4, 4] },
        { type: "shape", nodeId: "cell_trace", expectedAxes: ["B", "H", "T", "T"], expectedDims: [1, 3, 4, 4] },
        { type: "allclose", nodeId: "score_board", referenceNodeId: "reference", atol: 1e-5 }
      ]
    },
    {
      id: "carry_axes_preserved",
      title: "B/H carrier axes preserved",
      visibility: "hidden",
      inputSeed: "qk-hidden-carry",
      inputs: {
        q: runtimeValue(hiddenCarryQ),
        k: runtimeValue(hiddenCarryK),
        reference: runtimeValue(hiddenCarryReference)
      },
      assertions: [
        { type: "shape", nodeId: "score_board", expectedAxes: ["B", "H", "T", "T"], expectedDims: [3, 2, 5, 5] },
        { type: "shape", nodeId: "cell_trace", expectedAxes: ["B", "H", "T", "T"], expectedDims: [3, 2, 5, 5] },
        { type: "allclose", nodeId: "score_board", referenceNodeId: "reference", atol: 1e-5 }
      ]
    },
    {
      id: "single_head_d_mutation",
      title: "single-head D mutation",
      visibility: "hidden",
      inputSeed: "qk-hidden-single-head",
      inputs: {
        q: runtimeValue(hiddenSingleHeadQ),
        k: runtimeValue(hiddenSingleHeadK),
        reference: runtimeValue(hiddenSingleHeadReference)
      },
      assertions: [
        { type: "shape", nodeId: "score_board", expectedAxes: ["B", "H", "T", "T"], expectedDims: [2, 1, 3, 3] },
        { type: "shape", nodeId: "cell_trace", expectedAxes: ["B", "H", "T", "T"], expectedDims: [2, 1, 3, 3] },
        { type: "allclose", nodeId: "score_board", referenceNodeId: "reference", atol: 1e-5 }
      ]
    }
  ],
  onboarding: {
    story: "QK score machine is almost complete. K is connected directly, so the token axis and feature axis are in the wrong slots.",
    startingProblem: "MatMul needs K as [B,H,D,T], but the current K input is [B,H,T,D].",
    firstAction: "Run Visible, then inspect qk_matmul and the score board.",
    targetRecipe: [
      "q.out -> qk_matmul.left",
      "k.out -> k_transpose.x",
      "k_transpose.out -> axis_lock.x",
      "axis_lock.out -> qk_matmul.right",
      "qk_matmul.out -> score_board.scores",
      "score_board.out -> cell_trace.scores",
      "score_board.out -> reference.x"
    ],
    winCondition: "score_board and cell_trace output [B,H,T,T] and match the reference."
  },
  debrief: {
    completeTitle: "Transpose Trap Restored",
    fixedProblem: "K now swaps only its last two axes before QK MatMul, while B/H carrier axes stay locked.",
    learned: "QK scores compare query tokens to key tokens: scores[b,h,i,j] = dot(Q[b,h,i,:], K[b,h,j,:]). Cell Trace turns a numeric mismatch into a concrete cell-level proof.",
    nextUse: "Mask Add and Softmax depend on the score board's query/key axis orientation."
  }
};

export function createCh0TransposeSolutionGraph(): GraphSpec {
  return {
    levelId: "ch0_3_transpose_graph",
    version: 1,
    nodes: [
      { id: "q", moduleId: "InputTensor", params: { inputKey: "q" }, position: { x: 80, y: 110 } },
      { id: "k", moduleId: "InputTensor", params: { inputKey: "k" }, position: { x: 80, y: 290 } },
      { id: "k_transpose", moduleId: "TransposeSwitch", params: { axisA: -2, axisB: -1 }, position: { x: 330, y: 290 } },
      { id: "axis_lock", moduleId: "AxisLock", params: { expectedPrefixAxes: ["B", "H"] }, position: { x: 560, y: 290 } },
      { id: "qk_matmul", moduleId: "MatMulGate", params: {}, position: { x: 560, y: 140 } },
      { id: "score_board", moduleId: "ScoreBoard", params: { expectedAxes: ["B", "H", "T", "T"] }, position: { x: 810, y: 140 } },
      { id: "cell_trace", moduleId: "CellTrace", params: { b: 0, h: 0, i: 0, j: 1 }, position: { x: 1060, y: 130 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 1060, y: 340 } }
    ],
    edges: [
      { id: "e_q_matmul", from: { nodeId: "q", portId: "out" }, to: { nodeId: "qk_matmul", portId: "left" } },
      { id: "e_k_transpose", from: { nodeId: "k", portId: "out" }, to: { nodeId: "k_transpose", portId: "x" } },
      { id: "e_transpose_axis_lock", from: { nodeId: "k_transpose", portId: "out" }, to: { nodeId: "axis_lock", portId: "x" } },
      { id: "e_axis_lock_matmul", from: { nodeId: "axis_lock", portId: "out" }, to: { nodeId: "qk_matmul", portId: "right" } },
      { id: "e_matmul_score_board", from: { nodeId: "qk_matmul", portId: "out" }, to: { nodeId: "score_board", portId: "scores" } },
      { id: "e_score_board_cell_trace", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "cell_trace", portId: "scores" } },
      { id: "e_q_cell_trace", from: { nodeId: "q", portId: "out" }, to: { nodeId: "cell_trace", portId: "q" } },
      { id: "e_k_cell_trace", from: { nodeId: "k", portId: "out" }, to: { nodeId: "cell_trace", portId: "k" } },
      { id: "e_score_board_ref", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["score_board", "cell_trace", "reference"]
  };
}

export function createCh0TransposeNoKTransposeGraph(): GraphSpec {
  const graph = createCh0TransposeSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => node.id !== "k_transpose" && node.id !== "axis_lock"),
    edges: [
      { id: "e_q_matmul", from: { nodeId: "q", portId: "out" }, to: { nodeId: "qk_matmul", portId: "left" } },
      { id: "e_k_matmul", from: { nodeId: "k", portId: "out" }, to: { nodeId: "qk_matmul", portId: "right" } },
      { id: "e_matmul_score_board", from: { nodeId: "qk_matmul", portId: "out" }, to: { nodeId: "score_board", portId: "scores" } },
      { id: "e_score_board_cell_trace", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "cell_trace", portId: "scores" } },
      { id: "e_q_cell_trace", from: { nodeId: "q", portId: "out" }, to: { nodeId: "cell_trace", portId: "q" } },
      { id: "e_k_cell_trace", from: { nodeId: "k", portId: "out" }, to: { nodeId: "cell_trace", portId: "k" } },
      { id: "e_score_board_ref", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ]
  };
}

export function createCh0TransposeWrongOperandOrderGraph(): GraphSpec {
  const graph = createCh0TransposeSolutionGraph();
  return {
    ...graph,
    edges: graph.edges.map((edge) => {
      if (edge.id === "e_q_matmul") {
        return { ...edge, to: { nodeId: "qk_matmul", portId: "right" } };
      }
      if (edge.id === "e_axis_lock_matmul") {
        return { ...edge, to: { nodeId: "qk_matmul", portId: "left" } };
      }
      return edge;
    })
  };
}

function runtimeValue(tensor: TinyTensor): RuntimeValue {
  return {
    dtype: tensor.dtype,
    shape: { dtype: tensor.dtype, dims: tensor.dims, axes: tensor.axes },
    data: tensor.data
  };
}
