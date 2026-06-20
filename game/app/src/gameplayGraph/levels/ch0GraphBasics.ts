import type { GraphSpec, LevelSpec, RuntimeValue } from "../types";
import { seededTensor, type TinyTensor } from "../runtime/tinyTensor";

const visibleTensor = seededTensor([1, 2, 3], ["B", "T", "C"], "graph-basics-visible");
const hiddenTensor = seededTensor([2, 1, 4], ["B", "T", "C"], "graph-basics-hidden");

export const ch0GraphBasics: LevelSpec = {
  id: "ch0_0_graph_basics",
  title: "0-G Graph Basics",
  mode: "graph_challenge",
  chapter: "Chapter 0",
  goal: "Repair one missing cable: input.out must flow into shape_gate.x.",
  modulePalette: ["InputTensor", "OutputContractGate"],
  initialGraph: createCh0GraphBasicsBrokenGraph(),
  targetGraph: createCh0GraphBasicsSolutionGraph(),
  constraints: {
    maxNodes: 3,
    maxEdges: 2
  },
  onboarding: {
    story: "This tiny machine is almost ready. One cable is unplugged.",
    startingProblem: "shape_gate needs an input tensor, but its x port is empty.",
    firstAction: "Click Run Visible first, then select the red node in Trace.",
    targetRecipe: ["input.out -> shape_gate.x"],
    winCondition: "shape_gate outputs a tensor with axes [B,T,C].",
    allowedMistakes: ["Dragging the canvas", "Clicking an output port before choosing an input port", "Resetting and trying again"]
  },
  visibleTests: [
    {
      id: "connect_basic_visible",
      title: "connect one missing cable",
      visibility: "visible",
      inputSeed: "graph-basics-visible",
      inputs: { input: runtimeValue(visibleTensor) },
      assertions: [{ type: "shape", nodeId: "shape_gate", expectedAxes: ["B", "T", "C"], expectedDims: [1, 2, 3] }]
    }
  ],
  hiddenTests: [
    {
      id: "connect_basic_hidden",
      title: "connection generalizes to another shape",
      visibility: "hidden",
      inputSeed: "graph-basics-hidden",
      inputs: { input: runtimeValue(hiddenTensor) },
      assertions: [{ type: "shape", nodeId: "shape_gate", expectedAxes: ["B", "T", "C"], expectedDims: [2, 1, 4] }]
    }
  ],
  debrief: {
    completeTitle: "Graph Basics Restored",
    fixedProblem: "The input tensor now reaches the output contract gate.",
    learned: "Nodes are machines, ports are sockets, and edges move typed values between them.",
    nextUse: "Next repairs add small modules between already connected machine parts."
  }
};

export function createCh0GraphBasicsBrokenGraph(): GraphSpec {
  return {
    levelId: "ch0_0_graph_basics",
    version: 1,
    nodes: [
      { id: "input", moduleId: "InputTensor", params: { inputKey: "input" }, position: { x: 120, y: 180 } },
      { id: "shape_gate", moduleId: "OutputContractGate", params: { expectedAxes: ["B", "T", "C"] }, position: { x: 430, y: 180 } }
    ],
    edges: [],
    outputNodes: ["shape_gate"]
  };
}

export function createCh0GraphBasicsSolutionGraph(): GraphSpec {
  return {
    ...createCh0GraphBasicsBrokenGraph(),
    edges: [{ id: "e_input_shape_gate", from: { nodeId: "input", portId: "out" }, to: { nodeId: "shape_gate", portId: "x" } }]
  };
}

function runtimeValue(tensor: TinyTensor): RuntimeValue {
  return {
    dtype: tensor.dtype,
    shape: { dtype: tensor.dtype, dims: tensor.dims, axes: tensor.axes },
    data: tensor.data
  };
}
