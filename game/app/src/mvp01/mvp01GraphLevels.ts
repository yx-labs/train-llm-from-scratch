import type { AxisName, CertificationControlValue, CertificationTestSpec, GraphSpec, LevelCertificationSpec, LevelSpec, RuntimeValue, TestAssertion, TestCase } from "../gameplayGraph/types";
import { referenceQKScores } from "../gameplayGraph/runtime/reference";
import { addTensors, broadcastTo, makeTensor, matmul, seededTensor, type TinyTensor } from "../gameplayGraph/runtime/tinyTensor";
import { mvp01CourseInputCount, mvp01CourseLevels, type Mvp01CourseLevelDef } from "./mvp01CourseCatalog";

export type Mvp01ComponentFlowSpec = {
  levelId: string;
  componentId: string;
  title: string;
  version: number;
  exportModuleId: string;
  requires: string[];
  unlocks: string[];
  shelf: string;
};

const playableCourseCodes = new Set(["0-0", "0-1", "0-3", "0-4", "0-5", "0-6", "0-7", "1-1", "3-2", "4-3"]);

const componentRequirementOverrides: Record<string, string[]> = {
  mvp01_2_vector_rail: ["component.scalar_cell.v1"],
  mvp01_ch0_04_dot_product: ["component.vector_rail.v1"],
  mvp01_3_matrix_struct: ["component.dot_product.v1"],
  mvp01_5_matmul_gate: ["component.matrix_struct.v1"],
  mvp01_4_tensor_box: ["component.matmul_gate.v1"],
  mvp01_ch1_01_splitter: ["component.tensor_box.v1"],
  mvp01_6_linear: ["component.matmul_gate.v1"],
  mvp01_ch4_03_qk_score: ["component.linear.v1"]
};

export const mvp01ComponentFlowSpecs: Mvp01ComponentFlowSpec[] = mvp01CourseLevels
  .filter((level) => level.buildable && playableCourseCodes.has(level.code))
  .map((level) => ({
    levelId: level.id,
    componentId: level.componentId,
    title: `${level.component} v1`,
    version: 1,
    exportModuleId: level.componentId,
    requires: componentRequirementOverrides[level.id] ?? level.requires,
    unlocks: level.unlocks,
    shelf: level.shelf
  }));

const scalarVisible = scalar(0.5);
const scalarHidden = scalar(0.5);
const wirePacketVisible = makeTensor("float32", [1, 2], ["B", "T"], [0.25, -0.75]);
const wirePacketHidden = makeTensor("float32", [1, 2], ["B", "T"], [1.25, 0.5]);

const vectorValues = [0.5, -1, 2];
const vectorVisible = vector(vectorValues);
const vectorHidden = vector(vectorValues);

const dotQueryVisible = vector([0.2, -0.5, 1.0]);
const dotKeyVisible = vector([0.4, 0.1, -0.3]);
const dotVisible = dotProduct(dotQueryVisible, dotKeyVisible);
const dotQueryHidden = vector([0.4, -0.2, 1.1, 0.75, -0.6]);
const dotKeyHidden = vector([-0.5, 0.25, 0.9, -1.2, 0.3]);
const dotHidden = dotProduct(dotQueryHidden, dotKeyHidden);

const matrixCol0 = vector([0.4, 1.1, -0.7]);
const matrixCol1 = vector([-0.2, 0.3, 0.8]);
const matrixVisible = matrixFromColumns(matrixCol0, matrixCol1);
const matrixHiddenCol0 = vector([1, -2, 0.5, 3]);
const matrixHiddenCol1 = vector([-1.5, 0.25, 2.5, -0.75]);
const matrixHidden = matrixFromColumns(matrixHiddenCol0, matrixHiddenCol1);

const token0 = vector([1, 0, 2]);
const token1 = vector([-1, 3, 0.5]);
const tensorVisible = tensorFromTokens(token0, token1);
const hiddenToken0 = vector([0.25, -0.5, 1, 2]);
const hiddenToken1 = vector([1.5, 0, -1, 0.75]);
const tensorHidden = tensorFromTokens(hiddenToken0, hiddenToken1);

const matmulVisible = matmul(tensorVisible, matrixVisible);
const matmulHiddenInput = seededTensor([2, 3, 4], ["B", "T", "C"], "mvp01-matmul-hidden-input");
const matmulHiddenWeight = seededTensor([4, 3], ["C", "O"], "mvp01-matmul-hidden-weight");
const matmulHidden = matmul(matmulHiddenInput, matmulHiddenWeight);

const linearBiasVisible = makeTensor("float32", [2], ["O"], [0.2, -0.1]);
const linearVisible = linear(tensorVisible, matrixVisible, linearBiasVisible);
const linearHiddenInput = seededTensor([2, 3, 4], ["B", "T", "C"], "mvp01-linear-hidden-input");
const linearHiddenWeight = seededTensor([4, 3], ["C", "O"], "mvp01-linear-hidden-weight");
const linearHiddenBias = seededTensor([3], ["O"], "mvp01-linear-hidden-bias");
const linearHidden = linear(linearHiddenInput, linearHiddenWeight, linearHiddenBias);

const splitterVisibleText = "tokenizers are useful!";
const splitterVisiblePieces = ["tokenizers", "are", "useful", "!"];
const splitterHiddenText = "shape, token buffer";
const splitterHiddenPieces = ["shape", ",", "token", "buffer"];

const qkVisibleQ = seededTensor([1, 1, 3, 2], ["B", "H", "T", "D"], "mvp01-qk-visible-q");
const qkVisibleK = seededTensor([1, 1, 3, 2], ["B", "H", "T", "D"], "mvp01-qk-visible-k");
const qkVisibleReference = referenceQKScores(qkVisibleQ, qkVisibleK);
const qkHiddenEqualQ = seededTensor([1, 1, 3, 3], ["B", "H", "T", "D"], "mvp01-qk-hidden-equal-q");
const qkHiddenEqualK = seededTensor([1, 1, 3, 3], ["B", "H", "T", "D"], "mvp01-qk-hidden-equal-k");
const qkHiddenEqualReference = referenceQKScores(qkHiddenEqualQ, qkHiddenEqualK);
const qkHiddenCarryQ = seededTensor([2, 2, 4, 3], ["B", "H", "T", "D"], "mvp01-qk-hidden-carry-q");
const qkHiddenCarryK = seededTensor([2, 2, 4, 3], ["B", "H", "T", "D"], "mvp01-qk-hidden-carry-k");
const qkHiddenCarryReference = referenceQKScores(qkHiddenCarryQ, qkHiddenCarryK);

const manualCourseLevelFactories: Record<string, () => LevelSpec> = {
  "0-0": wireAndProbeTutorialLevel,
  "0-1": scalarCellLevel,
  "0-3": vectorRailLevel,
  "0-4": dotProductLevel,
  "0-5": matrixStructLevel,
  "0-6": matMulGateLevel,
  "0-7": tensorBoxLevel,
  "1-1": splitterLevel,
  "3-2": linearBuildLevel,
  "4-3": qkScoreLevel
};

export const mvp01GraphLevels: LevelSpec[] = mvp01CourseLevels.map((courseLevel) => {
  const playable = playableCourseCodes.has(courseLevel.code);
  const level = manualCourseLevelFactories[courseLevel.code]?.() ?? courseCatalogLevel(courseLevel);
  return playable ? { ...level, routeStatus: "playable" } : { ...level, routeStatus: "roadmap", modulePalette: [], certification: undefined };
});

function wireAndProbeTutorialLevel(): LevelSpec {
  return {
    id: "mvp01_0_wire_probe",
    title: "MVP0.1-0 Wire & Probe",
    mode: "graph_challenge",
    chapter: "MVP0.1 Component Arc",
    goal: "Learn port direction, data flow, and why prebuilt probe nodes exist before building components.",
    modulePalette: [],
    initialGraph: createMvp01WireProbeInitialGraph(),
    targetGraph: createMvp01WireProbeSolutionGraph(),
    constraints: { maxNodes: 3, maxEdges: 2 },
    visibleTests: [
      shapeAndReferenceCase(
        "wire_probe_visible",
        "wire packet through contract into probe",
        {
          case: textBatch([
            "Concept: A graph wire carries a shaped value from an output port to an input port.",
            "Task: Wire packet_input.out to port_gate.x, then wire port_gate.out to reference.x.",
            "Check: port_gate only passes the packet when its port contract is connected; reference is a prebuilt probe that supplies the expected packet for tests."
          ]),
          packet: runtimeValue(wirePacketVisible),
          reference: runtimeValue(wirePacketVisible)
        },
        "port_gate",
        ["B", "T"],
        [1, 2]
      )
    ],
    hiddenTests: [
      shapeAndReferenceCase(
        "wire_probe_hidden",
        "probe connection remains part of the test circuit",
        {
          packet: runtimeValue(wirePacketHidden),
          reference: runtimeValue(wirePacketHidden)
        },
        "port_gate",
        ["B", "T"],
        [1, 2]
      )
    ],
    onboarding: {
      story: "Before building components, first learn how this graph workbench tests a value.",
      startingProblem: "The packet input, port contract gate, and reference probe are already on the canvas, but no wires move data between their ports yet.",
      firstAction: "Connect packet_input.out to port_gate.x, then connect port_gate.out to reference.x.",
      targetRecipe: ["packet_input.out -> port_gate.x", "port_gate.out -> reference.x"],
      winCondition: "The packet reaches the port gate, then the prebuilt reference probe supplies the expected packet for the test comparison."
    },
    caseStudy: {
      title: "Wire the test circuit",
      narrative:
        "This warm-up is only about Graph OS wiring. The packet already exists; you are not creating a reusable data component here. Data leaves an output port and enters an input port. The blue contract and probe nodes are prebuilt test equipment: port_gate checks that the packet reaches the contract, while reference provides the expected packet for tests. They are not your reusable component.",
      visibleInputFocus: "Task: Wire packet_input.out to port_gate.x, then wire port_gate.out to reference.x.",
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: "Task: Wire packet_input.out to port_gate.x, then wire port_gate.out to reference.x." },
        { type: "tensor_preview", title: "Incoming packet", inputKey: "packet" },
        { type: "tensor_preview", title: "Expected probe value", inputKey: "reference" }
      ],
      playerQuestion: "Can the packet travel through the port gate and into the reference probe?",
      successObservation: "Both wires exist, port_gate receives float32[B,T], and the reference probe can provide the expected packet for comparison."
    },
    debrief: {
      completeTitle: "Wire and probe circuit ready",
      fixedProblem: "The test circuit now has a real data path from source to contract to probe.",
      learned: "Probe nodes are prebuilt testing equipment. They help certify your graph, but they are not your reusable component.",
      nextUse: "Next you will build the first reusable value component: ScalarCell."
    }
  };
}

function scalarCellLevel(): LevelSpec {
  return {
    id: "mvp01_1_scalar_cell",
    title: "MVP0.1-1 Scalar Cell",
    mode: "graph_challenge",
    chapter: "MVP0.1 Component Arc",
    goal: "Build ScalarCell v1 from a raw finite float32 literal, then certify its rank-0 output contract.",
    modulePalette: ["Float32Literal", "OutputContractGate", "ReferenceChecker"],
    initialGraph: createMvp01ScalarInitialGraph(),
    targetGraph: createMvp01ScalarSolutionGraph(),
    constraints: { maxNodes: 3, maxEdges: 2 },
    visibleTests: [
      shapeContractCase(
        "scalar_visible",
        "scalar value and rank-0 contract",
        {
          case: textBatch([
            "Concept: A Scalar is one single value with no row, column, or extra axis.",
            "Task: Create a valid float32 scalar. The starting example is 0.5, but any finite number is valid.",
            "Check: It must be finite and its output shape must be rank-0: []."
          ]),
          reference: runtimeValue(scalarVisible)
        },
        "scalar_out",
        [],
        []
      )
    ],
    hiddenTests: [
      shapeContractCase("scalar_hidden", "scalar stays finite under certification", { reference: runtimeValue(scalarHidden) }, "scalar_out", [], [])
    ],
    certification: scalarCertification(),
    onboarding: {
      story: "The component forge starts by wrapping a raw number into the first reusable component.",
      startingProblem: "The ScalarCell blueprint has an output contract, but no internal primitive produces the value yet.",
      firstAction: "Add Float32Literal, connect scalar_source.out to scalar_out.x, then connect scalar_out.out to reference.x.",
      targetRecipe: ["Float32Literal.out -> scalar_out.x", "scalar_out.out -> reference.x"],
      winCondition: "scalar_out is a finite float32[] value."
    },
    caseStudy: {
      title: "Create a valid Scalar",
      narrative: "A Scalar is the smallest data unit: one value, no axes, no length. In this level, make the literal into a valid float32[] scalar; 0.5 is only the starting example.",
      visibleInputFocus: "Task: Create a valid float32 scalar. The starting example is 0.5, but any finite number is valid.",
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: "Task: Create a valid float32 scalar. The starting example is 0.5, but any finite number is valid." },
        { type: "tensor_preview", title: "Starting example", inputKey: "reference" }
      ],
      playerQuestion: "Is this value a finite float32 scalar?",
      successObservation: "The output is float32[] and the value is finite, so the Scalar is valid."
    },
    debrief: {
      completeTitle: "ScalarCell certified",
      fixedProblem: "The ScalarCell blueprint now wraps a finite float32 literal behind a clean output contract.",
      learned: "Constructing a component is different from using it: ScalarCell becomes available only after the current task check and certification pass.",
      nextUse: "VectorRail will combine several certified scalar cells into a C axis."
    }
  };
}

function vectorRailLevel(): LevelSpec {
  return {
    id: "mvp01_2_vector_rail",
    title: "MVP0.1-2 Vector Rail",
    mode: "graph_challenge",
    chapter: "MVP0.1 Component Arc",
    goal: "Combine three ScalarCell values into one feature vector with semantic axis C.",
    modulePalette: ["component.scalar_cell.v1", "VectorRail", "OutputContractGate", "ReferenceChecker"],
    initialGraph: createMvp01VectorInitialGraph(),
    targetGraph: createMvp01VectorSolutionGraph(),
    constraints: { maxNodes: 6, maxEdges: 5 },
    visibleTests: [
      shapeAndReferenceCase(
        "vector_visible",
        "three scalars become vector[C]",
        {
          case: textBatch([
            "Concept: A Vector is an ordered row of scalars.",
            "Task: Put three scalars into vector[C] in C0, C1, C2 order.",
            "Check: The output must have one C axis with length 3."
          ]),
          reference: runtimeValue(vectorVisible)
        },
        "vector_out",
        ["C"],
        [3]
      )
    ],
    hiddenTests: [
      shapeAndReferenceCase("vector_hidden", "vector keeps scalar order c0,c1,c2", { reference: runtimeValue(vectorHidden) }, "vector_out", ["C"], [3])
    ],
    certification: vectorCertification(),
    onboarding: {
      story: "A feature vector is a row of scalar cells with a named C rail.",
      startingProblem: "The scalar cells exist, but nothing stacks them into vector[C].",
      firstAction: "Add VectorRail and wire c0, c1, c2 in order.",
      targetRecipe: ["c0.out -> vector.c0", "c1.out -> vector.c1", "c2.out -> vector.c2", "vector.out -> vector_out.x", "vector_out.out -> reference.x"],
      winCondition: "vector_out has shape float32[C]=[3] and matches the reference order."
    },
    caseStudy: {
      title: "Create a valid Vector",
      narrative: "A Vector is not three loose numbers; it is an ordered group of scalars. In this level, place 0.5, -1, and 2 onto the C axis.",
      visibleInputFocus: "Task: Put three scalars into vector[C] in C0, C1, C2 order.",
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: "Task: Put three scalars into vector[C] in C0, C1, C2 order." },
        { type: "tensor_preview", title: "Expected vector", inputKey: "reference" }
      ],
      playerQuestion: "Do the three scalars enter the C axis in the correct order?",
      successObservation: "The output is float32[C=3], so the Vector is valid."
    },
    debrief: {
      completeTitle: "VectorRail certified",
      fixedProblem: "Three scalar cells now form one semantic feature rail.",
      learned: "Composition is a graph operation: value order becomes axis position.",
      nextUse: "MatrixStruct will stack vectors into a rank-2 component."
    }
  };
}

function dotProductLevel(): LevelSpec {
  return {
    id: "mvp01_ch0_04_dot_product",
    title: "MVP0.1-4 DotProduct",
    mode: "graph_challenge",
    chapter: "MVP0.1 Component Arc",
    goal: "Build DotProduct from elementwise multiply plus SumReduce; do not use a prebuilt DotProduct node.",
    modulePalette: ["InputTensor", "ElementwiseMultiply", "SumReduce", "OutputContractGate", "ReferenceChecker"],
    initialGraph: createMvp01DotProductInitialGraph(),
    targetGraph: createMvp01DotProductSolutionGraph(),
    constraints: { maxNodes: 6, maxEdges: 5, forbiddenModules: ["component.dot_product.v1"] },
    visibleTests: [
      shapeAndReferenceCase(
        "dot_product_visible",
        "multiply matching C cells, then reduce to scalar",
        {
          case: textBatch([
            "Concept: DotProduct multiplies matching vector cells and sums over C.",
            "Task: Build score = sum(query[C] * key[C]).",
            "Check: The output must be one scalar float32[] and match the numeric reference."
          ]),
          query: runtimeValue(dotQueryVisible),
          key: runtimeValue(dotKeyVisible),
          reference: runtimeValue(dotVisible)
        },
        "dot_out",
        [],
        [],
        dotProductStructureAssertions()
      )
    ],
    hiddenTests: [
      shapeAndReferenceCase(
        "dot_product_hidden_c5",
        "DotProduct generalizes when C changes",
        { query: runtimeValue(dotQueryHidden), key: runtimeValue(dotKeyHidden), reference: runtimeValue(dotHidden) },
        "dot_out",
        [],
        [],
        dotProductStructureAssertions()
      )
    ],
    certification: dotProductCertification(),
    onboarding: {
      story: "A dot product is the first numeric component that must compute, not just reshape.",
      startingProblem: "query and key vectors are on the canvas, but the graph has no multiply-then-sum implementation.",
      firstAction: "Add ElementwiseMultiply and SumReduce. Feed query/key into multiply, then reduce the product over C.",
      targetRecipe: ["query.out -> multiply.left", "key.out -> multiply.right", "multiply.out -> sum.x", "sum.out -> dot_out.x", "dot_out.out -> reference.x"],
      winCondition: "dot_out is a rank-0 float32 scalar and allclose to the reference."
    },
    caseStudy: {
      title: "Create a valid DotProduct",
      narrative: "DotProduct compares two equal-length vectors. Each C cell is multiplied with its matching partner, then all C products collapse into one score.",
      visibleInputFocus: "Task: Build score = sum(query[C] * key[C]).",
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: "Task: Build score = sum(query[C] * key[C])." },
        { type: "tensor_preview", title: "Query vector", inputKey: "query" },
        { type: "tensor_preview", title: "Key vector", inputKey: "key" },
        { type: "tensor_preview", title: "Expected scalar score", inputKey: "reference" }
      ],
      playerQuestion: "Did the graph multiply every matching C cell and then remove the C axis by summing?",
      successObservation: "The output is float32[] and numerically matches sum(query * key)."
    },
    debrief: {
      completeTitle: "DotProduct certified",
      fixedProblem: "The score is now computed by a real multiply-then-reduce graph.",
      learned: "A component certification must prove behavior, not just dtype and shape.",
      nextUse: "MatrixStruct and MatMulGate will build larger tables from repeated dot-product thinking."
    }
  };
}

function matrixStructLevel(): LevelSpec {
  return {
    id: "mvp01_3_matrix_struct",
    title: "MVP0.1-3 Matrix Struct",
    mode: "graph_challenge",
    chapter: "MVP0.1 Component Arc",
    goal: "Stack two vector[C] inputs into a matrix[C,O] with shape metadata.",
    modulePalette: ["InputTensor", "MatrixStruct", "OutputContractGate", "ReferenceChecker"],
    initialGraph: createMvp01MatrixInitialGraph(),
    targetGraph: createMvp01MatrixSolutionGraph(),
    constraints: { maxNodes: 5, maxEdges: 4 },
    visibleTests: [
      shapeAndReferenceCase(
        "matrix_visible",
        "two columns become matrix[C,O]",
        {
          case: textBatch([
            "Concept: A Matrix is a 2D structure where values are located by row and column.",
            "Task: Use two vector[C] inputs as two columns to build matrix[C,O].",
            "Check: The output must have C and O axes with shape [3,2]."
          ]),
          col0: runtimeValue(matrixCol0),
          col1: runtimeValue(matrixCol1),
          reference: runtimeValue(matrixVisible)
        },
        "matrix_out",
        ["C", "O"],
        [3, 2]
      )
    ],
    hiddenTests: [
      shapeAndReferenceCase(
        "matrix_hidden_c4",
        "C dimension can mutate while O stays structural",
        { col0: runtimeValue(matrixHiddenCol0), col1: runtimeValue(matrixHiddenCol1), reference: runtimeValue(matrixHidden) },
        "matrix_out",
        ["C", "O"],
        [4, 2]
      )
    ],
    certification: matrixCertification(),
    onboarding: {
      story: "The component now needs rank, dims, axes, and stride-like metadata.",
      startingProblem: "Two vector sources are present, but the matrix contract has no internal graph.",
      firstAction: "Add MatrixStruct and feed the two vectors as O0 and O1 columns.",
      targetRecipe: ["col0.out -> matrix.o0", "col1.out -> matrix.o1", "matrix.out -> matrix_out.x", "matrix_out.out -> reference.x"],
      winCondition: "matrix_out is float32[C,O] and hidden C mutations still pass."
    },
    caseStudy: {
      title: "Create a valid Matrix",
      narrative: "A Matrix is a two-dimensional data structure. In this level, the first axis is C and the second axis is O; the two input vectors become the O0 and O1 columns.",
      visibleInputFocus: "Task: Use two vector[C] inputs as two columns to build matrix[C,O].",
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: "Task: Use two vector[C] inputs as two columns to build matrix[C,O]." },
        { type: "tensor_preview", title: "O0 column", inputKey: "col0" },
        { type: "tensor_preview", title: "O1 column", inputKey: "col1" },
        { type: "tensor_preview", title: "Expected matrix", inputKey: "reference" }
      ],
      playerQuestion: "Do these two vectors form a 2D matrix instead of being joined into one long vector?",
      successObservation: "The output is float32[C=3,O=2], so the Matrix is valid."
    },
    debrief: {
      completeTitle: "MatrixStruct certified",
      fixedProblem: "Two vector rails are now represented as one rank-2 structure.",
      learned: "A matrix component is a value plus a shape contract.",
      nextUse: "TensorBox will introduce batch and token axes."
    }
  };
}

function tensorBoxLevel(): LevelSpec {
  return {
    id: "mvp01_4_tensor_box",
    title: "MVP0.1-4 Tensor Box",
    mode: "graph_challenge",
    chapter: "MVP0.1 Component Arc",
    goal: "Stack token vectors into a tensor[B,T,C] while preserving C as the innermost feature rail.",
    modulePalette: ["InputTensor", "TensorBox", "OutputContractGate", "ReferenceChecker"],
    initialGraph: createMvp01TensorInitialGraph(),
    targetGraph: createMvp01TensorSolutionGraph(),
    constraints: { maxNodes: 5, maxEdges: 4 },
    visibleTests: [
      shapeAndReferenceCase(
        "tensor_visible",
        "two token vectors become [B,T,C]",
        {
          case: textBatch([
            "Concept: A Tensor is a multi-dimensional data structure with named axes.",
            "Task: Place two vector[C] rows into B and T axes to build tensor[B,T,C].",
            "Check: The output must have B, T, C axes with shape [1,2,3]."
          ]),
          t0: runtimeValue(token0),
          t1: runtimeValue(token1),
          reference: runtimeValue(tensorVisible)
        },
        "tensor_out",
        ["B", "T", "C"],
        [1, 2, 3]
      )
    ],
    hiddenTests: [
      shapeAndReferenceCase(
        "tensor_hidden_c4",
        "feature width mutates while B/T semantics stay fixed",
        { t0: runtimeValue(hiddenToken0), t1: runtimeValue(hiddenToken1), reference: runtimeValue(tensorHidden) },
        "tensor_out",
        ["B", "T", "C"],
        [1, 2, 4]
      )
    ],
    certification: tensorCertification(),
    onboarding: {
      story: "The vector rail becomes a token row inside a batch.",
      startingProblem: "Token vectors are available, but no component introduces B and T axes.",
      firstAction: "Add TensorBox and connect t0 and t1 as token rows.",
      targetRecipe: ["t0.out -> tensor.t0", "t1.out -> tensor.t1", "tensor.out -> tensor_out.x", "tensor_out.out -> reference.x"],
      winCondition: "tensor_out has axes [B,T,C] and hidden C width mutations pass."
    },
    caseStudy: {
      title: "Create a valid Tensor",
      narrative: "A Tensor can have three or more axes. This level only practices rank-3: B means a batch of samples, T means position, and C means the value channels at each position.",
      visibleInputFocus: "Task: Place two vector[C] rows into B and T axes to build tensor[B,T,C].",
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: "Task: Place two vector[C] rows into B and T axes to build tensor[B,T,C]." },
        { type: "tensor_preview", title: "C vector 0", inputKey: "t0" },
        { type: "tensor_preview", title: "C vector 1", inputKey: "t1" },
        { type: "tensor_preview", title: "Expected tensor", inputKey: "reference" }
      ],
      playerQuestion: "Does the C axis stay in the final dimension?",
      successObservation: "The output is float32[B=1,T=2,C=3], so the Tensor is valid."
    },
    debrief: {
      completeTitle: "TensorBox certified",
      fixedProblem: "Token vectors now form a batch-token-feature tensor.",
      learned: "Tensor rank is not decoration; later MatMul depends on C being the innermost axis.",
      nextUse: "MatMulGate will consume C and emit O."
    }
  };
}

function splitterLevel(): LevelSpec {
  return {
    id: "mvp01_ch1_01_splitter",
    title: "Chapter 1-1 Splitter",
    mode: "graph_challenge",
    chapter: "Chapter 1: Text to Token Pipeline",
    goal: "Build BoundarySplitter from raw text into ordered string_piece[T] without using a prebuilt Splitter component.",
    modulePalette: ["TextInput", "BoundarySplitter", "PieceBuffer", "TypeContractGate"],
    initialGraph: createMvp01SplitterInitialGraph(),
    targetGraph: createMvp01SplitterSolutionGraph(),
    constraints: { maxNodes: 4, maxEdges: 3, forbiddenModules: ["component.splitter.v1"] },
    visibleTests: [
      splitterCase(
        "splitter_visible",
        "split words and punctuation into pieces",
        splitterVisibleText,
        splitterVisiblePieces,
        [
          "Concept: BoundarySplitter cuts raw text into ordered pieces along word and punctuation boundaries.",
          "Task: Split the sentence into string_piece[T] without dropping punctuation.",
          "Check: The output pieces must be tokenizers | are | useful | !"
        ]
      )
    ],
    hiddenTests: [
      splitterCase("splitter_hidden_punctuation", "punctuation remains its own piece", splitterHiddenText, splitterHiddenPieces)
    ],
    certification: splitterCertification(),
    onboarding: {
      story: "Text cannot enter tensor modules as one opaque string; it first needs a stable boundary split.",
      startingProblem: "A raw text source and a string_piece contract are present, but there is no splitter path between them.",
      firstAction: "Add BoundarySplitter and PieceBuffer. Route raw text into the splitter, then buffer the produced pieces before the contract.",
      targetRecipe: ["text.out -> splitter.text", "splitter.pieces -> piece_buffer.pieces", "piece_buffer.out -> pieces_out.x"],
      winCondition: "pieces_out is string_piece[T] and exactly matches the expected word/punctuation pieces.",
      allowedMistakes: ["Dropping punctuation", "Connecting raw_text directly into the contract", "Skipping PieceBuffer so the implementation path cannot be certified"]
    },
    caseStudy: {
      title: "Create a valid BoundarySplitter",
      narrative:
        "BoundarySplitter is the first text component. It does not tokenize into IDs yet; it only turns one raw sentence into an ordered T-axis list of string pieces.",
      visibleInputFocus: "Task: Split the sentence into string_piece[T] without dropping punctuation.",
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: "Task: Split the sentence into string_piece[T] without dropping punctuation." },
        { type: "text_batch", title: "Raw text", inputKey: "texts" },
        { type: "tensor_preview", title: "Expected pieces", inputKey: "expected" }
      ],
      playerQuestion: "Does every word and punctuation mark become a separate ordered piece?",
      successObservation: "The output is string_piece[T=4] and the sequence is tokenizers | are | useful | !."
    },
    debrief: {
      completeTitle: "BoundarySplitter certified",
      fixedProblem: "Raw text now has a reusable graph-built boundary splitter.",
      learned: "Text components also need behavior certification: dtype and T shape are not enough unless the actual pieces are correct.",
      nextUse: "Merge rules and tokenizer sockets will build on these ordered pieces."
    }
  };
}

function matMulGateLevel(): LevelSpec {
  return {
    id: "mvp01_5_matmul_gate",
    title: "MVP0.1-5 MatMulGate",
    mode: "graph_challenge",
    chapter: "MVP0.1 Component Arc",
    goal: "Connect hidden[B,T,C] and weight[C,O] through MatMulGate to produce projected[B,T,O].",
    modulePalette: ["InputTensor", "WeightPlate", "MatMulGate", "OutputContractGate", "ReferenceChecker"],
    initialGraph: createMvp01MatMulInitialGraph(),
    targetGraph: createMvp01MatMulSolutionGraph(),
    constraints: { maxNodes: 5, maxEdges: 4 },
    visibleTests: [
      shapeAndReferenceCase(
        "matmul_visible",
        "MatMul consumes C and emits O",
        {
          case: textBatch([
            "Concept: MatMulGate multiplies tensors by matching the left last C axis with the right first C axis.",
            "Task: Compute left[B,T,C] with right[C,O] to produce output[B,T,O].",
            "Check: C is consumed and O becomes the new output axis."
          ]),
          hidden: runtimeValue(tensorVisible),
          weight: runtimeValue(matrixVisible),
          reference: runtimeValue(matmulVisible)
        },
        "matmul_out",
        ["B", "T", "O"],
        [1, 2, 2]
      )
    ],
    hiddenTests: [
      shapeAndReferenceCase(
        "matmul_hidden_mutation",
        "B/T/C/O mutation keeps the same contract",
        { hidden: runtimeValue(matmulHiddenInput), weight: runtimeValue(matmulHiddenWeight), reference: runtimeValue(matmulHidden) },
        "matmul_out",
        ["B", "T", "O"],
        [2, 3, 3]
      )
    ],
    certification: matMulCertification(),
    onboarding: {
      story: "The tensor and matrix components meet at the C rail.",
      startingProblem: "hidden and weight are present, but the graph has no compute gate between them.",
      firstAction: "Add MatMulGate, connect hidden to left and weight to right.",
      targetRecipe: ["hidden.out -> matmul.left", "weight.out -> matmul.right", "matmul.out -> matmul_out.x", "matmul_out.out -> reference.x"],
      winCondition: "matmul_out is [B,T,O] and certification mutations pass."
    },
    caseStudy: {
      title: "Create a valid MatMulGate",
      narrative: "The current MatMulGate rule is simple: the last axis of left must match the first axis of right. Here C matches C, while the output keeps B and T and creates O.",
      visibleInputFocus: "Task: Compute left[B,T,C] with right[C,O] to produce output[B,T,O].",
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: "Task: Compute left[B,T,C] with right[C,O] to produce output[B,T,O]." },
        { type: "tensor_preview", title: "Left input left[B,T,C]", inputKey: "hidden" },
        { type: "tensor_preview", title: "Right input right[C,O]", inputKey: "weight" },
        { type: "tensor_preview", title: "Expected output", inputKey: "reference" }
      ],
      playerQuestion: "Are left.C and right.C connected correctly?",
      successObservation: "The output is float32[B,T,O], so the MatMulGate is valid."
    },
    debrief: {
      completeTitle: "MatMulGate certified",
      fixedProblem: "The projection gate now consumes hidden and weight with the correct operand roles.",
      learned: "MatMul is an axis contract, not just a table multiply.",
      nextUse: "Linear will compose MatMul plus bias broadcast and add."
    }
  };
}

function linearBuildLevel(): LevelSpec {
  return {
    id: "mvp01_6_linear",
    title: "MVP0.1-6 Linear",
    mode: "graph_challenge",
    chapter: "MVP0.1 Component Arc",
    goal: "Build Linear from MatMulGate + BroadcastRail + AddGate without using a prebuilt Linear node.",
    modulePalette: ["InputTensor", "WeightPlate", "component.matmul_gate.v1", "BroadcastRail", "AddGate", "OutputContractGate", "ReferenceChecker"],
    initialGraph: createMvp01LinearInitialGraph(),
    targetGraph: createMvp01LinearSolutionGraph(),
    constraints: { maxNodes: 8, maxEdges: 8, forbiddenModules: ["LinearModule"] },
    visibleTests: [
      shapeAndReferenceCase(
        "linear_visible",
        "Linear = matmul + bias broadcast + add",
        {
          case: textBatch([
            "Concept: Linear is a composed structure: MatMul first, then add bias[O] to each O channel.",
            "Task: Build output = MatMul(left, weight) + bias.",
            "Check: bias must align to O, then expand across B and T."
          ]),
          hidden: runtimeValue(tensorVisible),
          weight: runtimeValue(matrixVisible),
          bias: runtimeValue(linearBiasVisible),
          reference: runtimeValue(linearVisible)
        },
        "linear_out",
        ["B", "T", "O"],
        [1, 2, 2]
      )
    ],
    hiddenTests: [
      shapeAndReferenceCase(
        "linear_hidden_mutation",
        "Linear generalizes across B/T/C/O sizes",
        {
          hidden: runtimeValue(linearHiddenInput),
          weight: runtimeValue(linearHiddenWeight),
          bias: runtimeValue(linearHiddenBias),
          reference: runtimeValue(linearHidden)
        },
        "linear_out",
        ["B", "T", "O"],
        [2, 3, 3]
      )
    ],
    certification: linearCertification(),
    onboarding: {
      story: "The first complete component is built by composing smaller certified pieces.",
      startingProblem: "The interface ports are present, but the internal graph for Linear is empty.",
      firstAction: "Add MatMulGate, BroadcastRail, and AddGate. Broadcast bias along O before adding.",
      targetRecipe: [
        "hidden.out -> matmul.left",
        "weight.out -> matmul.right",
        "matmul.out -> bias_broadcast.target",
        "bias.out -> bias_broadcast.small",
        "matmul.out -> linear_add.left",
        "bias_broadcast.out -> linear_add.right",
        "linear_add.out -> linear_out.x",
        "linear_out.out -> reference.x"
      ],
      winCondition: "linear_out passes the task check and certification as [B,T,O]."
    },
    caseStudy: {
      title: "Create a valid Linear",
      narrative: "Linear currently expresses one composed formula: the MatMul result plus bias. The focus is not a new formula; bias[O] must align to the output O axis.",
      visibleInputFocus: "Task: Build output = MatMul(left, weight) + bias.",
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: "Task: Build output = MatMul(left, weight) + bias." },
        { type: "tensor_preview", title: "Left input left[B,T,C]", inputKey: "hidden" },
        { type: "tensor_preview", title: "Weight weight[C,O]", inputKey: "weight" },
        { type: "tensor_preview", title: "Bias bias[O]", inputKey: "bias" },
        { type: "tensor_preview", title: "Expected Linear output", inputKey: "reference" }
      ],
      playerQuestion: "Does bias align to the O axis instead of the wrong B or T axis?",
      successObservation: "The output is float32[B,T,O], so the Linear is valid."
    },
    debrief: {
      completeTitle: "Linear certified",
      fixedProblem: "Linear is now a graph-built component, not a prebuilt shortcut.",
      learned: "Available components can preserve their internal graph while exposing a clean interface.",
      nextUse: "The same component system can support upgrades, certification variants, and Expand Inside."
    }
  };
}

function qkScoreLevel(): LevelSpec {
  return {
    id: "mvp01_ch4_03_qk_score",
    title: "Chapter 4-3 QK Score",
    mode: "graph_challenge",
    chapter: "Chapter 4: Attention Core",
    goal: "Build QKScore from K transpose plus certified MatMulGate, then prove the score board numerically.",
    modulePalette: ["InputTensor", "TransposeSwitch", "component.matmul_gate.v1", "ScoreBoard", "CellTrace", "ReferenceChecker"],
    initialGraph: createMvp01QkScoreInitialGraph(),
    targetGraph: createMvp01QkScoreSolutionGraph(),
    constraints: { maxNodes: 7, maxEdges: 8, forbiddenModules: ["component.qk_score.v1"] },
    visibleTests: [
      shapeAndReferenceCase(
        "qk_score_visible",
        "K transpose creates token-to-token score board",
        {
          case: textBatch([
            "Concept: QKScore compares every query token with every key token.",
            "Task: Transpose only K's last two axes, then compute Q @ K^T.",
            "Check: The score board must be float32[B,H,T,T] and match the reference."
          ]),
          q: runtimeValue(qkVisibleQ),
          k: runtimeValue(qkVisibleK),
          reference: runtimeValue(qkVisibleReference)
        },
        "score_board",
        ["B", "H", "T", "T"],
        [1, 1, 3, 3],
        qkScoreStructureAssertions()
      )
    ],
    hiddenTests: [
      shapeAndReferenceCase(
        "qk_score_hidden_equal_td",
        "T==D still requires a real K transpose",
        { q: runtimeValue(qkHiddenEqualQ), k: runtimeValue(qkHiddenEqualK), reference: runtimeValue(qkHiddenEqualReference) },
        "score_board",
        ["B", "H", "T", "T"],
        [1, 1, 3, 3],
        qkScoreStructureAssertions()
      ),
      shapeAndReferenceCase(
        "qk_score_hidden_carry_axes",
        "B/H carrier axes survive the QK score",
        { q: runtimeValue(qkHiddenCarryQ), k: runtimeValue(qkHiddenCarryK), reference: runtimeValue(qkHiddenCarryReference) },
        "score_board",
        ["B", "H", "T", "T"],
        [2, 2, 4, 4],
        qkScoreStructureAssertions()
      )
    ],
    certification: qkScoreCertification(),
    onboarding: {
      story: "Attention scores are not Q @ K directly; K must present feature D as the inner dimension and key token T as the output column.",
      startingProblem: "Q and K sources are ready, but the score board cannot be certified until K is transposed before MatMul.",
      firstAction: "Add TransposeSwitch, certified MatMulGate, ScoreBoard, CellTrace, and ReferenceChecker. Route K through the transpose before qk_matmul.right.",
      targetRecipe: [
        "q.out -> qk_matmul.left",
        "k.out -> k_transpose.x",
        "k_transpose.out -> qk_matmul.right",
        "qk_matmul.out -> score_board.scores",
        "score_board.out -> reference.x",
        "score_board.out -> cell_trace.scores",
        "q.out -> cell_trace.q",
        "k.out -> cell_trace.k"
      ],
      winCondition: "score_board and cell_trace output [B,H,T,T] and allclose to the reference.",
      allowedMistakes: ["Connecting K directly to MatMul", "Transposing Q instead of K", "Dropping B/H carrier axes"]
    },
    caseStudy: {
      title: "Create a valid QKScore",
      narrative:
        "QKScore builds an attention score board. Each output cell compares one query token row against one key token row by dotting over D.",
      visibleInputFocus: "Task: Transpose only K's last two axes, then compute Q @ K^T.",
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: "Task: Transpose only K's last two axes, then compute Q @ K^T." },
        { type: "tensor_preview", title: "Query Q[B,H,T,D]", inputKey: "q", maxRows: 4, maxCols: 6 },
        { type: "tensor_preview", title: "Key K[B,H,T,D]", inputKey: "k", maxRows: 4, maxCols: 6 },
        { type: "tensor_preview", title: "Expected scores[B,H,T,T]", inputKey: "reference", maxRows: 4, maxCols: 6 }
      ],
      playerQuestion: "Did K become [B,H,D,T] before MatMul while Q stayed [B,H,T,D]?",
      successObservation: "The score board is float32[B,H,T,T] and matches Q @ K^T."
    },
    debrief: {
      completeTitle: "QKScore certified",
      fixedProblem: "The attention score board now uses the correct K transpose path and certified MatMulGate.",
      learned: "QKScore certification needs a numerical hidden trap: when T==D, shape alone can look plausible while the score cells are wrong.",
      nextUse: "Scale, mask, softmax, and weighted value sum will consume this score board."
    }
  };
}

function courseCatalogLevel(courseLevel: Mvp01CourseLevelDef): LevelSpec {
  const inputCount = mvp01CourseInputCount(courseLevel);
  const targetGraph = createCourseTargetGraph(courseLevel);
  const startsWithComponent = courseLevel.index % 2 === 1;
  const targetRecipe = [
    ...Array.from({ length: inputCount }, (_, index) => `input_${index}.out -> component.${courseInputPortId(index)}`),
    "component.out -> contract.x"
  ];

  return {
    id: courseLevel.id,
    title: `Chapter ${courseLevel.code} ${courseLevel.title}`,
    mode: "graph_challenge",
    chapter: `Chapter ${courseLevel.chapter}: ${courseLevel.chapterTitle}`,
    goal: courseLevel.goal,
    modulePalette: [courseLevel.moduleId],
    initialGraph: startsWithComponent ? createCourseWireRepairGraph(courseLevel) : createCourseInitialGraph(courseLevel),
    targetGraph,
    constraints: { maxNodes: inputCount + 2, maxEdges: inputCount + 1 },
    visibleTests: [courseContractCase(courseLevel, "visible", courseLevel.output.dims, "visible")],
    hiddenTests: [courseContractCase(courseLevel, "hidden", courseLevel.output.dims, "hidden")],
    certification: courseCertification(courseLevel),
    onboarding: {
      story: courseLevel.concept,
      startingProblem: startsWithComponent
        ? `${courseLevel.component} is on the canvas, but the contract does not receive its output yet.`
        : `${courseLevel.component} is missing from the graph. The source and contract are waiting for a valid component path.`,
      firstAction: startsWithComponent ? "Connect component.out to contract.x." : `Add ${courseLevel.component}, then wire the source ports into it and connect its output to contract.x.`,
      targetRecipe,
      winCondition: courseLevel.check,
      allowedMistakes: ["Forgetting one input edge", "Wiring a source directly into the contract", "Ignoring the expected dtype or axis names"]
    },
    caseStudy: {
      title: `Build ${courseLevel.component}`,
      narrative: courseLevel.concept,
      visibleInputFocus: courseLevel.task,
      dataPanels: [
        { type: "text_batch", title: "Task Brief", inputKey: "case", focusText: courseLevel.task },
        ...Array.from({ length: inputCount }, (_, index) => ({ type: "tensor_preview" as const, title: `Input ${courseInputPortId(index)}`, inputKey: `input_${index}` }))
      ],
      playerQuestion: `Can this graph turn the prepared case into a certified ${courseLevel.component} output?`,
      successObservation: courseLevel.check
    },
    debrief: {
      completeTitle: `${courseLevel.component} certified`,
      fixedProblem: `${courseLevel.component} now has a graph path from source inputs to a checked output contract.`,
      learned: courseLevel.learn,
      nextUse: courseLevel.futureUse
    }
  };
}

function courseCertification(courseLevel: Mvp01CourseLevelDef): LevelCertificationSpec {
  return {
    title: "Certification variant",
    narrative: `The current task checks one ${courseLevel.component} case. Certification mutates the generated case while preserving the component contract.`,
    publicVariantLabel: "Public variant",
    publicVariantDescription: "Adjust a compact generated variant. The system changes shape data without asking you to handwrite the full tensor.",
    systemVariantDescription: "System variants keep the same dtype and axis contract while changing the generated case seed.",
    controls: [
      ...(courseLevel.output.dims.length
        ? [{ id: "width", label: "variant width", kind: "integer" as const, defaultValue: Math.max(1, courseLevel.output.dims[courseLevel.output.dims.length - 1] + 1), min: 1, max: 12, help: "Mutates the last output dimension while preserving axis semantics." }]
        : []),
      { id: "seed", label: "variant seed", kind: "integer", defaultValue: courseLevel.chapter * 100 + courseLevel.index + 1, min: 1, max: 999, help: "Changes generated values without changing the component contract." }
    ],
    makePublicTests: (graph, values) => {
      const variantDims = courseVariantDims(courseLevel, values);
      const seed = String(controlInteger(values, "seed", courseLevel.chapter * 100 + courseLevel.index + 1));
      return [
        {
          graph: withNodeParams(graph, {
            component: { outputDims: variantDims, seed: `${courseLevel.slug}-${seed}` },
            contract: { expectedDims: variantDims }
          }),
          testCase: courseContractCase(courseLevel, "public_variant", variantDims, seed)
        }
      ];
    }
  };
}

function courseVariantDims(courseLevel: Mvp01CourseLevelDef, values: Record<string, CertificationControlValue>) {
  if (!courseLevel.output.dims.length) return [];
  const dims = [...courseLevel.output.dims];
  dims[dims.length - 1] = controlInteger(values, "width", dims[dims.length - 1]);
  return dims;
}

function courseContractCase(courseLevel: Mvp01CourseLevelDef, visibility: "visible" | "hidden" | "public_variant", dims: number[], seed: string): TestCase {
  const inputCount = mvp01CourseInputCount(courseLevel);
  const testVisibility = visibility === "visible" ? "visible" : "hidden";
  return {
    id: `${courseLevel.id}_${visibility}`,
    title: `${courseLevel.component} ${visibility.replace("_", " ")}`,
    visibility: testVisibility,
    inputSeed: `${courseLevel.slug}-${seed}`,
    inputs: {
      case: textBatch([`Concept: ${courseLevel.concept}`, `Task: ${courseLevel.task}`, `Check: ${courseLevel.check}`]),
      ...Object.fromEntries(
        Array.from({ length: inputCount }, (_, index) => [
          `input_${index}`,
          courseRuntimeValue(courseLevel, `input-${index}-${seed}`, index === 0 ? courseLevel.output.dims : courseInputDims(courseLevel.output.dims, index))
        ])
      )
    },
    assertions: [
      { type: "dtype", nodeId: "contract", expected: courseLevel.output.dtype },
      ...(courseLevel.output.dtype === "raw_text" ? [] : [{ type: "shape" as const, nodeId: "contract", expectedAxes: courseLevel.output.axes, expectedDims: dims }])
    ]
  };
}

function createCourseInitialGraph(courseLevel: Mvp01CourseLevelDef): GraphSpec {
  return {
    levelId: courseLevel.id,
    version: 1,
    nodes: [...courseInputNodes(courseLevel), courseContractNode(courseLevel)],
    edges: [],
    outputNodes: ["contract"]
  };
}

function createCourseWireRepairGraph(courseLevel: Mvp01CourseLevelDef): GraphSpec {
  const inputCount = mvp01CourseInputCount(courseLevel);
  return {
    levelId: courseLevel.id,
    version: 1,
    nodes: [...courseInputNodes(courseLevel), courseComponentNode(courseLevel), courseContractNode(courseLevel)],
    edges: Array.from({ length: inputCount }, (_, index) => ({
      id: `e_input_${index}_component`,
      from: { nodeId: `input_${index}`, portId: "out" },
      to: { nodeId: "component", portId: courseInputPortId(index) }
    })),
    outputNodes: ["contract"]
  };
}

function createCourseTargetGraph(courseLevel: Mvp01CourseLevelDef): GraphSpec {
  const inputCount = mvp01CourseInputCount(courseLevel);
  return {
    levelId: courseLevel.id,
    version: 1,
    nodes: [...courseInputNodes(courseLevel), courseComponentNode(courseLevel), courseContractNode(courseLevel)],
    edges: [
      ...Array.from({ length: inputCount }, (_, index) => ({
        id: `e_input_${index}_component`,
        from: { nodeId: `input_${index}`, portId: "out" },
        to: { nodeId: "component", portId: courseInputPortId(index) }
      })),
      { id: "e_component_contract", from: { nodeId: "component", portId: "out" }, to: { nodeId: "contract", portId: "x" } }
    ],
    outputNodes: ["contract"]
  };
}

function courseInputNodes(courseLevel: Mvp01CourseLevelDef) {
  return Array.from({ length: mvp01CourseInputCount(courseLevel) }, (_, index) => ({
    id: `input_${index}`,
    moduleId: "CourseCaseSource",
    params: {
      inputKey: `input_${index}`,
      dtype: courseLevel.output.dtype,
      dims: index === 0 ? courseLevel.output.dims : courseInputDims(courseLevel.output.dims, index),
      axes: courseLevel.output.axes,
      seed: `${courseLevel.slug}-input-${index}`
    },
    position: { x: 80, y: 90 + index * 155 },
    locked: true
  }));
}

function courseComponentNode(courseLevel: Mvp01CourseLevelDef) {
  return {
    id: "component",
    moduleId: courseLevel.moduleId,
    params: {
      outputDType: courseLevel.output.dtype,
      outputDims: courseLevel.output.dims,
      outputAxes: courseLevel.output.axes,
      seed: courseLevel.slug
    },
    position: { x: 410, y: 160 }
  };
}

function courseContractNode(courseLevel: Mvp01CourseLevelDef) {
  return {
    id: "contract",
    moduleId: "TypeContractGate",
    params: {
      expectedDType: courseLevel.output.dtype,
      expectedDims: courseLevel.output.dims,
      expectedAxes: courseLevel.output.axes
    },
    position: { x: 760, y: 170 },
    locked: true
  };
}

function courseInputPortId(index: number) {
  return index === 0 ? "x" : index === 1 ? "y" : "z";
}

function courseInputDims(dims: number[], index: number) {
  if (!dims.length) return [];
  return dims.map((dim, dimIndex) => Math.max(1, dim - (index === dimIndex % 2 ? 0 : 1)));
}

function courseRuntimeValue(courseLevel: Mvp01CourseLevelDef, seed: string, dims: number[]): RuntimeValue {
  if (courseLevel.output.dtype === "raw_text") {
    const text = `case ${seed}: ${courseLevel.component} data`;
    return { dtype: "raw_text", data: text, meta: { batch: [text] } };
  }
  if (courseLevel.output.dtype === "string_piece" || courseLevel.output.dtype === "token_piece") {
    const count = Math.max(1, dims.reduce((total, dim) => total * dim, 1));
    return {
      dtype: courseLevel.output.dtype,
      shape: { dtype: courseLevel.output.dtype, dims, axes: courseLevel.output.axes },
      data: Array.from({ length: count }, (_, index) => `${courseLevel.slug}_${index}`)
    };
  }
  const tensorDType = courseLevel.output.dtype === "bool" || courseLevel.output.dtype === "mask" || courseLevel.output.dtype === "int" ? courseLevel.output.dtype : "float32";
  const tensor = seededTensor(dims, courseLevel.output.axes, seed, tensorDType);
  return runtimeValue(tensor);
}

function scalarCertification(): LevelCertificationSpec {
  return {
    title: "Certification variant",
    narrative: "The current task checks the shown scalar. Certification asks whether the component still satisfies the scalar contract when the value changes.",
    publicVariantLabel: "Public variant",
    publicVariantDescription: "Choose another finite float32 value. The system will temporarily use it as the scalar source during certification.",
    systemVariantDescription: "System variants also check finite rank-0 output without revealing their exact values.",
    controls: [
      { id: "value", label: "Scalar value", kind: "number", defaultValue: 0.6, step: 0.1, help: "Any finite float32 value should remain a rank-0 scalar." }
    ],
    makePublicTests: (graph, values) => {
      const value = controlNumber(values, "value", 0.6);
      return [
        {
          graph: withNodeParams(graph, { scalar_source: { value } }),
          testCase: shapeContractCase("scalar_hidden_public_variant", "public certification: finite scalar variant", { reference: runtimeValue(scalar(value)) }, "scalar_out", [], [])
        }
      ];
    }
  };
}

function vectorCertification(): LevelCertificationSpec {
  return {
    title: "Certification variant",
    narrative: "The current task checks one vector. Certification changes the scalar values and verifies that order still maps to C positions.",
    publicVariantLabel: "Public variant",
    publicVariantDescription: "Edit three scalar values. Certification will temporarily feed them through c0, c1, and c2.",
    systemVariantDescription: "System variants keep checking the vector contract and the c0,c1,c2 order.",
    controls: [
      { id: "c0", label: "c0 value", kind: "number", defaultValue: 1.25, step: 0.1, help: "First C-axis slot." },
      { id: "c1", label: "c1 value", kind: "number", defaultValue: 0.25, step: 0.1, help: "Second C-axis slot." },
      { id: "c2", label: "c2 value", kind: "number", defaultValue: -2, step: 0.1, help: "Third C-axis slot." }
    ],
    makePublicTests: (graph, values) => {
      const vectorValues = [controlNumber(values, "c0", 1.25), controlNumber(values, "c1", 0.25), controlNumber(values, "c2", -2)];
      return [
        {
          graph: withNodeParams(graph, {
            scalar_c0: { value: vectorValues[0] },
            scalar_c1: { value: vectorValues[1] },
            scalar_c2: { value: vectorValues[2] }
          }),
          testCase: shapeAndReferenceCase(
            "vector_hidden_public_variant",
            "public certification: changed scalar values keep C order",
            { reference: runtimeValue(vector(vectorValues)) },
            "vector_out",
            ["C"],
            [3]
          )
        }
      ];
    }
  };
}

function dotProductCertification(): LevelCertificationSpec {
  return {
    title: "Certification variant",
    narrative: "The current task checks one dot product. Certification changes the C width and data while requiring the same multiply-then-sum graph.",
    publicVariantLabel: "Public variant",
    publicVariantDescription: "Choose C width and a data seed. The system generates compatible query/key vectors.",
    systemVariantDescription: "System variants mutate C again and still require numeric allclose, not just scalar shape.",
    controls: [cWidthControl(5), seedControl("dot-public")],
    makePublicTests: (_graph, values) => {
      const c = controlInteger(values, "c", 5);
      const seed = controlString(values, "seed", "dot-public");
      const query = seededTensor([c], ["C"], `${seed}-query`);
      const key = seededTensor([c], ["C"], `${seed}-key`);
      const reference = dotProduct(query, key);
      return [
        {
          testCase: shapeAndReferenceCase(
            "dot_product_public_variant",
            "public certification: generated C-width dot product",
            { query: runtimeValue(query), key: runtimeValue(key), reference: runtimeValue(reference) },
            "dot_out",
            [],
            [],
            dotProductStructureAssertions()
          )
        }
      ];
    }
  };
}

function splitterCertification(): LevelCertificationSpec {
  return {
    title: "Certification variant",
    narrative: "The current task checks one sentence. Certification changes the raw text while requiring the same boundary split graph and exact pieces.",
    publicVariantLabel: "Public variant",
    publicVariantDescription: "Choose a compact text fixture. The system supplies raw text and expected pieces so you do not handwrite arrays.",
    systemVariantDescription: "System variants keep punctuation and word boundaries private while checking the same string_piece[T] contract.",
    controls: [
      {
        id: "text_case",
        label: "Text fixture",
        kind: "select",
        defaultValue: "shape-buffer",
        help: "Changes the raw text case while keeping the BoundarySplitter contract.",
        options: [
          { value: "shape-buffer", label: "shape, token buffer" },
          { value: "we-train", label: "we train llm." },
          { value: "tokenizers", label: "tokenizers are useful!" }
        ]
      }
    ],
    makePublicTests: (_graph, values) => {
      const fixture = splitterFixture(controlString(values, "text_case", "shape-buffer"));
      return [
        {
          testCase: splitterCase("splitter_public_variant", "public certification: text fixture split", fixture.text, fixture.pieces)
        }
      ];
    }
  };
}

function matrixCertification(): LevelCertificationSpec {
  return {
    title: "Certification variant",
    narrative: "The current task checks C=3. Certification changes C width while preserving the two O columns.",
    publicVariantLabel: "Public variant",
    publicVariantDescription: "Choose a C width and data seed. The system generates two vector[C] columns for you.",
    systemVariantDescription: "System variants mutate C again and still require matrix[C,O].",
    controls: [cWidthControl(5), seedControl("matrix-public")],
    makePublicTests: (_graph, values) => {
      const c = controlInteger(values, "c", 5);
      const seed = controlString(values, "seed", "matrix-public");
      const col0 = seededTensor([c], ["C"], `${seed}-col0`);
      const col1 = seededTensor([c], ["C"], `${seed}-col1`);
      const reference = matrixFromColumns(col0, col1);
      return [
        {
          testCase: shapeAndReferenceCase(
            "matrix_hidden_public_variant",
            "public certification: generated C width variant",
            { col0: runtimeValue(col0), col1: runtimeValue(col1), reference: runtimeValue(reference) },
            "matrix_out",
            ["C", "O"],
            [c, 2]
          )
        }
      ];
    }
  };
}

function tensorCertification(): LevelCertificationSpec {
  return {
    title: "Certification variant",
    narrative: "The current task checks one token pair. Certification changes feature width C while keeping B and T semantics.",
    publicVariantLabel: "Public variant",
    publicVariantDescription: "Choose a C width and data seed. The system generates two token vector[C] rows.",
    systemVariantDescription: "System variants keep C mutable while requiring tensor[B,T,C].",
    controls: [cWidthControl(5), seedControl("tensor-public")],
    makePublicTests: (_graph, values) => {
      const c = controlInteger(values, "c", 5);
      const seed = controlString(values, "seed", "tensor-public");
      const t0 = seededTensor([c], ["C"], `${seed}-t0`);
      const t1 = seededTensor([c], ["C"], `${seed}-t1`);
      const reference = tensorFromTokens(t0, t1);
      return [
        {
          testCase: shapeAndReferenceCase(
            "tensor_hidden_public_variant",
            "public certification: generated token feature variant",
            { t0: runtimeValue(t0), t1: runtimeValue(t1), reference: runtimeValue(reference) },
            "tensor_out",
            ["B", "T", "C"],
            [1, 2, c]
          )
        }
      ];
    }
  };
}

function matMulCertification(): LevelCertificationSpec {
  return {
    title: "Certification variant",
    narrative: "The current task checks one [B,T,C] @ [C,O]. Certification changes the dimensions while preserving the C-to-O contract.",
    publicVariantLabel: "Public variant",
    publicVariantDescription: "Choose B/T/C/O and a seed. The system generates compatible hidden and weight tensors.",
    systemVariantDescription: "System variants use additional B/T/C/O sizes that are not shown here.",
    controls: [bSizeControl(2), tLengthControl(2), cWidthControl(4), oWidthControl(3), seedControl("matmul-public")],
    makePublicTests: (_graph, values) => {
      const b = controlInteger(values, "b", 2);
      const t = controlInteger(values, "t", 2);
      const c = controlInteger(values, "c", 4);
      const o = controlInteger(values, "o", 3);
      const seed = controlString(values, "seed", "matmul-public");
      const hidden = seededTensor([b, t, c], ["B", "T", "C"], `${seed}-hidden`);
      const weight = seededTensor([c, o], ["C", "O"], `${seed}-weight`);
      const reference = matmul(hidden, weight);
      return [
        {
          testCase: shapeAndReferenceCase(
            "matmul_hidden_public_variant",
            "public certification: generated B/T/C/O variant",
            { hidden: runtimeValue(hidden), weight: runtimeValue(weight), reference: runtimeValue(reference) },
            "matmul_out",
            ["B", "T", "O"],
            [b, t, o]
          )
        }
      ];
    }
  };
}

function linearCertification(): LevelCertificationSpec {
  return {
    title: "Certification variant",
    narrative: "The current task checks one Linear shape. Certification changes dimensions and verifies MatMul plus bias broadcast together.",
    publicVariantLabel: "Public variant",
    publicVariantDescription: "Choose B/T/C/O and a seed. The system generates hidden, weight, and bias tensors.",
    systemVariantDescription: "System variants add private B/T/C/O mutations before Linear becomes available.",
    controls: [bSizeControl(2), tLengthControl(2), cWidthControl(4), oWidthControl(3), seedControl("linear-public")],
    makePublicTests: (_graph, values) => {
      const b = controlInteger(values, "b", 2);
      const t = controlInteger(values, "t", 2);
      const c = controlInteger(values, "c", 4);
      const o = controlInteger(values, "o", 3);
      const seed = controlString(values, "seed", "linear-public");
      const hidden = seededTensor([b, t, c], ["B", "T", "C"], `${seed}-hidden`);
      const weight = seededTensor([c, o], ["C", "O"], `${seed}-weight`);
      const bias = seededTensor([o], ["O"], `${seed}-bias`);
      const reference = linear(hidden, weight, bias);
      return [
        {
          testCase: shapeAndReferenceCase(
            "linear_hidden_public_variant",
            "public certification: generated Linear dimension variant",
            { hidden: runtimeValue(hidden), weight: runtimeValue(weight), bias: runtimeValue(bias), reference: runtimeValue(reference) },
            "linear_out",
            ["B", "T", "O"],
            [b, t, o]
          )
        }
      ];
    }
  };
}

function qkScoreCertification(): LevelCertificationSpec {
  return {
    title: "Certification variant",
    narrative: "The current task checks one QK score board. Certification mutates B/H/T/D and seed while requiring the same K-transpose implementation path.",
    publicVariantLabel: "Public variant",
    publicVariantDescription: "Choose B/H/T/D and a seed. The system generates compatible Q/K tensors and reference scores.",
    systemVariantDescription: "System variants include equal T/D traps and carrier-axis mutations.",
    controls: [bSizeControl(1), hSizeControl(2), tLengthControl(3), dWidthControl(3), seedControl("qk-public")],
    makePublicTests: (_graph, values) => {
      const b = controlInteger(values, "b", 1);
      const h = controlInteger(values, "h", 2);
      const t = controlInteger(values, "t", 3);
      const d = controlInteger(values, "d", 3);
      const seed = controlString(values, "seed", "qk-public");
      const q = seededTensor([b, h, t, d], ["B", "H", "T", "D"], `${seed}-q`);
      const k = seededTensor([b, h, t, d], ["B", "H", "T", "D"], `${seed}-k`);
      const reference = referenceQKScores(q, k);
      return [
        {
          testCase: shapeAndReferenceCase(
            "qk_score_public_variant",
            "public certification: generated QK score board",
            { q: runtimeValue(q), k: runtimeValue(k), reference: runtimeValue(reference) },
            "score_board",
            ["B", "H", "T", "T"],
            [b, h, t, t],
            qkScoreStructureAssertions()
          )
        }
      ];
    }
  };
}

function cWidthControl(defaultValue: number) {
  return { id: "c", label: "C width", kind: "integer" as const, defaultValue, min: 2, max: 6, step: 1, help: "Feature/channel width used by the generated certification variant." };
}

function bSizeControl(defaultValue: number) {
  return { id: "b", label: "B size", kind: "integer" as const, defaultValue, min: 1, max: 3, step: 1, help: "Batch size for the generated certification variant." };
}

function tLengthControl(defaultValue: number) {
  return { id: "t", label: "T length", kind: "integer" as const, defaultValue, min: 1, max: 4, step: 1, help: "Token length for the generated certification variant." };
}

function hSizeControl(defaultValue: number) {
  return { id: "h", label: "H heads", kind: "integer" as const, defaultValue, min: 1, max: 3, step: 1, help: "Attention head count for the generated certification variant." };
}

function dWidthControl(defaultValue: number) {
  return { id: "d", label: "D width", kind: "integer" as const, defaultValue, min: 2, max: 5, step: 1, help: "Feature width reduced inside QK dot products." };
}

function oWidthControl(defaultValue: number) {
  return { id: "o", label: "O width", kind: "integer" as const, defaultValue, min: 1, max: 5, step: 1, help: "Output channel width for the generated certification variant." };
}

function seedControl(defaultValue: string) {
  return {
    id: "seed",
    label: "Data seed",
    kind: "select" as const,
    defaultValue,
    help: "Changes generated values without hand-writing full tensors.",
    options: [
      { value: defaultValue, label: "Seed A" },
      { value: `${defaultValue}-b`, label: "Seed B" },
      { value: `${defaultValue}-edge`, label: "Edge seed" }
    ]
  };
}

export function createMvp01ScalarSolutionGraph(): GraphSpec {
  return {
    levelId: "mvp01_1_scalar_cell",
    version: 1,
    nodes: [
      { id: "scalar_source", moduleId: "Float32Literal", params: { value: 0.5 }, position: { x: 120, y: 190 } },
      { id: "scalar_out", moduleId: "OutputContractGate", params: { expectedAxes: [] }, position: { x: 390, y: 190 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 660, y: 190 } }
    ],
    edges: [
      { id: "e_scalar_out", from: { nodeId: "scalar_source", portId: "out" }, to: { nodeId: "scalar_out", portId: "x" } },
      { id: "e_scalar_ref", from: { nodeId: "scalar_out", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["scalar_out", "reference"]
  };
}

export function createMvp01WireProbeSolutionGraph(): GraphSpec {
  return {
    levelId: "mvp01_0_wire_probe",
    version: 1,
    nodes: [
      { id: "packet_input", moduleId: "InputTensor", params: { inputKey: "packet", shape: [1, 2], axes: ["B", "T"] }, position: { x: 120, y: 190 } },
      { id: "port_gate", moduleId: "OutputContractGate", params: { expectedAxes: ["B", "T"] }, position: { x: 390, y: 190 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 660, y: 190 } }
    ],
    edges: [
      { id: "e_wire_probe_contract", from: { nodeId: "packet_input", portId: "out" }, to: { nodeId: "port_gate", portId: "x" } },
      { id: "e_wire_probe_reference", from: { nodeId: "port_gate", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["port_gate", "reference"]
  };
}

export function createMvp01WireProbeInitialGraph(): GraphSpec {
  const graph = createMvp01WireProbeSolutionGraph();
  return {
    ...graph,
    edges: []
  };
}

export function createMvp01ScalarInitialGraph(): GraphSpec {
  const graph = createMvp01ScalarSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => node.id !== "scalar_source"),
    edges: []
  };
}

export function createMvp01VectorSolutionGraph(): GraphSpec {
  return {
    levelId: "mvp01_2_vector_rail",
    version: 1,
    nodes: [
      { id: "scalar_c0", moduleId: "component.scalar_cell.v1", params: { value: vectorValues[0] }, position: { x: 80, y: 80 } },
      { id: "scalar_c1", moduleId: "component.scalar_cell.v1", params: { value: vectorValues[1] }, position: { x: 80, y: 230 } },
      { id: "scalar_c2", moduleId: "component.scalar_cell.v1", params: { value: vectorValues[2] }, position: { x: 80, y: 380 } },
      { id: "vector", moduleId: "VectorRail", params: {}, position: { x: 350, y: 220 } },
      { id: "vector_out", moduleId: "OutputContractGate", params: { expectedAxes: ["C"] }, position: { x: 620, y: 220 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 890, y: 220 } }
    ],
    edges: [
      { id: "e_c0_vector", from: { nodeId: "scalar_c0", portId: "out" }, to: { nodeId: "vector", portId: "c0" } },
      { id: "e_c1_vector", from: { nodeId: "scalar_c1", portId: "out" }, to: { nodeId: "vector", portId: "c1" } },
      { id: "e_c2_vector", from: { nodeId: "scalar_c2", portId: "out" }, to: { nodeId: "vector", portId: "c2" } },
      { id: "e_vector_out", from: { nodeId: "vector", portId: "out" }, to: { nodeId: "vector_out", portId: "x" } },
      { id: "e_vector_ref", from: { nodeId: "vector_out", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["vector_out", "reference"]
  };
}

export function createMvp01VectorInitialGraph(): GraphSpec {
  const graph = createMvp01VectorSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => node.id !== "vector"),
    edges: []
  };
}

export function createMvp01DotProductSolutionGraph(): GraphSpec {
  return {
    levelId: "mvp01_ch0_04_dot_product",
    version: 1,
    nodes: [
      { id: "query", moduleId: "InputTensor", params: { inputKey: "query", shape: [3], axes: ["C"] }, position: { x: 90, y: 130 } },
      { id: "key", moduleId: "InputTensor", params: { inputKey: "key", shape: [3], axes: ["C"] }, position: { x: 90, y: 310 } },
      { id: "multiply", moduleId: "ElementwiseMultiply", params: {}, position: { x: 370, y: 220 } },
      { id: "sum", moduleId: "SumReduce", params: { axis: "C" }, position: { x: 640, y: 220 } },
      { id: "dot_out", moduleId: "OutputContractGate", params: { expectedAxes: [] }, position: { x: 910, y: 220 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 1180, y: 220 } }
    ],
    edges: [
      { id: "e_query_multiply", from: { nodeId: "query", portId: "out" }, to: { nodeId: "multiply", portId: "left" } },
      { id: "e_key_multiply", from: { nodeId: "key", portId: "out" }, to: { nodeId: "multiply", portId: "right" } },
      { id: "e_multiply_sum", from: { nodeId: "multiply", portId: "out" }, to: { nodeId: "sum", portId: "x" } },
      { id: "e_sum_out", from: { nodeId: "sum", portId: "out" }, to: { nodeId: "dot_out", portId: "x" } },
      { id: "e_dot_ref", from: { nodeId: "dot_out", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["dot_out", "reference"]
  };
}

export function createMvp01DotProductInitialGraph(): GraphSpec {
  const graph = createMvp01DotProductSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => !["multiply", "sum"].includes(node.id)),
    edges: []
  };
}

export function createMvp01MatrixSolutionGraph(): GraphSpec {
  return {
    levelId: "mvp01_3_matrix_struct",
    version: 1,
    nodes: [
      { id: "col0", moduleId: "InputTensor", params: { inputKey: "col0", shape: [3], axes: ["C"] }, position: { x: 90, y: 140 } },
      { id: "col1", moduleId: "InputTensor", params: { inputKey: "col1", shape: [3], axes: ["C"] }, position: { x: 90, y: 310 } },
      { id: "matrix", moduleId: "MatrixStruct", params: {}, position: { x: 360, y: 220 } },
      { id: "matrix_out", moduleId: "OutputContractGate", params: { expectedAxes: ["C", "O"] }, position: { x: 630, y: 220 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 900, y: 220 } }
    ],
    edges: [
      { id: "e_col0_matrix", from: { nodeId: "col0", portId: "out" }, to: { nodeId: "matrix", portId: "o0" } },
      { id: "e_col1_matrix", from: { nodeId: "col1", portId: "out" }, to: { nodeId: "matrix", portId: "o1" } },
      { id: "e_matrix_out", from: { nodeId: "matrix", portId: "out" }, to: { nodeId: "matrix_out", portId: "x" } },
      { id: "e_matrix_ref", from: { nodeId: "matrix_out", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["matrix_out", "reference"]
  };
}

export function createMvp01MatrixInitialGraph(): GraphSpec {
  const graph = createMvp01MatrixSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => node.id !== "matrix"),
    edges: []
  };
}

export function createMvp01TensorSolutionGraph(): GraphSpec {
  return {
    levelId: "mvp01_4_tensor_box",
    version: 1,
    nodes: [
      { id: "t0", moduleId: "InputTensor", params: { inputKey: "t0", shape: [3], axes: ["C"] }, position: { x: 90, y: 140 } },
      { id: "t1", moduleId: "InputTensor", params: { inputKey: "t1", shape: [3], axes: ["C"] }, position: { x: 90, y: 310 } },
      { id: "tensor", moduleId: "TensorBox", params: {}, position: { x: 360, y: 220 } },
      { id: "tensor_out", moduleId: "OutputContractGate", params: { expectedAxes: ["B", "T", "C"] }, position: { x: 630, y: 220 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 900, y: 220 } }
    ],
    edges: [
      { id: "e_t0_tensor", from: { nodeId: "t0", portId: "out" }, to: { nodeId: "tensor", portId: "t0" } },
      { id: "e_t1_tensor", from: { nodeId: "t1", portId: "out" }, to: { nodeId: "tensor", portId: "t1" } },
      { id: "e_tensor_out", from: { nodeId: "tensor", portId: "out" }, to: { nodeId: "tensor_out", portId: "x" } },
      { id: "e_tensor_ref", from: { nodeId: "tensor_out", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["tensor_out", "reference"]
  };
}

export function createMvp01TensorInitialGraph(): GraphSpec {
  const graph = createMvp01TensorSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => node.id !== "tensor"),
    edges: []
  };
}

export function createMvp01SplitterSolutionGraph(): GraphSpec {
  return {
    levelId: "mvp01_ch1_01_splitter",
    version: 1,
    nodes: [
      { id: "text", moduleId: "TextInput", params: { inputKey: "texts" }, position: { x: 90, y: 220 } },
      { id: "splitter", moduleId: "BoundarySplitter", params: { policy: "word", preservePunctuation: true, expectedT: 4 }, position: { x: 360, y: 220 } },
      { id: "piece_buffer", moduleId: "PieceBuffer", params: { maxPieces: 12 }, position: { x: 640, y: 220 } },
      {
        id: "pieces_out",
        moduleId: "TypeContractGate",
        params: { expectedDType: "string_piece", expectedDims: [4], expectedAxes: ["T"] },
        position: { x: 920, y: 220 }
      }
    ],
    edges: [
      { id: "e_text_splitter", from: { nodeId: "text", portId: "out" }, to: { nodeId: "splitter", portId: "text" } },
      { id: "e_splitter_buffer", from: { nodeId: "splitter", portId: "pieces" }, to: { nodeId: "piece_buffer", portId: "pieces" } },
      { id: "e_buffer_out", from: { nodeId: "piece_buffer", portId: "out" }, to: { nodeId: "pieces_out", portId: "x" } }
    ],
    outputNodes: ["pieces_out"]
  };
}

export function createMvp01SplitterInitialGraph(): GraphSpec {
  const graph = createMvp01SplitterSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => !["splitter", "piece_buffer"].includes(node.id)),
    edges: []
  };
}

export function createMvp01MatMulSolutionGraph(): GraphSpec {
  return {
    levelId: "mvp01_5_matmul_gate",
    version: 1,
    nodes: [
      { id: "hidden", moduleId: "InputTensor", params: { inputKey: "hidden", shape: [1, 2, 3], axes: ["B", "T", "C"] }, position: { x: 90, y: 140 } },
      { id: "weight", moduleId: "WeightPlate", params: { inputKey: "weight", shape: [3, 2], axes: ["C", "O"], orientation: "C,O" }, position: { x: 90, y: 320 } },
      { id: "matmul", moduleId: "MatMulGate", params: {}, position: { x: 380, y: 230 } },
      { id: "matmul_out", moduleId: "OutputContractGate", params: { expectedAxes: ["B", "T", "O"] }, position: { x: 650, y: 230 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 920, y: 230 } }
    ],
    edges: [
      { id: "e_hidden_matmul", from: { nodeId: "hidden", portId: "out" }, to: { nodeId: "matmul", portId: "left" } },
      { id: "e_weight_matmul", from: { nodeId: "weight", portId: "out" }, to: { nodeId: "matmul", portId: "right" } },
      { id: "e_matmul_out", from: { nodeId: "matmul", portId: "out" }, to: { nodeId: "matmul_out", portId: "x" } },
      { id: "e_matmul_ref", from: { nodeId: "matmul_out", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["matmul_out", "reference"]
  };
}

export function createMvp01MatMulInitialGraph(): GraphSpec {
  const graph = createMvp01MatMulSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => node.id !== "matmul"),
    edges: []
  };
}

export function createMvp01LinearSolutionGraph(): GraphSpec {
  return {
    levelId: "mvp01_6_linear",
    version: 1,
    nodes: [
      { id: "hidden", moduleId: "InputTensor", params: { inputKey: "hidden", shape: [1, 2, 3], axes: ["B", "T", "C"] }, position: { x: 80, y: 120 } },
      { id: "weight", moduleId: "WeightPlate", params: { inputKey: "weight", shape: [3, 2], axes: ["C", "O"], orientation: "C,O" }, position: { x: 80, y: 300 } },
      { id: "bias", moduleId: "InputTensor", params: { inputKey: "bias", shape: [2], axes: ["O"] }, position: { x: 80, y: 500 } },
      { id: "matmul", moduleId: "component.matmul_gate.v1", params: {}, position: { x: 370, y: 205 } },
      { id: "bias_broadcast", moduleId: "BroadcastRail", params: { alignAxes: ["O"] }, position: { x: 620, y: 380 } },
      { id: "linear_add", moduleId: "AddGate", params: {}, position: { x: 880, y: 240 } },
      { id: "linear_out", moduleId: "OutputContractGate", params: { expectedAxes: ["B", "T", "O"] }, position: { x: 1140, y: 240 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 1400, y: 240 } }
    ],
    edges: [
      { id: "e_hidden_matmul", from: { nodeId: "hidden", portId: "out" }, to: { nodeId: "matmul", portId: "left" } },
      { id: "e_weight_matmul", from: { nodeId: "weight", portId: "out" }, to: { nodeId: "matmul", portId: "right" } },
      { id: "e_bias_broadcast_small", from: { nodeId: "bias", portId: "out" }, to: { nodeId: "bias_broadcast", portId: "small" } },
      { id: "e_matmul_broadcast_target", from: { nodeId: "matmul", portId: "out" }, to: { nodeId: "bias_broadcast", portId: "target" } },
      { id: "e_matmul_add", from: { nodeId: "matmul", portId: "out" }, to: { nodeId: "linear_add", portId: "left" } },
      { id: "e_broadcast_add", from: { nodeId: "bias_broadcast", portId: "out" }, to: { nodeId: "linear_add", portId: "right" } },
      { id: "e_add_out", from: { nodeId: "linear_add", portId: "out" }, to: { nodeId: "linear_out", portId: "x" } },
      { id: "e_linear_ref", from: { nodeId: "linear_out", portId: "out" }, to: { nodeId: "reference", portId: "x" } }
    ],
    outputNodes: ["linear_out", "reference"]
  };
}

export function createMvp01LinearInitialGraph(): GraphSpec {
  const graph = createMvp01LinearSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => !["matmul", "bias_broadcast", "linear_add"].includes(node.id)),
    edges: []
  };
}

export function createMvp01QkScoreSolutionGraph(): GraphSpec {
  return {
    levelId: "mvp01_ch4_03_qk_score",
    version: 1,
    nodes: [
      { id: "q", moduleId: "InputTensor", params: { inputKey: "q", shape: [1, 1, 3, 2], axes: ["B", "H", "T", "D"] }, position: { x: 80, y: 130 } },
      { id: "k", moduleId: "InputTensor", params: { inputKey: "k", shape: [1, 1, 3, 2], axes: ["B", "H", "T", "D"] }, position: { x: 80, y: 340 } },
      { id: "k_transpose", moduleId: "TransposeSwitch", params: { axisA: -2, axisB: -1 }, position: { x: 360, y: 340 } },
      { id: "qk_matmul", moduleId: "component.matmul_gate.v1", params: {}, position: { x: 640, y: 230 } },
      { id: "score_board", moduleId: "ScoreBoard", params: { expectedAxes: ["B", "H", "T", "T"] }, position: { x: 910, y: 230 } },
      { id: "cell_trace", moduleId: "CellTrace", params: { b: 0, h: 0, i: 0, j: 1 }, position: { x: 1170, y: 110 } },
      { id: "reference", moduleId: "ReferenceChecker", params: { referenceKey: "reference" }, position: { x: 1170, y: 330 } }
    ],
    edges: [
      { id: "e_q_matmul", from: { nodeId: "q", portId: "out" }, to: { nodeId: "qk_matmul", portId: "left" } },
      { id: "e_k_transpose", from: { nodeId: "k", portId: "out" }, to: { nodeId: "k_transpose", portId: "x" } },
      { id: "e_transpose_matmul", from: { nodeId: "k_transpose", portId: "out" }, to: { nodeId: "qk_matmul", portId: "right" } },
      { id: "e_matmul_score", from: { nodeId: "qk_matmul", portId: "out" }, to: { nodeId: "score_board", portId: "scores" } },
      { id: "e_score_reference", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "reference", portId: "x" } },
      { id: "e_score_trace", from: { nodeId: "score_board", portId: "out" }, to: { nodeId: "cell_trace", portId: "scores" } },
      { id: "e_q_trace", from: { nodeId: "q", portId: "out" }, to: { nodeId: "cell_trace", portId: "q" } },
      { id: "e_k_trace", from: { nodeId: "k", portId: "out" }, to: { nodeId: "cell_trace", portId: "k" } }
    ],
    outputNodes: ["score_board", "cell_trace", "reference"]
  };
}

export function createMvp01QkScoreInitialGraph(): GraphSpec {
  const graph = createMvp01QkScoreSolutionGraph();
  return {
    ...graph,
    nodes: graph.nodes.filter((node) => !["k_transpose", "qk_matmul", "score_board", "cell_trace", "reference"].includes(node.id)),
    edges: []
  };
}

function shapeAndReferenceCase(
  id: string,
  title: string,
  inputs: Record<string, RuntimeValue>,
  outputNodeId: string,
  expectedAxes: AxisName[],
  expectedDims: number[],
  extraAssertions: TestAssertion[] = []
): TestCase {
  return {
    id,
    title,
    visibility: id.includes("hidden") ? "hidden" : "visible",
    inputSeed: id,
    inputs,
    assertions: [
      ...extraAssertions,
      { type: "shape", nodeId: outputNodeId, expectedAxes, expectedDims },
      { type: "allclose", nodeId: outputNodeId, referenceNodeId: "reference", atol: 1e-5 }
    ]
  };
}

function splitterCase(id: string, title: string, text: string, expected: string[], caseLines?: string[]): TestCase {
  return {
    id,
    title,
    visibility: id.includes("hidden") ? "hidden" : "visible",
    inputSeed: id,
    inputs: {
      ...(caseLines ? { case: textBatch(caseLines) } : {}),
      texts: rawText(text),
      expected: stringPieces(expected)
    },
    assertions: [
      ...splitterStructureAssertions(),
      { type: "dtype", nodeId: "pieces_out", expected: "string_piece" },
      { type: "shape", nodeId: "pieces_out", expectedAxes: ["T"], expectedDims: [expected.length] },
      { type: "pieces_equal", nodeId: "pieces_out", expected }
    ]
  };
}

function dotProductStructureAssertions(): TestAssertion[] {
  return [
    { type: "requires_node", nodeId: "multiply", moduleId: "ElementwiseMultiply" },
    { type: "requires_node", nodeId: "sum", moduleId: "SumReduce" },
    { type: "requires_edge_path", from: "query", through: "multiply", to: "sum" },
    { type: "requires_edge_path", from: "key", through: "multiply", to: "sum" },
    { type: "requires_edge_path", from: "sum", through: "dot_out", to: "reference" }
  ];
}

function splitterStructureAssertions(): TestAssertion[] {
  return [
    { type: "requires_node", nodeId: "splitter", moduleId: "BoundarySplitter" },
    { type: "requires_node", nodeId: "piece_buffer", moduleId: "PieceBuffer" },
    { type: "requires_edge_path", from: "text", through: "splitter", to: "piece_buffer" },
    { type: "requires_edge_path", from: "splitter", through: "piece_buffer", to: "pieces_out" }
  ];
}

function qkScoreStructureAssertions(): TestAssertion[] {
  return [
    { type: "requires_node", nodeId: "k_transpose", moduleId: "TransposeSwitch" },
    { type: "requires_node", nodeId: "qk_matmul", moduleId: "component.matmul_gate.v1" },
    { type: "requires_node", nodeId: "score_board", moduleId: "ScoreBoard" },
    { type: "requires_node", nodeId: "cell_trace", moduleId: "CellTrace" },
    { type: "requires_edge_path", from: "k", through: "k_transpose", to: "qk_matmul" },
    { type: "requires_edge_path", from: "q", through: "qk_matmul", to: "score_board" },
    { type: "requires_edge_path", from: "qk_matmul", through: "score_board", to: "reference" },
    { type: "requires_edge_path", from: "score_board", through: "cell_trace", to: "cell_trace" },
    { type: "requires_edge_path", from: "q", through: "cell_trace", to: "cell_trace" },
    { type: "requires_edge_path", from: "k", through: "cell_trace", to: "cell_trace" }
  ];
}

function shapeContractCase(
  id: string,
  title: string,
  inputs: Record<string, RuntimeValue>,
  outputNodeId: string,
  expectedAxes: AxisName[],
  expectedDims: number[]
): TestCase {
  return {
    id,
    title,
    visibility: id.includes("hidden") ? "hidden" : "visible",
    inputSeed: id,
    inputs,
    assertions: [{ type: "shape", nodeId: outputNodeId, expectedAxes, expectedDims }]
  };
}

function splitterFixture(id: string) {
  const fixtures: Record<string, { text: string; pieces: string[] }> = {
    "shape-buffer": { text: "shape, token buffer", pieces: ["shape", ",", "token", "buffer"] },
    "we-train": { text: "we train llm.", pieces: ["we", "train", "llm", "."] },
    tokenizers: { text: "tokenizers are useful!", pieces: ["tokenizers", "are", "useful", "!"] }
  };
  return fixtures[id] ?? fixtures["shape-buffer"];
}

function rawText(text: string): RuntimeValue {
  return { dtype: "raw_text", data: text, meta: { batch: [text] } };
}

function stringPieces(pieces: string[]): RuntimeValue {
  return {
    dtype: "string_piece",
    shape: { dtype: "string_piece", dims: [pieces.length], axes: ["T"] },
    data: pieces
  };
}

function scalar(value: number) {
  return makeTensor("float32", [], [], [value]);
}

function vector(values: number[]) {
  return makeTensor("float32", [values.length], ["C"], values);
}

function dotProduct(left: TinyTensor, right: TinyTensor) {
  if (left.dims.length !== 1 || right.dims.length !== 1 || left.dims[0] !== right.dims[0]) {
    throw new Error("dotProduct requires two same-length vector[C] tensors");
  }
  const sum = left.data.reduce((total, value, index) => total + value * right.data[index], 0);
  return makeTensor("float32", [], [], [Number(sum.toFixed(6))]);
}

function matrixFromColumns(...columns: TinyTensor[]) {
  if (!columns.length) throw new Error("matrixFromColumns requires at least one column");
  const c = columns[0].dims[0];
  const data: number[] = [];
  for (let row = 0; row < c; row += 1) {
    columns.forEach((column) => data.push(column.data[row]));
  }
  return makeTensor("float32", [c, columns.length], ["C", "O"], data);
}

function tensorFromTokens(...tokens: TinyTensor[]) {
  if (!tokens.length) throw new Error("tensorFromTokens requires at least one token");
  const c = tokens[0].dims[0];
  return makeTensor("float32", [1, tokens.length, c], ["B", "T", "C"], tokens.flatMap((item) => item.data));
}

function linear(hidden: TinyTensor, weight: TinyTensor, bias: TinyTensor) {
  const projected = matmul(hidden, weight);
  const expandedBias = broadcastTo(bias, projected.dims, projected.axes, ["O"]);
  return addTensors(projected, expandedBias);
}

function withNodeParams(graph: GraphSpec, paramsByNodeId: Record<string, Record<string, unknown>>): GraphSpec {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (paramsByNodeId[node.id] ? { ...node, params: { ...node.params, ...paramsByNodeId[node.id] } } : node))
  };
}

function controlNumber(values: Record<string, CertificationControlValue>, id: string, fallback: number) {
  const raw = values[id] ?? fallback;
  const parsed = typeof raw === "number" ? raw : Number(raw);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function controlInteger(values: Record<string, CertificationControlValue>, id: string, fallback: number) {
  return Math.max(1, Math.round(controlNumber(values, id, fallback)));
}

function controlString(values: Record<string, CertificationControlValue>, id: string, fallback: string) {
  const value = String(values[id] ?? fallback);
  return value.trim() ? value : fallback;
}

function runtimeValue(tensor: TinyTensor): RuntimeValue {
  return {
    dtype: tensor.dtype,
    shape: { dtype: tensor.dtype, dims: tensor.dims, axes: tensor.axes },
    data: tensor.data
  };
}

function textBatch(lines: string[]): RuntimeValue {
  return {
    dtype: "raw_text",
    data: lines
  };
}
