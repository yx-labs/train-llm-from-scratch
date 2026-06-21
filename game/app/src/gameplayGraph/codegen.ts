import { encodeToyBatch, encodeToyText, type TokenizerPolicy, type ToyTokenizerOptions } from "../toyTokenizer";
import type { CodeLine, GraphEdge, GraphNode, GraphSpec, LevelSpec, ModuleDef, RuntimeValue, TestAssertion, TestCase } from "./types";

export type TokenizerCasePreview = {
  focusText: string;
  policy: string;
  applyMerges: boolean;
  fallback: string;
  pieces: string[];
  tokens: string[];
  ids: number[];
  mask: number[];
  rawTokenCount: number;
  maxBudget: number;
  withinBudget: boolean;
  truncated: boolean;
  unresolved: string[];
};

export type GraphCodeSections = {
  caseCode: CodeLine[];
  graphCode: CodeLine[];
  testCode: CodeLine[];
};

export function createTokenizerPreview(
  level: LevelSpec,
  graph: GraphSpec,
  modules: ModuleDef[],
  testCase: TestCase | undefined,
  tokenizerNodeId: string,
  textInputKey: string,
  focusText?: string
): TokenizerCasePreview | undefined {
  const tokenizerNode = graph.nodes.find((node) => node.id === tokenizerNodeId);
  if (!tokenizerNode) return undefined;
  const tokenizerModule = modules.find((module) => module.id === tokenizerNode.moduleId);
  const params = { ...(tokenizerModule?.defaultParams ?? {}), ...tokenizerNode.params };
  const texts = getCaseTexts(testCase, textInputKey);
  const selectedText = focusText ?? level.caseStudy?.visibleInputFocus ?? texts[0];
  if (!selectedText) return undefined;

  const options = tokenizerOptionsFromParams(params);
  const raw = encodeToyText(selectedText, { ...options, maxLength: undefined, padToLength: undefined });
  const batch = encodeToyBatch([selectedText], options);
  const maxBudget = getTokenBudget(testCase, Number(params.maxLength ?? 8));
  const specialCount = (options.addBos ? 1 : 0) + (options.addEos ? 1 : 0);
  const rawTokenCount = raw.pieces.length + specialCount;

  return {
    focusText: selectedText,
    policy: String(params.policy ?? "subword"),
    applyMerges: Boolean(params.applyMerges ?? true),
    fallback: String(params.fallback ?? "unk"),
    pieces: raw.pieces,
    tokens: batch.tokens[0] ?? [],
    ids: batch.tokenIds[0] ?? [],
    mask: batch.attentionMask[0] ?? [],
    rawTokenCount,
    maxBudget,
    withinBudget: rawTokenCount <= maxBudget,
    truncated: batch.truncated,
    unresolved: batch.unresolved
  };
}

export function generateGraphCodeSections(level: LevelSpec, graph: GraphSpec, modules: ModuleDef[], testCase?: TestCase): GraphCodeSections {
  const activeTest = testCase ?? level.visibleTests[0];
  return {
    caseCode: buildCaseCode(level, activeTest, graph),
    graphCode: buildGraphCode(graph, modules, activeTest),
    testCode: buildTestCode(activeTest)
  };
}

export function generateCaseCodeLines(level: LevelSpec, testCase?: TestCase, graph?: GraphSpec): CodeLine[] {
  return buildCaseCode(level, testCase ?? level.visibleTests[0], graph);
}

export function getCaseTexts(testCase: TestCase | undefined, inputKey: string) {
  return extractTexts(testCase?.inputs[inputKey]);
}

function buildCaseCode(level: LevelSpec, testCase: TestCase | undefined, graph?: GraphSpec): CodeLine[] {
  if (!testCase) return [{ id: "case-empty", text: "# No visible task input selected." }];
  if (level.id.startsWith("mvp01_")) return buildMvp01CaseCode(level, testCase, graph);

  const textPanel = level.caseStudy?.dataPanels.find((panel) => panel.type === "text_batch");
  if (textPanel?.type === "text_batch") {
    const texts = getCaseTexts(testCase, textPanel.inputKey);
    const focusText = textPanel.focusText ?? level.caseStudy?.visibleInputFocus;
    const lines: CodeLine[] = [
      { id: "case-texts-open", text: `${textPanel.inputKey} = [` },
      ...texts.map((text, index) => ({
        id: `case-text-${index}`,
        text: `    ${toPythonLiteral(text)},${focusText === text ? "  # focus task" : ""}`
      })),
      { id: "case-texts-close", text: "]" }
    ];
    if (focusText) lines.push({ id: "case-focus", text: `focus_text = ${toPythonLiteral(focusText)}` });
    return lines;
  }

  const lines: CodeLine[] = [];
  Object.entries(testCase.inputs).forEach(([key, value]) => {
    if (value.shape) {
      lines.push({ id: `case-shape-${key}`, text: `${safeVar(key)} = Tensor(shape=${toPythonLiteral(value.shape.dims)}, axes=${toPythonLiteral(value.shape.axes)})` });
      return;
    }
    lines.push({ id: `case-input-${key}`, text: `${safeVar(key)} = ${toPythonLiteral(value.data ?? value.meta ?? "runtime input")}` });
  });
  return lines.length ? lines : [{ id: "case-empty-inputs", text: "# This task has no explicit inputs." }];
}

function buildMvp01CaseCode(level: LevelSpec, testCase: TestCase, graph: GraphSpec | undefined): CodeLine[] {
  if (level.id === "mvp01_1_scalar_cell") {
    const value = graphParam(graph, "scalar_source", "value") ?? tensorFirstValue(testCase.inputs.reference) ?? 0.5;
    return [tensorCaseLine("case-scalar-starting-value", "starting_value", [], [], [value])];
  }

  if (level.id === "mvp01_2_vector_rail") {
    const referenceValues = Array.isArray(testCase.inputs.reference?.data) ? testCase.inputs.reference.data : [];
    return ["c0", "c1", "c2"].map((slot, index) => {
      const value = graphParam(graph, `scalar_${slot}`, "value") ?? referenceValues[index] ?? 0;
      return tensorCaseLine(`case-vector-${slot}`, slot, [], [], [value]);
    });
  }

  const inputLines = Object.entries(testCase.inputs)
    .filter(([key, value]) => key !== "case" && key !== "reference" && Boolean(value.shape))
    .map(([key, value]) => tensorCaseLine(`case-${key}`, key, value.shape?.dims ?? [], value.shape?.axes ?? [], previewTensorValues(value)));

  if (inputLines.length) return inputLines;

  const reference = testCase.inputs.reference;
  if (reference?.shape) {
    return [tensorCaseLine("case-reference", "starting_example", reference.shape.dims, reference.shape.axes, previewTensorValues(reference))];
  }

  return [{ id: "case-empty-inputs", text: "# This task has no explicit tensor inputs." }];
}

function tensorCaseLine(id: string, name: string, shape: number[], axes: string[], values?: unknown[]): CodeLine {
  const args = [`shape=${toPythonLiteral(shape)}`, `axes=${toPythonLiteral(axes)}`];
  if (values?.length) args.push(`values=${toPythonLiteral(values.map(normalizeCaseValue))}`);
  return { id, text: `${safeVar(name)} = Tensor(${args.join(", ")})` };
}

function previewTensorValues(value: RuntimeValue | undefined) {
  if (!Array.isArray(value?.data)) return undefined;
  if (value.data.length > 8) return undefined;
  return value.data;
}

function tensorFirstValue(value: RuntimeValue | undefined) {
  return Array.isArray(value?.data) ? value.data[0] : undefined;
}

function graphParam(graph: GraphSpec | undefined, nodeId: string, key: string) {
  return graph?.nodes.find((node) => node.id === nodeId)?.params[key];
}

function normalizeCaseValue(value: unknown) {
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  if (!trimmed) return value;
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed)) return value;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : value;
}

function buildGraphCode(graph: GraphSpec, modules: ModuleDef[], testCase: TestCase | undefined): CodeLine[] {
  const ordered = topologicalNodes(graph);
  const lines: CodeLine[] = [];
  ordered.forEach((node) => {
    const module = modules.find((item) => item.id === node.moduleId);
    if (!module) {
      lines.push({ id: `graph-${node.id}-unknown`, nodeId: node.id, text: `# Unknown module: ${node.moduleId}` });
      return;
    }

    if (node.moduleId === "TextInput") {
      const inputKey = String(node.params.inputKey ?? "texts");
      lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = ${safeVar(inputKey)}` });
      return;
    }

    if (node.moduleId === "TokenizerSocket") {
      const textVar = incomingVar(graph, node, "text");
      const params = { ...module.defaultParams, ...node.params };
      const tokenizerVar = safeVar(node.id);
      lines.push({ id: `graph-${node.id}-ctor`, nodeId: node.id, text: `${tokenizerVar} = ToyTokenizer(` });
      tokenizerParamLines(params).forEach((line, index) => lines.push({ id: `graph-${node.id}-param-${index}`, nodeId: node.id, text: line }));
      lines.push({ id: `graph-${node.id}-ctor-close`, nodeId: node.id, text: ")" });
      lines.push({ id: `graph-${node.id}-pieces`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "pieces")} = ${tokenizerVar}.split_batch(${textVar})` });
      lines.push({ id: `graph-${node.id}-encode`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")}, ${graphOutputVar(graph, node.id, "mask")} = ${tokenizerVar}.encode_batch(${textVar})` });
      return;
    }

    if (node.moduleId === "BoundarySplitter") {
      const textVar = incomingVar(graph, node, "text");
      const preservePunctuation = Boolean(node.params.preservePunctuation ?? true);
      lines.push({
        id: `graph-${node.id}`,
        nodeId: node.id,
        text: `${graphOutputVar(graph, node.id, "pieces")} = split_boundaries(${textVar}, keep_punctuation=${toPythonLiteral(preservePunctuation)})`
      });
      return;
    }

    if (node.moduleId === "PieceBuffer") {
      const maxPieces = Number(node.params.maxPieces ?? 12);
      lines.push({
        id: `graph-${node.id}`,
        nodeId: node.id,
        text: `${graphOutputVar(graph, node.id, "out")} = expect_piece_buffer(${incomingVar(graph, node, "pieces")}, max_pieces=${maxPieces})`
      });
      return;
    }

    if (node.moduleId === "EmbeddingReadyProbe") {
      const idsVar = incomingVar(graph, node, "ids");
      lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = embedding_ready(${idsVar})` });
      return;
    }

    if (node.moduleId === "InputTensor") {
      const inputKey = String(node.params.inputKey ?? node.id);
      const value = testCase?.inputs[inputKey];
      lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = ${tensorInputExpression(inputKey, value, node.params)}` });
      return;
    }

    if (node.moduleId === "WeightPlate") {
      const inputKey = String(node.params.inputKey ?? node.id);
      const orientation = String(node.params.orientation ?? "C,O");
      const value = testCase?.inputs[inputKey];
      lines.push({
        id: `graph-${node.id}`,
        nodeId: node.id,
        text: `${graphOutputVar(graph, node.id, "out")} = ${tensorInputExpression(inputKey, value, node.params)}  # storage orientation: ${orientation}`
      });
      return;
    }

    if (node.moduleId === "TransposeSwitch") {
      const xVar = incomingVar(graph, node, "x");
      const axisA = Number(node.params.axisA ?? -2);
      const axisB = Number(node.params.axisB ?? -1);
      lines.push({
        id: `graph-${node.id}`,
        nodeId: node.id,
        text: `${graphOutputVar(graph, node.id, "out")} = ${xVar}.transpose(${axisA}, ${axisB})`
      });
      return;
    }

    if (node.moduleId === "MatMulGate" || node.moduleId === "component.matmul_gate.v1") {
      lines.push({
        id: `graph-${node.id}`,
        nodeId: node.id,
        text: `${graphOutputVar(graph, node.id, "out")} = torch.matmul(${incomingVar(graph, node, "left")}, ${incomingVar(graph, node, "right")})`
      });
      return;
    }

    if (node.moduleId === "OutputContractGate") {
      const expectedAxes = Array.isArray(node.params.expectedAxes) ? node.params.expectedAxes : [];
      lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = expect_axes(${incomingVar(graph, node, "x")}, ${toPythonLiteral(expectedAxes)})` });
      return;
    }

    if (node.moduleId === "ReferenceChecker") {
      const referenceKey = String(node.params.referenceKey ?? "reference");
      lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = reference_tensor(${toPythonLiteral(referenceKey)})` });
      return;
    }

    if (node.moduleId === "AxisLock") {
      const expectedPrefixAxes = Array.isArray(node.params.expectedPrefixAxes) ? node.params.expectedPrefixAxes : [];
      lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = expect_prefix_axes(${incomingVar(graph, node, "x")}, ${toPythonLiteral(expectedPrefixAxes)})` });
      return;
    }

    if (node.moduleId === "AxisAlignmentRuler") {
      const expectedAxes = Array.isArray(node.params.expectedAxes) ? node.params.expectedAxes : [];
      lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = expect_axes(${incomingVar(graph, node, "x")}, ${toPythonLiteral(expectedAxes)})` });
      return;
    }

    if (node.moduleId === "BroadcastRail") {
      const alignAxes = Array.isArray(node.params.alignAxes) ? node.params.alignAxes : [];
      lines.push({
        id: `graph-${node.id}`,
        nodeId: node.id,
        text: `${graphOutputVar(graph, node.id, "out")} = broadcast_to(${incomingVar(graph, node, "small")}, like=${incomingVar(graph, node, "target")}, align_axes=${toPythonLiteral(alignAxes)})`
      });
      return;
    }

    if (node.moduleId === "GhostExpansionPreview" || node.moduleId === "SemanticWarningLens") {
      lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = ${incomingVar(graph, node, "x")}` });
      return;
    }

    if (node.moduleId === "AddGate") {
      lines.push({
        id: `graph-${node.id}`,
        nodeId: node.id,
        text: `${graphOutputVar(graph, node.id, "out")} = ${incomingVar(graph, node, "left")} + ${incomingVar(graph, node, "right")}`
      });
      return;
    }

    if (node.moduleId === "CausalMask") {
      lines.push({
        id: `graph-${node.id}`,
        nodeId: node.id,
        text: `${graphOutputVar(graph, node.id, "out")} = causal_mask_like(${incomingVar(graph, node, "target")}, orientation=${toPythonLiteral(node.params.maskOrientation ?? "query_key")}, masked_value=${toPythonLiteral(node.params.maskedValue ?? -10000)})`
      });
      return;
    }

    if (node.moduleId === "ScoreBoard") {
      const expectedAxes = Array.isArray(node.params.expectedAxes) ? node.params.expectedAxes : [];
      lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = expect_axes(${incomingVar(graph, node, "scores")}, ${toPythonLiteral(expectedAxes)})` });
      return;
    }

    if (node.moduleId === "CellTrace") {
      const inputs = ["scores", "q", "k", "left", "small"]
        .map((portId) => graph.edges.some((edge) => edge.to.nodeId === node.id && edge.to.portId === portId) ? `${portId}=${incomingVar(graph, node, portId)}` : undefined)
        .filter(Boolean)
        .join(", ");
      lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = trace_cell(${inputs})` });
      return;
    }

    const inputArgs = module.inputs.map((port) => `${port.id}=${incomingVar(graph, node, port.id)}`).join(", ");
    lines.push({ id: `graph-${node.id}`, nodeId: node.id, text: `${graphOutputVar(graph, node.id, "out")} = ${pythonCallName(node.moduleId)}(${inputArgs})` });
  });

  return lines.length ? lines : [{ id: "graph-empty", text: "# Empty graph." }];
}

function buildTestCode(testCase: TestCase | undefined): CodeLine[] {
  if (!testCase) return [{ id: "test-empty", text: "# Run a visible task input to see test code." }];
  return testCase.assertions.map((assertion, index) => ({ id: `test-${index}`, nodeId: assertionNodeId(assertion), text: assertionToCode(assertion) }));
}

function tokenizerParamLines(params: Record<string, unknown>) {
  const keys = ["policy", "applyMerges", "fallback", "maxLength", "padToLength", "padSide", "maskPolicy", "addBos", "addEos"];
  return keys.map((key) => `    ${toSnakeCase(key)}=${toPythonLiteral(params[key])},`);
}

function tokenizerOptionsFromParams(params: Record<string, unknown>): ToyTokenizerOptions {
  return {
    policy: stringParam(params, "policy", "subword") as TokenizerPolicy,
    applyMerges: boolParam(params, "applyMerges", true),
    fallback: stringParam(params, "fallback", "unk") as "none" | "char" | "unk",
    preservePunctuation: boolParam(params, "preservePunctuation", true),
    addBos: boolParam(params, "addBos", true),
    addEos: boolParam(params, "addEos", true),
    maxLength: numberParam(params, "maxLength", 8),
    padToLength: numberParam(params, "padToLength", numberParam(params, "maxLength", 8)),
    padSide: stringParam(params, "padSide", "right") as "left" | "right",
    maskPolicy: stringParam(params, "maskPolicy", "pad-aware") as "pad-aware" | "all-ones"
  };
}

function getTokenBudget(testCase: TestCase | undefined, fallback: number) {
  const assertion = testCase?.assertions.find((item): item is Extract<TestAssertion, { type: "token_budget" }> => item.type === "token_budget");
  return assertion?.maxT ?? fallback;
}

function incomingVar(graph: GraphSpec, node: GraphNode, portId: string) {
  const edge = graph.edges.find((item) => item.to.nodeId === node.id && item.to.portId === portId);
  if (!edge) return `${safeVar(node.id)}_${safeVar(portId)}_missing`;
  return graphOutputVar(graph, edge.from.nodeId, edge.from.portId);
}

function graphOutputVar(graph: GraphSpec, nodeId: string, portId: string) {
  const node = graph.nodes.find((item) => item.id === nodeId);
  if (node?.moduleId === "TokenizerSocket") {
    if (portId === "out") return `${safeVar(nodeId)}_ids`;
    if (portId === "mask") return `${safeVar(nodeId)}_attention_mask`;
    if (portId === "pieces") return `${safeVar(nodeId)}_pieces`;
  }
  if (portId === "out") return safeVar(nodeId);
  return `${safeVar(nodeId)}_${safeVar(portId)}`;
}

function tensorInputExpression(inputKey: string, value: RuntimeValue | undefined, params: Record<string, unknown>) {
  const shape = value?.shape?.dims ?? arrayParam(params.shape);
  const axes = value?.shape?.axes ?? arrayParam(params.axes);
  const args = [toPythonLiteral(inputKey)];
  if (shape.length) args.push(`shape=${toPythonLiteral(shape)}`);
  if (axes.length) args.push(`axes=${toPythonLiteral(axes)}`);
  return `Tensor(${args.join(", ")})`;
}

function arrayParam(value: unknown) {
  return Array.isArray(value) ? value : [];
}

function pythonCallName(moduleId: string) {
  return toSnakeCase(moduleId.replace(/Module$/, ""));
}

function topologicalNodes(graph: GraphSpec) {
  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));
  const incoming = new Map(graph.nodes.map((node) => [node.id, 0]));
  graph.edges.forEach((edge) => incoming.set(edge.to.nodeId, (incoming.get(edge.to.nodeId) ?? 0) + 1));

  const ready = graph.nodes.filter((node) => (incoming.get(node.id) ?? 0) === 0);
  const ordered: GraphNode[] = [];
  const used = new Set<string>();
  while (ready.length) {
    const node = ready.shift() as GraphNode;
    if (used.has(node.id)) continue;
    ordered.push(node);
    used.add(node.id);
    graph.edges
      .filter((edge) => edge.from.nodeId === node.id)
      .forEach((edge) => {
        const count = Math.max(0, (incoming.get(edge.to.nodeId) ?? 0) - 1);
        incoming.set(edge.to.nodeId, count);
        const next = nodesById.get(edge.to.nodeId);
        if (next && count === 0) ready.push(next);
      });
  }

  graph.nodes.forEach((node) => {
    if (!used.has(node.id)) ordered.push(node);
  });
  return ordered;
}

function assertionNodeId(assertion: TestAssertion) {
  switch (assertion.type) {
    case "allclose":
      return assertion.nodeId;
    case "mask_pad":
      return assertion.idsNodeId;
    default:
      return "nodeId" in assertion ? assertion.nodeId : undefined;
  }
}

function assertionToCode(assertion: TestAssertion) {
  switch (assertion.type) {
    case "dtype":
      return `assert dtype(${nodeRefToVar(assertion.nodeId)}) == ${toPythonLiteral(assertion.expected)}`;
    case "shape":
      return `assert shape(${nodeRefToVar(assertion.nodeId)}) == ${toPythonLiteral(assertion.expectedDims ?? assertion.expectedAxes)}`;
    case "axis_semantics":
      return `assert axes(${nodeRefToVar(assertion.nodeId)}) == ${toPythonLiteral(assertion.expectedAxes)}`;
    case "allclose":
      return `assert allclose(${nodeRefToVar(assertion.nodeId)}, ${nodeRefToVar(assertion.referenceNodeId)}, atol=${assertion.atol})`;
    case "requires_node":
      return `assert graph_has_node(${toPythonLiteral(assertion.nodeId)}${assertion.moduleId ? `, module=${toPythonLiteral(assertion.moduleId)}` : ""})`;
    case "requires_edge_path":
      return `assert path_exists(graph, ${toPythonLiteral(assertion.from)}, through=${toPythonLiteral(assertion.through)}, to=${toPythonLiteral(assertion.to)})`;
    case "pieces_non_empty":
      return `assert len(${nodeRefToVar(assertion.nodeId)}) > 0`;
    case "pieces_equal":
      return `assert ${nodeRefToVar(assertion.nodeId)} == ${toPythonLiteral(assertion.expected)}`;
    case "no_oov":
      return `assert no_oov(${nodeRefToVar(assertion.nodeId)})`;
    case "tokens_include":
      return `assert ${toPythonLiteral(assertion.token)} in tokens(${nodeRefToVar(assertion.nodeId)})`;
    case "eos_preserved":
      return `assert eos_preserved(${nodeRefToVar(assertion.nodeId)}, eos=${toPythonLiteral(assertion.eosToken ?? "<eos>")})`;
    case "token_budget":
      return `assert token_count(${nodeRefToVar(assertion.nodeId)}) <= ${assertion.maxT}`;
    case "mask_pad":
      return `assert mask_matches_pad(${nodeRefToVar(assertion.idsNodeId)}, ${nodeRefToVar(assertion.maskNodeId)}, pad_id=${assertion.padId})`;
    case "future_attention_zero":
      return `assert future_attention_zero(${nodeRefToVar(assertion.nodeId)}, threshold=${assertion.threshold})`;
    case "row_sum":
      return `assert row_sum(${nodeRefToVar(assertion.nodeId)}, dim=${toPythonLiteral(assertion.dim)}) ~= ${assertion.target}`;
    default:
      return `# Unsupported assertion ${JSON.stringify(assertion)}`;
  }
}

function nodeRefToVar(ref: string) {
  return safeVar(ref.replace(".", "_"));
}

function extractTexts(value: RuntimeValue | undefined) {
  if (!value) return [];
  if (typeof value.data === "string") return [value.data];
  if (Array.isArray(value.data) && value.data.every((item): item is string => typeof item === "string")) return value.data;
  const batch = value.meta?.batch;
  if (Array.isArray(batch) && batch.every((item): item is string => typeof item === "string")) return batch;
  return [];
}

function stringParam(params: Record<string, unknown>, key: string, fallback: string) {
  const value = params[key];
  return typeof value === "string" ? value : fallback;
}

function boolParam(params: Record<string, unknown>, key: string, fallback: boolean) {
  const value = params[key];
  return typeof value === "boolean" ? value : fallback;
}

function numberParam(params: Record<string, unknown>, key: string, fallback: number) {
  const value = params[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function safeVar(value: string) {
  const normalized = value.replace(/[^A-Za-z0-9_]/g, "_").replace(/^([0-9])/, "_$1");
  return normalized || "value";
}

function toSnakeCase(value: string) {
  return value.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`);
}

function toPythonLiteral(value: unknown): string {
  if (value === undefined) return "None";
  if (value === null) return "None";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (Array.isArray(value)) return `[${value.map((item) => toPythonLiteral(item)).join(", ")}]`;
  if (typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `${toPythonLiteral(key)}: ${toPythonLiteral(item)}`)
      .join(", ")}}`;
  }
  return JSON.stringify(String(value));
}
