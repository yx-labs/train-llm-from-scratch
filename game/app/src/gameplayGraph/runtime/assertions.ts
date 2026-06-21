import type { AxisName, GraphSpec, GraphExecutionResult, RuntimeValue, TestAssertion, TestResult, TestVisibility } from "../types";
import { valueKey } from "../types";
import { allclose, makeTensor, sameAxes, sameDims, type TinyTensor } from "./tinyTensor";

export function evaluateAssertion(
  assertion: TestAssertion,
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  graph?: GraphSpec
): TestResult {
  if (assertion.type === "requires_node") {
    return assertRequiresNode(assertion.nodeId, assertion.moduleId, graph, visibility, id, assertion);
  }
  if (assertion.type === "requires_edge_path") {
    return assertRequiresEdgePath(assertion.from, assertion.through, assertion.to, graph, visibility, id, assertion);
  }

  if (execution.error) {
    return {
      id,
      visibility,
      status: "blocked",
      message: execution.error.message,
      firstBadNodeId: execution.error.nodeId,
      traceFrame: execution.trace.find((frame) => frame.error)?.step,
      assertion,
      diagnostic: {
        errorType: execution.error.type,
        expected: execution.error.expected,
        received: execution.error.received,
        possibleCause: possibleCauseForRuntimeError(execution.error.type),
        suggestedProbe: execution.error.suggestedProbe ?? suggestedProbeForRuntimeError(execution.error.type),
        sample: execution.trace.find((frame) => frame.error)?.samples
      }
    };
  }

  switch (assertion.type) {
    case "dtype":
      return assertDType(assertion.nodeId, assertion.expected, execution, visibility, id, assertion);
    case "shape":
      return assertShape(assertion.nodeId, assertion.expectedAxes, assertion.expectedDims, execution, visibility, id, assertion);
    case "axis_semantics":
      return assertShape(assertion.nodeId, assertion.expectedAxes, undefined, execution, visibility, id, assertion);
    case "allclose":
      return assertAllclose(assertion.nodeId, assertion.referenceNodeId, assertion.atol, execution, visibility, id, assertion);
    case "pieces_non_empty":
      return assertPiecesNonEmpty(assertion.nodeId, execution, visibility, id, assertion);
    case "pieces_equal":
      return assertPiecesEqual(assertion.nodeId, assertion.expected, execution, visibility, id, assertion);
    case "no_oov":
      return assertNoOov(assertion.nodeId, execution, visibility, id, assertion);
    case "tokens_include":
      return assertTokensInclude(assertion.nodeId, assertion.token, execution, visibility, id, assertion);
    case "eos_preserved":
      return assertEosPreserved(assertion.nodeId, assertion.eosToken ?? "<eos>", assertion.padToken ?? "<pad>", execution, visibility, id, assertion);
    case "token_budget":
      return assertTokenBudget(assertion.nodeId, assertion.maxT, execution, visibility, id, assertion);
    case "mask_pad":
      return assertMaskPad(assertion.idsNodeId, assertion.maskNodeId, assertion.padId, execution, visibility, id, assertion);
    case "future_attention_zero":
      return assertFutureAttentionZero(assertion.nodeId, assertion.threshold, execution, visibility, id, assertion);
    default:
      return {
        id,
        visibility,
        status: "blocked",
        message: `Assertion ${(assertion as TestAssertion).type} is not implemented yet`,
        assertion
      };
  }
}

function assertRequiresNode(
  nodeId: string,
  moduleId: string | undefined,
  graph: GraphSpec | undefined,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const node = graph?.nodes.find((item) => item.id === nodeId);
  const passed = Boolean(node) && (!moduleId || node?.moduleId === moduleId);
  return {
    id,
    visibility,
    status: passed ? "pass" : "fail",
    message: passed ? `required node ${nodeId} is present` : `Required node ${nodeId}${moduleId ? `:${moduleId}` : ""} is missing`,
    firstBadNodeId: passed ? undefined : nodeId,
    assertion,
    diagnostic: passed
      ? undefined
      : {
          errorType: "assertion_failed",
          expected: { nodeId, moduleId },
          received: node ? { nodeId: node.id, moduleId: node.moduleId } : "missing",
          possibleCause: "The graph bypasses the required implementation node.",
          suggestedProbe: "Use the required primitive/component instead of wiring inputs directly to the contract."
        }
  };
}

function assertRequiresEdgePath(
  from: string,
  through: string,
  to: string,
  graph: GraphSpec | undefined,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const passed = Boolean(graph && hasDirectedPath(graph, from, through) && hasDirectedPath(graph, through, to));
  return {
    id,
    visibility,
    status: passed ? "pass" : "fail",
    message: passed ? `path ${from} -> ${through} -> ${to} exists` : `Missing required path ${from} -> ${through} -> ${to}`,
    firstBadNodeId: passed ? undefined : through,
    assertion,
    diagnostic: passed
      ? undefined
      : {
          errorType: "assertion_failed",
          expected: { from, through, to },
          received: graph?.edges.map((edge) => `${edge.from.nodeId}->${edge.to.nodeId}`) ?? "missing graph",
          possibleCause: "The output contract can receive data without the intended implementation graph.",
          suggestedProbe: "Trace the data path and make sure it flows through the required component or primitive."
        }
  };
}

function hasDirectedPath(graph: GraphSpec, from: string, to: string) {
  if (from === to) return true;
  const seen = new Set<string>();
  const queue = [from];
  while (queue.length) {
    const current = queue.shift()!;
    if (seen.has(current)) continue;
    seen.add(current);
    const nextNodes = graph.edges.filter((edge) => edge.from.nodeId === current).map((edge) => edge.to.nodeId);
    if (nextNodes.includes(to)) return true;
    queue.push(...nextNodes.filter((nodeId) => !seen.has(nodeId)));
  }
  return false;
}

function assertPiecesNonEmpty(
  nodeId: string,
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const value = outputValue(execution, nodeId);
  const pieces = Array.isArray(value?.data) ? value.data : [];
  const passed = pieces.length > 0;
  return {
    id,
    visibility,
    status: passed ? "pass" : "fail",
    message: passed ? `pieces emitted: ${pieces.length}` : "Tokenizer emitted no pieces",
    firstBadNodeId: passed ? undefined : nodeId,
    assertion,
    diagnostic: passed
      ? undefined
      : {
          errorType: "assertion_failed",
          expected: "one or more token pieces",
          received: pieces,
          possibleCause: "Text did not reach TokenizerSocket or the split policy dropped all content.",
          suggestedProbe: "Inspect TokenizerSocket pieces output and preservePunctuation policy."
        }
  };
}

function assertPiecesEqual(
  nodeId: string,
  expected: string[],
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const value = outputValue(execution, nodeId);
  if (!value) return missingValue(id, visibility, nodeId, assertion);
  const pieces = Array.isArray(value.data) ? value.data : [];
  const passed = pieces.length === expected.length && pieces.every((piece, index) => piece === expected[index]);
  return {
    id,
    visibility,
    status: passed ? "pass" : "fail",
    message: passed ? `pieces matched: ${expected.join(" | ")}` : "Pieces differ from the expected split",
    firstBadNodeId: passed ? undefined : nodeId,
    assertion,
    diagnostic: passed
      ? undefined
      : {
          errorType: "assertion_failed",
          expected,
          received: pieces,
          possibleCause: "The boundary split policy kept, dropped, or merged a text piece incorrectly.",
          suggestedProbe: "Inspect BoundarySplitter output pieces and punctuation settings."
        }
  };
}

function assertNoOov(
  nodeId: string,
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const value = outputValue(execution, nodeId);
  if (!value) return missingValue(id, visibility, nodeId, assertion);
  const unresolved = Array.isArray(value.meta?.unresolved) ? value.meta.unresolved : [];
  const passed = unresolved.length === 0;
  return {
    id,
    visibility,
    status: passed ? "pass" : "fail",
    message: passed ? "all token pieces resolved by vocab or fallback" : `unresolved OOV pieces: ${unresolved.join(", ")}`,
    firstBadNodeId: passed ? undefined : nodeId,
    assertion,
    diagnostic: passed
      ? undefined
      : {
          errorType: "assertion_failed",
          expected: "no unresolved OOV pieces",
          received: unresolved,
          possibleCause: "Fallback is disabled or tokenizer pieces do not map to vocab rows.",
          suggestedProbe: "Use fallback=unk or fallback=char, then inspect token IDs."
        }
  };
}

function assertTokensInclude(
  nodeId: string,
  token: string,
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const value = outputValue(execution, nodeId);
  if (!value) return missingValue(id, visibility, nodeId, assertion);
  const tokens = flattenTokens(value.meta?.tokens);
  const passed = tokens.includes(token);
  return {
    id,
    visibility,
    status: passed ? "pass" : "fail",
    message: passed ? `token ${token} present` : `token ${token} missing`,
    firstBadNodeId: passed ? undefined : nodeId,
    assertion,
    diagnostic: passed
      ? undefined
      : {
          errorType: "assertion_failed",
          expected: token,
          received: tokens.slice(0, 24),
          possibleCause: "Split, merge, punctuation, or fallback policy produced the wrong token surface.",
          suggestedProbe: "Inspect TokenizerSocket Values for pieces and tokens."
        }
  };
}

function assertEosPreserved(
  nodeId: string,
  eosToken: string,
  padToken: string,
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const value = outputValue(execution, nodeId);
  if (!value) return missingValue(id, visibility, nodeId, assertion);
  const tokenRows = tokenRowsFromMeta(value.meta?.tokens);
  const failedRow = tokenRows.find((row) => {
    const content = row.filter((token) => token !== padToken);
    return content.length > 0 && content[content.length - 1] !== eosToken;
  });
  const passed = tokenRows.length > 0 && !failedRow;
  return {
    id,
    visibility,
    status: passed ? "pass" : "fail",
    message: passed ? `${eosToken} preserved at the end of each non-pad sequence` : `${eosToken} missing before padding`,
    firstBadNodeId: passed ? undefined : nodeId,
    assertion,
    diagnostic: passed
      ? undefined
      : {
          errorType: "assertion_failed",
          expected: `last non-pad token is ${eosToken}`,
          received: failedRow ?? tokenRows,
          possibleCause: "Truncation removed EOS or special token injection is disabled.",
          suggestedProbe: "Inspect addEos, maxLength, and truncation settings."
        }
  };
}

function assertFutureAttentionZero(
  nodeId: string,
  threshold: number,
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const value = tensorFromValue(outputValue(execution, nodeId));
  if (!value) return missingValue(id, visibility, nodeId, assertion);
  if (value.dims.length !== 4) {
    return {
      id,
      visibility,
      status: "fail",
      message: `Expected rank-4 attention scores, received rank ${value.dims.length}`,
      firstBadNodeId: nodeId,
      assertion,
      diagnostic: {
        errorType: "assertion_failed",
        expected: "rank-4 [B,H,T,T] masked scores",
        received: value.dims,
        possibleCause: "Mask was applied to the wrong tensor or score board contract was lost.",
        suggestedProbe: "Inspect ScoreBoard and BroadcastRail output shapes."
      }
    };
  }

  const [batch, heads, querySize, keySize] = value.dims;
  for (let b = 0; b < batch; b += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let query = 0; query < querySize; query += 1) {
        for (let key = query + 1; key < keySize; key += 1) {
          const flat = ((b * heads + h) * querySize + query) * keySize + key;
          const received = value.data[flat];
          if (received > threshold) {
            return {
              id,
              visibility,
              status: "fail",
              message: `Future key was not masked at [b=${b},h=${h},query=${query},key=${key}]: ${received} > ${threshold}`,
              firstBadNodeId: nodeId,
              assertion,
              diagnostic: {
                errorType: "assertion_failed",
                expected: `future key cells <= ${threshold}`,
                received: { b, h, query, key, value: received },
                possibleCause: "Causal mask direction is likely reversed: query/key axes are swapped.",
                suggestedProbe: "Inspect CausalMask orientation and CellTrace for a future key cell."
              }
            };
          }
        }
      }
    }
  }

  return {
    id,
    visibility,
    status: "pass",
    message: `future key cells are masked below threshold ${threshold}`,
    assertion
  };
}

function assertDType(
  nodeId: string,
  expected: RuntimeValue["dtype"],
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const value = outputValue(execution, nodeId);
  if (!value) return missingValue(id, visibility, nodeId, assertion);
  const passed = value.dtype === expected;
  return {
    id,
    visibility,
    status: passed ? "pass" : "fail",
    message: passed ? `dtype ${expected} matched` : `Expected dtype ${expected}, received ${value.dtype}`,
    firstBadNodeId: passed ? undefined : nodeId,
    assertion,
    diagnostic: passed
      ? undefined
      : {
          errorType: "assertion_failed",
          expected,
          received: value.dtype,
          possibleCause: "The node emitted a value with the wrong runtime dtype.",
          suggestedProbe: "Inspect the source module and the input port contract."
        }
  };
}

function assertShape(
  nodeId: string,
  expectedAxes: AxisName[],
  expectedDims: number[] | undefined,
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const value = outputValue(execution, nodeId);
  if (!value?.shape) return missingValue(id, visibility, nodeId, assertion);
  const dimsOk = expectedDims ? sameDims(value.shape.dims, expectedDims) : true;
  const axesOk = sameAxes(value.shape.axes, expectedAxes as never);
  const passed = dimsOk && axesOk;
  return {
    id,
    visibility,
    status: passed ? "pass" : "fail",
    message: passed
      ? `shape [${value.shape.axes.join(",")}] [${value.shape.dims.join(",")}] matched`
      : `Expected axes [${expectedAxes.join(",")}]${expectedDims ? ` dims [${expectedDims.join(",")}]` : ""}, received axes [${value.shape.axes.join(",")}] dims [${value.shape.dims.join(",")}]`,
    firstBadNodeId: passed ? undefined : nodeId,
    assertion,
    diagnostic: passed
      ? undefined
      : {
          errorType: "assertion_failed",
          expected: { axes: expectedAxes, dims: expectedDims },
          received: { axes: value.shape.axes, dims: value.shape.dims },
          possibleCause: "A port is connected to a tensor with the wrong axis order or dimensions.",
          suggestedProbe: "Inspect the failing node shape and the incoming edge contracts."
        }
  };
}

function assertAllclose(
  nodeId: string,
  referenceNodeId: string,
  atol: number,
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const value = tensorFromValue(outputValue(execution, nodeId));
  const reference = tensorFromValue(outputValue(execution, referenceNodeId));
  if (!value || !reference) return missingValue(id, visibility, nodeId, assertion);
  const result = allclose(value, reference, atol);
  return {
    id,
    visibility,
    status: result.ok ? "pass" : "fail",
    message: result.ok ? `allclose passed, max_abs_error=${result.maxAbsError}` : `allclose failed, max_abs_error=${result.maxAbsError}`,
    firstBadNodeId: result.ok ? undefined : nodeId,
    assertion,
    diagnostic: result.ok
      ? undefined
      : {
          errorType: "assertion_failed",
          expected: { referenceNodeId, atol },
          received: { maxAbsError: result.maxAbsError },
          possibleCause: "Shape is plausible, but numeric values differ from the reference.",
          suggestedProbe: "Inspect orientation, transpose settings, or operand order."
        }
  };
}

function assertTokenBudget(
  nodeId: string,
  maxT: number,
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const value = outputValue(execution, nodeId);
  const t = value?.shape?.dims[value.shape.axes.indexOf("T")];
  const passed = t !== undefined && t <= maxT;
  return {
    id,
    visibility,
    status: passed ? "pass" : "fail",
    message: passed ? `T=${t} within maxT=${maxT}` : `Token budget exceeded or T missing: T=${t ?? "unknown"}, maxT=${maxT}`,
    firstBadNodeId: passed ? undefined : nodeId,
    assertion,
    diagnostic: passed
      ? undefined
      : {
          errorType: "assertion_failed",
          expected: { maxT },
          received: { T: t ?? "unknown" },
          possibleCause: "Tokenizer params created too many tokens for the model budget.",
          suggestedProbe: "Inspect policy, merge, fallback, maxLength, and padToLength."
        }
  };
}

function assertMaskPad(
  idsNodeId: string,
  maskNodeId: string,
  padId: number,
  execution: GraphExecutionResult,
  visibility: TestVisibility,
  id: string,
  assertion: TestAssertion
): TestResult {
  const ids = outputValue(execution, idsNodeId);
  const mask = outputValue(execution, maskNodeId);
  if (!ids?.data || !mask?.data || !ids.shape || !mask.shape) return missingValue(id, visibility, idsNodeId, assertion);
  const idsData = ids.data as number[];
  const maskData = mask.data as number[];
  const sameShape = sameDims(ids.shape.dims, mask.shape.dims);
  const matches = sameShape && idsData.every((tokenId, index) => maskData[index] === (tokenId === padId ? 0 : 1));
  return {
    id,
    visibility,
    status: matches ? "pass" : "fail",
    message: matches ? "attention_mask matches PAD positions" : "attention_mask mismatch: PAD positions should be 0 and content positions should be 1",
    firstBadNodeId: matches ? undefined : maskNodeId,
    assertion,
    diagnostic: matches
      ? undefined
      : {
          errorType: "assertion_failed",
          expected: "mask=0 for PAD and mask=1 for content",
          received: { idsShape: ids.shape, maskShape: mask.shape },
          possibleCause: "Padding and attention mask policy are inconsistent.",
          suggestedProbe: "Inspect token ids and mask samples together."
        }
  };
}

function tokenRowsFromMeta(tokens: unknown) {
  if (!Array.isArray(tokens)) return [];
  if (tokens.every((item) => typeof item === "string")) return [tokens as string[]];
  return tokens.filter((row): row is string[] => Array.isArray(row) && row.every((item) => typeof item === "string"));
}

function flattenTokens(tokens: unknown) {
  return tokenRowsFromMeta(tokens).flat();
}

function outputValue(execution: GraphExecutionResult, nodeId: string) {
  const [realNodeId, portId = "out"] = nodeId.split(".");
  return execution.values[valueKey(realNodeId, portId)];
}

function tensorFromValue(value: RuntimeValue | undefined): TinyTensor | undefined {
  if (!value?.shape || !Array.isArray(value.data) || value.dtype !== "float32") return undefined;
  return makeTensor("float32", value.shape.dims, value.shape.axes, value.data as number[]);
}

function missingValue(id: string, visibility: TestVisibility, nodeId: string, assertion: TestAssertion): TestResult {
  return {
    id,
    visibility,
    status: "blocked",
    message: `Missing output for node ${nodeId}`,
    firstBadNodeId: nodeId,
    assertion,
    diagnostic: {
      errorType: "missing_input",
      expected: "node output value",
      received: "missing",
      possibleCause: "The graph did not execute far enough or the required output port is disconnected.",
      suggestedProbe: "Inspect incoming edges and required input ports for the node."
    }
  };
}

function possibleCauseForRuntimeError(type: string) {
  switch (type) {
    case "dtype_mismatch":
      return "An edge connects incompatible runtime value types.";
    case "shape_mismatch":
      return "A tensor axis or inner dimension contract is violated.";
    case "axis_semantic_error":
      return "The tensor shape may exist, but the axis semantics are wrong.";
    case "numeric_mismatch":
      return "The numeric result differs from the reference even if shape is plausible.";
    case "budget_exceeded":
      return "Tokenizer output exceeds the allowed sequence budget.";
    case "oov_unresolved":
      return "Text contains pieces that the current tokenizer strategy cannot encode.";
    case "mask_error":
      return "Mask values or orientation do not match the token or score board contract.";
    case "missing_input":
      return "A required input port is not connected or did not produce a value.";
    case "graph_structure":
      return "The graph topology or port references are invalid.";
    default:
      return "The runtime stopped before assertions could be evaluated.";
  }
}

function suggestedProbeForRuntimeError(type: string) {
  switch (type) {
    case "shape_mismatch":
      return "Open Shape and compare each input axis at the failing node.";
    case "axis_semantic_error":
      return "Inspect axis labels and transpose parameters.";
    case "dtype_mismatch":
      return "Check the source output dtype and target input accepts list.";
    case "oov_unresolved":
      return "Inspect fallback policy and token pieces.";
    case "missing_input":
      return "Trace the missing port backward to the nearest disconnected edge.";
    default:
      return "Step through the trace and inspect the first failing node.";
  }
}
