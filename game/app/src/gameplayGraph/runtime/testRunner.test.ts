import { describe, expect, it } from "vitest";
import { createGameplayRegistry } from "../modules";
import {
  ch0MatMulGraph,
  createCh0MatMulNoTransposeGraph,
  createCh0MatMulSolutionGraph,
  createCh0MatMulWrongOutputAxesGraph
} from "../levels/ch0MatMulGraph";
import {
  ch0TransposeGraph,
  createCh0TransposeNoKTransposeGraph,
  createCh0TransposeSolutionGraph,
  createCh0TransposeWrongOperandOrderGraph
} from "../levels/ch0TransposeGraph";
import {
  ch0BroadcastGraph,
  createCh0BroadcastSolutionGraph,
  createCh0BroadcastWithoutRailGraph,
  createCh0BroadcastWrongAxisGraph
} from "../levels/ch0BroadcastGraph";
import {
  ch0MaskGraph,
  createCh0MaskSolutionGraph,
  createCh0MaskWithoutBroadcastGraph,
  createCh0MaskWrongOrientationGraph
} from "../levels/ch0MaskGraph";
import {
  ch1OovFallback,
  ch1PaddingMask,
  ch1SplitMergeBudget,
  ch1TextTypeGate,
  ch1TokenizerMachine,
  createCh1OovFallbackSolutionGraph,
  createCh1OovNoFallbackGraph,
  createCh1PaddingMaskAllOnesGraph,
  createCh1PaddingMaskSolutionGraph,
  createCh1SplitMergeCharGraph,
  createCh1SplitMergeSolutionGraph,
  createCh1TextTypeSolutionGraph
} from "../levels/ch1TokenizerMachine";
import { runTests } from "./testRunner";

const registry = createGameplayRegistry();

describe("gameplay graph test runner", () => {
  it("passes Tokenizer Machine visible and hidden tests", () => {
    const visible = runTests(ch1TokenizerMachine.initialGraph, registry, ch1TokenizerMachine.visibleTests);
    const hidden = runTests(ch1TokenizerMachine.initialGraph, registry, ch1TokenizerMachine.hiddenTests);

    expect(visible.status).toBe("pass");
    expect(hidden.status).toBe("pass");
  });

  it("blocks Text Type Gate when raw text is wired directly into embedding probe", () => {
    const visible = runTests(ch1TextTypeGate.initialGraph, registry, ch1TextTypeGate.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("token_ids");
  });

  it("passes Text Type Gate after inserting tokenizer before embedding probe", () => {
    const graph = createCh1TextTypeSolutionGraph();
    const visible = runTests(graph, registry, ch1TextTypeGate.visibleTests);
    const hidden = runTests(graph, registry, ch1TextTypeGate.hiddenTests);

    expect(visible.status).toBe("pass");
    expect(hidden.status).toBe("pass");
  });

  it("catches Split/Merge Budget when char split destroys merged vocabulary pieces", () => {
    const graph = createCh1SplitMergeCharGraph();
    const visible = runTests(graph, registry, ch1SplitMergeBudget.visibleTests);

    expect(visible.status).toBe("fail");
    expect(visible.results.some((result) => result.assertion?.type === "tokens_include" && result.firstBadNodeId === "tokenizer")).toBe(true);
  });

  it("passes Split/Merge Budget with subword merges in visible and hidden sets", () => {
    const graph = createCh1SplitMergeSolutionGraph();
    const visible = runTests(graph, registry, ch1SplitMergeBudget.visibleTests);
    const hidden = runTests(graph, registry, ch1SplitMergeBudget.hiddenTests);

    expect(visible.status).toBe("pass");
    expect(hidden.status).toBe("pass");
  });

  it("blocks OOV Fallback when fallback is disabled for unseen text", () => {
    const graph = createCh1OovNoFallbackGraph();
    const hidden = runTests(graph, registry, ch1OovFallback.hiddenTests);

    expect(hidden.status).toBe("blocked");
    expect(hidden.results[0].firstBadNodeId).toBe("tokenizer");
  });

  it("passes OOV Fallback with stable unk fallback", () => {
    const graph = createCh1OovFallbackSolutionGraph();
    const visible = runTests(graph, registry, ch1OovFallback.visibleTests);
    const hidden = runTests(graph, registry, ch1OovFallback.hiddenTests);

    expect(visible.status).toBe("pass");
    expect(hidden.status).toBe("pass");
  });

  it("catches Padding/Mask when PAD positions are marked as valid", () => {
    const graph = createCh1PaddingMaskAllOnesGraph();
    const visible = runTests(graph, registry, ch1PaddingMask.visibleTests);

    expect(visible.status).toBe("fail");
    expect(visible.results.some((result) => result.assertion?.type === "mask_pad" && result.firstBadNodeId === "tokenizer.mask")).toBe(true);
  });

  it("passes Padding/Mask with right padding, EOS preservation, and pad-aware mask", () => {
    const graph = createCh1PaddingMaskSolutionGraph();
    const visible = runTests(graph, registry, ch1PaddingMask.visibleTests);
    const hidden = runTests(graph, registry, ch1PaddingMask.hiddenTests);

    expect(visible.status).toBe("pass");
    expect(hidden.status).toBe("pass");
  });

  it("blocks the empty MatMul challenge graph before player assembly", () => {
    const visible = runTests(ch0MatMulGraph.initialGraph, registry, ch0MatMulGraph.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("projected");
  });

  it("passes MatMul visible and hidden mutation tests with the complete solution graph", () => {
    const graph = createCh0MatMulSolutionGraph();
    const visible = runTests(graph, registry, ch0MatMulGraph.visibleTests);
    const hidden = runTests(graph, registry, ch0MatMulGraph.hiddenTests);

    expect(visible.status).toBe("pass");
    expect(hidden.status).toBe("pass");
    expect(hidden.cases).toHaveLength(3);
  });

  it("fails MatMul without the required stored-weight transpose", () => {
    const graph = createCh0MatMulNoTransposeGraph();
    const visible = runTests(graph, registry, ch0MatMulGraph.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("matmul");
    expect(visible.results[0].diagnostic?.suggestedProbe).toContain("Shape");
  });

  it("rejects a MatMul graph with the wrong output axis contract", () => {
    const graph = createCh0MatMulWrongOutputAxesGraph();
    const visible = runTests(graph, registry, ch0MatMulGraph.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("projected");
  });

  it("blocks the empty Transpose Trap graph before player assembly", () => {
    const visible = runTests(ch0TransposeGraph.initialGraph, registry, ch0TransposeGraph.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("score_board");
  });

  it("fails Transpose Trap before K transpose", () => {
    const graph = createCh0TransposeNoKTransposeGraph();
    const visible = runTests(graph, registry, ch0TransposeGraph.visibleTests);
    const hiddenEqualTrap = runTests(graph, registry, [ch0TransposeGraph.hiddenTests[0]]);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("qk_matmul");
    expect(hiddenEqualTrap.status).toBe("blocked");
    expect(hiddenEqualTrap.results[0].firstBadNodeId).toBe("score_board");
  });

  it("passes Transpose Trap visible and hidden tests with the complete solution graph", () => {
    const graph = createCh0TransposeSolutionGraph();
    const visible = runTests(graph, registry, ch0TransposeGraph.visibleTests);
    const hidden = runTests(graph, registry, ch0TransposeGraph.hiddenTests);

    expect(visible.status).toBe("pass");
    expect(hidden.status).toBe("pass");
    expect(hidden.cases).toHaveLength(3);
  });

  it("rejects Transpose Trap when K.transpose and Q are wired in the wrong operand order", () => {
    const graph = createCh0TransposeWrongOperandOrderGraph();
    const visible = runTests(graph, registry, ch0TransposeGraph.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("score_board");
  });

  it("blocks the empty Broadcast Add graph before player assembly", () => {
    const visible = runTests(ch0BroadcastGraph.initialGraph, registry, ch0BroadcastGraph.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("axis_ruler");
  });

  it("passes Broadcast Add visible and hidden semantic-axis tests with the complete solution graph", () => {
    const graph = createCh0BroadcastSolutionGraph();
    const visible = runTests(graph, registry, ch0BroadcastGraph.visibleTests);
    const hidden = runTests(graph, registry, ch0BroadcastGraph.hiddenTests);

    expect(visible.status).toBe("pass");
    expect(hidden.status).toBe("pass");
    expect(hidden.cases).toHaveLength(3);
  });

  it("catches Broadcast Add when bias is aligned to the token axis", () => {
    const graph = createCh0BroadcastWrongAxisGraph();
    const visible = runTests(graph, registry, ch0BroadcastGraph.visibleTests);
    const hiddenEqualTrap = runTests(graph, registry, [ch0BroadcastGraph.hiddenTests[1]]);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("broadcast");
    expect(hiddenEqualTrap.status).toBe("fail");
    expect(hiddenEqualTrap.results[hiddenEqualTrap.results.length - 1]?.firstBadNodeId).toBe("biased");
  });

  it("rejects Broadcast Add when bias is added without BroadcastRail", () => {
    const graph = createCh0BroadcastWithoutRailGraph();
    const visible = runTests(graph, registry, ch0BroadcastGraph.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("biased");
  });

  it("blocks the empty Mini Mask graph before player assembly", () => {
    const visible = runTests(ch0MaskGraph.initialGraph, registry, ch0MaskGraph.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("score_board");
  });

  it("passes Mini Mask visible and hidden orientation tests with the complete solution graph", () => {
    const graph = createCh0MaskSolutionGraph();
    const visible = runTests(graph, registry, ch0MaskGraph.visibleTests);
    const hidden = runTests(graph, registry, ch0MaskGraph.hiddenTests);

    expect(visible.status).toBe("pass");
    expect(hidden.status).toBe("pass");
    expect(hidden.cases).toHaveLength(2);
  });

  it("catches Mini Mask when causal orientation is reversed", () => {
    const graph = createCh0MaskWrongOrientationGraph();
    const visible = runTests(graph, registry, ch0MaskGraph.visibleTests);
    const hidden = runTests(graph, registry, [ch0MaskGraph.hiddenTests[0]]);

    expect(visible.status).toBe("fail");
    expect(visible.results.some((result) => result.assertion?.type === "future_attention_zero" && result.firstBadNodeId === "masked_scores")).toBe(true);
    expect(hidden.status).toBe("fail");
    expect(hidden.results.some((result) => result.assertion?.type === "future_attention_zero" && result.firstBadNodeId === "masked_scores")).toBe(true);
  });

  it("rejects Mini Mask when mask is added without BroadcastRail", () => {
    const graph = createCh0MaskWithoutBroadcastGraph();
    const visible = runTests(graph, registry, ch0MaskGraph.visibleTests);

    expect(visible.status).toBe("blocked");
    expect(visible.results[0].firstBadNodeId).toBe("masked_scores");
  });
});
