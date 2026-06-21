import { describe, expect, it } from "vitest";
import { createTokenizerPreview, generateCaseCodeLines, generateGraphCodeSections } from "./codegen";
import { ch0MatMulGraph, createCh0MatMulSolutionGraph } from "./levels/ch0MatMulGraph";
import { ch1SplitMergeBudget, createCh1SplitMergeCharGraph, createCh1SplitMergeSolutionGraph } from "./levels/ch1TokenizerMachine";
import { createGameplayRegistry } from "./modules";

const registry = createGameplayRegistry();
const modules = registry.list();

describe("gameplay graph codegen", () => {
  it("defines Split / Merge Budget as a case-first tokenizer repair", () => {
    expect(ch1SplitMergeBudget.caseStudy?.visibleInputFocus).toBe("tokenizers are useful!");
    expect(ch1SplitMergeBudget.caseStudy?.dataPanels.map((panel) => panel.type)).toEqual(["text_batch", "tokenizer_preview"]);
  });

  it("previews the char split failure and the subword merge repair", () => {
    const charPreview = createTokenizerPreview(ch1SplitMergeBudget, createCh1SplitMergeCharGraph(), modules, ch1SplitMergeBudget.visibleTests[0], "tokenizer", "texts");
    const repairedPreview = createTokenizerPreview(ch1SplitMergeBudget, createCh1SplitMergeSolutionGraph(), modules, ch1SplitMergeBudget.visibleTests[0], "tokenizer", "texts");

    expect(charPreview?.withinBudget).toBe(false);
    expect(charPreview?.rawTokenCount).toBeGreaterThan(8);
    expect(charPreview?.pieces.slice(0, 3)).toEqual(["t", "o", "k"]);
    expect(repairedPreview?.withinBudget).toBe(true);
    expect(repairedPreview?.pieces).toContain("tokenizers");
    expect(repairedPreview?.pieces).toContain("useful");
  });

  it("emits only the visible case input code for the code panel", () => {
    const caseCode = generateCaseCodeLines(ch1SplitMergeBudget, ch1SplitMergeBudget.visibleTests[0]).map((line) => line.text).join("\n");

    expect(caseCode).toContain("texts = [");
    expect(caseCode).toContain('"tokenizers are useful!",  # focus case');
    expect(caseCode).toContain('focus_text = "tokenizers are useful!"');
    expect(caseCode).not.toContain("assert ");
    expect(caseCode).not.toContain("ToyTokenizer(");
  });

  it("emits tokenizer graph code as Python instead of overwriting the tokenizer object", () => {
    const graphCode = generateGraphCodeSections(ch1SplitMergeBudget, createCh1SplitMergeCharGraph(), modules, ch1SplitMergeBudget.visibleTests[0])
      .graphCode.map((line) => line.text)
      .join("\n");

    expect(graphCode).toContain("tokenizer = ToyTokenizer(");
    expect(graphCode).toContain("tokenizer_pieces = tokenizer.split_batch(text)");
    expect(graphCode).toContain("tokenizer_ids, tokenizer_attention_mask = tokenizer.encode_batch(text)");
    expect(graphCode).toContain("token_ids = embedding_ready(tokenizer_ids)");
    expect(graphCode).not.toContain("tokenizer, tokenizer_mask");
  });

  it("emits MatMul graph code as Python/PyTorch-style tensor operations", () => {
    const graphCode = generateGraphCodeSections(ch0MatMulGraph, createCh0MatMulSolutionGraph(), modules, ch0MatMulGraph.visibleTests[0])
      .graphCode.map((line) => line.text)
      .join("\n");

    expect(graphCode).toContain('hidden = Tensor("hidden", shape=[2, 4, 3], axes=["B", "T", "C"])');
    expect(graphCode).toContain('weight = Tensor("weight", shape=[5, 3], axes=["O", "C"])  # storage orientation: O,C');
    expect(graphCode).toContain("weight_transpose = weight.transpose(0, 1)");
    expect(graphCode).toContain("matmul = torch.matmul(hidden, weight_transpose)");
    expect(graphCode).not.toContain("WeightPlate(");
    expect(graphCode).not.toContain("MatMulGate(");
  });
});
