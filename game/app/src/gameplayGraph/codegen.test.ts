import { describe, expect, it } from "vitest";
import { generateCaseCodeLines, createTokenizerPreview } from "./codegen";
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
});
