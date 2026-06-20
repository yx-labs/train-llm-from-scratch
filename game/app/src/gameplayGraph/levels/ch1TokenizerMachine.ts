import type { GraphSpec, LevelSpec, RuntimeValue } from "../types";

type TokenizerParams = {
  policy: "char" | "word" | "subword";
  applyMerges: boolean;
  fallback: "none" | "char" | "unk";
  preservePunctuation: boolean;
  addBos: boolean;
  addEos: boolean;
  maxLength: number;
  padToLength: number;
  padSide: "left" | "right";
  maskPolicy: "pad-aware" | "all-ones";
};

const textTypeParams: TokenizerParams = {
  policy: "word",
  applyMerges: true,
  fallback: "unk",
  preservePunctuation: true,
  addBos: false,
  addEos: false,
  maxLength: 4,
  padToLength: 4,
  padSide: "right",
  maskPolicy: "pad-aware"
};

const splitMergeParams: TokenizerParams = {
  policy: "subword",
  applyMerges: true,
  fallback: "unk",
  preservePunctuation: true,
  addBos: false,
  addEos: false,
  maxLength: 8,
  padToLength: 8,
  padSide: "right",
  maskPolicy: "pad-aware"
};

const oovParams: TokenizerParams = {
  ...splitMergeParams,
  fallback: "unk"
};

const paddingParams: TokenizerParams = {
  policy: "subword",
  applyMerges: true,
  fallback: "unk",
  preservePunctuation: true,
  addBos: true,
  addEos: true,
  maxLength: 8,
  padToLength: 8,
  padSide: "right",
  maskPolicy: "pad-aware"
};

const textTypeVisible = rawTexts(["we train llm"]);
const splitMergeVisible = rawTexts(["we train llm.", "tokenizers are useful!", "shape tells truth"]);
const splitMergeHidden = rawTexts(["unknown glyph ?", "a loooooong tokenized sequence", "中文 mixed text!", "code_snake_case + punctuation"]);
const oovVisible = rawTexts(["we train llm!"]);
const oovHidden = rawTexts(["unseen glyph ?"]);
const paddingVisible = rawTexts(["we train llm", "shape"]);
const paddingHidden = rawTexts(["a very very long sequence that exceeds budget", "pad"]);
const gauntletVisible = rawTexts(["we train llm!", "shape, token buffer"]);
const gauntletHidden = rawTexts([
  "we train llm!",
  "unknown glyph ?",
  "shape, token buffer",
  "emoji 😀 test",
  "中文 mixed text",
  "a very very long sequence that exceeds budget"
]);

export const ch1TextTypeGate: LevelSpec = {
  id: "ch1_1_text_type_gate",
  title: "1-1 Text Type Gate",
  mode: "graph_challenge",
  chapter: "Chapter 1",
  goal: "Repair Raw Text -> Embedding Ready by inserting TokenizerSocket before integer token IDs.",
  modulePalette: ["TextInput", "TokenizerSocket", "EmbeddingReadyProbe"],
  initialGraph: {
    levelId: "ch1_1_text_type_gate",
    version: 1,
    nodes: [
      { id: "text", moduleId: "TextInput", params: { inputKey: "texts" }, position: { x: 90, y: 180 } },
      { id: "token_ids", moduleId: "EmbeddingReadyProbe", params: {}, position: { x: 420, y: 180 } }
    ],
    edges: [{ id: "e_raw_embedding", from: { nodeId: "text", portId: "out" }, to: { nodeId: "token_ids", portId: "ids" } }],
    outputNodes: ["token_ids"]
  },
  constraints: { maxNodes: 4, maxEdges: 4 },
  visibleTests: [
    {
      id: "raw_text_requires_token_ids",
      title: "raw text must become integer token IDs",
      visibility: "visible",
      inputSeed: "ch1-1-visible",
      inputs: { texts: textTypeVisible },
      assertions: [
        { type: "dtype", nodeId: "token_ids", expected: "int" },
        { type: "shape", nodeId: "token_ids", expectedAxes: ["B", "T"], expectedDims: [1, 4] },
        { type: "mask_pad", idsNodeId: "tokenizer", maskNodeId: "tokenizer.mask", padId: 0 }
      ]
    }
  ],
  hiddenTests: [
    {
      id: "embedding_contract_repeat",
      title: "embedding contract remains int IDs",
      visibility: "hidden",
      inputSeed: "ch1-1-hidden",
      inputs: { texts: rawTexts(["shape"]) },
      assertions: [
        { type: "dtype", nodeId: "token_ids", expected: "int" },
        { type: "shape", nodeId: "token_ids", expectedAxes: ["B", "T"], expectedDims: [1, 4] }
      ]
    }
  ],
  debrief: {
    completeTitle: "Text Type Gate Restored",
    fixedProblem: "Raw text no longer enters the embedding probe directly.",
    learned: "Embedding lookup consumes stable integer token IDs, not string objects.",
    nextUse: "Split and merge policy controls how many token IDs the model receives."
  }
};

export const ch1SplitMergeBudget: LevelSpec = {
  id: "ch1_2_split_merge_budget",
  title: "1-2 Split / Merge Budget",
  mode: "graph_challenge",
  chapter: "Chapter 1",
  goal: "Tune tokenizer policy and merges so useful pieces stay intact while T stays within budget.",
  modulePalette: ["TextInput", "TokenizerSocket", "EmbeddingReadyProbe"],
  initialGraph: createTokenizerGraph("ch1_2_split_merge_budget", { ...splitMergeParams, policy: "char", applyMerges: false }),
  constraints: { maxNodes: 5, maxEdges: 5 },
  visibleTests: [
    {
      id: "merge_visible_set",
      title: "visible text set preserves merged tokens",
      visibility: "visible",
      inputSeed: "ch1-2-visible",
      inputs: { texts: splitMergeVisible },
      assertions: [
        { type: "pieces_non_empty", nodeId: "tokenizer.pieces" },
        { type: "token_budget", nodeId: "token_ids", maxT: 8 },
        { type: "tokens_include", nodeId: "tokenizer", token: "tokenizers" },
        { type: "tokens_include", nodeId: "tokenizer", token: "useful" },
        { type: "no_oov", nodeId: "tokenizer" }
      ]
    }
  ],
  hiddenTests: [
    {
      id: "merge_hidden_set",
      title: "hidden text set stays deterministic and in budget",
      visibility: "hidden",
      inputSeed: "ch1-2-hidden",
      inputs: { texts: splitMergeHidden },
      assertions: [
        { type: "pieces_non_empty", nodeId: "tokenizer.pieces" },
        { type: "token_budget", nodeId: "token_ids", maxT: 8 },
        { type: "no_oov", nodeId: "tokenizer" }
      ]
    }
  ],
  debrief: {
    completeTitle: "Split / Merge Budget Restored",
    fixedProblem: "Tokenizer policy now keeps useful merged pieces while respecting max T.",
    learned: "Character split is robust but expensive; subword merges keep recognizable pieces under budget.",
    nextUse: "Unknown pieces still need a stable vocab fallback."
  }
};

export const ch1OovFallback: LevelSpec = {
  id: "ch1_3_oov_fallback",
  title: "1-3 Vocab / OOV / Fallback",
  mode: "graph_challenge",
  chapter: "Chapter 1",
  goal: "Resolve unknown token pieces with a stable fallback instead of leaving OOV holes.",
  modulePalette: ["TextInput", "TokenizerSocket", "EmbeddingReadyProbe"],
  initialGraph: createTokenizerGraph("ch1_3_oov_fallback", { ...oovParams, fallback: "none" }),
  constraints: { maxNodes: 5, maxEdges: 5 },
  visibleTests: [
    {
      id: "punctuation_visible",
      title: "punctuation and known vocab rows are stable",
      visibility: "visible",
      inputSeed: "ch1-3-visible",
      inputs: { texts: oovVisible },
      assertions: [
        { type: "dtype", nodeId: "token_ids", expected: "int" },
        { type: "tokens_include", nodeId: "tokenizer", token: "!" },
        { type: "no_oov", nodeId: "tokenizer" }
      ]
    }
  ],
  hiddenTests: [
    {
      id: "unseen_glyph_hidden",
      title: "unseen glyph resolves through fallback",
      visibility: "hidden",
      inputSeed: "ch1-3-hidden",
      inputs: { texts: oovHidden },
      assertions: [
        { type: "dtype", nodeId: "token_ids", expected: "int" },
        { type: "tokens_include", nodeId: "tokenizer", token: "<unk>" },
        { type: "no_oov", nodeId: "tokenizer" }
      ]
    }
  ],
  debrief: {
    completeTitle: "OOV Fallback Restored",
    fixedProblem: "Unknown pieces now resolve into valid IDs through fallback.",
    learned: "A tokenizer must produce stable IDs even when text contains unseen pieces.",
    nextUse: "Padding and masks turn variable-length ID lists into a rectangular batch."
  }
};

export const ch1PaddingMask: LevelSpec = {
  id: "ch1_4_padding_mask",
  title: "1-4 Special Tokens / Padding / Mask",
  mode: "graph_challenge",
  chapter: "Chapter 1",
  goal: "Build token_ids[B,T] and attention_mask[B,T] with BOS/EOS, right padding, and pad-aware masks.",
  modulePalette: ["TextInput", "TokenizerSocket", "EmbeddingReadyProbe"],
  initialGraph: createTokenizerGraph("ch1_4_padding_mask", { ...paddingParams, maskPolicy: "all-ones" }),
  constraints: { maxNodes: 5, maxEdges: 5 },
  visibleTests: [
    {
      id: "batch_padding_visible",
      title: "two texts become rectangular B/T tensors",
      visibility: "visible",
      inputSeed: "ch1-4-visible",
      inputs: { texts: paddingVisible },
      assertions: [
        { type: "shape", nodeId: "token_ids", expectedAxes: ["B", "T"], expectedDims: [2, 8] },
        { type: "shape", nodeId: "tokenizer.mask", expectedAxes: ["B", "T"], expectedDims: [2, 8] },
        { type: "mask_pad", idsNodeId: "tokenizer", maskNodeId: "tokenizer.mask", padId: 0 },
        { type: "eos_preserved", nodeId: "tokenizer" }
      ]
    }
  ],
  hiddenTests: [
    {
      id: "long_and_short_hidden",
      title: "long text truncates with EOS and short text pads",
      visibility: "hidden",
      inputSeed: "ch1-4-hidden",
      inputs: { texts: paddingHidden },
      assertions: [
        { type: "token_budget", nodeId: "token_ids", maxT: 8 },
        { type: "mask_pad", idsNodeId: "tokenizer", maskNodeId: "tokenizer.mask", padId: 0 },
        { type: "eos_preserved", nodeId: "tokenizer" },
        { type: "no_oov", nodeId: "tokenizer" }
      ]
    }
  ],
  debrief: {
    completeTitle: "Padding / Mask Restored",
    fixedProblem: "Variable-length texts now form token_ids[B,T] with a pad-aware attention mask.",
    learned: "PAD is data in token_ids, but it must be masked out before attention.",
    nextUse: "The gauntlet combines type, budget, fallback, and mask checks."
  }
};

export const ch1TokenizerMachine: LevelSpec = {
  id: "ch1_tokenizer_machine",
  title: "1-X Tokenizer Gauntlet",
  mode: "graph_challenge",
  chapter: "Chapter 1",
  goal: "Convert messy raw text batches into stable token_ids[B,T] and attention_mask[B,T].",
  modulePalette: ["TextInput", "TokenizerSocket", "EmbeddingReadyProbe"],
  initialGraph: createTokenizerGraph("ch1_tokenizer_machine", paddingParams),
  constraints: {
    maxNodes: 6,
    maxEdges: 6
  },
  visibleTests: [
    {
      id: "gauntlet_visible",
      title: "visible tokenizer gauntlet",
      visibility: "visible",
      inputSeed: "tokenizer-visible",
      inputs: { texts: gauntletVisible },
      assertions: [
        { type: "dtype", nodeId: "token_ids", expected: "int" },
        { type: "shape", nodeId: "token_ids", expectedAxes: ["B", "T"], expectedDims: [2, 8] },
        { type: "no_oov", nodeId: "tokenizer" },
        { type: "mask_pad", idsNodeId: "tokenizer", maskNodeId: "tokenizer.mask", padId: 0 },
        { type: "eos_preserved", nodeId: "tokenizer" }
      ]
    }
  ],
  hiddenTests: [
    {
      id: "gauntlet_hidden",
      title: "hidden tokenizer gauntlet",
      visibility: "hidden",
      inputSeed: "tokenizer-hidden",
      inputs: { texts: gauntletHidden },
      assertions: [
        { type: "dtype", nodeId: "token_ids", expected: "int" },
        { type: "shape", nodeId: "token_ids", expectedAxes: ["B", "T"], expectedDims: [6, 8] },
        { type: "token_budget", nodeId: "token_ids", maxT: 8 },
        { type: "no_oov", nodeId: "tokenizer" },
        { type: "mask_pad", idsNodeId: "tokenizer", maskNodeId: "tokenizer.mask", padId: 0 },
        { type: "eos_preserved", nodeId: "tokenizer" }
      ]
    }
  ],
  debrief: {
    completeTitle: "Tokenizer Machine Restored",
    fixedProblem: "Raw text is now converted to stable int[B,T] token IDs with pad-aware masks.",
    learned: "Tokenizer policy, fallback, padding, and mask are part of the model input contract.",
    nextUse: "Embedding Lookup consumes token_ids[B,T] in the next pipeline stage."
  }
};

export function createCh1TextTypeSolutionGraph() {
  return createTokenizerGraph("ch1_1_text_type_gate", textTypeParams);
}

export function createCh1SplitMergeSolutionGraph() {
  return createTokenizerGraph("ch1_2_split_merge_budget", splitMergeParams);
}

export function createCh1SplitMergeCharGraph() {
  return createTokenizerGraph("ch1_2_split_merge_budget", { ...splitMergeParams, policy: "char", applyMerges: false });
}

export function createCh1OovFallbackSolutionGraph() {
  return createTokenizerGraph("ch1_3_oov_fallback", oovParams);
}

export function createCh1OovNoFallbackGraph() {
  return createTokenizerGraph("ch1_3_oov_fallback", { ...oovParams, fallback: "none" });
}

export function createCh1PaddingMaskSolutionGraph() {
  return createTokenizerGraph("ch1_4_padding_mask", paddingParams);
}

export function createCh1PaddingMaskAllOnesGraph() {
  return createTokenizerGraph("ch1_4_padding_mask", { ...paddingParams, maskPolicy: "all-ones" });
}

function createTokenizerGraph(levelId: string, params: TokenizerParams): GraphSpec {
  return {
    levelId,
    version: 1,
    nodes: [
      { id: "text", moduleId: "TextInput", params: { inputKey: "texts" }, position: { x: 80, y: 160 } },
      { id: "tokenizer", moduleId: "TokenizerSocket", params, position: { x: 340, y: 160 } },
      { id: "token_ids", moduleId: "EmbeddingReadyProbe", params: {}, position: { x: 640, y: 160 } }
    ],
    edges: [
      { id: "e_text_tokenizer", from: { nodeId: "text", portId: "out" }, to: { nodeId: "tokenizer", portId: "text" } },
      { id: "e_tokenizer_probe", from: { nodeId: "tokenizer", portId: "out" }, to: { nodeId: "token_ids", portId: "ids" } }
    ],
    outputNodes: ["tokenizer", "token_ids"]
  };
}

function rawTexts(texts: string[]): RuntimeValue {
  return {
    dtype: "raw_text",
    data: texts,
    meta: { batch: texts }
  };
}
