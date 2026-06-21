import type { ModuleDef, RuntimeError, RuntimeValue, TensorShape } from "../types";
import { encodeToyBatch, encodeToyText, splitTextToPieces, type TokenizerPolicy } from "../../toyTokenizer";

function error(type: RuntimeError["type"], nodeId: string, message: string, expected?: unknown, received?: unknown, portId?: string): RuntimeError {
  return {
    type,
    nodeId,
    portId,
    message,
    expected,
    received
  };
}

export const textInputModule: ModuleDef = {
  id: "TextInput",
  label: "Text Input",
  category: "data",
  inputs: [],
  outputs: [{ id: "out", label: "raw text", direction: "out", emits: "raw_text" }],
  defaultParams: { inputKey: "texts", text: "we train llm" },
  summary: "Reads raw text or a text batch from the active test case.",
  execute: ({ node, testInputs }) => {
    const inputKey = String(node.params.inputKey ?? "texts");
    const testValue = testInputs[inputKey];
    if (testValue) return { outputs: { out: testValue }, samples: { raw: testValue.data } };
    return {
      outputs: {
        out: {
          dtype: "raw_text",
          data: String(node.params.text ?? ""),
          meta: { batch: [String(node.params.text ?? "")] }
        }
      },
      samples: { raw: node.params.text }
    };
  },
  infer: () => ({ outputs: { out: { dtype: "raw_text", dims: [], axes: [] } } })
};

export const tokenizerSocketModule: ModuleDef = {
  id: "TokenizerSocket",
  label: "Tokenizer Socket",
  category: "tokenizer",
  inputs: [{ id: "text", label: "text", direction: "in", accepts: ["raw_text"], required: true }],
  outputs: [
    { id: "pieces", label: "pieces", direction: "out", emits: "token_piece" },
    { id: "out", label: "ids", direction: "out", emits: "int" },
    { id: "mask", label: "mask", direction: "out", emits: "mask" }
  ],
  defaultParams: {
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
  },
  summary: "Turns raw text into token pieces, token IDs, and an attention mask.",
  execute: ({ node, inputs }) => {
    const textValue = inputs.text;
    const texts = extractTexts(textValue);
    if (!texts.length) {
      return { outputs: {}, error: error("missing_input", node.id, "TokenizerSocket requires raw text", "raw_text", textValue, "text") };
    }
    const maxLength = Number(node.params.maxLength ?? 8);
    const padToLength = Number(node.params.padToLength ?? maxLength);
    const batch = encodeToyBatch(texts, {
      policy: node.params.policy as TokenizerPolicy,
      applyMerges: Boolean(node.params.applyMerges ?? true),
      fallback: node.params.fallback as "none" | "char" | "unk",
      preservePunctuation: Boolean(node.params.preservePunctuation ?? true),
      addBos: Boolean(node.params.addBos ?? true),
      addEos: Boolean(node.params.addEos ?? true),
      maxLength,
      padToLength,
      padSide: node.params.padSide as "left" | "right",
      maskPolicy: node.params.maskPolicy as "pad-aware" | "all-ones"
    });
    const b = batch.tokenIds.length;
    const t = batch.tokenIds[0]?.length ?? 0;
    const unresolved = batch.unresolved;
    if (unresolved.length && node.params.fallback === "none") {
      return {
        outputs: {},
        error: error("oov_unresolved", node.id, `OOV unresolved: ${unresolved.join(", ")}`, "known vocab or fallback", unresolved)
      };
    }
    const idsValue: RuntimeValue = {
      dtype: "int",
      shape: { dtype: "int", dims: [b, t], axes: ["B", "T"] },
      data: batch.tokenIds.flat(),
      meta: { tokens: batch.tokens, unresolved, truncated: batch.truncated }
    };
    const maskValue: RuntimeValue = {
      dtype: "mask",
      shape: { dtype: "mask", dims: [b, t], axes: ["B", "T"] },
      data: batch.attentionMask.flat(),
      meta: { policy: node.params.maskPolicy ?? "pad-aware" }
    };
    return {
      outputs: {
        pieces: {
          dtype: "token_piece",
          data: texts.flatMap(
            (text) =>
              encodeToyText(text, {
                policy: node.params.policy as TokenizerPolicy,
                applyMerges: Boolean(node.params.applyMerges ?? true),
                preservePunctuation: Boolean(node.params.preservePunctuation ?? true)
              }).pieces
          )
        },
        out: idsValue,
        mask: maskValue
      },
      samples: { tokens: batch.tokens, ids: batch.tokenIds, mask: batch.attentionMask }
    };
  },
  infer: ({ node, inputs }) => {
    const texts = extractTexts(inputs.text);
    const b = Math.max(texts.length, 1);
    const t = Number(node.params.padToLength ?? node.params.maxLength ?? 8);
    return {
      outputs: {
        pieces: { dtype: "token_piece", dims: [t], axes: ["T"] },
        out: { dtype: "int", dims: [b, t], axes: ["B", "T"] },
        mask: { dtype: "mask", dims: [b, t], axes: ["B", "T"] }
      }
    };
  }
};

export const boundarySplitterModule: ModuleDef = {
  id: "BoundarySplitter",
  label: "Boundary Splitter",
  category: "tokenizer",
  inputs: [{ id: "text", label: "text", direction: "in", accepts: ["raw_text"], required: true }],
  outputs: [{ id: "pieces", label: "pieces", direction: "out", emits: "string_piece" }],
  defaultParams: { policy: "word", preservePunctuation: true, expectedT: 4 },
  summary: "Splits raw text into ordered string pieces along word and punctuation boundaries.",
  pseudoCode: "pieces = split_words_and_punctuation(text)\nshape(pieces) == string_piece[T]",
  execute: ({ node, inputs }) => {
    const texts = extractTexts(inputs.text);
    if (!texts.length) {
      return { outputs: {}, error: error("missing_input", node.id, "BoundarySplitter requires raw text", "raw_text", inputs.text, "text") };
    }
    const policy = (node.params.policy ?? "word") as TokenizerPolicy;
    const preservePunctuation = Boolean(node.params.preservePunctuation ?? true);
    const pieces = texts.flatMap((text) => splitTextToPieces(text, policy, preservePunctuation));
    const value: RuntimeValue = {
      dtype: "string_piece",
      shape: { dtype: "string_piece", dims: [pieces.length], axes: ["T"] },
      data: pieces,
      meta: { texts, policy, preservePunctuation }
    };
    return { outputs: { pieces: value }, samples: { pieces } };
  },
  infer: ({ node }) => ({
    outputs: {
      pieces: {
        dtype: "string_piece",
        dims: [Number(node.params.expectedT ?? 4)],
        axes: ["T"]
      }
    }
  })
};

export const pieceBufferModule: ModuleDef = {
  id: "PieceBuffer",
  label: "Piece Buffer",
  category: "contract",
  inputs: [{ id: "pieces", label: "pieces", direction: "in", accepts: ["string_piece"], required: true }],
  outputs: [{ id: "out", label: "pieces", direction: "out", emits: "string_piece" }],
  defaultParams: { maxPieces: 12 },
  summary: "Checks that a splitter produced a finite ordered string_piece[T] buffer.",
  pseudoCode: "assert dtype(pieces) == string_piece\nassert len(pieces) <= max_pieces",
  execute: ({ node, inputs }) => {
    const value = inputs.pieces;
    if (value?.dtype !== "string_piece") {
      return {
        outputs: {},
        error: error("dtype_mismatch", node.id, "PieceBuffer expects string_piece input", "string_piece[T]", value?.dtype, "pieces")
      };
    }
    const pieces = Array.isArray(value.data) ? value.data : [];
    if (!pieces.every((piece) => typeof piece === "string")) {
      return {
        outputs: {},
        error: error("dtype_mismatch", node.id, "PieceBuffer only accepts string pieces", "string[]", pieces, "pieces")
      };
    }
    const maxPieces = Number(node.params.maxPieces ?? 12);
    if (pieces.length > maxPieces) {
      return {
        outputs: {},
        error: error("budget_exceeded", node.id, `PieceBuffer maxPieces exceeded: ${pieces.length} > ${maxPieces}`, maxPieces, pieces.length, "pieces")
      };
    }
    const shape = value.shape ?? { dtype: "string_piece" as const, dims: [pieces.length], axes: ["T" as const] };
    return { outputs: { out: { ...value, shape } }, samples: { pieces } };
  },
  infer: ({ node, inputs }) => {
    const shape = inputs.pieces?.shape;
    return {
      outputs: {
        out: shape ?? {
          dtype: "string_piece",
          dims: [Number(node.params.maxPieces ?? 12)],
          axes: ["T"]
        }
      }
    };
  }
};

export const embeddingReadyProbeModule: ModuleDef = {
  id: "EmbeddingReadyProbe",
  label: "Embedding Ready Probe",
  category: "contract",
  inputs: [{ id: "ids", label: "ids", direction: "in", accepts: ["int"], required: true }],
  outputs: [{ id: "out", label: "ids", direction: "out", emits: "int" }],
  defaultParams: {},
  summary: "Validates that model input is integer token IDs.",
  execute: ({ node, inputs }) => {
    const value = inputs.ids;
    if (value?.dtype !== "int") {
      return {
        outputs: {},
        error: error("dtype_mismatch", node.id, "Embedding Lookup expects integer token IDs.", "int token_ids[B,T]", value?.dtype, "ids")
      };
    }
    return { outputs: { out: value }, samples: { shape: value.shape } };
  },
  infer: ({ inputs }) => {
    const shape = inputs.ids?.shape;
    return shape ? { outputs: { out: shape as TensorShape } } : { outputs: {} };
  }
};

export const tokenizerModules = [textInputModule, tokenizerSocketModule, boundarySplitterModule, pieceBufferModule, embeddingReadyProbeModule];

function extractTexts(value: RuntimeValue | undefined) {
  if (!value) return [];
  if (typeof value.data === "string") return [value.data];
  if (Array.isArray(value.data) && value.data.every((item) => typeof item === "string")) return value.data as string[];
  const batch = value.meta?.batch;
  if (Array.isArray(batch) && batch.every((item) => typeof item === "string")) return batch as string[];
  return [];
}
