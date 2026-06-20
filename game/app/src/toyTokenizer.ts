export type TokenizerPolicy = "word" | "char" | "subword";

export type ToyTokenizerOptions = {
  policy?: TokenizerPolicy;
  applyMerges?: boolean;
  fallback?: "none" | "char" | "unk";
  addSpecialTokens?: boolean;
  addBos?: boolean;
  addEos?: boolean;
  preservePunctuation?: boolean;
  maxLength?: number;
  padToLength?: number;
  padSide?: "left" | "right";
  maskPolicy?: "pad-aware" | "all-ones";
};

export type ToyTokenizerResult = {
  text: string;
  pieces: string[];
  tokens: string[];
  ids: number[];
  attentionMask: number[];
  unresolved: string[];
  truncated: boolean;
};

export const toyVocab: Record<string, number> = {
  "<pad>": 0,
  "<bos>": 1,
  "<eos>": 2,
  "<unk>": 3,
  "_": 4,
  "!": 5,
  ",": 6,
  ".": 7,
  we: 8,
  train: 9,
  llm: 10,
  shape: 11,
  token: 12,
  izer: 13,
  tokenizer: 14,
  tokenizers: 15,
  ing: 16,
  training: 17,
  use: 18,
  ful: 19,
  useful: 20,
  mask: 21,
  data: 22,
  batch: 23,
  buffer: 24,
  ids: 25,
  pad: 26,
  unknown: 27,
  fallback: 28,
  test: 29,
  text: 30,
  to: 31,
  embedding: 32,
  flow: 33,
  tokenization: 34,
  budget: 35,
  symbols: 36,
  preserve: 37,
  alignment: 38
};

export const toyMergeRules: Array<[string, string, string]> = [
  ["token", "izer", "tokenizer"],
  ["tokenizer", "s", "tokenizers"],
  ["train", "ing", "training"],
  ["use", "ful", "useful"]
];

export const visibleTextSet = ["we train llm.", "tokenizers are useful!", "shape, token buffer"];
export const hiddenTextSet = ["we train useful tokenizers.", "unknown glyph ?", "tokenization budget test", "padding mask alignment"];

const punctuation = new Set([",", ".", "!", "?", ":", ";", "(", ")", "[", "]"]);

export function splitTextToPieces(text: string, policy: TokenizerPolicy = "subword", preservePunctuation = true) {
  if (policy === "char") {
    return Array.from(text)
      .filter((char) => preservePunctuation || !punctuation.has(char))
      .map((char) => (char === " " ? "_" : char));
  }

  const normalized = text.trim().toLowerCase();
  const words: string[] = [];
  let current = "";
  Array.from(normalized).forEach((char) => {
    if (char === " ") {
      if (current) words.push(current);
      current = "";
      return;
    }
    if (punctuation.has(char)) {
      if (current) words.push(current);
      if (preservePunctuation) words.push(char);
      current = "";
      return;
    }
    current += char;
  });
  if (current) words.push(current);

  if (policy === "word") return words;

  return words.flatMap((word) => {
    if (punctuation.has(word)) return [word];
    if (toyVocab[word] !== undefined) return [word];
    if (word.endsWith("izers")) return [word.slice(0, -5), "izer", "s"].filter(Boolean);
    if (word.endsWith("ization")) return [word.slice(0, -7), "ization"].filter(Boolean);
    if (word.endsWith("ing")) return [word.slice(0, -3), "ing"].filter(Boolean);
    if (word.endsWith("ful")) return [word.slice(0, -3), "ful"].filter(Boolean);
    return [word];
  });
}

export function applyToyMerges(pieces: string[]) {
  let current = [...pieces];
  let changed = true;
  while (changed) {
    changed = false;
    const next: string[] = [];
    for (let index = 0; index < current.length; index += 1) {
      const pair = toyMergeRules.find(([left, right]) => current[index] === left && current[index + 1] === right);
      if (pair) {
        next.push(pair[2]);
        index += 1;
        changed = true;
      } else {
        next.push(current[index]);
      }
    }
    current = next;
  }
  return current;
}

export function resolveToyPieces(pieces: string[], fallback: ToyTokenizerOptions["fallback"] = "unk") {
  const tokens: string[] = [];
  const unresolved: string[] = [];

  pieces.forEach((piece) => {
    if (toyVocab[piece] !== undefined) {
      tokens.push(piece);
      return;
    }
    if (fallback === "char") {
      const chars = Array.from(piece);
      chars.forEach((char) => {
        if (toyVocab[char] !== undefined) tokens.push(char);
        else if (fallback === "char" && toyVocab[char.toLowerCase()] !== undefined) tokens.push(char.toLowerCase());
        else tokens.push("<unk>");
      });
      return;
    }
    if (fallback === "unk") {
      tokens.push("<unk>");
      return;
    }
    unresolved.push(piece);
  });

  return { tokens, unresolved };
}

export function encodeToyText(text: string, options: ToyTokenizerOptions = {}): ToyTokenizerResult {
  const policy = options.policy ?? "subword";
  const initialPieces = splitTextToPieces(text, policy, options.preservePunctuation ?? true);
  const pieces = options.applyMerges === false ? initialPieces : applyToyMerges(initialPieces);
  const resolved = resolveToyPieces(pieces, options.fallback ?? "unk");
  const addBos = options.addBos ?? options.addSpecialTokens !== false;
  const addEos = options.addEos ?? options.addSpecialTokens !== false;
  let tokens = [...(addBos ? ["<bos>"] : []), ...resolved.tokens, ...(addEos ? ["<eos>"] : [])];
  let truncated = false;

  if (options.maxLength !== undefined && tokens.length > options.maxLength) {
    tokens = tokens.slice(0, options.maxLength);
    truncated = true;
    if (addEos && tokens.length > 0 && tokens[tokens.length - 1] !== "<eos>") tokens[tokens.length - 1] = "<eos>";
  }

  const padToLength = options.padToLength;
  if (padToLength !== undefined && tokens.length < padToLength) {
    const pads = Array.from({ length: padToLength - tokens.length }, () => "<pad>");
    tokens = options.padSide === "left" ? [...pads, ...tokens] : [...tokens, ...pads];
  }
  const attentionMask = options.maskPolicy === "all-ones" ? tokens.map(() => 1) : tokens.map((token) => (token === "<pad>" ? 0 : 1));

  return {
    text,
    pieces,
    tokens,
    ids: tokens.map((token) => toyVocab[token] ?? toyVocab["<unk>"]),
    attentionMask,
    unresolved: resolved.unresolved,
    truncated
  };
}

export function encodeToyBatch(texts: string[], options: ToyTokenizerOptions = {}) {
  const encoded = texts.map((text) => encodeToyText(text, { ...options, padToLength: undefined }));
  const targetLength = options.padToLength ?? Math.max(...encoded.map((item) => item.tokens.length));
  const padded = texts.map((text) => encodeToyText(text, { ...options, padToLength: targetLength }));
  return {
    tokens: padded.map((item) => item.tokens),
    tokenIds: padded.map((item) => item.ids),
    attentionMask: padded.map((item) => item.attentionMask),
    unresolved: padded.flatMap((item) => item.unresolved),
    truncated: padded.some((item) => item.truncated)
  };
}
