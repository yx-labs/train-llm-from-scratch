import type { LevelSpec } from "./types";
import { ch0GraphBasics } from "./levels/ch0GraphBasics";
import { ch0MatMulGraph } from "./levels/ch0MatMulGraph";
import { ch0TransposeGraph } from "./levels/ch0TransposeGraph";
import { ch0BroadcastGraph } from "./levels/ch0BroadcastGraph";
import { ch0MaskGraph } from "./levels/ch0MaskGraph";
import { ch1OovFallback, ch1PaddingMask, ch1SplitMergeBudget, ch1TextTypeGate, ch1TokenizerMachine } from "./levels/ch1TokenizerMachine";

export const graphLevels: LevelSpec[] = [
  ch0GraphBasics,
  ch1TextTypeGate,
  ch1SplitMergeBudget,
  ch0MatMulGraph,
  ch0TransposeGraph,
  ch0BroadcastGraph,
  ch0MaskGraph,
  ch1OovFallback,
  ch1PaddingMask,
  ch1TokenizerMachine
];

export function getGraphLevel(levelId: string) {
  return graphLevels.find((level) => level.id === levelId);
}
