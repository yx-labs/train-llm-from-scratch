import type { AxisName } from "../types";
import { addTensors, broadcastTo, matmul, makeTensor, transpose, type TinyTensor } from "./tinyTensor";

export function referenceLinear(hidden: TinyTensor, weight: TinyTensor, orientation: "C,O" | "O,C") {
  const computeWeight = orientation === "O,C" ? transpose(weight, 0, 1) : weight;
  return matmul(hidden, computeWeight);
}

export function referenceQKScores(q: TinyTensor, k: TinyTensor) {
  return matmul(q, transpose(k, -2, -1));
}

export function referenceBroadcastAdd(base: TinyTensor, small: TinyTensor, alignAxes: AxisName[]) {
  return addTensors(base, broadcastTo(small, base.dims, base.axes, alignAxes));
}

export function causalMask(size: number, orientation: "query_key" | "key_query" = "query_key") {
  const data: number[] = [];
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const queryIndex = orientation === "query_key" ? row : col;
      const keyIndex = orientation === "query_key" ? col : row;
      data.push(keyIndex > queryIndex ? -10000 : 0);
    }
  }
  return makeTensor("float32", [size, size], ["T", "T"], data);
}
