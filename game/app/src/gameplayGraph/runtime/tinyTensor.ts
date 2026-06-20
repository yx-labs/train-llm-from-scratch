import type { AxisName, DType, TensorShape } from "../types";

export type TinyTensor = {
  dtype: Extract<DType, "float32" | "int" | "bool" | "mask">;
  dims: number[];
  axes: AxisName[];
  data: number[];
};

export function elementCount(dims: number[]) {
  return dims.reduce((total, dim) => total * dim, 1);
}

export function shapeOf(tensor: TinyTensor): TensorShape {
  return {
    dtype: tensor.dtype,
    dims: tensor.dims,
    axes: tensor.axes
  };
}

export function makeTensor(dtype: TinyTensor["dtype"], dims: number[], axes: AxisName[], data: number[]): TinyTensor {
  const expected = elementCount(dims);
  if (expected !== data.length) {
    throw new Error(`Tensor data length mismatch: expected ${expected}, received ${data.length}`);
  }
  if (dims.length !== axes.length) {
    throw new Error(`Tensor axes mismatch: dims rank ${dims.length}, axes rank ${axes.length}`);
  }
  return { dtype, dims, axes, data };
}

export function seededTensor(dims: number[], axes: AxisName[], seed: string, dtype: TinyTensor["dtype"] = "float32") {
  const count = elementCount(dims);
  let state = hashSeed(seed);
  const data = Array.from({ length: count }, () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    const normalized = state / 0xffffffff;
    if (dtype === "int" || dtype === "bool" || dtype === "mask") return Math.floor(normalized * 10);
    return Number((normalized * 2 - 1).toFixed(4));
  });
  return makeTensor(dtype, dims, axes, data);
}

export function transpose(tensor: TinyTensor, axisA: number, axisB: number) {
  const rank = tensor.dims.length;
  const a = normalizeAxis(axisA, rank);
  const b = normalizeAxis(axisB, rank);
  const outDims = [...tensor.dims];
  const outAxes = [...tensor.axes];
  [outDims[a], outDims[b]] = [outDims[b], outDims[a]];
  [outAxes[a], outAxes[b]] = [outAxes[b], outAxes[a]];

  const outData = Array.from({ length: tensor.data.length }, (_, outFlatIndex) => {
    const outIndex = unravelIndex(outFlatIndex, outDims);
    const inputIndex = [...outIndex];
    [inputIndex[a], inputIndex[b]] = [inputIndex[b], inputIndex[a]];
    return getValue(tensor, inputIndex);
  });

  return makeTensor(tensor.dtype, outDims, outAxes, outData);
}

export function matmul(left: TinyTensor, right: TinyTensor) {
  if (left.dtype !== "float32" || right.dtype !== "float32") {
    throw new Error("MatMul expects float32 tensors");
  }
  if (left.dims.length < 1 || right.dims.length < 2) {
    throw new Error("MatMul expects left [..., C] and right [C, O] or [..., C, O]");
  }

  if (right.dims.length === 2) return vectorizedMatmul(left, right);
  return batchedMatrixMatmul(left, right);
}

export function broadcastTo(tensor: TinyTensor, targetDims: number[], targetAxes: AxisName[], alignAxes: AxisName[]) {
  if (tensor.dtype !== "float32") {
    throw new Error("Broadcast expects float32 tensors");
  }
  if (tensor.dims.length !== alignAxes.length) {
    throw new Error(`Broadcast axis rank mismatch: tensor rank ${tensor.dims.length}, align axes rank ${alignAxes.length}`);
  }

  const targetSlots = resolveTargetSlots(targetAxes, alignAxes);
  targetSlots.forEach((targetIndex, index) => {
    const axis = alignAxes[index];
    if (targetIndex < 0) {
      throw new Error(`Broadcast axis ${axis} is not present in target axes [${targetAxes.join(",")}]`);
    }
    const targetDim = targetDims[targetIndex];
    const sourceDim = tensor.dims[index];
    if (targetDim !== sourceDim) {
      throw new Error(`Broadcast dim mismatch on ${axis}: source ${sourceDim}, target ${targetDim}`);
    }
  });

  const data = Array.from({ length: elementCount(targetDims) }, (_, flatIndex) => {
    const targetIndex = unravelIndex(flatIndex, targetDims);
    const sourceIndex = targetSlots.map((targetSlot) => targetIndex[targetSlot]);
    return getValue(tensor, sourceIndex);
  });
  return makeTensor(tensor.dtype, targetDims, targetAxes, data);
}

function resolveTargetSlots(targetAxes: AxisName[], alignAxes: AxisName[]) {
  const usedTargetSlots = new Set<number>();
  return alignAxes.map((axis) => {
    const targetIndex = targetAxes.findIndex((targetAxis, index) => targetAxis === axis && !usedTargetSlots.has(index));
    if (targetIndex >= 0) usedTargetSlots.add(targetIndex);
    return targetIndex;
  });
}

export function addTensors(left: TinyTensor, right: TinyTensor) {
  if (left.dtype !== "float32" || right.dtype !== "float32") {
    throw new Error("AddGate expects float32 tensors");
  }
  if (!sameDims(left.dims, right.dims) || !sameAxes(left.axes, right.axes)) {
    throw new Error(
      `AddGate shape mismatch: left ${left.dtype}[${left.axes.join(",")}]=[${left.dims.join(",")}] vs right ${right.dtype}[${right.axes.join(",")}]=[${right.dims.join(",")}]`
    );
  }
  const data = left.data.map((value, index) => Number((value + right.data[index]).toFixed(6)));
  return makeTensor("float32", left.dims, left.axes, data);
}

function vectorizedMatmul(left: TinyTensor, right: TinyTensor) {
  const leftC = left.dims[left.dims.length - 1];
  const rightC = right.dims[0];
  if (leftC !== rightC) {
    throw new Error(`MatMul inner dimension mismatch: ${leftC} vs ${rightC}`);
  }

  const carrierDims = left.dims.slice(0, -1);
  const carrierAxes = left.axes.slice(0, -1);
  const outO = right.dims[1];
  const outDims = [...carrierDims, outO];
  const outAxes = [...carrierAxes, right.axes[1] ?? "O"];
  const carrierCount = elementCount(carrierDims);
  const data: number[] = [];

  for (let carrierFlat = 0; carrierFlat < carrierCount; carrierFlat += 1) {
    const carrierIndex = carrierDims.length ? unravelIndex(carrierFlat, carrierDims) : [];
    for (let o = 0; o < outO; o += 1) {
      let sum = 0;
      for (let c = 0; c < leftC; c += 1) {
        sum += getValue(left, [...carrierIndex, c]) * getValue(right, [c, o]);
      }
      data.push(Number(sum.toFixed(6)));
    }
  }

  return makeTensor("float32", outDims, outAxes, data);
}

function batchedMatrixMatmul(left: TinyTensor, right: TinyTensor) {
  if (left.dims.length < 2 || right.dims.length < 2) {
    throw new Error("Batched MatMul expects left [..., M, C] and right [..., C, O]");
  }
  const leftCarrierDims = left.dims.slice(0, -2);
  const rightCarrierDims = right.dims.slice(0, -2);
  if (!sameDims(leftCarrierDims, rightCarrierDims)) {
    throw new Error(`MatMul carrier dimensions mismatch: [${leftCarrierDims.join(",")}] vs [${rightCarrierDims.join(",")}]`);
  }

  const leftC = left.dims[left.dims.length - 1];
  const rightC = right.dims[right.dims.length - 2];
  if (leftC !== rightC) {
    throw new Error(`MatMul inner dimension mismatch: ${leftC} vs ${rightC}`);
  }

  const carrierAxes = left.axes.slice(0, -2);
  const m = left.dims[left.dims.length - 2];
  const o = right.dims[right.dims.length - 1];
  const outDims = [...leftCarrierDims, m, o];
  const outAxes = [...carrierAxes, left.axes[left.axes.length - 2], right.axes[right.axes.length - 1]];
  const carrierCount = elementCount(leftCarrierDims);
  const data: number[] = [];

  for (let carrierFlat = 0; carrierFlat < carrierCount; carrierFlat += 1) {
    const carrierIndex = leftCarrierDims.length ? unravelIndex(carrierFlat, leftCarrierDims) : [];
    for (let row = 0; row < m; row += 1) {
      for (let col = 0; col < o; col += 1) {
        let sum = 0;
        for (let c = 0; c < leftC; c += 1) {
          sum += getValue(left, [...carrierIndex, row, c]) * getValue(right, [...carrierIndex, c, col]);
        }
        data.push(Number(sum.toFixed(6)));
      }
    }
  }

  return makeTensor("float32", outDims, outAxes, data);
}

export function allclose(left: TinyTensor, right: TinyTensor, atol: number) {
  if (!sameDims(left.dims, right.dims)) return { ok: false, maxAbsError: Number.POSITIVE_INFINITY };
  let maxAbsError = 0;
  for (let index = 0; index < left.data.length; index += 1) {
    maxAbsError = Math.max(maxAbsError, Math.abs(left.data[index] - right.data[index]));
  }
  return { ok: maxAbsError <= atol, maxAbsError };
}

export function sameDims(left: number[], right: number[]) {
  return left.length === right.length && left.every((dim, index) => dim === right[index]);
}

export function sameAxes(left: AxisName[], right: AxisName[]) {
  return left.length === right.length && left.every((axis, index) => axis === right[index]);
}

export function getValue(tensor: TinyTensor, index: number[]) {
  return tensor.data[ravelIndex(index, tensor.dims)];
}

export function sampleValues(tensor: TinyTensor, limit = 6) {
  return tensor.data.slice(0, limit);
}

function normalizeAxis(axis: number, rank: number) {
  const normalized = axis < 0 ? rank + axis : axis;
  if (normalized < 0 || normalized >= rank) {
    throw new Error(`Axis ${axis} is out of range for rank ${rank}`);
  }
  return normalized;
}

function ravelIndex(index: number[], dims: number[]) {
  let flat = 0;
  for (let dimIndex = 0; dimIndex < dims.length; dimIndex += 1) {
    flat = flat * dims[dimIndex] + index[dimIndex];
  }
  return flat;
}

function unravelIndex(flatIndex: number, dims: number[]) {
  const index = Array.from({ length: dims.length }, () => 0);
  let remainder = flatIndex;
  for (let dimIndex = dims.length - 1; dimIndex >= 0; dimIndex -= 1) {
    index[dimIndex] = remainder % dims[dimIndex];
    remainder = Math.floor(remainder / dims[dimIndex]);
  }
  return index;
}

function hashSeed(seed: string) {
  let hash = 2166136261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
