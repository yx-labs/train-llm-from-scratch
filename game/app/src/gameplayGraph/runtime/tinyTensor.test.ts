import { describe, expect, it } from "vitest";
import { addTensors, allclose, broadcastTo, makeTensor, matmul, transpose } from "./tinyTensor";

describe("tinyTensor", () => {
  it("transposes a 2D tensor with shape and value mapping", () => {
    const input = makeTensor("float32", [2, 3], ["R", "C"], [1, 2, 3, 4, 5, 6]);
    const output = transpose(input, 0, 1);

    expect(output.dims).toEqual([3, 2]);
    expect(output.axes).toEqual(["C", "R"]);
    expect(output.data).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("runs [..., C] @ [C, O] and preserves carrier axes", () => {
    const left = makeTensor("float32", [2, 2, 3], ["B", "T", "C"], [
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
      10, 11, 12
    ]);
    const right = makeTensor("float32", [3, 2], ["C", "O"], [
      1, 10,
      2, 20,
      3, 30
    ]);

    const output = matmul(left, right);

    expect(output.dims).toEqual([2, 2, 2]);
    expect(output.axes).toEqual(["B", "T", "O"]);
    expect(output.data).toEqual([14, 140, 32, 320, 50, 500, 68, 680]);
  });

  it("runs batched [..., M, C] @ [..., C, O]", () => {
    const q = makeTensor("float32", [1, 1, 2, 2], ["B", "H", "T", "D"], [
      1, 0,
      0, 1
    ]);
    const kt = makeTensor("float32", [1, 1, 2, 3], ["B", "H", "D", "T"], [
      2, 0, 1,
      0, 3, 1
    ]);

    const scores = matmul(q, kt);

    expect(scores.dims).toEqual([1, 1, 2, 3]);
    expect(scores.axes).toEqual(["B", "H", "T", "T"]);
    expect(scores.data).toEqual([2, 0, 1, 0, 3, 1]);
  });

  it("exposes the T == D numeric trap when K is not transposed", () => {
    const q = makeTensor("float32", [1, 1, 2, 2], ["B", "H", "T", "D"], [
      1, 2,
      3, 4
    ]);
    const k = makeTensor("float32", [1, 1, 2, 2], ["B", "H", "T", "D"], [
      5, 6,
      7, 8
    ]);

    const wrong = matmul(q, k);
    const correct = matmul(q, transpose(k, -2, -1));

    expect(wrong.dims).toEqual(correct.dims);
    expect(allclose(wrong, correct, 1e-5).ok).toBe(false);
  });

  it("computes allclose with max abs error", () => {
    const left = makeTensor("float32", [2], ["T"], [1, 2]);
    const right = makeTensor("float32", [2], ["T"], [1, 2.000001]);

    expect(allclose(left, right, 1e-5)).toEqual({ ok: true, maxAbsError: 0.000001000000000139778 });
  });

  it("broadcasts a bias vector by semantic axis before add", () => {
    const base = makeTensor("float32", [1, 2, 3], ["B", "T", "O"], [
      1, 2, 3,
      4, 5, 6
    ]);
    const bias = makeTensor("float32", [3], ["O"], [10, 20, 30]);
    const expanded = broadcastTo(bias, base.dims, base.axes, ["O"]);
    const out = addTensors(base, expanded);

    expect(expanded.dims).toEqual([1, 2, 3]);
    expect(expanded.axes).toEqual(["B", "T", "O"]);
    expect(expanded.data).toEqual([10, 20, 30, 10, 20, 30]);
    expect(out.data).toEqual([11, 22, 33, 14, 25, 36]);
  });

  it("broadcasts duplicate T axes by occurrence order for masks", () => {
    const mask = makeTensor("float32", [2, 2], ["T", "T"], [
      0, -100,
      0, 0
    ]);

    const expanded = broadcastTo(mask, [1, 1, 2, 2], ["B", "H", "T", "T"], ["T", "T"]);

    expect(expanded.dims).toEqual([1, 1, 2, 2]);
    expect(expanded.axes).toEqual(["B", "H", "T", "T"]);
    expect(expanded.data).toEqual([0, -100, 0, 0]);
  });
});
