# LLM Complete MVP 0.02

This directory now hosts the independent v0.02 prototype. MVP 0.01 has been archived under `archive/mvp0.01/`.

## Scope

- PixiJS v8 canvas foundation for the 3D Tensor Workbench direction.
- Chapter 0 Tensor Bootcamp with four playable shape-repair levels:
  - 0-1 Shape Reader
  - 0-2 MatMul Gate
  - 0-3 Transpose Trap
  - 0-4 Broadcast Add
- Config-driven broken boards with typed ports, shape labels, tensor cuboids, operation nodes, repair slots, and repair paths.
- Repair Console with draggable/clickable tags, Probe tools, visible/reference/hidden contract tests, and score feedback.
- React side panels for level selection, Tensor Inspector evidence, briefing, Repair Console, and Test Runner.

## Run

```powershell
pnpm -C game/app install
pnpm -C game/app dev
```

Open `http://127.0.0.1:5173`.

## Verify

```powershell
pnpm -C game/app typecheck
pnpm -C game/app build
```
