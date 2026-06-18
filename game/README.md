# LLM Complete MVP 0.02

This directory now hosts the independent v0.02 prototype. MVP 0.01 has been archived under `archive/mvp0.01/`.

## Scope

- PixiJS v8 canvas foundation for the 3D Tensor Workbench direction.
- Fixed Transformer attention trace board with typed ports, shape labels, tensor cuboids, attention heatmaps, mask diagnostics, and gradient path.
- React side panels for Module Shelf, Tensor Inspector, and Test Runner.

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
