# LLM Learning Game Workspace

This folder contains all game-specific planning, prototypes, tools, and runtime code.

## Local Environment

Python uses the project-level virtual environment created from:

```powershell
C:\python312\python.exe -m venv .venv
```

Backend dependencies are installed into `.venv` from `game/requirements.txt`.

Frontend uses the Vite React app in `game/app`.

## API Keys

Local secrets live in `game/.env.local` and are ignored by git. The current preferred AI provider is OpenAI:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_LLM_MODEL`
- `OPENAI_IMAGE_MODEL`

Do not commit real keys. Use `game/env.example` as the template for new machines.

## Start Backend

```powershell
.venv\Scripts\python.exe -m uvicorn server.main:app --app-dir game --reload --port 8010
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8010/api/health
```

## Start Frontend

```powershell
pnpm -C game/app install
pnpm -C game/app dev
```

The frontend defaults to `http://127.0.0.1:5173`.

## MVP Contents

The first playable MVP includes:

- Pipeline map with 8 missions from data cleaning to SFT masking.
- Mission workbench interactions for sample filtering, token merging, target shift, causal mask, module assembly, training simulation, generation controls, and SFT loss mask.
- Rule-based failure reports for dirty data, future leakage, unstable optimization, bad sampling, and SFT mask mistakes.
- Local progress stored in `game/progress/local.sqlite`.
- Artifact registry with `pretrain_good` and `sft` sample-output fallbacks.
- Arena comparison for base vs SFT behavior.

## Verify

```powershell
.venv\Scripts\python.exe -m pytest game\tests
pnpm -C game/app typecheck
pnpm -C game/app build
```
