# Self-Consistency Eval (DeepInfra / OpenAI-Compatible)

This folder contains a small evaluation harness to measure **pass@1** (temp=0) and **pass@k** (temp=0.7, k=8..256) on **MATH-500**, plus a lightweight **live HTML dashboard**.

## Setup

Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `mathruler` is included to match EasyR1's grading behavior (e.g. treating `\\frac{11}{2}` and `5.5` as equal). If it’s unavailable, the script falls back to a lightweight numeric equivalence.

Set credentials:

- **DeepInfra**: set `DEEPINFRA_API_KEY`
- Or any OpenAI-compatible provider: set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`

## Run evaluation

```bash
export DEEPINFRA_API_KEY="YOUR_KEY"
python eval_self_consistency.py \
  --model "YOUR_MODEL_NAME"
```

## Evaluate both AIME24 + MATH-500 (suite)

```bash
export DEEPINFRA_API_KEY="YOUR_KEY"
python eval_suite.py \
  --suite-id "my_suite_run" \
  --model "YOUR_MODEL_NAME"
```

Outputs to `runs_suite/my_suite_run/{math500,aime24}/` plus `runs_suite/my_suite_run/suite_summary.json`.

## Sweep checkpoints with DeepInfra deploy/delete (serverless)

This will:
- deploy each HF checkpoint as a DeepInfra LLM deployment
- evaluate AIME24 + MATH-500
- delete the deployment afterwards
- keep at most **N** deployments (instances) active concurrently

```bash
export DEEPINFRA_API_KEY="YOUR_KEY"
python sweep_deepinfra_checkpoints.py \
  --hf-prefix "nicoledy/qwen2.5-math-base-limr-step-" \
  --start 1 --end 41 \
  --max-active 4
```

Defaults for sweep:
- **GPU**: auto-choose with preference **B200-180GB** (single GPU per checkpoint)
- **Concurrency**: `--max-active 4`

Then serve and open `training_dynamics.html` to visualize curves:

```bash
python -m http.server 8000
```

Open:
- `http://localhost:8000/training_dynamics.html?sweep=sweeps/<sweep_id>`

Outputs go to `runs/<run_id>/`:

- `config.json`: run config
- `samples.json`: the 50 sampled rows (raw)
- `results.jsonl`: one JSON per sample (raw sample + parsed generations + correctness)
- `requests.jsonl`: **raw API request/response payloads** (all chunks)
- `progress.json`: small file updated frequently for the dashboard
- `summary.json`: final aggregate metrics

## View dashboard (live)

Serve the folder over HTTP (recommended; `file://` won’t work reliably due to browser CORS):

```bash
python -m http.server 8000
```

Then open either:

- Run-local dashboard: `http://localhost:8000/runs/<run_id>/dashboard.html`
- Global dashboard (auto-detects latest run): `http://localhost:8000/dashboard.html`
- Or specify a run explicitly: `http://localhost:8000/dashboard.html?run=runs/<run_id>`


