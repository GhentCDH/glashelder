# Glaswacht

Assesses the content safety of images using multiple vision-language models (LLavaGuard, Qwen3, Pixtral) with majority voting.

## Commands

```bash
# launch the Glaswacht TUI monitor, runs all three models
uv run glashelder glaswacht tui --path ./images

# run one or more models directly
uv run glashelder glaswacht run --path ./images --model lavaguard
uv run glashelder glaswacht run --path ./images --model qwen3 --model pixtral

# start the validation web interface (port 5003 by default)
uv run glashelder glaswacht serve --images-dir ./images
```

## Hardware

Runs large VLMs, so a powerful GPU is required.

## Rerun demo

Watch the three models write their verdicts token by token, in [Rerun](https://rerun.io).
Each token is highlighted by how sure the model was of it, and beside each model a chart
shows the alternatives it weighed at that step.

```bash
# 1. choose plates covering every verdict pattern (prints --images flags)
uv run glashelder glaswacht rerun pick --results results.csv --images ./assets --count 25

# 2. (GPU, ~5 min for 25 plates) run the models with instrumentation; paste the flags
uv run glashelder glaswacht rerun probe --images a.jpg --images b.jpg --out-dir demo/probe

# 3. (CPU, ~2 min) build the recording, then open it
uv run glashelder glaswacht rerun demo --probe-dir demo/probe --results results.csv --out demo/glaswacht.rrd
rerun demo/glaswacht.rrd
```

Press play (space) on the `deep_tick` timeline. Tabs: token by token, verdicts, what the
models see, policy and prompt, plate index with batch findings, and cost. The probe decodes
greedily, while the archive run samples LlavaGuard, so verdicts can differ; the demo
shows both.

`rerun replay` (archive statistics for a whole results CSV) and `rerun live` (stream a
real run) are also available. `demo/` is gitignored because it holds scan thumbnails.
