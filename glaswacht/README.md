# Glaswacht

Assesses the content safety of images using multiple vision-language models with majority
voting: Qwen3, Ministral 3 and Gemma 3 by default, LLavaGuard and Pixtral on request.

## Commands

```bash
# launch the Glaswacht TUI monitor, runs Qwen3, Ministral 3 and Gemma 3;
# the image folder can also be pasted in its IMAGES box (tab to edit, esc when done)
uv run glashelder glaswacht tui --path ./images

# run one or more models directly (LLavaGuard and Pixtral are off by default, select them to run them)
uv run glashelder glaswacht run --path ./images --model lavaguard
uv run glashelder glaswacht run --path ./images --model qwen3 --model ministral3

# run the default models plus one that is off by default
uv run glashelder glaswacht run --path ./images --enable lavaguard

# start the validation web interface (port 5003 by default), showing the newest run
uv run glashelder glaswacht serve --images-dir ./images
```

## Output

Every run writes two files to `glaswacht_output/` in the glashelder folder, whichever
folder you start the command from. Both are named after the date and time the run
started: `glaswacht_<YYYYMMDD_HHMMSS>.csv` with the results and a `.log` next to it.
Choose another folder with `--output`, the TUI's OUTPUT box, or `output` under
`[glaswacht]` in `glashelder.toml`; a relative folder is also in the glashelder folder.

## Long runs and HPC jobs

`run` never sends an image to a model that already assessed it. Every assessment
goes, the moment it is made, to a ledger: `ledger.jsonl` in the output folder, one
JSON line per attempt with the rating, categories, rationale, model id, time and HPC
job id. A run that is cut off loses at most the image it was busy with; the next run
continues from there.

```bash
# at most 2000 images not finished earlier, all models unquantized in GPU memory at once
uv run glashelder glaswacht run --path $VSC_DATA/images --load-all --full-precision \
    --max-images 2000 --output $VSC_DATA/glaswacht_output

# one CSV with everything in the ledger, which `serve` then shows as the newest run
uv run glashelder glaswacht export --output $VSC_DATA/glaswacht_output
```

- `--max-images N` takes the first N images (in name order) that some model still has
  to assess; an image counts once, however many models still need it.
- `--load-all` loads every model at once and assesses the images one by one with all of
  them, so a stopped run leaves no image half done. Qwen3, Ministral 3 and Gemma 3
  together take about 34 GB, more than a 4090 holds: there, leave it off and the models
  load one after another as before.
- `--full-precision` loads Gemma 3 in bfloat16 instead of 4-bit, and Ministral 3 from
  Mistral's bfloat16 checkpoint (`Ministral-3-8B-Instruct-2512-BF16`) instead of the FP8
  one. With Qwen3 that is about 56 GB of weights: for a large GPU such as an H100, not a
  4090. Download the BF16 checkpoint beforehand if the compute nodes have no internet.
  Each ledger line records the model id and precision used.
- Images are keyed by their path relative to `--path`, so keep one ledger per image
  folder: pass `--ledger` to give another folder its own. `--recursive` includes
  subfolders.
- A model that raises on an image (out of memory, an unreadable file) is tried again in
  later runs, up to 3 times, then gives up on that image. An answer of `Error` or
  `Unknown` counts as assessed.
- On SIGTERM or SIGUSR1 a run stops after the image at hand and still writes its CSV.
  To get that signal some minutes before the time limit, add
  `#SBATCH --signal=B:USR1@600` to the job script and start the run with `exec uv run ...`,
  so the signal reaches it.

## Comparing with the annotations

`glaswacht labelstudio` scores runs against the human ratings in the Label Studio export
`glaswacht/src/labelstudio/project75_export.json` (not committed). A relative `--path`
is looked up from where you run the command, then from the glashelder folder.

```bash
# pick 20 images where all annotators agree with each other but not with the old models,
# from every kind of disagreement in turn; writes glaswacht/src/labelstudio/disagreements/
uv run glashelder glaswacht labelstudio select --count 20

# assess them, then score that run (the newest in the output folder) against the annotators
uv run glashelder glaswacht run --path glaswacht/src/labelstudio/disagreements/images
uv run glashelder glaswacht labelstudio compare
```

`compare` shows, per model and for the majority vote, how many images it rates as the
annotators did and how many it rates too strictly or too leniently, next to the old models,
followed by a table per image. Pass `--results` to score an older run.

It also writes `<run>_comparison.html` next to the run's CSV and opens it in the browser
(`--no-open` to skip): the same scores, the new majority per kind of old disagreement and
against the annotators, and every image with its annotations and each model's rating,
categories and rationale, filterable to the images a model gets wrong. The page is one
self-contained file with the images embedded, so it can be shared.

## Hardware

Runs large VLMs, so a powerful GPU is required.
