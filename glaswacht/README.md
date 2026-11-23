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
