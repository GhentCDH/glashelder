# Glaswacht

Assesses the content safety of images using multiple vision-language models (LLavaGuard, Qwen3, Pixtral, Gemma 3) with majority voting.

## Commands

```bash
# launch the Glaswacht TUI monitor, runs all models except LLavaGuard;
# the image folder can also be pasted in its IMAGES box (tab to edit, esc when done)
uv run glashelder glaswacht tui --path ./images

# run one or more models directly (LLavaGuard is off by default, select it to run it)
uv run glashelder glaswacht run --path ./images --model lavaguard
uv run glashelder glaswacht run --path ./images --model qwen3 --model pixtral

# run the default models plus one that is off by default
uv run glashelder glaswacht run --path ./images --enable lavaguard

# start the validation web interface (port 5003 by default), showing the newest run
uv run glashelder glaswacht serve --images-dir ./images
```

## Output

Every run writes two files to `glaswacht_output/`, named after the date and time it
started: `glaswacht_<YYYYMMDD_HHMMSS>.csv` with the results and a `.log` next to it.
Choose another folder with `--output`, the TUI's OUTPUT box, or `output` under
`[glaswacht]` in `glashelder.toml`.

## Hardware

Runs large VLMs, so a powerful GPU is required.
