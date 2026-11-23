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

## Hardware

Runs large VLMs, so a powerful GPU is required.
