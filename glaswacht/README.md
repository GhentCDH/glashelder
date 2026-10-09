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
