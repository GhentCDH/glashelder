# Risk Assessment

Assesses the content safety of images using multiple vision-language models (LLavaGuard, Qwen3, Pixtral) with majority voting.

## Commands

```bash
# launch the Glaswacht TUI monitor — runs all three models
uv run glashelder risk-assessment glaswacht --path ./images

# run a single model
uv run glashelder risk-assessment run-lavaguard --path ./images
uv run glashelder risk-assessment run-qwen --path ./images
uv run glashelder risk-assessment run-pixtral --path ./images

# start the validation web interface
uv run glashelder risk-assessment serve --images-dir ./images
```

## Hardware

Runs large VLMs, so a powerful GPU is required.
