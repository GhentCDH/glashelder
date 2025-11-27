# Glashelder AI Experiments

## CLI

Use the CLI to run experimenst through:

```sh
uv run glashelder <command> [arguments]
```

get an overview of available experiments / commands using `--help`:

```sh
uv run glashelder --help
```

## Adding new commands to the CLI

to add a new subgroup to the CLI, make a `@click.group()` function in your code,
and register the commands in the subgroup in your code. e.g.:

```py
# ./newexperiment/src/newexperiment/cli.py
import click

@click.group()
def newexperiment():
    pass


@newexperiment.command()
@click.option(...)
def experiment_1(...):
    pass
```
