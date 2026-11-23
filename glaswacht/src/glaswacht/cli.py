from pathlib import Path

import click

from common.cli import serve_options
from glaswacht.models.safety_model import SafetyModel
from glaswacht.models.utils import WEB_PORT, output_folder

MODEL_REGISTRY: dict[str, type[SafetyModel]] = {}


def _register_defaults() -> None:
    from glaswacht.models.model_Gemma3 import Gemma3
    from glaswacht.models.model_LLavaguard import LavaGuard
    from glaswacht.models.model_Ministral3 import Ministral3
    from glaswacht.models.model_Pixtral import Pixtral
    from glaswacht.models.model_Qwen3 import Qwen3

    for cls in [LavaGuard, Qwen3, Ministral3, Gemma3, Pixtral]:
        MODEL_REGISTRY[cls.name] = cls


_register_defaults()


def _model_classes_from_names(
    model_names: tuple[str, ...], enabled: tuple[str, ...] = ()
) -> list[type[SafetyModel]]:
    """Resolve model names to classes, adding the enabled ones."""
    names = list(dict.fromkeys([*model_names, *enabled]))
    return [MODEL_REGISTRY[name] for name in names]


def model_options(func):
    """Add the --model and --enable options, named as in MODEL_REGISTRY."""
    choices = click.Choice(list(MODEL_REGISTRY.keys()))
    off_by_default = [
        name for name, cls in MODEL_REGISTRY.items() if not cls.enabled_by_default
    ]
    func = click.option(
        "--enable",
        "enabled",
        multiple=True,
        type=choices,
        help="Model(s) to run on top of the --model ones, such as those off by "
        f"default ({', '.join(off_by_default) or 'none'}). Repeat to enable multiple.",
    )(func)
    return click.option(
        "--model",
        "model_names",
        multiple=True,
        type=choices,
        default=[
            name for name, cls in MODEL_REGISTRY.items() if cls.enabled_by_default
        ],
        show_default=True,
        help="Model(s) to run. Repeat to select multiple.",
    )(func)


def output_option(func):
    """Add the --output option: where a run writes its results CSV and log."""
    return click.option(
        "--output",
        "output_dir",
        default="glaswacht_output",
        show_default=True,
        type=click.Path(file_okay=False, path_type=Path),
        callback=lambda ctx, param, value: output_folder(value),
        help="Folder for each run's results CSV and log, named after the run's "
        "date and time; a relative one is in the glashelder folder "
        "[glaswacht.output].",
    )(func)


@click.group()
def glaswacht():
    """Assess image content risk using multiple vision-language models."""
    pass


@glaswacht.command()
@click.option(
    "--path",
    default=None,
    type=click.Path(exists=True),
    help="Image folder to assess; it can also be typed or pasted in the TUI.",
)
@output_option
@model_options
def tui(path, output_dir, model_names, enabled):
    """Launch the Glaswacht TUI monitor."""
    from glaswacht.orchestrator import Orchestrator
    from glaswacht.tui import MonitorApp

    model_classes = _model_classes_from_names(model_names, enabled)
    orchestrator = Orchestrator(model_classes)
    app = MonitorApp(orchestrator, image_path=path, output_dir=output_dir)
    app.run(mouse=False)
    app._stop_webviewer()


@glaswacht.command()
@click.option(
    "--path", required=True, help="Path to image folder or single image file."
)
@output_option
@model_options
def run(path, output_dir, model_names, enabled):
    """Run selected vision-language models on images."""
    from glaswacht.orchestrator import Orchestrator

    model_classes = _model_classes_from_names(model_names, enabled)
    orchestrator = Orchestrator(model_classes)
    if orchestrator.run_all(path, output_dir) is None:
        raise click.ClickException(f"run aborted, see {orchestrator.log_path}")


@glaswacht.command()
@serve_options(WEB_PORT)
@click.option(
    "--csv-path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Results CSV of a run [glaswacht.results]; the newest one in "
    "--output by default.",
)
@output_option
@click.option(
    "--images-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder with the assessed images [glaswacht.images].",
)
def serve(csv_path, output_dir, images_dir, **server):
    """Serve the validation web interface."""
    import logging

    from common import run_app
    from glaswacht.models.utils import latest_results
    from glaswacht.web import create_app

    if csv_path is None:
        csv_path = latest_results(output_dir)
    if csv_path is None:
        raise click.UsageError(
            f"no results csv in {output_dir}: run `glaswacht run` first, "
            "or pass --csv-path"
        )
    if images_dir is None:
        raise click.UsageError(
            "no images dir: pass --images-dir or set glaswacht.images"
        )

    logging.basicConfig(level=logging.INFO)
    run_app(create_app(csv_path, images_dir), **server)
