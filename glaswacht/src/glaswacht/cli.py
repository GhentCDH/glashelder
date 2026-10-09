from pathlib import Path

import click

from common.cli import serve_options
from glaswacht.models.safety_model import SafetyModel
from glaswacht.models.utils import WEB_PORT

MODEL_REGISTRY: dict[str, type[SafetyModel]] = {}


def _register_defaults() -> None:
    from glaswacht.models.model_LLavaguard import LavaGuard
    from glaswacht.models.model_Pixtral import Pixtral
    from glaswacht.models.model_Qwen3 import Qwen3

    for cls in [LavaGuard, Qwen3, Pixtral]:
        MODEL_REGISTRY[cls.name] = cls


_register_defaults()


def _model_classes_from_names(model_names: tuple[str, ...]) -> list[type[SafetyModel]]:
    return [MODEL_REGISTRY[name] for name in model_names]


@click.group()
def glaswacht():
    """Assess image content risk using multiple vision-language models."""
    pass


@glaswacht.command()
@click.option(
    "--path",
    default=None,
    type=click.Path(exists=True),
    help="Path to image folder to assess.",
)
@click.option(
    "--model",
    "model_names",
    multiple=True,
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    default=list(MODEL_REGISTRY.keys()),
    show_default=True,
    help="Model(s) to run. Repeat to select multiple.",
)
def tui(path, model_names):
    """Launch the Glaswacht TUI monitor."""
    from glaswacht.orchestrator import Orchestrator
    from glaswacht.tui import MonitorApp

    model_classes = _model_classes_from_names(model_names)
    orchestrator = Orchestrator(model_classes)
    app = MonitorApp(orchestrator, image_path=path)
    app.run(mouse=False)
    app._stop_webviewer()


@glaswacht.command()
@click.option(
    "--path", required=True, help="Path to image folder or single image file."
)
@click.option(
    "--model",
    "model_names",
    multiple=True,
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    default=list(MODEL_REGISTRY.keys()),
    show_default=True,
    help="Model(s) to run. Repeat to select multiple.",
)
def run(path, model_names):
    """Run selected vision-language models on images."""
    from glaswacht.orchestrator import Orchestrator

    model_classes = _model_classes_from_names(model_names)
    orchestrator = Orchestrator(model_classes)
    orchestrator.run_all(path)


@glaswacht.command()
@serve_options(WEB_PORT)
@click.option(
    "--csv-path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Combined results CSV written by `glaswacht run` [glaswacht.results].",
)
@click.option(
    "--images-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder with the assessed images [glaswacht.images].",
)
def serve(csv_path, images_dir, **server):
    """Serve the validation web interface."""
    import logging

    from common import run_app
    from glaswacht.web import create_app

    if csv_path is None:
        raise click.UsageError(
            "no results csv: pass --csv-path or set glaswacht.results"
        )
    if images_dir is None:
        raise click.UsageError(
            "no images dir: pass --images-dir or set glaswacht.images"
        )

    logging.basicConfig(level=logging.INFO)
    run_app(create_app(csv_path, images_dir), **server)
