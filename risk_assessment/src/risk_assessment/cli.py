import click

from risk_assessment.models.safety_model import SafetyModel

MODEL_REGISTRY: dict[str, type[SafetyModel]] = {}


def _register_defaults() -> None:
    from risk_assessment.models.model_LLavaguard import LavaGuard
    from risk_assessment.models.model_Pixtral import Pixtral
    from risk_assessment.models.model_Qwen3 import Qwen3

    for cls in [LavaGuard, Qwen3, Pixtral]:
        MODEL_REGISTRY[cls.name] = cls


_register_defaults()


def _model_classes_from_names(model_names: tuple[str, ...]) -> list[type[SafetyModel]]:
    return [MODEL_REGISTRY[name] for name in model_names]


@click.group()
def risk_assessment():
    """Assess image content risk using multiple vision-language models."""
    pass


@risk_assessment.command()
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
def glaswacht(path, model_names):
    """Launch the Glaswacht TUI monitor."""
    from risk_assessment.glaswacht import MonitorApp
    from risk_assessment.orchestrator import Orchestrator

    model_classes = _model_classes_from_names(model_names)
    orchestrator = Orchestrator(model_classes)
    app = MonitorApp(orchestrator, image_path=path)
    app.run(mouse=False)
    app._stop_webviewer()


@risk_assessment.command()
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
    from risk_assessment.orchestrator import Orchestrator

    model_classes = _model_classes_from_names(model_names)
    orchestrator = Orchestrator(model_classes)
    orchestrator.run_all(path)


@risk_assessment.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind to.")
@click.option("--port", default=5001, show_default=True, help="Port to listen on.")
@click.option(
    "--debug/--no-debug", default=True, show_default=True, help="Enable debug mode."
)
@click.option("--csv-path", default=None, help="Override CSV_PATH from web.py.")
@click.option("--images-dir", default=None, help="Override IMAGES_DIR from web.py.")
def serve(host, port, debug, csv_path, images_dir):
    """Serve the validation web interface."""
    import logging
    import os

    if csv_path is not None:
        os.environ["RA_CSV_PATH"] = csv_path
    if images_dir is not None:
        os.environ["RA_IMAGES_DIR"] = images_dir
    else:
        click.echo(
            "Warning: no --images-dir set, using default path from web.py.", err=True
        )

    from risk_assessment.web import app

    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        f"Starting validation server on http://{host}:{port}"
    )
    app.run(host=host, port=port, debug=debug)
