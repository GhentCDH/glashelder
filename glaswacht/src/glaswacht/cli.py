import click

from glaswacht.models.safety_model import SafetyModel

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
    app.stop_viewers()


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
@click.option(
    "--image",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Single image to trace generation for.",
)
@click.option(
    "--model",
    "model_names",
    multiple=True,
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    default=list(MODEL_REGISTRY.keys()),
    show_default=True,
    help="Model(s) to diagnose. Repeat to select multiple.",
)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    help="Number of alternative tokens to record per generation step.",
)
@click.option(
    "--max-new-tokens",
    default=None,
    type=int,
    help="Override the generation length each model uses by default.",
)
@click.option(
    "--out-dir",
    default="diagnostics",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory the JSON traces are written to.",
)
@click.option(
    "--serve/--no-serve",
    default=True,
    show_default=True,
    help="Serve the diagnostics viewer once the traces are written.",
)
@click.option(
    "--port", default=5002, show_default=True, help="Port for the diagnostics viewer."
)
def diagnose(image, model_names, top_k, max_new_tokens, out_dir, serve, port):
    """Trace token-level generation for one image across the selected models."""
    import os

    from glaswacht.orchestrator import Orchestrator

    model_classes = _model_classes_from_names(model_names)
    orchestrator = Orchestrator(model_classes)
    traces = orchestrator.run_diagnostics(
        image, out_dir, top_k=top_k, max_new_tokens=max_new_tokens
    )
    for trace in traces:
        click.echo(f"Wrote {trace}")

    if not serve:
        return
    if not traces:
        click.echo("No traces written, not serving the viewer.", err=True)
        return

    os.environ["GW_DIAG_DIR"] = os.path.abspath(out_dir)
    os.environ["GW_DIAG_PORT"] = str(port)

    from glaswacht.diagnostics_web import main as serve_diagnostics

    click.echo(f"Serving diagnostics on http://localhost:{port}")
    serve_diagnostics()


@glaswacht.command("diagnose-tui")
@click.option(
    "--model",
    "model_names",
    multiple=True,
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    default=list(MODEL_REGISTRY.keys()),
    show_default=True,
    help="Model(s) preselected at launch. Toggle inside the TUI to change.",
)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    help="Initial number of alternative tokens to record per generation step.",
)
@click.option(
    "--max-new-tokens",
    default=None,
    type=int,
    help="Initial override of each model's default generation length.",
)
@click.option(
    "--out-dir",
    default="diagnostics",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Initial directory the JSON traces are written to.",
)
@click.option(
    "--port",
    default=5002,
    show_default=True,
    help="Initial port for the diagnostics viewer.",
)
def diagnose_tui(model_names, top_k, max_new_tokens, out_dir, port):
    """Launch the interactive diagnostics TUI (per-token generation traces)."""
    from glaswacht.diagnostics_tui import DiagnosticsApp

    app = DiagnosticsApp(
        model_registry=MODEL_REGISTRY,
        initial_model_names=model_names,
        initial_top_k=top_k,
        initial_max_new_tokens=max_new_tokens,
        initial_out_dir=out_dir,
        initial_port=port,
    )
    app.run(mouse=False)
    app.stop_viewers()


@glaswacht.command()
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

    from glaswacht.web import app

    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        f"Starting validation server on http://{host}:{port}"
    )
    app.run(host=host, port=port, debug=debug)
