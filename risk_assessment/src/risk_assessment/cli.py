import click


@click.group()
def risk_assessment():
    """Assess image content risk using multiple vision-language models."""
    pass


@risk_assessment.command()
def glaswacht():
    """Launch the Glaswacht TUI monitor."""
    from risk_assessment.glaswacht import MonitorApp
    from risk_assessment.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    app = MonitorApp(orchestrator)
    app.run(mouse=False)
    app._stop_webviewer()


@risk_assessment.command()
@click.option(
    "--path", required=True, help="Path to image folder or single image file."
)
def run_lavaguard(path):
    """Run only the LLavaGuard model."""
    from risk_assessment.models.model_LLavaguard import run_lavaguard as _run
    from risk_assessment.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    _run(orchestrator, path)


@risk_assessment.command()
@click.option(
    "--path", required=True, help="Path to image folder or single image file."
)
def run_qwen(path):
    """Run only the Qwen3 model."""
    from risk_assessment.models.model_Qwen3 import run_qwen as _run
    from risk_assessment.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    _run(orchestrator, path)


@risk_assessment.command()
@click.option(
    "--path", required=True, help="Path to image folder or single image file."
)
def run_pixtral(path):
    """Run only the Pixtral model."""
    from risk_assessment.models.model_Pixtral import run_pixtral as _run
    from risk_assessment.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    _run(orchestrator, path)


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
