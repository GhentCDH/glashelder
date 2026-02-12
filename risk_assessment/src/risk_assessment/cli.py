import click


@click.group()
def risk_assessment():
    pass


@risk_assessment.command()
def glaswacht():
    """Launch the Glaswacht TUI monitor."""
    from risk_assessment.orchestrator import Orchestrator
    from risk_assessment.glaswacht import MonitorApp

    orchestrator = Orchestrator()
    app = MonitorApp(orchestrator)
    app.run(mouse=False)
    app._stop_webviewer()


@risk_assessment.command()
@click.option("--path", required=True, help="Path to image folder or single image file.")
def run_lavaguard(path):
    """Run only the LLavaGuard model."""
    from risk_assessment.orchestrator import Orchestrator
    from risk_assessment.models.model_LLavaguard import run_lavaguard as _run

    orchestrator = Orchestrator()
    _run(orchestrator, path)


@risk_assessment.command()
@click.option("--path", required=True, help="Path to image folder or single image file.")
def run_qwen(path):
    """Run only the Qwen3 model."""
    from risk_assessment.orchestrator import Orchestrator
    from risk_assessment.models.model_Qwen3 import run_qwen as _run

    orchestrator = Orchestrator()
    _run(orchestrator, path)


@risk_assessment.command()
@click.option("--path", required=True, help="Path to image folder or single image file.")
def run_pixtral(path):
    """Run only the Pixtral model."""
    from risk_assessment.orchestrator import Orchestrator
    from risk_assessment.models.model_Pixtral import run_pixtral as _run

    orchestrator = Orchestrator()
    _run(orchestrator, path)
