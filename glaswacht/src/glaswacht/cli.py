from pathlib import Path

import click

from common.cli import serve_options
from glaswacht.models.safety_model import SafetyModel
from glaswacht.models.utils import WEB_PORT, image_path, output_folder

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


def _resolve_image_path(ctx, param, value):
    if value is None:
        return None
    resolved = image_path(value)
    if resolved is None:
        raise click.BadParameter(
            f"{value} not found, neither from here nor from the glashelder folder"
        )
    return resolved


def path_option(required: bool, help: str):
    """Add the --path option: the images to assess, resolved as `image_path`
    does and checked before any model loads."""
    return click.option(
        "--path",
        required=required,
        default=None,
        callback=_resolve_image_path,
        help=f"{help} A relative path is looked up from here, then from the "
        "glashelder folder.",
    )


@click.group()
def glaswacht():
    """Assess image content risk using multiple vision-language models."""
    pass


@glaswacht.command()
@path_option(
    required=False,
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
    app = MonitorApp(
        orchestrator,
        image_path=str(path) if path else None,
        output_dir=output_dir,
    )
    app.run(mouse=False)
    app._stop_webviewer()


@glaswacht.command()
@path_option(required=True, help="Image folder or single image file to assess.")
@output_option
@model_options
def run(path, output_dir, model_names, enabled):
    """Run selected vision-language models on images."""
    import logging

    from glaswacht.orchestrator import Orchestrator

    # show each image's assessment; `glashelder` sets up logging itself, the
    # bare `python -m glaswacht` does not. Rich renders the bold rating.
    if not logging.getLogger().handlers:
        from rich.logging import RichHandler

        logging.basicConfig(
            level=logging.INFO, format="%(message)s", handlers=[RichHandler()]
        )

    model_classes = _model_classes_from_names(model_names, enabled)
    orchestrator = Orchestrator(model_classes)
    if orchestrator.run_all(str(path), output_dir) is None:
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


# the Label Studio export and the selection made from it, next to each other
_LABELSTUDIO_DIR = Path(__file__).parent.parent / "labelstudio"


@glaswacht.group()
def labelstudio():
    """Compare runs with the human annotations of a Label Studio export."""


@labelstudio.command("select")
@click.option(
    "--export",
    "export_path",
    default=_LABELSTUDIO_DIR / "project75_export.json",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Label Studio JSON export with the annotations.",
)
@click.option(
    "--count", default=20, show_default=True, help="How many images to select."
)
@click.option(
    "--out",
    "out_dir",
    default=_LABELSTUDIO_DIR / "disagreements",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Folder for the selected images and their reference.csv.",
)
def labelstudio_select(export_path, count, out_dir):
    """Select images where the annotators and the old models disagree.

    Only images all annotators rated the same are used, taken in turn from
    each kind of disagreement (models too strict and too lenient). Writes the
    images to OUT/images, to assess with `glaswacht run --path OUT/images`,
    and the human ratings to OUT/reference.csv.
    """
    from collections import Counter

    from glaswacht.labelstudio import load_tasks, select_disagreements, write_selection

    click.echo(f"reading {export_path} ...")
    tasks = load_tasks(export_path)
    selected = select_disagreements(tasks, count)
    if not selected:
        raise click.ClickException("no images where annotators and models disagree")
    reference = write_selection(selected, out_dir)
    click.echo(f"selected {len(selected)} of {len(tasks)} annotated images:")
    for kind, n in Counter(t.disagreement for t in selected).most_common():
        click.echo(f"  {n:3}  {kind}")
    click.echo(f"images:    {out_dir / 'images'}")
    click.echo(f"reference: {reference}")


@labelstudio.command("compare")
@click.option(
    "--reference",
    "reference_csv",
    default=_LABELSTUDIO_DIR / "disagreements" / "reference.csv",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="reference.csv written by `glaswacht labelstudio select`.",
)
@click.option(
    "--results",
    "results_csv",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Results CSV of a run; the newest one in --output by default.",
)
@output_option
@click.option(
    "--open/--no-open",
    "open_report",
    default=True,
    show_default=True,
    help="Open the HTML report in the browser.",
)
def labelstudio_compare(reference_csv, results_csv, output_dir, open_report):
    """Score a run against the human ratings, next to the old models.

    Prints a summary and writes an HTML report next to the run's CSV, with
    every image, its annotations and each model's rating, categories and
    rationale.
    """
    import webbrowser

    from glaswacht.labelstudio import build_comparison, text_report, write_html_report
    from glaswacht.models.utils import latest_results

    if results_csv is None:
        results_csv = latest_results(output_dir)
    if results_csv is None:
        raise click.UsageError(f"no results csv in {output_dir}: pass --results")
    try:
        comparison = build_comparison(reference_csv, results_csv)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo(text_report(comparison))
    report = write_html_report(
        comparison, results_csv.with_name(f"{results_csv.stem}_comparison.html")
    )
    click.echo(f"\nHTML report: {report}")
    if open_report:
        webbrowser.open(report.resolve().as_uri())
