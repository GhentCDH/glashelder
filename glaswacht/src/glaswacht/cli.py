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


@glaswacht.group("rerun")
def rerun_group():
    """Rerun demo: replay archive results, probe model internals, or stream a live run."""


def _sink_options(f):
    f = click.option(
        "--out",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Write a .rrd recording here (plus a .rbl blueprint beside it).",
    )(f)
    return click.option(
        "--connect",
        default=None,
        help="Stream to a running viewer, e.g. rerun+http://127.0.0.1:9876/proxy. "
        "Deliberately not read from RERUN_ADDR, which would override --out.",
    )(f)


@rerun_group.command("replay")
@click.option(
    "--results",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Combined results CSV (`model_N_*` or `<model>_*` columns).",
)
@click.option(
    "--images",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder with the assessed scans.",
)
@click.option("--limit", type=int, default=None, help="Only the first N plates.")
@click.option(
    "--thumb", type=int, default=320, show_default=True, help="Thumbnail size in px."
)
@click.option(
    "--quality", type=int, default=65, show_default=True, help="JPEG quality."
)
@click.option(
    "--blur-flagged",
    is_flag=True,
    help="Blur plates the vote flagged (for audiences who should not see them).",
)
@click.option(
    "--archive-total",
    type=int,
    default=None,
    help="Scans in the whole archive, for the coverage line.",
)
@_sink_options
def rerun_replay(
    results,
    images,
    limit,
    thumb,
    quality,
    blur_flagged,
    archive_total,
    out,
    connect,
):
    """Replay a finished run (no GPU) into a recording or a viewer."""
    import logging

    from glaswacht.rerundemo.replay import replay

    logging.basicConfig(level=logging.INFO)
    if bool(out) == bool(connect):
        raise click.UsageError("pass exactly one of --out FILE.rrd or --connect URL")
    replay(
        results,
        images,
        out=out,
        connect=connect,
        thumb=thumb,
        quality=quality,
        limit=limit,
        blur_flagged=blur_flagged,
        archive_total=archive_total,
    )


@rerun_group.command("probe")
@click.option(
    "--images",
    "image_paths",
    multiple=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A plate to probe. Repeat for several (about 25 makes a good demo).",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where traces are saved, for `rerun replay --probe-dir`.",
)
@click.option(
    "--model",
    "model_names",
    multiple=True,
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    default=list(MODEL_REGISTRY.keys()),
    show_default=True,
)
def rerun_probe(image_paths, out_dir, model_names):
    """Run the models with instrumentation (GPU). Greedy decoding, token by token."""
    import logging

    from glaswacht.rerundemo.deepdive import probe

    logging.basicConfig(level=logging.INFO)
    probe(list(image_paths), _model_classes_from_names(model_names), out_dir)


@rerun_group.command("live")
@click.option("--path", required=True, help="Folder or single image to assess.")
@click.option(
    "--model",
    "model_names",
    multiple=True,
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    default=list(MODEL_REGISTRY.keys()),
    show_default=True,
)
@click.option("--blur-flagged", is_flag=True)
@_sink_options
def rerun_live(path, model_names, blur_flagged, out, connect):
    """Run the models for real, streaming each rating into Rerun as it arrives."""
    from glaswacht.orchestrator import Orchestrator
    from glaswacht.rerundemo.observer import RerunObserver

    classes = _model_classes_from_names(model_names)
    orchestrator = Orchestrator(classes)
    observer = RerunObserver(
        tuple(c.name for c in classes),
        out=out,
        connect=connect,
        blur_flagged=blur_flagged,
    )
    orchestrator.add_observer(observer)
    try:
        orchestrator.run_all(path)
    finally:
        observer.close()


@rerun_group.command("pick")
@click.option(
    "--results",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--images",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--count", default=25, show_default=True)
@click.option("--seed", default=7, show_default=True)
def rerun_pick(results, images, count, seed):
    """Print a diverse set of plates for `rerun probe` (one --images flag each)."""
    from glaswacht.rerundemo.data import load_results, pattern, pick_diverse

    for row in pick_diverse(load_results(results), count, seed):
        click.echo(f"--images {images / row.image}   # {pattern(row)}")


@rerun_group.command("demo")
@click.option(
    "--probe-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Output of `rerun probe`.",
)
@click.option(
    "--results",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Archive results CSV: adds each plate's archive verdicts and batch-wide findings.",
)
@_sink_options
def rerun_demo(probe_dir, results, out, connect):
    """Build the demo: the probed plates, generated token by token. No GPU needed."""
    import logging

    from glaswacht.rerundemo.demo import build_demo

    logging.basicConfig(level=logging.INFO)
    if bool(out) == bool(connect):
        raise click.UsageError("pass exactly one of --out FILE.rrd or --connect URL")
    build_demo(probe_dir, results, out=out, connect=connect)
