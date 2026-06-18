from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

from preprocessing.bounding import _find_bounding_box, save_bounding_boxes


@click.group()
def preprocessing():
    """Prepare datasets and train YOLO models for object detection."""
    pass


@preprocessing.command()
@click.option(
    "-g",
    "--ground-truth-dir",
    required=True,
    help="This is the path to the directory containing the GT output from e.g. LabelStudio in YOLO-format.",
)
@click.option(
    "-o",
    "--output-folder",
    default="dataset_output",
    help="This is the path to the folder where the YOLO dataset split will be saved. Default: dataset_output",
)
@click.option(
    "-t",
    "--train-split",
    default=0.8,
    help="Set this float to determine the size of the train and test set. Default: 0.8 (80-20)",
)
@click.option(
    "-s",
    "--seed",
    default=1,
    help="This is the seed used to randomize the split test and train. Default: 1",
)
def create_yolo_dataset(ground_truth_dir, output_folder, train_split, seed):
    """Create a YOLO dataset from LabelStudio annotations."""
    from .dataset_prep import create_yolo_dataset as cyd

    cyd(ground_truth_dir, output_folder, train_split, seed)


@preprocessing.command()
@click.option(
    "-m",
    "--model",
    default="yolo11m.pt",
    help="This is the base yolo model from which fine-tuning is happening. Default: yolo11m.pt",
)
@click.option(
    "-d",
    "--data-yaml",
    required=True,
    help="This is the path to the data_yaml file generated in the GT.",
)
@click.option(
    "-e",
    "--epochs",
    default=30,
    help="This is the amount of epochs the fine-tuning goes through (depends on the model). Default: 30",
)
@click.option(
    "-p",
    "--project-folder",
    required=True,
    help="This is the path to where the project should reside",
)
@click.option(
    "-n", "--project-name", default="modeltje", help="This is the name of the model."
)
def train_yolo_model(model, data_yaml, epochs, project_folder, project_name):
    """Fine-tune a YOLO model on a custom dataset."""
    from .yolo_train import train_yolo_model as tym

    tym(model, data_yaml, epochs, project_folder, project_name)


@preprocessing.command()
@click.option(
    "-m",
    "--model-path",
    required=True,
    help="This is the path to the trained YOLO model.",
)
@click.option(
    "-i",
    "--image-folder",
    required=True,
    help="This is the path to the folder containing images to be cropped.",
)
@click.option(
    "-o",
    "--output-folder",
    required=True,
    help="This is the path to the folder where cropped images will be saved.",
)
def preprocessing_on_image(model_path, image_folder, output_folder):
    """Detect and crop objects from images using a trained YOLO model."""
    from .yolo_cropper import run_crop_on_batch_of_images as rcbi

    rcbi(model_path, image_folder, output_folder)


@preprocessing.command()
@click.option(
    "-i",
    "--image-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where the uncropped images are located",
)
@click.option(
    "-c",
    "--crops-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where the cropped images are located",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Json path where output bounding boxes will be saved to",
)
def bounding_box(image_dir: Path, crops_dir: Path, output: Path):
    """Generate a JSON of bounding boxes for all image -> crop pairs.

    Assumes that all images in `images_dir` have a corresponding image in `crops_dir` with the same filename.
    """

    import logging

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
    )

    log = logging.getLogger(__name__)

    bounding_boxes = {}

    img_paths = list(image_dir.glob("*.jpg"))

    log.info(f"total {len(img_paths)} to check")

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_find_bounding_box, p, crops_dir): p for p in img_paths
        }
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task(
                    "finding bounding boxes from crops", total=len(img_paths)
                )

                for future in as_completed(futures):
                    stem, box, messages = future.result()
                    for level, msg in messages:
                        getattr(log, level)(msg)
                    if box is not None:
                        bounding_boxes[stem] = box
                    progress.advance(task)
        except KeyboardInterrupt:
            log.warning("interrupted, shutting down workers...")
            executor.shutdown(wait=False, cancel_futures=True)

    save_bounding_boxes(bounding_boxes, output)
