import click


@click.group()
def preprocessing():
    """Prepare datasets and train YOLO models for object detection."""
    pass


@preprocessing.command()
@click.option(
    "-g",
    "--ground-truth-folder",
    required=True,
    help="This is the path to the folder containing the GT output from LabelStudio in YOLO-format.",
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
def create_yolo_dataset(ground_truth_folder, output_folder, train_split, seed):
    """Create a YOLO dataset from LabelStudio annotations."""
    from .dataset_prep import create_yolo_dataset as cyd

    cyd(ground_truth_folder, output_folder, train_split, seed)


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
