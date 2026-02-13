import click


@click.group()
def qwenvlm():
    """Generate image descriptions using Qwen Vision-Language models."""
    pass


@qwenvlm.command()
@click.option(
    "-i",
    "--image-path",
    required=True,
    help="This is the path to the image to be processed.",
)
@click.option(
    "-p",
    "--prompt",
    required=True,
    help="This is the prompt to use for generating descriptions.",
)
def describe_image(image_path, prompt):
    """Generate a text description for a single image."""
    from qwenvlm.qwen_3_vl import qwen_use

    results = qwen_use(image_path, prompt, None)
    print(results)


@qwenvlm.command()
@click.option(
    "-i",
    "--input-folder",
    required=True,
    help="This is the path to the folder containing the images to be processed.",
)
@click.option(
    "-o",
    "--output-folder",
    default="guarded_output",
    help="This is the path to the folder where the guarded results will be saved. Default: guarded_output",
)
@click.option(
    "-p",
    "--prompt",
    required=True,
    help="This is the prompt to use for generating descriptions.",
)
@click.option(
    "-t",
    "--tokens",
    default=128,
    help="This is the max amount of tokens to generate. Default: 128",
)
def guard_image_folder(input_folder, prompt, output_folder, tokens):
    """Process a folder of images and save guarded descriptions."""
    from qwenvlm.qwen_guarded_vl import process_and_guard_image_folder as pgif

    pgif(input_folder, prompt, output_folder, tokens)
