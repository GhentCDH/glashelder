import click

IMAGE_DIR = "/mnt/UGent_Share/glashelder/assets"

@click.group()
def classification():
    pass


@classification.command()
@click.option("--image_dir", default=IMAGE_DIR, help="Directory where the images reside")
def classify(option):
    pass
