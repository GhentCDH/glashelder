import click

@click.group()
def qwenvlm():
    pass

from qwenvlm.qwen import Qwen_use

@qwenvlm.command()
def command():
    Qwen_use("/media/lbth/Disk/1c1tf36b1k_tiff.jpg", "Give a detailed description of this image.")
