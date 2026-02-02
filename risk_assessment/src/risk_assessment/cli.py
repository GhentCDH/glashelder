import click

@click.group()
def risk_assessment():
    pass


@risk_assessment.command()
@click.option("--option", default="change me", help="change this help message")
def command(option):
    pass
