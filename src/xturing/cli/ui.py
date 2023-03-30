import click

from xturing.ui.playground import Playground


@click.command(name="ui")
def ui_command():
    Playground().launch()
