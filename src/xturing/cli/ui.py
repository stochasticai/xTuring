import click


@click.command(name="ui")
def ui_command():
    from xturing.ui.playground import Playground

    Playground().launch()
