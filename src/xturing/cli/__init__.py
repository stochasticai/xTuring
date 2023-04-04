import click

from xturing.__about__ import __version__
from xturing.cli.api import api_command
from xturing.cli.chat import chat_command
from xturing.cli.ui import ui_command


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(version=__version__, prog_name="xturing")
@click.pass_context
def xturing(ctx: click.Context):
    click.secho("xTuring\n\n", fg="white", bold=True)


xturing.add_command(chat_command)
xturing.add_command(ui_command)
xturing.add_command(api_command)
