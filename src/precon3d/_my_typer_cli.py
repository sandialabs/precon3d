"""
This module provides a custom command-line interface (CLI) toolkit,
utilizing the Typer and Click libraries for enhanced user interaction
and command management.
"""

import sys
from enum import StrEnum
from typing import Final

import click
import typer
from rich.console import Console
from rich.table import Table
from rich import box


LOGO_TEXT: Final = r"""
        +-------------+
       /             /|
      /             / |
     +-------------+  |    PRECON3D:
     |   slice 1   |  |    (PRE)pare and re(CON)struct 
     |      2      |  |    serial sectioning montage data 
     |      3      |  |    into 3D volume 🧊
     |     ...     |  +
     |   slice n   | /
     +-------------+/

"""
console = Console()


class Color(StrEnum):
    """Colors for custom typer formmatting

    full list of colors for customization here:
    https://rich.readthedocs.io/en/stable/appendix/colors.html
    """

    FRONT_MATTER: str = "dark_sea_green1"
    MODULE: str = "magenta"
    COMMAND: str = "orange1"
    ERROR: str = "red"
    STATEMENT: str = "white"
    CURRENT_LEVEL: str = "yellow"
    HEADER: str = "bold cyan"
    PARAMETER: str = "green"
    TYPE: str = "blue"
    REQUIRED: str = "yellow"
    DEAFULT: str = "magenta"
    SHORT_HELP: str = "white"


class CustomCLIGroup(typer.core.TyperGroup):
    """
    CustomCLIGroup is an enhanced command group for module-level help
    and error handling in the CLI.

    This class extends Typer's base group functionality to provide a
    rich, color-coded help and usage display using the Rich library. It
    is designed to improve user interaction by offering:

      1. A dynamic "Usage:" line that adapts based on the current
         command context.
      2. Display of a short help message (if available) for the current
         command group.
      3. A formatted table listing subcommands (or modules) with
         color-coded entries:
           - Module names are shown in a designated module color.
           - Commands are shown in a designated command color.
      4. Full help documentation (docstring) display when available.

    Additional Features:
      - When the main command is invoked without additional subcommands,
        an ASCII art logo is printed to brand the CLI.
      - The resolve_command method is overridden to catch cases where a
        user supplies a non-existent command or module. In such cases,
        it prints a descriptive error message, indicates available
        options, and then displays the group help.
      - Overrides several methods (such as format_help, get_help, main)
        to integrate rich formatting and improved error management.

    Methods:
      list_commands(ctx):
          Returns a sorted list of available command names.
      format_help(ctx, formatter):
          Displays the help message for the current group.
      get_help(ctx):
          Displays help and exits the program.
      show_help(ctx):
          Formats and prints the usage, short help, subcommand table,
          and full help for the group.
      format_exception(ctx, exc):
          Overrides exception formatting (returns an empty string, as
          errors are handled separately).
      resolve_command(ctx, args):
          Attempts to resolve a command; on failure, prints an error
          message, shows available options, and exits.
      main(*args, **kwargs):
          Executes the main command loop with standalone mode disabled
          to allow error bubbling.
    """

    def list_commands(self, _ctx):
        """List all the commands (functions) added to the module"""
        return sorted(self.commands.keys())

    def format_help(self, ctx, _formatter):
        """Displays the help message for the current group."""
        self.show_help(ctx)

    def get_help(self, ctx):
        """Displays help and exits the program to handle errors."""
        self.show_help(ctx)
        ctx.exit()
        return ""

    def show_help(self, ctx: click.Context):
        """Custom show help function displaying table for groups"""
        # 1. Print usage header.
        # Determine the usage line based on the current command path:
        if ctx.command_path.strip() == "precon3d":
            console.print(
                f"[{Color.FRONT_MATTER}]"
                f"{LOGO_TEXT}"
                f"[/{Color.FRONT_MATTER}]"
            )
            # If only "precon3d" is provided, show [MODULE]:
            usage_line = (
                f"[{Color.CURRENT_LEVEL}]precon3d[/{Color.CURRENT_LEVEL}] "
                f"[[{Color.MODULE}]MODULE[/{Color.MODULE}]] "
                f"[[{Color.COMMAND}]COMMAND[/{Color.COMMAND}]] "
                f"[[{Color.PARAMETER}]PARAMETERS[/{Color.PARAMETER}]]"
            )
            column_label = "Module"
        else:
            # Otherwise, just append [COMMAND] [OPTIONS]:
            usage_line = (
                f"[{Color.CURRENT_LEVEL}]{ctx.command_path}[/{Color.CURRENT_LEVEL}] "
                f"[[{Color.COMMAND}]COMMAND[/{Color.COMMAND}]] "
                f"[[{Color.PARAMETER}]PARAMETERS[/{Color.PARAMETER}]]"
            )
            column_label = "Command"
        console.print(f"[bold]Usage:[/] {usage_line}\n")

        # 2. Print short help for this group if available.
        short_help = getattr(self, "short_help", "")
        if short_help:
            console.print(short_help + "\n", style=f"{Color.SHORT_HELP}")
        # 3. Build a table listing subcommands.
        table = Table(
            show_header=True, header_style=f"{Color.HEADER}", box=None
        )
        table.add_column(column_label, justify="left")
        table.add_column("Short help", justify="left")
        for name, cmd in self.commands.items():
            # Distinguish modules from commands:
            if isinstance(cmd, typer.core.TyperGroup):
                # For modules, display the name in {Color.MODULE}.
                display_name = f"[{Color.MODULE}]{name}[/{Color.MODULE}]"
            else:
                # For function commands, display the name in {Color.COMMAND}.
                display_name = f"[{Color.COMMAND}]{name}[/{Color.COMMAND}]"
            short = getattr(cmd, "short_help", "") or ""
            table.add_row(display_name, short)
        console.print(table)

    def format_exception(self, _ctx, _exc):
        """Format exceptions generated by typer"""
        # We override this method but do not print anything here.
        return ""

    def resolve_command(self, ctx, args):
        """Attempts to resolve a command; on failure, prints an error
        message, shows available options, and exits."""
        try:
            return super().resolve_command(ctx, args)
        except click.exceptions.UsageError as e:
            # Print error message.
            console.print(f"Error: {e}\n", style=f"{Color.ERROR}")
            # Print a message indicating viable options.
            console.print(
                f"\n[{Color.STATEMENT}]Availble options at this level ([/{Color.STATEMENT}]"
                f"[{Color.CURRENT_LEVEL}]{ctx.command_path}[/{Color.CURRENT_LEVEL}]"
                f"[{Color.STATEMENT}]) are:\n[/{Color.STATEMENT}]"
            )
            # Show this group's help then exit.
            self.show_help(ctx)
            ctx.exit(1)
            return None  # This return is never reached but silences pylint.

    def main(self, *args, **kwargs):
        """Executes the main command loop with standalone mode disabled
        to allow error bubbling."""
        # Force standalone_mode=False so errors bubble up.
        kwargs.setdefault("standalone_mode", False)
        try:
            return super().main(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            # catch broad exceptions
            console.print(f"Error: {e}\n", style=f"{Color.ERROR}")
            sys.exit(1)


class CustomCLICommand(click.Command):
    """
    CustomCLICommand is an enhanced command class for individual CLI
    commands that provides detailed, color-formatted help and error
    handling using the Rich library.

    This class extends Click's Command functionality to:
    - Print a "Usage:" line with the current command path and its parameters.
    - Display the command's short help message (if provided) in a highlighted style.
    - Create a formatted table of parameters, detailing each parameter's:
        - Name
        - Type
        - Required status
        - Default value
        - Associated short help
    - Output the full detailed help (the command's docstring) when available.

    Additionally, CustomCLICommand intercepts missing parameter exceptions
    during argument parsing or command invocation. When such exceptions occur,
    it prints an informative error message, displays the help for the command,
    and exits the execution, ensuring that users are guided toward correct usage.

    Methods:
    - parse_args(ctx, args): Parses command-line arguments and handles missing parameters.
    - invoke(ctx): Invokes the command and handles missing parameters.
    - format_help(ctx, formatter): Displays the help message for the command.
    - get_help(ctx): Displays the help and exits the process.
    - show_help(ctx): Formats and prints:
        - A usage header
        - The command's short help (if available)
        - A detailed table of parameters
        - The full detailed help (if provided)
    - format_exception(ctx, exc): Returns an empty string for higher-level error handling.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("rich_markup_mode", None)
        kwargs.pop("rich_help_panel", None)
        super().__init__(*args, **kwargs)

    def parse_args(self, ctx, args):
        try:
            return super().parse_args(ctx, args)
        except click.MissingParameter as e:
            console.print(f"Error: {e}\n", style=f"{Color.ERROR}")
            self.show_help(ctx)
            ctx.exit(1)

    def invoke(self, ctx):
        try:
            return super().invoke(ctx)
        except click.MissingParameter as e:
            console.print(f"Error: {e}\n", style=f"{Color.ERROR}")
            self.show_help(ctx)
            ctx.exit(1)

    def format_help(self, ctx, _formatter):
        self.show_help(ctx)

    def get_help(self, ctx):
        self.show_help(ctx)
        ctx.exit()

    def show_help(self, ctx: click.Context):
        """Custom show help function displaying table for commands"""
        # Print usage header for the command.
        console.print(
            f"[bold]Usage:[/] "
            f"[{Color.CURRENT_LEVEL}]{ctx.command_path}[/{Color.CURRENT_LEVEL}]  "
            f"[[{Color.PARAMETER}]PARAMETERS[/{Color.PARAMETER}]]\n"
        )

        # Print short help if available.
        short_help = getattr(self, "short_help", "")
        if short_help:
            console.print(short_help + "\n", style=f"{Color.SHORT_HELP}")

        # Build table for parameters with additional detail.
        console.print(f"[{Color.HEADER}][bold]Parameters:[/][/{Color.HEADER}]")
        if self.params:
            for param in self.params:
                param_type = (
                    "Argument"
                    if isinstance(param, click.Argument)
                    else "Option"
                )
                required = "Yes" if param.required else "No"
                default = (
                    str(param.default) if param.default is not None else ""
                )
                # help_text = getattr(param, "help", "") or ""
                names = (
                    ", ".join(param.opts)
                    if isinstance(param, click.Option)
                    else param.name
                )

                # Determine the expected type
                if isinstance(param.type, click.Path):
                    expected_type = "Path"
                elif isinstance(
                    param.type, type
                ):  # Check if it's a built-in type
                    expected_type = param.type.__name__
                else:
                    expected_type = str(
                        param.type
                    )  # Fallback to string representation

                console.print(
                    f"[[{Color.PARAMETER}]{names}[/{Color.PARAMETER}]] ({param_type})"
                )
                console.print(f"  Required: {required}")
                console.print(f"  Default: {default}")
                console.print(f"  Type: {expected_type}")
                # console.print(f"  Help: {help_text}\n")
        else:
            console.print("No parameters.\n")

    def format_exception(self, _ctx, _exc):
        """Format exceptions generated by typer"""
        return ""
