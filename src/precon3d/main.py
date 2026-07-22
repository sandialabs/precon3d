import typer

import precon3d.aligner, precon3d.shading_correction, precon3d.stitcher, precon3d.post_stitch_utils
import precon3d.downscaler, precon3d.cropper, precon3d.resampler, precon3d.utility, precon3d.czi_info

from precon3d._my_typer_cli import Color, CustomCLIGroup

# app = typer.Typer(no_args_is_help=True)
app = typer.Typer(
    cls=CustomCLIGroup,
    no_args_is_help=True,
    add_completion=False,
    short_help=(
        f"[{Color.FRONT_MATTER}]"
        "PRECON3D: (PRE)pare and re(CON)struct serial sectioning montage data into 3D volume 🧊"
        f"[/{Color.FRONT_MATTER}]"
    ),
)

# Add subcommands to the routines group
app.add_typer(precon3d.czi_info.app, name="czi_info")
app.add_typer(precon3d.shading_correction.app, name="shading_correction")
app.add_typer(precon3d.stitcher.app, name="stitcher")
app.add_typer(precon3d.post_stitch_utils.app, name="post_stitch")
app.add_typer(precon3d.aligner.app, name="aligner")
app.add_typer(precon3d.cropper.app, name="cropper")
app.add_typer(precon3d.downscaler.app, name="downscaler")
app.add_typer(precon3d.resampler.app, name="resampler")
app.add_typer(precon3d.utility.app, name="utility")


@app.callback()
def callback():
    """
    PRECON3D: (PRE)pare and re(CON)struct serial sectioning montage data into 3D volume 🧊
    """


if __name__ == "__main__":
    app()
