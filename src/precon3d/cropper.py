import os
import time
import numpy as np
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from typing import NamedTuple, Dict, Any
import typer
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
)  # Import Rich progress componentsimport yaml  # Assuming the configuration is in YAML format


import precon3d.utility as ut
import precon3d.factory

# pylint: disable=wildcard-import
from precon3d.custom_types import *
from precon3d._my_typer_cli import CustomCLIGroup, CustomCLICommand

# CLI
app = typer.Typer(
    cls=CustomCLIGroup,
    no_args_is_help=True,
    short_help="Crop images to regions of interest",
)


class CropperAttrs(NamedTuple):
    """Represents a rectangular subarea defined by its position and dimensions.

    Attributes:
        subarea_x (int): The x-coordinate of the top-left corner of the subarea.
        subarea_y (int): The y-coordinate of the top-left corner of the subarea.
        subarea_width (int): The width of the subarea.
        subarea_height (int): The height of the subarea.
    """

    subarea_x: int = None
    subarea_y: int = None
    subarea_width: int = None
    subarea_height: int = None


def create_cropper_attrs(data: Dict[str, Any]) -> CropperAttrs:
    """Create CropperAttrs from a dictionary."""
    return CropperAttrs(
        subarea_x=int(data["cropper_attrs"]["subarea_x"]),
        subarea_y=int(data["cropper_attrs"]["subarea_y"]),
        subarea_width=int(data["cropper_attrs"]["subarea_width"]),
        subarea_height=int(data["cropper_attrs"]["subarea_height"]),
    )


def crop_image_stack(config: Dict):
    """crop a stack of images based on the provided configuration.

    Args:
        config (dict): Configuration parameters including image directory, output directory,
                       and downscaling settings.
    """

    general_attrs = precon3d.factory.create_general_attrs(config)
    crop_attrs = create_cropper_attrs(config)
    scale_factor = float(config["scale_factor"])

    output_dir = general_attrs.output_directory.joinpath("Cropped")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the list of images to process
    image_list = ut.sorted_files(
        general_attrs.input_directory, general_attrs.file_extension
    )
    print(f"Found {len(image_list)} images in {general_attrs.input_directory}")

    if config["start_slice"] is None:
        user_image_range_start = 0
    else:
        user_image_range_start = int(config["start_slice"])
    if config["end_slice"] is None:
        user_image_range_end = len(image_list)
    else:
        user_image_range_end = int(config["end_slice"])

    user_image_list = image_list[user_image_range_start:user_image_range_end]

    # Use Rich's Progress for displaying progress
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("[bold blue]{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task(
            "Cropping Images...", total=len(user_image_list)
        )

        # Process each image in the list
        for image_file in user_image_list:
            # for counter, image_file in enumerate(image_list):
            image_filepath_out = output_dir.joinpath(image_file.name)
            # if counter % 10 == 0:
            #     print(f"\t\tCropping image {counter + 1} of {len(image_list)}")
            with Image.open(image_file) as image_input:
                image_cropped = crop_image(
                    image_input, crop_attrs, scale_factor
                )
                image_cropped.save(image_filepath_out)
            progress.update(task, advance=1)  # Update the progress bar


def crop_image(
    img: Image.Image, cropper_attrs: CropperAttrs, scale_factor: float = 1.0
):
    """_summary_

    Args:
        img (Image.Image): _description_
        cropper_attrs (CropperAttrs): _description_
        scale_factor (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    #    with Image.open(img_filepath) as img:

    # Define the cropping box
    crop_box = (
        cropper_attrs.subarea_x,
        cropper_attrs.subarea_y,
        cropper_attrs.subarea_x + cropper_attrs.subarea_width,
        cropper_attrs.subarea_y + cropper_attrs.subarea_height,
    )
    img_roi = img.crop(crop_box)

    if scale_factor != 1.0:

        scaled_width = int(cropper_attrs.subarea_width * scale_factor)
        scaled_height = int(cropper_attrs.subarea_height * scale_factor)

        img_roi = img_roi.resize((scaled_width, scaled_height))

    return img_roi


@app.command(
    cls=CustomCLICommand,
    name="process_images",
    short_help="crop images from user config",
)
def process_images(configfile: Path):
    """
    crop images from user config
    """

    # cast to Path object if not already a Path object
    if not isinstance(configfile, Path):
        configfile = Path(configfile)

    # read yaml content
    config_dict = ut.read_config(configfile)

    ut.current_date_and_time()
    typer.echo(
        f"*** Starting precon3d_downscale:process_images, {configfile.name} *** \n"
    )

    tic = time.perf_counter()

    crop_image_stack(config_dict)

    toc = time.perf_counter()

    typer.echo(
        f"""total time: 
        {toc - tic:0.2f} seconds 
        {(toc - tic)/60:0.2f} minutes 
        {(toc - tic)/3600:0.2f} hours"""
    )


@app.callback()
def callback():
    """
    precon3d.cropper crops images to user defined regions of interest
    """


if __name__ == "__main__":
    app()
