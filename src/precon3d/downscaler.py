"""Downscale images"""

# Default python modules
import numpy as np
from pathlib import Path
import math
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
)  # Import Rich progress componentsimport yaml  # Assuming the configuration is in YAML format
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import typer
import time
from typing import NamedTuple, Dict, Any

# Local scripts
import precon3d.utility as ut
import precon3d.factory

# pylint: disable=wildcard-import
from precon3d.custom_types import *
from precon3d._my_typer_cli import CustomCLIGroup, CustomCLICommand

# CLI
app = typer.Typer(
    cls=CustomCLIGroup,
    no_args_is_help=True,
    short_help="Downscale images",
)


class DownscalerAttrs(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """

    downscale_factor: float = None
    downscale_tolerance: float = None
    image_limit_factor: float = None


def create_downscaler_attrs(data: Dict[str, Any]) -> DownscalerAttrs:
    """Create DownscalerAttrs from a dictionary."""
    return DownscalerAttrs(
        downscale_factor=data["downscaler_attrs"]["resolution_output"][
            "inplane"
        ]
        / data["downscaler_attrs"]["resolution_input"]["inplane"],
        downscale_tolerance=data["downscaler_attrs"]["downscale_tolerance"],
        image_limit_factor=data["downscaler_attrs"]["image_limit_factor"],
    )


def downscale_image_stack(config):
    """Downscale a stack of images based on the provided configuration.

    Args:
        config (dict): Configuration parameters including image directory, output directory,
                       and downscaling settings.
    """

    general_attrs = precon3d.factory.create_general_attrs(config)
    downscaler_attrs = create_downscaler_attrs(config)

    # Get the list of images to process
    input_image_list = ut.sorted_files(
        general_attrs.input_directory, general_attrs.file_extension
    )
    print(
        f"found {len(input_image_list)} images in {general_attrs.input_directory}"
    )

    downscaled_output_dir = general_attrs.output_directory
    if downscaled_output_dir.is_dir() and any(downscaled_output_dir.iterdir()):
        downscaled_filepaths = ut.sorted_files(
            directory=general_attrs.output_directory,
            file_extension=general_attrs.file_extension,
        )

        filtered_image_list = ut.remove_matching_filenames(
            input_image_list, downscaled_filepaths
        )
    else:
        filtered_image_list = input_image_list

        # Create the target directory if it doesn't exist
        downscaled_output_dir.mkdir(parents=True, exist_ok=True)

    new_width, new_height = find_new_image_dimensions(
        filtered_image_list[0], downscaler_attrs
    )
    print(f"scaling images by {downscaler_attrs.downscale_factor:.2f}")

    # Use Rich's Progress for displaying progress
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("[bold blue]{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task(
            "Processing Images...", total=len(filtered_image_list)
        )

        # Process each image in the list
        for image_filepath in filtered_image_list:
            # if counter % 50 == 0:
            #     print(f"\t\tDownscaling image {counter + 1} of {len(image_list)}")
            relative_path = image_filepath.parent.relative_to(
                general_attrs.input_directory
            )
            downscale_target_dir = downscaled_output_dir.joinpath(
                relative_path
            )

            downscale_target_dir.mkdir(parents=True, exist_ok=True)

            downscale_image(
                image_filepath,
                scale_factor=downscaler_attrs.downscale_factor,
                new_width=new_width,
                new_height=new_height,
                downscale_target_dir=downscale_target_dir,
            )
            progress.update(task, advance=1)  # Update the progress bar


def find_new_image_dimensions(test_img_filepath: Path, attrs: DownscalerAttrs):
    """Find the new dimensions for downscaling an image.

    Args:
        test_img_filepath (Path): The path to the test image.
        attrs (DownscalerAttrs): Configuration parameters including resolution settings.

    Returns:
        tuple: new width, new height.
    """

    with Image.open(test_img_filepath) as img:
        width, height = img.size

    new_width = calculate_new_dimension(
        width,
        attrs.downscale_factor,
        attrs.downscale_tolerance,
        attrs.image_limit_factor,
        attrs.image_limit_factor,
    )
    new_height = calculate_new_dimension(
        height,
        attrs.downscale_factor,
        attrs.downscale_tolerance,
        attrs.image_limit_factor,
        attrs.image_limit_factor,
    )

    return new_width, new_height


def calculate_new_dimension(
    size: int,
    downscale_factor: float,
    downscale_tolerance: float,
    min_size_factor: float,
    max_size_factor: float,
) -> int:
    """Calculate new dimension that meets downscaling criteria.

    Args:
        size (int): The original size (width or height).
        downscale_factor (float): The factor by which to downscale.
        downscale_tolerance (float): The tolerance for downscaling.
        min_size_factor (float): Minimum size factor for downscaling.
        max_size_factor (float): Maximum size factor for downscaling.

    Returns:
        int: The new size that meets the downscaling criteria.
    """
    min_size = size * min_size_factor
    max_size = size * max_size_factor

    def adjust_size(new_size):
        """Adjust the size to find a valid downscaled dimension."""
        while not math.isclose(
            new_size % downscale_factor, 0, abs_tol=downscale_tolerance
        ):
            if new_size < min_size:
                return size  # Return original size if no smaller solution
            new_size -= 1
        return new_size

    new_size = adjust_size(size)
    if new_size == size:  # If no smaller solution found, try increasing
        new_size = size
        while not math.isclose(
            new_size % downscale_factor, 0, abs_tol=downscale_tolerance
        ):
            if new_size > max_size:
                raise ValueError(
                    "Could not find even dimensions within size limit range and downscale tolerance specified. Please relax constraints"
                )
            new_size += 1
    return new_size


def downscale_image(
    image: Path,
    scale_factor: float,
    new_width: int,
    new_height: int,
    downscale_target_dir: Path,
):
    """Downscale a single image and save it to the target directory.

    Args:
        image (Path): The path to the image file.
        width (int): Original width of the image.
        new_width (int): New width after downscaling.
        height (int): Original height of the image.
        new_height (int): New height after downscaling.
        new_slice_num (int): The index for naming the output file.
        downscale_target_dir (str): Directory to save the downscaled image.
        config (dict): Configuration parameters.
    """

    with Image.open(image) as img:
        # width, height = img.size

        # # # Crop original image prior to downscaling
        # left, right = calculate_crop(width, new_width)
        # top, bottom = calculate_crop(height, new_height)

        # cropped_img = img.crop((left, top, right, bottom))

        downscaled_img = img.resize(
            (int(new_width / scale_factor), int(new_height / scale_factor)),
            Image.Resampling.LANCZOS,
        )
        downscaled_img.save(downscale_target_dir.joinpath(image.name))


def calculate_crop(original_size: int, new_size: int):
    """Calculate the crop dimensions.

    Args:
        original_size (int): The original size (width or height).
        new_size (int): The new size after downscaling.

    Returns:
        tuple: crop dimensions for each size (e.g. left or right).
    """

    # crop
    if new_size < original_size:
        left = int(math.floor((original_size - new_size) / 2))
        right = left + new_size
    else:
        left = 0
        right = original_size

    return left, right


@app.command(
    cls=CustomCLICommand,
    name="process_images",
    short_help="downscale images files from user config",
)
def process_images(configfile: Path):
    """
    Downscale image stack using the provided configuration file.

    Parameters
    ----------
    configfile : Path
        The path to the YAML configuration file containing parameters for downscaling images.

    Returns
    -------
    None
        This function does not return a value. It performs image processing and outputs timing information.

    Notes
    -----
    This function reads the configuration from the specified YAML file and uses it to downscale images.
    The total time taken for the operation is printed upon completion.

    Examples
    --------
    >>> process_images(Path("path/to/config.yml"))
    *** Starting precon3d_downscale:process_images, config.yml ***
    total time:
        1969.47 seconds
        32.82 minutes
        0.55 hours

    CLI
    ---
    >>> python -m precon3d.precon3d_downscale process_images "path/to/config.yml"
    """

    # cast to Path object if not already a Path object
    if not isinstance(configfile, Path):
        configfile = Path(configfile)

    # parse and run prep
    config = ut.read_config(configfile)
    ut.current_date_and_time()
    typer.echo(
        f"*** Starting precon3d_downscale:process_images, {configfile.name} *** \n"
    )

    tic = time.perf_counter()

    downscale_image_stack(config)

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
    precon3d.downscaler downscales images.
    """


if __name__ == "__main__":
    app()
