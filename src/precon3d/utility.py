"""This module holds utilties that can be reused across other modules, for example the UTC date/time stamp."""

# Default python modules
import re
import glob
from pathlib import Path
import os
import platform
import pytest
import typer
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from concurrent.futures import ThreadPoolExecutor

# import time
import skimage
import numpy as np
import datetime
import yaml
from enum import Enum, IntEnum
from typing import NamedTuple, Final, Dict, Any, List

# pylint: disable=wildcard-import
from precon3d.custom_types import *
from precon3d._my_typer_cli import CustomCLIGroup, CustomCLICommand

# CLI
app = typer.Typer(
    cls=CustomCLIGroup,
    no_args_is_help=True,
    short_help="Utilities for file management",
)


def valid_enum_entry(obj: Any, check_type: Enum) -> bool:
    """
    Determine if an object is a member of an Enum class.

    This function checks if an object is a member of an Enum class.

    Parameters
    ----------
    obj : Any
        The object to check.
    check_type : Enum
        The Enum class to check against.

    Returns
    -------
    bool
        True if the object is a member of the Enum class, False otherwise.
    """
    return obj in check_type._value2member_map_


def user_settings() -> UserSettings:
    """Constructs a UserSettings NamedTuple."""
    fin: Final[str] = "precon3d_user_settings.yml"
    user_home = Path.home()
    pin = user_home.joinpath(fin)
    # assert pin.is_file(), f"File not found: {pin}"

    config_dict = read_config(pin)
    # for k, v in config_dict.items():
    #     print(f"\nChecking key: {k}, value: {v}")
    #     assert Path(v).exists(), f"Does not exist: {v}"

    us = UserSettings(
        fiji_app=Path(config_dict["fiji_app"]),
        home=Path(config_dict["home"]),
        scratch=Path(config_dict["scratch"]),
    )

    return us


# UTILITY TYPES - enumerate these to avoid magic numbers
class RGB(IntEnum):
    """Constant axis ordering in Python for color images
    stored in 3D arrays (rows,cols,channels) ordering.
    The channels of the color are ordered R, G, B."""

    R: int = 0
    G: int = 1
    B: int = 2


# UTILITY FUNCTIONS
def current_date_and_time() -> str:
    """print date and time for logging"""

    now = datetime.now()
    formatted_now = now.strftime("%m/%d/%Y %H:%M:%S")

    print(formatted_now)

    return formatted_now


def natural_key(string_):
    """https://stackoverflow.com/questions/2545532/python-analog-of-phps-natsort-function-sort-a-list-using-a-natural-order-alg"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def convert_gray_to_8bit(image_array, gamma=1.0):
    """
    Convert a NumPy array image to 8-bit grayscale while preserving the histogram and applying gamma correction.
    """
    if image_array.dtype == np.uint16:
        # Normalize the 16-bit array to the range 0-1
        image_array = image_array / 65535.0
    elif image_array.dtype == np.uint32:
        # Normalize the 32-bit array to the range 0-1
        image_array = image_array / 4294967295.0
    elif image_array.dtype == np.float32 or image_array.dtype == np.float64:
        if np.max(image_array) < 256:
            image_array = image_array / 255.0
        elif np.max(image_array) > 255 and np.max(image_array) < 65535:
            image_array = image_array / 65535.0
        elif np.max(image_array) > 65536 and np.max(image_array) < 4294967295:
            image_array = image_array / 4294967295.0
        else:
            image_array = image_array / np.max(image_array)
        # Ensure the float array is in the range 0-1
        image_array = np.clip(image_array, 0, 1)
    else:
        # If the image is already 8-bit, no conversion is needed
        return image_array.astype(np.uint8)

    # Apply histogram equalization
    # image_array = skimage.exposure.equalize_hist(image_array)

    # Apply gamma correction
    if gamma != 1.0:
        image_array = skimage.exposure.adjust_gamma(image_array, gamma)

        # Rescale intensity to use the full range of 8-bit
        p_low, p_high = np.percentile(image_array, (0.01, 99.99))
        image_array = skimage.exposure.rescale_intensity(
            image_array, in_range=(p_low, p_high)
        )

    # Scale to 8-bit
    image_array = (image_array * 255).astype(np.uint8)
    return image_array


def read_config(config_file_path: Path) -> dict:
    """
    Read a YAML configuration file and return its contents as a dictionary.

    Parameters
    ----------
    config_file_path : Path
        The path to the YAML configuration file to be read.

    Returns
    -------
    dict
        A dictionary containing the contents of the YAML file.
        If the file cannot be read or is not valid YAML, returns None.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    Examples
    --------
    >>> config = read_config(Path("config.yml"))
    >>> print(config)
    {'key': 'value'}
    """
    with open(config_file_path, "r", encoding="utf-8") as stream:
        try:
            config_dict = yaml.safe_load(stream)
            # print(config_dict)
        except yaml.YAMLError as exc:
            raise exc
        return config_dict


@app.command(
    cls=CustomCLICommand,
    name="n_files",
    short_help="Get the number of files with the corresponding extension in the directory.",
)
def n_files(directory: Path, file_extension: str):
    """
    how many files of are in the directory with the corresponding file extension
    """

    n_files_found = len(sorted_files(directory, file_extension))
    print(f"Found {n_files_found} {file_extension} files in {directory}")

    return n_files_found


def sorted_files(directory: Path, file_extension: str) -> list[Path]:
    """
    Sort files in a directory using a natural key (ie 1,2,...10,11,12... not 1,10,11,12,...,2...)
    returning list of Path objects

    Args:
        directory: folder containing images
        file_extension: a suffix at the end of a computer file.
            It comes after the period and is usually two to four characters

    Returns:
        file_list: sorted sequence of Path objects of specified file extension

    """

    if directory.exists() is False:
        raise FileNotFoundError(f"Sorry, {directory} not found.")

    file_list = list(directory.rglob(f"*{file_extension}"))
    file_list = [str(item) for item in file_list]
    file_list = sorted(file_list, key=natural_key)
    file_list = [Path(item) for item in file_list]

    return file_list


def remove_matching_filenames(
    filelist_source: list[Path], filelist_dest: list[Path]
) -> list[Path]:
    """
    Removes files from filelist_source that have a matching file name in filelist_dest.

    Parameters:
        filelist_source (List[Path]): List of source file paths.
        filelist_dest (List[Path]): List of destination file paths to compare against.

    Returns:
        List[Path]: A list of file paths from filelist_source with no matching file names in filelist_dest.
    """

    # Create a set of file names (stems) from the destination file list
    dest_filenames = {file.stem for file in filelist_dest}

    # Filter out files from the source list whose names appear in the destination list
    # filtered_filelist = [file for file in filelist_source if file.stem not in dest_filenames]

    # Filter out files from the source list whose names are substrings of any name in the destination list
    filtered_filelist = [
        file
        for file in filelist_source
        if not any(file.stem in dest_name for dest_name in dest_filenames)
    ]

    print(
        f"Filtering the input files for existence is destination directory:\n\t{len(filelist_source)} files in {filelist_source[0].parent}\n\t{len(filelist_dest)} files in {filelist_dest[0].parent}\n\t{len(filtered_filelist)} files for processing after filtering"
    )

    return filtered_filelist


def downselect_paths(path: Path, keyword: str) -> list[Path]:
    """downselect the list of Paths and only
    keep the Paths containing the keyword"""

    # paths = path.rglob("*")  #
    paths = glob.glob(f"{str(path.as_posix())}/**/", recursive=True)

    # downselected_paths = glob.glob(f"{paths}/*{keyword}*/")
    downselected_paths = []
    for each_path in paths:
        glob_path = glob.glob(f"{each_path}/*{keyword}*/")
        if glob_path:
            for path in glob_path:
                downselected_paths.append(Pat)

    # for path in downselected_paths:
    #     path = Pat

    return downselected_paths


def rmdir(directory: Path) -> None:
    """
    Recursively delete a directory and all its contents.
    (credit to: https://stackoverflow.com/questions/13118029/deleting-folders-in-python-recursively/49782093#49782093)


    This function deletes the specified directory and all its contents, including
    subdirectories and files. If the directory does not exist, the function does nothing.

    Parameters
    ----------
    directory : Path
        The path to the directory to be deleted.

    Returns
    -------
    None

    Examples
    --------
    >>> from pathlib import Path
    >>> directory = Path("path/to/directory")
    >>> rmdir(directory)
    """

    if not directory.exists():
        return

    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


# Custom function to convert the string to a list of floats
def string_to_float_list(s):
    """
    Convert a string representation of a list into a list of floats.

    Parameters
    ----------
    s : str
        A string representation of a list of numbers, e.g., "[1.0, 2.5, 3.3]".

    Returns
    -------
    list of float
        A list containing the float values extracted from the input string.

    Examples
    --------
    >>> string_to_float_list("[1.0, 2.5, 3.3]")
    [1.0, 2.5, 3.3]

    >>> string_to_float_list("[4.0, 5.1]")
    [4.0, 5.1]

    >>> string_to_float_list("[]")
    []

    >>> string_to_float_list("[10, 20, 30]")
    [10.0, 20.0, 30.0]
    """
    numbers_str = s.strip("[]").split()
    return [float(num) for num in numbers_str]


def yes_no(question):
    """Simple Yes/No Function."""
    prompt = f"{question} (y/n): "
    ans = input(prompt).strip().lower()
    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return yes_no(question)
    if ans == "y":
        return True
    return False


@app.command(
    cls=CustomCLICommand,
    name="prepend_filenames",
    short_help="Rename files in directory by adding some prefix.",
)
def prepend_filenames(directory: Path, extension: str, prefix: str):
    """Rename files in directory

    CLI
    ---
    >>> python -m precon3d.utility prepend_filenames "path/to/config.yml" .extension 'prefix'
    """

    file_paths = sorted_files(directory=directory, file_extension=extension)

    print(f"Found {len(file_paths)} '{extension}' files")

    for each_file in file_paths:
        # Check if it's a file (not a directory)
        if os.path.isfile(each_file):
            new_file_name = f"{prefix}{each_file.stem}{extension}"
            new_file_path = each_file.parent.joinpath(new_file_name)

            os.rename(each_file, new_file_path)

    print(f"all files prepended with '{prefix}' ")


@app.command(
    cls=CustomCLICommand,
    name="rename_files_sequentially",
    short_help="Rename files in directory sequentially (e.g. 0001.tif, 0002.tif, etc.).",
)
def rename_files_sequentially(directory: Path, extension: str):
    """Rename files in directory starting from 0 to n_files

    e.g. my_file_01.tif, my_file_01.tif,... to
    0001.tif, 0002.tif

    Args:
        directory: _description_
        extension: _description_

    CLI
    ---
    >>> python -m precon3d.utility rename_files_sequentially "path/to/config.yml" .extension
    """

    file_paths = sorted_files(directory=directory, file_extension=extension)

    print(f"Found {len(file_paths)} '{extension}' files")

    for count, each_file in enumerate(file_paths):
        # Check if it's a file (not a directory)
        if os.path.isfile(each_file):
            new_file_name = f"{count:04d}{extension}"
            new_file_path = os.path.join(directory, new_file_name)

            os.rename(each_file, new_file_path)

    print("all files sequentially renamed")


@app.command(
    cls=CustomCLICommand,
    name="check_image_sizes",
    short_help="Check if all images in the directory have the same dimensions.",
)
def check_image_sizes(input_dir: Path) -> tuple[int, int] | None:
    """
    Check if all images in the directory have the same dimensions.

    Args:
        input_dir (Path): The directory containing the images to check.

    Returns:
        tuple[int, int] | None: The dimensions (width, height) if all images have the same size,
                                None if images have different sizes.

    Raises:
        FileNotFoundError: If the input directory does not exist or is not accessible.
        ValueError: If no images are found in the specified directory.
    """
    # Retrieve the list of image file paths with the specified extension
    filepaths_list = sorted_files(input_dir, ".tif")

    # Count the number of images found
    n_images = len(filepaths_list)
    print(f"Found {n_images} images")

    # Helper function to retrieve the size of an image
    def get_image_size(filepath: Path) -> tuple[int, int]:
        with Image.open(filepath) as img:
            return img.size  # Returns (width, height)

    # Use ThreadPoolExecutor to parallelize the retrieval of image sizes
    with ThreadPoolExecutor() as executor:
        all_image_sizes = list(executor.map(get_image_size, filepaths_list))

    # Check if all images have the same size
    first_image_size = all_image_sizes[0]
    all_same_size = all(size == first_image_size for size in all_image_sizes)

    if all_same_size:
        print(f"All images have the same size: {first_image_size}")
        return first_image_size
    else:
        print("Images have different sizes:")
        for idx, size in enumerate(all_image_sizes, start=1):
            print(f"\tImage {idx}: {size}")
        return None


def determine_image_type(file_name: str) -> str:
    """
    Determines whether the image is 'Bright' or 'Dark' based on the filename.

    Parameters:
        file_name: The name of the file.

    Returns:
        str: 'Bright' if the filename contains 'Bright', 'Dark' if it contains 'Dark', or 'Unknown'.
    """
    if "bright" in file_name.lower():
        return "Bright"
    elif "dark" in file_name.lower():
        return "Dark"
    return "Unknown"


@app.command(
    cls=CustomCLICommand,
    name="reorganize_images_by_type",
    short_help="Separating images into 'Bright' and 'Dark' subdirectories",
)
def reorganize_images_by_type(base_dir: Path):
    """
    Reorganizes images by separating them into 'Bright' and 'Dark' subdirectories
    without resizing and without prepending any values to the filenames.

    Parameters:
        base_dir: The base directory containing the images.
    """

    # Iterate through all .tif files in the base directory (not subdirectories)
    for file_path in base_dir.glob("*.tif"):
        # Determine whether the image is 'Bright' or 'Dark'
        image_type = determine_image_type(file_path.name)

        # Skip files with unknown type
        if image_type == "Unknown":
            print(f"Skipping file with unknown type: {file_path}")
            continue

        # Create the subdirectory for the image type (Bright/Dark)
        type_subdir = base_dir / image_type
        type_subdir.mkdir(parents=True, exist_ok=True)

        # Define the new file path in the appropriate subdirectory
        new_file_path = type_subdir / file_path.name

        # Move the file to the new location
        try:
            file_path.rename(new_file_path)
            print(f"Moved: {file_path} to {new_file_path}")
        except FileNotFoundError as e:
            print(f"Error moving {file_path}: {e}")
        except Exception as e:
            print(f"An error occurred while moving {file_path}: {e}")


@app.command(
    cls=CustomCLICommand,
    name="reorganize_images_by_keyword",
    short_help="Reorganize images by creating a subdirectory based on a user-defined keyword found in the filenames",
)
def reorganize_images_by_keyword(base_dir: Path, keyword: str):
    """
    Reorganize images by creating a subdirectory based on a user-defined keyword found in the filenames.

    Parameters:
        base_dir: The base directory containing the images.
        keyword: The keyword to look for in the filenames.
    """
    # Create the subdirectory for the specified keyword
    keyword_subdir = base_dir / keyword
    keyword_subdir.mkdir(parents=True, exist_ok=True)

    # Iterate through all .tif files in the base directory and its subdirectories
    for file_path in base_dir.rglob("*.tif"):
        # Check if the keyword is in the filename
        if keyword in file_path.name:
            # Define the new file path in the appropriate subdirectory
            new_file_path = keyword_subdir / file_path.name

            # Move the file to the new location
            try:
                file_path.rename(new_file_path)
                print(f"Moved: {file_path} to {new_file_path}")
            except FileNotFoundError as e:
                print(f"Error moving {file_path}: {e}")
            except Exception as e:
                print(f"An error occurred while moving {file_path}: {e}")
        else:
            print(
                f"Keyword '{keyword}' not found in filename: {file_path.name}"
            )


def find_missing_files(
    directory: str, pattern: str = r"Slice_(\d+)_.*\.tif"
) -> None:
    """
    Scans a directory for files matching a specific pattern and identifies any missing files in a numeric sequence.

    Parameters:
    directory: The path to the directory containing the files to be checked.
    pattern: The regular expression pattern used to extract the numeric part from the filenames.

    Returns:
    None: Outputs the results directly to the console.
    """

    print(f"Checking {directory} for missing files")
    try:
        # List all files in the directory
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"The directory {directory} does not exist.")
        return
    except PermissionError:
        print(f"Permission denied to access the directory {directory}.")
        return

    # Use regular expression to find numbers after 'Slice_'
    regex = re.compile(pattern)
    numbers: List[int] = []

    for file in files:
        match = regex.search(file)
        if match:
            number = int(
                match.group(1)
            )  # Convert the number part to an integer
            numbers.append(number)

    # Find missing numbers in the sequence
    if numbers:
        start, end = min(numbers), max(numbers)
        full_set = set(range(start, end + 1))
        missing = full_set - set(numbers)
        if missing:
            print("Missing files:")
            for m in sorted(missing):
                print(f"Slice_{m:04d}_...")
        else:
            print("No files are missing.")
    else:
        print("No relevant files found.")


def extract_significant_part(filename: str) -> str:
    """
    Extracts the significant parts of the filename using a regular expression.
    Assumes the significant parts include 'run' and 'Slice_' with the sequence number.

    Parameters:
    filename: The filename from which to extract the significant parts.

    Returns:
    str: The significant parts of the filename.
    """
    # match = re.search(r'(run\d+).*?(Slice_\d+)', filename)
    match = re.search(r"(Slice_\d+)", filename)
    if match:
        # Concatenate the significant parts for comparison
        return "_".join(match.groups())
    return ""


@app.command(
    cls=CustomCLICommand,
    name="compare_directories",
    short_help="Compares files in two directories to identify any missing files",
)
def compare_directories(dir1: Path, dir2: Path) -> None:
    """
    Compares files in two directories and prints out which files are missing in the second directory.

    Parameters:
        dir1 (Path): Path to the first directory.
        dir2 (Path): Path to the second directory.

    Returns:
        None: Outputs the results directly to the console.
    """
    # Ensure the inputs are Path objects
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    # Extract significant parts of filenames for comparison
    files1 = {extract_significant_part(f.name) for f in dir1.glob("*.tif")}
    files2 = {extract_significant_part(f.name) for f in dir2.glob("*.tif")}

    # Find missing files
    missing_files = files1 - files2

    print(
        f"Checking which files ({len(missing_files)}) in \n\t{dir1}\nare missing in \n\t{dir2}"
    )

    if missing_files:
        print(f"\nMissing files in {dir2}:")
        for file in sorted(missing_files):
            print(file)
    else:
        print(f"\nNo files are missing in {dir2}. \n{len(files2)} total files")


def crop_to_dim(image_dir: Path, new_shape: tuple):
    """
    Crop all .tif images in the specified directory evenly to the specified dimensions.

    This function processes all `.tif` image files in the given directory, cropping them
    to the specified width and height. The cropping is performed symmetrically
    from the center of the image, with adjustments for odd dimensions by cropping
    extra pixels from the right and bottom.

    Parameters:
        dir: The directory containing the `.tif` images to be cropped. Must be a valid Path object.
        new_shape (tuple): A tuple specifying the desired dimensions (width, height) for the cropped images.

    Returns:
        None: The function modifies the images in place and saves the cropped versions
        back to the same directory.

    Raises:
        ValueError: If `new_shape` is not a tuple of two positive integers.
        FileNotFoundError: If the specified directory does not exist.
        Exception: If an image file cannot be processed.

    Example:
        >>> from pathlib import Path
        >>> crop_to_dim(Path("/path/to/images"), (200, 200))
        Crops all `.tif` images in the directory "/path/to/images" to 200x200 pixels.

    Notes:
        - This function only processes `.tif` images.
        - Ensure you have write permissions for the directory to save the cropped images.

    """
    # Validate inputs
    if not isinstance(image_dir, Path):
        raise TypeError("The 'image_dir' parameter must be a Path object.")
    if not image_dir.exists():
        raise FileNotFoundError(f"The directory '{image_dir}' does not exist.")

    height, width = new_shape

    # Process each .tif image in the directory
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() == ".tif":
            try:
                with Image.open(image_path) as img:
                    # Get original dimensions
                    img_width, img_height = img.size

                    # Calculate cropping box
                    left = (img_width - width) // 2
                    top = (img_height - height) // 2
                    right = left + width
                    bottom = top + height

                    # Adjust for odd dimensions
                    if img_width % 2 != 0:
                        right -= 1  # Crop extra pixel from the right
                    if img_height % 2 != 0:
                        bottom -= 1  # Crop extra pixel from the bottom

                    # Ensure the crop box is within bounds
                    if (
                        left < 0
                        or top < 0
                        or right > img_width
                        or bottom > img_height
                    ):
                        raise ValueError(
                            f"Cannot crop image {image_path.name} to dimensions {new_shape}."
                        )

                    # Perform cropping
                    cropped_img = img.crop((left, top, right, bottom))

                    # Save the cropped image back to the same file
                    cropped_img.save(image_path)
            except Exception as e:
                print(f"Error processing file {image_path}: {e}")


def image_min_extent(input_dir: Path):
    """Retrieve the minimum image dimensions from a list of file paths."""

    filepaths_list = sorted_files(input_dir, ".tif")

    n_images = len(filepaths_list)
    print(f"Found {n_images} images")

    def get_image_size(filepath):
        with Image.open(filepath) as img:
            return img.size  # Returns (width, height)

    # Use ThreadPoolExecutor to parallelize the retrieval of image sizes
    with ThreadPoolExecutor() as executor:
        all_image_sizes = list(executor.map(get_image_size, filepaths_list))

    # Convert (width, height) to (height, width) and find min dimensions
    all_image_sizes = [(height, width) for width, height in all_image_sizes]
    min_image_extent = np.min(all_image_sizes, axis=0)
    print(f"\tMin extent: {min_image_extent} pixels")

    return tuple(min_image_extent)


def get_unique_lowest_level_subfolder_names(directory):
    """
    Retrieve the unique names of the lowest level subfolders within a specified directory
    and return them as a list.

    This function traverses all subdirectories of the given directory and identifies
    the lowest level subfolders (i.e., those that do not contain any other subdirectories).
    It returns a list of unique names of these lowest level subfolders, ensuring that
    each name appears only once in the list.

    Parameters:
        directory: The path to the directory to search within.

    Returns:
        list: A list of unique subfolder names at the lowest level.

    """

    # Convert the input path to a Path object
    base_path = Path(directory)

    # Set to hold the unique names of the lowest level subfolders
    unique_subfolder_names = set()

    # Verify that the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        print(
            f"The specified path {directory} does not exist or is not a directory."
        )
        return set()

    # Walk through the directory tree using rglob to find all directories
    for path in base_path.rglob("*"):
        # Check if the current path is a directory and does not contain any subdirectories
        if path.is_dir() and not any(p.is_dir() for p in path.iterdir()):
            # Add the name of the directory to the set (automatically handles duplicates)
            unique_subfolder_names.add(path.name)

    # Return the set of unique subfolder names
    return sorted(list(unique_subfolder_names))


@app.command(
    cls=CustomCLICommand,
    name="merge_bf_df",
    short_help="Merge brightfield images and darkfield images",
)
def merge_bf_df(
    bf_dir: Path, df_dir: Path, output_dir: Path, file_extension: str = ".tif"
) -> None:
    """
    Merge brightfield images and darkfield images by taking the max of the pixels.

    Args:
        bf_dir (Path): Directory containing brightfield images.
        df_dir (Path): Directory containing darkfield images.
        output_dir (Path): Directory to save the merged images.
        file_extension (str): File extension of the images (default is '.tif').

    Raises:
        ValueError: If the images in BF and DF directories have mismatched sizes or counts.
    """
    # Check all images have the same size
    bf_sizes = check_image_sizes(bf_dir)
    df_sizes = check_image_sizes(df_dir)

    if bf_sizes != df_sizes:
        raise ValueError(
            f"Image size mismatch: BF directory has size {bf_sizes}, DF directory has size {df_sizes}"
        )

    # Check equivalent number of files
    bf_imgs = sorted_files(bf_dir, file_extension)
    df_imgs = sorted_files(df_dir, file_extension)

    if len(bf_imgs) != len(df_imgs):
        compare_directories(bf_dir, df_dir)
        raise ValueError(
            f"BF directory has {len(bf_imgs)} images, DF directory has {len(df_imgs)} images"
        )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process and merge images
    for bf_img_path, df_img_path in zip(bf_imgs, df_imgs):
        # Open BF and DF images
        with (
            Image.open(bf_img_path) as bf_img,
            Image.open(df_img_path) as df_img,
        ):
            # Convert images to NumPy arrays for pixel-wise operations
            bf_array = np.array(bf_img)
            df_array = np.array(df_img)

            # Take the maximum of each pixel
            merged_array = np.maximum(bf_array, df_array)

            # Convert the merged array back to an image
            merged_img = Image.fromarray(merged_array)

            # Save the merged image to the output directory
            output_file_name = bf_img_path.name  # Use the BF image's filename
            output_file_path = output_dir / output_file_name
            merged_img.save(output_file_path)

    print(f"Merged images saved to {output_dir}")


def run_on_local_machine(func):
    """pytest only on local machine

    Args:
        func (_type_): function to be tested
    """

    def wrapper_func():
        current_machine = platform.uname().node.lower()
        test_machines = ["s1059904"]
        if current_machine not in test_machines:
            pytest.skip("Run on Local Machine Only.")
        func()

    return wrapper_func


@app.callback()
def callback():
    """
    precon3d.utility provides tools for file management.
    """


if __name__ == "__main__":
    app()
