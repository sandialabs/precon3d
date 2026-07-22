"""Resample slices"""

# Default python modules
from enum import Enum
from numpy.typing import NDArray
from typing import NamedTuple, Final, Tuple, List
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from tqdm import tqdm
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from datetime import datetime
import typer

import csv
import os
import re
import sys
import shutil
import time

# 3rd party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from aicspylibczi import CziFile
import seaborn as sns

# Local scripts
import precon3d.czi_info as ci
import precon3d.utility as ut
import precon3d.factory as gen_types

# pylint: disable=wildcard-import
from precon3d.custom_types import *
from precon3d._my_typer_cli import CustomCLIGroup, CustomCLICommand

# CLI
app = typer.Typer(
    cls=CustomCLIGroup,
    no_args_is_help=True,
    short_help="Resample slices",
)

UM_TO_MM: Final[float] = 1.0e-3


class ResamplePlotOptions(NamedTuple):
    """Displays the resample plot with the specified plot options."""

    display: bool
    raw: bool
    skip_removed: bool
    resampled: bool
    fit: bool
    legend: bool
    # segments_legend: bool = False


class ResampleMethod(Enum):
    """Defines the available resample methods as an enumeration."""

    SIMPLE = 1
    ROBUST = 2


# Example using the ResampleMethod enum:
# Person = namedtuple('Person', ['name', 'age', 'method'])
# Andrew = Person("Andrew", 33, ResampleMethod.ROBUST)


def tileregion_support_points(czi_file_path: Path) -> List[SupportPoints]:
    """Extract scene support points data from CZI metadata.

    Args:
        czi_file_path (Path): Path to the `.czi` file.

    Returns:
        List[SupportPoints]: List of scene support points data.
    """
    parent_parent_dir_name = czi_file_path.parent.parent.name
    parent_dir_name = czi_file_path.parent.name
    file_basename = czi_file_path.stem

    czi_fname = f"{parent_parent_dir_name}_{parent_dir_name}_{file_basename}"

    # Remove all whitespace from the string
    czi_fname = re.sub(r"\s+", "", czi_fname)

    # Regular expression to match numerical values
    def zero_pad(match):
        return match.group().zfill(3)  # Pad the matched number with zeros

    # Replace all numerical values with zero-padded versions
    czi_fname_zeropadded = re.sub(r"\d+", zero_pad, czi_fname)

    metadata = CziFile(czi_file_path).meta

    all_scene_sp = []
    for each_scene in metadata.findall(".//TileRegion"):
        if each_scene.find(".//IsUsedForAcquisition").text == "true":
            scene_name = each_scene.attrib["Name"]
            scene_z = each_scene.find(".//Z").text

            sp_list = []
            for each_sp in each_scene.findall(".//SupportPoints//SupportPoint"):
                sp_list.append(
                    SupportPoint(
                        x=float(each_sp.find("X").text),
                        y=float(each_sp.find("Y").text),
                        z=float(each_sp.find("Z").text),
                    )
                )

            all_scene_sp.append(
                SupportPoints(
                    z_height=scene_z,
                    positions=sp_list,
                    czi_fname=czi_fname_zeropadded,
                    scene_name=scene_name,
                )
            )

    return all_scene_sp


def fit_plane(points: list[SupportPoints]) -> tuple[np.ndarray, float]:
    """
    Fits a plane to the given support points and calculates the normal vector and tilt angle.

    Args:
        points (list[tuple[float, float, float]]): List of support points as (x, y, z) tuples.

    Returns:
        tuple[np.ndarray, float]: A tuple containing the unit normal vector (as a NumPy array)
    """
    # Extract coordinates from the support points
    X = np.array([p.x for p in points])
    Y = np.array([p.y for p in points])
    Z = np.array([p.z for p in points])

    # Fit a plane using least squares
    A = np.c_[X, Y, np.ones(X.shape)]
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)

    # Plane equation: Z = C[0]*X + C[1]*Y + C[2]
    normal_vector = np.array([C[0], C[1], -1])
    normal_vector_magnitude = np.linalg.norm(normal_vector)
    normal_vector_unit = normal_vector / normal_vector_magnitude

    # Round the vector to remove extra whitespace and ensure clean formatting
    normal_vector_unit = np.round(normal_vector_unit, decimals=6)

    # Calculate the tilt angle from the z-axis
    z_axis = np.array([0, 0, -1])
    cos_theta = np.dot(normal_vector_unit, z_axis)
    tilt_angle = np.arccos(cos_theta) * (180 / np.pi)  # Convert to degrees

    # Return the unit normal vector and the tilt angle
    return normal_vector_unit, float(tilt_angle)


def fit_plane_and_plot(points: list[SupportPoints]) -> None:
    """
    Fits a plane to the given support points, plots the plane, normal vector, and calculates the tilt angle from the z-axis.

    Args:
        points (list[SupportPoint]): List of support points with x, y, z coordinates.

    Returns:
        None
    """
    # Extract coordinates from the support points
    X = np.array([p.x for p in points])
    Y = np.array([p.y for p in points])
    Z = np.array([p.z for p in points])

    # Fit a plane using least squares
    A = np.c_[X, Y, np.ones(X.shape)]
    C, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)

    # Plane equation: Z = C[0]*X + C[1]*Y + C[2]
    normal_vector = np.array([C[0], C[1], -1])
    normal_vector_magnitude = np.linalg.norm(normal_vector)
    normal_vector_unit = normal_vector / normal_vector_magnitude

    # Calculate the tilt angle from the z-axis
    z_axis = np.array([0, 0, -1])
    cos_theta = np.dot(normal_vector_unit, z_axis)
    tilt_angle = np.arccos(cos_theta) * (180 / np.pi)  # Convert to degrees

    # Create a grid to plot the plane
    x_range = np.linspace(X.min(), X.max(), 10)
    y_range = np.linspace(Y.min(), Y.max(), 10)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    Z_grid = C[0] * X_grid + C[1] * Y_grid + C[2]

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the original points
    ax.scatter(X, Y, Z, color="r", label="Support Points")

    # Plot the fitted plane
    ax.plot_surface(
        X_grid, Y_grid, Z_grid, alpha=0.25, color="b", rstride=100, cstride=100
    )

    # Plot the normal vector
    origin = np.mean(X), np.mean(Y), np.mean(Z)
    ax.quiver(
        *origin,
        *normal_vector_unit,
        length=20,
        color="k",
        label="Normal Vector",
    )

    ax.quiver(*origin, 1, 0, 0, length=100, color="r", label="X")
    ax.quiver(*origin, 0, 1, 0, length=100, color="g", label="Y")
    ax.quiver(*origin, 0, 0, 1, length=100, color="b", label="Z")

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Fitted Plane and Normal Vector")
    ax.legend()

    ax.set_box_aspect((1, 1, 0.1))
    # ax.auto_scale_xyz([0, 500], [0, 500], [0, 0.15])

    # Display the equation of the plane and tilt angle
    print(f"Plane equation: Z = {C[0]:.4f}*X + {C[1]:.4f}*Y + {C[2]:.4f}")
    print(f"Magnitude of tilt from Z-axis: {tilt_angle:.4f} degrees")

    plt.show()


@app.command(
    cls=CustomCLICommand,
    name="main",
    short_help="save support point information",
)
def save_czi_tileregion_support_points(configfile: Path):
    """
    Extracts support points from CZI files in the input directory and saves them to a CSV file.

    Parameters
    ----------
    params : Path to config file with general attrs
    """

    # cast to Path object if not already a Path object
    if not isinstance(configfile, Path):
        configfile = Path(configfile)

    config_dict = ut.read_config(configfile)
    params = gen_types.create_resampler_parameters(config_dict)

    # Validate file extension
    if params.file_extension != ".czi":
        raise ValueError("At this time, only .czi file extension is supported")

    # Input and output directories
    czi_input_directory = params.input_directory
    output_directory = params.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    # Sort the CZI files in the specified input directory
    sorted_czi_files = ut.sorted_files(
        directory=czi_input_directory, file_extension=params.file_extension
    )

    # Prepare the output file path
    output_file_name = re.sub(
        r"\s+",
        "",
        f"{czi_input_directory.parent.parent.name}_{czi_input_directory.parent.name}_{czi_input_directory.name}_support_points.csv",
    )
    output_file_path = output_directory / output_file_name

    print(f"Found {len(sorted_czi_files)} CZI files in {czi_input_directory}\n")
    print(f"Support point information will be saved in {output_file_path}\n")
    print(f'{" Reading the files ":*^28}')

    # Progress bar setup
    total_files = len(sorted_czi_files)
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[bold blue]{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Reading...", total=total_files)

        # Open the CSV file for writing
        with open(output_file_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header row to the CSV file
            csv_writer.writerow(
                [
                    "File Name",
                    "Scene",
                    "z height from metadata",
                    "Support Point #",
                    "Support Point X",
                    "Support Point Y",
                    "Support Point Z",
                    "normal vector",
                    "angle from z",
                ]
            )

            # Process each CZI file
            for each_czi_path in sorted_czi_files:
                try:
                    # Extract support points from the current CZI file
                    tr_sp = tileregion_support_points(each_czi_path)

                    # Write support points to the CSV file
                    for scene_sp in tr_sp:
                        normal_vec, angle_from_z = fit_plane(scene_sp.positions)
                        for idx, each_sp in enumerate(scene_sp.positions, start=1):
                            csv_writer.writerow(
                                [
                                    scene_sp.czi_fname,
                                    scene_sp.scene_name,
                                    scene_sp.z_height,
                                    idx,
                                    each_sp.x,
                                    each_sp.y,
                                    each_sp.z,
                                    normal_vec,
                                    angle_from_z,
                                ]
                            )

                except RuntimeError as error:
                    # Print an error message if the file cannot be processed
                    print(f"{each_czi_path} has an error: {error}")
                    continue
                finally:
                    progress.update(task, advance=1)

    print(f"Support points successfully saved to {output_file_path}")


def plot_support_points(csv_file: Path):
    """
    Reads support point data from a CSV file and plots the z-positions of the support points
    for each slice using Seaborn.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing support point data.

    Returns
    -------
    None
    """
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)

    # Ensure the data is sorted by slice and support point number
    df = df.sort_values(by=["File Name", "Support Point #"])

    # Plot the z-positions for each slice
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x="File Name",  # Slice (x-axis)
        y="Support Point Z",  # Z position (y-axis)
        hue="Support Point #",  # Support Point # (overlaid lines)
        marker="o",
    )

    # Customize the plot
    plt.title("Z Positions of Support Points Across Slices", fontsize=16)
    plt.xlabel("Slice (File Name)", fontsize=14)
    plt.ylabel("Z Position", fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title="Support Point #", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_support_points_and_metadata(csv_file: Path):
    """
    Reads support point data from a CSV file and creates multiple plots:
    1. Z positions of support points across slices.
    2. Z height from metadata across slices.
    3. Angle from z-axis across slices.
    4. Polar plot visualizing normal vectors looking down the z-axis.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing support point data.

    Returns
    -------
    None
    """
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)

    # Ensure the data is sorted by slice and support point number
    df = df.sort_values(by=["File Name", "Support Point #"])

    # Preprocess the normal vector column to fix missing commas
    def parse_normal_vector(vec: str):
        """
        Parses a string representation of a normal vector into a NumPy array,
        ensuring no leading commas and replacing internal whitespace with commas.
        """

        # Remove leading and trailing whitespace
        vec = vec.strip()

        # Use regular expressions to replace internal whitespace with commas
        vec = re.sub(r"\s+", ",", vec)  # Replace all consecutive whitespace with commas

        if vec.startswith("[,"):
            vec = "[" + vec[2:]

        return np.array(eval(vec))  # Convert string to NumPy array

    df["normal vector"] = df["normal vector"].apply(parse_normal_vector)

    # Create subplots
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.suptitle(f"{csv_file.stem}\nSupport Points Analysis", fontsize=16)

    def reduce_filename_length_vectorized(filenames: pd.Series) -> pd.Series:
        """
        Reduces filename length by removing repeated substrings between underscores
        using vectorized operations.

        Parameters
        ----------
        filenames : pd.Series
            Pandas Series containing filenames.

        Returns
        -------
        pd.Series
            Pandas Series with reduced filenames.
        """
        # Extract all substrings between underscores
        substrings = filenames.str.extractall(r"_(.*?)_")[0]

        # Find common substrings across all filenames
        common_substrings = substrings.value_counts()[
            substrings.value_counts() == len(filenames)
        ].index

        # Remove common substrings from filenames using vectorized string replacement
        reduced_filenames = filenames
        for common in common_substrings:
            reduced_filenames = reduced_filenames.str.replace(
                f"_{common}_", "_", regex=False
            )

        return reduced_filenames

    # Reduce filename length
    df["Reduced File Name"] = reduce_filename_length_vectorized(df["File Name"])

    # Adjust x-ticks spacing
    x_ticks = df["Reduced File Name"].unique()

    # Plot 1: Z positions of support points across slices
    sns.lineplot(
        data=df,
        x="Reduced File Name",
        y="Support Point Z",
        hue="Support Point #",
        marker=".",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Z Positions of Support Points Across Slices", fontsize=14)
    axes[0, 0].set_xlabel("Slice (File Name)", fontsize=12)
    axes[0, 0].set_ylabel("Z Position", fontsize=12)
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Adjust x-ticks spacing
    axes[0, 0].set_xticks(x_ticks[::50])  # Show every second slice for better spacing

    # Plot 2: Z height from metadata across slices
    sns.lineplot(
        data=df,
        x="Reduced File Name",
        y="z height from metadata",
        markers=False,
        ax=axes[0, 1],
    )
    axes[0, 1].scatter(
        df["Reduced File Name"][::10],  # Select every 2nd point for x-axis
        df["z height from metadata"][::10],  # Select every 2nd point for y-axis
        color="blue",
        alpha=0.7,
        s=10,
    )
    axes[0, 1].set_title("Z Height from Metadata Across Slices", fontsize=14)
    axes[0, 1].set_xlabel("Slice (File Name)", fontsize=12)
    axes[0, 1].set_ylabel("Z Height", fontsize=12)
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Adjust x-ticks spacing
    axes[0, 1].set_xticks(x_ticks[::50])  # Show every second slice for better spacing

    # Plot 3: Angle from z-axis across slices
    sns.lineplot(
        data=df,
        x="Reduced File Name",
        y="angle from z",
        marker=".",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Angle from Z-Axis Across Slices", fontsize=14)
    axes[1, 0].set_xlabel("Slice (File Name)", fontsize=12)
    axes[1, 0].set_ylabel("Angle (Degrees)", fontsize=12)
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Adjust x-ticks spacing
    axes[1, 0].set_xticks(x_ticks[::50])  # Show every second slice for better spacing

    # Plot 4: Polar plot of normal vectors
    axes[1, 1] = plt.subplot(2, 2, 4, polar=True)
    # for _, row in df.iterrows():
    #     normal_vec = row["normal vector"]
    #     angle = np.arctan2(normal_vec[1], normal_vec[0])  # Angle in radians
    #     magnitude = np.linalg.norm(normal_vec[:2])  # Magnitude in the XY plane
    #     axes[1, 1].scatter(angle, magnitude, color='r', s=5)
    # Extract normal vectors as a NumPy

    # Randomly sample 100 rows (or fewer if the DataFrame has less than 100 rows)
    sampled_df = df.sample(n=min(100, len(df)), random_state=42)

    # Extract normal vectors as a NumPy array
    normal_vectors = np.array(sampled_df["normal vector"].tolist())

    # Calculate angles and magnitudes using vectorized operations
    angles = np.arctan2(normal_vectors[:, 1], normal_vectors[:, 0])  # Angles in radians
    magnitudes = np.linalg.norm(
        normal_vectors[:, :2], axis=1
    )  # Magnitudes in the XY plane

    axes[1, 1].scatter(angles, magnitudes, color="r", s=5)
    axes[1, 1].spines["polar"].set_visible(False)  # This is key for polar plots
    axes[1, 1].set_title(
        "Polar Plot of Normal Vectors (Looking Down Z-Axis)", fontsize=14
    )
    axes[1, 1].set_theta_zero_location("N")  # North points up
    axes[1, 1].set_theta_direction(-1)  # Clockwise direction
    # axes[1, 1].axis('off')

    # axes[1, 1].legend(loc="upper right", fontsize=8)

    # Adjust layout and show the plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def save_support_point_info(params: GeneralAttrs) -> Path:
    """
    Save support point information extracted from CZI files.

    This function reads CZI files from the specified input directory, extracts support points,
    and saves the information into a CSV file in the output directory. Currently, only files
    with the '.czi' extension are supported.

    Parameters
    ----------
    params : ut.GeneralAttrs
        An object containing the following attributes:
        - file_extension (str): The file extension to filter files (must be '.czi').
        - input_directory (Path): The directory containing the input CZI files.
        - output_directory (Path): The directory where the output CSV file will be saved.

    Returns
    -------
    Path
        The path to the CSV file containing the support point information.

    Raises
    ------
    ValueError
        If the file extension is not '.czi'.

    Notes
    -----
    - The output CSV file will be named 'support_points.csv'.
    - If an error occurs while processing a CZI file, a message will be printed, and the
      function will continue with the next file.
    - The output directory will be created if it does not exist.
    """

    ## Tile reading, CZI only
    file_extension = params.file_extension
    if file_extension != ".czi":
        raise ValueError("At this time, only .czi file extension is supported")

    czi_input_directory = params.input_directory
    # Sort the CZI files in the specified input directory
    sorted_czi_files = ut.sorted_files(
        directory=czi_input_directory, file_extension=file_extension
    )

    output_directory = params.output_directory
    # Create the output directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file_name = "support_points.csv"
    file_path_czi_sp_out = output_directory.joinpath(output_file_name)
    file_path_czi_info = output_directory.joinpath("czi_info.csv")

    print(f"Found {len(sorted_czi_files)} czi files in {czi_input_directory}\n")
    print(f"Support point information will be saved in {file_path_czi_sp_out}\n")

    print(f'{" Reading the files ":*^28}')

    total_files = len(sorted_czi_files)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[bold blue]{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Reading...", total=total_files)

        for each_czi_path in sorted_czi_files:
            try:
                # Extract support points from the current CZI file
                all_scene_sp = ci.scene_support_points(each_czi_path)

                # check if empty
                if all_scene_sp:  # case one, there are tile regions and tile arrays
                    if all_scene_sp[0].z_heights.size == 0:
                        all_scene_sp = ci.scene_support_points_global(each_czi_path)
                else:  # case two, empty, only tile arrays
                    all_scene_sp = ci.scene_support_points_global(each_czi_path)
            except RuntimeError as error:
                # Print an error message if the file cannot be processed
                # This scenario usually occurs if the czi is corrupted
                print(f"{each_czi_path} has an error: {error}")
                continue
            finally:
                progress.update(task, advance=1)
            # Dump the support point data to a CSV file
            dump_czi_support_point_info(all_scene_sp, file_path_czi_sp_out)
            dump_file_info_to_csv(each_czi_path, file_path_czi_info)
            # Uncomment the following lines to dump scene-specific support point data
            # for each_scene_sp in all_scene_sp:
            #     # Determine the file name based on the scene name
            #     file_path = output_directory / f"{each_scene_sp.scene}_raw_sp_info.csv"
            #     dump_scene_support_point_info(each_scene_sp, file_path)

    return file_path_czi_sp_out


def dump_czi_support_point_info(
    all_sp_info: list[SceneSupportPoints], csv_out_file: Path
):
    """to come"""
    czi_fname = all_sp_info[0].czi_fname

    all_z_heights = np.array([])
    for each_scene_sp in all_sp_info:
        # Ensure each_scene_sp.z_heights is a NumPy array
        each_scene_sp_z_height = np.array(each_scene_sp.z_heights)
        # Concatenate the current z_heights with the accumulated ones
        all_z_heights = np.concatenate((all_z_heights, each_scene_sp_z_height))

    # compute slice average
    average_z_height = np.mean(all_z_heights)

    # Check if the file exists and is empty to decide on writing the header
    file_exists = csv_out_file.exists()

    with open(csv_out_file, "a", newline="", encoding="utf-8") as csvfile:
        sp_writer = csv.writer(csvfile)

        # If the file didn't exist or was empty, write the header
        if not file_exists or csv_out_file.stat().st_size == 0:
            sp_writer.writerow(
                [
                    "File name",
                    "Average height (um)",
                    "All heights (um)",
                ]
            )

        # Write the data row
        sp_writer.writerow([czi_fname, average_z_height, all_z_heights])


def dump_file_info_to_csv(file_path: Path, csv_file: Path):
    # Ensure the file_path is a Path object
    # file_path = Path(file_path)

    # Check if the file exists
    if not file_path.exists():
        print(f"The file {file_path} does not exist.")
        return

    # Extracting information
    parent1 = file_path.parent.name
    parent2 = file_path.parent.parent.name if file_path.parent.parent else ""
    file_name = file_path.name
    file_size = file_path.stat().st_size
    # creation_time = datetime.fromtimestamp(file_path.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S')
    creation_time = ci.acquisition_date_and_time(file_path)

    # Data to write
    data = [parent2, parent1, file_name, file_size, creation_time]

    # Check if the file exists and is empty to decide on writing the header
    file_exists = csv_file.exists()
    # Writing to CSV
    with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # If the file didn't exist or was empty, write the header
        if not file_exists or csv_file.stat().st_size == 0:
            writer.writerow(
                [
                    "Parent Dir2",
                    "Parent Dir1",
                    "Filename",
                    "File size",
                    "Creation time",
                ]
            )
        writer.writerow(data)

    # print(f"Data for {file_name} has been written to {csv_file}")


def dump_scene_support_point_info(sp_info: SceneSupportPoints, csv_out_file: Path):
    """Dump scene support point information to a CSV file.

    Parameters
    ----------
    sp_info : SceneSupportPoints
        An object containing information about the scene, including the filename,
        scene name, and z heights of support points.
    csv_out_file : Path
        The path to the CSV file where the information will be written.

    Notes
    -----
    If the specified CSV file does not exist or is empty, a header row will be
    written. Each subsequent call will append a new row with the scene information.
    """
    czi_fname = sp_info.czi_fname
    scene_name = sp_info.scene
    if len(sp_info.z_heights) == 0:  # empty
        average_z_height = sp_info.z
    else:
        average_z_height = np.mean(sp_info.z_heights)
    # breakpoint()

    # Check if the file exists and is empty to decide on writing the header
    file_exists = csv_out_file.exists()

    with open(csv_out_file, "a", newline="", encoding="utf-8") as csvfile:
        sp_writer = csv.writer(csvfile)

        # If the file didn't exist or was empty, write the header
        if not file_exists or csv_out_file.stat().st_size == 0:
            sp_writer.writerow(
                [
                    "File name",
                    "Scene name",
                    "Average height (um)",
                    "All heights (um)",
                    # "All Support Points (x,y,z)",
                ]
            )

        # Write the data row
        sp_writer.writerow([czi_fname, scene_name, average_z_height, sp_info.z_heights])


def raw_z_heights(csv_file_path: Path, pol_images: bool) -> np.array:
    """Returns the z height position from the csv file.

    Args:
        csv_file_path:  The fully pathed csv file.

    Returns:
        The z-height positions.
    """

    df = pd.read_csv(csv_file_path)

    # Check if the "Average height (um)" column exists
    if "Average height (um)" in df.columns:
        height_data = df.get("Average height (um)")
    else:
        # If not found, get unique z heights for each unique filename
        unique_heights = df[["File Name", "z height from metadata"]].drop_duplicates()
        height_data = unique_heights["z height from metadata"]

    height_arr = height_data.to_numpy()

    if pol_images:
        return height_arr[1::13]

    return height_arr


def process_scene_resample(
    raw_z_csv: Path,
    target_thickness: float,  # units of micron
    plot_opts: ResamplePlotOptions,
    pol_images: bool,
    skip_slice_idx: np.array = np.array(0),
) -> bool:
    """Resamples an individual scene for a dataset to provide 3D slices with
    nearly uniform slice thickness.

    Assumes we are working with directories of images, not HDF files to store the origin data

    Resample the slice images to be near uniform thickness

    Args:
        raw_z_csv:  The csv file containing the raw z values.

    Returns:
        True if the function succeeded, False otherwise.
    """

    # 1

    processed = False  # by default, the fuction has not yet been processed

    all_slice_z = raw_z_heights(raw_z_csv, pol_images)

    # all_slice_czi_paths = ut.sorted_files(Path(czi_input_dir), "czi")
    # all_slice_tif_paths = io.sorted_files(scene_input_dir, "tif")

    # check the file indices match properly, same sizes
    # assert len(all_slice_czi_paths) == len(
    #     all_slice_tif_paths
    # ), "number of czi files used to calculate z heights do not match the number of image slices attempting to resample."

    # IGNORE FOR NOW
    # assert all_slice_z.shape[0] == len(
    #     all_slice_czi_paths
    # ), "number of z heights in the csv do not match the number of czi files used to calculate z heights."

    # assert all_slice_z.shape[0] == len(
    #     all_slice_tif_paths
    # ), "number of z heights in the csv do not match the number of image slices attempting to resample."

    # 2 return new slice indices
    resampled_idx = scene_resample_simple(all_slice_z, target_thickness, skip_slice_idx)

    # plot and verify depending on plot options

    # for each channel in the scene, place files into new directory
    # (could be with mv or cp depending on how much we trust ourselves)

    if plot_opts.display:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.set_ylabel("Z Position (mm)", fontweight="bold")
        ax.set_xlabel("Slice Number", labelpad=5, fontweight="bold")

        if plot_opts.raw:
            plt.scatter(
                np.arange(0, len(all_slice_z)),
                all_slice_z * UM_TO_MM,
                s=3,
                label="All Slices",
                color="k",
            )
        if plot_opts.skip_removed:
            skip_remove_z = np.delete(all_slice_z, skip_slice_idx)
            plt.scatter(
                np.arange(0, len(skip_remove_z)),
                skip_remove_z * UM_TO_MM,
                s=3,
                label="Skipped Slice Removed",
                color="orange",
                alpha=0.5,
            )
        if plot_opts.resampled:
            plt.scatter(
                np.arange(0, len(resampled_idx)),
                all_slice_z[resampled_idx] * UM_TO_MM,
                s=3,
                label="Resampled Height",
                color="gray",
            )
        if plot_opts.fit:
            q = fit_linear_segments(
                x_values=np.arange(0, len(resampled_idx)),
                y_values=all_slice_z[resampled_idx] * UM_TO_MM,
                segment_count=1,
            )
            qx = np.array([q[0][1], q[0][0]])
            qy = np.array([q[1][1], q[1][0]])
            resampled_slope = (q[1][1] - q[1][0]) / (q[0][1] - q[0][0])

            plt.plot(
                qx,
                qy,
                label=f"Overall {np.around(resampled_slope, 4)*(1/(UM_TO_MM))} um/slice",
                color="k",
                linestyle="--",
            )
        title = f"Resampling for target thickness of\n{target_thickness} micron"
        if plot_opts.legend:
            plt.legend()
        plt.grid()
        ax.set_axisbelow(True)
        plt.title(title, fontweight="bold")
        plt.tight_layout()
        plt.show()

        if not ut.yes_no("\tContinue with resampling?"):
            print("Exiting data processor...")
            sys.exit(0)

    # new_slice_idx = 0
    # for slice_idx in resampled_idx:
    #     file_dest = os.path.join(scene_output_dir, f"Slice_{new_slice_idx:>04}")

    #     # COPY FILES
    #     shutil.copyfile(all_slice_tif_paths[slice_idx], file_dest)

    #     new_slice_idx += 1

    processed = True  # overwrite, if we get to this point, the functiion succeede

    return processed


def scene_resample_simple(
    z_heights: NDArray,  # Array of original z heights
    target_thickness: float,  # Desired thickness of the slices in microns
    skip_slice_idx: NDArray = np.array([], dtype=int),  # Indices of slices to skip
) -> NDArray:  # Indices of the resampled heights in the original z_heights array
    """Resamples an individual scene for a dataset to provide 3D slices with nearly uniform slice thickness.

    Args:
        z_heights (np.ndarray): Array of original z heights.
        target_thickness (float): Desired thickness of the slices in microns.
        skip_slice_idx (np.ndarray, optional): Array of indices to skip during resampling. Defaults to an empty array.

    Returns:
        np.ndarray: Indices of the resampled heights in the original z_heights array.
    """

    # Ensure skip_slice_idx is a 1D array
    skip_slice_idx = np.asarray(skip_slice_idx).flatten()

    # Remove skipped slice indices from z_heights
    if skip_slice_idx.size > 0:
        sel_z_heights = np.delete(z_heights, skip_slice_idx)
    else:
        sel_z_heights = z_heights

    # Generate target heights based on the original z_heights range
    target_heights = np.arange(
        sel_z_heights[0], sel_z_heights[-1] + target_thickness, target_thickness
    )

    # Resample heights based on the target heights
    resampled_heights = []
    for height in target_heights:
        diff = np.abs(sel_z_heights - height)
        index = diff.argmin()
        resampled_heights.append(sel_z_heights[index])

    # Convert the result to a NumPy array
    resampled_heights = np.asarray(resampled_heights)

    # Get the indices of the resampled heights in the original z_heights
    resampled_heights_indices = np.array(
        [np.argwhere(z_heights == val)[0][0] for val in resampled_heights]
    )

    return resampled_heights_indices


def fit_linear_segments(
    x_values: NDArray[np.float64],
    y_values: NDArray[np.float64],
    segment_count: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fit linear segments to a set of points.
    """
    if segment_count >= len(x_values):
        raise ValueError(
            "The number of segments must be less than the number of data points."
        )

    # Calculate the range of x values
    x_min = x_values.min()
    x_max = x_values.max()

    # Improved initial guess for segment boundaries
    # Distribute segments evenly across the x range
    segment_boundaries_x = np.linspace(x_min, x_max, segment_count + 1)

    # For y-values at segment boundaries, interpolate based on the existing points
    segment_boundaries_y = np.interp(segment_boundaries_x, x_values, y_values)

    def error_function(params):
        """
        Calculate the mean squared error between the model and the actual y-values.
        """
        # Reconstruct the y-values at segment boundaries from params
        y_values_at_boundaries = params
        # Use fixed x segment boundaries and interpolated y-values for error calculation
        y_predicted = np.interp(x_values, segment_boundaries_x, y_values_at_boundaries)
        return np.mean((y_values - y_predicted) ** 2)

    # Optimize only the y-values at segment boundaries
    initial_guess = segment_boundaries_y
    result = optimize.minimize(error_function, x0=initial_guess, method="Nelder-Mead")

    # Use the optimized y-values with the original x segment boundaries
    optimized_y = result.x

    return segment_boundaries_x, optimized_y


@app.callback()
def callback():
    """
    precon3d.resampler tries to evenly space out the slices collected based on the microscope focus height.
    """
