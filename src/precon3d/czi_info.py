"""Module for extracting metadata and tile information from CZI files.

This module provides CLI commands and functions to process `.czi` files,
extract metadata, check dimensions, and save tiles.
"""

# Standard library imports
from datetime import datetime
from pathlib import Path
from typing import Final, Tuple, List, NamedTuple, Union, Optional
import csv
import xml.etree.ElementTree as ET
import math

# Third-party library imports
import numpy as np
import pytz
from dateutil import parser
import typer
from aicspylibczi import CziFile

# Local imports
import precon3d.mosaic_utils
import precon3d.utility as ut
from precon3d.custom_types import *
from precon3d._my_typer_cli import CustomCLIGroup, CustomCLICommand

# CLI setup
app = typer.Typer(
    cls=CustomCLIGroup,
    no_args_is_help=True,
    short_help="Access information about the CZI file.",
)


@app.command(
    cls=CustomCLICommand,
    name="czidims",
    short_help="Get the dimensions of the CZI file or folder.",
)
def czidims(czi_path: Path) -> bool:
    """Check that one or many CZI files have consistent dimensions.

    Args:
        czi_path (Path): Path to a `.czi` file or directory of `.czi` files.

    Returns:
        bool: True if a single file is valid or if all files in a directory share the same dimensions.
              False if two files in the directory differ in shape.

    Raises:
        FileNotFoundError: If `czi_path` is a directory but contains no `.czi` files.
        ValueError: If `czi_path` is neither a file nor a directory.
    """
    if czi_path.is_file():
        dims = get_dims(czi_path)
        print(f"{czi_path.stem} dimensions: {dims}")
        return True

    if czi_path.is_dir():
        czi_files = ut.sorted_files(czi_path, file_extension=".czi")
        if not czi_files:
            raise FileNotFoundError(f"No .czi files found in directory: {czi_path}")

        ref_dims = get_dims(czi_files[0])
        print(f"{czi_files[0].stem} dimensions: {ref_dims}")

        for file_path in czi_files[1:]:
            dims = get_dims(file_path)
            print(f"{file_path.stem} dimensions: {dims}")
            if dims != ref_dims:
                print("Dimensions are not consistent")
                return False

        print("Dimensions are consistent")
        return True

    raise ValueError(f"Path is not a file or directory: {czi_path}")


def get_dims(czi_file_path: Path) -> dict[str, int]:
    """Return the dimension‐shape mapping for a single CZI file.

    Args:
        czi_file_path (Path): Path to the `.czi` file.

    Returns:
        dict[str, int]: A mapping from dimension names (e.g., 'X', 'Y', 'Z', 'C', 'T') to their respective sizes.
    """
    return CziFile(czi_file_path).get_dims_shape()


@app.command(
    cls=CustomCLICommand,
    name="acquisition_date_and_time",
    short_help="Get the acquisition date and time from the CZI file.",
)
def acquisition_date_and_time(czi_file_path: Path) -> str:
    """Retrieve the acquisition date and time for the CZI file.

    Args:
        czi_file_path (Path): Path to the `.czi` file.

    Returns:
        str: Formatted acquisition date and time string.
    """
    datetime_str = (
        CziFile(czi_file_path).meta.find(".//Image//AcquisitionDateAndTime").text
    )
    dt = parser.isoparse(datetime_str)
    formatted_datetime = dt.strftime("%Y-%m-%d %H:%M:%S")
    typer.echo(formatted_datetime)
    return formatted_datetime


@app.command(
    cls=CustomCLICommand,
    name="simple_metadata",
    short_help="Retrieve simple metadata from the CZI file.",
)
def simple_metadata(czi_file_path: Path) -> CZIMetadata:
    """Retrieve simple metadata from the CZI file.

    Args:
        czi_file_path (Path): Path to the `.czi` file.

    Raises:
        TypeError: If the file is not a mosaic CZI file.

    Returns:
        CZIMetadata: Named tuple containing the metadata.
    """
    if not isinstance(czi_file_path, Path):
        czi_file_path = Path(czi_file_path)

    czi = CziFile(czi_file_path)
    if not czi.is_mosaic():
        raise TypeError("Only CZI mosaic files are supported")

    metadata = czi.meta
    pixel_type = czi.pixel_type

    all_channels = metadata.findall(".//Image//Dimensions//Channels//Channel")
    all_scenes = metadata.findall(".//Image//Dimensions//S//Scenes//Scene")

    simple_czi_metadata = CZIMetadata(
        filename=czi_file_path.stem,
        author=metadata.find(".//Document//UserName").text,
        date_created=metadata.find(".//Image//AcquisitionDateAndTime").text,
        dims=czi.dims,
        dims_shape=czi.get_dims_shape(),
        scene_names=[scene.get("Name") for scene in all_scenes],
        scene_metadata=scene_metadata(czi_file_path),
        scene_shapes_are_consistent=czi.shape_is_consistent,
        channel_names=[channel.get("Name") for channel in all_channels],
        channel_metadata=channel_metadata(czi_file_path),
        tile_width=metadata.find(".//SampleHolder//TileDimension//Width").text,
        tile_height=metadata.find(".//SampleHolder//TileDimension//Height").text,
        tile_overlap=metadata.find(".//SampleHolder//Overlap").text,
        tile_anchor_mode=metadata.find(".//SampleHolder//TileRegionAnchorMode").text,
        tile_scan_mode=metadata.find(".//SampleHolder//ScanMode").text,
        microscope_name=metadata.find(".//Instrument//Microscopes//Microscope").get(
            "Name"
        ),
        microscope_type=metadata.find(
            ".//Instrument//Microscopes//Microscope//Type"
        ).text,
        camera_name=metadata.find(
            ".//Instrument//Detectors//Detector//Manufacturer//Model"
        ).text,
        camera_adapter=metadata.find(
            ".//Instrument//Detectors//Adapter//Manufacturer//Model"
        ).text,
        camera_pixel_size=metadata.find(
            ".//Scaling//AutoScaling//CameraPixelDistance"
        ).text,
        camera_pixel_unit=metadata.find(
            ".//Scaling//Items//Distance//DefaultUnitFormat"
        ).text,
        objective_name=metadata.find(".//Objective//Manufacturer//Model").text,
        objective_magnification=metadata.find(
            ".//Objectives//Objective//NominalMagnification"
        ).text,
        effective_pixel_size=metadata.find(".//Scaling//Items//Distance//Value").text,
        color_mode=str(pixel_type),
    )

    typer.echo(simple_czi_metadata)
    return simple_czi_metadata


def average_slice_heights(all_czi_file_paths: List[Path]) -> np.ndarray:
    """Calculate and save the average slice heights for each scene in CZI files.

    Args:
        all_czi_file_paths (List[Path]): List of paths to `.czi` files.

    Returns:
        np.ndarray: Array of average slice heights for each scene.
    """
    avg_slice_heights = []
    csv_fname_root = "raw_slice_heights.csv"
    row_number = 1

    print(f'{" Reading the files: ":*^28}')
    for each_czi_path in all_czi_file_paths:
        print(each_czi_path.stem)

        czi = CziFile(each_czi_path)
        if czi.is_mosaic():
            print("Found a mosaic CZI file with scene support points.")
            all_scene_sp = scene_support_points(each_czi_path)
        else:
            print("Found a tilearray CZI file with global support points.")
            all_scene_sp = scene_support_points_global(each_czi_path)

        # If no support points found in metadata
        if len(all_scene_sp) == 0:
            print(f"No scene support points found in {each_czi_path.stem}. Skipping.")
            raise TypeError("CZI file must contain support points")

        for each_scene_sp in all_scene_sp:
            czi_fname = each_scene_sp.czi_fname
            scene_name = each_scene_sp.scene

            # no z_heights scenario
            if each_scene_sp.z_heights.size == 0:
                average_z_height = float(each_scene_sp.z)
            else:
                average_z_height = np.mean(each_scene_sp.z_heights)

            avg_slice_heights.append(average_z_height)

            with open(
                f"{each_czi_path.parent.joinpath(scene_name)}_{csv_fname_root}",
                "a",
                newline="",
                encoding="utf-8",
            ) as csvfile:
                sp_writer = csv.writer(csvfile)
                if row_number == 1:
                    sp_writer.writerow(
                        [
                            "File name",
                            "Scene name",
                            "Average height (um)",
                            "All heights (um)",
                        ]
                    )
                sp_writer.writerow(
                    [
                        czi_fname,
                        scene_name,
                        average_z_height,
                        each_scene_sp.z_heights,
                    ]
                )

        row_number += 1

    return np.array(avg_slice_heights)


def scene_support_points(czi_file_path: Path) -> List[SceneSupportPoints]:
    """Extract scene support points data from CZI metadata.

    Args:
        czi_file_path (Path): Path to the `.czi` file.

    Returns:
        List[SceneSupportPoints]: List of scene support points data.
    """
    file_basename = czi_file_path.stem
    metadata = CziFile(czi_file_path).meta

    all_scene_sp = []
    for each_scene in metadata.findall(".//TileRegion"):
        if each_scene.find(".//IsUsedForAcquisition").text == "true":
            scene_name = each_scene.attrib["Name"]
            scene_z = each_scene.find(".//Z").text

            sp_xy = []
            sp_z = []
            for each_sp in each_scene.findall(".//SupportPoints//SupportPoint"):
                sp_xy.append(
                    TilePosition(x=each_sp.find("X").text, y=each_sp.find("Y").text)
                )
                sp_z.append(each_sp.find("Z").text)

            all_scene_sp.append(
                SceneSupportPoints(
                    z=scene_z,
                    z_heights=np.array(sp_z, dtype="float32"),
                    xy_positions=sp_xy,
                    czi_fname=file_basename,
                    scene=scene_name,
                )
            )

    return all_scene_sp


def scene_support_points_global(
    czi_file_path: Path,
) -> List[SceneSupportPoints]:
    """Extract global scene support points data from CZI metadata.

    Args:
        czi_file_path (Path): Path to the `.czi` file.

    Returns:
        List[SceneSupportPoints]: List of global scene support points data.
    """
    file_basename = czi_file_path.stem
    metadata = CziFile(czi_file_path).meta

    all_scene_sp = []
    for each_scene in metadata.findall(".//SampleHolder//Template"):
        scene_name = each_scene.attrib["Name"]

        sp_xy = []
        sp_z = []
        for each_sp in each_scene.findall(".//SupportPoints//SupportPoint"):
            sp_xy.append(
                TilePosition(x=each_sp.find("X").text, y=each_sp.find("Y").text)
            )
            sp_z.append(each_sp.find("Z").text)

        sp_all_z = np.array(sp_z, dtype="float32")
        all_scene_sp.append(
            SceneSupportPoints(
                z=np.mean(sp_all_z),
                z_heights=sp_all_z,
                xy_positions=sp_xy,
                czi_fname=file_basename,
                scene=scene_name,
            )
        )

    return all_scene_sp


def channel_metadata(czi_file_path: Path) -> List[ChannelMetadata]:
    """Extract channel metadata from CZI files.

    Args:
        czi_file_path (Path): Path to the `.czi` file.

    Returns:
        List[ChannelMetadata]: List of channel metadata.
    """
    czi = CziFile(czi_file_path)
    metadata = czi.meta

    all_channels = metadata.findall(".//Image//Dimensions//Channels//Channel")
    all_channel_metadata = []
    for each_channel in all_channels:
        illumination_wavelength_element = each_channel.find(
            "IlluminationWavelength/SinglePeak"
        )
        illumination_wavelength = (
            int(illumination_wavelength_element.text)
            if illumination_wavelength_element is not None
            else None
        )

        illumination_wavelength_range_element = each_channel.find(
            "IlluminationWavelength/Ranges"
        )
        illumination_wavelength_range = (
            illumination_wavelength_range_element.text
            if illumination_wavelength_range_element is not None
            else None
        )

        each_channel_metadata = ChannelMetadata(
            channel_name=each_channel.get("Name"),
            mode=each_channel.find("Fluor").text,
            exposure_time=int(each_channel.find("ExposureTime").text),
            reflector=each_channel.find("Reflector").text,
            contrast=each_channel.find("ContrastMethod").text,
            pixel_type=each_channel.find("PixelType").text,
            pixel_bit_count=int(each_channel.find("ComponentBitCount").text),
            illumination_wavelength=illumination_wavelength,
            illumination_wavelength_range=illumination_wavelength_range,
            illumination_intensity=each_channel.find(
                "LightSourcesSettings/LightSourceSettings/Intensity"
            ).text,
            detector_binning=each_channel.find("DetectorSettings/Binning").text,
        )
        all_channel_metadata.append(each_channel_metadata)

    return all_channel_metadata


def scene_metadata(czi_file_path: Path) -> List[SceneMetadata]:
    """Extract scene metadata from CZI files.

    Args:
        czi_file_path (Path): Path to the `.czi` file.

    Returns:
        List[SceneMetadata]: List of scene metadata.
    """
    czi = CziFile(czi_file_path)
    metadata = czi.meta

    active_tile_regions = [
        tile_region
        for tile_region in metadata.findall(".//SampleHolder//TileRegions//TileRegion")
        if tile_region.find("IsUsedForAcquisition").text == "true"
    ]

    all_bbox = czi.get_all_mosaic_scene_bounding_boxes()

    all_scenes_metadata = []
    for idx, each_tile_region in enumerate(active_tile_regions):
        each_scene_metadata = SceneMetadata(
            scene_name=each_tile_region.get("Name"),
            scene_index=idx,
            bounding_box=BoundingBox(
                h=all_bbox[idx].h,
                w=all_bbox[idx].w,
                x=all_bbox[idx].x,
                y=all_bbox[idx].y,
            ),
            n_columns=int(each_tile_region.find("Columns").text),
            n_rows=int(each_tile_region.find("Rows").text),
            contour_size=each_tile_region.find("ContourSize").text,
            center_position=each_tile_region.find("CenterPosition").text,
        )
        all_scenes_metadata.append(each_scene_metadata)

    return all_scenes_metadata


@app.command(
    cls=CustomCLICommand,
    name="metadata_as_xml",
    short_help="Save xml of metadata.",
)
def metadata_as_xml(czi_file_path: Path) -> Path:
    """Save CZI metadata as an XML file.

    Args:
        czi_file_path (Path): Path to the `.czi` file.

    Returns:
        Path: Path to the saved XML file.
    """
    directory_path = czi_file_path.parent
    czi_file_metadata = CziFile(czi_file_path).meta
    output_xml_path = directory_path.joinpath(czi_file_path.stem + "_metadata.xml")

    xml_string = ET.tostring(czi_file_metadata, encoding="unicode", method="xml")
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_string)

    return output_xml_path


@app.command(
    cls=CustomCLICommand,
    name="save_tiles",
    short_help="Save all tiles from the CZI files in the specified directory.",
)
def save_tiles(czi_input_dir: Path, output_directory: Path):
    """Save tiles from CZI files as TIFF images.

    Args:
        czi_input_dir (Path): Directory containing `.czi` files.
        output_directory (Path): Directory to save the extracted tiles.
    """
    print(f"\tExtracting tiles from .czi files in {czi_input_dir}\n")

    czi_filepaths = ut.sorted_files(directory=czi_input_dir, file_extension=".czi")

    for each_czi_file in czi_filepaths:
        if CziFile(each_czi_file).is_mosaic():
            tilestack, tilestack_fname = precon3d.mosaic_utils.ai_tilestack(
                each_czi_file
            )

            for each_scene_index in tilestack.scenes:
                tilestack_list = precon3d.mosaic_utils.simple_tilestack(
                    tilestack, each_scene_index, tilestack_fname
                )

                for each_tilestack in tilestack_list:
                    precon3d.mosaic_utils.save_tilestack(
                        tilestack=each_tilestack,
                        output_directory=output_directory,
                    )
        else:
            tilestack_list = precon3d.mosaic_utils.tile_region_array_data(each_czi_file)
            # breakpoint()
            # for each_tilestack_scene_list in tilestack_list:
            # for each_tilestack_channel in each_tilestack_scene_list:
            # for each_tilestack_channel in tilestack_list:
            # breakpoint()
            merged_tilestack_channel = precon3d.mosaic_utils.merge_tile_arrays(
                tilestack_list
            )
            precon3d.mosaic_utils.save_tilestack(
                tilestack=merged_tilestack_channel,
                output_directory=output_directory,
            )

        print(f'\tTiles from "{each_czi_file.stem}" were saved.')


@app.command(
    cls=CustomCLICommand,
    name="save_tile_by_idx",
    short_help="Save single tile based on index from the CZI files in the specified directory.",
)
def save_tile_by_idx(czi_input_dir: Path, tile_idx: int, output_directory: Path):
    """Save tiles from CZI files as TIFF images.

    Args:
        czi_input_dir (Path): Directory containing `.czi` files.
        output_directory (Path): Directory to save the extracted tiles.
    """
    print(f"\tExtracting tiles from .czi files in {czi_input_dir}\n")

    czi_filepaths = ut.sorted_files(directory=czi_input_dir, file_extension=".czi")

    for each_czi_file in czi_filepaths:
        if CziFile(each_czi_file).is_mosaic():
            tilestack, tilestack_fname = precon3d.mosaic_utils.ai_tilestack(
                each_czi_file
            )

            for each_scene_index in tilestack.scenes:
                tilestack_list = precon3d.mosaic_utils.simple_tilestack(
                    tilestack, each_scene_index, tilestack_fname
                )

                for each_tilestack_channel in tilestack_list:
                    precon3d.mosaic_utils.save_tile_by_idx(
                        tilestack=each_tilestack_channel,
                        tile_idx=tile_idx,
                        output_directory=output_directory,
                    )
        else:
            tilestack_list = precon3d.mosaic_utils.tile_region_array_data(each_czi_file)

            for each_tilestack_scene_list in tilestack_list:
                for each_tilestack_channel in each_tilestack_scene_list:
                    merged_tilestack_channel = precon3d.mosaic_utils.merge_tile_arrays(
                        each_tilestack_channel
                    )
                    precon3d.mosaic_utils.save_tile_by_idx(
                        tilestack=merged_tilestack_channel,
                        tile_idx=tile_idx,
                        output_directory=output_directory,
                    )

        print(f'\tTile {tile_idx} from "{each_czi_file.stem}" was saved.')


@app.callback()
def callback():
    """CLI callback for precon3d.czi_info.

    Provides access to metadata and tile information from CZI files.
    """


if __name__ == "__main__":
    app()
