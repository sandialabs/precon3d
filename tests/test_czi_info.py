# python standard libraries
from pathlib import Path
import numpy as np
from typing import Dict
from numpy import float32

# 3rd party libraries
import pytest
import xml.etree.ElementTree as ET

# local libraries
from tests.test_files import *
import precon3d.czi_info
from precon3d.utility import rmdir

# import the types individually
from precon3d.czi_info import (
    CZIMetadata,
    ChannelMetadata,
    SceneMetadata,
    SceneSupportPoints,
    TilePosition,
    BoundingBox,
)


def test_czidims():
    """Test czi dims"""
    all_test_files = load_test_files(TEST_FILES_DIRECTORY)

    # Known dims for each test file
    known_dims_mapping = {
        "BW_Pol_1TA.czi": [{"X": (0, 896), "Y": (0, 896), "C": (0, 1), "S": (0, 4)}],
        "BW_Pol_1TR.czi": [
            {"X": (0, 896), "Y": (0, 896), "C": (0, 1), "M": (0, 4), "S": (0, 1)}
        ],
        "BW_Pol_1TR_1TA.czi": [
            {"X": (0, 896), "Y": (0, 896), "C": (0, 1), "M": (0, 2), "S": (0, 1)},
            {"X": (0, 896), "Y": (0, 896), "C": (0, 1), "M": (0, 1), "S": (1, 2)},
            {"X": (0, 896), "Y": (0, 896), "C": (0, 1), "M": (0, 1), "S": (2, 3)},
            {"X": (0, 896), "Y": (0, 896), "C": (0, 1), "M": (0, 1), "S": (3, 4)},
            {"X": (0, 896), "Y": (0, 896), "C": (0, 1), "M": (0, 1), "S": (4, 5)},
        ],
        "BW_Pol_Bright_1TA.czi": [
            {"X": (0, 896), "Y": (0, 896), "C": (0, 2), "S": (0, 4)}
        ],
        "BW_Pol_Bright_1TR.czi": [
            {"X": (0, 896), "Y": (0, 896), "C": (0, 2), "M": (0, 4), "S": (0, 1)}
        ],
        "BW_Pol_Bright_1TR_1TA.czi": [
            {"X": (0, 896), "Y": (0, 896), "C": (0, 2), "M": (0, 2), "S": (0, 1)},
            {"X": (0, 896), "Y": (0, 896), "C": (0, 2), "M": (0, 1), "S": (1, 2)},
            {"X": (0, 896), "Y": (0, 896), "C": (0, 2), "M": (0, 1), "S": (2, 3)},
            {"X": (0, 896), "Y": (0, 896), "C": (0, 2), "M": (0, 1), "S": (3, 4)},
            {"X": (0, 896), "Y": (0, 896), "C": (0, 2), "M": (0, 1), "S": (4, 5)},
        ],
        "RGB_Dark_1TA_5x.czi": [
            {"A": (0, 3), "X": (0, 448), "Y": (0, 448), "C": (0, 1), "S": (0, 4)}
        ],
        "RGB_Dark_1TR_5x.czi": [
            {
                "A": (0, 3),
                "X": (0, 448),
                "Y": (0, 448),
                "C": (0, 1),
                "M": (0, 4),
                "S": (0, 1),
            }
        ],
        "RGB_Dark_1TR_1TA_5x.czi": [
            {
                "A": (0, 3),
                "X": (0, 448),
                "Y": (0, 448),
                "C": (0, 1),
                "M": (0, 4),
                "S": (0, 1),
            },
            {
                "A": (0, 3),
                "X": (0, 448),
                "Y": (0, 448),
                "C": (0, 1),
                "M": (0, 1),
                "S": (1, 2),
            },
            {
                "A": (0, 3),
                "X": (0, 448),
                "Y": (0, 448),
                "C": (0, 1),
                "M": (0, 1),
                "S": (2, 3),
            },
            {
                "A": (0, 3),
                "X": (0, 448),
                "Y": (0, 448),
                "C": (0, 1),
                "M": (0, 1),
                "S": (3, 4),
            },
        ],
        "RGB_Dark_Bright_1TA_5x.czi": [
            {"A": (0, 3), "X": (0, 448), "Y": (0, 448), "C": (0, 2), "S": (0, 4)}
        ],
        "RGB_Dark_Bright_1TR_5x.czi": [
            {
                "A": (0, 3),
                "X": (0, 448),
                "Y": (0, 448),
                "C": (0, 2),
                "M": (0, 4),
                "S": (0, 1),
            }
        ],
        "RGB_Dark_Bright_1TR_1TA_5x.czi": [
            {
                "A": (0, 3),
                "X": (0, 448),
                "Y": (0, 448),
                "C": (0, 2),
                "M": (0, 2),
                "S": (0, 1),
            },
            {
                "A": (0, 3),
                "X": (0, 448),
                "Y": (0, 448),
                "C": (0, 2),
                "M": (0, 1),
                "S": (1, 2),
            },
            {
                "A": (0, 3),
                "X": (0, 448),
                "Y": (0, 448),
                "C": (0, 2),
                "M": (0, 1),
                "S": (2, 3),
            },
            {
                "A": (0, 3),
                "X": (0, 448),
                "Y": (0, 448),
                "C": (0, 2),
                "M": (0, 1),
                "S": (3, 4),
            },
        ],
    }

    # Iterate through each test file and compare dims
    # all_dims = []
    for each_test_file in all_test_files:
        # Extract dims from the file
        found_dims = precon3d.czi_info.get_dims(each_test_file.path)

        # Get the known dims for the file
        known_dims = known_dims_mapping.get(each_test_file.name)

        # Assert that the found dims matches the known dims
        assert found_dims == known_dims, (
            f"Mismatch for file {each_test_file.name}: "
            f"found {found_dims}, expected {known_dims}"
        )

    # single czi
    assert precon3d.czi_info.czidims(all_test_files[0].path) == True
    # Czi are all different
    assert precon3d.czi_info.czidims(TEST_FILES_DIRECTORY) == False


def test_average_slice_heights():
    """to come"""
    all_test_files = load_test_files(TEST_FILES_DIRECTORY)

    known_average_slice_heights_mapping = {
        "BW_Pol_1TA.czi": [500.005],
        "BW_Pol_1TR.czi": [531.0652],
        "BW_Pol_1TR_1TA.czi": [291.8],
        "BW_Pol_Bright_1TA.czi": [538.3342],
        "BW_Pol_Bright_1TR.czi": [500.01],
        "BW_Pol_Bright_1TR_1TA.czi": [291.8],
        "RGB_Dark_1TA_5x.czi": [5039.6953],
        "RGB_Dark_1TR_1TA_5x.czi": [5051.629],
        "RGB_Dark_1TR_5x.czi": [5215.95],
        "RGB_Dark_Bright_1TA_5x.czi": [5215.95],
        "RGB_Dark_Bright_1TR_1TA_5x.czi": [5011.0703],
        "RGB_Dark_Bright_1TR_5x.czi": [5215.95],
    }

    for each_test_file in all_test_files:
        # Extract dims from the file
        found_heights = precon3d.czi_info.average_slice_heights([each_test_file.path])

        print(f"{each_test_file.path}, {found_heights}")
        assert found_heights > 0

        # Get the known dims for the file
        known_average_slice_heights = known_average_slice_heights_mapping.get(
            each_test_file.name
        )

        # Assert that the found dims matches the known dims
        assert np.allclose(found_heights, known_average_slice_heights), (
            f"Mismatch for file {each_test_file.name}: "
            f"found {found_heights}, expected {known_average_slice_heights}"
        )

    # remove csv
    for csv_file in TEST_FILES_DIRECTORY.glob("*.csv"):
        csv_file.unlink()  # Delete the file
        print(f"Deleted: {csv_file}")


def test_acquisition_date_and_time():
    """Test acquisition date and time can be correctly ready from the file"""

    all_test_files = load_test_files(TEST_FILES_DIRECTORY)

    # Known acquisition date and time for each test file
    known_date_time_mapping = {
        "BW_Pol_1TA.czi": "2025-05-02 17:56:17",
        "BW_Pol_1TR.czi": "2025-05-02 17:51:50",
        "BW_Pol_1TR_1TA.czi": "2025-05-02 18:26:40",
        "BW_Pol_Bright_1TA.czi": "2025-05-02 17:57:02",
        "BW_Pol_Bright_1TR.czi": "2025-05-02 17:50:43",
        "BW_Pol_Bright_1TR_1TA.czi": "2025-05-02 18:25:47",
        "RGB_Dark_1TA_5x.czi": "2025-05-02 18:11:49",
        "RGB_Dark_1TR_1TA_5x.czi": "2025-05-02 18:18:56",
        "RGB_Dark_1TR_5x.czi": "2025-05-02 18:13:26",
        "RGB_Dark_Bright_1TA_5x.czi": "2025-05-02 18:10:25",
        "RGB_Dark_Bright_1TR_1TA_5x.czi": "2025-05-02 18:20:36",
        "RGB_Dark_Bright_1TR_5x.czi": "2025-05-02 18:14:17",
    }

    # Iterate through each test file and compare acquisition date/time
    for each_test_file in all_test_files:
        # Extract acquisition date and time from the file
        found_date_time = precon3d.czi_info.acquisition_date_and_time(
            each_test_file.path
        )

        # Get the known date/time for the file
        known_date_time = known_date_time_mapping.get(each_test_file.name)

        # Assert that the found date/time matches the known date/time
        assert found_date_time == known_date_time, (
            f"Mismatch for file {each_test_file.name}: "
            f"found {found_date_time}, expected {known_date_time}"
        )


def test_czi_scene_metadata():
    """Tests the czi metadata is correctly read from the file"""

    # GRAY
    grayscale_tileregion_files = [
        file
        for file in load_test_files(TEST_FILES_DIRECTORY)
        if file.type == "grayscale" and file.format == "tileregion"
    ]

    known_grayscale_scene_metadata_mapping: Dict[str, SceneMetadata] = {
        "BW_Pol_1TR.czi": [
            SceneMetadata(
                scene_name="TR2",
                scene_index=0,
                bounding_box=BoundingBox(h=1613, w=1613, x=0, y=0),
                n_columns=2,
                n_rows=2,
                contour_size="7652.275,8687.214",
                center_position="10557.918,-28368.83",
            )
        ],
        "BW_Pol_Bright_1TR.czi": [
            SceneMetadata(
                scene_name="TR2",
                scene_index=0,
                bounding_box=BoundingBox(h=1613, w=1613, x=0, y=0),
                n_columns=2,
                n_rows=2,
                contour_size="7652.275,8687.214",
                center_position="10557.918,-28368.83",
            )
        ],
    }

    # Iterate through each test file and compare scene metadata
    for each_test_file in grayscale_tileregion_files:
        # Extract scene metadata from the file
        found_metadata = precon3d.czi_info.scene_metadata(each_test_file.path)

        # Get the known metadata for the file
        known_metadata = known_grayscale_scene_metadata_mapping.get(each_test_file.name)

        # # Assert that the found metadata matches the known metadata
        assert found_metadata == known_metadata, (
            f"Mismatch for file {each_test_file.name}:\n"
            f"Found: {found_metadata}\nExpected: {known_metadata}"
        )

    # RGB
    color_tileregion_files = [
        file
        for file in load_test_files(TEST_FILES_DIRECTORY)
        if file.type == "color" and file.format == "tileregion"
    ]

    known_color_scene_metadata_mapping: Dict[str, SceneMetadata] = {
        "RGB_Dark_1TR_5x.czi": [
            SceneMetadata(
                scene_name="TR1",
                scene_index=0,
                bounding_box=BoundingBox(h=762, w=762, x=0, y=0),
                n_columns=2,
                n_rows=2,
                contour_size="3412.161,3637.965",
                center_position="15617.043,-28075.598",
            )
        ],
        "RGB_Dark_Bright_1TR_5x.czi": [
            SceneMetadata(
                scene_name="TR1",
                scene_index=0,
                bounding_box=BoundingBox(h=762, w=762, x=0, y=0),
                n_columns=2,
                n_rows=2,
                contour_size="3412.161,3637.965",
                center_position="15617.043,-28075.598",
            )
        ],
    }

    # Iterate through each test file and compare scene metadata
    for each_test_file in color_tileregion_files:
        # Extract scene metadata from the file
        found_metadata = precon3d.czi_info.scene_metadata(each_test_file.path)

        # Get the known metadata for the file
        known_metadata = known_color_scene_metadata_mapping.get(each_test_file.name)

        # Assert that the found metadata matches the known metadata
        assert found_metadata == known_metadata, (
            f"Mismatch for file {each_test_file.name}:\n"
            f"Found: {found_metadata}\nExpected: {known_metadata}"
        )


def test_simple_metadata():
    """Tests the czi metadata is correctly read from the file"""

    # TileRegions
    tileregion_files = [
        file
        for file in load_test_files(TEST_FILES_DIRECTORY)
        if file.format == "tileregion"
    ]

    known_simple_metadata_mapping: Dict[str, CZIMetadata] = {
        "BW_Pol_1TR.czi": CZIMetadata(
            filename="BW_Pol_1TR",
            author="Robomet",
            date_created="2025-05-02T17:51:50.2890399Z",
            dims="SCMYX",
            dims_shape=[
                {
                    "X": (0, 896),
                    "Y": (0, 896),
                    "C": (0, 1),
                    "M": (0, 4),
                    "S": (0, 1),
                }
            ],
            scene_names=["TR2"],
            scene_metadata=[
                SceneMetadata(
                    scene_name="TR2",
                    scene_index=0,
                    bounding_box=BoundingBox(h=1613, w=1613, x=0, y=0),
                    n_columns=2,
                    n_rows=2,
                    contour_size="7652.275,8687.214",
                    center_position="10557.918,-28368.83",
                )
            ],
            scene_shapes_are_consistent=True,
            channel_names=["Pol"],
            channel_metadata=[
                ChannelMetadata(
                    channel_name="Pol",
                    mode="RL Pol",
                    exposure_time=60000000,
                    reflector="Pol Refl.light",
                    contrast="PolarizedLight",
                    pixel_type="Gray8",
                    pixel_bit_count=8,
                    illumination_wavelength=630,
                    illumination_wavelength_range="615-648",
                    illumination_intensity="100.00 %",
                    detector_binning="5,5",
                )
            ],
            tile_width="4910.08",
            tile_height="4910.08",
            tile_overlap="0.2",
            tile_anchor_mode="TopLeft",
            tile_scan_mode="Meander",
            microscope_name="Axio Observer.Z1 / 7",
            microscope_type="Inverted",
            camera_name="Axiocam820c",
            camera_adapter="1x Camera Adapter",
            camera_pixel_size="13.7,13.7",
            camera_pixel_unit="µm",
            objective_name="EC Epiplan-Neofluar 2,5x/0.06 HD M27",
            objective_magnification="2.5",
            effective_pixel_size="5.48E-06",
            color_mode="gray8",
        ),
        "BW_Pol_Bright_1TR.czi": CZIMetadata(
            filename="BW_Pol_Bright_1TR",
            author="Robomet",
            date_created="2025-05-02T17:50:43.3214197Z",
            dims="SCMYX",
            dims_shape=[
                {
                    "X": (0, 896),
                    "Y": (0, 896),
                    "C": (0, 2),
                    "M": (0, 4),
                    "S": (0, 1),
                }
            ],
            scene_names=["TR2"],
            scene_metadata=[
                SceneMetadata(
                    scene_name="TR2",
                    scene_index=0,
                    bounding_box=BoundingBox(h=1613, w=1613, x=0, y=0),
                    n_columns=2,
                    n_rows=2,
                    contour_size="7652.275,8687.214",
                    center_position="10557.918,-28368.83",
                )
            ],
            scene_shapes_are_consistent=True,
            channel_names=["Bright", "Pol"],
            channel_metadata=[
                ChannelMetadata(
                    channel_name="Bright",
                    mode="RL Brightfield",
                    exposure_time=1000000,
                    reflector="Brightfield Refl.light",
                    contrast="Brightfield",
                    pixel_type="Gray8",
                    pixel_bit_count=8,
                    illumination_wavelength=None,
                    illumination_wavelength_range="370-400,450-488,540-570,615-648",
                    illumination_intensity="5.00 %",
                    detector_binning="5,5",
                ),
                ChannelMetadata(
                    channel_name="Pol",
                    mode="RL Pol",
                    exposure_time=60000000,
                    reflector="Pol Refl.light",
                    contrast="PolarizedLight",
                    pixel_type="Gray8",
                    pixel_bit_count=8,
                    illumination_wavelength=630,
                    illumination_wavelength_range="615-648",
                    illumination_intensity="100.00 %",
                    detector_binning="5,5",
                ),
            ],
            tile_width="4910.08",
            tile_height="4910.08",
            tile_overlap="0.2",
            tile_anchor_mode="TopLeft",
            tile_scan_mode="Meander",
            microscope_name="Axio Observer.Z1 / 7",
            microscope_type="Inverted",
            camera_name="Axiocam820c",
            camera_adapter="1x Camera Adapter",
            camera_pixel_size="13.7,13.7",
            camera_pixel_unit="µm",
            objective_name="EC Epiplan-Neofluar 2,5x/0.06 HD M27",
            objective_magnification="2.5",
            effective_pixel_size="5.48E-06",
            color_mode="gray8",
        ),
        "RGB_Dark_1TR_5x.czi": CZIMetadata(
            filename="RGB_Dark_1TR_5x",
            author="Robomet",
            date_created="2025-05-02T18:13:26.0211578Z",
            dims="SCMYXA",
            dims_shape=[
                {
                    "A": (0, 3),
                    "X": (0, 448),
                    "Y": (0, 448),
                    "C": (0, 1),
                    "M": (0, 4),
                    "S": (0, 1),
                }
            ],
            scene_names=["TR1"],
            scene_metadata=[
                SceneMetadata(
                    scene_name="TR1",
                    scene_index=0,
                    bounding_box=BoundingBox(h=762, w=762, x=0, y=0),
                    n_columns=2,
                    n_rows=2,
                    contour_size="3412.161,3637.965",
                    center_position="15617.043,-28075.598",
                )
            ],
            scene_shapes_are_consistent=True,
            channel_names=["Dark"],
            channel_metadata=[
                ChannelMetadata(
                    channel_name="Dark",
                    mode="RL Darkfield",
                    exposure_time=500000000,
                    reflector="Darkfield Refl.light",
                    contrast="Darkfield",
                    pixel_type="Bgr24",
                    pixel_bit_count=8,
                    illumination_wavelength=None,
                    illumination_wavelength_range="370-400,450-488,540-570,615-648",
                    illumination_intensity="100.00 %",
                    detector_binning="5,5",
                )
            ],
            tile_width="2455.04",
            tile_height="2455.04",
            tile_overlap="0.3",
            tile_anchor_mode="TopLeft",
            tile_scan_mode="Meander",
            microscope_name="Axio Observer.Z1 / 7",
            microscope_type="Inverted",
            camera_name="Axiocam820c",
            camera_adapter="1x Camera Adapter",
            camera_pixel_size="27.4,27.4",
            camera_pixel_unit="µm",
            objective_name="EC Epiplan-Neofluar 5x/0.13 HD DIC M27",
            objective_magnification="5",
            effective_pixel_size="5.48E-06",
            color_mode="bgr24",
        ),
        "RGB_Dark_Bright_1TR_5x.czi": CZIMetadata(
            filename="RGB_Dark_Bright_1TR_5x",
            author="Robomet",
            date_created="2025-05-02T18:14:17.2530257Z",
            dims="SCMYXA",
            dims_shape=[
                {
                    "A": (0, 3),
                    "X": (0, 448),
                    "Y": (0, 448),
                    "C": (0, 2),
                    "M": (0, 4),
                    "S": (0, 1),
                }
            ],
            scene_names=["TR1"],
            scene_metadata=[
                SceneMetadata(
                    scene_name="TR1",
                    scene_index=0,
                    bounding_box=BoundingBox(h=762, w=762, x=0, y=0),
                    n_columns=2,
                    n_rows=2,
                    contour_size="3412.161,3637.965",
                    center_position="15617.043,-28075.598",
                )
            ],
            scene_shapes_are_consistent=True,
            channel_names=["Bright", "Dark"],
            channel_metadata=[
                ChannelMetadata(
                    channel_name="Bright",
                    mode="RL Brightfield",
                    exposure_time=2000000,
                    reflector="Brightfield Refl.light",
                    contrast="Brightfield",
                    pixel_type="Bgr24",
                    pixel_bit_count=8,
                    illumination_wavelength=None,
                    illumination_wavelength_range="370-400,450-488,540-570,615-648",
                    illumination_intensity="20.00 %",
                    detector_binning="5,5",
                ),
                ChannelMetadata(
                    channel_name="Dark",
                    mode="RL Darkfield",
                    exposure_time=500000000,
                    reflector="Darkfield Refl.light",
                    contrast="Darkfield",
                    pixel_type="Bgr24",
                    pixel_bit_count=8,
                    illumination_wavelength=None,
                    illumination_wavelength_range="370-400,450-488,540-570,615-648",
                    illumination_intensity="100.00 %",
                    detector_binning="5,5",
                ),
            ],
            tile_width="2455.04",
            tile_height="2455.04",
            tile_overlap="0.3",
            tile_anchor_mode="TopLeft",
            tile_scan_mode="Meander",
            microscope_name="Axio Observer.Z1 / 7",
            microscope_type="Inverted",
            camera_name="Axiocam820c",
            camera_adapter="1x Camera Adapter",
            camera_pixel_size="27.4,27.4",
            camera_pixel_unit="µm",
            objective_name="EC Epiplan-Neofluar 5x/0.13 HD DIC M27",
            objective_magnification="5",
            effective_pixel_size="5.48E-06",
            color_mode="bgr24",
        ),
    }

    # Iterate through each test file and compare scene metadata
    for each_test_file in tileregion_files:
        # Extract scene metadata from the file
        found_metadata = precon3d.czi_info.simple_metadata(each_test_file.path)

        # Get the known metadata for the file
        known_metadata = known_simple_metadata_mapping.get(each_test_file.name)

        # Assert that the found metadata matches the known metadata
        assert found_metadata == known_metadata, (
            f"Mismatch for file {each_test_file.name}:\n"
            f"Found: {found_metadata}\nExpected: {known_metadata}"
        )


def assert_support_points(
    test_file: FileMetadata,
    found_support_points: SceneSupportPoints,
    known_support_points: SceneSupportPoints,
):
    """assert the support points are equal"""

    # Assert that the found metadata matches the known metadata
    assert len(found_support_points) == len(known_support_points), (
        f"Mismatch in number of support points for file {test_file.name}:\n"
        f"Found: {len(found_support_points)}\nExpected: {len(known_support_points)}"
    )

    for found, known in zip(found_support_points, known_support_points):
        # Compare non-array attributes
        assert found.czi_fname == known.czi_fname, (
            f"Mismatch in czi_fname for file {test_file.name}:\n"
            f"Found: {found.czi_fname}\nExpected: {known.czi_fname}"
        )
        assert found.scene == known.scene, (
            f"Mismatch in scene for file {test_file.name}:\n"
            f"Found: {found.scene}\nExpected: {known.scene}"
        )
        assert abs(float(found.z) - float(known.z)) <= 1e-2, (
            f"Mismatch in z for file {test_file.name}:\n"
            f"Found: {found.z}\nExpected: {known.z}"
        )

        # Compare numpy arrays for z_heights
        assert np.allclose(found.z_heights, known.z_heights, atol=1e-2), (
            f"Mismatch in z_heights for file {test_file.name}:\n"
            f"Found: {found.z_heights}\nExpected: {known.z_heights}"
        )

        # Compare xy_positions (list of TilePosition objects)
        assert len(found.xy_positions) == len(known.xy_positions), (
            f"Mismatch in number of xy_positions for file {test_file.name}:\n"
            f"Found: {len(found.xy_positions)}\nExpected: {len(known.xy_positions)}"
        )
        for found_pos, known_pos in zip(found.xy_positions, known.xy_positions):
            assert found_pos.x == known_pos.x, (
                f"Mismatch in x position for file {test_file.name}:\n"
                f"Found: {found_pos.x}\nExpected: {known_pos.x}"
            )
            assert found_pos.y == known_pos.y, (
                f"Mismatch in y position for file {test_file.name}:\n"
                f"Found: {found_pos.y}\nExpected: {known_pos.y}"
            )


def test_scene_support_points():
    """
    Test that scene support points are correctly extracted from CZI files.

    This test verifies that the scene support points metadata, including Z heights,
    XY positions, and scene names, are correctly read from CZI files with the "tileregion"
    format. The test compares the extracted metadata against known expected values.

    The metadata includes:
    - Scene name
    - Z heights (as a NumPy array)
    - XY positions (as a list of `TilePosition` objects)
    - Scene index
    - Center position

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return anything. It raises an assertion error if the
        extracted metadata does not match the expected metadata.

    Raises
    ------
    AssertionError
        If the extracted scene support points metadata does not match the expected
        metadata for any of the test files.

    Notes
    -----
    The test uses the following known metadata for comparison:
    - `known_scene_support_point`: A dictionary mapping filenames to their expected
      `SceneSupportPoints` metadata.

    Examples
    --------
    Test files with the "tileregion" format are filtered and their metadata is validated:

    >>> test_scene_support_points()
    Test passed successfully.

    If there is a mismatch in metadata, an assertion error is raised:

    AssertionError: Mismatch for file BW_Pol_1TR.czi:
    Found: [SceneSupportPoints(...)]
    Expected: [SceneSupportPoints(...)]
    """

    # TileRegions
    tileregion_files = [
        file
        for file in load_test_files(TEST_FILES_DIRECTORY)
        if file.format == "tileregion"
    ]

    known_scene_support_point: Dict[str, list] = {
        "BW_Pol_1TR.czi": [
            SceneSupportPoints(
                czi_fname="BW_Pol_1TR",
                scene="TR2",
                z="500.01",
                z_heights=np.array([540.201, 548.167, 518.7, 517.193], dtype=float32),
                xy_positions=[
                    TilePosition(x="8007.16", y="-31264.568"),
                    TilePosition(x="13108.677", y="-31264.568"),
                    TilePosition(x="8007.16", y="-25473.092"),
                    TilePosition(x="13108.677", y="-25473.092"),
                ],
            ),
        ],
        "BW_Pol_Bright_1TR.czi": [
            SceneSupportPoints(
                czi_fname="BW_Pol_Bright_1TR",
                scene="TR2",
                z="500.01",
                z_heights=np.array([500.01, 500.01, 500.01, 500.01], dtype=float32),
                xy_positions=[
                    TilePosition(x="8007.16", y="-31264.568"),
                    TilePosition(x="13108.677", y="-31264.568"),
                    TilePosition(x="8007.16", y="-25473.092"),
                    TilePosition(x="13108.677", y="-25473.092"),
                ],
            ),
        ],
        "RGB_Dark_1TR_5x.czi": [
            SceneSupportPoints(
                czi_fname="RGB_Dark_1TR_5x",
                scene="TR1",
                z="5215.95",
                z_heights=np.array([5215.95, 5215.95, 5215.95, 5215.95], dtype=float32),
                xy_positions=[
                    TilePosition(x="14479.656", y="-29288.253"),
                    TilePosition(x="16754.43", y="-29288.253"),
                    TilePosition(x="14479.656", y="-26862.943"),
                    TilePosition(x="16754.43", y="-26862.943"),
                ],
            )
        ],
        "RGB_Dark_Bright_1TR_5x.czi": [
            SceneSupportPoints(
                czi_fname="RGB_Dark_Bright_1TR_5x",
                scene="TR1",
                z="5215.95",
                z_heights=np.array([5215.95, 5215.95, 5215.95, 5215.95], dtype=float32),
                xy_positions=[
                    TilePosition(x="14479.656", y="-29288.253"),
                    TilePosition(x="16754.43", y="-29288.253"),
                    TilePosition(x="14479.656", y="-26862.943"),
                    TilePosition(x="16754.43", y="-26862.943"),
                ],
            )
        ],
    }

    # Iterate through each test file and compare scene metadata
    for each_test_file in tileregion_files:
        # Extract scene metadata from the file
        found_support_points = precon3d.czi_info.scene_support_points(
            each_test_file.path
        )

        # Get the known metadata for the file
        known_support_points = known_scene_support_point.get(each_test_file.name)

        assert_support_points(
            each_test_file, found_support_points, known_support_points
        )


def test_scene_support_points_global():
    """Verify the support point for each scene is properly extracted from the zeiss czi metadata"""

    # TileArray
    tilearray_files = [
        file
        for file in load_test_files(TEST_FILES_DIRECTORY)
        if file.format == "tilearray"
    ]

    known_global_ta_support_points: Dict[str, list] = {
        "BW_Pol_1TA.czi": [
            SceneSupportPoints(
                czi_fname="BW_Pol_1TA",
                scene="robomet-polarized",
                z=500.005,
                z_heights=np.array([500.01, 500.01, 500.0, 500.0], dtype=float32),
                xy_positions=[
                    TilePosition(x="9969", y="-30442"),
                    TilePosition(x="13398", y="-30442"),
                    TilePosition(x="13398", y="-26956"),
                    TilePosition(x="13398", y="-26956"),
                ],
            )
        ],
        "BW_Pol_Bright_1TA.czi": [
            SceneSupportPoints(
                czi_fname="BW_Pol_Bright_1TA",
                scene="robomet-polarized",
                z=538.3342,
                z_heights=np.array([546.707, 545.525, 530.614, 530.491], dtype=float32),
                xy_positions=[
                    TilePosition(x="9969", y="-30442"),
                    TilePosition(x="13398", y="-30442"),
                    TilePosition(x="13398", y="-26956"),
                    TilePosition(x="13398", y="-26956"),
                ],
            )
        ],
        "RGB_Dark_1TA_5x.czi": [
            SceneSupportPoints(
                czi_fname="RGB_Dark_1TA_5x",
                scene="robomet-polarized",
                z=5039.6953,
                z_heights=np.array(
                    [5083.329, 5031.603, 4995.992, 5047.857], dtype=float32
                ),
                xy_positions=[
                    TilePosition(x="15059", y="-28964"),
                    TilePosition(x="16737", y="-28964"),
                    TilePosition(x="16737", y="-27323"),
                    TilePosition(x="15119", y="-27323"),
                ],
            )
        ],
        "RGB_Dark_Bright_1TA_5x.czi": [
            SceneSupportPoints(
                czi_fname="RGB_Dark_Bright_1TA_5x",
                scene="robomet-polarized",
                z=5215.95,
                z_heights=np.array([5215.95, 5215.95, 5215.95, 5215.95], dtype=float32),
                xy_positions=[
                    TilePosition(x="15059", y="-28964"),
                    TilePosition(x="16737", y="-28964"),
                    TilePosition(x="16737", y="-27323"),
                    TilePosition(x="15119", y="-27323"),
                ],
            )
        ],
    }

    # Iterate through each test file and compare scene support points
    for each_test_file in tilearray_files:
        # Extract scene support points from the file
        found_support_points = precon3d.czi_info.scene_support_points_global(
            each_test_file.path
        )

        # Get the known support points for the file
        known_support_points = known_global_ta_support_points.get(each_test_file.name)

        assert_support_points(
            each_test_file, found_support_points, known_support_points
        )

    # Mixed
    mixed_tiles_files = [
        file for file in load_test_files(TEST_FILES_DIRECTORY) if file.format == "mixed"
    ]

    known_global_mixed_support_points: Dict[str, list] = {
        "BW_Pol_1TR_1TA.czi": [
            SceneSupportPoints(
                czi_fname="BW_Pol_1TR_1TA",
                scene="robomet-polarized",
                z=540.0235,
                z_heights=np.array([523.595, 554.989, 537.821, 543.689], dtype=float32),
                xy_positions=[
                    TilePosition(x="7619", y="-34166"),
                    TilePosition(x="15753", y="-29243"),
                    TilePosition(x="17081", y="-25735"),
                    TilePosition(x="12995", y="-28044"),
                ],
            )
        ],
        "BW_Pol_Bright_1TR_1TA.czi": [
            SceneSupportPoints(
                czi_fname="BW_Pol_Bright_1TR_1TA",
                scene="robomet-polarized",
                z=480.59076,
                z_heights=np.array([291.8, 554.989, 537.821, 537.753], dtype=float32),
                xy_positions=[
                    TilePosition(x="7619", y="-34166"),
                    TilePosition(x="15753", y="-29243"),
                    TilePosition(x="17081", y="-25735"),
                    TilePosition(x="12995", y="-28044"),
                ],
            )
        ],
        "RGB_Dark_1TR_1TA_5x.czi": [
            SceneSupportPoints(
                czi_fname="RGB_Dark_1TR_1TA_5x",
                scene="robomet-polarized",
                z=5038.0635,
                z_heights=np.array(
                    [5033.699, 4987.019, 4950.974, 5002.625, 5216.0],
                    dtype=float32,
                ),
                xy_positions=[
                    TilePosition(x="15059", y="-28964"),
                    TilePosition(x="16737", y="-28964"),
                    TilePosition(x="16737", y="-27323"),
                    TilePosition(x="15119", y="-27323"),
                    TilePosition(x="7619", y="-26917"),
                ],
            )
        ],
        "RGB_Dark_Bright_1TR_1TA_5x.czi": [
            SceneSupportPoints(
                czi_fname="RGB_Dark_Bright_1TR_1TA_5x",
                scene="robomet-polarized",
                z=5045.16,
                z_heights=np.array(
                    [5034.205, 4987.349, 4951.029, 5002.632, 5250.586],
                    dtype=float32,
                ),
                xy_positions=[
                    TilePosition(x="15059", y="-28964"),
                    TilePosition(x="16737", y="-28964"),
                    TilePosition(x="16737", y="-27323"),
                    TilePosition(x="15119", y="-27323"),
                    TilePosition(x="7619", y="-26917"),
                ],
            )
        ],
    }

    # Iterate through each test file and compare scene support points
    for each_test_file in mixed_tiles_files:
        # Extract scene support points from the file
        found_support_points = precon3d.czi_info.scene_support_points_global(
            each_test_file.path
        )

        # Get the known support points for the file
        known_support_points = known_global_mixed_support_points.get(
            each_test_file.name
        )

        assert_support_points(
            each_test_file, found_support_points, known_support_points
        )


def test_metadata_as_xml():
    """to come"""

    all_test_files = load_test_files(TEST_FILES_DIRECTORY)

    # Iterate through each test file and compare acquisition date/time
    for each_test_file in all_test_files:
        # Extract acquisition date and time from the file
        found_metadata_xml = precon3d.czi_info.metadata_as_xml(each_test_file.path)

        assert found_metadata_xml.exists()  # file created

        found_metadata_xml.unlink()  # Delete the file


def test_save_tiles():
    """Test that .tif tiles are correctly saved into a nested folder structure."""

    # Define the output directory for tiles
    output_directory = TEST_FILES_DIRECTORY / "output_tiles"

    # Ensure the output directory exists before testing
    output_directory.mkdir(exist_ok=True)

    # must be directory with czi files
    precon3d.czi_info.save_tiles(TEST_FILES_DIRECTORY, output_directory)

    # Assert that the output directory contains subdirectories or files
    assert output_directory.exists(), f"Output directory does not exist"
    assert any(output_directory.iterdir()), f"No tiles were saved"

    # Collect all saved .tif files
    saved_tiles = list(output_directory.rglob("*.tif"))

    # Assert that at least one .tif file was created
    assert len(saved_tiles) > 0, f"No .tif tiles were created"

    # Cleanup
    rmdir(output_directory)
