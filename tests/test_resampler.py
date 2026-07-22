""" "This module tests the resampler services."""

# Default python modules
from pathlib import Path
import numpy as np
import tempfile

# 3rd party modules
import pytest
import pandas as pd
import ast

# local modules
import precon3d.resampler as rs
import precon3d.factory as gen_types
import precon3d.utility as ut
from precon3d.czi_info import scene_support_points

# pylint: disable=wildcard-import
from precon3d.custom_types import *
from tests.test_files import *


# def test_resampler_params(test_yml_params):
#     """Test the resampler_params is able to parse the yml file"""

#     config_dict = ut.read_config(test_yml_params)

#     user_resampler_params = gen_types.create_resampler_parameters(config_dict)

#     assert (
#         user_resampler_params.input_directory
#         == Path("tests/files/czi").expanduser()
#     )
#     assert user_resampler_params.file_extension == ".czi"
#     assert (
#         user_resampler_params.output_directory
#         == Path("tests/files/output").expanduser()
#     )


def test_dump_scene_support_point_info():
    """test the scene support points can be saved properly to a csv file"""

    all_test_files = load_test_files(TEST_FILES_DIRECTORY)

    for each_test_file in all_test_files:
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
            temp_csv_path = Path(temp_csv.name)

        try:
            # Create a SceneSupportPoints instance
            sp_info = scene_support_points(each_test_file.path)

            for each_scene_sp_info in sp_info:
                # Call the function to dump the scene support point info
                rs.dump_scene_support_point_info(each_scene_sp_info, temp_csv_path)

                # Read the contents of the CSV file
                with open(temp_csv_path, "r", encoding="utf-8") as csvfile:
                    lines = csvfile.readlines()

                # Check if the header is written correctly
                assert (
                    lines[0].strip()
                    == "File name,Scene name,Average height (um),All heights (um)"
                )

                # Check if the data row is written correctly
                # expected_average_height = np.mean(each_scene_sp_info.z_heights)

                # Assert that the average height matches the calculated mean
                # assert np.isclose(
                #     float(each_scene_sp_info.z), expected_average_height, rtol=1e-2
                # ), f"Mismatch in {each_test_file.name}: {each_scene_sp_info.z} != {expected_average_height}"

                # Call the function again to test appending
                rs.dump_scene_support_point_info(each_scene_sp_info, temp_csv_path)

                # Read the contents of the CSV file again
                with open(temp_csv_path, "r", encoding="utf-8") as csvfile:
                    lines = csvfile.readlines()

                    # Check if a second data row is appended correctly
                    assert len(lines) == 3  # Two data rows now

        finally:
            # Clean up the temporary file
            temp_csv_path.unlink()


# def test_raw_z_heights(test_yml_params):
#     """Test the resampler_params is able to parse the yml file
#     and save the support point information from the test czi files"""

#     config_dict = ut.read_config(test_yml_params)
#     user_resampler_params = gen_types.create_resampler_parameters(config_dict)
#     sp_filepath = rs.save_support_point_info(user_resampler_params)

#     # read in the heights
#     found_z_heights = rs.raw_z_heights(sp_filepath)

#     known_z_heights = np.array([1300.17004395, 7654.19075521])

#     # Assert that the z heights match
#     assert np.all(np.isclose(found_z_heights, known_z_heights, rtol=1e-2))

#     # Clean up and remove the test files
#     ut.rmdir(sp_filepath.parent)


# def test_save_support_point_info(test_yml_params):
#     """Test the resampler_params is able to parse the yml file
#     and save the support point information from the test czi files"""

#     config_dict = ut.read_config(test_yml_params)

#     user_resampler_params = gen_types.create_resampler_parameters(config_dict)

#     # execute function
#     sp_filepath = rs.save_support_point_info(user_resampler_params)

#     assert sp_filepath.exists()

#     # check the support point informaiton is correct
#     found_df = pd.read_csv(sp_filepath)

#     # Known list of filenames
#     known_filenames = [
#         "test-color-multichannel-singlescene",
#         "test-gray-singlechannel-singlescene",
#     ]

#     known_support_point_heights = [
#         [1300.17004395, 1300.17004395],
#         [7650.04199219, 7666.42822266, 7646.10205078],
#     ]

#     # # Apply the custom function to the column containing the lists
#     # found_data = found_df["All heights (um)"].apply(ut.string_to_float_list).tolist()
#     # # Flatten the list of lists into a single list
#     # flattened_found_data = [item for item in found_data]

#     for index, row in found_df.iterrows():
#         file_name = row["File name"]
#         average_height = row["Average height (um)"]
#         all_heights = row["All heights (um)"]  # Keep as string for conversion

#         # Assert that the filename is in the known list
#         assert file_name in known_filenames, f"Unknown filename: {file_name}"
#         assert file_name == known_filenames[index]

#         # Convert the string representation of the list to a list of floats
#         heights_list = ut.string_to_float_list(all_heights)

#         # Assert the support points are correct
#         assert heights_list == known_support_point_heights[index]

#         # Check the mean is properly calculated
#         # Calculate the mean of all heights
#         calculated_mean = np.mean(heights_list)

#         # Assert that the average height matches the calculated mean
#         assert np.isclose(
#             average_height, calculated_mean, rtol=1e-5
#         ), f"Mismatch in {file_name}: {average_height} != {calculated_mean}"

#     # Clean up and remove the test files
#     ut.rmdir(sp_filepath.parent)


# def test_resample_method():
#     """Assures that the ResampleMethod is as anticipated."""
#     aa = rs.ResampleMethod

#     assert aa.SIMPLE.value == 1
#     assert aa.ROBUST.value == 2


def test_basic_functionality_of_fit_linear_segments():
    """Test fitting linear segments to a simple linear dataset."""
    x_values = np.array([0, 1, 2, 3, 4, 5])
    y_values = np.array([0, 0.8, 2.1, 3.2, 4.1, 5.1])
    count = 3

    px, py = rs.fit_linear_segments(x_values, y_values, count)

    assert px.shape[0] == count + 1
    assert py.shape[0] == count + 1
    assert np.all(px[0] <= px[1]) and np.all(px[1] <= px[2])


def test_non_linear_data():
    """Test fitting linear segments to a quadratic dataset."""
    x_values = np.array([0, 1, 2, 3, 4, 5])
    y_values = np.array([0, 1, 4, 9, 16, 25])  # Quadratic data
    count = 3

    px, py = rs.fit_linear_segments(x_values, y_values, count)

    assert px.shape[0] == count + 1
    assert py.shape[0] == count + 1


def test_large_dataset():
    """Test fitting linear segments to a large dataset with noise."""
    np.random.seed(0)  # For reproducibility
    x_values = np.linspace(0, 100, 1000)
    y_values = 0.5 * x_values + np.random.normal(
        0, 5, size=x_values.shape
    )  # Linear with noise
    count = 5

    px, py = rs.fit_linear_segments(x_values, y_values, count)

    assert px.shape[0] == count + 1
    assert py.shape[0] == count + 1


def test_empty_input():
    """Test that the function raises an error with empty input arrays."""
    x_values = np.array([])
    y_values = np.array([])
    count = 1

    with pytest.raises(ValueError):
        rs.fit_linear_segments(x_values, y_values, count)

    z_heights = np.array([])
    target_thickness = 10
    with pytest.raises(IndexError):
        rs.scene_resample_simple(z_heights, target_thickness)


def test_count_greater_than_points():
    """Test that the function raises an error when the count exceeds the number of points."""
    x_values = np.array([0, 1, 2])
    y_values = np.array([0, 1, 2])
    count = 5  # More segments than points

    with pytest.raises(ValueError):
        rs.fit_linear_segments(x_values, y_values, count)


def test_fit_linear_segments_output_type():
    """Test that the output of the function is of the correct type."""
    x_values = np.array([0, 1, 2, 3, 4, 5])
    y_values = np.array([0, 0.8, 2.1, 3.2, 4.1, 5.1])
    count = 3

    px, py = rs.fit_linear_segments(x_values, y_values, count)

    assert isinstance(px, np.ndarray)
    assert isinstance(py, np.ndarray)


def test_segment_values():
    """Test that the segments are placed correctly for a linear dataset."""
    x_values = np.array([0, 1, 2, 3, 4, 5])
    y_values = np.array([0, 1, 2, 3, 4, 5])  # Linear data
    count = 2

    px, py = rs.fit_linear_segments(x_values, y_values, count)

    # Check that the segments are at the expected positions
    assert np.isclose(px[0], 0)
    assert np.isclose(px[-1], 5)
    assert np.isclose(py[0], 0)
    assert np.isclose(py[-1], 5)


# def test_high_noise_data():
#     """Test fitting linear segments to a dataset with high noise."""
#     np.random.seed(1)  # For reproducibility
#     x_values = np.linspace(0, 100, 1000)
#     y_values = 0.5 * x_values + np.random.normal(
#         0, 20, size=x_values.shape
#     )  # Linear with high noise
#     count = 5

#     px, py = rs.fit_linear_segments(x_values, y_values, count)

#     assert px.shape[0] == count + 1  # Number of segments
#     assert py.shape[0] == count + 1  # Number of segment values


def test_scene_resample_simple():
    """Given an individual scene for a dataset, assures that 3D slides with
    with uniform slice thicknesses is calculated."""

    test_heights = np.array([0, 2, 3, 5, 10, 20, 31, 32, 33, 40, 50])
    test_target_thickness = 10.0
    skip_slice_idx = np.array([1, 3, 5, -1])

    result = rs.scene_resample_simple(test_heights, test_target_thickness)

    assert np.allclose(result, np.array([0, 4, 5, 6, 9, 10]))

    result = rs.scene_resample_simple(
        test_heights, test_target_thickness, skip_slice_idx
    )

    assert np.allclose(result, np.array([0, 4, 4, 6, 9]))


def test_basic_functionality_of_scene_resample_simple():
    """Test basic functionality with a simple input."""
    z_heights = np.array([0, 10, 20, 30, 40, 50])
    target_thickness = 10

    expected_indices = np.array([0, 1, 2, 3, 4, 5])  # All indices should be retained
    result = rs.scene_resample_simple(z_heights, target_thickness)

    assert np.array_equal(result, expected_indices)


def test_skip_slice_indices():
    """Test functionality when some slice indices are skipped."""
    z_heights = np.array([0, 10, 20, 30, 40, 50])
    target_thickness = 10
    skip_slice_idx = np.array([1, 3])  # Skip indices 1 and 3 (10 and 30)

    expected_indices = np.array([0, 0, 2, 2, 4, 5])  # Should retain 0, 20, 40, 50
    result = rs.scene_resample_simple(z_heights, target_thickness, skip_slice_idx)

    assert np.array_equal(result, expected_indices)


def test_target_thickness_larger_than_range():
    """Test functionality when target thickness is larger than the range of heights."""
    z_heights = np.array([0, 5, 10])
    target_thickness = 20  # Larger than the range of heights

    expected_indices = np.array([0, 2])  # Should retain only the first and last indices
    result = rs.scene_resample_simple(z_heights, target_thickness)

    assert np.array_equal(result, expected_indices)


def test_single_height():
    """Test functionality with a single height."""
    z_heights = np.array([10])
    target_thickness = 5

    expected_indices = np.array([0])  # Should retain the only index available
    result = rs.scene_resample_simple(z_heights, target_thickness)

    assert np.array_equal(result, expected_indices)


def test_non_uniform_heights():
    """Test functionality with non-uniform heights."""
    z_heights = np.array([0, 1, 3, 6, 10])
    target_thickness = 2

    expected_indices = np.array(
        [0, 1, 2, 3, 3, 4]
    )  # Should retain indices corresponding to 0, 3, 6, 10
    result = rs.scene_resample_simple(z_heights, target_thickness)

    assert np.array_equal(result, expected_indices)


def test_scene_resample_simple_output_type():
    """Test that the output of the function is of the correct type."""
    z_heights = np.array([0, 10, 20, 30, 40, 50])
    target_thickness = 10

    result = rs.scene_resample_simple(z_heights, target_thickness)

    assert isinstance(result, np.ndarray)
