## python standard libraries
from pathlib import Path

# 3rd party libraries
import pytest
import yaml
from unittest.mock import patch

# Local
from tests.test_files import *

import precon3d.utility as ut


# @pytest.fixture
# def test_czi_files_dir() -> str:
#     """The relative path and file string locating the default yml test file."""

#     return Path(__file__).parent.joinpath("files", "czi")


def test_sorted_files():
    """Tests that files in the directory have been found."""

    found_file_list = ut.sorted_files(TEST_FILES_DIRECTORY, ".czi")

    assert len(found_file_list) == 12


# read_config
# def test_read_config_valid_yaml(tmp_path):
#     """Test reading a valid YAML configuration file."""
#     yaml_content = """
#     key1: value1
#     key2:
#       - item1
#       - item2
#     """
#     config_file = tmp_path / "config.yml"
#     config_file.write_text(yaml_content)

#     config = ut.read_config(config_file)

#     expected_config = {"key1": "value1", "key2": ["item1", "item2"]}

#     assert config == expected_config


# def test_read_config_invalid_yaml(tmp_path):
#     """Test reading an invalid YAML configuration file."""
#     invalid_yaml_content = """
#     key1: value1
#     key2: incorrect
#       - item1
#       - item2
#     key3: value3
#     key4:
#     key5: value5: value6  # Invalid syntax
#     """
#     config_file = tmp_path / "invalid_config.yml"
#     config_file.write_text(invalid_yaml_content)

#     with pytest.raises(yaml.YAMLError):
#         ut.read_config(config_file)


# yes_no
def test_yes_no_valid_yes():
    """Test the yes_no function with a valid 'yes' input."""
    with patch("builtins.input", return_value="y"):
        assert ut.yes_no("Do you want to continue?") is True


def test_yes_no_valid_no():
    """Test the yes_no function with a valid 'no' input."""
    with patch("builtins.input", return_value="n"):
        assert ut.yes_no("Do you want to continue?") is False


def test_yes_no_invalid_input():
    """Test the yes_no function with invalid input followed by valid input."""
    # First input is invalid, second is valid 'yes'
    with patch("builtins.input", side_effect=["x", "y"]):
        assert ut.yes_no("Do you want to continue?") is True

    # First input is invalid, second is valid 'no'
    with patch("builtins.input", side_effect=["x", "n"]):
        assert ut.yes_no("Do you want to continue?") is False


# string_to_float_list
def test_string_to_float_list_valid_numbers():
    """Test the function with valid string representations of lists."""
    assert ut.string_to_float_list("[1.0 2.5 3.3]") == [1.0, 2.5, 3.3]
    assert ut.string_to_float_list("[4.0 5.1]") == [4.0, 5.1]
    assert ut.string_to_float_list("[10 20 30]") == [10.0, 20.0, 30.0]


def test_string_to_float_list_empty():
    """Test the function with an empty list representation."""
    assert ut.string_to_float_list("[]") == []


def test_string_to_float_list_whitespace():
    """Test the function with whitespace in the input string."""
    assert ut.string_to_float_list("[ 1.0   2.5   3.3 ]") == [1.0, 2.5, 3.3]
    assert ut.string_to_float_list("[ 10  20  30 ]") == [10.0, 20.0, 30.0]


def test_string_to_float_list_single_value():
    """Test the function with a single value in the list."""
    assert ut.string_to_float_list("[42]") == [42.0]
    assert ut.string_to_float_list("[0]") == [0.0]


def test_string_to_float_list_invalid_input():
    """Test the function with invalid input to ensure it raises a ValueError."""
    with pytest.raises(ValueError):
        ut.string_to_float_list("[1.0 two 3.3]")  # Non-numeric value
    with pytest.raises(ValueError):
        ut.string_to_float_list("[1.0 2.5, ]")  # Trailing comma


# rmdir
def test_rmdir_empty_directory():
    """Test that an empty directory is removed."""
    empty_dir = TEST_FILES_DIRECTORY / "empty_dir"
    empty_dir.mkdir()

    ut.rmdir(empty_dir)

    assert not empty_dir.exists()


def test_rmdir_directory_with_files():
    """Test that a directory with files is removed."""
    dir_with_files = TEST_FILES_DIRECTORY / "dir_with_files"
    dir_with_files.mkdir()

    file1 = dir_with_files / "file1.txt"
    file1.write_text("Hello, World!")

    file2 = dir_with_files / "file2.txt"
    file2.write_text("Goodbye, World!")

    ut.rmdir(dir_with_files)

    assert not dir_with_files.exists()
    assert not file1.exists()
    assert not file2.exists()


def test_rmdir_directory_with_subdirectories():
    """Test that a directory with subdirectories is removed."""
    parent_dir = TEST_FILES_DIRECTORY / "parent_dir"
    parent_dir.mkdir()

    sub_dir = parent_dir / "sub_dir"
    sub_dir.mkdir()

    file_in_sub = sub_dir / "file_in_sub.txt"
    file_in_sub.write_text("This is a file in a subdirectory.")

    ut.rmdir(parent_dir)

    assert not parent_dir.exists()
    assert not sub_dir.exists()
    assert not file_in_sub.exists()


def test_rmdir_non_existent_directory():
    """Test that calling rmdir on a non-existent directory does nothing."""
    non_existent_dir = TEST_FILES_DIRECTORY / "non_existent_dir"

    # Ensure it does not exist
    assert not non_existent_dir.exists()

    # Call rmdir and check that it still does not exist
    ut.rmdir(non_existent_dir)

    assert not non_existent_dir.exists()


# def test_parse_type(test_dir):
# def test_file_open(test_dir):
#     """Tests parsing of input image type"""

#     image = "bw.tif"
#     image_path = test_dir.joinpath(image)
#     converted_image = ut.file_open(image_path)

#     assert converted_image.data[40, 58] == 117

#     image = "rgb_merged.tif"  # overwrite
#     image_path = test_dir.joinpath(image)
#     converted_image = ut.file_open(image_path)

#     assert converted_image.R.data[32, 19] == 39
#     assert converted_image.G.data[32, 19] == 60
#     assert converted_image.B.data[32, 19] == 51

#     image = "rgb_stack.tif"  # overwrite
#     image_path = test_dir.joinpath(image)
#     converted_image = ut.file_open(image_path)

#     assert converted_image.R.data[150, 184] == 143
#     assert converted_image.G.data[150, 184] == 105
#     assert converted_image.B.data[150, 184] == 84
