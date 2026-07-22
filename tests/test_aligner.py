# test aligner.py
from pathlib import Path
from typing import Final, Tuple
import math

# import numpy as np
import pytest

import precon3d.aligner as al
from precon3d.stitcher import FijiAttrs
import precon3d.utility as ut
from precon3d.utility import user_settings


def align_gray_test_files() -> Tuple[Path, Path]:
    """The relative path and file string locating the test greyscale czi file."""

    return Path(__file__).parent.joinpath(
        "files", "align", "gray", "test_gray_slice_1.tif"
    ), Path(__file__).parent.joinpath(
        "files", "align", "gray", "test_gray_slice_2.tif"
    )


def align_color_test_files() -> Tuple[Path, Path]:
    """The relative path and file string locating the test greyscale czi file."""

    return Path(__file__).parent.joinpath(
        "files", "align", "color", "small", "color_slice_1_500pix.tif"
    ), Path(__file__).parent.joinpath(
        "files", "align", "color", "small", "color_slice_2_500pix.tif"
    )


@ut.run_on_local_machine
def test_calc_shifts_turboreg_gray():
    """Test the calc_shifts_turboreg function."""

    us = user_settings()

    reference_image, moving_image = align_gray_test_files()

    # subarea FALSE
    # Create Alignment and Fiji attributes
    alignment_attrs = al.AlignmentAttrs(
        registration_method="turboreg", use_subarea=False
    )
    fiji_attrs = FijiAttrs(fiji_app=us.fiji_app)

    # Call the function
    found_shifts_gray = al.calc_shifts_turboreg(
        reference_image, moving_image, alignment_attrs, fiji_attrs
    )

    # Check if the output is as expected
    assert isinstance(
        found_shifts_gray, al.DeltaTranslation
    ), "Output should be a DeltaTranslation."

    known_shift_dx = 42.019
    known_shift_dy = -20.02
    assert math.isclose(found_shifts_gray.dx, known_shift_dx, rel_tol=1e-1)
    assert math.isclose(found_shifts_gray.dy, known_shift_dy, rel_tol=1e-1)

    # subarea TRUE
    # makeRectangle(213, 331, 552, 378);
    alignment_attrs_2 = al.AlignmentAttrs(
        registration_method="turboreg",
        use_subarea=True,
        subarea_x=213,
        subarea_y=331,
        subarea_width=552,
        subarea_height=378,
    )

    found_shifts_gray_2 = al.calc_shifts_turboreg(
        reference_image, moving_image, alignment_attrs_2, fiji_attrs
    )

    known_shift_dx = 42.019
    known_shift_dy = -20.02
    assert math.isclose(found_shifts_gray_2.dx, known_shift_dx, rel_tol=1e-1)
    assert math.isclose(found_shifts_gray_2.dy, known_shift_dy, rel_tol=1e-1)


@ut.run_on_local_machine
def test_calc_shifts_turboreg_color():
    """Test the calc_shifts_turboreg function."""

    us = user_settings()

    reference_image, moving_image = align_color_test_files()

    # Create Alignment and Fiji attributes
    alignment_attrs = al.AlignmentAttrs(
        registration_method="turboreg", use_subarea=False
    )

    fiji_attrs = FijiAttrs(fiji_app=us.fiji_app)

    # Call the function
    found_shifts_color = al.calc_shifts_turboreg(
        reference_image, moving_image, alignment_attrs, fiji_attrs
    )

    # Check if the output is as expected
    assert isinstance(
        found_shifts_color, al.DeltaTranslation
    ), "Output should be a DeltaTranslation."

    known_shift_dx = 21.01
    known_shift_dy = -10.013
    assert math.isclose(found_shifts_color.dx, known_shift_dx, rel_tol=1e-1)
    assert math.isclose(found_shifts_color.dy, known_shift_dy, rel_tol=1e-1)

    # subarea TRUE
    # makeRectangle(96, 186, 290, 158);
    alignment_attrs_2 = al.AlignmentAttrs(
        registration_method="turboreg",
        use_subarea=True,
        subarea_x=96,
        subarea_y=186,
        subarea_width=290,
        subarea_height=158,
    )

    found_shifts_color_2 = al.calc_shifts_turboreg(
        reference_image, moving_image, alignment_attrs_2, fiji_attrs
    )

    known_shift_dx = 21.01
    known_shift_dy = -10.013
    assert math.isclose(found_shifts_color_2.dx, known_shift_dx, rel_tol=1e-1)
    assert math.isclose(found_shifts_color_2.dy, known_shift_dy, rel_tol=1e-1)


# def test_calc_shifts_using_keypoints_gray(test_files_gray):
#     """Test the calc_shifts_using_keypoints function."""
#     reference_image, moving_image = test_files_gray

#     # Create Alignment attributes
#     alignment_attrs = al.AlignmentAttrs(
#         registration_method="keypoints", use_subarea=False
#     )

#     found_shifts = al.calc_shifts_using_keypoints(
#         reference_image, moving_image, alignment_attrs
#     )

#     # Check if the output is as expected
#     assert isinstance(
#         found_shifts, al.DeltaTranslation
#     ), "Output should be a DeltaTranslation."

#     known_shift_dx = 42.0
#     known_shift_dy = -20.0

#     assert math.isclose(found_shifts.dx, known_shift_dx, rel_tol=1e-1)
#     assert math.isclose(found_shifts.dy, known_shift_dy, rel_tol=1e-1)


def test_calc_shifts_phase_corr_gray():
    """Test the calc_shifts_phase_corr function."""

    reference_image, moving_image = align_gray_test_files()

    # Create Alignment attributes
    alignment_attrs = al.AlignmentAttrs(
        registration_method="phasecorr", use_subarea=False
    )

    found_shifts = al.calc_shifts_phase_corr(
        reference_image, moving_image, alignment_attrs
    )

    # Check if the output is as expected
    assert isinstance(
        found_shifts, al.DeltaTranslation
    ), "Output should be a DeltaTranslation."

    known_shift_dx = 42.0
    known_shift_dy = -20.0

    assert math.isclose(found_shifts.dx, known_shift_dx, rel_tol=1e-1)
    assert math.isclose(found_shifts.dy, known_shift_dy, rel_tol=1e-1)


def test_calc_shifts_phase_corr_pyramid_gray():
    """Test the calc_shifts_phase_corr function."""

    reference_image, moving_image = align_gray_test_files()

    # Create Alignment attributes
    alignment_attrs = al.AlignmentAttrs(
        registration_method="phasecorr", use_subarea=False
    )

    found_shifts = al.calc_shifts_phase_corr_pyramid(
        reference_image, moving_image, alignment_attrs
    )

    # Check if the output is as expected
    assert isinstance(
        found_shifts, al.DeltaTranslation
    ), "Output should be a DeltaTranslation."

    known_shift_dx = 41.9
    known_shift_dy = -20.0

    assert math.isclose(found_shifts.dx, known_shift_dx, rel_tol=1e-1)
    assert math.isclose(found_shifts.dy, known_shift_dy, rel_tol=1e-1)
