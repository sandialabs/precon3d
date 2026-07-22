"""Tests the shading_correction module."""

# python standard libraries
from pathlib import Path
from typing import NamedTuple

# 3rd party libraries
import pytest
import yaml
import skimage.io as skio

# local libraries
import precon3d.utility as ut
import precon3d.shading_correction as shading


# TODO, generate example images and create tests.

# TODO, don't use local files.
# @pytest.fixture
# def test_files() -> Path:
#     """The relative path and file string locating the simple test files."""

#     return Path("/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-2/Pol")


# @pytest.fixture
# def ref_file() -> Path:
#     """reference tile"""

#     return Path("/Users/pchao/Documents/2024/Ti/AM_Ti/forBasic/Pol0000.tif")
# def test_img_attrs(ref_file):
#     """Test the img_attrs function"""

#     img = skio.imread(ref_file)

#     img_stats = shading.img_attrs(img)

#     assert img_stats.mean == pytest.approx(1721.790214695328)


# def test_imagestats(ref_path, test_files_dir, test_file_index):
#     """
#     Helper function to provide ImageStats from paths

#     return  -> tuple(shading.ImageStats, shading.ImageStats)

#     """
#     ref_img = skio.imread(ref_path)
#     ref_img_stats = shading.img_attrs(ref_img)

#     test_files_paths = ut.sorted_files(test_files_dir, ".tif")
#     test_img = skio.imread(test_files_paths[test_file_index])
#     test_img_stats = shading.img_attrs(test_img)

#     return (ref_img_stats, test_img_stats)


# def test_subimages_similar_by_ssim():
#     # use local files for now

#     ref_image_path = Path(
#         "c:/Users/pchao/Documents/_raw_pol_data/Ti7Al_240925_original_analyzer/precon3d_manual_output/shading_correction/Bright/Slice-1_CenterFOV_Bright_0027.tif"
#     )
#     found_image_path = Path(
#         "c:/Users/pchao/Documents/_raw_pol_data/Ti7Al_240925_original_analyzer/precon3d_manual_output/shading_correction/Bright/Slice-1_CenterFOV_Bright_0062.tif"
#     )
#     ref_image = skio.imread(ref_image_path)
#     found_image = skio.imread(found_image_path)
#     result = shading.subimages_similar_by_ssim(ref_image, found_image, 10, 0.95)

#     assert result is False

#     found_image_path = Path(
#         "c:/Users/pchao/Documents/_raw_pol_data/Ti7Al_240925_original_analyzer/precon3d_manual_output/shading_correction/Bright/Slice-1_CenterFOV_Bright_0012.tif"
#     )
#     found_image = skio.imread(found_image_path)
#     result = shading.subimages_similar_by_ssim(ref_image, found_image, 10, 0.95)

#     assert result is False

#     result = shading.subimages_similar_by_ssim(ref_image, found_image, 10, 0.9)

#     assert result is True


# def test_similar_images(ref_file, test_files):
#     """
#     test image comparison function
#     """

#     # not on sample
#     [img_stats_ref, img_stats_test] = test_imagestats(ref_file, test_files, 1)
#     assert shading.similar_images(img_stats_ref, img_stats_test) is False

#     # partially on sample
#     [img_stats_ref, img_stats_test] = test_imagestats(ref_file, test_files, 14)
#     assert shading.similar_images(img_stats_ref, img_stats_test) is False

#     # on sample
#     [img_stats_ref, img_stats_test] = test_imagestats(ref_file, test_files, 51)
#     assert shading.similar_images(img_stats_ref, img_stats_test) is True


# def test_downselect_tiles(ref_file, test_files):
#     """_summary_

#     Args:
#         ref_file (_type_): _description_
#         test_files (_type_): _description_
#     """

#     output_dir = Path("/Users/pchao/Documents/2024/Ti/AM_Ti").joinpath("shading_images")

#     test_dir_list = [
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-2/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-3/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-4/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-5/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-6/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-7/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-8/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-9/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-10/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-11/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-12/Pol",
#         "/Users/pchao/Documents/2024/Ti/AM_Ti/precon3d/Tiles/TR1/Slice-13/Pol",
#     ]
#     for each_dir_path in test_dir_list:

#         shading.downselect_tiles(
#             ref_image_path=ref_file,
#             tiles_dir=Path(each_dir_path),
#             output_dir=output_dir,
#         )


# def test_downselect_tiles2():
#     """to come

#     Args:
#         ref_file (_type_): _description_
#         test_files (_type_): _description_
#     """
#     ref_img = Path(
#         "F:/AM_BiTe/precon3d_output/Run 1/Tiles/Region_1/Slice-22/Pol/Slice-22_Region_1_Pol_0005.tif"
#     )

#     output_dir = Path("F:/AM_BiTe/precon3d_output/").joinpath("shading_images", "Run 2")

#     test_dir = Path("F:/AM_BiTe/precon3d_output/Run 2")

#     downselected_paths = ut.downselect_paths(test_dir, "Pol")

#     for each_dir_path in downselected_paths:

#         shading.downselect_tiles(
#             ref_image_path=ref_img,
#             tiles_dir=Path(each_dir_path),
#             output_dir=output_dir,
#         )
