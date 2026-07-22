"""Tests the Mosaic module."""

# python standard libraries
from functools import singledispatch
from pathlib import Path
from typing import NamedTuple, Dict
import os
import tifffile

# 3rd party libraries
import pytest
import yaml
import skimage
import numpy as np
import imageio

# local libraries
import precon3d.czi_info as ci
import precon3d.mosaic_utils
from precon3d.mosaic_utils import TilePosition
import precon3d.utility as ut

# pylint: disable=wildcard-import
from precon3d.custom_types import *
from tests.test_files import *


# def test_simple_mosaic_gray():
#     """To come"""

#     # GRAY
#     grayscale_tileregion_files = [
#         file
#         for file in load_test_files(TEST_FILES_DIRECTORY)
#         if file.type == "grayscale" and file.format == "tileregion"
#     ]

#     known_simple_mosaic_mapping: Dict[str, list] = {
#         "BW_Pol_1TR.czi": [[
#             (1, 1613, 1613, 1), 4, 'BW_Pol_1TR', 'Pol', 'TR2', 0
#         ]],
#         "BW_Pol_Bright_1TR.czi": [[(1, 1613, 1613, 1), 4, 'BW_Pol_Bright_1TR', 'Bright', 'TR2', 0], [(1, 1613, 1613, 1), 4, 'BW_Pol_Bright_1TR', 'Pol', 'TR2', 0]]}

#     for each_test_file in grayscale_tileregion_files:

#         ai_img, fname = precon3d.mosaic.ai_mosaic(each_test_file.path)

#         found_simple_mosaic = precon3d.mosaic.simple_mosaic(
#             ai_img, ai_img.scenes[0], fname
#         )

#         found_channel_attrs = []
#         for each_channel in found_simple_mosaic:
#             d_shp = each_channel.data.shape
#             n_tiles = len(each_channel.positions)
#             fnm = each_channel.czi_fname
#             ch = each_channel.channel
#             sc = each_channel.scene
#             sc_idx = each_channel.scene_idx
#             simple_mosaic_attr = [d_shp, n_tiles, fnm, ch, sc, sc_idx]
#             found_channel_attrs.append(simple_mosaic_attr)

#         known_simple_mosaic = known_simple_mosaic_mapping.get(each_test_file.name)

#         assert found_channel_attrs == known_simple_mosaic, (
#             f"Mismatch for file {each_test_file.name}:\n"
#             f"Found: {found_channel_attrs}\nExpected: {known_simple_mosaic}"
#         )


# # @ut.run_on_local_machine  # Use with local machine running python 3.12.1 to avoid dask error
# def test_simple_mosaic_color():
#     """To come"""

#     # RGB
#     color_tileregion_files = [
#         file
#         for file in load_test_files(TEST_FILES_DIRECTORY)
#         if file.type == "color" and file.format == "tileregion"
#     ]

#     known_simple_mosaic_mapping: Dict[str, list] = {
#         "RGB_Dark_1TR_5x.czi": [[(4, 448, 448, 3), 4, 'RGB_Dark_1TR_5x', 'Dark', 'TR1', 0]],
#         "RGB_Dark_Bright_1TR_5x.czi": [[(1, 1613, 1613, 1), 4, 'BW_Pol_Bright_1TR', 'Bright', 'TR2', 0], [(1, 1613, 1613, 1), 4, 'BW_Pol_Bright_1TR', 'Pol', 'TR2', 0]]}

#     for each_test_file in color_tileregion_files:

#         ai_img, fname = precon3d.mosaic.ai_mosaic(each_test_file.path)

#         found_simple_mosaic = precon3d.mosaic.simple_mosaic(
#             ai_img, ai_img.scenes[0], fname
#         )

#         found_channel_attrs = []
#         for each_channel in found_simple_mosaic:
#             d_shp = each_channel.data.shape
#             n_tiles = len(each_channel.positions)
#             fnm = each_channel.czi_fname
#             ch = each_channel.channel
#             sc = each_channel.scene
#             sc_idx = each_channel.scene_idx
#             simple_mosaic_attr = [d_shp, n_tiles, fnm, ch, sc, sc_idx]
#             found_channel_attrs.append(simple_mosaic_attr)

#         # known_simple_mosaic = known_simple_mosaic_mapping.get(each_test_file.name)

#         bb = 4
#         # assert found_channel_attrs == known_simple_mosaic, (
#         #     f"Mismatch for file {each_test_file.name}:\n"
#         #     f"Found: {found_channel_attrs}\nExpected: {known_simple_mosaic}"
#         # )


def test_tilestack():
    """To come"""

    tileregion_files = [
        file
        for file in load_test_files(TEST_FILES_DIRECTORY)
        if file.format == "tileregion"
    ]

    known_tilestack_mapping: Dict[str, list] = {
        "BW_Pol_1TR.czi": [
            [
                (4, 896, 896, 1),
                [
                    TilePosition(x=0, y=0),
                    TilePosition(x=717, y=0),
                    TilePosition(x=717, y=717),
                    TilePosition(x=0, y=717),
                ],
                "BW_Pol_1TR",
                "Pol",
                "TR2",
                0,
            ]
        ],
        "BW_Pol_Bright_1TR.czi": [
            [
                (4, 896, 896, 1),
                [
                    TilePosition(x=0, y=0),
                    TilePosition(x=717, y=0),
                    TilePosition(x=717, y=717),
                    TilePosition(x=0, y=717),
                ],
                "BW_Pol_Bright_1TR",
                "Bright",
                "TR2",
                0,
            ],
            [
                (4, 896, 896, 1),
                [
                    TilePosition(x=0, y=0),
                    TilePosition(x=717, y=0),
                    TilePosition(x=717, y=717),
                    TilePosition(x=0, y=717),
                ],
                "BW_Pol_Bright_1TR",
                "Pol",
                "TR2",
                0,
            ],
        ],
        "RGB_Dark_1TR_5x.czi": [
            [
                (4, 448, 448, 3),
                [
                    TilePosition(x=0, y=0),
                    TilePosition(x=314, y=0),
                    TilePosition(x=314, y=314),
                    TilePosition(x=0, y=314),
                ],
                "RGB_Dark_1TR_5x",
                "Dark",
                "TR1",
                0,
            ]
        ],
        "RGB_Dark_Bright_1TR_5x.czi": [
            [
                (4, 448, 448, 3),
                [
                    TilePosition(x=0, y=0),
                    TilePosition(x=314, y=0),
                    TilePosition(x=314, y=314),
                    TilePosition(x=0, y=314),
                ],
                "RGB_Dark_Bright_1TR_5x",
                "Bright",
                "TR1",
                0,
            ],
            [
                (4, 448, 448, 3),
                [
                    TilePosition(x=0, y=0),
                    TilePosition(x=314, y=0),
                    TilePosition(x=314, y=314),
                    TilePosition(x=0, y=314),
                ],
                "RGB_Dark_Bright_1TR_5x",
                "Dark",
                "TR1",
                0,
            ],
        ],
    }

    for each_test_file in tileregion_files:

        (ai_img, fname) = precon3d.mosaic_utils.ai_tilestack(each_test_file.path)

        found_tilestack = precon3d.mosaic_utils.simple_tilestack(
            ai_img, ai_img.scenes[0], fname
        )

        found_channel_attrs = []
        for each_channel in found_tilestack:
            d_shp = each_channel.data.shape
            positions = each_channel.positions
            fnm = each_channel.czi_fname
            ch = each_channel.channel
            sc = each_channel.scene
            sc_idx = each_channel.scene_idx
            found_tilestack_attr = [d_shp, positions, fnm, ch, sc, sc_idx]
            found_channel_attrs.append(found_tilestack_attr)

        known_tilestack = known_tilestack_mapping.get(each_test_file.name)

        assert found_channel_attrs == known_tilestack, (
            f"Mismatch for file {each_test_file.name}:\n"
            f"Found: {found_channel_attrs}\nExpected: {known_tilestack}"
        )


# def test_save_tilestack_color():
#     """To come"""

#     # connector_czi = Path('g:\Robomet Archive Data\FY2025\SA4234_Connector_FY25_pchao\Run 3\Mosaic\Slice-1.czi')
#     # test_file_color = connector_czi

#     (ai_img, fname) = precon3d.mosaic.ai_tilestack(czi_test_files.color.value)

#     # There should be one TileRegion
#     # assert ai_img.scenes == ("TR1",)

#     found_tilestack = precon3d.mosaic.simple_tilestack(
#         ai_img, ai_img.scenes[0], fname
#     )

#     # There should be 2 Channels (Bright, Pol)
#     assert len(found_tilestack) == 2

#     # temp_outdir = Path(__file__).parent.joinpath("temp")
#     # test_config = ut.ConfigSettings(
#     #     fiji_location=None,
#     #     flatfield_location=None,
#     #     data_location=None,
#     #     output_directory=temp_outdir,
#     #     file_extension=None,
#     #     record_log=None,
#     #     save_verbose_data=None,
#     #     blue_only=None,
#     #     data_type=None,
#     #     reference_channel="Bright",
#     # )

#     # for each_tilestack in found_tilestack:
#     #     mosaic.save_tilestack(each_tilestack, test_config)


def test_extract_substring():
    """Test for extract substring"""
    test_string = "Chad and Andrew = (Paul, Chao)"

    known_string = "Paul, Chao"
    found_string = precon3d.mosaic_utils.extract_substring(
        start="(", end=")", string=test_string
    )

    assert known_string == found_string


# @ut.run_on_local_machine
# def test_align_tilestack_positions():
#     """_summary_"""

#     fiji_app = ut.user_settings().fiji_app

#     (ai_img, fname) = precon3d.mosaic.ai_tilestack(czi_test_files.gray.value)
#     #     Path("g:/CMU_DED_Ti64_21T/2D_pol/Mosaic/Slice-2.czi")
#     # )

#     found_tilestack = precon3d.mosaic.simple_tilestack(
#         ai_img, ai_img.scenes[0], fname
#     )

#     unaligned_positions = [
#         TilePosition(x=28331, y=21563),
#         TilePosition(x=29235, y=21563),
#         TilePosition(x=30139, y=21563),
#         TilePosition(x=30139, y=22241),
#         TilePosition(x=29235, y=22241),
#         TilePosition(x=28331, y=22241),
#     ]
#     assert found_tilestack[0].positions == unaligned_positions

#     aligned_tilestack = precon3d.mosaic.align_tilestack_positions(
#         found_tilestack[0], fiji_app=fiji_app
#     )

#     aligned_positions = [
#         TilePosition(x=0.0, y=0.0),
#         TilePosition(x=925.0, y=28.0),
#         TilePosition(x=1850.0, y=56.0),
#         TilePosition(x=1830.0, y=750.0),
#         TilePosition(x=905.0, y=722.0),
#         TilePosition(x=-20.0, y=694.0),
#     ]

#     assert aligned_tilestack.positions == aligned_positions

#     # stitched_result = mosaic.stitch_tiles_from_positions(
#     #     aligned_tilestack, with_dask=True
#     # )

#     # output_small_stitched_file = Path(
#     #     f"g:/CMU_DED_Ti64_21T/{fname}_stitch_tiles_from_positions_supersmall.tif"
#     # )
#     # small_img = skimage.transform.rescale(
#     #     stitched_result.data.squeeze(), 0.01, anti_aliasing=False
#     # )
#     # imageio.v3.imwrite(output_small_stitched_file, small_img)

#     # output_stitched_file = Path(
#     #     f"g:/CMU_DED_Ti64_21T/{fname}_stitch_tiles_from_positions.tif"
#     # )

#     # with tifffile.TiffWriter(output_stitched_file, bigtiff=True) as tiff:
#     #     tiff.write(stitched_result.data.squeeze())

#     # def test_stitch_tiles_from_positions(get_tilestack):
#     #     """_summary_"""

#     #     test_tilestack = get_tilestack

#     #     stitched_result = mosaic.stitch_tiles_from_positions(
#     #         test_tilestack, with_dask=False
#     #     )

#     #     output_small_stitched_file = Path(
#     #         "g:/CMU_DED_Ti64_21T/stitch_tiles_from_positions_supersmall.tif"
#     #     )
#     #     small_img = skimage.transform.rescale(
#     #         stitched_result.data.squeeze(), 0.01, anti_aliasing=False
#     #     )
#     #     imageio.v3.imwrite(output_small_stitched_file, small_img)

#     #     output_stitched_file = Path("g:/CMU_DED_Ti64_21T/stitch_tiles_from_positions.tif")

#     #     with tifffile.TiffWriter(output_stitched_file, bigtiff=True) as tiff:
#     #         tiff.write(stitched_result.data.squeeze())

#     # imageio.v3.imwrite(output_stitched_file, stitched_result.data)
#     # imageio.v3.imwrite(
#     #     output_stitched_file,
#     #     stitched_result.data.squeeze(),
#     #     bigtiff=True,
#     # )

#     # output_stitched_file_pil = Path(
#     #     "g:/CMU_DED_Ti64_21T/stitch_tiles_from_positions_pillow.tif"
#     # )

#     # pil_img = Image.fromarray(stitched_result.data.squeeze())
#     # pil_img.save(output_stitched_file_pil)

#     assert True
#     # Compute the result and display the mosaic
#     # mosaic_image_computed = stitched_result.compute()
#     # plt.imshow(mosaic_image_computed)
#     # plt.axis("off")  # Hide axes
#     # plt.show()
