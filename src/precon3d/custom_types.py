#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
precon3d_types.py

Type definitions and data structures for the precon3d image
processing and stitching pipeline. This module defines NamedTuples
for configuration settings, image metadata, tile and mosaic data,
and related utilities.
"""

from __future__ import annotations

import pytz
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, NamedTuple, Union

import numpy.typing as npt


class ValueExpectedTrueError(Exception):
    """Exception raised when a value expected to be True is False."""  # noqa: D101

    def __init__(
        self,
        key: str,
        message: str = "Expected value to be True but got False.",
    ) -> None:
        """
        Initialize the exception.

        Args:
            key: The configuration key or parameter name.
            message: Explanation of the error.
        """
        self.key = key
        super().__init__(message)


class UserSettings(NamedTuple):
    """
    User and machine dependent settings.

    Attributes:
        fiji_app: Path to the Fiji executable.
        home: User home directory.
        scratch: Scratch directory for temporary files.
    """

    fiji_app: Path
    home: Path
    scratch: Path


class GeneralAttrs(NamedTuple):
    """
    General attributes for data processing.

    Attributes:
        input_directory: Directory containing input files.
        file_extension: File extension to filter input (e.g. '.czi').
        output_directory: Directory for writing output data.
    """

    input_directory: Path
    file_extension: str
    output_directory: Path


class ManualShadingAttrs(NamedTuple):
    """
    Manual shading correction parameters.

    Attributes:
        extract_tiles: If True, extract individual tiles.
        reorganize_tiles_by_channels: If True, group tiles by channel.
        channel_keywords: List of substrings to identify channels.
        downselection_reference_channel: Channel used for SSIM downselection.
        downselection_reference_image: Path to a reference image.
        downselection_ssim_threshold: SSIM threshold for downselection.
        downselection_nxn_subimages: Number of subimages per axis for SSIM.
    """

    extract_tiles: bool
    reorganize_tiles_by_channels: bool
    channel_keywords: list[str]
    downselection_reference_channel: str
    downselection_reference_image: Path
    downselection_ssim_threshold: float
    downselection_nxn_subimages: int


class ShadingConfig(NamedTuple):
    """
    Shading correction configuration.

    Attributes:
        general_attrs: General attributes for shading.
        manual_attrs: Manual shading correction parameters.
    """

    general_attrs: GeneralAttrs
    manual_attrs: ManualShadingAttrs


class FijiAttrs(NamedTuple):
    """
    Fiji application settings for stitching.

    Attributes:
        fiji_app: Path to the Fiji executable.
    """

    fiji_app: Path


class NormalizationAttrs(NamedTuple):
    """
    Flatfield normalization parameters.

    Attributes:
        use_flatfield: If True, apply flatfield normalization.
        channel_flatfields_parent_directory: Directory containing flatfield images.
        channel_flatfields_filename: Mapping from channel name to flatfield filename.
    """

    use_flatfield: bool
    channel_flatfields_parent_directory: Path | None
    channel_flatfields_filename: Dict[str, str] | None


class StitchingAttrs(NamedTuple):
    """
    Parameters controlling stitching operations.

    Attributes:
        simple_mosaic: If True, generate a quick downsampled mosaic.
        simple_mosaic_downscale_bin_size: Integer bin size for quick mosaics.
        stitch_tiles: If True, perform full tile-based stitching.
        stitch_reference_channel: Channel name to use as reference.
    """

    simple_mosaic: bool
    simple_mosaic_downscale_bin_size: int
    stitch_tiles: bool
    stitch_reference_channel: str


class StitchingParams(NamedTuple):
    """
    All parameters required for a stitching run.

    Attributes:
        fiji_attrs: Fiji application settings.
        general_attrs: General processing attributes.
        normalization_attrs: Flatfield normalization parameters.
        stitching_attrs: Stitching-specific parameters.
    """

    fiji_attrs: FijiAttrs
    general_attrs: GeneralAttrs
    normalization_attrs: NormalizationAttrs
    stitching_attrs: StitchingAttrs


class TilePosition(NamedTuple):
    """
    2D position of a tile in pixels.

    Attributes:
        x: Horizontal offset.
        y: Vertical offset.
    """

    x: int
    y: int


class Mosaic(NamedTuple):
    """
    A stitched mosaic image and its metadata.

    Attributes:
        data: Image array (HxW or HxWxC).
        positions: Tile positions used in the mosaic.
        channel: Channel name.
        czi_fname: Base filename (without extension).
        scene: Scene identifier.
        scene_idx: Scene index.
    """

    data: npt.NDArray[Any]
    positions: list[TilePosition]
    channel: str
    czi_fname: str
    scene: str
    scene_idx: int


class TileStack(NamedTuple):
    """
    A stack of image tiles for a single channel and scene.

    Attributes:
        data: Stack array (NxHxW or NxHxWxC).
        positions: List of tile positions.
        channel: Channel name.
        czi_fname: Base filename (without extension).
        scene: Scene identifier.
        scene_idx: Scene index.
    """

    data: npt.NDArray[Any]
    positions: list[TilePosition]
    channel: str
    czi_fname: str
    scene: str
    scene_idx: int


class SupportPoint(NamedTuple):
    """
    A 3D support/control point.

    Attributes:
        x: X coordinate in pixels.
        y: Y coordinate in pixels.
        z: Z coordinate (height).
    """

    x: float
    y: float
    z: float


class SupportPoints(NamedTuple):
    """
    Collection of support points for a scene.

    Attributes:
        czi_fname: CZI base filename.
        scene_name: Scene identifier.
        z_height: Global Z coordinate for the scene.
        positions: List of 3D support points.
    """

    czi_fname: str
    scene_name: str
    z_height: float
    positions: list[SupportPoint]


class SceneSupportPoints(NamedTuple):
    """
    Support points organized by scene.

    Attributes:
        czi_fname: CZI base filename.
        scene: Scene identifier.
        z: Global Z coordinate for the scene.
        z_heights: Array of Z positions.
        xy_positions: XY tile positions.
    """

    czi_fname: str
    scene: str
    z: float
    z_heights: npt.NDArray[Any]
    xy_positions: list[TilePosition]


class BoundingBox(NamedTuple):
    """
    2D bounding box specification.

    Attributes:
        h: Height in pixels.
        w: Width in pixels.
        x: X offset of the top-left corner.
        y: Y offset of the top-left corner.
    """

    h: int
    w: int
    x: int
    y: int

    def __str__(self) -> str:
        """String representation of the bounding box."""
        return f"BoundingBox(h={self.h}, w={self.w}, x={self.x}, y={self.y})"


class ChannelMetadata(NamedTuple):
    """
    Metadata for a single imaging channel.

    Attributes:
        channel_name: Human-readable channel name.
        mode: Acquisition mode (e.g., 'Brightfield', 'Fluorescence').
        exposure_time: Exposure time in milliseconds.
        reflector: Reflector configuration.
        contrast: Contrast method.
        pixel_type: Pixel data type.
        pixel_bit_count: Bits per pixel.
        illumination_wavelength: Wavelength in nanometers.
        illumination_wavelength_range: Range string (e.g., '450-490').
        illumination_intensity: Intensity description.
        detector_binning: Binning setting.
    """

    channel_name: str | None = None
    mode: str | None = None
    exposure_time: int | None = None
    reflector: str | None = None
    contrast: str | None = None
    pixel_type: str | None = None
    pixel_bit_count: int | None = None
    illumination_wavelength: int | None = None
    illumination_wavelength_range: str | None = None
    illumination_intensity: str | None = None
    detector_binning: str | None = None

    def __str__(self) -> str:
        """Formatted channel metadata."""
        return (
            f"Channel '{self.channel_name}': mode={self.mode}, reflector={self.reflector}, "
            f"contrast={self.contrast}, exposure={self.exposure_time} ms, "
            f"illum={self.illumination_wavelength} nm ({self.illumination_wavelength_range}), "
            f"intensity={self.illumination_intensity}, binning={self.detector_binning}, "
            f"pixel={self.pixel_type} ({self.pixel_bit_count} bits)"
        )


class SceneMetadata(NamedTuple):
    """
    Metadata for a single scene in the mosaic.

    Attributes:
        scene_name: Human-readable scene name.
        scene_index: Numeric index of the scene.
        bounding_box: Scene bounding box.
        n_columns: Number of tile columns.
        n_rows: Number of tile rows.
        contour_size: Contour size descriptor.
        center_position: Center coordinate as string.
    """

    scene_name: str | None = None
    scene_index: int | None = None
    bounding_box: BoundingBox | None = None
    n_columns: int | None = None
    n_rows: int | None = None
    contour_size: str | None = None
    center_position: str | None = None

    def __str__(self) -> str:
        """Formatted scene metadata."""
        return (
            f"Scene {self.scene_index} ('{self.scene_name}'): "
            f"{self.n_columns}×{self.n_rows} tiles, {self.bounding_box}, "
            f"contour_size={self.contour_size}, center={self.center_position}"
        )


class CZIMetadata(NamedTuple):
    """
    Comprehensive metadata extracted from a CZI file.

    Attributes:
        filename: Base CZI filename.
        author: File author.
        date_created: ISO timestamp of creation.
        dims: Dimension order string (e.g., 'XYZCT').
        dims_shape: Tuple of dimension sizes.
        scene_names: List of scene identifiers.
        scene_metadata: List of per-scene metadata.
        scene_shapes_are_consistent: True if all scenes share the same shape.
        channel_names: List of channel identifiers.
        channel_metadata: List of per-channel metadata.
        tile_width: Width of each tile in pixels.
        tile_height: Height of each tile in pixels.
        tile_overlap: Overlap between tiles as string or number.
        tile_anchor_mode: Anchor mode description.
        tile_scan_mode: Scan mode description.
        microscope_name: Microscope make/model.
        microscope_type: Microscope type.
        camera_name: Camera model.
        camera_adapter: Adapter description.
        camera_pixel_size: Pixel size string.
        camera_pixel_unit: Pixel size unit.
        objective_name: Objective lens name.
        objective_magnification: Magnification string.
        effective_pixel_size: Effective pixel size.
        color_mode: Color mode description.
    """

    filename: str | None = None
    author: str | None = None
    date_created: str | None = None
    dims: str | None = None
    dims_shape: str | None = None
    scene_names: list[str] | None = None
    scene_metadata: list[SceneMetadata] | None = None
    scene_shapes_are_consistent: bool | None = None
    channel_names: list[str] | None = None
    channel_metadata: list[ChannelMetadata] | None = None
    tile_width: Union[int, str] | None = None
    tile_height: Union[int, str] | None = None
    tile_overlap: Union[int, str] | None = None
    tile_anchor_mode: str | None = None
    tile_scan_mode: str | None = None
    microscope_name: str | None = None
    microscope_type: str | None = None
    camera_name: str | None = None
    camera_adapter: str | None = None
    camera_pixel_size: Union[int, str] | None = None
    camera_pixel_unit: str | None = None
    objective_name: str | None = None
    objective_magnification: Union[int, str] | None = None
    effective_pixel_size: Union[int, str] | None = None
    color_mode: str | None = None

    def __str__(self) -> str:
        """
        Return a human-readable summary of the CZI metadata.

        Converts the ISO timestamp to local mountain time.
        """
        # Parse and convert creation time
        created = ""
        if self.date_created:
            try:
                dt_utc = datetime.strptime(
                    self.date_created[:19] + "Z", "%Y-%m-%dT%H:%M:%S%z"
                )
                local = dt_utc.astimezone(pytz.timezone("America/Denver"))
                created = local.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                created = self.date_created

        # Build scene and channel blocks
        scenes = "\n".join(str(md) for md in (self.scene_metadata or []))
        channels = "\n".join(str(md) for md in (self.channel_metadata or []))

        return (
            f"{self.filename}.czi by {self.author} on {created}\n"
            f"Dimensions: {self.dims} shape={self.dims_shape}, "
            f"consistent_shapes={self.scene_shapes_are_consistent}\n"
            f"Tiles: {self.tile_width}×{self.tile_height}, overlap={self.tile_overlap}, "
            f"anchor={self.tile_anchor_mode}, scan_mode={self.tile_scan_mode}\n"
            f"Scenes ({len(self.scene_names or [])}): {self.scene_names}\n"
            f"Channels ({len(self.channel_names or [])}): {self.channel_names}\n"
            f"\nScene Details:\n{scenes}\n\nChannel Details:\n{channels}"
        )
