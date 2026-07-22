#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stitcher.py

A unified CLI for stitching ZEISS CZI mosaics and tile-region arrays.

Features:
  * Save “simple” downscaled mosaics (color or grayscale).
  * Stitch per-channel mosaics in parallel using a reference channel.
  * Optional flatfield normalization for mosaics.
  * Handle non-mosaic CZI files as tile-region arrays.
  * Commands for checking tile arrays and processing directories.
"""

import time
from pathlib import Path
from typing import List

import typer
import numpy as np
import skimage.io
import imageio
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn
from joblib import Parallel, delayed
from aicspylibczi import CziFile

# Local imports
import precon3d.mosaic_utils as mosaic_utils
import precon3d.utility as ut
import precon3d.factory as factory
from precon3d._my_typer_cli import CustomCLIGroup, CustomCLICommand
from precon3d.custom_types import (
    StitchingParams,
    GeneralAttrs,
    StitchingAttrs,
    FijiAttrs,
    NormalizationAttrs,
    TileStack,
    Mosaic,
    TilePosition,
)

app = typer.Typer(
    cls=CustomCLIGroup,
    no_args_is_help=True,
    help="Stitch tiles and mosaics from ZEISS CZI files.",
)


def save_simple_color_mosaic(
    czi_file: Path,
    general_attrs: GeneralAttrs,
    stitching_attrs: StitchingAttrs,
    fiji_attrs: FijiAttrs,
) -> None:
    """
    Save downscaled, per-channel color mosaics for each scene.

    Args:
        czi_file (Path): Path to the CZI file.
        general_attrs (GeneralAttrs): General pipeline settings.
        stitching_attrs (StitchingAttrs): Stitching parameters.
        fiji_attrs (FijiAttrs): Fiji application settings.
    """
    tilestack, fname = mosaic_utils.ai_tilestack(czi_file)
    bin_size = stitching_attrs.simple_mosaic_downscale_bin_size

    for scene_idx in tilestack.scenes:
        stacks = mosaic_utils.simple_tilestack(tilestack, scene_idx, fname)
        for ts in stacks:
            # stitch the tiles using Fiji (per-channel)
            stitched = mosaic_utils.stitch_tilestack(
                ts, fiji_attrs.fiji_app, cleanup=True, tileconfig=ts.positions
            )
            # prepare output directory and filename
            out_dir = (
                general_attrs.output_directory
                / "Simple_Mosaic"
                / stitched.scene
                / stitched.channel
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            fname_out = (
                f"{stitched.czi_fname}_{stitched.scene}_"
                f"{stitched.channel}_binsz{bin_size:02d}_mosaic.tif"
            )
            # downscale
            img = stitched.data.squeeze()
            new_shape = (
                img.shape[0] // bin_size,
                img.shape[1] // bin_size,
                img.shape[2],
            )
            img_resized = skimage.transform.resize(
                img, new_shape, anti_aliasing=True
            )
            # save 8-bit TIFF
            skimage.io.imsave(
                out_dir / fname_out,
                ut.convert_gray_to_8bit(img_resized.squeeze()),
                check_contrast=False,
            )
    typer.echo(f'✓ Simple color mosaics from "{fname}" saved.')


def save_simple_mosaic(
    czi_file: Path,
    general_attrs: GeneralAttrs,
    stitching_attrs: StitchingAttrs,
    fiji_attrs: FijiAttrs,
) -> None:
    """
    Save downscaled mosaics (color or grayscale) generated
    directly from acquisition stage positions.

    Args:
        czi_file (Path): Path to the CZI file.
        general_attrs (GeneralAttrs): General pipeline settings.
        stitching_attrs (StitchingAttrs): Stitching parameters.
        fiji_attrs (FijiAttrs): Fiji application settings.
    """
    ai_obj, _ = mosaic_utils.ai_tilestack(czi_file)
    dims = dict(ai_obj.dims)
    if dims.get("S", False):
        save_simple_color_mosaic(
            czi_file, general_attrs, stitching_attrs, fiji_attrs
        )
        return

    # grayscale path
    czi_mos, fname = mosaic_utils.ai_mosaic(czi_file)
    bin_size = stitching_attrs.simple_mosaic_downscale_bin_size

    for scene_idx in czi_mos.scenes:
        czi_mos.set_scene(scene_idx)
        channels = dims.get("C", 1)
        channel_mosaics = []
        for ch in range(channels):
            arr = czi_mos.get_image_data("MYXS", C=ch).squeeze()
            positions = czi_mos.reader.get_mosaic_tile_positions()
            tiles = [TilePosition(y=y, x=x) for y, x in positions]
            channel_mosaics.append(
                TileStack(
                    data=arr,
                    positions=tiles,
                    czi_fname=fname,
                    channel=czi_mos.channel_names[ch],
                    scene=czi_mos.current_scene,
                    scene_idx=czi_mos.current_scene_index,
                )
            )
        # downscale & save each channel
        for ts in channel_mosaics:
            out_dir = (
                general_attrs.output_directory
                / "Simple_Mosaic"
                / ts.scene
                / ts.channel
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            fname_out = (
                f"{ts.czi_fname}_{ts.scene}_{ts.channel}"
                f"_binsz{bin_size:02d}_mosaic.tif"
            )
            img = ts.data.squeeze()
            pil = Image.fromarray(img)
            pil = pil.resize(
                (img.shape[1] // bin_size, img.shape[0] // bin_size),
                Image.Resampling.LANCZOS,
            )
            skimage.io.imsave(
                out_dir / fname_out,
                ut.convert_gray_to_8bit(np.array(pil).squeeze()),
                check_contrast=False,
            )
    typer.echo(f'✓ Simple grayscale mosaics from "{fname}" saved.')


def normalize_simple_tilestacks(
    stacks: List[TileStack],
    norm_attrs: NormalizationAttrs,
) -> List[TileStack]:
    """
    Normalize each TileStack by its channel-specific flatfield.

    Args:
        stacks (List[TileStack]): Tile stacks to normalize.
        norm_attrs (NormalizationAttrs): Flatfield settings.

    Returns:
        List[TileStack]: New TileStacks with normalized data.

    Raises:
        ValueError: If a required flatfield file is missing.
    """
    parent = Path(norm_attrs.channel_flatfields_parent_directory)
    normalized = []

    for ts in stacks:
        ff_name = norm_attrs.channel_flatfields_filename.get(ts.channel)
        if ff_name is None:
            raise ValueError(f"No flatfield entry for channel '{ts.channel}'")
        ff_path = parent / ff_name
        if not ff_path.exists():
            raise ValueError(f"Flatfield file '{ff_path}' not found")
        ff_img = imageio.v3.imread(ff_path).squeeze()
        if ff_img.ndim == 2:
            ff_img = ff_img[..., None]

        data = ts.data
        norm_tiles = np.zeros_like(data, dtype=np.uint16)
        for i, tile in enumerate(data):
            norm_tiles[i] = mosaic_utils.normalize_images(tile, ff_img)
        normalized.append(
            TileStack(
                data=norm_tiles,
                positions=ts.positions,
                channel=ts.channel,
                czi_fname=ts.czi_fname,
                scene=ts.scene,
                scene_idx=ts.scene_idx,
            )
        )
    return normalized


def stitch_simple_tilestacks_with_reference_channel(
    stacks: List[TileStack],
    fiji_attrs: FijiAttrs,
    ref_channel: str,
) -> List[Mosaic]:
    """
    Stitch a list of TileStacks, using one channel as reference.

    The reference channel is stitched first, its tile positions
    are then applied to other channels to guarantee alignment.

    Args:
        stacks (List[TileStack]): Per-channel TileStacks.
        fiji_attrs (FijiAttrs): Fiji application settings.
        ref_channel (str): Channel name to use as reference.

    Returns:
        List[Mosaic]: Stitched mosaics for all channels.
    """
    ref_ts = next((ts for ts in stacks if ts.channel == ref_channel), None)
    if ref_ts is None:
        raise ValueError(f"Reference channel '{ref_channel}' not found")

    ref_mos = mosaic_utils.stitch_tilestack(
        ref_ts, fiji_attrs.fiji_app, cleanup=True
    )
    out = [ref_mos]
    others = [ts for ts in stacks if ts.channel != ref_channel]

    for ts in others:
        m = mosaic_utils.stitch_tilestack(
            ts,
            fiji_attrs.fiji_app,
            cleanup=True,
            tileconfig=ref_mos.positions,
        )
        out.append(m)
    return out


def stitch_and_save_mosaic(czi_file: Path, params: StitchingParams) -> None:
    """
    Full pipeline: per-scene tilestack → normalize → stitch → save.

    Args:
        czi_file (Path): Path to a mosaic CZI file.
        params (StitchingParams): All pipeline parameters.
    """
    tilestack, fname = mosaic_utils.ai_tilestack(czi_file)

    for scene_idx in tilestack.scenes:
        stacks = mosaic_utils.simple_tilestack(tilestack, scene_idx, fname)

        if params.normalization_attrs.use_flatfield:
            stacks = normalize_simple_tilestacks(
                stacks, params.normalization_attrs
            )

        mosaics = stitch_simple_tilestacks_with_reference_channel(
            stacks,
            params.fiji_attrs,
            params.stitching_attrs.stitch_reference_channel,
        )
        # save
        for m in mosaics:
            # color check
            out_dir = (
                params.general_attrs.output_directory
                / "Stitched"
                / m.scene
                / m.channel
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            fname_out = f"{m.czi_fname}_{m.scene}_{m.channel}_mosaic.tif"
            img = m.data
            if img.shape[-1] == 3:
                # RGB: convert BGR→RGB and gamma‐correct
                rgb = img[:, :, ::-1]
                rgb = skimage.exposure.adjust_gamma(rgb, 0.5)
                skimage.io.imsave(
                    out_dir / fname_out, rgb, check_contrast=False
                )
            else:
                skimage.io.imsave(
                    out_dir / fname_out,
                    ut.convert_gray_to_8bit(img.squeeze()),
                    check_contrast=False,
                )


def stitch_save_tilearray_by_idx(
    czi_file: Path,
    params: StitchingParams,
    grouping_idx: int,
) -> None:
    """
    Stitch and save a tile‐region array CZI by grouping index.

    Args:
        czi_file (Path): Path to a tile‐region CZI file.
        params (StitchingParams): Pipeline parameters.
        grouping_idx (int): Tile‐array group index.
    """
    stacks = mosaic_utils.retrieve_tile_region_array_data(czi_file)
    grouped = mosaic_utils.group_tile_arrays(stacks, idx=grouping_idx)
    for ts in grouped:
        mos = mosaic_utils.stitch_tiles_from_positions(ts)
        mosaic_utils.save_stitched_mosaic(
            mos, params.general_attrs.output_directory
        )


def start_stitcher_parallel(
    params: StitchingParams,
    ncores: int = 4,
    grouping_idx: int = 0,
) -> None:
    """
    Process all CZI files in input directory in parallel.

    Depending on `stitch_tiles` and whether the file is a mosaic,
    either runs full‐mosaic stitching or tile‐region array stitching.

    Args:
        params (StitchingParams): Pipeline parameters.
        ncores (int): Number of parallel jobs.
        grouping_idx (int): Group index for tile‐region arrays.
    """
    ext = params.general_attrs.file_extension
    if ext != ".czi":
        raise ValueError("Only .czi is supported")

    inp = params.general_attrs.input_directory
    all_files = ut.sorted_files(inp, ext)
    out_stitched = params.general_attrs.output_directory / "Stitched"

    if out_stitched.exists():
        done = ut.sorted_files(params.general_attrs.output_directory, ".tif")
        todo = ut.remove_matching_filenames(all_files, done)
    else:
        todo = all_files

    if params.stitching_attrs.stitch_tiles:
        first = todo[0]
        if CziFile(first).is_mosaic():
            # full mosaic path
            Parallel(n_jobs=ncores)(
                delayed(stitch_and_save_mosaic)(f, params) for f in todo
            )
        else:
            # tile‐region arrays
            Parallel(n_jobs=ncores)(
                delayed(stitch_save_tilearray_by_idx)(f, params, grouping_idx)
                for f in todo
            )


@app.command(
    cls=CustomCLICommand,
    name="process_images",
    help="Stitch all CZI in parallel per config.",
)
def process_images(
    configfile: Path = typer.Argument(..., exists=True, help="YAML config."),
    ncores: int = typer.Option(4, help="Number of parallel jobs."),
    grouping_idx: int = typer.Option(
        0, help="Tile-array group index for non-mosaics."
    ),
) -> None:
    """
    Read a YAML config and process all CZI files in the input directory.

    Example:
      python stitcher.py process_images config.yml --ncores 8
    """
    config = ut.read_config(configfile)
    params = factory.create_stitching_parameters(config)
    ut.current_date_and_time()
    typer.secho(f"*** Starting processing: {configfile.name} ***", fg="green")
    t0 = time.perf_counter()
    start_stitcher_parallel(params, ncores=ncores, grouping_idx=grouping_idx)
    dt = time.perf_counter() - t0
    typer.secho(f"Total time: {dt:.2f}s ({dt/60:.2f}min)", fg="green")


@app.command(
    cls=CustomCLICommand,
    name="stitch_czi",
    help="Stitch a single CZI with progress spinner.",
)
def stitch_czi(
    czifile: Path = typer.Argument(..., exists=True, help="CZI to stitch."),
    configfile: Path = typer.Argument(..., exists=True, help="YAML config."),
) -> None:
    """
    Stitch one CZI (must be a mosaic) using the provided config.

    Example:
      python stitcher.py stitch_czi sample.czi config.yml
    """
    czi = czifile
    cfg = ut.read_config(configfile)
    params = factory.create_stitching_parameters(cfg)
    ut.current_date_and_time()
    typer.secho(f"*** Stitching {czi.name} ***", fg="cyan")
    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as prog:
        task = prog.add_task("Stitching...", total=None)
        stitch_and_save_mosaic(czi, params)
        prog.update(task, completed=True)
    dt = time.perf_counter() - t0
    typer.secho(f"Done in {dt:.2f}s ({dt/60:.2f}min)", fg="cyan")


@app.command(
    cls=CustomCLICommand,
    name="check_tilearray",
    help="Visualize tile-region arrays in a CZI.",
)
def check_tilearray(
    czi_path: Path = typer.Argument(..., exists=True, help="CZI to inspect."),
) -> None:
    """
    Load tile-region array data and display a quick overview.

    Example:
      python stitcher.py check_tilearray sample.czi
    """
    stacks = mosaic_utils.tile_region_array_data(czi_path)
    flat = mosaic_utils.flatten_list(stacks)
    mosaic_utils.visualize_tilestacks(flat)


if __name__ == "__main__":
    app()
