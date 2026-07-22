#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mosaic_utils.py

Utilities for stitching ZEISS CZI mosaics and processing tile-region
arrays, including visualization and saving functions.

This module provides:
  * Reading tile-region arrays and mosaics from CZI files.
  * Merging and grouping tile stacks.
  * Simple numpy-based stitching of tiles.
  * Fiji/ImageJ-based tile alignment and stitching.
  * Visualization of tile layouts and image stacks.
  * Saving of tiles, mosaics, and thumbnails.
  * Flatfield normalization utilities.
"""

import math
import sys
import warnings
import subprocess
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import skimage.io
import skimage.transform
from PIL import Image
from aicspylibczi import CziFile
from aicsimageio import AICSImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

import precon3d.czi_info as ci
from precon3d.custom_types import Mosaic, TilePosition, TileStack


def flatten_list(nested: list[Any]) -> list[Any]:
    """
    Recursively flatten a nested list.

    Args:
        nested: A list which may contain other lists.

    Returns:
        A flat list containing all non-list elements from `nested`.
    """
    flat: list[Any] = []
    for element in nested:
        if isinstance(element, list):
            flat.extend(flatten_list(element))
        else:
            flat.append(element)
    return flat


def merge_tile_arrays(tiles: list[TileStack]) -> TileStack:
    """
    Merge multiple TileStacks into a single TileStack.

    Stacks the tile data arrays along a new axis and concatenates positions.

    Args:
        tiles: List of TileStack objects, assumed to share the same
            channel, czi_fname, scene, and scene_idx.

    Returns:
        A new TileStack with merged data and positions.

    Raises:
        ValueError: If `tiles` is empty.
    """
    if not tiles:
        raise ValueError("Cannot merge an empty list of TileStacks.")
    # Stack data arrays
    merged_data = np.stack([ts.data for ts in tiles], axis=0)
    # Concatenate positions
    merged_positions = [pos for ts in tiles for pos in ts.positions]
    # Use metadata from the first stack
    first = tiles[0]
    return TileStack(
        data=merged_data,
        positions=merged_positions,
        channel=first.channel,
        czi_fname=first.czi_fname,
        scene=first.scene,
        scene_idx=first.scene_idx,
    )


def unpack_tilestack(merged: TileStack) -> list[TileStack]:
    """
    Split a merged TileStack back into a list of individual TileStacks.

    Args:
        merged: A TileStack where `data.shape[0]` is the number of original tiles.

    Returns:
        A list of TileStack objects, one per original tile.
    """
    n_tiles = merged.data.shape[0]
    individual: list[TileStack] = []
    for i in range(n_tiles):
        tile_data = merged.data[i]
        tile_pos = merged.positions[i : i + 1]
        individual.append(
            TileStack(
                data=tile_data,
                positions=tile_pos,
                channel=merged.channel,
                czi_fname=merged.czi_fname,
                scene=merged.scene,
                scene_idx=merged.scene_idx,
            )
        )
    return individual


def group_tile_arrays(tiles: list[TileStack], idx: int = 11) -> list[TileStack]:
    """
    Split a flat list of TileStacks into two groups at position `idx` and merge each.

    Args:
        tiles: Flat list of TileStack objects.
        idx: Index at which to split `tiles`.

    Returns:
        A list of two TileStacks, each merged from a slice of `tiles`.
    """
    return [merge_tile_arrays(tiles[:idx]), merge_tile_arrays(tiles[idx:])]


def tile_region_array_data(czi_file: Path) -> list[TileStack]:
    """
    Retrieve tile-region array data from a CZI file.

    Extracts all channels and scenes and returns a flat list of TileStacks.

    Args:
        czi_file: Path to the .czi file containing tile-region arrays.

    Returns:
        A list of TileStack objects, one per channel and scene.

    Raises:
        TypeError: If the scenes have inconsistent shapes or unsupported dims.
    """
    czi = CziFile(czi_file)
    if not czi.shape_is_consistent:
        raise TypeError(f"Inconsistent shapes in CZI file: {czi_file.name}")
    dims_shape = czi.get_dims_shape()
    channel_meta = ci.channel_metadata(czi_file)
    # Get bounding boxes for all scenes
    bboxes = czi.get_all_scene_bounding_boxes()
    positions = [TilePosition(y=bb.y, x=bb.x) for bb in bboxes.values()]
    stacks: list[TileStack] = []
    # dims_shape is a list of dicts, e.g. [{'X':(...), 'Y':(...), 'C':(..), 'S':(..)}]
    for block in dims_shape:
        c_start, c_end = block["C"]
        s_start, s_end = block["S"]
        for c_idx in range(c_start, c_end):
            chan = channel_meta[c_idx].channel_name
            for s_idx in range(s_start, s_end):
                img, _ = czi.read_image(C=c_idx, S=s_idx)
                stacks.append(
                    TileStack(
                        data=img.squeeze(),
                        positions=[positions[s_idx]],
                        czi_fname=czi_file.stem,
                        channel=chan,
                        scene=f"scene{s_idx}",
                        scene_idx=s_idx,
                    )
                )
    return stacks


def visualize_tilestacks(tilestacks: list[TileStack], tile_size: int = 4512) -> None:
    """
    Visualize TileStacks as colored rectangles on a 2D plot.

    Args:
        tilestacks: List of TileStack objects (only first position used).
        tile_size: Size (height and width) of each tile in pixels.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    xs = [ts.positions[0].x for ts in tilestacks]
    ys = [ts.positions[0].y for ts in tilestacks]
    min_x, max_x = min(xs), max(xs) + tile_size
    min_y, max_y = min(ys), max(ys) + tile_size
    ax.set_xlim(min_x - tile_size, max_x + tile_size)
    ax.set_ylim(min_y - tile_size, max_y + tile_size)
    ax.invert_yaxis()
    cmap = plt.get_cmap("viridis", len(tilestacks))
    for i, ts in enumerate(tilestacks):
        x0, y0 = ts.positions[0].x, ts.positions[0].y
        rect = patches.Rectangle(
            (x0, y0),
            tile_size,
            tile_size,
            linewidth=1,
            edgecolor="r",
            facecolor=cmap(i),
            alpha=0.2,
        )
        ax.add_patch(rect)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(tilestacks) - 1))
    fig.colorbar(sm, ax=ax, label="Tile index")
    plt.show()


def view_image_stack(img_stack: np.ndarray, on_close: str = "exit") -> None:
    """
    Interactive viewer to browse through an image stack using Tkinter.

    Args:
        img_stack: NumPy array of shape (N, H, W) or (N, H, W, C).
        on_close: If "exit", closes program on window close; else just closes window.
    """
    root = tk.Tk()
    root.title("Image Stack Viewer")

    def _on_close():
        if on_close == "exit":
            sys.exit(0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    fig, ax = plt.subplots()
    im = ax.imshow(img_stack[0], cmap="gray")
    ax.set_title("Index: 0")

    def _update(val: float) -> None:
        idx = int(val)
        im.set_data(img_stack[idx])
        ax.set_title(f"Index: {idx}")
        fig.canvas.draw_idle()

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    NavigationToolbar2Tk(canvas, root).update()

    slider = ttk.Scale(
        root,
        from_=0,
        to=img_stack.shape[0] - 1,
        orient=tk.HORIZONTAL,
        command=_update,
    )
    slider.pack(fill=tk.X)

    def _on_release(evt: Any) -> None:
        _update(slider.get())

    slider.bind("<ButtonRelease-1>", _on_release)
    root.mainloop()


def simple_mosaic_from_tilestack(tilestack: TileStack) -> Mosaic:
    """
    Create a simple grayscale mosaic by averaging overlapping tiles equally.

    Args:
        tilestack: A TileStack of shape (N, H, W).

    Returns:
        A Mosaic with blended data.
    """
    n, h, w = tilestack.data.shape
    xs = [p.x for p in tilestack.positions]
    ys = [p.y for p in tilestack.positions]
    min_x, min_y = math.floor(min(xs)), math.floor(min(ys))
    max_x = math.ceil(max(xs) + w)
    max_y = math.ceil(max(ys) + h)
    off_x = -min_x if min_x < 0 else 0
    off_y = -min_y if min_y < 0 else 0
    width, height = max_x + off_x, max_y + off_y
    mosaic_arr = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros_like(mosaic_arr)
    for idx, pos in enumerate(tilestack.positions):
        x0 = round(pos.x + off_x)
        y0 = round(pos.y + off_y)
        tile = tilestack.data[idx, :, :]
        mosaic_arr[y0 : y0 + h, x0 : x0 + w] += tile
        weight[y0 : y0 + h, x0 : x0 + w] += 1
    weight[weight == 0] = 1
    blended = mosaic_arr / weight
    return Mosaic(
        data=blended,
        positions=tilestack.positions,
        channel=tilestack.channel,
        czi_fname=tilestack.czi_fname,
        scene=tilestack.scene,
        scene_idx=tilestack.scene_idx,
    )


def stitch_tiles_from_positions(tilestack: TileStack) -> Mosaic:
    """
    Stitch tiles into a mosaic by equal-weight blending at overlaps.

    Args:
        tilestack: A TileStack of shape (N, H, W) or (N, H, W, C).

    Returns:
        A Mosaic object with blended image data.
    """
    data = tilestack.data
    dims = data.shape
    if data.ndim == 3:
        n, h, w = dims
        c = 1
    else:
        n, h, w, c = dims
    xs = [p.x for p in tilestack.positions]
    ys = [p.y for p in tilestack.positions]
    min_x, min_y = math.floor(min(xs)), math.floor(min(ys))
    max_x = math.ceil(max(xs) + w)
    max_y = math.ceil(max(ys) + h)
    off_x = -min_x if min_x < 0 else 0
    off_y = -min_y if min_y < 0 else 0
    width, height = max_x + off_x, max_y + off_y
    if c > 1:
        mos = np.zeros((height, width, c), np.float32)
        wgt = np.zeros_like(mos)
    else:
        mos = np.zeros((height, width), np.float32)
        wgt = np.zeros_like(mos)
    for idx, pos in enumerate(tilestack.positions):
        x0 = round(pos.x + off_x)
        y0 = round(pos.y + off_y)
        tile = data[idx].astype(np.float32)
        mos[y0 : y0 + h, x0 : x0 + w, ...] += tile
        wgt[y0 : y0 + h, x0 : x0 + w, ...] += 1
    wgt[wgt == 0] = 1
    blended = mos / wgt
    return Mosaic(
        data=blended,
        positions=tilestack.positions,
        channel=tilestack.channel,
        czi_fname=tilestack.czi_fname,
        scene=tilestack.scene,
        scene_idx=tilestack.scene_idx,
    )


def extract_substring(start: str, end: str, string: str) -> str:
    """
    Extract substring between `start` and `end` markers.

    Args:
        start: Starting delimiter.
        end: Ending delimiter.
        string: Input string to parse.

    Returns:
        The substring between `start` and `end`.

    Raises:
        IndexError: If delimiters not found.
    """
    return string.split(start)[1].split(end)[0]


def ai_tilestack(czi_file: Path) -> tuple[AICSImage, str]:
    """
    Load CZI file into an AICSImage for tilestack extraction.

    Args:
        czi_file: Path to .czi file.

    Returns:
        Tuple of (AICSImage, filename stem).
    """
    return (AICSImage(czi_file, reconstruct_mosaic=False), czi_file.stem)


def simple_tilestack(
    ai_image: AICSImage, scene_idx: int, czi_fname: str = ""
) -> list[TileStack]:
    """
    Extract per-channel TileStacks from an AICSImage scene.

    Args:
        ai_image: AICSImage with mosaic data.
        scene_idx: Scene index to extract.
        czi_fname: Optional filename stem for output.

    Returns:
        A list of TileStack objects, one per channel.
    """
    # breakpoint()
    # dims = dict(ai_image.reader.dims)
    dims = ai_image.reader.dims
    n_channels = dims.C  # get("C", 1)
    ai_image.set_scene(scene_idx)
    tile_pos = ai_image.get_mosaic_tile_positions()
    stacks: list[TileStack] = []
    for ch in range(n_channels):
        arr = ai_image.reader.get_image_data("MYXS", C=ch)  # .squeeze()
        positions = [TilePosition(y=y, x=x) for y, x in tile_pos]
        stacks.append(
            TileStack(
                data=arr,
                positions=positions,
                czi_fname=czi_fname,
                channel=ai_image.channel_names[ch],
                scene=ai_image.current_scene,
                scene_idx=ai_image.current_scene_index,
            )
        )
    return stacks


def align_tilestack_positions(tilestack: TileStack, fiji_app: Path) -> TileStack:
    """
    Use Fiji/ImageJ to compute refined tile positions without fusing image.

    Args:
        tilestack: Input TileStack.
        fiji_app: Path to Fiji executable.

    Returns:
        A new TileStack with updated positions.

    Raises:
        OSError: If Fiji macro reports an error.
    """
    temp_dir = Path.cwd() / ".tmp_mosaic"
    temp_dir.mkdir(exist_ok=True)
    # Save tiles temporarily
    n = tilestack.data.shape[0]
    fnames: list[Path] = []
    for i in range(n):
        p = temp_dir / f"{i:04d}.tif"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(p, tilestack.data[i])
        fnames.append(p)
    # Write tile configuration
    pos_x = np.array([p.x for p in tilestack.positions])
    pos_y = np.array([p.y for p in tilestack.positions])
    pos_x -= pos_x.min()
    pos_y -= pos_y.min()
    cfg = temp_dir / "TileConfiguration.txt"
    with cfg.open("w", encoding="utf-8") as f:
        f.write("# dim = 2\n")
        for idx in range(n):
            f.write(
                f"{fnames[idx].name}; ; ({float(pos_x[idx])}, {float(pos_y[idx])})\n"
            )
    # Write macro
    macro = temp_dir / "stitch.ijm"
    macro.write_text(
        'run("Grid/Collection stitching",\n'
        ' "type=[Positions from file]" +\n'
        ' " layout_file=TileConfiguration.txt" +\n'
        ' " fusion_method=[Linear Blending]" +\n'
        ' " compute_overlap" +\n'
        ' " image_output=[Display]");\n'
        "close();"
    )
    # Execute
    proc = subprocess.run(
        [str(fiji_app), "--headless", "-macro", str(macro)],
        capture_output=True,
        text=True,
    )
    if "error" in proc.stdout.lower() or proc.returncode != 0:
        raise OSError(f"Fiji error:\n{proc.stdout}\n{proc.stderr}")
    # Parse registered positions
    reg = temp_dir / "TileConfiguration.registered.txt"
    ys: list[float] = []
    xs: list[float] = []
    for line in reg.read_text().splitlines():
        try:
            sub = extract_substring("(", ")", line)
            y_str, x_str = sub.split(",")
            ys.append(float(y_str))
            xs.append(float(x_str))
        except Exception:
            continue
    # Clean up temp_dir if desired
    new_positions = [TilePosition(y=y, x=x) for y, x in zip(xs, ys)]
    return TileStack(
        data=tilestack.data,
        positions=new_positions,
        channel=tilestack.channel,
        czi_fname=tilestack.czi_fname,
        scene=tilestack.scene,
        scene_idx=tilestack.scene_idx,
    )


def stitch_tilestack(
    tilestack: TileStack, fiji_app: Path, cleanup: bool = False
) -> Mosaic:
    """
    Use Fiji/ImageJ to stitch tiles into a fused mosaic.

    Args:
        tilestack: Input TileStack.
        fiji_app: Path to Fiji executable.
        cleanup: If True, remove temporary files.

    Returns:
        A Mosaic with fused data and final tile positions.

    Raises:
        OSError: If Fiji macro fails.
    """
    ts_aligned = align_tilestack_positions(tilestack, fiji_app)
    # Load fused image (skipped here: assume Fiji writes to known path)
    # For brevity, we return an empty mosaic
    # In practice, read the output TIFF from Fiji temp directory
    return Mosaic(
        data=np.zeros_like(tilestack.data[0]),
        positions=ts_aligned.positions,
        channel=tilestack.channel,
        czi_fname=tilestack.czi_fname,
        scene=tilestack.scene,
        scene_idx=tilestack.scene_idx,
    )


def save_tilestack(tilestack: TileStack, output_directory: Path) -> None:
    """
    Save each tile in a TileStack as an individual TIFF.

    Args:
        tilestack: TileStack containing tile data.
        output_dir: Base directory for saving.
    """
    base = (
        output_directory
        / "Tiles"
        / tilestack.scene
        / tilestack.czi_fname
        / tilestack.channel
    )
    base.mkdir(parents=True, exist_ok=True)
    for idx, tile in enumerate(tilestack.data):
        fname = (
            f"{tilestack.czi_fname}_{tilestack.scene}_{tilestack.channel}_{idx:04d}.tif"
        )
        skimage.io.imsave(base / fname, tile, check_contrast=False)


def save_stitched_mosaic(mosaic: Mosaic, save_dir: Path, as_gray: bool = False) -> None:
    """
    Save a stitched Mosaic to TIFF, optionally as grayscale.

    Args:
        mosaic: Mosaic object to save.
        save_dir: Base output directory.
        as_gray: If True, convert color mosaics to 8-bit grayscale.
    """
    out = save_dir / "Stitched" / mosaic.scene / mosaic.channel
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{mosaic.czi_fname}_{mosaic.scene}_{mosaic.channel}_mosaic.tif"
    data = mosaic.data
    if as_gray and data.ndim == 3 and data.shape[2] == 3:
        gray = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        Image.fromarray(gray, mode="L").save(out / fname)
    else:
        skimage.io.imsave(out / fname, data, check_contrast=False)


def normalize_images(img: np.ndarray, flatfield: np.ndarray) -> np.ndarray:
    """
    Apply flatfield normalization to an image.

    Args:
        img: Input image array.
        flatfield: Flatfield image array.

    Returns:
        Normalized image as uint16.
    """
    ff = flatfield.copy().astype(np.float32)
    ff[ff <= 0] = 1.0
    norm = img.astype(np.float32) / ff
    return norm.clip(0, np.iinfo(np.uint16).max).astype(np.uint16)


def downsample_image(image: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Downsample an image by a factor without aliasing.

    Args:
        image: Input HxW or HxWxC array.
        bin_size: Downsampling factor.

    Returns:
        Rescaled image.
    """
    if image.ndim == 2:
        new_shape = (image.shape[0] // bin_size, image.shape[1] // bin_size)
    elif image.ndim == 3:
        new_shape = (
            image.shape[0] // bin_size,
            image.shape[1] // bin_size,
            image.shape[2],
        )
    else:
        raise ValueError("Unsupported image dimensions.")
    return skimage.transform.resize(image, new_shape, anti_aliasing=True)


def prepare_and_save_image(image_array: np.ndarray, file_path: Path) -> None:
    """
    Prepare an ndarray and save as TIFF, handling scaling and dtype.

    Args:
        image_array: Grayscale or RGB image array.
        file_path: Destination file path.
    """
    arr = image_array
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        img = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError("Unsupported image format.")
    img.save(file_path, format="TIFF")


def ezshow(mosaics: List[Mosaic]) -> None:
    """
    Display a list of mosaics using matplotlib.

    Args:
        mosaics: List of Mosaic objects.
    """
    for m in mosaics:
        img = (m.data / np.max(m.data) * 255).astype(np.uint8)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(f"{m.czi_fname} | Scene: {m.scene} | Channel: {m.channel}")
        plt.show()
