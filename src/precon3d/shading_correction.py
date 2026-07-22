# shading correction
import os
from sys import platform
import shutil
import skimage
import skimage.io as skio
import imageio
from pathlib import Path
import numpy as np
import time
import random
from dataclasses import dataclass, field
from joblib import Parallel, delayed


from numpy.typing import NDArray
from typing import List, Dict, Any, NamedTuple
from scipy.stats import wasserstein_distance, chisquare
from sklearn.decomposition import NMF
from subprocess import Popen, DEVNULL, PIPE
import subprocess
import typer

# import dask.array as da
# import dask_image.imread

# from PIL import Image
from tqdm import tqdm

# local libraries
import precon3d.utility as ut
import precon3d.mosaic_utils
import precon3d.factory as gen_types

# pylint: disable=wildcard-import
from precon3d.custom_types import *
from precon3d._my_typer_cli import CustomCLIGroup, CustomCLICommand

# CLI
app = typer.Typer(
    cls=CustomCLIGroup,
    no_args_is_help=True,
    short_help="Estimate shading correction from tiles",
)


def subimages_similar_by_ssim(
    ref_image, found_image, n_subimages, ssim_thresh
):
    """Compare subimages are also similar

    divide into equal sized sub images and compare they are all similar
    """
    height, width = ref_image.shape[:2]
    subimg_height = height // n_subimages
    subimg_width = width // n_subimages

    # Pre-calculate the slicing indices to avoid doing it in the loop
    slice_indices = [
        (y, x)
        for y in range(0, height, subimg_height)
        for x in range(0, width, subimg_width)
        if (y + subimg_height <= height and x + subimg_width <= width)
    ]

    ssim_scores = []
    for y, x in slice_indices:
        if found_image.shape[-1] == 3:  # COLOR
            ref_subimg = ref_image[
                y : y + subimg_height, x : x + subimg_width, :
            ].squeeze()
            found_subimg = found_image[
                y : y + subimg_height, x : x + subimg_width, :
            ].squeeze()

            # Calculate SSIM for the current pair of subimages
            score, _ = skimage.metrics.structural_similarity(
                ref_subimg,
                found_subimg,
                data_range=65535,
                full=True,
                channel_axis=-1,
            )
        else:
            ref_subimg = ref_image[
                y : y + subimg_height, x : x + subimg_width
            ].squeeze()
            found_subimg = found_image[
                y : y + subimg_height, x : x + subimg_width
            ].squeeze()

            # Calculate SSIM for the current pair of subimages
            score, _ = skimage.metrics.structural_similarity(
                ref_subimg,
                found_subimg,
                data_range=65535,
                full=True,
            )
        ssim_scores.append(score)

    # Check if all SSIM scores are above the threshold
    return all(score > ssim_thresh for score in ssim_scores)


def downselect_tiles(
    reorganized_tile_dir: Path,
    ref_ch: str,
    ref_img_idx: int = 0,
    ssim_thresh: float = 0.9,
    n_subimages: int = 10,
) -> bool:
    """Downselect tiles organized already by channels"""

    # get channel subdirectories
    channel_subdirs = [
        ch for ch in reorganized_tile_dir.iterdir() if ch.is_dir()
    ]

    # get the file paths
    all_file_paths = {
        subdir.name: ut.sorted_files(subdir, ".tif")
        for subdir in channel_subdirs
    }

    print(f"Found {len(channel_subdirs)} channels")
    # Dictionary to hold the count of file paths for each subdir
    file_path_counts = {
        subdir: len(paths) for subdir, paths in all_file_paths.items()
    }

    # Printing the number of file paths for each subdir
    for subdir, count in file_path_counts.items():
        print(f"\t{subdir}: {count} file(s)")

    # Get an iterator of all list lengths
    lengths = (len(paths) for paths in all_file_paths.values())

    # Get the length of the first list to compare against others
    first_length = next(
        lengths, None
    )  # Returns None if no elements are present

    # Check if all lengths are the same as the first length
    all_same_length = all(length == first_length for length in lengths)
    if all_same_length:
        print("all folders have the same number of files")
    else:
        # Raise an exception with a message indicating the problem
        # raise ValueError('Folders have different numbers of files. Each folder must contain the same number of files.')

        file_counts = {
            subdir: len(paths) for subdir, paths in all_file_paths.items()
        }
        error_message = (
            "Folders have different numbers of files: "
            + ", ".join(f"{k}: {v} files" for k, v in file_counts.items())
        )
        raise ValueError(error_message)

    ref_filepath = all_file_paths.get(ref_ch)[ref_img_idx]
    ref_image = skio.imread(ref_filepath)

    print(
        f"Downselecting tiles in {reorganized_tile_dir} by {ref_ch} channel\nUsing {ref_filepath} as reference"
    )

    for each_ch_subdir in channel_subdirs:
        all_file_paths[each_ch_subdir.name] = ut.sorted_files(
            each_ch_subdir, "tif"
        )  # each_ch_subdir.iterdir()# sort naturally for the files in the directories

    # downselect based on the reference channel
    for idx, each_image in enumerate(all_file_paths.get(ref_ch)):
        found_image = skio.imread(each_image)

        if found_image.shape[-1] == 3:  # COLOR
            ssim_score = skimage.metrics.structural_similarity(
                ref_image.squeeze(),
                found_image.squeeze(),
                data_range=65535,
                full=True,
                channel_axis=-1,
            )
        else:
            ssim_score = skimage.metrics.structural_similarity(
                ref_image.squeeze(),
                found_image.squeeze(),
                data_range=65535,
                full=True,
            )

        # remove if not similar
        if (
            ssim_score[0] <= ssim_thresh
        ):  # not similar based on the ssim for subimages
            print(
                f"DIFFERENT: SSIM score = {ssim_score[0]} ({each_image.stem})"
            )
            # delete specific file from all dirs
            for subdir_name, file_path in all_file_paths.items():
                file_to_delete = file_path[idx]  # use index to specify

                print(f"\t{file_to_delete.stem} deleted from {subdir_name}")

                file_to_delete.unlink()

        else:  # overall, images look similar, then divide into subimages and compare subimages
            print(
                f"SIMILAR, check subimages: SSIM score = {ssim_score[0]} ({each_image.stem})"
            )

            subimages_similar = subimages_similar_by_ssim(
                ref_image, found_image, n_subimages, ssim_thresh
            )

            if subimages_similar is False:
                # not similar based on the ssim for subtiles

                print(
                    f"\tFound differences when tiles divided into {n_subimages*n_subimages} smaller pieces"
                )

                # breakpoint()

                # delete specific file from all dirs
                for subdir_name, file_path in all_file_paths.items():
                    file_to_delete = file_path[idx]  # use index to specify

                    print(
                        f"\t{file_to_delete.stem} deleted from {subdir_name}"
                    )
                    # print(f"\t\t{ssim_list}")

                    file_to_delete.unlink()


@app.command(
    cls=CustomCLICommand,
    name="downselect_tiles_in_directory",
    short_help="Use a SSIM threshold to downselect tiles",
)
def downselect_tiles_in_directory(
    reorganized_tile_dir: Path,
    ref_img_idx: int = 0,
    ssim_thresh: float = 0.9,
    n_subimages: int = 10,
) -> bool:
    """Downselect tiles organized already by channels"""

    # get the file paths

    all_file_paths = ut.sorted_files(
        reorganized_tile_dir, ".tif"
    )  # each_ch_subdir.iterdir()# sort naturally for the files in the directories
    ref_image = skio.imread(all_file_paths[ref_img_idx])

    # downselect based on the reference channel
    for each_image in all_file_paths:
        found_image = skio.imread(each_image)

        if found_image.shape[-1] == 3:  # COLOR
            ssim_score = skimage.metrics.structural_similarity(
                ref_image.squeeze(),
                found_image.squeeze(),
                data_range=65535,
                full=True,
                channel_axis=-1,
            )
        else:
            ssim_score = skimage.metrics.structural_similarity(
                ref_image.squeeze(),
                found_image.squeeze(),
                data_range=65535,
                full=True,
            )

        # remove if not similar
        if (
            ssim_score[0] <= ssim_thresh
        ):  # not similar based on the ssim for subimages
            print(
                f"DIFFERENT: SSIM score = {ssim_score[0]} ({each_image.stem})"
            )
            # delete specific file
            each_image.unlink()

        else:  # overall, images look similar, then divide into subimages and compare subimages
            print(
                f"SIMILAR, check subimages: SSIM score = {ssim_score[0]} ({each_image.stem})"
            )

            subimages_similar = subimages_similar_by_ssim(
                ref_image, found_image, n_subimages, ssim_thresh
            )

            if subimages_similar is False:
                # not similar based on the ssim for subtiles

                print(
                    f"\tFound differences when tiles divided into {n_subimages*n_subimages} smaller pieces"
                )

                # delete specific file
                each_image.unlink()


def downselect_tiles_in_directory_parallel(
    reorganized_tile_dir: Path,
    ref_img_idx: int = 0,
    ssim_thresh: float = 0.9,
    n_subimages: int = 10,
    n_cores: int = 8,
) -> bool:
    """Downselect tiles organized already by channels"""

    # get the file paths

    all_file_paths = ut.sorted_files(
        reorganized_tile_dir, ".tif"
    )  # each_ch_subdir.iterdir()# sort naturally for the files in the directories
    ref_image = skio.imread(all_file_paths[ref_img_idx])

    # downselect based on the reference channel, parallel
    def _compare_image_with_ssim(comparison_image: Path):
        """tocome"""

        found_image = skio.imread(comparison_image)

        if found_image.shape[-1] == 3:  # COLOR
            ssim_score = skimage.metrics.structural_similarity(
                ref_image.squeeze(),
                found_image.squeeze(),
                data_range=65535,
                full=True,
                channel_axis=-1,
            )
        else:
            ssim_score = skimage.metrics.structural_similarity(
                ref_image.squeeze(),
                found_image.squeeze(),
                data_range=65535,
                full=True,
            )

        # remove if not similar
        if (
            ssim_score[0] <= ssim_thresh
        ):  # not similar based on the ssim for subimages
            print(
                f"DIFFERENT: SSIM score = {ssim_score[0]} ({comparison_image.stem})"
            )
            # delete specific file
            comparison_image.unlink()

        else:  # overall, images look similar, then divide into subimages and compare subimages
            print(
                f"SIMILAR, check subimages: SSIM score = {ssim_score[0]} ({each_image.stem})"
            )

            subimages_similar = subimages_similar_by_ssim(
                ref_image, found_image, n_subimages, ssim_thresh
            )

            if subimages_similar is False:
                # not similar based on the ssim for subtiles

                print(
                    f"\tFound differences when tiles divided into {n_subimages*n_subimages} smaller pieces"
                )

                # delete specific file
                comparison_image.unlink()

    Parallel(n_jobs=n_cores)(
        delayed(_compare_image_with_ssim)(each_image)
        for each_image in all_file_paths
    )


def downselect_tiles_using_reference_channel(
    reorganized_tile_dir: Path,
    ref_ch: str,
    ref_image: Path,
    ssim_thresh: float = 0.9,
    n_subimages: int = 10,
) -> bool:
    """Downselect tiles organized already by channels"""

    ref_image = skio.imread(ref_image)

    # get channel subdirectories
    channel_subdirs = [
        ch for ch in reorganized_tile_dir.iterdir() if ch.is_dir()
    ]

    # get the file paths
    all_file_paths = {subdir.name: [] for subdir in channel_subdirs}
    for each_ch_subdir in channel_subdirs:
        all_file_paths[each_ch_subdir.name] = ut.sorted_files(
            each_ch_subdir, "tif"
        )  # each_ch_subdir.iterdir()# sort naturally for the files in the directories

    # check number of files are the same
    n_files = [len(file_paths) for file_paths in all_file_paths.values()]
    if len(set(n_files)) == 1:
        print(
            f"The channel dirs: {list(all_file_paths.keys())}, all contain {list(set(n_files))} files\n"
        )
    else:
        print(
            f"The channel dirs: {list(all_file_paths.keys())}, contain {n_files} files\n"
        )
        return

    # downselect based on the reference channel
    for idx, each_image in enumerate(all_file_paths.get(ref_ch)):
        found_image = skio.imread(each_image)

        if found_image.shape[-1] == 3:  # COLOR
            ssim_score = skimage.metrics.structural_similarity(
                ref_image.squeeze(),
                found_image.squeeze(),
                data_range=65535,
                full=True,
                channel_axis=-1,
            )
        else:
            ssim_score = skimage.metrics.structural_similarity(
                ref_image.squeeze(),
                found_image.squeeze(),
                data_range=65535,
                full=True,
            )

        # remove if not similar
        if (
            ssim_score[0] <= ssim_thresh
        ):  # not similar based on the ssim for subimages
            print(
                f"DIFFERENT: SSIM score = {ssim_score[0]} ({each_image.stem})"
            )
            # delete specific file from all dirs
            for subdir_name, file_path in all_file_paths.items():
                file_to_delete = file_path[idx]  # use index to specify

                print(f"\t{file_to_delete.stem} deleted from {subdir_name}")

                file_to_delete.unlink()

        else:  # overall, images look similar, then divide into subimages and compare subimages
            print(
                f"SIMILAR, check subimages: SSIM score = {ssim_score[0]} ({each_image.stem})"
            )

            if n_subimages == 1:
                continue

            subimages_similar = subimages_similar_by_ssim(
                ref_image, found_image, n_subimages, ssim_thresh
            )

            if subimages_similar is False:
                # not similar based on the ssim for subtiles

                print(
                    f"\tFound differences when tiles divided into {n_subimages*n_subimages} smaller pieces"
                )

                # breakpoint()

                # delete specific file from all dirs
                for subdir_name, file_path in all_file_paths.items():
                    file_to_delete = file_path[idx]  # use index to specify

                    print(
                        f"\t{file_to_delete.stem} deleted from {subdir_name}"
                    )
                    # print(f"\t\t{ssim_list}")

                    file_to_delete.unlink()


# def nmf_on_images(images: da.Array) -> np.ndarray:
#     """
#     Apply NMF to a set of images and return the flatfield correction.

#     Parameters:
#     - images: A Dask array of images.

#     Returns:
#     - A NumPy array representing the flatfield correction.
#     """
#     # Reshape images for NMF: (n_samples, n_features)
#     reshaped_images = images.reshape((images.shape[0], -1))

#     # Convert to NumPy array (compute the Dask array)
#     reshaped_images_np = reshaped_images.compute()

#     # Apply NMF
#     model = NMF(n_components=1, init="random", random_state=0)
#     _ = model.fit_transform(reshaped_images_np)
#     h = model.components_

#     # Assume the first image's shape is representative
#     flatfield = h.reshape(images.shape[1:])

#     # Normalize the flatfield
#     flatfield_mean = np.mean(flatfield)
#     if flatfield_mean != 0:
#         flatfield /= flatfield_mean

#     return flatfield.astype(np.float32)


# def process_in_parts(shading_tiles_dir: Path, n_parts: int) -> np.ndarray:
#     """
#     Split the reading and processing of images into n_parts, apply NMF to each part,
#     and average the results to get the final flatfield correction.

#     Parameters:
#     - shading_tiles_dir: Path to the directory containing the shading images.
#     - n_parts: Number of parts to split the dataset into.

#     Returns:
#     - A NumPy array representing the averaged flatfield correction.
#     """
#     # Load all image paths
#     # image_paths = list(shading_tiles_dir.glob('*.tif'))
#     image_paths = ut.sorted_files(shading_tiles_dir, ".tif")
#     total_images = len(image_paths)
#     images_per_part = total_images // n_parts

#     flatfields = []

#     for part in range(n_parts):
#         print(f"Part {part+1} of {n_parts}")

#         start_idx = part * images_per_part
#         end_idx = (
#             (part + 1) * images_per_part
#             if part < n_parts - 1
#             else total_images
#         )

#         # Use Dask to lazily load a subset of images
#         # images = dask_image.imread.imread(image_paths[start_idx:end_idx])
#         images = [
#             dask_image.imread.imread(each_path)[None, ...]
#             for each_path in image_paths[start_idx:end_idx]
#         ]
#         image_stack = da.concatenate(images, axis=0)
#         # Apply NMF and get the flatfield correction for this part
#         flatfield = nmf_on_images(image_stack)
#         flatfields.append(flatfield)

#     return flatfields
#     # # Average the flatfield corrections from all parts
#     # averaged_flatfield = np.mean(flatfields, axis=0)

#     # return averaged_flatfield


# def estimate_shading_correction_with_dask(shading_tiles_dir: Path) -> NDArray:
#     """
#     Estimate the shading correction for a set of large images using Non-negative Matrix Factorization (NMF),
#     with optimizations for memory constraints by using Dask arrays for lazy loading and processing.

#     Parameters
#     ----------
#     shading_tiles_dir : Path
#         The directory path where the large shading images are stored. These images are used to compute the shading correction.

#     Returns
#     -------
#     NDArray
#         A 2D numpy array representing the estimated flatfield correction image. This image is used to correct
#         other images for shading effects, normalized so its mean is centered around 1.

#     Notes
#     -----
#     - This function is optimized for handling large images that do not fit into memory by using Dask for lazy evaluation.
#     - The function assumes that all shading images are of the same dimensions and are stored in '.tif' format.
#     - The flatfield correction is derived from the NMF decomposition of the shading images, specifically from the H matrix,
#       and is normalized to have a mean of 1 for consistent lighting correction.
#     """

#     # Use dask_image to lazily load the images as a Dask array
#     images = dask_image.imread.imread(str(shading_tiles_dir / '*.tif'))

#     # Since Dask operates lazily, computations are only triggered upon calling compute().
#     # Here, we reshape the images into a 2D array where each row is a flattened image.
#     # Note: This step might need adjustments based on the actual memory constraints and image sizes.
#     reshaped_images = images.reshape((images.shape[0], -1)).compute()

#     # Apply NMF on the reshaped images to find the flatfield correction
#     # Note: NMF is performed on the in-memory NumPy array since sklearn's NMF does not directly support Dask arrays.
#     model = NMF(n_components=1, init='random', random_state=0)
#     W = model.fit_transform(reshaped_images)
#     H = model.components_

#     # Reshape the H matrix to get the flatfield correction image
#     # The first image's shape is used as a reference for reshaping
#     flatfield_shape = images.shape[1:]  # Ignoring the number of images, just getting one image's shape
#     flatfield = H.reshape(flatfield_shape)

#     # Normalize the flatfield so its mean is centered around 1
#     flatfield_mean = np.mean(flatfield)
#     if flatfield_mean != 0:
#         flatfield /= flatfield_mean

#     print(f'Estimated the flatfield ({flatfield.shape[0]}x{flatfield.shape[1]} pixels) for normalization with non-negative matrix factorization.')

#     return flatfield


@app.command(
    cls=CustomCLICommand,
    name="estimate_shading_correction",
    short_help="Estimate flatfield with NNMF",
)
def estimate_shading_correction(
    shading_tiles_dir: Path, n_images_limit: int = 800
):
    """
    Estimate the shading correction for a set of images using Non-negative Matrix Factorization (NMF).

    This function takes a directory containing shading images, loads them, and applies NMF to estimate
    a flatfield correction image. The flatfield image is normalized so that its mean is centered around 1.
    This normalization is crucial for subsequent image processing steps, ensuring that the corrected images
    have uniform lighting conditions.

    Parameters
    ----------
    shading_tiles_dir : Path
        The directory path where the shading images are stored. These images are used to compute the shading correction.

    Returns
    -------
    NDArray
        A 2D numpy array representing the estimated flatfield correction image. This image is used to correct
        other images for shading effects.

    Notes
    -----
    - The function assumes that all shading images are of the same dimensions.
    - The shading images should be in '.tif' format.
    - The function uses Non-negative Matrix Factorization (NMF) to decompose the shading images into a basis
      component (W) and coefficient matrix (H). The flatfield correction is derived from these components.

    Examples
    --------
    >>> shading_correction = estimate_shading_correction(Path('/path/to/shading/images'))
    >>> print(shading_correction.shape)
    (1024, 1024)  # Example output, the actual size depends on the input images.
    """

    # Retrieve a sorted list of shading image files from the specified directory
    shading_images = ut.sorted_files(shading_tiles_dir, ".tif")

    if len(shading_images) > n_images_limit:
        print(
            f"Woah, found {len(shading_images)} images in {shading_tiles_dir}, selecting random subset of {n_images_limit} images\n"
        )
        shading_images = random.sample(shading_images, n_images_limit)
    else:
        print(f"Found {len(shading_images)} images in {shading_tiles_dir}\n")

    # Load the shading images into a list, using imageio for image reading
    images = [
        imageio.v3.imread(each_shading_image)
        for each_shading_image in tqdm(shading_images)
    ]

    ff_shape = images[0].shape
    # Reshape the loaded images into a 2D array where each row represents a flattened image
    reshaped_images = np.array([img.flatten() for img in images])

    del images

    # Initialize and fit the NMF model to the reshaped images
    # n_components=1 indicates we are reducing the image set to 1 component
    model = NMF(n_components=1, init="random", random_state=0)
    W = model.fit_transform(reshaped_images)

    del reshaped_images

    # Basis matrix
    H = model.components_  # Coefficient matrix

    # Reshape the H matrix to the original image shape to get the flatfield correction
    flatfield = H.reshape(ff_shape)

    # Normalize the flatfield so its mean is centered around 1
    flatfield_mean = np.mean(flatfield)
    if flatfield_mean != 0:
        flatfield /= flatfield_mean

    print(
        f"Estimated the flatfield ({flatfield.shape[0]}x{flatfield.shape[1]} pixels) for normalization with non-negative matrix factorization."
    )

    imageio.v3.imwrite(
        shading_tiles_dir.parent.joinpath("nnmf_shading_correction.tif"),
        flatfield.astype(np.float32),
    )

    # return flatfield


import imageio.v3 as iio
import tempfile


def estimate_shading_correction_v2(
    shading_tiles_dir: Path,
    n_images_limit: int = 800,
) -> np.ndarray:
    """
    Estimate the flatfield (shading correction) from a directory of TIFF tiles
    using rank‐1 NMF, but without ever loading all images into RAM at once.

    Parameters
    ----------
    shading_tiles_dir : Path
        Directory containing the '.tif' shading‐tile images.
    n_images_limit : int, optional
        Maximum number of images to sample for the NMF fitting.

    Returns
    -------
    flatfield : np.ndarray
        2D array of the shading‐correction (mean scaled to 1).
    """

    # 1) Gather and possibly subsample the list of files
    shading_images = ut.sorted_files(shading_tiles_dir, ".tif")
    n_total = len(shading_images)
    if n_total > n_images_limit:
        print(
            f"Found {n_total} images, sampling {n_images_limit} of them for memory‐savings."
        )
        shading_images = np.random.choice(
            shading_images, n_images_limit, replace=False
        )
    else:
        print(f"Found {n_total} images.")

    n_images = len(shading_images)

    # 2) Peek at the first image to get shape
    first_img = iio.imread(shading_images[0]).astype(np.float32)

    height, width, _ = first_img.shape
    del first_img

    # 3) Create a temporary file on disk and memory‐map it
    tmp = tempfile.NamedTemporaryFile(
        dir=shading_tiles_dir.parent, suffix=".dat", delete=False
    )
    tmp_filename = tmp.name
    tmp.close()

    # A memmap of shape (n_images, height*width) dtype float32
    X = np.memmap(
        tmp_filename,
        dtype=np.float32,
        mode="w+",
        shape=(n_images, height * width),
    )

    # 4) Stream each TIFF into the memmap
    for idx, img_path in enumerate(
        tqdm(shading_images, desc="Building memmap")
    ):
        img = iio.imread(img_path).astype(np.float32)
        X[idx, :] = img.ravel()
    X.flush()

    # 5) Fit a rank‐1 NMF
    #    This will only hold (n_images × 1) W and (1 × n_pixels) H in RAM
    model = NMF(
        n_components=1,
        init="random",
        random_state=0,
        max_iter=200,
        tol=1e-4,
    )
    W = model.fit_transform(X)  # shape = (n_images, 1)
    H = model.components_[0]  # shape = (height*width,)

    # 6) Tear down the memmap file
    del X
    os.remove(tmp_filename)

    # 7) Reshape H back into a 2D flatfield and normalize its mean to 1
    flatfield = H.reshape((height, width))
    m = flatfield.mean()
    if m != 0:
        flatfield /= m

    # 8) Optionally write it out
    out_path = (
        shading_tiles_dir.parent
        / f"nnmf_shading_correction_{n_images_limit}.tif"
    )
    iio.imwrite(out_path, flatfield.astype(np.float32))

    print(
        f"Estimated flatfield of size {flatfield.shape} and wrote to {out_path}"
    )


# def save_shading_correction(shading_correction_img: np.ndarray, save_location: Path):
#     """Save the shading correction"""

#     # make the directory if it doesn't exist
#     parent_dir = save_location.parent.mkdir(parents=True,exist_ok=True)

#     # save
#     imageio.imwrite(save_location,shading_correction_img)


@app.command(
    cls=CustomCLICommand,
    name="estimate_basic_shading_correction",
    short_help="Estimate BaSiC (fiji plugin) flatfield",
)
def estimate_basic_shading_correction(
    shading_tiles_dir: Path, fiji_app: Path
) -> NDArray:
    """To come"""

    # write ImageJ macro
    shading_macro = shading_tiles_dir.parent.joinpath(
        f"{shading_tiles_dir.name}_shading_macro.ijm"
    )

    print(f"Estimating basic shading correction: {shading_macro}")

    if platform == "win32":
        shading_tiles_dir_str = str(shading_tiles_dir).replace(os.sep, "/")
    else:
        shading_tiles_dir_str = str(shading_tiles_dir)

    with open(shading_macro, "w", encoding="utf-8") as f:
        # f.write('setBatchMode(true);\n')
        f.write(f'File.openSequence("{shading_tiles_dir_str}");\n')
        # f.write("imageTitle = getTitle();\n")
        f.write(
            f'run("BaSiC ", "processing_stack={shading_tiles_dir.name} flat-field=None dark-field=None shading_estimation=[Estimate shading profiles] shading_model=[Estimate flat-field only (ignore dark-field)] setting_regularisationparametes=Manual temporal_drift=Ignore correction_options=[Compute shading only] lambda_flat=5 lambda_dark=0.50");\n'
        )
        # f.write('close(imageTitle);\n')
        f.write(f'selectImage("Flat-field:{shading_tiles_dir.name}");\n')
        f.write(
            f'saveAs("Tiff", "{str(shading_tiles_dir.parent.joinpath(f"BaSiC_flatfield_estimate_{shading_tiles_dir.name}.tif")).replace(os.sep, "/")}");\n'
        )
        f.write('run("Close All");\n')
        f.write('run("Quit");')
        # f.write('setBatchMode(false);')
        f.close()

    # result = subprocess.run([str(fiji_app), "--run", str(shading_macro)], capture_output=True, text=True)
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)

    ## Don't add "--headless",  to not run in headless
    # https://forum.image.sc/t/set-memory-from-command-line-upon-startup-headless/8668/4
    run_macro = Popen(
        [str(fiji_app), "-macro", str(shading_macro)],
        shell=False,
        stdout=PIPE,
        stderr=PIPE,
    )

    # stream_data = run_macro.communicate()[0]
    stdout, stderr = run_macro.communicate()

    # print("STDOUT:", stdout.decode())
    # print("STDERR:", stderr.decode())

    if "error" in str(stdout).lower():
        raise OSError(
            f"Shading estimation failed on current slice, Fiji output:\n{stdout}"
        )


@app.command(
    cls=CustomCLICommand,
    name="reorganize_tiles_by_channel",
    short_help="Reorganize tiles by channel keyword",
)
def reorganize_tiles_by_channel(
    tiles_dir: Path, channel_keywords: List[str], move_files: bool = True
):
    """Reorganize by moving files based on keywords such as Pol, Bright, or Dark."""

    dest_folder = tiles_dir.parent.expanduser()

    for file_path in tiles_dir.rglob(
        "*"
    ):  # Use rglob to find all files recursively
        if file_path.is_file():
            for keyword in channel_keywords:
                if keyword in file_path.name:
                    keyword_dest_directory = dest_folder / keyword
                    keyword_dest_directory.mkdir(
                        parents=True, exist_ok=True
                    )  # Create directory if it doesn't exist
                    if move_files:
                        shutil.move(
                            file_path, keyword_dest_directory / file_path.name
                        )
                    else:
                        shutil.copy2(
                            file_path, keyword_dest_directory / file_path.name
                        )
                    break  # Move to the next file after finding the first matching keyword

    print("\tTiles reorganized by channels.\n")

    # remove origin directory
    if move_files:
        ut.rmdir(tiles_dir)


def combine_rgb_channels(
    red_channel_path, green_channel_path, blue_channel_path, output_path
):
    """_summary_

    Args:
        red_channel_path (_type_): _description_
        green_channel_path (_type_): _description_
        blue_channel_path (_type_): _description_
        output_path (_type_): _description_
    """
    # Read the individual channel images
    r_channel = imageio.v3.imread(red_channel_path)
    g_channel = imageio.v3.imread(green_channel_path)
    b_channel = imageio.v3.imread(blue_channel_path)

    # Check if all channels have the same shape
    if (
        r_channel.shape != g_channel.shape
        or r_channel.shape != b_channel.shape
    ):
        print("Error: All channel images must have the same dimensions.")
        return

    # Stack the channels along the last dimension to create an RGB image
    rgb_image = np.stack((r_channel, g_channel, b_channel), axis=-1)

    # Save the combined image as a 32-bit RGB image
    imageio.imwrite(
        output_path, rgb_image, format="TIFF"
    )  # Use TIFF for float support


def separate_color_channels(input_dir: Path) -> list[Path]:
    """To come"""

    print(f"\tReading images in {input_dir}\n")
    parent_dir = input_dir.parent

    # Create directories for each channel in the parent directory
    red_channel_dir = parent_dir / f"{input_dir.name}_channel_1_red"
    green_channel_dir = parent_dir / f"{input_dir.name}_channel_2_green"
    blue_channel_dir = parent_dir / f"{input_dir.name}_channel_3_blue"

    red_channel_dir.mkdir(exist_ok=True)
    green_channel_dir.mkdir(exist_ok=True)
    blue_channel_dir.mkdir(exist_ok=True)

    print(
        f"Splitting RGB images into \n\t\t{red_channel_dir}\n\t\t{green_channel_dir}\n\t\t{blue_channel_dir}"
    )

    all_color_images_paths = ut.sorted_files(input_dir, ".tif")

    for each_color_image_path in tqdm(all_color_images_paths):
        # color_image = Image.open(each_color_image_path)
        color_image = imageio.v3.imread(each_color_image_path)

        # # Check if the image is in RGB mode
        # if color_image.mode != 'RGB':
        #     print(f"The image is in {color_image.mode} mode, not RGB. Please provide an RGB image.")
        #     ut.rmdir(red_channel_dir)
        #     ut.rmdir(green_channel_dir)
        #     ut.rmdir(blue_channel_dir)
        #     return

        # # Split the image into its RGB channels
        # r, g, b = color_image.split()

        # Check if the image is RGB
        if color_image.ndim != 3 or color_image.shape[2] != 3:
            print("The image is not in RGB format.")
            ut.rmdir(red_channel_dir)
            ut.rmdir(green_channel_dir)
            ut.rmdir(blue_channel_dir)
            return

        # Split the image into its RGB channels
        r_channel = color_image[:, :, 0]  # Red channel
        g_channel = color_image[:, :, 1]  # Green channel
        b_channel = color_image[:, :, 2]  # Blue channel

        # Save each channel as an image
        imageio.imwrite(
            red_channel_dir / f"{each_color_image_path.stem}_channel1_red.tif",
            r_channel,
        )
        imageio.imwrite(
            green_channel_dir
            / f"{each_color_image_path.stem}_channel2_green.tif",
            g_channel,
        )
        imageio.imwrite(
            blue_channel_dir
            / f"{each_color_image_path.stem}_channel3_blue.tif",
            b_channel,
        )

    return [red_channel_dir, green_channel_dir, blue_channel_dir]
