# aligner.py is aa refactor of Alignment.py

from sys import platform
import os
import time
import shutil
from pathlib import Path
from enum import Enum
from typing import NamedTuple
import tempfile
from subprocess import Popen, DEVNULL
import numpy as np
from PIL import Image, ImageEnhance
import scipy.ndimage
import scipy.optimize
import scipy.optimize

Image.MAX_IMAGE_PIXELS = None
import imageio
from skimage import feature, transform, registration, measure, exposure
import matplotlib.pyplot as plt
import typer
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
)  # Import Rich progress componentsimport yaml  # Assuming the configuration is in YAML format

from pystackreg import StackReg

from skimage.transform import pyramid_gaussian

# from skimage.color import rgb2gray

from precon3d.stitcher import FijiAttrs
import precon3d.utility as ut
import precon3d.factory as gen_types

# pylint: disable=wildcard-import
from precon3d.custom_types import *
from precon3d._my_typer_cli import CustomCLIGroup, CustomCLICommand

# CLI
app = typer.Typer(
    cls=CustomCLIGroup,
    no_args_is_help=True,
    short_help="Image registration/alignment methods",
)


class RegistrationMethod(Enum):
    """
    Sepcific registration type

    Attributes supported
    --------------------
    TURBOREG: str
        Fiji's turboreg method
    SIFT: str
        skimage SIFT (keypoints)
    PHASECORR: str
        skimage phasecorrelation (fft)
    """

    TURBOREG: str = "turboreg"
    SIFT: str = "sift"
    PHASECORR: str = "phasecorr"


class AlignmentAttrs(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """

    registration_method: RegistrationMethod = "turboreg"
    use_subarea: bool = False
    subarea_x: int = None
    subarea_y: int = None
    subarea_width: int = None
    subarea_height: int = None


def create_aligner_attrs(data: Dict[str, Any]) -> AlignmentAttrs:
    """Create AlignmentAttrs from a dictionary."""
    return AlignmentAttrs(
        registration_method=str(data["aligner_attrs"]["registration_method"]),
        use_subarea=bool(data["aligner_attrs"]["use_subarea"]),
        subarea_x=int(data["aligner_attrs"]["subarea_x"]),
        subarea_y=int(data["aligner_attrs"]["subarea_y"]),
        subarea_width=int(data["aligner_attrs"]["subarea_width"]),
        subarea_height=int(data["aligner_attrs"]["subarea_height"]),
    )


class DeltaTranslation(NamedTuple):
    """Represents a translation in a 2D space.

    This class is used to define a translation vector with respect to the x and y axes.

    Args:
        dx (float): The translation distance along the x-axis.
        dy (float): The translation distance along the y-axis.
    """

    dx: float
    dy: float

    def __str__(self):
        """Return a string representation of the DeltaTranslation instance."""
        return f"(dx={self.dx}, dy={self.dy})"


def apply_shift(
    mov_img: np.ndarray, dt: DeltaTranslation, scale_factor: float = 1.0
) -> np.array:
    # Apply shift to the second image
    # shift = np.array(dt.dx, dt.dy)
    tform = transform.AffineTransform(
        translation=[scale_factor * dt.dx, scale_factor * dt.dy]
    )
    registered_image = transform.warp(
        mov_img, tform.inverse, clip=False, preserve_range=True
    )
    return registered_image


def pad_arrays_to_match(ref_array: np.ndarray, mov_array: np.ndarray):
    """Pad the smaller array among ref_array and mov_array to match the size of the larger array."""
    # Calculate the size difference
    delta_height = ref_array.shape[0] - mov_array.shape[0]
    delta_width = ref_array.shape[1] - mov_array.shape[1]

    # Determine padding for height and width
    pad_height = abs(delta_height // 2), abs(
        delta_height // 2 + delta_height % 2
    )
    pad_width = abs(delta_width // 2), abs(delta_width // 2 + delta_width % 2)

    # Apply padding to the smaller array
    if delta_height < 0 or delta_width < 0:
        # If reference array is smaller, pad it
        ref_array = np.pad(ref_array, (pad_height, pad_width), mode="constant")
    else:
        # If moving array is smaller or they are equal (no padding needed), pad moving array
        mov_array = np.pad(mov_array, (pad_height, pad_width), mode="constant")

    # Ensure both arrays have the same shape by padding the other side if necessary
    if ref_array.shape != mov_array.shape:
        # If there's still a difference, this means one dimension was equal and the other was not
        # Determine new deltas
        delta_height = ref_array.shape[0] - mov_array.shape[0]
        delta_width = ref_array.shape[1] - mov_array.shape[1]

        # New padding calculations
        pad_height = abs(delta_height), 0
        pad_width = abs(delta_width), 0

        # Apply additional padding to the smaller array
        if delta_height != 0:
            if delta_height > 0:
                # Moving array is smaller in height
                mov_array = np.pad(
                    mov_array, (pad_height, (0, 0)), mode="constant"
                )
            else:
                # Reference array is smaller in height
                ref_array = np.pad(
                    ref_array, (pad_height, (0, 0)), mode="constant"
                )

        if delta_width != 0:
            if delta_width > 0:
                # Moving array is smaller in width
                mov_array = np.pad(
                    mov_array, ((0, 0), pad_width), mode="constant"
                )
            else:
                # Reference array is smaller in width
                ref_array = np.pad(
                    ref_array, ((0, 0), pad_width), mode="constant"
                )

    return ref_array, mov_array


from scipy.optimize import minimize
import scipy


def calc_shift_max(
    ref_array, mov_array
):  # (reference_image: Path, moving_image: Path):
    """Maximize the product of the images"""

    # # Read and convert images to grayscale
    # ref_img = Image.open(reference_image).convert("L")
    # mov_img = Image.open(moving_image).convert("L")

    # # Convert images to numpy arrays
    # ref_array = np.array(ref_img)
    # mov_array = np.array(mov_img)

    # mov_array = scipy.ndimage.rotate(mov_array,15)

    # # Pad
    # ref_array, mov_array = pad_arrays_to_match(ref_image_gray, mov_image_gray)

    # Define a function to calculate the negative product sum (since we'll be minimizing)
    def neg_product_sum(shift, ref_array, mov_array):
        """Calculate negative of product sum for given shift."""
        dx, dy = int(shift[0]), int(shift[1])
        shifted_mov_array = np.roll(mov_array, shift=(dx, dy), axis=(0, 1))

        # For simplicity, we'll ignore the edges where the images don't overlap
        # product_sum = np.sum(ref_array * shifted_mov_array)
        # return -math.log(product_sum)  # Negative for minimization
        # Calculate the log of each product to avoid overflow, then sum
        # Adding a small constant epsilon to avoid log(0)
        epsilon = 1e-10
        log_product_sum = np.sum(
            np.log(np.abs(ref_array * shifted_mov_array) + epsilon)
        )

        return -log_product_sum  # Return negative because we're minimizing

    # Initial guess for the shift
    initial_shift = [0, 0]

    # Use scipy's minimize function to find the optimal shift
    result = minimize(
        neg_product_sum,
        initial_shift,
        args=(ref_array, mov_array),
        method="Powell",
    )

    optimal_shift = result.x
    print(f"Optimal shift: {optimal_shift}")

    return optimal_shift
    # breakpoint()

    # # Apply the optimal shift to the moving image
    # optimal_dx, optimal_dy = int(optimal_shift[0]), int(optimal_shift[1])
    # optimal_shifted_mov_array = np.roll(mov_array, shift=(optimal_dx, optimal_dy), axis=(0, 1))

    # # Convert back to PIL image for any further processing or saving
    # optimal_shifted_mov_img = Image.fromarray(optimal_shifted_mov_array)

    # return optimal_shifted_mov_img


def pystackreg_shift(reference, moving):
    """Optimize the shift between two images using pystackreg."""

    # read in
    ref_img = Image.open(reference).convert("L")
    mov_img = Image.open(moving).convert("L")

    # # Convert images to numpy arrays
    ref_array = np.array(ref_img)
    mov_array = np.array(mov_img)

    sr = StackReg(StackReg.TRANSLATION)

    # register images
    tmats = sr.register(ref_array, mov_array)

    return DeltaTranslation(dx=tmats[0, 2], dy=tmats[1, 2])


def optimize_shift(ref_array, mov_array):
    """Optimize the shift between two images using image pyramids."""

    # Convert images to grayscale
    # ref_image_gray = rgb2gray(ref_image)
    # mov_image_gray = rgb2gray(mov_image)

    reduction_factor = 10

    # Generate image pyramids
    pyramid_ref = tuple(
        pyramid_gaussian(ref_array, downscale=reduction_factor)
    )
    pyramid_mov = tuple(
        pyramid_gaussian(mov_array, downscale=reduction_factor)
    )

    # Start with an initial guess for the shift
    initial_shift = np.array([0, 0])

    # Iterate over pyramid levels starting from the smallest image
    for ref, mov in zip(reversed(pyramid_ref), reversed(pyramid_mov)):
        # Scale the initial shift for the current resolution
        initial_shift *= reduction_factor

        # Define the objective function for optimization
        # def objective_function(shift):
        #     shifted_mov = np.roll(mov, shift.astype(int), axis=(0, 1))
        #     # Calculate the negative of the sum of squared differences (SSD)
        #     return -np.sum((ref - shifted_mov) ** 2)
        # Define a function to calculate the negative product sum (since we'll be minimizing)
        def objective_function(shift):
            """Calculate negative of product sum for given shift."""
            dx, dy = int(shift[0]), int(shift[1])
            shifted_mov_array = np.roll(mov, shift=(dx, dy), axis=(0, 1))

            # For simplicity, we'll ignore the edges where the images don't overlap
            # product_sum = np.sum(ref_array * shifted_mov_array)
            # return -math.log(product_sum)  # Negative for minimization
            # Calculate the log of each product to avoid overflow, then sum
            # Adding a small constant epsilon to avoid log(0)
            epsilon = 1e-10
            log_product_sum = np.sum(
                np.log(np.abs(ref * shifted_mov_array) + epsilon)
            )

            return -log_product_sum  # Return negative because we're minimizing

        # Optimize the shift
        result = minimize(objective_function, initial_shift, method="Powell")
        initial_shift = result.x

    print(f"Optimal shift: {initial_shift}")
    return initial_shift


# import cupy as cp


# def pyramid_down_gpu(image, downscale=10):
#     """Downsample an image by a given factor using CuPy."""
#     # Assuming 'image' is a CuPy array
#     # Simple downsampling, for demonstration purposes
#     return image[::downscale, ::downscale]


# def generate_pyramid_gpu(image, levels):
#     """Generate an image pyramid as a list of images of decreasing sizes using CuPy."""
#     pyramid = [image]
#     for level in range(1, levels):
#         pyramid.append(pyramid_down_gpu(pyramid[level - 1]))
#     return pyramid


# def optimize_shift_pyd_gpu(ref_image_gpu, mov_image_gpu, levels=3):
#     """Optimize the shift between two images using image pyramids and CuPy for GPU acceleration.

#     Find the optimal shift to align mov_image_gpu with ref_image_gpu.

#     Parameters:
#     - ref_image_gpu: CuPy array of the reference image.
#     - mov_image_gpu: CuPy array of the moving image.

#     Returns:
#     - Optimal shift as a tuple (y_shift, x_shift).
#     """
#     # Convert images to grayscale and to CuPy arrays
#     # ref_image_gpu = cp.asarray(ref_image)
#     # mov_image_gpu = cp.asarray(mov_image)

#     # Generate image pyramids
#     pyramid_ref = generate_pyramid_gpu(ref_image_gpu, levels)
#     pyramid_mov = generate_pyramid_gpu(mov_image_gpu, levels)

#     initial_shift = cp.array([0, 0], dtype=cp.float64)

#     for level in reversed(range(levels)):
#         # Scale the initial shift for the current resolution
#         initial_shift *= 10
#         ref = pyramid_ref[level]
#         mov = pyramid_mov[level]

#         # Define the objective function for optimization
#         def objective_function(shift):
#             # Ensure shift is integer for indexing
#             shift = cp.asarray(shift).astype(cp.int32)
#             shifted_mov = cp.roll(mov, shift, axis=(0, 1))
#             # Calculate the negative of the sum of squared differences (SSD)
#             # ssd = -cp.sum((ref - shifted_mov) ** 2).get()  # Use .get() to transfer result to CPU
#             # return ssd
#             # Adding a small constant epsilon to avoid log(0)
#             epsilon = 1e-5
#             log_product_sum = cp.sum(
#                 cp.log(cp.abs(ref * shifted_mov) + epsilon)
#             ).get()

#             return -log_product_sum  # Return negative because we're minimizing

#         result = scipy.optimize.basinhopping(
#             objective_function,
#             initial_shift.get(),
#             niter=200,
#             stepsize=50,
#             niter_success=50,
#         )

#         # Optimize the shift using scipy (CPU-based)
#         # Note: The objective function transfers the final SSD value back to the CPU
#         # result = minimize(objective_function, initial_shift.get(), method='Powell')  # Use .get() for CPU compatibility
#         initial_shift = cp.array(
#             result.x
#         )  # Transfer optimized shift back to GPU

#     return initial_shift.get()  # Return final shift as a NumPy array


# def optimize_shift_gpu(ref_image_gpu, mov_image_gpu, levels=3):
#     """Optimize the shift between two images using image pyramids and CuPy for GPU acceleration.

#     Find the optimal shift to align mov_image_gpu with ref_image_gpu.

#     Parameters:
#     - ref_image_gpu: CuPy array of the reference image.
#     - mov_image_gpu: CuPy array of the moving image.

#     Returns:
#     - Optimal shift as a tuple (y_shift, x_shift).
#     """
#     # Convert images to grayscale and to CuPy arrays
#     # ref_image_gpu = cp.asarray(ref_image)
#     # mov_image_gpu = cp.asarray(mov_image)

#     # Generate image pyramids
#     # pyramid_ref = generate_pyramid_gpu(ref_image_gpu, levels)
#     # pyramid_mov = generate_pyramid_gpu(mov_image_gpu, levels)

#     ref = ref_image_gpu
#     mov = mov_image_gpu

#     initial_shift = cp.array([0, 0], dtype=cp.float64)

#     # Define the objective function for optimization
#     def objective_function(shift):
#         # Ensure shift is integer for indexing
#         shift = cp.asarray(shift).astype(cp.int32)
#         shifted_mov = cp.roll(mov, shift, axis=(0, 1))
#         # Calculate the negative of the sum of squared differences (SSD)
#         # ssd = -cp.sum((ref - shifted_mov) ** 2).get()  # Use .get() to transfer result to CPU
#         # return ssd
#         # Adding a small constant epsilon to avoid log(0)
#         epsilon = 1e-5
#         log_product_sum = cp.sum(
#             cp.log(cp.abs(ref * shifted_mov) + epsilon)
#         ).get()

#         return -log_product_sum  # Return negative because we're minimizing

#     result = scipy.optimize.basinhopping(
#         objective_function,
#         initial_shift.get(),
#         niter=200,
#         stepsize=50,
#         niter_success=20,
#     )

#     return result.x
#     # Optimize the shift using scipy (CPU-based)
#     # Note: The objective function transfers the final SSD value back to the CPU
#     # result = minimize(objective_function, initial_shift.get(), method='Nelder-Mead')  # Use .get() for CPU compatibility
#     # initial_shift = cp.array(result.x)  # Transfer optimized shift back to GPU

#     result = minimize(
#         objective_function, initial_shift.get(), method="Nelder-Mead"
#     )  # method="BFGS")#method='Powell')  # Use .get() for CPU compatibility

#     brute_range = ((-100, 100), (-100, 100))
#     result = scipy.optimize.brute(func=objective_function, ranges=brute_range)
#     return result

#     # initial_shift = cp.array(result.x)  # Transfer optimized shift back to GPU

#     # return initial_shift.get()  # Return final shift as a NumPy array


def calc_shifts_phase_corr(
    reference_image: Path,
    moving_image: Path,
    alignment_attrs: AlignmentAttrs,
) -> DeltaTranslation:
    """_summary_

    Args:
        reference_image (Path): _description_
        moving_image (Path): _description_
        alignment_attrs (AlignmentAttrs): _description_

    Returns:
        DeltaTranslation: _description_
    """

    # Read and convert images to grayscale
    ref_img = Image.open(reference_image).convert("L")
    mov_img = Image.open(moving_image).convert("L")

    # Crop the images if use_subarea is set
    if alignment_attrs.use_subarea:
        # Define the cropping box
        crop_box = (
            alignment_attrs.subarea_x,
            alignment_attrs.subarea_y,
            alignment_attrs.subarea_x + alignment_attrs.subarea_width,
            alignment_attrs.subarea_y + alignment_attrs.subarea_height,
        )
        ref_img = ref_img.crop(crop_box)
        mov_img = mov_img.crop(crop_box)

    ref_img = np.array(ref_img)
    mov_img = np.array(mov_img)

    shift, _, _ = registration.phase_cross_correlation(ref_img, mov_img)

    delta_translation = DeltaTranslation(dx=shift[1], dy=shift[0])

    return delta_translation


def coarse_calc_shifts_phase_corr(
    reference_image: Path,
    moving_image: Path,
    alignment_attrs: AlignmentAttrs,
    scale_factor: float = 0.05,
) -> DeltaTranslation:
    """_summary_

    Args:
        reference_image (Path): _description_
        moving_image (Path): _description_
        alignment_attrs (AlignmentAttrs): _description_

    Returns:
        DeltaTranslation: _description_
    """

    # Read and convert images to grayscale
    ref_img = Image.open(reference_image).convert("L")
    mov_img = Image.open(moving_image).convert("L")

    # Crop the images if use_subarea is set
    if alignment_attrs.use_subarea:
        # Define the cropping box
        crop_box = (
            alignment_attrs.subarea_x,
            alignment_attrs.subarea_y,
            alignment_attrs.subarea_x + alignment_attrs.subarea_width,
            alignment_attrs.subarea_y + alignment_attrs.subarea_height,
        )
        ref_img = ref_img.crop(crop_box)
        mov_img = mov_img.crop(crop_box)

    ref_img_scaled = scipy.ndimage.zoom(
        np.array(ref_img), (scale_factor, scale_factor), order=3
    )
    mov_img_scaled = scipy.ndimage.zoom(
        np.array(mov_img), (scale_factor, scale_factor), order=3
    )

    ref_img_8bit = ut.convert_gray_to_8bit(ref_img_scaled)
    mov_img_8bit = ut.convert_gray_to_8bit(mov_img_scaled)

    shift, _, _ = registration.phase_cross_correlation(
        ref_img_8bit, mov_img_8bit
    )

    delta_translation = DeltaTranslation(
        dx=shift[1] / scale_factor, dy=shift[0] / scale_factor
    )

    print(f"{moving_image.name} shifts {delta_translation}")

    return delta_translation


def calc_shifts_phase_corr_pyramid(
    reference_image: Path,
    moving_image: Path,
    alignment_attrs: AlignmentAttrs,
    max_layer=3,
) -> DeltaTranslation:
    """_summary_

    Args:
        reference_image (Path): _description_
        moving_image (Path): _description_
        alignment_attrs (AlignmentAttrs): _description_

    Returns:
        DeltaTranslation: _description_
    """

    # Read and convert images to grayscale
    ref_img = Image.open(reference_image).convert("L")
    mov_img = Image.open(moving_image).convert("L")

    # Crop the images if use_subarea is set
    if alignment_attrs.use_subarea:
        # Define the cropping box
        crop_box = (
            alignment_attrs.subarea_x,
            alignment_attrs.subarea_y,
            alignment_attrs.subarea_x + alignment_attrs.subarea_width,
            alignment_attrs.subarea_y + alignment_attrs.subarea_height,
        )
        ref_img = ref_img.crop(crop_box)
        mov_img = mov_img.crop(crop_box)

    ref_img = np.array(ref_img)
    mov_img = np.array(mov_img)

    # Generate Gaussian pyramids for both images
    pyramid1 = tuple(transform.pyramid_gaussian(ref_img, max_layer=max_layer))
    pyramid2 = tuple(transform.pyramid_gaussian(mov_img, max_layer=max_layer))

    shift = np.array([0, 0])
    for layer in range(max_layer, -1, -1):
        # Calculate shift at current pyramid level
        shift_temp, _, _ = registration.phase_cross_correlation(
            pyramid1[layer], pyramid2[layer], upsample_factor=10
        )
        # Upscale shift for next level
        shift = (shift + shift_temp) * 2 if layer > 0 else shift + shift_temp

    # average
    shift = shift / (max_layer + 1)

    delta_translation = DeltaTranslation(dx=shift[1], dy=shift[0])

    return delta_translation


def calc_shifts_using_keypoints(
    reference_image: Path,
    moving_image: Path,
    alignment_attrs: AlignmentAttrs,
    show_plots: bool = False,
) -> DeltaTranslation:
    """_summary_

    Args:
        reference_image (Path): _description_
        moving_image (Path): _description_
        alignment_attrs (AlignmentAttrs): _description_
        fiji_attrs (FijiAttrs): _description_

    Returns:
        DeltaTranslation: _description_
    """

    # Read and convert images to grayscale
    ref_img = Image.open(reference_image)  # .convert("L")
    mov_img = Image.open(moving_image)  # .convert("L")

    # Crop the images if use_subarea is set
    if alignment_attrs.use_subarea:
        # Define the cropping box
        crop_box = (
            alignment_attrs.subarea_x,
            alignment_attrs.subarea_y,
            alignment_attrs.subarea_x + alignment_attrs.subarea_width,
            alignment_attrs.subarea_y + alignment_attrs.subarea_height,
        )
        ref_img = ref_img.crop(crop_box)
        mov_img = mov_img.crop(crop_box)

    # convert to gray (8bit)
    ref_img = Image.fromarray(
        ut.convert_gray_to_8bit(np.array(ref_img), gamma=0.5)
    ).convert("L")
    mov_img = Image.fromarray(
        ut.convert_gray_to_8bit(np.array(mov_img), gamma=0.5)
    ).convert("L")

    # Create a Contrast enhancer object
    enhancer_ref_img = ImageEnhance.Contrast(ref_img)
    enhancer_mov_img = ImageEnhance.Contrast(mov_img)

    # Enhance the contrast (e.g., by a factor of 3)
    ref_img = enhancer_ref_img.enhance(3)
    mov_img = enhancer_mov_img.enhance(3)

    ref_img = np.array(ref_img)
    mov_img = np.array(mov_img)

    # Enhance contrast using CLAHE
    # ref_img = exposure.equalize_adapthist(ref_img, clip_limit=0.03)
    # mov_img = exposure.equalize_adapthist(mov_img, clip_limit=0.03)

    # Detect SIFT keypoints and descriptors
    sift = feature.SIFT()
    sift.detect_and_extract(ref_img)
    keypoints1 = sift.keypoints
    descriptors1 = sift.descriptors

    sift.detect_and_extract(mov_img)
    keypoints2 = sift.keypoints
    descriptors2 = sift.descriptors

    # # Detect ORB keypoints and descriptors
    # orb = feature.ORB(n_keypoints=500, fast_threshold=0.05)

    # orb.detect_and_extract(ref_img)
    # keypoints1 = orb.keypoints
    # descriptors1 = orb.descriptors

    # orb.detect_and_extract(mov_img)
    # keypoints2 = orb.keypoints
    # descriptors2 = orb.descriptors

    # Match descriptors using Hamming distance
    matches = feature.match_descriptors(
        descriptors1, descriptors2, cross_check=True
    )

    src = keypoints2[matches[:, 1]][:, ::-1]
    dst = keypoints1[matches[:, 0]][:, ::-1]

    # Use RANSAC to estimate a robust transform
    model_robust, inliers = measure.ransac(
        (src, dst),
        transform.EuclideanTransform,
        min_samples=4,
        residual_threshold=2,
        max_trials=1000,
    )

    # Extract matched keypoints
    # matched_keypoints1 = keypoints1[matches[:, 0]]
    # matched_keypoints2 = keypoints2[matches[:, 1]]

    # # Estimate translation using RANSAC
    # model_robust = transform.estimate_transform('euclidean', matched_keypoints1, matched_keypoints2)
    # # model_robust, _ = transform.estimate_transform('euclidean', matched_keypoints1, matched_keypoints2, return_inliers=True)

    # # Extract translation amount
    # translation = model_robust.translation

    best_translation = DeltaTranslation(
        model_robust.translation[0], model_robust.translation[1]
    )

    # Visualize matches
    if show_plots:

        print(f"Estimated translation: {best_translation}")
        inlier_matches = matches[inliers]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

        # Plot the images next to each other
        img3 = np.concatenate((ref_img, mov_img), axis=1)
        ax.imshow(img3, cmap="gray")
        ax.plot(
            keypoints1[matches[:, 0], 1],
            keypoints1[matches[:, 0], 0],
            "ro",
            markersize=4,
            alpha=0.1,
        )
        ax.plot(
            keypoints2[matches[:, 1], 1] + ref_img.shape[1],
            keypoints2[matches[:, 1], 0],
            "ro",
            markersize=4,
            alpha=0.1,
        )

        ax.plot(
            keypoints1[inlier_matches[:, 0], 1],
            keypoints1[inlier_matches[:, 0], 0],
            "co",
            markersize=8,
            alpha=0.2,
        )
        ax.plot(
            keypoints2[inlier_matches[:, 1], 1] + ref_img.shape[1],
            keypoints2[inlier_matches[:, 1], 0],
            "mo",
            markersize=8,
            alpha=0.2,
        )

        # Draw lines between matched keypoints
        for match in inlier_matches:
            pt1 = (keypoints1[match[0]][1], keypoints1[match[0]][0])
            pt2 = (
                keypoints2[match[1]][1] + ref_img.shape[1],
                keypoints2[match[1]][0],
            )
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "g", alpha=0.1)
            # ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "go", alpha=0.05)

        # Turn off axis
        ax.axis("off")
        plt.title(f"Estimated translation: {best_translation}")
        plt.tight_layout()
        plt.show()

    # breakpoint()

    return best_translation


# def calc_shifts_turboreg(
#     reference_image: Path,
#     moving_image: Path,
#     alignment_attrs: AlignmentAttrs,
#     fiji_attrs: FijiAttrs,
# ) -> DeltaTranslation:
#     """
#     Calculate shifts using TurboReg.

#     Parameters
#     ----------
#     reference_image : Path
#         Path to the reference image.
#     moving_image : Path
#         Path to the moving image.
#     alignment_attrs : AlignmentAttrs
#         Attributes related to alignment, including cropping.
#     fiji_attrs : FijiAttrs
#         Attributes related to Fiji application settings.

#     Returns
#     -------
#     DeltaTranslation
#         The calculated shifts (dx, dy).
#     """

#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Define the destination path for the temporary image
#         temp_reference_image = os.path.join(temp_dir, os.path.basename(reference_image))
#         temp_moving_image = os.path.join(temp_dir, os.path.basename(moving_image))

#         # Read the image using imageio
#         ref_img = imageio.v3.imread(reference_image)
#         mov_img = imageio.v3.imread(moving_image)

#         # Convert to gray if color
#         if ref_img.ndim == 2 and mov_img.ndim == 2:
#             pass
#         elif ref_img.ndim == 3 and ref_img.shape[-1] == 3 and mov_img.ndim == 3 and mov_img.shape[-1] == 3:
#             ref_img = skimage.color.rgb2gray(ref_img)
#             mov_img = skimage.color.rgb2gray(mov_img)
#         elif ref_img.ndim == 3 and ref_img.shape[-1] == 1 and  mov_img.ndim == 3 and mov_img.shape[-1] == 1:
#             ref_img = ref_img.squeeze()
#             mov_img = mov_img.squeeze()
#         else:
#             raise TypeError(f'Images must be 3-channel color or gray, not shaped as {ref_img.shape} and {mov_img.shape}')


def calc_shifts_turboreg(
    reference_image: Path,
    moving_image: Path,
    alignment_attrs: AlignmentAttrs,
    fiji_attrs: FijiAttrs,
) -> DeltaTranslation:
    """
    Calculate shifts using TurboReg.

    Parameters
    ----------
    reference_image : Path
        Path to the reference image.
    moving_image : Path
        Path to the moving image.
    alignment_attrs : AlignmentAttrs
        Attributes related to alignment, including cropping.
    fiji_attrs : FijiAttrs
        Attributes related to Fiji application settings.

    Returns
    -------
    DeltaTranslation
        The calculated shifts (dx, dy).
    """

    with tempfile.TemporaryDirectory() as temp_dir:

        # Define the destination path for the temporary image
        temp_reference_image = os.path.join(
            temp_dir, os.path.basename(reference_image)
        )
        temp_moving_image = os.path.join(
            temp_dir, os.path.basename(moving_image)
        )

        # Read and convert images to grayscale
        ref_img = Image.open(reference_image)  # .convert("L")
        mov_img = Image.open(moving_image)  # .convert("L")

        # Crop the images if use_subarea is set
        if alignment_attrs.use_subarea:
            # Define the cropping box
            crop_box = (
                alignment_attrs.subarea_x,
                alignment_attrs.subarea_y,
                alignment_attrs.subarea_x + alignment_attrs.subarea_width,
                alignment_attrs.subarea_y + alignment_attrs.subarea_height,
            )
            ref_img = ref_img.crop(crop_box)
            mov_img = mov_img.crop(crop_box)

            # adjust the contrast

            # Create a Contrast enhancer object
            enhancer_ref_img = ImageEnhance.Contrast(ref_img)
            enhancer_mov_img = ImageEnhance.Contrast(mov_img)

            # Enhance the contrast (e.g., by a factor of 3)
            ref_img = enhancer_ref_img.enhance(3)
            mov_img = enhancer_mov_img.enhance(3)

        # Save the processed images to the temporary directory
        ref_img.save(temp_reference_image)
        mov_img.save(temp_moving_image)

        # Get the width and height of the processed image
        height, width = ref_img.size

        # # Read the image using Pillow
        # ref_img = Image.open(reference_image)
        # mov_img = Image.open(moving_image)

        # # Convert to grayscale if the images are in color
        # if ref_img.mode == 'L' and mov_img.mode == 'L':
        #     pass  # Both images are already grayscale
        # elif ref_img.mode in ['RGB', 'RGBA'] and mov_img.mode in ['RGB', 'RGBA']:
        #     ref_img = ref_img.convert('L')  # Convert to grayscale
        #     mov_img = mov_img.convert('L')  # Convert to grayscale
        # else:
        #     raise TypeError(f'Images must be 3-channel color or gray, not shaped as {ref_img.size} and {mov_img.size}')

        # # Save the temporary grayscale images if needed
        # ref_img.save(temp_reference_image)
        # mov_img.save(temp_moving_image)

        # if alignment_attrs.use_subarea:
        #     # Crop the image using NumPy slicing
        #     cropped_ref_img = ref_img[
        #         alignment_attrs.subarea_y : alignment_attrs.subarea_y
        #         + alignment_attrs.subarea_height,
        #         alignment_attrs.subarea_x : alignment_attrs.subarea_x
        #         + alignment_attrs.subarea_width,
        #     ]

        #     cropped_mov_img = mov_img[
        #         alignment_attrs.subarea_y : alignment_attrs.subarea_y
        #         + alignment_attrs.subarea_height,
        #         alignment_attrs.subarea_x : alignment_attrs.subarea_x
        #         + alignment_attrs.subarea_width,
        #     ]

        #     # save the cropped image to the temporary directory
        #     imageio.v3.imwrite(temp_reference_image, cropped_ref_img)
        #     imageio.v3.imwrite(temp_moving_image, cropped_mov_img)

        # else:
        #     # save the uncropped image to the temporary directory
        #     imageio.v3.imwrite(temp_reference_image, ref_img)
        #     imageio.v3.imwrite(temp_moving_image, mov_img)

        # # Get the width and height of the cropped image
        # height = imageio.v3.imread(temp_reference_image).shape[0]
        # width = imageio.v3.imread(temp_reference_image).shape[1]

        # define the location of the shift file
        temp_calculated_shifts = os.path.join(
            temp_dir, "calculated_shifts.csv"
        )

        # modify file path for the macro if using windows
        if platform == "win32":
            temp_reference_image = temp_reference_image.replace(os.sep, "/")
            temp_moving_image = temp_moving_image.replace(os.sep, "/")
            temp_calculated_shifts = temp_calculated_shifts.replace(
                os.sep, "/"
            )

        # write ImageJ macro
        turbo_reg_macro = os.path.join(temp_dir, "turbo_reg_macro.ijm")
        Path(turbo_reg_macro).touch()  # Creates an empty file
        with open(turbo_reg_macro, "w", encoding="utf-8") as f:
            f.write(f'target = "{temp_reference_image}"; //reference\n')
            f.write(f'source = "{temp_moving_image}"; //thing to be aligned\n')
            f.write(f"width = {width};\n")
            f.write(f"height = {height};\n")

            f.write('run("TurboReg ","-align " +\n')
            f.write('\t\t\t\t"-file " + source  + \n')
            f.write(
                '\t\t\t\t" 0 0 " + width + " " + height + //cropping (start x, start y, width, height)\n'
            )
            f.write('\t\t\t\t" -file " + target + \n')
            f.write(
                '\t\t\t\t" 0 0 " + width + " " + height + //cropping (start x, start y, width, height)\n'
            )
            f.write('\t\t\t\t" -translation " +\n')
            f.write(
                '\t\t\t\twidth/2 + " " + height/2 + " " + //landmark x,y (center of image)\n'
            )
            f.write(
                '\t\t\t\twidth/2 + " " + height/2 + 		 //landmark x,y (center of image)\n'
            )
            f.write('\t\t\t\t" -hideOutput");\n')
            # f.write('\t\t\t\t" -showOutput");\n')

            # interpret result
            f.write('sourceX0 = getResult("sourceX", 0);\n')
            f.write('sourceY0 = getResult("sourceY", 0);\n')
            f.write('targetX0 = getResult("targetX", 0);\n')
            f.write('targetY0 = getResult("targetY", 0);\n')
            f.write("dx = targetX0 - sourceX0;\n")
            f.write("dy = targetY0 - sourceY0;\n")

            # write out teh result
            f.write(f'f = File.open("{temp_calculated_shifts}");\n')
            f.write('print(f, d2s(dx,2) + ", " + d2s(dy,3));\n')
            f.close()

        fiji_app = fiji_attrs.fiji_app
        ## managing memory in headless
        # https://forum.image.sc/t/set-memory-from-command-line-upon-startup-headless/8668/4
        run_macro = Popen(
            [fiji_app, "--headless", "-macro", turbo_reg_macro],
            shell=False,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        stream_data = run_macro.communicate()[0]
        if "error" in str(stream_data).lower():
            raise OSError(
                f"Stitching failed on current slice, Fiji output:\n{stream_data}"
            )

        ##convert to numpy file
        shift_data = np.genfromtxt(
            temp_calculated_shifts, delimiter=","
        )  # [:, 2:].astype(int)
        delta_translation = DeltaTranslation(
            dx=shift_data[0], dy=shift_data[1]
        )

        return delta_translation


# def apply_all_shifts_deprecated(image_dir: Path, shifts_path: Path):
#     """
#     to come
#     """

#     image_file_list = ut.sorted_files(image_dir, 'tif')

#     calculated_shifts = np.load(shifts_path)
#     c_shifts = np.cumsum(calculated_shifts,axis=0)

#     # check sizes make sense
#     if len(image_file_list)-1 != len(calculated_shifts):
#         raise ValueError(f"Found {len(image_file_list)} files, expected {len(calculated_shifts)} from calculated shifts")


#     save_dir = image_dir.parent.joinpath(f"aligned_xy_{image_dir.stem}")
#     save_dir.mkdir(exist_ok=True)

#     # copy first image
#     shutil.copyfile(image_file_list[0],save_dir.joinpath(image_file_list[0].name))

#     for i, each_shift in enumerate(c_shifts, start=1):
#         # convert to a DeltaTranslation
#         dt = DeltaTranslation(dx=each_shift[0],dy=each_shift[1])
#         orig_img = np.array(Image.open(image_file_list[i]))
#         shifted_orig_img = apply_shift(orig_img,dt)
#         new_img = Image.fromarray(shifted_orig_img.astype(orig_img.dtype))
#         new_img.save(save_dir.joinpath(image_file_list[i].name))


@app.command(
    cls=CustomCLICommand,
    name="apply_all_shifts",
    short_help="apply known shifts",
)
def apply_all_shifts(
    image_dir: Path, shifts_path: Path, scale_factor: float = 1.0
):
    """
    to come
    """

    # cast to Path object if not already a Path object
    if not isinstance(image_dir, Path):
        image_dir = Path(image_dir)
    if not isinstance(shifts_path, Path):
        shifts_path = Path(shifts_path)

    image_file_list = ut.sorted_files(image_dir, "tif")

    calculated_shifts = np.load(shifts_path)
    c_shifts = np.cumsum(calculated_shifts, axis=0)

    # check sizes make sense
    # if len(image_file_list)-1 != len(calculated_shifts):
    #     raise ValueError(f"Found {len(image_file_list)} files, expected {len(calculated_shifts)} from calculated shifts")
    if len(image_file_list) % (len(calculated_shifts) + 1) != 0:
        raise ValueError(
            f"Oops, found {len(image_file_list)} files, and {len(calculated_shifts)} from calculated shifts"
        )
    else:
        print(
            f"Found {len(image_file_list)} files, and {len(calculated_shifts)} from calculated shifts"
        )

    save_dir = image_dir.parent.joinpath(f"aligned_xy_{image_dir.stem}")
    save_dir.mkdir(exist_ok=True)

    # copy first image
    # first_imgs = image_file_list[0::13]
    # for each_fist_img in first_imgs:
    #     shutil.copyfile(each_fist_img,save_dir.joinpath(each_fist_img.name))

    # Use list comprehension to exclude every 1st, 14th, 27th, etc., element
    # filtered_list = [item for index, item in enumerate(image_file_list, start=1) if (index - 1) % 13 != 0]

    shutil.copyfile(
        image_file_list[0], save_dir.joinpath(image_file_list[0].name)
    )

    for shift_idx, each_img in enumerate(image_file_list[1:]):

        shift = c_shifts[shift_idx]

        # convert to a DeltaTranslation
        dt = DeltaTranslation(dx=shift[0], dy=shift[1])
        orig_img = np.array(Image.open(each_img))
        shifted_orig_img = apply_shift(orig_img, dt, scale_factor)
        new_img = Image.fromarray(shifted_orig_img.astype(orig_img.dtype))
        new_img.save(save_dir.joinpath(each_img.name))

        print(f"\tapplied shift to {each_img.name}")


def list_to_numpy_array(delta_translations):
    # Extract dx and dy from each DeltaTranslation object and create a list of tuples
    data = [(dt.dx, dt.dy) for dt in delta_translations]
    # Convert the list of tuples into a NumPy array
    return np.array(data)


def numpy_array_to_list(numpy_array):
    # Create a list of DeltaTranslation objects from the NumPy array
    return [DeltaTranslation(dx=row[0], dy=row[1]) for row in numpy_array]


# def run_aligner(configfile: Path):
#     """align image stacks using the user defined yaml

#     Args:
#         configfile (Path): _description_
#     """

#     # cast to Path object if not already a Path object
#     if not isinstance(configfile, Path):
#         configfile = Path(configfile)

#     # read yaml content
#     config_dict = ut.read_config(configfile)

#     user_aligner_attrs = create_aligner_attrs(config_dict)
#     general_attrs = precon3d.factory.create_general_attrs(config_dict)
#     ref_channel = str(config_dict["ref_channel"])

#     # check the ref channel is a folder of the general_attrs:input_directory


# def align_scene_using_ref(config: Dict):
#     """
#     to come
#     """


def align_image_stack_turboreg(
    image_dir: Path, alignment_attrs: AlignmentAttrs, fiji_attrs: FijiAttrs
):
    """_summary_"""

    # align
    if image_dir.is_dir():

        processed_imagelist = ut.sorted_files(image_dir, ".tif")

        print(f"Aligning {len(processed_imagelist)} images in {image_dir}")

        optimal_shifts = [
            calc_shifts_turboreg(
                ref_image, mov_image, alignment_attrs, fiji_attrs
            )
            for ref_image, mov_image in zip(
                processed_imagelist, processed_imagelist[1:]
            )
        ]

        calculated_shifts = image_dir.parent.joinpath(
            f"{image_dir.name}_optimized_shifts_turboreg.npy"
        )
        np.save(calculated_shifts, list_to_numpy_array(optimal_shifts))

        # apply shifts
        apply_all_shifts(image_dir, calculated_shifts)

    print("Alignment done.")


def align_image_stack_sift(image_dir: Path, alignment_attrs: AlignmentAttrs):
    """_summary_"""

    # align
    if image_dir.is_dir():

        processed_imagelist = ut.sorted_files(image_dir, ".tif")

        print(f"Aligning {len(processed_imagelist)} images in {image_dir}")

        optimal_shifts = [
            calc_shifts_using_keypoints(ref_image, mov_image, alignment_attrs)
            for ref_image, mov_image in zip(
                processed_imagelist, processed_imagelist[1:]
            )
        ]

        calculated_shifts = image_dir.parent.joinpath(
            f"{image_dir.name}_optimized_shifts_sift.npy"
        )
        np.save(calculated_shifts, list_to_numpy_array(optimal_shifts))

        # apply shifts
        apply_all_shifts(image_dir, calculated_shifts)

    print("Alignment done.")


def align_image_stack_phasecorr(
    image_dir: Path, alignment_attrs: AlignmentAttrs
):
    """_summary_"""

    # align
    if image_dir.is_dir():

        processed_imagelist = ut.sorted_files(image_dir, ".tif")

        print(f"Aligning {len(processed_imagelist)} images in {image_dir}")

        optimal_shifts = [
            calc_shifts_phase_corr(ref_image, mov_image, alignment_attrs)
            for ref_image, mov_image in zip(
                processed_imagelist, processed_imagelist[1:]
            )
        ]

        calculated_shifts = image_dir.parent.joinpath(
            f"{image_dir.name}_optimized_shifts_phasecorr.npy"
        )
        np.save(calculated_shifts, list_to_numpy_array(optimal_shifts))

        # apply shifts
        apply_all_shifts(image_dir, calculated_shifts)

    print("Alignment done.")


@app.command(
    cls=CustomCLICommand,
    name="process_images",
    short_help="align images from user defined config",
)
def process_images(config_filepath: Path):
    """
    align images from user defined config
    """

    # Read the dict in the config file
    config_dict = ut.read_config(config_filepath)
    alignment_attrs = create_aligner_attrs(config_dict)
    general_attrs = gen_types.create_general_attrs(config_dict)

    if not ut.valid_enum_entry(
        alignment_attrs.registration_method, RegistrationMethod
    ):
        raise KeyError(
            f"Unsupported registration method {alignment_attrs.registration_method}, supported types include {[i.value for i in RegistrationMethod]}"
        )
    print(f"Using: {alignment_attrs.registration_method} method to align")

    tic = time.perf_counter()

    ut.current_date_and_time()
    print("*** Starting precon3d.aligner *** \n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description="Aligning...", total=None)
        # parse and run prep
        if alignment_attrs.registration_method == "turboreg":
            fiji_attrs = FijiAttrs(**config_dict["fiji_attrs"])
            align_image_stack_turboreg(
                general_attrs.input_directory, alignment_attrs, fiji_attrs
            )
        elif alignment_attrs.registration_method == "sift":
            align_image_stack_sift(
                general_attrs.input_directory, alignment_attrs
            )
        elif alignment_attrs.registration_method == "phasecorr":
            align_image_stack_phasecorr(
                general_attrs.input_directory, alignment_attrs
            )
        progress.update(task, completed=True)  # Mark the task as completed

    print("Done!\n")

    toc = time.perf_counter()

    print(
        f"""Data (pre)paration/(con)truction in 3D took
        {toc - tic:0.2f} seconds
        {(toc - tic)/60:0.2f} minutes
        {(toc - tic)/3600:0.2f} hours"""
    )


@app.callback()
def callback():
    """
    precon3d.aligner aligns image stacks.
    """
