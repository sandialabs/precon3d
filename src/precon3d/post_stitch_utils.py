import numpy.typing as npt
from typing import NamedTuple
from pathlib import Path
import skimage
import numpy as np
import time
import typer
import imageio
import scipy
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import tifffile
import re
from itertools import islice
from itertools import pairwise

from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

import precon3d.utility as ut

# pylint: disable=wildcard-import
from precon3d._my_typer_cli import CustomCLIGroup, CustomCLICommand

# CLI
app = typer.Typer(
    cls=CustomCLIGroup,
    no_args_is_help=True,
    short_help="Prepare polarized images",
)


def rotate_image_deg(image_path: Path, output_dir: Path):
    """Use XXXdeg in filename to rotate images

    Args:
        image_path (Path): The path to the image file.

    Returns:
        Image: The rotated image.
    """

    # Convert Path to string if not already, to use regex
    fname_str = str(image_path.stem)

    # Extract the degree information from the filename using regex
    match = re.search(r"(\d{1,3})deg", fname_str)
    if match:
        degrees = int(match.group(1))
    else:
        raise ValueError("No degree information found in filename.")

    # Load the image
    with Image.open(image_path) as img:
        print(f"rotating {image_path.stem} ccw {degrees} degrees")

        # Rotate the image. The expand flag is used to resize the output to fit the new orientation
        rotated_img = img.rotate(degrees, expand=False)

        # save the rotated image or return it
        save_path = output_dir.joinpath(f"rotated_{image_path.name}")

        rotated_img.save(save_path)


def rotate_image_deg_imageio(image_path: Path, degrees):
    # Read the image
    img = imageio.v3.imread(image_path)

    rot_img = scipy.ndimage.rotate(img, degrees, reshape=True, mode="constant")

    # # Calculate the rotation in radians
    # radians = np.deg2rad(degrees)

    # # Perform the rotation
    # rotated_img = imageio.core.functions.rotate(img, radians, resize=True)
    fpath_npy = image_path.parent.joinpath(f"rotated_{image_path.stem}.npy")
    # fpath_tif = image_path.parent.joinpath(f"rotated_{image_path.name}")

    # Save the rotated image
    # imageio.v3.imwrite(fpath_tif, rot_img)
    np.save(fpath_npy, rot_img)


def rotate_and_save_tiff(
    image_path, output_path, angle, clip_to_uint8=False, reshape=True
):
    """
    Rotate a high bit-depth TIFF image and save the result.

    Args:
    - image_path (str): Path to the input TIFF image.
    - output_path (str): Path where the rotated image will be saved.
    - angle (float): Rotation angle in degrees. Positive for counter-clockwise rotation.
    - reshape (bool): Whether to expand the output image's shape to fit the entire rotated image.
    """
    # Read the image
    with tifffile.TiffFile(image_path) as tif:
        img_array = tif.asarray()

    # Rotate the image
    rotated_img_array = scipy.ndimage.rotate(
        img_array, angle, reshape=reshape, order=1, mode="constant"
    )

    # Find the bounding box of non-zero values
    non_zero_coords = np.argwhere(
        rotated_img_array > 0
    )  # Assuming non-zero values are the content
    y_min, x_min = non_zero_coords.min(axis=0)
    y_max, x_max = (
        non_zero_coords.max(axis=0) + 1
    )  # +1 because slice end index is exclusive

    # Crop the image
    cropped_img_array = rotated_img_array[y_min:y_max, x_min:x_max]

    # Clip values to the 0-255 range and convert to uint8
    if clip_to_uint8:
        cropped_img_array = np.clip(cropped_img_array, 0, 255).astype(np.uint8)

    # Save the rotated image
    tifffile.imwrite(output_path, cropped_img_array, photometric="minisblack")


@app.command(name="rotate_images_by_filename")
def rotate_images_by_filename(input_dir: Path, output_dir: Path):
    """
    rotate images in a directory using the XXXdeg in the filename
    """

    mosaic_image_file_list = ut.sorted_files(input_dir, "tif")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for each_image in mosaic_image_file_list:

        # rotate_image_deg(each_image)
        # Convert Path to string if not already, to use regex
        fname_str = str(each_image.stem)
        # print(f"Rotating {fname_str}")

        # Extract the degree information from the filename using regex
        match = re.search(r"(\d{1,3})deg", fname_str)
        if match:
            degrees = int(match.group(1))
        else:
            raise ValueError("No degree information found in filename.")

        print(f"Applying {degrees} deg rotation to {fname_str}")

        # output_path = each_image.parent.joinpath("Rotated",f"rotated_{each_image.name}")
        output_filepath = output_dir.joinpath(f"rotated_{each_image.name}")

        # rotate_image_deg_imageio(each_image, degrees)
        rotate_and_save_tiff(
            each_image, output_path=output_filepath, angle=degrees
        )


def rotate_and_save_single_image_by_filename(
    input_image: Path, output_dir: Path
):
    """
    rotate images in a directory using the XXXdeg in the filename
    """

    # rotate_image_deg(each_image)
    # Convert Path to string if not already, to use regex
    fname_str = str(input_image.stem)
    # print(f"Rotating {input_image.name}")

    # Extract the degree information from the filename using regex
    match = re.search(r"(\d{1,3})deg", fname_str)
    if match:
        degrees = int(match.group(1))
    else:
        raise ValueError("No degree information found in filename.")

    print(f"Applying {degrees} deg rotation to {input_image.name}")

    # output_path = each_image.parent.joinpath("Rotated",f"rotated_{each_image.name}")
    output_filepath = output_dir.joinpath(f"rotated_{input_image.name}")

    # rotate_image_deg_imageio(each_image, degrees)
    rotate_and_save_tiff(
        input_image, output_path=output_filepath, angle=degrees
    )


def rotate_save_pad_single_image_by_filename(
    input_image: Path, max_image_extent: np.ndarray, output_dir: Path
):
    """
    rotate images in a directory using the XXXdeg in the filename
    """

    # rotate_image_deg(each_image)
    # Convert Path to string if not already, to use regex
    fname_str = str(input_image.stem)
    # print(f"Rotating {input_image.name}")

    # Extract the degree information from the filename using regex
    match = re.search(r"(\d{1,3})deg", fname_str)
    if match:
        degrees = int(match.group(1))
    else:
        raise ValueError("No degree information found in filename.")

    # output_path = each_image.parent.joinpath("Rotated",f"rotated_{each_image.name}")
    output_filepath = output_dir.joinpath(
        f"{input_image.stem}_pad_rot{input_image.suffix}"
    )

    # rotate_image_deg_imageio(each_image, degrees)
    # rotate_and_save_tiff(input_image, output_path=output_filepath,angle=degrees)
    # Read the image
    with tifffile.TiffFile(input_image) as tif:
        img_array = tif.asarray()

    # Rotate the image
    rotated_img_array = scipy.ndimage.rotate(
        img_array, degrees, reshape=True, order=1, mode="constant"
    )

    # Ensure the image does not exceed the maximum dimensions
    cropped_img_array = rotated_img_array[
        : min(rotated_img_array.shape[0], max_image_extent[0]),
        : min(rotated_img_array.shape[1], max_image_extent[1]),
    ]

    # # Find the bounding box of non-zero values
    # non_zero_coords = np.argwhere(rotated_img_array > 0)  # Assuming non-zero values are the content
    # y_min, x_min = non_zero_coords.min(axis=0)
    # y_max, x_max = non_zero_coords.max(axis=0) + 1  # +1 because slice end index is exclusive

    # # Crop the image
    # cropped_img_array = rotated_img_array[y_min:y_max, x_min:x_max]

    # pad_height = max_image_extent[0] - cropped_img_array.shape[0]
    # pad_width = max_image_extent[1] - cropped_img_array.shape[1]

    # # Calculate padding for height and width
    # pad_top = pad_height // 2
    # pad_bottom = pad_height - pad_top
    # pad_left = pad_width // 2
    # pad_right = pad_width - pad_left

    # Calculate necessary padding
    pad_height = max(0, max_image_extent[0] - cropped_img_array.shape[0])
    pad_width = max(0, max_image_extent[1] - cropped_img_array.shape[1])

    # Calculate padding for height and width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Ensure padding does not exceed the image dimensions
    pad_top = min(pad_top, max_image_extent[0])
    pad_bottom = min(pad_bottom, max_image_extent[0] - pad_top)
    pad_left = min(pad_left, max_image_extent[1])
    pad_right = min(pad_right, max_image_extent[1] - pad_left)

    # Apply padding
    padded_image = np.pad(
        cropped_img_array,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )

    # Save the padded image
    skimage.io.imsave(output_filepath, padded_image, check_contrast=False)

    print(
        f"Saved {output_filepath.name} after a {degrees} deg rotation and padded to {max_image_extent}"
    )


def pad_images_to_max_extent(input_dir: Path, output_dir: Path):
    """Pad all images to the same size, the size is the max extent found in the images

    Args:
        input_dir (Path): Directory containing input images.
        output_dir (Path): Directory where padded images will be saved.
    """
    filepaths_list = ut.sorted_files(
        directory=input_dir, file_extension=".tif"
    )
    n_images = len(filepaths_list)
    print(f"Found {n_images} images")

    # Collect all images and their shapes
    all_images_shape = []
    for item in filepaths_list:
        each_image = skimage.io.imread(item)
        all_images_shape.append(each_image.shape)

    # Determine the maximum extent in each dimension
    max_image_extent = np.max(all_images_shape, axis=0)
    print(f"\tPadding images to {max_image_extent} pixels")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pad and save images
    for each_image_path in filepaths_list:
        each_image = skimage.io.imread(each_image_path)
        pad_height = max_image_extent[0] - each_image.shape[0]
        pad_width = max_image_extent[1] - each_image.shape[1]

        # Calculate padding for height and width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply padding
        padded_image = np.pad(
            each_image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

        # Save the padded image
        tif_out_path = output_dir.joinpath(each_image_path.name)
        skimage.io.imsave(tif_out_path, padded_image, check_contrast=False)

    # print("Padding and saving completed.")


def image_max_extent(filepaths_list: list[Path]):
    """
    Retrieve the maximum image dimensions from a list of file paths.

    Leverages PIL Image package for quick access to image width and height.
    """
    n_images = len(filepaths_list)
    print(f"Found {n_images} images")

    def get_image_size(filepath):
        with Image.open(filepath) as img:
            return img.size  # Returns (width, height)

    # Use ThreadPoolExecutor to parallelize the retrieval of image sizes
    with ThreadPoolExecutor() as executor:
        all_image_sizes = list(executor.map(get_image_size, filepaths_list))

    # Convert (width, height) to (height, width) and find max dimensions
    all_image_sizes = [(height, width) for width, height in all_image_sizes]
    max_image_extent = np.max(all_image_sizes, axis=0)
    print(f"Padding images to {max_image_extent} pixels")

    return max_image_extent


def pad_and_save_image_equal(
    filename: Path, max_image_extent: np.ndarray, output_dir: Path
):
    """To come"""

    input_image = skimage.io.imread(filename)
    pad_height = max_image_extent[0] - input_image.shape[0]
    pad_width = max_image_extent[1] - input_image.shape[1]

    # Calculate padding for height and width
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding
    if input_image.ndim == 3 and input_image.shape[2] == 3:
        padded_image = np.pad(
            input_image,
            (
                (pad_top, pad_bottom),
                (pad_left, pad_right),
                (0, 0),
            ),  # No padding for the channel dimension
            mode="constant",
            constant_values=0,
        )  # grayscale
    else:
        padded_image = np.pad(
            input_image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )

    # Save the padded image
    output_filename = f"padded_{filename.name}"
    tif_out_path = output_dir.joinpath(output_filename)
    skimage.io.imsave(tif_out_path, padded_image, check_contrast=False)


def filter_and_sort_file_paths(
    file_paths: list[Path], keyword: str
) -> list[Path]:
    """
    Filters a list of file paths to include only those that contain a specific directory keyword and sorts the result.

    Args:
    file_paths (list of str): The list of file paths to filter.
    keyword (str): The directory keyword to filter by.

    Returns:
    list of Path: A sorted list containing only the file paths that include the keyword.
    """
    # Convert strings to Path objects and filter
    filtered_paths = [
        Path(path) for path in file_paths if keyword in Path(path).parts
    ]

    # Sort the filtered paths
    sorted_filtered_paths = sorted(filtered_paths, key=lambda x: x.as_posix())

    return sorted_filtered_paths


def append_deg_suffix_and_rename(filenames: list[Path], directory_path: Path):
    # Define the degree suffixes in a list
    deg_suffixes = [
        "000deg",
        "015deg",
        "030deg",
        "045deg",
        "060deg",
        "075deg",
        "090deg",
        "105deg",
        "120deg",
        "135deg",
        "150deg",
        "165deg",
        "180deg",
    ]

    # Process each filename
    updated_filenames = []
    for i, filename in enumerate(filenames):
        # Calculate the modulo 13 of the index (i + 1) to match the desired mapping
        mod_value = i % 13
        suffix = deg_suffixes[mod_value]

        # Remove existing duplicate suffixes
        stem = filename.stem
        while stem.endswith(suffix):
            stem = stem[
                : -(len(suffix) + 1)
            ]  # Remove the suffix and underscore

        # Append the corresponding degree suffix to the cleaned filename
        new_filename = f"{stem}_{suffix}{filename.suffix}"
        updated_filenames.append(new_filename)

        # Full path for old and new filenames
        old_file_path = directory_path / filename
        new_file_path = directory_path / new_filename

        # Rename the file
        old_file_path.rename(new_file_path)

    return updated_filenames


@app.command(
    cls=CustomCLICommand,
    name="rotate_images_by_filename_parallel",
    short_help="rotate images by filename in parallel",
)
def rotate_images_by_filename_parallel(
    input_dir: Path, output_dir: Path, ncores: int = 4
):
    """
    rotate images in a directory using the XXXdeg in the filename
    """

    image_file_list = ut.sorted_files(input_dir, "tif")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # rotate and save images, parallel
    # Parallel(n_jobs=ncores)(delayed(rotate_and_save_single_image_by_filename)(each_image_path, output_dir) for each_image_path in image_file_list)
    # Parallel(n_jobs=ncores)(delayed(rotate_image_deg)(each_image_path, output_dir) for each_image_path in image_file_list)
    Parallel(n_jobs=ncores)(
        delayed(rotate_and_save_single_image_by_filename)(
            each_image_path, output_dir
        )
        for each_image_path in image_file_list
    )

    print("Image rotations completed.")


@app.command(
    cls=CustomCLICommand,
    name="pad_images_to_max_extent_parallel",
    short_help="pad images to maximum extent in parallel",
)
def pad_images_to_max_extent_parallel(
    input_dir: Path, output_dir: Path, ncores: int = 4
):
    """Pad all images to the same size, the size is the max extent determined from all the images

    Args:
        input_dir (Path): Directory containing input images.
        output_dir (Path): Directory where padded images will be saved.
    """
    # filepaths_list = ut.sorted_files(
    #     directory=input_dir, file_extension=".tif"
    # )

    filepaths_list = ut.sorted_files(
        directory=input_dir, file_extension=".tif"
    )

    max_image_extent = image_max_extent(filepaths_list)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pad and save images, parallel

    Parallel(n_jobs=ncores)(
        delayed(pad_and_save_image_equal)(
            each_image_path, max_image_extent, output_dir
        )
        for each_image_path in filepaths_list
    )

    print("Padding and saving completed.")


def combined_image_padding_and_rotation_parallel(
    input_dir: Path,
    max_image_extent: np.ndarray,
    output_dir: Path,
    ncores: int = 4,
):
    """To come"""

    filepaths_list = ut.sorted_files(
        directory=input_dir, file_extension=".tif"
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pad and save images, parallel
    Parallel(n_jobs=ncores)(
        delayed(rotate_save_pad_single_image_by_filename)(
            each_image_path, max_image_extent, output_dir
        )
        for each_image_path in filepaths_list
    )

    print("Padding, rotation, and saving completed.")


@app.callback()
def callback():
    """
    precon3d.pol_prep can prepare polarized images by padding and rotating in a parallel operation. Colorization also available.
    """


import matplotlib.pyplot as plt


def visualize_rgb(rgb_image: npt.NDArray):
    """plot the images side by side"""

    H, W, _ = rgb_image.shape

    # 1) Normalize each component map to [0,1] for display
    C_norm = np.empty_like(rgb_image, dtype=float)
    for i in range(3):
        comp = rgb_image[:, :, i]
        lo, hi = comp.min(), comp.max()
        if hi > lo:
            C_norm[:, :, i] = (comp - lo) / (hi - lo)
        else:
            C_norm[:, :, i] = comp

    # 2) Build the RGB composite
    rgb_composite = C_norm  # channel 0→R, 1→G, 2→B

    # 3) Build three “pure‐R/G/B” versions of each component
    colored = []
    for i in range(3):
        # zero‐plane
        zero = np.zeros((H, W), dtype=float)
        if i == 0:
            # comp 0 in red channel
            rgb = np.stack([C_norm[:, :, 0], zero, zero], axis=-1)
        elif i == 1:
            # comp 1 in green
            rgb = np.stack([zero, C_norm[:, :, 1], zero], axis=-1)
        else:
            # comp 2 in blue
            rgb = np.stack([zero, zero, C_norm[:, :, 2]], axis=-1)
        colored.append(rgb)

    # 4) Plot the 1×4 montage
    fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 4))

    # Panel 1: RGB composite
    ax = axes[0]
    ax.imshow(rgb_composite, vmin=0, vmax=1)
    ax.set_title("Composite (RGB)")
    ax.axis("off")

    # Panels 2–4: individual comps in R/G/B
    titles = ["Component 1 → Red", "Component 2 → Green", "Component 3 → Blue"]
    for ax, img, title in zip(axes[1:], colored, titles):
        ax.imshow(img, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_spectra(spectra: npt.NDArray):
    """plot the spectra, color as RGB"""

    n_channels = spectra.shape[0]
    channels = np.arange(n_channels)  # or your actual channel‐wavelength array

    plt.figure(figsize=(6, 4))
    plt.plot(channels, spectra[:, 0], color="red", lw=2, label="Comp 1")
    plt.plot(channels, spectra[:, 1], color="green", lw=2, label="Comp 2")
    plt.plot(channels, spectra[:, 2], color="blue", lw=2, label="Comp 3")

    plt.xlabel("Channel")
    plt.ylabel("Intensity")
    plt.title("Recovered Spectra")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import os
from skimage import io
from skimage.util import img_as_float


def load_pol_images(
    img_dir: Path, file_extension: str = ".tif"
) -> npt.NDArray:
    """load and reshape images stack data from directory"""

    # Create an ImageCollection
    # image_collection = io.ImageCollection(os.path.join(img_dir, '*.tif'))  # Adjust the file extension as needed
    # image_collection = io.ImageCollection(ut.sorted_files(img_dir, file_extension))

    # Optionally, convert images to float format
    # image_collection_float = [img_as_float(image) for image in image_collection]

    # Example: Print the number of images loaded and their shapes
    # print(f'Number of images loaded: {len(image_collection)}')
    # for i, img in enumerate(image_collection_float):
    #     print(f'Image {i}: shape {img.shape}')

    image_stack_paths = ut.sorted_files(img_dir, file_extension)

    # Load the shading images into a list, using imageio for image reading
    imagestack = [
        imageio.v3.imread(each_image) for each_image in image_stack_paths
    ]

    # Reshape the loaded images into a 2D array where each row represents a flattened image
    reshaped_images = np.array([img_as_float(img) for img in imagestack])

    img = np.asarray(reshaped_images)

    img = np.transpose(img, (1, 2, 0))

    return img


from sklearn.decomposition import PCA, NMF


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """
    Perform varimax (orthogonal) rotation on the loadings matrix Phi.
    Input:
      Phi   = (p, k) loadings
      gamma = 1.0 for classic varimax
      q     = max iterations
      tol   = convergence tolerance
    Returns:
      rotated loadings, rotation matrix R
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        Lambda = Phi.dot(R)
        # the varimax update
        u, s, vh = np.linalg.svd(
            Phi.T.dot(
                np.power(Lambda, 3)
                - (gamma / p) * Lambda.dot(np.diag(np.sum(Lambda**2, axis=0)))
            )
        )
        R = u.dot(vh)
        d_new = np.sum(s)
        if d != 0 and (d_new - d) < tol:
            break
        d = d_new
    return Phi.dot(R), R


@app.command(
    cls=CustomCLICommand,
    name="colorize_pca_varimax",
    short_help="Colorize with PCA",
)
def colorize_pca_varimax(img_dir: Path):
    """colorize the polarized iamge stack with pca and varimax"""

    # cast to Path object if not already a Path object
    if not isinstance(img_dir, Path):
        img_dir = Path(img_dir)

    img = load_pol_images(img_dir)

    H, W, C = img.shape
    # reshape into (n_pixels, n_channels)
    X = img.reshape(-1, 13)  # shape = (H*W, 13)

    n_components = 3  # for RGB

    # 3a. fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)  # (H*W, 3) principal component scores
    loadings = pca.components_.T  # (13, 3) unrotated loadings

    # 3b. rotate loadings
    rot_loadings, R = varimax(loadings)  # (13, 3), rotation R
    # rotated scores
    rot_scores = X.dot(rot_loadings)  # (H*W, 3)

    # reshape back to images
    score_imgs = rot_scores.reshape(H, W, n_components)

    visualize_rgb(score_imgs)
    visualize_spectra(rot_loadings)

    # Normalize before saving
    score_imgs_norm = np.empty_like(score_imgs, dtype=float)
    for i in range(3):
        comp = score_imgs[:, :, i]
        lo, hi = comp.min(), comp.max()
        if hi > lo:
            score_imgs_norm[:, :, i] = (comp - lo) / (hi - lo)
        else:
            score_imgs_norm[:, :, i] = comp

    # io.imsave(img_dir.parent.joinpath('rgb_pca_image.png'), (score_imgs * 255).astype(np.uint8), check_contrast=False)
    imageio.v3.imwrite(
        img_dir.parent.joinpath(f"{img_dir.name}_rgb_pca_image.tif"),
        (score_imgs_norm * 255).astype(np.uint8),
    )


def mcr_als(X, C_init, S_init, max_iter=100, tol=1e-6):
    """
    Basic non‐negative MCR‐ALS.
    X      : (n_samples, n_vars)
    C_init : (n_samples, n_comp)
    S_init : (n_vars,    n_comp)
    """
    C = C_init.copy()
    S = S_init.copy()
    for it in range(max_iter):
        S_old = S.copy()

        # 1) update spectra:   X = C @ S.T  ⇒  S.T = pinv(C) @ X
        S = (np.linalg.pinv(C) @ X).T
        S[S < 0] = 0

        # 2) update concentrations:  X = C @ S.T  ⇒  C = X @ pinv(S.T)
        C = X @ np.linalg.pinv(S.T)
        C[C < 0] = 0

        # 3) convergence on S
        delta = np.linalg.norm(S - S_old) / np.linalg.norm(S_old)
        if delta < tol:
            print(f"MCR‐ALS converged in {it+1} iter (Δ={delta:.2e})")
            break

    return C, S


def mcr_als_lstsq(X, C_init, S_init, max_iter=100, tol=1e-6):
    C = C_init.copy()
    S = S_init.copy()
    for it in range(max_iter):
        S_old = S.copy()

        # 1) solve C @ S.T = X  ⇒  S.T = lstsq(C, X)
        S = np.linalg.lstsq(C, X, rcond=None)[0].T
        S[S < 0] = 0

        # 2) solve S @ C.T = X.T ⇒  C.T = lstsq(S, X.T)
        C = np.linalg.lstsq(S, X.T, rcond=None)[0].T
        C[C < 0] = 0

        delta = np.linalg.norm(S - S_old) / np.linalg.norm(S_old)
        if delta < tol:
            print(f"MCR‐ALS converged in {it+1} iter (Δ={delta:.2e})")
            break

    return C, S


@app.command(
    cls=CustomCLICommand,
    name="colorize_mcr_als",
    short_help="Colorize with MCR",
)
def colorize_mcr_als(img_dir: Path):
    """colorize the polarized iamge stack with mcr and als"""

    # cast to Path object if not already a Path object
    if not isinstance(img_dir, Path):
        img_dir = Path(img_dir)

    img = load_pol_images(img_dir)

    H, W, C = img.shape
    # reshape into (n_pixels, n_channels)
    X = img.reshape(-1, 13)  # shape = (H*W, 13)

    n_components = 3  # for RGB

    # 4a. initialize with NMF (non‐negative matrix factorization)
    nmf = NMF(
        n_components=n_components,
        init="random",
        random_state=0,
        max_iter=500,
        tol=1e-4,
    )
    C_init = nmf.fit_transform(X)  # (H*W, 3) initial concentration profiles
    S_init = nmf.components_  # (3, 13) initial spectra

    # transpose S_init so spectra are columns (13, 3)
    S_init = S_init.T

    C_mcr, S_mcr = mcr_als(X, C_init, S_init)

    # reshape concentrations back to image form
    C_imgs = C_mcr.reshape(H, W, n_components)

    visualize_rgb(C_imgs)
    visualize_spectra(S_mcr)

    # io.imsave(img_dir.parent.joinpath('rgb_mcr_image.png'), (C_imgs * 255).astype(np.uint8), check_contrast=False)
    imageio.v3.imwrite(
        img_dir.parent.joinpath(f"{img_dir.name}_rgb_mcr_image.tif"),
        (C_imgs * 255).astype(np.uint8),
    )


def apply_loading(img_dir: Path, loadings: npt.NDArray):
    """use a known loading to color polarized images"""

    # cast to Path object if not already a Path object
    if not isinstance(img_dir, Path):
        img_dir = Path(img_dir)

    img = load_pol_images(img_dir)

    H, W, C = img.shape
    # reshape into (n_pixels, n_channels)
    X = img.reshape(-1, 13)  # shape = (H*W, 13)

    rng = np.random.default_rng()
    sampled_element = rng.choice(X, size=500, axis=0)

    n_components = 3  # for RGB

    # 4a. initialize with NMF (non‐negative matrix factorization)
    nmf = NMF(
        n_components=n_components,
        init="random",
        random_state=0,
        max_iter=500,
        tol=1e-4,
    )
    C_init = nmf.fit_transform(
        sampled_element
    )  # (H*W, 3) initial concentration profiles
    S_init = nmf.components_  # (3, 13) initial spectra

    # transpose S_init so spectra are columns (13, 3)
    S_init = S_init.T

    C_mcr, S_mcr = mcr_als(sampled_element, C_init, S_init)

    rot_scores = X.dot(S_mcr)  # (H*W, 3)
    score_imgs = rot_scores.reshape(H, W, n_components)
