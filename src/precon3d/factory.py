"""Use factory to generate the precon3d_types"""

from pathlib import Path

import precon3d.utility as ut

# pylint: disable=wildcard-import
from precon3d.custom_types import *


def create_general_attrs(data: Dict[str, Any]) -> GeneralAttrs:
    """Create GeneralAttrs from a dictionary."""
    return GeneralAttrs(
        input_directory=Path(data["general_attrs"]["input_directory"]),
        file_extension=data["general_attrs"]["file_extension"],
        output_directory=Path(data["general_attrs"]["output_directory"]),
    )


def create_stitching_parameters(data: Dict[str, Any]) -> StitchingParams:
    """
    Create an instance of StitchingParams from a dictionary of attributes.

    This function leverages Python's unpacking feature to pass dictionary keys
    as keyword arguments to the constructors of various NamedTuples. This
    approach allows for a concise and efficient way to create instances of
    the NamedTuples directly from a structured dictionary.

    Parameters
    ----------
    data : dict
        A dictionary containing the following keys:

        - 'fiji_attrs': A dictionary of attributes for Fiji processing.
        - 'general_attrs': A dictionary of general attributes for data processing.
        - 'normalization_attrs': A dictionary of normalization attributes.
        - 'stitching_attrs': A dictionary of stitching attributes.

        Each of these dictionaries should contain the necessary keys to
        instantiate their respective NamedTuples.

    Returns
    -------
    StitchingParams
        An instance of StitchingParams containing the constructed NamedTuple
        instances for Fiji attributes, general attributes, normalization
        attributes, and stitching attributes.

    Notes
    -----
    The unpacking pattern `**data["key"]` allows the function to extract
    the key-value pairs from the dictionary and pass them as keyword
    arguments to the NamedTuple constructors. This eliminates the need
    for explicitly specifying each argument, making the code cleaner
    and more maintainable.
    """

    fiji_attrs = FijiAttrs(**data["fiji_attrs"])
    # type hint filepaths/directories with Path
    general_attrs = ut.GeneralAttrs(
        input_directory=Path(data["general_attrs"]["input_directory"]),
        file_extension=data["general_attrs"]["file_extension"],
        output_directory=Path(data["general_attrs"]["output_directory"]),
    )

    channel_dir = data["normalization_attrs"][
        "channel_flatfields_parent_directory"
    ]
    if channel_dir is not None:
        channel_flatfields_parent_directory = Path(channel_dir)
    else:
        channel_flatfields_parent_directory = None

    normalization_attrs = NormalizationAttrs(
        use_flatfield=bool(data["normalization_attrs"]["use_flatfield"]),
        channel_flatfields_parent_directory=channel_flatfields_parent_directory,
        channel_flatfields_filename=data["normalization_attrs"][
            "channel_flatfields_filename"
        ],
    )
    # genaric types can be automatically unpacked
    stitching_attrs = StitchingAttrs(**data["stitching_attrs"])

    return StitchingParams(
        fiji_attrs=fiji_attrs,
        general_attrs=general_attrs,
        normalization_attrs=normalization_attrs,
        stitching_attrs=stitching_attrs,
    )


def create_shading_parameters(data: Dict[str, Any]) -> ShadingConfig:
    """
    Create a ShadingConfig from a dictionary.

    Parameters
    ----------
    data : Dict[str, Any]
        A dictionary containing the user defined configuration parameters.

    Returns
    -------
    ShadingConfig
        An instance of the ShadingConfig class initialized with the provided data.
    Raises
    ------
    ValueError
        If the file_extension is not '.czi' or '.tif'.

    """
    file_extension = data["general_attrs"]["file_extension"]

    # Validate file_extension
    if file_extension not in [".czi", ".tif"]:
        raise ValueError(
            f"Invalid file extension: {file_extension}. Only '.czi' and '.tif' are accepted."
        )

    # Validate directory paths using pathlib
    input_dir = Path(data["general_attrs"]["input_directory"])
    output_dir = Path(data["general_attrs"]["output_directory"])
    if not input_dir.is_dir():
        raise ValueError(f"Invalid directory path: {input_dir}")
    # Create output_dir if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count files and subfolders in input_dir
    files = list(input_dir.iterdir())
    num_files = sum(1 for f in files if f.is_file())
    num_subfolders = sum(1 for f in files if f.is_dir())

    print(
        f"Input Directory '{input_dir}' contains {num_files} files and {num_subfolders} subfolders."
    )
    # Validate extract_tiles based on file_extension
    extract_tiles = data["manual_attrs"]["extract_tiles"]
    if extract_tiles and file_extension != ".czi":
        raise ValueError(
            "extract_tiles can only be True if file_extension is '.czi'."
        )

    # Check if tiles have already been extracted
    tiles_output_dir = output_dir.joinpath("Tiles")
    if tiles_output_dir.is_dir() and any(tiles_output_dir.iterdir()):
        print(f"Tiles have already been extracted to '{tiles_output_dir}'.")
        # Print subfolder names and file counts
        print("\nContents of the 'Tiles' directory:")
        for subfolder in tiles_output_dir.iterdir():
            if subfolder.is_dir():
                file_count = len(
                    list(subfolder.glob("*"))
                )  # Count files in the subfolder
                print(
                    f"  - Subfolder '{subfolder.name}' contains {file_count} files."
                )

        if extract_tiles:
            user_input = (
                input(
                    "\nTiles already exist. Do you want to continue with extraction? (yes/no): "
                )
                .strip()
                .lower()
            )
            if user_input not in ["yes", "y"]:
                print("Extraction aborted by the user.")
                extract_tiles = False  # Set to False to prevent extraction
    # else:
    #     print(f"'Tiles' folder doesn't exist in {output_dir}")

    general_attrs = GeneralAttrs(
        input_directory=input_dir,
        file_extension=file_extension,
        output_directory=output_dir,
    )

    # Validate reference image
    downselection_reference_image = Path(
        data["manual_attrs"]["downselection_reference_image"]
    )
    # TODO: pchao, what to do if it is created later...
    # if not downselection_reference_image.is_file():
    #     raise ValueError(
    #         f"Invalid reference image path: {downselection_reference_image}"
    #     )

    # Check for existing channel folders if reorganizing by channels
    reorganize_tiles_by_channels = data["manual_attrs"][
        "reorganize_tiles_by_channels"
    ]
    if reorganize_tiles_by_channels:
        existing_folders = [
            f
            for f in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, f))
        ]
        print(f"\nCheck the existing folders in the {output_dir}:")
        mismatches = []
        channel_exists = False
        for channel in data["manual_attrs"]["channel_keywords"]:
            if channel in existing_folders:
                print(f"  - {channel} (exists)")
                channel_exists = True
            else:
                print(f"  - {channel} (does not exist)")
                mismatches.append(channel)

        # Prompt user if there are mismatches
        # if mismatches:
        if channel_exists:
            user_input = (
                input(
                    "Some channel folders already exists. Do you want to continue? (yes/no): "
                )
                .strip()
                .lower()
            )
            if user_input not in ["yes", "y"]:
                print("Operation aborted by the user.")
                reorganize_tiles_by_channels = (
                    False  # Set to False to prevent reorganization
                )

    # Check if downselection_reference_channel is in channel_keywords
    downselection_reference_channel = data["manual_attrs"][
        "downselection_reference_channel"
    ]
    if (
        downselection_reference_channel
        not in data["manual_attrs"]["channel_keywords"]
    ):
        raise ValueError(
            f"downselection_reference_channel '{downselection_reference_channel}' must be one of the channel_keywords."
        )

    # Validate downselection_ssim_threshold is between 0 and 1
    downselection_ssim_threshold = float(
        data["manual_attrs"]["downselection_ssim_threshold"]
    )
    if not 0 <= downselection_ssim_threshold <= 1:
        raise ValueError(
            f"downselection_ssim_threshold '{downselection_ssim_threshold}' must be between 0.0 and 1.0"
        )

    # Validate downselection_nxn_subimages is greater than 1, less than 100
    downselection_nxn_subimages = int(
        data["manual_attrs"]["downselection_nxn_subimages"]
    )
    if not 1 <= downselection_nxn_subimages <= 100:
        raise ValueError(
            f"Currently, downselection_nxn_subimages '{downselection_nxn_subimages}' only accepts values between 1 and 100."
        )

    manual_attrs = ManualShadingAttrs(
        extract_tiles=extract_tiles,
        reorganize_tiles_by_channels=reorganize_tiles_by_channels,
        channel_keywords=data["manual_attrs"]["channel_keywords"],
        downselection_reference_channel=downselection_reference_channel,
        downselection_reference_image=downselection_reference_image,
        downselection_ssim_threshold=downselection_ssim_threshold,
        downselection_nxn_subimages=downselection_nxn_subimages,
    )

    return ShadingConfig(
        general_attrs=general_attrs, manual_attrs=manual_attrs
    )


def create_resampler_parameters(config: dict) -> GeneralAttrs:
    """Factory to handle user specified precon settings inputted in the yml file"""

    print("Inspecting czi files for resampling: \n")
    user_resampler_params = GeneralAttrs(
        input_directory=Path(
            config["general_attrs"]["input_directory"]
        ).expanduser(),
        output_directory=Path(
            config["general_attrs"]["output_directory"]
        ).expanduser(),
        file_extension=config["general_attrs"]["file_extension"],
    )

    # # execute function
    # save_support_point_info(user_resample_settings)

    return user_resampler_params
