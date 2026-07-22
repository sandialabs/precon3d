"""Test fixtures defining the file locations."""

from enum import Enum
import pytest

from typing import Tuple

from typing import NamedTuple, Final
from pathlib import Path

TEST_FILES_DIRECTORY: Final[Path] = Path(__file__).parent.joinpath(
    "files", "test_czi"
)


class FileMetadata(NamedTuple):
    """Represents a test file with metadata."""

    name: str  # File name
    path: Path  # Full path to the file
    type: str  # File type (e.g., "grayscale" or "color")
    format: str  # Format level (e.g., "tileregion" or "tilearray" or "mixed")

    def description(self) -> str:
        """Return a human-readable description of the file."""
        return f"{self.name} ({self.type}, {self.format})"


def parse_file_metadata(file_name: str) -> tuple[str, str, str]:
    """Parse file metadata from the file name."""
    # Example naming convention: BW_Pol_Bright_1TA.czi
    parts = file_name.split("_")
    file_type = "grayscale" if parts[0] == "BW" else "color"
    # Determine format based on "TA" and "TR" in the file name
    if "TA" in file_name and "TR" in file_name:
        format_level = "mixed"  # Both TileArray and TileRegion
    elif "TA" in file_name:
        format_level = "tilearray"  # TileArray only
    elif "TR" in file_name:
        format_level = "tileregion"  # TileRegion only
    else:
        format_level = "unknown"  # No TA or TR in the file name

    return file_type, format_level


def load_test_files(directory: Path) -> list[FileMetadata]:
    """Load test files from the given directory."""
    test_files = []
    for file_path in directory.glob("*.czi"):  # Find all .czi files
        file_type, format_level = parse_file_metadata(file_path.name)
        test_files.append(
            FileMetadata(
                name=file_path.name,
                path=file_path,
                type=file_type,
                format=format_level,
            )
        )
    return test_files


def validate_files(files: list[FileMetadata]) -> None:
    """Validate that all test files exist."""
    missing_files = [f.path for f in files if not f.path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing test files: {missing_files}")
