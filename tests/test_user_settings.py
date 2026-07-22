"""This module tests that the user settings file can be found and used.

Example:
cd ~/precon3d
source .venv/source/activate
python -m pytest precon3d/tests/test_user_settings.py

"""

from typing import Final
from pathlib import Path

import precon3d.utility as ut


@ut.run_on_local_machine
def test_user_settings_file_exists_and_functions():
    """Assures that a user settings file exists on the local machine."""
    fin: Final[str] = "precon3d_user_settings.yml"
    user_home = Path.home()
    pin = user_home.joinpath(fin)
    assert pin.is_file(), f"File not found: {pin}"

    required_keys = ["fiji_app", "home", "scratch"]
    config_dict = ut.read_config(pin)

    for k in config_dict.keys():
        # print(f"\nChecking key: {k}, value: {v}")
        assert (
            k in required_keys
        ), f"key: {k}, not found. Please define {required_keys}"

    user_settings = ut.UserSettings(
        fiji_app=Path(config_dict["fiji_app"]),
        home=Path(config_dict["home"]),
        scratch=Path(config_dict["scratch"]),
    )

    assert Path(
        user_settings.fiji_app
    ).exists(), f"Does not exist: {user_settings.fiji_app}"
    assert Path(
        user_settings.home
    ).exists(), f"Does not exist: {user_settings.home}"
    assert Path(
        user_settings.scratch
    ).exists(), f"Does not exist: {user_settings.scratch}"
