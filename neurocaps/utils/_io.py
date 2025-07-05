"""
Internal IO functions for creating directories, issuing warnings, serializing (with
Joblib or pickle), unserializing (with Joblib), and additional utilities.
"""

import os, copy, pickle
from typing import Any, Union

import joblib

from ._helpers import list_to_str
from ._logging import setup_logger
from neurocaps.exceptions import UnsupportedFileExtensionError

LG = setup_logger(__name__)


def makedir(output_dir: str) -> None:
    """Creates non-existent directory."""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


def issue_file_warning(param_name: str, param: str, output_dir: str) -> None:
    """Issues warning."""
    if param is not None and not output_dir:
        LG.warning(
            f"`{param_name}` supplied but no `output_dir` specified. Files will not be saved."
        )


def filename(
    base_name: str, add_name: Union[str, None], pos: str, ext: Union[None, str] = None
) -> str:
    """
    Adds the file extension and the prefix or suffix to the file name depending on if ``pos``
    is "prefix" or "suffix".
    """
    if not add_name:
        filename = f"{base_name}.{ext}" if ext else base_name

        return filename.replace(" ", "_")

    if pos == "prefix":
        # Use - since prefix is currently for ``CAP.calculate_metrics``
        add_name = os.path.splitext(add_name.rstrip())[0].rstrip()
        filename = f"{add_name}-{base_name}"
    else:
        filename = f"{base_name}_{add_name.rstrip().replace(' ', '_')}"

    filename = filename.replace(" ", "_")

    return f"{filename}.{ext}" if ext else filename


def serialize(obj: Any, output_dir: str, save_filename: str, use_joblib=False) -> None:
    """
    Serialize an object with standard pickle or Joblib. Internally, plots use pickle for
    compatibility with pickle or Joblib.
    """
    with open(os.path.join(output_dir, save_filename), "wb") as f:
        joblib.dump(obj, f) if use_joblib else pickle.dump(obj, f)


def get_obj(obj: Any, needs_deepcopy=True) -> Any:
    """Determines if object needs deepcopy or needs to be unserialized."""
    if not isinstance(obj, str):
        obj = copy.deepcopy(obj) if needs_deepcopy else obj
        return obj
    else:
        return unserialize(obj)


def unserialize(filename: str) -> Any:
    """Opens serialized objects such as ``subject_timeseries`` or ``parcel_approach``."""
    check_file_exist(filename)

    base_ext = (".pkl", ".pickle", ".joblib")
    supported_ext = (*base_ext, *[f"{x}.gz" for x in base_ext])
    check_ext(filename, supported_ext)

    with open(filename, "rb") as f:
        obj = joblib.load(f)

    return obj


def check_file_exist(filename: str) -> None:
    """Check is file exists."""
    # Dont end with periods
    if not isinstance(filename, str):
        raise ValueError(f"The following file must be a string: {filename}")

    if not os.path.isfile(filename):
        raise FileExistsError(f"The following file does not exist: {filename}")


def check_ext(
    filename: str, supported_ext: list[str], return_ext: bool = False
) -> Union[str, None]:
    """Checks if a file as a valid extension."""
    if filename.endswith(".gz"):
        _, ext = os.path.splitext(filename.removesuffix(".gz"))
        ext += ".gz"
    else:
        _, ext = os.path.splitext(filename)

    # Dont end with periods
    if ext not in supported_ext:
        raise UnsupportedFileExtensionError(
            f"The following file has an unsupported extension: {filename}\n"
            f"Only the following extensions are supported: {list_to_str(supported_ext)}"
        )

    return ext if return_ext else None
