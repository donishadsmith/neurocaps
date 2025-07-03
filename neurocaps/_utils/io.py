"""
Internal IO functions for creating directories, issuing warnings, serializing (with
Joblib or pickle), unserializing (with Joblib), and additional utilities.
"""

import os, copy, pickle
from typing import Any, Union

import joblib

from .logging import _logger
from ..exceptions import UnsupportedFileExtensionError

LG = _logger(__name__)


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


def unserialize(file: str) -> Any:
    """Opens serialized objects such as ``subject_timeseries`` or ``parcel_approach``."""
    base_ext = (".pkl", ".pickle", ".joblib")
    supported_ext = (*base_ext, *[f"{x}.gz" for x in base_ext])
    if file.endswith(supported_ext):
        with open(file, "rb") as f:
            obj = joblib.load(f)
    else:
        raise UnsupportedFileExtensionError(
            "Serialized files must end with one of the following extensions: "
            "'.pkl', '.pickle', '.joblib'."
        )

    return obj
