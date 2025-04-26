"""
Internal IO class and functions for creating directories, issuing warnings, serializing (with Joblib or pickle),
unserializing (with Joblib), and additional utilities.
"""

import os, copy, pickle
from typing import Any, Union

import joblib

from .logger import _logger
from ..exceptions import UnsupportedFileExtensionError

LG = _logger(__name__)


class _IO:
    @staticmethod
    def makedir(output_dir: str) -> None:
        """Creates non-existent directory."""
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @staticmethod
    def issue_file_warning(param_name: str, param: str, output_dir: str) -> None:
        """Issues warning."""
        if param is not None and not output_dir:
            LG.warning(f"`{param_name}` supplied but no `output_dir` specified. Files will not be saved.")

    @staticmethod
    def filename(base_name: str, add_name: Union[str, None], pos: str, ext: Union[None, str] = None) -> str:
        """
        Adds the file extension and the prefix or suffix to the file name depending on if ``pos`` is "prefix" or
        "suffix".
        """
        if not add_name:
            return f"{base_name}.{ext}" if ext else base_name

        if pos == "prefix":
            # Use - since prefix is currently for ``CAP.calculate_metrics``
            add_name = os.path.splitext(add_name.rstrip())[0].rstrip()
            filename = f"{add_name}-{base_name}"
        else:
            filename = f"{base_name}_{add_name.rstrip().replace(' ', '_')}"

        filename = filename.replace(" ", "_")

        return f"{filename}.{ext}" if ext else filename

    @staticmethod
    def serialize(obj: Any, output_dir: str, save_filename: str, use_joblib=False) -> None:
        """
        Serialize an object with standard pickle or Joblib. Internally, plots use pickle for compatibility with
        pickle or Joblib.
        """
        with open(os.path.join(output_dir, save_filename), "wb") as f:
            joblib.dump(obj, f) if use_joblib else pickle.dump(obj, f)

    @staticmethod
    def get_obj(obj: Any, needs_deepcopy=True) -> Any:
        """Determines if object needs deepcopy or needs to be unserialized."""
        if not isinstance(obj, str):
            obj = copy.deepcopy(obj) if needs_deepcopy else obj
            return obj
        else:
            return _IO.unserialize(obj)

    @staticmethod
    def unserialize(file: str) -> Any:
        """Opens serialized objects such as ``subject_timeseries`` or ``parcel_approach``."""
        base_ext = (".pkl", ".pickle", ".joblib")
        supported_ext = (*base_ext, *[f"{x}.gz" for x in base_ext])
        if file.endswith(supported_ext):
            with open(file, "rb") as f:
                obj = joblib.load(f)
        else:
            raise UnsupportedFileExtensionError(
                "Serialized files must end with one of the following extensions: '.pkl', '.pickle', '.joblib'."
            )

        return obj

    @staticmethod
    def dicts_to_pickles(
        output_dir: str,
        dict_list: list[dict],
        call: str,
        filenames: Union[str, None] = None,
        message: Union[str, None] = None,
        save_reduced_dicts: bool = False,
    ) -> None:
        """
        Saves dictionaries containing NumPy arrays as pickles for the ``merge_dicts``, ``standardize``, and
        ``change_dtypes`` functions.
        """
        if not filenames:
            saved_filenames = _create_default_filenames(dict_list, call, save_reduced_dicts)
        else:
            saved_filenames = [
                f"{os.path.splitext(os.path.basename(name).rstrip())[0].rstrip()}.pkl" for name in filenames
            ]

        if message is None:
            message = "Length of `filenames` list  must be the same length as `subject_timeseries_list`."

        if call == "merge" and save_reduced_dicts is False:
            dict_list = {"merged": dict_list["merged"]}

        if filenames:
            assert len(filenames) == len(list(dict_list)), message

        _IO.makedir(output_dir)

        for i, dict_name in enumerate(list(dict_list)):
            with open(os.path.join(output_dir, saved_filenames[i]), "wb") as f:
                joblib.dump(dict_list[dict_name], f)


def _create_default_filenames(dict_list: list[dict], call: str, save_reduced_dicts: bool) -> list[str]:
    """
    Generates default names for dictionaries created by the  ``merge_dicts``, ``standardize``, and ``change_dtypes``
    functions.
    """
    if call == "merge":
        if save_reduced_dicts:
            default_filenames = [
                (
                    f"subject_timeseries_{key.split('_')[-1]}_reduced.pkl"
                    if "dict" in key
                    else "merged_subject_timeseries.pkl"
                )
                for key in list(dict_list)
            ]
        else:
            default_filenames = ["merged_subject_timeseries.pkl"]
    else:
        if call == "standardize":
            suffix = "standardized"
        else:
            dict_name = list(dict_list)[0]
            first_sub = list(dict_list[dict_name])[0]
            run_name = list(dict_list[dict_name][first_sub])[0]
            suffix = f"dtype-{str(dict_list[dict_name][first_sub][run_name].dtype)}"

        default_filenames = [f"subject_timeseries_{key.split('_')[-1]}_{suffix}.pkl" for key in list(dict_list)]

    return default_filenames
