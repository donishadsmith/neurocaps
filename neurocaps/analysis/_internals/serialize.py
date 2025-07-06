import os
from typing import Union

import joblib

from neurocaps.utils import _io as io_utils


def dicts_to_pickles(
    output_dir: str,
    dict_list: list[dict],
    call: str,
    filenames: Union[str, None] = None,
    message: Union[str, None] = None,
    save_reduced_dicts: bool = False,
) -> None:
    """
    Saves dictionaries containing NumPy arrays as pickles for the ``merge_dicts``,
    ``standardize``, and ``change_dtypes`` functions.
    """
    if not filenames:
        saved_filenames = create_default_filenames(dict_list, call, save_reduced_dicts)
    else:
        saved_filenames = [
            f"{os.path.splitext(os.path.basename(name).rstrip())[0].rstrip()}.pkl"
            for name in filenames
        ]

    if message is None:
        message = (
            "Length of `filenames` list  must be the same length as `subject_timeseries_list`."
        )

    if call == "merge" and save_reduced_dicts is False:
        dict_list = {"merged": dict_list["merged"]}

    if filenames:
        assert len(filenames) == len(list(dict_list)), message

    io_utils.makedir(output_dir)

    for i, dict_name in enumerate(list(dict_list)):
        with open(os.path.join(output_dir, saved_filenames[i]), "wb") as f:
            joblib.dump(dict_list[dict_name], f)


def create_default_filenames(
    dict_list: list[dict], call: str, save_reduced_dicts: bool
) -> list[str]:
    """
    Generates default names for dictionaries created by the  ``merge_dicts``, ``standardize``, and
    ``change_dtypes`` functions.
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

        default_filenames = [
            f"subject_timeseries_{key.split('_')[-1]}_{suffix}.pkl" for key in list(dict_list)
        ]

    return default_filenames
