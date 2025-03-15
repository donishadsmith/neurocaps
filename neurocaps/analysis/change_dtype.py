import copy
from typing import Union, Optional

import numpy as np

from .._utils import _convert_pickle_to_dict, _dicts_to_pickles, _logger
from ..typing import SubjectTimeseries

LG = _logger(__name__)


def change_dtype(
    subject_timeseries_list: Union[list[SubjectTimeseries], list[str]],
    dtype: Union[str, np.floating],
    return_dicts: bool = True,
    output_dir: Optional[str] = None,
    filenames: Optional[list[str]] = None,
) -> Union[dict[str, SubjectTimeseries], None]:
    """
    Perform Participant-wise Dtype Conversion.

    Changes the dtypes of each participants NumPy array. This function uses the ``.astype()`` method from NumPy.
    This function can help reduce memory usage. For example, converting a NumPy array from "float64" to "float32" can
    halve the memory required, which is particularly useful when analyzing large datasets on a local machine.

    Parameters
    ----------
    subject_timeseries_list: :obj:`list[dict[str, dict[str, np.ndarray]]]` or :obj:`list[str]`
        A list where each element consist of a dictionary mapping subject IDs to their run IDs and associated
        timeseries (TRs x ROIs) as a NumPy array. Can also be a list consisting of paths to pickle files
        containing this same structure. The expected structure of each dictionary is as follows:

        ::

            subject_timeseries = {
                    "101": {
                        "run-0": np.array([...]), # Shape: TRs x ROIs
                        "run-1": np.array([...]), # Shape: TRs x ROIs
                        "run-2": np.array([...]), # Shape: TRs x ROIs
                    },
                    "102": {
                        "run-0": np.array([...]), # Shape: TRs x ROIs
                        "run-1": np.array([...]), # Shape: TRs x ROIs
                    }
                }

    dtype: :obj:`bool` or :obj:`np.floating`
        Target data type (e.g "float32" or ``np.float32``) to convert each participant's NumPy arrays into.

    return_dicts: :obj:`bool`, default=True
        If True, returns a single dictionary containing the converted input dictionaries. Keys are named "dict_{0}"
        where {0} corresponds to the dictionary's position in the input list.

    output_dir: :obj:`str` or :obj:`None`, default=None
        Directory to save the converted ``subject_timeseries`` as pickle files. The directory will be created if it
        does not exist. Dictionaries will not be saved if None.

    filenames: :obj:`list[str]` or :obj:`None`, default=None
        A list of names to save the dictionaries with changed dtypes as. Names are matched to dictionaries by position
        (e.g., a file name in the 0th position will be the file name for the dictionary in the 0th position of
        ``subject_timeseries_list``). If None and ``output_dir`` is specified, uses default file names -
        "subject_timeseries_{0}_float{1}.pkl" (where {0} indicates the original input order and {1} is the dtype.

    Returns
    -------
    dict[str, SubjectTimeseries]
        A nested dictionary containing the converted subject timeseries if ``return_dicts`` is True.

    See Also
    --------
    :data:`neurocaps.typing.SubjectTimeseries`
        The type definition for the subject timeseries dictionary structure.

    Warning
    -------
    **Floating Point Precision**: The minimum recommended floating-point dtype is *float32*, as lower precision may
    introduce rounding errors that affect calculations.
    """
    assert isinstance(subject_timeseries_list, list), "`subject_timeseries_list` must be a list."

    if filenames is not None and output_dir is None:
        LG.warning("`filenames` supplied but no `output_dir` specified. Files will not be saved.")

    changed_dtype_dicts = {}

    for indx, curr_dict in enumerate(subject_timeseries_list):
        if isinstance(curr_dict, str) and curr_dict.endswith(".pkl"):
            curr_dict = _convert_pickle_to_dict(curr_dict)
        else:
            curr_dict = copy.deepcopy(curr_dict)

        for subj_id in curr_dict:
            for run in curr_dict[subj_id]:
                curr_dict[subj_id][run] = curr_dict[subj_id][run].astype(dtype)

        changed_dtype_dicts[f"dict_{indx}"] = curr_dict

    if output_dir:
        _dicts_to_pickles(
            output_dir=output_dir, dict_list=changed_dtype_dicts, filenames=filenames, call="change_dtype"
        )

    if return_dicts:
        return changed_dtype_dicts
