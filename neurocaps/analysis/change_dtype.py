import copy, os
from typing import Union, Optional
import numpy as np
from .._utils import _convert_pickle_to_dict, _dicts_to_pickles

def change_dtype(subject_timeseries_list: Union[list[dict[str, dict[str, np.ndarray]]], list[os.PathLike]],
                 dtype: Union[str, np.floating],
                 return_dicts: bool=True,
                 output_dir: Optional[os.PathLike]=None,
                 file_names: Optional[list[str]]=None) -> dict[str, dict[str, dict[str, np.ndarray]]]:

    """
    **Perform Participant-wise Dtype Conversion**

    Changes the dtypes of each participants numpy array. This function uses the ``.astype()`` method from numpy.
    This function can help reduce memory usage. For example, converting a numpy array from "float64" to "float32" can
    halve the memory required, which is particularly useful when analyzing large datasets on a local machine.

    Parameters
    ----------
    subject_timeseries_list : :obj:`list[dict[str, dict[str, np.ndarray]]]` or :obj:`list[os.PathLike]`
        A list of dictionaries or pickle files containing the nested subject timeseries dictionary saved by the
        ``TimeSeriesExtractor`` class or a list of nested subject timeseries dictionaries produced by the
        ``TimeSeriesExtractor`` class. The first level of the nested dictionary must consist of the subject ID as a
        string, the second level must consist of the run numbers in the form of "run-#"
        (where # is the corresponding number of the run), and the last level must consist of the timeseries
        (as a numpy array) associated with that run. The structure is as follows:
        ::

            subject_timeseries = {
                    "101": {
                        "run-0": np.array([...]), # 2D array
                        "run-1": np.array([...]), # 2D array
                        "run-2": np.array([...]), # 2D array
                    },
                    "102": {
                        "run-0": np.array([...]), # 2D array
                        "run-1": np.array([...]), # 2D array
                    }
                }

        .. versionchanged:: 0.15 changed from ``subject_timeseries`` to ``subject_timeseries_list``

    dtype : :obj:`bool` or :obj:`np.floating`
        Target data type (e.g "float32" or ``np.float32``) to convert each participant's numpy arrays into.

    return_dicts : :obj:`bool`, default=True
        If True, returns the converted ``subject_timeseries``.

        .. versionchanged:: 0.15 changed from ``return_dict`` to ``return_dicts``

    output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
        Directory to save the converted ``subject_timeseries`` to. Will be saved as a pickle file. The directory will
        be created if it does not exist.

    file_names : :obj:`list[str]` or :obj:`None`, default=None
        A list of names to save the  dictionaries with changed dtypes as. The assignment of file names to dictionaries
        depends on the index position (a file name in the 0th position will be the file name for the dictionary in the
        0th position of ``subject_timeseries_list``). If no names are provided and ``output_dir`` is specified,
        default names will be used.

        .. versionchanged:: 0.15.0 from ``file_name`` to ``file_names``

    Returns
    -------
    `dict[str, dict[str, dict[str, np.ndarray]]]`

    Warning
    -------
    While this function allows conversion to any valid numpy dtype, it is recommended to use floating-point dtypes.
    Reducing the dtype could introduce rounding errors that may lower the precision of subsequent analyses as decimal
    digits are reduced when lower dtypes are requested. Thus, the lowest recommended floating-point dtype would be
    "float32", since it allows for memory usage reduction while limiting rounding errors that could significantly
    alter calculations.
    """
    assert isinstance(subject_timeseries_list, list), "`subject_timeseries_list` must be a list."
    changed_dtype_dicts = {}

    for indx, curr_dict in enumerate(subject_timeseries_list):
        if isinstance(curr_dict, str) and curr_dict.endswith(".pkl"):
            curr_dict = _convert_pickle_to_dict(curr_dict)
        else:
            curr_dict =  copy.deepcopy(curr_dict)
        for subj_id in curr_dict:
            for run in curr_dict[subj_id]:
                curr_dict[subj_id][run] = curr_dict[subj_id][run].astype(dtype)
        changed_dtype_dicts[f"dict_{indx}"] = curr_dict

    if output_dir:
        _dicts_to_pickles(output_dir=output_dir, dict_list=changed_dtype_dicts, file_names=file_names, call="change_dtype")

    if return_dicts: return changed_dtype_dicts
