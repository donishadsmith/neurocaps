import copy, os
from typing import Union, Optional
import numpy as np, joblib
from .._utils import _convert_pickle_to_dict

def change_dtype(subject_timeseries: Union[dict[str, dict[str, np.ndarray]], os.PathLike],
                 dtype: Union[str, np.floating],
                 return_dict=True,
                 output_dir: Optional[os.PathLike]=None,
                 file_name: Optional[str]=None) -> dict[str, np.ndarray]:

    """
    **Change Dtype of Subject Timeseries**

    Changes the dtypes of each participants numpy array. This function uses the ``.astype()`` method from numpy.
    This function can assist with memory usage. For instance, converting a numpy array from "float64" to "float32"
    can halve the memory required, which is especially useful when performing analyses on larger datasets on a
    local machine.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    subject_timeseries : :obj:`dict[str, dict[str, np.ndarray]]` or :obj:`os.PathLike`
        A pickle file containing the nested subject timeseries dictionary saved by the
        ``TimeSeriesExtractor`` class or a nested subject timeseries dictionary produced by the
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

    dtype : :obj:`bool` or :obj:`np.floating`
        Target data type (e.g "float32" or ``np.float32``) to convert each participant's numpy arrays into.

    return_dict : :obj:`bool`, default=True
        If True, returns the converted ``subject_timeseries``.

    output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
        Directory to save the converted ``subject_timeseries`` to. Will be saved as a pickle file. The directory will
        be created if it does not exist.

    file_name : :obj:`str` or :obj:`None`, default=None
        Name to save the converted ``subject_timeseries`` as.

    Returns
    -------
    `dict[str, np.ndarray]`.

    Warning
    -------
    While this function allows conversion to any valid numpy dtype, it is recommended to use floating-point dtypes.
    Reducing the dtype could introduce rounding errors that may lower the precision of subsequent analyses as decimal
    digits are reduced when lower dtypes are requested. Thus, the lowest recommended floating-point dtype would be
    "float32", since it allows for memory usage reduction while limiting rounding errors that could significantly
    alter calculations.
    """
    # Deep Copy
    subject_timeseries = copy.deepcopy(subject_timeseries)

    if isinstance(subject_timeseries, str) and subject_timeseries.endswith(".pkl"):
        subject_timeseries = _convert_pickle_to_dict()

    for subj_id in subject_timeseries:
        for run in subject_timeseries[subj_id]:
            subject_timeseries[subj_id][run] = subject_timeseries[subj_id][run].astype(dtype)

    if output_dir:
        if file_name: save_file_name = f"{os.path.splitext(file_name.rstrip())[0].rstrip()}.pkl"
        else: save_file_name = file_name if file_name else f"subject_timeseries_desc-{dtype}.pkl"
        with open(os.path.join(output_dir, save_file_name), "wb") as f:
            joblib.dump(subject_timeseries,f)

    if return_dict: return subject_timeseries
