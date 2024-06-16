"""Function to standardize timeseries within subject runs"""
import copy, os
from typing import Union
import numpy as np, joblib
from .._utils import _convert_pickle_to_dict

def standardize(subject_timeseries: Union[dict[str, dict[str, np.ndarray]], os.PathLike]) -> dict[str, np.ndarray]:
    """
    **Standardize Subject Timeseries**

    Standardizes the columns/ROIs of each run independently for all subjects in the subject timeseries. This function
    uses sample standard deviation, meaning Bessel's correction, `n-1` is used in the denominator.

    Parameters
    ----------
    subject_timeseries: :obj:`dict[str, dict[str, np.ndarray]]` or :obj:`os.PathLike`
        A pickle file containing the nested subject timeseries dictionary saved by the
        ``TimeSeriesExtractor`` class or a nested subject timeseries dictionary produced by the
        ``TimeSeriesExtractor`` class. The first level of the nested dictionary must consist of the subject ID as a
        string, the second level must consist of the run numbers in the form of "run-#"
        (where # is the corresponding number of the run), and the last level must consist of the timeseries
        (as a numpy array) associated with that run. The structure is as follows:
        ::

            subject_timeseries = {
                    "101": {
                        "run-0": np.array([timeseries]), # 2D array
                        "run-1": np.array([timeseries]), # 2D array
                        "run-2": np.array([timeseries]), # 2D array
                    },
                    "102": {
                        "run-0": np.array([timeseries]), # 2D array
                        "run-1": np.array([timeseries]), # 2D array
                    }
                }

    Returns
    -------
        `dict[str, np.ndarray]`.
    """
    # Deep Copy
    subject_timeseries = copy.deepcopy(subject_timeseries)

    if isinstance(subject_timeseries, str) and subject_timeseries.endswith(".pkl"):
        subject_timeseries = _convert_pickle_to_dict()

    for subject in subject_timeseries:
        for run in subject_timeseries[subject]:
            std = np.std(subject_timeseries[subject][run], axis=0, ddof=1)
            # Taken from nilearn pipeline, used for numerical stability purposes to avoid numpy division error
            std[std < np.finfo(np.float64).eps] = 1.0
            mean = np.mean(subject_timeseries[subject][run], axis=0)
            subject_timeseries[subject][run] = (subject_timeseries[subject][run] - mean)/std

    return subject_timeseries
