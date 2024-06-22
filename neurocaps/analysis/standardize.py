"""Function to standardize timeseries within subject runs"""
import copy, os
from typing import Union, Optional
import numpy as np, joblib
from .._utils import _convert_pickle_to_dict

def standardize(subject_timeseries: Union[dict[str, dict[str, np.ndarray]], os.PathLike],
                return_dict: bool=True,
                output_dir: Optional[os.PathLike]=None,
                file_name: Optional[str]=None) -> dict[str, dict[str, np.ndarray]]:

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
                        "run-0": np.array([...]), # 2D array
                        "run-1": np.array([...]), # 2D array
                        "run-2": np.array([...]), # 2D array
                    },
                    "102": {
                        "run-0": np.array([...]), # 2D array
                        "run-1": np.array([...]), # 2D array
                    }
                }

    return_dict: :obj:`bool`, default=True
        If True, returns the standardized ``subject_timeseries``.

        .. versionadded:: 0.11.0

    output_dir: :obj:`os.PathLike` or :obj:`None`, default=None
        Directory to save the standardized ``subject_timeseries`` to. Will be saved as a pickle file. The directory will
        be created if it does not exist.

        .. versionadded:: 0.11.0

    file_name: :obj:`str` or :obj:`None`, default=None
        Name to save the standardized ``subject_timeseries`` as.

        .. versionadded:: 0.11.0

    Returns
    -------
    `dict[str, dict[str, np.ndarray]]`.
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

    if output_dir:

        if file_name: save_file_name = f"{os.path.splitext(file_name.rstrip())[0].rstrip()}.pkl"
        else: save_file_name = f"standardized_subject_timeseries.pkl"

        with open(os.path.join(output_dir,save_file_name), "wb") as f:
            joblib.dump(subject_timeseries,f)

    if return_dict: return subject_timeseries
