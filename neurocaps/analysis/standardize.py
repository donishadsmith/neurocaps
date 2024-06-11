"""Function to standardize timeseries within subject runs"""
from typing import Union, Dict
import numpy as np, joblib

def standardize(subject_timeseries: Union[Dict[str, Dict[str, np.ndarray]], str]) -> Dict[str, np.ndarray]:
    """
    **Standardize Subject Timeseries**

    Standardizes each run independently for all subjects in the subject timeseries.

    Parameters
    ----------
        subject_timeseries_list: List[Dict]] or List[str]
            A list of pickle files containing the nested subject timeseries dictionary saved by the
            ``TimeSeriesExtractor`` class or a list of nested subject timeseries dictionaries produced by the
            ``TimeSeriesExtractor`` class. The first level of the nested dictionary must consist of the subject ID as a
            string, the second level must consist of the run numbers in the form of 'run-#' (where # is the
            corresponding number of the run), and the last level must consist of the timeseries (as a numpy array)
            associated with that run.  The structure is as follows:
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
        Dict[str, Dict[str, np.ndarray]]
    """

    if ".pkl" in subject_timeseries:
        with open(subject_timeseries, "rb") as pickle_file:
            subject_timeseries = joblib.load(pickle_file)

    for subject in subject_timeseries:
        for run in subject_timeseries[subject]:
            subject_timeseries[subject][run] -= subject_timeseries[subject][run].mean(axis=0)
            std = subject_timeseries[subject][run].std(axis=0, ddof=1)
            # Taken from nilearn pipeline, used for numerical stability purposes to avoid numpy division error
            std[std < np.finfo(np.float64).eps] = 1.0
            subject_timeseries[subject][run] /= std

    return subject_timeseries
