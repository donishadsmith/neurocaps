import numpy as np, pickle
from typing import Union, Dict

def standardize(subject_timeseries: Union[Dict[str, Dict[str, np.ndarray]], str]) -> dict:
    """Standardize subject timeseries 
    
    Standardizes each run independently for all subjects in the subject timeseries.

    Parameters
    ----------
    subject_timeseries_list: Dict[str, Dict[str, np.ndarray]] or str
        A list of pickle files containing the nested subject timeseries dictionary saved by the `TimeSeriesExtractor` class or a list of nested subject 
        timeseries dictionaries produced by the `TimeSeriesExtractor` class. The first level of the nested dictionary must consist of the subject ID as a string, 
        the second level must consist of the run numbers in the form of 'run-#' (where # is the corresponding number of the run), and the last level must consist of the timeseries 
        (as a numpy array) associated with that run.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
    """

    if ".pkl" in subject_timeseries:
        with open(subject_timeseries, "rb") as foo:
            subject_timeseries = pickle.load(foo)

    for subject in subject_timeseries.keys():
        for run in subject_timeseries[subject].keys():
            subject_timeseries[subject][run] -= subject_timeseries[subject][run].mean(axis=0)
            std = subject_timeseries[subject][run].std(axis=0, ddof=1)
            # Taken from nilearn pipeline, used for numerical stability purposes to avoide numpy division error
            std[std < np.finfo(np.float64).eps] = 1.0             
            subject_timeseries[subject][run] /= std

    return subject_timeseries