import numpy as np, pickle
from typing import Union

def standardize(subject_timeseries: Union[dict,str]) -> dict:
    """Standardize subject timeseries 
    
    Standardizes each run independently for all subjects in the subject timeseries.

    Parameters
    ----------
    subject_timeseries_list: dict or str
        The list of pickle files containing the nested subject timeseries dictionary saved by the TimeSeriesExtractor class or a liist of the
        the nested subject timeseries dictionary produced by the TimeseriesExtractor class. The first level of the nested dictionary must consist of the subject
        ID as a string, the second level must consist of the the run numbers in the form of 'run-#', where # is the corresponding number of the run, and the last level 
        must consist of the timeseries associated with that run.

    Returns
    -------
    dict
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