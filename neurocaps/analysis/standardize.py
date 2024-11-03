"""Function to standardize timeseries within subject runs"""
import copy, os
from typing import Union, Optional
import numpy as np
from .._utils import _convert_pickle_to_dict, _dicts_to_pickles

def standardize(subject_timeseries_list: Union[list[dict[str, dict[str, np.ndarray]]], list[os.PathLike]],
                return_dicts: bool=True,
                output_dir: Optional[os.PathLike]=None,
                file_names: Optional[list[str]]=None) -> dict[str, dict[str, dict[str, np.ndarray]]]:

    """
    **Perform Participant-wise Timeseries Standardization**

    Standardizes the columns/ROIs of each run independently for all subjects in the subject timeseries. This function
    uses sample standard deviation, meaning Bessel's correction, `n-1` is used in the denominator. Note,
    this function is intended for use  when standardization was not performed during timeseries extraction using
    ``TimeseriesExtractor.get_bold``, as requested by the user.

    Parameters
    ----------
    subject_timeseries_list : :obj:`list[dict[str, dict[str, np.ndarray]]]` or :obj:`list[os.PathLike]`
        A list where each element consist of a dictionary mapping subject IDs to their run IDs and associated
        timeseries (TRs x ROIs) as a numpy array. Can also be a list consisting of paths to pickle files
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

    return_dicts : :obj:`bool`, default=True
        If True, returns the standardized dictionaries in the order they were inputted in ``subject_timeseries_list``.

    output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
        Directory to save the standardized dictionaries to. Will be saved as a pickle file. The directory will
        be created if it does not exist.

    file_names : :obj:`list[str]` or :obj:`None`, default=None
        A list of names to save the standardized dictionaries as. The assignment of file names to dictionaries depends
        on the index position (a file name in the 0th position will be the file name for the dictionary in the
        0th position of ``subject_timeseries_list``). If no names are provided and ``output_dir`` is specified,
        default names will be used.

    Returns
    -------
    `dict[str, dict[str, dict[str, np.ndarray]]]`
    """
    assert isinstance(subject_timeseries_list, list) and len(subject_timeseries_list) > 0, \
        "`subject_timeseries_list` must be a list greater than length 0."
    # Initialize  dict
    standardized_dicts = {}

    for indx, curr_dict in enumerate(subject_timeseries_list):
        if isinstance(curr_dict, str) and curr_dict.endswith(".pkl"): curr_dict = _convert_pickle_to_dict(curr_dict)
        else: curr_dict =  copy.deepcopy(curr_dict)

        for subj_id in curr_dict:
            for run in curr_dict[subj_id]:
                std = np.std(curr_dict[subj_id][run], axis=0, ddof=1)
                eps = np.finfo(std.dtype).eps
                # Taken from nilearn pipeline, used for numerical stability purposes to avoid numpy division error
                std[std < eps] = 1.0
                mean = np.mean(curr_dict[subj_id][run], axis=0)
                curr_dict[subj_id][run] = (curr_dict[subj_id][run] - mean)/std

        standardized_dicts[f"dict_{indx}"] = curr_dict

    if output_dir:
        _dicts_to_pickles(output_dir=output_dir, dict_list=standardized_dicts, file_names=file_names,
                          call="standardize")

    if return_dicts: return standardized_dicts
