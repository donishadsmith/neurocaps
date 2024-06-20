import os
from typing import Union, Optional
import numpy as np, joblib
from .._utils import _convert_pickle_to_dict

def merge_dicts(subject_timeseries_list: Union[list[dict], list[os.PathLike]], return_combined_dict: bool=True,
                return_reduced_dicts: bool=False, output_dir: Optional[Union[str, os.PathLike]]=None,
                file_name: Optional[str]=None) -> dict[str, np.ndarray]:
    """
    **Merge subject timeseries**

    Merge subject timeseries dictionaries or pickle files into the first dictionary or pickle file in the list.
    Repetition times/frames from the same subject and run are merged together. The combined dictionary will only
    include subjects that are present in all dictionaries.

    Parameters
    ----------
    subject_timeseries_list: :obj:`list[dict]]` or :obj:`list[os.PathLike]`
        A list of pickle files containing the nested subject timeseries dictionary saved by the
        ``TimeSeriesExtractor`` class or a list of nested subject timeseries dictionaries produced by the
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

    return_combined_dict: :obj:`bool`, default=True,
        If True, returns the merged dictionary.

    return_reduced_dicts: :obj:`bool`, default=False
        If True, returns the list of dictionaries provided with only the subjects present in the combined
        dictionary. The dictionaries are returned in the same order as listed in the ``subject_timeseries_list``
        parameter. The keys will be names "dict_#", with "#" indicating the index of the dictionary or pickle file
        in the ``subject_timeseries_list`` parameter.

    output_dir: :obj:`os.PathLike` or :obj:`None`, default=None
        Directory to save the merged dictionary to. Will be saved as a pickle file. The directory will be created
        if it does not exist.

    file_name: :obj:`str` or :obj:`None`, default=None
        Name to save the merged dictionary as.

    Returns
    -------
        `dict[str, dict[str, np.ndarray]]` or `dict[str, dict[str, dict[str, np.ndarray]]]` if ``return_reduced_dicts``
        is True.
    """
    assert len(subject_timeseries_list) > 1, "Merging cannot be done with less than two dictionaries or files."

    if isinstance(subject_timeseries_list[0],dict): subject_timeseries_combined = subject_timeseries_list[0]
    else: subject_timeseries_combined = _convert_pickle_to_dict(pickle_file=subject_timeseries_list[0])

    # Get common subject ids
    subject_set = {}

    for curr_dict in subject_timeseries_list:
        if isinstance(curr_dict, str) and curr_dict.endswith(".pkl"):
            curr_dict = _convert_pickle_to_dict(pickle_file=curr_dict)
        if len(subject_set) == 0: subject_set = set(curr_dict)
        subject_set = subject_set.intersection(list(curr_dict))

    # Order subjects
    intersect_subjects = sorted(list(subject_set))

    subject_timeseries_combined = {}

    for curr_dict in subject_timeseries_list:
        if isinstance(curr_dict, str) and curr_dict.endswith(".pkl"):
            curr_dict = _convert_pickle_to_dict(pickle_file=curr_dict)
        for subj_id in intersect_subjects:
            if subj_id not in subject_timeseries_combined: subject_timeseries_combined.update({subj_id: {}})
            # Get run names in the current iteration
            subject_runs = curr_dict[subj_id]
            for curr_run in subject_runs:
                # If run is in combined dict, stack. If not, add
                if curr_run in subject_timeseries_combined[subj_id]:
                    subject_timeseries_combined[subj_id][curr_run] = np.vstack([subject_timeseries_combined[subj_id][curr_run],
                                                                                curr_dict[subj_id][curr_run]])
                else:
                    subject_timeseries_combined[subj_id].update({curr_run: curr_dict[subj_id][curr_run]})

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir,file_name + ".pkl"), "wb") as f:
            joblib.dump(subject_timeseries_combined,f)

    if return_reduced_dicts:
        all_dicts = {}
        for indx, curr_dict in enumerate(subject_timeseries_list):
            if "pkl" in curr_dict: curr_dict = _convert_pickle_to_dict(pickle_file=curr_dict)
            if any([elem in subject_timeseries_combined for elem in curr_dict]):
                all_dicts[f"dict_{indx}"] = {}
                for subj_id in subject_timeseries_combined:
                    if subj_id in curr_dict:
                        all_dicts[f"dict_{indx}"].update({subj_id : curr_dict[subj_id]})
        if not return_combined_dict: return all_dicts

    if return_combined_dict:
        if not return_reduced_dicts: return subject_timeseries_combined
        else:
            all_dicts["combined"] = subject_timeseries_combined
            return all_dicts
