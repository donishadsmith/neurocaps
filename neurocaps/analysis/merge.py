import copy, os
from typing import Union, Optional
import numpy as np
from .._utils import _convert_pickle_to_dict, _dicts_to_pickles

def merge_dicts(subject_timeseries_list: Union[list[dict[str, dict[str, np.ndarray]]], list[os.PathLike]],
                return_merged_dict: bool=True, return_reduced_dicts: bool=False,
                output_dir: Optional[Union[str, os.PathLike]]=None,
                file_names: Optional[list[str]]=None,
                save_reduced_dicts: bool=False) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    **Merge Participant Timeseries Across Multiple Tasks**

    Merge subject timeseries dictionaries or pickle files into the first dictionary or pickle file in the list.
    For each subject, timeseries (numpy arrays) with the same run ID will be concatenated, while unique run IDs will
    still be included.

    For example, if three subject timeseries are specified in ``subject_timeseries_list``, and subject 1 has:

        - run-1 in the first dictionary (representing the extracted timeseries from resting-state),
        - run-1 and run-2 in the second dictionary (representing the extracted timeseries from a Stroop task),
        - run-3 in the third dictionary (representing the extracted timeseries from a N-back task)

    Then subject 1 in the final merged dictionary will contain:

        - run-1 (concatenated from the first dictionary and second dictionary 2, resting-state and the Stroop task),
        - run-2 (from the second dictionary, the Stroop task),
        - run-3 (from the third dictionary, the N-back task).

    This function is intended for use in workflows where the final merged dictionary, returned by setting
    ``return_merged_dict`` to True, can be input into ``CAP.get_caps`` to identify similar CAPs across different tasks
    or the same task over time. Additionally, the reduced dictionaries — the input dictionaries that only contain the
    subjects present in the final merged dictionary — are returned by setting ``return_reduced_dicts`` to True. These
    reduced dictionaries can then be used in ``CAP.calculate_metrics`` to compute participant-wise CAP metrics for each
    task.

    This facilitates analysis of the temporal dynamics of similar CAPs across tasks or the same task at different
    time points.

    **Note**, Only subjects with at least one functional run present in all dictionaries are included in the final
    merged dictionary.

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

    return_merged_dict : :obj:`bool`, default=True
        If True, returns the merged dictionary.

    return_reduced_dicts : :obj:`bool`, default=False
        If True, returns the list of dictionaries provided with only the subjects present in the merged
        dictionary. The dictionaries are returned in the same order as listed in the ``subject_timeseries_list``
        parameter. The keys will be names "dict_#", with "#" indicating the index of the dictionary or pickle file
        in the ``subject_timeseries_list`` parameter.

    output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
        Directory to save the merged or reduced dictionaries, as pickle files, to. The directory will be created
        if it does not exist. For the reduced dictionaries to be saved, ``save_reduced_dicts`` must be set to True.
        If ``save_reduced_dicts`` is False and ``output_dir`` is provided, only the merged dictionary will be saved.

    file_names : :obj:`list[str]` or :obj:`None`, default=None
        A list of names to save the dictionaries as if ``output_dir`` is provided. If ``save_reduced_dicts`` is False,
        only a list with a single name should be supplied, which will be used to save the merged dictionary. If
        ``save_reduced_dicts`` is True, then the length of the file_name must match the length of
        ``subject_timeseries_list`` plus an additional name for the merged dictionary. For instance, if for
        dictionaries or pickle files are provided in ``subject_timeseries_list`` and ``save_reduced_dicts`` is True,
        five names need to be provided. Additionally, the assignment of file names to dictionaries depends on the
        index position (file name in 0th index in the list will be assigned to the reduced version of the dictionary in
        the 0th index of ``subject_timeseries_list``. The last file name is always assigned to the merged dictionary.
        For this parameter, ``os.path.basename`` is used to get the basename of the files (if a full path is supplied)
        and ``os.path.splitext`` is used to ignore extensions. Default names are provided if this variable is None.

    save_reduced_dicts : :obj:`bool` or None, default=False
        If True and the ``output_dir`` is provided, then the reduced dictionaries are saved.

    Returns
    -------
    `dict[str, dict[str, dict[str, np.ndarray]]]`
    """

    assert isinstance(subject_timeseries_list, list), "`subject_timeseries_list` must be a list."
    assert len(subject_timeseries_list) > 1, "Merging cannot be done with less than two dictionaries or files."

    if isinstance(subject_timeseries_list[0],dict): subject_timeseries_merged = subject_timeseries_list[0]
    else: subject_timeseries_merged = _convert_pickle_to_dict(pickle_file=subject_timeseries_list[0])

    # Get common subject ids
    subject_set = {}

    for curr_dict in subject_timeseries_list:
        if isinstance(curr_dict, str) and curr_dict.endswith(".pkl"):
            curr_dict = _convert_pickle_to_dict(pickle_file=curr_dict)

        if not subject_set: subject_set = set(curr_dict)

        subject_set = subject_set.intersection(list(curr_dict))

    # Order subjects
    intersect_subjects = sorted(list(subject_set))

    subject_timeseries_merged = {}

    for curr_dict in subject_timeseries_list:
        if isinstance(curr_dict, str) and curr_dict.endswith(".pkl"):
            curr_dict = _convert_pickle_to_dict(pickle_file=curr_dict)

        for subj_id in intersect_subjects:
            if subj_id not in subject_timeseries_merged: subject_timeseries_merged.update({subj_id: {}})
            # Get run names in the current iteration
            for curr_run in curr_dict[subj_id]:
                # If run is in merged dict, stack. If not, add
                if curr_run in subject_timeseries_merged[subj_id]:
                    subject_timeseries_merged[subj_id][curr_run] = np.vstack(
                        [subject_timeseries_merged[subj_id][curr_run], curr_dict[subj_id][curr_run]])
                else:
                    subject_timeseries_merged[subj_id].update({curr_run: curr_dict[subj_id][curr_run]})

            # Sort runs lexicographically
            if list(subject_timeseries_merged[subj_id]) != sorted(subject_timeseries_merged[subj_id].keys()):
                subject_timeseries_merged[subj_id] = {run_id: subject_timeseries_merged[subj_id][run_id] for run_id
                                                      in sorted(subject_timeseries_merged[subj_id].keys())}

    modified_dicts = {}

    if return_reduced_dicts or (save_reduced_dicts and output_dir):
        for indx, curr_dict in enumerate(subject_timeseries_list):
            if "pkl" in curr_dict: curr_dict = _convert_pickle_to_dict(pickle_file=curr_dict)
            else: curr_dict = copy.deepcopy(curr_dict)

            if any([elem in subject_timeseries_merged for elem in curr_dict]):
                modified_dicts[f"dict_{indx}"] = {}
                for subj_id in subject_timeseries_merged:
                    if subj_id in curr_dict: modified_dicts[f"dict_{indx}"].update({subj_id : curr_dict[subj_id]})

    if return_merged_dict or output_dir: modified_dicts["merged"] = subject_timeseries_merged

    if output_dir:
        message = ("Length of `file_names` must be equal to 1 if `save_reduced_dicts`is False or the length of "
                   "`subject_timeseries_list` + 1 if `save_reduced_dicts` is True.")

        _dicts_to_pickles(output_dir=output_dir, dict_list=modified_dicts, call="merge", file_names=file_names,
                          message=message,save_reduced_dicts=save_reduced_dicts)

    if return_merged_dict or return_reduced_dicts: return modified_dicts
