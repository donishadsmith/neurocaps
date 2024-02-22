import numpy as np
from typing import Union
from .._utils import _convert_pickle_to_dict

def merge_dicts(subject_timeseries_list: Union[list[dict], list[str]], return_dict: bool=True, output_dir: str=None, file_name: str=None) -> dict:
    """Merge subject timeseries

    Merge subject timeseries dictionaries or pickle files to the first dictionary or pickle file in the list.
    Repetition times from the same subject and run are merged together. The combined dicitonary will only include subjects
    that are present in all dictionaries.

    Parameters
    ----------

    subject_timeseries_list: dict
        The list of pickle files containing the nested subject timeseries dictionary saved by the TimeSeriesExtractor class or a liist of the
        the nested subject timeseries dictionary produced by the TimeseriesExtractor class. The first level of the nested dictionary must consist of the subject
        ID as a string, the second level must consist of the the run numbers in the form of 'run-#', where # is the corresponding number of the run, and the last level 
        must consist of the timeseries associated with that run.
    return_dict: bool, default=True,
        Returns the merged dictionaries if True
    output_dir: str, default=None
        Directory to save the merged dictionary to. Will be saved as a pickle file.
    file_name: str, default=None
        Name to save merged dictionary as.

    Raises
    ------
    AssertionError
        If the length of `subject_timeseries_list` is less than two.

    Returns
    -------
    dict
    
    """
    assert len(subject_timeseries_list) > 1, "Merging cannot be done with less than two dictionaries or files."

    if isinstance(subject_timeseries_list[0],dict): subject_timeseries_combined = subject_timeseries_list[0] 
    else: subject_timeseries_combined = _convert_pickle_to_dict(pickle_file=subject_timeseries_list[0])

    # Get common subject ids
    subject_set = {}

    for curr_dict in subject_timeseries_list:      
        if "pkl" in curr_dict: curr_dict = _convert_pickle_to_dict(pickle_file=curr_dict)
        if len(subject_set) == 0: subject_set = set(curr_dict.keys())    
        subject_set = subject_set.intersection(list(curr_dict.keys()))

    # Order subjects
    intersect_subjects = sorted(list(subject_set))

    subject_timeseries_combined = {}

    for curr_dict in subject_timeseries_list:
        if "pkl" in curr_dict: curr_dict = _convert_pickle_to_dict(pickle_file=curr_dict)
        for subj_id in intersect_subjects:
            if subj_id not in subject_timeseries_combined.keys(): subject_timeseries_combined.update({subj_id: {}})
            # Get run names in the current iteration
            subject_runs = curr_dict[subj_id].keys()
            for curr_run in subject_runs:
                # If run is in combined dict, stack. If not, add
                if curr_run in subject_timeseries_combined[subj_id].keys():
                    subject_timeseries_combined[subj_id][curr_run] = np.vstack([subject_timeseries_combined[subj_id][curr_run], curr_dict[subj_id][curr_run]])
                else:
                    subject_timeseries_combined[subj_id].update({curr_run: curr_dict[subj_id][curr_run]})
                
    if output_dir:
        import pickle, os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        with open(os.path.join(output_dir,file_name + ".pkl"), "wb") as f:
            pickle.dump(subject_timeseries_combined,f)
        
    if return_dict:
        return subject_timeseries_combined
