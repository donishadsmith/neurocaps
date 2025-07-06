"""Function for merging timeseries data across dictionaries."""

from typing import Union, Optional

import numpy as np

from ._internals.serialize import dicts_to_pickles
from neurocaps.typing import SubjectTimeseries
from neurocaps.utils import _io as io_utils


def merge_dicts(
    subject_timeseries_list: Union[list[SubjectTimeseries], list[str]],
    return_merged_dict: bool = True,
    return_reduced_dicts: bool = False,
    output_dir: Optional[Union[str, str]] = None,
    filenames: Optional[list[str]] = None,
    save_reduced_dicts: bool = False,
) -> Union[dict[str, SubjectTimeseries], None]:
    """
    Merge Participant Timeseries Across Multiple Sessions or Tasks.

    Merge subject timeseries data across dictionaries, concatenating matching run IDs. Only subjects
    present across all input dictionaries are included in the merged output.

    For example, if three dictionaries are provided contain subject 1 with:

        - dict 1: run-1 (resting-state)
        - dict 2: run-1 and run-2 (stroop)
        - dict 3: run-3 (n-back)

    Then subject 1 in the final merged dictionary will contain:

        - run-1: concatenated timeseries from dict 1 and dict 2 (resting-state + stroop)
        - run-2: timeseries from dict 2 (stroop)
        - run-3: timeseries from dict 3 (n-back)

    This function supports workflows for identifying similar CAPs across tasks or sessions.
    Specifically, using the merged dictionary as input for ``CAP.get_caps`` and the filtered input
    dictionaries, containing only subjects present in the merged dictionary, as inputs for
    ``CAP.calculate_metrics`` to compute participant-wise CAP metrics for each task.

    Parameters
    ----------
    subject_timeseries_list: :obj:`list[SubjectTimeseries]` or :obj:`list[str]`
        A list where each element consist of a dictionary mapping subject IDs to their run IDs and
        associated timeseries (TRs x ROIs) as a NumPy array. Can also be a list consisting of paths
        to pickle files containing this same structure. Refer to documentation for
        ``SubjectTimeseries`` in the "See Also" section for an example structure.

    return_merged_dict: :obj:`bool`, default=True
        If True, returns a single dictionary containing the merged dictionary under a key named
        "merged".

    return_reduced_dicts: :obj:`bool`, default=False
        If True, returns a single dictionary containing the input dictionaries filtered to only
        include subjects present in the merged dictionary. Keys are named "dict_{0}" where {0}
        corresponds to the dictionary's position in the input list.

    output_dir: :obj:`str` or :obj:`None`, default=None
        Directory to save the merged or reduced dictionaries as pickle files. The directory will be
        created if it does not exist. For the reduced dictionaries to be saved, ``save_reduced_dicts``
        must be set to True. If ``save_reduced_dicts`` is False and ``output_dir`` is provided, only
        the merged dictionary will be saved. Dictionaries will not be saved if None.

    filenames: :obj:`list[str]` or :obj:`None`, default=None
        A list of file names for saving dictionaries when ``output_dir`` is provided.

        If ``save_reduced_dicts`` is False:

            - Provide a single name, which will be used to save the merged dictionary.

        If ``save_reduced_dicts`` is True:

            - Provide N+1 names (where N is the length of ``subject_timeseries_list``): N names for
            individual reduced dictionaries followed by one name for the merged dictionary. Names
            are assigned by input position order.

        *Note*: Full paths are handled using basename and extensions are ignored. If None, uses
        default names - "subject_timeseries_{0}_reduced.pkl" (where {0} indicates the original input
        order) and "merged_subject_timeseries.pkl" for the merged dictionary.

    save_reduced_dicts: :obj:`bool` or None, default=False
        If True and the ``output_dir`` is provided, then the reduced dictionaries are saved as
        pickle files.

    Returns
    -------
    dict[str, SubjectTimeseries]
        A nested dictionary containing the merged subject timeseries and reduced subject timeseries
        if either ``return_merged_dict`` or ``return_reduced_dict`` are True.

    See Also
    --------
    :data:`neurocaps.typing.SubjectTimeseries`
        Type definition for the subject timeseries dictionary structure.
        (See: `SubjectTimeseries Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.SubjectTimeseries.html#neurocaps.typing.SubjectTimeseries>`_)

    References
    ----------
    Kupis, L., Romero, C., Dirks, B., Hoang, S., Parladé, M. V., Beaumont, A. L., Cardona, S. M.,
    Alessandri, M., Chang, C., Nomi, J. S., & Uddin, L. Q. (2020). Evoked and intrinsic brain
    network dynamics in children with autism spectrum disorder. NeuroImage: Clinical, 28, 102396.
    https://doi.org/10.1016/j.nicl.2020.102396
    """

    assert isinstance(subject_timeseries_list, list), "`subject_timeseries_list` must be a list."
    assert (
        len(subject_timeseries_list) > 1
    ), "Merging cannot be done with less than two dictionaries or files."

    io_utils.issue_file_warning("filenames", filenames, output_dir)

    # Only perform IO operation once
    new_timeseries_list = []
    for curr_dict in subject_timeseries_list:
        curr_dict = io_utils.get_obj(curr_dict, needs_deepcopy=False)
        new_timeseries_list.append(curr_dict)

    # Get common subject ids
    subject_set = {}
    for curr_dict in new_timeseries_list:
        if not subject_set:
            subject_set = set(curr_dict)

        subject_set = subject_set.intersection(list(curr_dict))

    # Order subjects
    intersect_subjects = sorted(list(subject_set))

    # Collect timeseries array in a temporary dictionary as lists to reduce repeated np.vstack calls
    temp_dict = {}
    for curr_dict in new_timeseries_list:
        for subj_id in intersect_subjects:
            if subj_id not in temp_dict:
                temp_dict[subj_id] = {}

            for curr_run, arr in curr_dict[subj_id].items():
                if curr_run not in temp_dict[subj_id]:
                    temp_dict[subj_id][curr_run] = [arr]
                else:
                    temp_dict[subj_id][curr_run].append(arr)

    # Now use temporary dict to create the final merged dictionary
    subject_timeseries_merged = {}
    for subj_id, runs_dict in temp_dict.items():
        subject_timeseries_merged[subj_id] = {}
        for curr_run, arr_list in runs_dict.items():
            # Ensure memory address of numpy array is different when only a single array
            if len(arr_list) == 1:
                subject_timeseries_merged[subj_id][curr_run] = arr_list[0].copy()
            else:
                subject_timeseries_merged[subj_id][curr_run] = np.vstack(arr_list)

        # Sort runs lexicographically
        if list(subject_timeseries_merged[subj_id]) != sorted(
            subject_timeseries_merged[subj_id].keys()
        ):
            subject_timeseries_merged[subj_id] = {
                run_id: subject_timeseries_merged[subj_id][run_id]
                for run_id in sorted(subject_timeseries_merged[subj_id].keys())
            }

    del temp_dict

    modified_dicts = {}
    # Obtain the reduced dictionaries
    if return_reduced_dicts or (save_reduced_dicts and output_dir):
        for indx, curr_dict in enumerate(new_timeseries_list):
            modified_dicts[f"dict_{indx}"] = {}

            for subj_id in subject_timeseries_merged:
                # Diff memory address from original array
                modified_dicts[f"dict_{indx}"].update({subj_id: curr_dict[subj_id].copy()})

    if return_merged_dict or output_dir:
        modified_dicts["merged"] = subject_timeseries_merged

    if output_dir:
        message = (
            "Length of `file_names` must be equal to 1 if `save_reduced_dicts`is False or the "
            "length of `subject_timeseries_list` + 1 if `save_reduced_dicts` is True."
        )

        # Convert to list if string
        if isinstance(filenames, str):
            filenames = [filenames]

        dicts_to_pickles(
            output_dir=output_dir,
            dict_list=modified_dicts,
            call="merge",
            filenames=filenames,
            message=message,
            save_reduced_dicts=save_reduced_dicts,
        )

    if return_merged_dict or return_reduced_dicts:
        return modified_dicts
