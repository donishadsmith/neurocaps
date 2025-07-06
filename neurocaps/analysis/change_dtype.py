"""Function for changing the dtype of timeseries data."""

from typing import Union, Optional

import numpy as np

from ._internals import serialize
from neurocaps.utils import _io as io_utils
from neurocaps.typing import SubjectTimeseries


def change_dtype(
    subject_timeseries_list: Union[list[SubjectTimeseries], list[str]],
    dtype: Union[str, np.floating],
    return_dicts: bool = True,
    output_dir: Optional[str] = None,
    filenames: Optional[list[str]] = None,
) -> Union[dict[str, SubjectTimeseries], None]:
    """
    Perform Participant-wise Dtype Conversion.

    Changes the dtypes of each participants NumPy array.

    Parameters
    ----------
    subject_timeseries_list: :obj:`list[dict[str, dict[str, np.ndarray]]]` or :obj:`list[str]`
        A list where each element consist of a dictionary mapping subject IDs to their run IDs and
        associated timeseries (TRs x ROIs) as a NumPy array. Can also be a list consisting of paths
        to pickle files containing this same structure. Refer to documentation for
        ``SubjectTimeseries`` in the "See Also" section for an example structure.

    dtype: :obj:`bool` or :obj:`np.floating`
        Target data type (e.g "float32" or ``np.float32``) to convert each participant's NumPy
        arrays into.

    return_dicts: :obj:`bool`, default=True
        If True, returns a single dictionary containing the converted input dictionaries. Keys are
        named "dict_{0}" where {0} corresponds to the dictionary's position in the input list.

    output_dir: :obj:`str` or :obj:`None`, default=None
        Directory to save the converted ``subject_timeseries`` as pickle files. The directory will
        be created if it does not exist. Dictionaries will not be saved if None.

    filenames: :obj:`list[str]` or :obj:`None`, default=None
        A list of names to save the dictionaries with changed dtypes as. Names are matched to
        dictionaries by position (e.g., a file name in the 0th position will be the file name for
        the dictionary in the 0th position of ``subject_timeseries_list``). If None and
        ``output_dir`` is specified, uses default file names - "subject_timeseries_{0}_float{1}.pkl"
        (where {0} indicates the original input order and {1} is the dtype.

    Returns
    -------
    dict[str, SubjectTimeseries]
        A nested dictionary containing the converted subject timeseries if ``return_dicts`` is True.

    See Also
    --------
    :data:`neurocaps.typing.SubjectTimeseries`
        Type definition for the subject timeseries dictionary structure.

    Warning
    -------
    **Floating Point Precision**: The minimum recommended floating-point dtype is *float32*, as
    lower precision may introduce rounding errors that affect calculations.
    """
    assert isinstance(subject_timeseries_list, list), "`subject_timeseries_list` must be a list."

    io_utils.issue_file_warning("filenames", filenames, output_dir)

    changed_dtype_dicts = {}

    for indx, curr_dict in enumerate(subject_timeseries_list):
        curr_dict = io_utils.get_obj(curr_dict)

        for subj_id in curr_dict:
            for run in curr_dict[subj_id]:
                curr_dict[subj_id][run] = curr_dict[subj_id][run].astype(dtype)

        changed_dtype_dicts[f"dict_{indx}"] = curr_dict

    if output_dir:
        serialize.dicts_to_pickles(
            output_dir=output_dir,
            dict_list=changed_dtype_dicts,
            filenames=filenames,
            call="change_dtype",
        )

    if return_dicts:
        return changed_dtype_dicts
