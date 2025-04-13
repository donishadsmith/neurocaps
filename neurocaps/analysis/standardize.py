"""Function to standardize timeseries within subject runs."""

import copy
from typing import Union, Optional

from ..typing import SubjectTimeseries
from .._utils import _convert_pickle_to_dict, _dicts_to_pickles, _logger, _standardize

LG = _logger(__name__)


def standardize(
    subject_timeseries_list: Union[list[SubjectTimeseries], list[str]],
    return_dicts: bool = True,
    output_dir: Optional[str] = None,
    filenames: Optional[list[str]] = None,
) -> Union[dict[str, SubjectTimeseries], None]:
    """
    Perform Participant-wise Timeseries Standardization Within Runs.

    Standardizes the columns/ROIs of each run independently for all subjects in the subject timeseries. Uses sample
    standard deviation with Bessel's correction (`n-1` in denominator). Primarily to be used when standardizing was not
    done in ``TimeseriesExtractor``.

    .. note:: Standard deviations below ``np.finfo(std.dtype).eps`` are replaced with 1 for numerical stability.

    Parameters
    ----------
    subject_timeseries_list: :obj:`list[SubjectTimeseries]` or :obj:`list[str]`
        A list where each element consist of a dictionary mapping subject IDs to their run IDs and associated
        timeseries (TRs x ROIs) as a NumPy array. Can also be a list consisting of paths to pickle files
        containing this same structure. Refer to documentation for ``SubjectTimeseries`` in the "See Also" section for
        an example structure.

    return_dicts: :obj:`bool`, default=True
        If True, returns a single dictionary containing the standardized input dictionaries. Keys are named "dict_{0}"
        where {0} corresponds to the dictionary's position in the input list.

    output_dir: :obj:`str` or :obj:`None`, default=None
        Directory to save the standardized dictionaries as pickle files. The directory will be created if it does not
        exist. Dictionaries will not be saved if None.

    filenames: :obj:`list[str]` or :obj:`None`, default=None
        A list of names to save the standardized dictionaries as. Names are matched to dictionaries by position (e.g.,
        a file name in the 0th position will be the file name for the dictionary in the 0th position of
        ``subject_timeseries_list``). If None and ``output_dir`` is specified, uses default file names -
        "subject_timeseries_{0}_standardized.pkl" (where {0} indicates the original input order).

    Returns
    -------
    dict[str, SubjectTimeseries]
        A nested dictionary containing the standardized subject timeseries if ``return_dicts`` is True.

    See Also
    --------
    :data:`neurocaps.typing.SubjectTimeseries`
        Type definition for the subject timeseries dictionary structure. Refer to the `SubjectTimeseries
        documentation <https://neurocaps.readthedocs.io/en/stable/generated/neurocaps.typing.SubjectTimeseries.html#neurocaps.typing.SubjectTimeseries>`_.
    """
    assert (
        isinstance(subject_timeseries_list, list) and len(subject_timeseries_list) > 0
    ), "`subject_timeseries_list` must be a list greater than length 0."

    if filenames is not None and output_dir is None:
        LG.warning("`filenames` supplied but no `output_dir` specified. Files will not be saved.")

    standardized_dicts = {}

    for indx, curr_dict in enumerate(subject_timeseries_list):
        if isinstance(curr_dict, str):
            curr_dict = _convert_pickle_to_dict(curr_dict)
        else:
            curr_dict = copy.deepcopy(curr_dict)

        for subj_id in curr_dict:
            for run in curr_dict[subj_id]:
                curr_dict[subj_id][run] = _standardize(curr_dict[subj_id][run])

        standardized_dicts[f"dict_{indx}"] = curr_dict

    if output_dir:
        _dicts_to_pickles(output_dir=output_dir, dict_list=standardized_dicts, filenames=filenames, call="standardize")

    if return_dicts:
        return standardized_dicts
