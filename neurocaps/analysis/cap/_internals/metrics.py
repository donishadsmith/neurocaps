"""Internal module for computing temporal dynamic metrics."""

import itertools, os
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurocaps.utils import _io as io_utils
from neurocaps.utils._helpers import list_to_str
from neurocaps.utils._logging import setup_logger

LG = setup_logger(__name__)


def filter_metrics(metrics: Union[list[str], tuple[str], None]) -> list[str]:
    """
    Filters metrics to ensure only the supported metrics ("temporal_fraction", "persistence",
    "counts", "transition_frequency") are in the list. Maintains the order of the original
    user-specified metrics.
    """
    metrics = (
        ("temporal_fraction", "persistence", "counts", "transition_frequency")
        if metrics is None
        else metrics
    )
    metrics = [metrics] if isinstance(metrics, str) else metrics
    metrics_set = set(metrics)

    valid_metrics = {
        "temporal_fraction",
        "persistence",
        "counts",
        "transition_frequency",
        "transition_probability",
    }
    set_diff = metrics_set - valid_metrics
    metrics_set = metrics_set.intersection(valid_metrics)
    # Ensure original order maintained
    ordered_metrics = [metric for metric in metrics if metric in metrics_set]

    if set_diff:
        LG.warning(f"The following invalid metrics will be ignored: {list_to_str(set_diff)}.")

    if not ordered_metrics:
        raise ValueError(
            f"No valid metrics in `metrics` list. Valid metrics are: {list_to_str(valid_metrics)}."
        )

    return ordered_metrics


def extract_caps_info(
    cap_dict: dict[str, dict[str, NDArray]],
) -> tuple[list[str], int, dict[str, int]]:
    """
    Extracts CAP-related information, specifically the names of the CAPs (i.e. CAP-1, CAP-2,
    etc), the maximum number of CAPs found across all groups (e.g if Group A has 4 CAPs and
    group B has 5 CAPs then 5 is returned), and the number of CAPs for each group.
    """
    group_cap_counts = {}

    for group_name in cap_dict:
        # Store the length of caps in each group
        group_cap_counts.update({group_name: len(cap_dict[group_name])})

    # CAP names based on groups with the most CAPs
    cap_names = list(cap_dict[max(group_cap_counts, key=group_cap_counts.get)])
    max_cap = max(group_cap_counts.values())

    return cap_names, max_cap, group_cap_counts


def create_transition_pairs(
    cap_dict: dict[str, dict[str, NDArray]],
) -> dict[str, list[tuple[int, int]]]:
    """Obtains all possible transition pairs."""
    group_caps = {
        group_name: [get_cap_id(cap_name) for cap_name in cap_dict[group_name]]
        for group_name in cap_dict
    }

    return {
        group_name: list(itertools.product(cap_id, cap_id))
        for group_name, cap_id in group_caps.items()
    }


def get_cap_id(cap_name: str) -> int:
    """
    Extracts the CAP name, assumed to be in the form `CAP-{n}` (i.e. CAP-1) and returns the int.
    """
    return int(cap_name.split("-")[-1])


def create_columns_names(
    metrics: list[str],
    group_names: list[str],
    cap_names: list[str],
    pairs: dict[str, list[tuple[int, int]]],
) -> dict[str, Union[list[str], dict[str, list[str]]]]:
    """
    Creates the column names for each requested metric. Used downstream has the column
    names for each metrics dataframe.
    """
    columns_names_dict = {}
    base_cols = ["Subject_ID", "Group", "Run"]

    for metric in metrics:
        if metric not in ["transition_frequency", "transition_probability"]:
            columns_names_dict.update({metric: base_cols + list(cap_names)})
        elif metric == "transition_probability":
            columns_names_dict[metric] = {}
            for group_name in group_names:
                col_names = base_cols + [f"{x}.{y}" for x, y in pairs[group_name]]
                columns_names_dict[metric].update({group_name: col_names})
        else:
            columns_names_dict.update({metric: base_cols + ["Transition_Frequency"]})

    return columns_names_dict


def initialize_all_metrics_dict(
    metrics: list[str], group_names: list[str]
) -> dict[str, Union[list, dict[str, list]]]:
    """
    Initializes a dictionary intended to store all computations for a metric across all
    subjects. The dictionary will be converted to a dataframe downstream.
    """
    all_metrics_dict = {}
    for metric in metrics:
        if metric != "transition_probability":
            all_metrics_dict[metric] = []
        else:
            all_metrics_dict["transition_probability"] = {
                group_name: [] for group_name in group_names
            }

    return all_metrics_dict


def create_distributed_dict(
    subject_table: dict[str, str], predicted_subject_timeseries: dict[str, NDArray]
) -> dict[str, list[tuple[str, str]]]:
    """Creates a dictionary mapping for each subject and run pair to iterate over."""
    distributed_dict = {}

    for subj_id, group_name in subject_table.items():
        distributed_dict[subj_id] = []
        for curr_run in predicted_subject_timeseries[subj_id]:
            distributed_dict[subj_id].append((group_name, curr_run))

    return distributed_dict


def compute_temporal_fraction(arr: NDArray, n_caps: int) -> dict[str, float]:
    """
    Computes temporal fraction for the subject and run specified in ``sub_info`` and inserts new
    row in the dataframe. Assumes one-based values in ``arr``.
    """
    frequency_dict = {key: np.where(arr == key, 1, 0).sum() for key in range(1, n_caps + 1)}
    proportion_dict = {key: value / (len(arr)) for key, value in frequency_dict.items()}

    return proportion_dict


def compute_counts(arr: NDArray, n_caps: int) -> dict[str, int]:
    """
    Computes counts for the subject and run specified in ``sub_info`` and inserts new row in the
    dataframe. Assumes one-based values in ``arr``.
    """
    count_dict = {}
    for target in range(1, n_caps + 1):
        if target in arr:
            _, counts = segments(target, arr)
            count_dict.update({target: counts})
        else:
            count_dict.update({target: 0})

    return count_dict


def compute_persistence(arr: NDArray, n_caps: int, tr: Union[float, int, None]) -> dict[str, float]:
    """
    Computes persistence for the subject and run specified in ``sub_info`` and inserts new row
    in the dataframe. Assumes one-based values in ``arr``.
    """
    persistence_dict = {}

    # Iterate through caps
    for target in range(1, n_caps + 1):
        # Floor n_segments at one to prevent nan due to division by 0
        binary_arr, n_segments = segments(arr, target, floor_at_one=True)
        persistence_dict.update({target: (binary_arr.sum() / n_segments) * (tr if tr else 1)})

    return persistence_dict


def segments(
    timeseries: NDArray, target: int, floor_at_one: bool = False
) -> tuple[NDArray[np.bool_], int]:
    """
    Computes the number of segments for persistence and counts computation. If ``floor_at_one``,
    then a minimum of 1 for number of segments is returned to prevent NaN due to divide by 0
    when computing persistence in which 0 is preferred to represent the absence of a CAP.

    Example Computation
    -------------------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 1, 1, 1, 3])
    >>> target = 1
    >>> binary_arr = np.where(timeseries == target, 1, 0) # [1, 0, 1, 1, 1, 0]
    >>> target_indices = np.where(binary_arr == 1)[0] # [0, 2, 3, 4]
    >>> diff_arr = np.diff(target_indices, n=1) # [2, 1, 1]
    >>> n_transitions = np.where(diff_arr > 1, 1, 0).sum() # 1
    >>> n_segments += 1 # Account for first segment
    """
    binary_arr = np.where(timeseries == target, 1, 0)
    target_indices = np.where(binary_arr == 1)[0]
    n_segments = np.where(np.diff(target_indices, n=1) > 1, 1, 0).sum() + 1

    if not floor_at_one:
        n_segments = n_segments if binary_arr.sum() != 0 else 0

    return binary_arr, n_segments


def add_nans_to_dict(
    max_cap: int, n_group_caps: int, curr_dict: dict[str, Union[float, int]]
) -> dict[str, Union[float, int]]:
    """Adds NaN for groups with less CAPs than the group with the greatest number of CAPs."""
    if max_cap > n_group_caps:
        for i in range(n_group_caps + 1, max_cap + 1):
            curr_dict.update({i: float("nan")})

    return curr_dict


def convert_dict_to_df(
    columns_names_dict: dict[str, Union[list[str], dict[str, list[str]]]],
    all_metrics_dict: dict[str, Union[list, dict[str, list]]],
) -> dict[str, Union[pd.DataFrame, dict[str, pd.DataFrame]]]:
    """
    Appends the data ``all_metrics_dict`` to its respective dataframe in ``df_dict``.
    """
    df_dict = {}
    for metric_name in columns_names_dict:
        if metric_name != "transition_probability":
            df = pd.DataFrame.from_records(
                all_metrics_dict[metric_name], columns=columns_names_dict[metric_name]
            )
            df_dict[metric_name] = df
        else:
            df_dict["transition_probability"] = {}
            for group_name in columns_names_dict["transition_probability"]:
                df = pd.DataFrame.from_records(
                    all_metrics_dict[metric_name][group_name],
                    columns=columns_names_dict[metric_name][group_name],
                )
                df_dict[metric_name].update({group_name: df})

    return df_dict


def compute_transition_frequency(arr: NDArray) -> dict[str, int]:
    """
    Computes transition frequency for the subject and run specified in ``sub_info`` and inserts
    new row in the dataframe.

    Example Computation
    -------------------
    >>> import numpy as np
    >>> arr = np.array([1, 2, 1, 1, 1, 3])
    >>> n_trans = np.where(np.diff(arr, n=1) != 0, 1, 0).sum()
    """
    transition_frequency = np.where(np.diff(arr, n=1) != 0, 1, 0).sum()

    return {"transition_frequency": transition_frequency}


def compute_transition_probability(
    arr: NDArray, cap_pairs: list[tuple[int, int]]
) -> dict[str, float]:
    """
    Computes transition probability for the subject and run specified in ``sub_info`` and
    inserts new row in the dataframe.
    """
    trans_prob_dict = {}

    # Arrays for transitioning from and to element
    trans_from = arr[:-1]
    trans_to = arr[1:]

    # Iterate through pairs and calculate probability
    for e1, e2 in cap_pairs:
        # Get total number of possible transitions for first element
        total_trans = np.sum(trans_from == e1)
        # Compute sum of adjacent pairs of A -> B and divide
        trans_prob_dict[f"{e1}.{e2}"] = (
            np.sum((trans_from == e1) & (trans_to == e2)) / total_trans if total_trans > 0 else 0
        )

    return trans_prob_dict


def save_metrics(
    output_dir: str,
    group_names: list[str],
    df_dict: dict[str, Union[pd.DataFrame, dict[str, pd.DataFrame]]],
    prefix_filename: Union[str, None],
) -> None:
    """Saves the metric dataframes as csv files."""
    if not output_dir:
        return None

    for metric in df_dict:
        filename = io_utils.filename(base_name=f"{metric}", add_name=prefix_filename, pos="prefix")
        if metric != "transition_probability":
            df_dict[f"{metric}"].to_csv(
                path_or_buf=os.path.join(output_dir, f"{filename}.csv"), sep=",", index=False
            )
        else:
            for group_name in group_names:
                df_dict[f"{metric}"][group_name].to_csv(
                    path_or_buf=os.path.join(
                        output_dir, f"{filename}-{group_name.replace(' ', '_')}.csv"
                    ),
                    sep=",",
                    index=False,
                )
