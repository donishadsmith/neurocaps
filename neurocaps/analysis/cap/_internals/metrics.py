"""Internal module for computing temporal dynamic metrics."""

import itertools, os
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurocaps._utils import io as io_utils
from neurocaps._utils.logging import setup_logger

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
        formatted_string = ", ".join(["'{a}'".format(a=x) for x in set_diff])
        LG.warning(f"The following invalid metrics will be ignored: {formatted_string}.")

    if not ordered_metrics:
        formatted_string = ", ".join(["'{a}'".format(a=x) for x in valid_metrics])
        raise ValueError(
            f"No valid metrics in `metrics` list. Valid metrics are: {formatted_string}."
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
    max_cap = max([int(name.split("-")[-1]) for name in cap_names])

    return cap_names, max_cap, group_cap_counts


def create_transition_pairs(
    cap_dict: dict[str, dict[str, NDArray]],
) -> dict[str, list[tuple[int, int]]]:
    """Obtains all possible transition pairs."""
    group_caps = {}
    all_pairs = {}

    for group_name in cap_dict:
        group_caps.update({group_name: [int(name.split("-")[-1]) for name in cap_dict[group_name]]})
        all_pairs.update(
            {group_name: list(itertools.product(group_caps[group_name], group_caps[group_name]))}
        )

    return all_pairs


def build_df(
    metrics: list[str],
    group_names: list[str],
    cap_names: list[str],
    pairs: dict[str, list[tuple[int, int]]],
) -> dict[str, pd.DataFrame]:
    """Initializes the output dataframes with column names for each requested metric."""
    df_dict = {}
    base_cols = ["Subject_ID", "Group", "Run"]

    for metric in metrics:
        if metric not in ["transition_frequency", "transition_probability"]:
            df_dict.update({metric: pd.DataFrame(columns=base_cols + list(cap_names))})
        elif metric == "transition_probability":
            df_dict[metric] = {}
            for group in group_names:
                col_names = base_cols + [f"{x}.{y}" for x, y in pairs[group]]
                df_dict[metric].update({group: pd.DataFrame(columns=col_names)})
        else:
            df_dict.update({metric: pd.DataFrame(columns=base_cols + ["Transition_Frequency"])})

    return df_dict


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


def compute_temporal_fraction(
    arr: NDArray, sub_info: list[str], df: pd.DataFrame, n_group_caps: int, max_cap: int
) -> pd.DataFrame:
    """
    Computes temporal fraction for the subject and run specified in ``sub_info`` and inserts new
    row in the dataframe.
    """
    frequency_dict = {key: np.where(arr == key, 1, 0).sum() for key in range(1, n_group_caps + 1)}

    frequency_dict = add_nans_to_dict(max_cap, n_group_caps, frequency_dict)

    proportion_dict = {key: value / (len(arr)) for key, value in frequency_dict.items()}

    return append_df(df, sub_info, proportion_dict)


def compute_counts(
    arr: NDArray, sub_info: list[str], df: pd.DataFrame, n_group_caps: int, max_cap: int
) -> pd.DataFrame:
    """
    Computes counts for the subject and run specified in ``sub_info`` and inserts new row in the
    dataframe.
    """
    count_dict = {}
    for target in range(1, n_group_caps + 1):
        if target in arr:
            _, counts = segments(target, arr)
            count_dict.update({target: counts})
        else:
            count_dict.update({target: 0})

    count_dict = add_nans_to_dict(max_cap, n_group_caps, count_dict)

    return append_df(df, sub_info, count_dict)


def compute_persistence(
    arr: NDArray,
    sub_info: list[str],
    df: pd.DataFrame,
    n_group_caps: int,
    max_cap: int,
    tr: Union[float, int, None],
) -> pd.DataFrame:
    """
    Computes persistence for the subject and run specified in ``sub_info`` and inserts new row
    in the dataframe.
    """
    persistence_dict = {}

    # Iterate through caps
    for target in range(1, n_group_caps + 1):
        binary_arr, n_segments = segments(target, arr)
        # ``n_segments`` returns minimum of 1 so persistence is 0 instead of NaN when CAP not in
        # timeseries
        persistence_dict.update({target: (binary_arr.sum() / n_segments) * (tr if tr else 1)})

    persistence_dict = add_nans_to_dict(max_cap, n_group_caps, persistence_dict)

    return append_df(df, sub_info, persistence_dict)


def segments(target: int, timeseries: NDArray) -> tuple[NDArray[np.bool_], int]:
    """
    Computes the number of segments for persistence and counts computation. Always returns
    1 for number of segments to prevent NaN due to divide by 0.

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

    return binary_arr, n_segments


def add_nans_to_dict(
    max_cap: int, n_group_caps: int, curr_dict: dict[str, Union[float, int]]
) -> dict[str, Union[float, int]]:
    """Adds NaN for groups with less caps than the group with the greatest number of caps."""
    if max_cap > n_group_caps:
        for i in range(n_group_caps + 1, max_cap + 1):
            curr_dict.update({i: float("nan")})

    return curr_dict


def append_df(
    df: pd.DataFrame, sub_info: list[str], metric_dict: dict[str, Union[float, int]]
) -> pd.DataFrame:
    """Appends new row in dataframe."""
    df.loc[len(df)] = sub_info + [items for items in metric_dict.values()]
    return df


def compute_transition_frequency(
    arr: NDArray, sub_info: list[str], df: pd.DataFrame
) -> pd.DataFrame:
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

    df.loc[len(df)] = sub_info + [transition_frequency]

    return df


def compute_transition_probability(
    arr: NDArray, sub_info: list[str], df: pd.DataFrame, cap_pairs: list[tuple[int, int]]
) -> pd.DataFrame:
    """
    Computes transition probability for the subject and run specified in ``sub_info`` and
    inserts new row in the dataframe.
    """
    df.loc[len(df)] = sub_info + [0.0] * (df.shape[-1] - 3)

    # Arrays for transitioning from and to element
    trans_from = arr[:-1]
    trans_to = arr[1:]

    # Iterate through pairs and calculate probability
    for e1, e2 in cap_pairs:
        # Get total number of possible transitions for first element
        total_trans = np.sum(trans_from == e1)
        column = f"{e1}.{e2}"
        # Compute sum of adjacent pairs of A -> B and divide
        df.loc[df.index[-1], column] = (
            np.sum((trans_from == e1) & (trans_to == e2)) / total_trans if total_trans > 0 else 0
        )

    return df


def save_metrics(
    output_dir: str,
    group_names: list[str],
    df_dict: dict[str, pd.DataFrame],
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
