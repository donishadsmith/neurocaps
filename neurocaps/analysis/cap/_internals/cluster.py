"""Internal module containing helper functions for ``CAP.get_caps``."""

from typing import Any, Union

import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from joblib import Parallel, delayed
from numpy.typing import NDArray
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from tqdm.auto import tqdm

from neurocaps.extraction._internals.postprocess import standardize_rois
from neurocaps.exceptions import NoElbowDetectedError
from neurocaps.typing import SubjectTimeseries
from neurocaps.utils import _io as io_utils
from neurocaps.utils._helpers import resolve_kwargs
from neurocaps.utils._logging import setup_logger
from neurocaps.utils._plotting_utils import PlotDefaults, PlotFuncs

LG = setup_logger(__name__)


def perform_kmeans(n_cluster: int, configs: dict, concatenated_timeseries: NDArray, method: str):
    """
    Uses scikit-learn to perform k-means clustering on concatenated timeseries data in both
    sequential and parallel contexts. Also uses scikit-learn to provide cluster performance metrics.
    """
    model = KMeans(n_clusters=n_cluster, **configs, verbose=0).fit(concatenated_timeseries)

    # Only return model when no cluster selection chosen
    if method is None:
        return model

    cluster_labels = model.labels_
    if method == "davies_bouldin":
        performance = {n_cluster: davies_bouldin_score(concatenated_timeseries, cluster_labels)}
    elif method == "elbow":
        performance = {n_cluster: model.inertia_}
    elif method == "silhouette":
        performance = {
            n_cluster: silhouette_score(concatenated_timeseries, cluster_labels, metric="euclidean")
        }
    else:
        # Variance Ratio
        performance = {n_cluster: calinski_harabasz_score(concatenated_timeseries, cluster_labels)}

    model_dict = {n_cluster: model}

    return performance, model_dict


def setup_groups(
    subject_timeseries: SubjectTimeseries, groups_dict: Union[dict[str, str], None]
) -> tuple[dict[str, str], dict[str, str]]:
    """Used to resolve ``self._groups`` and ``self._subject_table``."""
    if groups_dict is None:
        groups_dict = create_default_group(subject_timeseries)

    groups_dict = sort_subject_ids(groups_dict)

    subject_table = generate_lookup_table(groups_dict)

    return groups_dict, subject_table


def create_default_group(subject_timeseries: SubjectTimeseries):
    """
    Creates a dictionary mapping the default group "All Subjects" to the subject IDs in the
    SubjectTimeseries.
    """
    # Create dictionary for "All Subjects" if no groups are specified
    LG.info(
        "No groups specified. Using default group 'All Subjects' containing all subject "
        "IDs from `subject_timeseries`. The `groups` dictionary will remain fixed "
        "unless the `CAP` class is re-initialized or `clear_groups()` is used."
    )

    return {"All Subjects": list(subject_timeseries)}


def sort_subject_ids(group_dict: dict[str, str]) -> dict[str, str]:
    """Sort IDs lexicographically (also done in ``TimeseriesExtractor``)."""
    return {group: sorted(group_dict[group]) for group in group_dict}


def generate_lookup_table(group_dict: dict[str, str]) -> dict[str, str]:
    """Creates dictionary mapping subject IDs to their associated group."""
    subject_table = {}

    for group in group_dict:
        for subj_id in group_dict[group]:
            if subj_id in subject_table:
                LG.warning(
                    f"[SUBJECT: {subj_id}] Appears more than once. Only the first instance of "
                    "this subject will be included in the analysis."
                )
            else:
                subject_table.update({subj_id: group})

    return subject_table


def create_group_map(subject_table: dict[str, str], group_dict: dict[str, str]) -> dict[str, str]:
    """
    Create a new group_dict by intersecting ``subject_table`` and ``group_dict``.
    Done since ``subject_table`` will contain unique subject IDs but ``group_dict`` may have
    subject IDs that repeat across groups. Output used for tqdm progress bars.
    """
    # Intersect subjects in subjects table and the groups for tqdm
    group_map = {
        group_name: sorted(set(subject_table).intersection(group_dict[group_name]))
        for group_name in group_dict
    }

    return group_map


def concatenate_timeseries(
    subject_timeseries: SubjectTimeseries,
    group_dict: dict[str, str],
    runs: Union[list[int], list[str], None],
    progress_bar: bool,
) -> dict[str, NDArray]:
    """
    Concatenates the timeseries data of all subjects into a single numpy array if ``groups`` are
    None or group-specific numpy arrays.
    """
    # Collect timeseries data in lists
    group_arrays = {group_name: [] for group_name in group_dict}
    for group_name in group_dict:
        for subj_id in tqdm(
            group_dict[group_name],
            desc=f"Collecting Subject Timeseries Data [GROUP: {group_name}]",
            disable=not progress_bar,
        ):
            subject_runs, miss_runs = get_runs(runs, list(subject_timeseries[subj_id]))

            if miss_runs:
                LG.warning(
                    f"[SUBJECT: {subj_id}] Does not have the requested runs: "
                    f"{', '.join(miss_runs)}."
                )

            if not subject_runs:
                LG.warning(
                    f"[SUBJECT: {subj_id}] Excluded from the concatenated timeseries due to "
                    "having no runs."
                )
                continue

            subj_arrays = [subject_timeseries[subj_id][run_id] for run_id in subject_runs]
            group_arrays[group_name].extend(subj_arrays)

    # Only stack once per group; avoid bottleneck due to repeated calls on large data
    concatenated_timeseries = {group_name: None for group_name in group_dict}
    for group_name in tqdm(
        group_arrays, desc="Concatenating Timeseries Data Per Group", disable=not progress_bar
    ):
        concatenated_timeseries[group_name] = np.vstack(group_arrays[group_name])

    del group_arrays

    return concatenated_timeseries


def scale(concatenated_timeseries: dict[str, NDArray]) -> dict[str, NDArray]:
    """Scales the concatenated timeseries data."""
    mean_vec = {group_name: None for group_name in concatenated_timeseries}
    stdev_vec = {group_name: None for group_name in concatenated_timeseries}

    for group_name in concatenated_timeseries:
        concatenated_timeseries[group_name], mean_vec[group_name], stdev_vec[group_name] = (
            standardize_rois(concatenated_timeseries[group_name], return_parameters=True)
        )

    return concatenated_timeseries, mean_vec, stdev_vec


def get_runs(
    requested_runs: Union[list[int], list[str], None], curr_runs: list[str]
) -> tuple[list[str], Union[list[str], None]]:
    """
    Filters the current runs available for a subject if specific runs are requested.
    Also returns a list of missing runs that were requested
    """
    if requested_runs:
        requested_runs = [str(run).removeprefix("run-") for run in requested_runs]
        requested_runs = [f"run-{run}" for run in requested_runs]

    runs = [run for run in requested_runs if run in curr_runs] if requested_runs else curr_runs
    miss_runs = list(set(requested_runs) - set(runs)) if requested_runs else None

    return runs, miss_runs


def select_optimal_clusters(
    concatenated_timeseries_dict,
    method: str,
    n_clusters: list[int],
    n_cores: Union[int, None],
    configs: dict[str, Any],
    show_figs: bool,
    output_dir: Union[str, None],
    progress_bar: bool,
    as_pickle: bool,
    **kwargs,
) -> None:
    """Selects optimal number of clusters based on the specific ``cluster_selection_method``."""
    cluster_scores = {}
    optimal_n_clusters = {}
    kmeans = {}
    performance_dict = {}

    for group_name in concatenated_timeseries_dict:
        performance_dict[group_name] = {}
        model_dict = {}

        if n_cores is None:
            for n_cluster in tqdm(
                n_clusters, desc=f"Clustering [GROUP: {group_name}]", disable=not progress_bar
            ):
                output_score, model = perform_kmeans(
                    n_cluster, configs, concatenated_timeseries_dict[group_name], method
                )
                performance_dict[group_name].update(output_score)
                model_dict.update(model)
        else:
            parallel = Parallel(
                return_as="generator",
                n_jobs=n_cores,
                backend="loky",
                max_nbytes=kwargs.get("max_nbytes", "1M"),
            )
            outputs = tqdm(
                parallel(
                    delayed(perform_kmeans)(
                        n_cluster, configs, concatenated_timeseries_dict[group_name], method
                    )
                    for n_cluster in n_clusters
                ),
                desc=f"Clustering [GROUP: {group_name}]",
                total=len(n_clusters),
                disable=not progress_bar,
            )

            output_scores, models = zip(*outputs)
            for output in output_scores:
                performance_dict[group_name].update(output)
            for model in models:
                model_dict.update(model)

        # Select optimal clusters
        if method == "elbow":
            kneedle = KneeLocator(
                x=list(performance_dict[group_name]),
                y=list(performance_dict[group_name].values()),
                curve="convex",
                direction="decreasing",
                S=kwargs.get("S", 1.0),
            )

            optimal_n_clusters[group_name] = kneedle.elbow

            if optimal_n_clusters[group_name] is None:
                raise NoElbowDetectedError(
                    f"[GROUP: {group_name}] - No elbow detected. Try adjusting the sensitivity "
                    "parameter (`S`) to increase or decrease sensitivity (higher values "
                    "are less sensitive), expanding the list of `n_clusters` to test, or "
                    "using another `cluster_selection_method`."
                )
        elif method == "davies_bouldin":
            # Get minimum for davies bouldin
            optimal_n_clusters[group_name] = min(
                performance_dict[group_name], key=performance_dict[group_name].get
            )
        else:
            # Get max for silhouette and variance ratio
            optimal_n_clusters[group_name] = max(
                performance_dict[group_name], key=performance_dict[group_name].get
            )

        # Get the optimal kmeans model
        kmeans[group_name] = model_dict[optimal_n_clusters[group_name]]

        LG.info(
            f"[GROUP: {group_name} | METHOD: {method}] Optimal cluster size is "
            f"{optimal_n_clusters[group_name]}."
        )

        if show_figs or output_dir is not None:
            # Create plot dictionary
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["S", "max_nbytes"]}
            plot_dict = resolve_kwargs(PlotDefaults.get_caps(), **filtered_kwargs)

            fig = plot_cluster_performance(
                method,
                group_name,
                performance_dict[group_name],
                optimal_n_clusters[group_name] if method == "elbow" else None,
                show_figs,
                plot_dict,
            )

            save_cluster_performance_figure(
                fig, output_dir, group_name, method, as_pickle, plot_dict
            )

    cluster_scores = {"Cluster_Selection_Method": method}
    cluster_scores.update({"Scores": performance_dict})

    return optimal_n_clusters, kmeans, cluster_scores


def plot_cluster_performance(
    method: str,
    group_name: str,
    performance_dict: dict[str, float],
    optimal_n_clusters: Union[int, None],
    show_figs: bool,
    plot_dict: dict[str, Any],
) -> None:
    """Plots results of the specific ``cluster_selection_method``."""
    y_titles = {
        "elbow": "Inertia",
        "davies_bouldin": "Davies Bouldin Score",
        "silhouette": "Silhouette Score",
        "variance_ratio": "Variance Ratio Score",
    }

    plt.figure(figsize=plot_dict["figsize"])

    x_values = list(performance_dict)
    y_values = [y for _, y in performance_dict.items()]
    plt.plot(x_values, y_values)

    if plot_dict["step"]:
        x_ticks = range(x_values[0], x_values[-1] + 1, plot_dict["step"])
        plt.xticks(x_ticks)

    plt.title(group_name)
    plt.xlabel("K")

    y_title = y_titles[method]
    plt.ylabel(y_title)
    # Add vertical line for elbow method
    if y_title == "Inertia":
        plt.vlines(
            optimal_n_clusters,
            plt.ylim()[0],
            plt.ylim()[1],
            linestyles="--",
            label="elbow",
        )

    fig = plt.gcf()

    PlotFuncs.show(show_figs)

    return fig


def save_cluster_performance_figure(
    fig: Figure,
    output_dir: Union[str, None],
    group_name: str,
    method_name: str,
    as_pickle: bool,
    plot_dict: dict[str, Any],
) -> None:
    """Saves the cluster performance plot if ``output_dir`` is not falsy."""
    if not output_dir:
        return None

    io_utils.makedir(output_dir)

    save_name = f"{group_name.replace(' ', '_')}_{method_name}.png"
    PlotFuncs.save_fig(fig, output_dir, save_name, plot_dict, as_pickle)


def compute_variance_explained(
    concatenated_timeseries_dict: dict[str, NDArray], kmeans: dict[str, KMeans]
) -> dict[str, float]:
    """Computes variance explained in the concatenated timeseries by clustering."""
    variance_explained_dict = {}

    for group_name in concatenated_timeseries_dict:
        mean_vec = np.mean(concatenated_timeseries_dict[group_name], axis=0)
        total_var = np.sum((concatenated_timeseries_dict[group_name] - mean_vec) ** 2)
        explained_var = 1 - (kmeans[group_name].inertia_ / total_var)
        variance_explained_dict[group_name] = explained_var

    return variance_explained_dict


def create_caps_dict(kmeans_dict: dict[str, KMeans]) -> dict[str, NDArray]:
    """Maps groups to their CAPs (cluster centroids)."""
    caps_dict = {}

    for group_names in kmeans_dict:
        caps_dict[group_names] = {}
        cluster_centroids = zip(
            [num for num in range(1, len(kmeans_dict[group_names].cluster_centers_) + 1)],
            kmeans_dict[group_names].cluster_centers_,
        )
        caps_dict[group_names].update(
            {
                f"CAP-{state_number}": state_vector
                for state_number, state_vector in cluster_centroids
            }
        )

    return caps_dict
