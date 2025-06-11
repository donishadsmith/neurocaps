"""Contains the CAP class for performing co-activation patterns analyses"""

import collections, itertools, os, re, sys, tempfile
from typing import Any, Callable, Literal, Optional, Union
from typing_extensions import Self

import nibabel as nib, numpy as np, matplotlib.pyplot as plt, pandas as pd, seaborn, surfplot
import plotly.express as px, plotly.graph_objects as go, plotly.offline as pyo
from kneed import KneeLocator
from joblib import Parallel, delayed
from nilearn.plotting.cm import _cmap_d
from neuromaps.transforms import mni152_to_fslr, fslr_to_fslr
from neuromaps.datasets import fetch_fslr
from numpy.typing import ArrayLike, NDArray
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

import neurocaps._utils.io as io_utils
from ..exceptions import NoElbowDetectedError
from ..typing import ParcelConfig, ParcelApproach, SubjectTimeseries
from .._utils import (
    _CAPGetter,
    _MatrixVisualizer,
    _PlotDefaults,
    _PlotFuncs,
    _cap2statmap,
    _check_kwargs,
    _check_parcel_approach,
    _collapse_aal_node_names,
    _logger,
    _run_kmeans,
    _standardize,
)

LG = _logger(__name__)


class CAP(_CAPGetter):
    """
    Co-Activation Patterns (CAPs) Class.

    Performs k-means clustering for CAP identification, computes temporal dynamics metrics, and
    provides visualization tools for analyzing brain activation patterns.

    Parameters
    ----------
    parcel_approach: :obj:`ParcelConfig`, :obj:`ParcelApproach`, or :obj:`str`, default=None
        Specifies the parcellation approach to use. Options are "Schaefer", "AAL", or "Custom". Can
        be initialized with parameters, as a nested dictionary, or loaded from a pickle file. For
        detailed documentation on the expected structure, see the type definitions for ``ParcelConfig``
        and ``ParcelApproach`` in the "See Also" section.

    groups: :obj:`dict[str, list[str]]` or :obj:`None`, default=None
        Optional mapping of group names to lists of subject IDs for group-specific analyses. If
        None, on the first call of ``self.get_caps()``, "All Subjects" will be set as the default
        group name and be populated with the subject IDs in ``subject_timeseries``. Groups remain
        fixed for the entire instance of the class unless ``self.clear_groups()`` is used.


    Properties
    ----------
    parcel_approach: :obj:`ParcelApproach`
        Parcellation information with "maps" (path to parcellation file), "nodes" (labels), and
        "regions" (anatomical regions or networks). This property is also settable (accepts a
        dictionary or pickle file). Returns a deep copy.

    groups: :obj:`dict[str, list[str]]` or :obj:`None`:
        Mapping of groups names to lists of subject IDs. Returns a deep copy.

    subject_table: :obj:`dict[str, str]` or :obj:`None`
        Lookup table mapping subject IDs to their groups. Derived from ``self.groups`` each time
        ``self.get_caps()`` is ran. While this property can be modified using its setter, any
        changes will be overwritten based on ``self.groups`` on the subsequent call to
        ``self.get_caps()``. Returns a deep copy.

    n_clusters: :obj:`int`, :obj:`list[int]`, or :obj:`None`
        An integer or list of integers representing the number of clusters used for k-means.
        Defined after running ``self.get_caps()``.

    cluster_selection_method: :obj:`str` or :obj:`None`:
        Method used to identify the optimal number of clusters. Defined after running
        ``self.get_caps()``.

    n_cores: :obj:`int` or :obj:`None`
        Number of cores specified used for multiprocessing with Joblib. Defined after running
        ``self.get_caps()``.

    runs: :obj:`int`, :obj:`list[int | str]`, or :obj:`None`
        Run IDs specified in the analysis. Defined after running ``self.get_caps()``.

    standardize: :obj:`bool` or :obj:`None`
        Whether region-of-interests (ROIs)/columns were standardized during analysis.
        Defined after running ``self.get_caps()``.

    means: :obj:`dict[str, np.array]` or :obj:`None`
        Group-specific feature means if standardization was applied. Defined after running
        ``self.get_caps()``. Returns a deep copy.

        ::

            {"GroupName": np.array(shape=[1, ROIs])}

    stdev: :obj:`dict[str, np.array]` or :obj:`None`
        Group-specific feature standard deviations if standardization was applied. Defined after
        running ``self.get_caps()``. Returns a deep copy.

        ::

            {"GroupName": np.array(shape=[1, ROIs])}

        .. note:: Standard deviations below ``np.finfo(std.dtype).eps`` are replaced with 1 for\
        numerical stability.

    concatenated_timeseries: :obj:`dict[str, np.array]` or :obj:`None`
        Group-specific concatenated timeseries data. Can be deleted using
        ``del self.concatenated_timeseries``. Defined after running ``self.get_caps()``. Returns a
        reference.

        ::

            {"GroupName": np.array(shape=[(participants x TRs), ROIs])}

        .. note:: For versions >= 0.25.0, subject IDs are sorted lexicographically prior to\
        concatenation and the order is determined by ``self.groups``.

    kmeans: :obj:`dict[str, sklearn.cluster.KMeans]` or :obj:`None`
        Group-specific k-means models. Defined after running ``self.get_caps()``. Returns a deep
        copy.

        ::

            {"GroupName": sklearn.cluster.KMeans}

    caps: :obj:`dict[str, dict[str, np.array]]` or :obj:`None`
        Cluster centroids for each group and CAP. Defined after running ``self.get_caps()``. Returns
        a deep copy.

        ::

            {"GroupName": {"CAP-1": np.array(shape=[1, ROIs]), ...}}

    cluster_scores: :obj:`dict[str, str | dict[str, float]]` or :obj:`None`
        Scores for different cluster sizes by group. Defined after running ``self.get_caps()``.

        ::

            {"Cluster_Selection_Method": str, "Scores": {"GroupName": {2: float, 3: float}}}

    optimal_n_clusters: :obj:`dict[str, int]` or :obj:`None`
        Optimal number of clusters by group if cluster selection was used. Defined after running
        ``self.get_caps()``.

        ::

            {"GroupName": int}

    variance_explained: :obj:`dict[str, float]` or :obj:`None`
        Total variance explained by each group's model. Defined after running ``self.get_caps()``.

        ::

            {"GroupName": float}

    region_means: :obj:`dict[str, dict[str, list[str] | np.array]]` or :obj:`None`
        Region-averaged values used for visualization. Defined after running ``self.caps2plot()``.

        ::

            {"GroupName": {"Regions": [...], "CAP-1": np.array(shape=[1, Regions]), ...)}}

    outer_products: :obj:`dict[str, dict[str, np.array]]` or :obj:`None`
        Outer product matrices for visualization. Defined after running ``self.caps2plot()``.

        ::

            {"GroupName": {"CAP-1": np.array(shape=[ROIs, ROIs]), ...}}

    cosine_similarity: :obj:`dict[str, dict[str, list[str] | np.array]]` or :obj:`None`
        Cosine similarities between CAPs and the regions specified in ``parcel_approach``.
        Defined after running ``self.caps2radar()``.

        ::

            {
                "GroupName": {
                    "Regions": [...],
                    "CAP-1": {
                        "High Amplitude": np.array(shape=[1, Regions]),
                        "Low Amplitude": np.array(shape=[1, Regions]),
                    }
                }
            }

    See Also
    --------
    :class:`neurocaps.typing.ParcelConfig`
        Type definition representing the configuration options and structure for the Schaefer
        and AAL parcellations.
    :class:`neurocaps.typing.ParcelApproach`
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation
        approaches.

    Important
    ---------
    **Data/Property Persistence**: Each time certain functions are called, properties related to
    that function are automatically initialized/overwritten to create a clean state for the subsequent
    analysis. For instance, when ``self.get_caps()`` is ran, then properties such as ``self.caps``,
    ``self.kmeans``, ``self.concatenated_timeseries``, ``self.stdev``, etc are automatically
    re-initialized to store the new results. The same occurs for ``self.cosine_similarity``, when
    ``self.caps2radar()`` is ran and for other properties and their associated functions.

    Note
    ----
    **Default Group Name**: The default group name is "All Subjects" when no groups are specified.

    **Custom Parcellations**: Refer to the `NeuroCAPs' Parcellation Documentation
    <https://neurocaps.readthedocs.io/en/stable/parcellations.html>`_ for detailed explanations and
    example structures for Custom parcellations.
    """

    def __init__(
        self,
        parcel_approach: Optional[Union[ParcelConfig, ParcelApproach, str]] = None,
        groups: Optional[dict[str, list[str]]] = None,
    ) -> None:
        if parcel_approach is not None:
            parcel_approach = _check_parcel_approach(parcel_approach=parcel_approach, call="CAP")

        self._parcel_approach = parcel_approach

        # Raise error if self groups is not a dictionary
        if groups:
            if not isinstance(groups, dict):
                raise TypeError(
                    "`groups` must be a dictionary where the keys are the group names and the "
                    "items correspond to subject IDs in the groups."
                )

            for group_name in groups:
                assert groups[group_name], f"{group_name} has zero subject IDs."

            # Convert ids to strings
            for group in set(groups):
                groups[group] = [
                    str(subj_id) if not isinstance(subj_id, str) else subj_id
                    for subj_id in groups[group]
                ]

        self._groups = groups

    def get_caps(
        self,
        subject_timeseries: Union[SubjectTimeseries, str],
        runs: Optional[Union[int, str, list[int], list[str]]] = None,
        n_clusters: Union[int, list[int], range] = 5,
        cluster_selection_method: Optional[
            Literal["elbow", "davies_bouldin", "silhouette", "variance_ratio"]
        ] = None,
        random_state: Optional[int] = None,
        init: Union[Literal["k-means++", "random"], Callable, ArrayLike] = "k-means++",
        n_init: Union[Literal["auto"], int] = "auto",
        max_iter: int = 300,
        tol: float = 0.0001,
        algorithm: Literal["lloyd", "elkan"] = "lloyd",
        standardize: bool = True,
        n_cores: Optional[int] = None,
        show_figs: bool = False,
        output_dir: Optional[str] = None,
        progress_bar: bool = False,
        as_pickle: bool = False,
        **kwargs,
    ) -> Self:
        """
        Perform K-Means Clustering to Identify CAPs.

        Concatenates the timeseries of each subject into a single NumPy array with dimensions
        (participants x TRs) x ROI and uses ``sklearn.cluster.KMeans`` on the concatenated data.
        Separate ``KMeans`` models are generated for all groups.

        Parameters
        ----------
        subject_timeseries: :obj:`SubjectTimeseries` or :obj:`str`
            A dictionary mapping subject IDs to their run IDs and their associated timeseries
            (TRs x ROIs) as a NumPy array. Can also be a path to a pickle file containing this same
            structure. Refer to documentation for ``SubjectTimeseries`` in the "See Also" section for
            an example structure.

        runs: :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            Specific run IDs to perform the CAPs analysis with (e.g. ``runs=[0, 1]`` or
            ``runs=["01", "02"]``). If None, all runs will be used.

        n_clusters: :obj:`int` or :obj:`list[int]`, default=5
            Number of clusters to use. Can be a single integer or a list of integers (if
            ``cluster_selection_method`` is not None).

        cluster_selection_method: {"elbow", "davies_bouldin", "silhouette", "variance_ratio"}\
        or :obj:`None`, default=None
            Method to find the optimal number of clusters. Options are "elbow", "davies_bouldin",
            "silhouette", and "variance_ratio".

        random_state: :obj:`int` or :obj:`None`, default=None
            Random state (seed) value to use.

        init: {"k-means++", "random"}, :obj:`Callable`, or `ArrayLike`, default="k-means++"
            Method for choosing initial cluster centroid. Options are "k-means++", "random", or
            callable or array-like of shape (n_clusters, n_features).

        n_init: {"auto"} or :obj:`int`, default="auto"
            Number of times k-means is ran with different initial clusters. The model with lowest
            inertia from these runs will be selected.

        max_iter: :obj:`int`, default=300
            Maximum number of iterations for a single run of k-means.

        tol: :obj:`float`, default=1e-4
            Stopping criterion if the change in cluster centers (measured using Frobenius norm) is
            below this value, assuming ``max_iter`` has not been reached.

        algorithm: {"lloyd", "elkan"}, default="lloyd"
            The algorithm to use. Options are "lloyd" and "elkan".

        standardize: :obj:`bool`, default=True
            Standardizes the columns (ROIs) of the concatenated timeseries data. Uses sample
            standard deviation (`n-1`).

            .. note:: Standard deviations below ``np.finfo(std.dtype).eps`` are replaced with 1 for\
            numerical stability.

        n_cores: :obj:`int` or :obj:`None`, default=None
            Number of cores to use for multiprocessing, with Joblib, to run multiple k-means models
            if ``cluster_selection_method`` is not None. The "loky" backend is used.

        show_figs: :obj:`bool`, default=False
            Displays the plots for the specified ``cluster_selection_method`` for all groups.

        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save plots as png files if ``cluster_selection_method`` is not None. The
            directory will be created if it does not exist. If None, plots will not be saved.

        progress_bar: :obj:`bool`, default=False
            If True and ``cluster_selection_method`` is not None, displays a progress bar.

        as_pickle: :obj:`bool`, default=False
            When ``output_dir`` and ``cluster_selection_method`` is specified, plots are saved as
            pickle filess, which can be further modified, instead of png images.

            .. versionadded:: 0.26.5

        **kwargs:
            Additional keyword arguments when ``cluster_selection_method`` is specified:

            - S: :obj:`int` or :obj:`float`, default=1.0 -- Adjusts the sensitivity of finding the
              elbow. Larger values are more conservative and less sensitive to small fluctuations.
              Passed to ``KneeLocator`` from the kneed package.
            - dpi: :obj:`int`, default=300 -- Dots per inch for the figure.
            - figsize: :obj:`tuple`, default=(8, 6) -- Adjusts the size of the plots.
            - bbox_inches: :obj:`str` or :obj:`None`, default="tight" -- Alters size of the
              whitespace in the saved image.
            - step: :obj:`int`, default=None -- An integer value that controls the progression of
              the x-axis in plots.
            - max_nbytes: :obj:`int`, :obj:`str`, or :obj:`None`, default="1M" -- If ``n_cores`` is
              not None, serves as the threshold to trigger Joblib's automated memory mapping for
              large arrays.

              .. versionadded:: 0.28.5

        See Also
        --------
        :data:`neurocaps.typing.SubjectTimeseries`
            Type definition for the subject timeseries dictionary structure.

        Returns
        -------
        self

        Raises
        ------
        NoElbowDetectionError
            Occurs when ``cluster_selection_method`` is set to elbow but kneed's ``KneeLocator``
            does not detect an elbow in the convex curve.

        Notes
        -----
        **KMeans Algorithm:** Refer to `scikit-learn's Documentation
        <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ for
        additional information about the ``KMeans`` algorithm used in this method.

        The ``n_clusters``, ``random_state``, ``init``, ``n_init``, ``max_iter``, ``tol``, and
        ``algorithm`` parameters are passed to ``sklearn.cluster.KMeans``. Only ``n_clusters``
        differs from scikit-learn's default value, changing from 8 to 5.

        **Default Group Naming:** When ``group`` is None during initialization of the ``CAP`` class,
        then "All Subjects" is the default group name. On the first call of this function, the
        subject IDs in ``subject_timeseries`` will be automatically detected and stored in
        ``self.group``. This mapping persists until the ``CAP`` class is re-initialized or unless
        ``self.clear_groups()`` is used.

        **Concatenated Timeseries**: The concatenated timeseries is stored in
        ``self.concatenated_timeseries`` for user convenience and can be deleted using
        ``del self.concatenated_timeseries`` without disruption to the any other function.
        Additionally, for versions >= 0.25.0, the concatenation of subjects is performed
        lexicographically based on their subject IDs.
        """
        # Ensure all unique values if n_clusters is a list
        n_clusters = n_clusters if isinstance(n_clusters, int) else sorted(list(set(n_clusters)))

        if isinstance(n_clusters, list):
            n_clusters = (
                n_clusters[0]
                if all([isinstance(n_clusters, list), len(n_clusters) == 1])
                else n_clusters
            )
            # Raise error if n_clusters is a list and no cluster selection method is specified
            if len(n_clusters) > 1 and not cluster_selection_method:
                raise ValueError(
                    "`cluster_selection_method` must be specified  since `n_clusters` is a list."
                )

        self._n_clusters = n_clusters

        valid_methods = ["elbow", "davies_bouldin", "silhouette", "variance_ratio"]
        if cluster_selection_method and cluster_selection_method not in valid_methods:
            formatted_string = ", ".join(["'{a}'".format(a=x) for x in valid_methods])
            raise ValueError(f"Options for `cluster_selection_method` are: {formatted_string}.")

        # Raise error if silhouette_method is requested when n_clusters is an integer
        if cluster_selection_method and isinstance(self._n_clusters, int):
            raise ValueError(
                "`cluster_selection_method` only valid if `n_clusters` is a range of unique "
                "integers."
            )

        if not cluster_selection_method and n_cores:
            raise ValueError(
                "Parallel processing will not run since `cluster_selection_method` is None."
            )

        self._n_cores = n_cores
        configs = {
            "random_state": random_state,
            "init": init,
            "n_init": n_init,
            "max_iter": max_iter,
            "tol": tol,
            "algorithm": algorithm,
        }

        if runs and not isinstance(runs, list):
            runs = [runs]

        self._runs = runs
        self._standardize = standardize

        subject_timeseries = io_utils._get_obj(subject_timeseries)
        self._concatenated_timeseries = self._concatenate_timeseries(
            subject_timeseries, runs, progress_bar
        )

        if cluster_selection_method:
            self._select_optimal_clusters(
                cluster_selection_method,
                configs,
                show_figs,
                output_dir,
                progress_bar,
                as_pickle,
                **kwargs,
            )
        else:
            self._kmeans = {}
            for group in self._groups:
                self._kmeans[group] = _run_kmeans(
                    self._n_clusters, configs, self._concatenated_timeseries[group], method=None
                )

        self._var_explained()

        self._create_caps_dict()

        return self

    def clear_groups(self) -> None:
        """
        Clears the ``groups`` property.

        Sets ``groups`` to None to allow ``self.get_caps`` to create a new "All Subjects" group
        with different subject IDs based on the inputted ``subject_timeseries``.

        .. version_added:: 0.28.5
        """
        self._groups = None

    def _generate_lookup_table(self) -> None:
        """
        Creates dictionary mapping subject IDs to their associated group. This function is
        called whenever ``self._concatenate_timeseries`` is called."""
        self._subject_table = {}

        for group in self._groups:
            for subj_id in self._groups[group]:
                if subj_id in self._subject_table:
                    LG.warning(
                        f"[SUBJECT: {subj_id}] Appears more than once. Only the first instance of "
                        "this subject will be included in the analysis."
                    )
                else:
                    self._subject_table.update({subj_id: group})

    def _concatenate_timeseries(
        self,
        subject_timeseries: SubjectTimeseries,
        runs: Union[list[int], list[str], None],
        progress_bar: bool,
    ) -> dict[str, NDArray]:
        """
        Concatenates the timeseries data of all subjects into a single numpy array if ``groups`` are
        None or group-specific numpy arrays.
        """
        # Create dictionary for "All Subjects" if no groups are specified
        if not self._groups:
            LG.info(
                "No groups specified. Using default group 'All Subjects' containing all subject "
                "IDs from `subject_timeseries`. The `self.groups` dictionary will remain fixed "
                "unless the `CAP` class is re-initialized or `self.clear_groups()` is used."
            )
            self._groups = {"All Subjects": list(subject_timeseries)}

        # Sort IDs lexicographically (also done in `TimeseriesExtractor`)
        self._groups = {group: sorted(self._groups[group]) for group in self._groups}

        self._generate_lookup_table()

        # Intersect subjects in subjects table and the groups for tqdm
        group_map = {
            group: sorted(set(self._subject_table).intersection(self._groups[group]))
            for group in self._groups
        }
        # Collect timeseries data in lists
        group_arrays = {group: [] for group in self._groups}
        for group in group_map:
            for subj_id in tqdm(
                group_map[group],
                desc=f"Collecting Subject Timeseries Data [GROUP: {group}]",
                disable=not progress_bar,
            ):
                subject_runs, miss_runs = self._get_runs(runs, list(subject_timeseries[subj_id]))

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
                group_arrays[group].extend(subj_arrays)

        # Only stack once per group; avoid bottleneck due to repeated calls on large data
        concatenated_timeseries = {group: None for group in self._groups}
        for group in tqdm(
            group_arrays, desc="Concatenating Timeseries Data Per Group", disable=not progress_bar
        ):
            concatenated_timeseries[group] = np.vstack(group_arrays[group])

        del group_arrays

        self._mean_vec = {group: None for group in self._groups}
        self._stdev_vec = {group: None for group in self._groups}
        if self._standardize:
            concatenated_timeseries = self._scale(concatenated_timeseries)

        return concatenated_timeseries

    def _scale(self, concatenated_timeseries: NDArray) -> NDArray:
        """Scales the concatenated timeseries data."""
        for group in self._groups:
            concatenated_timeseries[group], self._mean_vec[group], self._stdev_vec[group] = (
                _standardize(concatenated_timeseries[group], return_parameters=True)
            )

        return concatenated_timeseries

    @staticmethod
    def _get_runs(
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

    def _select_optimal_clusters(
        self,
        method: str,
        configs: dict[str, Any],
        show_figs: bool,
        output_dir: Union[str, None],
        progress_bar: bool,
        as_pickle: bool,
        **kwargs,
    ) -> None:
        """Selects optimal number of clusters based on the specific ``cluster_selection_method``."""
        self._cluster_scores = {}
        self._optimal_n_clusters = {}
        self._kmeans = {}
        performance_dict = {}

        for group in self._groups:
            performance_dict[group] = {}
            model_dict = {}

            if self._n_cores is None:
                for n_cluster in tqdm(
                    self._n_clusters, desc=f"Clustering [GROUP: {group}]", disable=not progress_bar
                ):
                    output_score, model = _run_kmeans(
                        n_cluster, configs, self._concatenated_timeseries[group], method
                    )
                    performance_dict[group].update(output_score)
                    model_dict.update(model)
            else:
                parallel = Parallel(
                    return_as="generator",
                    n_jobs=self._n_cores,
                    backend="loky",
                    max_nbytes=kwargs.get("max_nbytes", "1M"),
                )
                outputs = tqdm(
                    parallel(
                        delayed(_run_kmeans)(
                            n_cluster, configs, self._concatenated_timeseries[group], method
                        )
                        for n_cluster in self._n_clusters
                    ),
                    desc=f"Clustering [GROUP: {group}]",
                    total=len(self._n_clusters),
                    disable=not progress_bar,
                )

                output_scores, models = zip(*outputs)
                for output in output_scores:
                    performance_dict[group].update(output)
                for model in models:
                    model_dict.update(model)

            # Select optimal clusters
            if method == "elbow":
                kneedle = KneeLocator(
                    x=list(performance_dict[group]),
                    y=list(performance_dict[group].values()),
                    curve="convex",
                    direction="decreasing",
                    S=kwargs.get("S", 1.0),
                )

                self._optimal_n_clusters[group] = kneedle.elbow

                if self._optimal_n_clusters[group] is None:
                    raise NoElbowDetectedError(
                        f"[GROUP: {group}] - No elbow detected. Try adjusting the sensitivity "
                        "parameter (`S`) to increase or decrease sensitivity (higher values "
                        "are less sensitive), expanding the list of `n_clusters` to test, or "
                        "using another `cluster_selection_method`."
                    )
            elif method == "davies_bouldin":
                # Get minimum for davies bouldin
                self._optimal_n_clusters[group] = min(
                    performance_dict[group], key=performance_dict[group].get
                )
            else:
                # Get max for silhouette and variance ratio
                self._optimal_n_clusters[group] = max(
                    performance_dict[group], key=performance_dict[group].get
                )

            # Get the optimal kmeans model
            self._kmeans[group] = model_dict[self._optimal_n_clusters[group]]

            LG.info(
                f"[GROUP: {group} | METHOD: {method}] Optimal cluster size is "
                f"{self._optimal_n_clusters[group]}."
            )

            if show_figs or output_dir is not None:
                self._plot_method(
                    method, performance_dict, group, show_figs, output_dir, as_pickle, **kwargs
                )

        self._cluster_scores = {"Cluster_Selection_Method": method}
        self._cluster_scores.update({"Scores": performance_dict})

    def _plot_method(
        self,
        method: str,
        performance_dict: dict[str, dict[str, float]],
        group: str,
        show_figs: bool,
        output_dir: Union[str, None],
        as_pickle: bool,
        **kwargs,
    ) -> None:
        """Plots results of the specific ``cluster_selection_method``."""
        y_titles = {
            "elbow": "Inertia",
            "davies_bouldin": "Davies Bouldin Score",
            "silhouette": "Silhouette Score",
            "variance_ratio": "Variance Ratio Score",
        }

        # Create plot dictionary
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ["S", "max_nbytes"]}
        plot_dict = _check_kwargs(_PlotDefaults.get_caps(), **filtered_kwargs)

        plt.figure(figsize=plot_dict["figsize"])

        y_values = [y for _, y in performance_dict[group].items()]
        plt.plot(self._n_clusters, y_values)

        if plot_dict["step"]:
            x_ticks = range(self._n_clusters[0], self._n_clusters[-1] + 1, plot_dict["step"])
            plt.xticks(x_ticks)

        plt.title(group)
        plt.xlabel("K")

        y_title = y_titles[method]
        plt.ylabel(y_title)
        # Add vertical line for elbow method
        if y_title == "Inertia":
            plt.vlines(
                self._optimal_n_clusters[group],
                plt.ylim()[0],
                plt.ylim()[1],
                linestyles="--",
                label="elbow",
            )

        if output_dir:
            io_utils._makedir(output_dir)

            save_name = f"{group.replace(' ', '_')}_{method}.png"
            _PlotFuncs.save_fig(plt.gcf(), output_dir, save_name, plot_dict, as_pickle)

        _PlotFuncs.show(show_figs)

    def _var_explained(self) -> None:
        """Computes variance explained in the concatenated timeseries by clustering."""
        self._variance_explained = {}

        for group in self._groups:
            mean_vec = np.mean(self._concatenated_timeseries[group], axis=0)
            total_var = np.sum((self._concatenated_timeseries[group] - mean_vec) ** 2)
            explained_var = 1 - (self._kmeans[group].inertia_ / total_var)
            self._variance_explained[group] = explained_var

    def _create_caps_dict(self) -> None:
        """Maps groups to their CAPs (cluster centroids)."""
        self._caps = {}

        for group in self._groups:
            self._caps[group] = {}
            cluster_centroids = zip(
                [num for num in range(1, len(self._kmeans[group].cluster_centers_) + 1)],
                self._kmeans[group].cluster_centers_,
            )
            self._caps[group].update(
                {
                    f"CAP-{state_number}": state_vector
                    for state_number, state_vector in cluster_centroids
                }
            )

    @staticmethod
    def _raise_error(attr_name: str) -> None:
        """Raises an attribute error if a specific attribute is not present."""
        if attr_name == "_caps":
            raise AttributeError(
                "Cannot plot caps since `self.caps` is None. Run `self.get_caps()` first."
            )
        elif attr_name == "_parcel_approach":
            raise AttributeError(
                "`self.parcel_approach` is None. Add `parcel_approach` using "
                "`self.parcel_approach=parcel_approach` to use this function."
            )
        else:
            raise AttributeError(
                "Cannot calculate metrics since `self.kmeans` is None. Run `self.get_caps()` first."
            )

    def _check_required_attrs(self, attr_names: list[str]) -> None:
        """Checks if certain attributes, needed for a specific function, are present."""
        for attr_name in attr_names:
            if getattr(self, attr_name, None) is None:
                self._raise_error(attr_name)

    def calculate_metrics(
        self,
        subject_timeseries: Union[SubjectTimeseries, str],
        tr: Optional[float] = None,
        runs: Optional[Union[int, str, list[int], list[str]]] = None,
        continuous_runs: bool = False,
        metrics: Union[
            Literal[
                "temporal_fraction",
                "persistence",
                "counts",
                "transition_frequency",
                "transition_probability",
            ],
            list[str],
            tuple[str],
            None,
        ] = ("temporal_fraction", "persistence", "counts", "transition_frequency"),
        return_df: bool = True,
        output_dir: Optional[str] = None,
        prefix_filename: Optional[str] = None,
        progress_bar: bool = False,
    ) -> Union[dict[str, pd.DataFrame], None]:
        """
        Compute Participant-wise CAP Metrics.

        Computes the following temporal dynamic metrics (as described by Liu et al., 2018 and
        Yang et al., 2021):

         - "temporal_fraction" (fraction of time): Proportion of total volumes spent in a single CAP
           over all volumes in a run.

           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
                target = 1
                temporal_fraction = 4 / 6

         - "persistence" (dwell time): Average time spent in a single CAP before transitioning to
           another CAP (average consecutive/uninterrupted time).

           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
                target = 1

                # Sequences for 1 are [1] and [1, 1, 1]; There are 2 contiguous sequences
                persistence = (1 + 3) / 2

                # Turns average frames into average time = 4
                tr = 2
                if tr: persistence = ((1 + 3) / 2) * 2

         - "counts" (state initiation): Total number of initiations of a specific CAP across an
           entire run. An initiation is defined as the first occurrence of a CAP. If the same CAP is
           maintained in consecutively (indicating stability), it is still counted as a single
           initiation.

           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
                target = 1

                # Initiations of CAP-1 occur at indices 0 and 2
                counts = 2

         - "transition_frequency": Total number of transitions to different CAPs across the entire
           run.

           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]

                # Transitions between unique CAPs occur at indices 0 -> 1, 1 -> 2, and 4 -> 5
                transition_frequency = 3

         - "transition_probability": Probability of transitioning from one CAP to another CAP
           (or the same CAP). This is calculated as (Number of transitions from A to B) /
           (Total transitions from A). Note that the transition probability from CAP-A -> CAP-B is
           not the same as CAP-B -> CAP-A.

           ::

                # Note last two numbers in the predicted timeseries are switched for this example
                predicted_subject_timeseries = [1, 2, 1, 1, 3, 1]

                # If three CAPs were identified in the analysis
                combinations = [
                    (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)
                ]

                # Represents transition from CAP-1 -> CAP-2
                target = (1, 2)

                # There are 4 ones in the timeseries but only three transitions from 1
                # 1 -> 2, 1 -> 1, 1 -> 3
                n_transitions_from_1 = 3

                # There is only one 1 -> 2 transition
                transition_probability = 1 / 3

        .. note:: In the supplementary material for Yang et al., the mathematical relationship\
        between temporal fraction, counts, and persistence is ``temporal_fraction = (persistence * counts)/total_volumes``.\
        If persistence has been converted into time units (seconds), then\
        ``temporal_fraction = (persistence * counts) / (total_volumes * tr)``.

        Parameters
        ----------
        subject_timeseries: :obj:`SubjectTimeseries` or :obj:`str`
            A dictionary mapping subject IDs to their run IDs and their associated timeseries
            (TRs x ROIs) as a NumPy array. Can also be a path to a pickle file containing this same
            structure. Refer to documentation for ``SubjectTimeseries`` in the "See Also" section
            for an example structure.

        tr: :obj:`float` or :obj:`None`, default=None
            Repetition time (TR) in seconds. If provided, persistence will be calculated as the
            average uninterrupted time, in seconds, spent in each CAP. If not provided, persistence
            will be calculated as the average uninterrupted volumes (TRs), in TR units, spent in
            each state.

        runs: :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            The run IDs to calculate CAP metrics for (e.g. ``runs=[0, 1]`` or ``runs=["01", "02"]``).
            If None, CAP metrics will be calculated for all detected run IDs even if only specific
            runs were used during ``self.get_caps()``.

        continuous_runs: :obj:`bool`, default=False
            If True, all runs will be treated as a single, uninterrupted run.

            ::

                # CAP assignment of frames from for run_1 and run_2
                run_1 = [0, 1, 1]
                run_2 = [2, 3, 3]

                # Computation of each CAP metric will be conducted on the combined vector
                continuous_runs = [0, 1, 1, 2, 3, 3]

            .. note::
                - This parameter can be used together with ``runs`` to filter the runs to combine.
                - The run-ID in the dataframe will be converted to run-continuous to denote that
                  runs were combined.
                - If only a single run available for a subject, the original run ID (as opposed to
                  "run-continuous") will be used.

        metrics: {"temporal_fraction", "persistence", "counts", "transition_frequency", "transition_probability"},\
                 :obj:`list[str]`, :obj:`tuple[str]`, or :obj:`None`,\
                 default=("temporal_fraction", "persistence", "counts", "transition_frequency")
            The metrics to calculate. Available options include "temporal_fraction", "persistence",
            "counts", "transition_frequency", and "transition_probability". Defaults to
            ``("temporal_fraction", "persistence", "counts", "transition_frequency")`` if None.

            .. versionchanged:: 0.28.6 Default changed to tuple to provide better clarity; however,\
            the default metrics remains the same and is backwards compatible.

        return_df: :obj:`str`, default=True
            If True, returns ``pandas.DataFrame`` inside a dictionary, mapping each dataframe to
            their metric.

        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save ``pandas.DataFrame`` as csv files. The directory will be created if
            it does not exist. Dataframes will not be saved if None.

        prefix_filename: :obj:`str` or :obj:`None`, default=None
            A prefix to append to the saved file names for each ``pandas.DataFrame``, if
            ``output_dir`` is provided.

        progress_bar: :obj:`bool`, default=False
            If True, displays a progress bar.

        See Also
        --------
        :data:`neurocaps.typing.SubjectTimeseries`
            Type definition for the subject timeseries dictionary structure.

        Returns
        -------
        dict[str, pd.DataFrame] or dict[str, dict[str, pd.DataFrame]]
            Dictionary containing `pandas.DataFrame` - one for each requested metric. In the case of
            "transition_probability", each group has a separate dataframe which is returned in the
            from of `dict[str, dict[str, pd.DataFrame]]`. Only returned if ``return_df`` is True.


        Important
        ---------
        **Scaling:** If standardization was requested in ``self.get_caps()``, then the columns/ROIs
        of the ``subject_timeseries`` provided to this method will be scaled using group-specific
        means and standard deviations. These statistics are derived from the concatenated data of
        each group. This scaling ensures the subject's data matches the distribution of the input
        data used for group-specific clustering, which is needed for accurate predictions when using
        group-specific k-means models. The group assignment for each subject is determined based on
        the ``self.subject_table`` property.

        **Group-Specific CAPs**: If groups were specified, then each group uses their respective
        k-means models to compute metrics. The inclusion of all groups within the same dataframe
        (for "temporal_fraction", "persistence", "counts", and "transition_frequency") is primarily
        to reduce the number of dataframes generated. Hence, each CAP (e.g., "CAP-1") is specific to
        its respective groups.

        For instance, if their are two groups, Group A and Group B, each with their own CAPs:

        - **A** has 2 CAPs: "CAP-1" and "CAP-2"
        - **B** has 3 CAPs: "CAP-1", "CAP-2", and "CAP-3"

        The resulting `"temporal_fraction"` dataframe ("persistence" and "counts" have a similar
        structure but "transition frequency" will only contain the "Subject_ID", "Group", and "Run"
        columns in addition to a "Transition_Frequency" column):

        **With Groups**

        +------------+---------+-------+-------+-------+-------+
        | Subject_ID | Group   | Run   | CAP-1 | CAP-2 | CAP-3 |
        +============+=========+=======+=======+=======+=======+
        | 101        |    A    | run-1 | 0.40  | 0.60  | NaN   |
        +------------+---------+-------+-------+-------+-------+
        | 102        |    B    | run-1 | 0.30  | 0.50  | 0.20  |
        +------------+---------+-------+-------+-------+-------+
        | ...        | ...     | ...   | ...   | ...   | ...   |
        +------------+---------+-------+-------+-------+-------+

        The "NaN" indicates that "CAP-3" is not applicable for Group A. Additionally, "NaN" will
        only be observed in instances when two or more groups are specified and have different
        number of CAPs. As mentioned previously, "CAP-1", "CAP-2", and "CAP-3" for Group A is
        distinct from Group B due to using separate k-means models.

        **Without Groups**

        +------------+--------------+-------+-------+-------+-------+-------+
        | Subject_ID | Group        | Run   | CAP-1 | CAP-2 | CAP-3 | CAP-4 |
        +============+==============+=======+=======+=======+=======+=======+
        | 101        | All Subjects | run-1 | 0.20  | 0     | 0     | 0.80  |
        +------------+--------------+-------+-------+-------+-------+-------+
        | 102        | All Subjects | run-1 | 0.50  | 0.25  | 0.25  | 0     |
        +------------+--------------+-------+-------+-------+-------+-------+
        | ...        | ...          | ...   | ...   | ...   | ...   | ...   |
        +------------+--------------+-------+-------+-------+-------+-------+

        **Transition Probability:** For "transition_probability", each group has a separate
        dataframe to containing the CAP transitions for each group.

        - **Group A Transition Probability:** Stored in ``df_dict["transition_probability"]["A"]``
        - **Group B Transition Probability:** Stored in ``df_dict["transition_probability"]["B"]``

        **The resulting "transition_probability" for Group A**:

        +------------+---------+-------+-------+-------+-------+-------+-------+
        | Subject_ID | Group   | Run   |  1.1  |  1.2  |  1.3  | 2.1   | ...   |
        +============+=========+=======+=======+=======+=======+=======+=======+
        | 101        |    A    | run-1 | 0.40  | 0.60  |   0   | 0.2   | ...   |
        +------------+---------+-------+-------+-------+-------+-------+-------+
        | ...        | ...     | ...   | ...   | ...   | ...   | ...   | ...   |
        +------------+---------+-------+-------+-------+-------+-------+-------+

        **The resulting "transition_probability" for Group B**:

        +------------+---------+-------+-------+-------+-------+-------+
        | Subject_ID | Group   | Run   |  1.1  |  1.2  |  2.1  | ...   |
        +============+=========+=======+=======+=======+=======+=======+
        | 102        |    B    | run-1 | 0.70  | 0.30  | 0.10  | ...   |
        +------------+---------+-------+-------+-------+-------+-------+
        | ...        | ...     | ...   | ...   | ...   | ...   | ...   |
        +------------+---------+-------+-------+-------+-------+-------+

        Here the columns indicate {from}.{to}. For instance, column 1.2 indicates the probability
        of transitioning from CAP-1 to CAP-2.

        If no groups are specified, then the dataframe is stored in
        ``df_dict["transition_probability"]["All Subjects"]``.

        References
        ----------
        Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state
        fMRI signals. NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

        Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible
        coactivation patterns of functional brain networks reveal the aberrant dynamic state
        transition in schizophrenia. NeuroImage, 237, 118193.
        https://doi.org/10.1016/j.neuroimage.2021.118193
        """
        self._check_required_attrs(["_kmeans"])

        io_utils._issue_file_warning("prefix_filename", prefix_filename, output_dir)
        io_utils._makedir(output_dir)

        if runs and not isinstance(runs, list):
            runs = [runs]

        metrics = self._filter_metrics(metrics)

        subject_timeseries = io_utils._get_obj(subject_timeseries, needs_deepcopy=False)

        # Assign each subject's frame to a CAP, adding shift is not necessary for the current
        # iteration of the code, done for conceptual reasons due to the naming for the CAPs start
        # at 1
        predicted_subject_timeseries = self.return_cap_labels(
            subject_timeseries, runs, continuous_runs, shift_labels=True
        )

        cap_names, max_cap, group_cap_counts = self._get_caps_info()

        # Get combination of transitions in addition to building the base dataframe dictionary
        cap_pairs = self._get_pairs() if "transition_probability" in metrics else None
        df_dict = self._build_df(metrics, cap_names, cap_pairs)

        # Create function map
        metric_map = {
            "temporal_fraction": self._compute_temporal_fraction,
            "counts": self._compute_counts,
            "persistence": self._compute_persistence,
            "transition_frequency": self._compute_transition_frequency,
            "transition_probability": self._compute_transition_probability,
        }

        # Generate list for iteration
        distributed_dict = self._distribute(predicted_subject_timeseries)

        for subj_id in tqdm(
            distributed_dict, desc="Computing Metrics for Subjects", disable=not progress_bar
        ):
            for group, curr_run in distributed_dict[subj_id]:
                for metric in metrics:
                    sub_info = [subj_id, group, curr_run]

                    df = (
                        df_dict[metric]
                        if metric != "transition_probability"
                        else df_dict[metric][group]
                    )

                    # Common base arguments for all functions
                    args = [predicted_subject_timeseries[subj_id][curr_run], sub_info, df]

                    if metric in ["temporal_fraction", "counts", "persistence"]:
                        args.extend([group_cap_counts[group], max_cap])
                        args += [tr] if metric == "persistence" else []

                    if metric == "transition_probability":
                        args += [cap_pairs[group]]

                    df = metric_map[metric](*args)

                    if metric == "transition_probability":
                        df_dict[metric][group] = df
                    else:
                        df_dict[metric] = df

        if output_dir:
            self._save_metrics(output_dir, df_dict, prefix_filename)

        if return_df:
            return df_dict

    @staticmethod
    def _filter_metrics(metrics: Union[list[str], tuple[str], None]) -> list[str]:
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

    def _get_caps_info(self) -> tuple[list[str], int, dict[str, int]]:
        """
        Extracts CAP-related information, specifically the names of the CAPs (i.e. CAP-1, CAP-2,
        etc), the maximum number of CAPs found across all groups (e.g if Group A has 4 CAPs and
        group B has 5 CAPs then 5 is returned), and the number of CAPs for each group.
        """
        group_cap_counts = {}

        for group in self._groups:
            # Store the length of caps in each group
            group_cap_counts.update({group: len(self._caps[group])})

        # CAP names based on groups with the most CAPs
        cap_names = list(self._caps[max(group_cap_counts, key=group_cap_counts.get)])
        max_cap = max([int(name.split("-")[-1]) for name in cap_names])

        return cap_names, max_cap, group_cap_counts

    def return_cap_labels(
        self,
        subject_timeseries: Union[SubjectTimeseries, str],
        runs: Optional[Union[int, str, list[int], list[str]]] = None,
        continuous_runs: bool = False,
        shift_labels: bool = False,
    ) -> dict[str, dict[str, NDArray]]:
        """
        Return CAP Labels for Each Subject.

        Uses the group-specific k-means models in ``self.kmeans`` to assign each frames (TR) to
        CAPs for each subject in ``self.subject_table``.

        .. versionadded:: 0.29.0

        The process involves the following steps:

            1. Retrieve the timeseries for a specific subject's run from ``subject_timeseries``.

            2. Determine their group assignment using ``self.subject_table`` and scale their
               timeseries data (if ``standardize`` was set to True in ``self.get_caps()``) using the
               means and standard deviation derived from the group specific concatenated dataframes
               (``self.means`` and ``self.stdev``).

                .. note:: This scaling ensures the subject's data matches the distribution of the\
                input data used for group-specific clustering, which is needed for accurate\
                predictions when using group-specific k-means models.

            3. Use group-specific k-means model (``self.kmeans``) and the ``predict()`` function
               from scikit-learn's ``KMeans`` to assign each frame (TR).

            4. If ``shift_labels`` is True, apply a one unit shift for the minimum label to
               start at "1" instead of "0".

            5. Repeat 1-4 to the remaining runs (all if ``runs`` is None or specific runs) for the
               subject.

            6. If ``continuous_runs`` is True, then stack each numpy array horizontally to create a
               single array containing the predicted labels for a subject.

            7. Repeat 1-6 for the remaining subjects.

        Parameters
        ----------
        subject_timeseries: :obj:`SubjectTimeseries` or :obj:`str`
            A dictionary mapping subject IDs to their run IDs and their associated timeseries
            (TRs x ROIs) as a NumPy array. Can also be a path to a pickle file containing this same
            structure. Refer to documentation for ``SubjectTimeseries`` in the "See Also" section
            for an example structure.

        runs: :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            The run IDs to return CAP labels for (e.g. ``runs=[0, 1]`` or ``runs=["01", "02"]``).
            If None, CAP labels will be returned for all detected run IDs even if only specific runs
            were used during ``self.get_caps()``.

        continuous_runs: :obj:`bool`, default=False
            If True, all runs will be treated as a single, uninterrupted run.

            ::

                # CAP assignment of frames from for run_1 and run_2
                run_1 = [0, 1, 1]
                run_2 = [2, 3, 3]

                # Computation of each CAP metric will be conducted on the combined vector
                continuous_runs = [0, 1, 1, 2, 3, 3]

            .. note::
                - This parameter can be used together with ``runs`` to filter the runs to combine.
                - The run-ID for each subject in the dictionary will be converted to run-continuous
                  to denote that runs were combined.
                - If only a single run available for a subject, the original run ID (as opposed to
                  "run-continuous") will be used.

        shift_labels: :obj:`bool`, default=False
            If True, shifts each label by up one unit for the minimum CAP label to start at "1" as
            opposed to "0" (scikit-learn's default), if preferred.

            ::

                predicted_labels = [0, 2, 5]
                # Add plus one shift
                predicted_labels = [1, 3, 6]

        See Also
        --------
        :data:`neurocaps.typing.SubjectTimeseries`
            Type definition for the subject timeseries dictionary structure.

        Returns
        -------
            dict[str, dict[str, np.ndarray]]
                Dictionary mapping each subject to their run IDs and a 1D numpy array containing
                the predicted CAP for each frame (TR).
        """
        self._check_required_attrs(["_kmeans"])

        subject_timeseries = io_utils._get_obj(subject_timeseries, needs_deepcopy=False)

        for subj_id, group in self._subject_table.items():
            if "predicted_subject_timeseries" not in locals():
                predicted_subject_timeseries = {}

            predicted_subject_timeseries[subj_id] = {}
            subject_runs, miss_runs = self._get_runs(runs, list(subject_timeseries[subj_id]))

            if miss_runs:
                LG.warning(
                    f"[SUBJECT: {subj_id}] Does not have the requested runs: "
                    f"{', '.join(miss_runs)}."
                )

            if not subject_runs:
                LG.warning(
                    f"[SUBJECT: {subj_id}] Excluded from the dictionary due to having no runs."
                )
                continue

            prediction_dict = {}
            for run_id in subject_runs:
                # Standardize or not
                if self._standardize:
                    timeseries = (
                        subject_timeseries[subj_id][run_id] - self._mean_vec[group]
                    ) / self._stdev_vec[group]
                else:
                    timeseries = subject_timeseries[subj_id][run_id]

                # Add 1 to the prediction vector since labels start at 0
                shift = 1 if shift_labels else 0
                prediction_dict.update({run_id: self._kmeans[group].predict(timeseries) + shift})

            if len(prediction_dict) > 1 and continuous_runs:
                # Horizontally stack predicted runs
                predicted_subject_timeseries[subj_id].update(
                    {
                        "run-continuous": np.hstack(
                            [prediction_dict[run_id] for run_id in subject_runs]
                        )
                    }
                )
            else:
                predicted_subject_timeseries[subj_id].update(prediction_dict)

        return predicted_subject_timeseries

    def _get_pairs(self) -> dict[str, list[tuple[int, int]]]:
        """Obtains all possible transition pairs."""
        group_caps = {}
        all_pairs = {}

        for group in self._groups:
            group_caps.update({group: [int(name.split("-")[-1]) for name in self._caps[group]]})
            all_pairs.update({group: list(itertools.product(group_caps[group], group_caps[group]))})

        return all_pairs

    def _build_df(
        self, metrics: list[str], cap_names: list[str], pairs: dict[str, list[tuple[int, int]]]
    ) -> dict[str, pd.DataFrame]:
        """Initializes the output dataframes with column names for each requested metric."""
        df_dict = {}
        base_cols = ["Subject_ID", "Group", "Run"]

        for metric in metrics:
            if metric not in ["transition_frequency", "transition_probability"]:
                df_dict.update({metric: pd.DataFrame(columns=base_cols + list(cap_names))})
            elif metric == "transition_probability":
                df_dict[metric] = {}
                for group in self._groups:
                    col_names = base_cols + [f"{x}.{y}" for x, y in pairs[group]]
                    df_dict[metric].update({group: pd.DataFrame(columns=col_names)})
            else:
                df_dict.update({metric: pd.DataFrame(columns=base_cols + ["Transition_Frequency"])})

        return df_dict

    def _distribute(
        self, predicted_subject_timeseries: dict[str, NDArray]
    ) -> dict[str, list[tuple[str, str]]]:
        """Creates a dictionary mapping for each subject and run pair to iterate over."""
        distributed_dict = {}

        for subj_id, group in self._subject_table.items():
            distributed_dict[subj_id] = []
            for curr_run in predicted_subject_timeseries[subj_id]:
                distributed_dict[subj_id].append((group, curr_run))

        return distributed_dict

    def _compute_temporal_fraction(
        self, arr: NDArray, sub_info: list[str], df: pd.DataFrame, n_group_caps: int, max_cap: int
    ) -> pd.DataFrame:
        """
        Computes temporal fraction for the subject and run specified in ``sub_info`` and inserts new
        row in the dataframe.
        """
        frequency_dict = {
            key: np.where(arr == key, 1, 0).sum() for key in range(1, n_group_caps + 1)
        }

        frequency_dict = self._update_dict(max_cap, n_group_caps, frequency_dict)

        proportion_dict = {key: value / (len(arr)) for key, value in frequency_dict.items()}

        return self._append_df(df, sub_info, proportion_dict)

    def _compute_counts(
        self, arr: NDArray, sub_info: list[str], df: pd.DataFrame, n_group_caps: int, max_cap: int
    ) -> pd.DataFrame:
        """
        Computes counts for the subject and run specified in ``sub_info`` and inserts new row in the
        dataframe.
        """
        count_dict = {}
        for target in range(1, n_group_caps + 1):
            if target in arr:
                _, counts = self._segments(target, arr)
                count_dict.update({target: counts})
            else:
                count_dict.update({target: 0})

        count_dict = self._update_dict(max_cap, n_group_caps, count_dict)

        return self._append_df(df, sub_info, count_dict)

    def _compute_persistence(
        self,
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
            binary_arr, n_segments = self._segments(target, arr)
            # ``n_segments`` returns minimum of 1 so persistence is 0 instead of NaN when CAP not in
            # timeseries
            persistence_dict.update({target: (binary_arr.sum() / n_segments) * (tr if tr else 1)})

        persistence_dict = self._update_dict(max_cap, n_group_caps, persistence_dict)

        return self._append_df(df, sub_info, persistence_dict)

    @staticmethod
    def _segments(target: int, timeseries: NDArray) -> tuple[NDArray[np.bool_], int]:
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

    @staticmethod
    def _update_dict(
        max_cap: int, n_group_caps: int, curr_dict: dict[str, Union[float, int]]
    ) -> dict[str, Union[float, int]]:
        """Adds NaN for groups with less caps than the group with the greatest number of caps."""
        if max_cap > n_group_caps:
            for i in range(n_group_caps + 1, max_cap + 1):
                curr_dict.update({i: float("nan")})

        return curr_dict

    @staticmethod
    def _append_df(
        df: pd.DataFrame, sub_info: list[str], metric_dict: dict[str, Union[float, int]]
    ) -> pd.DataFrame:
        """Appends new row in dataframe."""
        df.loc[len(df)] = sub_info + [items for items in metric_dict.values()]
        return df

    @staticmethod
    def _compute_transition_frequency(
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

    @staticmethod
    def _compute_transition_probability(
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
                np.sum((trans_from == e1) & (trans_to == e2)) / total_trans
                if total_trans > 0
                else 0
            )

        return df

    def _save_metrics(
        self, output_dir: str, df_dict: dict[str, pd.DataFrame], prefix_filename: Union[str, None]
    ) -> None:
        """Saves the metric dataframes as csv files."""
        for metric in df_dict:
            filename = io_utils._filename(
                base_name=f"{metric}", add_name=prefix_filename, pos="prefix"
            )
            if metric != "transition_probability":
                df_dict[f"{metric}"].to_csv(
                    path_or_buf=os.path.join(output_dir, f"{filename}.csv"), sep=",", index=False
                )
            else:
                for group in self._groups:
                    df_dict[f"{metric}"][group].to_csv(
                        path_or_buf=os.path.join(
                            output_dir, f"{filename}-{group.replace(' ', '_')}.csv"
                        ),
                        sep=",",
                        index=False,
                    )

    def caps2plot(
        self,
        output_dir: Optional[str] = None,
        suffix_title: Optional[str] = None,
        suffix_filename: Optional[str] = None,
        plot_options: Union[
            Literal["outer_product", "heatmap"], list[Literal["outer_product", "heatmap"]]
        ] = "outer_product",
        visual_scope: Union[
            Literal["regions", "nodes"], list[Literal["regions", "nodes"]]
        ] = "regions",
        show_figs: bool = True,
        subplots: bool = False,
        as_pickle: bool = False,
        **kwargs,
    ) -> Self:
        """
        Generate Heatmaps and Outer Product Plots for CAPs.

        Plot CAPs as heatmaps or outer products at the node or region levels. Separate plots are
        generated for each group.

        Parameters
        ----------
        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory for saving plots as png files. The directory will be created if it does not
            exist. If None, plots will not be saved.

        suffix_title: :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the filename of each saved plot if ``output_dir`` is provided.

        plot_options: {"outer_product", "heatmap"} or :obj:`list["outer_product", "heatmap"]`,\
            default="outer_product"
            Type of plots to create. Options are "outer_product" or "heatmap".

        visual_scope: {"regions", "nodes"} or :obj:`list["regions", "nodes"]`, default="regions"
            Determines whether plotting is done at the region level or node level. For "regions",
            the values of all nodes in the same regions (including both hemispheres) are averaged
            together then plotted. For "nodes", plots individual node values separately.

        show_figs: :obj:`bool`, default=True
           Display figures.

        subplots: :obj:`bool`, default=True
            Produce subplots for outer product plots, combining all plots into a single figure.

        as_pickle: :obj:`bool`, default=False
            When ``output_dir`` is specified, plots are saved as pickle files, which can be further
            modified, instead of png images.

            .. versionadded:: 0.26.5

        **kwargs:
            Keyword arguments used when saving figures. Valid keywords include:

            - dpi: :obj:`int`, default=300 -- Dots per inch for the figure.
            - figsize: :obj:`tuple`, default=(8, 6) -- Size of the figure in inches.
            - fontsize: :obj:`int`, default=14 -- Font size for the title of individual plots or
              subplots.
            - hspace: :obj:`float`, default=0.4 -- Height space between subplots.
            - wspace: :obj:`float`, default=0.4 -- Width space between subplots.
            - xticklabels_size: :obj:`int`, default=8 -- Font size for x-axis tick labels.
            - yticklabels_size: :obj:`int`, default=8 -- Font size for y-axis tick labels.
            - shrink: :obj:`float`, default=0.8 -- Fraction by which to shrink the colorbar.
            - cbarlabels_size: :obj:`int`, default=8 -- Font size for the colorbar labels.
            - nrow: :obj:`int`, default=varies (max 5) -- Number of rows for subplots.
              Default varies but the maximum is 5.
            - ncol: :obj:`int` or :obj:`None`, default=None -- Number of columns for subplots.
              Default varies but the maximum is 5.
            - suptitle_fontsize: :obj:`float`, default=0.7 -- Font size for the main title when
              subplot is True.
            - tight_layout: :obj:`bool`, default=True -- Use tight layout for subplots.
            - rect: :obj:`list[int]`, default=[0, 0.03, 1, 0.95] -- Rectangle parameter for
              ``tight_layout`` when ``subplots`` are True. Fixes whitespace issues.
            - sharey: :obj:`bool`, default=True -- Share y-axis labels for subplots.
            - xlabel_rotation: :obj:`int`, default=0 -- Rotation angle for x-axis labels.
            - ylabel_rotation: :obj:`int`, default=0 -- Rotation angle for y-axis labels.
            - annot: :obj:`bool`, default=False -- Add values to cells.
            - annot_kws: :obj:`dict`, default=None -- Customize the annotations.
            - fmt: :obj:`str`, default=".2g" -- Modify how the annotated vales are presented.
            - linewidths: :obj:`float`, default=0 -- Padding between each cell in the plot.
            - borderwidths: :obj:`float`, default=0 -- Width of the border around the plot.
            - linecolor: :obj:`str`, default="black" -- Color of the line that seperates each cell.
            - edgecolors: :obj:`str` or :obj:`None`, default=None -- Color of the edges.
            - alpha: :obj:`float` or :obj:`None`, default=None -- Controls transparency and ranges
              from 0 (transparent) to 1 (opaque).
            - bbox_inches: :obj:`str` or :obj:`None`, default="tight" -- Alters size of the
              whitespace in the saved image.
            - hemisphere_labels: :obj:`bool`, default=False -- When ``visual_scope="nodes"``,
              shows simplified left/right hemisphere division line instead of individual node labels
              (only for Custom and Schaefer parcellations). For Custom parcellations, assumes labels
              are ordered such that all left hemisphere nodes come before right hemisphere nodes.
              Ignores edgecolors; linewidths/linecolor only affect division line.
            - cmap: :obj:`str`, :obj:`callable` default="coolwarm" -- Color map for the plot cells.
              Options include strings to call seaborn's pre-made palettes, ``seaborn.diverging_palette``
              function to generate custom palettes, and ``matplotlib.color.LinearSegmentedColormap``
              to generate custom palettes.
            - vmin: :obj:`float` or :obj:`None`, default=None -- The minimum value to display in colormap.
            - vmax: :obj:`float` or :obj:`None`, default=None -- The maximum value to display in colormap.

        Returns
        -------
        self

        Note
        ----
        **Parcellation Approach**: the "nodes" and "regions" subkeys are required in
        ``parcel_approach`` for this method.

        **Color Palettes**: Refer to `seaborn's Color Palettes\
        <https://seaborn.pydata.org/tutorial/color_palettes.html>`_ for valid pre-made palettes.
        """
        self._check_required_attrs(["_parcel_approach", "_caps"])

        # Check if parcellation_approach is custom
        if "Custom" in self._parcel_approach and any(
            key not in self._parcel_approach["Custom"] for key in ["nodes", "regions"]
        ):
            _check_parcel_approach(parcel_approach=self._parcel_approach, call="caps2plot")

        # Check labels
        check_caps = self._caps[list(self._caps)[0]]
        check_caps = check_caps[list(check_caps)[0]]
        # Get parcellation name
        parcellation_name = list(self._parcel_approach)[0]
        if check_caps.shape[0] != len(self._parcel_approach[parcellation_name]["nodes"]):
            raise ValueError(
                "Number of nodes used for CAPs does not equal the number of nodes specified in "
                "`parcel_approach`."
            )

        io_utils._issue_file_warning("suffix_filename", suffix_filename, output_dir)
        io_utils._makedir(output_dir)

        if isinstance(plot_options, str):
            plot_options = [plot_options]
        if isinstance(visual_scope, str):
            visual_scope = [visual_scope]

        # Check inputs for plot_options and visual_scope
        if not any(["heatmap" in plot_options, "outer_product" in plot_options]):
            raise ValueError("Valid inputs for `plot_options` are 'heatmap' and 'outer_product'.")

        if not any(["regions" in visual_scope, "nodes" in visual_scope]):
            raise ValueError("Valid inputs for `visual_scope` are 'regions' and 'nodes'.")

        if "regions" in visual_scope:
            # Compute means of regions/networks for each cap
            self._compute_region_means(parcellation_name)

        # Create plot dictionary
        plot_dict = _check_kwargs(_PlotDefaults.caps2plot(), **kwargs)
        if plot_dict["hemisphere_labels"] is True:
            if "nodes" not in visual_scope:
                raise ValueError(
                    "`hemisphere_labels` is only available when `visual_scope == 'nodes'`."
                )
            if parcellation_name == "AAL":
                raise ValueError(
                    "`hemisphere_labels` is only available for 'Custom' and 'Schaefer'."
                )

        # Ensure plot_options and visual_scope are lists
        plot_options = plot_options if isinstance(plot_options, list) else list(plot_options)
        visual_scope = visual_scope if isinstance(visual_scope, list) else list(visual_scope)
        # Initialize outer product attribute
        if "outer_product" in plot_options:
            self._outer_products = {}

        distributed_list = list(itertools.product(plot_options, visual_scope, self._groups))

        for plot_option, scope, group in distributed_list:
            # Get correct labels depending on scope
            cap_dict, labels = self._extract_scope_information(scope, parcellation_name)

            # Generate plot for each group
            input_keys = dict(
                group=group,
                plot_dict=plot_dict,
                cap_dict=cap_dict,
                full_labels=labels,
                output_dir=output_dir,
                suffix_title=suffix_title,
                suffix_filename=suffix_filename,
                show_figs=show_figs,
                as_pickle=as_pickle,
                scope=scope,
                parcellation_name=parcellation_name,
            )

            # Generate plot for each group
            if plot_option == "outer_product":
                self._generate_outer_product_plots(**input_keys, subplots=subplots)
            elif plot_option == "heatmap":
                self._generate_heatmap_plots(**input_keys)

        return self

    def _compute_region_means(self, parcellation_name: str) -> None:
        """
        Creates an attribute called ``self._region_means``, representing the average values of
        all nodes in a corresponding region to create region heatmaps or outer product plots.
        """
        self._region_means = {group: {} for group in self._groups}

        region_dict = (
            self._parcel_approach["Custom"]["regions"] if parcellation_name == "Custom" else None
        )
        # List of regions remains list for Schaefer and AAL but converts keys to list for Custom
        regions = list(self._parcel_approach[parcellation_name]["regions"])

        group_caps = [(group, cap) for group in self._groups for cap in self._caps[group]]
        for group, cap in group_caps:
            region_means = None
            for region in regions:
                if parcellation_name != "Custom":
                    region_indxs = np.array(
                        [
                            index
                            for index, node in enumerate(
                                self._parcel_approach[parcellation_name]["nodes"]
                            )
                            if region in node
                        ]
                    )
                else:
                    region_indxs = np.array(
                        list(region_dict[region]["lh"]) + list(region_dict[region]["rh"])
                    )

                if region_means is None:
                    region_means = np.array([np.average(self._caps[group][cap][region_indxs])])
                else:
                    region_means = np.hstack(
                        [region_means, np.average(self._caps[group][cap][region_indxs])]
                    )

            # Append regions and their means
            self._region_means[group].update({"Regions": regions})
            self._region_means[group].update({cap: region_means})

    def _extract_scope_information(
        self, scope: str, parcellation_name: str
    ) -> tuple[dict[str, dict[str, NDArray]], list[str]]:
        """
        Extracting region means of each CAP from ``self._region_means`` if scope is "region" else
        extracts the CAP vectors from ``self._caps``. Also extracts region labels or nodes
        from the ``parcel_approach``.
        """
        if scope == "regions":
            cap_dict = {
                group: {k: v for k, v in self._region_means[group].items() if k != "Regions"}
                for group in self._region_means
            }
            labels = list(self._parcel_approach[parcellation_name]["regions"])
        elif scope == "nodes":
            if parcellation_name in ["Schaefer", "AAL"]:
                cap_dict, labels = self._caps, self._parcel_approach[parcellation_name]["nodes"]
            else:
                cap_dict = self._caps
                labels = [
                    x[0] + " " + x[1]
                    for x in list(
                        itertools.product(["LH", "RH"], self._parcel_approach["Custom"]["regions"])
                    )
                ]

        return cap_dict, labels

    def _generate_outer_product_plots(
        self,
        group: str,
        plot_dict: dict[str, Any],
        cap_dict: dict[str, dict[str, NDArray]],
        full_labels: list[str],
        subplots: bool,
        output_dir: Union[str, None],
        suffix_title: Union[str, None],
        suffix_filename: Union[str, None],
        show_figs: bool,
        as_pickle: bool,
        scope: str,
        parcellation_name: str,
    ) -> None:
        """
        Generates the outer product plots (either individual plots for each CAP or a single subplot
        if ``subplot`` is True).
        """
        self._outer_products[group] = {}

        plot_labels = (
            {"xticklabels": full_labels, "yticklabels": full_labels}
            if scope == "regions"
            else {"xticklabels": [], "yticklabels": []}
        )

        # Create base grid for subplots
        if subplots:
            fig, axes, axes_coord, shape = self._initialize_outer_product_subplot(
                cap_dict, group, plot_dict, suffix_title
            )
            axes_x, axes_y = axes_coord
            ncol, nrow = shape

        for cap in cap_dict[group]:
            # Calculate outer product
            self._outer_products[group].update(
                {cap: np.outer(cap_dict[group][cap], cap_dict[group][cap])}
            )

            # Create labels if nodes requested for scope
            if scope == "nodes" and plot_dict["hemisphere_labels"] is False:
                reduced_labels, _ = self._collapse_node_labels(
                    parcellation_name,
                    self._parcel_approach,
                    node_ids=full_labels if parcellation_name == "Custom" else None,
                )

            if subplots:
                ax = axes[axes_y] if nrow == 1 else axes[axes_x, axes_y]
                # Modify tick labels based on scope
                if scope == "regions":
                    display = seaborn.heatmap(
                        ax=ax,
                        data=self._outer_products[group][cap],
                        **plot_labels,
                        **_PlotFuncs.base_kwargs(plot_dict),
                    )
                else:
                    # No labels
                    display = seaborn.heatmap(
                        ax=ax,
                        data=self._outer_products[group][cap],
                        **_PlotFuncs.base_kwargs(
                            plot_dict, *_PlotFuncs.extra_kwargs(plot_dict["hemisphere_labels"])
                        ),
                    )

                    if plot_dict["hemisphere_labels"] is False:
                        ax = _PlotFuncs.set_ticks(ax, reduced_labels)
                    else:
                        ax, division_line, plot_dict["linewidths"] = _PlotFuncs.division_line(
                            display=ax,
                            linewidths=plot_dict["linewidths"],
                            n_labels=len(self._parcel_approach[parcellation_name]["nodes"]),
                            add_labels=["LH", "RH"],
                        )

                        ax.axhline(
                            division_line,
                            color=plot_dict["linecolor"],
                            linewidth=plot_dict["linewidths"],
                        )
                        ax.axvline(
                            division_line,
                            color=plot_dict["linecolor"],
                            linewidth=plot_dict["linewidths"],
                        )

                # Add border; if "borderwidths" is Falsy, returns display unmodified
                display = _PlotFuncs.border(
                    display, plot_dict, axhline=self._outer_products[group][cap].shape[0]
                )

                # Modify label sizes for x axis
                display = _PlotFuncs.label_size(display, plot_dict, set_x=True, set_y=False)

                # Modify label sizes for y axis; if share_y, only set y for plots at axes == 0
                if plot_dict["sharey"]:
                    display = (
                        _PlotFuncs.label_size(display, plot_dict, False, set_y=True)
                        if axes_y == 0
                        else display
                    )
                else:
                    display = _PlotFuncs.label_size(display, plot_dict, False, set_y=True)

                # Set title of subplot
                ax.set_title(cap, fontsize=plot_dict["fontsize"])

                # If modulus is zero, move onto the new column back (to zero index of new column)
                if (axes_y % ncol == 0 and axes_y != 0) or axes_y == ncol - 1:
                    axes_x += 1
                    axes_y = 0
                else:
                    axes_y += 1
            else:
                # Create new plot for each iteration when not subplot
                plt.figure(figsize=plot_dict["figsize"])

                if scope == "regions":
                    display = seaborn.heatmap(
                        self._outer_products[group][cap],
                        **plot_labels,
                        **_PlotFuncs.base_kwargs(plot_dict),
                    )
                else:
                    display = seaborn.heatmap(
                        self._outer_products[group][cap],
                        **plot_labels,
                        **_PlotFuncs.base_kwargs(
                            plot_dict, *_PlotFuncs.extra_kwargs(plot_dict["hemisphere_labels"])
                        ),
                    )

                    if plot_dict["hemisphere_labels"] is False:
                        display = _PlotFuncs.set_ticks(display, reduced_labels)
                    else:
                        display, division_line, plot_dict["linewidths"] = _PlotFuncs.division_line(
                            display=display,
                            linewidths=plot_dict["linewidths"],
                            n_labels=len(self._parcel_approach[parcellation_name]["nodes"]),
                            add_labels=["LH", "RH"],
                        )

                        plt.axhline(
                            division_line,
                            color=plot_dict["linecolor"],
                            linewidth=plot_dict["linewidths"],
                        )
                        plt.axvline(
                            division_line,
                            color=plot_dict["linecolor"],
                            linewidth=plot_dict["linewidths"],
                        )

                # Add border; if "borderwidths" is Falsy, returns display unmodified
                display = _PlotFuncs.border(
                    display, plot_dict, axhline=self._outer_products[group][cap].shape[0]
                )

                # Set title
                display = _PlotFuncs.set_title(display, f"{group} {cap}", suffix_title, plot_dict)

                # Modify label sizes
                display = _PlotFuncs.label_size(display, plot_dict)

                # Save individual plots
                if output_dir:
                    filename = io_utils._filename(
                        f"{group}_{cap}_outer_product-{scope}", suffix_filename, "suffix", "png"
                    )
                    _PlotFuncs.save_fig(display, output_dir, filename, plot_dict, as_pickle)

        if subplots:
            # Remove subplots with no data
            [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]

            if output_dir:
                filename = io_utils._filename(
                    f"{group}_CAPs_outer_product-{scope}", suffix_filename, "suffix", "png"
                )
                _PlotFuncs.save_fig(display, output_dir, filename, plot_dict, as_pickle)

        _PlotFuncs.show(show_figs)

    @staticmethod
    def _initialize_outer_product_subplot(
        cap_dict: dict[str, dict[str, NDArray]],
        group: str,
        plot_dict: dict[str, Any],
        suffix_title: Union[str, None],
    ) -> tuple[plt.Figure, plt.Axes, tuple[int, int], tuple[int, int]]:
        """
        Initializes the subplot for "outer_product".

        Returns
        -------
        tuple
            Contains the matplotlib figure, matplolib axes, tuple representing the row and
            column position of the current suplot (0, 0), and tuple representing the number of
            rows and columns in the subplot.
        """
        # Max five subplots per row for default
        default_col = len(cap_dict[group]) if len(cap_dict[group]) <= 5 else 5
        ncol = plot_dict["ncol"] if plot_dict["ncol"] is not None else default_col
        ncol = min(ncol, len(cap_dict[group]))

        # Determine number of rows needed based on ceiling if not specified
        nrow = (
            plot_dict["nrow"]
            if plot_dict["nrow"] is not None
            else int(np.ceil(len(cap_dict[group]) / ncol))
        )
        subplot_figsize = (
            (8 * ncol, 6 * nrow) if plot_dict["figsize"] == (8, 6) else plot_dict["figsize"]
        )
        fig, axes = plt.subplots(
            nrow, ncol, sharex=False, sharey=plot_dict["sharey"], figsize=subplot_figsize
        )

        fig = _PlotFuncs.set_title(fig, f"{group}", suffix_title, plot_dict, is_subplot=True)
        fig.subplots_adjust(hspace=plot_dict["hspace"], wspace=plot_dict["wspace"])

        if plot_dict["tight_layout"]:
            fig.tight_layout(rect=plot_dict["rect"])

        return fig, axes, (0, 0), (ncol, nrow)

    def _generate_heatmap_plots(
        self,
        group: str,
        plot_dict: dict[str, Any],
        cap_dict: dict[str, dict[str, NDArray]],
        full_labels: list[str],
        output_dir: Union[str, None],
        suffix_title: Union[str, None],
        suffix_filename: Union[str, None],
        show_figs: bool,
        as_pickle: bool,
        scope: str,
        parcellation_name: str,
    ) -> None:
        """Generates one heatmap per group."""
        plt.figure(figsize=plot_dict["figsize"])
        plot_labels = {"xticklabels": True, "yticklabels": True}

        if scope == "regions":
            display = seaborn.heatmap(
                pd.DataFrame(cap_dict[group], index=full_labels),
                **plot_labels,
                **_PlotFuncs.base_kwargs(plot_dict),
            )
        else:
            # Create Labels
            if plot_dict["hemisphere_labels"] is False:
                reduced_labels, collapsed_node_names = self._collapse_node_labels(
                    parcellation_name, self._parcel_approach, full_labels
                )

                display = seaborn.heatmap(
                    pd.DataFrame(cap_dict[group], columns=list(cap_dict[group])),
                    **plot_labels,
                    **_PlotFuncs.base_kwargs(plot_dict),
                )

                plt.yticks(
                    ticks=[indx for indx, label in enumerate(reduced_labels) if label],
                    labels=collapsed_node_names,
                )
            else:
                display = seaborn.heatmap(
                    pd.DataFrame(cap_dict[group], columns=list(cap_dict[group])),
                    **plot_labels,
                    **_PlotFuncs.base_kwargs(plot_dict, line=False, edge=False),
                )

                display, division_line, plot_dict["linewidths"] = _PlotFuncs.division_line(
                    display=display,
                    linewidths=plot_dict["linewidths"],
                    n_labels=len(self._parcel_approach[parcellation_name]["nodes"]),
                    set_x=False,
                    add_labels=["LH", "RH"],
                )

                plt.axhline(
                    division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"]
                )

        # Add border; if "borderwidths" is Falsy, returns display unmodified
        display = _PlotFuncs.border(
            display,
            plot_dict,
            axhline=len(cap_dict[group][list(cap_dict[group])[0]]),
            axvline=len(self._caps[group]),
        )

        # Modify label sizes
        display = _PlotFuncs.label_size(display, plot_dict)

        # Set title
        display = _PlotFuncs.set_title(display, f"{group} CAPs", suffix_title, plot_dict)

        if output_dir:
            filename = io_utils._filename(
                f"{group}_CAPs_heatmap-{scope}", suffix_filename, "suffix", "png"
            )
            _PlotFuncs.save_fig(display, output_dir, filename, plot_dict, as_pickle)

        _PlotFuncs.show(show_figs)

    @staticmethod
    def _collapse_node_labels(
        parcellation_name: str,
        parcel_approach: ParcelApproach,
        node_ids: Union[list[str], None] = None,
    ) -> tuple[list[str], list[str]]:
        """
        Collapses node labels names (based on unique node names and hemisphere) for plotting
        purposes. For instance in the Schaefer parcellation, nodes containing "Vis" have a left and
        right hemisphere version, instead of ["LH_Vis_1", "LH_Vis_2", "RH_Vis_1", "RH_Vis_2", ...],
        the unique names would be "LH Vis" and "RH Vis". The frequencies of nodes containing the
        unique node name and hemisphere combination are computed to reduce plot clutter when "nodes"
        are plotted.

        Returns
        -------
        tuple
            Consists of a two lists. The first list is the same length as "nodes" in
            ``self._parcel_approach`` and contains the unique node and hemisphere combination at
            certain indices, with the remaining indices being empty strings. The second list only
            contains the names of the unique node and hemisphere comvination.
        """
        # Get frequency of each major hemisphere and region in Schaefer, AAL, or Custom atlas
        if parcellation_name == "Schaefer":
            nodes = parcel_approach[parcellation_name]["nodes"]
            # Retain only the hemisphere and primary Schaefer network
            # Node string in form of {hemisphere}_{region}_{number} (e.g. LH_Cont_Par_1)
            # Below code returns a list where each element is a list of [Hemisphere, Network]
            # (e.g ["LH", "Vis"])
            hemi_network_pairs = [node.split("_")[:2] for node in nodes]
            frequency_dict = collections.Counter([" ".join(node) for node in hemi_network_pairs])
        elif parcellation_name == "AAL":
            nodes = parcel_approach[parcellation_name]["nodes"]
            # AAL in the form of {region}_{hemisphere}_{number} or {region}_{hemisphere)
            # (e.g. Frontal_Inf_Orb_2_R, Precentral_L); _collapse_aal_node_names would return these
            # as Frontal and Pre
            collapsed_aal_nodes = _collapse_aal_node_names(nodes, return_unique_names=False)
            frequency_dict = collections.Counter(collapsed_aal_nodes)
        else:
            frequency_dict = {}
            for node_id in node_ids:
                # For custom, columns comes in the form of "{Hemisphere} {Region}"
                hemisphere_id = "LH" if node_id.startswith("LH ") else "RH"
                region_id = re.split("LH |RH ", node_id)[-1]
                node_indices = parcel_approach["Custom"]["regions"][region_id][
                    hemisphere_id.lower()
                ]
                frequency_dict.update({node_id: len(node_indices)})

        # Get the names, which indicate the hemisphere and region; reverting Counter objects to
        # list retains original ordering of nodes in list as of Python 3.7
        collapsed_node_labels = list(frequency_dict)
        tick_labels = ["" for _ in range(len(parcel_approach[parcellation_name]["nodes"]))]

        starting_value = 0

        # Iterate through names_list and assign the starting indices corresponding to unique region
        # and hemisphere key
        for num, collapsed_node_label in enumerate(collapsed_node_labels):
            if num == 0:
                tick_labels[0] = collapsed_node_label
            else:
                # Shifting to previous frequency of the preceding network to obtain the new starting
                # value of the subsequent region and hemisphere pair
                starting_value += frequency_dict[collapsed_node_labels[num - 1]]
                tick_labels[starting_value] = collapsed_node_label

        return tick_labels, collapsed_node_labels

    def caps2corr(
        self,
        output_dir: Optional[str] = None,
        suffix_title: Optional[str] = None,
        suffix_filename: Optional[str] = None,
        show_figs: bool = True,
        save_plots: bool = True,
        return_df: bool = False,
        save_df: bool = False,
        as_pickle: bool = False,
        method: str = "pearson",
        **kwargs,
    ) -> Union[dict[str, pd.DataFrame], None]:
        """
        Generate a Correlation Matrix for CAPs.

        Produces a correlation matrix of all CAPs. Separate correlation matrices are created for
        each group.

        Parameters
        ----------
        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save plots (if ``save_plots`` is True) and correlation matrices DataFrames
            (if ``save_df`` is True). The directory will be created if it does not exist. If None,
            plots and dataFrame will not be saved.

        suffix_title: :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the filename of each saved plot if ``output_dir`` is provided.

        show_figs: :obj:`bool`, default=True
            Display figures.

        save_plots: :obj:`bool`, default=True
            If True, plots are saves as png images. For this to be used, ``output_dir`` must be
            specified.

        return_df: :obj:`bool`, default=False
            If True, returns a dictionary with a correlation matrix for each group.

        save_df: :obj:`bool`, default=False
            If True, saves the correlation matrix contained in the DataFrames as csv files. For
            this to be used, ``output_dir`` must be specified.

        as_pickle: :obj:`bool`, default=False
            When ``output_dir`` is specified, plots are saved as pickle files, which can be further
            modified, instead of png images.

            .. versionadded:: 0.26.5

        method: :obj:`str`, default="pearson"
            Type of correlation method to use. Options are "pearson" or "spearman".

            .. versionadded:: 0.29.6

        **kwargs
            Keyword arguments used when modifying figures. Valid keywords include:

            - dpi: :obj:`int`, default=300 -- Dots per inch for the figure.
            - figsize: :obj:`tuple`, default=(8, 6) -- Size of the figure in inches.
            - fontsize: :obj:`int`, default=14 -- Font size for the title each plot.
            - xticklabels_size: :obj:`int`, default=8 -- Font size for x-axis tick labels.
            - yticklabels_size: :obj:`int`, default=8 -- Font size for y-axis tick labels.
            - shrink: :obj:`float`, default=0.8 -- Fraction by which to shrink the colorbar.
            - cbarlabels_size: :obj:`int`, default=8 -- Font size for the colorbar labels.
            - xlabel_rotation: :obj:`int`, default=0 -- Rotation angle for x-axis labels.
            - ylabel_rotation: :obj:`int`, default=0 -- Rotation angle for y-axis labels.
            - annot: :obj:`bool`, default=False -- Add values to each cell.
            - annot_kws: :obj:`dict`, default=None -- Customize the annotations.
            - fmt: :obj:`str`, default=".2g" -- Modify how the annotated vales are presented.
            - linewidths: :obj:`float`, default=0 -- Padding between each cell in the plot.
            - borderwidths: :obj:`float`, default=0 -- Width of the border around the plot.
            - linecolor: :obj:`str`, default="black" -- Color of the line that seperates each cell.
            - edgecolors: :obj:`str` or :obj:`None`, default=None -- Color of the edges.
            - alpha: :obj:`float` or :obj:`None`, default=None -- Controls transparency and ranges
              from 0 (transparent) to 1 (opaque).
            - bbox_inches: :obj:`str` or :obj:`None`, default="tight" -- Alters size of the
              whitespace in the saved image.
            - cmap: :obj:`str`, :obj:`callable` default="coolwarm" -- Color map for the plot cells.
              Options include strings to call seaborn's pre-made palettes, ``seaborn.diverging_palette``
              function to generate custom palettes, and ``matplotlib.color.LinearSegmentedColormap``
              to generate custom palettes.
            - vmin: :obj:`float` or :obj:`None`, default=None -- The minimum value to display in colormap.
            - vmax: :obj:`float` or :obj:`None`, default=None -- The maximum value to display in colormap.

        Returns
        -------
        dict[str, pd.DataFrame]
            A dictionary mapping an instance of a pandas DataFrame for each group. Only returned if
            ``return_df`` is True.

        Note
        ----
        **Color Palettes**: Refer to `seaborn's Color Palettes\
        <https://seaborn.pydata.org/tutorial/color_palettes.html>`_ for valid pre-made palettes.

        **Significance Values**: If ``return_df`` is True, each element will contain its uncorrected
        p-value in parenthesis with a single asterisk if < 0.05, a double asterisk if < 0.01, and a
        triple asterisk < 0.001.
        """
        self._check_required_attrs(["_caps"])

        assert method in [
            "spearman",
            "pearson",
        ], "Options for `method` are 'pearson' or 'spearman'."

        io_utils._issue_file_warning("suffix_filename", suffix_filename, output_dir)

        corr_dict = {group: None for group in self._groups} if return_df or save_df else None

        # Create plot dictionary
        plot_dict = _check_kwargs(_PlotDefaults.caps2corr(), **kwargs)

        # Dictionary of scipy correlation functions
        corr_func = {"pearson": pearsonr, "spearman": spearmanr}

        for group in self._groups:
            df = pd.DataFrame(self._caps[group])
            corr_df = df.corr(method=method)

            display = _MatrixVisualizer.create_display(
                corr_df, plot_dict, suffix_title, group, "corr"
            )

            if corr_dict:
                # Get p-values; use np.eye to make main diagonals equal zero; implementation of
                # tozCSS from:
                # https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
                pval_df = df.corr(method=lambda x, y: corr_func[method](x, y)[1]) - np.eye(
                    *corr_df.shape
                )
                # Add asterisk to values that meet the threshold
                pval_df = pval_df.map(
                    lambda x: f'({format(x, plot_dict["fmt"])})'
                    + "".join(["*" for code in [0.05, 0.01, 0.001] if x < code])
                )
                # Add the p-values to the correlation matrix
                corr_dict[group] = (
                    corr_df.map(lambda x: f'{format(x, plot_dict["fmt"])}') + " " + pval_df
                )

            if output_dir:
                _MatrixVisualizer.save_contents(
                    output_dir=output_dir,
                    suffix_filename=suffix_filename,
                    group=group,
                    curr_dict=corr_dict,
                    plot_dict=plot_dict,
                    save_plots=save_plots,
                    save_df=save_df,
                    display=display,
                    as_pickle=as_pickle,
                    call="corr",
                )

            _PlotFuncs.show(show_figs)

        if return_df:
            return corr_dict

    def caps2niftis(
        self,
        output_dir: str,
        suffix_filename: Optional[str] = None,
        fwhm: Optional[float] = None,
        knn_dict: Optional[dict[str, Union[int, list[int], NDArray[np.integer], str]]] = None,
        progress_bar: bool = False,
    ) -> Self:
        """
        Convert CAPs to NIfTI Statistical Maps.

        Projects CAPs onto the parcellation in defined in ``parcel_approach`` to create compressed
        NIfTI (.nii.gz) statistical maps. One image is generated per CAP and separate images are
        generated per group.

        Parameters
        ----------
        output_dir: :obj:`str`
            Directory to save nii.gz files. The directory will be created if it does not exist.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the name of the saved file.

        fwhm: :obj:`float` or :obj:`None`, default=None
            Strength of spatial smoothing to apply (in millimeters) to the statistical map prior to
            interpolating from MNI152 space to fsLR surface space. Uses Nilearn's ``smooth_img``.

        knn_dict: :obj:`dict[str, int | bool]`, default=None
            Use KNN (k-nearest neighbors) interpolation with reference atlas masking to fill in
            non-background coordinates that are assigned zero. Useful when custom parcellation does
            not project well from volumetric to surface space. The following subkeys are recognized:

            - "k": An integer (Default=1). Determines the number of nearest neighbors to consider.
            - "reference_atlas": A string (Default="Schaefer"). Specifies the atlas to use for
              reference masking ("AAL" or "Schaefer").
            - "resolution_mm": An integer (Default=1). Spatial resolution of the Schaefer
              parcellation (in millimeters) (1 or 2).
            - "remove_labels": A list or array (Default=None). The label IDs as integers of the
              regions in the parcellation to not interpolate.
            - "method": A string (Default="majority_vote"). Method used to assign new values to
              non-background voxels ("majority_vote" or "distance_weighted"). For majority vote,
              the most frequently appearing value among "k" choices (or chosen neighbors) is, while
              the distance weighted approach uses inverse distance weighting (1/distance) to
              estimate the new averaged value for the non-background voxel.

              .. versionadded:: 0.29.5 "method" key added.

            .. note:: KNN interpolation is applied before ``fwhm``.

        progress_bar: :obj:`bool`, default=False
            If True, displays a progress bar.

        Returns
        -------
        self

        Important
        ---------
        **Parcellation Approach**: ``parcel_approach`` must have the "maps" subkey containing the
        path to the parcellation NIfTI file.

        **Assumption**: This function assumes that the background label for the parcellation is
        zero. When the numerical labels from the parcellation map are extracted, the first element
        (assumed to be zero/the background label after sorting) is skipped. Then the remaining
        sorted labels are iterated over to map each element of the CAP cluster centroid onto the
        corresponding non-zero label IDs in the parcellation.
        """
        self._check_required_attrs(["_parcel_approach", "_caps"])

        knn_dict = self._validate_knn_dict(knn_dict)

        io_utils._makedir(output_dir)

        parcellation_name = list(self._parcel_approach)[0]
        for group in self._groups:
            for cap in tqdm(
                self._caps[group],
                desc=f"Generating Statistical Maps [GROUP: {group}]",
                disable=not progress_bar,
            ):
                stat_map = _cap2statmap(
                    atlas_file=self._parcel_approach[parcellation_name]["maps"],
                    cap_vector=self._caps[group][cap],
                    fwhm=fwhm,
                    knn_dict=knn_dict,
                )

                filename = io_utils._filename(
                    f"{group.replace(' ', '_')}_{cap}", suffix_filename, "suffix", "nii.gz"
                )
                nib.save(stat_map, os.path.join(output_dir, filename))

        return self

    @staticmethod
    def _validate_knn_dict(
        knn_dict: Union[dict[str, Any], None],
    ) -> Union[dict[str, Union[int, list[int], NDArray[np.integer], str]], None]:
        """Validates the ``knn_dict``."""
        if not knn_dict:
            return None

        valid_atlases = ("Schaefer", "AAL")
        valid_methods = ("majority_vote", "distance_weighted")

        if knn_dict.get("reference_atlas") and knn_dict.get("reference_atlas") not in valid_atlases:
            raise ValueError(
                "In `knn_dict`, 'reference_atlas' must be a string ('Schaefer' or 'AAL')."
            )
        else:
            knn_dict["reference_atlas"] = "Schaefer"
            LG.warning(
                "'reference_atlas' not specified in `knn_dict`. The default reference atlas is "
                "'Schaefer'."
            )

        if knn_dict["reference_atlas"] == "Schaefer":
            if not knn_dict.get("resolution_mm"):
                knn_dict["resolution_mm"] = 1
                LG.warning(
                    "Defaulting to 1mm resolution for the Schaefer atlas since 'resolution_mm' "
                    "was not specified in `knn_dict`."
                )
        else:
            if "resolution_mm" in knn_dict:
                LG.warning(
                    "In `knn_dict`, 'resolution_mm' only used when 'reference_atlas' is . "
                    "set to 'Schaefer'."
                )
            knn_dict["resolution_mm"] = None

        if not knn_dict.get("k"):
            knn_dict["k"] = 3
            LG.warning("Defaulting to k=3 since 'k' was not specified in `knn_dict`.")

        if not knn_dict.get("method"):
            knn_dict["method"] = "majority_vote"
            LG.warning(
                "Defaulting to 'majority_vote' since 'method' was not specified in `knn_dict`."
            )
        elif knn_dict.get("method") not in valid_methods:
            raise ValueError(
                "In `knn_dict`, 'method' must be 'majority_vote' or 'distance_weighted'."
            )

        return knn_dict

    def caps2surf(
        self,
        output_dir: Optional[str] = None,
        suffix_title: Optional[str] = None,
        suffix_filename: Optional[str] = None,
        show_figs: bool = True,
        fwhm: Optional[float] = None,
        fslr_density: Literal["4k", "8k", "32k", "164k"] = "32k",
        method: Literal["linear", "nearest"] = "linear",
        save_stat_maps: bool = False,
        fslr_giftis_dict: Optional[dict] = None,
        knn_dict: Optional[dict[str, Union[int, list[int], NDArray[np.integer], str]]] = None,
        progress_bar: bool = False,
        as_pickle: bool = False,
        **kwargs,
    ) -> Self:
        """
        Project CAPs onto Surface Plots.

        Projects CAPs onto the parcellation defined in ``parcel_approach`` to create NIfTI
        statistical maps. Then transforms these maps from volumetric to surface space and generates
        visualizations. One surface plot image is generated per CAP and separate images are produced
        per group.

        Parameters
        ----------
        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save plots as png files. The directory will be created if it does not exist.
            If None, plots will not be saved.

        suffix_title: :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the filename of each saved plot if ``output_dir`` is provided.

        show_figs: :obj:`bool`, default=True
            Display figures.

        fwhm: :obj:`float` or :obj:`None`, defualt=None
            Strength of spatial smoothing to apply (in millimeters) to the statistical map prior to
            interpolating from MNI152 space to fsLR surface space.

        fslr_density: {"4k", "8k", "32k", "164k"}, default="32k"
            Density of the fsLR surface when converting from MNI152 space to fsLR surface. Options
            are "32k" or "164k". If using ``fslr_giftis_dict`` options are "4k", "8k", "32k", and "164k".

        method: {"linear", "nearest"}, default="linear"
            Interpolation method to use when converting from MNI152 space to fsLR surface or from
            fsLR to fsLR. Options are "linear" or "nearest".

        save_stat_maps: :obj:`bool`, default=False
            If True, saves the statistical map for each CAP for all groups as a Nifti1Image if
            ``output_dir`` is provided.

        fslr_giftis_dict: :obj:`dict` or :obj:`None`, default=None
            Dictionary specifying precomputed GifTI files in fsLR space for plotting statistical maps.

            ::

                {
                    "GroupName": {
                        "CAP-1": {
                            "lh": "path/to/left_hemisphere_gifti",
                            "rh": "path/to/right_hemisphere_gifti",
                        },
                    }
                }

        knn_dict: :obj:`dict[str, int | bool]`, default=None
            Use KNN (k-nearest neighbors) interpolation with reference atlas masking to fill in
            non-background coordinates that are assigned zero. Useful when custom parcellation does
            not project well from volumetric to surface space. The following subkeys are recognized:

            - "k": An integer (Default=1). Determines the number of nearest neighbors to consider.
            - "reference_atlas": A string (Default="Schaefer"). Specifies the atlas to use for
              reference masking ("AAL" or "Schaefer").
            - "resolution_mm": An integer (Default=1). Spatial resolution of the Schaefer
              parcellation (in millimeters) (1 or 2).
            - "remove_labels": A list or array (Default=None). The label IDs as integers of the
              regions in the parcellation to not interpolate.
            - "method": A string (Default="majority_vote"). Method used to assign new values to
              non-background voxels ("majority_vote" or "distance_weighted"). For majority vote,
              the most frequently appearing value among "k" choices (or chosen neighbors) is, while
              the distance weighted approach uses inverse distance weighting (1/distance) to
              estimate the new averaged value for the non-background voxel.

              .. versionadded:: 0.29.5 "method" key added.

            .. note:: KNN interpolation is applied before ``fwhm``.

        progress_bar: :obj:`bool`, default=False
            If True, displays a progress bar.

        as_pickle: :obj:`bool`, default=False
            When ``output_dir`` is specified, plots are saved as pickle files, which can be further
            modified, instead of png images.

            .. versionadded:: 0.26.5

        **kwargs
            Additional parameters to pass to modify certain plot parameters. Options include:

            - dpi: :obj:`int`, default=300 -- Dots per inch for the plot.
            - title_pad: :obj:`int`, default=-3 -- Padding for the plot title.
            - cmap: :obj:`str` or :obj:`callable`, default="cold_hot" -- Colormap to be used for the plot.
            - cbar_kws: :obj:`dict`, default={"location": "bottom", "n_ticks": 3} -- Customize colorbar.
              Refer to ``_add_colorbars`` for ``surfplot.plotting.Plot`` in `Surfplot's Plot
              Documentation <https://surfplot.readthedocs.io/en/latest/generated/surfplot.plotting.Plot.html#surfplot.plotting.Plot._add_colorbars>`_
              for valid parameters.
            - alpha: :obj:`float`, default=1 -- Transparency level of the colorbar.
            - as_outline: :obj:`bool`, default=False -- Plots only an outline of contiguous vertices
              with the same value.
            - outline_alpha: :obj:`float`, default=1 -- Transparency level of the colorbar for
              outline if ``as_outline`` is True.
            - zero_transparent: :obj:`bool`, default=True -- Turns vertices with a value of 0
              transparent.
            - size: :obj:`tuple`, default=(500, 400) -- Size of the plot in pixels.
            - layout: :obj:`str`, default="grid" -- Layout of the plot.
            - zoom: :obj:`float`, default=1.5 -- Zoom level for the plot.
            - views: {"lateral", "medial"} or :obj:`list[{"lateral", "medial}]`,\
              default=["lateral", "medial"] -- Views to be displayed in the plot.
            - brightness: :obj:`float`, default=0.5 -- Brightness level of the plot.
            - figsize: :obj:`tuple` or :obj:`None`, default=None -- Size of the figure.
            - scale: :obj:`tuple`, default=(2, 2) -- Scale factors for the plot.
            - surface: {"inflated", "veryinflated"}, default="inflated" -- The surface atlas that
              is used for plotting. Options are "inflated" or "veryinflated".
            - color_range: :obj:`tuple` or :obj:`None`, default=None -- The minimum and maximum
              value to display in plots. For instance, (-1, 1) where minimum value is first. If
              None, the minimum and maximum values from the image will be used.
            - bbox_inches: :obj:`str` or :obj:`None`, default="tight" -- Alters size of the
              whitespace in the saved image.

        Returns
        -------
        self

        Important
        ---------
        **Parcellation Approach**: ``parcel_approach`` must have the "maps" subkey containing the
        path to the NIfTI file of the parcellation.

        **Assumptions**: This function assumes that the background label for the parcellation is
        zero. During extraction of the numerical labels from the parcellation map, the first element
        (assumed to be zero/the background label after sorting) is skipped. Then the remaining sorted
        labels are iterated over to map each element of the CAP cluster centroid onto the
        corresponding non-zero label IDs in the parcellation.

        Additionally, this funcition assumes that the parcellation map is in volumetric MNI space
        unless ``fslr_giftis_dict`` is defined, then this function assumes the maps are in surface
        space.
        """
        if fslr_giftis_dict is None:
            check_params = ["_parcel_approach", "_caps"]
            self._check_required_attrs(check_params)

            parcellation_name = list(self._parcel_approach)[0]

        knn_dict = self._validate_knn_dict(knn_dict)

        io_utils._issue_file_warning("suffix_filename", suffix_filename, output_dir)
        io_utils._makedir(output_dir)

        # Create plot dictionary
        plot_dict = _check_kwargs(_PlotDefaults.caps2surf(), **kwargs)

        groups = (
            self._caps if hasattr(self, "_caps") and fslr_giftis_dict is None else fslr_giftis_dict
        )
        for group in groups:
            caps = (
                self._caps[group]
                if hasattr(self, "_caps") and fslr_giftis_dict is None
                else fslr_giftis_dict[group]
            )
            for cap in tqdm(
                caps, desc=f"Generating Surface Plots [GROUP: {group}]", disable=not progress_bar
            ):
                if fslr_giftis_dict is None:
                    stat_map = _cap2statmap(
                        atlas_file=self._parcel_approach[parcellation_name]["maps"],
                        cap_vector=self._caps[group][cap],
                        fwhm=fwhm,
                        knn_dict=knn_dict,
                    )

                    # Fix for python 3.12, saving stat map so that it is path instead of a NIfTI
                    try:
                        gii_lh, gii_rh = mni152_to_fslr(
                            stat_map, method=method, fslr_density=fslr_density
                        )
                    except TypeError:
                        temp_nifti = self._create_temp_nifti(stat_map)
                        gii_lh, gii_rh = mni152_to_fslr(
                            temp_nifti.name, method=method, fslr_density=fslr_density
                        )
                        # Delete
                        os.unlink(temp_nifti.name)
                else:
                    gii_lh, gii_rh = fslr_to_fslr(
                        (fslr_giftis_dict[group][cap]["lh"], fslr_giftis_dict[group][cap]["rh"]),
                        target_density=fslr_density,
                        method=method,
                    )

                fig = self._generate_surface_plot(
                    plot_dict, gii_lh, gii_rh, group, cap, suffix_title
                )

                if output_dir:
                    filename = io_utils._filename(
                        f"{group.replace(' ', '_')}_{cap}_surface", suffix_filename, "suffix", "png"
                    )
                    _PlotFuncs.save_fig(fig, output_dir, filename, plot_dict, as_pickle)

                    if save_stat_maps:
                        nib.save(
                            stat_map,
                            os.path.join(output_dir, filename.split("_surface")[0] + ".nii.gz"),
                        )

                try:
                    plt.show(fig) if show_figs else plt.close(fig)
                except:
                    _PlotFuncs.show(show_figs)

        return self

    @staticmethod
    def _create_temp_nifti(stat_map: nib.Nifti1Image) -> tempfile._TemporaryFileWrapper:
        """Creates a temporary NIfTI image as a workaround for Python 3.12 issue."""
        # Create temp file
        temp_nifti = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
        LG.warning(
            "TypeError raised by neuromaps due to changes in pathlib.py in Python 3.12 "
            "Converting NIfTI image into a temporary nii.gz file (which will be "
            f"automatically deleted afterwards) [TEMP FILE: {temp_nifti.name}]"
        )

        # Ensure file is closed
        temp_nifti.close()
        # Save temporary nifti to temp file
        nib.save(stat_map, temp_nifti.name)

        return temp_nifti

    @staticmethod
    def _generate_surface_plot(
        plot_dict: dict[str, Any],
        gii_lh: tuple[nib.gifti.GiftiImage],
        gii_rh: tuple[nib.gifti.GiftiImage],
        group: str,
        cap: str,
        suffix_title: Union[str, None],
    ) -> plt.Figure:
        """Creates the surface plot."""
        # Code adapted from example on https://surfplot.readthedocs.io/
        surfaces = fetch_fslr()

        if plot_dict["surface"] not in ["inflated", "veryinflated"]:
            LG.warning(
                f"{plot_dict['surface']} is an invalid option for `surface`. Available options "
                "include 'inflated' or 'verinflated'. Defaulting to 'inflated'."
            )
            plot_dict["surface"] = "inflated"

        lh, rh = surfaces[plot_dict["surface"]]
        lh = str(lh) if not isinstance(lh, str) else lh
        rh = str(rh) if not isinstance(rh, str) else rh
        sulc_lh, sulc_rh = surfaces["sulc"]
        sulc_lh = str(sulc_lh) if not isinstance(sulc_lh, str) else sulc_lh
        sulc_rh = str(sulc_rh) if not isinstance(sulc_rh, str) else sulc_rh

        p = surfplot.Plot(
            lh,
            rh,
            size=plot_dict["size"],
            layout=plot_dict["layout"],
            zoom=plot_dict["zoom"],
            views=plot_dict["views"],
            brightness=plot_dict["brightness"],
        )

        # Add base layer
        p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)
        cmap = (
            _cmap_d[plot_dict["cmap"]] if isinstance(plot_dict["cmap"], str) else plot_dict["cmap"]
        )
        # Add stat map layer
        p.add_layer(
            {"left": gii_lh, "right": gii_rh},
            cmap=cmap,
            alpha=plot_dict["alpha"],
            color_range=plot_dict["color_range"],
            zero_transparent=plot_dict["zero_transparent"],
            as_outline=False,
        )

        if plot_dict["as_outline"] is True:
            p.add_layer(
                {"left": gii_lh, "right": gii_rh},
                cmap="gray",
                cbar=False,
                alpha=plot_dict["outline_alpha"],
                as_outline=True,
            )

        # Color bar
        fig = p.build(
            cbar_kws=plot_dict["cbar_kws"], figsize=plot_dict["figsize"], scale=plot_dict["scale"]
        )
        fig_name = f"{group} {cap} {suffix_title}" if suffix_title else f"{group} {cap}"
        fig.axes[0].set_title(fig_name, pad=plot_dict["title_pad"])

        return fig

    def caps2radar(
        self,
        output_dir: Optional[str] = None,
        suffix_title: Optional[str] = None,
        suffix_filename: Optional[str] = None,
        show_figs: bool = True,
        use_scatterpolar: bool = False,
        as_html: bool = False,
        as_json: bool = False,
        **kwargs,
    ) -> Self:
        """
        Generate Radar Plots for CAPs using Cosine Similarity.

        Calculates the cosine similarity between the "High Amplitude" (positive/above the mean) and
        "Low Amplitude" (negative/below the mean) activations of the CAP cluster centroid and each
        a-priori region or network in the parcellation defined by ``parcel_approach``. One image is
        generated per CAP and separate images are generated for each group.

        .. important::

          - This function assumes the mean for each ROI is 0 due to standardization.
          - The absolute values of the negative activations are computed for visualization purposes.

        The process involves the following steps:

            1. Extract Cluster Centroids:

              - Each CAP is represented by a cluster centroid, which is a 1 x ROI
                (Region of Interest) vector.

            2. Generate Binary Vectors:

              - For each region create a binary vector (1 x ROI) where "1" indicates that the ROI is
                part of the specific region and "0" otherwise.
              - In this example, the binary vector acts as a 1-D mask to isolate ROIs in the Visual
                Network by setting the corresponding indices to "1".

                ::

                    import numpy as np

                    # Define nodes with their corresponding label IDs
                    nodes = ["LH_Vis1", "LH_Vis2", "LH_SomSot1", "LH_SomSot2",
                             "RH_Vis1", "RH_Vis2", "RH_SomSot1", "RH_SomSot2"]

                    # Binary mask for the Visual Network (Vis)
                    binary_vector = np.array([1, 1, 0, 0, 1, 1, 0, 0])

            3. Isolate Positive and Negative Activations in CAP Centroid:

              - Positive activations are defined as the values in the CAP centroid that are greater
                than zero. These values represent the "High Amplitude" activations for that CAP.
              - Negative activations are defined as the values in the CAP centroid that are less
                than zero. These values represent the "Low Amplitude" activations for that CAP.

                ::

                    # Example cluster centroid for CAP 1
                    cap_1_cluster_centroid = np.array([-0.3, 1.5, 2.0, -0.2, 0.7, 1.3, -0.5, 0.4])

                    # Assign values less than 0 as 0 to isolate the high amplitude activations
                    high_amp = np.where(cap_1_cluster_centroid > 0, cap_1_cluster_centroid, 0)

                    # Assign values less than 0 as 0 to isolate the low amplitude activations
                    # Also invert the sign to restrict similarity to [0, 1]
                    low_amp = np.where(cap_1_cluster_centroid < 0, -cap_1_cluster_centroid, 0)

            4. Calculate Cosine Similarity:

              - Normalize the dot product by the product of the Euclidean norms of the cluster
                centroid and the binary vector to obtain the cosine similarity:

                ::

                    # Compute dot product between the binary vector each activation vector
                    high_dot = np.dot(high_amp, binary_vector)
                    low_dot = np.dot(low_amp, binary_vector)

                    # Compute the norms
                    high_norm = np.linalg.norm(high_amp)
                    low_norm = np.linalg.norm(low_amp)
                    bin_norm = np.linalg.norm(binary_vector)

                    # Calculate cosine similarity
                    high_cos = high_dot / (high_norm * bin_norm)
                    low_cos = low_dot / (low_norm * bin_norm)

            5. Generate Radar Plots of Each CAPs:

              - Each radar plot visualizes the cosine similarity for both "High Amplitude"
                (positive) and "Low Amplitude" (negative) activations of the CAP.

        Parameters
        ----------
        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save plots as png or html images. The directory will be created if it does
            not exist. If None, plots will not be saved.

        suffix_title: :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the filename of each saved plot if ``output_dir`` is provided.

        show_figs: :obj:`bool`, default=True
            Display figures. If the current Python session is non-interactive, then ``plotly.offline``
            is used to generate an html file named "temp-plot.html", which opens each plot in the
            default browser.

        use_scatterpolar: :obj:`bool`, default=False
            Uses ``plotly.graph_objects.Scatterpolar`` instead of ``plotly.express.line_polar``.

        as_html: :obj:`bool`, default=False
            When ``output_dir`` is specified, plots are saved as html file instead of png images.

        as_json: :obj:`bool`, default=False
            When ``output_dir`` is specified, plots are saved as json file, which can be further
            modified, instead of png images.

            .. versionadded:: 0.26.5

        **kwargs:
            Additional parameters to pass to modify certain plot parameters. Options include:

            - scale: :obj:`int`, default=2 -- If ``output_dir`` provided, controls resolution of
              image when saving. Serves a similar purpose as dpi.
            - savefig_options: :obj:`dict[str]`, default={"width": 3, "height": 3, "scale": 1} -- If
              ``output_dir`` provided, controls the width (in inches), height (in inches), and scale
              of the plot.
            - height: :obj:`int`, default=800 -- Height of the plot.
            - width: :obj:`int`, defualt=1200 -- Width of the plot.
            - line_close: :obj:`bool`, default=True -- Whether to close the lines
            - bgcolor: :obj:`str`, default="white" -- Color of the background
            - scattersize: :obj:`int`, default=8 -- Controls size of the dots when markers are used.
            - connectgaps: :obj:`bool`, default=True -- If ``use_scatterpolar=True``, controls if
              missing values are connected.
            - linewidth: :obj:`int`, default = 2 -- The width of the line connecting the values if
              ``use_scatterpolar=True``.
            - opacity: :obj:`float`, default=0.5 -- If ``use_scatterpolar=True``, sets the opacity
              of the trace.
            - fill: :obj:`str`, default="toself" -- If "toself" the are of the dots and within the
              boundaries of the line will be filled.

              .. versionchanged:: 0.26.0 Default changed from "none" to "toself".

            - mode: :obj:`str`, default="markers+lines" -- Determines how the trace is drawn.
              Can include "lines", "markers", "lines+markers", "lines+markers+text".
            - radialaxis: :obj:`dict`, default={"showline": False, "linewidth": 2, \
              "linecolor": "rgba(0, 0, 0, 0.25)", "gridcolor": "rgba(0, 0, 0, 0.25)", \
              ticks": "outside", "tickfont": {"size": 14, "color": "black"}} --
              Customizes the radial axis. Refer to `Plotly's radialaxis Documentation\
              <https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.polar.radialaxis.html>`_\
              or `Plotly's polar Documentation <https://plotly.com/python/reference/layout/polar/>`_
              for valid keys.
            - angularaxis: :obj:`dict`, default={"showline": True, "linewidth": 2, \
              "linecolor": "rgba(0, 0, 0, 0.25)", "gridcolor": "rgba(0, 0, 0, 0.25)", \
              "tickfont": {"size": 16, "color": "black"}} --
              Customizes the angular axis. Refer to `Plotly's angularaxis Documentation\
              <https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.polar.angularaxis.html>`_\
              or `Plotly's polar Documentation <https://plotly.com/python/reference/layout/polar/>`_ for valid keys.
            - color_discrete_map: :obj:`dict`, default={"High Amplitude": "red", "Low Amplitude": "blue"} --
              Change the color of the "High Amplitude" and "Low Amplitude" groups. Must use the keys
              "High Amplitude" and "Low Amplitude".
            - title_font: :obj:`dict`, default={"family": "Times New Roman", "size": 30, "color": "black"} --
              Modifies the font of the title. Refer to `Plotly's layout Documentation\
              <https://plotly.com/python/reference/layout/>`_ for valid keys.
            - title_x: :obj:`float`, default=0.5 -- Modifies x position of title.
            - title_y: :obj:`float`, default=None -- Modifies y position of title.
            - legend: :obj:`dict`, default={"yanchor": "top", "xanchor": "left", "y": 0.99,\
              "x": 0.01, title_font_family": "Times New Roman", "font": {"size": 12, "color": "black"}} --
              Customizes the legend. Refer to `Plotly's layout Documentation\
              <https://plotly.com/python/reference/layout/>`_ for valid keys
            - engine: {"kaleido", "orca"}, default="kaleido" -- Engine used for saving plots.

        Returns
        -------
        self

        Note
        -----
        **Handling Division by Zero:** NumPy automatically handles division by zero errors. Thi
        may occur if the network or the "High Amplitude" or "Low Amplitude" vectors are all zeroes.
        In such cases, NumPy assigns `NaN` to the cosine similarity for the affected network(s),
        indicating that the similarity is undefined.

        **Parcellation Approach**: If using "Custom" for ``parcel_approach`` the "regions" subkey is
        required.

        **Saving Plots**: By default, this function uses "kaleido" (a dependency of NeuroCAPs) to
        save plots. For other engines such as "orca", those packages must be installed seperately.

        **Tick Values**: if the ``tickvals`` or  ``range`` subkeys in this code are not specified in
        the ``radialaxis`` kwarg, then four values are shown - 0.25*(max value), 0.50*(max value),
        0.75*(max value), and the max value. These values are also rounded to the second decimal place.

        References
        ----------
        Zhang, R., Yan, W., Manza, P., Shokri-Kojori, E., Demiral, S. B., Schwandt, M., Vines, L.,
        Sotelo, D., Tomasi, D., Giddens, N. T., Wang, G., Diazgranados, N., Momenan, R., & Volkow,
        N. D. (2023). Disrupted brain state dynamics in opioid and alcohol use disorder: attenuation
        by nicotine use. Neuropsychopharmacology, 49(5), 876–884. https://doi.org/10.1038/s41386-023-01750-w

        Ingwersen, T., Mayer, C., Petersen, M., Frey, B. M., Fiehler, J., Hanning, U., Kühn, S.,
        Gallinat, J., Twerenbold, R., Gerloff, C., Cheng, B., Thomalla, G., & Schlemm, E. (2024).
        Functional MRI brain state occupancy in the presence of cerebral small vessel disease — A
        pre-registered replication analysis of the Hamburg City Health Study. Imaging Neuroscience,
        2, 1–17. https://doi.org/10.1162/imag_a_00122
        """
        self._check_required_attrs(["_parcel_approach", "_caps"])

        io_utils._issue_file_warning("suffix_filename", suffix_filename, output_dir)
        io_utils._makedir(output_dir)

        if not self._standardize:
            LG.warning(
                "To better aid interpretation, the matrix subjected to kmeans clustering in "
                "`self.get_caps()` should be standardized so that each ROI in the CAP cluster "
                "centroid represents activation or de-activation relative to the mean (0)."
            )

        # Create plot dictionary
        plot_dict = _check_kwargs(_PlotDefaults.caps2radar(), **kwargs)

        self._cosine_similarity = {}

        parcellation_name = list(self._parcel_approach)[0]
        # Create radar dict
        for group in self._groups:
            radar_dict = {"Regions": list(self._parcel_approach[parcellation_name]["regions"])}
            radar_dict = self._update_radar_dict(group, parcellation_name, radar_dict)
            self._cosine_similarity[group] = radar_dict

            for cap in self._caps[group]:
                fig = self._generate_radar_plot(
                    use_scatterpolar, radar_dict, cap, plot_dict, group, suffix_title
                )

                if show_figs:
                    if bool(getattr(sys, "ps1", sys.flags.interactive)):
                        fig.show()
                    else:
                        pyo.plot(fig, auto_open=True)

                if output_dir:
                    filename = io_utils._filename(
                        f"{group.replace(' ', '_')}_{cap}_radar", suffix_filename, "suffix", "png"
                    )
                    if as_html:
                        fig.write_html(os.path.join(output_dir, filename.replace(".png", ".html")))
                    elif as_json:
                        fig.write_json(os.path.join(output_dir, filename.replace(".png", ".json")))
                    else:
                        fig.write_image(
                            os.path.join(output_dir, filename),
                            scale=plot_dict["scale"],
                            engine=plot_dict["engine"],
                        )

        return self

    def _update_radar_dict(self, group: str, parcellation_name: str, radar_dict: dict) -> dict:
        """
        Updates a dictionary containing information about the cosine similarity between each region
        specified in a parcellation and the positive and negative activations in a CAP vector for
        all CAPs.
        """
        for cap in self._caps[group]:
            cap_vector = self._caps[group][cap]
            radar_dict[cap] = {"High Amplitude": [], "Low Amplitude": []}

            for region in radar_dict["Regions"]:
                region_mask = self._create_region_mask_1d(parcellation_name, region, cap_vector)

                # Get high and low amplitudes
                high_amp_vector = np.where(cap_vector > 0, cap_vector, 0)
                # Invert vector for low_amp so that cosine similarity is positive
                low_amp_vector = np.where(cap_vector < 0, -cap_vector, 0)

                # Get cosine similarity between the high amplitude and low amplitude vectors
                high_amp_cosine = self._compute_cosine_similarity(high_amp_vector, region_mask)
                low_amp_cosine = self._compute_cosine_similarity(low_amp_vector, region_mask)

                radar_dict[cap]["High Amplitude"].append(high_amp_cosine)
                radar_dict[cap]["Low Amplitude"].append(low_amp_cosine)

        return radar_dict

    def _create_region_mask_1d(
        self, parcellation_name: str, region: str, cap_vector: NDArray[np.floating]
    ) -> NDArray[np.bool_]:
        """
        Creates a 1D binary mask where 1 denotes the indices in ``cap_vector`` belonging to a
        specific network/region and "0" otherwise.
        """
        # Get the index values of nodes in each network/region
        if parcellation_name == "Custom":
            lh = list(self._parcel_approach[parcellation_name]["regions"][region]["lh"])
            rh = list(self._parcel_approach[parcellation_name]["regions"][region]["rh"])
            indxs = lh + rh
        else:
            indxs = np.array(
                [
                    value
                    for value, node in enumerate(self._parcel_approach[parcellation_name]["nodes"])
                    if region in node
                ]
            )

        # Create mask and set ROIs not in regions to zero and ROIs in regions to 1
        region_mask = np.zeros_like(cap_vector)
        region_mask[indxs] = 1

        return region_mask

    @staticmethod
    def _compute_cosine_similarity(
        amp_vector: NDArray[np.floating], region_mask: NDArray[np.bool_]
    ) -> np.floating:
        """Compute the cosine similarity between a vector and 1D binary mask."""
        dot_product = np.dot(amp_vector, region_mask)
        norm_region_mask = np.linalg.norm(region_mask)
        norm_amp_vector = np.linalg.norm(amp_vector)
        cosine_similarity = dot_product / (norm_amp_vector * norm_region_mask)

        return cosine_similarity

    @staticmethod
    def _generate_radar_plot(
        use_scatterpolar: bool,
        radar_dict: dict,
        cap: str,
        plot_dict: dict[str, Any],
        group: str,
        suffix_title: Union[str, None],
    ) -> go.Figure:
        """Generates radar plots."""
        if use_scatterpolar:
            # Create dataframe
            df = pd.DataFrame({"Regions": radar_dict["Regions"]})
            df = pd.concat([df, pd.DataFrame(radar_dict[cap])], axis=1)
            regions = df["Regions"].values

            # Initialize figure
            fig = go.Figure(layout=go.Layout(width=plot_dict["width"], height=plot_dict["height"]))

            for i in ["High Amplitude", "Low Amplitude"]:
                values = df[i].values
                fig.add_trace(
                    go.Scatterpolar(
                        r=list(values),
                        theta=regions,
                        connectgaps=plot_dict["connectgaps"],
                        name=i,
                        opacity=plot_dict["opacity"],
                        marker=dict(
                            color=plot_dict["color_discrete_map"][i], size=plot_dict["scattersize"]
                        ),
                        line=dict(
                            color=plot_dict["color_discrete_map"][i], width=plot_dict["linewidth"]
                        ),
                    )
                )
        else:
            n = len(radar_dict["Regions"])
            df = pd.DataFrame(
                {
                    "Regions": radar_dict["Regions"] * 2,
                    "Amp": radar_dict[cap]["High Amplitude"] + radar_dict[cap]["Low Amplitude"],
                }
            )
            df["Groups"] = ["High Amplitude"] * n + ["Low Amplitude"] * n

            fig = px.line_polar(
                df,
                r=df["Amp"].values,
                theta="Regions",
                line_close=plot_dict["line_close"],
                color=df["Groups"].values,
                width=plot_dict["width"],
                height=plot_dict["height"],
                category_orders={"Regions": df["Regions"]},
                color_discrete_map=plot_dict["color_discrete_map"],
            )

        if use_scatterpolar:
            fig.update_traces(fill=plot_dict["fill"], mode=plot_dict["mode"])
        else:
            fig.update_traces(
                fill=plot_dict["fill"],
                mode=plot_dict["mode"],
                marker=dict(size=plot_dict["scattersize"]),
            )

        # Set max value
        if "tickvals" not in plot_dict["radialaxis"] and "range" not in plot_dict["radialaxis"]:
            if use_scatterpolar:
                max_value = max(df[["High Amplitude", "Low Amplitude"]].max())
            else:
                max_value = df["Amp"].max()

            default_ticks = [max_value / 4, max_value / 2, 3 * max_value / 4, max_value]
            plot_dict["radialaxis"]["tickvals"] = [round(x, 2) for x in default_ticks]

        title_text = f"{group} {cap} {suffix_title}" if suffix_title else f"{group} {cap}"

        # Add additional customization
        fig.update_layout(
            title=dict(text=title_text, font=plot_dict["title_font"]),
            title_x=plot_dict["title_x"],
            title_y=plot_dict["title_y"],
            showlegend=bool(plot_dict["legend"]),
            legend=plot_dict["legend"],
            legend_title_text="Cosine Similarity",
            polar=dict(
                bgcolor=plot_dict["bgcolor"],
                radialaxis=plot_dict["radialaxis"],
                angularaxis=plot_dict["angularaxis"],
            ),
        )

        return fig
