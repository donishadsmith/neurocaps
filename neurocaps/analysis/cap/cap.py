"""Contains the CAP class for performing co-activation patterns analyses"""

import itertools
from typing import Any, Callable, Literal, Optional, Union
from typing_extensions import Self

import numpy as np
import pandas as pd

from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm

from ._internals import metrics as metrics_utils
from ._internals import cluster, correlation, matrix, radar, spatial, surface
from ._internals.getter import CAPGetter
from neurocaps.typing import ParcelConfig, ParcelApproach, SubjectTimeseries
from neurocaps.utils import _io as io_utils
from neurocaps.utils._helpers import list_to_str, resolve_kwargs
from neurocaps.utils._logging import setup_logger
from neurocaps.utils._parcellation_validation import check_parcel_approach, get_parc_name
from neurocaps.utils._plotting_utils import PlotDefaults, PlotFuncs, MatrixVisualizer

LG = setup_logger(__name__)


class CAP(CAPGetter):
    """
    Co-Activation Patterns (CAPs) Class.

    Performs k-means clustering for CAP identification, computes temporal dynamics metrics, and
    provides visualization tools for analyzing brain activation patterns.

    .. important::
       Parcellation maps are expected to be in a standard MNI space. This is due to the
       ``CAP.caps2surf`` assuming that the parcellation map is in MNI standard space when
       transforming data between volumetric and surface spaces.

    Parameters
    ----------
    parcel_approach: :obj:`ParcelConfig`, :obj:`ParcelApproach`, or :obj:`str`, default=None
        Specifies the parcellation approach to use. Options are "Schaefer", "AAL", or "Custom". Can
        be initialized with parameters, as a nested dictionary, or loaded from a serialized file
        (i.e. pickle, joblib, json). For detailed documentation on the expected structure, see the
        type definitions for ``ParcelConfig`` and ``ParcelApproach`` in the "See Also" section.

        .. versionchanged:: 0.31.0
           The default "regions" names for "AAL" has changed, which will group nodes differently.

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

        .. note::
           Standard deviations below ``np.finfo(std.dtype).eps`` are replaced with 1 for
           numerical stability.

    concatenated_timeseries: :obj:`dict[str, np.array]` or :obj:`None`
        Group-specific concatenated timeseries data. Can be deleted using
        ``del self.concatenated_timeseries``. Defined after running ``self.get_caps()``. Returns a
        reference.

        ::

            {"GroupName": np.array(shape=[(participants x TRs), ROIs])}

        .. note::
           For versions >= 0.25.0, subject IDs are sorted lexicographically prior to
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
        (See `ParcelConfig Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.ParcelConfig.html#neurocaps.typing.ParcelConfig>`_)

    :class:`neurocaps.typing.ParcelApproach`
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation
        approaches.
        (See `ParcelApproach Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.ParcelApproach.html#neurocaps.typing.ParcelApproach>`_)

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
            parcel_approach = check_parcel_approach(parcel_approach=parcel_approach, call="CAP")

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

            .. note::
               Standard deviations below ``np.finfo(std.dtype).eps`` are replaced with 1 for
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
            (See: `SubjectTimeseries Documentation
            <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.SubjectTimeseries.html#neurocaps.typing.SubjectTimeseries>`_)

        Returns
        -------
        self

        Raises
        ------
        NoElbowDetectionError
            Occurs when ``cluster_selection_method`` is set to elbow but kneed's ``KneeLocator``
            does not detect an elbow in the convex curve.

        Note
        ----
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
            raise ValueError(
                f"Options for `cluster_selection_method` are: {list_to_str(valid_methods)}."
            )

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

        self._runs = [runs] if runs and not isinstance(runs, list) else runs
        self._standardize = standardize

        # Get `subject_timeseries`, could be in memory or serialized file
        subject_timeseries = io_utils.get_obj(subject_timeseries)

        # Updates `group` if None, sorts IDs, and creates the subject table
        self._groups, self._subject_table = cluster.setup_groups(subject_timeseries, self._groups)

        # Create the temporally concatenated/stacked timeseries
        self._concatenated_timeseries = cluster.concatenate_timeseries(
            subject_timeseries,
            cluster.create_group_map(self._subject_table, self._groups),
            self._runs,
            progress_bar,
        )

        # Scale the temporally concatenated timeseries data
        self._mean_vec, self._stdev_vec = None, None
        if self._standardize:
            self._concatenated_timeseries, self._mean_vec, self._stdev_vec = cluster.scale(
                self._concatenated_timeseries
            )

        # Ensure objects are cleared
        self._optimal_n_clusters, self._kmeans, self._cluster_scores = None, None, None
        if cluster_selection_method:
            self._optimal_n_clusters, self._kmeans, self._cluster_scores = (
                cluster.select_optimal_clusters(
                    self._concatenated_timeseries,
                    cluster_selection_method,
                    self._n_clusters,
                    self._n_cores,
                    configs,
                    show_figs,
                    output_dir,
                    progress_bar,
                    as_pickle,
                    **kwargs,
                )
            )
        else:
            self._kmeans = {}
            for group_name in self._groups:
                self._kmeans[group_name] = cluster.perform_kmeans(
                    self._n_clusters,
                    configs,
                    self._concatenated_timeseries[group_name],
                    method=None,
                )

        self._variance_explained = cluster.compute_variance_explained(
            self._concatenated_timeseries, self._kmeans
        )
        self._caps = cluster.create_caps_dict(self._kmeans)

        return self

    def clear_groups(self) -> None:
        """
        Clears the ``groups`` property.

        Sets ``groups`` to None to allow ``self.get_caps`` to create a new "All Subjects" group
        with different subject IDs based on the inputted ``subject_timeseries``.

        .. versionadded:: 0.28.5
        """
        self._groups = None

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
        Calculate Participant-wise CAP Metrics.

        Calculates temporal dynamic metrics for each CAP across all participants. The metrics are
        calculated as described by Liu et al. (2018) and Yang et al. (2021) and include the
        following:

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

        .. note::
           In the supplementary material for Yang et al., the mathematical relationship
           between temporal fraction, counts, and persistence is
           ``temporal_fraction = (persistence * counts)/total_volumes``. If persistence has been
           converted into time units (seconds), then
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

            .. versionchanged:: 0.28.6
               Default changed to tuple to provide better clarity; however, the default metrics
               remains the same and is backwards compatible.

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
            (See: `SubjectTimeseries Documentation
            <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.SubjectTimeseries.html#neurocaps.typing.SubjectTimeseries>`_)

        Returns
        -------
        dict[str, pd.DataFrame] or dict[str, dict[str, pd.DataFrame]]
            Dictionary containing `pandas.DataFrame`: Only returned if ``return_df`` is True.

            - For "temporal_fraction", "counts", "persistence", and "transition_frequency": one
            dataframe is returned for each requested metric.

            - For "transition_probability": each group has a separate dataframe which is returned
            in the from of `dict[str, dict[str, pd.DataFrame]]`.


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

        io_utils.issue_file_warning("prefix_filename", prefix_filename, output_dir)
        io_utils.makedir(output_dir)

        runs = [runs] if runs and not isinstance(runs, list) else runs

        metrics = metrics_utils.filter_metrics(metrics)

        subject_timeseries = io_utils.get_obj(subject_timeseries, needs_deepcopy=False)

        # Assign each subject's frame to a CAP, adding shift is not necessary for the current
        # iteration of the code, done for conceptual reasons due to the naming for the CAPs start
        # at 1
        predicted_subject_timeseries = self.return_cap_labels(
            subject_timeseries, runs, continuous_runs, shift_labels=True
        )

        cap_names, max_cap, group_cap_counts = metrics_utils.extract_caps_info(self._caps)

        # Get combination of transitions in addition to building the base dataframe dictionary
        cap_pairs = (
            metrics_utils.create_transition_pairs(self._caps)
            if "transition_probability" in metrics
            else None
        )
        columns_names_dict = metrics_utils.create_columns_names(
            metrics, list(self._groups), cap_names, cap_pairs
        )

        # Create function map
        metric_map = {
            "temporal_fraction": metrics_utils.compute_temporal_fraction,
            "counts": metrics_utils.compute_counts,
            "persistence": metrics_utils.compute_persistence,
            "transition_frequency": metrics_utils.compute_transition_frequency,
            "transition_probability": metrics_utils.compute_transition_probability,
        }

        # Generate list for iteration
        distributed_dict = metrics_utils.create_distributed_dict(
            self._subject_table, predicted_subject_timeseries
        )

        all_metrics_dict = metrics_utils.initialize_all_metrics_dict(metrics, list(self._groups))
        for subj_id in tqdm(
            distributed_dict, desc="Computing Metrics for Subjects", disable=not progress_bar
        ):
            for group_name, curr_run in distributed_dict[subj_id]:
                for metric in metrics:
                    # Create a list of base arguments
                    args = [predicted_subject_timeseries[subj_id][curr_run]]
                    if metric in ["temporal_fraction", "counts", "persistence"]:
                        args.append(group_cap_counts[group_name])
                        args += [tr] if metric == "persistence" else []
                    elif metric == "transition_probability":
                        args += [cap_pairs[group_name]]

                    metric_dict = metric_map[metric](*args)
                    # Pad with nan values to align data from groups with a different number of CAPs
                    # Only for metrics that have different groups in a single dataframe
                    if metric in ["temporal_fraction", "counts", "persistence"]:
                        metric_dict = metrics_utils.add_nans_to_dict(
                            max_cap, group_cap_counts[group_name], metric_dict
                        )

                    subject_data = [subj_id, group_name, curr_run] + list(metric_dict.values())
                    if metric == "transition_probability":
                        all_metrics_dict[metric][group_name].append(subject_data)
                    else:
                        all_metrics_dict[metric].append(subject_data)

        df_dict = metrics_utils.convert_dict_to_df(columns_names_dict, all_metrics_dict)
        metrics_utils.save_metrics(output_dir, list(self._groups), df_dict, prefix_filename)

        if return_df:
            return df_dict

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

                .. note::
                   This scaling ensures the subject's data matches the distribution of the
                   input data used for group-specific clustering, which is needed for accurate
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
            (See: `SubjectTimeseries Documentation
            <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.SubjectTimeseries.html#neurocaps.typing.SubjectTimeseries>`_)

        Returns
        -------
            dict[str, dict[str, np.ndarray]]
                Dictionary mapping each subject to their run IDs and a 1D numpy array containing
                the predicted CAP for each frame (TR).
        """
        self._check_required_attrs(["_kmeans"])

        subject_timeseries = io_utils.get_obj(subject_timeseries, needs_deepcopy=False)

        for subj_id, group in self._subject_table.items():
            if "predicted_subject_timeseries" not in locals():
                predicted_subject_timeseries = {}

            predicted_subject_timeseries[subj_id] = {}
            subject_runs, miss_runs = cluster.get_runs(runs, list(subject_timeseries[subj_id]))

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

        Produces a 2D plot for each CAP, visualized as either a heatmap or an outer product.
        The visualization can be generated at two levels of granularity (nodes or regions).
        Separate plots are generated for each group.

        Parameters
        ----------
        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory for saving plots as png files. The directory will be created if it does not
            exist. If None, plots will not be saved.

        suffix_title: :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the filename of each saved plot if ``output_dir`` is provided.

        plot_options: {"heatmap", "outer_product"} or :obj:`list["heatmap", "outer_product"]`,\
            default="outer_product"
            Type of plots to create. Options are:

            - "heatmap": Displays the activation value (z-score if data was standardized prior to
              clustering) of each node or the average activation of each predefined region across
              all CAPs. Each column represents a different CAP, while each row represents a
              node/region.

            - "outer_product": Computed as the outer product of the CAP vector (cluster centroid)
              with itself (i.e. ``np.outer(CAP_1, CAP_1)``). This shows the pairwise interactions
              between all nodes/regions, highlighting pairs that co-activate/co-deactivate together
              (positive values) or diverge where one node/region activates while the other
              deactivates (negative values). The main diagonal represents self-interactions (the
              squared activation of each node/region), while off-diagonal elements represent
              pairwise interactions between different nodes/regions.

        visual_scope: {"regions", "nodes"} or :obj:`list["regions", "nodes"]`, default="regions"
            Determines the level of granularity of the plots. Options are:

            - "nodes": Visualizes each parcel from the brain parcellation individually.

            - "regions": Averages parcels into larger groups based on their network membership
              before plotting. These groupings must be defined in the `parcel_approach`
              configuration under the "regions" subkey.

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
            - cmap: :obj:`str`, :obj:`callable` default="coolwarm" -- Color map for the plot cells.
              Options include strings to call seaborn's pre-made palettes, ``seaborn.diverging_palette``
              function to generate custom palettes, and ``matplotlib.color.LinearSegmentedColormap``
              to generate custom palettes.
            - vmin: :obj:`float` or :obj:`None`, default=None -- The minimum value to display in colormap.
            - vmax: :obj:`float` or :obj:`None`, default=None -- The maximum value to display in colormap.
            - add_custom_node_labels: :obj:`bool`, default=False -- When ``visual_scope`` is set to
              "nodes" and a "Custom" ``parcel_approach`` is used, adds simplified node names to the
              plot's axes. Instead of labeling every individual node, the node list is "collapsed"
              by region. A single label is then placed at the beginning of the group of nodes
              corresponding to that region (e.g., "LH Visual" or "Hippocampus"), while the other
              nodes in that group are not explicitly labeled. This is done to minimize
              cluttering of the axes labels.

              .. versionadded:: 0.30.0

              .. important::
                 This feature should be used with caution. It is recommended to leave this
                 argument as ``False`` for the following conditions:

                 1. **Large Number of Nodes**: Enabling labels for a parcellation with many
                    nodes can clutter the plot axes and make them unreadable.
                 2. **Non-Consecutive Node Indices**: The labeling logic assumes that the
                    numerical indices for all nodes within a given region are defined as a
                    consecutive block (e.g., ``"RegionA": [0, 1, 2]``, ``"RegionB": [3, 4]``).
                    If the indices are non-consecutive or interleaved (e.g.,
                    ``"RegionA": [0, 2]``, ``"RegionB": [1, 3]``), the axis labels will be
                    misplaced. Note that this issue only affects the visual labeling on the plot;
                    the underlying data matrix remains correctly ordered and plotted.

        Returns
        -------
        self

        Note
        ----
        **Parcellation Approach**: the "nodes" and "regions" subkeys are required in
        ``parcel_approach`` for this method.

        **Property Persistence**: the function creates and recomputes ``self.region_means`` for
        each call if "regions" is in ``visual_scope`` and and creates and recomputes
        ``self.outer_products`` for each call if "outer_products" is in ``plot_options``. For
        ``self.outer_products``, the final values stored are associated with the last
        string in the ``visual_scope`` list.

        **Color Palettes**: Refer to `seaborn's Color Palettes
        <https://seaborn.pydata.org/tutorial/color_palettes.html>`_ for valid pre-made palettes.
        """
        self._check_required_attrs(["_parcel_approach", "_caps"])

        # Check if parcellation_approach is custom
        if "Custom" in self._parcel_approach and any(
            key not in self._parcel_approach["Custom"] for key in ["nodes", "regions"]
        ):
            check_parcel_approach(parcel_approach=self._parcel_approach, call="caps2plot")

        # Check if number of nodes in parcel_approach match length of first cap vector
        matrix.check_cap_length(self._caps, self._parcel_approach)

        io_utils.issue_file_warning("suffix_filename", suffix_filename, output_dir)
        io_utils.makedir(output_dir)

        # Check inputs for plot_options and visual_scope
        if not any(["heatmap" in plot_options, "outer_product" in plot_options]):
            raise ValueError("Valid inputs for `plot_options` are 'heatmap' and 'outer_product'.")

        if not any(["regions" in visual_scope, "nodes" in visual_scope]):
            raise ValueError("Valid inputs for `visual_scope` are 'regions' and 'nodes'.")

        # Create plot dictionary
        plot_dict = resolve_kwargs(PlotDefaults.caps2plot(), **kwargs)

        # Ensure plot_options and visual_scope are lists
        plot_options = [plot_options] if isinstance(plot_options, str) else plot_options
        visual_scope = [visual_scope] if isinstance(visual_scope, str) else visual_scope

        if "regions" in visual_scope:
            # Compute means of regions/networks for each cap
            self._region_means = matrix.compute_region_means(
                self._parcel_approach, list(self._groups), self._caps
            )
        # Initialize outer product attribute
        if "outer_product" in plot_options:
            self._outer_products = {group_name: None for group_name in self._groups}

        distributed_list = list(itertools.product(plot_options, visual_scope, self._groups))
        for plot_option, scope, group_name in distributed_list:
            # Get correct labels depending on scope
            cap_dict, labels = matrix.extract_scope_information(
                scope,
                self._parcel_approach,
                plot_dict["add_custom_node_labels"],
                self._caps,
                getattr(self, "_region_means", None),
            )

            # Generate plot for each group
            input_keys = dict(
                group_name=group_name,
                plot_dict=plot_dict,
                cap_dict=cap_dict,
                full_labels=labels,
                output_dir=output_dir,
                suffix_title=suffix_title,
                suffix_filename=suffix_filename,
                show_figs=show_figs,
                as_pickle=as_pickle,
                scope=scope,
                parcel_approach=self._parcel_approach,
            )

            # Generate plot for each group
            if plot_option == "outer_product":
                # Always ends up storing the last iterations
                self._outer_products[group_name] = matrix.compute_outer_products(
                    cap_dict[group_name]
                )
                matrix.generate_outer_product_plots(
                    **input_keys, subplots=subplots, outer_products=self._outer_products
                )
            elif plot_option == "heatmap":
                matrix.generate_heatmap_plots(**input_keys)

        return self

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
        **Color Palettes**: Refer to `seaborn's Color Palettes
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

        io_utils.issue_file_warning("suffix_filename", suffix_filename, output_dir)

        create_corr_dict = return_df or save_df

        # Create plot dictionary
        plot_dict = resolve_kwargs(PlotDefaults.caps2corr(), **kwargs)

        for group_name in self._groups:
            df = pd.DataFrame(self._caps[group_name])
            corr_df = df.corr(method=method)

            display = MatrixVisualizer.create_display(
                corr_df, plot_dict, suffix_title, group_name, "corr"
            )

            if create_corr_dict:
                if "corr_dict" not in locals():
                    corr_dict = {}

                corr_dict[group_name] = correlation.add_significance_values(
                    df, corr_df, method, plot_dict["fmt"]
                )
            else:
                corr_dict = None

            MatrixVisualizer.save_contents(
                output_dir=output_dir,
                suffix_filename=suffix_filename,
                group_name=group_name,
                curr_dict=corr_dict,
                plot_dict=plot_dict,
                save_plots=save_plots,
                save_df=save_df,
                display=display,
                as_pickle=as_pickle,
                call="corr",
            )

            PlotFuncs.show(show_figs)

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

        io_utils.makedir(output_dir)

        parc_name = get_parc_name(self._parcel_approach)
        for group_name in self._groups:
            for cap_name in tqdm(
                self._caps[group_name],
                desc=f"Generating Statistical Maps [GROUP: {group_name}]",
                disable=not progress_bar,
            ):
                stat_map = spatial.cap_to_img(
                    atlas_file=self._parcel_approach[parc_name]["maps"],
                    cap_vector=self._caps[group_name][cap_name],
                    fwhm=fwhm,
                    knn_dict=knn_dict,
                )

                filename = io_utils.filename(
                    f"{group_name.replace(' ', '_')}_{cap_name}",
                    suffix_filename,
                    "suffix",
                    "nii.gz",
                )
                surface.save_nifti_img(stat_map, output_dir, filename)

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
              Refer to ``_add_colorbars`` for ``surfplot.plotting.Plot`` in `Surfplot's Plot\
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

        knn_dict = self._validate_knn_dict(knn_dict)

        io_utils.issue_file_warning("suffix_filename", suffix_filename, output_dir)
        io_utils.makedir(output_dir)

        # Create plot dictionary
        plot_dict = resolve_kwargs(PlotDefaults.caps2surf(), **kwargs)

        group_names = (
            self._caps if hasattr(self, "_caps") and fslr_giftis_dict is None else fslr_giftis_dict
        )
        for group_name in group_names:
            cap_dict = (
                self._caps[group_name]
                if hasattr(self, "_caps") and fslr_giftis_dict is None
                else fslr_giftis_dict[group_name]
            )
            for cap_name in tqdm(
                cap_dict,
                desc=f"Generating Surface Plots [GROUP: {group_name}]",
                disable=not progress_bar,
            ):
                params = {"method": method, "fslr_density": fslr_density}
                if fslr_giftis_dict is None:
                    parc_name = get_parc_name(self._parcel_approach)
                    stat_map = spatial.cap_to_img(
                        atlas_file=self._parcel_approach[parc_name]["maps"],
                        cap_vector=self._caps[group_name][cap_name],
                        fwhm=fwhm,
                        knn_dict=knn_dict,
                    )
                    gii_lh, gii_rh = surface.convert_volume_to_surface(stat_map=stat_map, **params)
                else:
                    stat_map = None
                    gii_lh, gii_rh = surface.resample_surface(
                        fslr_giftis_dict=fslr_giftis_dict[group_name],
                        group_name=group_name,
                        cap_name=cap_name,
                        **params,
                    )

                fig = surface.generate_surface_plot(
                    gii_lh, gii_rh, group_name, cap_name, suffix_title, plot_dict
                )

                surface.save_surface_plot(
                    output_dir,
                    stat_map,
                    fig,
                    group_name,
                    cap_name,
                    suffix_filename,
                    save_stat_maps,
                    as_pickle,
                    plot_dict,
                )

                surface.show_surface_plot(fig, show_figs)

        return self

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
        "Low Amplitude" (negative/below the mean) activations of the CAP and each a-priori region
        or network in the parcellation defined by ``parcel_approach`` (e.g. DMN, Vis, etc). One
        radar plot is generated per CAP and separate images are generated for each group.

        .. important::

          - This function assumes the mean for each ROI is 0 due to standardization.
          - The absolute values of the negative activations are computed for visualization purposes.

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
            modified, instead of png images. Note that option is ignored if ``as_html`` is also
            True.

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
        ----
        **Interpretation**: Each radar plot provides a high-level representation of CAP by computing
        and visualizing how closely each CAP's positive ("High Amplitude") and negative
        ("Low Amplitude") patterns aligns large-scale, canonical networks or regions.

        For each radar plot, two traces are shown:
            - "High Amplitude": Spatial alignment with the positively activating nodes in a CAP.
              Traces indicate a given region is active during that CAP.
            - "Low Amplitude": Spatial alignment with deactivating nodes in a CAP. Traces indicate
              that a given region is suppressed during that CAP.

        This provides a quantitative method to label CAPs based on the dominant active and
        suppressed networks or regions. For instance, if the dorsal attention network (DAN) has
        the highest (largest trace) "High Amplitude" cosine similarity value and the ventral
        attention network (VAN) has a highest "Low Amplitude" cosine similarity value, then that CAP
        can be described as (DAN +/VAN -). Or if the highest "High Amplitude" cosine similarity
        value is assoiciated with the default mode network (DMN), while the highest "Low Amplitude"
        cosine similarity values are associates with the DAN and the frontoparietal network (FPN),
        then the CAP can represent a classic task-negative pattern.

        **Methodology**: The process involves the following steps for computing the "High Amplitude"
        and "Low Amplitude" values for each CAP and network/region combination

        1. Extract Cluster Centroids:

            - Each CAP is represented by a cluster centroid, which is a 1 x ROI
            (Region of Interest) vector.

        2. Generate Binary Vectors:

            - For each network/region create a binary vector (1 x ROI) where "1" indicates that
            the ROI is part of the specific region and "0" otherwise.
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

                # Assign values greater than 0 as 0 to isolate the low amplitude activations
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

                # Calculate cosine similarity; Produces the alignment of the region/network
                # with the active and suppressed patterns of the CAP
                high_cos = high_dot / (high_norm * bin_norm)
                low_cos = low_dot / (low_norm * bin_norm)

        5. Generate Radar Plots of Each CAPs:

            - Each radar plot visualizes the cosine similarity for both "High Amplitude"
            (positive) and "Low Amplitude" (negative) activations of the CAP.

        **Handling Division by Zero:** NumPy automatically handles division by zero errors. This
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

        io_utils.issue_file_warning("suffix_filename", suffix_filename, output_dir)
        io_utils.makedir(output_dir)

        if not self._standardize:
            LG.warning(
                "To better aid interpretation, the matrix subjected to kmeans clustering in "
                "`self.get_caps()` should be standardized so that each ROI in the CAP cluster "
                "centroid represents activation or de-activation relative to the mean (0)."
            )

        # Create plot dictionary
        plot_dict = resolve_kwargs(PlotDefaults.caps2radar(), **kwargs)

        self._cosine_similarity = {}

        # Create radar dict
        for group_name in self._groups:
            regions = list(self._parcel_approach[get_parc_name(self._parcel_approach)]["regions"])
            radar_dict = {"Regions": regions}
            radar_dict = radar.update_radar_dict(
                self._caps[group_name], radar_dict, self._parcel_approach
            )
            self._cosine_similarity[group_name] = radar_dict

            for cap_name in self._caps[group_name]:
                fig = radar.generate_radar_plot(
                    use_scatterpolar, radar_dict, cap_name, group_name, suffix_title, plot_dict
                )

                radar.show_radar_plot(fig, show_figs)

                radar.save_radar_plot(
                    fig,
                    output_dir,
                    group_name,
                    cap_name,
                    suffix_filename,
                    as_html,
                    as_json,
                    plot_dict["scale"],
                    plot_dict["engine"],
                )

        return self
