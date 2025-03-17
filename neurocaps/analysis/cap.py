import collections, copy, itertools, os, re, sys, tempfile
from typing import Callable, Literal, Optional, Union

# Conditional import based on major and minor version of Python
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import nibabel as nib, numpy as np, matplotlib.pyplot as plt, pandas as pd, seaborn, surfplot
import plotly.express as px, plotly.graph_objects as go, plotly.offline as pyo
from kneed import KneeLocator
from joblib import Parallel, delayed
from nilearn.plotting.cm import _cmap_d
from neuromaps.transforms import mni152_to_fslr, fslr_to_fslr
from neuromaps.datasets import fetch_fslr
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from ..typing import ParcelConfig, ParcelApproach, SubjectTimeseries
from .._utils import (
    _CAPGetter,
    _PlotDefaults,
    _cap2statmap,
    _check_kwargs,
    _create_display,
    _convert_pickle_to_dict,
    _check_parcel_approach,
    _handle_aal,
    _logger,
    _run_kmeans,
    _save_contents,
)

LG = _logger(__name__)


class CAP(_CAPGetter):
    """
    Co-Activation Patterns (CAPs) Class.

    Performs k-means clustering for CAP identification, computes various temporal dynamics metrics
    (including counts, temporal fraction, persistence, transition frequency, and transition probability),
    provides multiple visualizations (such as heatmaps, outer products, correlation matrices, and cosine
    similarity radar plots that shows the network correspondence to both positive and negative CAP activations),
    and enables conversion of CAPs to NIfTI statistical maps.

    Parameters
    ----------
    parcel_approach: :obj:`ParcelConfig`, :obj:`ParcelApproach`, or :obj:`str`, default=None
        Specifies the parcellation approach for segmenting NifTI images into distinct regions-of-interests (ROIs).
        Supported parcellation aproaches includes "Schaefer", "AAL", and "Custom" (user-defined).

        The ``parcel_approach`` requires a nested dictionary with:

          - First Level Key: The parcellation name (e.g., "Schaefer", "AAL", "Custom")
          - Second Level Keys (subkeys):

              - "maps": Mapping of regions to coordinates.
              - "nodes": Node definitions for each region.
              - "regions": List of anatomical region names (see "Custom Parcellations" in Notes section for detailed\
              structure requirements).

        There are *3* valid input options for this parameter:

          - Use the initialization parameters from ``TimeseriesExtractor`` (e.g., ``n_rois``, ``yeo_networks``,\
          ``resolution_mm`` for "Schaefer", or ``version`` for "AAL") to generate the above structure.
          - Provide a nested dictionary with the required structure (first level key of the parcellation name with\
          subkeys - "maps", "nodes", and "regions").
          - Load configuration from a pickle file containing the valid nested dictionary.

        *Note*: This parameter is required for several visualization functions in this class but can be set later
        using the ``parcel_approach`` property.

    groups: :obj:`dict[str, list[str]]` or :obj:`None`, default=None
        A mapping of group names to unique subject IDs. If specified, then separate analyses are performed on groups
        (i.e., group-specific k-means models, group-specific visualization, etc). If None, then the analyses will be
        performed on all subjects and the default group name is "All Subjects". The structure should be as follows:

        ::

            {
                "GroupName1": ["1", "2", "3"],
                "GroupName2": ["4", "5", "6"],
            }


    Properties
    ----------
    parcel_approach: :obj:`ParcelApproach`
        A dictionary containing information about the parcellation. Can also be used as a setter, which accepts a
        dictionary or a dictionary saved as a pickle file. The structure is as follows:

        ::

            # Structure of Schaefer
            {
                "Schaefer":
                {
                    "maps": "path/to/parcellation.nii.gz",
                    "nodes": ["LH_Vis1", "LH_SomSot1", "RH_Vis1", "RH_SomSot1"],
                    "regions": ["Vis", "SomSot"]
                }
            }

            # Structure of AAL
            {
                "AAL":
                {
                    "maps": "path/to/parcellation.nii.gz",
                    "nodes": ["Precentral_L", "Precentral_R", "Frontal_Sup_L", "Frontal_Sup_R"],
                    "regions": ["Precentral", "Frontal"]
                }
            }

        Refer to the example for "Custom" in the Note section below for the expected structure.

    groups: :obj:`dict[str, list[str]]` or :obj:`None`:
        A mapping of groups names to unique subject IDs.

    subject_table: :obj:`dict[str, str]` or :obj:`None`
        A dictionary generated when ``self.get_caps()`` is used. Operates as a lookup table that pairs each subject ID
        with the associated group. Also can be used as a setter. The structure is as follows.

        ::

            {
                "Subject-ID": "GroupName",
                "Subject-ID": "GroupName",
            }

    n_clusters: :obj:`int`, :obj:`list[int]`, or :obj:`None`
        An integer or list of integers (if ``cluster_selection_method`` is not None) that was used for
        ``sklearn.cluster.KMeans`` in ``self.get_caps()``.

    cluster_selection_method: :obj:`str` or :obj:`None`:
        The cluster selection method used in ``self.get_caps()`` to identify the optimal number of clusters.

    n_cores: :obj:`int` or :obj:`None`
        Number of cores specified in ``self.get_caps()`` to use for multiprocessing with Joblib.

    runs: :obj:`int`, :obj:`list[int | str]`, or :obj:`None`
        The run IDs specified in ``self.get_caps()``.

    standardize: :obj:`bool` or :obj:`None`
        A boolean denoting whether the features of the concatenated timeseries data are standardized if standardization
        was requested in ``self.get_caps()``.

    means: :obj:`dict[str, np.array]` or :obj:`None`
        A dictionary mapping groups to their associated NumPy array containing the means of each feature (ROI) if
        ``standardize`` is True in ``self.get_caps()``. The structure is as follows:

        ::

            {
                "GroupName": np.array([...]), # Shape: 1 x ROIs
            }

    stdev: :obj:`dict[str, np.array]` or :obj:`None`
        A dictionary mapping groups to their associated NumPy array containing the sample standard deviation of each
        feature (ROI) if ``standardize`` is True in ``self.get_caps()``. The structure is as follows:

        ::

            {
                "GroupName": np.array([...]), # Shape: 1 x ROIs
            }

    concatenated_timeseries: :obj:`dict[str, np.array]` or :obj:`None`
        A dictionary mapping each group to their associated concatenated NumPy array [(participants x TRs) x ROIs] when
        ``self.get_caps()`` is used. Note, if there are memory issues, ``del self.concatenated_timeseries``
        can be used to delete property and have it only return None. The structure is as follows:

        ::

            {
                "GroupName": np.array([...]), # Shape: (participants x TRs) x ROIs
            }

    kmeans: :obj:`dict[str, sklearn.cluster.KMeans]` or :obj:`None`
        A dictionary mapping group-specific ``sklearn.cluster.KMeans`` model to each group when ``self.get_caps()`` is
        used. If ``cluster_selection_method`` is used, the model stored in this property is the optimal k-means model.
        The structure is as follows:

        ::

            {
                "GroupName": sklearn.cluster.KMeans,
            }

    caps: :obj:`dict[str, dict[str, np.array]]` or :obj:`None`
        A dictionary mapping the cluster centroids, extracted from the k-means model, to each group after
        ``self.get_caps()`` is used. The structure is as follows:

        ::

            {
                "GroupName": {
                    "CAP-1": np.array([...]), # Shape: 1 x ROIs
                    "CAP-2": np.array([...]), # Shape: 1 x ROIs
                }

            }

    cluster_scores: :obj:`dict[str, str | dict[str, float]]` or :obj:`None`
        A dictionary mapping groups to the assessed cluster sizes and corresponding score of a specific
        ``cluster_selection_method`` available in ``self.get_caps()``. The structure is as follows:

        ::

            {
                "Cluster_Selection_Method": str,  # e.g., "elbow", "davies_bouldin", "silhouette", or "variance_ratio"
                "Scores": {
                    "GroupName": {
                        2: float,  # Score for 2 clusters
                        3: float,  # Score for 3 clusters
                        4: float,  # Score for 4 clusters
                    },
                }
            }

    optimal_n_clusters: :obj:`dict[str, int]` or :obj:`None`
        A dictionary mapping groups to their optimal cluster sizes if ``cluster_selection_method`` is not None in
        ``self.get_caps()``. The structure is as follows:

        ::

            {
                "GroupName": int,
            }

    variance_explained: :obj:`dict[str, float]` or :obj:`None`
        A dictionary mapping groups to a float representing the total variance explained by their respective model
        stored in ``self.kmeans``. The structure is as follows:

        ::

            {
                "GroupName": float,
            }

    region_means: :obj:`dict[str, dict[str, list[str] | np.array]]` or :obj:`None`
        A dictionary mapping group to their CAPs and corresponding NumPy array (1 x regions) containing the averaged
        value of each region or network if ``visual_scope`` set to "regions" in ``self.caps2plot()``.
        The position of elements corresponds to the region in the “Regions” list (mean value at the 0th index
        corresponds to the region at the 0th index). The structure is as follows:

        ::

            {
                "GroupName": {
                    "Regions": [...],
                    "CAP-1": np.array([...]), # Shape: 1 x regions
                    "CAP-2": np.array([...]), # Shape: 1 x regions
                }

            }

        .. versionchanged:: 0.23.4 Replaces ``self.region_caps`` and adds the "Regions" key for each group.\
        For backwards compatibility, ``self.region_caps``, which doesn't include the is the "Regions" key is\
        still available.

    outer_products: :obj:`dict[str, dict[str, np.array]]` or :obj:`None`
        A dictionary mapping group to their CAPs and corresponding NumPy array (ROIs x ROIs) containing the outer
        product if ``plot_options`` set to "outer_product" ``self.caps2plot()``. The structure is as follows:

        ::

            {
                "GroupName": {
                    "CAP-1": np.array([...]), # Shape: ROIs x ROIs
                    "CAP-2": np.array([...]), # Shape: ROIs x ROIs
                }

            }

    cosine_similarity: :obj:`dict[str, dict[str, list[str] | np.array]]` or :obj:`None`
        A dictionary mapping each group to their CAPs and their associated "High Amplitude" and "Low Amplitude"
        list containing the cosine similarities. Each group contains a "Regions" key, consisting of a list of
        regions or networks. The position of the cosine similarities in the "High Amplitude" and "Low Amplitude"
        lists, corresponds to the region in the "Regions" list (cosine similarity value in 0th index belongs to
        region in 0th index). The structure is as follows:

        ::

            {
                "GroupName": {
                    "Regions": [...],
                    "CAP-1": {
                        "High Amplitude": [...], # Shape: 1 x Regions
                        "Low Amplitude": [...]   # Shape: 1 x Regions
                    }
                    "CAP-2": {
                        "High Amplitude": [...], # Shape: 1 x Regions
                        "Low Amplitude": [...]   # Shape: 1 x Regions
                    }
                }
            }

    See Also
    --------
    :class:`neurocaps.typing.ParcelConfig`
        Type definition representing the configuration options and structure for the Schaefer and AAL parcellations.
    :class:`neurocaps.typing.ParcelApproach`
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation approaches.

    Note
    ----
    **Default Group Name**: If no groups were specified, **the default group name will always be "All Subjects"** to
    denote that the data of all subjects in the analysis were used to derive the CAPs. This is done to retain the
    same nesting structure for each property regardless if groups are specified.

    **Custom Parcellations**: If using a "Custom" parcellation approach, ensure that the parcellation is
    lateralized (where each region/network has nodes in the left and right hemisphere). This is due to certain
    visualization functions assuming that each region consists of left and right hemisphere nodes. Additionally,
    certain visualization functions in this class also assume that the background label is 0. Therefore, do not add a
    background label in the "nodes" or "regions" keys.

    The recognized subkeys for the "Custom" parcellation approach includes:

    - "maps": Directory path containing the parcellation in a supported format (e.g., .nii or .nii.gz for NifTI).
    - "nodes": A list or numpy array of all node labels arranged in ascending order based on their numerical IDs from
      the parcellation. The 0th index should contain the label corresponding to the lowest, non-background numerical ID.
    - "regions": A dictionary defining major brain regions or networks, with each region containing "lh"
      (left hemisphere) and "rh" (right hemisphere) subkeys listing node indices.

    Refer to the `NeuroCAPs' Parcellation Documentation <https://neurocaps.readthedocs.io/en/stable/parcellations.html>`_
    for more detailed explanations and example structures for the "nodes" and "regions" subkeys.

    **Note**: Different subkeys are required depending on the function used. Refer to the Note section under each
    function for information regarding the subkeys required for that specific function.
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
                    "`groups` must be a dictionary where the keys are the group names and the items "
                    "correspond to subject ids in the groups."
                )

            for group_name in groups:
                assert groups[group_name], f"{group_name} has zero subject ids."

            # Convert ids to strings
            for group in set(groups):
                groups[group] = [str(subj_id) if not isinstance(subj_id, str) else subj_id for subj_id in groups[group]]

        self._groups = groups

    def get_caps(
        self,
        subject_timeseries: Union[SubjectTimeseries, str],
        runs: Optional[Union[int, str, list[int], list[str]]] = None,
        n_clusters: Union[int, list[int], range] = 5,
        cluster_selection_method: Optional[Literal["elbow", "davies_bouldin", "silhouette", "variance_ratio"]] = None,
        random_state: Optional[int] = None,
        init: Union[Literal["k-means++", "random"], Callable, np.typing.ArrayLike] = "k-means++",
        n_init: Union[Literal["auto"], int] = "auto",
        max_iter: int = 300,
        tol: float = 0.0001,
        algorithm: Literal["lloyd", "elkan"] = "lloyd",
        standardize: bool = True,
        n_cores: Optional[int] = None,
        show_figs: bool = False,
        output_dir: Optional[str] = None,
        progress_bar: bool = False,
        **kwargs,
    ) -> Self:
        """
        Perform K-Means Clustering to Identify CAPs.

        Concatenates the timeseries of each subject into a single NumPy array with dimensions
        (participants x TRs) x ROI and uses ``sklearn.cluster.KMeans`` on the concatenated data. Note,
        ``KMeans`` uses Euclidean distance. Additionally, the Elbow method is determined using ``KneeLocator`` from
        the kneed package and the Davies Bouldin, Silhouette, and Variance Ratio methods are calculated using
        scikit-learn's ``davies_bouldin_score``, ``silhouette_score``, and ``calinski_harabasz_score`` functions,
        respectively. Note, if groups were given when the ``CAP`` class was initialized, separate ``KMeans`` models
        and plots will be generated for all groups.

        Parameters
        ----------
        subject_timeseries: :obj:`SubjectTimeseries` or :obj:`str`
            A dictionary mapping subject IDs to their run IDs and their associated timeseries (TRs x ROIs) as a NumPy
            array. Can also be a path to a pickle file containing this same structure. The expected structure of is as
            follows:

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

        runs: :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            The run numbers to perform the CAPs analysis with (e.g. ``runs=[0, 1]`` or ``runs=["01", "02"]``). If None,
            all runs in the subject timeseries will be concatenated into a single dataframe and subjected to k-means
            clustering.

        n_clusters: :obj:`int` or :obj:`list[int]`, default=5
            The number of clusters to use for ``sklearn.cluster.KMeans``. Can be a single integer or a list of
            integers (if ``cluster_selection_method`` is not None).

        cluster_selection_method: {"elbow", "davies_bouldin", "silhouette", "variance_ratio"} or :obj:`None`, default=None
            Method to find the optimal number of clusters. Options are "elbow", "davies_bouldin", "silhouette", and
            "variance_ratio".

        random_state: :obj:`int` or :obj:`None`, default=None
            The random state to use for ``sklearn.cluster.KMeans``. Ensures reproducible results.

        init: {"k-means++", "random"}, :obj:`Callable`, or `ArrayLike`, default="k-means++"
            Method for choosing initial cluster centroid for ``sklearn.cluster.KMeans``. Options are "k-means++",
            "random", or callable or array-like of shape (n_clusters, n_features).

        n_init: {"auto"} or :obj:`int`, default="auto"
            Number of times ``sklearn.cluster.KMeans`` is ran with different initial clusters.
            The model with lowest inertia from these runs will be selected.

        max_iter: :obj:`int`, default=300
            Maximum number of iterations for a single run of ``sklearn.cluster.KMeans``.

        tol: :obj:`float`, default=1e-4
            Stopping criterion for ``sklearn.cluster.KMeans`` if the change in inertia is below this value, assuming
            ``max_iter`` has not been reached.

        algorithm: {"lloyd", "elkan"}, default="lloyd"
            The type of algorithm to use for ``sklearn.cluster.KMeans``. Options are "lloyd" and "elkan".

        standardize: :obj:`bool`, default=True
            Standardizes the columns (ROIs) of the concatenated timeseries data. Uses sample standard deviation with
            Bessel's correction (`n-1` in denominator).

        n_cores: :obj:`int` or :obj:`None`, default=None
            The number of cores to use for multiprocessing, with Joblib, to run multiple ``sklearn.cluster.KMeans``
            models if ``cluster_selection_method`` is not None. The "loky" backend is used.

        show_figs: :obj:`bool`, default=False
            Displays the plots for the specified ``cluster_selection_method`` for all groups
            if ``cluster_selection_method`` is not None.

        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save plots as png files if ``cluster_selection_method`` is not None. The directory will be
            created if it does not exist. If None, plots will not be saved.

        progress_bar: :obj:`bool`, default=False
            If True and ``cluster_selection_method`` is not None, displays a progress bar.

            .. versionadded:: 0.21.5

        **kwargs:
            Dictionary to adjust certain parameters when ``cluster_selection_method`` is not None. Additional\
            parameters include:

            - S: :obj:`int`, default=1 -- Adjusts the sensitivity of finding the elbow. Larger values are more\
                conservative and less sensitive to small fluctuations. Passed to ``KneeLocator`` from the kneed package.
            - dpi: :obj:`int`, default=300 -- Dots per inch for the figure.
            - figsize: :obj:`tuple`, default=(8, 6) -- Adjusts the size of the plots.
            - bbox_inches: :obj:`str` or :obj:`None`, default="tight" -- Alters size of the whitespace in the saved image.
            - step: :obj:`int`, default=None -- An integer value that controls the progression of the x-axis in plots\
                for the specified ``cluster_selection_method``. When set, only integer values will be displayed on the x-axis.

        See Also
        --------
        :data:`neurocaps.typing.SubjectTimeseries`
            The type definition for the subject timeseries dictionary structure.

        Returns
        -------
        self

        Note
        ----
        **KMeans Algorithm:** Refer to `scikit-learn's Documentation
        <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_ for additional information
        about the ``KMeans`` algorithm used in this method.
        """
        self._n_cores = n_cores
        # Ensure all unique values if n_clusters is a list
        self._n_clusters = n_clusters if isinstance(n_clusters, int) else sorted(list(set(n_clusters)))
        self._cluster_selection_method = cluster_selection_method
        configs = {
            "random_state": random_state,
            "init": init,
            "n_init": n_init,
            "max_iter": max_iter,
            "tol": tol,
            "algorithm": algorithm,
        }

        if isinstance(n_clusters, list):
            self._n_clusters = (
                self._n_clusters[0]
                if all([isinstance(self._n_clusters, list), len(self._n_clusters) == 1])
                else self._n_clusters
            )
            # Raise error if n_clusters is a list and no cluster selection method is specified
            if all([len(n_clusters) > 1, self._cluster_selection_method is None]):
                raise ValueError("`cluster_selection_method` cannot be None since `n_clusters` is a list.")

        # Raise error if silhouette_method is requested when n_clusters is an integer
        if all([self._cluster_selection_method is not None, isinstance(self._n_clusters, int)]):
            raise ValueError("`cluster_selection_method` only valid if `n_clusters` is a range of unique integers.")

        if self._n_cores and self._cluster_selection_method is None:
            raise ValueError("Parallel processing will not run since `cluster_selection_method` is None.")

        if runs and not isinstance(runs, list):
            runs = [runs]

        self._runs = runs
        self._standardize = standardize

        subject_timeseries = self._process_subject_timeseries(subject_timeseries)
        self._concatenated_timeseries = self._concatenate_timeseries(subject_timeseries, runs)

        if self._cluster_selection_method is not None:
            valid_methods = ["elbow", "davies_bouldin", "silhouette", "variance_ratio"]
            if self._cluster_selection_method not in valid_methods:
                formatted_string = ", ".join(["'{a}'".format(a=x) for x in valid_methods])
                raise ValueError(f"Options for `cluster_selection_method` are - {formatted_string}.")
            else:
                self._select_optimal_clusters(configs, show_figs, output_dir, progress_bar, **kwargs)
        else:
            self._kmeans = {}
            for group in self._groups:
                self._kmeans[group] = {}
                self._kmeans[group] = _run_kmeans(
                    self._n_clusters, configs, self._concatenated_timeseries[group], method=None
                )

        # Create variance explained dict
        self._var_explained()

        # Create states dict
        self._create_caps_dict()

        return self

    @staticmethod
    def _process_subject_timeseries(subject_timeseries):
        if isinstance(subject_timeseries, str) and subject_timeseries.endswith(".pkl"):
            subject_timeseries = _convert_pickle_to_dict(subject_timeseries)
        elif isinstance(subject_timeseries, dict) and len(list(subject_timeseries)) == 1:
            # Potential mutability issue if only a single subject in dictionary potentially due to the variable
            # for the concatenated data being set to the subject timeseries if concatenated data is empty.
            subject_timeseries = copy.deepcopy(subject_timeseries)

        return subject_timeseries

    def _generate_lookup_table(self):
        self._subject_table = {}

        for group in self._groups:
            for subj_id in self._groups[group]:
                if subj_id in self._subject_table:
                    LG.warning(
                        f"[SUBJECT: {subj_id}] Appears more than once. Only the first instance of this "
                        "subject will be included in the analysis."
                    )
                else:
                    self._subject_table.update({subj_id: group})

    def _concatenate_timeseries(self, subject_timeseries, runs):
        # Create dictionary for "All Subjects" if no groups are specified to reuse the same loop instead of having to
        # create logic for grouped and non-grouped version of the same code
        if not self._groups:
            self._groups = {"All Subjects": [subject for subject in subject_timeseries]}

        concatenated_timeseries = {group: None for group in self._groups}

        self._generate_lookup_table()

        self._mean_vec = {group: None for group in self._groups}
        self._stdev_vec = {group: None for group in self._groups}

        for subj_id, group in self._subject_table.items():
            subject_runs, miss_runs = self._get_runs(runs, list(subject_timeseries[subj_id]))

            if miss_runs:
                LG.warning(f"[SUBJECT: {subj_id}] Does not have the requested runs: {', '.join(miss_runs)}.")

            if not subject_runs:
                LG.warning(f"[SUBJECT: {subj_id}] Excluded from the concatenated timeseries due to having no runs.")
                continue

            for curr_run in subject_runs:
                if concatenated_timeseries[group] is None:
                    concatenated_timeseries[group] = subject_timeseries[subj_id][curr_run]
                else:
                    concatenated_timeseries[group] = np.vstack(
                        [concatenated_timeseries[group], subject_timeseries[subj_id][curr_run]]
                    )

        if self._standardize:
            concatenated_timeseries = self._scale(concatenated_timeseries)

        return concatenated_timeseries

    def _scale(self, concatenated_timeseries):
        for group in self._groups:
            self._mean_vec[group] = np.mean(concatenated_timeseries[group], axis=0)
            self._stdev_vec[group] = np.std(concatenated_timeseries[group], ddof=1, axis=0)
            eps = np.finfo(self._stdev_vec[group].dtype).eps
            # Taken from nilearn pipeline, used for numerical stability purposes to avoid numpy division error
            self._stdev_vec[group][self._stdev_vec[group] < eps] = 1.0
            diff = concatenated_timeseries[group] - self._mean_vec[group]
            concatenated_timeseries[group] = diff / self._stdev_vec[group]

        return concatenated_timeseries

    @staticmethod
    def _get_runs(requested_runs, curr_runs):
        if requested_runs:
            requested_runs = [f"run-{run}" for run in requested_runs]

        runs = [run for run in requested_runs if run in curr_runs] if requested_runs else curr_runs
        miss_runs = list(set(requested_runs) - set(runs)) if requested_runs else None

        return runs, miss_runs

    def _select_optimal_clusters(self, configs, show_figs, output_dir, progress_bar, **kwargs):
        # Initialize attributes
        self._cluster_scores = {}
        self._optimal_n_clusters = {}
        self._kmeans = {}
        self._cluster_metric = {}
        performance_dict = {}

        method = self._cluster_selection_method

        for group in self._groups:
            performance_dict[group] = {}
            # Don't want to store model dict for all groups; re-initialize for each group
            model_dict = {}

            if self._n_cores is None:
                for n_cluster in tqdm(self._n_clusters, desc=f"Clustering [GROUP: {group}]", disable=not progress_bar):
                    output_score, model = _run_kmeans(n_cluster, configs, self._concatenated_timeseries[group], method)
                    performance_dict[group].update(output_score)
                    model_dict.update(model)
            else:
                parallel = Parallel(return_as="generator", n_jobs=self._n_cores, backend="loky")
                output = tqdm(
                    parallel(
                        delayed(_run_kmeans)(n_cluster, configs, self._concatenated_timeseries[group], method)
                        for n_cluster in self._n_clusters
                    ),
                    desc=f"Clustering [GROUP: {group}]",
                    total=len(self._n_clusters),
                    disable=not progress_bar,
                )

                output_scores, models = zip(*output)
                for output in output_scores:
                    performance_dict[group].update(output)
                for model in models:
                    model_dict.update(model)

            # Select optimal clusters
            if method == "elbow":
                knee_dict = {"S": kwargs["S"] if "S" in kwargs else 1}
                kneedle = KneeLocator(
                    x=list(performance_dict[group]),
                    y=list(performance_dict[group].values()),
                    curve="convex",
                    direction="decreasing",
                    S=knee_dict["S"],
                )

                self._optimal_n_clusters[group] = kneedle.elbow

                if self._optimal_n_clusters[group] is None:
                    raise ValueError(
                        f"[GROUP: {group}] - No elbow detected. Try adjusting the sensitivity parameter, "
                        "`S`, to increase or decrease sensitivity (higher values are less sensitive), "
                        "expanding the list of `n_clusters` to test, or using another "
                        "`cluster_selection_method`."
                    )
            elif method == "davies_bouldin":
                # Get minimum for davies bouldin
                self._optimal_n_clusters[group] = min(performance_dict[group], key=performance_dict[group].get)
            else:
                # Get max for silhouette and variance ratio
                self._optimal_n_clusters[group] = max(performance_dict[group], key=performance_dict[group].get)

            # Get the optimal kmeans model
            self._kmeans[group] = model_dict[self._optimal_n_clusters[group]]

            LG.info(f"[GROUP: {group} | METHOD: {method}] Optimal cluster size is {self._optimal_n_clusters[group]}.")

            # Plot
            if show_figs or output_dir is not None:
                self._plot_method(method, performance_dict, group, show_figs, output_dir, **kwargs)

        self._cluster_scores = {"Cluster_Selection_Method": method}
        self._cluster_scores.update({"Scores": performance_dict})

    def _plot_method(self, method, performance_dict, group, show_figs, output_dir, **kwargs):
        y_titles = {
            "elbow": "Inertia",
            "davies_bouldin": "Davies Bouldin Score",
            "silhouette": "Silhouette Score",
            "variance_ratio": "Variance Ratio Score",
        }

        # Create plot dictionary
        plot_dict = _check_kwargs(_PlotDefaults.get_caps(), **kwargs)

        # Create visualization for method
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
            plt.vlines(self._optimal_n_clusters[group], plt.ylim()[0], plt.ylim()[1], linestyles="--", label="elbow")

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            save_name = f"{group.replace(' ', '_')}_{self._cluster_selection_method}.png"
            plt.savefig(os.path.join(output_dir, save_name), dpi=plot_dict["dpi"], bbox_inches=plot_dict["bbox_inches"])

        plt.show() if show_figs else plt.close()

    def _var_explained(self):
        self._variance_explained = {}

        for group in self._groups:
            mean_vec = np.mean(self._concatenated_timeseries[group], axis=0)
            total_var = np.sum((self._concatenated_timeseries[group] - mean_vec) ** 2)
            explained_var = 1 - (self._kmeans[group].inertia_ / total_var)
            self._variance_explained[group] = explained_var

    def _create_caps_dict(self):
        # Initialize dictionary
        self._caps = {}

        for group in self._groups:
            self._caps[group] = {}
            cluster_centroids = zip(
                [num for num in range(1, len(self._kmeans[group].cluster_centers_) + 1)],
                self._kmeans[group].cluster_centers_,
            )
            self._caps[group].update(
                {f"CAP-{state_number}": state_vector for state_number, state_vector in cluster_centroids}
            )

    @staticmethod
    def _raise_error(attr_name):
        if attr_name == "_caps":
            raise AttributeError("Cannot plot caps since `self.caps` is None. Run `self.get_caps()` first.")
        elif attr_name == "_parcel_approach":
            raise AttributeError(
                "`self.parcel_approach` is None. Add `parcel_approach` using "
                "`self.parcel_approach=parcel_approach` to use this method."
            )
        else:
            raise AttributeError("Cannot calculate metrics since `self.kmeans` is None. Run `self.get_caps()` first.")

    def calculate_metrics(
        self,
        subject_timeseries: Union[SubjectTimeseries, str],
        tr: Optional[float] = None,
        runs: Optional[Union[int, str, list[int], list[str]]] = None,
        continuous_runs: bool = False,
        metrics: Union[
            Literal["temporal_fraction", "persistence", "counts", "transition_frequency", "transition_probability"],
            list[
                Literal["temporal_fraction", "persistence", "counts", "transition_frequency", "transition_probability"]
            ],
        ] = ["temporal_fraction", "persistence", "counts", "transition_frequency"],
        return_df: bool = True,
        output_dir: Optional[str] = None,
        prefix_filename: Optional[str] = None,
        progress_bar: bool = False,
    ) -> Union[dict[str, pd.DataFrame], None]:
        """
        Compute Participant-wise CAP Metrics.

        Uses the k-means model (or group-specific k-means models if ``groups`` specified during initialization of the
        ``CAP`` class) to assign each subject's TRs to a CAP. Also, creates a single ``pandas.DataFrame`` per
        CAP metric for all participants (with the exception of "transition_probability" which creates a single
        dataframe per group). As described by Liu et al., 2018 and Yang et al., 2021. The metrics include:

         - ``"temporal_fraction"``: The proportion of total volumes spent in a single CAP over all volumes in a run.

           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
                target = 1
                temporal_fraction = 4 / 6

         - ``"persistence"``: The average time spent in a single CAP before transitioning to another CAP
           (average consecutive/uninterrupted time).

           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
                target = 1

                # Sequences for 1 are [1] and [1, 1, 1]; There are 2 contiguous sequences
                persistence = (1 + 3) / 2

                # Turns average frames into average time = 4
                tr = 2
                if tr: persistence = ((1 + 3) / 2) * 2

         - ``"counts"``: The total number of initiations of a specific CAP across an entire run. An initiation is
           defined as the first occurrence of a CAP. If the same CAP is maintained in contiguous segment
           (indicating stability), it is still counted as a single initiation.

           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
                target = 1

                # Initiations of CAP-1 occur at indices 0 and 2
                counts = 2

         - ``"transition_frequency"``: The total number of transitions to different CAPs across the entire run.

           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]

                # Transitions between unique CAPs occur at indices 0 -> 1, 1 -> 2, and 4 -> 5
                transition_frequency = 3

         - ``"transition_probability"``: The probability of transitioning from one CAP to another CAP (or the same CAP).
           This is calculated as (Number of transitions from A to B) / (Total transitions from A). Note that the
           transition probability from CAP-A -> CAP-B is not the same as CAP-B -> CAP-A.

           ::

                # Note last two numbers in the predicted timeseries are switched for this example
                predicted_subject_timeseries = [1, 2, 1, 1, 3, 1]

                # If three CAPs were identified in the analysis
                combinations = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]

                # Represents transition from CAP-1 -> CAP-2
                target = (1, 2)

                # There are 4 ones in the timeseries but only three transitions from 1; 1 -> 2, 1 -> 1, 1 -> 3
                n_transitions_from_1 = 3

                # There is only one 1 -> 2 transition
                transition_probability = 1 / 3

            **Note**: In the supplementary material for Yang et al., the mathematical relationship between
            temporal fraction, counts, and persistence is ``temporal fraction = (persistence * counts)/total volumes``.
            If persistence has been converted into time units (seconds), then
            ``temporal fraction = (persistence * counts) / (total volumes * TR)``.

        Parameters
        ----------
        subject_timeseries: :obj:`SubjectTimeseries` or :obj:`str`
            A dictionary mapping subject IDs to their run IDs and their associated timeseries (TRs x ROIs) as a NumPy
            array. Can also be a path to a pickle file containing this same structure. The expected structure of is as
            follows:

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

        tr: :obj:`float` or :obj:`None`, default=None
            The repetition time (TR) in seconds. If provided, persistence will be calculated as the average
            uninterrupted time, in seconds, spent in each CAP. If not provided, persistence will be calculated as the
            average uninterrupted volumes (TRs), in TR units, spent in each state.

        runs: :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            The run numbers to calculate CAP metrics for (e.g. ``runs=[0, 1]`` or ``runs=["01", "02"]``). If None, CAP
            metrics will be calculated for each run.

        continuous_runs: :obj:`bool`, default=False
            If True, all runs will be treated as a single, uninterrupted run.

            ::

                # CAP assignment of frames from for run_1 and run_2
                run_1 = [0, 1, 1]
                run_2 = [2, 3, 3]

                # Computation of each CAP metric will be conducted on the combined vector
                continuous_runs = [0, 1, 1, 2, 3, 3]

        metrics: {"temporal_fraction", "persistence", "counts", "transition_frequency", "transition_probability"} \
                 or :obj:`list["temporal_fraction", "persistence", "counts", "transition_frequency", "transition_probability"]`, \
                 default=["temporal_fraction", "persistence", "counts", "transition_frequency"]
            The metrics to calculate. Available options include "temporal_fraction", "persistence",
            "counts", "transition_frequency", and "transition_probability".

        return_df: :obj:`str`, default=True
            If True, returns ``pandas.DataFrame`` inside a dictionary`, mapping each dataframe to their metric.

        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save ``pandas.DataFrame`` as csv files. The directory will be created if it does not exist.
            Dataframes will not be saved if None.

        prefix_filename: :obj:`str` or :obj:`None`, default=None
            A prefix to append to the saved file names for each ``pandas.DataFrame``, if ``output_dir`` is provided.

        progress_bar: :obj:`bool`, default=False
            If True, displays a progress bar.

            .. versionadded:: 0.21.5

        See Also
        --------
        :data:`neurocaps.typing.SubjectTimeseries`
            The type definition for the subject timeseries dictionary structure.

        Returns
        -------
        dict[str, pd.DataFrame] or dict[str, dict[str, pd.DataFrame]]
            Dictionary containing `pandas.DataFrame` - one for each requested metric. In the case of "transition_probability",
            each group has a separate dataframe which is returned in the from of `dict[str, dict[str, pd.DataFrame]]`.
            Only returned if ``return_df`` is True.


        Note
        ----
        **Scaling:** If standardizing was requested in ``self.get_caps()``, then the columns/ROIs of the
        ``subject_timeseries`` provided to this method will be scaled using the group-specific mean and sample
        standard deviation derived from the concatenated data.

        **Group-Specific CAPs**: When the ``groups`` parameter is used during initialization of the ``CAP`` class,
        ``self.get_caps()`` computes separate k-means model for each group. This means that each group has its own
        specific k-means model that is used for CAP metric calculations. The inclusion of all groups within the same
        dataframe (for "temporal_fraction", "persistence", "counts", and "transition_frequency") is primarily to
        reduce the number of dataframes generated. Hence, each CAP (e.g., "CAP-1") is specific to its respective groups.
        For instance, "CAP-1" under Group A is distinct from "CAP-1" under Group B.

        For instance, if their are two groups, Group A and Group B, each with their own CAPs:

        - **A** has 2 CAPs: "CAP-1" and "CAP-2"
        - **B** has 3 CAPs: "CAP-1", "CAP-2", and "CAP-3"

        The resulting `"temporal_fraction"` dataframe ("persistence" and "counts" have a similar structure but
        "transition frequency" will only contain the "Subject_ID", "Group", and "Run" columns in addition to
        a "Transition_Frequency" column):

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

        The "NaN" indicates that "CAP-3" is not applicable for Group A. Additionally, "NaN" will only be observed
        in instances when two or more groups are specified and have different number of CAPs. As mentioned previously,
        "CAP-1", "CAP-2", and "CAP-3" for Group A is distinct from Group B due to using separate k-means models.

        If no groups were specified during initialization of the ``CAP`` class, the resulting `"temporal_fraction"`
        dataframe (assuming four CAPs were identified in the k-means model using all participants):

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

        **Transition Probability:** For `"transition_probability"`, each group has a separate dataframe to containing
        the CAP transitions for each group.

        - **Group A Transition Probability:** Stored in ``df_dict["transition_probability"]["A"]``
        - **Group B Transition Probability:** Stored in ``df_dict["transition_probability"]["B"]``

        **The resulting `"transition_probability"` for Group A**:

        +------------+---------+-------+-------+-------+-------+-------+-------+
        | Subject_ID | Group   | Run   |  1.1  |  1.2  |  1.3  | 2.1   | ...   |
        +============+=========+=======+=======+=======+=======+=======+=======+
        | 101        |    A    | run-1 | 0.40  | 0.60  |   0   | 0.2   | ...   |
        +------------+---------+-------+-------+-------+-------+-------+-------+
        | ...        | ...     | ...   | ...   | ...   | ...   | ...   | ...   |
        +------------+---------+-------+-------+-------+-------+-------+-------+

        **The resulting `"transition_probability"` for Group B**:

        +------------+---------+-------+-------+-------+-------+-------+
        | Subject_ID | Group   | Run   |  1.1  |  1.2  |  2.1  | ...   |
        +============+=========+=======+=======+=======+=======+=======+
        | 102        |    B    | run-1 | 0.70  | 0.30  | 0.10  | ...   |
        +------------+---------+-------+-------+-------+-------+-------+
        | ...        | ...     | ...   | ...   | ...   | ...   | ...   |
        +------------+---------+-------+-------+-------+-------+-------+

        Here the columns indicate {from}.{to}. For instance, column 1.2 indicates the probability of transitioning
        from CAP-1 to CAP-2.

        If no groups are specified, then the dataframe is stored in ``df_dict["transition_probability"]["All Subjects"]``.

        References
        ----------
        Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals.
        NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

        Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation
        patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia.
        NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193
        """
        if not hasattr(self, "_kmeans"):
            self._raise_error("_kmeans")

        if prefix_filename is not None and output_dir is None:
            LG.warning("`prefix_filename` supplied but no `output_dir` specified. Files will not be saved.")

        if runs and not isinstance(runs, list):
            runs = [runs]

        metrics = self._filter_metrics(metrics)

        subject_timeseries = self._process_subject_timeseries(subject_timeseries)

        # Assign each subject's frame to a CAP
        predicted_subject_timeseries = self._build_prediction_dict(subject_timeseries, runs, continuous_runs)

        # Get CAP information
        cap_names, cap_numbers, group_cap_counts = self._cap_info()

        # Get combination of transitions in addition to building the base dataframe dictionary
        if "transition_probability" in metrics:
            cap_pairs = self._get_pairs()
            df_dict, temp_dict = self._build_df(metrics, cap_names, cap_pairs)
        else:
            df_dict = self._build_df(metrics, cap_names)

        # Generate list for iteration
        distributed_dict = self._distribute(predicted_subject_timeseries)

        for subj_id in tqdm(distributed_dict, desc="Computing Metrics for Subjects", disable=not progress_bar):
            for group, curr_run in distributed_dict[subj_id]:
                if "temporal_fraction" in metrics:
                    # Get frequency
                    frequency_dict = {
                        key: np.where(predicted_subject_timeseries[subj_id][curr_run] == key, 1, 0).sum()
                        for key in range(1, group_cap_counts[group] + 1)
                    }

                    self._update_dict(cap_numbers, group_cap_counts[group], frequency_dict)

                    if "temporal_fraction" in metrics:
                        proportion_dict = {
                            key: value / (len(predicted_subject_timeseries[subj_id][curr_run]))
                            for key, value in frequency_dict.items()
                        }

                        # Populate Dataframe
                        new_row = [subj_id, group, curr_run] + [items for items in proportion_dict.values()]
                        df_dict["temporal_fraction"].loc[len(df_dict["temporal_fraction"])] = new_row

                if "counts" in metrics:
                    count_dict = {}
                    for target in cap_numbers:
                        # + 1 is always added to segments to handle the + 1 needed to account for transitions and to avoid
                        # a NaN for persistence. This ensures counts is 0 if target not present
                        if target in predicted_subject_timeseries[subj_id][curr_run]:
                            _, counts = self._segments(target, predicted_subject_timeseries[subj_id][curr_run])
                        else:
                            counts = 0

                        count_dict.update({target: counts})

                    self._update_dict(cap_numbers, group_cap_counts[group], count_dict)

                    # Populate Dataframe
                    new_row = [subj_id, group, curr_run] + [items for items in count_dict.values()]
                    df_dict["counts"].loc[len(df_dict["counts"])] = new_row

                if "persistence" in metrics:
                    # Initialize variable
                    persistence_dict = {}

                    # Iterate through caps
                    for target in cap_numbers:
                        binary_arr, n_segments = self._segments(target, predicted_subject_timeseries[subj_id][curr_run])
                        # Sum of ones in the binary array divided by segments, then multiplied by 1 or the tr; segment is
                        # always 1 at minimum due to + 1; binary_arr.sum() is 0 when empty or the condition isn't met;
                        # thus, persistence is 0 instead of NaN in this case
                        persistence_dict.update({target: (binary_arr.sum() / n_segments) * (tr if tr else 1)})

                    self._update_dict(cap_numbers, group_cap_counts[group], persistence_dict)

                    # Populate Dataframe
                    new_row = [subj_id, group, curr_run] + [items for _, items in persistence_dict.items()]
                    df_dict["persistence"].loc[len(df_dict["persistence"])] = new_row

                if "transition_frequency" in metrics:
                    # Sum the differences that are not zero - [1, 2, 1, 1, 1, 3] becomes [1, -1, 0, 0, 2]
                    # binary representation for values not zero is [1, 1, 0, 0, 1] = 3 transitions
                    transition_frequency = np.where(
                        np.diff(predicted_subject_timeseries[subj_id][curr_run], n=1) != 0, 1, 0
                    ).sum()

                    # Populate DataFrame
                    new_row = [subj_id, group, curr_run, transition_frequency]
                    df_dict["transition_frequency"].loc[len(df_dict["transition_frequency"])] = new_row

                if "transition_probability" in metrics:
                    base_row = [subj_id, group, curr_run] + [0.0] * (temp_dict[group].shape[-1] - 3)
                    temp_dict[group].loc[len(temp_dict[group])] = base_row

                    # Arrays for transitioning from and to element
                    trans_from = predicted_subject_timeseries[subj_id][curr_run][:-1]
                    trans_to = predicted_subject_timeseries[subj_id][curr_run][1:]

                    indx = temp_dict[group].index[-1]

                    # Iterate through pairs and calculate probability
                    for e1, e2 in cap_pairs[group]:
                        # Get total number of possible transitions for first element
                        total_trans = np.sum(trans_from == e1)
                        column = f"{e1}.{e2}"
                        # Compute sum of adjacent pairs of A -> B and divide
                        temp_dict[group].loc[indx, column] = (
                            np.sum((trans_from == e1) & (trans_to == e2)) / total_trans if total_trans > 0 else 0
                        )

        # Add temporary dict for transition probability to `df_dict`
        if "transition_probability" in metrics:
            df_dict["transition_probability"] = temp_dict

        if output_dir:
            self._save_metrics(output_dir, df_dict, prefix_filename)

        if return_df:
            return df_dict

    @staticmethod
    def _filter_metrics(metrics):
        metrics = [metrics] if isinstance(metrics, str) else metrics
        # Change metrics to set
        metrics = set(metrics)
        valid_metrics = {"temporal_fraction", "persistence", "counts", "transition_frequency", "transition_probability"}
        set_diff = metrics - valid_metrics
        metrics = metrics.intersection(valid_metrics)

        if set_diff:
            formatted_string = ", ".join(["'{a}'".format(a=x) for x in set_diff])
            LG.warning(f"The following invalid metrics will be ignored: {formatted_string}.")

        if not metrics:
            formatted_string = ", ".join(["'{a}'".format(a=x) for x in valid_metrics])
            raise ValueError(f"No valid metrics in `metrics` list. Valid metrics are {formatted_string}.")

        return metrics

    def _cap_info(self):
        group_cap_counts = {}

        for group in self._groups:
            # Store the length of caps in each group
            group_cap_counts.update({group: len(self._caps[group])})

        # CAP names based on groups with the most CAPs
        cap_names = list(self._caps[max(group_cap_counts, key=group_cap_counts.get)])
        cap_numbers = [int(name.split("-")[-1]) for name in cap_names]

        return cap_names, cap_numbers, group_cap_counts

    def _build_prediction_dict(self, subject_timeseries, runs, continuous_runs):
        for subj_id, group in self._subject_table.items():
            # Initialize predicted timeseries dictionary if it does not exist
            if "predicted_subject_timeseries" not in locals():
                predicted_subject_timeseries = {}

            predicted_subject_timeseries[subj_id] = {}
            subject_runs, miss_runs = self._get_runs(runs, list(subject_timeseries[subj_id]))

            if miss_runs:
                LG.warning(f"[SUBJECT: {subj_id}] Does not have the requested runs: {', '.join(miss_runs)}.")

            if not subject_runs:
                LG.warning(f"[SUBJECT: {subj_id}] Excluded from the concatenated timeseries due to having no runs.")
                continue

            for curr_run in subject_runs:
                # Standardize or not
                if self._standardize:
                    timeseries = (subject_timeseries[subj_id][curr_run] - self._mean_vec[group]) / self._stdev_vec[
                        group
                    ]
                else:
                    timeseries = subject_timeseries[subj_id][curr_run]

                # Set run_id
                run_id = curr_run if not continuous_runs or len(subject_runs) == 1 else "run-continuous"
                # Add 1 to the prediction vector since labels start at 0, ensures that the labels map onto the caps
                prediction_vector = self._kmeans[group].predict(timeseries) + 1
                if run_id != "run-continuous":
                    predicted_subject_timeseries[subj_id].update({run_id: prediction_vector})
                else:
                    # Horizontally stack predicted runs
                    if curr_run == subject_runs[0]:
                        predicted_continuous_timeseries = prediction_vector
                    else:
                        predicted_continuous_timeseries = np.hstack(
                            [predicted_continuous_timeseries, prediction_vector]
                        )

            if run_id == "run-continuous":
                predicted_subject_timeseries[subj_id].update({run_id: predicted_continuous_timeseries})

        return predicted_subject_timeseries

    # Get all pairs of all possible transitions
    def _get_pairs(self):
        group_caps = {}
        all_pairs = {}

        for group in self._groups:
            group_caps.update({group: [int(name.split("-")[-1]) for name in self._caps[group]]})
            all_pairs.update({group: list(itertools.product(group_caps[group], group_caps[group]))})

        return all_pairs

    def _build_df(self, metrics, cap_names, pairs=None):
        df_dict = {}
        base_cols = ["Subject_ID", "Group", "Run"]

        for metric in metrics:
            if metric not in ["transition_frequency", "transition_probability"]:
                df_dict.update({metric: pd.DataFrame(columns=base_cols + list(cap_names))})
            elif metric == "transition_probability":
                temp_dict = {}
                for group in self._groups:
                    temp_dict.update({group: pd.DataFrame(columns=base_cols + [f"{x}.{y}" for x, y in pairs[group]])})
            else:
                df_dict.update({metric: pd.DataFrame(columns=base_cols + ["Transition_Frequency"])})

        if "transition_probability" in metrics:
            return df_dict, temp_dict
        else:
            return df_dict

    def _distribute(self, predicted_subject_timeseries):
        distributed_dict = {}

        for subj_id, group in self._subject_table.items():
            distributed_dict[subj_id] = []
            for curr_run in predicted_subject_timeseries[subj_id]:
                distributed_dict[subj_id].append((group, curr_run))

        return distributed_dict

    # Replace zeros with nan for groups with less caps than the group with the max caps
    @staticmethod
    def _update_dict(cap_numbers, group_cap_counts, curr_dict):
        if max(cap_numbers) > group_cap_counts:
            for i in range(group_cap_counts + 1, max(cap_numbers) + 1):
                curr_dict.update({i: float("nan")})

    @staticmethod
    def _segments(target, timeseries):
        # Binary representation of numpy array - if [1, 2, 1, 1, 1, 3] and target is 1, then it is [1, 0, 1, 1, 1, 0]
        binary_arr = np.where(timeseries == target, 1, 0)
        # Get indices of values that equal 1; [0, 2, 3, 4]
        target_indices = np.where(binary_arr == 1)[0]
        # Count the transitions, indices where diff > 1 is a transition; diff of indices = [2, 1, 1];
        # binary for diff > 1 = [1, 0, 0]; thus, n_segments = transitions + first_sequence(1) = 2
        n_segments = np.where(np.diff(target_indices, n=1) > 1, 1, 0).sum() + 1

        return binary_arr, n_segments

    def _save_metrics(self, output_dir, df_dict, prefix_filename):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for metric in df_dict:
            if prefix_filename:
                filename = os.path.splitext(prefix_filename.rstrip())[0].rstrip() + f"-{metric}"
            else:
                filename = f"{metric}"

            if metric != "transition_probability":
                df_dict[f"{metric}"].to_csv(
                    path_or_buf=os.path.join(output_dir, f"{filename}.csv"), sep=",", index=False
                )
            else:
                for group in self._groups:
                    df_dict[f"{metric}"][group].to_csv(
                        path_or_buf=os.path.join(output_dir, f"{filename}-{group.replace(' ', '_')}.csv"),
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
        visual_scope: Union[Literal["regions", "nodes"], list[Literal["regions", "nodes"]]] = "regions",
        show_figs: bool = True,
        subplots: bool = False,
        **kwargs,
    ) -> Self:
        """
        Generate Heatmaps and Outer Product Plots for CAPs.

        Plot CAPs as heatmaps or outer products at the node or region/network levels. This function produces a
        ``seaborn.heatmap`` for each CAP. Note, if groups were given when the ``CAP`` class was initialized, separate
        plots will be generated for all groups.

        Parameters
        ----------
        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory for saving plots as png files. The directory will be created if it does not exist. If None,
            plots will not be saved.

        suffix_title: :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the filename of each saved plot if ``output_dir`` is provided.

        plot_options: {"outer_product", "heatmap"} or :obj:`list["outer_product", "heatmap"]`, default="outer_product"
            Type of plots to create. Options are "outer_product" or "heatmap".

        visual_scope: {"regions", "nodes"} or :obj:`list["regions", "nodes"]`, default="regions"
            Determines whether plotting is done at the region level or node level. For "regions", the values of all
            nodes in the same regions (including both hemispheres) are averaged together then plotted. For "nodes",
            plots individual node values separately.

        show_figs: :obj:`bool`, default=True
           Display figures.

        subplots: :obj:`bool`, default=True
            Produce subplots for outer product plots, combining all plots into a single figure.

        **kwargs:
            Keyword arguments used when saving figures. Valid keywords include:

            - dpi: :obj:`int`, default=300 -- Dots per inch for the figure.
            - figsize: :obj:`tuple`, default=(8, 6) -- Size of the figure in inches.
            - fontsize: :obj:`int`, default=14 -- Font size for the title of individual plots or subplots.
            - hspace: :obj:`float`, default=0.4 -- Height space between subplots.
            - wspace: :obj:`float`, default=0.4 -- Width space between subplots.
            - xticklabels_size: :obj:`int`, default=8 -- Font size for x-axis tick labels.
            - yticklabels_size: :obj:`int`, default=8 -- Font size for y-axis tick labels.
            - shrink: :obj:`float`, default=0.8 -- Fraction by which to shrink the colorbar.
            - cbarlabels_size: :obj:`int`, default=8 -- Font size for the colorbar labels.
            - nrow: :obj:`int`, default=varies (max 5) -- Number of rows for subplots. Default varies but the maximum is 5.
            - ncol: :obj:`int` or :obj:`None`, default=None -- Number of columns for subplots. Default varies but the maximum is 5.
            - suptitle_fontsize: :obj:`float`, default=0.7 -- Font size for the main title when subplot is True.
            - tight_layout: :obj:`bool`, default=True -- Use tight layout for subplots.
            - rect: :obj:`list[int]`, default=[0, 0.03, 1, 0.95] -- Rectangle parameter for ``tight_layout`` when\
                ``subplots`` are True. Fixes whitespace issues.
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
            - alpha: :obj:`float` or :obj:`None`, default=None -- Controls transparency and ranges from 0 (transparent)\
                to 1 (opaque).
            - bbox_inches: :obj:`str` or :obj:`None`, default="tight" -- Alters size of the whitespace in the saved image.
            - hemisphere_labels: :obj:`bool`, default=False -- When ``visual_scope="nodes"``, shows simplified\
                left/right hemisphere division line instead of individual node labels (only for Custom and Schaefer\
                parcellations). For Custom parcellations, assumes labels are ordered such that all left hemisphere nodes\
                come before right hemisphere nodes. Ignores edgecolors; linewidths/linecolor only affect division line.
            - cmap: :obj:`str`, :obj:`callable` default="coolwarm" -- Color map for the plot cells. Options include\
                strings to call seaborn's pre-made palettes, ``seaborn.diverging_palette`` function to generate custom\
                palettes, and ``matplotlib.color.LinearSegmentedColormap`` to generate custom palettes.
            - vmin: :obj:`float` or :obj:`None`, default=None -- The minimum value to display in colormap.
            - vmax: :obj:`float` or :obj:`None`, default=None -- The maximum value to display in colormap.

        Returns
        -------
        self

        Note
        ----
        **Parcellation Approach**: the "nodes" and "regions" subkeys are required in ``parcel_approach`` for this
        function.

        **Color Palettes**: Refer to `seaborn's Color Palettes <https://seaborn.pydata.org/tutorial/color_palettes.html>`_
        for valid pre-made palettes.

        """
        if self._parcel_approach is None:
            self._raise_error("_parcel_approach")

        if not hasattr(self, "_caps"):
            self._raise_error("_caps")

        if suffix_filename is not None and output_dir is None:
            LG.warning("`suffix_filename` supplied but no `output_dir` specified. Files will not be saved.")

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
                "Number of nodes used for CAPs does not equal the number of nodes specified in `parcel_approach`."
            )

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Convert to list
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
                raise ValueError("`hemisphere_labels` is only available when `visual_scope == 'nodes'`.")
            if parcellation_name == "AAL":
                raise ValueError("`hemisphere_labels` is only available for 'Custom' and 'Schaefer'.")

        # Ensure plot_options and visual_scope are lists
        plot_options = plot_options if isinstance(plot_options, list) else list(plot_options)
        visual_scope = visual_scope if isinstance(visual_scope, list) else list(visual_scope)
        # Initialize outer product attribute
        if "outer_product" in plot_options:
            self._outer_products = {}

        distributed_list = list(itertools.product(plot_options, visual_scope, self._groups))

        for plot_option, scope, group in distributed_list:
            # Get correct labels depending on scope
            if scope == "regions":
                cap_dict = {
                    group: {k: v for k, v in self._region_means[group].items() if k != "Regions"}
                    for group in self._region_means.keys()
                }
                columns = list(self._parcel_approach[parcellation_name]["regions"])
            elif scope == "nodes":
                if parcellation_name in ["Schaefer", "AAL"]:
                    cap_dict, columns = self._caps, self._parcel_approach[parcellation_name]["nodes"]
                else:
                    cap_dict = self._caps
                    columns = [
                        x[0] + " " + x[1]
                        for x in list(itertools.product(["LH", "RH"], self._parcel_approach["Custom"]["regions"]))
                    ]

            # Generate plot for each group
            input_keys = dict(
                group=group,
                plot_dict=plot_dict,
                cap_dict=cap_dict,
                columns=columns,
                output_dir=output_dir,
                suffix_title=suffix_title,
                suffix_filename=suffix_filename,
                show_figs=show_figs,
                scope=scope,
                parcellation_name=parcellation_name,
            )

            # Generate plot for each group
            if plot_option == "outer_product":
                self._generate_outer_product_plots(**input_keys, subplots=subplots)
            elif plot_option == "heatmap":
                self._generate_heatmap_plots(**input_keys)

        return self

    def _compute_region_means(self, parcellation_name):
        # Internal function to create an attribute called `_region_means` (previously `_region_caps`). Purpose is to
        # average the values of all nodes in a corresponding region to create region heatmaps or outer product plots
        self._region_means = {group: {} for group in self._groups}

        region_dict = self._parcel_approach["Custom"]["regions"] if parcellation_name == "Custom" else None
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
                            for index, node in enumerate(self._parcel_approach[parcellation_name]["nodes"])
                            if region in node
                        ]
                    )
                else:
                    region_indxs = np.array(list(region_dict[region]["lh"]) + list(region_dict[region]["rh"]))

                if region_means is None:
                    region_means = np.array([np.average(self._caps[group][cap][region_indxs])])
                else:
                    region_means = np.hstack([region_means, np.average(self._caps[group][cap][region_indxs])])

            # Append regions and their means
            self._region_means[group].update({"Regions": regions})
            self._region_means[group].update({cap: region_means})

    def _generate_outer_product_plots(
        self,
        group,
        plot_dict,
        cap_dict,
        columns,
        subplots,
        output_dir,
        suffix_title,
        suffix_filename,
        show_figs,
        scope,
        parcellation_name,
    ):
        # Nested dictionary for group
        self._outer_products[group] = {}

        # Create base grid for subplots
        if subplots:
            # Max five subplots per row for default
            default_col = len(cap_dict[group]) if len(cap_dict[group]) <= 5 else 5
            ncol = plot_dict["ncol"] if plot_dict["ncol"] is not None else default_col

            if ncol > len(cap_dict[group]):
                ncol = len(cap_dict[group])

            # Pad nrow, since int will round down, padding is needed for cases
            # where len(cap_dict[group])/ncol is a float. This will add the extra row needed
            x_pad = 0 if len(cap_dict[group]) / ncol <= 1 else 1
            nrow = plot_dict["nrow"] if plot_dict["nrow"] is not None else x_pad + int(len(cap_dict[group]) / ncol)
            subplot_figsize = (8 * ncol, 6 * nrow) if plot_dict["figsize"] == (8, 6) else plot_dict["figsize"]
            fig, axes = plt.subplots(nrow, ncol, sharex=False, sharey=plot_dict["sharey"], figsize=subplot_figsize)

            suptitle = f"{group} {suffix_title}" if suffix_title else f"{group}"
            fig.suptitle(suptitle, fontsize=plot_dict["suptitle_fontsize"])
            fig.subplots_adjust(hspace=plot_dict["hspace"], wspace=plot_dict["wspace"])

            if plot_dict["tight_layout"]:
                fig.tight_layout(rect=plot_dict["rect"])

            # Current subplot
            axes_x, axes_y = [0, 0]

        # Iterate over CAPs
        for cap in cap_dict[group]:
            # Calculate outer product
            self._outer_products[group].update({cap: np.outer(cap_dict[group][cap], cap_dict[group][cap])})
            # Create labels if nodes requested for scope
            if scope == "nodes" and plot_dict["hemisphere_labels"] is False:
                labels, _ = self._create_node_labels(parcellation_name, self._parcel_approach, columns)

            if subplots:
                ax = axes[axes_y] if nrow == 1 else axes[axes_x, axes_y]
                # Modify tick labels based on scope
                if scope == "regions":
                    display = seaborn.heatmap(
                        ax=ax,
                        data=self._outer_products[group][cap],
                        xticklabels=columns,
                        yticklabels=columns,
                        **self._base_kwargs(plot_dict),
                    )
                else:
                    if plot_dict["hemisphere_labels"] is False:
                        display = seaborn.heatmap(
                            ax=ax, data=self._outer_products[group][cap], **self._base_kwargs(plot_dict)
                        )
                    else:
                        display = seaborn.heatmap(
                            ax=ax,
                            data=self._outer_products[group][cap],
                            **self._base_kwargs(plot_dict, line=False, edge=False),
                        )

                    if plot_dict["hemisphere_labels"] is False:
                        ax = self._set_ticks(ax, labels)
                    else:
                        ax, division_line, plot_dict["linewidths"] = self._division_line(
                            ax, parcellation_name, plot_dict["linewidths"]
                        )

                        ax.axhline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])
                        ax.axvline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])

                # Add border
                if plot_dict["borderwidths"] != 0:
                    border_length = self._outer_products[group][cap].shape[0]
                    display = self._border(display, plot_dict, border_length)

                # Modify label sizes
                display = self._label_size(display, plot_dict, set_x=True, set_y=False)
                # Modify label sizes; if share_y, only set y for plots at axes == 0
                if plot_dict["sharey"]:
                    if axes_y == 0:
                        display = self._label_size(display, plot_dict, set_x=False, set_y=True)
                else:
                    display = self._label_size(display, plot_dict, set_x=False, set_y=True)

                # Set title of subplot
                ax.set_title(cap, fontsize=plot_dict["fontsize"])

                # If modulus is zero, move onto the new column back to zero
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
                        xticklabels=columns,
                        yticklabels=columns,
                        **self._base_kwargs(plot_dict),
                    )
                else:
                    if plot_dict["hemisphere_labels"] is False:
                        display = seaborn.heatmap(
                            self._outer_products[group][cap],
                            xticklabels=[],
                            yticklabels=[],
                            **self._base_kwargs(plot_dict),
                        )
                    else:
                        display = seaborn.heatmap(
                            self._outer_products[group][cap],
                            xticklabels=[],
                            yticklabels=[],
                            **self._base_kwargs(plot_dict, line=False, edge=False),
                        )

                    if plot_dict["hemisphere_labels"] is False:
                        display = self._set_ticks(display, labels)
                    else:
                        display, division_line, plot_dict["linewidths"] = self._division_line(
                            display, parcellation_name, plot_dict["linewidths"]
                        )

                        plt.axhline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])
                        plt.axvline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])

                # Add border
                if plot_dict["borderwidths"] != 0:
                    border_length = self._outer_products[group][cap].shape[0]
                    display = self._border(display, plot_dict, border_length)

                # Modify label sizes
                plot_title = f"{group} {cap} {suffix_title}" if suffix_title else f"{group} {cap}"
                display.set_title(plot_title, fontdict={"fontsize": plot_dict["fontsize"]})
                display = self._label_size(display, plot_dict)

                # Save individual plots
                if output_dir:
                    partial_filename = f"{group}_{cap}"
                    self._save_heatmap(
                        display, scope, partial_filename, suffix_filename, plot_dict, output_dir, call="outer_product"
                    )

        # Remove subplots with no data
        if subplots:
            [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]

        # Save subplot
        if subplots and output_dir:
            partial_filename = f"{group}_CAPs"
            self._save_heatmap(
                display, scope, partial_filename, suffix_filename, plot_dict, output_dir, call="outer_product"
            )

        # Display figures
        plt.show() if show_figs else plt.close()

    def _generate_heatmap_plots(
        self,
        group,
        plot_dict,
        cap_dict,
        columns,
        output_dir,
        suffix_title,
        suffix_filename,
        show_figs,
        scope,
        parcellation_name,
    ):
        # Initialize new grid
        plt.figure(figsize=plot_dict["figsize"])

        if scope == "regions":
            display = seaborn.heatmap(
                pd.DataFrame(cap_dict[group], index=columns),
                xticklabels=True,
                yticklabels=True,
                **self._base_kwargs(plot_dict),
            )
        else:
            # Create Labels
            if plot_dict["hemisphere_labels"] is False:
                labels, names_list = self._create_node_labels(parcellation_name, self._parcel_approach, columns)

                display = seaborn.heatmap(
                    pd.DataFrame(cap_dict[group], columns=list(cap_dict[group])),
                    xticklabels=True,
                    yticklabels=True,
                    **self._base_kwargs(plot_dict),
                )

                plt.yticks(ticks=[pos for pos, label in enumerate(labels) if label], labels=names_list)
            else:
                display = seaborn.heatmap(
                    pd.DataFrame(cap_dict[group], columns=list(cap_dict[group])),
                    xticklabels=True,
                    yticklabels=True,
                    **self._base_kwargs(plot_dict, line=False, edge=False),
                )

                display, division_line, plot_dict["linewidths"] = self._division_line(
                    display, parcellation_name, plot_dict["linewidths"], call="heatmap"
                )

                plt.axhline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])

        if plot_dict["borderwidths"] != 0:
            y_length = len(cap_dict[group][list(cap_dict[group])[0]])
            display = self._border(display, plot_dict, y_length, len(self._caps[group]))

        # Modify label sizes
        display = self._label_size(display, plot_dict)
        plot_title = f"{group} CAPs {suffix_title}" if suffix_title else f"{group} CAPs"
        display.set_title(plot_title, fontdict={"fontsize": plot_dict["fontsize"]})

        # Save plots
        if output_dir:
            partial_filename = f"{group}_CAPs"
            self._save_heatmap(display, scope, partial_filename, suffix_filename, plot_dict, output_dir, call="heatmap")

        # Display figures
        plt.show() if show_figs else plt.close()

    @staticmethod
    def _base_kwargs(plot_dict, line=True, edge=True):
        kwargs = {
            "cmap": plot_dict["cmap"],
            "cbar_kws": {"shrink": plot_dict["shrink"]},
            "annot": plot_dict["annot"],
            "annot_kws": plot_dict["annot_kws"],
            "fmt": plot_dict["fmt"],
            "alpha": plot_dict["alpha"],
            "vmin": plot_dict["vmin"],
            "vmax": plot_dict["vmax"],
        }

        if line:
            kwargs.update({"linewidths": plot_dict["linewidths"], "linecolor": plot_dict["linecolor"]})
        if edge:
            kwargs.update({"edgecolors": plot_dict["edgecolors"]})

        return kwargs

    @staticmethod
    def _create_node_labels(parcellation_name, parcel_approach, columns):
        # Get frequency of each major hemisphere and region in Schaefer, AAL, or Custom atlas
        if parcellation_name == "Schaefer":
            nodes = parcel_approach[parcellation_name]["nodes"]
            # Retain only the hemisphere and primary Schaefer network
            nodes = [node.split("_")[:2] for node in nodes]
            frequency_dict = collections.Counter([" ".join(node) for node in nodes])
        elif parcellation_name == "AAL":
            nodes = parcel_approach[parcellation_name]["nodes"]
            regions = _handle_aal(nodes, unique=False)
            frequency_dict = collections.Counter(regions)
        else:
            frequency_dict = {}
            for names_id in columns:
                # For custom, columns comes in the form of "Hemisphere Region"
                hemisphere_id = "LH" if names_id.startswith("LH ") else "RH"
                region_id = re.split("LH |RH ", names_id)[-1]
                node_indices = parcel_approach["Custom"]["regions"][region_id][hemisphere_id.lower()]
                frequency_dict.update({names_id: len(node_indices)})

        # Get the names, which indicate the hemisphere and region
        # Reverting Counter objects to list retains original ordering of nodes in list as of Python 3.7
        names_list = list(frequency_dict)
        labels = ["" for _ in range(0, len(parcel_approach[parcellation_name]["nodes"]))]

        starting_value = 0

        # Iterate through names_list and assign the starting indices corresponding to unique region and hemisphere key
        for num, name in enumerate(names_list):
            if num == 0:
                labels[0] = name
            else:
                # Shifting to previous frequency of the preceding network to obtain the new starting value of
                # the subsequent region and hemisphere pair
                starting_value += frequency_dict[names_list[num - 1]]
                labels[starting_value] = name

        return labels, names_list

    def _division_line(self, display, parcellation_name, linewidths, call="outer"):
        n_labels = len(self._parcel_approach[parcellation_name]["nodes"])
        division_line = n_labels // 2
        left_hemisphere_tick = (0 + division_line) // 2
        right_hemisphere_tick = (division_line + n_labels) // 2

        if call == "outer":
            display.set_xticks([left_hemisphere_tick, right_hemisphere_tick])
            display.set_xticklabels(["LH", "RH"])

        display.set_yticks([left_hemisphere_tick, right_hemisphere_tick])
        display.set_yticklabels(["LH", "RH"])

        line_widths = linewidths if linewidths != 0 else 1

        return display, division_line, line_widths

    @staticmethod
    def _set_ticks(display, labels):
        ticks = [i for i, label in enumerate(labels) if label]

        display.set_xticks(ticks)
        display.set_xticklabels([label for label in labels if label])
        display.set_yticks(ticks)
        display.set_yticklabels([label for label in labels if label])

        return display

    @staticmethod
    def _border(display, plot_dict, length, axvline=None):
        display.axhline(y=0, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        display.axhline(y=length, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        display.axvline(x=0, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])

        if axvline:
            display.axvline(x=axvline, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        else:
            display.axvline(x=length, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])

        return display

    @staticmethod
    def _label_size(display, plot_dict, set_x=True, set_y=True):
        if set_x:
            display.set_xticklabels(
                display.get_xticklabels(), size=plot_dict["xticklabels_size"], rotation=plot_dict["xlabel_rotation"]
            )

        if set_y:
            display.set_yticklabels(
                display.get_yticklabels(), size=plot_dict["yticklabels_size"], rotation=plot_dict["ylabel_rotation"]
            )

        if plot_dict["cbarlabels_size"]:
            cbar = display.collections[0].colorbar
            cbar.ax.tick_params(labelsize=plot_dict["cbarlabels_size"])

        return display

    # Function to save plots for outer_product and heatmap called by caps2plot
    @staticmethod
    def _save_heatmap(display, scope, partial, suffix, plot_dict, output_dir, call):
        full_filename = partial + f"_{call}-{scope}"

        if suffix:
            full_filename += f"_{suffix}".replace(" ", "_")

        full_filename += ".png"
        display.get_figure().savefig(
            os.path.join(output_dir, full_filename), dpi=plot_dict["dpi"], bbox_inches=plot_dict["bbox_inches"]
        )

    def caps2corr(
        self,
        output_dir: Optional[str] = None,
        suffix_title: Optional[str] = None,
        suffix_filename: Optional[str] = None,
        show_figs: bool = True,
        save_plots: bool = True,
        return_df: bool = False,
        save_df: bool = False,
        **kwargs,
    ) -> Union[dict[str, pd.DataFrame], None]:
        """
        Generate Pearson Correlation Matrix for CAPs.

        Produces a correlation matrix of all CAPs and visualizes it using ``seaborn.heatmap``. Can also produce
        a pandas Dataframe of the correlation matrix where each element contains its uncorrected p-value in parenthesis,
        with a single asterisk if < 0.05, a double asterisk if < 0.01, and a triple asterisk < 0.001. Note, if groups
        were given when the ``CAP`` class was initialized, separate correlation matrices will be generated for all
        groups.

        Parameters
        ----------
        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save plots (if ``save_plots`` is True) and correlation matrices DataFrames (if ``save_df`` is
            True). The directory will be created if it does not exist. If None, plots and dataFrame will not be saved.

        suffix_title: :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the filename of each saved plot if ``output_dir`` is provided.

        show_figs: :obj:`bool`, default=True
            Display figures.

        save_plots: :obj:`bool`, default=True
            If True, plots are saves as png images. For this to be used, ``output_dir`` must be specified.

        return_df: :obj:`bool`, default=False
            If True, returns a dictionary with a correlation matrix for each group.

        save_df: :obj:`bool`, default=False,
            If True, saves the correlation matrix contained in the DataFrames as csv files. For this to be used,
            ``output_dir`` must be specified.

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
            - alpha: :obj:`float` or :obj:`None`, default=None -- Controls transparency and ranges from 0\
                (transparent) to 1 (opaque).
            - bbox_inches: :obj:`str` or :obj:`None`, default="tight" -- Alters size of the whitespace in the saved image.
            - cmap: :obj:`str`, :obj:`callable` default="coolwarm" -- Color map for the plot cells. Options include\
                strings to call seaborn's pre-made palettes, ``seaborn.diverging_palette`` function to generate custom\
                palettes, and ``matplotlib.color.LinearSegmentedColormap`` to generate custom palettes.
            - vmin: :obj:`float` or :obj:`None`, default=None -- The minimum value to display in colormap.
            - vmax: :obj:`float` or :obj:`None`, default=None -- The maximum value to display in colormap.

        Returns
        -------
        dict[str, pd.DataFrame]
            A dictionary mapping an instance of a pandas DataFrame for each group. Only returned if ``return_df`` is
            True.

        Note
        ----
        **Color Palettes**: Refer to `seaborn's Color Palettes <https://seaborn.pydata.org/tutorial/color_palettes.html>`_
        for valid pre-made palettes.
        """
        corr_dict = {group: None for group in self._groups} if return_df or save_df else None

        if not hasattr(self, "_caps"):
            self._raise_error("_caps")

        if suffix_filename is not None and output_dir is None:
            LG.warning("`suffix_filename` supplied but no `output_dir` specified. Files will not be saved.")

        # Create plot dictionary
        plot_dict = _check_kwargs(_PlotDefaults.caps2corr(), **kwargs)

        for group in self._groups:
            df = pd.DataFrame(self._caps[group])
            corr_df = df.corr(method="pearson")

            display = _create_display(corr_df, plot_dict, suffix_title, group, "corr")

            if corr_dict:
                # Get p-values; use np.eye to make main diagonals equal zero; implementation of tozCSS from
                # https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
                pval_df = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*corr_df.shape)
                # Add asterisk to values that meet the threshold
                pval_df = pval_df.map(
                    lambda x: f'({format(x, plot_dict["fmt"])})'
                    + "".join(["*" for code in [0.05, 0.01, 0.001] if x < code])
                )
                # Add the p-values to the correlation matrix
                corr_dict[group] = corr_df.map(lambda x: f'{format(x, plot_dict["fmt"])}') + " " + pval_df

            # Save figure
            if output_dir:
                _save_contents(
                    output_dir, suffix_filename, group, corr_dict, plot_dict, save_plots, save_df, display, call="corr"
                )

            # Display figures
            plt.show() if show_figs else plt.close()

        if return_df:
            return corr_dict

    # Create basename for certain files
    @staticmethod
    def _basename(group, cap, desc=None, suffix=None, ext="png"):
        base = f"{group.replace(' ', '_')}_{cap}"

        if desc:
            base += f"_{desc}"

        if suffix:
            base += f"_{suffix.rstrip().replace(' ', '_')}"

        return base + f".{ext}"

    def caps2niftis(
        self,
        output_dir: str,
        suffix_filename: Optional[str] = None,
        fwhm: Optional[float] = None,
        knn_dict: Optional[dict[str, Union[int, list[int], np.typing.NDArray[np.integer]]]] = None,
        progress_bar: bool = False,
    ) -> Self:
        """
        Standalone Method to Convert CAPs to NifTI Statistical Maps.

        Projects CAPs onto the parcellation in ``self.parcel_approach`` to create NifTI statistical maps by replacing
        parcellation labels with their corresponding CAP (cluster centroid) values. Creates compressed NifTI (.nii.gz)
        files. One image is generated per CAP. Note, if groups were given when the ``CAP`` class was initialized,
        separate NifTI images will be generated per CAP for all groups.

        Parameters
        ----------
        output_dir: :obj:`str`
            Directory to save nii.gz files. The directory will be created if it does not exist.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the name of the saved file.

        fwhm: :obj:`float` or :obj:`None`, default=None
            Strength of spatial smoothing to apply (in millimeters) to the statistical map prior to interpolating
            from MNI152 space to fsLR surface space. Uses Nilearn's ``image.smooth_img``.

        knn_dict: :obj:`dict[str, int | bool]`, default=None
            Use KNN (k-nearest neighbors) interpolation with reference atlas masking to fill in non-background
            coordinates that are assigned zero. Useful when custom parcellation does not project well from volumetric
            to surface space. The following subkeys are recognized:

            - "k": An integer (Default=1). Determines the number of nearest neighbors to consider.
            - "reference_atlas": A string (Default="Schaefer"). Specifies the atlas to use for reference masking ("AAL" or "Schaefer").
            - "resolution_mm": An integer (Default=1). Spatial resolution of the Schaefer parcellation (in millimeters) (1 or 2).
            - "remove_labels": A list or array (Default=None). The label IDs as integers of the regions in the parcellation to not interpolate.

            *Note*: This method is applied before the ``fwhm``.

        progress_bar: :obj:`bool`, default=False
            If True, displays a progress bar.

            .. versionadded:: 0.21.5

        Returns
        -------
        self

        Note
        ----
        **Assumption**: This function assumes that the background label for the parcellation is zero. Additionaly,
        the following approach is used to map each CAP onto the parcellation.

        ::

            atlas = nib.load(atlas_file)
            atlas_fdata = atlas.get_fdata()
            # Create array of zeroes with same dimensions as atlas
            atlas_array = np.zeros_like(atlas_fdata)

            # Get array containing all labels in parcellation in order
            target_array = sorted(np.unique(atlas_fdata))

            # Start at 1 to avoid assigment to the background label
            for indx, value in enumerate(cap_vector, start=1):
                atlas_array[atlas_fdata == target_array[indx]] = value
        """
        if self._parcel_approach is None:
            self._raise_error("_parcel_approach")

        if not hasattr(self, "_caps"):
            self._raise_error("_caps")

        # Check `knn_dict`
        if knn_dict:
            knn_dict = self._validate_knn_dict(knn_dict)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        parcellation_name = list(self._parcel_approach)[0]
        for group in self._groups:
            for cap in tqdm(
                self._caps[group], desc=f"Generating Statistical Maps [GROUP: {group}]", disable=not progress_bar
            ):
                stat_map = _cap2statmap(
                    atlas_file=self._parcel_approach[parcellation_name]["maps"],
                    cap_vector=self._caps[group][cap],
                    fwhm=fwhm,
                    knn_dict=knn_dict,
                )

                filename = self._basename(group, cap, suffix=suffix_filename, ext="nii.gz")

                nib.save(stat_map, os.path.join(output_dir, filename))

        return self

    @staticmethod
    def _validate_knn_dict(knn_dict):
        valid_atlases = ["Schaefer", "AAL"]

        if "reference_atlas" not in knn_dict:
            knn_dict["reference_atlas"] = "Schaefer"
            LG.warning("'reference_atlas' not specified in `knn_dict`. The default reference atlas is 'Schaefer'.")
        else:
            if not isinstance(knn_dict["reference_atlas"], str):
                raise TypeError("In `knn_dict`, 'reference_atlas' must be a string.")

            if not any(knn_dict["reference_atlas"] == atlas for atlas in valid_atlases):
                raise ValueError("Only 'Schaefer' and 'AAL' are valid options for 'reference_atlas'.")

        if "resolution_mm" not in knn_dict and knn_dict["reference_atlas"] == "Schaefer":
            knn_dict["resolution_mm"] = 1
            LG.warning(
                "Defaulting to 1mm resolution for the Schaefer atlas since 'resolution_mm' was not specified in "
                "`knn_dict`."
            )
        else:
            if "resolution_mm" in knn_dict and not knn_dict["reference_atlas"] == "Schaefer":
                knn_dict["resolution_mm"] = None
                LG.warning("'resolution_mm' only used when 'reference_atlas' is set to 'Schaefer'.")

            if "resolution_mm" not in knn_dict:
                knn_dict["resolution_mm"] = None

        if "k" not in knn_dict:
            knn_dict["k"] = 3
            LG.warning("Defaulting to k=3 since 'k' was not specified in `knn_dict`.")

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
        knn_dict: Optional[dict[str, Union[int, list[int], np.typing.NDArray[np.integer]]]] = None,
        progress_bar: bool = False,
        **kwargs,
    ) -> Self:
        """
        Project CAPs onto Surface Plots.

        Plot CAPs on cortical surface in fsLR space. First, projects CAPs onto parcellation to create
        NifTI statistical maps by replacing parcellation labels with their corresponding CAP (cluster centroid)
        values. Then uses neuromap's ``transforms.mni152_to_fslr`` for coordinate system transformation and surfplot's
        ``Plot`` for plotting. If CAPs where already converted to NifTI (``self.caps2niftis()``) and transformed to
        fsLR GifTI files externally, these can be provided using the ``fslr_giftis_dict`` parameter and will be
        converted to a suitable format for surfplot's ``Plot`` function by using neuromap's ``transforms.fslr_to_fslr``
        function. Note, if groups were given when the ``CAP`` class was initialized, surface plots will be generated
        per CAP for all groups.

        Parameters
        ----------
        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save plots as png files. The directory will be created if it does not exist. If None, plots
            will not be saved.

        suffix_title: :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the filename of each saved plot if ``output_dir`` is provided.

        show_figs: :obj:`bool`, default=True
            Display figures.

        fwhm: :obj:`float` or :obj:`None`, defualt=None
            Strength of spatial smoothing to apply (in millimeters) to the statistical map prior to interpolating
            from MNI152 space to fsLR surface space. Uses Nilearn's ``image.smooth``.

        fslr_density: {"4k", "8k", "32k", "164k"}, default="32k"
            Density of the fsLR surface when converting from MNI152 space to fsLR surface. Options are "32k" or
            "164k". If using ``fslr_giftis_dict`` options are "4k", "8k", "32k", and "164k".

        method: {"linear", "nearest"}, default="linear"
            Interpolation method to use when converting from MNI152 space to fsLR surface or from fsLR to fsLR. Options
            are "linear" or "nearest".

        save_stat_maps: :obj:`bool`, default=False
            If True, saves the statistical map for each CAP for all groups as a Nifti1Image if ``output_dir`` is
            provided.

        fslr_giftis_dict: :obj:`dict` or :obj:`None`, default=None
            Dictionary specifying precomputed GifTI files in fsLR space for plotting statistical maps. This parameter
            should be used if the statistical CAP NIfTI files (can be obtained using ``self.caps2niftis()``) were
            converted to GifTI files using a tool such as Connectome Workbench. The dictionary structure is:

            ::

                {
                    "GroupName": {
                        "CAP-Name": {
                            "lh": "path/to/left_hemisphere_gifti",
                            "rh": "path/to/right_hemisphere_gifti"
                        }
                    }
                }

            "GroupName" can be "All Subjects" or any specific group name. CAP-Name is the name of the CAP. This
            parameter allows plotting without re-running the analysis. Initialize the CAP class and use this method
            if using this parameter.

        knn_dict: :obj:`dict[str, int | bool]`, default=None
            Use KNN (k-nearest neighbors) interpolation with reference atlas masking to fill in non-background
            coordinates that are assigned zero. Useful when custom parcellation does not project well from volumetric
            to surface space. The following subkeys are recognized:

            - "k": An integer (Default=1). Determines the number of nearest neighbors to consider.
            - "reference_atlas": A string (Default="Schaefer"). Specifies the atlas to use for reference masking ("AAL" or "Schaefer").
            - "resolution_mm": An integer (Default=1). Spatial resolution of the Schaefer parcellation (in millimeters) (1 or 2).
            - "remove_labels": A list or array (Default=None). The label IDs as integers of the regions in the parcellation to not interpolate.

            *Note*: This method is applied before the ``fwhm``.

        progress_bar: :obj:`bool`, default=False
            If True, displays a progress bar.

            .. versionadded:: 0.21.5

        **kwargs
            Additional parameters to pass to modify certain plot parameters. Options include:

            - dpi: :obj:`int`, default=300 -- Dots per inch for the plot.
            - title_pad: :obj:`int`, default=-3 -- Padding for the plot title.
            - cmap: :obj:`str` or :obj:`callable`, default="cold_hot" -- Colormap to be used for the plot.
            - cbar_kws: :obj:`dict`, default={"location": "bottom", "n_ticks": 3} -- Customize colorbar.  Refer to \
                ``_add_colorbars`` for ``surfplot.plotting.Plot`` in `Surfplot's Plot Documentation\
                <https://surfplot.readthedocs.io/en/latest/generated/surfplot.plotting.Plot.html#surfplot.plotting.Plot._add_colorbars>`_\
                for valid parameters.
            - alpha: :obj:`float`, default=1 -- Transparency level of the colorbar.
            - as_outline: :obj:`bool`, default=False -- Plots only an outline of contiguous vertices with the same value.
            - outline_alpha: :obj:`float`, default=1 -- Transparency level of the colorbar for outline if ``as_outline`` is True.
            - zero_transparent: :obj:`bool`, default=True -- Turns vertices with a value of 0 transparent.
            - size: :obj:`tuple`, default=(500, 400) -- Size of the plot in pixels.
            - layout: :obj:`str`, default="grid" -- Layout of the plot.
            - zoom: :obj:`float`, default=1.5 -- Zoom level for the plot.
            - views: {"lateral", "medial"} or :obj:`list[{"lateral", "medial}]`, default=["lateral", "medial"] -- \
                Views to be displayed in the plot.
            - brightness: :obj:`float`, default=0.5 -- Brightness level of the plot.
            - figsize: :obj:`tuple` or :obj:`None`, default=None -- Size of the figure.
            - scale: :obj:`tuple`, default=(2, 2) -- Scale factors for the plot.
            - surface: {"inflated", "veryinflated"}, default="inflated" -- The surface atlas that is used for plotting.\
                Options are "inflated" or "veryinflated".
            - color_range: :obj:`tuple` or :obj:`None`, default=None -- The minimum and maximum value to display in\
                plots. For instance, (-1, 1) where minimum value is first. If None, the minimum and maximum values from\
                the image will be used.
            - bbox_inches: :obj:`str` or :obj:`None`, default="tight" -- Alters size of the whitespace in the saved image.

        Returns
        -------
        self

        Note
        ----
        **Parcellation Approach**: ``parcel_approach`` must have the "maps" subkey containing the path to th
        NifTI file of the parcellation.

        **Assumptions**: This function assumes that the background label for the parcellation is zero and that is in
        MNI space. Additionally, the following approach is taken to map the each CAP onto the parcellation

        ::

            atlas = nib.load(atlas_file)
            atlas_fdata = atlas.get_fdata()
            # Create array of zeroes with same dimensions as atlas
            atlas_array = np.zeros_like(atlas_fdata)

            # Get array containing all labels in parcellation in order
            target_array = sorted(np.unique(atlas_fdata))

            # Start at 1 to avoid assigment to the background label
            for indx, value in enumerate(cap_vector, start=1):
                atlas_array[atlas_fdata == target_array[indx]] = value
        """
        if self._parcel_approach is None and fslr_giftis_dict is None:
            self._raise_error("_parcel_approach")

        if not hasattr(self, "_caps") and fslr_giftis_dict is None:
            self._raise_error("_caps")

        if suffix_filename is not None and output_dir is None:
            LG.warning("`suffix_filename` supplied but no `output_dir` specified. Files will not be saved.")

        # Check `knn_dict`
        if knn_dict:
            knn_dict = self._validate_knn_dict(knn_dict)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create plot dictionary
        plot_dict = _check_kwargs(_PlotDefaults.caps2surf(), **kwargs)

        groups = self._caps if hasattr(self, "_caps") and fslr_giftis_dict is None else fslr_giftis_dict

        if fslr_giftis_dict is None:
            parcellation_name = list(self._parcel_approach)[0]

        for group in groups:
            caps = self._caps[group] if hasattr(self, "_caps") and fslr_giftis_dict is None else fslr_giftis_dict[group]

            for cap in tqdm(caps, desc=f"Generating Surface Plots [GROUP: {group}]", disable=not progress_bar):
                if fslr_giftis_dict is None:
                    stat_map = _cap2statmap(
                        atlas_file=self._parcel_approach[parcellation_name]["maps"],
                        cap_vector=self._caps[group][cap],
                        fwhm=fwhm,
                        knn_dict=knn_dict,
                    )

                    # Fix for python 3.12, saving stat map so that it is path instead of a NifTi
                    try:
                        gii_lh, gii_rh = mni152_to_fslr(stat_map, method=method, fslr_density=fslr_density)
                    except TypeError:
                        # Create temp
                        temp_nifti = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
                        LG.warning(
                            "TypeError raised by neuromaps due to changes in pathlib.py in Python 3.12 "
                            "Converting NifTI image into a temporary nii.gz file (which will be "
                            f"automatically deleted afterwards) [TEMP FILE: {temp_nifti.name}]"
                        )

                        # Ensure file is closed
                        temp_nifti.close()
                        # Save temporary nifti to temp file
                        nib.save(stat_map, temp_nifti.name)
                        gii_lh, gii_rh = mni152_to_fslr(temp_nifti.name, method=method, fslr_density=fslr_density)
                        # Delete
                        os.unlink(temp_nifti.name)
                else:
                    gii_lh, gii_rh = fslr_to_fslr(
                        (fslr_giftis_dict[group][cap]["lh"], fslr_giftis_dict[group][cap]["rh"]),
                        target_density=fslr_density,
                        method=method,
                    )

                fig = self._generate_surface_plot(plot_dict, gii_lh, gii_rh, group, cap, suffix_title)

                if output_dir:
                    filename = self._basename(group, cap, "surface", suffix_filename, "png")

                    fig.savefig(
                        os.path.join(output_dir, filename), dpi=plot_dict["dpi"], bbox_inches=plot_dict["bbox_inches"]
                    )

                    # Save stat map
                    if save_stat_maps:
                        stat_map_name = filename.split("_surface")[0] + ".nii.gz"
                        nib.save(stat_map, os.path.join(output_dir, stat_map_name))

                try:
                    plt.show(fig) if show_figs else plt.close(fig)
                except:
                    plt.show() if show_figs else plt.close()

        return self

    @staticmethod
    def _generate_surface_plot(plot_dict, gii_lh, gii_rh, group, cap, suffix_title):
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

        # Check cmap
        cmap = _cmap_d[plot_dict["cmap"]] if isinstance(plot_dict["cmap"], str) else plot_dict["cmap"]
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
        fig = p.build(cbar_kws=plot_dict["cbar_kws"], figsize=plot_dict["figsize"], scale=plot_dict["scale"])
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
        **kwargs,
    ) -> Self:
        """
        Generate Radar Plots for CAPs using Cosine Similarity.

        Calculates the cosine similarity between the "High Amplitude" (positive/above the mean) and "Low Amplitude"
        (negative/below the mean) activations of the CAP cluster centroid and each a-priori region or network in a
        parcellation. This function assumes the mean for each ROI is 0 due to standardization.

        Cosine similarity is computed separately for high and low amplitudes by comparing them to the binary vector of
        the a-priori region, representing the region or network of interest. This provides a measure of how closely
        the CAP's positive and negative activation patterns align with each selected region.

        The process involves the following steps:

            1. Extract Cluster Centroids:

              - Each CAP is represented by a cluster centroid, which is a 1 x ROI (Region of Interest) vector.

            2. Generate Binary Vectors:

              - For each region create a binary vector (1 x ROI) where `1` indicates that the ROI is part of the\
              specific region and `0` otherwise.
              - In this example, the binary vector acts as a 1D mask to isolate ROIs in the Visual Network by setting\
              the corresponding indices to `1`.

                ::

                    import numpy as np

                    # Define nodes with their corresponding label IDs
                    nodes = ["LH_Vis1", "LH_Vis2", "LH_SomSot1", "LH_SomSot2",
                             "RH_Vis1", "RH_Vis2", "RH_SomSot1", "RH_SomSot2"]

                    # Binary mask for the Visual Network (Vis)
                    binary_vector = np.array([1, 1, 0, 0, 1, 1, 0, 0])

            3. Isolate Positive and Negative Activations in CAP Centroid:

              - Positive activations are defined as the values in the CAP centroid that are greater than zero.\
              These values represent the "High Amplitude" activations for that CAP.
              - Negative activations are defined as the values in the CAP centroid that are less than zero.\
              These values represent the "Low Amplitude" activations for that CAP.
              - To simplify the comparison between positive and negative activations using cosine similarity,\
              the negative activations are inverted (i.e., multiplied by -1). This inversion converts the negative\
              values into positive ones, allowing the cosine similarity calculation to return a positive value.\
              The positive similarity score represents how closely the a-priori region aligns with both the high and\
              low amplitude aspects of the CAP.

                ::

                  # Example cluster centroid for CAP 1
                  cap_1_cluster_centroid = np.array([-0.3, 1.5, 2.0, -0.2, 0.7, 1.3, -0.5, 0.4])

                  # Assign values less than 0 as 0 to isolate the high amplitude activations
                  high_amp = np.where(cap_1_cluster_centroid > 0, cap_1_cluster_centroid, 0)

                  # Assign values less than 0 as 0 to isolate the low amplitude activations; Also invert the sign
                  low_amp = high_amp = np.where(cap_1_cluster_centroid < 0, -cap_1_cluster_centroid, 0)

            4. Calculate Cosine Similarity:

              - Normalize the dot product by the product of the Euclidean norms of the cluster centroid and the binary\
              vector to obtain the cosine similarity:

                ::

                    # Compute dot product between the binary vector with the positive and negative activations
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

              - Each radar plot visualizes the cosine similarity for both "High Amplitude" (positive) and\
              "Low Amplitude" (negative) activations of the CAP. The cosine similarity values range from 0 to 1,\
              representing how closely the a-priori region aligns with the positive and negative activations\
              of the CAP centroid.

            Note, if groups were given when the ``CAP`` class was initialized, separate radar plots will be generated
            per CAP for all groups.

        Parameters
        ----------
        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save plots as png or html images. The directory will be created if it does not exist.
            If None, plots will not be saved.

        suffix_title: :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot.

        suffix_filename: :obj:`str` or :obj:`None`, default=None
            Appended to the filename of each saved plot if ``output_dir`` is provided.

        show_figs: :obj:`bool`, default=True
            Display figures. If the current Python session is non-interactive, then ``plotly.offline`` is used to
            generate an html file named "temp-plot.html", which opens each plot in the default browser.

        use_scatterpolar: :obj:`bool`, default=False
            Uses ``plotly.graph_objects.Scatterpolar`` instead of ``plotly.express.line_polar``.

        as_html: :obj:`bool`, default=False
            When ``output_dir`` is specified, plots are saved as html images instead of png images.

        **kwargs:
            Additional parameters to pass to modify certain plot parameters. Options include:

            - scale: :obj:`int`, default=2 -- If ``output_dir`` provided, controls resolution of image when saving.\
                Serves a similar purpose as dpi.
            - savefig_options: :obj:`dict[str]`, default={"width": 3, "height": 3, "scale": 1} -- If ``output_dir``\
                provided, controls the width (in inches), height (in inches), and scale of the plot.
            - height: :obj:`int`, default=800 -- Height of the plot.
            - width: :obj:`int`, defualt=1200 -- Width of the plot.
            - line_close: :obj:`bool`, default=True -- Whether to close the lines
            - bgcolor: :obj:`str`, default="white" -- Color of the background
            - scattersize: :obj:`int`, default=8 -- Controls size of the dots when markers are used.
            - connectgaps: :obj:`bool`, default=True -- If ``use_scatterpolar=True``, controls if missing values are connected.
            - linewidth: :obj:`int`, default = 2 -- The width of the line connecting the values if ``use_scatterpolar=True``.
            - opacity: :obj:`float`, default=0.5 -- If ``use_scatterpolar=True``, sets the opacity of the trace.
            - fill: :obj:`str`, default="none" -- If "toself" the are of the dots and within the boundaries of the\
                line will be filled.
            - mode: :obj:`str`, default="markers+lines" -- Determines how the trace is drawn. Can include "lines",\
                "markers", "lines+markers", "lines+markers+text".
            - radialaxis: :obj:`dict`, default={"showline": False, "linewidth": 2, \
                                                "linecolor": "rgba(0, 0, 0, 0.25)", \
                                                "gridcolor": "rgba(0, 0, 0, 0.25)", \
                                                "ticks": "outside", "tickfont": {"size": 14, "color": "black"}} --\
                Customizes the radial axis. Refer to `Plotly's radialaxis Documentation\
                <https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.polar.radialaxis.html>`_\
                or `Plotly's polar Documentation <https://plotly.com/python/reference/layout/polar/>`_ for valid keys.
            - angularaxis: :obj:`dict`, default={"showline": True, "linewidth": 2, \
                                                 "linecolor": "rgba(0, 0, 0, 0.25)", \
                                                 "gridcolor": "rgba(0, 0, 0, 0.25)", \
                                                 "tickfont": {"size": 16, "color": "black"}} --\
                Customizes the angular axis. Refer to `Plotly's angularaxis Documentation\
                <https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.polar.angularaxis.html>`_\
                or `Plotly's polar Documentation <https://plotly.com/python/reference/layout/polar/>`_ for valid keys.
            - color_discrete_map: :obj:`dict`, default={"High Amplitude": "red", "Low Amplitude": "blue"} --\
                Change the color of the "High Amplitude" and "Low Amplitude" groups. Must use the keys "High Amplitude"\
                and "Low Amplitude".
            - title_font: :obj:`dict`, default={"family": "Times New Roman", "size": 30, "color": "black"} --\
                Modifies the font of the title. Refer to `Plotly's layout Documentation\
                <https://plotly.com/python/reference/layout/>`_ for valid keys.
            - title_x: :obj:`float`, default=0.5 -- Modifies x position of title.
            - title_y: :obj:`float`, default=None -- Modifies y position of title.
            - legend: :obj:`dict`, default={"yanchor": "top", "xanchor": "left", "y": 0.99, "x": 0.01, \
                                            "title_font_family": "Times New Roman", "font": {"size": 12, \
                                            "color": "black"}} --\
                Customizes the legend. Refer to `Plotly's layout Documentation\
                <https://plotly.com/python/reference/layout/>`_ for valid keys
            - engine: {"kaleido", "orca"}, default="kaleido" -- Engine used for saving plots.

        Returns
        -------
        self

        Note
        -----
        **Handling Division by Zero:** NumPy automatically handles division by zero errors. This may occur if the
        network or the "High Amplitude" or "Low Amplitude" vectors are all zeroes. In such cases, NumPy assigns `NaN`
        to the cosine similarity for the affected network(s), indicating that the similarity is undefined.
        Plotly is capable of handling `NaN` values.

        **Parcellation Approach**: If using "Custom" for ``parcel_approach`` the "regions" subkey is required.

        **Saving Plots**: By default, this function uses "kaleido" (which is also a dependency in this package)
        to save plots. For other engines such as "orca", those packages must be installed seperately.

        **Tick Values**: if the ``tickvals`` or  ``range`` subkeys in this code are not specified in the ``radialaxis``
        kwarg, then four values are shown - 0.25*(max value), 0.50*(max value), 0.75*(max value), and the max value.
        These values are also rounded to the second decimal place.

        References
        ----------
        Zhang, R., Yan, W., Manza, P., Shokri-Kojori, E., Demiral, S. B., Schwandt, M., Vines, L., Sotelo, D., Tomasi,
        D., Giddens, N. T., Wang, G., Diazgranados, N., Momenan, R., & Volkow, N. D. (2023). Disrupted brain state
        dynamics in opioid and alcohol use disorder: attenuation by nicotine use. Neuropsychopharmacology, 49(5),
        876–884. https://doi.org/10.1038/s41386-023-01750-w

        Ingwersen, T., Mayer, C., Petersen, M., Frey, B. M., Fiehler, J., Hanning, U., Kühn, S., Gallinat, J.,
        Twerenbold, R., Gerloff, C., Cheng, B., Thomalla, G., & Schlemm, E. (2024). Functional MRI brain state
        occupancy in the presence of cerebral small vessel disease — A pre-registered replication analysis of the
        Hamburg City Health Study. Imaging Neuroscience, 2, 1–17. https://doi.org/10.1162/imag_a_00122
        """
        if self._parcel_approach is None:
            self._raise_error("_parcel_approach")

        if not hasattr(self, "_caps"):
            self._raise_error("_caps")

        if not self._standardize:
            LG.warning(
                "To better aid interpretation, the matrix subjected to kmeans clustering in "
                "`self.get_caps()` should be standardized so that each ROI in the CAP cluster centroid. "
                "represents activation or de-activation relative to the mean."
            )

        if suffix_filename is not None and output_dir is None:
            LG.warning("`suffix_filename` supplied but no `output_dir` specified. Files will not be saved.")

        # Create plot dictionary
        plot_dict = _check_kwargs(_PlotDefaults.caps2radar(), **kwargs)

        # Initialize cosine_similarity attribute
        self._cosine_similarity = {}

        parcellation_name = list(self._parcel_approach)[0]
        # Create radar dict
        for group in self._groups:
            radar_dict = {"Regions": list(self._parcel_approach[parcellation_name]["regions"])}
            self._update_radar_dict(group, parcellation_name, radar_dict)
            self._cosine_similarity[group] = radar_dict

            for cap in self._caps[group]:
                if use_scatterpolar:
                    # Create dataframe
                    df = pd.DataFrame({"Regions": radar_dict["Regions"]})
                    df = pd.concat([df, pd.DataFrame(radar_dict[cap])], axis=1)
                    regions = df["Regions"].values

                    # Initialize figure
                    fig = go.Figure(layout=go.Layout(width=plot_dict["width"], height=plot_dict["height"]))

                    for i in ["High Amplitude", "Low Amplitude"]:
                        values = df[i].values

                        # Add traces
                        fig.add_trace(
                            go.Scatterpolar(
                                r=list(values),
                                theta=regions,
                                connectgaps=plot_dict["connectgaps"],
                                name=i,
                                opacity=plot_dict["opacity"],
                                marker=dict(color=plot_dict["color_discrete_map"][i], size=plot_dict["scattersize"]),
                                line=dict(color=plot_dict["color_discrete_map"][i], width=plot_dict["linewidth"]),
                            )
                        )
                else:
                    n = len(radar_dict["Regions"])
                    # Create dataframe
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
                        fill=plot_dict["fill"], mode=plot_dict["mode"], marker=dict(size=plot_dict["scattersize"])
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

                # Customize
                fig.update_layout(
                    title=dict(text=title_text, font=plot_dict["title_font"]),
                    title_x=plot_dict["title_x"],
                    title_y=plot_dict["title_y"],
                    showlegend=True if plot_dict["legend"] else False,
                    legend=plot_dict["legend"],
                    legend_title_text="Cosine Similarity",
                    polar=dict(
                        bgcolor=plot_dict["bgcolor"],
                        radialaxis=plot_dict["radialaxis"],
                        angularaxis=plot_dict["angularaxis"],
                    ),
                )

                if show_figs:
                    if bool(getattr(sys, "ps1", sys.flags.interactive)):
                        fig.show()
                    else:
                        pyo.plot(fig, auto_open=True)

                if output_dir:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    filename = self._basename(group, cap, "radar", suffix_filename, "png")
                    if not as_html:
                        fig.write_image(
                            os.path.join(output_dir, filename), scale=plot_dict["scale"], engine=plot_dict["engine"]
                        )
                    else:
                        filename = filename.replace(".png", ".html")
                        fig.write_html(os.path.join(output_dir, filename))

        return self

    def _update_radar_dict(self, group, parcellation_name, radar_dict):
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

                # Store value in dict
                radar_dict[cap]["High Amplitude"].append(high_amp_cosine)
                radar_dict[cap]["Low Amplitude"].append(low_amp_cosine)

    def _create_region_mask_1d(self, parcellation_name, region, cap_vector):
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
    def _compute_cosine_similarity(amp_vector, region_mask):
        dot_product = np.dot(amp_vector, region_mask)
        norm_region_mask = np.linalg.norm(region_mask)
        norm_amp_vector = np.linalg.norm(amp_vector)
        cosine_similarity = dot_product / (norm_amp_vector * norm_region_mask)

        return cosine_similarity
