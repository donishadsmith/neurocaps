import collections, copy, itertools, os, re, sys, tempfile
from typing import Union, Literal, Optional

import numpy as np, nibabel as nib, matplotlib.pyplot as plt, pandas as pd, seaborn, surfplot
import plotly.express as px, plotly.graph_objects as go, plotly.offline as pyo
from kneed import KneeLocator
from joblib import Parallel, delayed
from nilearn.plotting.cm import _cmap_d
from neuromaps.transforms import mni152_to_fslr, fslr_to_fslr
from neuromaps.datasets import fetch_fslr
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

from .._utils import (_CAPGetter, _cap2statmap, _check_kwargs, _create_display, _convert_pickle_to_dict,
                      _check_parcel_approach, _logger, _run_kmeans, _save_contents)

LG = _logger(__name__)

class CAP(_CAPGetter):
    """
    **Co-Activation Patterns (CAPs) Class**

    Initializes the CAPs (Co-activation Patterns) class.

    Parameters
    ----------
    parcel_approach : :obj:`dict[str, dict[str, os.PathLike | list[str]]]`, :obj:`dict[str, dict[str, str | int]]`, or :obj:`os.PathLike`, default=None
        The approach used to parcellate BOLD images. Similar to ``TimeseriesExtractor``, "Schaefer" and "AAL"
        can be initialized here to create the appropriate ``parcel_approach`` that includes the sub-keys
        "maps", "nodes", and "regions", which are needed for plotting.

        - For "Schaefer", available sub-keys include "n_rois", "yeo_networks", and "resolution_mm". Refer to documentation for ``nilearn.datasets.fetch_atlas_schaefer_2018`` for valid inputs.
        - For "AAL", the only sub-key is "version". Refer to documentation for ``nilearn.datasets.fetch_atlas_aal`` for valid inputs.
        - For "Custom", the sub-keys should include:

            - "maps" : Directory path to the location of the parcellation file.
            - "nodes" : A list of node names in the order of the label IDs in the parcellation.
            - "regions" : The regions or networks in the parcellation.

        If the "Schaefer" or "AAL" option was used in the ``TimeSeriesExtractor`` class, you can initialize
        the ``TimeSeriesExtractor`` class with the ``parcel_approach`` that was initially used, then set this
        parameter to ``TimeSeriesExtractor.parcel_approach``. For this parameter, only "Schaefer", "AAL", and
        "Custom" are supported. Note, this parameter is not needed for using ``self.get_caps()``; however, for
        certain plotting functions it will be needed. This class contains a ``parcel_approach`` property that also acts
        as a setter so ``self.parcel_approach=parcel_approach`` can be used to set the ``parcel_approach`` later on.

    groups : :obj:`dict[str, list[str]]` or :obj:`None`, default=None
        A mapping of group names to subject IDs. Each group contains subject IDs for separate CAP analysis.
        Additionally, if duplicate IDs are detected within or across groups, a warning is issued and only the first
        instance is retained. This parameter is used to create a dictionary (``self.subject_table``), which pairs each
        subject ID (keys) with their group name (values) and is used for concatenating data and calculating metrics.
        This is done to avoid issues with duplicate IDs. If ``groups`` left as None, CAPs are not separated by group
        and are performed on all subjects. The structure should be as follows:
        ::

            {
                "GroupName1": ["1", "2", "3"],
                "GroupName2": ["4", "5", "6"],
            }


    Property
    --------
    n_clusters : :obj:`int` or :obj:`list[int]`
        A single integer or list of integers if ``cluster_selection_method`` is not None) that will used for
        ``sklearn.cluster.KMeans``. Is None until ``self.get_caps()`` is used.

    groups : :obj:`dict[str, list[str]]` or :obj:`None`:
        A mapping of groups names to subject IDs.

    cluster_selection_method : :obj:`str` or :obj:`None`:
        The cluster selection method to identify the optimal number of clusters. Is None until ``self.get_caps()``
        is used.

    parcel_approach : :obj:`dict[str, dict[str, os.PathLike | list[str]]]`
        Nested dictionary containing information about the parcellation. Can also be used as a setter, which accepts
        a dictionary or a dictionary saved as a pickle file. If "Schaefer" or "AAL" was specified during initialization
        of the ``TimeseriesExtractor`` class, then ``nilearn.datasets.fetch_atlas_schaefer_2018`` and
        ``nilearn.datasets.fetch_atlas_aal`` will be used to obtain the "maps" and the "nodes". Then string splitting
        is used on the "nodes" to obtain the "regions":
        ::

            {
                "Schaefer":
                {
                    "maps": "path/to/parcellation.nii.gz",
                    "nodes": ["LH_Vis1", "LH_SomSot1", "RH_Vis1", "RH_SomSot1"],
                    "regions": ["Vis", "SomSot"]
                }
            }


            {
                "AAL":
                {
                    "maps": "path/to/parcellation.nii.gz",
                    "nodes": ["Precentral_L", "Precentral_R", "Frontal_Sup_L", "Frontal_Sup_R"],
                    "regions": ["Precentral", "Frontal"]
                }
            }

        If "Custom" is specified, only checks are done to ensure that the dictionary contains the proper sub-keys
        such as "maps", "nodes", and "regions". Unlike "Schaefer" and "AAL", "regions" must be
        a nested dictionary specifying the name of the region as the first level key and the indices in the "nodes"
        list belonging to the "lh" and "rh" for that region. Refer to the example for "Custom" in
        the Note section below.

    n_cores : :obj:`int`
        Number of cores to use for multiprocessing with joblib. Is None until ``self.get_caps()`` is used
        and ``n_cores`` is specified.

    runs : :obj:`int` or :obj:`list[int]`
        The runs used for the CAPs analysis. Is None until ``self.get_caps()`` is used and ``runs`` is specified.

    caps : :obj:`dict[str, dict[str, np.array]]`
        The extracted cluster centroids, representing each CAP from the k-means model. It is a nested dictionary
        containing the group name, CAP names, and 1D numpy array. Is None until ``self.get_caps``
        is used. The structure is as follows:
        ::

            {
                "GroupName": {
                    "CAP-1": np.array([...]), # 1 x ROI array
                    "CAP-2": np.array([...]), # 1 x ROI array
                }

            }

    kmeans : :obj:`dict[str, sklearn.cluster.KMeans]`
        Dictionary containing the ``sklearn.cluster.KMeans`` model used for each group. If ``cluster_selection__method``
        is not None, the ``sklearn.cluster.KMeans`` model will be the optimal model.  Is None until
        ``self.get_caps()`` is used. The structure is as follows:
        ::

            {
                "GroupName": sklearn.cluster.KMeans,
            }

    davies_bouldin : :obj:`dict[str, dict[str, float]]`
        If ``cluster_selection_method`` is "davies_bouldin", this property will be a nested dictionary containing the
        group name, cluster number, and davies_bouldin scores. Is None until ``self.get_caps()`` is used.
        The structure is as follows:
        ::

            {
                "GroupName": {
                    "2": float,
                    "3": float,
                    "4": float,
                }
            }

    inertia : :obj:`dict[str, dict[str, float]]`
        If ``cluster_selection_method`` is "elbow", this property will be a nested dictionary containing the
        group name, cluster number, and inertia values. Is None until ``self.get_caps()`` is used. The structure is as
        follows:
        ::

            {
                "GroupName": {
                    "2": float,
                    "3": float,
                    "4": float,
                }
            }

    silhouette_scores : :obj:`dict[str, dict[str, float]]`
        If ``cluster_selection_method`` is "silhouette", this property will be a nested dictionary containing
        the group name, cluster number, and silhouette scores. Is None until ``self.get_caps()`` is used. The
        structure is as follows:
        ::

            {
                "GroupName": {
                    "2": float,
                    "3": float,
                    "4": float,
                }
            }

    variance_ratio : :obj:`dict[str, dict[str, float]]`
        If ``cluster_selection_method`` is "variance_ratio", this property will be a nested dictionary containing
        the group name, cluster number, and variance ratio scores. Is None until ``self.get_caps()`` is used. The
        structure is as follows:
        ::

            {
                "GroupName": {
                    "2": float,
                    "3": float,
                    "4": float,
                }
            }

    optimal_n_clusters : :obj:`dict[str, dict[int]]`
        If ``cluster_selection_method`` is not None, this property is a nested dictionary containing the group
        name and the optimal number of clusters. The structure is as follows:
        ::

            {
                "GroupName": int,
            }

    standardize : :obj:`bool`
        Boolean denoting whether the features of the concatenated timeseries data was z-scored. Is None until
        ``self.get_caps()`` is used.

    means : :obj:`dict[str, np.array]`
        If ``standardize`` is True in ``self.get_caps()``, this property is a nested dictionary containing the
        group names and a numpy array (participants x TR) x ROIs of the means of the features. The structure is as
        follows:
        ::

            {
                "GroupName": np.array([...]), # Dimensions: 1 x ROIs
            }

    stdev : :obj:`dict[str, np.array]`
        If ``standardize`` is True in ``self.get_caps()``, this property is a nested dictionary containing the
        group names and a numpy array (participants x TR) x ROIs of the sample standard deviation of the features.
        The structure is as follows:
        ::

            {
                "GroupName": np.array([...]), # Dimensions: 1 x ROIs
            }

    concatenated_timeseries : :obj:`dict[str, np.array]`
        Nested dictionary containing the group name and their respective concatenated numpy arrays
        (participants x TR) x ROIs. Is None until ``self.get_caps()`` is used. If this property needs to be deleted due
        to space issues, ``delattr(self,"_concatenated_timeseries")`` can be used to delete the array.
        The structure is as follows:
        ::

            {
                "GroupName": np.array([...]), # Dimensions: (participants x TR) x ROIs
            }

    region_caps : :obj:`dict[str, np.array]`
        If ``visual_scope`` set to "regions" in ``self.caps2plot()``, this property is a nested dictionary
        containing the group name, CAP names, and numpy array (1 x region) of the averaged z-score value for each
        region. The structure is as follows:
        ::

            {
                "GroupName": {
                    "CAP-1": np.array([...]), # 1 x region array
                    "CAP-2": np.array([...]), # 1 x region array
                }

            }

    outer_products : :obj:`dict[str, dict[str, np.array]]`
        If ``plot_options`` set to "outer_product" ``self.caps2plot()``, this property is a nested dictionary
        containing the group name, CAP names, and numpy array (ROI x ROI) of the outer product. The structure is
        as follows:
        ::

            {
                "GroupName": {
                    "CAP-1": np.array([...]), # ROI x ROI array
                    "CAP-2": np.array([...]), # ROI x ROI array
                }

            }

    subject_table : :obj:`dict[str, str]`
        A dictionary generated when ``self.get_caps()`` is used. Operates as a lookup table that pairs each subject ID
        with the associated group. Also can be used as a setter. The structure is as follows.
        ::

            {
                "Subject-ID": "GroupName",
                "Subject-ID": "GroupName",
            }

    cosine_similarity : :obj: `dict[str, dict[list]]`
        Nested dictionary that is generates when ``self.caps2radar`` is used. Each group contains the "regions" key,
        containing a list of region, these are the regions that the nodes used to generate the binary vector are from.
        This key is followed by the cap names, which consist of a list of cosine similarities, where the index of the
        cosine similarity corresponds to the index of the region in the "regions" key. The structure is as followd:
        ::

            {
                "GroupName": {
                    "regions": [...],
                    "CAP-1": [...], # Length of list corresponds to length of "regions" list
                    "CAP-2": [...], # Length of list corresponds to length of "regions" list
                }

            }

    Note
    ----
    **If no groups were specified, the default group name will be "All Subjects".**

    **If using a "Custom" parcellation approach**, ensure that the atlas is lateralized (where each region/network has
    nodes in the left and right hemisphere). Certain visualization functions in this class assume that the background
    label is "0". Do not add a background label in the "nodes" or "regions" key; the zero index should correspond to
    the first ID that is not "0".

    - "maps": Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g.,
      .nii for NIfTI files).
    - "nodes": List of all node labels used in your study, arranged in the exact order they correspond to indices in
      your parcellation files. Each label should match the parcellation index it represents. For example, if the
      parcellation label "1" corresponds to the left hemisphere visual cortex area 1, then "LH_Vis1" should occupy
      the 0th index in this list. This ensures that data extraction and analysis accurately reflect the anatomical
      regions intended.
    - "regions": Dictionary defining major brain regions. Each region should list node indices under
      "lh" and "rh" to specify left and right hemisphere nodes.

    **Different sub-keys are required depending on the function used. Refer to the Note section under each function
    for information regarding the sub-keys required for that function.**

    The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis)
    and hippocampus regions:
    ::

        parcel_approach = {
            "Custom": {
                "maps": "/location/to/parcellation.nii.gz",
                "nodes": [
                    "LH_Vis1",
                    "LH_Vis2",
                    "LH_Hippocampus",
                    "RH_Vis1",
                    "RH_Vis2",
                    "RH_Hippocampus"
                ],
                "regions": {
                    "Vis": {
                        "lh": [0, 1],
                        "rh": [3, 4]
                    },
                    "Hippocampus": {
                        "lh": [2],
                        "rh": [5]
                    }
                }
            }
        }
    """
    def __init__(self,
                 parcel_approach: Union[
                     dict[str, dict[str, Union[os.PathLike, list[str]]]],
                     dict[str, dict[str, Union[str,int]]],
                     os.PathLike
                     ]=None,
                 groups: dict[str, list[str]]=None) -> None:
        self._groups = groups
        # Raise error if self groups is not a dictionary
        if self._groups:
            if not isinstance(self._groups, dict):
                raise TypeError("`groups` must be a dictionary where the keys are the group names and the items "
                                "correspond to subject ids in the groups.")

            for group_name in self._groups:
                assert self._groups[group_name], f"{group_name} has zero subject ids."

            # Convert ids to strings
            for group in set(self._groups):
                self._groups[group] = [str(subj_id) if not isinstance(subj_id, str)
                                       else subj_id for subj_id in self._groups[group]]

        if parcel_approach is not None:
            parcel_approach = _check_parcel_approach(parcel_approach=parcel_approach, call="CAP")

        self._parcel_approach = parcel_approach

    def get_caps(self,
                 subject_timeseries: Union[dict[str, dict[str, np.ndarray]], os.PathLike],
                 runs: Optional[Union[int, str, list[int], list[str]]]=None,
                 n_clusters: Union[int, list[int]]=5,
                 cluster_selection_method: Literal["elbow", "davies_bouldin", "silhouette", "variance_ratio"]=None,
                 random_state: Optional[int]=None,
                 init: Union[np.array, Literal["k-means++", "random"]]="k-means++",
                 n_init: Union[Literal["auto"],int]="auto",
                 max_iter: int=300, tol: float=0.0001, algorithm: Literal["lloyd", "elkan"]="lloyd",
                 standardize: bool=True,
                 n_cores: Optional[int]=None,
                 show_figs: bool=False,
                 output_dir: Optional[os.PathLike]=None,
                 **kwargs) -> plt.figure:
        """
        **Perform K-Means Clustering to Identify CAPs**

        Concatenates the timeseries of each subject into a single numpy array with dimensions
        (participants x TRs) x ROI and uses ``sklearn.cluster.KMeans`` on the concatenated data. **Note**,
        ``KMeans`` uses euclidean distance. Additionally, the Elbow method is determined using ``KneeLocator`` from
        the kneed package and the Davies Bouldin, Silhouette, and Variance Ratio methods are calculated using
        scikit-learn's ``davies_bouldin_score``, ``silhouette_score``, and ``calinski_harabasz_score`` functions,
        respectively.

        Parameters
        ----------
        subject_timeseries : :obj:`dict[str, dict[str, np.ndarray]]` or :obj:`os.PathLike`
            Path of the pickle file containing the nested subject timeseries dictionary saved by the
            ``TimeSeriesExtractor`` class or the nested subject timeseries dictionary produced by the
            ``TimeseriesExtractor`` class. The first level of the nested dictionary must consist of the subject ID
            as a string, the second level must consist of the run numbers in the form of "run-#" (where # is the
            corresponding number of the run), and the last level must consist of the timeseries (as a numpy array)
            associated with that run. The structure is as follows:
            ::

                subject_timeseries = {
                        "101": {
                            "run-0": np.array([...]), # 2D array
                            "run-1": np.array([...]), # 2D array
                            "run-2": np.array([...]), # 2D array
                        },
                        "102": {
                            "run-0": np.array([...]), # 2D array
                            "run-1": np.array([...]), # 2D array
                        }
                    }

        runs : :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            The run numbers to perform the CAPs analysis with. If None, all runs in the subject timeseries will be
            concatenated into a single dataframe and subjected to k-means clustering.

        n_clusters : :obj:`int | list[int]`, default=5
            The number of clusters to use for ``sklearn.cluster.KMeans``. Can be a single integer or a list of
            integers (if ``cluster_selection_method`` is not None).

        cluster_selection_method : {"elbow", "davies_bouldin", "silhouette", "variance_ratio"} or :obj:`None`, default=None
            Method to find the optimal number of clusters. Options are "elbow", "davies_bouldin", "silhouette", and
            "variance_ratio".

        random_state : :obj:`int` or :obj:`None`, default=None
            The random state to use for ``sklearn.cluster.KMeans``. Ensures reproducible results.

        init : {"k-means++","random"}, or :class:`np.ndarray`, default="k-means++"
            Method for choosing initial cluster centroid for ``sklearn.cluster.KMeans``. Options are "k-means++",
            "random", or np.array.

        n_init : {"auto"} or :obj:`int`, default="auto"
            Number of times ``sklearn.cluster.KMeans`` is ran with different initial clusters.
            The model with lowest inertia from these runs will be selected.

        max_iter : :obj:`int`, default=300
            Maximum number of iterations for a single run of ``sklearn.cluster.KMeans``.

        tol : :obj:`float`, default=1e-4,
            Stopping criterion for ``sklearn.cluster.KMeans``if the change in inertia is below this value, assuming
            ``max_iter`` has not been reached.

        algorithm : {"lloyd", "elkan"}, default="lloyd"
            The type of algorithm to use for ``sklearn.cluster.KMeans``. Options are "lloyd" and "elkan".

        standardize : :obj:`bool`, default=True
            Whether to z-score the columns/ROIs of the concatenated timeseries data. The sample standard deviation will
            be used, meaning Bessel's correction, `n-1`, will be used in the denominator.

        n_cores : :obj:`int` or :obj:`None`, default=None
            The number of CPU cores to use for multiprocessing, with joblib, to run multiple ``sklearn.cluster.KMeans``
            models if ``cluster_selection_method`` is not None.

        show_figs : :obj:`bool`, default=False
            Display the plots of inertia or silhouette scores for all groups if ``cluster_selection_method`` is not
            None.

        output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
            Directory to save plot to if ``cluster_selection_method`` is set to "elbow". The directory will be
            created if it does not exist. Outputs as png file.

        kwargs : :obj:`dict`
            Dictionary to adjust certain parameters related to ``cluster_selection_method`` when set to "elbow".
            Additional parameters include:

            - S : :obj:`int`, default=1
                Adjusts the sensitivity of finding the elbow. Larger values are more conservative and less
                sensitive to small fluctuations. This package uses ``KneeLocator`` from the kneed package to
                identify the elbow. Default is 1.
            - dpi : :obj:`int`, default=300
                Adjusts the dpi of the elbow plot. Default is 300.
            - figsize : :obj:`tuple`, default=(8,6)
                Adjusts the size of the elbow plots.
            - bbox_inches : :obj:`str` or :obj:`None`, default="tight"
                Alters size of the whitespace in the saved image.
            - step : :obj:`int`, default=None
                An integer value that controls the progression of the x-axis in plots for the silhouette or elbow
                method. When set, only integer values will be displayed on the x-axis.

        Returns
        -------
        `matplotlib.Figure`
            An instance of `matplotlib.Figure`.
        """
        self._n_cores = n_cores
        # Ensure all unique values if n_clusters is a list
        self._n_clusters = n_clusters if isinstance(n_clusters, int) else sorted(list(set(n_clusters)))
        self._cluster_selection_method = cluster_selection_method

        if isinstance(n_clusters, list):
            self._n_clusters =  self._n_clusters[0] if all([isinstance(self._n_clusters, list),
                                                            len(self._n_clusters) == 1]) else self._n_clusters
            # Raise error if n_clusters is a list and no cluster selection method is specified
            if all([len(n_clusters) > 1, self._cluster_selection_method is None]):
                raise ValueError("`cluster_selection_method` cannot be None since `n_clusters` is a list.")

        # Raise error if silhouette_method is requested when n_clusters is an integer
        if all([self._cluster_selection_method is not None, isinstance(self._n_clusters, int)]):
            raise ValueError("`cluster_selection_method` only valid if `n_clusters` is a range of unique integers.")

        if self._n_cores and self._cluster_selection_method is None:
            raise ValueError("Parallel processing will not run since `cluster_selection_method` is None.")

        if runs and not isinstance(runs, list): runs = [runs]

        self._runs = runs
        self._standardize = standardize

        subject_timeseries = self._process_subject_timeseries(subject_timeseries)

        self._concatenated_timeseries = self._concatenate_timeseries(subject_timeseries, runs)

        valid_methods = ["elbow", "davies_bouldin", "silhouette", "variance_ratio"]

        if self._cluster_selection_method is not None:
            if self._cluster_selection_method not in valid_methods:
                formatted_string = ', '.join(["'{a}'".format(a=x) for x in valid_methods])
                raise ValueError(f"Options for `cluster_selection_method` are - {formatted_string}.")
            else:
                self._select_optimal_clusters(random_state=random_state, init=init, n_init=n_init, max_iter=max_iter,
                                              tol=tol, algorithm=algorithm, show_figs=show_figs, output_dir=output_dir,
                                              **kwargs)
        else:
            self._kmeans = {}
            for group in self._groups:
                self._kmeans[group] = {}
                self._kmeans[group] = KMeans(n_clusters=self._n_clusters, random_state=random_state, init=init,
                                             n_init=n_init, max_iter=max_iter, tol=tol,
                                             algorithm=algorithm).fit(self._concatenated_timeseries[group])

        # Create states dict
        self._create_caps_dict()

    @staticmethod
    def _process_subject_timeseries(subject_timeseries):
        if isinstance(subject_timeseries, str) and subject_timeseries.endswith(".pkl"):
            subject_timeseries = _convert_pickle_to_dict(pickle_file=subject_timeseries)
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
                    LG.warning(f"[SUBJECT: {subj_id}] Appears more than once. Only the first instance of this "
                               "subject will be included in the analysis.")
                else:
                    self._subject_table.update({subj_id : group})

    def _concatenate_timeseries(self, subject_timeseries, runs):
        # Create dictionary for "All Subjects" if no groups are specified to reuse the same loop instead of having to
        # create logic for grouped and non-grouped version of the same code
        if not self._groups: self._groups = {"All Subjects": [subject for subject in subject_timeseries]}

        concatenated_timeseries = {group: None for group in self._groups}

        self._generate_lookup_table()

        self._mean_vec = {group: None for group in self._groups}
        self._stdev_vec = {group: None for group in self._groups}

        for subj_id, group in self._subject_table.items():
            requested_runs = [f"run-{run}" for run in runs] if runs else list(subject_timeseries[subj_id])
            subject_runs = [subject_run for subject_run in subject_timeseries[subj_id]
                            if subject_run in requested_runs]
            if not subject_runs:
                LG.warning(f"[SUBJECT: {subj_id}] Does not have the requested run numbers: "
                           f"{','.join(requested_runs)}.")
                continue
            for curr_run in subject_runs:
                if concatenated_timeseries[group] is None:
                    concatenated_timeseries[group] = subject_timeseries[subj_id][curr_run]
                else:
                    concatenated_timeseries[group] = np.vstack([concatenated_timeseries[group],
                                                                subject_timeseries[subj_id][curr_run]])
        # Standardize
        if self._standardize: concatenated_timeseries = self._scale(concatenated_timeseries)

        return concatenated_timeseries

    def _scale(self, concatenated_timeseries):
        for group in self._groups:
            self._mean_vec[group] = np.mean(concatenated_timeseries[group], axis=0)
            self._stdev_vec[group] = np.std(concatenated_timeseries[group], ddof=1, axis=0)
            eps = np.finfo(self._stdev_vec[group].dtype).eps
            # Taken from nilearn pipeline, used for numerical stability purposes to avoid numpy division error
            self._stdev_vec[group][self._stdev_vec[group] < eps] = 1.0
            diff = concatenated_timeseries[group] - self._mean_vec[group]
            concatenated_timeseries[group] = diff/self._stdev_vec[group]

        return concatenated_timeseries

    def _select_optimal_clusters(self, random_state, init, n_init, max_iter, tol, algorithm,
                                 show_figs, output_dir, **kwargs):
        # Initialize attributes
        self._davies_bouldin = {}
        self._inertia = {}
        self._silhouette_scores = {}
        self._variance_ratio = {}
        self._optimal_n_clusters = {}
        self._kmeans = {}
        self._cluster_metric = {}
        performance_dict = {}

        method = self._cluster_selection_method

        # Defaults
        defaults = {"dpi": 300, "figsize": (8,6), "step": None, "bbox_inches": "tight"}
        plot_dict = _check_kwargs(defaults, **kwargs)

        for group in self._groups:
            performance_dict[group] = {}
            # Don't want to store model dict for all groups; re-initialize for each group
            model_dict = {}
            if self._n_cores is None:
                for n_cluster in self._n_clusters:
                    output_score, model = _run_kmeans(n_cluster=n_cluster, random_state=random_state, init=init,
                                                      n_init=n_init, max_iter=max_iter, tol=tol, algorithm=algorithm,
                                                      concatenated_timeseries=self._concatenated_timeseries[group],
                                                      method=method)
                    performance_dict[group].update(output_score)
                    model_dict.update(model)
            else:
                parallel = Parallel(return_as="generator", n_jobs=self._n_cores)
                output = parallel(delayed(_run_kmeans)(n_cluster, random_state, init, n_init, max_iter, tol,
                                                       algorithm, self._concatenated_timeseries[group],
                                                       method) for n_cluster in self._n_clusters)

                output_scores, models = zip(*output)
                for output in output_scores: performance_dict[group].update(output)
                for model in models: model_dict.update(model)

            # Select optimal clusters
            if method == "elbow":
                knee_dict = {"S": kwargs["S"] if "S" in kwargs else 1}
                kneedle = KneeLocator(x=list(performance_dict[group]),
                                      y=list(performance_dict[group].values()),
                                      curve="convex", direction="decreasing", S=knee_dict["S"])
                self._optimal_n_clusters[group] = kneedle.elbow
                if self._optimal_n_clusters[group] is None:
                    raise ValueError(f"[GROUP: {group}] - No elbow detected. Try adjusting the sensitivity parameter, "
                                     "`S`, to increase or decrease sensitivity (higher values are less sensitive), "
                                     "expanding the list of `n_clusters` to test, or using another "
                                     "`cluster_selection_method`.")
            elif method == "davies_bouldin":
                # Get minimum for davies bouldin
                self._optimal_n_clusters[group] = min(performance_dict[group],
                                                      key=performance_dict[group].get)
            else:
                # Get max for silhouette and variance ratio
                self._optimal_n_clusters[group] = max(performance_dict[group],
                                                      key=performance_dict[group].get)

            # Get the optimal kmeans model
            self._kmeans[group] = model_dict[self._optimal_n_clusters[group]]

            LG.info(f"[GROUP: {group} | METHOD: {method}] Optimal cluster size is {self._optimal_n_clusters[group]}.")

            # Plot
            if show_figs or output_dir is not None:
                self._plot_method(method, performance_dict, group, plot_dict, show_figs, output_dir)

        if method == "elbow": self._inertia = performance_dict
        elif method == "davies_bouldin": self._davies_bouldin = performance_dict
        elif method == "silhouette": self._silhouette_scores = performance_dict
        else: self._variance_ratio = performance_dict

    def _plot_method(self, method, performance_dict, group, plot_dict, show_figs, output_dir):
        y_titles = {"elbow": "Inertia", "davies_bouldin": "Davies Bouldin Score", "silhouette": "Silhouette Score",
                    "variance_ratio": "Variance Ratio Score"}

        y_title = y_titles[method]

        # Create visualization for method
        plt.figure(figsize=plot_dict["figsize"])
        y_values = [y for _ , y in  performance_dict[group].items()]
        plt.plot(self._n_clusters, y_values)

        if plot_dict["step"]:
            x_ticks = range(self._n_clusters[0], self._n_clusters[-1] + 1, plot_dict["step"])
            plt.xticks(x_ticks)

        plt.xlabel("K")
        plt.ylabel(y_title)
        plt.title(group)

        # Add vertical line for elbow method
        if y_title == "Inertia":
            plt.vlines(self._optimal_n_clusters[group], plt.ylim()[0], plt.ylim()[1], linestyles="--",
                        label="elbow")

        if output_dir:
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            save_name = f"{group.replace(' ','_')}_{self._cluster_selection_method}.png"
            plt.savefig(os.path.join(output_dir,save_name), dpi=plot_dict["dpi"],
                        bbox_inches=plot_dict["bbox_inches"])

        plt.show() if show_figs else plt.close()

    def _create_caps_dict(self):
        # Initialize dictionary
        self._caps = {}

        for group in self._groups:
            self._caps[group] = {}
            cluster_centroids = zip([num for num in range(1,len(self._kmeans[group].cluster_centers_)+1)],
                                    self._kmeans[group].cluster_centers_)
            self._caps[group].update({f"CAP-{state_number}": state_vector
                                      for state_number, state_vector in cluster_centroids})

    def calculate_metrics(self,
                          subject_timeseries: Union[dict[str, dict[str, np.ndarray]], os.PathLike],
                          tr: Optional[float]=None,
                          runs: Optional[Union[int, str, list[int], list[str]]]=None,
                          continuous_runs: bool=False,
                          metrics: Union[
                              Literal[
                                  "temporal_fraction", "persistence", "counts",
                                  "transition_frequency", "transition_probability"],
                              list[Literal["temporal_fraction", "persistence","counts",
                                           "transition_frequency", "transition_probability"]]
                              ]=["temporal_fraction", "persistence", "counts", "transition_frequency"],
                          return_df: bool=True,
                          output_dir: Optional[os.PathLike]=None,
                          prefix_file_name: Optional[str]=None) -> dict[str, pd.DataFrame]:
        """
        **Compute Participant-wise CAP Metrics**

        Creates a single ``pandas.DataFrame`` per CAP metric for all participants (with the exception of
        "transition_probability" which creates a single dataframe per group). As described by
        Liu et al., 2018 and Yang et al., 2021. The metrics include:

         - ``"temporal_fraction"`` : The proportion of total volumes spent in a single CAP over all volumes in a run.
           Additionally, in the supplementary material of Yang et al., the stated relationship between
           temporal fraction, counts, and persistence is temporal fraction = (persistence*counts)/total volumes
           If persistence and temporal fraction is converted into time units,
           then temporal fraction = (persistence*counts)/(total volumes * TR)
           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
                target = 1
                temporal_fraction = 4/6

         - ``"persistence"`` : The average time spent in a single CAP before transitioning to another CAP
           (average consecutive/uninterrupted time).
           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
                target = 1
                # Sequences for 1 are [1] and [1,1,1]
                persistence = (1 + 3)/2 # Average number of frames = 2
                tr = 2
                if tr:
                    persistence = ((1 + 3)/2)*2 # Turns average frames into average time = 4

         - ``"counts"`` : The total number of initiations of a specific CAP across an entire run. An initiation is
           defined as the first occurrence of a CAP. If the same CAP is maintained in contiguous segment
           (indicating stability), it is still counted as a single initiation.
           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
                target = 1
                # Initiations of CAP-1 occur at indices 0 and 2
                counts = 2

         - ``"transition_frequency"`` : The total number of transitions to different CAPs across the entire run.
           ::

                predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
                # Transitions between unique CAPs occur at indices 0 -> 1, 1 -> 2, and 4 -> 5
                transition_frequency = 3

         - ``"transition_probability"`` : The probability of transitioning from one CAP to another CAP (or the same CAP).
           This is calculated as (Number of transitions from A to B)/ (Total transitions from A). Note that the
           transition probability from CAP-A -> CAP-B is not the same as CAP-B -> CAP-A.
           ::

                # Note last two numbers in the predicted timeseries are switched for this example
                predicted_subject_timeseries = [1, 2, 1, 1, 3, 1]
                # If three CAPs were identified in the analysis
                combinations = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
                target = (1,2) # Represents transition from CAP-1 -> CAP-2
                # There are 4 ones in the timeseries but only three transitions from 1; 1 -> 2, 1 -> 1, 1 -> 3
                n_transitions_from_1 = 3
                # There is only one 1 -> 2 transition.
                transition_probability = 1/3
                # 1 -> 1 has a probability of 1/3 and 1 -> 3 has a probability of 1/3

        Parameters
        ----------
        subject_timeseries : :obj:`dict[str, dict[str, np.ndarray]]` or :obj:`os.PathLike`
            Path of the pickle file containing the nested subject timeseries dictionary saved by the
            ``TimeSeriesExtractor`` class or the nested subject timeseries dictionary produced by the
            ``TimeseriesExtractor`` class. The first level of the nested dictionary must consist of the subject ID
            as a string, the second level must consist of the run numbers in the form of "run-#" (where # is the
            corresponding number of the run), and the last level must consist of the timeseries (as a numpy array)
            associated with that run. **This does not need to be the same subject timeseries dictionary used for
            generating the k-means model but should have the same number of columns/ROIs**. If your
            ``subject_timeseries`` does not contain the same subject IDs then use the ``self.subject_table`` setter to
            generate the appropriate subject ID and group name mapping. Note, if standardizing was requested in
            ``self.get_caps()``, then the columns/ROIs of the ``subject_timeseries`` provided to this method will be
            scaled using the mean and sample standard deviation derived from the concatenated data used to generate the
            k-means model. This is to ensure that each subject's frames are correctly assigned to the cluster centroid
            it is closest to as it will be on the same scale of the data used to generate the k-means model. If you
            specify the same ``subject_timeseries`` (or any of the individual dictionaries that were combined to form
            the ``subject_timeseries``) used for creating the k-means model, the predicted label assignments will align
            with the original label assignments from the k-means model. The structure of is as follows:
            ::

                subject_timeseries = {
                        "101": {
                            "run-0": np.array([...]), # 2D array
                            "run-1": np.array([...]), # 2D array
                            "run-2": np.array([...]), # 2D array
                        },
                        "102": {
                            "run-0": np.array([...]), # 2D array
                            "run-1": np.array([...]), # 2D array
                        }
                    }

        tr : :obj:`float` or :obj:`None`, default=None
            The repetition time (TR). If provided, persistence will be calculated as the average uninterrupted time
            spent in each CAP. If not provided, persistence will be calculated as the average uninterrupted volumes
            (TRs) spent in each state.

        runs : :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            The run numbers to calculate CAP metrics for. If None, CAP metrics will be calculated for each run.

        continuous_runs : :obj:`bool`, default=False
            If True, all runs will be treated as a single, uninterrupted run.
            ::

                run_1 = [0,1,1]
                run_2 = [2,3,3]
                continuous_runs = [0,1,1,2,3,3]

        metrics : {"temporal_fraction", "persistence", "counts", "transition_frequency", "transition_probability"} or :obj:`list["temporal_fraction", "persistence", "counts", "transition_frequency", "transition_probability"]`, default=["temporal_fraction", "persistence", "counts", "transition_frequency"]
            The metrics to calculate. Available options include "temporal_fraction", "persistence",
            "counts", "transition_frequency", and "transition_probability".

            .. versionadded:: 0.16.2 "transition_probabilities"
            .. versionchanged:: 0.16.6 calculation for "counts" has changed to align with the equation,
               temporal fraction = (persistence*counts)/total volumes, which can be found in the supplemental
               material of Yang et al., 2022

        return_df : :obj:`str`, default=True
            If True, returns ``pandas.DataFrame`` inside a ``dict``, where the key corresponds to the
            metric requested.

        output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
            Directory to save ``pandas.DataFrame`` as csv file to. The directory will be created if it does not exist.
            Will not be saved if None.

        prefix_file_name : :obj:`str` or :obj:`None`, default=None
            Will serve as a prefix to append to the saved file names for each ``pandas.DataFrame``, if
            ``output_dir`` is provided.

        Returns
        -------
        dict[str, pd.DataFrame] or dict[str, dict[str, pd.DataFrame]]
            Dictionary containing `pandas.DataFrame` - one for each requested metric. In the case of "transition_probability",
            each group has a separate dataframe which is returned in the from of `dict[str, dict[str, pd.DataFrame]]`.

        Note
        ----
        Information for all groups will be in a single file, there is a "Group" column to denote the group the
        subject belongs to, this column exists even when no group is specified, all subjects all listed as being
        in the "All Subjects" group.

        The presence of 0 for specific CAPs in the "temporal_fraction", "persistence", or "counts" dataFrames
        indicates that the participant had zero instances of a specific CAP. In the case of "transition_probability",
        a 0 indicates that a specific transition between two CAPs where not observed. If performing an analysis on groups
        where each group has a different number of CAPs, then for "temporal_fraction", "persistence", and "counts",
        "nan" values will be seen for CAP numbers that exceed the group's number of CAPs.

        For instance, if group "A" has 2 CAPs but group "B" has 4 CAPs, the DataFrame will contain columns for CAP-1,
        CAP-2, CAP-3, and CAP-4. However, for all members in group "A", CAP-3 and CAP-4 will contain "nan" values to
        indicate that these CAPs are not applicable to the group. This differentiation helps distinguish between CAPs
        that are not applicable to the group and CAPs that are applicable but had zero instances for a specific member.

        For "transition_frequency", a 0 indicates no transitions due to all participant's frames being assigned to a
        single CAP.

        **For "transition_probability", the cap-to-cap transitions are indicated by the column names. For instance,
        column "1.2" indicate the probability from transitioning from CAP-1 to CAP-2, while column "2.1" represents
        the probability of transitioning from CAP-2 to CAP-1.**

        References
        ----------
        Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals.
        NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

        Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation
        patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia.
        NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

        """
        if not hasattr(self, "_kmeans"):
            raise AttributeError("Cannot calculate metrics since `self._kmeans` attribute does not exist. Run "
                                 "`self.get_caps()` first.")

        if prefix_file_name is not None and output_dir is None:
            LG.warning("`prefix_file_name` supplied but no `output_dir` specified. Files will not be saved.")

        if runs and not isinstance(runs, list): runs = [runs]

        metrics = self._filter_metrics(metrics)

        subject_timeseries = self._process_subject_timeseries(subject_timeseries)

        # Assign each subject's frame to a CAP
        predicted_subject_timeseries = self._build_prediction_dict(subject_timeseries, runs, continuous_runs)

        # Get CAP information
        cap_names, cap_numbers, group_cap_counts = self._cap_info()

        # Get combination of transitions in addition to building the base dataframe dictionary
        if "transition_probability" in metrics:
            group_caps, products, products_unique = self._combinations()
            df_dict, temp_dict = self._build_df(metrics, cap_names, products)
        else:
            df_dict = self._build_df(metrics, cap_names)

        # Generate list for iteration
        distributed_list = self._distribute(predicted_subject_timeseries)

        for subj_id, group, curr_run in distributed_list:
            group_name = group.replace(" ", "_")

            if "temporal_fraction" in metrics:
                # Get frequency
                frequency_dict = {key: np.where(predicted_subject_timeseries[subj_id][curr_run] == key,1,0).sum()
                                  for key in range(1, group_cap_counts[group] + 1)}
                self._update_dict(cap_numbers, group_cap_counts[group], frequency_dict)

                if "temporal_fraction" in metrics:
                    proportion_dict = {key: value/(len(predicted_subject_timeseries[subj_id][curr_run]))
                                       for key, value in frequency_dict.items()}
                    # Populate Dataframe
                    new_row = [subj_id, group_name, curr_run] + [items for items in proportion_dict.values()]
                    df_dict["temporal_fraction"].loc[len(df_dict["temporal_fraction"])] = new_row

            if "counts" in metrics:
                count_dict = {}
                for target in cap_numbers:
                    _ , segments = self._segments(target, predicted_subject_timeseries[subj_id][curr_run])
                    count_dict.update({target: segments})

                self._update_dict(cap_numbers, group_cap_counts[group], count_dict)

                # Populate Dataframe
                new_row = [subj_id, group_name, curr_run] + [items for items in count_dict.values()]
                df_dict["counts"].loc[len(df_dict["counts"])] = new_row

            if "persistence" in metrics:
                # Initialize variable
                persistence_dict = {}
                # Iterate through caps
                for target in cap_numbers:
                    binary_arr, segments = self._segments(target, predicted_subject_timeseries[subj_id][curr_run])
                    # Sum of ones in the binary array divided by segments, then multiplied by 1 or the tr; segment is
                    # always 1 at minimum due to + 1; np.where(np.diff(target_indices, n=1) > 1, 1,0).sum() is 0
                    # when empty or the condition isn't met
                    persistence_dict.update({target: (binary_arr.sum()/segments) * (tr if tr else 1)})

                self._update_dict(cap_numbers, group_cap_counts[group], persistence_dict)

                # Populate Dataframe
                new_row = [subj_id, group_name, curr_run] + [items for _ , items in persistence_dict.items()]
                df_dict["persistence"].loc[len(df_dict["persistence"])] = new_row

            if "transition_frequency" in metrics:
                # Sum the differences that are not zero - [1,2,1,1,1,3] becomes [1,-1,0,0,2], binary representation for
                # values not zero is [1,1,0,0,1] = 3 transitions
                transition_frequency = np.where(
                    np.diff(predicted_subject_timeseries[subj_id][curr_run], n=1) != 0, 1, 0).sum()
                # Populate DataFrame
                new_row = [subj_id, group_name, curr_run, transition_frequency]
                df_dict["transition_frequency"].loc[len(df_dict["transition_frequency"])] = new_row

            if "transition_probability" in metrics:
                base_row = [subj_id, group, curr_run] + [0.0] * (temp_dict[group].shape[-1] - 3)
                temp_dict[group].loc[len(temp_dict[group])] = base_row

                # Get number of transitions
                trans_dict = {target: np.sum(np.where(predicted_subject_timeseries[subj_id][curr_run][:-1] == target,1,0))
                              for target in group_caps[group]}
                indx = temp_dict[group].index[-1]

                # Iterate through products and calculate all symmetric pairs/off-diagonals
                for prod in products_unique[group]:
                    target1, target2 = prod[0], prod[1]
                    trans_array = predicted_subject_timeseries[subj_id][curr_run].copy()
                    # Set all values not equal to target1 or target2 to zero
                    trans_array[(trans_array != target1) & (trans_array != target2)] = 0
                    trans_array[np.where(trans_array == target1)] = 1
                    trans_array[np.where(trans_array == target2)] = 3
                    # 2 indicates forward transition target1 -> target2; -2 means reverse/backward transition
                    # target2 -> target1
                    diff_array = np.diff(trans_array, n=1)

                    # Avoid division by zero errors and calculate both the forward and reverse transition
                    if trans_dict[target1] != 0:
                        temp_dict[group].loc[indx,f"{target1}.{target2}"] = float(
                            np.sum(np.where(diff_array == 2,1,0))/trans_dict[target1])

                    if trans_dict[target2] != 0:
                        temp_dict[group].loc[indx,f"{target2}.{target1}"] = float(
                            np.sum(np.where(diff_array == -2,1,0))/trans_dict[target2])

                # Calculate the probability for the self transitions/diagonals
                for target in group_caps[group]:
                    if trans_dict[target] == 0: continue
                    # Will include the {target}.{target} column, but the value is initially set to zero
                    columns = temp_dict[group].filter(regex=fr"^{target}\.").columns.tolist()
                    cumulative = temp_dict[group].loc[indx,columns].values.sum()
                    temp_dict[group].loc[indx, f"{target}.{target}"] = 1.0 - cumulative

        if "transition_probability" in metrics: df_dict["transition_probability"] = temp_dict

        if output_dir: self._save_metrics(output_dir, df_dict, prefix_file_name)

        if return_df: return df_dict

    @staticmethod
    def _filter_metrics(metrics):
        metrics = [metrics] if isinstance(metrics, str) else metrics

        # Change metrics to set
        metrics = set(metrics)

        valid_metrics = {"temporal_fraction", "persistence", "counts", "transition_frequency", "transition_probability"}

        set_diff = metrics - valid_metrics

        metrics = metrics.intersection(valid_metrics)

        if set_diff:
            formatted_string = ', '.join(["'{a}'".format(a=x) for x in set_diff])
            LG.warning(f"Invalid metrics will be ignored: {formatted_string}.")
        if not metrics:
            formatted_string = ', '.join(["'{a}'".format(a=x) for x in valid_metrics])
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
            # Initialize predicted timeseries dict if first subject in table
            if subj_id == list(self._subject_table)[0]: predicted_subject_timeseries = {}

            predicted_subject_timeseries[subj_id] = {}
            requested_runs = [f"run-{run}" for run in runs] if runs else list(subject_timeseries[subj_id])
            subject_runs = [subject_run for subject_run in subject_timeseries[subj_id] if subject_run in requested_runs]

            if not subject_runs:
                LG.warning(f"[SUBJECT: {subj_id}] Does not have the requested run numbers: {','.join(requested_runs)}.")
                continue

            for curr_run in subject_runs:
                # Standardize or not
                if self._standardize:
                    timeseries = (subject_timeseries[subj_id][curr_run] - self._mean_vec[group])/self._stdev_vec[group]
                else:
                    timeseries = subject_timeseries[subj_id][curr_run]

                # Set run_id
                run_id = curr_run if not continuous_runs or len(subject_runs) == 1 else "continuous_runs"

                # Add 1 to the prediction vector since labels start at 0, needed to ensure that the labels map onto the
                # caps
                prediction_vector = self._kmeans[group].predict(timeseries) + 1

                if run_id != "continuous_runs":
                    predicted_subject_timeseries[subj_id].update({run_id: prediction_vector})
                else:
                    # Horizontally stack predicted runs
                    if curr_run == subject_runs[0]: predicted_continuous_timeseries = prediction_vector
                    else: predicted_continuous_timeseries = np.hstack([predicted_continuous_timeseries,
                                                                       prediction_vector])
            if run_id == "continuous_runs":
                predicted_subject_timeseries[subj_id].update({run_id: predicted_continuous_timeseries})

        return predicted_subject_timeseries

    # Get all combinations of transitions
    def _combinations(self):
        group_caps = {}
        products, products_unique = {}, {}

        for group in self._groups:
            group_caps.update({group: [int(name.split("-")[-1]) for name in self._caps[group]]})
            products.update({group: list(itertools.product(group_caps[group], group_caps[group]))})
            # Filter out all reversed products and products with the self transitions
            products_unique[group] = []
            for prod in products[group]:
                if prod[0] == prod[1]: continue
                # Include only the first instance of symmetric pairs
                if (prod[1], prod[0]) not in products_unique[group]: products_unique[group].append(prod)

        return group_caps, products, products_unique

    def _build_df(self, metrics, cap_names, products=None):
        df_dict = {}

        base_cols = ["Subject_ID", "Group","Run"]

        for metric in metrics:
            if metric not in ["transition_frequency", "transition_probability"]:
                df_dict.update({metric: pd.DataFrame(columns=base_cols + list(cap_names))})
            elif metric == "transition_probability":
                temp_dict = {}
                for group in self._groups:
                    temp_dict.update(
                        {group: pd.DataFrame(columns=base_cols + [f"{x}.{y}" for x, y in products[group]])} )
            else:
                df_dict.update({metric: pd.DataFrame(columns=base_cols + ["Transition_Frequency"])})

        if "transition_probability" in metrics: return df_dict, temp_dict
        else: return df_dict

    def _distribute(self, predicted_subject_timeseries):
        distributed_list = []

        for subj_id, group in self._subject_table.items():
            for curr_run in predicted_subject_timeseries[subj_id]:
                distributed_list.append([subj_id, group, curr_run])

        return distributed_list

    # Replace zeros with nan for groups with less caps than the group with the max caps
    @staticmethod
    def _update_dict(cap_numbers, group_cap_counts, curr_dict):
        if max(cap_numbers) > group_cap_counts:
            for i in range(group_cap_counts + 1, max(cap_numbers) + 1): curr_dict.update({i: float("nan")})

    @staticmethod
    def _segments(target, timeseries):
        # Binary representation of numpy array - if [1,2,1,1,1,3] and target is 1, then it is [1,0,1,1,1,0]
        binary_arr = np.where(timeseries == target,1,0)
        # Get indices of values that equal 1; [0,2,3,4]
        target_indices = np.where(binary_arr == 1)[0]
        # Count the transitions, indices where diff > 1 is a transition; diff of indices = [2,1,1];
        # binary for diff > 1 = [1,0,0]; thus, segments = transitions + first_sequence(1) = 2
        segments = np.where(np.diff(target_indices, n=1) > 1,1,0).sum() + 1

        return binary_arr, segments

    def _save_metrics(self, output_dir, df_dict, prefix_file_name):
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        for metric in df_dict:
            if prefix_file_name:
                file_name = os.path.splitext(prefix_file_name.rstrip())[0].rstrip() + f"-{metric}"
            else:
                file_name = f"{metric}"

            if metric != "transition_probability":
                df_dict[f"{metric}"].to_csv(path_or_buf=os.path.join(output_dir, f"{file_name}.csv"),
                                            sep=",", index=False)
            else:
                for group in self._groups:
                    df_dict[f"{metric}"][group].to_csv(
                        path_or_buf=os.path.join(output_dir, f"{file_name}-{group.replace(' ', '_')}.csv"),
                        sep=",", index=False
                        )

    def caps2plot(self,
                  output_dir: Optional[os.PathLike]=None,
                  suffix_title: Optional[str]=None,
                  plot_options: Union[Literal["outer_product", "heatmap"],
                                      list[Literal["outer_product", "heatmap"]]]="outer_product",
                  visual_scope: Union[Literal["regions", "nodes"],
                                      list[Literal["regions", "nodes"]]]="regions",
                  show_figs: bool=True,
                  subplots: bool=False,
                  **kwargs) -> seaborn.heatmap:
        """
        **Generate Heatmaps and Outer Product Plots for CAPs**

        This function produces a ``seaborn.heatmap`` for each CAP. If groups were given when the CAP class was
        initialized, plotting will be done for all CAPs for all groups.

        Parameters
        ----------
        output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
            Directory to save plots to. The directory will be created if it does not exist. If None, plots will not
            be saved. Outputs as png file.

        suffix_title : :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot as well as the name of the saved file if ``output_dir`` is provided.

        plot_options : {"outer_product", "heatmap"} or :obj:`list["outer_product", "heatmap"]`, default="outer_product"
            Type of plots to create. Options are "outer_product" or "heatmap".

        visual_scope : {"regions", "nodes"} or :obj:`list["regions", "nodes"]`, default="regions"
            Determines whether plotting is done at the region level or node level.
            For region level, the value of each nodes in the same regions (both left and right hemisphere nodes in the
            same region) are averaged together then plotted. Options are "regions" or "nodes".

        show_figs : :obj:`bool`, default=True
            Whether to display figures.

        subplots : :obj:`bool`, default=True
            Whether to produce subplots for outer product plots.

        kwargs : :obj:`dict`
            Keyword arguments used when saving figures. Valid keywords include:

            - dpi : :obj:`int`, default=300
                Dots per inch for the figure. Default is 300 if ``output_dir`` is provided and ``dpi`` is not specified.
            - figsize : :obj:`tuple`, default=(8, 6)
                Size of the figure in inches.
            - fontsize : :obj:`int`, default=14
                Font size for the title of individual plots or subplots.
            - hspace : :obj:`float`, default=0.4
                Height space between subplots.
            - wspace : :obj:`float`, default=0.4
                Width space between subplots.
            - xticklabels_size : :obj:`int`, default=8
                Font size for x-axis tick labels.
            - yticklabels_size : :obj:`int`, default=8
                Font size for y-axis tick labels.
            - shrink : :obj:`float`, default=0.8
                Fraction by which to shrink the colorbar.
            - nrow : :obj:`int`, default=varies (max 5)
                Number of rows for subplots. Default varies but the maximum is 5.
            - ncol : :obj:`int` or :obj:`None`, default=None
                Number of columns for subplots. Default varies but the maximum is 5.
            - suptitle_fontsize : :obj:`float`, default=0.7
                Font size for the main title when subplot is True.
            - tight_layout : :obj:`bool`, default=True
                Use tight layout for subplots.
            - rect : :obj:`list[int]`, default=[0, 0.03, 1, 0.95]
                Rectangle parameter for tight layout when subplots are True to fix whitespace issues.
            - sharey : :obj:`bool`, default=True
                Share y-axis labels for subplots.
            - xlabel_rotation : :obj:`int`, default=0
                Rotation angle for x-axis labels.
            - ylabel_rotation : :obj:`int`, default=0
                Rotation angle for y-axis labels.
            - annot : :obj:`bool`, default=False
                Add values to cells.
            - annot_kws : :obj:`dict`, default=None,
                Customize the annotations.
            - fmt : :obj:`str`, default=".2g"
                Modify how the annotated vales are presented.
            - linewidths : :obj:`float`, default=0
                Padding between each cell in the plot.
            - borderwidths : :obj:`float`, default=0
                Width of the border around the plot.
            - linecolor : :obj:`str`, default="black"
                Color of the line that seperates each cell.
            - edgecolors : :obj:`str` or :obj:`None`, default=None
                Color of the edges.
            - alpha : :obj:`float` or :obj:`None`, default=None
                Controls transparancy and ranges from 0 (transparant) to 1 (opaque).
            - bbox_inches : :obj:`str` or :obj:`None`, default="tight"
                Alters size of the whitespace in the saved image.
            - hemisphere_labels : :obj:`bool`, default=False
                This option is only available when ``visual_scope="nodes"``. Instead of listing all individual labels
                this parameter simplifies the labels to indicate only the left and right hemispheres, with a
                division line separating the cells belonging to each hemisphere. If set to True, ``edgecolors``
                will not be used, and both ``linewidths`` and ``linecolor`` will be applied only to the division
                line. This option is available exclusively for "Custom" and "Schaefer" parcellations.
                **Note, for the "Custom" option, the parcellation should be organized such that the first half of the
                nodes belong to the left hemisphere and the latter half to the right hemisphere.**
            - cmap : :obj:`str` or :obj:`callable` default="coolwarm"
                Color map for the cells in the plot. For this parameter, you can use premade color palettes or
                create custom ones. Below is a list of valid options:

                    - Strings to call seaborn's premade palettes.
                    - ``seaborn.diverging_palette`` function to generate custom palettes.
                    - ``matplotlib.colors.LinearSegmentedColormap`` to generate custom palettes.

            - vmin : :obj:`float` or :obj:`None`, default=None
                The minimum value to display in plots.
            - vmax : :obj:`float` or :obj:`None`, default=None
                The maximum value to display in plots.

        Returns
        -------
        `seaborn.heatmap`
            An instance of `seaborn.heatmap`.

        Note
        ----
        **If using "Custom" parcellation approach**, the "nodes" and "regions" sub-keys are required for this
        function.

        For valid premade palettes for seaborn, refer to https://seaborn.pydata.org/tutorial/color_palettes.html

        """
        if not self._parcel_approach:
            raise AttributeError("`self.parcel_approach` is None. Add `parcel_approach` using "
                                 "`self.parcel_approach=parcel_approach` to use this method.")

        if not hasattr(self,"_caps"):
            raise AttributeError("Cannot plot caps since `self._caps` attribute does not exist. Run `self.get_caps()` "
                                 "first.")

        # Check if parcellation_approach is custom
        if "Custom" in self._parcel_approach and any(key not in self._parcel_approach["Custom"] for key in ["nodes", "regions"]):
            _check_parcel_approach(parcel_approach=self._parcel_approach, call="caps2plot")

        # Get parcellation name
        parcellation_name = list(self._parcel_approach)[0]

        # Check labels
        check_caps = self._caps[list(self._caps)[0]]
        check_caps = check_caps[list(check_caps)[0]]

        if check_caps.shape[0] != len(self._parcel_approach[parcellation_name]["nodes"]):
            raise ValueError("Number of nodes used for CAPs does not equal the number of nodes specified in "
                             "`parcel_approach`.")

        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

        # Convert to list
        if isinstance(plot_options, str): plot_options = [plot_options]
        if isinstance(visual_scope, str): visual_scope = [visual_scope]

        # Check inputs for plot_options and visual_scope
        if not any(["heatmap" in plot_options, "outer_product" in plot_options]):
            raise ValueError("Valid inputs for `plot_options` are 'heatmap' and 'outer_product'.")

        if not any(["regions" in visual_scope, "nodes" in visual_scope]):
            raise ValueError("Valid inputs for `visual_scope` are 'regions' and 'nodes'.")

        if "regions" in visual_scope: self._create_regions(parcellation_name=parcellation_name)

        # Create plot dictionary
        defaults= {"dpi": 300, "figsize": (8, 6), "fontsize": 14, "hspace": 0.2, "wspace": 0.2, "xticklabels_size": 8,
                   "yticklabels_size": 8, "shrink": 0.8, "nrow": None, "ncol": None, "suptitle_fontsize": 20,
                   "tight_layout": True, "rect": [0, 0.03, 1, 0.95], "sharey": True, "xlabel_rotation": 0,
                   "ylabel_rotation": 0, "annot": False, "annot_kws": None, "fmt": ".2g", "linewidths": 0,
                   "linecolor": "black", "cmap": "coolwarm", "edgecolors": None, "alpha": None,
                   "hemisphere_labels": False, "borderwidths": 0, "vmin": None, "vmax": None, "bbox_inches": "tight"}

        plot_dict = _check_kwargs(defaults, **kwargs)

        if plot_dict["hemisphere_labels"] is True:
            if "nodes" not in visual_scope:
                raise ValueError("`hemisphere_labels` is only available when `visual_scope == 'nodes'`.")
            if parcellation_name == "AAL":
                raise ValueError("`hemisphere_labels` is only available for 'Custom' and 'Schaefer'.")

        # Ensure plot_options and visual_scope are lists
        plot_options = plot_options if isinstance(plot_options, list) else list(plot_options)
        visual_scope = visual_scope if isinstance(visual_scope, list) else list(visual_scope)
        # Initialize outer product attribute
        if "outer_product" in plot_options: self._outer_products = {}

        distributed_list = list(itertools.product(plot_options, visual_scope, self._groups))

        for plot_option, scope, group in distributed_list:
            # Get correct labels depending on scope
            if scope == "regions":
                if parcellation_name in ["Schaefer", "AAL"]:
                    cap_dict, columns = self._region_caps, self._parcel_approach[parcellation_name]["regions"]
                else:
                    cap_dict, columns = self._region_caps, list(self._parcel_approach["Custom"]["regions"])
            elif scope == "nodes":
                if parcellation_name in ["Schaefer", "AAL"]:
                    cap_dict, columns = self._caps, self._parcel_approach[parcellation_name]["nodes"]
                else:
                    cap_dict = self._caps
                    columns =  [x[0] + " " + x[1] for x in
                                list(itertools.product(["LH", "RH"], self._parcel_approach["Custom"]["regions"]))]

            #  Generate plot for each group
            input_keys = dict(group=group, plot_dict=plot_dict, cap_dict=cap_dict, columns=columns,
                              output_dir=output_dir,suffix_title=suffix_title,show_figs=show_figs,scope=scope,
                              parcellation_name=parcellation_name)

            #  Generate plot for each group
            if plot_option == "outer_product": self._generate_outer_product_plots(**input_keys,subplots=subplots)
            elif plot_option == "heatmap": self._generate_heatmap_plots(**input_keys)

    def _create_regions(self, parcellation_name):
        # Internal function to create an attribute called `region_caps`. Purpose is to average the values of all nodes
        # in a corresponding region to create region heatmaps or outer product plots
        self._region_caps = {group: {} for group in self._groups}

        for group in self._groups:
            for cap in self._caps[group]:
                region_caps = None
                if parcellation_name != "Custom":
                    for region in self._parcel_approach[parcellation_name]["regions"]:
                        region_indxs = np.array([index for index, node in
                                                 enumerate(self._parcel_approach[parcellation_name]["nodes"])
                                                 if region in node])
                        if region_caps is None:
                            region_caps = np.array([np.average(self._caps[group][cap][region_indxs])])
                        else:
                            region_caps = np.hstack([region_caps, np.average(self._caps[group][cap][region_indxs])])
                else:
                    region_dict = self._parcel_approach["Custom"]["regions"]
                    region_keys = list(region_dict)
                    for region in region_keys:
                        roi_indxs = np.array(region_dict[region]["lh"] + region_dict[region]["rh"])
                        if region_caps is None:
                            region_caps= np.array([np.average(self._caps[group][cap][roi_indxs])])
                        else:
                            region_caps= np.hstack([region_caps, np.average(self._caps[group][cap][roi_indxs])])

                self._region_caps[group].update({cap: region_caps})

    def _generate_outer_product_plots(self, group, plot_dict, cap_dict, columns, subplots, output_dir, suffix_title,
                                      show_figs, scope, parcellation_name):
        # Nested dictionary for group
        self._outer_products[group] = {}

        # Create base grid for subplots
        if subplots:
            # Max five subplots per row for default
            default_col = len(cap_dict[group]) if len(cap_dict[group]) <= 5 else 5
            ncol = plot_dict["ncol"] if plot_dict["ncol"] is not None else default_col
            if ncol > len(cap_dict[group]): ncol = len(cap_dict[group])
            # Pad nrow, since int will round down, padding is needed for cases
            # where len(cap_dict[group])/ncol is a float. This will add the extra row needed
            x_pad = 0 if len(cap_dict[group])/ncol <= 1 else 1
            nrow = plot_dict["nrow"] if plot_dict["nrow"] is not None else x_pad + int(len(cap_dict[group])/ncol)

            subplot_figsize = (8 * ncol, 6 * nrow) if plot_dict["figsize"] == (8,6) else plot_dict["figsize"]

            fig, axes = plt.subplots(nrow, ncol, sharex=False, sharey=plot_dict["sharey"], figsize=subplot_figsize)
            suptitle = f"{group} {suffix_title}" if suffix_title else f"{group}"
            fig.suptitle(suptitle, fontsize=plot_dict["suptitle_fontsize"])
            fig.subplots_adjust(hspace=plot_dict["hspace"], wspace=plot_dict["wspace"])
            if plot_dict["tight_layout"]: fig.tight_layout(rect=plot_dict["rect"])

            # Current subplot
            axes_x, axes_y = [0,0]

        # Iterate over CAPs
        for cap in cap_dict[group]:
            # Calculate outer product
            self._outer_products[group].update({cap: np.outer(cap_dict[group][cap],cap_dict[group][cap])})
            # Create labels if nodes requested for scope
            if scope == "nodes" and plot_dict["hemisphere_labels"] is False:
                labels, _ = self._create_node_labels(parcellation_name, self._parcel_approach, columns)

            if subplots:
                ax = axes[axes_y] if nrow == 1 else axes[axes_x, axes_y]
                # Modify tick labels based on scope
                if scope == "regions":
                    display = seaborn.heatmap(ax=ax, data=self._outer_products[group][cap], xticklabels=columns,
                                              yticklabels=columns, **self._base_kwargs(plot_dict))
                else:
                    if plot_dict["hemisphere_labels"] is False:
                        display = seaborn.heatmap(ax=ax, data=self._outer_products[group][cap],
                                                  **self._base_kwargs(plot_dict))
                    else:
                        display = seaborn.heatmap(ax=ax, data=self._outer_products[group][cap],
                                                  **self._base_kwargs(plot_dict, False, False))

                    if plot_dict["hemisphere_labels"] is False:
                        ax = self._set_ticks(ax, labels)
                    else:
                        ax, division_line, plot_dict["linewidths"] = self._division_line(ax,
                                                                                         parcellation_name,
                                                                                         plot_dict["linewidths"])

                        ax.axhline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])
                        ax.axvline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])

                # Add border
                if plot_dict["borderwidths"] != 0:
                    border_length = self._outer_products[group][cap].shape[0]
                    display = self._border(display, plot_dict, border_length)

                display = self._label_size(display, plot_dict, True, False)

                # Modify label sizes; if share_y, only set y for plots at axes == 0
                if plot_dict["sharey"]:
                    if axes_y == 0: display = self._label_size(display, plot_dict, False, True)
                else:
                    display = self._label_size(display, plot_dict, False, True)

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
                    display = seaborn.heatmap(self._outer_products[group][cap],  xticklabels=columns,
                                              yticklabels=columns, **self._base_kwargs(plot_dict))
                else:
                    if plot_dict["hemisphere_labels"] is False:
                        display = seaborn.heatmap(self._outer_products[group][cap], xticklabels=[], yticklabels=[],
                                                  **self._base_kwargs(plot_dict))
                    else:
                        display = seaborn.heatmap(self._outer_products[group][cap], xticklabels=[], yticklabels=[],
                                                  **self._base_kwargs(plot_dict, False, False))

                    if plot_dict["hemisphere_labels"] is False:
                        display = self._set_ticks(display, labels)
                    else:
                        display, division_line, plot_dict["linewidths"] = self._division_line(display,
                                                                                              parcellation_name,
                                                                                              plot_dict["linewidths"])

                        plt.axhline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])
                        plt.axvline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])

                # Add border
                if plot_dict["borderwidths"] != 0:
                    border_length = self._outer_products[group][cap].shape[0]
                    display = self._border(display, plot_dict, border_length)

                # Modify label sizes
                plot_title = f"{group} {cap} {suffix_title}" if suffix_title else f"{group} {cap}"
                display.set_title(plot_title, fontdict= {"fontsize": plot_dict["fontsize"]})
                display = self._label_size(display, plot_dict)

                # Save individual plots
                if output_dir:
                    partial_filename = f"{group}_{cap}_{suffix_title}" if suffix_title else f"{group}_{cap}"
                    self._save_heatmap(display, scope, partial_filename, plot_dict, output_dir, call="outer_product")

        # Remove subplots with no data
        if subplots: [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]

        # Save subplot
        if subplots and output_dir:
            partial_filename = f"{group}_CAPs_{suffix_title}" if suffix_title else f"{group}_CAPs"
            self._save_heatmap(display, scope, partial_filename, plot_dict, output_dir, call="outer_product")

        # Display figures
        plt.show() if show_figs else plt.close()

    def _generate_heatmap_plots(self, group, plot_dict, cap_dict, columns, output_dir, suffix_title, show_figs,
                                scope, parcellation_name):
        # Initialize new grid
        plt.figure(figsize=plot_dict["figsize"])

        if scope == "regions":
            display = seaborn.heatmap(pd.DataFrame(cap_dict[group], index=columns), xticklabels=True, yticklabels=True,
                                      **self._base_kwargs(plot_dict))
        else:
            # Create Labels
            if plot_dict["hemisphere_labels"] is False:
                labels, names_list = self._create_node_labels(parcellation_name, self._parcel_approach, columns)

                display = seaborn.heatmap(pd.DataFrame(cap_dict[group], columns=list(cap_dict[group])),
                                          xticklabels=True, yticklabels=True, **self._base_kwargs(plot_dict))

                plt.yticks(ticks=[pos for pos, label in enumerate(labels) if label], labels=names_list)

            else:
                display = seaborn.heatmap(pd.DataFrame(cap_dict[group], columns=list(cap_dict[group])),
                                          xticklabels=True, yticklabels=True,
                                          **self._base_kwargs(plot_dict, False, False))

                display, division_line, plot_dict["linewidths"] = self._division_line(display,
                                                                                      parcellation_name,
                                                                                      plot_dict["linewidths"],
                                                                                      call="heatmap")

                plt.axhline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])

        if plot_dict["borderwidths"] != 0:
            y_length = len(cap_dict[group][list(cap_dict[group])[0]])
            display = self._border(display, plot_dict, y_length, len(self._caps[group]))

        # Modify label sizes
        plot_title = f"{group} CAPs {suffix_title}" if suffix_title else f"{group} CAPs"
        display.set_title(plot_title, fontdict= {"fontsize": plot_dict["fontsize"]})

        # Save plots
        if output_dir:
            partial_filename = f"{group}_CAPs_{suffix_title}" if suffix_title else f"{group}_CAPs"
            self._save_heatmap(display, scope, partial_filename, plot_dict, output_dir, call="heatmap")

        # Display figures
        plt.show() if show_figs else plt.close()

    @staticmethod
    def _base_kwargs(plot_dict, line = True, edge = True):
        kwargs = {"cmap": plot_dict["cmap"], "cbar_kws": {"shrink": plot_dict["shrink"]}, "annot": plot_dict["annot"],
                  "annot_kws": plot_dict["annot_kws"], "fmt": plot_dict["fmt"], "alpha": plot_dict["alpha"],
                  "vmin": plot_dict["vmin"], "vmax": plot_dict["vmax"]}

        if line: kwargs.update({"linewidths": plot_dict["linewidths"], "linecolor": plot_dict["linecolor"]})

        if edge: kwargs.update({"edgecolors": plot_dict["edgecolors"]})

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
            frequency_dict = collections.Counter([node.split("_")[0] for node in nodes])
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
        labels = ["" for _ in range(0,len(parcel_approach[parcellation_name]["nodes"]))]

        starting_value = 0

        # Iterate through names_list and assign the starting indices corresponding to unique region and hemisphere key
        for num, name in enumerate(names_list):
            if num == 0:
                labels[0] = name
            else:
                # Shifting to previous frequency of the preceding network to obtain the new starting value of
                # the subsequent region and hemisphere pair
                starting_value += frequency_dict[names_list[num-1]]
                labels[starting_value] = name

        return labels, names_list

    def _division_line(self, display, parcellation_name, linewidths, call = "outer"):
        n_labels = len(self._parcel_approach[parcellation_name]["nodes"])
        division_line = n_labels//2
        left_hemisphere_tick = (0 + division_line)//2
        right_hemisphere_tick = (division_line + n_labels)//2

        if call == "outer":
            display.set_xticks([left_hemisphere_tick,right_hemisphere_tick])
            display.set_xticklabels(["LH", "RH"])

        display.set_yticks([left_hemisphere_tick,right_hemisphere_tick])
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
    def _border(display, plot_dict, length, axvline = None):
        display.axhline(y=0, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        display.axhline(y=length, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        display.axvline(x=0, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        if axvline: display.axvline(x=axvline, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])
        else: display.axvline(x=length, color=plot_dict["linecolor"], linewidth=plot_dict["borderwidths"])

        return display

    @staticmethod
    def _label_size(display, plot_dict, set_x = True, set_y = True):
        if set_x:
            display.set_xticklabels(display.get_xticklabels(), size = plot_dict["xticklabels_size"],
                                    rotation=plot_dict["xlabel_rotation"])
        if set_y:
            display.set_yticklabels(display.get_yticklabels(), size = plot_dict["yticklabels_size"],
                                    rotation=plot_dict["ylabel_rotation"])

        return display

    # Function to save plots for outer_product and heatmap called by caps2plot
    @staticmethod
    def _save_heatmap(display, scope, partial, plot_dict, output_dir, call):
        full_filename = f"{partial.replace(' ', '_')}_{call}-{scope}.png"
        display.get_figure().savefig(os.path.join(output_dir, full_filename), dpi=plot_dict["dpi"],
                                     bbox_inches=plot_dict["bbox_inches"])

    def caps2corr(self,
                  output_dir: Optional[os.PathLike]=None,
                  suffix_title: Optional[str]=None,
                  show_figs: bool=True,
                  save_plots: bool=True,
                  return_df: bool=False,
                  save_df: bool=False,
                  **kwargs) -> Union[seaborn.heatmap, dict[str, pd.DataFrame]]:
        """
        **Generate Pearson Correlation Matrix for CAPs**

        Produces the correlation matrix of all CAPs and visualizes it as a ``seaborn.heatmap``. If groups were
        given when the CAP class was initialized, a correlation matrix will be generated for each group.
        Additionally, DataFrames of the correlation matrix with thier corresponding uncorrected p-value can be
        generated too. For correlation matrices, each element in the correlation matrix will contain its
        associated uncorrected p-value in parenthesis, with a single asterisk if < 0.05, a double asterisk if
        < 0.01, and a triple asterisk < 0.001 - ``{"<0.05": "*", "<0.01": "**", "<0.001": "***"}``. Additionally,
        all elements will be rounded using the formatting style provided by the ``fmt`` kwarg. Checking significance is
        done before formatting.

        Parameters
        ----------
        output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
            Directory to save plots and correlation matrices DataFrames to. The directory will be created if it does
            not exist. If None, plots and dataFrame will not be saved.

        suffix_title : :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot as well as the name of the saved file if ``output_dir``
            is provided.

        show_figs : :obj:`bool`, default=True
            Whether to display figures.

        save_plots : :obj:`bool`, default=True
            If True, plots are saves as png images. For this to be used, ``output_dir`` must be specified.

        return_df : :obj:`bool`, default=False
            If True, returns a dictionary with a correlation matrix for each group.

        save_df : :obj:`bool`, default=False,
            If True, saves the correlation matrix contained in the DataFrames as csv files. For this to be used,
            ``output_dir`` must be specified.

        kwargs : :obj:`dict`
            Keyword arguments used when modifying figures. Valid keywords include:

            - dpi : :obj:`int`, default=300
                Dots per inch for the figure. Default is 300 if ``output_dir`` is provided and ``dpi`` is not
                specified.
            - figsize : :obj:`tuple`, default=(8, 6)
                Size of the figure in inches.
            - fontsize : :obj:`int`, default=14
                Font size for the title each plot.
            - xticklabels_size : :obj:`int`, default=8
                Font size for x-axis tick labels.
            - yticklabels_size : :obj:`int`, default=8
                Font size for y-axis tick labels.
            - shrink : :obj:`float`, default=0.8
                Fraction by which to shrink the colorbar.
            - xlabel_rotation : :obj:`int`, default=0
                Rotation angle for x-axis labels.
            - ylabel_rotation : :obj:`int`, default=0
                Rotation angle for y-axis labels.
            - annot : :obj:`bool`, default=False
                Add values to each cell.
            - annot_kws : :obj:`dict`, default=None,
                Customize the annotations.
            - fmt : :obj:`str`, default=".2g",
                Modify how the annotated vales are presented.
            - linewidths : :obj:`float`, default=0
                Padding between each cell in the plot.
            - borderwidths : :obj:`float`, default=0
                Width of the border around the plot.
            - linecolor : :obj:`str`, default="black"
                Color of the line that seperates each cell.
            - edgecolors : :obj:`str` or :obj:`None`, default=None
                Color of the edges.
            - alpha : :obj:`float` or :obj:`None`, default=None
                Controls transparancy and ranges from 0 (transparant) to 1 (opaque).
            - bbox_inches : :obj:`str` or :obj:`None`, default="tight"
                Alters size of the whitespace in the saved image.
            - cmap : :obj:`str`, :obj:`callable` default="coolwarm"
                Color map for the cells in the plot. For this parameter, you can use premade color palettes or
                create custom ones.
                Below is a list of valid options:

                    - Strings to call seaborn's premade palettes.
                    - ``seaborn.diverging_palette`` function to generate custom palettes.
                    - ``matplotlib.color.LinearSegmentedColormap`` to generate custom palettes.

        Returns
        -------
        `seaborn.heatmap`
            An instance of `seaborn.heatmap`.
        `dict[str, pd.DataFrame]`
            An instance of a pandas DataFrame for each group.

        Note
        ----
        For valid premade palettes for ``seaborn``, refer to https://seaborn.pydata.org/tutorial/color_palettes.html
        """
        corr_dict = {group: None for group in self._groups} if return_df or save_df else None

        if not hasattr(self,"_caps"):
            raise AttributeError(
                "Cannot plot caps since `self._caps` attribute does not exist. Run `self.get_caps()` first."
                )

        # Create plot dictionary
        defaults = {"dpi": 300, "figsize": (8, 6), "fontsize": 14, "xticklabels_size": 8, "yticklabels_size": 8,
                    "shrink": 0.8, "xlabel_rotation": 0, "ylabel_rotation": 0, "annot": False, "linewidths": 0,
                    "linecolor": "black", "cmap": "coolwarm", "fmt": ".2g", "borderwidths": 0, "edgecolors": None,
                    "alpha": None, "bbox_inches": "tight", "annot_kws": None}

        plot_dict = _check_kwargs(defaults, **kwargs)

        for group in self._caps:
            df = pd.DataFrame(self._caps[group])

            corr_df = df.corr(method="pearson")

            display = _create_display(corr_df, plot_dict, suffix_title, group, "corr")

            if corr_dict:
                # Get p-values; use np.eye to make main diagonals equal zero; implementation of tozCSS from
                # https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance
                pval_df = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*corr_df.shape)
                # Add asterisk to values that meet the threshold
                pval_df = pval_df.map(
                    lambda x: f'({format(x, plot_dict["fmt"])})' + "".join(["*" for code in [0.05, 0.01, 0.001] if x < code])
                    )
                # Add the p-values to the correlation matrix
                corr_dict[group] = corr_df.map(lambda x: f'{format(x, plot_dict["fmt"])}') + " " + pval_df

            # Save figure
            if output_dir:
                _save_contents(output_dir, suffix_title, group, corr_dict, plot_dict, save_plots, save_df, display,
                               "corr")

            # Display figures
            plt.show() if show_figs else plt.close()

        if return_df: return corr_dict

    # Create basename for certain files
    @staticmethod
    def _basename(group, cap, desc=None, suffix=None, ext="png"):
        base = f"{group.replace(' ', '_')}_{cap}"

        if desc: base += f"_{desc}"

        if suffix: base += f"_{suffix.rstrip().replace(' ', '_')}"

        return base + f".{ext}"

    def caps2niftis(self,
                    output_dir: os.PathLike,
                    suffix_file_name: Optional[str]=None,
                    fwhm: Optional[float]=None,
                    knn_dict: dict[str, Union[int, list[int], np.array]]=None) -> nib.Nifti1Image:
        """
        **Standalone Method to Convert CAPs to NifTI Statistical Maps**

        Converts atlas into a stat map by replacing labels with the corresponding from the cluster centroids then saves
        them as compressed NifTI (nii.gz) files. Below is the internal function that converts the cluster centroids
        into a NifTI statistical map.
        ::

            def _cap2statmap(atlas_file, cap_vector, fwhm):
                atlas = nib.load(atlas_file)
                atlas_fdata = atlas.get_fdata()
                # Get array containing all labels in atlas to avoid issue if the first non-zero atlas label is not 1
                target_array = sorted(np.unique(atlas_fdata))
                for indx, value in enumerate(cap_vector, start=1):
                    atlas_fdata[np.where(atlas_fdata == target_array[indx])] = value
                stat_map = nib.Nifti1Image(atlas_fdata, atlas.affine, atlas.header)

        Parameters
        ----------
        output_dir : :obj:`os.PathLike`, default=None
            Directory to save plots to. The directory will be created if it does not exist.

        suffix_title : :obj:`str` or :obj:`None`, default=None
            Appended to the name of the saved file.

        fwhm : :obj:`float` or :obj:`None`, default=None
            Strength of spatial smoothing to apply (in millimeters) to the statistical map prior to interpolating
            from MNI152 space to fslr surface space. Note, this can assist with coverage issues in the plot.
            Uses ``nilearn.image.smooth_img``.

        knn_dict : :obj:`dict[str, int | bool]`, default=None
            Use KNN (k-nearest neighbors) interpolation to fill in non-background values that are assigned zero. This is
            primarily used as a fix for when a custom parcellation does not project well from volumetric to surface
            space. This method involves resampling the Schaefer parcellation, a volumetric parcellation that projects
            well onto surface space, to the target parcellation specified in the "maps" sub-key in ``self.parcel_approach``.
            The background indices are extracted from the Schaefer parcellation, and these indices are used to obtain
            the non-background indices (parcels) that are set to zero in the target parcellation.

            These indices are then iterated over, and the zero values are replaced with the value of the nearest neighbor,
            determined by the sub-key "k". The dictionary contains the following sub-keys:

            - "k" : An integer that determines the number of nearest neighbors to consider, with the majority vote determining the new value. If not specified, the default is 1.
            - "resolution_mm" : An integer (1 or 2) that determines the resolution of the Schaefer parcellation. If not specified, the default is 1.
            - "remove_subcortical": A list or  array of label ids as integers of the subcortical regions in the parcellation.

            This method is applied before the `fwhm` method.

        Returns
        -------
        `NifTI1Image`
            `NifTI` statistical map.
        """
        if not self._parcel_approach:
            raise AttributeError("`self.parcel_approach` is None. Set "
                                 "`self.parcel_approach=parcel_approach` to use this method.")

        if not hasattr(self,"_caps"):
            raise AttributeError(
                "Cannot plot caps since `self._caps` attribute does not exist. Run `self.get_caps()` first."
                )

        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

        parcellation_name = list(self._parcel_approach)[0]

        for group in self._caps:
            for cap in self._caps[group]:
                stat_map = _cap2statmap(atlas_file=self._parcel_approach[parcellation_name]["maps"],
                                        cap_vector=self._caps[group][cap], fwhm=fwhm, knn_dict=knn_dict)

                file_name = self._basename(group, cap, suffix=suffix_file_name, ext="nii.gz")

                nib.save(stat_map, os.path.join(output_dir, file_name))

    def caps2surf(self,
                  output_dir: Optional[os.PathLike]=None,
                  suffix_title: Optional[str]=None,
                  show_figs: bool=True,
                  fwhm: Optional[float]=None,
                  fslr_density: Literal["4k", "8k", "32k", "164k"]="32k",
                  method: Literal["linear", "nearest"]="linear",
                  save_stat_maps: bool=False,
                  fslr_giftis_dict: Optional[dict]=None,
                  knn_dict: dict[str, Union[int, list[int], np.array]]=None,
                  **kwargs) -> surfplot.Plot:
        """
        **Project CAPs onto Surface Plots**

        If not using precomputed GifTI file, this function converts the parcellation used for spatial dimensionality
        reduction into a NifTI statistical map by replacing labels with the corresponding value from the cluster
        centroids, converts the NifTI stastical map into fsLR (surface) space (using ``neuromaps.transforms.mni152_to_fslr``),
        which also turns it into a tuple-of-nib.GiftiImage that can be plotted using ``surfplot.plotting.Plot``.
        If ``self.caps2niftis()`` was used to convert the parcellation into a NifTI statistical map and an external
        tool such as Connectome Workbench was used to convert these NifTI files into GifTI files in fsLR (surface)
        space, then the ``fslr_giftis_dict`` parameter can be used for plotting. Here, ``neuromaps.transforms.fslr_to_fslr``
        is used to convert the image into a `tuple-of-nib.GiftiImage` form that can be plotted by ``surfplot.plotting.Plot``.
        Below is the internal function that converts the cluster centroids into a NifTI statistical map.
        ::

            def _cap2statmap(atlas_file, cap_vector, fwhm):
                atlas = nib.load(atlas_file)
                atlas_fdata = atlas.get_fdata()
                # Get array containing all labels in atlas to avoid issue if the first non-zero atlas label is not 1
                target_array = sorted(np.unique(atlas_fdata))
                for indx, value in enumerate(cap_vector, start=1):
                    atlas_fdata[np.where(atlas_fdata == target_array[indx])] = value
                stat_map = nib.Nifti1Image(atlas_fdata, atlas.affine, atlas.header)

        Parameters
        ----------
        output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
            Directory to save plots to. The directory will be created if it does not exist. If None, plots will not
            be saved. Outputs as png file.

        suffix_title : :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot as well as the name of the saved file if ``output_dir`` is provided.

        show_figs : :obj:`bool`, default=True
            Whether to display figures.

        fwhm : :obj:`float` or :obj:`None`, defualt=None
            Strength of spatial smoothing to apply (in millimeters) to the statistical map prior to interpolating
            from MNI152 space to fsLR surface space. Uses nilearn's ``image.smooth``.
            Note, this can assist with coverage issues in the plot.

        fslr_density : {"4k", "8k", "32k", "164k"}, default="32k"
            Density of the fsLR surface when converting from MNI152 space to fsLR surface. Options are "32k" or
            "164k". If using ``fslr_giftis_dict`` options are "4k", "8k", "32k", and "164k".

        method : {"linear", "nearest"}, default="linear"
            Interpolation method to use when converting from MNI152 space to fsLR surface or from fsLR to fsLR. Options
            are "linear" or "nearest".

        save_stat_maps : :obj:`bool`, default=False
            If True, saves the statistical map for each CAP for all groups as a Nifti1Image if ``output_dir`` is
            provided.

             .. versionchanged:: 0.16.0 changed from ``save_stat_map`` to ``save_stat_maps``.

        fslr_giftis_dict : :obj:`dict` or :obj:`None`, default=None
            Dictionary specifying precomputed GifTI files in fsLR space for plotting stat maps. This parameter
            should be used if the statistical CAP NIfTI files (can be obtained using ``self.caps2niftis()``) were
            converted to GifTI files using a tool such as Connectome Workbench.The dictionary structure is:
            ::

                {
                    "GroupName": {
                        "CAP-Name": {
                            "lh": "path/to/left_hemisphere_gifti",
                            "rh": "path/to/right_hemisphere_gifti"
                        }
                    }
                }

            GroupName can be "All Subjects" or any specific group name. CAP-Name is the name of the CAP. This
            parameter allows plotting without re-running the analysis. Initialize the CAP class and use this method
            if using this parameter.

        knn_dict : :obj:`dict[str, int | bool]`, default=None
            Use KNN (k-nearest neighbors) interpolation to fill in non-background values that are assigned zero. This is
            primarily used as a fix for when a custom parcellation does not project well from volumetric to surface
            space. This method involves resampling the Schaefer parcellation, a volumetric parcellation that projects
            well onto surface space, to the target parcellation specified in the "maps" sub-key in ``self.parcel_approach``.
            The background indices are extracted from the Schaefer parcellation, and these indices are used to obtain
            the non-background indices (parcels) that are set to zero in the target parcellation.

            These indices are then iterated over, and the zero values are replaced with the value of the nearest neighbor,
            determined by the sub-key "k". The dictionary contains the following sub-keys:

            - "k" : An integer that determines the number of nearest neighbors to consider, with the majority vote determining the new value. If not specified, the default is 1.
            - "resolution_mm" : An integer (1 or 2) that determines the resolution of the Schaefer parcellation. If not specified, the default is 1.
            - "remove_subcortical" : A list or array of label ids as integers of the subcortical regions in the parcellation.

            This method is applied before the `fwhm` method.

        kwargs : :obj:`dict`
            Additional parameters to pass to modify certain plot parameters. Options include:

            - dpi : :obj:`int`, default=300
                Dots per inch for the plot.
            - title_pad : int, default=-3
                Padding for the plot title.
            - cmap : :obj:`str` or :obj:`callable`, default="cold_hot"
                Colormap to be used for the plot. For this parameter, you can use premade color palettes or create
                custom ones. Below is a list of valid options:

                - Strings to call ``nilearn.plotting.cm._cmap_d`` fuction.
                - ``matplotlib.colors.LinearSegmentedColormap`` to generate custom colormaps.
            - cbar_kws : :obj:`dict`, default={"location": "bottom", "n_ticks": 3}
                Customize colorbar. Refer to ``_add_colorbars`` at for valid kwargs in ``surfplot.plotting.Plot``
                documentation listed in the Note section.
            - alpha : :obj:`float`, default=1
                Transparency level of the colorbar.
            - outline_alpha : :obj:`float`, default=1
                Transparency level of the colorbar for outline if ``as_outline`` is True.
            - zero_transparent : :obj:`bool`, default=True
                Turns vertices with a value of 0 transparent.
            - as_outline : :obj:`bool`, default=False
                Plots only an outline of contiguous vertices with the same value.
            - size : :obj:`tuple`, default=(500, 400)
                Size of the plot in pixels.
            - layout : :obj:`str`, default="grid"
                Layout of the plot.
            - zoom : :obj:`float`, default=1.5
                Zoom level for the plot.
            - views : {"lateral", "medial"} or :obj:`list[{"lateral", "medial}]`, default=["lateral", "medial"]
                Views to be displayed in the plot.
            - brightness : :obj:`float`, default=0.5
                Brightness level of the plot.
            - figsize : :obj:`tuple` or :obj:`None`, default=None
                Size of the figure.
            - scale : :obj:`tuple`, default=(2, 2)
                Scale factors for the plot.
            - surface : {"inflated", "veryinflated"}, default="inflated"
                The surface atlas that is used for plotting. Options are "inflated" or "veryinflated".
            - color_range : :obj:`tuple` or :obj:`None`, default=None
                The minimum and maximum value to display in plots. For instance, (-1,1) where minimum
                value is first. If None, the minimum and maximum values from the image will be used.
            - bbox_inches : :obj:`str` or :obj:`None`, default="tight"
                Alters size of the whitespace in the saved image.

        Returns
        -------
        `NifTI1Image`
            `NifTI` statistical map.
        `surfplot.plotting.Plot`
            An instance of `surfplot.plotting.Plot`.

        Note
        ----
        For this to work, ``parcel_approach`` must have the "maps" sub-key containing the path to the NifTI file of
        the atlas. Assumes that atlas background label is zero and atlas is in MNI space. Also assumes that the indices
        from the cluster centroids are related to the atlas by an offset of one. For instance, index 0 of the cluster
        centroid vector is the first nonzero label, which is assumed to be at the first index of the array in
        ``sorted(np.unique(atlas_fdata))``.
        """
        if not self._parcel_approach and fslr_giftis_dict is None:
            raise AttributeError("`self.parcel_approach` is None. Add parcel_approach using "
                                 "`self.parcel_approach=parcel_approach` to use this method.")

        if not hasattr(self,"_caps") and fslr_giftis_dict is None:
            raise AttributeError("Cannot plot caps since `self._caps` attribute does not exist. Run `self.get_caps()` "
                                 "first.")

        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

        # Create plot dictionary
        defaults = {"dpi": 300, "title_pad": -3, "cmap": "cold_hot", "cbar_kws":  {"location": "bottom", "n_ticks": 3},
                    "size": (500, 400), "layout": "grid", "zoom": 1.5, "views": ["lateral", "medial"], "alpha": 1,
                    "zero_transparent": True, "as_outline": False,"brightness": 0.5, "figsize": None, "scale": (2, 2),
                    "surface": "inflated", "color_range": None, "bbox_inches": "tight", "outline_alpha": 1}

        plot_dict = _check_kwargs(defaults, **kwargs)

        groups = self._caps if hasattr(self,"_caps") and fslr_giftis_dict is None else fslr_giftis_dict

        if fslr_giftis_dict is None: parcellation_name = list(self._parcel_approach)[0]

        for group in groups:
            caps = self._caps[group] if hasattr(self,"_caps") and fslr_giftis_dict is None else fslr_giftis_dict[group]
            for cap in caps:
                if fslr_giftis_dict is None:
                    stat_map = _cap2statmap(atlas_file=self._parcel_approach[parcellation_name]["maps"],
                                            cap_vector=self._caps[group][cap], fwhm=fwhm, knn_dict=knn_dict)

                # Fix for python 3.12, saving stat map so that it is path instead of a NifTi
                    try:
                        gii_lh, gii_rh = mni152_to_fslr(stat_map, method=method, fslr_density=fslr_density)
                    except TypeError:
                        # Create temp
                        temp_nifti = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
                        LG.warning("TypeError raised by neuromaps due to changes in pathlib.py in Python 3.12 "
                                   "Converting `statistical map` into a temporary nii.gz file (which will be "
                                   f"automatically deleted afterwards) [TEMP FILE: {temp_nifti.name}]")
                        # Ensure file is closed
                        temp_nifti.close()
                        # Save temporary nifti to temp file
                        nib.save(stat_map, temp_nifti.name)
                        gii_lh, gii_rh = mni152_to_fslr(temp_nifti.name, method=method, fslr_density=fslr_density)
                        # Delete
                        os.unlink(temp_nifti.name)
                else:
                    gii_lh, gii_rh = fslr_to_fslr((fslr_giftis_dict[group][cap]["lh"],
                                                   fslr_giftis_dict[group][cap]["rh"]),
                                                   target_density=fslr_density, method=method)
                # Code slightly adapted from surfplot example 2
                surfaces = fetch_fslr()

                if plot_dict["surface"] not in ["inflated", "veryinflated"]:
                    LG.warning(f"{plot_dict['surface']} is an invalid option for `surface`. Available options "
                               "include 'inflated' or 'verinflated'. Defaulting to 'inflated'.")
                    plot_dict["surface"] = "inflated"

                lh, rh = surfaces[plot_dict["surface"]]
                lh = str(lh) if not isinstance(lh, str) else lh
                rh = str(rh) if not isinstance(rh, str) else rh
                sulc_lh, sulc_rh = surfaces["sulc"]
                sulc_lh = str(sulc_lh) if not isinstance(sulc_lh, str) else sulc_lh
                sulc_rh = str(sulc_rh) if not isinstance(sulc_rh, str) else sulc_rh
                p = surfplot.Plot(lh, rh, size=plot_dict["size"], layout=plot_dict["layout"], zoom=plot_dict["zoom"],
                                  views=plot_dict["views"], brightness=plot_dict["brightness"])

                # Add base layer
                p.add_layer({"left": sulc_lh, "right": sulc_rh}, cmap="binary_r", cbar=False)

                # Check cmap
                cmap = _cmap_d[plot_dict["cmap"]] if isinstance(plot_dict["cmap"],str) else plot_dict["cmap"]
                # Add stat map layer
                p.add_layer({"left": gii_lh, "right": gii_rh}, cmap=cmap,
                            alpha=plot_dict["alpha"], color_range=plot_dict["color_range"],
                            zero_transparent=plot_dict["zero_transparent"], as_outline=False)

                if plot_dict["as_outline"] is True:
                    p.add_layer({"left": gii_lh, "right": gii_rh}, cmap="gray", cbar=False,
                                alpha=plot_dict["outline_alpha"], as_outline=True)

                # Color bar
                fig = p.build(cbar_kws=plot_dict["cbar_kws"], figsize=plot_dict["figsize"], scale=plot_dict["scale"])
                fig_name = f"{group} {cap} {suffix_title}" if suffix_title else f"{group} {cap}"
                fig.axes[0].set_title(fig_name, pad=plot_dict["title_pad"])

                if output_dir:
                    file_name = self._basename(group, cap, "surface", suffix_title, "png")

                    fig.savefig(os.path.join(output_dir, file_name), dpi=plot_dict["dpi"],
                                bbox_inches=plot_dict["bbox_inches"])
                    # Save stat map
                    if save_stat_maps:
                        stat_map_name = file_name.split("_surface")[0] + ".nii.gz"
                        nib.save(stat_map, os.path.join(output_dir, stat_map_name))

                try: plt.show(fig) if show_figs else plt.close(fig)
                except: plt.show() if show_figs else plt.close()

    def caps2radar(self,
                   output_dir: Optional[os.PathLike]=None,
                   suffix_title: Optional[str]=None,
                   show_figs: bool=True,
                   use_scatterpolar: bool=False,
                   as_html: bool=False,
                   **kwargs) -> Union[px.line_polar, go.Scatterpolar]:
        """
        **Generate Radar Plots for CAPs using Cosine Similarity**

        Calculates the cosine similarity between the High Amplitude (positive/above the mean) and Low Amplitude
        (negative/below the mean) activations of the CAP cluster centroid and each a-priori region or network in a
        parcellation. **Note** that this function assumes the mean for each ROI is 0 due to standardization.

        Cosine similarity is computed separately for high and low amplitudes by comparing them to the binary vector of
        the a-priori region, representing the region or network of interest. This provides a measure of how closely
        the CAP's positive and negative activation patterns align with each selected region.

        The process involves the following steps:

            1. **Extract Cluster Centroids**:

              - Each CAP is represented by a cluster centroid, which is a 1 x ROI (Region of Interest) vector.

            2. **Generate Binary Vectors**:

              - For each region create a binary vector (1 x ROI) where `1` indicates that the ROI is part of the specific
                region and `0` otherwise.
              - In this example, the binary vector acts as a 1D mask to isolate ROIs in the Visual Network by setting
                the corresponding indices to `1`.

                ::

                    import numpy as np

                    # Define nodes with their corresponding label IDs
                    nodes = ["LH_Vis1", "LH_Vis2", "LH_SomSot1", "LH_SomSot2",
                             "RH_Vis1", "RH_Vis2", "RH_SomSot1", "RH_SomSot2"]
                    # Binary mask for the Visual Network (Vis)
                    binary_vector = np.array([1, 1, 0, 0, 1, 1, 0, 0])

            3. **Isolate Positive and Negative Activations in CAP Centroid**:

              - Positive activations are defined as the values in the CAP centroid that are greater than zero.
                These values represent the "High Amplitude" activations for that CAP.
              - Negative activations are defined as the values in the CAP centroid that are less than zero.
                These values represent the "Low Amplitude" activations for that CAP.
              - To simplify the comparison between positive and negative activations using cosine similarity,
                the negative activations are inverted (i.e., multiplied by -1). This inversion converts the negative
                values into positive ones, allowing the cosine similarity calculation to return a positive value.
                The positive similarity score represents how closely the a-priori region aligns with both the high and
                low amplitude aspects of the CAP.

                ::

                  # Example cluster centroid for CAP 1
                  cap_1_cluster_centroid = np.array([-0.3, 1.5, 2.0, -0.2, 0.7, 1.3, -0.5, 0.4])
                  # Assign values less than 0 as 0 to isolate the high amplitude activations
                  high_amp = np.where(cap_1_cluster_centroid > 0, cap_1_cluster_centroid, 0)
                  # Assign values less than 0 as 0 to isolate the low amplitude activations; Also invert the sign
                  low_amp = high_amp = np.where(cap_1_cluster_centroid < 0, -cap_1_cluster_centroid, 0)

            4. **Calculate Cosine Similarity**:

              - Normalize the dot product by the product of the Euclidean norms of the cluster centroid and the binary
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

            5. **Generate Radar Plots of Each CAPs**:

              - Each radar plot visualizes the cosine similarity for both "High Amplitude" (positive) and
                "Low Amplitude" (negative) activations of the CAP. The cosine similarity values range from 0 to 1,
                representing how closely the a-priori region aligns with the positive and negative activations
                of the CAP centroid.

        Parameters
        ----------
        output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
            Directory to save plots to. The directory will be created if it does not exist. Outputs as png file.

        suffix_title : :obj:`str` or :obj:`None`, default=None
            Appended to the title of each plot as well as the name of the saved file if ``output_dir`` is provided.

        show_figs : :obj:`bool`, default=True
            Whether to display figures. If this function detects that it is not being ran in an interactive Python
            environment, then it uses ``plotly.offline``, creates an html file named "temp-plot.html", and opens each
            plot in the default browser.

        use_scatterpolar : :obj:`bool`, default=False
            Uses ``plotly.graph_objects.Scatterpolar`` instead of ``plotly.express.line_polar``. The primary difference
            is that ``plotly.graph_objects.Scatterpolar`` shows the scatter dots. However, this can be acheived with
            ``plotly.express.line_polar`` by setting ``mode`` to "markers+lines". There also seems to be a difference
            in default opacity behavior.

        as_html : :obj:`bool`, default=False
            When ``output_dir`` is specified, plots are saved as html images instead of png images. The advantage is
            that plotly's radar plots will retain its interactive properties, cna be opened in a browser, and also
            still be saved as a png in the browser.

        kwargs: :obj:`dict`
            Additional parameters to pass to modify certain plot parameters. Options include:

            - scale : :obj:`int`, default=2
                If ``output_dir`` provided, controls resolution of image when saving. Serves a similar purpose as dpi.
            - savefig_options : :obj:`dict[str]`, default={"width": 3, "height": 3, "scale": 1}
                If ``output_dir`` provided, controls the width (in inches), height (in inches), and scale of the
                plot. The height and width are multiplied by the dpi.
            - height : :obj:`int`, default=800
                Height of the plot. Value is multiplied by the dpi when saving.
            - width : :obj:`int`, defualt=1200
                Width of the plot. Value is multiplied by the dpi when saving.
            - line_close : :obj:`bool`, default=True
                Whether to close the lines
            - bgcolor : :obj:`str`, default="white"
                Color of the background
            - scattersize : :obj:`int`, default=8
                Controls size of the dots when markers are used.
            - connectgaps : :obj:`bool`, default=True
                If ``use_scatterpolar=True``, controls if missing values are connected.
            - linewidth : :obj:`int`, default = 2
                The width of the line connecting the values if ``use_scatterpolar=True``.
            - opacity : :obj:`float`, default=0.5,
                If ``use_scatterpolar=True``, sets the opacity of the trace.
            - fill : :obj:`str`, default="none".
                If "toself" the are of the dots and within the boundaries of the line will be filled.
            - mode : :obj:`str`, default="markers+lines",
                Determines how the trace is drawn. Can include "lines", "markers", "lines+markers", "lines+markers+text".
            - radialaxis : :obj:`dict`, default={"showline": False, "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)", "gridcolor": "rgba(0, 0, 0, 0.25)", "ticks": "outside","tickfont": {"size": 14, "color": "black"}}
                Customizes the radial axis.
            - angularaxis : :obj:`dict`, default={"showline": True, "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)", "gridcolor": "rgba(0, 0, 0, 0.25)", "tickfont": {"size": 16, "color": "black"}}
                Customizes the angular axis.
            - color_discrete_map : :obj:`dict`, default={"High Amplitude": "red", "Low Amplitude": "blue"},
                Change color of the "High Amplitude" and "Low Amplitude" groups. Must use the keys
                "High Amplitude" and "Low Amplitude" to work.
            - title_font : :obj:`dict`, default={"family": "Times New Roman", "size": 30, "color": "black"}
                Modifies the font of the title.
            - title_x : :obj:`float`, default=0.5
                Modifies x position of title.
            - title_y : :obj:`float`, default=None
                Modifies y position of title.
            - legend : :obj:`dict`, default={"yanchor": "top", "xanchor": "left", "y": 0.99, "x": 0.01,"title_font_family": "Times New Roman", "font": {"size": 12, "color": "black"}}
                Customizes the legend.
            - engine : {"kaleido", "orca"}, default="kaleido"
                Engine used for saving plots.

        Returns
        -------
        `plotly.express.line_polar`
            An instance of `plotly.express.line_polar`.
        `plotly.graph_objects.Scatterpolar`
            An instance of `plotly.graph_objects.Scatterpolar`.

        Note
        -----
        By default, this function uses "kaleido" (which is also a dependency in this package) to save plots.
        For other engines such as "orca", those packages must be installed seperately.

        **If using "Custom" parcellation approach**, the "regions" sub-key is required.

        In this code, if the ``tickvals`` or  ``range`` sub-keys in this code are not specified in the ``radialaxis``
        kwarg, then four values are shown - 0.25*(max value), 0.50*(max value), 0.75*(max value), and the max value.
        These values are also rounded to the second decimal place.

        For valid keys for ``radialaxis`` refer to plotly's documentation at
        https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.polar.radialaxis.html or
        https://plotly.com/python/reference/layout/polar/ for valid kwargs.

        For valid keys for ``angularaxis`` refer to plotly's documentation at
        https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.polar.angularaxis.html or
        https://plotly.com/python/reference/layout/polar/ for valid kwargs.

        For valid keys for ``legend`` and ``title_font``, refer to plotly's documentation at
        https://plotly.com/python/reference/layout/ for valid kwargs.

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
        if not self._parcel_approach:
            raise AttributeError("self.parcel_approach` is None. Add parcel_approach using "
                                 "`self.parcel_approach=parcel_approach` to use this method.")

        if not hasattr(self,"_caps"):
            raise AttributeError("Cannot plot caps since `self._caps` attribute does not exist. Run `self.get_caps()` "
                                 "first.")

        if not self._standardize:
            LG.warning("To better aid interpretation, the matrix subjected to kmeans clustering in "
                       "`self.get_caps` should be standardized so that each ROI in the CAP cluster centroid. "
                       "represents activation or de-activation relative to the mean.")

        defaults = {"scale": 2, "height": 800, "width": 1200, "line_close": True, "bgcolor": "white", "fill": "none",
                    "scattersize": 8, "connectgaps": True, "opacity": 0.5, "linewidth": 2,
                    "radialaxis": {"showline": False, "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)",
                                   "gridcolor": "rgba(0, 0, 0, 0.25)","ticks": "outside",
                                   "tickfont": {"size": 14, "color": "black"}},
                    "angularaxis": {"showline": True, "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)",
                                    "gridcolor": "rgba(0, 0, 0, 0.25)", "tickfont": {"size": 16, "color": "black"}},
                    "color_discrete_map": {"High Amplitude": "rgba(255, 0, 0, 1)",
                                           "Low Amplitude": "rgba(0, 0, 255, 1)"},
                    "title_font": {"family": "Times New Roman", "size": 30, "color": "black"},
                    "title_x": 0.5, "title_y": None,
                    "legend": {"yanchor": "top", "xanchor": "left", "y": 0.99, "x": 0.01,
                               "title_font_family": "Times New Roman", "font": {"size": 12, "color": "black"}},
                    "mode": "markers+lines", "engine": "kaleido"}

        plot_dict = _check_kwargs(defaults, **kwargs)

        parcellation_name = list(self.parcel_approach)[0]

        # Initialize cosine_similarity attribute
        self._cosine_similarity = {}

        # Create radar dict
        for group in self._caps:
            radar_dict = {"Regions": list(self.parcel_approach[parcellation_name]["regions"])}
            self._update_radar_dict(group, parcellation_name, radar_dict)
            self._cosine_similarity[group] = radar_dict

            for cap in self._caps[group]:
                if use_scatterpolar:
                    # Create dataframe
                    df = pd.DataFrame({"Regions": radar_dict["Regions"]})
                    df = pd.concat([df, pd.DataFrame(radar_dict[cap])], axis = 1)
                    # Initialize figure
                    fig = go.Figure(layout=go.Layout(width=plot_dict["width"], height=plot_dict["height"]))
                    regions = df["Regions"].values

                    for i in ["High Amplitude", "Low Amplitude"]:
                        values = df[i].values

                        # Add traces
                        fig.add_trace(go.Scatterpolar(
                        r=list(values),
                        theta=regions,
                        connectgaps=plot_dict["connectgaps"],
                        name=i,
                        opacity=plot_dict["opacity"],
                        marker=dict(color=plot_dict["color_discrete_map"][i],
                                    size=plot_dict["scattersize"]),
                        line = dict(color=plot_dict["color_discrete_map"][i], width=plot_dict["linewidth"])))
                else:
                    # Create dataframe
                    n = len(radar_dict["Regions"])
                    df = pd.DataFrame({"Regions": radar_dict["Regions"]*2,
                                       "Amp": radar_dict[cap]["High Amplitude"] + radar_dict[cap]["Low Amplitude"]})
                    df["Groups"] = ["High Amplitude"]*n + ["Low Amplitude"]*n

                    fig = px.line_polar(df, r=df["Amp"].values, theta="Regions", line_close=plot_dict["line_close"],
                                        color=df["Groups"].values, width=plot_dict["width"],
                                        height=plot_dict["height"], category_orders={"Regions": df["Regions"]},
                                        color_discrete_map = plot_dict["color_discrete_map"])

                if use_scatterpolar:
                    fig.update_traces(fill=plot_dict["fill"], mode=plot_dict["mode"])
                else:
                    fig.update_traces(fill=plot_dict["fill"], mode=plot_dict["mode"],
                                      marker=dict(size=plot_dict["scattersize"]))

                # Set max value
                if "tickvals" not in plot_dict["radialaxis"] and "range" not in plot_dict["radialaxis"]:
                    if use_scatterpolar: max_value = max(df[["High Amplitude", "Low Amplitude"]].max())
                    else: max_value = df["Amp"].max()

                    default_ticks = [max_value/4, max_value/2, 3*max_value/4, max_value]
                    plot_dict["radialaxis"]["tickvals"] = [round(x, 2) for x in default_ticks]

                title_text = f"{group} {cap} {suffix_title}" if suffix_title else f"{group} {cap}"

                # Customize
                fig.update_layout(
                    title=dict(text=title_text, font=plot_dict["title_font"]),
                    title_x=plot_dict["title_x"],
                    title_y=plot_dict["title_y"],
                    legend=plot_dict["legend"],
                    legend_title_text="Cosine Similarity",
                    polar=dict(
                        bgcolor=plot_dict["bgcolor"],
                        radialaxis=plot_dict["radialaxis"],
                        angularaxis=plot_dict["angularaxis"]
                    ))

                if show_figs:
                    if bool(getattr(sys, "ps1", sys.flags.interactive)): fig.show()
                    else: pyo.plot(fig, auto_open=True)

                if output_dir:
                    if not os.path.exists(output_dir): os.makedirs(output_dir)

                    file_name = self._basename(group, cap, "radar", suffix_title, "png")

                    if not as_html:
                        fig.write_image(os.path.join(output_dir,file_name), scale=plot_dict["scale"],
                                        engine=plot_dict["engine"])
                    else:
                        file_name = file_name.replace(".png", ".html")
                        fig.write_html(os.path.join(output_dir,file_name))

    def _update_radar_dict(self, group, parcellation_name, radar_dict):
        for cap in self._caps[group]:
            cap_vector = self._caps[group][cap]
            radar_dict[cap] = {"High Amplitude": [], "Low Amplitude": []}

            for region in radar_dict["Regions"]:
                if parcellation_name == "Custom":
                    lh = self._parcel_approach[parcellation_name]["regions"][region]["lh"]
                    rh = self._parcel_approach[parcellation_name]["regions"][region]["rh"]
                    indxs = lh + rh
                else:
                    indxs = np.array([value for value, node in
                                      enumerate(self._parcel_approach[parcellation_name]["nodes"])
                                      if region in node])

                # Create mask to set ROIs not in regions to zero and ROIs in regions as 1
                binary_vector = np.zeros_like(cap_vector)
                binary_vector[indxs] = 1

                # Calculate binary norm
                norm_binary_vector = np.linalg.norm(binary_vector)

                # Get high and low amplitudes
                high_amp = np.where(cap_vector > 0, cap_vector, 0)
                # Invert vector for low_amp so that cosine similarity is positive
                low_amp = np.where(cap_vector < 0, -cap_vector, 0)
                vecs = {"High Amplitude": high_amp, "Low Amplitude": low_amp}

                warning_msg = (f"[GROUP: {group} | REGION: {region} | CAP: {cap}] - "
                               "Division by zero error when calculating cosine similarity for the "
                               "{0} activations. Setting cosine similarity to zero.")

                for vec in vecs:
                    dot_product = np.dot(vecs[vec], binary_vector)
                    norm_vec = np.linalg.norm(vecs[vec])

                    try:
                        cosine_similarity = dot_product/(norm_vec * norm_binary_vector)
                    except ZeroDivisionError:
                        LG.warning(warning_msg.format(vec))
                        cosine_similarity = 0

                    # Store value in dict
                    radar_dict[cap][vec].append(cosine_similarity)
