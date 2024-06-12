
import collections, itertools, os, re, sys, warnings
from typing import Union, Literal, List, Dict, Optional
import numpy as np, nibabel as nib, matplotlib.pyplot as plt, pandas as pd, seaborn, surfplot
import plotly.express as px, plotly.graph_objects as go, plotly.offline as pyo
from kneed import KneeLocator
from joblib import cpu_count, delayed, Parallel
from nilearn.plotting.cm import _cmap_d
from neuromaps.transforms import mni152_to_fslr, fslr_to_fslr
from neuromaps.datasets import fetch_fslr
from sklearn.cluster import KMeans
from .._utils import (_CAPGetter, _cap2statmap, _check_kwargs, _convert_pickle_to_dict,
                      _check_parcel_approach, _run_kmeans)

class CAP(_CAPGetter):
    """
    **Co-Activation Patterns (CAPs) Class**

    Initializes the CAPs (Co-activation Patterns) class.

    Parameters
    ----------
        parcel_approach : Dict[Dict], default=None
            The approach used to parcellate BOLD images. This should be a nested dictionary with the first key being
            the atlas name. The sub-keys should include:

             - ``maps``: Directory path to the location of the parcellation file.
             - ``nodes``: A list of node names in the order of the label IDs in the parcellation.
             - ``regions``: The regions or networks in the parcellation.

            If the "Schaefer" or "AAL" option was used in the ``TimeSeriesExtractor`` class, you can initialize
            the ``TimeSeriesExtractor`` class with the ``parcel_approach``  that was initially used, then set this
            parameter to ``TimeSeriesExtractor.parcel_approach``. For this parameter, only "Schaefer", "AAL", and
            "Custom" are supported. **Note**, this is not needed for using ``self.get_caps``; however, for plotting
            it will be needed. ``self.parcel_approach=parcel_approach`` can be used.
        n_clusters : Union[int, List[int]], default=5
            The number of clusters to use. Can be a single integer or a list of integers
            (if ``cluster_selection_method`` is not None).
        cluster_selection_method : str or None, default=None
            Method to find the optimal number of clusters. Options are "silhouette" or "elbow".
        groups : Dict[str, List] or None, default=None
            A mapping of group names to subject IDs. Each group contains subject IDs for separate CAP analysis.
            If None, CAPs are not separated by group. The structure should be as follows:
            ::

                {
                    "GroupName1": ["1", "2", "3"],
                    "GroupName2": ["4", "5", "6"]
                }

            **Note**, if left as None, when ``get_caps()`` is used, the group name will default to "All Subjects" and
            will contain a list will all subject IDs. The structure will be as follows
            ::

                {
                    "All Subjects": ["1", "2", "3", "4", "5", "6"]
                }

    Property
    --------
        n_clusters : int or List[int]
            A single integer or list of integers if ``cluster_selection_method`` is not None) that will used for
            k-means clustering.
        groups : Dict[str, List] or None:
            A mapping og groups names to and subject IDS. **Note**, if left as None, when ``get_caps()`` is used, the
            group name will default to "All Subjects" and will contain a list will all subject IDs.
        cluster_selection_method : str or None:
            The cluster selection method to use to identify the optimal cluster.
        parcel_approach : Dict[str, Dict]
            Nested dictionary containing information about the parcellation. Can also be used as a setter. If "Schaefer"
            or ""AAL" was specified during initialization of the ``TimeseriesExtractor`` class, then `Nilearn's`
            ``datasets.fetch_atlas_schaefer_2018` and ``datasets.fetch_atlas_aal`` will be used to obtain the "maps"
            and the "labels"/"nodes". Then string splitting is used on the "labels"/"nodes" to obtain the
            "regions"/"networks:
            ::

                {
                    "Schaefer":
                    {
                        "maps": "path/to/parcellation.nii.gz",
                        "nodes": ["LH_Vis1", "LH_SomSot1", "RH_Vis1", "RH_Somsot1"],
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
            such as ``maps``, ``nodes``, and ``regions``. Unlike "Schaefer" and "AAL", "region" must be a nested
            dictionary specifying the name of the region as the first level key and the indices in the "nodes" list
            belonging to the "lh" and "rh" for that region. Refer to the structure example for "Custom" in the Notes
            for ``parcel_approach`` section below.

        n_cores : int
            Number of cores to use for multiprocessing with ``joblib``.  Is None until ``self.get_caps`` is used
            and ``n_cores`` is specified.
        runs : int or List[int]
            The runs used for the CAPs analysis. Is None until ``self.get_caps`` is used and ``runs``is specified.
        caps : Dict[str, Dict[np.array]]
            The extracted cluster centroids, representing each CAP from the K-means model. It is a nested
            dictionary containing the group name, CAP names, and 1D numpy array. *Note*, if no groups were
            specified, the default group name is "All Subjects".  Is None until ``self.get_caps`` is used.
            The structure is as follows:
            ::

                {
                    "GroupName": {
                        "CAP-1": np.array([...]) # 1 x ROI array,
                        "CAP-2": np.array([...]) # 1 x ROI array,
                    }

                }

        kmeans : Dict[str, ``sklearn.cluster.KMeans``]
            Dictionary containing the ``KMeans`` model used for each group. If ``cluster_selection__method`` is not
            None, the ``KMeans`` model will be the optimal model.  Is None until ``self.get_caps`` is used.
            The structure is as follows:
            ::

                {
                    "GroupName": sklearn.cluster.KMeans
                }

        silhouette_scores : Dict[str, Dict[str, float]]
            If ``cluster_selection_method`` is "silhouette", this property will be a nested dictionary containing
            the group name, cluster number, and silhouette score.  Is None until ``self.get_caps`` is used.
            The structure is as follows:
            ::

                {
                    "GroupName": {
                        "2": float,
                        "3": float,
                        "4": float
                    }
                }

        inertia : Dict[str, Dict[str, float]]
            If ``cluster_selection_method`` is "elbow", this property will be a nested dictionary containing the
            group name, cluster number, and inertia.  Is None until ``self.get_caps`` is used.
            The structure is as follows:
            ::

                {
                    "GroupName": {
                        "2": float,
                        "3": float,
                        "4": float
                    }
                }

        optimal_n_clusters : Dict[str, Dict[int]]
            If ``cluster_selection_method`` is not None, this property is a nested dictionary containing the group
            name and the optimal number of clusters. The structure is as follows:
            ::

                {
                    "GroupName": int
                }

        standardize : bool
            Boolean denoting whether the features of the concatenated timeseries data was z-scored. Is None until
            ``self.get_caps`` is used.
        epsilon : float
            A small number to added to the denominator when z-scoring for numerical stability.
        means : Dict[str, np.array]
            If ``standardize`` is True, this property is a nested dictionary containing the group names and a
            numpy array (participants x TR) x ROIs of the means of the features.  Is None until ``self.get_caps`` is
            used. The structure is as follows:
            ::

                {
                    "GroupName": np.array([...])  # Dimensions: 1 x ROIs
                }

        stdev : Dict[str, np.array]
            If ``standardize`` is True, this property is a nested dictionary containing the group names and a numpy
            array (participants x TR) x ROIs of the sample standard deviation of the features.  Is None until
            ``self.get_caps`` is used. The structure is as follows:
            ::

                {
                    "GroupName": np.array([...])  # Dimensions: 1 x ROIs
                }

        concatenated_timeseries : Dict[str, np.array]
            Nested dictionary containing the group name and their respective concatenated numpy arrays
            (participants x TR) x ROIs. Is None until ``self.get_caps`` is used. The structure is as follows:
            ::

                {
                    "GroupName": np.array([...])  # Dimensions: (participants x TR) x ROIs
                }

        region_caps : Dict[str, np.array]:
            If ``visual_scope`` set to "regions" in ``self.caps2plot``, this property is a nested dictionary containing
            the group name, CAP names, and numpy array (1 x region) of the averaged z-score value for each region.
            The structure is as follows:
            ::

                {
                        "GroupName": {
                            "CAP-1": np.array([...]) # 1 x region array,
                            "CAP-2": np.array([...]) # 1 x region array,
                        }

                    }

        outer_products : Dict[str, Dict[str, np.array]]
            If ``plot_options`` set to " outer product", this property is a nested dictionary containing the group name,
            CAP names, and numpy array (ROI x ROI) of the outer product. The structure is as follows:
            ::

                {
                        "GroupName": {
                            "CAP-1": np.array([...]) # ROI x ROI array,
                            "CAP-2": np.array([...]) # ROI x ROI array,
                        }

                    }

    Note
    ----
    **If no groups were specified, the default group name will be "All Subjects".**

    **If using a "Custom" parcellation approach**, ensure each node in your dataset includes both left (lh) and right
    (rh) hemisphere versions. This function assumes that the background label is "zero". Do not add a background label
    in the "nodes" or "networks" key; the zero index should correspond to the first ID that is not zero.

    - ``maps``: Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g.,
      .nii for NIfTI files).
    - ``nodes``: List of all node labels used in your study, arranged in the exact order they correspond to indices in
      your parcellation files. Each label should match the parcellation index it represents. For example, if the
      parcellation label "1" corresponds to the left hemisphere visual cortex area 1, then "LH_Vis1" should occupy
      the 0th index in this list. This ensures that data extraction and analysis accurately reflect the anatomical
      regions intended.
    - ``regions``: Dictionary defining major brain regions. Each region should list node indices under "lh" and "rh" to
      specify left and right hemisphere nodes.

    **Different sub-keys are required depending on the function used.**

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
    def __init__(self, parcel_approach: Dict[str, Dict]=None, n_clusters: Union[int, List[int]]=5,
                 cluster_selection_method: Literal["elbow","silhouette"]=None, groups: Dict[str, List]=None) -> None:
        # Ensure all unique values if n_clusters is a list
        self._n_clusters = n_clusters if isinstance(n_clusters, int) else sorted(list(set(n_clusters)))
        self._cluster_selection_method = cluster_selection_method

        if isinstance(n_clusters, list):
            self._n_clusters =  self._n_clusters[0] if all([isinstance(self._n_clusters, list),
                                                            len(self._n_clusters) == 1]) else self._n_clusters
            # Raise error if n_clusters is a list and no cluster selection method is specified
            if all([len(n_clusters) > 1, self._cluster_selection_method is None]):
                raise ValueError("`cluster_selection_method` cannot be None since n_clusters is a list.")

        # Raise error if silhouette_method is requested when n_clusters is an integer
        if all([self._cluster_selection_method is not None, isinstance(self._n_clusters, int)]):
            raise ValueError("`cluster_selection_method` only valid if n_clusters is a range of unique integers.")

        self._groups = groups
        # Raise error if self groups is not a dictionary
        if self._groups:
            if not isinstance(self._groups, dict):
                raise TypeError("""
                                `groups` must be a dictionary where the keys are the group names and the items
                                correspond to subject ids in the groups.
                                """)

            for group_name in self._groups:
                assert len(self._groups[group_name]) > 0, f"{group_name} has zero subject ids."

            # Convert ids to strings
            for group in self._groups:
                self._groups[group] = [str(subj_id) if not isinstance(subj_id,str)
                                       else subj_id for subj_id in self._groups[group]]

        if parcel_approach is not None:
            _check_parcel_approach(parcel_approach=parcel_approach, call="CAP")

        self._parcel_approach = parcel_approach

    def get_caps(self, subject_timeseries: Union[Dict[str, Dict[str, np.ndarray]], str],
                 runs: Optional[Union[int, List[int]]]=None, random_state: Optional[int]=None,
                 init: Union[np.array, Literal["k-means++", "random"]]="k-means++",
                 n_init: Union[Literal["auto"],int]='auto',
                 max_iter: int=300, tol: float=0.0001, algorithm: Literal["lloyd", "elkan"]="lloyd",
                 show_figs: bool=False,
                 output_dir: Optional[Union[str, os.PathLike]]=None, standardize: bool=True,
                 epsilon: Union[int,float]=0, n_cores: Optional[int]=None, **kwargs) -> plt.figure:
        """
        **Perform K-Means Clustering to Generate CAPs**

        Concatenates the timeseries of each subject into a single numpy array with dimensions (participants x TRs) x
        ROI and performs k-means clustering on the concatenated data. **Note, this uses `Scikit's` ``KMeans`` so the
        distance metric used is euclidean. Additionally, the elbow method is determined using ``KneeLocator`` from
        the ``kneed`` package and the silhouette scores are calculates with `Scikit's` ``silhouette_score``.

        Parameters
        ----------
            subject_timeseries : Dict[str, Dict[str, np.ndarray]] or str
                Path of the pickle file containing the nested subject timeseries dictionary saved by the
                ``TimeSeriesExtractor`` class or the nested subject timeseries dictionary produced by the
                ``TimeseriesExtractor`` class. The first level of the nested dictionary must consist of the subject ID
                as a string, the second level must consist of the run numbers in the form of 'run-#' (where # is the
                corresponding number of the run), and the last level must consist of the timeseries (as a numpy array)
                associated with that run. The structure is as follows:
                    ::

                        subject_timeseries = {
                                "101": {
                                    "run-0": np.array([timeseries]), # 2D array
                                    "run-1": np.array([timeseries]), # 2D array
                                    "run-2": np.array([timeseries]), # 2D array
                                },
                                "102": {
                                    "run-0": np.array([timeseries]), # 2D array
                                    "run-1": np.array([timeseries]), # 2D array
                                }
                            }

            runs : int or List[int] or None, default=None
                The run numbers to perform the CAPs analysis with. If None, all runs in the subject timeseries will be
                concatenated into a single dataframe and subjected to k-means clustering.
            random_state : int or None, default=None
                The random state to use for ``scikit-learn`` ``KMeans`` function.
            init : "k-means++", "random", or array, default="k-means++"
                Method for choosing initial cluster centroid. Refer to ``scikit-learn` ``KMeans`` documentation for more
                information.
            n_init : "auto" or int, default="auto"
                Number of times KMeans is ran with different initial clusters. The model with lowest inertia from these
                runs will be selected. Refer to ``scikit-learn` ``KMeans`` documentation for more information.
            max_iter : int, default=300
                Maximum number of iterations for a single run of ``KMeans``.
            tol : float, default=1e-4,
                Stopping criterion if the change in inertia is below this value, assuming ``max_iter`` has not been
                reached.
            algorithm : "lloyd or "elkan", default="lloyd"
                The type of algorithm to use. Refer to ``scikit-learn` ``KMeans`` documentation for more information.
            show_figs : bool, default=False
                Display the plots of inertia scores for all groups if ``cluster_selection_method`` is set to "elbow".
            output_dir : `os.Pathlike` or None, default=None
                Directory to save plot to if ``cluster_selection_method`` is set to "elbow". The directory will be
                created if it does not exist.
            standardize : bool, default=True
                Whether to z-score the features of the concatenated timeseries data. The sample standard deviation will
                be used, meaning ``n-1`` in the denominator.
            epsilon : int or float, default=0
                A small number to add to the denominator when z-scoring for numerical stability.
            n_cores : int or None, default=None
                The number of CPU cores to use for multiprocessing, with ``joblib``, to run multiple ``KMeans`` models
                if ``cluster_selection_method`` is not None.
            kwargs : Dict
                Dictionary to adjust certain parameters related to ``cluster_selection_method`` when set to "elbow".
                Additional parameters include:

                - S : int, default=1
                    Adjusts the sensitivity of finding the elbow. Larger values are more conservative and less
                    sensitive to small fluctuations. This package uses ``KneeLocator`` from the ``kneed`` package to
                    identify the elbow. Default is 1.
                - dpi : int, default=300
                    Adjusts the dpi of the elbow plot. Default is 300.
                - figsize : Tuple, default=(8,6)
                    Adjusts the size of the elbow plots.

        Returns
        -------
            `matplotlib.Figure`
                An instance of a `matplotlib` figure.
        """
        if n_cores and self._cluster_selection_method is not None:
            if n_cores > cpu_count():
                raise ValueError(f"""
                                 More cores specified than available -
                                 Number of cores specified: {n_cores};
                                 Max cores available: {cpu_count()}.
                                 """)
            if isinstance(n_cores, int): self._n_cores = n_cores
            else: raise ValueError("`n_cores` must be an integer.")
        else:
            if n_cores and self._cluster_selection_method is None:
                warnings.warn("Multiprocessing will not run since `cluster_selection_method` is None.")
            self._n_cores = None

        if runs:
            if isinstance(runs,int): runs = list(runs)

        self._runs = runs if runs else "all"
        self._standardize = standardize
        self._epsilon = epsilon

        if isinstance(subject_timeseries, str) and "pkl" in subject_timeseries:
            subject_timeseries = _convert_pickle_to_dict(pickle_file=subject_timeseries)

        self._concatenated_timeseries = self._get_concatenated_timeseries(subject_timeseries=subject_timeseries,
                                                                          runs=runs)

        if self._cluster_selection_method == "silhouette":
            self._perform_silhouette_method(random_state=random_state, init=init, n_init=n_init, max_iter=max_iter,
                                            tol=tol, algorithm=algorithm)
        elif self._cluster_selection_method == "elbow":
            self._perform_elbow_method(random_state=random_state, show_figs=show_figs, output_dir=output_dir,
                                       init=init, n_init=n_init, max_iter=max_iter, tol=tol, algorithm=algorithm,
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

    def _perform_silhouette_method(self, random_state, init, n_init, max_iter, tol, algorithm) -> None:
        # Initialize attribute
        self._silhouette_scores = {}
        self._optimal_n_clusters = {}
        self._kmeans = {}

        for group in self._groups:
            self._silhouette_scores[group] = {}
            if self._n_cores is None:
                for n_cluster in self._n_clusters:
                    silhouette_dict = _run_kmeans(n_cluster=n_cluster, random_state=random_state, init=init,
                                                  n_init=n_init, max_iter=max_iter, tol=tol, algorithm=algorithm,
                                                  concatenated_timeseries=self._concatenated_timeseries[group],
                                                  method="elbow")
                    self._silhouette_scores[group].update(silhouette_dict)
            else:
                with Parallel(n_jobs=self._n_cores) as parallel:
                    outputs = parallel(delayed(_run_kmeans)(n_cluster, random_state, init, n_init, max_iter, tol,
                                                            algorithm, self._concatenated_timeseries[group],
                                                            "silhouette") for n_cluster in self._n_clusters)
                for output in outputs:
                    self._silhouette_scores[group].update(output)

            # Get max score
            self._optimal_n_clusters[group] = max(self._silhouette_scores[group],key=self._silhouette_scores[group].get)
            self._kmeans[group] = KMeans(n_clusters=self._optimal_n_clusters[group], random_state=random_state,
                                         init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                                         algorithm=algorithm).fit(self._concatenated_timeseries[group])
            print(f"Optimal cluster size for {group} is {self._optimal_n_clusters[group]}.")

    def _perform_elbow_method(self, random_state, show_figs, output_dir, init, n_init, max_iter, tol, algorithm,
                              **kwargs) -> None:
        # Initialize attribute
        self._inertia = {}
        self._optimal_n_clusters = {}
        self._kmeans = {}

        knee_dict = {"S": kwargs["S"] if "S" in kwargs else 1}

        for group in self._groups:
            self._inertia[group] = {}
            if self._n_cores is None:
                for n_cluster in self._n_clusters:
                    inertia_dict = _run_kmeans(n_cluster=n_cluster, random_state=random_state, init=init, n_init=n_init,
                                               max_iter=max_iter, tol=tol, algorithm=algorithm,
                                               concatenated_timeseries=self._concatenated_timeseries[group],
                                               method="elbow")
                    self._inertia[group].update(inertia_dict)
            else:
                with Parallel(n_jobs=self._n_cores) as parallel:
                    outputs = parallel(delayed(_run_kmeans)(n_cluster, random_state, init, n_init, max_iter,
                                                            tol, algorithm,self._concatenated_timeseries[group],
                                                            "elbow") for n_cluster in self._n_clusters)
                for output in outputs:
                    self._inertia[group].update(output)

            # Get optimal cluster size
            kneedle = KneeLocator(x=list(self._inertia[group]),
                                                        y=list(self._inertia[group].values()),
                                                        curve='convex',
                                                        direction='decreasing', S=knee_dict["S"])

            self._optimal_n_clusters[group] = kneedle.elbow
            if not self._optimal_n_clusters[group]:
                warnings.warn("""
                               No elbow detected so optimal cluster size is None. Try adjusting the sensitivity
                               parameter, `S`, to increase or decrease sensitivity (higher values are less sensitive),
                               expanding the list of clusters to test, or setting `cluster_selection_method` to
                               'silhouette'.
                               """)
            else:
                self._kmeans[group] = KMeans(n_clusters=self._optimal_n_clusters[group], random_state=random_state,
                                             init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                                             algorithm=algorithm).fit(self._concatenated_timeseries[group])
                print(f"Optimal cluster size for {group} is {self._optimal_n_clusters[group]}.\n")

                if show_figs or output_dir is not None:
                    # Defaults
                    defaults = {"dpi": 300,"figsize": (8,6)}

                    plot_dict = _check_kwargs(defaults, **kwargs)

                    plt.figure(figsize=plot_dict["figsize"])
                    inertia_values = [y for x,y in self._inertia[group].items()]
                    plt.plot(self._n_clusters, inertia_values)
                    plt.vlines(self._optimal_n_clusters[group], plt.ylim()[0], plt.ylim()[1],
                               linestyles="--", label="elbow")
                    plt.legend(loc="best")
                    plt.xlabel("K")
                    plt.ylabel("Inertia")
                    plt.title(group)

                    if output_dir:
                        if not os.path.exists(output_dir): os.makedirs(output_dir)
                        plt.savefig(os.path.join(output_dir,f"{group.replace(' ','_')}_elbow.png"),
                                    dpi=plot_dict["dpi"])

                    if show_figs is False: plt.close()
                    else: plt.show()

    def _create_caps_dict(self) -> None:
        # Initialize dictionary
        self._caps = {}
        for group in self._groups:
            self._caps[group] = {}
            cluster_centroids = zip([num for num in range(1,len(self._kmeans[group].cluster_centers_)+1)],
                                    self._kmeans[group].cluster_centers_)
            self._caps[group].update({f"CAP-{state_number}": state_vector
                                        for state_number, state_vector in cluster_centroids})

    def _get_concatenated_timeseries(self, subject_timeseries, runs) -> None:
        # Create dictionary for "All Subjects" if no groups are specified to reuse the same loop instead of having to
        # create logic for grouped and non-grouped version of the same code
        if not self._groups: self._groups = {"All Subjects": [subject for subject in subject_timeseries]}

        concatenated_timeseries = {group: {} for group in self._groups}

        self._generate_lookup_table()

        self._mean_vec = {group: {} for group in self._groups}
        self._stdev_vec = {group: {} for group in self._groups}

        for subj_id, group in self._subject_table.items():
            requested_runs = [f"run-{run}" for run in runs] if runs else list(subject_timeseries[subj_id])
            subject_runs = [subject_run for subject_run in subject_timeseries[subj_id]
                            if subject_run in requested_runs]
            if len(subject_runs) == 0:
                warnings.warn(f"""
                              Skipping subject {subj_id} since they do not have the
                              requested run numbers {','.join(requested_runs)}
                              """)
                continue
            for curr_run in subject_runs:
                if len(concatenated_timeseries[group]) == 0:
                    if subj_id in list(set(self._groups[group])):
                        concatenated_timeseries[group] = subject_timeseries[subj_id][curr_run]
                else:
                    if subj_id in list(set(self._groups[group])):
                        concatenated_timeseries[group] = np.vstack([concatenated_timeseries[group],
                                                                    subject_timeseries[subj_id][curr_run]])
        # Standardize
        if self._standardize:
            for group in self._groups:
                self._mean_vec[group] = np.mean(concatenated_timeseries[group], axis=0)
                self._stdev_vec[group] = np.std(concatenated_timeseries[group], ddof=1, axis=0)
                concatenated_timeseries[group] = (concatenated_timeseries[group] - self._mean_vec[group])/(self._stdev_vec[group] + self._epsilon)

        return concatenated_timeseries

    def _generate_lookup_table(self) -> None:
        self._subject_table = {}
        for group in self._groups:
            for subj_id in self._groups[group]:
                if subj_id in self._subject_table:
                    warnings.warn(f"""
                                  Subject: {subj_id} appears more than once, only including the first instance
                                  of this subject in the analysis.
                                  """)
                else:
                    self._subject_table.update({subj_id : group})

    def calculate_metrics(self, subject_timeseries: Union[Dict[str, Dict[str, np.ndarray]], str],
                          tr: Optional[float]=None, runs: Optional[Union[int]]=None, continuous_runs: bool=False,
                          metrics: Union[str, List[str]]=["temporal fraction", "persistence", "counts", "transition frequency"],
                          return_df: bool=True, output_dir: Optional[Union[str, os.PathLike]]=None,
                          prefix_file_name: Optional[str]=None) -> Dict[str, pd.DataFrame]:
        """
        **Get CAPs metrics**

        Creates a single pandas DataFrame containing CAP metrics for all participants, as described in Liu et al.,
        2018 and Yang et al., 2021. The metrics include:

         - `temporal fraction`: The proportion of total volumes spent in a single CAP over all volumes in a run.
         - `persistence`: The average time spent in a single CAP before transitioning to another CAP
           (average consecutive/uninterrupted time).
         - `counts`: The frequency of each CAP observed in a run.
         - `transition frequency`: The number of switches between different CAPs across the entire run.

        Parameters
        ----------
            subject_timeseries : Dict[str, Dict[str, np.ndarray]] or str
                Path of the pickle file containing the nested subject timeseries dictionary saved by the
                ``TimeSeriesExtractor`` class or the nested subject timeseries dictionary produced by the
                ``TimeseriesExtractor`` class. The first level of the nested dictionary must consist of the subject ID
                as a string, the second level must consist of the run numbers in the form of 'run-#' (where # is the
                corresponding number of the run), and the last level must consist of the timeseries (as a numpy array)
                associated with that run. The structure is as follows:
                    ::

                        subject_timeseries = {
                                "101": {
                                    "run-0": np.array([timeseries]), # 2D array
                                    "run-1": np.array([timeseries]), # 2D array
                                    "run-2": np.array([timeseries]), # 2D array
                                },
                                "102": {
                                    "run-0": np.array([timeseries]), # 2D array
                                    "run-1": np.array([timeseries]), # 2D array
                                }
                            }

            tr : float or None, default=None
                The repetition time (TR). If provided, persistence will be calculated as the average uninterrupted time
                spent in each CAP. If not provided, persistence will be calculated as the average uninterrupted volumes
                (TRs) spent in each state.
            runs : int or List[int], default=None
                The run numbers to calculate CAP metrics for. If None, CAP metrics will be calculated for each run.
            continuous_runs : bool, default=False
                If True, all runs will be treated as a single, uninterrupted run.
            metrics : str or List[str], default=["temporal fraction", "persistence", "counts", "transition frequency"]
                The metrics to calculate. Available options include "temporal fraction", "persistence", "counts", and
                "transition frequency".
            return_df : str, default=True
                If True, returns the dataframe.
            output_dir : `os.Pathlike` or None, default=None
                Directory to save dataframe to. The directory will be created if it does not exist. If None, dataframe
                will not be saved.
            prefix_file_name : str or None, default=None
                Will serve as a prefix to append to the saved file names for the dataframes, if ``output_dir`` is
                provided.

        Returns
        -------
            Dict[str, ``pd.DataFrame``]
                Dictionary containing ``pandas`` DataFrames - one for each requested metric.

        Note
        ----
        The presence of 0 for specific CAPs in the "temporal fraction", "persistence", or "counts" dataframes indicates
        that the participant had zero instances of a specific CAP.

        References
        ----------
        Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals.
        NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

        Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation
        patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia.
        NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

        """
        if not hasattr(self,"_kmeans"):
            raise AttributeError("""
                                 Cannot calculate metrics since `self._kmeans` attribute does not exist.
                                 Run `self.get_caps()` first.
                                 """)

        if prefix_file_name is not None and output_dir is None:
            warnings.warn("`prefix_name` supplied but no `output_dir` specified. Files will not be saved.")

        metrics = [metrics] if isinstance(metrics, str) else metrics

        valid_metrics = ["temporal fraction", "persistence", "counts", "transition frequency"]

        boolean_list = [element in valid_metrics for element in metrics]

        if any(boolean_list):
            invalid_metrics = [metrics[indx] for indx,boolean in enumerate(boolean_list) if boolean is False]
            if len(invalid_metrics) > 0:
                warnings.warn(f"invalid metrics will be ignored: {' '.join(invalid_metrics)}")
        else:
            raise ValueError(f"No valid metrics in `metrics` list. Valid metrics are {', '.join(valid_metrics)}")

        if isinstance(subject_timeseries, str) and "pkl" in subject_timeseries:
            subject_timeseries = _convert_pickle_to_dict(pickle_file=subject_timeseries)

        group_cap_dict = {}
        # Get group with most CAPs
        for group in self._groups:
            group_cap_dict.update({group: len(self._caps[group])})

        cap_names = list(self._caps[max(group_cap_dict, key=group_cap_dict.get)])
        cap_numbers = [int(name.split("-")[-1]) for name in cap_names]

        # Assign each subject TRs to CAP
        predicted_subject_timeseries = {}

        for subj_id, group in self._subject_table.items():
            predicted_subject_timeseries[subj_id] = {}
            requested_runs = [f"run-{run}" for run in runs] if runs else list(subject_timeseries[subj_id])
            subject_runs = [subject_run for subject_run in subject_timeseries[subj_id] if subject_run in requested_runs]
            if len(subject_runs) == 0:
                warnings.warn(f"""
                              Skipping subject {subj_id} since they do not have the requested run numbers
                              {','.join(requested_runs)}
                              """)
                continue
            if not continuous_runs or len(requested_runs) == 1:
                for curr_run in subject_runs:
                    if self._standardize:
                        timeseries = (subject_timeseries[subj_id][curr_run] - self._mean_vec[group])/(self._stdev_vec[group] + self._epsilon)
                    else:
                        timeseries = subject_timeseries[subj_id][curr_run]
                    predicted_subject_timeseries[subj_id].update({curr_run: self._kmeans[group].predict(timeseries) + 1})
            else:
                subject_runs = "Continuous Runs"
                timeseries = {subject_runs: {}}
                for curr_run in subject_timeseries[subj_id]:
                    if len(timeseries[subject_runs]) != 0:
                        timeseries[subject_runs] = np.vstack([timeseries[subject_runs],
                                                              subject_timeseries[subj_id][curr_run]])
                    else: timeseries[subject_runs] = subject_timeseries[subj_id][curr_run]
                if self._standardize:
                    timeseries = (timeseries[subject_runs] - self._mean_vec[group])/(self._stdev_vec[group] + self._epsilon)
                else:
                    timeseries = timeseries[subject_runs]
                predicted_subject_timeseries[subj_id].update({subject_runs: self._kmeans[group].predict(timeseries) + 1})

        df_dict = {}

        for metric in metrics:
            if metric in valid_metrics:
                if metric != "transition frequency":
                    df_dict.update({metric: pd.DataFrame(columns=["Subject_ID", "Group","Run"] + list(cap_names))})
                else:
                    df_dict.update({metric: pd.DataFrame(columns=["Subject_ID", "Group","Run","Transition_Frequency"])})

        distributed_list = []
        for subj_id, group in self._subject_table.items():
            for curr_run in predicted_subject_timeseries[subj_id]:
                distributed_list.append([subj_id,group,curr_run])

        for subj_id, group, curr_run in distributed_list:
            if "temporal fraction" in metrics or "counts" in metrics:
                frequency_dict = dict(collections.Counter(predicted_subject_timeseries[subj_id][curr_run]))
                sorted_frequency_dict = {key: frequency_dict[key] for key in sorted(list(frequency_dict))}
                if len(sorted_frequency_dict) != len(cap_numbers):
                    sorted_frequency_dict = {cap_number: sorted_frequency_dict[cap_number] if cap_number in
                                             list(sorted_frequency_dict) else 0 for cap_number in cap_numbers}
                if "temporal fraction" in metrics:
                    proportion_dict = {key: item/(len(predicted_subject_timeseries[subj_id][curr_run]))
                                       for key, item in sorted_frequency_dict.items()}
                    # Populate Dataframe
                    new_row = [subj_id, group, curr_run] +[items for _ , items in proportion_dict.items()]
                    df_dict["temporal fraction"].loc[len(df_dict["temporal fraction"])] = new_row
                if "counts" in metrics:
                    # Populate Dataframe
                    new_row = [subj_id, group, curr_run] + [items for _ , items in sorted_frequency_dict.items()]
                    df_dict["counts"].loc[len(df_dict["counts"])] = new_row
            if "persistence" in metrics:
                # Initialize variable
                persistence_dict = {}
                uninterrupted_volumes = []
                count = 0

                # Iterate through caps
                for target in cap_numbers:
                    # Iterate through each element and count uninterrupted volumes that equal target
                    for index in range(0,len(predicted_subject_timeseries[subj_id][curr_run])):
                        if predicted_subject_timeseries[subj_id][curr_run][index] == target:
                            count +=1
                        # Store count in list if interrupted and not zero
                        else:
                            if count != 0:
                                uninterrupted_volumes.append(count)
                            # Reset counter
                            count = 0
                    # In the event, a participant only occupies one CAP and to ensure final counts are added
                    if count > 0:
                        uninterrupted_volumes.append(count)
                    # If uninterrupted_volumes not zero, multiply elements in the list by repetition time, sum and divide
                    if len(uninterrupted_volumes) > 0:
                        if tr:
                            persistence_dict.update({target: np.sum(np.array(uninterrupted_volumes)*tr)/(len(uninterrupted_volumes))})
                        else:
                            persistence_dict.update({target: np.sum(np.array(uninterrupted_volumes))/(len(uninterrupted_volumes))})
                    else:
                        persistence_dict.update({target: 0})
                    # Reset variables
                    count = 0
                    uninterrupted_volumes = []
                # Populate Dataframe
                new_row = [subj_id, group, curr_run] + [items for _ , items in persistence_dict.items()]
                df_dict["persistence"].loc[len(df_dict["persistence"])] = new_row
            if "transition frequency" in metrics:
                count = 0
                # Iterate through predicted values
                for index in range(0,len(predicted_subject_timeseries[subj_id][curr_run])):
                    if index != 0:
                        # If the subsequent element does not equal the previous element, this is considered a transition
                        if predicted_subject_timeseries[subj_id][curr_run][index-1] != predicted_subject_timeseries[subj_id][curr_run][index]:
                            count +=1
                new_row = [subj_id, group, curr_run, count]
                df_dict["transition frequency"].loc[len(df_dict["transition frequency"])] = new_row

        if output_dir:
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            for metric in df_dict:
                if prefix_file_name:
                    prefix_file_name = os.path.splitext(prefix_file_name.rstrip())[0].rstrip() + f"-{metric.replace(' ','_')}"
                else:
                    prefix_file_name = f"{metric.replace(' ','_')}"
                df_dict[f"{metric}"].to_csv(path_or_buf=os.path.join(output_dir,f"{prefix_file_name}.csv"), sep=",",
                                            index=False)

        if return_df: return df_dict

    def caps2plot(self, output_dir: Optional[Union[str, os.PathLike]]=None, suffix_title: Optional[str]=None,
                  plot_options: Union[str, List[str]]="outer product", visual_scope: List[str]="regions",
                  show_figs: bool=True, subplots: bool=False, **kwargs) -> seaborn.heatmap:
        """
        **Generate heatmaps and outer product plots of CAPs**

        This function produces seaborn heatmaps for each CAP. If groups were given when the CAP class was initialized,
        plotting will be done for all CAPs for all groups.

        Parameters
        ----------
            output_dir : Path or None, default=None
                Directory to save plots to. The directory will be created if it does not exist. If None, plots will not
                be saved.
            suffix_title : str or None, default=None
                Appended to the title of each plot as well as the name of the saved file if ``output_dir`` is provided.
            plot_options : str or List[str], default="outer product"
                Type of plots to create. Options are "outer product" or "heatmap".
            visual_scope : str or List[str], default="regions"
                Determines whether plotting is done at the region level or node level.
                For region level, the value of each nodes in the same regions are averaged together then plotted.
                Options are "regions" or "nodes".
            show_figs : bool, default=True
                Whether to display figures.
            subplots : bool, default=True
                Whether to produce subplots for outer product plots.
            kwargs : Dict
                Keyword arguments used when saving figures. Valid keywords include:

                - dpi : int, default=300
                    Dots per inch for the figure. Default is 300 if ``output_dir`` is provided and ``dpi`` is not
                    pecified.
                - figsize : Tuple, default=(8, 6)
                    Size of the figure in inches.
                - fontsize : int, default=14
                    Font size for the title of individual plots or subplots.
                - hspace : float, default=0.4
                    Height space between subplots.
                - wspace : float, default=0.4
                    Width space between subplots.
                - xticklabels_size : int, default=8
                    Font size for x-axis tick labels.
                - yticklabels_size : int, default=8
                    Font size for y-axis tick labels.
                - shrink : float, default=0.8
                    Fraction by which to shrink the colorbar.
                - nrow : int, varies;
                    Number of rows for subplots. Default varies.
                - ncol : int, default varies (max 5)
                    Number of columns for subplots. Default varies but the maximum is 5.
                - suptitle_fontsize : float, default=0.7
                    Font size for the main title when subplot is True.
                - tight_layout : bool, default=True
                    Use tight layout for subplots.
                - rect : List[int], default=[0, 0.03, 1, 0.95]
                    Rectangle parameter for tight layout when subplots are True to fix whitespace issues.
                - sharey : bool, default=True
                    Share y-axis labels for subplots.
                - xlabel_rotation : int, default=0
                    Rotation angle for x-axis labels.
                - ylabel_rotation : int, default=0
                    Rotation angle for y-axis labels.
                - annot : bool, default=False
                    Add values to cells.
                - fmt : str, default=".2g"
                    Modify how the annotated vales are presented.
                - linewidths : float, default=0
                    Padding between each cell in the plot.
                - borderwidths : float, default=0
                    Width of the border around the plot.
                - linecolor : str, default="black"
                    Color of the line that seperates each cell.
                - edgecolors : str or None, default=None
                    Color of the edges.
                - alpha : float or None, default=None
                    Controls transparancy and ranges from 0 (transparant) to 1 (opaque).
                - bbox_inches : str or None, default="tight"
                    Alters size of the whitespace in the saved image.
                - hemisphere_labels : bool, default=False
                    This option is only available when visual_scope="nodes". Instead of listing all individual labels
                    this parameter simplifies the labels to indicate only the left and right hemispheres, with a
                    division line separating the cells belonging to each hemisphere. If set to True, "edgecolors" will
                    not be used, and both "linewidths" and "linecolor" will be applied only to the division line.
                    This option is available exclusively for "Custom" and "Schaefer" parcellations. **Note, for the
                    "Custom" option, the parcellation should be organized such that the first half of the labels/nodes
                    belong to the left hemisphere and the latter half to the right hemisphere.**
                - cmap : str, Class, or Function, default="coolwarm"
                    Color map for the cells in the plot. For this parameter, you can use premade color palettes or
                    create custom ones. Below is a list of valid options:

                     - Strings to call `seaborn's` premade palettes. Refer to `seaborn's` documentation for valid
                       options.
                     - `Seaborn's` ``diverging_palette`` function to generate custom palettes.
                     - `Matplotlib's` ``LinearSegmentedColormap`` to generate custom palettes.
                     - Other classes or functions compatible with `seaborn`.

                - vmin : float, default=None
                    The minimum value to display in plots.
                - vmax : float, default=None
                    The maximum value to display in plots.

    Returns
    -------
        ``seaborn.heatmap``
            An instance of a ``seaborn`` ``heatmap``.

    Note
    ----
    **If using "Custom" parcellation approach**,, the ``nodes`` and ``regions`` sub-keys are required for this function.

    """
        if not self._parcel_approach:
            raise AttributeError("""
                                 `self.parcel_approach` is None. Add parcel_approach
                                 using `self.parcel_approach=parcel_approach` to use this
                                 method.
                                 """)

        if not hasattr(self,"_caps"):
            raise AttributeError("""
                                 Cannot plot caps since `self._caps` attribute does not exist.
                                 Run `self.get_caps()` first.
                                 """)

        # Check if parcellation_approach is custom
        if "Custom" in self._parcel_approach and ("nodes" not in self._parcel_approach["Custom"] or "regions" not in self._parcel_approach["Custom"]):
            _check_parcel_approach(parcel_approach=self._parcel_approach, call="caps2plot")

        # Get parcellation name
        parcellation_name = list(self._parcel_approach)[0]

        # Check labels
        check_caps = self._caps[list(self._caps)[0]]
        check_caps = check_caps[list(check_caps)[0]]
        if check_caps.shape[0] != len(self._parcel_approach[parcellation_name]["nodes"]):
            raise ValueError("""
                                Number of rois/nodes used for CAPs does not equal the
                                number of rois/nodes specified in `parcel_approach`.
                                """)

        if output_dir:
            if not os.path.exists(output_dir): os.makedirs(output_dir)

        # Convert to list
        if isinstance(plot_options, str): plot_options = [plot_options]
        if isinstance(visual_scope, str): visual_scope = [visual_scope]

        # Check inputs for plot_options and visual_scope
        if not any(["heatmap" in plot_options, "outer product" in plot_options]):
            raise ValueError("Valid inputs for `plot_options` are 'heatmap' and 'outer product'.")

        if not any(["regions" in visual_scope, "nodes" in visual_scope]):
            raise ValueError("Valid inputs for `visual_scope` are 'regions' and 'nodes'.")

        if "regions" in visual_scope: self._create_regions(parcellation_name=parcellation_name)

        # Create plot dictionary
        defaults= {"dpi": 300, "figsize": (8, 6), "fontsize": 14, "hspace": 0.2, "wspace": 0.2, "xticklabels_size": 8,
                    "yticklabels_size": 8, "shrink": 0.8, "nrow": None, "ncol": None, "suptitle_fontsize": 20,
                    "tight_layout": True, "rect": [0, 0.03, 1, 0.95], "sharey": True, "xlabel_rotation": 0,
                    "ylabel_rotation": 0, "annot": False, "fmt": ".2g", "linewidths": 0, "linecolor": "black",
                    "cmap": "coolwarm", "edgecolors": None, "alpha": None, "hemisphere_labels": False,
                    "borderwidths": 0, "vmin": None, "vmax": None, "bbox_inches": "tight"}

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
        if "outer product" in plot_options: self._outer_products = {}

        distributed_list = list(itertools.product(plot_options,visual_scope,self._groups))

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
                    columns =  [x[0] + " " + x[1] for x in list(itertools.product(["LH", "RH"],
                                                                                  self._parcel_approach["Custom"]["regions"]))]

            #  Generate plot for each group
            if plot_option == "outer product": self._generate_outer_product_plots(group=group, plot_dict=plot_dict,
                                                                                    cap_dict=cap_dict,
                                                                                    columns=columns,
                                                                                    subplots=subplots,
                                                                                    output_dir=output_dir,
                                                                                    suffix_title=suffix_title,
                                                                                    show_figs=show_figs, scope=scope,
                                                                                    parcellation_name=parcellation_name)
            elif plot_option == "heatmap": self._generate_heatmap_plots(group=group, plot_dict=plot_dict,
                                                                        cap_dict=cap_dict, columns=columns,
                                                                        output_dir=output_dir,
                                                                        suffix_title=suffix_title,
                                                                        show_figs=show_figs, scope=scope,
                                                                        parcellation_name=parcellation_name)

    def _create_regions(self, parcellation_name):
        # Internal function to create an attribute called `region_caps`. Purpose is to average the values of all nodes
        # in a corresponding region to create region heatmaps or outer product plots
        self._region_caps = {group: {} for group in self._groups}
        for group in self._groups:
            for cap in self._caps[group]:
                region_caps = {}
                if parcellation_name != "Custom":
                    for region in self._parcel_approach[parcellation_name]["regions"]:
                        if len(region_caps) == 0:
                            region_indxs = np.array([index for index, node in
                                                     enumerate(self._parcel_approach[parcellation_name]["nodes"])
                                                     if region in node])
                            region_caps = np.array([np.average(self._caps[group][cap][region_indxs])])
                        else:
                            region_indxs = np.array([index for index, node in
                                                     enumerate(self._parcel_approach[parcellation_name]["nodes"])
                                                     if region in node])
                            region_caps = np.hstack([region_caps, np.average(self._caps[group][cap][region_indxs])])
                else:
                    region_dict = self._parcel_approach["Custom"]["regions"]
                    region_keys = list(region_dict)
                    for region in region_keys:
                        roi_indxs = np.array(region_dict[region]["lh"] + region_dict[region]["rh"])
                        if len(region_caps) == 0:
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
                # Get frequency of each major hemisphere and region in Schaefer,
                # AAL, or Custom atlas
                if parcellation_name == "Schaefer":
                    names_list = [name.split("_")[0:2] for name in self._parcel_approach[parcellation_name]["nodes"]]
                    frequency_dict = dict(collections.Counter([names[0] + " " + names[1] for names in names_list]))
                elif parcellation_name == "AAL":
                    frequency_dict = collections.Counter([name.split("_")[0] for name in
                                                          self._parcel_approach[parcellation_name]["nodes"]])
                else:
                    frequency_dict = {}
                    for names_id in columns:
                        hemisphere_id = "LH" if names_id.startswith("LH ") else "RH"
                        region_id = re.split("LH |RH ", names_id)[-1]
                        frequency_dict.update({names_id: len(self._parcel_approach["Custom"]["regions"][region_id][hemisphere_id.lower()])})
                # Get the names, which indicate the hemisphere and region
                names_list = list(frequency_dict)
                labels = ["" for _ in range(0,len(self._parcel_approach[parcellation_name]["nodes"]))]

                starting_value = 0

                # Iterate through names_list and assign the starting indices corresponding to unique region and
                # hemisphere key
                for num, name in enumerate(names_list):
                    if num == 0:
                        labels[0] = name
                    else:
                        # Shifting to previous frequency of the preceding netwerk to obtain the new starting value of
                        # the subsequent region and hemosphere pair
                        starting_value += frequency_dict[names_list[num-1]]
                        labels[starting_value] = name

            if subplots:
                ax = axes[axes_y] if nrow == 1 else axes[axes_x,axes_y]
                # Modify tick labels based on scope
                if scope == "regions":
                    display = seaborn.heatmap(ax=ax, data=self._outer_products[group][cap], cmap=plot_dict["cmap"],
                                      linewidths=plot_dict["linewidths"], linecolor=plot_dict["linecolor"],
                                      xticklabels=columns, yticklabels=columns,
                                      cbar_kws={"shrink": plot_dict["shrink"]}, annot=plot_dict["annot"],
                                      fmt=plot_dict["fmt"], edgecolors=plot_dict["edgecolors"],
                                      alpha=plot_dict["alpha"], vmin=plot_dict["vmin"], vmax=plot_dict["vmax"])
                else:
                    if plot_dict["hemisphere_labels"] is False:
                        display = seaborn.heatmap(ax=ax, data=self._outer_products[group][cap], cmap=plot_dict["cmap"],
                                          linewidths=plot_dict["linewidths"], linecolor=plot_dict["linecolor"],
                                          cbar_kws={"shrink": plot_dict["shrink"]}, annot=plot_dict["annot"],
                                          fmt=plot_dict["fmt"], edgecolors=plot_dict["edgecolors"],
                                          alpha=plot_dict["alpha"], vmin=plot_dict["vmin"], vmax=plot_dict["vmax"])
                    else:
                        display = seaborn.heatmap(ax=ax, data=self._outer_products[group][cap], cmap=plot_dict["cmap"],
                                          cbar_kws={"shrink": plot_dict["shrink"]}, annot=plot_dict["annot"],
                                          fmt=plot_dict["fmt"], alpha=plot_dict["alpha"], vmin=plot_dict["vmin"],
                                          vmax=plot_dict["vmax"])

                    if plot_dict["hemisphere_labels"] is False:
                        ticks = [i for i, label in enumerate(labels) if label]

                        ax.set_xticks(ticks)
                        ax.set_xticklabels([label for label in labels if label])
                        ax.set_yticks(ticks)
                        ax.set_yticklabels([label for label in labels if label])
                    else:
                        n_labels = len(self._parcel_approach[parcellation_name]["nodes"])
                        division_line = n_labels//2
                        left_hemisphere_tick = (0 + division_line)//2
                        right_hemisphere_tick = (division_line + n_labels)//2

                        ax.set_xticks([left_hemisphere_tick,right_hemisphere_tick])
                        ax.set_xticklabels(["LH", "RH"])
                        ax.set_yticks([left_hemisphere_tick,right_hemisphere_tick])
                        ax.set_yticklabels(["LH", "RH"])

                        plot_dict["linewidths"] = plot_dict["linewidths"] if plot_dict["linewidths"] != 0 else 1

                        ax.axhline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])
                        ax.axvline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])

                # Add border
                if plot_dict['borderwidths'] != 0:
                    border_length = self._outer_products[group][cap].shape[0]

                    display.axhline(y=0, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
                    display.axhline(y=border_length, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
                    display.axvline(x=0, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
                    display.axvline(x=border_length, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])

                # Modify label sizes
                display.set_xticklabels(display.get_xticklabels(),
                                        size = plot_dict["xticklabels_size"],
                                        rotation=plot_dict["xlabel_rotation"])

                if plot_dict["sharey"] is True:
                    if axes_y == 0: display.set_yticklabels(display.get_yticklabels(),
                                                            size = plot_dict["yticklabels_size"],
                                                            rotation=plot_dict["ylabel_rotation"])
                else:
                    display.set_yticklabels(display.get_yticklabels(),
                                            size = plot_dict["yticklabels_size"],
                                            rotation=plot_dict["ylabel_rotation"])

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

                plot_title = f"{group} {cap} {suffix_title}" if suffix_title else f"{group} {cap}"
                if scope == "regions": display = seaborn.heatmap(self._outer_products[group][cap],
                                                                 cmap=plot_dict["cmap"],
                                                                 linewidths=plot_dict["linewidths"],
                                                                 linecolor=plot_dict["linecolor"],
                                                                 xticklabels=columns, yticklabels=columns,
                                                                 cbar_kws={'shrink': plot_dict["shrink"]},
                                                                 annot=plot_dict["annot"], fmt=plot_dict["fmt"],
                                                                 edgecolors=plot_dict["edgecolors"],
                                                                 alpha=plot_dict["alpha"],
                                                                 vmin=plot_dict["vmin"], vmax=plot_dict["vmax"])
                else:
                    if plot_dict["hemisphere_labels"] is False:
                        display = seaborn.heatmap(self._outer_products[group][cap], cmap=plot_dict["cmap"],
                                          linewidths=plot_dict["linewidths"], linecolor=plot_dict["linecolor"],
                                          xticklabels=[], yticklabels=[], cbar_kws={'shrink': plot_dict["shrink"]},
                                          annot=plot_dict["annot"], fmt=plot_dict["fmt"],
                                          edgecolors=plot_dict["edgecolors"], alpha=plot_dict["alpha"],
                                          vmin=plot_dict["vmin"], vmax=plot_dict["vmax"])
                    else:
                        display = seaborn.heatmap(self._outer_products[group][cap], cmap=plot_dict["cmap"], xticklabels=[],
                                          yticklabels=[], cbar_kws={'shrink': plot_dict["shrink"]},
                                          annot=plot_dict["annot"], fmt=plot_dict["fmt"], alpha=plot_dict["alpha"],
                                          vmin=plot_dict["vmin"], vmax=plot_dict["vmax"])

                    if plot_dict["hemisphere_labels"] is False:
                        ticks = [i for i, label in enumerate(labels) if label]

                        display.set_xticks(ticks)
                        display.set_xticklabels([label for label in labels if label])
                        display.set_yticks(ticks)
                        display.set_yticklabels([label for label in labels if label])

                    else:
                        n_labels = len(self._parcel_approach[parcellation_name]["nodes"])
                        division_line = n_labels//2
                        left_hemisphere_tick = (0 + division_line)//2
                        right_hemisphere_tick = (division_line + n_labels)//2

                        display.set_xticks([left_hemisphere_tick,right_hemisphere_tick])
                        display.set_xticklabels(["LH", "RH"])
                        display.set_yticks([left_hemisphere_tick,right_hemisphere_tick])
                        display.set_yticklabels(["LH", "RH"])

                        plot_dict["linewidths"] = plot_dict["linewidths"] if plot_dict["linewidths"] != 0 else 1

                        plt.axhline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])
                        plt.axvline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])

                # Add border
                if plot_dict['borderwidths'] != 0:
                    border_length = self._outer_products[group][cap].shape[0]

                    display.axhline(y=0, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
                    display.axhline(y=border_length, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
                    display.axvline(x=0, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
                    display.axvline(x=border_length, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])

                display.set_title(plot_title, fontdict= {'fontsize': plot_dict["fontsize"]})

                display.set_xticklabels(display.get_xticklabels(),
                                        size = plot_dict["xticklabels_size"],
                                        rotation=plot_dict["xlabel_rotation"])
                display.set_yticklabels(display.get_yticklabels(),
                                        size = plot_dict["yticklabels_size"],
                                        rotation=plot_dict["ylabel_rotation"])

                # Save individual plots
                if output_dir:
                    partial_filename = f"{group}_{cap}_{suffix_title}" if suffix_title else f"{group}_{cap}"
                    if scope == "regions":
                        full_filename = f"{partial_filename.replace(' ','_')}_outer_product_heatmap-regions.png"
                    else:
                        full_filename = f"{partial_filename.replace(' ','_')}_outer_product_heatmap-nodes.png"

                    display.get_figure().savefig(os.path.join(output_dir,full_filename), dpi=plot_dict["dpi"],
                                                 bbox_inches=plot_dict["bbox_inches"])

        # Remove subplots with no data
        if subplots: [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]

        # Save subplot
        if subplots and output_dir:
            partial_filename = f"{group}_CAPs_{suffix_title}" if suffix_title else f"{group}_CAPs"
            if scope == "regions":
                full_filename = f"{partial_filename.replace(' ','_')}_outer_product_heatmap-regions.png"
            else:
                full_filename = f"{partial_filename.replace(' ','_')}_outer_product_heatmap-nodes.png"
            display.get_figure().savefig(os.path.join(output_dir,full_filename), dpi=plot_dict["dpi"],
                                         bbox_inches=plot_dict["bbox_inches"])

        # Display figures
        if not show_figs: plt.close()

    def _generate_heatmap_plots(self, group, plot_dict, cap_dict, columns, output_dir, suffix_title, show_figs,
                                scope, parcellation_name):
        # Initialize new grid
        plt.figure(figsize=plot_dict["figsize"])

        if scope == "regions":
            display = seaborn.heatmap(pd.DataFrame(cap_dict[group], index=columns), xticklabels=True, yticklabels=True,
                              cmap=plot_dict["cmap"], linewidths=plot_dict["linewidths"],
                              linecolor=plot_dict["linecolor"], cbar_kws={'shrink': plot_dict["shrink"]},
                              fmt=plot_dict["fmt"], edgecolors=plot_dict["edgecolors"], alpha=plot_dict["alpha"],
                              vmin=plot_dict["vmin"], vmax=plot_dict["vmax"])
        else:
            # Create Labels
            if plot_dict["hemisphere_labels"] is False:
                if parcellation_name == "Schaefer":
                    names_list = [name.split("_")[0:2] for name in self._parcel_approach[parcellation_name]["nodes"]]
                    frequency_dict = dict(collections.Counter([names[0] + " " + names[1] for names in names_list]))
                elif parcellation_name == "AAL":
                    frequency_dict = collections.Counter([name.split("_")[0] for name in self._parcel_approach[parcellation_name]["nodes"]])
                else:
                    frequency_dict = {}
                    for names_id in columns:
                        hemisphere_id = "LH" if names_id.startswith("LH ") else "RH"
                        region_id = re.split("LH |RH ", names_id)[-1]
                        frequency_dict.update({names_id: len(self._parcel_approach["Custom"]["regions"][region_id][hemisphere_id.lower()])})
                names_list = list(frequency_dict)
                labels = ["" for _ in range(0,len(self._parcel_approach[parcellation_name]["nodes"]))]

                starting_value = 0

                # Iterate through names_list and assign the starting indices corresponding to unique region
                # and hemisphere key
                for num, name in enumerate(names_list):
                    if num == 0:
                        labels[0] = name
                    else:
                        # Shifting to previous frequency of the preceding netwerk to obtain the new starting value of
                        # the subsequent region and hemosphere pair
                        starting_value += frequency_dict[names_list[num-1]]
                        labels[starting_value] = name

                display = seaborn.heatmap(pd.DataFrame(cap_dict[group], columns=list(cap_dict[group])),
                                          xticklabels=True, yticklabels=True, cmap=plot_dict["cmap"],
                                          linewidths=plot_dict["linewidths"], linecolor=plot_dict["linecolor"],
                                          cbar_kws={'shrink': plot_dict["shrink"]}, annot=plot_dict["annot"],
                                          fmt=plot_dict["fmt"], edgecolors=plot_dict["edgecolors"],
                                          alpha=plot_dict["alpha"], vmin=plot_dict["vmin"], vmax=plot_dict["vmax"])

                plt.yticks(ticks=[pos for pos, label in enumerate(labels) if label], labels=names_list)

            else:
                display = seaborn.heatmap(pd.DataFrame(cap_dict[group], columns=list(cap_dict[group])),
                                          xticklabels=True, yticklabels=True, cmap=plot_dict["cmap"],
                                          cbar_kws={'shrink': plot_dict["shrink"]}, annot=plot_dict["annot"],
                                          fmt=plot_dict["fmt"], alpha=plot_dict["alpha"], vmin=plot_dict["vmin"],
                                          vmax=plot_dict["vmax"])

                n_labels = len(self._parcel_approach[parcellation_name]["nodes"])
                division_line = n_labels//2
                left_hemisphere_tick = (0 + division_line)//2
                right_hemisphere_tick = (division_line + n_labels)//2

                display.set_yticks([left_hemisphere_tick,right_hemisphere_tick])
                display.set_yticklabels(["LH", "RH"])

                plot_dict["linewidths"] = plot_dict["linewidths"] if plot_dict["linewidths"] != 0 else 1

                plt.axhline(division_line, color=plot_dict["linecolor"], linewidth=plot_dict["linewidths"])

        if plot_dict['borderwidths'] != 0:
            y_length = len(cap_dict[group][list(cap_dict[group])[0]])

            display.axhline(y=0, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
            display.axhline(y=y_length, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
            display.axvline(x=0, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
            display.axvline(x=len(self._caps[group]), color=plot_dict["linecolor"],
                            linewidth=plot_dict["borderwidths"])

        display.set_xticklabels(display.get_xticklabels(),
                                size = plot_dict["xticklabels_size"],
                                rotation=plot_dict["xlabel_rotation"])
        display.set_yticklabels(display.get_yticklabels(),
                                size = plot_dict["yticklabels_size"],
                                rotation=plot_dict["ylabel_rotation"])

        plot_title = f"{group} CAPs {suffix_title}" if suffix_title else f"{group} CAPs"
        display.set_title(plot_title, fontdict= {'fontsize': plot_dict["fontsize"]})

        # Save plots
        if output_dir:
            partial_filename = f"{group}_CAPs_{suffix_title}" if suffix_title else f"{group}_CAPs"
            if scope == "regions":
                full_filename = f"{partial_filename.replace(' ','_')}_heatmap-regions.png"
            else:
                full_filename = f"{partial_filename.replace(' ','_')}_heatmap-nodes.png"
            display.get_figure().savefig(os.path.join(output_dir,full_filename), dpi=plot_dict["dpi"],
                                         bbox_inches=plot_dict["bbox_inches"])

        # Display figures
        if not show_figs: plt.close()

    def caps2corr(self, output_dir: Optional[Union[str, os.PathLike]]=None, suffix_title: Optional[str]=None,
                  show_figs: bool=True, **kwargs) -> seaborn.heatmap:
        """
        **Generate Correlation Matrix for CAPs**

        Produces the correlation matrix of all CAPs. If groups were given when the CAP class was initialized, a
        correlation matrix will be generated for each group.

        Parameters
        ----------
            output_dir : `os.Pathlike` or None, default=None
                Directory to save plots to. The directory will be created if it does not exist. If None,
                plots will not be saved.
            suffix_title : str or None, default=None
                Appended to the title of each plot as well as the name of the saved file if ``output_dir``
                is provided.
            show_figs : bool, default=True
                Whether to display figures.
            kwargs : Dict
                Keyword arguments used when saving figures. Valid keywords include:

                - dpi : int, default=300
                    Dots per inch for the figure. Default is 300 if ``output_dir`` is provided and ``dpi`` is not
                    specified.
                - figsize : Tuple, default=(8, 6)
                    Size of the figure in inches.
                - fontsize : int, default=14
                    Font size for the title of individual plots or subplots.
                - xticklabels_size : int, default=8
                    Font size for x-axis tick labels.
                - yticklabels_size : int, default=8
                    Font size for y-axis tick labels.
                - shrink : float, default=0.8
                    Fraction by which to shrink the colorbar.
                - xlabel_rotation : int, default=0
                    Rotation angle for x-axis labels.
                - ylabel_rotation : int, default=0
                    Rotation angle for y-axis labels.
                - annot : bool, default=False
                    Add values to each cell.
                - fmt : str, default=".2g",
                    Modify how the annotated vales are presented.
                - linewidths : float, default=0
                    Padding between each cell in the plot.
                - borderwidths : float, default=0
                    Width of the border around the plot.
                - linecolor : str, default="black"
                    Color of the line that seperates each cell.
                - edgecolors : str, default=None
                    Color of the edges.
                - alpha : float, default=None
                    Controls transparancy and ranges from 0 (transparant) to 1 (opaque).
                - bbox_inches : str or None, default="tight"
                    Alters size of the whitespace in the saved image.
                - cmap : str, Class, or function, default="coolwarm"
                    Color map for the cells in the plot. For this parameter, you can use premade color palettes or
                    create custom ones.
                    Below is a list of valid options:

                     - Strings to call seaborn's premade palettes. Refer to seaborn's documentation for valid options.
                     - Seaborn's diverging_palette function to generate custom palettes.
                     - Matplotlib's LinearSegmentedColormap to generate custom palettes.
                     - Other classes or functions compatible with seaborn.

        Returns
        -------
            `seaborn.heatmap`
                An instance of a `seaborn` `heatmap`.
        """
        if not hasattr(self,"_caps"):
            raise AttributeError("""
                                 Cannot plot caps since `self._caps` attribute does not exist.
                                 Run `self.get_caps()` first.
                                 """)

        # Create plot dictionary
        defaults = {"dpi": 300, "figsize": (8, 6), "fontsize": 14, "xticklabels_size": 8, "yticklabels_size": 8,
                    "shrink": 0.8, "xlabel_rotation": 0, "ylabel_rotation": 0, "annot": False, "linewidths": 0,
                    "linecolor": "black", "cmap": "coolwarm", "fmt": ".2g", "borderwidths": 0, "edgecolors": None,
                    "alpha": None, "bbox_inches": "tight"}

        plot_dict = _check_kwargs(defaults, **kwargs)

        for group in self._caps:
            # Refresh grid for each iteration
            plt.figure(figsize=plot_dict["figsize"])

            df = pd.DataFrame(self._caps[group])
            display = seaborn.heatmap(df.corr(), xticklabels=True, yticklabels=True, cmap=plot_dict["cmap"],
                              linewidths=plot_dict["linewidths"], linecolor=plot_dict["linecolor"],
                              cbar_kws={'shrink': plot_dict["shrink"]}, annot=plot_dict["annot"],
                              fmt=plot_dict["fmt"], edgecolors=plot_dict["edgecolors"], alpha=plot_dict["alpha"])
            # Add Border
            if plot_dict["borderwidths"] != 0:
                display.axhline(y=0, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
                display.axhline(y=df.corr().shape[1], color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
                display.axvline(x=0, color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])
                display.axvline(x=df.corr().shape[0], color=plot_dict["linecolor"],linewidth=plot_dict["borderwidths"])

            # Modify label sizes
            display.set_xticklabels(display.get_xticklabels(),
                                    size = plot_dict["xticklabels_size"],
                                    rotation=plot_dict["xlabel_rotation"])
            display.set_yticklabels(display.get_yticklabels(),
                                    size = plot_dict["yticklabels_size"],
                                    rotation=plot_dict["ylabel_rotation"])
            # Set plot name
            if suffix_title:
                plot_title = f"{group} CAPs Correlation Matrix {suffix_title}"
            else:
                plot_title = f"{group} CAPs Correlation Matrix"
            display.set_title(plot_title, fontdict= {'fontsize': plot_dict["fontsize"]})

            # Display figures
            if not show_figs: plt.close()
            # Save figure
            if output_dir:
                if not os.path.exists(output_dir): os.makedirs(output_dir)
                if suffix_title:
                    full_filename = f"{group.replace(' ', '_')}_correlation_matrix_{suffix_title}.png"
                else:
                    full_filename = f"{group.replace(' ', '_')}_correlation_matrix.png"
                display.get_figure().savefig(os.path.join(output_dir,full_filename), dpi=plot_dict["dpi"],
                                             bbox_inches=plot_dict["bbox_inches"])

    def caps2niftis(self, output_dir: Union[str, os.PathLike], suffix_file_name: Optional[str]=None,
                    fwhm: Optional[float]=None) -> nib.Nifti1Image:
        """
        **Standalone Method to Convert CAPs to NifTi Statistical Maps**

        Converts atlas into a stat map by replacing labels with the corresponding from the cluster centroids then saves
        them as compressed nii files.

        Parameters
        ----------
            output_dir: `os.Pathlike`, default=None
                Directory to save plots to. The directory will be created if it does not exist.
            suffix_title: str or None, default=None
                Appended to the name of the saved file.
            fwhm: float or None, default=None
                Strength of spatial smoothing to apply (in millimeters) to the statistical map prior to interpolating
                from MNI152 space to fslr surface space. Note, this can assist with coverage issues in the plot.

        Returns
        -------
            `NifTI1Image`
                `NifTI` statistical map.
        """
        if not self._parcel_approach:
            raise AttributeError("""
                                 `self.parcel_approach` is None. Add parcel_approach
                                 using `self.parcel_approach=parcel_approach` to use this
                                 method.
                                 """)

        if not hasattr(self,"_caps"):
            raise AttributeError("""
                                 Cannot plot caps since `self._caps` attribute does not exist.
                                 Run `self.get_caps()` first.
                                 """)

        if not os.path.exists(output_dir): os.makedirs(output_dir)

        parcellation_name = list(self._parcel_approach)[0]

        for group in self._caps:
            for cap in self._caps[group]:
                stat_map = _cap2statmap(atlas_file=self._parcel_approach[parcellation_name]["maps"],
                                        cap_vector=self._caps[group][cap], fwhm=fwhm)

                if suffix_file_name:
                    save_name = f"{group.replace(' ', '_')}_{cap.replace('-', '_')}_{suffix_file_name}.nii.gz"
                else:
                    save_name = f"{group.replace(' ', '_')}_{cap.replace('-', '_')}.nii.gz"
                nib.save(stat_map, os.path.join(output_dir,save_name))

    def caps2surf(self, output_dir: Optional[Union[str, os.PathLike]]=None, suffix_title: Optional[str]=None,
                  show_figs: bool=True, fwhm: Optional[float]=None, fslr_density: str="32k", method: str="linear",
                  save_stat_map: bool=False, fslr_giftis_dict: Optional[dict]=None, **kwargs) -> surfplot.Plot:
        """
        **Project CAPs onto Surface Plots**

        Converts atlas into a stat map by replacing labels with the corresponding from the cluster centroids then plots
        on a surface plot. This function uses ``surfplot`` and ``neuromaps`` for surface plotting.

        Parameters
        ----------
            output_dir: `os.Pathlike` or None, default=None
                Directory to save plots to. The directory will be created if it does not exist. If None, plots will not
                be saved.
            suffix_title: str or None, default=None
                Appended to the title of each plot as well as the name of the saved file if ``output_dir`` is provided.
            show_figs: bool, default=True
                Whether to display figures.
            fwhm: float or None, defualt=None
                Strength of spatial smoothing to apply (in millimeters) to the statistical map prior to interpolating
                from MNI152 space to fslr surface space.
                Note, this can assist with coverage issues in the plot.
            fslr_density: str, default="32k"
                Density of the fslr surface when converting from MNI152 space to fslr surface. Options are "32k" or
                "164k". If using ``fslr_giftis_dict`` options are "4k", "8k", "32k", and "164k".
            method: str, default="linear"
                Interpolation method to use when converting from MNI152 space to fslr surface. Options are "linear" or
                "nearest".
            save_stat_map: bool, default=False
                If True, saves the statistical map for each CAP for all groups as a Nifti1Image if ``output_dir`` is
                provided.
            fslr_giftis_dict: dict or None, default=None
                Dictionary specifying precomputed gifti files in fslr space for plotting stat maps. This parameter
                should be used if the statistical CAP NIfTI files (can be obtained using ``caps2niftis``) were
                converted to GifTi files using a tool such as Connectome Workbench.The dictionary structure is:
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
            kwargs : Dict
                Additional parameters to pass to modify certain plot parameters. Options include:

                - dpi : int, default=300
                    Dots per inch for the plot.
                - title_pad : int, default=-3
                    Padding for the plot title.
                - cmap : str or Class, default="cold_hot"
                    Colormap to be used for the plot. For this parameter, you can use premade color palettes orcreate
                    custom ones.Below is a list of valid options:

                     - Strings to call `nilearn's` ``_cmap_d`` fuction. Refer to documention for `nilearn's` ``_cmap_d``
                     for valid palettes.
                     - Matplotlib's LinearSegmentedColormap to generate custom colormaps.

                - cbar_kws : Dict, default={"location": "bottom", "n_ticks": 3}
                    Customize colorbar. Refer to ``_add_colorbars`` at
                    https://surfplot.readthedocs.io/en/latest/generated/surfplot.plotting.Plot.html#surfplot.plotting.Plot
                    for valid kwargs.
                - alpha : float, default=1
                    Transparency level of the colorbar.
                - zero_transparent : bool, default=True
                    Turns vertices with a value of 0 transparent.
                - as_outline : bool, default=False
                    Plots only an outline of contiguous vertices with the same value.
                - size : Tuple, default=(500, 400)
                    Size of the plot in pixels.
                - layout : str, default="grid"
                    Layout of the plot.
                - zoom : float, default=1.5
                    Zoom level for the plot.
                - views : List[str], default=["lateral", "medial"]
                    Views to be displayed in the plot.
                - brightness : float, default=0.5
                    Brightness level of the plot.
                - figsize : Tuple or None, default=None
                    Size of the figure.
                - scale : Tuple, default=(2, 2)
                    Scale factors for the plot.
                - surface : str, default="inflated"
                    The surface atlas that is used for plotting. Options are "inflated" or "veryinflated"
                - color_range : Tuple or None, default=None
                    The minimum and maximum value to display in plots. For instance, (-1,1) where minimum
                    value is first. If None, the minimum and maximum values from the image will be used.

                Please refer to surfplot's documentation for specifics:
                https://surfplot.readthedocs.io/en/latest/generated/surfplot.plotting.Plot.html#surfplot.plotting.Plot

        Returns
        -------
            `NifTI1Image`
                `NifTI` statistical map.
            `surfplot.Plot`
                An instance of a `surfplot` `Plot`.

        Note
        ----
        For this to work, ``parcel_approach`` must have the "maps" sub-key containing the path to the NifTi file of the
        atlas. Assumes that atlas background label is zero and atlas is in MNI space. Also assumes that the indices
        from the cluster centroids are related to the atlas by an offset of one. For instance, index 0 of the cluster
        centroid vector is the first nonzero label, which is assumed to be at the first index of the array in
        ``sorted(np.unique(atlas_fdata))``.
        """
        if not self._parcel_approach:
            raise AttributeError("""
                                 `self.parcel_approach` is None. Add parcel_approach
                                 using `self.parcel_approach=parcel_approach` to use this
                                 method.
                                 """)

        if not hasattr(self,"_caps") and fslr_giftis_dict is None:
            raise AttributeError("""
                                 Cannot plot caps since `self._caps` attribute does not exist. Run `self.get_caps()`
                                 first.
                                 """)

        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

        # Create plot dictionary
        defaults = {"dpi": 300, "title_pad": -3, "cmap": "cold_hot", "cbar_kws":  {"location": "bottom", "n_ticks": 3},
                    "size": (500, 400), "layout": "grid", "zoom": 1.5, "views": ["lateral", "medial"], "alpha": 1,
                    "zero_transparent": True, "as_outline": False,"brightness": 0.5, "figsize": None, "scale": (2, 2),
                    "surface": "inflated", "color_range": None}

        plot_dict = _check_kwargs(defaults, **kwargs)

        groups = self._caps if hasattr(self,"_caps") and fslr_giftis_dict is None else fslr_giftis_dict

        parcellation_name = list(self._parcel_approach)[0]

        for group in groups:
            caps = self._caps[group] if hasattr(self,"_caps") and fslr_giftis_dict is None else fslr_giftis_dict[group]
            for cap in caps:
                if fslr_giftis_dict is None:
                    stat_map = _cap2statmap(atlas_file=self._parcel_approach[parcellation_name]["maps"],
                                            cap_vector=self._caps[group][cap], fwhm=fwhm)
                    gii_lh, gii_rh = mni152_to_fslr(stat_map, method=method, fslr_density=fslr_density)
                else:
                    gii_lh, gii_rh = fslr_to_fslr([fslr_giftis_dict[group][cap]['lh'],
                                                   fslr_giftis_dict[group][cap]['rh']],
                                                   target_density=fslr_density, method=method)
                # Code slightly adapted from surfplot example 2
                surfaces = fetch_fslr()
                if plot_dict["surface"] not in ["inflated", "veryinflated"]:
                    warnings.warn(f"""
                                  {plot_dict['surface']} is an invalid option for `surface`. Available options
                                  include 'inflated' or 'verinflated'. Defaulting to 'inflated'
                                  """)
                    plot_dict["surface"] = "inflated"
                lh, rh = surfaces[plot_dict["surface"]]
                lh = str(lh) if not isinstance(lh, str) else lh
                rh = str(rh) if not isinstance(rh, str) else rh
                sulc_lh, sulc_rh = surfaces['sulc']
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
                            zero_transparent=plot_dict["zero_transparent"], as_outline=plot_dict["as_outline"])

                # Color bar
                fig = p.build(cbar_kws=plot_dict["cbar_kws"], figsize=plot_dict["figsize"], scale=plot_dict["scale"])
                fig_name = f"{group} {cap} {suffix_title}" if suffix_title else f"{group} {cap}"
                fig.axes[0].set_title(fig_name, pad=plot_dict["title_pad"])

                if show_figs: fig.show()

                if output_dir:
                    if suffix_title:
                        save_name = f"{group.replace(' ', '_')}_{cap.replace('-', '_')}_{suffix_title}.png"
                    else:
                        save_name = f"{group.replace(' ', '_')}_{cap.replace('-', '_')}.png"

                    fig.savefig(os.path.join(output_dir, save_name), dpi=plot_dict["dpi"])
                    # Save stat map
                    if save_stat_map:
                        stat_map_name = save_name.replace(".png", ".nii.gz")
                        nib.save(stat_map, stat_map_name)

    def caps2radar(self, output_dir: Optional[Union[str, os.PathLike]]=None, suffix_title: Optional[str]=None,
                   show_figs: bool=True, use_scatterpolar: bool=False,
                   **kwargs) -> Union[px.line_polar, go.Scatterpolar]:
        """
        **Generate Radar Plots**

        This method identifies networks/regions (across both hemispheres) in each CAP that show high amplitude (high
        activation relative to the mean zero if z-scored) and low amplitude (high deactivation relative to the mean
        zero if z-scored) by using cosine similarity. This is accomplished by extracting the cluster centroids (CAPs),
        a 1 x ROI vector, and generating a binary vector (a vector 1 x ROI vector consisting of 0's and 1's) where 1's
        indicate the indices/ROIs (Regions of Interest) in a specific region. For instance, if elements at indices 0,
        5, and 10 in the cluster centroid are nodes in the Visual Network, then a binary vector is generated where
        those indices are 1, and all others are 0. This binary vector essentially operates like a 1-dimensional binary
        mask to capture relevant ROIs in a given region/network.

        Once, the dot product of the cluster centroid and binary vector is then calculated it is normalized by the
        product of the norms of the cluster centroid and the binary vector to restrict the range to -1 and 1, hence
        cosine similarity. For example, with the Schaefer 7-network parcellation, if your analysis has five CAPs,
        then each cluster centroid (CAP) will be multiplied by seven different binary vectors (where the 1's represent
        the nodes in that network).

        Cosine similarities notably above zero suggest that many nodes in that network/region are highly activating
        together, cosine similarities around zero suggest that nodes in that network/region are co-activating and
        deactivating together, and cosine similarities notably below zero suggest that many nodes in that
        network/region are deactivating together.

        This method is a useful quantitative method to characterize CAPs based on nodes in region/networks that
        display high co-activation or high co-deactivation. For instance, if the dorsal attention network (DAN) has the
        highest cosine similarity and the ventral attention network (VAN) has a lowest cosine similarity, then that cap
        can be described as (DAN +/VAN -).

        **Note, the radar plots only display positive values. The "Low Amplitude" group are negative cosine similarity
        (below zero); however, the absolute value was taken to make them positive so that the radar plot starts at 0
        and direct magnitude comparisons between the "High Amplitude" (positive cosine similarity above zero) and
        "Low Amplitude" groups are easier.**

        Parameters
        ----------
            output_dir: `os.Pathlike` or None, default=None
                Directory to save plots to. The directory will be created if it does not exist.
            suffix_title: str or None, default=None
                Appended to the title of each plot as well as the name of the saved file if ``output_dir`` is provided.
            show_figs: bool, default=True
                Whether to display figures. If this function detects that it is not being ran in an interactive Python
                environment, then it uses plotly.offline, creates an html file named "temp-plot.html", and opens each
                plot in the default browser.
            use_scatterpolar: bool=False
                Uses plotly's ``Scatterpolar`` instead of plotly's ``line_polar``. The primary difference is that
                ``Scatterpolar`` shows the scatter dots. However, this can be acheived with ``line_polar`` by setting
                ``mode`` to "markers+lines". There also seems to be a difference in default opacity behavior.
            kwargs: Dict
                Additional parameters to pass to modify certain plot parameters. Options include:

                - scale : int, default=2
                    Controls resolution of image when saving.
                - savefig_options : Dict[str], default={"width": 3, "height": 3, "scale": 1}
                    If ``output_dir`` provided, controls the width (in inches), height (in inches), and scale of the
                    plot.
                    The height and width are multiplied by the dpi.
                - height : int, default=800
                    Height of the plot. Value is multiplied by the dpi when saving.
                - width : int, defualt=1200
                    Width of the plot. Value is multiplied by the dpi when saving.
                - line_close : int, default=True
                    Whether to close the lines
                - bgcolor : str, default="white"
                    Color of the background
                - scattersize : int, default=8
                    If ``use_scatterpolar=True``, controls the size of the dots.
                - connectgaps : bool, default=True
                    If ``use_scatterpolar=True``, controls if missing values are connected.
                - opacity : float, default=0.5,
                    If ``use_scatterpolar=True``, sets the opacity of the trace.
                - fill : str, default="none".
                    If "toself" the are of the dots and within the boundaries of the line will be filled.
                - mode : str, default="markers+lines",
                    Determines how the trace is drawn. Can include "lines", "markers", "lines+markers", "lines+markers+text".
                - radialaxis : Dict[str], default={"showline": False, "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)", "gridcolor": "rgba(0, 0, 0, 0.25)", "ticks": "outside","tickfont": {"size": 14, "color": "black"}}
                    Customizes the radial axis.
                    Refer to https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.polar.radialaxis.html or
                    https://plotly.com/python/reference/layout/polar/ for valid kwargs.
                - angularaxis : Dict[str], default= {"showline": True, "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)", "gridcolor": "rgba(0, 0, 0, 0.25)", "tickfont": {"size": 16, "color": "black"}}
                    Customizes the angular axis.
                    Refer to https://plotly.com/python-api-reference/generated/plotly.graph_objects.layout.polar.angularaxis.html
                    or https://plotly.com/python/reference/layout/polar/ for valid kwargs.
                - color_discrete_map : Dict[str], default={"High Amplitude": "red", "Low Amplitude": "blue"},
                    Change color of the "High Amplitude" and "Low Amplitude" groups. Must use the keys
                    "High Amplitude" and "Low Amplitude" to work.
                - title_font : Dict[str], default={"family": "Times New Roman", "size": 30, "color": "black"}
                    Modifies the font of the title. Refer to https://plotly.com/python/reference/layout/ for valid
                    kwargs.
                - title_x : float, default=0.5
                    Modifies x position of title.
                - title_y : float, default=None
                    Modifies y position of title.
                - legend : Dict[str], default={"yanchor": "top", "xanchor": "left", "y": 0.99, "x": 0.01,"title_font_family": "Times New Roman", "font": {"size": 12, "color": "black"}}
                    Customized legend. Refer to https://plotly.com/python/reference/layout/ for valid kwargs.
                - engine : str, default="kaleido"
                    Engine used for saving plots.

        Returns
        -------
            `plotly.express.line_polar`
                An instance of a `plotly` `line_polar` radar plot.
            `plotly.express.Scatterplot`
                An instance of a `plotly` `Scatterplot` radar plot.

        Note
        -----
        To save, the kaleido package is needed, which is a dependency in this package. The kaleido package on Windows
        seems to only work with plotly if it is a specific version, such as version 0.1.0.post1. Additionally, if
        another engine such as ``orca`` is used, then it must be installed.

        **If using "Custom" parcellation approach**, the ``regions`` sub-key is required

        References
        ----------
        Zhang, R., Yan, W., Manza, P., Shokri-Kojori, E., Demiral, S. B., Schwandt, M., Vines, L., Sotelo, D., Tomasi,
        D., Giddens, N. T., Wang, G., Diazgranados, N., Momenan, R., & Volkow, N. D. (2023). Disrupted brain state
        dynamics in opioid and alcohol use disorder: attenuation by nicotine use. Neuropsychopharmacology, 49(5),
        876–884. https://doi.org/10.1038/s41386-023-01750-w
        """
        if not self._parcel_approach:
            raise AttributeError("""
                                 `self.parcel_approach` is None. Add parcel_approach
                                 using `self.parcel_approach=parcel_approach` to use this
                                 method.
                                 """)

        if not hasattr(self,"_caps"):
            raise AttributeError("""
                                 Cannot plot caps since `self._caps` attribute does not exist. Run `self.get_caps()`
                                 first.""")

        defaults = {"scale": 2, "height": 800, "width": 1200, "line_close": True, "bgcolor": "white", "fill": "none",
                    "scattersize": 8, "connectgaps": True, "opacity": 0.5, "radialaxis": {"showline": False,
                    "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)", "gridcolor": "rgba(0, 0, 0, 0.25)", "ticks": "outside", "tickfont": {"size": 14, "color": "black"}},
                    "angularaxis": {"showline": True, "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)", "gridcolor": "rgba(0, 0, 0, 0.25)", "tickfont": {"size": 16, "color": "black"}},
                    "color_discrete_map": {"High Amplitude": "rgba(255, 0, 0, 1)", "Low Amplitude": "rgba(0, 0, 255, 1)"},
                    "title_font": {"family": "Times New Roman", "size": 30, "color": "black"}, "title_x": 0.5,
                    "title_y":None,"legend": {"yanchor": "top", "xanchor": "left", "y": 0.99, "x": 0.01, "title_font_family": "Times New Roman", "font": {"size": 12, "color": "black"}},
                    "mode": "markers+lines", "engine": "kaleido"}

        plot_dict = _check_kwargs(defaults, **kwargs)

        parcellation_name = list(self.parcel_approach)[0]

        # Create radar dict
        for group in self._caps:
            if parcellation_name == "Custom":
                radar_dict = {"regions": list(self.parcel_approach[parcellation_name]["regions"])}
            else:
                radar_dict = {"regions": list(self.parcel_approach[parcellation_name]["regions"])}
            for cap in self._caps[group]:
                cap_vector = self._caps[group][cap]
                radar_dict[cap] = []
                for region in radar_dict["regions"]:
                    if parcellation_name == "Custom":
                        indxs = self._parcel_approach[parcellation_name]["regions"][region]["lh"] + self._parcel_approach[parcellation_name]["regions"][region]["rh"]
                    else:
                        indxs = np.array([value for value, node in
                                          enumerate(self._parcel_approach[parcellation_name]["nodes"])
                                          if region in node])

                    # Create mask to set ROIs not in regions to zero and ROIs in regions as 1
                    binary_vector = np.zeros_like(cap_vector)
                    binary_vector[indxs] = 1

                    #Calculate cosine similarity
                    dot_product = np.dot(cap_vector, binary_vector)
                    norm_cap_vector = np.linalg.norm(cap_vector)
                    norm_binary_vector = np.linalg.norm(binary_vector)
                    cosine_similarity = dot_product/(norm_cap_vector * norm_binary_vector)

                    # Store value in dict
                    radar_dict[cap].append(cosine_similarity)

            # Create dataframe
            df = pd.DataFrame(radar_dict)

            for cap in df.columns[df.columns != "regions"]:

                groups = df[cap].apply(lambda x: 'High Amplitude' if x > 0 else ('Low Amplitude' if x < 0 else np.nan))
                df[cap] = df[cap].abs()

                if use_scatterpolar:
                    # Get high amplitude and low amplitude data
                    regions = df["regions"].values
                    # Set non high amplitude values as nan and low amplitude values as nan
                    high_amplitude_values = np.array([np.nan]*len(regions))
                    low_amplitude_values = np.array([np.nan]*len(regions))
                    high_amp_indxs = np.where(groups.values == "High Amplitude")
                    high_amplitude_values[high_amp_indxs] = df[cap].values[high_amp_indxs]
                    low_amp_indxs = np.where(groups.values == "Low Amplitude")
                    low_amplitude_values[low_amp_indxs] = df[cap].values[low_amp_indxs]

                    # Add high amplitude and low amplitude data as traces
                    fig = go.Figure(layout=go.Layout(width=plot_dict["width"], height=plot_dict["height"]))
                    fig.add_trace(go.Scatterpolar(
                    r=list(high_amplitude_values),
                    theta=regions,
                    connectgaps=plot_dict["connectgaps"],
                    name="High Amplitude",
                    opacity=plot_dict["opacity"],
                    marker=dict(color=plot_dict["color_discrete_map"]["High Amplitude"],
                                size=plot_dict["scattersize"])))

                    fig.add_trace(go.Scatterpolar(
                    r=list(low_amplitude_values),
                    theta=regions,
                    name="Low Amplitude",
                    connectgaps=plot_dict["connectgaps"],
                    opacity=plot_dict["opacity"],
                    marker=dict(color=plot_dict["color_discrete_map"]["Low Amplitude"],
                                size=plot_dict["scattersize"])))

                else:
                    fig = px.line_polar(df, r=cap, theta="regions", line_close=plot_dict["line_close"], color=groups,
                                        width=plot_dict["width"], height=plot_dict["height"],
                                        category_orders={"regions": df["regions"]},
                                        color_discrete_map = plot_dict["color_discrete_map"])

                fig.update_traces(fill=plot_dict["fill"], mode=plot_dict["mode"])

                # Set max value
                if "tickvals" not in plot_dict["radialaxis"] and "range" not in plot_dict["radialaxis"]:
                    max_value = df[cap].max()
                    plot_dict["radialaxis"]["tickvals"] = [max_value/4, max_value/2, 3*max_value/4, max_value]

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
                    )
                )

                if show_figs:
                    if bool(getattr(sys, 'ps1', sys.flags.interactive)): fig.show()
                    else: pyo.plot(fig, auto_open=True)

                if output_dir:
                    if not os.path.exists(output_dir): os.makedirs(output_dir)
                    if suffix_title:
                        file_name = f"{group}_{cap}_radar_{suffix_title}.png"
                    else: file_name = f"{group}_{cap}_radar.png"
                    fig.write_image(os.path.join(output_dir,file_name), scale=plot_dict["scale"],
                                    engine=plot_dict["engine"])
