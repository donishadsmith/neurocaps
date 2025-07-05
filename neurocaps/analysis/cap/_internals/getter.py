"""A class which is responsible for accessing attributes in ``CAP``."""

import copy, sys
from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from neurocaps.utils._parcellation_validation import check_parcel_approach
from neurocaps.typing import ParcelConfig, ParcelApproach


class CAPGetter:
    def __init__(self):
        pass

    ### Attributes exist when CAP initialized
    @property
    def parcel_approach(self) -> Union[ParcelApproach, None]:
        """
        Parcellation information with "maps" (path to parcellation file), "nodes" (labels), and
        "regions" (anatomical regions or networks). This property is also settable (accepts a
        dictionary or pickle file). Returns a deep copy.
        """
        return self._parcel_approach

    @parcel_approach.setter
    def parcel_approach(self, parcel_dict: Union[ParcelConfig, ParcelApproach, str]) -> None:
        self._parcel_approach = check_parcel_approach(parcel_approach=parcel_dict, call="setter")

    @property
    def groups(self) -> Union[dict[str, list[str]], None]:
        """Mapping of groups names to lists of subject IDs. Returns a deep copy."""
        return copy.deepcopy(self._groups)

    ### Attributes exist when CAP.get_caps() used
    @property
    def subject_table(self) -> Union[dict[str, str], None]:
        """
        Lookup table mapping subject IDs to their groups. Derived from ``self.groups`` each time
        ``self.get_caps()`` is ran. While this property can be modified using its setter, any
        changes will be overwritten based on ``self.groups`` on the subsequent call to
        ``self.get_caps()``. Returns a deep copy.
        """
        return copy.deepcopy(getattr(self, "_subject_table", None))

    @subject_table.setter
    def subject_table(self, subject_dict: dict[str, str]) -> None:
        if isinstance(subject_dict, dict):
            self._subject_table = copy.deepcopy(subject_dict)
        else:
            raise TypeError(
                "`self.subject_table` must be a dictionary where the keys are the subject IDs and "
                "the values are the group names."
            )

    @property
    def n_clusters(self) -> Union[int, list[int], None]:
        """
        An integer or list of integers representing the number of clusters used for k-means.
        Defined after running ``self.get_caps()``.
        """
        return getattr(self, "_n_clusters", None)

    @property
    def n_cores(self) -> Union[int, None]:
        """
        Number of cores specified used for multiprocessing with Joblib. Defined after running
        ``self.get_caps()``.
        """
        return getattr(self, "_n_cores", None)

    @property
    def runs(self) -> Union[list[Union[int, str]], None]:
        """
        Run IDs specified in the analysis. Defined after running ``self.get_caps()``.
        """
        return getattr(self, "_runs", None)

    @property
    def standardize(self) -> Union[bool, None]:
        """
        Whether region-of-interests (ROIs)/columns were standardized during analysis.
        Defined after running ``self.get_caps()``.
        """
        return getattr(self, "_standardize", None)

    @property
    def concatenated_timeseries(self) -> Union[dict[str, NDArray[np.floating]], None]:
        """
        Group-specific concatenated timeseries data. Can be deleted using
        ``del self.concatenated_timeseries``. Defined after running ``self.get_caps()``. Returns
        a reference.

        ::

            {"GroupName": np.array(shape=[(participants x TRs), ROIs])}

        .. note:: For versions >= 0.25.0, subject IDs are sorted lexicographically prior to\
        concatenation and the order is determined by ``self.groups``.
        """
        return getattr(self, "_concatenated_timeseries", None)

    @concatenated_timeseries.deleter
    def concatenated_timeseries(self) -> None:
        del self._concatenated_timeseries

    @property
    def means(self) -> Union[dict[str, NDArray[np.floating]], None]:
        """
        Group-specific feature means if standardization was applied. Defined after running
        ``self.get_caps()``. Returns a deep copy.

        ::

            {"GroupName": np.array(shape=[1, ROIs])}
        """
        return copy.deepcopy(getattr(self, "_mean_vec", None))

    @property
    def stdev(self) -> Union[dict[str, NDArray[np.floating]], None]:
        """
        Group-specific feature standard deviations if standardization was applied. Defined after
        running ``self.get_caps()``. Returns a deep copy.

        ::

            {"GroupName": np.array(shape=[1, ROIs])}

        .. note:: Standard deviations below ``np.finfo(std.dtype).eps`` are replaced with 1 for\
        numerical stability.
        """
        return copy.deepcopy(getattr(self, "_stdev_vec", None))

    @property
    def kmeans(self) -> Union[dict[str, KMeans], None]:
        """
        Group-specific k-means models. Defined after running ``self.get_caps()``. Returns a deep
        copy.

        ::

            {"GroupName": sklearn.cluster.KMeans}
        """
        return copy.deepcopy(getattr(self, "_kmeans", None))

    @property
    def caps(self) -> Union[dict[str, dict[str, NDArray[np.floating]]], None]:
        """
        Cluster centroids for each group and CAP. Defined after running ``self.get_caps()``. Returns
        a deep copy.
        """
        return copy.deepcopy(getattr(self, "_caps", None))

    @property
    def cluster_scores(self) -> Union[dict[str, Union[str, dict[str, float]]], None]:
        """
        Scores for different cluster sizes by group. Defined after running ``self.get_caps()``.

        ::

            {"Cluster_Selection_Method": str, "Scores": {"GroupName": {2: float, 3: float}}}
        """
        return getattr(self, "_cluster_scores", None)

    @property
    def cluster_selection_method(self) -> Union[str, None]:
        """
        Method used to identify the optimal number of clusters. Defined after running
        ``self.get_caps()``.
        """
        attr = getattr(self, "_cluster_scores", None)

        if attr:
            return attr["Cluster_Selection_Method"]
        else:
            return attr

    @property
    def optimal_n_clusters(self) -> Union[dict[str, int], None]:
        """
        Optimal number of clusters by group if cluster selection was used. Defined after running
        ``self.get_caps()``.

        ::

            {"GroupName": int}
        """
        return getattr(self, "_optimal_n_clusters", None)

    @property
    def variance_explained(self) -> Union[dict[str, float], None]:
        """
        Total variance explained by each group's model. Defined after running ``self.get_caps()``.

        ::

            {"GroupName": float}
        """
        return getattr(self, "_variance_explained", None)

    # Generated in `caps2plot`
    @property
    def region_means(
        self,
    ) -> Union[dict[str, dict[str, Union[list[str], NDArray[np.floating]]]], None]:
        """
        Region-averaged values used for visualization. Defined after running ``self.caps2plot()``.

        ::

            {"GroupName": {"Regions": [...], "CAP-1": np.array(shape=[1, Regions]), ...)}}
        """
        return getattr(self, "_region_means", None)

    @property
    def outer_products(self) -> Union[dict[str, dict[str, NDArray[np.floating]]], None]:
        """
        Outer product matrices for visualization. Defined after running ``self.caps2plot()``.

        ::

            {"GroupName": {"CAP-1": np.array(shape=[ROIs, ROIs]), ...}}
        """
        return getattr(self, "_outer_products", None)

    # Generated in `caps2radar`
    @property
    def cosine_similarity(
        self,
    ) -> Union[dict[str, dict[str, Union[list[str], NDArray[np.floating]]]], None]:
        """
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
        """
        return getattr(self, "_cosine_similarity", None)

    def _concatenated_timeseries_size(self) -> str:
        """Computes the size of the concatenated timeseries object in bytes."""
        if not self.concatenated_timeseries:
            return "0 bytes"

        total_bytes = sum(arr.nbytes for arr in self.concatenated_timeseries.values())
        total_bytes += sys.getsizeof(self.concatenated_timeseries)

        return f"{total_bytes} bytes"

    def __str__(self) -> str:
        """
        Print Current Object State.

        Provides a formatted summary of the ``CAP`` configuration when called with ``print(self)``.
        Returns a string containing the following information:

        - Parcellation approach used
        - Group definitions
        - Clustering configuration (e.g, number of clusters, selection method, etc)
        - Optimal number of clusters  per group (if a range of clusters were provided)
        - Number of CPU cores used for clustering (multiprocessing)
        - Run identifiers used in analysis
        - Estimated memory usage estimate for ``concatenated_timeseries`` (in bytes)
        - Standardization applied prior to clustering
        - Co-Activation Patterns (CAPs) per group
        - Variance explained by clustering

        Returns
        -------
        str
            A formatted string containing information about the object's current state.

        Example
        -------

        >>> cap_analysis = CAP()
        >>> print(cap_analysis)
        Current Object State:
        =====================
        Parcellation Approach                                       : None
        ...
        """
        parcellation_name = list(self.parcel_approach)[0] if self.parcel_approach else None
        # Get group names
        groups_names = ", ".join(f"{k}" for k in self.groups) if self.groups else None
        # Get total CAPs per group
        group_caps = {k: len(v) for k, v in self.caps.items()} if self.caps else None
        # Get additional information
        method = self.cluster_selection_method
        optimal_n = self.optimal_n_clusters
        data_size = self._concatenated_timeseries_size()
        var_explained = self.variance_explained

        object_properties = (
            f"Parcellation Approach                                       : {parcellation_name}\n"
            f"Groups                                                      : {groups_names}\n"
            f"Number of Clusters                                          : {self.n_clusters}\n"
            f"Cluster Selection Method                                    : {method}\n"
            f"Optimal Number of Clusters (if Range of Clusters Provided)  : {optimal_n}\n"
            f"CPU Cores Used for Clustering (Multiprocessing)             : {self.n_cores}\n"
            f"User-Specified Runs IDs Used for Clustering                 : {self.runs}\n"
            f"Concatenated Timeseries Bytes                               : {data_size}\n"
            f"Standardized Concatenated Timeseries                        : {self.standardize}\n"
            f"Co-Activation Patterns (CAPs)                               : {group_caps}\n"
            f"Variance Explained by Clustering                            : {var_explained}"
        )

        sep = "=" * len(object_properties.rsplit(": ")[0])

        return "Current Object State:\n" + sep + f"\n{object_properties}"
