"""A class which is responsible for accessing attributes in ``CAP``."""

import copy, sys
from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from ..check_parcel_approach import _check_parcel_approach
from ...typing import ParcelConfig, ParcelApproach


class _CAPGetter:
    def __init__(self):
        pass

    ### Attributes exist when CAP initialized
    @property
    def parcel_approach(self) -> Union[ParcelApproach, None]:
        return self._parcel_approach

    @parcel_approach.setter
    def parcel_approach(self, parcel_dict: Union[ParcelConfig, ParcelApproach, str]) -> None:
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_dict, call="setter")

    @property
    def groups(self) -> Union[dict[str, list[str]], None]:
        return copy.deepcopy(self._groups)

    ### Attributes exist when CAP.get_caps() used
    @property
    def subject_table(self) -> Union[dict[str, str], None]:
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
        return getattr(self, "_n_clusters", None)

    @property
    def n_cores(self) -> Union[int, None]:
        return getattr(self, "_n_cores", None)

    @property
    def runs(self) -> Union[list[Union[int, str]], None]:
        return getattr(self, "_runs", None)

    @property
    def standardize(self) -> Union[bool, None]:
        return getattr(self, "_standardize", None)

    @property
    def concatenated_timeseries(self) -> Union[dict[str, NDArray[np.floating]], None]:
        return getattr(self, "_concatenated_timeseries", None)

    @concatenated_timeseries.deleter
    def concatenated_timeseries(self) -> None:
        del self._concatenated_timeseries

    @property
    def means(self) -> Union[dict[str, NDArray[np.floating]], None]:
        return copy.deepcopy(getattr(self, "_mean_vec", None))

    @property
    def stdev(self) -> Union[dict[str, NDArray[np.floating]], None]:
        return copy.deepcopy(getattr(self, "_stdev_vec", None))

    @property
    def kmeans(self) -> Union[dict[str, KMeans], None]:
        return copy.deepcopy(getattr(self, "_kmeans", None))

    @property
    def caps(self) -> Union[dict[str, dict[str, NDArray[np.floating]]], None]:
        return copy.deepcopy(getattr(self, "_caps", None))

    @property
    def cluster_scores(self) -> Union[dict[str, Union[str, dict[str, float]]], None]:
        return getattr(self, "_cluster_scores", None)

    @property
    def cluster_selection_method(self) -> Union[str, None]:
        attr = getattr(self, "_cluster_scores", None)

        if attr:
            return attr["Cluster_Selection_Method"]
        else:
            return attr

    @property
    def optimal_n_clusters(self) -> Union[dict[str, int], None]:
        return getattr(self, "_optimal_n_clusters", None)

    @property
    def variance_explained(self) -> Union[dict[str, float], None]:
        return getattr(self, "_variance_explained", None)

    # Generated in `caps2plot`
    @property
    def region_means(
        self,
    ) -> Union[dict[str, dict[str, Union[list[str], NDArray[np.floating]]]], None]:
        return getattr(self, "_region_means", None)

    @property
    def outer_products(self) -> Union[dict[str, dict[str, NDArray[np.floating]]], None]:
        return getattr(self, "_outer_products", None)

    # Generated in `caps2radar`
    @property
    def cosine_similarity(
        self,
    ) -> Union[dict[str, dict[str, Union[list[str], NDArray[np.floating]]]], None]:
        return getattr(self, "_cosine_similarity", None)

    def _concatenated_timeseries_size(self) -> str:
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
