"""A class which is responsible for accessing all CAP metadata and to keep track of all attributes in CAP"""

import copy, os
from typing import Union

import numpy as np
from ..check_parcel_approach import _check_parcel_approach
from sklearn.cluster import KMeans


class _CAPGetter:
    def __init__(self):
        pass

    ### Attributes exist when CAP initialized
    @property
    def parcel_approach(
        self,
    ) -> Union[
        dict[str, dict[str, Union[os.PathLike, list[str]]]],
        dict[str, dict[str, Union[os.PathLike, list[str], dict[str, dict[str, list[int]]]]]],
    ]:
        return self._parcel_approach

    @parcel_approach.setter
    def parcel_approach(self, parcel_dict):
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_dict, call="setter")

    @property
    def groups(self) -> Union[dict[str, list[str]], None]:
        return self._groups

    ### Attributes exist when CAP.get_caps() used
    @property
    def n_clusters(self) -> Union[int, list[int], None]:
        return self._n_clusters if hasattr(self, "_n_clusters") else None

    @property
    def cluster_selection_method(self) -> Union[str, None]:
        return self._cluster_selection_method if hasattr(self, "_cluster_selection_method") else None

    @property
    def n_cores(self) -> Union[int, None]:
        return self._n_cores if hasattr(self, "_n_cores") else None

    @property
    def runs(self) -> Union[list[Union[int, str]], None]:
        return self._runs if hasattr(self, "_runs") else None

    @property
    def caps(self) -> Union[dict[str, dict[str, np.array]], None]:
        return self._caps if hasattr(self, "_caps") else None

    @property
    def kmeans(self) -> Union[dict[str, KMeans], None]:
        return self._kmeans if hasattr(self, "_kmeans") else None

    @property
    def cluster_scores(self) -> Union[dict[str, Union[str, dict[str, float]]], None]:
        return self._cluster_scores if hasattr(self, "_cluster_scores") else None

    @property
    def variance_explained(self) -> Union[dict[str, float], None]:
        return self._variance_explained if hasattr(self, "_variance_explained") else None

    @property
    def optimal_n_clusters(self) -> Union[dict[str, int], None]:
        return self._optimal_n_clusters if hasattr(self, "_optimal_n_clusters") else None

    @property
    def standardize(self) -> Union[bool, None]:
        return self._standardize if hasattr(self, "_standardize") else None

    @property
    def means(self) -> Union[dict[str, np.array], None]:
        return self._mean_vec if hasattr(self, "_mean_vec") else None

    @property
    def stdev(self) -> Union[dict[str, np.array], None]:
        return self._stdev_vec if hasattr(self, "_stdev_vec") else None

    @property
    def concatenated_timeseries(self) -> Union[dict[str, np.array], None]:
        return self._concatenated_timeseries if hasattr(self, "_concatenated_timeseries") else None

    @concatenated_timeseries.deleter
    def concatenated_timeseries(self):
        del self._concatenated_timeseries

    # Generated in `caps2plot`
    @property
    def region_caps(self) -> Union[dict[str, dict[str, np.array]], None]:
        return self._region_caps if hasattr(self, "_region_caps") else None

    @property
    def outer_products(self) -> Union[dict[str, dict[str, np.array]], None]:
        return self._outer_products if hasattr(self, "_outer_products") else None

    @property
    def subject_table(self) -> Union[dict[str, str], None]:
        return self._subject_table if hasattr(self, "_subject_table") else None

    @subject_table.setter
    def subject_table(self, subject_dict):
        if isinstance(subject_dict, dict):
            self._subject_table = copy.deepcopy(subject_dict)
        else:
            raise TypeError(
                "`self.subject_table` must be a dictionary where the keys are the subject IDs and the "
                "values are the group names."
            )

    @property
    def cosine_similarity(self) -> Union[dict[str, Union[list[str], dict[str, dict[str, float]]]], None]:
        return self._cosine_similarity if hasattr(self, "_cosine_similarity") else None
