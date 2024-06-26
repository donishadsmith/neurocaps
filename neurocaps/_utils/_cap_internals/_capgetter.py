"""A class which is responsible for accessing all CAP metadata and to keep track of all attributes in CAP"""
import copy
from .._check_parcel_approach import _check_parcel_approach
from .._pickle_to_dict import _convert_pickle_to_dict

class _CAPGetter:
    def __init__(self):
        pass

    ### Attributes exist when CAP initialized
    @property
    def groups(self):
        return self._groups

    @property
    def parcel_approach(self):
        return self._parcel_approach

    @parcel_approach.setter
    def parcel_approach(self, parcel_dict):
        if isinstance(parcel_dict, str) and parcel_dict.endswith(".pkl"):
            parcel_dict = _convert_pickle_to_dict(parcel_dict)
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_dict, call="setter")

    ### Attributes exist when CAP.get_caps() used
    @property
    def n_clusters(self):
        return self._n_clusters if hasattr(self, "_n_clusters") else None

    @property
    def cluster_selection_method(self):
        return self._cluster_selection_method if hasattr(self, "_cluster_selection_method") else None

    @property
    def n_cores(self):
        return self._n_cores if hasattr(self, "_n_cores") else None

    @property
    def runs(self):
        return self._runs if hasattr(self, "_runs") else None

    @property
    def caps(self):
        return self._caps if hasattr(self, "_caps") else None

    @property
    def kmeans(self):
        return self._kmeans if hasattr(self, "_kmeans") else None

    @property
    def davies_bouldin(self):
        return self._davies_bouldin if hasattr(self, "_davies_bouldin") else None

    @property
    def silhouette_scores(self):
        return self._silhouette_scores if hasattr(self, "_silhouette_scores") else None

    @property
    def inertia(self):
        return self._inertia if hasattr(self, "_inertia") else None

    @property
    def variance_ratio(self):
        return self._variance_ratio if hasattr(self, "_variance_ratio") else None

    @property
    def optimal_n_clusters(self):
        return self._optimal_n_clusters if hasattr(self, "_optimal_n_clusters")else None

    @property
    def standardize(self):
        return self._standardize if hasattr(self, "_standardize") else None

    @property
    def means(self):
        return self._mean_vec if hasattr(self, "_mean_vec") else None

    @property
    def stdev(self):
        return self._stdev_vec if hasattr(self, "_stdev_vec") else None

    @property
    def concatenated_timeseries(self):
        return self._concatenated_timeseries if hasattr(self, "_concatenated_timeseries") else None

    # Generated in `caps2plot`
    @property
    def region_caps(self):
        return self._region_caps if hasattr(self, "_region_caps") else None

    @property
    def outer_products(self):
        return self._outer_products if hasattr(self, "_outer_product") else None

    @property
    def subject_table(self):
        return self._subject_table if hasattr(self, "_subject_table") else None

    @subject_table.setter
    def subject_table(self, subject_dict):
        if isinstance(subject_dict, dict):
            self._subject_table = copy.deepcopy(subject_dict)

    @property
    def cosine_similarity(self):
        return self._cosine_similarity if hasattr(self, "_cosine_similarity") else None
