"""A class which is responsible for accessing all CAP metadata and to keep track of all attributes in CAP"""
from .._timeseriesextractor_internals._check_parcel_approach import _check_parcel_approach

class _CAPGetter:
    def __init__(self):
        pass

    ### Attributes exist when CAP initialized
    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def cluster_selection_method(self):
        return self._cluster_selection_method

    @property
    def groups(self):
        return self._groups

    @property
    def parcel_approach(self):
        return self._parcel_approach

    @parcel_approach.setter
    def parcel_approach(self, parcel_dict):
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_dict, call="setter")

    ### Attributes exist when CAP.get_caps() used
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
    def silhouette_scores(self):
        return self._silhouette_scores if hasattr(self, "_silhouette_scores") else None

    @property
    def inertia(self):
        return self._inertia if hasattr(self, "_inertia") else None

    @property
    def optimal_n_clusters(self):
        return self._optimal_n_clusters if hasattr(self, "_optimal_n_clusters")else None

    @property
    def standardize(self):
        return self._standardize if hasattr(self, "_standardizee") else None

    @property
    def epsilon(self):
        return self._epsilon if hasattr(self, "_epsilon") else None

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
