# A class which is responsible for accessing all CAPMetadata and to keep track of all attributes in CAP
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
    def node_labels(self):
        return self._node_labels
    
    @property
    def node_networks(self):
        return self._node_networks
    
    ### Attributes exist when CAP.get_caps() used
    @property
    def runs(self):
        if hasattr(self, "_runs"): return self._runs
        else: return None

    @property
    def caps(self):
        if hasattr(self, "_caps"): return self._caps
        else: return None
    
    @property
    def kmeans(self):
        if hasattr(self, "_kmeans"): return self._kmeans
        else: return None
    
    @property
    def silhouette_scores(self):
        if hasattr(self, "_silhouette_scores"): return self._silhouette_scores
        else: return None

    @property
    def inertia(self):
        if hasattr(self, "_inertia"): return self._inertia
        else: return None

    @property
    def optimal_n_clusters(self):
        if hasattr(self, "_optimal_n_clusters"): return self._optimal_n_clusters
        else: return None
    
    @property
    def network_caps(self):
        if hasattr(self, "_network_caps"): return self._network_caps
        else: return None

    @property
    def outer_product(self):
        if hasattr(self, "_outer_product"): return self._outer_product
        else: return None
    
    @property
    def standardize(self):
        if hasattr(self, "_standardizee"): return self._standardize
        else: return None

    @property
    def epsilon(self):
        if hasattr(self, "_epsilon"): return self._epsilon
        else: return None


    


    
     
     
     
     
         
