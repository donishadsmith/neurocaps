# A class which is responsible for acessing all TimeseriesExtractorGetter and to keep track of all attributes in TimeSeriesExtractor
class _TimeseriesExtractorGetter:
    def __init__(self):
        pass
    
    #### Exists upon initialization of TimeseriesExtractor
    @property
    def space(self):
        return self._space
    
    @property
    def standardize(self):
        return self._standardize
    
    @property
    def detrend(self):
        return self._detrend
    
    @property
    def low_pass(self):
        return self._low_pass
    
    @property
    def high_pass(self):
        return self._high_pass
    
    @property
    def n_rois(self):
        return self._n_rois
    
    @property
    def n_networks(self):
        return self._n_networks
    
    @property
    def use_confounds(self):
        return self._use_confounds
    
    @property
    def confound_names(self):
        return self._confound_names
    
    @property
    def atlas(self):
        return self._atlas

    @property
    def atlas_labels(self):
        return self._atlas_labels
    
    @property
    def atlas_indices(self):
        return enumerate(self._atlas_labels)
    
    @property
    def atlas_networks(self):
        return self._atlas_networks
    
    ### Does not exists upon initialization of Timeseries Extractor

    # Exist when TimeSeriesExtractor.get_bold() used
    @property
    def task(self):
        if hasattr(self, "_task"): return self._task
        else: return None
            
    @property
    def condition(self):
        if hasattr(self, "_condition"): return self._condition
        else: return None
    
    @property
    def session(self):
        if hasattr(self, "_session"): return self._session
        else: return None
    
    @property
    def run(self):
        if hasattr(self, "_run"): return self._run
        else: return None
    
    @property
    def tr(self):
        if hasattr(self, "_tr"): return self._tr
        else: return None
    
    # Gets initialized and populated in TimeSeriesExtractor.get_bold(), 
    @property
    def subject_ids(self):
        if hasattr(self, "_subject_ids"): return self._subject_ids
        else: return None
    
    # Gets initialized in TimeSeriesExtractor.get_bold(), gets populated when TimeseriesExtractor._timeseries_aggregator
    # gets called in TimeseriesExtractor._extract_timeseries
    @property
    def subject_timeseries(self):
        if hasattr(self, "_subject_timeseries"): return self._subject_timeseries
        else: return None
    
    
        