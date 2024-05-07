import numpy as np
from ._check_parcel_approach import _check_parcel_approach
# A class which is responsible for acessing all TimeseriesExtractorGetter and to keep track of all attributes in TimeSeriesExtractor
class _TimeseriesExtractorGetter:
    def __init__(self):
        pass
    
    #### Exists upon initialization of TimeseriesExtractor
    @property
    def space(self):
        return self._space
    
    @property
    def signal_clean_info(self):
        return self._signal_clean_info
    
    @property
    def parcel_approach(self):
        return self._parcel_approach
    
    @parcel_approach.setter
    def parcel_approach(self, parcel_dict):
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_dict, call="setter")

    ### Does not exists upon initialization of Timeseries Extractor

    # Exist when TimeSeriesExtractor.get_bold() used
    @property
    def task_info(self):
        if hasattr(self, "_task_info"): return self._task_info
        else: return None
    
    # Gets initialized and populated in TimeSeriesExtractor.get_bold(), 
    @property
    def subject_ids(self):
        if hasattr(self, "_subject_ids"): return self._subject_ids
        else: return None
    
    @property
    def n_cores(self):
        if hasattr(self, "_n_cores"): return self._n_cores
        else: return None

    # Gets initialized in TimeSeriesExtractor.get_bold(), gets populated when TimeseriesExtractor._timeseries_aggregator
    # gets called in TimeseriesExtractor._extract_timeseries
    @property
    def subject_timeseries(self):
        if hasattr(self, "_subject_timeseries"): return self._subject_timeseries
        else: return None
    
    @subject_timeseries.setter
    def subject_timeseries(self, subject_dict):
        error_message = "Must be a nested dictionary where the first level is the subject id, second level is the run number in the form of 'run-#', and the final level is the timeseris as a numpy array."
        if isinstance(subject_dict, dict): 
            first_level_indx = list(subject_dict.keys())[0]
            if isinstance(subject_dict[first_level_indx], dict) and len(subject_dict[first_level_indx].keys()) != 0 and "run" in list(subject_dict[first_level_indx].keys())[0]: 
                run = list(subject_dict[first_level_indx].keys())[0]
                if isinstance(subject_dict[first_level_indx][run],np.ndarray): 
                    self._subject_timeseries = subject_dict
                else: raise TypeError(error_message) 
            else: raise TypeError(error_message) 
        else: raise TypeError(error_message)
    
    
        