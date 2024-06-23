"""# A class which is responsible for accessing all TimeseriesExtractorGetter and to keep track of all
attributes in TimeSeriesExtractor"""
import copy, textwrap
import numpy as np
from .._check_parcel_approach import _check_parcel_approach
from .._pickle_to_dict import _convert_pickle_to_dict

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
        if isinstance(parcel_dict, str) and parcel_dict.endswith(".pkl"):
            parcel_dict = _convert_pickle_to_dict(parcel_dict)
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_dict, call="setter")

    ### Does not exists upon initialization of Timeseries Extractor

    # Exist when TimeSeriesExtractor.get_bold() used
    @property
    def task_info(self):
        return self._task_info if hasattr(self, "_task_info") else None

    # Gets initialized and populated in TimeSeriesExtractor.get_bold(),
    @property
    def subject_ids(self):
        return self._subject_ids if hasattr(self, "_subject_ids") else None

    @property
    def n_cores(self):
        return self._n_cores if hasattr(self, "_n_cores") else None

    # Gets initialized in TimeSeriesExtractor.get_bold(), gets populated when
    # TimeseriesExtractor._timeseries_aggregator gets called in TimeseriesExtractor._extract_timeseries
    @property
    def subject_timeseries(self):
        return self._subject_timeseries if hasattr(self, "_subject_timeseries") else None

    @subject_timeseries.setter
    def subject_timeseries(self, subject_dict):
        error_message = textwrap.dedent("""
                        A valid pickle file/be a nested dictionary where the first level is the subject id, second level
                        is the run number in the form of 'run-#', and the final level is the timeseries as a numpy
                        array.
                        """)
        if isinstance(subject_dict, str) and subject_dict.endswith(".pkl"):
            self._subject_timeseries = _convert_pickle_to_dict(subject_dict)
        elif isinstance(subject_dict, dict):
            first_level_indx = list(subject_dict)[0]
            if isinstance(subject_dict[first_level_indx], dict) and len(subject_dict[first_level_indx]) != 0 and "run" in list(subject_dict[first_level_indx])[0]:
                run = list(subject_dict[first_level_indx])[0]
                if isinstance(subject_dict[first_level_indx][run],np.ndarray):
                    self._subject_timeseries = copy.deepcopy(subject_dict)
                else: raise TypeError(error_message)
            else: raise TypeError(error_message)
        else: raise TypeError(error_message)
