"""A class which is responsible for accessing all TimeseriesExtractorGetter and to keep track of all attributes in
TimeSeriesExtractor"""
import copy
import numpy as np
from ..check_parcel_approach import _check_parcel_approach
from ..pickle_utils import _convert_pickle_to_dict

class _TimeseriesExtractorGetter:
    def __init__(self):
        pass

    #### Exists upon initialization of TimeseriesExtractor
    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, new_space):
        if not isinstance(new_space, str): raise TypeError("`space` must be a string.")
        self._space = new_space

    @property
    def parcel_approach(self):
        return self._parcel_approach

    @property
    def signal_clean_info(self):
        return self._signal_clean_info

    @parcel_approach.setter
    def parcel_approach(self, parcel_dict):
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
        need_deepcopy = True

        if isinstance(subject_dict, str) and subject_dict.endswith(".pkl"):
            subject_dict = _convert_pickle_to_dict(subject_dict)
            need_deepcopy = False

        self._validate_timeseries(subject_dict)

        if need_deepcopy: self._subject_timeseries = copy.deepcopy(subject_dict)
        else: self._subject_timeseries = subject_dict

    @subject_timeseries.deleter
    def subject_timeseries(self):
        del self._subject_timeseries

    @staticmethod
    def _validate_timeseries(subject_dict):
        error_msg = ("A valid pickle file/subject timeseries should contain a nested dictionary where the "
                     "first level is the subject id, second level is the run number in the form of 'run-#', and "
                     "the final level is the timeseries as a numpy array. ")

        error_dict = {"Sub": error_msg + "The error occurred at [SUBJECT: {0}]. ",
                      "Run": error_msg + "The error occurred at [SUBJECT: {0} | RUN: {1}]. "}

        if not isinstance(subject_dict, dict): raise TypeError(error_msg)

        for sub in subject_dict:
            if not isinstance(subject_dict[sub], dict):
                raise TypeError(error_dict["Sub"].format(sub) + "The subject must be a dictionary with second level "
                                "'run-#' keys.")

            runs = list(subject_dict[sub])

            if not all("run" in x for x in runs):
                raise TypeError(error_dict["Sub"].format(sub) + "Not all second level keys follow the form of 'run-#'.")

            for run in runs:
                if not isinstance(subject_dict[sub][run], np.ndarray):
                    raise TypeError(error_dict["Run"].format(sub, run) + "All 'run-#' keys must contain a numpy array.")
