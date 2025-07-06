"""A class which is responsible for accessing attributes in ``TimeSeriesExtractor``."""

import copy, sys
from typing import Union

import numpy as np

from neurocaps.typing import ParcelConfig, ParcelApproach, SubjectTimeseries
from neurocaps.utils import _io as io_utils
from neurocaps.utils._parcellation_validation import check_parcel_approach


class TimeseriesExtractorGetter:
    def __init__(self):
        pass

    #### Exists upon initialization of TimeseriesExtractor
    @property
    def space(self) -> str:
        """
        The standard template space that the preprocessed BOLD data is registered to. This property
        is also settable.
        """
        return self._space

    @space.setter
    def space(self, new_space: str) -> None:
        if not isinstance(new_space, str):
            raise TypeError("`space` must be a string.")
        self._space = new_space

    @property
    def parcel_approach(self) -> ParcelApproach:
        """
        Parcellation information with "maps" (path to parcellation file), "nodes" (labels), and
        "regions" (anatomical regions or networks). Returns a deep copy.
        """
        return copy.deepcopy(self._parcel_approach)

    @parcel_approach.setter
    def parcel_approach(self, parcel_dict: Union[ParcelConfig, ParcelApproach, str]) -> None:
        self._parcel_approach = check_parcel_approach(parcel_approach=parcel_dict, call="setter")

    @property
    def signal_clean_info(self) -> Union[dict[str, Union[bool, int, float, str]], None]:
        """
        Dictionary containing signal cleaning parameters. Returns a deep copy.
        """
        return copy.deepcopy(self._signal_clean_info)

    ### Does not exists upon initialization of Timeseries Extractor

    # Exist when TimeSeriesExtractor.get_bold() used
    @property
    def task_info(self) -> Union[dict[str, Union[str, int]], None]:
        """
        Dictionary containing all task-related information such. Defined after running
        ``self.get_bold()``.
        """
        return getattr(self, "_task_info", None)

    # Gets initialized and populated in TimeSeriesExtractor.get_bold(),
    @property
    def subject_ids(self) -> Union[list[str], None]:
        """
        A list containing all subject IDs retrieved from ``BIDSLayout`` for timeseries extraction.
        Defined after running ``self.get_bold()``.
        """
        return getattr(self, "_subject_ids", None)

    @property
    def n_cores(self) -> Union[int, None]:
        """
        Number of cores used for multiprocessing with Joblib. Defined after running
        ``self.get_bold()``.
        """
        return getattr(self, "_n_cores", None)

    @property
    def subject_timeseries(self) -> Union[SubjectTimeseries, None]:
        """
        A dictionary mapping subject IDs to their run IDs and their associated timeseries
        (TRs x ROIs) as a NumPy array. Can be deleted using ``del self.subject_timeseries``.
        Defined after running ``self.get_bold()``. This property is also settable (accepts a
        dictionary or pickle file). Returns a reference.
        """
        return getattr(self, "_subject_timeseries", None)

    @subject_timeseries.setter
    def subject_timeseries(self, subject_dict: Union[SubjectTimeseries, str]) -> None:
        subject_dict = io_utils.get_obj(subject_dict)

        self._validate_timeseries(subject_dict)

        self._subject_timeseries = subject_dict

    @subject_timeseries.deleter
    def subject_timeseries(self) -> None:
        del self._subject_timeseries

    @property
    def qc(self) -> Union[dict[str, dict[str, dict[str, Union[float, int]]]], None]:
        """
        A dictionary reporting quality control, which maps subject IDs to their run IDs and
        information related to framewise displacement and dummy scans. Returns a reference.

        ::

            {"subjectID": {"run-ID": {"mean_fd": float, "std_fd": float, ...}}}
        """
        return getattr(self, "_qc", None)

    @staticmethod
    def _validate_timeseries(subject_dict: SubjectTimeseries) -> None:
        """Validates ``subject_timeseries`` structure."""
        error_msg = (
            "A valid pickle file/subject timeseries should contain a nested dictionary where the "
            "first level is the subject id, second level is the run number in the form of 'run-#', "
            "and the final level is the timeseries as a numpy array. "
        )

        error_dict = {
            "Sub": error_msg + "The error occurred at [SUBJECT: {0}]. ",
            "Run": error_msg + "The error occurred at [SUBJECT: {0} | RUN: {1}]. ",
        }

        if not isinstance(subject_dict, dict):
            raise TypeError(error_msg)

        for sub in subject_dict:
            if not isinstance(subject_dict[sub], dict):
                raise TypeError(
                    error_dict["Sub"].format(sub)
                    + "The subject must be a dictionary with second level "
                    "'run-#' keys."
                )

            runs = list(subject_dict[sub])

            if not all("run" in x for x in runs):
                raise TypeError(
                    error_dict["Sub"].format(sub)
                    + "Not all second level keys follow the form of 'run-#'."
                )

            for run in runs:
                if not isinstance(subject_dict[sub][run], np.ndarray):
                    raise TypeError(
                        error_dict["Run"].format(sub, run)
                        + "All 'run-#' keys must contain a numpy array."
                    )

    def _subject_timeseries_size(self) -> str:
        """Computes the byte size of ``self.subject_timeseries``."""
        if not self.subject_timeseries:
            return "0 bytes"

        total_bytes = sum(
            arr.nbytes for subject in self.subject_timeseries.values() for arr in subject.values()
        )
        # Adding size of dictionary
        total_bytes += sys.getsizeof(self.subject_timeseries)

        return f"{total_bytes} bytes"

    def __str__(self) -> str:
        """
        Print Current Object State.

        Provides a formatted summary of the ``TimeseriesExtractor`` configuration when called with
        ``print(self)``. Returns a string containing the following information:

        - Preprocessed BOLD template space
        - Parcellation approach used
        - Signal cleaning parameters applied
        - Task information
        - Number of subjects in ``subject_timeseries``
        - Number of CPU cores used for extraction (multiprocessing)
        - Estimated memory usage estimate for ``subject_timeseries`` (in bytes)

        Returns
        -------
        str
            A formatted string containing information about the object's current state.

        Example
        -------
        >>> from neurocaps.extraction import TimeseriesExtractor
        >>> extractor = TimeseriesExtractor()
        >>> print(extractor)
            Current Object State:
            =====================
            Preprocessed BOLD Template Space                           : "MNI152NLin2009cAsym"
            ...
        """
        # Store some information in variables
        n_subjects = len(self.subject_ids) if self.subject_ids else None
        parcel_approach = list(self.parcel_approach)[0]
        clean_params = self.signal_clean_info
        data_size = self._subject_timeseries_size()

        object_properties = (
            f"Preprocessed BOLD Template Space                           : {self.space}\n"
            f"Parcellation Approach                                      : {parcel_approach}\n"
            f"Signal Cleaning Parameters                                 : {clean_params}\n"
            f"Task Information                                           : {self.task_info}\n"
            f"Number of Subjects                                         : {n_subjects}\n"
            f"CPU Cores Used for Timeseries Extraction (Multiprocessing) : {self.n_cores}\n"
            f"Subject Timeseries Byte Size                               : {data_size}"
        )

        sep = "=" * len(object_properties.rsplit(": ")[0])

        return "Current Object State:\n" + sep + f"\n{object_properties}"
