import json, os, re
from functools import lru_cache
from typing import Callable, Literal, Optional, Union

import matplotlib.pyplot as plt, numpy as np
from joblib import Parallel, delayed, dump

from .._utils import (_TimeseriesExtractorGetter, _check_kwargs, _check_confound_names,
                      _check_parcel_approach, _extract_timeseries, _logger)

LG = _logger(__name__)

# Custom error class
class BIDSQueryError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"{self.message}"

class TimeseriesExtractor(_TimeseriesExtractorGetter):
    """
    Timeseries Extractor Class.

    Initializes the Timeseries Extractor class.

    Parameters
    ----------
    space: :obj:`str`, default="MNI152NLin2009cAsym"
        The standard template space that the preprocessed bold data is registered to. Used for querying with pybids
        to locate preprocessed BOLD-related files.

    parcel_approach: :obj:`dict[str, dict[str, Union[str, int]]]` or :obj:`os.PathLike`, \
                     default={"Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}}
        The approach to parcellate NifTI images. This must be a nested dictionary with the first key being the
        parcellation name. Currently, only "Schaefer", "AAL", and "Custom" are supported. Recognized second level
        keys (sub-keys) are listed below:

        - For "Schaefer":

            - "n_rois": The number of ROIs (100, 200, 300, 400, 500, 600, 700, 800, 900, or 1000). Defaults to 400.
            - "yeo_networks": The number of Yeo networks (7 or 17). Defaults to 7.
            - "resolution_mm": The spatial resolution of the parcellation in millimeters (1 or 2). Defaults to 1.

        - For "AAL":

            - "version": The version of the AAL atlas used ("SPM5", "SPM8", "SPM12", or "3v2"). Defaults to "SPM12" if ``{"AAL": {}}`` is supplied.

        - For "Custom":

            - "maps": Directory path to the location of the parcellation file.
            - "nodes": A list of node names in the order of the label IDs in the parcellation.
            - "regions": The regions or networks in the parcellation.

        Refer to documentation from nilearn's ``datasets.fetch_atlas_schaefer_2018`` and ``datasets.fetch_atlas_aal``
        functions for more information about the "Schaefer" and "AAL" sub-keys. Also, refer to the "Note" section below
        for an explanation of the "Custom" sub-keys.

    standardize: {"zscore_sample", "zscore", "psc", True, False}, default="zscore_sample"
        Standardizes the timeseries. Refer to ``nilearn.maskers.NiftiLabelsMasker`` for an explanation of each
        available option.

    detrend: :obj:`bool`, default=True
        Detrends the timeseries during extraction.

    low_pass: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Filters out signals above the specified cutoff frequency.

    high_pass: :obj:`float`, :obj:`int`, or :obj:`None``, default=None
        Filters out signals below the specified cutoff frequency.

    fwhm: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Applies spatial smoothing to data (in millimeters). Note that using parcellations already averages voxels
        within parcel boundaries, which can improve signal-to-noise ratio (SNR) assuming Gaussian noise
        distribution. However, smoothing may also blur parcel boundaries.

    use_confounds: :obj:`bool`, default=True
        Perform nuisance regression using the default or user-specified confounds in ``confound_names`` when
        extracting timeseries. Note, the confound tsv files must be located in the same directory as the preprocessed
        BOLD images.

    confound_names: :obj:`list[str]` or :obj:`None`, default=None
        The names of the confounds to extract from the confound tsv files. If None, default confounds are used, which
        consists of all cosine-basis parameters, the six-head motion parameters and their first-order derivatives,
        and the first six combined acompcor components. Additionally, the names of these confounds follow the
        naming scheme of confounds in fMRIPrep versions ``>= 1.2.0``. Note, an asterisk ("*") can be used to find
        confound names that start with the term preceding the asterisk. For instance, "cosine*" will find all confound
        names in the confound files starting with "cosine".

    fd_threshold: :obj:`float`, :obj:`dict[str, float]`, or :obj:`None`, default=None
        Sets a threshold for removing exceeding volumes. This requires a column named `framewise_displacement` in the
        confounds file and ``use_confounds`` set to True. Additionally, `framewise_displacement` should be
        specified in ``confound_names`` if using this parameter. By default, censoring is done after nuisance
        regression; however, this behavior can be modified with the "use_sample_mask" key to censor prior to
        nuisance regression. If, ``fd_threshold`` is a dictionary, the following keys can be specified:

        - "threshold": A float value. Volumes with a `framewise_displacement` value exceeding this threshold are removed.
        - "outlier_percentage": A float value between 0 and 1 representing a percentage. Runs where the proportion of
          volumes exceeding the "threshold" is higher than this percentage are removed. If ``condition`` is specified
          in ``self.get_bold``, only the runs where the proportion of volumes exceeds this value for the specific
          condition of interest are removed. Note, this proportion is calculated after dummy scans have been removed.
          A warning is issued whenever a run is flagged.
        - "n_before": An integer value indicating the number of volumes to scrub before the flagged volume. Hence,
          if frame 5 is flagged and "n_before" is 2, then volumes 3, 4, and 5 are scrubbed.
        - "n_after": An integer indicating the number of volumes to scrub after to the flagged volume. Hence,
          if frame 5 is flagged and "n_after" is 2, then volumes 5, 6, and 7 are scrubbed.
        - "use_sample_mask": A boolean value. If True, a sample mask is generated and passed to the ``sample_mask``
          parameter in nilearn's ``NiftiLabelsMasker`` to censor prior to nuisance regression. Internally,
          ``clean__extrapolate`` is set to False and passed to ``NiftiLabelsMasker``, which prevents censored
          volumes at the end from being interpolated prior to applying the butterworth filter. See
          documentation from ``nilearn.signal_clean`` and  ``nilearn.maskers.NiftiLabelsMasker`` for how nilearn
          handles censored volumes when ``sample_mask`` is used.  If this key is set to False, data is only censored
          after nuisance regression, which is the default behavior.

        .. versionadded:: 0.18.8 "use_sample_mask"

    n_acompcor_separate: :obj:`int` or :obj:`None`, default=None
        Specifies the number of separate acompcor components derived from white-matter (WM) and cerebrospinal
        fluid (CSF) masks to use. For example, if set to 5, the first five components from the WM mask
        and the first five from the CSF mask will be used, totaling ten acompcor components. If this parameter is
        not None, any acompcor components listed in ``confound_names`` will be disregarded. To use acompcor
        components derived from combined masks (WM & CSF), leave this parameter as None and list the specific
        acompcors of interest in ``confound_names``.

    dummy_scans: :obj:`int`, :obj:`dict[str, Union[bool, int]]`, or :obj:`None`, default=None
        Removes the first `n` volumes before extracting the timeseries. If, ``dummy_scans`` is a dictionary,
        the following keys can be used:

        - "auto": A boolean value. If True, the number of dummy scans removed depend on the number of
          "non_steady_state_outlier_XX" columns in the participants fMRIPrep confounds tsv file. For instance, if
          there are two "non_steady_state_outlier_XX" columns detected, then ``dummy_scans`` is set to two since
          there is one "non_steady_state_outlier_XX" per outlier volume for fMRIPrep. This is assessed for each run of
          all participants so ``dummy_scans`` depends on the number number of "non_steady_state_outlier_XX" in the
          confound file associated with the specific participant, task, and run number.
        - "min": An integer value indicating the minimum dummy scans to discard. The "auto" sub-key must be True
          for this to work. If, for instance, only two "non_steady_state_outlier_XX" columns are detected but the
          "min" is set to three, then three dummy volumes will be discarded.
        - "max": An integer value indicating the maximum dummy scans to discard. The "auto" sub-key must be True
          for this to work. If, for instance, six "non_steady_state_outlier_XX" columns are detected but the
          "max" is set to five, then five dummy volumes will be discarded.

    dtype: :obj:`str` or "auto", default=None
        The numpy dtype the NIfTI images are converted to when passed to nilearn's ``load_img`` function.


    Properties
    ----------
    space: str
        The standard template space that the preprocessed BOLD data is registered to. The space can also be set after
        class initialization using ``self.space = "New Space"`` if the template space needs to be changed.

    parcel_approach: :obj:`dict[str, dict[str, Union[os.PathLike, list[str]]]]` or \
                     :obj:`dict[str, dict[str, Union[os.PathLike, list[str], dict[str, dict[str, list[int]]]]]]`
        A dictionary containing information about the parcellation. Can also be used as a setter, which accepts a
        dictionary or a dictionary saved as pickle file. If "Schaefer" or "AAL" was specified during
        initialization of the ``TimeseriesExtractor`` class, then ``nilearn.datasets.fetch_atlas_schaefer_2018``
        and ``nilearn.datasets.fetch_atlas_aal`` will be used to obtain the "maps" and the "nodes". Then string
        splitting is used on the "nodes" to obtain the "regions":

        ::

            # Structure of Schaefer
            {
                "Schaefer":
                {
                    "maps": "path/to/parcellation.nii.gz",
                    "nodes": ["LH_Vis1", "LH_SomSot1", "RH_Vis1", "RH_Somsot1"],
                    "regions": ["Vis", "SomSot"]
                }
            }

            # Structure of AAL
            {
                "AAL":
                {
                    "maps": "path/to/parcellation.nii.gz",
                    "nodes": ["Precentral_L", "Precentral_R", "Frontal_Sup_L", "Frontal_Sup_R"],
                    "regions": ["Precentral", "Frontal"]
                }
            }

        Refer to the example for "Custom" in the Note section below for the expected structure.

    signal_clean_info: :obj:`dict[str, Union[bool, int, float, str]]` or :obj:`None`
        Dictionary containing parameters for signal cleaning specified during initialization of the
        ``TimeseriesExtractor`` class. This information includes ``standardize``, ``detrend``, ``low_pass``,
        ``high_pass``, ``fwhm``, ``dummy_scans``, ``use_confounds``, ``n_compcor_separate``, and ``fd_threshold``.

    task_info: :obj:`dict[str, Union[str, int]]` or :obj:`None`
        If ``self.get_bold()`` ran, is a dictionary containing all task-related information such as ``task``,
        ``condition``, ``session``, ``runs``, and ``tr`` (if specified) else None.

    subject_ids: :obj:`list[str]` or :obj:`None`
        A list containing all subject IDs that have retrieved from pybids and subjected to timeseries
        extraction.

    n_cores: :obj:`int` or :obj:`None`
        Number of cores used for multiprocessing with joblib.

    subject_timeseries: :obj:`dict[str, dict[str, np.ndarray]]` or :obj:`None`
        A dictionary mapping subject IDs to their run IDs and their associated timeseries (TRs x ROIs) as a numpy array.
        Can also be a path to a pickle file containing this same structure. If this property needs to be deleted due
        to memory issues,  ``delattr(self, "_subject_timeseries")`` (version < 0.18.10) or
        ``del self.subject_timeseries`` (version >= 0.18.10) can be used to delete this property and only have it
        return None. The structure is as follows:

        ::

            subject_timeseries = {
                    "101": {
                        "run-0": np.array([...]), # Shape: TRs x ROIs
                        "run-1": np.array([...]), # Shape: TRs x ROIs
                        "run-2": np.array([...]), # Shape: TRs x ROIs
                    },
                    "102": {
                        "run-0": np.array([...]), # Shape: TRs x ROIs
                        "run-1": np.array([...]), # Shape: TRs x ROIs
                    }
                }

    Note
    ----
    **Passed Parameters**: ``standardize``, ``detrend``, ``low_pass``, ``high_pass``, ``fwhm``, and nuisance
    regression (``confound_names``) uses ``nilearn.maskers.NiftiLabelsMasker``. The ``dtype`` parameter is used by
    ``nilearn.image.load_img``. For framewise displacement, if the "use_sample_mask" key is set to True in the
    ``fd_threshold`` dictionary, then a boolean sample mask is generated (setting indices corresponding to high motion
    volumes as False) and is passed to the ``sample_mask`` parameter in ``nilearn.maskers.NiftiLabelsMasker``.

    **Custom Parcellations**: If using a "Custom" parcellation approach, ensure that the parcellation is
    lateralized (where each region/network has nodes in the left and right hemisphere). This is due to certain
    visualization functions assuming that each region consists of left and right hemisphere nodes. Additionally,
    certain visualization functions in this class also assume that the background label is 0. Therefore, do not add a
    background label in the "nodes" or "regions" keys.

    The recognized sub-keys for the "Custom" parcellation approach includes:

    - "maps": Directory path containing the parcellation file in a supported format (e.g., .nii or .nii.gz for NifTI).
    - "nodes": A list of all node labels. The node labels should be arranged in ascending order based on their
      numerical IDs from the parcellation files. The node with the lowest numerical label in the parcellation file
      should occupy the 0th index in the list, regardless of its actual numerical value. For instance, if the numerical
      IDs are sequential, and the lowest, non-background numerical ID in the parcellation is "1" which corresponds
      to "left hemisphere visual cortex area" ("LH_Vis1"), then "LH_Vis1" should occupy the 0th element in this list.
      Even if the numerical IDs are non-sequential and the earliest non-background, numerical ID is "2000"
      (assuming "0" is the background), then the node label corresponding to "2000" should occupy the 0th element of
      this list.

      ::

            # Example of numerical label IDs and their organization in the "nodes" key
            "nodes": {
                "LH_Vis1",          # Corresponds to parcellation label 2000; lowest non-background numerical ID
                "LH_Vis2",          # Corresponds to parcellation label 2100; second lowest non-background numerical ID
                "LH_Hippocampus",   # Corresponds to parcellation label 2150; third lowest non-background numerical ID
                "RH_Vis1",          # Corresponds to parcellation label 2200; fourth lowest non-background numerical ID
                "RH_Vis2",          # Corresponds to parcellation label 2220; fifth lowest non-background numerical ID
                "RH_Hippocampus"    # Corresponds to parcellation label 2300; sixth lowest non-background numerical ID
            }

    - "regions": A dictionary defining major brain regions or networks. Each region should list node indices under
      "lh" (left hemisphere) and "rh" (right hemisphere) to specify the respective nodes. Both the "lh" and "rh"
      sub-keys should contain the indices of the nodes belonging to each region/hemisphere pair, as determined
      by the order/index in the "nodes" list. The naming of the sub-keys defining the major brain regions or networks
      have zero naming requirements and simply define the nodes belonging to the same name.

      ::

            # Example of the "regions" sub-keys
            "regions": {
                "Visual": {
                    "lh": [0, 1], # Corresponds to "LH_Vis1" and "LH_Vis2"
                    "rh": [3, 4]  # Corresponds to "RH_Vis1" and "RH_Vis2"
                },
                "Hippocampus": {
                    "lh": [2], # Corresponds to "LH_Hippocampus"
                    "rh": [5]  # Corresponds to "RH_Hippocampus"
                }
            }

    The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis)
    and hippocampus regions in full:

    ::

        parcel_approach = {
            "Custom": {
                "maps": "/location/to/parcellation.nii.gz",
                "nodes": [
                    "LH_Vis1",
                    "LH_Vis2",
                    "LH_Hippocampus",
                    "RH_Vis1",
                    "RH_Vis2",
                    "RH_Hippocampus"
                ],
                "regions": {
                    "Visual": {
                        "lh": [0, 1],
                        "rh": [3, 4]
                    },
                    "Hippocampus": {
                        "lh": [2],
                        "rh": [5]
                    }
                }
            }
        }

    **Note**: Different sub-keys are required depending on the function used. Refer to the Note section under each
    function for information regarding the sub-keys required for that specific function.
    """
    def __init__(self,
                 space: str="MNI152NLin2009cAsym",
                 parcel_approach: Union[
                    dict[str, dict[str, Union[str, int]]], os.PathLike
                    ]={"Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}},
                 standardize: Union[bool, Literal["zscore_sample", "zscore", "psc"]]="zscore_sample",
                 detrend: bool=True,
                 low_pass: Optional[Union[float, int]]=None,
                 high_pass: Optional[Union[float, int]]=None,
                 fwhm: Optional[Union[float, int]]=None,
                 use_confounds: bool=True,
                 confound_names: Optional[list[str]]=None,
                 fd_threshold: Optional[Union[float, dict[str, float]]]=None,
                 n_acompcor_separate: Optional[int]=None,
                 dummy_scans: Optional[Union[int, dict[str, Union[bool, int]]]]=None,
                 dtype: Union[str, Literal["auto"]]=None) -> None:

        self._space = space

        if isinstance(dummy_scans, dict):
            if "auto" not in dummy_scans:
                raise KeyError("'auto' sub-key must be included when `dummy_scans` is a dictionary.")
            if not use_confounds:
                raise ValueError("`use_confounds` must be True to use 'auto' for `dummy_scans` because the "
                                 "fMRIPrep confounds tsv file is needed to detect the number of "
                                 "'non_steady_state_outlier_XX' columns.")

        if not use_confounds and fd_threshold:
            LG.warning("`fd_threshold` specified but `use_confounds` is not True so removal of volumes after "
                       "nuisance regression will not be done since the fMRIPrep confounds tsv file is needed.")

        if isinstance(fd_threshold, dict):
            if "threshold" not in fd_threshold:
                raise KeyError("'threshold' sub-key must be included in `fd_threshold` dictionary.")
            if "outlier_percentage" in fd_threshold:
                if fd_threshold["outlier_percentage"] >= 1 or fd_threshold["outlier_percentage"] <= 0:
                    raise ValueError("'outlier_percentage' must be a positive float between 0 and 1.")
            if "n_before" in fd_threshold and not isinstance(fd_threshold["n_before"], int):
                raise ValueError("'n_before' must be an integer value.")
            if "n_after" in fd_threshold and not isinstance(fd_threshold["n_after"], int):
                raise ValueError("'n_after' must be an integer value.")
            if "use_sample_mask" in fd_threshold and not isinstance(fd_threshold["use_sample_mask"], bool):
                raise ValueError("'use_sample_mask' must be an boolean value.")

        self._signal_clean_info = {"masker_init": {"standardize": standardize,
                                                   "detrend": detrend,
                                                   "low_pass": low_pass,
                                                   "high_pass": high_pass,
                                                   "smoothing_fwhm": fwhm},
                                    "dummy_scans": dummy_scans,
                                    "n_acompcor_separate": n_acompcor_separate,
                                    "fd_threshold": None,
                                    "use_confounds": use_confounds,
                                    "dtype": dtype}
        # Check parcel_approach
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_approach)

        if self._signal_clean_info["use_confounds"]:
            self._signal_clean_info["confound_names"] = _check_confound_names(high_pass, confound_names,
                                                                              n_acompcor_separate)

            self._signal_clean_info["fd_threshold"] = fd_threshold

    def get_bold(self,
                 bids_dir: os.PathLike,
                 task: str,
                 session: Optional[Union[int,str]]=None,
                 runs: Optional[Union[int, str, list[int], list[str]]]=None,
                 condition: Optional[str]=None,
                 tr: Optional[Union[int, float]]=None,
                 run_subjects: Optional[list[str]]=None,
                 exclude_subjects: Optional[list[str]]= None,
                 exclude_niftis: Optional[list[str]]=None,
                 pipeline_name: Optional[str]=None,
                 n_cores: Optional[int]=None,
                 parallel_log_config: Optional[dict[str, Union[Callable, int]]]=None,
                 verbose: bool=True,
                 flush: bool=False) -> None:
        """
        Retrieve Preprocessed BOLD Data from BIDS Datasets.

        This function uses pybids for querying and requires the BOLD data directory (specified in ``bids_dir``) to be
        BIDS-compliant, including a "dataset_description.json" file. It assumes the dataset contains a derivatives
        folder with BOLD data preprocessed using a standard pipeline, specifically fMRIPrep. The pipeline directory
        must also include a "dataset_description.json" file for proper querying.

        The timeseries data of all subjects are appended to a single dictionary ``self.subject_timeseries``. Additional
        information regarding the structure of this dictionary can be found in the "Note" section.

        Basic BIDS directory:

        ::

            bids_root/
            ├── dataset_description.json
            ├── sub-<subject_label>/
            │   └── func/
            │       └── *task-*_events.tsv
            ├── derivatives/
            │   └── fmriprep-<version_label>/
            │       ├── dataset_description.json
            │       └── sub-<subject_label>/
            │           └── func/
            │               ├── *confounds_timeseries.tsv
            │               ├── *brain_mask.nii.gz
            │               └── *preproc_bold.nii.gz

        BIDS directory with session-level organization:

        ::

            bids_root/
            ├── dataset_description.json
            ├── sub-<subject_label>/
            │   └── ses-<session_label>/
            │       └── func/
            │           └── *task-*_events.tsv
            ├── derivatives/
            │   └── fmriprep-<version_label>/
            │       ├── dataset_description.json
            │       └── sub-<subject_label>/
            │           └── ses-<session_label>/
            │               └── func/
            │                   ├── *confounds_timeseries.tsv
            │                   ├── *brain_mask.nii.gz
            │                   └── *preproc_bold.nii.gz

        *Note: Only the preprocessed BOLD file is required. Additional files such as the confounds tsv (needed for
        denoising), mask, and task timing tsv file (needed for filtering a specific task condition) depend on the
        specific analyses. As mentioned previously, the "dataset_description.json" is required in both the bids root
        and pipeline directories for querying with pybids.*

        **This pipeline is most optimized for BOLD data preprocessed by fMRIPrep.**

        Parameters
        ----------
        bids_dir: :obj:`os.PathLike`
            Path to a BIDS compliant directory. A "dataset_description.json" file must be located in this directory
            or an error will be raised.

        task: :obj:`str`
            Name of task to extract timeseries data from (i.e "rest", "n-back", etc).

        session: :obj:`int`, :obj:`str`, or :obj:`None`, default=None
            Session ID to extract timeseries data from. Only a single session can be extracted at a time. While files
            having session IDs are not mandatory, this parameter must be specified if the dataset has multiple sessions
            . If ``session`` is None and multiple sessions are detected when the preprocessed NifTI files are queried,
            an error will be raised. The value can be an integer (e.g. ``session=2``) or a string (e.g.
            ``session="001"``).

        runs: :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            List of run numbers to extract timeseries data from. Extracts all runs if unspecified. For instance,
            extract only "run-0" and "run-1", use ``runs=[0, 1]``. For non-integer run IDs, use strings:
            ``runs=["000", "001"]``.

        condition: :obj:`str` or :obj:`None`, default=None
            Isolates the timeseries data corresponding to a specific condition, only after the timeseries has been
            extracted and subjected to nuisance regression. Only a single condition can be extracted at a time.

        tr: :obj:`int`, :obj:`float`, or :obj:`None`, default=None
            Repetition time (TR) for the specified task. If not provided, the TR will be automatically extracted from
            the first BOLD metadata file found for the task, searching first in the pipeline directory, then in
            the ``bids_dir`` if not found.

        run_subjects: :obj:`list[str]` or :obj:`None`, default=None
            List of subject IDs to process (e.g. ``run_subjects=["01", "02"]``). Processes all subjects if None.

        exclude_subjects: :obj:`list[str]` or :obj:`None`, default=None
            List of subject IDs to exclude (e.g. ``exclude_subjects=["01", "02"]``).

        exclude_niftis: :obj:`list[str]` or :obj:`None`, default=None
            List of the specific preprocessed NIfTI files to exclude, preventing their timeseries data from being
            extracted. Used if there are specific runs across different participants that need to be excluded.

            .. versionchanged:: 0.18.0 moved from being the second to last parameter, to being underneath
               ``exclude_subjects``

        pipeline_name: :obj:`str` or :obj:`None`, default=None
            The name of the pipeline folder in the derivatives folder containing the preprocessed data. If None,
            ``BIDSLayout`` will default to using the ``bids_dir`` with ``derivatives=True``. This parameter should be
            used if multiple pipelines exist or when the pipeline folder containing the "dataset_description.json" file
            is nested within another folder. The specified folder must contain the "dataset_description.json" file
            in its root level. For instance, if the json file is in "path/to/bids/derivatives/fmriprep/fmriprep-20.0.0",
            then ``pipeline_name = "fmriprep/fmriprep-20.0.0"``.

        n_cores: :obj:`int` or :obj:`None`, default=None
            The number of cores to use for multiprocessing with joblib. The default backend for joblib is used.

        parallel_log_config: :obj:`dict[str, Union[multiprocessing.Manager.Queue, int]]`
            Passes a user-defined managed queue and logging level to the internal timeseries extraction function
            when parallel processing (``n_cores``) is used. Note, if parallel processing is used, global logging
            configurations won't be passed to the child processes. Thus, to prevent the child processes from using the
            default logging behavior, this parameter must be used. Additionally, this parameter must be a dictionary
            and the available keys are:

            - "queue": The instance of ``multiprocessing.Manager.Queue`` to pass to ``QueueHandler``. If not specified,
              all logs will output to ``sys.stdout``.
            - "level": The logging level (e.g. ``logging.INFO``, ``logging.WARNING``). If not specified, the default
              level is ``logging.INFO``.

            ::

                import logging
                from logging.handlers import QueueListener
                from multiprocessing import Manager

                # Configure root with FileHandler
                root_logger = logging.getLogger()
                root_logger.setLevel(logging.INFO)
                file_handler = logging.FileHandler('neurocaps.log')
                file_handler.setFormatter(logging.Formatter('%(asctime)s %(name)s [%(levelname)s] %(message)s'))
                root_logger.addHandler(file_handler)

                if __name__ == "__main__":
                    # Import the TimeseriesExtractor
                    from neurocaps.extraction import TimeseriesExtractor

                    # Setup managed queue
                    manager = Manager()
                    queue = manager.Queue()

                    # Set up the queue listener
                    listener = QueueListener(queue, *root_logger.handlers)

                    # Start listener
                    listener.start()

                    extractor = TimeseriesExtractor()

                    # Use the `parallel_log_config` parameter to pass queue and the logging level
                    extractor.get_bold(
                        bids_dir="path/to/bids/dir",
                        task="rest",
                        tr=2,
                        n_cores=5,
                        parallel_log_config = {"queue": queue, "level": logging.WARNING}
                    )

                    # Stop listener
                    listener.stop()

            .. versionchanged:: 0.18.0 moved from being the last parameter, to being underneath ``n_cores``

        verbose: :obj:`bool`, default=True
            If True, logs detailed subject-specific information including: subjects skipped due to missing required
            files, current subject being processed for timeseries extraction, confounds identified for nuisance
            regression in addition to requested confounds that are missing for a subject, and additional warnings
            encountered during the timeseries extraction process.

        flush: :obj:`bool`, default=False
            If True, flushes the logged subject-specific information produced during the timeseries extraction process.

        Returns
        -------
        self


            .. versionadded:: 0.19.3

        Note
        ----
        **Subject Timeseries Dictionary**: This method stores the extracted timeseries of all subjects
        in ``self.subject_timeseries``. The structure is a dictionary mapping subject IDs to their run IDs and
        their associated timeseries (TRs x ROIs) as a numpy array:

        ::

            subject_timeseries = {
                    "101": {
                        "run-0": np.array([timeseries]), # Shape: TRs x ROIs
                        "run-1": np.array([timeseries]), # Shape: TRs x ROIs
                        "run-2": np.array([timeseries]), # Shape: TRs x ROIs
                    },
                    "102": {
                        "run-0": np.array([timeseries]), # Shape: TRs x ROIs
                        "run-1": np.array([timeseries]), # Shape: TRs x ROIs
                    }
                }

        By default, "run-0", will be used if run IDs are not specified in the NifTI file.

        **Parcellation & Nuisance Regression**: For timeseries extraction, nuisance regression, and spatial
        dimensionality reduction using a parcellation,  nilearn's ``NiftiLabelsMasker`` function is used. If requested,
        dummy scans are removed from the NIfTI images and confound dataset prior to timeseries extraction. For volumes
        exceeding a specified framewise displacement (FD) threshold, if the "use_sample_mask" key in the
        ``fd_threshold`` dictionary is set to True, then a boolean sample mask is generated (where False indicates the
        high motion volumes) and passed to the ``sample_mask`` parameter in nilearn's ``NiftiLabelsMasker``. If,
        "use_sample_mask" key is False or not specified in the ``fd_threshold`` dictionary, then censoring is done
        after nuisance regression, which is the default behavior.

        **Extraction of Task Conditions**: when extracting specific conditions, ``int`` to round down for the
        beginning scan index ``start_scan = int(onset/tr)`` and ``math.ceil`` is used to round up for the ending scan
        index ``end_scan = math.ceil((onset + duration)/tr)``. Filtering a specific condition from the
        timeseries is done after nuisance regression. Additionally,  if the "use_sample_mask" key in the
        ``fd_threshold`` dictionary is set to True, then the truncated 2D timeseries is temporarily padded to
        ensure the correct rows corresponding to the condition are obtained.
        """
        if runs and not isinstance(runs, list): runs = [runs]

        # Update attributes
        self._task_info = {"task": task, "condition": condition, "session": session, "runs": runs, "tr": tr}

        # Initialize new attributes
        self._subject_ids = []
        self._subject_timeseries = {}
        self._subject_info = {}
        self._n_cores = n_cores

        layout = self._call_layout(bids_dir, pipeline_name)
        subj_ids = sorted(layout.get(return_type="id", target="subject", task=task, space=self._space, suffix="bold"))

        if not subj_ids:
            msg = ("No subject IDs found - potential reasons: "
                   "1. Incorrect template space (default: 'MNI152NLin2009cAsym'). "
                   "Fix: Set correct template space using `self.space = 'TEMPLATE_SPACE'` (e.g. 'MNI152NLin6Asym') "
                   "2. Incorrect task name specified in `task` parameter.")
            raise BIDSQueryError(msg)

        if exclude_subjects:
            subj_ids = sorted([subj_id for subj_id in subj_ids if subj_id not in map(str, exclude_subjects)])

        if run_subjects:
            subj_ids = sorted([subj_id for subj_id in subj_ids if subj_id in map(str, run_subjects)])

        # Setup extraction
        self._setup_extraction(layout, subj_ids, exclude_niftis, verbose)

        if self._n_cores:
            # Generate list of tuples for each subject
            args_list = [(subj_id,
                          self._subject_info[subj_id]["prepped_files"],
                          self._subject_info[subj_id]["run_list"],
                          self._parcel_approach,
                          self._signal_clean_info,
                          self._task_info,
                          self._subject_info[subj_id]["tr"],
                          verbose,
                          flush,
                          parallel_log_config) for subj_id in self._subject_ids]

            parallel = Parallel(return_as="generator", n_jobs=self._n_cores)
            outputs = parallel(delayed(_extract_timeseries)(*args) for args in args_list)

            for output in outputs:
                if isinstance(output, dict): self._subject_timeseries.update(output)
        else:
            if parallel_log_config:
                LG.warning("`parallel_log_config` is only used for parallel processing. The default logger can be "
                           "modified by configuring either the root logger or a logger for a specific module prior to "
                           "package import.")

            for subj_id in self._subject_ids:
                subject_timeseries=_extract_timeseries(subj_id=subj_id,
                                                       **self._subject_info[subj_id],
                                                       parcel_approach=self._parcel_approach,
                                                       signal_clean_info=self._signal_clean_info,
                                                       task_info=self._task_info,
                                                       verbose=verbose,
                                                       flush=flush)

                # Aggregate new timeseries
                if isinstance(subject_timeseries, dict): self._subject_timeseries.update(subject_timeseries)

        return self

    @staticmethod
    @lru_cache(maxsize=4)
    def _call_layout(bids_dir, pipeline_name):
        try:
            import bids
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "This function relies on the pybids package to query subject-specific files. "
                "If on Windows, pybids does not install by default to avoid long path error issues "
                "during installation. Try using `pip install pybids` or `pip install neurocaps[windows]`.")

        bids_dir = os.path.normpath(bids_dir).rstrip(os.path.sep)

        if bids_dir.endswith("derivatives"): bids_dir = os.path.dirname(bids_dir)

        if pipeline_name:
            pipeline_name = os.path.normpath(pipeline_name).lstrip(os.path.sep).rstrip(os.path.sep)

            if pipeline_name.startswith("derivatives"):
                pipeline_name = pipeline_name[len("derivatives"):].lstrip(os.path.sep)

            layout = bids.BIDSLayout(bids_dir, derivatives=os.path.join(bids_dir, "derivatives", pipeline_name))
        else:
            layout = bids.BIDSLayout(bids_dir, derivatives=True)

        LG.info(f"{layout}")

        return layout

    # Get valid subjects to iterate through
    def _setup_extraction(self, layout, subj_ids, exclude_niftis, verbose):
        base_dict = {"layout": layout, "subj_id": None}

        for subj_id in subj_ids:
            base_dict["subj_id"] = subj_id
            files = self._build_dict(base_dict)

            # Remove excluded file from the niftis list, which will prevent it from being processed
            if exclude_niftis and files["niftis"]: files["niftis"] = self._exclude(files["niftis"], exclude_niftis)

            # Get subject header
            subject_header = self._header(subj_id)

            # Check files
            skip, msg = self._check_files(files)

            if msg and verbose: LG.warning(subject_header + msg)
            if skip: continue

            # Ensure only a single session is present if session is None
            if not self._task_info["session"]: self._check_sess(files["niftis"], subject_header)

            # Generate a list of runs to iterate through based on runs in niftis
            check_runs = self._gen_runs(files["niftis"])

            if check_runs:
                run_list = self._intersect_runs(check_runs, files)

                # Skip subject if no run has all needed files present
                if not run_list:
                    if verbose:
                        LG.warning(f"{subject_header}"
                                   "Timeseries Extraction Skipped: None of the necessary files (i.e NifTIs, masks, "
                                   "confound tsv files, confound json files, event files) are from the same run.")
                    continue

                if len(run_list) != len(check_runs):
                    if verbose:
                        LG.warning(f"{subject_header}"
                                   "Only the following runs available contain all required files: "
                                   f"{', '.join(run_list)}.")
            else:
                # Allows for nifti files that do not have the run- description
                run_list = [None]

            # Add subject list to subject attribute. These are subjects that will be ran
            self._subject_ids.append(subj_id)

            # Get repetition time for the subject
            tr = self._get_tr(files["bold_meta"], subject_header, verbose)

            # Store subject specific information
            self._subject_info[subj_id] = {"prepped_files": files,
                                           "tr": tr,
                                           "run_list": run_list}

    def _get_files(self, layout, extension, subj_id, scope="derivatives", suffix=None, desc=None,
                   event=False, space="attr"):
        query_dict = {"scope": scope, "return_type": "file", "task": self._task_info["task"], "extension": extension,
                      "subject": subj_id}

        if desc: query_dict.update({"desc": desc})
        if suffix: query_dict.update({"suffix": suffix})
        if self._task_info["session"]: query_dict.update({"session": self._task_info["session"]})
        if not event and not desc: query_dict.update({"space": self._space if space == "attr" else space})

        return sorted(layout.get(**query_dict))

    def _build_dict(self, base):
        files = {}
        files["niftis"] = self._get_files(**base, suffix="bold", extension="nii.gz")
        files["masks"] = self._get_files(**base, suffix="mask", extension="nii.gz")
        files["bold_meta"] = self._get_files(**base, suffix="bold", extension="json")

        if not files["bold_meta"]:
            files["bold_meta"] = self._get_files(**base, scope="raw", suffix="bold", extension="json", space=None)
        if self._task_info["condition"]:
            files["events"] = self._get_files(**base, scope="raw", suffix="events", extension="tsv", event=True)
        else:
            files["events"] = []

        files["confounds"] = self._get_files(**base, desc="confounds", extension="tsv")
        files["confounds_metas"] = self._get_files(**base, extension="json", desc="confounds")

        return files

    @staticmethod
    def _exclude(niftis, exclude_niftis):
        return [nifti for nifti in niftis if os.path.basename(nifti) not in exclude_niftis]

    def _header(self, subj_id):
        # Base subject message
        sub_message = f'[SUBJECT: {subj_id} | SESSION: {self._task_info["session"]} | TASK: {self._task_info["task"]}]'
        subject_header = f"{sub_message} "
        return subject_header

    def _check_files(self, files):
        skip, msg = None, None

        if not files["niftis"]:
            skip, msg = True, "Timeseries Extraction Skipped: No NifTI files were found or all NifTI files were excluded."

        if not files["masks"]:
            skip, msg = False, "Missing mask file but timeseries extraction will continue."

        if self._signal_clean_info["use_confounds"]:
            if not files["confounds"]:
                skip, msg = True, "Timeseries Extraction Skipped: `use_confounds` is requested but no confound files found."

            if not files["confounds_metas"] and self._signal_clean_info["n_acompcor_separate"]:
                skip = True
                msg = ("Timeseries Extraction Skipped: No confound metadata file found, which is needed to locate the "
                       "first n components of the white-matter and cerebrospinal fluid masks separately.")

            if self._task_info["condition"] and not files["events"]:
                skip, msg = True, "Timeseries Extraction Skipped: `condition` is specified but no event files found."

        return skip, msg

    def _check_sess(self, niftis, subject_header):
        ses_list = []

        for nifti in niftis:
            if "ses-" in os.path.basename(nifti):
                ses_list.append(re.search(r"ses-(\S+?)[-_]", os.path.basename(nifti))[0][:-1])

        ses_list = set(ses_list)

        if len(ses_list) > 1:
            raise ValueError(f"{subject_header}"
                            "`session` not specified but subject has more than one session: "
                            f"{', '.join(ses_list)}\n" + "In order to continue timeseries extraction, the "
                            "specific session to extract must be specified using `session`.")

    def _gen_runs(self, niftis):
        check_runs = []

        for nifti in niftis:
            if self._task_info["runs"]:
                check_runs.extend([f"run-{run}" for run in self._task_info["runs"]])
            else:
                if "run-" in os.path.basename(niftis[0]):
                    check_runs.append(re.search(r"run-(\S+?)[-_]", os.path.basename(nifti))[0][:-1])

        return check_runs

    def _intersect_runs(self, check_runs, files):
        run_list = []

        # Check if at least one run has all files present
        for run in check_runs:
            curr_list = []

            # Assess is any of these returns True
            curr_list.append(any([run in file for file in files["niftis"]]))
            if self._task_info["condition"]: curr_list.append(any([run in file for file in files["events"]]))
            if self._signal_clean_info["use_confounds"]:
                curr_list.append(any([run in file for file in files["confounds"]]))
                if self._signal_clean_info["n_acompcor_separate"]:
                    curr_list.append(any([run in file for file in files["confounds_metas"]]))

            if files["masks"]: curr_list.append(any([run in file for file in files["masks"]]))
            # Append runs that contain all needed files
            if all(curr_list): run_list.append(run)

        return run_list

    def _get_tr(self, bold_meta, subject_header, verbose):
        try:
            if self._task_info["tr"]:
                tr = self._task_info["tr"]
            else:
                with open(bold_meta[0], "r") as json_file:
                    tr = json.load(json_file)["RepetitionTime"]
        except (IndexError, KeyError, json.JSONDecodeError) as e:
            base_msg = "`tr` not specified and could not be extracted since using the first BOLD metadata file"
            base_msg += " due to " + (f"there being no BOLD metadata files for [TASK: {self._task_info['task']}]"
                                      if str(type(e).__name__) == "IndexError" else f"{type(e).__name__} - {str(e)}")

            if self._task_info["condition"]:
                raise ValueError(f"{subject_header}"
                                 f"{base_msg}" + " The `tr` must be given when `condition` is specified.")
            elif any([self._signal_clean_info["masker_init"]["high_pass"], self._signal_clean_info["masker_init"]["low_pass"]]):
                raise ValueError(f"{subject_header}"
                                 f"{base_msg}" + " The `tr` must be given when `high_pass` or `low_pass` is specified.")
            else:
                tr = None

                if verbose:
                    LG.warning(f"{subject_header}"
                               f"{base_msg}" + " `tr` has been set to None but extraction will continue.")

        return tr

    @staticmethod
    def _raise_error(msg):
        raise AttributeError(f"{msg} since `self.subject_timeseries` is None, either run "
                             "`self.get_bold()` or assign a valid timeseries dictionary to `self.subject_timeseries`.")

    def timeseries_to_pickle(self, output_dir: Union[str, os.PathLike], filename: Optional[str]=None) -> None:
        """
        Save the Extracted Subject Timeseries.

        Saves the extracted timeseries stored in the ``self.subject_timeseries`` dictionary (obtained from running
        ``self.get_bold``) as a pickle file. This allows for data persistence and easy conversion back into
        dictionary form for later use.

        Parameters
        ----------
        output_dir: :obj:`os.PathLike`
            Directory to save ``self.subject_timeseries`` dictionary as a pickle file. The directory will be created if
            it does not exist.

        filename: :obj:`str` or :obj:`None`, default=None
            Name of the file with or without the "pkl" extension.

            .. versionchanged:: 0.19.0  ``file_name`` to ``filename``

        Returns
        -------
        self


            .. versionadded:: 0.19.3
        """
        if not self.subject_timeseries: self._raise_error("Cannot save pickle file")

        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

        if filename is None: save_filename = "subject_timeseries.pkl"
        else: save_filename = f"{os.path.splitext(filename.rstrip())[0].rstrip()}.pkl"

        with open(os.path.join(output_dir, save_filename), "wb") as f:
            dump(self._subject_timeseries, f)

        return self

    def visualize_bold(self,
                       subj_id: Union[int, str],
                       run: Union[int, str],
                       roi_indx: Optional[Union[Union[int, str], Union[list[str], list[int]]]]=None,
                       region: Optional[str]=None,
                       show_figs: bool=True,
                       output_dir: Optional[Union[str, os.PathLike]]=None,
                       filename: Optional[str]=None,
                       **kwargs) -> None:
        """
        Plot the Extracted Subject Timeseries.

        Uses the ``self.subject_timeseries`` to visualize the extracted BOLD timeseries data of  data Regions of
        Interest (ROIs) or regions for a specific subject and run.

        Parameters
        ----------
        subj_id: :obj:`str` or :obj:`int`
            The ID of the subject.

        run: :obj:`int` or :obj:`str`
            The run ID of the subject to plot.

        roi_indx: :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[int]` or :obj:`None`, default=None
            The indices of the parcellation nodes to plot. See "nodes" in ``self.parcel_approach`` for valid
            nodes.

        region: :obj:`str` or :obj:`None`, default=None
            The region of the parcellation to plot. If not None, all nodes in the specified region will be averaged
            then plotted. See "regions" in ``self.parcel_approach`` for valid region.

        show_figs: :obj:`bool`, default=True
            Display figures.

        output_dir: :obj:`os.PathLike` or :obj:`None`, default=None
            Directory to save plot as png image. The directory will be created if it does not exist. If None, plot will
            not be saved.

        filename: :obj:`str` or :obj:`None`, default=None
            Name of the file without the extension.

            .. versionchanged:: 0.19.0  ``file_name`` to ``filename``

        kwargs: :obj:`dict`
            Keyword arguments used when saving figures. Valid keywords include:

            - dpi: :obj:`int`, default=300
                Dots per inch for the figure. Default is 300 if ``output_dir`` is provided and ``dpi`` is not
                specified.
            - figsize: :obj:`tuple`, default=(11, 5)
                Size of the figure in inches. Default is (11, 5) if ``figsize`` is not specified.
            - bbox_inches: :obj:`str` or :obj:`None`, default="tight"
                Alters size of the whitespace in the saved image.

        Returns
        -------
        self


            .. versionadded:: 0.19.3

        Note
        ----
        **Parcellation Approach**: the "nodes" and "regions" sub-keys are required in ``parcel_approach``.
        """

        if not self.subject_timeseries: self._raise_error("Cannot plot bold data")

        if roi_indx is None and region is None:
            raise ValueError("either `roi_indx` or `region` must be specified.")

        if roi_indx is not None and region is not None:
            raise ValueError("`roi_indx` and `region` can not be used simultaneously.")

        if filename is not None and output_dir is None:
            LG.warning("`filename` supplied but no `output_dir` specified. Files will not be saved.")

        parcellation_name = list(self._parcel_approach)[0]

        # Defaults
        defaults = {"dpi": 300, "figsize": (11, 5), "bbox_inches": "tight"}

        plot_dict = _check_kwargs(defaults, **kwargs)

        # Obtain the column indices associated with the rois
        if roi_indx or roi_indx == 0: plot_indxs = self._get_roi_indices(roi_indx, parcellation_name)
        else: plot_indxs = self._get_region_indices(region, parcellation_name)

        plt.figure(figsize=plot_dict["figsize"])

        timeseries = self._subject_timeseries[str(subj_id)][f"run-{run}"]

        if roi_indx or roi_indx == 0:
            plt.plot(range(1, timeseries.shape[0] + 1), timeseries[:, plot_indxs])

            if isinstance(roi_indx, (int, str)) or (isinstance(roi_indx, list) and len(roi_indx) == 1):
                if isinstance(roi_indx, int): roi_title = self._parcel_approach[parcellation_name]["nodes"][roi_indx]
                elif isinstance(roi_indx, str): roi_title = roi_indx
                else: roi_title = roi_indx[0]
                plt.title(roi_title)
        else:
            plt.plot(range(1, timeseries.shape[0] + 1), np.mean(timeseries[:, plot_indxs], axis=1))
            plt.title(region)

        plt.xlabel("TR")

        if output_dir:
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            if filename: save_filename = f"{os.path.splitext(filename.rstrip())[0].rstrip()}.png"
            else: save_filename = f'subject-{subj_id}_run-{run}_timeseries.png'

            plt.savefig(os.path.join(output_dir, save_filename), dpi=plot_dict["dpi"],
                        bbox_inches=plot_dict["bbox_inches"])

        plt.show() if show_figs else plt.close()

        return self

    def _get_roi_indices(self, roi_indx, parcellation_name):
        if isinstance(roi_indx, int):
            plot_indxs = roi_indx
        elif isinstance(roi_indx, str):
            # Check if parcellation_approach is custom
            if "Custom" in self._parcel_approach and "nodes" not in self._parcel_approach["Custom"]:
                _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")

            plot_indxs = self._parcel_approach[parcellation_name]["nodes"].index(roi_indx)
        else:
            if all([isinstance(indx, int) for indx in roi_indx]):
                plot_indxs = np.array(roi_indx)
            elif all([isinstance(indx, str) for indx in roi_indx]):
                # Check if parcellation_approach is custom
                if "Custom" in self._parcel_approach and "nodes" not in self._parcel_approach["Custom"]:
                    _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")

                plot_indxs = np.array(
                    [self._parcel_approach[parcellation_name]["nodes"].index(index) for index in roi_indx]
                    )
            else:
                raise ValueError("All elements in `roi_indx` need to be all strings or all integers.")

        return plot_indxs

    def _get_region_indices(self, region, parcellation_name):
        if "Custom" in self._parcel_approach:
            if "regions" not in self._parcel_approach["Custom"]:
                _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")
            else:
                plot_indxs = np.array(self._parcel_approach["Custom"]["regions"][region]["lh"] +
                                      self._parcel_approach["Custom"]["regions"][region]["rh"])
        else:
            plot_indxs = np.array(
                [index for index, label in enumerate(self._parcel_approach[parcellation_name]["nodes"])
                 if region in label]
                )

        return plot_indxs
