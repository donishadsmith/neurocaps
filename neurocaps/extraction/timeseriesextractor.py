import json, os, re, sys
from functools import lru_cache
from typing import Callable, Literal, Optional, Union

# Conditional import based on major and minor version of Python
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import matplotlib.pyplot as plt, numpy as np
from joblib import Parallel, delayed, dump
from tqdm.auto import tqdm

from ..exceptions import BIDSQueryError
from ..typing import ParcelConfig, ParcelApproach
from .._utils import (
    _TimeseriesExtractorGetter,
    _PlotDefaults,
    _check_kwargs,
    _check_confound_names,
    _check_parcel_approach,
    _extract_timeseries,
    _logger,
)

LG = _logger(__name__)


class TimeseriesExtractor(_TimeseriesExtractorGetter):
    """
    Timeseries Extractor Class.

    Performs timeseries denoising, extraction, serialization (pickling), and BOLD visualization.

    Parameters
    ----------
    space: :obj:`str`, default="MNI152NLin2009cAsym"
        The standard template space that the preprocessed bold data is registered to. Used for querying with PyBIDS
        to locate preprocessed BOLD-related files.

    parcel_approach: :obj:`ParcelConfig`, :obj:`ParcelApproach`, or :obj:`str`,\
                     default={"Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}}
        The approach used to parcellate NifTI images into distinct regions-of-interests (ROIs).

        To initialize a ``parcel_approach``, the configuration requires a nested dictionary with:

          - First Level Key: The parcellation name ("Schaefer", "AAL", or "Custom").
          - Second Level Keys: Parameters specific to each parcellation method.

        Supported parcellation approaches and their parameters, includes:

        - "Schaefer":

            - "n_rois": Number of ROIs (Default=400). Options are 100, 200, 300, 400, 500, 600, 700, 800, 900, or 1000.
            - "yeo_networks": Number of Yeo networks (Default=7). Options are 7 or 17.
            - "resolution_mm": Spatial resolution in millimeters (Default=1). Options are 1 or 2.

        - "AAL":

            - "version": AAL parcellation version to use (Default="SPM12" if ``{"AAL": {}}`` is given). Options are\
            "SPM5", "SPM8", "SPM12", or "3v2".

        - "Custom" (user-defined):

            - "maps": Directory path to the location of the parcellation file.
            - "nodes": A list of node names in the order of the label IDs in the parcellation.
            - "regions": The regions or networks in the parcellation.

        *Notes*:

        - Input can also be a pickle file containing a processed parcel approach with required keys
          ("maps", "nodes", and "regions").
        - For detailed parameter information, see:

            - `Schaefer (Nilearn's Fetch Schaefer Documentation)\
            <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_schaefer_2018.html>`_
            - `AAL (Nilearn's Fetch AAL Documentation)\
            <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_aal.html>`_
            - Custom: See Notes section below for structure requirements.

    standardize: {"zscore_sample", "zscore", "psc", True, False}, default="zscore_sample"
        Standardizes the timeseries.

        *Note*: Refer to `nilearn.maskers.NiftiLabelsMasker
        <https://nilearn.github.io/stable/modules/generated/nilearn.maskers.NiftiLabelsMasker.html>`_ for an
        explanation of each available option.

    detrend: :obj:`bool`, default=True
        Detrends the timeseries.

    low_pass: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Filters out signals above the specified cutoff frequency.

    high_pass: :obj:`float`, :obj:`int`, or :obj:`None``, default=None
        Filters out signals below the specified cutoff frequency.

    fwhm: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Applies spatial smoothing to data (in millimeters).

    use_confounds: :obj:`bool`, default=True
        If True, performs nuisance regression during timeseries extraction using the default or user-specified
        confounds in ``confound_names``.

        *Note*: requires that confound tsv files to be in same directory as preprocessed BOLD images.

    confound_names: {"basic"}, :obj:`list[str]`, or :obj:`None`, default="basic"
        Names of confounds extracted from the confound tsv files if ``use_confounds=True``.

        If "basic", the following confounds are used by default:

        - All cosine-basis parameters.
        - Six head-motion parameters and their first-order derivatives.
        - First six combined aCompcor components.

        *Notes*:

        - Confound names follow fMRIPrep's naming scheme (versions >= 1.2.0).
        - Wildcards are supported: e.g., "cosine*" matches all confounds starting with "cosine".

        .. versionchanged:: 0.23.0 Changed default from ``None`` to ``"basic"``. The ``"basic"`` option provides the
           same functionality that ``None`` did in previous versions.

    fd_threshold: :obj:`float`, :obj:`dict[str, float | int]`, or :obj:`None`, default=None
        Threshold for volume censoring based on framewise displacement (FD).

        - *If float*, removes volumes where FD > threshold.
        - *If dict*, the following subkeys are available:

            - "threshold": A float (Default=None). Removes volumes where FD > threshold.
            - "outlier_percentage": A float in interval [0,1] (Default=None). Removes entire runs where proportion of
              censored volumes exceeds this threshold. Proportion calculated after dummy scan removal. Issues warning
              when runs are flagged. If ``condition`` specified in ``self.get_bold()``, only considers volumes
              associated with the condition.
            - "n_before": An integer indicating the number of volumes to remove before each flagged volume
              (Default=None). For instance, if volume 5 flagged and ``{"n_before": 2}``, then volumes 3,
              4, and 5 are discarded.
            - "n_after": An integer indicating the of volumes to remove after each flagged volume (Default=False). For
              instance, if volume 5 flagged and ``{"n_after": 2}``, then volumes 5, 6, and 7 are discarded.
            - "use_sample_mask": A boolean (Default=False). If True, censors before nuisance regression using Nilearn's
              ``NiftiLabelsMasker``. Also, sets ``clean__extrapolate=False`` to prevent interpolation of end volumes.
              If False, censors after nuisance regression.
            - "interpolate": A boolean (Default=None). If True, uses scipy's ``CubicSpline`` function with
              ``extrapolate=False`` to perform cubic spline interpolation on censored frames that are not at the ends
              of the timeseries. For example, given a ``censor_mask=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]`` where "0"
              indicates censored volumes, only the volumes at index 3, 5, 6, and 8 would be interpolated. When False or
              None (default behavior), no interpolation is performed and all censored frames are discarded.

              .. versionadded:: 0.22.3 "interpolate" key added.

        *Notes*:

        - A column named "framewise_displacement" must be available in the confounds file.
        - ``use_confounds`` must be set to True.
        - Do not specify "framewise_displacement" in ``confound_names``.
        - See Nilearn's documentation for details on censored volume handling:

            - `Signal Clean <https://nilearn.github.io/stable/modules/generated/nilearn.signal.clean.html>`_
            - `NiftiLabelsMasker <https://nilearn.github.io/stable/modules/generated/nilearn.maskers.NiftiLabelsMasker.html>`_

        - When ``{"use_sample_mask": False}`` and ``standardize=True``, applying an additional within-run
          standardization (using ``neurocaps.analysis.standardize``) is recommended after outlier removal.
        - If ``{"interpolation: True}``, then interpolation is only applied nuisance regression and parcellation steps
          have been completed. It is also applied prior to the condition being extracted from the timeseries.
        - See Scipy's documentation on their
          `CubicSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_
          function.

    n_acompcor_separate: :obj:`int` or :obj:`None`, default=None
        Number of aCompCor components to extract separately from the white-matter (WM) and CSF masks. Uses first "n"
        components from each mask separately. For instance, if ``n_acompcor_separate=5``, then the the first 5 WM
        components and the first 5 CSF components (totaling 10 components) are regressed out.

        *Notes*:
        - ``use_confounds`` must be set to True.
        - If specified, this parameter overrides any aCompCor components listed in ``confound_names``.

    dummy_scans: :obj:`int`, :obj:`dict[str, bool | int]`, or :obj:`None`, default=None
        Number of initial volumes to remove before timeseries extraction.

        - *If int*, removes first "n" volumes.
        - *If dict*, the following keys are supported:

            - "auto": A boolean (Default=None). If True, Automatically determines dummy scans from fMRIPrep confounds
              file by counting the number of "non_steady_state_outlier_XX" columns in confounds.tsv file. For instance,
              if two columns are found,then the first two columns are removed.
            - "min": An integer (Default=None). Minimum volumes to remove when auto is set to True. If "auto" finds 2
              outliers but ``{"min": 3}``, removes 3 volumes.
            - "max": An integer (Default=None). Maximum volumes to remove when auto=True. If "auto" finds 6 outliers but
              ``{"max": 5}``, removes 5 volumes.

        *Note*: "min" and "max" keys only apply when "auto" is True.

    dtype: :obj:`str` or "auto", default=None
        The NumPy dtype the NIfTI images are converted to when passed to Nilearn's ``load_img`` function.


    Properties
    ----------
    space: :obj:`str`
        The standard template space that the preprocessed BOLD data is registered to. The space can also be set after
        class initialization using ``self.space = "New Space"`` if the template space needs to be changed.

    parcel_approach: :obj:`ParcelApproach`
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

    signal_clean_info: :obj:`dict[str, bool | int | float | str]` or :obj:`None`
        Dictionary containing parameters for signal cleaning specified during initialization of the
        ``TimeseriesExtractor`` class. This information includes ``standardize``, ``detrend``, ``low_pass``,
        ``high_pass``, ``fwhm``, ``dummy_scans``, ``use_confounds``, ``n_compcor_separate``, and ``fd_threshold``.

    task_info: :obj:`dict[str, str | int]` or :obj:`None`
        If ``self.get_bold()`` ran, is a dictionary containing all task-related information such as ``task``,
        ``condition``, ``session``, ``runs``, and ``tr`` (if specified) else None.

    subject_ids: :obj:`list[str]` or :obj:`None`
        A list containing all subject IDs that have retrieved from PyBIDS and subjected to timeseries
        extraction.

    n_cores: :obj:`int` or :obj:`None`
        Number of cores used for multiprocessing with Joblib.

    subject_timeseries: :obj:`SubjectTimeseries` or :obj:`None`
        A dictionary mapping subject IDs to their run IDs and their associated timeseries (TRs x ROIs) as a NumPy array.
        Can also be a path to a pickle file containing this same structure. If this property needs to be deleted due
        to memory issues, ``del self.subject_timeseries`` can be used to delete this property and only have it
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

    See Also
    --------
    :class:`neurocaps.typing.ParcelConfig`
        Type definition representing the configuration options and structure for the Schaefer and AAL parcellations.
    :class:`neurocaps.typing.ParcelApproach`
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation approaches.
    :data:`neurocaps.typing.SubjectTimeseries`
        Type definition representing the structure of the subject timeseries.

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

    The recognized subkeys for the "Custom" parcellation approach includes:

    - "maps": Directory path containing the parcellation in a supported format (e.g., .nii or .nii.gz for NifTI).
    - "nodes": A list or numpy array of all node labels arranged in ascending order based on their numerical IDs from
      the parcellation. The 0th index should contain the label corresponding to the lowest, non-background numerical ID.
    - "regions": A dictionary defining major brain regions or networks, with each region containing "lh"
      (left hemisphere) and "rh" (right hemisphere) subkeys listing node indices.

    Refer to the `NeuroCAPs' Parcellation Documentation <https://neurocaps.readthedocs.io/en/stable/parcellations.html>`_
    for more detailed explanations and example structures for the "nodes" and "regions" subkeys.

    **Note**: Different subkeys are required depending on the function used. Refer to the Note section under each
    function for information regarding the subkeys required for that specific function.
    """

    def __init__(
        self,
        space: str = "MNI152NLin2009cAsym",
        parcel_approach: Union[ParcelConfig, ParcelApproach, str] = {
            "Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}
        },
        standardize: Union[bool, Literal["zscore_sample", "zscore", "psc"]] = "zscore_sample",
        detrend: bool = True,
        low_pass: Optional[Union[float, int]] = None,
        high_pass: Optional[Union[float, int]] = None,
        fwhm: Optional[Union[float, int]] = None,
        use_confounds: bool = True,
        confound_names: Optional[Union[list[str], Literal["basic"]]] = "basic",
        fd_threshold: Optional[Union[float, dict[str, Union[bool, float, int]]]] = None,
        n_acompcor_separate: Optional[int] = None,
        dummy_scans: Optional[Union[int, dict[str, Union[bool, int]]]] = None,
        dtype: Optional[Union[str, Literal["auto"]]] = None,
    ) -> None:

        self._space = space
        # Check parcel_approach
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_approach)

        if use_confounds:
            if confound_names:
                # Replace confounds if not None
                confound_names = _check_confound_names(high_pass, confound_names, n_acompcor_separate)

            if fd_threshold:
                self._validate_init_params("fd_threshold", fd_threshold)

            if dummy_scans:
                self._validate_init_params("dummy_scans", dummy_scans)
        else:
            if confound_names:
                LG.warning(
                    "`confound_names` is specified but `use_confounds` is not True so nuisance regression will not be done."
                )
                confound_names = None

            if fd_threshold:
                raise ValueError(
                    "`fd_threshold` specified but `use_confounds` is not True, so removal of volumes after "
                    "nuisance regression cannot be done since confounds tsv file generated by fMRIPrep is needed."
                )

            if isinstance(dummy_scans, dict) and dummy_scans.get("auto"):
                raise ValueError(
                    "'auto' is True in `dummy_scans` dictionary but `use_confounds` is not True, so automated dummy "
                    "scans detection cannot be done since confounds tsv file generated by fMRIPrep is needed."
                )

            if n_acompcor_separate:
                raise ValueError(
                    "`n_acompcor_separate` specified `use_confounds` is not True, so separate WM and CSF components "
                    "cannot be regressed out since confounds tsv file generated by fMRIPrep is needed."
                )

        self._signal_clean_info = {
            "masker_init": {
                "standardize": standardize,
                "detrend": detrend,
                "low_pass": low_pass,
                "high_pass": high_pass,
                "smoothing_fwhm": fwhm,
            },
            "use_confounds": use_confounds,
            "confound_names": confound_names,
            "n_acompcor_separate": n_acompcor_separate,
            "dummy_scans": dummy_scans,
            "fd_threshold": fd_threshold,
            "dtype": dtype,
        }

    @staticmethod
    def _validate_init_params(param, struct):

        mandatory_keys = {"dummy_scans": {"auto": bool}, "fd_threshold": {"threshold": (float, int)}}

        optional_keys = {
            "dummy_scans": {"min": int, "max": int},
            "fd_threshold": {
                "n_before": int,
                "n_after": int,
                "outlier_percentage": float,
                "use_sample_mask": bool,
                "interpolate": bool,
            },
        }

        valid_types = (dict, int) if param == "dummy_scans" else (dict, float, int)
        error_msg = "dictionary or integer" if param == "dummy_scans" else "dictionary, float, or integer"
        if not isinstance(struct, valid_types):
            raise TypeError(f"`{param}` must be a {error_msg}.")

        if isinstance(struct, dict):
            # Check mandatory keys
            key = list(mandatory_keys[param].keys())[0]
            if key not in struct:
                raise KeyError(f"'{key}' is a mandatory key when `{param}` is a dictionary.")

            if not isinstance(struct[key], mandatory_keys[param][key]):
                raise TypeError(f"'{key}' must be a {'boolean' if key == 'auto' else 'float or integer'}.")

            # Check optional keys
            for key in optional_keys[param]:
                if key in struct:
                    if not isinstance(struct[key], optional_keys[param][key]):
                        type_msg = (
                            "a boolean"
                            if key == "use_sample_mask"
                            else ("a float" if key == "outlier_percentage" else "an integer")
                        )
                        raise TypeError(f"'{key}' must be {type_msg}.")

                    # Additional check for "outlier_percentage"
                    if key == "outlier_percentage" and not 0 < struct[key] < 1:
                        raise ValueError("'outlier_percentage' must be float between 0 and 1.")

            # Check invalid keys
            set_diff = set(struct) - set(optional_keys[param]) - set(mandatory_keys[param])
            if set_diff:
                formatted_string = ", ".join(["'{a}'".format(a=x) for x in set_diff])
                LG.warning(
                    f"The following invalid keys have been found in `{param}` and will be ignored: {formatted_string}."
                )

    def get_bold(
        self,
        bids_dir: str,
        task: str,
        session: Optional[Union[int, str]] = None,
        runs: Optional[Union[int, str, list[int], list[str]]] = None,
        condition: Optional[str] = None,
        condition_tr_shift: int = 0,
        tr: Optional[Union[int, float]] = None,
        slice_time_ref: Union[int, float] = 0.0,
        run_subjects: Optional[list[str]] = None,
        exclude_subjects: Optional[list[str]] = None,
        exclude_niftis: Optional[list[str]] = None,
        pipeline_name: Optional[str] = None,
        n_cores: Optional[int] = None,
        parallel_log_config: Optional[dict[str, Union[Callable, int]]] = None,
        verbose: bool = True,
        flush: bool = False,
        progress_bar: bool = False,
    ) -> Self:
        """
        Retrieve Preprocessed BOLD Data from BIDS Datasets.

        This function uses PyBIDS for querying and requires the BOLD data directory (specified in ``bids_dir``) to be
        BIDS-compliant, including a "dataset_description.json" file. It assumes the dataset contains a derivatives
        folder with BOLD data preprocessed using a standard pipeline, specifically fMRIPrep. The pipeline directory
        must also include a "dataset_description.json" file for proper querying.

        The timeseries data of all subjects are appended to a single dictionary ``self.subject_timeseries``. Additional
        information regarding the structure of this dictionary can be found in the "Note" section.

        **This pipeline is most optimized for BOLD data preprocessed by fMRIPrep.** Refer to
        `NeuroCAPs' BIDS Structure and Entities Documentation <https://neurocaps.readthedocs.io/en/stable/bids.html>`_
        for additional information on the expected directory structure and file naming scheme (entities) needed for
        querying.

        Parameters
        ----------
        bids_dir: :obj:`str`
            Path to a BIDS compliant directory. A "dataset_description.json" file must be located in this directory
            or an error will be raised.

        task: :obj:`str`
            Name of task to extract timeseries data from (i.e "rest", "n-back", etc).

        session: :obj:`int`, :obj:`str`, or :obj:`None`, default=None
            The session ID to extract timeseries data from. Only a single session can be extracted at a time and an
            error will be raised if more than one session is detected during querying. The value can be an integer
            (e.g. ``session=2``) or a string (e.g. ``session="001"``).

        runs: :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            List of run numbers to extract timeseries data from. Extracts all runs if unspecified. For instance,
            to extract only "run-0" and "run-1", use ``runs=[0, 1]``. For non-integer run IDs, use strings:
            ``runs=["000", "001"]``.

        condition: :obj:`str` or :obj:`None`, default=None
            Isolates the timeseries data corresponding to a specific condition (listed in the "trial_type" column of
            the "events.tsv" file), only after the timeseries has been extracted and subjected to nuisance regression.
            Only a single condition can be extracted at a time.

        condition_tr_shift: :obj:`int`, default=0
            Number of TR units to units to offset both the start and end scan indices of a condition. This parameter
            only applies when a ``condition`` is specified. For more details about how this offset affects the
            calculation of task conditions, see the "Extraction of Task Conditions" section below.

            .. versionadded:: 0.20.0

        tr: :obj:`int`, :obj:`float` or :obj:`None`, default=None
            Repetition time (TR), in seconds, for the specified task. If not provided, the TR will be automatically
            extracted from the first BOLD metadata file found for the task, searching first in the pipeline directory,
            then in the ``bids_dir`` if not found.

        slice_time_ref: :obj:`int` or :obj:`float`, default=0.0
            The reference slice expressed as a fraction of the ``tr`` that is subtracted from the condition onset times
            to adjust for slice time correction when ``condition`` is not None (``onset - slice_time_ref * tr``). Values
            can range from 0 to 1.

            .. versionadded:: 0.21.0

        run_subjects: :obj:`list[str]` or :obj:`None`, default=None
            List of subject IDs to process (e.g. ``run_subjects=["01", "02"]``). Processes all subjects if None.

        exclude_subjects: :obj:`list[str]` or :obj:`None`, default=None
            List of subject IDs to exclude (e.g. ``exclude_subjects=["01", "02"]``).

        exclude_niftis: :obj:`list[str]` or :obj:`None`, default=None
            List of the specific preprocessed NIfTI files to exclude, preventing their timeseries data from being
            extracted. Used if there are specific runs across different participants that need to be excluded.

        pipeline_name: :obj:`str` or :obj:`None`, default=None
            The name of the pipeline folder in the derivatives folder containing the preprocessed data. Used if
            multiple pipeline folders exist in the derivatives folder or the pipeline folder is nested
            (e.g. "fmriprep/fmriprep-20.0.0").

        n_cores: :obj:`int` or :obj:`None`, default=None
            The number of cores to use for multiprocessing with Joblib. The "loky" backend is used.

        parallel_log_config: :obj:`dict[str, multiprocessing.Manager.Queue | int]`
            Passes a user-defined managed queue and logging level to the internal timeseries extraction function
            when parallel processing (``n_cores``) is used. Additionally, this parameter must be a dictionary
            and the available keys are:

            - "queue": The instance of ``multiprocessing.Manager.Queue`` to pass to ``QueueHandler``. If not specified,
              all logs will output to ``sys.stdout``.
            - "level": The logging level (e.g. ``logging.INFO``, ``logging.WARNING``). If not specified, the default
              level is ``logging.INFO``.

            Refer to the `NeuroCAPs' Logging Documentation <https://neurocaps.readthedocs.io/en/stable/logging.html>`_
            for a detailed example of setting up this parameter.

        verbose: :obj:`bool`, default=True
            If True, logs detailed subject-specific information including: subjects skipped due to missing required
            files, current subject being processed for timeseries extraction, confounds identified for nuisance
            regression in addition to requested confounds that are missing for a subject, and additional warnings
            encountered during the timeseries extraction process.

        flush: :obj:`bool`, default=False
            If True, flushes the logged subject-specific information produced during the timeseries extraction process.

        progress_bar: :obj:`bool`, default=False
            If True, displays a progress bar.

            .. versionadded:: 0.21.5

        Returns
        -------
        self

        Raises
        ------
        BIDSQueryError
            Subject IDs were not found during querying.

        Note
        ----
        **Subject Timeseries Dictionary**: This method stores the extracted timeseries of all subjects
        in ``self.subject_timeseries``. The structure is a dictionary mapping subject IDs to their run IDs and
        their associated timeseries (TRs x ROIs) as a NumPy array:

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

        **NifTI Files Without "run-" Entity**: By default, "run-0" will be used as a placeholder, if run IDs are not
        specified in the NifTI file.

        **Parcellation & Nuisance Regression**: For timeseries extraction, nuisance regression, and spatial
        dimensionality reduction using a parcellation, Nilearn's ``NiftiLabelsMasker`` function is used. If requested,
        dummy scans are removed from the NIfTI images and confound dataset prior to timeseries extraction. For volumes
        exceeding a specified framewise displacement (FD) threshold, if the "use_sample_mask" key in the
        ``fd_threshold`` dictionary is set to True, then a boolean sample mask is generated (where False indicates the
        high motion volumes) and passed to the ``sample_mask`` parameter in Nilearn's ``NiftiLabelsMasker``. If,
        "use_sample_mask" key is False or not specified in the ``fd_threshold`` dictionary, then censoring is done
        after nuisance regression, which is the default behavior.

        **Extraction of Task Conditions**: The formula used for computing the scan indices corresponding to the
        corresponding to a specific condition:

        ::

            adjusted_onset = onset - slice_time_ref * tr
            adjusted_onset = adjusted_onset if adjusted_onset >= 0 else 0
            start_scan = int(adjusted_onset / tr) + condition_tr_shift
            end_scan = math.ceil((adjusted_onset + duration) / tr) + condition_tr_shift
            scans.extend(list(range(onset_scan, end_scan)))
            scans = sorted(list(set(scans)))

        When partial scans are computed, ``int`` is used to round down for the beginning scan index and ``math.ceil``
        is used to round up for the ending scan index. Negative scan indices are set to 0 to avoid unintentional
        negative indexing. For simplicity, note that when ``slice_time_ref`` and ``condition_tr_shift`` are 0, the
        formula simplifies to:

        ::

            start_scan = int(onset / tr)
            end_scan = math.ceil((onset + duration) / tr)
            scans.extend(list(range(onset_scan, end_scan)))
            scans = sorted(list(set(scans)))

        Filtering a specific condition from the timeseries is done after nuisance regression and the indices are used
        to extract the TRs corresponding to the condition from the timeseries. Additionally, if the "use_sample_mask"
        key in the ``fd_threshold`` dictionary is set to True, then the truncated 2D timeseries is temporarily padded
        to ensure the correct rows corresponding to the condition are obtained.
        """
        if runs and not isinstance(runs, list):
            runs = [runs]

        # Update attributes
        self._task_info = {
            "task": task,
            "session": session,
            "runs": runs,
            "condition": condition,
            "condition_tr_shift": condition_tr_shift,
            "tr": tr,
            "slice_time_ref": slice_time_ref,
        }

        # Quick check condition_tr_shift and slice_time_ref to not cause issues later in the pipeline
        self._validate_get_bold_params()

        # Initialize new attributes
        self._subject_ids = []
        self._subject_timeseries = {}
        self._subject_info = {}
        self._n_cores = n_cores

        layout = self._call_layout(bids_dir, pipeline_name)
        subj_ids = sorted(layout.get(return_type="id", target="subject", task=task, space=self._space, suffix="bold"))

        if not subj_ids:
            msg = (
                "No subject IDs found - potential reasons:\n"
                "1. Incorrect template space (default: 'MNI152NLin2009cAsym'). "
                "Fix: Set correct template space using `self.space = 'TEMPLATE_SPACE'` (e.g. 'MNI152NLin6Asym')\n"
                "2. File names do not contain specific entities required for querying such as 'sub-', 'space-', "
                "'task-', or 'desc-' (e.g 'sub-01_ses-1_task-rest_space-MNI152NLin2009cAsym_desc-preproc-bold.nii.gz')\n"
                "3. Incorrect task name specified in `task` parameter."
            )
            raise BIDSQueryError(msg)

        if exclude_subjects:
            subj_ids = sorted([subj_id for subj_id in subj_ids if subj_id not in map(str, exclude_subjects)])

        if run_subjects:
            subj_ids = sorted([subj_id for subj_id in subj_ids if subj_id in map(str, run_subjects)])

        # Setup extraction
        self._setup_extraction(layout, subj_ids, exclude_niftis, verbose)

        if self._n_cores:
            # Generate list of tuples for each subject
            args_list = [
                (
                    subj_id,
                    self._subject_info[subj_id]["prepped_files"],
                    self._subject_info[subj_id]["run_list"],
                    self._parcel_approach,
                    self._signal_clean_info,
                    self._task_info,
                    self._subject_info[subj_id]["tr"],
                    verbose,
                    flush,
                    parallel_log_config,
                )
                for subj_id in self._subject_ids
            ]

            parallel = Parallel(return_as="generator", n_jobs=self._n_cores, backend="loky")
            outputs = tqdm(
                parallel(delayed(_extract_timeseries)(*args) for args in args_list),
                desc="Processing Subjects",
                total=len(args_list),
                disable=not progress_bar,
            )
            for output in outputs:
                if isinstance(output, dict):
                    self._subject_timeseries.update(output)
        else:
            if parallel_log_config:
                LG.warning(
                    "`parallel_log_config` is only used for parallel processing. The default logger can be "
                    "modified by configuring either the root logger or a logger for a specific module prior to "
                    "package import."
                )

            for subj_id in tqdm(self._subject_ids, desc="Processing Subjects", disable=not progress_bar):
                subject_timeseries = _extract_timeseries(
                    subj_id=subj_id,
                    **self._subject_info[subj_id],
                    parcel_approach=self._parcel_approach,
                    signal_clean_info=self._signal_clean_info,
                    task_info=self._task_info,
                    verbose=verbose,
                    flush=flush,
                )

                # Aggregate new timeseries
                if isinstance(subject_timeseries, dict):
                    self._subject_timeseries.update(subject_timeseries)

        return self

    def _validate_get_bold_params(self):
        if self._task_info["condition_tr_shift"] != 0:
            if not isinstance(self._task_info["condition_tr_shift"], int) or self._task_info["condition_tr_shift"] < 0:
                raise ValueError("`condition_tr_shift` must be a integer value equal to or greater than 0.")

            if self._task_info["condition"] is None:
                LG.warning("`condition_tr_shift` specified but `condition` is None.")

        if self._task_info["slice_time_ref"] != 0:
            if not isinstance(self._task_info["slice_time_ref"], (float, int)) or (
                self._task_info["slice_time_ref"] < 0 or (self._task_info["slice_time_ref"] > 1)
            ):
                raise ValueError("`slice_time_ref` must be a numerical value from 0 to 1.")

            if self._task_info["condition"] is None:
                LG.warning("`slice_time_ref` specified but `condition` is None.")

    @staticmethod
    @lru_cache(maxsize=4)
    def _call_layout(bids_dir, pipeline_name):
        try:
            import bids
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "This function relies on the pybids package to query subject-specific files. "
                "If on Windows, pybids does not install by default to avoid long path error issues "
                "during installation. Try using `pip install pybids` or `pip install neurocaps[windows]`."
            )

        bids_dir = os.path.normpath(bids_dir).rstrip(os.path.sep)
        bids_dir = bids_dir.removesuffix("derivatives").rstrip(os.sep)

        if pipeline_name:
            pipeline_name = os.path.normpath(pipeline_name).lstrip(os.path.sep).rstrip(os.path.sep)
            pipeline_name = pipeline_name.removeprefix("derivatives").lstrip(os.path.sep)

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
            if exclude_niftis and files["niftis"]:
                files["niftis"] = self._exclude(files["niftis"], exclude_niftis)

            # Get subject header
            subject_header = self._header(subj_id)

            # Check files
            skip, msg = self._check_files(files)
            if msg and verbose:
                LG.warning(subject_header + msg)
            if skip:
                continue

            # Ensure only a single session is present if session is None
            if not self._task_info["session"]:
                self._check_sess(files["niftis"], subject_header)

            # Generate a list of runs to iterate through based on runs in niftis
            check_runs = self._gen_runs(files["niftis"])
            if check_runs:
                run_list = self._intersect_runs(check_runs, files)

                # Skip subject if no run has all needed files present
                if not run_list:
                    if verbose:
                        LG.warning(
                            f"{subject_header}"
                            "Timeseries Extraction Skipped: None of the necessary files (i.e NifTIs, masks, "
                            "confound tsv files, confound json files, event files) are from the same run."
                        )
                    continue

                if len(run_list) != len(check_runs):
                    if verbose:
                        LG.warning(
                            f"{subject_header}"
                            "Only the following runs available contain all required files: "
                            f"{', '.join(run_list)}."
                        )
            elif not check_runs and self._task_info["runs"]:
                if verbose:
                    requested_runs = [
                        f"run-{str(run)}" if "run-" not in str(run) else run for run in self._task_info["runs"]
                    ]
                    LG.warning(
                        f"{subject_header}"
                        "Timeseries Extraction Skipped: Subject does not have any of the requested run IDs: "
                        f"{', '.join(requested_runs)}"
                    )
                    continue

            else:
                # Allows for nifti files that do not have the run- description
                run_list = [None]

            # Add subject list to subject attribute. These are subjects that will be ran
            self._subject_ids.append(subj_id)

            # Get repetition time for the subject
            tr = self._get_tr(files["bold_meta"], subject_header, verbose)

            # Store subject specific information
            self._subject_info[subj_id] = {"prepped_files": files, "tr": tr, "run_list": run_list}

    def _get_files(
        self, layout, extension, subj_id, scope="derivatives", suffix=None, desc=None, event=False, space="attr"
    ):
        query_dict = {
            "scope": scope,
            "return_type": "file",
            "task": self._task_info["task"],
            "extension": extension,
            "subject": subj_id,
        }

        if desc:
            query_dict.update({"desc": desc})

        if suffix:
            query_dict.update({"suffix": suffix})

        if self._task_info["session"]:
            query_dict.update({"session": self._task_info["session"]})

        if not event and not desc:
            query_dict.update({"space": self._space if space == "attr" else space})

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

        if self._signal_clean_info["n_acompcor_separate"]:
            files["confound_metas"] = self._get_files(**base, extension="json", desc="confounds")

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
            skip, msg = (
                True,
                "Timeseries Extraction Skipped: No NifTI files were found or all NifTI files were excluded.",
            )

        if not files["masks"]:
            skip, msg = False, "Missing mask file but timeseries extraction will continue."

        if self._signal_clean_info["use_confounds"]:
            if not files["confounds"]:
                skip, msg = (
                    True,
                    "Timeseries Extraction Skipped: `use_confounds` is requested but no confound files found.",
                )

            if self._signal_clean_info["n_acompcor_separate"] and not files.get("confound_metas"):
                skip = True
                msg = (
                    "Timeseries Extraction Skipped: No confound metadata file found, which is needed to locate the "
                    "first n components of the white-matter and cerebrospinal fluid masks separately."
                )

            if self._task_info["condition"] and not files["events"]:
                skip, msg = True, "Timeseries Extraction Skipped: `condition` is specified but no event files found."

        return skip, msg

    def _check_sess(self, niftis, subject_header):
        ses_list = []

        for nifti in niftis:
            if "ses-" in os.path.basename(nifti):
                ses_list.append(re.search(r"ses-(\S+?)_", os.path.basename(nifti))[0][:-1])

        ses_list = sorted(set(ses_list))
        if len(ses_list) > 1:
            raise ValueError(
                f"{subject_header}"
                "`session` not specified but subject has more than one session: "
                f"{', '.join(ses_list)}. In order to continue timeseries extraction, the "
                "specific session to extract must be specified using `session`."
            )

    def _gen_runs(self, niftis):
        check_runs = []

        for nifti in niftis:
            if "run-" in os.path.basename(niftis[0]):
                check_runs.append(re.search(r"run-(\S+?)_", os.path.basename(nifti))[0][:-1])

        check_runs = set(check_runs)

        requested_runs = {}
        if self._task_info["runs"]:
            requested_runs = {f"run-{run}" for run in self._task_info["runs"]}

        return sorted(check_runs.intersection(requested_runs)) if requested_runs else sorted(check_runs)

    def _intersect_runs(self, check_runs, files):
        run_list = []

        # Check if at least one run has all files present
        for run in check_runs:
            curr_list = []

            # Assess is any of these returns True
            curr_list.append(any(f"{run}_" in file for file in files["niftis"]))

            if self._task_info["condition"]:
                curr_list.append(any(f"{run}_" in file for file in files["events"]))

            if self._signal_clean_info["use_confounds"]:
                curr_list.append(any(f"{run}_" in file for file in files["confounds"]))

                if self._signal_clean_info["n_acompcor_separate"]:
                    curr_list.append(any(f"{run}_" in file for file in files["confound_metas"]))

            if files["masks"]:
                curr_list.append(any(f"{run}_" in file for file in files["masks"]))

            # Append runs that contain all needed files
            if all(curr_list):
                run_list.append(run)

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
            base_msg += " due to " + (
                f"there being no BOLD metadata files for [TASK: {self._task_info['task']}]"
                if str(type(e).__name__) == "IndexError"
                else f"{type(e).__name__} - {str(e)}"
            )

            if self._task_info["condition"]:
                raise ValueError(
                    f"{subject_header}" f"{base_msg}" + " The `tr` must be provided when `condition` is specified."
                )
            elif any(
                [
                    self._signal_clean_info["masker_init"]["high_pass"],
                    self._signal_clean_info["masker_init"]["low_pass"],
                ]
            ):
                raise ValueError(
                    f"{subject_header}"
                    f"{base_msg}" + " The `tr` must be provided when `high_pass` or `low_pass` is specified."
                )
            elif self._signal_clean_info["fd_threshold"].get("interpolate"):
                raise ValueError("`tr` must be provided when interpolation of censored volumes is required.")
            else:
                tr = None

                if verbose:
                    LG.warning(
                        f"{subject_header}" f"{base_msg}" + " `tr` has been set to None but extraction will continue."
                    )

        return tr

    @staticmethod
    def _raise_error(msg):
        raise AttributeError(
            f"{msg} since `self.subject_timeseries` is None, either run "
            "`self.get_bold()` or assign a valid timeseries dictionary to `self.subject_timeseries`."
        )

    def timeseries_to_pickle(self, output_dir: Union[str, str], filename: Optional[str] = None) -> Self:
        """
        Save the Extracted Subject Timeseries.

        Saves the extracted timeseries stored in the ``self.subject_timeseries`` dictionary (obtained from running
        ``self.get_bold()``) as a pickle file. This allows for data persistence and easy conversion back into
        dictionary form for later use.

        Parameters
        ----------
        output_dir: :obj:`str`
            Directory to save ``self.subject_timeseries`` dictionary as a pickle file. The directory will be created if
            it does not exist.

        filename: :obj:`str` or :obj:`None`, default=None
            Name of the file with or without the "pkl" extension.

        Returns
        -------
        self
        """
        if not hasattr(self, "_subject_timeseries"):
            self._raise_error("Cannot save pickle file")

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if filename is None:
            save_filename = "subject_timeseries.pkl"
        else:
            save_filename = f"{os.path.splitext(filename.rstrip())[0].rstrip()}.pkl"

        with open(os.path.join(output_dir, save_filename), "wb") as f:
            dump(self._subject_timeseries, f)

        return self

    def visualize_bold(
        self,
        subj_id: Union[int, str],
        run: Union[int, str],
        roi_indx: Optional[Union[Union[int, str], Union[list[str], list[int]]]] = None,
        region: Optional[str] = None,
        show_figs: bool = True,
        output_dir: Optional[Union[str, str]] = None,
        filename: Optional[str] = None,
        **kwargs,
    ) -> Self:
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

        output_dir: :obj:`str` or :obj:`None`, default=None
            Directory to save plot as png image. The directory will be created if it does not exist. If None, plot will
            not be saved.

        filename: :obj:`str` or :obj:`None`, default=None
            Name of the file without the extension.

        **kwargs:
            Keyword arguments used when saving figures. Valid keywords include:

            - dpi: :obj:`int`, default=300 -- Dots per inch for the figure.
            - figsize: :obj:`tuple`, default=(11, 5) -- Size of the figure in inches.
            - bbox_inches: :obj:`str` or :obj:`None`, default="tight" -- Alters size of the whitespace in the saved image.

        Returns
        -------
        self

        Note
        ----
        **Parcellation Approach**: the "nodes" and "regions" subkeys are required in ``parcel_approach``.
        """

        if not hasattr(self, "_subject_timeseries"):
            self._raise_error("Cannot plot bold data")

        if roi_indx is None and region is None:
            raise ValueError("either `roi_indx` or `region` must be specified.")

        if roi_indx is not None and region is not None:
            raise ValueError("`roi_indx` and `region` can not be used simultaneously.")

        if filename is not None and output_dir is None:
            LG.warning("`filename` supplied but no `output_dir` specified. Files will not be saved.")

        # Defaults
        plot_dict = _check_kwargs(_PlotDefaults.visualize_bold(), **kwargs)

        # Obtain the column indices associated with the rois
        parcellation_name = list(self._parcel_approach)[0]
        if roi_indx or roi_indx == 0:
            plot_indxs = self._get_roi_indices(roi_indx, parcellation_name)
        else:
            plot_indxs = self._get_region_indices(region, parcellation_name)

        plt.figure(figsize=plot_dict["figsize"])

        timeseries = self._subject_timeseries[str(subj_id)][f"run-{run}"]
        if roi_indx or roi_indx == 0:
            plt.plot(range(1, timeseries.shape[0] + 1), timeseries[:, plot_indxs])

            if isinstance(roi_indx, (int, str)) or (isinstance(roi_indx, list) and len(roi_indx) == 1):
                if isinstance(roi_indx, int):
                    roi_title = self._parcel_approach[parcellation_name]["nodes"][roi_indx]
                elif isinstance(roi_indx, str):
                    roi_title = roi_indx
                else:
                    roi_title = roi_indx[0]
                plt.title(roi_title)
        else:
            plt.plot(range(1, timeseries.shape[0] + 1), np.mean(timeseries[:, plot_indxs], axis=1))
            plt.title(region)

        plt.xlabel("TR")

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if filename:
                save_filename = f"{os.path.splitext(filename.rstrip())[0].rstrip()}.png"
            else:
                save_filename = f"subject-{subj_id}_run-{run}_timeseries.png"

            plt.savefig(
                os.path.join(output_dir, save_filename), dpi=plot_dict["dpi"], bbox_inches=plot_dict["bbox_inches"]
            )

        plt.show() if show_figs else plt.close()

        return self

    def _get_roi_indices(self, roi_indx, parcellation_name):
        if isinstance(roi_indx, int):
            plot_indxs = roi_indx
        elif isinstance(roi_indx, str):
            # Check if parcellation_approach is custom
            if "Custom" in self._parcel_approach and "nodes" not in self._parcel_approach["Custom"]:
                _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")

            plot_indxs = list(self._parcel_approach[parcellation_name]["nodes"]).index(roi_indx)
        else:
            if all(isinstance(indx, int) for indx in roi_indx):
                plot_indxs = np.array(roi_indx)
            elif all(isinstance(indx, str) for indx in roi_indx):
                # Check if parcellation_approach is custom
                if "Custom" in self._parcel_approach and "nodes" not in self._parcel_approach["Custom"]:
                    _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")

                plot_indxs = np.array(
                    [list(self._parcel_approach[parcellation_name]["nodes"]).index(index) for index in roi_indx]
                )
            else:
                raise ValueError("All elements in `roi_indx` need to be all strings or all integers.")

        return plot_indxs

    def _get_region_indices(self, region, parcellation_name):
        if "Custom" in self._parcel_approach:
            if "regions" not in self._parcel_approach["Custom"]:
                _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")
            else:
                plot_indxs = np.array(
                    list(self._parcel_approach["Custom"]["regions"][region]["lh"])
                    + list(self._parcel_approach["Custom"]["regions"][region]["rh"])
                )
        else:
            plot_indxs = np.array(
                [
                    index
                    for index, label in enumerate(self._parcel_approach[parcellation_name]["nodes"])
                    if region in label
                ]
            )

        return plot_indxs
