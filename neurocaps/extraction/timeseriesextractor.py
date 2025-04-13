"""Contains the TimeseriesExtractor class for extracting timeseries"""

import json, os, re, sys
from functools import lru_cache
from typing import Callable, Literal, Optional, Union

# Conditional import based on major and minor version of Python
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import matplotlib.pyplot as plt, numpy as np
from joblib import Parallel, delayed
from pandas import DataFrame
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
    _pickle_object,
)

LG = _logger(__name__)


class TimeseriesExtractor(_TimeseriesExtractorGetter):
    """
    Timeseries Extractor Class.

    Performs timeseries denoising, extraction, serialization (pickling), and visualization.

    Parameters
    ----------
    space: :obj:`str`, default="MNI152NLin2009cAsym"
        The standard template space that the preprocessed bold data is registered to.

    parcel_approach: :obj:`ParcelConfig`, :obj:`ParcelApproach`, or :obj:`str`,\
                     default={"Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}}
        Specifies the parcellation approach to use. Options are "Schaefer", "AAL", or "Custom". Can be initialized with
        parameters, as a nested dictionary, or loaded from a pickle file. For detailed documentation on the expected
        structure, see the type definitions for ``ParcelConfig`` and ``ParcelApproach`` in the "See Also" section.

    standardize: :obj:`bool` or or :obj:`None`, default=True
        Standardizes the timeseries (zero mean and unit variance using sample standard deviation). Always the
        final step in the pipeline.

        .. versionchanged:: 0.25.0 No longer passed to Nilearn's ``NiftiLabelsMasker`` and only performs standardization\
        using sample standard deviation. Default behavior of standardizing using sample standard deviation is the same;\
        however, when not None or False, standardizing is always done at the end of the pipeline to prevent any\
        external standardization from needing to be done when censoring or extracting condition.

        .. note:: Standard deviations below ``np.finfo(std.dtype).eps`` are replaced with 1 for numerical stability.

    detrend: :obj:`bool`, default=False
        Detrends the timeseries.

        .. versionchanged:: 0.26.0 Default changed from True to False due to the redundancy of detrending when\
        discrete cosine-basis regressors are used, which is included in the "basic" option for ``confound_names``.

    low_pass: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Filters out signals above the specified cutoff frequency.

    high_pass: :obj:`float`, :obj:`int`, or :obj:`None``, default=None
        Filters out signals below the specified cutoff frequency.

    fwhm: :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Applies spatial smoothing to data (in millimeters).

    use_confounds: :obj:`bool`, default=True
        If True, performs nuisance regression during timeseries extraction using the default or user-specified
        confounds in ``confound_names``.

        .. important:: Requires the confound tsv files to be in same directory as preprocessed BOLD images.

    confound_names: {"basic"}, :obj:`list[str]`, or :obj:`None`, default="basic"
        Names of confounds extracted from the confound tsv files if ``use_confounds=True``.

        If "basic", the following confounds are used by default:

        - All discrete cosine-basis regressors.
        - Six head-motion parameters and their first-order derivatives.
        - First six principal aCompCor components.

        .. important::
            - Confound names follow fMRIPrep's naming scheme (versions >= 1.2.0).
            - Wildcards are supported: e.g., "cosine*" matches all confounds starting with "cosine".

    fd_threshold: :obj:`float`, :obj:`dict[str, float | int]`, or :obj:`None`, default=None
        Threshold for volume censoring based on framewise displacement (FD). Computed only after dummy volumes are
        removed.

        - *If float*, removes volumes where FD > threshold.
        - *If dict*, the following subkeys are available **(all non-required subkeys are None by default)**:

            - "threshold": A float (required). Removes volumes where FD > threshold.
            - "outlier_percentage": A float in interval [0,1]. Removes entire runs where proportion of censored volumes\
              exceeds this threshold. Proportion calculated after dummy scan removal.

            .. note::
                - A warning is issued when a run is flagged.
                - If ``condition`` specified for task-based data in ``self.get_bold()``, only considers volumes associated with the condition.

            - "n_before": An integer. Indicates the number of volumes to remove before each flagged volume.
            - "n_after": An integer. Indicates the number of volumes to remove after each flagged volume.
            - "use_sample_mask": A boolean. Controls when censoring is applied in the processing pipeline.

            .. important::
                - When True:

                    - Passes a sample mask of censored volumes to Nilearn's ``NiftiLabelsMasker``.
                    - Sets ``clean__extrapolate=False`` to prevent interpolation of end volumes.
                    - Censoring is applied before nuisance regression.
                    - If ``condition`` is specified for task-based data in ``self.get_bold()``, the timeseries is\
                      temporarily padded to extract the correct frames.

                - When False or None:

                    - Full timeseries data is used during nuisance regression.
                    - Censoring is applied after nuisance regression.

            - "interpolate": A boolean. If True, uses scipy's ``CubicSpline`` function` to perform cubic spline\
            interpolation only on censored frames. **Only performs interpolation if True**.

            .. note:: Interpolation is only performed on frames that are bounded by non-censored frames on both ends.\
                For example, given a ``censor_mask=[0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]`` where "0" indicates\
                censored high motion volumes and "1" indicates non-censored, low motion volumes, only the volumes at\
                index 3, 5, 6, 7, and 9 would be interpolated.

        .. important::
            - A column named "framewise_displacement" must be available in the confounds file.
            - ``use_confounds`` must be set to True.
            - Do not specify "framewise_displacement" in ``confound_names``.
            - See Nilearn's documentation for details on censored volume handling when "use_sample_mask" is True:

                - `Signal Clean <https://nilearn.github.io/stable/modules/generated/nilearn.signal.clean.html>`_
                - `NiftiLabelsMasker <https://nilearn.github.io/stable/modules/generated/nilearn.maskers.NiftiLabelsMasker.html>`_

            - If "interpolate" is True, then interpolation is only applied after the nuisance regression and\
              parcellation steps have been completed.
            - See Scipy's `CubicSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_ documentation.

    n_acompcor_separate: :obj:`int` or :obj:`None`, default=None
        Number of aCompCor components to extract separately from the white-matter (WM) and CSF masks. Uses first "n"
        components from each mask separately. For instance, if ``n_acompcor_separate=5``, then the the first 5 WM
        components and the first 5 CSF components (totaling 10 components) are regressed out.

        .. important::
            - ``use_confounds`` must be set to True.
            - If specified, this parameter overrides any aCompCor components listed in ``confound_names``.

    dummy_scans: :obj:`int`, :obj:`dict[str, bool | int]`, "auto", or :obj:`None`, default=None
        Number of initial volumes to remove before timeseries extraction.

        - *If int*, removes first "n" volumes.
        - *If auto*, removes first "n" volumes based on "non_steady_state_outlier_XX" columns.
        - *If dict*, the following keys are supported **(all non-required subkeys are None by default)**:

            - "auto": A boolean (required). Removes first "n" volumes based on "non_steady_state_outlier_XX" columns.
            - "min": An integer. Minimum volumes to remove when auto is set to True. If "auto" finds 2 outliers but\
              ``{"min": 3}``, removes 3 volumes.
            - "max": An integer. Maximum volumes to remove when auto is set to True. If "auto" finds 6 outliers but\
              ``{"max": 5}``, removes 5 volumes.

        .. important::
            - "auto" and dictionary option requires ``use_confounds`` to be True and "non_steady_state_outlier_XX" to\
            be present in the confounds tsv file.
            - "min" and "max" keys only apply when "auto" is True.

    dtype: :obj:`str` or "auto", default=None
        The NumPy dtype to convert NIfTI images to.


    Properties
    ----------
    space: :obj:`str`
        The standard template space that the preprocessed BOLD data is registered to. This property is also settable.

    parcel_approach: :obj:`ParcelApproach`
        Parcellation information with "maps" (path to parcellation file), "nodes" (labels), and "regions"
        (anatomical regions or networks).

    signal_clean_info: :obj:`dict[str, bool | int | float | str]` or :obj:`None`
        Dictionary containing signal cleaning parameters.

    task_info: :obj:`dict[str, str | int]` or :obj:`None`
        Dictionary containing all task-related information such. Defined after running ``self.get_bold()``.

    subject_ids: :obj:`list[str]` or :obj:`None`
        A list containing all subject IDs that have retrieved from PyBIDS and subjected to timeseries extraction.
        Defined after running ``self.get_bold()``.

    n_cores: :obj:`int` or :obj:`None`
        Number of cores used for multiprocessing with Joblib. Defined after running ``self.get_bold()``.

    subject_timeseries: :obj:`SubjectTimeseries` or :obj:`None`
        A dictionary mapping subject IDs to their run IDs and their associated timeseries (TRs x ROIs) as a NumPy array.
        Can be deleted using ``del self.subject_timeseries``. Defined after running ``self.get_bold()``.
        This property is also settable.

    qc: :obj:`dict` or :obj:`None`
        A dictionary reporting quality control, which maps subject IDs to their run IDs and information related to the
        number of frames scrubbed and interpolated as well as the mean and standard deviation of continuous high
        motion segments. Statistics for each subject's run are only reported when ``fd_threshold`` is specified, a valid
        confound tsv file containing the "framewise_displacement" column is found, and the run is not skipped.

        ::

            {"subjectID": {"run-ID": {"mean_fd": float, "std_fd": float, ...}}}

    See Also
    --------
    :class:`neurocaps.typing.ParcelConfig`
        Type definition representing the configuration options and structure for the Schaefer and AAL parcellations.\
        Refer to `ParcelConfig documentation <https://neurocaps.readthedocs.io/en/stable/generated/neurocaps.typing.ParcelConfig.html>`_.
    :class:`neurocaps.typing.ParcelApproach`
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation approaches. Refer to\
        `ParcelApproach documentation <https://neurocaps.readthedocs.io/en/stable/generated/neurocaps.typing.ParcelApproach.html>`_.
    :data:`neurocaps.typing.SubjectTimeseries`
        Type definition representing the structure of the subject timeseries. Refer to the `SubjectTimeseries\
        documentation <https://neurocaps.readthedocs.io/en/stable/generated/neurocaps.typing.SubjectTimeseries.html#neurocaps.typing.SubjectTimeseries>`_.

    Note
    ----
    **Passed Parameters**: ``detrend``, ``low_pass``, ``high_pass``, ``fwhm``, and nuisance regressors
    (``confound_names``) uses ``nilearn.maskers.NiftiLabelsMasker``. The ``dtype`` parameter is used by
    ``nilearn.image.load_img``.

    **Custom Parcellations**: Refer to the `NeuroCAPs' Parcellation Documentation
    <https://neurocaps.readthedocs.io/en/stable/parcellations.html>`_ for detailed explanations and example structures
    for Custom parcellations.
    """

    def __init__(
        self,
        space: str = "MNI152NLin2009cAsym",
        parcel_approach: Union[ParcelConfig, ParcelApproach, str] = {
            "Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}
        },
        standardize: bool = True,
        detrend: bool = False,
        low_pass: Optional[Union[float, int]] = None,
        high_pass: Optional[Union[float, int]] = None,
        fwhm: Optional[Union[float, int]] = None,
        use_confounds: bool = True,
        confound_names: Optional[Union[list[str], Literal["basic"]]] = "basic",
        fd_threshold: Optional[Union[float, dict[str, Union[bool, float, int]]]] = None,
        n_acompcor_separate: Optional[int] = None,
        dummy_scans: Optional[Union[int, dict[str, Union[bool, int]], Literal["auto"]]] = None,
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

            if dummy_scans == "auto" or (isinstance(dummy_scans, dict) and dummy_scans.get("auto")):
                raise ValueError(
                    "'auto' specified in `dummy_scans` but `use_confounds` is not True, so automated dummy "
                    "scans detection cannot be done since confounds tsv file generated by fMRIPrep is needed."
                )

            if n_acompcor_separate:
                raise ValueError(
                    "`n_acompcor_separate` specified `use_confounds` is not True, so separate WM and CSF components "
                    "cannot be regressed out since confounds tsv file generated by fMRIPrep is needed."
                )

        self._signal_clean_info = {
            "masker_init": {
                "detrend": detrend,
                "low_pass": low_pass,
                "high_pass": high_pass,
                "smoothing_fwhm": fwhm,
            },
            "standardize": standardize,
            "use_confounds": use_confounds,
            "confound_names": confound_names,
            "n_acompcor_separate": n_acompcor_separate,
            "dummy_scans": dummy_scans,
            "fd_threshold": fd_threshold,
            "dtype": dtype,
        }

    @staticmethod
    def _validate_init_params(param, struct):

        mandatory_keys = {"dummy_scans": {"auto": (bool,)}, "fd_threshold": {"threshold": (float, int)}}

        optional_keys = {
            "dummy_scans": {"min": (int,), "max": (int,)},
            "fd_threshold": {
                "n_before": (int,),
                "n_after": (int,),
                "outlier_percentage": (float,),
                "use_sample_mask": (bool,),
                "interpolate": (bool,),
            },
        }

        type2str = lambda t: t.__name__
        str2text = lambda valid_types: ", ".join([type2str(t) for t in valid_types])

        valid_types = (dict, int, str) if param == "dummy_scans" else (dict, float, int)
        if not isinstance(struct, valid_types):
            raise TypeError(f"`{param}` must be one of the following types when not None: {str2text(valid_types)}.")

        if param == "dummy_scans" and (isinstance(struct, str) and not struct == "auto"):
            raise ValueError(f"'auto' is the only valid string for `dummy_scans`.")

        if isinstance(struct, dict):
            # Check mandatory keys
            key = list(mandatory_keys[param].keys())[0]
            if key not in struct:
                raise KeyError(f"'{key}' is a mandatory key when `{param}` is a dictionary.")
            if not isinstance(struct[key], mandatory_keys[param][key]):
                raise TypeError(
                    f"'{key}' must be of one of the following types: {str2text(mandatory_keys[param][key])}."
                )

            # Check optional keys
            for key in optional_keys[param]:
                if key in struct:
                    if struct[key] is not None and not isinstance(struct[key], optional_keys[param][key]):
                        raise TypeError(
                            f"'{key}' must be either None or of type {str2text(optional_keys[param][key])}."
                        )
                    # Additional check for "outlier_percentage"
                    if key == "outlier_percentage" and not 0 < struct["outlier_percentage"] < 1:
                        raise ValueError("'outlier_percentage' must be either None or a float between 0 and 1.")

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
        run_subjects: Optional[Union[str, list[str]]] = None,
        exclude_subjects: Optional[Union[str, list[str]]] = None,
        exclude_niftis: Optional[Union[str, list[str]]] = None,
        pipeline_name: Optional[str] = None,
        n_cores: Optional[int] = None,
        parallel_log_config: Optional[dict[str, Union[Callable, int]]] = None,
        verbose: bool = True,
        flush: bool = False,
        progress_bar: bool = False,
    ) -> Self:
        """
        Retrieve Preprocessed BOLD Data from BIDS Datasets.

        Extracts the timeseries data from preprocessed BOLD images located in the derivatives folder of a
        BIDS-compliant dataset. The timeseries data of all subjects are appended to a single dictionary
        ``self.subject_timeseries``.

        .. important::
            - For proper querying, a "dataset_description.json" file must be located in the root of the BIDs directory\
            and the pipeline directory (located in the derivatives folder).
            - Refer to `NeuroCAPs' BIDS Structure and Entities documentation <https://neurocaps.readthedocs.io/en/stable/bids.html>`_\
            for additional information on the expected directory structure and file naming scheme (entities) needed for querying.
            - This pipeline is most optimized for BOLD data preprocessed by fMRIPrep.

        Parameters
        ----------
        bids_dir: :obj:`str`
            Path to a BIDS compliant directory. A "dataset_description.json" file must be located in this directory
            or an error will be raised.

        task: :obj:`str`
            Name of task to extract timeseries data from (i.e "rest", "n-back", etc).

        session: :obj:`int`, :obj:`str`, or :obj:`None`, default=None
            The session ID to extract timeseries data from. Only a single session can be extracted at a time. The value
            can be an integer (e.g. ``session=2``) or a string (e.g. ``session="001"``).

        runs: :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            List of run numbers to extract timeseries data from (e.g. ``runs=["000", "001"]``). Extracts all runs if unspecified.

        condition: :obj:`str` or :obj:`None`, default=None
            Isolates the timeseries data corresponding to a specific condition (listed in the "trial_type" column of
            the "events.tsv" file) after the timeseries has been extracted and subjected to nuisance regression.
            Only a single condition can be extracted at a time.

        condition_tr_shift: :obj:`int`, default=0
            Number of TR units to units to offset both the start and end scan indices of a condition to account for
            a fixed hemodynamic delay. This parameter only applies when a ``condition`` is specified. For more details
            about how this offset affects the calculation of task conditions, see the "Extraction of Task Conditions"
            section below.

        tr: :obj:`int`, :obj:`float` or :obj:`None`, default=None
            Repetition time (TR), in seconds, for the specified task. If not provided, the TR will be automatically
            extracted from the first BOLD metadata file found for the task, searching first in the pipeline directory,
            then in the ``bids_dir`` if not found.

        slice_time_ref: :obj:`int` or :obj:`float`, default=0.0
            The reference slice expressed as a fraction of the ``tr`` that is subtracted from the condition onset times
            to adjust for slice time correction when ``condition`` is not None. Values can range from 0 to 1. For more
            details, see the "Extraction of Task Conditions" section below.

        run_subjects: :obj:`str`, :obj:`list[str]` or :obj:`None`, default=None
            A string (if single subject) or list of subject IDs to process (e.g. ``run_subjects=["01", "02"]``).
            Processes all subjects if None.

        exclude_subjects: :obj:`str`, :obj:`list[str]` or :obj:`None`, default=None
            A string (if single subject) or list of subject IDs to exclude (e.g. ``exclude_subjects=["01", "02"]``).

        exclude_niftis: :obj:`str`, :obj:`list[str]` or :obj:`None`, default=None
            A string (if single file) or List of the specific preprocessed NIfTI files to exclude, preventing their
            timeseries data from being extracted. Used if there are specific runs across different participants that
            need to be excluded.

        pipeline_name: :obj:`str` or :obj:`None`, default=None
            The name of the pipeline folder in the derivatives folder containing the preprocessed data. Used if
            multiple pipeline folders exist in the derivatives folder or the pipeline folder is nested
            (e.g. "fmriprep/fmriprep-20.0.0").

        n_cores: :obj:`int` or :obj:`None`, default=None
            The number of cores to use for multiprocessing with Joblib. The "loky" backend is used.

        parallel_log_config: :obj:`dict[str, multiprocessing.Manager.Queue | int]`
            Passes a user-defined managed queue and logging level to the internal timeseries extraction function
            when parallel processing (``n_cores``) is used. Available dictionary keys are:

            - "queue": The instance of ``multiprocessing.Manager.Queue`` to pass to ``QueueHandler``. If not specified,
              all logs will output to ``sys.stdout``.
            - "level": The logging level (e.g. ``logging.INFO``, ``logging.WARNING``). If not specified, the default
              level is ``logging.INFO``.

            Refer to the `NeuroCAPs' Logging documentation <https://neurocaps.readthedocs.io/en/stable/logging.html>`_
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

        Returns
        -------
        self

        Raises
        ------
        BIDSQueryError
            Occurs when subject IDs were not found during querying. Refer to `BIDSQueryError documentation
            <https://neurocaps.readthedocs.io/en/stable/generated/neurocaps.exceptions.BIDSQueryError.html#neurocaps.exceptions.BIDSQueryError>`_.

        See Also
        --------
        :data:`neurocaps.typing.SubjectTimeseries`
            Type definition representing the structure of the subject timeseries. Refer to the `SubjectTimeseries
            documentation <https://neurocaps.readthedocs.io/en/stable/generated/neurocaps.typing.SubjectTimeseries.html#neurocaps.typing.SubjectTimeseries>`_.

        Important
        ---------
        **Subject Timeseries Dictionary**: This function stores the extracted timeseries of all subjects
        in the ``subject_timeseries`` property and can be deleted using ``del self.subject_timeseries`` (Note that
        ``self.timeseries_to_pickle()`` and ``self.visualize_bold()`` need this property in order to be used). The
        structure is a dictionary mapping subject IDs to their run IDs and their associated timeseries (TRs x ROIs) as
        NumPy array. Refer to documentation for ``SubjectTimeseries`` in the "See Also" section for an example structure.

        **Data/Property Persistence**: Each time this function is called, it's associated properties such as
        ``self.subject_timeseries``, ``self.task_info``, ``self.qc``, etc, are automatically initialized/overwritten to
        create a clean state for the subsequent analysis. To save, the subject timeseries dictionary,
        ``self.timeseries_to_pickle()`` can be used. Additionally, to save the quality control dictionary,
        ``self.report_qc()`` can be used.

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
            adjusted_onset = max([0, adjusted_onset])
            start_scan = int(adjusted_onset / tr) + condition_tr_shift
            end_scan = math.ceil((adjusted_onset + duration) / tr) + condition_tr_shift
            scans.extend(range(onset_scan, end_scan))
            scans = sorted(list(set(scans)))

        When partial scans are computed, ``int`` is used to round down for the beginning scan index and ``math.ceil``
        is used to round up for the ending scan index. Negative scan indices are set to 0 to avoid unintentional
        negative indexing. For simplicity, note that when ``slice_time_ref`` and ``condition_tr_shift`` are 0, the
        formula simplifies to:

        ::

            start_scan = int(onset / tr)
            end_scan = math.ceil((onset + duration) / tr)
            scans.extend(range(onset_scan, end_scan))
            scans = sorted(list(set(scans)))

        Filtering a specific condition from the timeseries is done after nuisance regression. The indices corresponding
        to the condition are used to extract the TRs (the timepoints that fall within the the event window(s)
        adjusted by the slice timing reference (``slice_time_ref``) and a fixed hemodynamic delay
        (``condition_tr_shift``) if specified) from the timeseries.

        If the "use_sample_mask" key in the ``fd_threshold`` dictionary is set to True, the truncated 2D timeseries
        is temporarily padded to ensure the correct rows corresponding to the condition are obtained.

        If the "interpolate" key in the ``fd_threshold`` dictionary is set to True, interpolation is performed using
        the full timeseries data (excluding dummy volumes) to replace only the censored (high-motion) volumes. Then,
        the indices corresponding to the condition are extracted from the timeseries, excluding any frames that do not
        have non-censored data at both edges.
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
        self._qc = {}
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
                "3. Incorrect task name specified in `task` parameter.\n"
                "4. The cache may need to be cleared using ``TimeseriesExtractor._call_layout.cache_clear()`` if the "
                "directory has been changed (e.g. new files added, file names changed, etc) during the current Python "
                "session."
            )
            raise BIDSQueryError(msg)

        if exclude_subjects:
            exclude_subjects = exclude_subjects if isinstance(exclude_subjects, list) else [exclude_subjects]
            exclude_subjects = [subj_id.removeprefix("sub-") for subj_id in map(str, exclude_subjects)]
            subj_ids = sorted([subj_id for subj_id in subj_ids if subj_id not in exclude_subjects])

        if run_subjects:
            run_subjects = run_subjects if isinstance(run_subjects, list) else [run_subjects]
            run_subjects = [subj_id.removeprefix("sub-") for subj_id in map(str, run_subjects)]
            subj_ids = sorted([subj_id for subj_id in subj_ids if subj_id in run_subjects])

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
                subject_timeseries, qc = output
                self._expand_dicts(subject_timeseries, qc)
        else:
            if parallel_log_config:
                LG.warning(
                    "`parallel_log_config` is only used for parallel processing. The default logger can be "
                    "modified by configuring either the root logger or a logger for a specific module prior to "
                    "package import."
                )

            for subj_id in tqdm(self._subject_ids, desc="Processing Subjects", disable=not progress_bar):
                outputs = _extract_timeseries(
                    subj_id=subj_id,
                    **self._subject_info[subj_id],
                    parcel_approach=self._parcel_approach,
                    signal_clean_info=self._signal_clean_info,
                    task_info=self._task_info,
                    verbose=verbose,
                    flush=flush,
                )

                self._expand_dicts(*outputs)

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
                run_list = self._filter_runs(check_runs, files)

                # Skip subject if no run has all needed files present
                if not run_list:
                    if verbose:
                        LG.warning(
                            f"{subject_header}"
                            "Timeseries Extraction Skipped: None of the necessary files (i.e NifTIs, "
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

            # Get repetition time for the subject
            tr = self._get_tr(files["bold_meta"], subject_header, verbose)

            # Add subject list to subject attribute. These are subjects that will be ran
            self._subject_ids.append(subj_id)

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
        exclude_niftis = exclude_niftis if isinstance(exclude_niftis, list) else [exclude_niftis]
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

    def _filter_runs(self, check_runs, files):
        run_list = []

        # Check if at least one run has all files present
        for run in check_runs:
            bool_list = []

            # Assess is any of these returns True
            bool_list.append(any(f"{run}_" in file for file in files["niftis"]))

            if self._task_info["condition"]:
                bool_list.append(any(f"{run}_" in file for file in files["events"]))

            if self._signal_clean_info["use_confounds"]:
                bool_list.append(any(f"{run}_" in file for file in files["confounds"]))

                if self._signal_clean_info["n_acompcor_separate"]:
                    bool_list.append(any(f"{run}_" in file for file in files["confound_metas"]))

            # Append runs that contain all needed files
            if all(bool_list):
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
                if verbose:
                    LG.warning(
                        f"{subject_header}" f"{base_msg}" + " `tr` has been set to None but extraction will continue."
                    )
                    tr = None

        return tr

    def _expand_dicts(self, subject_timeseries, qc):
        # Aggregate new timeseries dictionary and qc
        if isinstance(subject_timeseries, dict):
            self._subject_timeseries.update(subject_timeseries)
            if qc:
                self._qc.update(qc)

    @staticmethod
    def _raise_error(prop_name, msg):
        if prop_name == "_subject_timeseries":
            raise AttributeError(
                f"{msg} since `self.subject_timeseries` is None, either run `self.get_bold()` or assign a valid "
                "timeseries dictionary to `self.subject_timeseries`."
            )
        else:
            raise AttributeError(f"{msg} since `self.qc` is None, run `self.get_bold()` first.")

    def timeseries_to_pickle(self, output_dir: str, filename: Optional[str] = None) -> Self:
        """
        Save the Extracted Subject Timeseries.

        Saves the extracted timeseries stored in the ``self.subject_timeseries`` dictionary as a pickle file with
        Joblib.

        Parameters
        ----------
        output_dir: :obj:`str`
            Directory to save a pickle file to. The directory will be created if it does not exist.

        filename: :obj:`str` or :obj:`None`, default=None
            Name of the file with or without the "pkl" extension. If None, will use "subject_timeseries.pkl" as default.

        Returns
        -------
        self
        """
        save_filename = self._prepare_output_file(
            output_dir, filename, prop_name="_subject_timeseries", call="timeseries_to_pickle"
        )

        _pickle_object(self._subject_timeseries, output_dir, save_filename)

        return self

    def report_qc(
        self, output_dir: Optional[str] = None, filename: Optional[str] = None, return_df: bool = True
    ) -> Union[DataFrame, None]:
        """
        Report Quality Control Information.

        Converts per-subject, per-run quality control metrics (frame scrubbing and interpolation) stored in
        ``self.qc`` dictionary to a pandas DataFrame for analysis and reporting.

        Parameters
        ----------
        output_dir: :obj:`str`, default=None
            Directory to save a pickle file to. The directory will be created if it does not exist. File will not be
            saved in an output directory is not provided.

        filename: :obj:`str` or :obj:`None`, default=None
            Name of the file with or without the "csv" extension.

        return_df: :obj:`bool`, default=True
            If True, returns the dataframe.

        Returns
        -------
        pandas.Dataframe
            Pandas dataframe containing the colums: "Subject_ID", "Run", "Frames_Scrubbed", "Frames_Interpolated",
            "Mean_High_Motion_Length", "Std_High_Motion_Length", "Mean_FD", "Std_FD",

        Important
        ---------
        **Reporting**: Statistics for each subject's run are only reported when ``fd_threshold`` is specified, a valid
        confound tsv file containing the "framewise_displacement" column is found, and the run is not skipped.
        Additionally, all reported statistics exclude any dummy volumes, as they are calculated after dummy volumes are
        removed from the analysis.

        **Note** that if a ``condition`` was specified in ``self.get_bold()``, all the reported statistics are specific
        to the frames assigned to the condition (as determined by the equation in "Extraction of Task Conditions"
        in the documentation of ``self.get_bold()``), which are treated as one continuous event window for computational
        simplicity.

        **Mean & Standard Deviation of Framewise Displacement:** "Mean_FD" and "Std_FD" represent the statistics prior
        to scrubbing or interpolating.

        **Censored & Interpolated Frames**: The metrics for scrubbed frames and interpolated frames are independent. For
        instance, 2 scrubbed frames and 3 interpolated frames means that 3 frames were interpolated while 2 were
        scrubbed due to excessive motion.

        **Note** that only censored frames with bounded by non-censored frames on both sides are interpolated, while
        censored frames at the edge of the timeseries (including frames that border censored edges) are always scrubbed
        and counted in "Frames_Scrubbed". For instance, if the full sample mask is computed as ``[0, 0, 1, 0, 0, 1, 0]``
        where "0" are censored and "1" is not censored, then when no interpolation is requested, then "Frames_Scrubbed"
        will be 5 and "Frames_Interpolated" will be 0. If interpolation is requested, then "Frames_Scrubbed" would be 3
        and "Frames_Interpolated" would be 2 (indexes 3 and 4).

        **High Motion Length Computation**: "Mean_High_Motion_Length" and "Std_High_Motion_Length" represent the average
        length and population standard deviation of contiguous segments of frames flagged for high-motion frames,
        respectively.
        """
        save_filename = self._prepare_output_file(output_dir, filename, prop_name="_qc", call="report_qc")
        assert self._qc, "No quality control information to report."

        # Build df
        df = DataFrame(
            columns=[
                "Subject_ID",
                "Run",
                "Mean_FD",
                "Std_FD",
                "Frames_Scrubbed",
                "Frames_Interpolated",
                "Mean_High_Motion_Length",
                "Std_High_Motion_Length",
            ]
        )
        for subject in self._qc:
            for run in self._qc[subject]:
                df.loc[len(df)] = [
                    subject,
                    run,
                    self._qc[subject][run]["mean_fd"],
                    self._qc[subject][run]["std_fd"],
                    self._qc[subject][run]["frames_scrubbed"],
                    self._qc[subject][run]["frames_interpolated"],
                    self._qc[subject][run]["mean_high_motion_length"],
                    self._qc[subject][run]["std_high_motion_length"],
                ]

        if output_dir:
            df.to_csv(os.path.join(output_dir, save_filename), sep=",", index=0)

        if return_df:
            return df

    def _prepare_output_file(self, output_dir, filename, prop_name, call):
        if not hasattr(self, prop_name):
            self._raise_error(
                prop_name,
                msg="Cannot save pickle file" if prop_name == "_subject_timeseries" else "Cannot save csv file",
            )

        if not output_dir:
            return None
        elif output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ext = "pkl" if call == "timeseries_to_pickle" else "csv"

        if filename is None:
            save_filename = "subject_timeseries.pkl" if call == "timeseries_to_pickle" else "report_qc.csv"
        else:
            save_filename = f"{os.path.splitext(filename.rstrip())[0].rstrip()}.{ext}"

        return save_filename

    def visualize_bold(
        self,
        subj_id: Union[int, str],
        run: Optional[Union[int, str]] = None,
        roi_indx: Optional[Union[int, str, list[str], list[int]]] = None,
        region: Optional[str] = None,
        show_figs: bool = True,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
        as_pickle: bool = False,
        **kwargs,
    ) -> Self:
        """
        Plot the Extracted Subject Timeseries.

        Visualize the extracted BOLD timeseries data of nodes (Region-of-Interests [ROIs]) or regions (anatomical
        regions/networks) for a specific subject and run.

        Parameters
        ----------
        subj_id: :obj:`str` or :obj:`int`
            The ID of the subject.

        run: :obj:`int`, :obj:`str`, or :obj:`None`, default=None
            The run ID of the subject to plot. Must be specified if multiple runs exist for a given subject.

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

        as_pickle: :obj:`bool`, default=False
            When ``output_dir`` is specified, plots are saved as pickle file, which can be further modified, instead of
            png images.

            .. versionadded:: 0.26.5

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
            self._raise_error(prop_name="_subject_timeseries", msg="Cannot plot bold data")

        subj_id = str(subj_id).removeprefix("sub-")
        if subj_id not in self._subject_timeseries:
            raise KeyError(f"Subject {subj_id} is not available in `self._subject_timeseries`.")

        if len(subject_runs := list(self._subject_timeseries[subj_id])) > 1 and not run:
            raise ValueError(
                f"`run` must be specified when multiple runs exist. Runs available for sub-{subj_id}: "
                f"{', '.join(subject_runs)}."
            )

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

        run = None if not run else str(run)
        run = subject_runs[0] if not run else f"run-{run.removeprefix('run-')}"
        timeseries = self._subject_timeseries[subj_id][run]
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
                save_filename = f"subject-{subj_id}_{run}_timeseries.png"

            if not as_pickle:
                plt.savefig(
                    os.path.join(output_dir, save_filename), dpi=plot_dict["dpi"], bbox_inches=plot_dict["bbox_inches"]
                )
            else:
                _pickle_object(plt.gcf(), output_dir, save_filename.replace(".png", ".pkl"))

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
