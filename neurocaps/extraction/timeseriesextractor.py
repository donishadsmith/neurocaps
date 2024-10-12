import json, os, re
from typing import Union, Optional, Literal
import matplotlib.pyplot as plt, numpy as np
from joblib import Parallel, delayed, dump
from .._utils import (_TimeseriesExtractorGetter, _check_kwargs, _check_confound_names,
                      _check_parcel_approach, _extract_timeseries, _logger)

LG = _logger(__name__)

class TimeseriesExtractor(_TimeseriesExtractorGetter):
    """
    **Timeseries Extractor Class**

    Initializes the Timeseries Extractor class.

    Parameters
    ----------
    space : :obj:`str`, default="MNI152NLin2009cAsym"
        The standard template space that the preprocessed bold data is registered to. Used for querying with pybids
        to locate preprocessed BOLD-related files.

    parcel_approach : :obj:`dict[str, dict[str, str | int]]` or :obj:`os.PathLike`, default={"Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}}
        The approach to parcellate BOLD images. This should be a nested dictionary with the first key being the
        atlas name. Currently, only "Schaefer", "AAL", and "Custom" are supported.

        - For "Schaefer", available sub-keys include "n_rois", "yeo_networks", and "resolution_mm".
          Refer to documentation for ``nilearn.datasets.fetch_atlas_schaefer_2018`` for valid inputs.
        - For "AAL", the only sub-key is "version". Refer to documentation for ``nilearn.datasets.fetch_atlas_aal``
          for valid inputs.
        - For "Custom", the key must include a sub-key called "maps" specifying the directory location of the
          parcellation.

    standardize : {"zscore_sample", "zscore", "psc", True, False}, default="zscore_sample"
        Determines whether to standardize the timeseries. Refer to ``nilearn.maskers.NiftiLabelsMasker`` for available
        options.

    detrend : :obj:`bool`, default=True
        Detrends the timeseries during extraction.

    low_pass : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Filters out signals above the specified cutoff frequency.

    high_pass : :obj:`float`, :obj:`int`, or :obj:`None``, default=None
        Filters out signals below the specified cutoff frequency.

    fwhm : :obj:`float`, :obj:`int`, or :obj:`None`, default=None
        Applies spatial smoothing to data (in millimeters). Note that using parcellations already averages voxels
        within parcel boundaries, which can improve signal-to-noise ratio (SNR) assuming Gaussian noise
        distribution. However, smoothing may also blur parcel boundaries.

    use_confounds : :obj:`bool`, default=True
        Determines whether to perform nuisance regression using confounds when extracting timeseries.

    confound_names : :obj:`list[str]` or :obj:`None`, default=None
        Specifies the names of confounds to use from confound files. If None, default confounds are used.
        Note, an asterisk ("*") can be used to find confound names that start with the term preceding the
        asterisk. For instance, "cosine*" will find all confound names in the confound files starting with "cosine".

    fd_threshold : :obj:`float`, :obj:`dict[str, float]`, or :obj:`None`, default=None
        Sets a threshold to remove volumes after nuisance regression and timeseries extraction. This requires a
        column named `framewise_displacement` in the confounds file and ``use_confounds`` set to True.
        Additionally, `framewise_displacement` should not need be specified in ``confound_names`` if using this
        parameter. If, ``fd_threshold`` is a dictionary, the following keys can be specified:

        - "threshold" : A float value. Volumes with a `framewise_displacement` value exceeding this threshold are removed.
        - "outlier_percentage" : A float value between 0 and 1 representing a percentage. Runs where the proportion of
          volumes exceeding the "threshold" is higher than this percentage are removed. If ``condition`` is specified
          in ``self.get_bold``, only the runs where the proportion of volumes exceeds this value for the specific
          condition of interest are removed. **Note**, this proportion is calculated after dummy scans have been removed.
          A warning is issued whenever a run is flagged.

    n_acompcor_separate : :obj:`int` or :obj:`None`, default=None
        Specifies the number of separate acompcor components derived from white-matter (WM) and cerebrospinal
        fluid (CSF) masks to use. For example, if set to 5, the first five components from the WM mask
        and the first five from the CSF mask will be used, totaling ten acompcor components. If this parameter is
        not None, any acompcor components listed in ``confound_names`` will be disregarded. To use acompcor
        components derived from combined masks (WM & CSF), leave this parameter as None and list the specific
        acompcors of interest in ``confound_names``.

    dummy_scans : :obj:`int`, :obj:`dict[str, bool | int]`, or :obj:`None`, default=None
        Removes the first n volumes before extracting the timeseries. If, ``dummy_scans`` is a dictionary,
        the following keys can be used:

        - "auto" : A boolean value. If True, the number of dummy scans removed depend on the number of
          "non_steady_state_outlier_XX" columns in the participants fMRIPrep confounds tsv file. For instance, if
          there are two "non_steady_state_outlier_XX" columns detected, then ``dummy_scans`` is set to two since
          there is one "non_steady_state_outlier_XX" per outlier volume for fMRIPrep. This is assessed for each run of
          all participants so ``dummy_scans`` depends on the number number of "non_steady_state_outlier_XX" in the
          confound file associated with the specific participant, task, and run number.
        - "min" : An integer value indicating the minimum dummy scans to discard. The "auto" sub-key must be True
          for this to work. If, for instance, only two "non_steady_state_outlier_XX" columns are detected but the
          "min" is set to three, then three dummy volumes will be discarded.
        - "max" : An integer value indicating the maximum dummy scans to discard. The "auto" sub-key must be True
          for this to work. If, for instance, six "non_steady_state_outlier_XX" columns are detected but the
          "max" is set to five, then five dummy volumes will be discarded.


    Property
    --------
    space : str
        The standard template space that the preprocessed BOLD data is registered to. The space can also be set after
        class initialization using ``self.space = "New Space"`` if the template space needs to be changed.

    parcel_approach : :obj:`dict[str, dict[str, os.PathLike | list[str]]]`
        Nested dictionary containing information about the parcellation. Can also be used as a setter, which accepts a
        dictionary or a dictionary saved as pickle file. If "Schaefer" or "AAL" was specified during
        initialization of the ``TimeseriesExtractor`` class, then ``nilearn.datasets.fetch_atlas_schaefer_2018``
        and ``nilearn.datasets.fetch_atlas_aal`` will be used to obtain the "maps" and the "nodes". Then string
        splitting is used on the "nodes" to obtain the "regions":
        ::

            {
                "Schaefer":
                {
                    "maps": "path/to/parcellation.nii.gz",
                    "nodes": ["LH_Vis1", "LH_SomSot1", "RH_Vis1", "RH_Somsot1"],
                    "regions": ["Vis", "SomSot"]
                }
            }

            {
                "AAL":
                {
                    "maps": "path/to/parcellation.nii.gz",
                    "nodes": ["Precentral_L", "Precentral_R", "Frontal_Sup_L", "Frontal_Sup_R"],
                    "regions": ["Precentral", "Frontal"]
                }
            }

        If "Custom" is specified, only checks are done to ensure that the dictionary contains the proper
        sub-keys such as "maps", "nodes", and "regions". Unlike "Schaefer" and "AAL",
        "regions" must be a nested dictionary specifying the name of the region as the first level key and the
        indices in the "nodes" list belonging to the "lh" and "rh" for that region. Refer to the structure
        example for "Custom" in the Note section below.

    signal_clean_info : :obj:`dict[str]`
        Dictionary containing parameters for signal cleaning specified during initialization of the
        ``TimeseriesExtractor`` class. This information includes ``standardize``, ``detrend``, ``low_pass``,
        ``high_pass``, ``fwhm``, ``dummy_scans``, ``use_confounds``, ``n_compcor_separate``, and ``fd_threshold``.

    task_info : :obj:`dict[str]`
        If ``self.get_bold()`` ran, is a dictionary containing all task-related information such as ``task``,
        ``condition``, ``session``, ``runs``, and ``tr`` (if specified) else None.

    subject_ids : :obj:`list[str]`
        A list containing all subject IDs that have retrieved from pybids and subjected to timeseries
        extraction.

    n_cores : :obj:`int`
        Number of cores used for multiprocessing with joblib.

    subject_timeseries : :obj:`dict[str, dict[str, np.ndarray]`
        Nested dictionary containing the subject ID, run ID, and the 2D numpy arrays for timeseries data. Can
        all be used as a setter, which accepts a dictionary or a dictionary saved as pickle file.
        If this property needs to be deleted due to space issues, ``delattr(self,"_subject_timeseries")``
        can be used to delete the array. The structure is as follows:
        ::

            subject_timeseries = {
                    "101": {
                        "run-0": np.array([...]), # 2D array
                        "run-1": np.array([...]), # 2D array
                        "run-2": np.array([...]), # 2D array
                    },
                    "102": {
                        "run-0": np.array([...]), # 2D array
                        "run-1": np.array([...]), # 2D array
                    }
                }

    Note
    ----
    ``standardize``, ``detrend``, ``low_pass``, ``high_pass``, ``fwhm``, and nuisance regression uses
    ``nilearn.maskers.NiftiLabelsMasker``.

    Default ``confound_names`` changes if ``high_pass`` is not None.

    If ``high_pass`` is not None, then the cosine parameters and acompcor parameters are not used because the cosine
    parameters are high-pass filters and the acompcor regressors are estimated on the high-pass filtered version of
    the data:
    ::

        confound_names = ["trans_x", "trans_x_derivative1","trans_y", "trans_y_derivative1",
                          "trans_z", "trans_z_derivative1",  "rot_x", "rot_x_derivative1",
                          "rot_y", "rot_y_derivative1", "rot_z", "rot_z_derivative1"]

    If ``high_pass`` is None, then:
    ::

        confound_names = ["trans_x", "trans_x_derivative1","trans_y", "trans_y_derivative1",
                          "trans_z", "trans_z_derivative1",  "rot_x", "rot_x_derivative1",
                          "rot_y", "rot_y_derivative1", "rot_z", "rot_z_derivative1"]

    else:
    ::

        confound_names = ["cosine*","trans_x", "trans_x_derivative1","trans_y", "trans_y_derivative1",
                          "trans_z", "trans_z_derivative1",  "rot_x", "rot_x_derivative1",
                          "rot_y", "rot_y_derivative1", "rot_z", "rot_z_derivative1", "a_comp_cor_00",
                          "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05"]


    **If using a "Custom" parcellation approach**, ensure that the atlas is lateralized (where each region/network has
    nodes in the left and right hemisphere). The visualization function in this class assume that the background
    label is "0". Do not add a background label in the "nodes" or "regions" key; the zero index should correspond to
    the first ID that is not "0".

    - "maps": Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g.,
      .nii for NIfTI files). **This key is required for timeseries extraction**.
    - nodes: List of all node labels used in your study, arranged in the exact order they correspond to indices in
      your parcellation files. Each label should match the parcellation index it represents. For example, if the
      parcellation label "1" corresponds to the left hemisphere visual cortex area 1, then "LH_Vis1" should occupy
      the 0th index in this list. This ensures that data extraction and analysis accurately reflect the anatomical
      regions intended. For timeseries extraction, this key is not required.
    - regions: Dictionary defining major brain regions. Each region should list node indices under "lh" and
      "rh" to specify left and right hemisphere nodes. For timeseries extraction, this key is not required.

    The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis)
    and hippocampus regions:
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
                    "Vis": {
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
    """
    def __init__(self,
                 space: str = "MNI152NLin2009cAsym",
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
                 dummy_scans: Optional[Union[int, dict[str, Union[bool, int]]]]=None) -> None:

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

        self._signal_clean_info = {"masker_init": {"standardize": standardize,
                                                   "detrend": detrend,
                                                   "low_pass": low_pass,
                                                   "high_pass": high_pass,
                                                   "smoothing_fwhm": fwhm},
                                    "dummy_scans": dummy_scans,
                                    "n_acompcor_separate": n_acompcor_separate,
                                    "fd_threshold": None,
                                    "use_confounds": use_confounds}
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
                 pipeline_name: Optional[str]=None,
                 n_cores: Optional[int]=None,
                 verbose: bool=True,
                 flush: bool=False,
                 exclude_niftis: Optional[list[str]]=None) -> None:
        """
        **Retrieve Preprocessed BOLD Data from BIDS Datasets**

        This function uses pybids for querying and requires the BOLD data directory (specified in ``bids_dir``) to be
        BIDS-compliant, including a "dataset_description.json" file. It assumes the dataset contains a derivatives
        folder with BOLD data preprocessed using a standard pipeline, specifically fMRIPrep. The pipeline directory
        must also include a "dataset_description.json" file for proper querying.

        For timeseries extraction, nuisance regression, and spatial dimensionality reduction using a parcellation,
        nilearn’s ``NiftiLabelsMasker`` function is used. If requested, dummy scans are removed from the NIfTI images
        and confound dataset prior to timeseries extraction. Indices exceeding framewise displacement thresholds are
        removed after timeseries extraction, and the extracted timeseries can be filtered to include only indices
        corresponding to a specific condition, if requested.

        The timeseries data of all subjects are appended to a single dictionary ``self.subject_timeseries``. Additional
        information regarding the structure of this dictionary can be found in the "Note" section.

        **This pipeline is most optimized for BOLD data preprocessed by fMRIPrep.**

        Parameters
        ----------
        bids_dir : :obj:`os.PathLike`
            Absolute path to a BIDS compliant directory.

        task : :obj:`str`
            Name of task to process (i.e "rest", "n-back", etc).

        session : :obj:`int`, :obj:`str`, or :obj:`None`, default=None
            Session to extract timeseries from. Only a single session can be extracted at a time. An error will be
            issued if more than one session is detected in the preprocessed NifTI files. Session should be in the
            for of an integer (e.g.  ``session=2``) or a string if the session id cannot be represented
            as an integer (e.g. ``session="001"``).

        runs : :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[str]`, or :obj:`None`, default=None
            List of run numbers to extract timeseries data from. Extracts all runs if unspecified.
            For instance, if only "run-0" and "run-1" should be extracted, then:
            ::

                runs=[0,1]
                # Or if run IDs are not integers
                runs=["000","001"]

        condition : :obj:`str` or :obj:`None`, default=None
            Specific condition in the task to extract from. Only a single condition can be extracted at a time.

        tr : :obj:`int`, :obj:`float`, or :obj:`None`, default=None
            Repetition time for the task. If the ``tr`` is not specified, for each subject, the tr will be extracted from
            the first BOLD metadata file, for the specified task, in the pipeline directory or the `bids_dir` if not
            in the pipeline directory.

        run_subjects : :obj:`list[str]` or :obj:`None`, default=None
            List of subject IDs to process. Processes all subjects if None.

        exclude_subjects : :obj:`list[str]` or :obj:`None`, default=None
            List of subject IDs to exclude.

        pipeline_name : :obj:`str` or :obj:`None`, default=None
            The name of the pipeline folder in the derivatives folder containing the preprocessed data. If None,
            ``BIDSLayout`` will use the name of ``bids_dir`` with ``derivatives=True``. This parameter should be
            used if there are multiple pipelines or pipelines are nested in folders in the derivatives folder. If
            specified, the first level of the folder should contain the "dataset_description.json" file or an
            error will occur. For instance, if the json file in "path/to/bids/derivatives/fmriprep/fmriprep-20.0.0",
            then ``pipeline_name = "fmriprep/fmriprep-20.0.0"``.

        n_cores : :obj:`int` or :obj:`None`, default=None
            The number of CPU cores to use for multiprocessing with joblib.

        verbose : :obj:`bool`, default=True
            Print subject-specific information such as confounds being extracted, and id and run of subject being
            processed during timeseries extraction.

        flush : :obj:`bool`, default=False
            Flush the printed subject-specific information produced during the timeseries extraction process.
            Used to immediately print out information such as the current subject being processed, confounds found,
            etc.

            .. versionchanged:: 0.17.0 Changed from ``flush_print`` to ``flush``. 

        exclude_niftis : :obj:`list[str]` or :obj:`None`, default=None
            List of specific preprocessed NIfTI files to exclude, preventing their timeseries from being extracted.
            Used if there are specific runs across different participants that need to be
            excluded.

        Note
        ----
        This method stores the extracted timeseries as a nested dictionary and stores it in ``self.subject_timeseries``.
        The first level of the nested dictionary consists of the subject ID as a string, the second level consists of
        the run numbers in the form of "run-#" (where # is the corresponding number of the run), and the last level
        consist of the timeseries (as a numpy array) associated with that run:
        ::

            subject_timeseries = {
                    "101": {
                        "run-0": np.array([timeseries]), # 2D array
                        "run-1": np.array([timeseries]), # 2D array
                        "run-2": np.array([timeseries]), # 2D array
                    },
                    "102": {
                        "run-0": np.array([timeseries]), # 2D array
                        "run-1": np.array([timeseries]), # 2D array
                    }
                }

        Additionally, if your files do not specify a run number due to your subjects only having a single run, the run
        id key for the second level of the nested dictionary defaults to "run-0".

        When extracting specific conditions, this function uses ``math.ceil`` when calculating the ending scan of a
        condition to round up and ``int`` to round down for the onset. This is to allow for partial scans. Any
        overlapping/duplicate TRs are removed using set then are sorted. Python uses 0-based indexing so the scan
        number corresponds to the index in the timeseries (i.e onset of 0 corresponds to the 0th index/row in the
        timeseries). Below is the general code to extract the condition indices from the timeseries. There are
        several checks, such as skipping timeseries extraction when an entire run is flagged due to framewise
        displacement or having zero condition indices, that are not shown below.
        ::

            event_df = pd.read_csv(event_file[0], sep="\t")
            # Get specific timing information for specific condition
            condition_df = event_df[event_df["trial_type"] == condition]

            # Convert times into scan numbers to obtain the scans taken when the participant was exposed to the
            # condition of interest; include partial scans
            for i in condition_df.index:
                onset_scan = int(condition_df.loc[i,"onset"]/tr)
                end_scan = math.ceil((condition_df.loc[i,"onset"] + condition_df.loc[i,"duration"])/tr)
                # Add one since range is not inclusive
                scan_list.extend(list(range(onset_scan, end_scan + 1)))

            # Get unique scans to not duplicate information
            scan_list = sorted(list(set(scan_list)))

            # Adjust for dummy before censoring since censoring is dummy adjusted above
            if dummy_scans: scan_list = [scan - dummy_scans for scan in scan_list if scan not in range(0, dummy_scans)]

            if censor:
                # Get length of scan list prior to assess outliers if requested
                before_censor = len(scan_list)
                scan_list = [volume for volume in scan_list if volume not in censor_volumes]
                if outlier_limit:
                    percentage = 1 - (len(scan_list)/before_censor)
                    flagged = True if percentage > outlier_limit else False

            timeseries = timeseries[scan_list,:]

        """
        try:
            import bids
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "This function relies on the pybids package to query subject-specific files. "
                "If on Windows, pybids does not install by default to avoid long path error issues "
                "during installation. Try using `pip install pybids` or `pip install neurocaps[windows]`.")

        if runs and not isinstance(runs,list): runs = [runs]

        # Update attributes
        self._task_info = {"task": task, "condition": condition, "session": session, "runs": runs, "tr": tr}

        # Initialize new attributes
        self._subject_ids = []
        self._subject_timeseries = {}
        self._subject_info = {}
        self._n_cores = n_cores

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
        subj_ids = sorted(layout.get(return_type="id", target="subject", task=task, space=self._space, suffix="bold"))

        if exclude_subjects:
            subj_ids = sorted([subj_id for subj_id in subj_ids if subj_id not in map(str, exclude_subjects)])

        if run_subjects:
            subj_ids = sorted([subj_id for subj_id in subj_ids if subj_id in map(str, run_subjects)])

        # Setup extraction
        self._setup_extraction(layout=layout, subj_ids=subj_ids, exclude_niftis=exclude_niftis)

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
                          flush) for subj_id in self._subject_ids]

            parallel = Parallel(return_as="generator", n_jobs=self._n_cores)
            outputs = parallel(delayed(_extract_timeseries)(*args) for args in args_list)

            for output in outputs:
                if isinstance(output, dict): self._subject_timeseries.update(output)
        else:
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

    # Get valid subjects to iterate through
    def _setup_extraction(self, layout, subj_ids, exclude_niftis):
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

            if msg: LG.warning(subject_header + msg)
            if skip: continue

            # Ensure only a single session is present if session is None
            if not self._task_info["session"]: self._check_sess(files["niftis"], subject_header)

            # Generate a list of runs to iterate through based on runs in niftis
            check_runs = self._gen_runs(files["niftis"])

            if check_runs:
                run_list = self._intersect_runs(check_runs, files)

                # Skip subject if no run has all needed files present
                if not run_list:
                    LG.warning(f"{subject_header}"
                               "Timeseries Extraction Skipped: None of the necessary files (i.e NifTIs, masks, "
                               "confound tsv files, confound json files, event files) are from the same run.")
                    continue

                if len(run_list) != len(check_runs):
                    LG.warning(f"{subject_header}"
                               "Only the following runs available contain all required files: "
                               f"{', '.join(run_list)}.")
            else:
            # Allows for nifti files that do not have the run- description
                run_list = [None]

            # Add subject list to subject attribute. These are subjects that will be ran
            self._subject_ids.append(subj_id)

            # Get repetition time for the subject
            tr = self._get_tr(files["bold_meta"], subject_header)

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
        sub_message = f'[SUBJECT: {subj_id} | SESSION: {self._task_info["session"]} | TASK: {self.task_info["task"]}]'
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

    def _get_tr(self, bold_meta, subject_header):
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
            elif any([self.signal_clean_info["masker_init"]["high_pass"], self.signal_clean_info["masker_init"]["low_pass"]]):
                raise ValueError(f"{subject_header}"
                                 f"{base_msg}" + " The `tr` must be given when `high_pass` or `low_pass` is specified.")
            else:
                LG.warning(f"{subject_header}"
                           f"{base_msg}" + " `tr` has been set to None but extraction will continue.")
                tr=None

        return tr

    def timeseries_to_pickle(self, output_dir: Union[str, os.PathLike], file_name: Optional[str]=None) -> None:
        """
        **Save the Extracted Subject Timeseries**

        Saves the extracted timeseries stored in the ``self.subject_timeseries`` dictionary (obtained from running
        ``self.get_bold()``) as a pickle file. This allows for data persistence and easy conversion back into
        dictionary form for later use.

        Parameters
        ----------
        output_dir : :obj:`os.PathLike`
            Directory to save to. The directory will be created if it does not exist.

        file_name : :obj:`str` or :obj:`None`, default=None
            Name of the file with or without the "pkl" extension.
        """
        if not hasattr(self, "_subject_timeseries"):
            raise AttributeError("Cannot save pickle file since `self._subject_timeseries` does not exist, either run "
                                 "`self.get_bold()` or assign a valid timeseries dictionary to `self.subject_timeseries`.")

        if output_dir:
            if not os.path.exists(output_dir): os.makedirs(output_dir)

        if file_name is None: save_file_name = "subject_timeseries.pkl"
        else: save_file_name = f"{os.path.splitext(file_name.rstrip())[0].rstrip()}.pkl"

        with open(os.path.join(output_dir,save_file_name), "wb") as f:
            dump(self._subject_timeseries,f)

    def visualize_bold(self,
                       subj_id: Union[int,str],
                       run: Union[int, str],
                       roi_indx: Optional[Union[Union[int,str], Union[list[str],list[int]]]]=None,
                       region: Optional[str]=None,
                       show_figs: bool=True,
                       output_dir: Optional[Union[str, os.PathLike]]=None,
                       file_name: Optional[str]=None,
                       **kwargs) -> plt.figure:
        """
        **Plot the Extracted Subject Timeseries**

        Uses the ``self.subject_timeseries`` to visualize the extracted BOLD timeseries data of  data Regions of
        Interest (ROIs) or regions for a specific subject and run.

        Parameters
        ----------
        subj_id : :obj:`str` or :obj:`int`
            The ID of the subject.

        run : :obj:`int` or :obj:`str`
            The run ID of the subject to plot.

        roi_indx : :obj:`int`, :obj:`str`, :obj:`list[int]`, :obj:`list[int]` or :obj:`None`, default=None
            The indices of the parcellation nodes to plot. See "nodes" in ``self.parcel_approach`` for valid
            nodes.

        region : :obj:`str` or :obj:`None`, default=None
            The region of the parcellation to plot. If not None, all nodes in the specified region will be averaged
            then plotted. See "regions" in ``self.parcel_approach`` for valid region.

        show_figs : :obj:`bool`, default=True
            Whether to show the figures.

        output_dir : :obj:`os.PathLike` or :obj:`None`, default=None
            Directory to save to. The directory will be created if it does not exist. Outputs as png file.

        file_name : :obj:`str` or :obj:`None`, default=None
            Name of the file without the extension.

        kwargs : :obj:`dict`
            Keyword arguments used when saving figures. Valid keywords include:

            - dpi : :obj:`int`, default=300
                Dots per inch for the figure. Default is 300 if ``output_dir`` is provided and ``dpi`` is not
                specified.
            - figsize : :obj:`tuple`, default=(11, 5)
                Size of the figure in inches. Default is (11, 5) if ``figsize`` is not specified.
            - bbox_inches : :obj:`str` or :obj:`None`, default="tight"
                Alters size of the whitespace in the saved image.

        Returns
        -------
        `matplotlib.Figure`
            An instance of `matplotlib.Figure`.

        Note
        ----
        **If using a "Custom" parcellation approach**, the "nodes" and "regions" sub-keys are required.
        """

        if not hasattr(self, "_subject_timeseries"):
            raise AttributeError("Cannot plot bold data since `self._subject_timeseries` does not exist, either run "
                                 "`self.get_bold()` or assign a valid timeseries structure to self.subject_timeseries.")

        if roi_indx is None and region is None:
            raise ValueError("either `roi_indx` or `region` must be specified.")

        if roi_indx is not None and region is not None:
            raise ValueError("`roi_indx` and `region` can not be used simultaneously.")

        if file_name is not None and output_dir is None:
            LG.warning("`file_name` supplied but no `output_dir` specified. Files will not be saved.")

        parcellation_name = list(self._parcel_approach)[0]
        # Defaults
        defaults = {"dpi": 300,"figsize": (11,5), "bbox_inches": "tight"}

        plot_dict = _check_kwargs(defaults, **kwargs)

        # Obtain the column indices associated with the rois
        if roi_indx or roi_indx == 0:
            if isinstance(roi_indx, int):
                plot_indxs = roi_indx
            elif isinstance(roi_indx, str):
                # Check if parcellation_approach is custom
                if "Custom" in self._parcel_approach and "nodes" not in self._parcel_approach["Custom"]:
                    _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")
                plot_indxs = self._parcel_approach[parcellation_name]["nodes"].index(roi_indx)
            else:
                if all([isinstance(indx,int) for indx in roi_indx]):
                    plot_indxs = np.array(roi_indx)
                elif all([isinstance(indx,str) for indx in roi_indx]):
                    # Check if parcellation_approach is custom
                    if "Custom" in self._parcel_approach and "nodes" not in self._parcel_approach["Custom"]:
                        _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")
                    plot_indxs = np.array([self._parcel_approach[parcellation_name]["nodes"].index(index)
                                           for index in roi_indx])
                else:
                    raise ValueError("All elements in `roi_indx` need to be all strings or all integers.")
        else:
            if "Custom" in self._parcel_approach:
                if "regions" not in self._parcel_approach["Custom"]:
                    _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")
                else:
                    plot_indxs =  np.array(self._parcel_approach["Custom"]["regions"][region]["lh"] +
                                           self._parcel_approach["Custom"]["regions"][region]["rh"])
            else:
                plot_indxs = np.array([index for index, label in
                                       enumerate(self._parcel_approach[parcellation_name]["nodes"])
                                       if region in label])

        plt.figure(figsize=plot_dict["figsize"])

        timeseries = self._subject_timeseries[str(subj_id)][f"run-{run}"]

        if roi_indx or roi_indx == 0:
            plt.plot(range(1, timeseries.shape[0] + 1), timeseries[:,plot_indxs])
            if isinstance(roi_indx, (int,str)) or (isinstance(roi_indx, list) and len(roi_indx) == 1):
                if isinstance(roi_indx, int): roi_title = self._parcel_approach[parcellation_name]["nodes"][roi_indx]
                elif isinstance(roi_indx, str): roi_title = roi_indx
                else: roi_title = roi_indx[0]
                plt.title(roi_title)
        else:
            plt.plot(range(1, timeseries.shape[0] + 1), np.mean(timeseries[:,plot_indxs], axis=1))
            plt.title(region)

        plt.xlabel("TR")

        if output_dir:
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            if file_name: save_file_name = f"{os.path.splitext(file_name.rstrip())[0].rstrip()}.png"
            else: save_file_name = f'subject-{subj_id}_run-{run}_timeseries.png'
            plt.savefig(os.path.join(output_dir,save_file_name), dpi=plot_dict["dpi"],
                        bbox_inches=plot_dict["bbox_inches"])

        plt.show() if show_figs else plt.close()
