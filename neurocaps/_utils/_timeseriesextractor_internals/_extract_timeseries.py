"""Internal function to extract timeseries with or without multiprocessing"""
import copy, json, math, os, re
from dataclasses import dataclass, field
from functools import cached_property
import numpy as np, pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import index_img, load_img
from .._logger import _logger

# Data container
@dataclass
class _Data:
    parcel_approach: dict = field(default_factory=dict)
    signal_clean_info: dict = field(default_factory=dict)
    task_info: dict = field(default_factory=dict)
    tr: int = None
    verbose: bool = False

    # Run-specific attributes
    files: dict = field(default_factory=dict)
    fail_fast: bool = False
    dummy_vols: int = None
    censor_vols: list = field(default_factory=list)
    # Event condition scan indices
    scans: list = field(default_factory=list)
    # Subject header
    head: str = None
    # Percentage of volumes that exceed fd threshold
    vols_exceed_percent: float = None

    @property
    def session(self):
        return self.task_info["session"]

    @property
    def task(self):
        return self.task_info["task"]

    @property
    def condition(self):
        return self.task_info["condition"]

    @property
    def use_confounds(self):
        return self.signal_clean_info["use_confounds"]

    @property
    def fd(self):
        if isinstance(self.signal_clean_info["fd_threshold"], dict):
            return self.signal_clean_info["fd_threshold"]["threshold"]
        else:
            return self.signal_clean_info["fd_threshold"]

    @property
    def scrub_lim(self):
        if isinstance(self.signal_clean_info["fd_threshold"], dict):
            if "outlier_percentage" in self.signal_clean_info["fd_threshold"]:
                return self.signal_clean_info["fd_threshold"]["outlier_percentage"]
        else:
            return None

    @property
    def maps(self):
        return self.parcel_approach[list(self.parcel_approach)[0]]["maps"]

    @cached_property
    def confound_df(self):
        if self.files["confound"]: return pd.read_csv(self.files["confound"], sep="\t")
        else: return None

def _extract_timeseries(subj_id, prepped_files, run_list, parcel_approach, signal_clean_info, task_info, tr, verbose,
                        flush):

    # Logger inside function to give logger to each child process if parallel processing is done
    LG = _logger(__name__, flush=flush)

    # Initialize subject dictionary
    subject_timeseries = {subj_id: {}}

    for run in run_list:
        # Initialize class; placing inside run loops allows re-initialization of defaults
        Data = _Data(parcel_approach, signal_clean_info, task_info, tr, verbose)

        # Due to prior checking in _setup_extraction within TimeseriesExtractor, run_list = [None] is assumed to be a
        # single file without the run- description
        run_id = "run-0" if run is None else run
        run = run if run is not None else ""

        # Get files from specific run
        Data.files = {
            "nifti": _grab(run, prepped_files["niftis"]),
            "mask": _grab(run, prepped_files["masks"]),
            "confound": _grab(run, prepped_files["confounds"]),
            "confound_meta": _grab(run, prepped_files["confounds_metas"]),
            "event": _grab(run, prepped_files["events"])
        }

        # Base message
        Data.head = _header(Data, run_id.split("-")[-1], subj_id)

        if Data.verbose:
            LG.info(f"{Data.head}" + f"Preparing for Timeseries Extraction using "
                    f"[FILE: {os.path.basename(Data.files['nifti'])}].")

        # Get dummy volumes
        Data.dummy_vols = _get_dummy(Data, LG)

        # Assess framewise displacement
        if Data.fd and Data.files["confound"] and "framewise_displacement" in Data.confound_df.columns:
            # Get censor volumes vector, outlier_limit, and threshold
            fd_array, Data.censor_vols = _censor(Data)

            if len(Data.censor_vols) == fd_array.shape[0]:
                LG.warning(f"{Data.head}" + "Timeseries Extraction Skipped: Timeseries will be empty due to "
                            f"all volumes exceeding a framewise displacement of {Data.fd}.")
                continue

            # Check if run fails fast due to percentage volumes exceeding user-specified scrub_lim
            if Data.scrub_lim and not Data.condition:
                Data.vols_exceed_percent, Data.fail_fast = _flag(len(Data.censor_vols), len(fd_array),
                                                                    Data.scrub_lim)

        elif Data.fd and Data.files["confound"] and "framewise_displacement" not in Data.confound_df.columns:
            LG.warning(f"{Data.head}" + "`fd_threshold` specified but 'framewise_displacement' column not "
                        "found in the confound tsv file. Removal of volumes after nuisance regression will not be "
                        "but timeseries extraction will continue.")

        # Get events
        if Data.files['event']:
            event_df = pd.read_csv(Data.files['event'], sep="\t")
            # Get specific timing information for specific condition
            condition_df = event_df[event_df["trial_type"] == Data.condition]

            if condition_df.empty:
                LG.warning(f"{Data.head}" + f"[CONDITION: {Data.condition}] Timeseries Extraction Skipped: The "
                           "requested condition does not exist in the 'trial_type' column of the event file.")
                continue

            # Get condition indices
            Data.scans = _get_condition(Data, condition_df)

            if Data.censor_vols:
                Data.scans, n = _filter_condition(Data, fd_array, LG)
                if Data.scrub_lim:
                    Data.vols_exceed_percent, Data.fail_fast = _flag(n - len(Data.scans), n, Data.scrub_lim)

            if not Data.scans:
                LG.warning(f"{Data.head}" + f"[CONDITION: {Data.condition}] Timeseries Extraction Skipped: Timeseries "
                           "will be empty when filtered to only include volumes from the requested condition. Possibly "
                           "due to the TRs corresponding to the condition being removed by `dummy_scans` or filtered "
                           "due to exceeding threshold for `fd_threshold`.")
                continue

        if Data.fail_fast:
            percent_msg = f"Percentage of volumes exceeding the threshold limit is {Data.vols_exceed_percent*100}%"
            if Data.condition: percent_msg = f"{percent_msg} for [CONDITION: {Data.condition}]"
            LG.warning(f"{Data.head}" + f"Timeseries Extraction Skipped: Run flagged due to more than "
                       f"{Data.scrub_lim*100}% of the volumes exceeding the framewise displacement threshold of "
                       f"{Data.fd}. {percent_msg}.")
            continue

        # Continue extraction if the continue keyword isn't hit when assessing the fd threshold or events
        timeseries = _continue_extraction(Data, LG)

        if timeseries.shape[0] == 0:
            LG.warning(f"{Data.head}" + f"Timeseries is empty and will not be appended to the "
                       "`subject_timeseries` dictionary.")
        else:
            subject_timeseries[subj_id].update({run_id: timeseries})

    if not subject_timeseries[subj_id]:
        LG.warning(f"{Data.head.split(' | RUN:')[0]}] Timeseries Extraction Skipped: No runs were extracted.")
        subject_timeseries = None

    return subject_timeseries

def _continue_extraction(Data, LG):
    # Extract confound information of interest and ensure confound file does not contain NAs
    if Data.use_confounds:

        confound_names = copy.deepcopy(Data.signal_clean_info["confound_names"])

        # Extract first "n" numbers of specified WM and CSF components
        if Data.files["confound_meta"]:
            confound_names = _acompcor(confound_names,
                                       Data.files["confound_meta"],
                                       Data.signal_clean_info["n_acompcor_separate"])

        # Get confounds
        confounds = _get_confounds(Data, confound_names, LG)

    # Create the masker for extracting time series
    masker = NiftiLabelsMasker(
        mask_img=Data.files["mask"],
        labels_img=Data.maps,
        resampling_target='data',
        t_r=Data.tr,
        **Data.signal_clean_info["masker_init"]
    )

    # Load and discard volumes if needed
    nifti_img = load_img(Data.files["nifti"])

    if Data.dummy_vols:
        nifti_img = index_img(nifti_img, slice(Data.dummy_vols, None))
        if Data.use_confounds:
            confounds.drop(list(range(0, Data.dummy_vols)), axis=0, inplace=True)

    # Extract timeseries
    if Data.use_confounds: timeseries = masker.fit_transform(nifti_img, confounds=confounds)
    else: timeseries = masker.fit_transform(nifti_img)

    if Data.censor_vols and not Data.condition: timeseries = np.delete(timeseries, Data.censor_vols, axis=0)

    if Data.condition:
        # If any out of bound indices, remove them instead of getting indexing error.
        if max(Data.scans) > timeseries.shape[0] - 1:
            Data.scans = _check_indices(Data, timeseries.shape[0], LG)
        # Extract condition
        timeseries = timeseries[Data.scans,:]

    return timeseries

# Grab files
def _grab(run, files):
    if files: return [file for file in files if run in os.path.basename(file)][0]
    else: return None

# Create subject header
def _header(Data, run_id, subj_id):
    if Data.session:
        sess_id = Data.session
    else:
        base_filename = os.path.basename(Data.files["nifti"])
        sess = re.search("ses-(\\S+?)[-_]", base_filename)[0][:-1] if "ses-" in base_filename else None
        sess_id = sess.split("-")[-1] if sess else None

    sub_head = f'[SUBJECT: {subj_id} | SESSION: {sess_id} | TASK: {Data.task} | RUN: {run_id}]'

    return f"{sub_head} "

# Get dummy scan number
def _get_dummy(Data, LG):
    info = Data.signal_clean_info["dummy_scans"]
    if isinstance(info, dict) and info["auto"] is True:
        n, flag = len([col for col in Data.confound_df.columns if "non_steady_state" in col]), "auto"
        if "min" in info and n < info["min"]: n, flag = info["min"], "min"
        if "max" in info and n > info["max"]: n, flag = info["max"], "max"
        if n == 0: n = None

        if Data.verbose:
            if flag == "auto":
                if n:
                    LG.info(f"{Data.head}" + "Number of dummy scans to be removed based on "
                            f"'non_steady_state_outlier_XX' columns: {n}.")
                else:
                    LG.info(f"{Data.head}" + "No 'non_steady_state_outlier_XX' columns were found so 0 "
                            "dummy scans will be removed.")
            else:
                LG.info(f"{Data.head}" + f"Default dummy scans set by '{flag}' will be used: {n}.")

    return n if isinstance(info, dict) else info

# Create censor vector
def _censor(Data):
    fd_array = Data.confound_df["framewise_displacement"].fillna(0).values
    # Truncate fd_array if dummy scans
    if Data.dummy_vols: fd_array = fd_array[Data.dummy_vols:]

    censor_volumes = list(np.where(fd_array > Data.fd)[0])

    return fd_array, censor_volumes

# Determine if run fails fast; when condition is not specified
def _flag(n_censor, n, scrub_lim):
    vols_exceed_percent = n_censor/n
    fail_fast = True if vols_exceed_percent > scrub_lim else False

    return vols_exceed_percent, fail_fast

# Get event condition
def _get_condition(Data, condition_df):
    scans = []
    # Convert times into scan numbers to obtain the scans taken when the participant was exposed to the
    # condition of interest; include partial scans
    for i in condition_df.index:
        onset_scan = int(condition_df.loc[i,"onset"]/Data.tr)
        end_scan = math.ceil((condition_df.loc[i,"onset"] + condition_df.loc[i,"duration"])/Data.tr)
        # Add one since range is not inclusive
        scans.extend(list(range(onset_scan, end_scan + 1)))

    # Get unique scans to not duplicate information
    scans = sorted(list(set(scans)))

    # Adjust for dummy
    if Data.dummy_vols:
        scans = [scan - Data.dummy_vols for scan in scans if scan not in range(0, Data.dummy_vols)]

    return scans

def _filter_condition(Data, fd_array, LG):
    # New `scans` list created if conditions met
    scans = Data.scans
    # Assess if any condition indices greater than fd array to not dilute outlier calculation
    if max(scans) > fd_array.shape[0] - 1:
        scans = _check_indices(Data, fd_array.shape[0], LG)
    # Get length of scan list prior to assess outliers if requested
    n_before_censor = len(scans)
    scans = [volume for volume in scans if volume not in Data.censor_vols]

    return scans, n_before_censor

# Extract first "n" numbers of specified WM and CSF components
def _acompcor(confound_names, confound_metadata_file, n):
    with open(confound_metadata_file, "r") as confounds_json:
        confound_metadata = json.load(confounds_json)

    acompcors = sorted([acompcor for acompcor in confound_metadata if "a_comp_cor" in acompcor])

    CSF = [CSF for CSF in acompcors if confound_metadata[CSF]["Mask"] == "CSF"][0:n]
    WM = [WM for WM in acompcors if confound_metadata[WM]["Mask"] == "WM"][0:n]

    confound_names.extend(CSF + WM)

    return confound_names

# Get confounds
def _get_confounds(Data, confound_names, LG):
    valid_confounds = []
    invalid_confounds = []
    for confound_name in confound_names:
        if "*" in confound_name:
            prefix = confound_name.split("*")[0]
            confounds_list = [col for col in Data.confound_df.columns if col.startswith(prefix)]
        else:
            confounds_list = [col for col in Data.confound_df.columns if col == confound_name]

        if confounds_list: valid_confounds.extend(confounds_list)
        else: invalid_confounds.extend([confound_name])

    # First index of some variables is na
    if valid_confounds: confounds = Data.confound_df[valid_confounds].fillna(0)
    else: confounds = None

    if invalid_confounds and Data.verbose:
        LG.info(f"{Data.head}" + f"The following confounds were not found: {', '.join(invalid_confounds)}.")

    if not confounds.empty and Data.verbose:
        LG.info(f"{Data.head}" + "The following confounds will be used for nuisance regression: "
                f"{', '.join(list(confounds.columns))}.")

    return confounds

# Check if indices valid
def _check_indices(Data, arr_shape, LG):
    LG.warning(f"{Data.head}" + f"[CONDITION: {Data.condition}] Max scan index for exceeds timeseries max index. "
               f"Max condition index is {max(Data.scans)}, while max timeseries index is {arr_shape - 1}. Timing may "
               "be misaligned or specified repetition time incorrect. If intentional, ignore warning. Extracting "
               "indices for condition only within timeseries range.")

    return [scan for scan in Data.scans if scan in range(0, arr_shape)]
