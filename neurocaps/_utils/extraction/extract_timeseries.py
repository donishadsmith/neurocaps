"""Internal functions for extracting timeseries with or without joblib"""

import copy, json, math, os, re
from dataclasses import dataclass, field
from functools import cached_property
from typing import Union

import numpy as np, pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import index_img, load_img
from scipy.interpolate import CubicSpline

from ..logger import _logger
from ...typing import ParcelApproach

# Logger initialization to check if any user-defined loggers where created prior to package import
LG = _logger(__name__)


# Data container
@dataclass
class _Data:
    # Add defaults so no field are required when using this class for testing
    parcel_approach: ParcelApproach = field(default_factory=dict)
    signal_clean_info: dict = field(default_factory=dict)
    task_info: dict = field(default_factory=dict)
    tr: Union[int, None] = None
    verbose: bool = False
    # Run-specific attributes
    files: dict[str, str] = field(default_factory=dict)
    fail_fast: bool = False
    dummy_vols: Union[int, None] = None
    censored_frames: list[int] = field(default_factory=list)
    sample_mask: Union[np.typing.NDArray[np.bool_], None] = None
    censored_ends: list[int] = field(default_factory=list)
    max_len: Union[int, None] = None
    # Event condition scan indices and qc information for condition
    scans: list[int] = field(default_factory=list)
    n_censored_condition_indxs: int = 0
    n_interpolated_condition_indxs: int = 0
    # Subject header
    head: Union[str, None] = None
    # Percentage of volumes that exceed fd threshold
    vols_exceed_percent: Union[float, None] = None

    @property
    def session(self) -> Union[int, str, None]:
        return self.task_info["session"]

    @property
    def task(self) -> str:
        return self.task_info["task"]

    @property
    def condition(self) -> Union[str, None]:
        return self.task_info["condition"]

    @property
    def tr_shift(self) -> int:
        return self.task_info["condition_tr_shift"]

    @property
    def slice_ref(self) -> float:
        return self.task_info["slice_time_ref"]

    @property
    def use_confounds(self) -> bool:
        return self.signal_clean_info["use_confounds"]

    @property
    def confound_names(self) -> Union[list[str], None]:
        return self.signal_clean_info["confound_names"]

    @property
    def n_acompcor_separate(self) -> Union[int, None]:
        return self.signal_clean_info["n_acompcor_separate"]

    @property
    def fd_thresh(self) -> Union[float, None]:
        if isinstance(self.signal_clean_info["fd_threshold"], dict):
            return self.signal_clean_info["fd_threshold"]["threshold"]
        else:
            return self.signal_clean_info["fd_threshold"]

    @property
    def out_percent(self) -> Union[float, None]:
        if isinstance(self.signal_clean_info["fd_threshold"], dict):
            return self.signal_clean_info["fd_threshold"].get("outlier_percentage")

    @property
    def n_before(self) -> Union[int, None]:
        if isinstance(self.signal_clean_info["fd_threshold"], dict):
            return self.signal_clean_info["fd_threshold"].get("n_before")

    @property
    def n_after(self) -> Union[int, None]:
        if isinstance(self.signal_clean_info["fd_threshold"], dict):
            return self.signal_clean_info["fd_threshold"].get("n_after")

    @property
    def pass_mask_to_nilearn(self) -> Union[bool, None]:
        if isinstance(self.signal_clean_info["fd_threshold"], dict):
            return True if self.signal_clean_info["fd_threshold"].get("use_sample_mask") is True else False
        else:
            return False

    @property
    def interpolate(self) -> Union[bool, None]:
        if isinstance(self.signal_clean_info["fd_threshold"], dict):
            return self.signal_clean_info["fd_threshold"].get("interpolate")

    @cached_property
    def confound_df(self) -> Union[pd.DataFrame, None]:
        if self.files["confound"]:
            return pd.read_csv(self.files["confound"], sep="\t")
        else:
            return None

    @property
    def maps(self) -> str:
        return self.parcel_approach[list(self.parcel_approach)[0]]["maps"]


# Entry point for timeseries extraction
def _extract_timeseries(
    subj_id,
    prepped_files,
    run_list,
    parcel_approach,
    signal_clean_info,
    task_info,
    tr,
    verbose,
    flush,
    parallel_log_config=None,
):

    # Logger inside function to give logger to each child process if parallel processing is done
    LG = _logger(__name__, flush=flush, top_level=False, parallel_log_config=parallel_log_config)
    # Initialize subject dictionary and quality control dictionary
    subject_timeseries = {subj_id: {}}
    qc = {subj_id: {}}

    for run in run_list:
        # Initialize class; placing inside run loops allows re-initialization of defaults
        data = _Data(parcel_approach, signal_clean_info, task_info, tr, verbose)

        # Due to prior checking in _setup_extraction within TimeseriesExtractor, run_list = [None] is assumed to be a
        # single file without the run- description
        run_id = "run-0" if run is None else run

        # Get files from specific run; Presence of confound metadata depends on if separate acompcor components requested
        data.files = {
            "nifti": _grab_file(run, prepped_files["niftis"]),
            "confound": _grab_file(run, prepped_files["confounds"]),
            "confound_meta": _grab_file(
                run, prepped_files["confound_metas"] if prepped_files.get("confound_metas") else None
            ),
            "event": _grab_file(run, prepped_files["events"]),
        }

        # Base message
        data.head = _subject_header(data, run_id.split("-")[-1], subj_id)

        if data.verbose:
            LG.info(
                f"{data.head}" + f"Preparing for Timeseries Extraction using "
                f"[FILE: {os.path.basename(data.files['nifti'])}]."
            )

        # Get dummy volumes
        data.dummy_vols = _get_dummy(data, LG)

        # Assess framewise displacement
        if data.fd_thresh and data.files["confound"] and "framewise_displacement" in data.confound_df.columns:
            # Get censor volumes vector, outlier_limit, and threshold
            data.max_len, data.censored_frames = _censor(data)

            if data.censored_frames and (data.n_before or data.n_after):
                data.censored_frames = _extended_censor(data)

            if len(data.censored_frames) == data.max_len:
                LG.warning(
                    f"{data.head}" + "Timeseries Extraction Skipped: Timeseries will be empty due to "
                    f"all volumes exceeding a framewise displacement of {data.fd_thresh}."
                )
                continue

            # Create sample mask
            if data.censored_frames and (data.pass_mask_to_nilearn or data.interpolate):
                data.sample_mask = _create_sample_mask(data)

            # Check if run fails fast due to percentage volumes exceeding user-specified out_percent
            if data.out_percent and not data.condition:
                data.vols_exceed_percent, data.fail_fast = _flag(
                    len(data.censored_frames), data.max_len, data.out_percent
                )

        elif data.fd_thresh and data.files["confound"] and "framewise_displacement" not in data.confound_df.columns:
            LG.warning(
                f"{data.head}" + "`fd_threshold` specified but 'framewise_displacement' column not "
                "found in the confound tsv file. Removal of volumes after nuisance regression will not be "
                "but timeseries extraction will continue."
            )

        # Determines indxs of ends to be censored if interpolation of censored volumes are requested
        if data.interpolate:
            data.censored_ends = _get_contiguous_ends(data)

        # Get events
        if data.files["event"]:
            event_df = pd.read_csv(data.files["event"], sep="\t")
            # Get specific timing information for specific condition
            condition_df = event_df[event_df["trial_type"] == data.condition]

            if condition_df.empty:
                LG.warning(
                    f"{data.head}" + f"[CONDITION: {data.condition}] Timeseries Extraction Skipped: The "
                    "requested condition does not exist in the 'trial_type' column of the event file."
                )
                continue

            # Get condition indices; removes any scans in dummy volumes
            data.scans = _get_condition(data, condition_df)
            total_condition_frames = len(data.scans)

            if data.censored_frames:
                # Removing censored or non-interpolated scan indxs; n_censored_condition_indxs is all indxs flagged
                # regardless if indx will be interpolated
                data.scans, data.n_censored_condition_indxs, data.n_interpolated_condition_indxs = _filter_condition(
                    data, LG
                )
                if data.out_percent:
                    n = data.n_censored_condition_indxs
                    data.vols_exceed_percent, data.fail_fast = _flag(n, total_condition_frames, data.out_percent)

            if not data.scans:
                LG.warning(
                    f"{data.head}" + f"[CONDITION: {data.condition}] Timeseries Extraction Skipped: Timeseries "
                    "will be empty when filtered to only include volumes from the requested condition. Possibly "
                    "due to the TRs corresponding to the condition being removed by `dummy_scans` or filtered "
                    "due to exceeding threshold for `fd_threshold`."
                )
                continue

        if data.fail_fast:
            percent_msg = f"Percentage of volumes exceeding the threshold limit is {data.vols_exceed_percent * 100}%"
            if data.condition:
                percent_msg = f"{percent_msg} for [CONDITION: {data.condition}]"

            LG.warning(
                f"{data.head}" + f"Timeseries Extraction Skipped: Run flagged due to more than "
                f"{data.out_percent * 100}% of the volumes exceeding the framewise displacement threshold of "
                f"{data.fd_thresh}. {percent_msg}."
            )
            continue

        # Continue extraction if the continue keyword isn't hit when assessing the fd threshold or events
        timeseries = _perform_extraction(data, LG)
        if timeseries.shape[0] == 0:
            LG.warning(
                f"{data.head}" + f"Timeseries is empty and will not be appended to the "
                "`subject_timeseries` dictionary."
            )
        else:
            subject_timeseries[subj_id].update({run_id: timeseries})
            # Report framewise displacement quality control
            qc[subj_id].update({run_id: _report_qc(data)})

    if not subject_timeseries[subj_id]:
        LG.warning(f"{data.head.split(' | RUN:')[0]}] Timeseries Extraction Skipped: No runs were extracted.")
        subject_timeseries = None
        qc = None

    return subject_timeseries, qc


# Perform the timeseries extraction
def _perform_extraction(data, LG):
    # Extract confound information of interest and ensure confound file does not contain NAs
    confounds = _process_confounds(data, LG)

    # Create the masker for extracting time series; strategy="mean" is the default for NiftiLabelsMasker,
    # added to make it clear in codebase that the mean is the default strategy used for reducing regions
    masker = NiftiLabelsMasker(
        labels_img=data.maps,
        resampling_target="data",
        strategy="mean",
        t_r=data.tr,
        **data.signal_clean_info["masker_init"],
        clean__extrapolate=False,
    )

    # Load and discard volumes if needed
    nifti_img = load_img(data.files["nifti"], dtype=data.signal_clean_info["dtype"])

    # Get length of timeseries and append scan indices to ensure no scans are out of bounds due to shift
    if (data.condition and data.tr_shift > 0) and (max(data.scans) > nifti_img.shape[-1] - 1):
        data.scans = _check_indices(data, nifti_img.shape[-1], LG, False)

    if data.dummy_vols:
        nifti_img = index_img(nifti_img, slice(data.dummy_vols, None))
        if confounds is not None:
            confounds.drop(list(range(data.dummy_vols)), axis=0, inplace=True)

    # Extract timeseries; Censor mask used only when `pass_mask_to_nilearn` is True
    timeseries = masker.fit_transform(
        nifti_img,
        confounds=confounds,
        sample_mask=data.sample_mask if (data.pass_mask_to_nilearn and data.censored_frames) else None,
    )

    # Process censored timeseries
    timeseries = _process_censored_timeseries(timeseries, data) if data.censored_frames else timeseries

    if data.condition:
        # If any out of bound indices, remove them instead of getting indexing error.
        if max(data.scans) > timeseries.shape[0] - 1:
            data.scans = _check_indices(data, timeseries.shape[0], LG)

        if data.verbose:
            LG.info(data.head + f"Nuisance regression completed; extracting [CONDITION: {data.condition}].")

        # Extract condition
        timeseries = timeseries[data.scans, :]

    return timeseries


# Grab single file
def _grab_file(run, files):
    if not files:
        return None

    if run:
        return [file for file in files if f"{run}_" in os.path.basename(file)][0]
    else:
        return files[0]


# Create subject header
def _subject_header(data, run_id, subj_id):
    if data.session:
        sess_id = data.session
    else:
        base_filename = os.path.basename(data.files["nifti"])
        sess = re.search("ses-(\\S+?)[-_]", base_filename)[0][:-1] if "ses-" in base_filename else None
        sess_id = sess.split("-")[-1] if sess else None

    sub_head = f"[SUBJECT: {subj_id} | SESSION: {sess_id} | TASK: {data.task} | RUN: {run_id}]"

    return f"{sub_head} "


# Get dummy scan number
def _get_dummy(data, LG):
    info = data.signal_clean_info["dummy_scans"]

    if isinstance(info, dict) and info.get("auto"):
        # If n=0, it will simply be treated as False
        n, flag = len([col for col in data.confound_df.columns if "non_steady_state" in col]), "auto"

        if info.get("min") and n < info["min"]:
            n, flag = info["min"], "min"
        if info.get("max") and n > info["max"]:
            n, flag = info["max"], "max"

        if data.verbose:
            if flag == "auto":
                if n:
                    LG.info(
                        f"{data.head}" + "Number of dummy scans to be removed based on "
                        f"'non_steady_state_outlier_XX' columns: {n}."
                    )
                else:
                    LG.info(
                        f"{data.head}" + "No 'non_steady_state_outlier_XX' columns were found so 0 "
                        "dummy scans will be removed."
                    )
            else:
                LG.info(f"{data.head}" + f"Default dummy scans set by '{flag}' will be used: {n}.")

    return n if isinstance(info, dict) else info


# Create censor vector
def _censor(data):
    fd_array = data.confound_df["framewise_displacement"].fillna(0).values

    # Truncate fd_array if dummy scans
    if data.dummy_vols:
        fd_array = fd_array[data.dummy_vols :]

    censor_volumes = sorted(list(np.where(fd_array > data.fd_thresh)[0]))

    return fd_array.shape[0], censor_volumes


# Created an extended censored_frames vector
def _extended_censor(data):
    ext_arr = []

    for i in data.censored_frames:
        if data.n_before:
            ext_arr.extend(list(range(i - data.n_before, i)))
        if data.n_after:
            ext_arr.extend(list(range(i + 1, i + data.n_after + 1)))

    # Filter; ensure no index is below zero to prevent backwards indexing and not above max to prevent error
    filtered_ext_arr = [x for x in ext_arr if x >= 0 and x < data.max_len]

    # Return new list that is sorted and only contains unique indices
    return sorted(list(set(data.censored_frames + filtered_ext_arr)))


# Determines the number of frames that were scrubbed and interpolated
def _report_qc(data):
    qc = {"frames_scrubbed": 0, "frames_interpolated": 0}

    if not data.censored_frames:
        return qc

    n_interpolated_frames = 0
    if not data.condition:
        n_scrubbed_frames = len(data.censored_frames)
        # Subtract interpolated frames from the number of censored_ends, which don't get interpolated
        if data.interpolate:
            n_interpolated_frames = (
                n_scrubbed_frames - len(data.censored_ends) if data.censored_ends else n_scrubbed_frames
            )

        n_scrubbed_frames -= n_interpolated_frames
    else:
        n_scrubbed_frames = data.n_censored_condition_indxs
        n_interpolated_frames = data.n_interpolated_condition_indxs

    qc["frames_scrubbed"] = n_scrubbed_frames
    qc["frames_interpolated"] = n_interpolated_frames

    return qc


# Determine if run fails fast; when condition is not specified
def _flag(n_censor, n, out_percent):
    vols_exceed_percent = n_censor / n
    fail_fast = True if vols_exceed_percent > out_percent else False

    return vols_exceed_percent, fail_fast


# Get event condition
def _get_condition(data, condition_df):
    scans = []

    # Convert times into scan numbers to obtain the scans taken when the participant was exposed to the
    # condition of interest; include partial scans
    for i in condition_df.index:
        adjusted_onset = condition_df.loc[i, "onset"] - data.slice_ref * data.tr
        # Avoid accidental negative indexing
        adjusted_onset = adjusted_onset if adjusted_onset >= 0 else 0
        # Int is always the floor for positive floats
        onset_scan = int(adjusted_onset / data.tr) + data.tr_shift
        end_scan = math.ceil((adjusted_onset + condition_df.loc[i, "duration"]) / data.tr) + data.tr_shift
        scans.extend(list(range(onset_scan, end_scan)))

    # Get unique scans to not duplicate information
    scans = sorted(list(set(scans)))

    # Adjust for dummy
    if data.dummy_vols:
        scans = [scan - data.dummy_vols for scan in scans if scan not in range(data.dummy_vols)]

    return scans


def _filter_condition(data, LG):
    # New `scans` list always created; Now just add deepcopy to be explicit
    scans = copy.deepcopy(data.scans)
    # Assess if any condition indices greater than fd array to not dilute outlier calculation
    if max(scans) > data.max_len - 1:
        scans = _check_indices(data, data.max_len, LG)

    # Remove all censored scans if no interpolation else only remove censored (contiguous) ends
    scrubbing_approach = data.censored_frames if not data.interpolate else data.censored_ends
    censored_indxs = set(scrubbing_approach).intersection(scans)
    # Determine the number of frames that will be interpolated for qc report
    # Get the intersection of scan indxs and the censored_frames, then filter out indxs not in censored_ends
    n_interpolated_indxs = (
        len(set(scans).intersection(data.censored_frames).difference(data.censored_ends)) if data.interpolate else 0
    )
    scans = [indx for indx in scans if indx not in censored_indxs]
    return scans, len(censored_indxs), n_interpolated_indxs


# Extract first "n" numbers of specified WM and CSF components
def _get_separate_acompcor(data):
    confound_metadata_file = data.files["confound_meta"]
    components_list = []

    with open(confound_metadata_file, "r") as confounds_json:
        confound_metadata = json.load(confounds_json)

    acompcors = sorted([acompcor for acompcor in confound_metadata if "a_comp_cor" in acompcor])
    CSF = [CSF for CSF in acompcors if confound_metadata[CSF]["Mask"] == "CSF"][0 : data.n_acompcor_separate]
    WM = [WM for WM in acompcors if confound_metadata[WM]["Mask"] == "WM"][0 : data.n_acompcor_separate]
    components_list.extend(CSF + WM)

    return components_list


def _process_confounds(data, LG):
    if not data.use_confounds:
        return
    else:
        confound_names = copy.deepcopy(data.signal_clean_info["confound_names"]) if data.confound_names else []

        # Extract first "n" numbers of specified WM and CSF components and extend confound_names list
        components_list = []
        if data.files["confound_meta"]:
            components_list = _get_separate_acompcor(data)

        # Extend confounds
        if components_list:
            confound_names.extend(components_list)

        # Get confounds
        confounds = _extract_valid_confounds(data, confound_names, LG) if confound_names else None

    return confounds


# Extract valid confounds
def _extract_valid_confounds(data, confound_names, LG):
    valid_confounds = []
    invalid_confounds = []

    for confound_name in confound_names:
        if "*" in confound_name:
            prefix = confound_name.split("*")[0]
            confounds_list = [col for col in data.confound_df.columns if col.startswith(prefix)]
        else:
            confounds_list = [col for col in data.confound_df.columns if col == confound_name]

        if confounds_list:
            valid_confounds.extend(confounds_list)
        else:
            invalid_confounds.extend([confound_name])

    # First index of some variables is NaN
    if valid_confounds:
        confounds = data.confound_df[valid_confounds].fillna(0)
    else:
        confounds = None

    if data.verbose:
        if confounds is None:
            LG.warning(
                f"{data.head}" + "None of the requested confounds were found so nuisance regression will not "
                "be done."
            )
        elif invalid_confounds:
            LG.warning(f"{data.head}" + f"The following confounds were not found: {', '.join(invalid_confounds)}.")

        if confounds is not None and not confounds.empty:
            LG.info(
                f"{data.head}" + "The following confounds will be used for nuisance regression: "
                f"{', '.join(list(confounds.columns))}."
            )

    return confounds


# Create sample mask for censoring high motion volumes for nuisance regression or if not using for nuisance regression
# used to determine frames to be interpolated. True for volumes kept and False for volumes censored
def _create_sample_mask(data):
    sample_mask = np.ones(data.max_len, dtype="bool")
    sample_mask[np.array(data.censored_frames)] = False

    return sample_mask


# If the sample mask was passed to NiftiLabelsMasker, temporarily pad the censored rows to either filter the correct
# rows corresponding to condition or to interpolate the correct rows
def _pad_timeseries(timeseries, data):
    # Early return if the sample mask was not passed to nilearn
    if not data.pass_mask_to_nilearn:
        return timeseries

    padded_timeseries = np.zeros((data.max_len, timeseries.shape[1]))
    padded_timeseries[data.sample_mask, :] = timeseries

    return padded_timeseries


# Replaces censored frames with values from cubic spline interpolation; References nilearn's _interpolate_volumes
# approach from nilearn.signal clean and scipy's example docs
def _interpolate_censored_volumes(timeseries, data):
    # Get frame times
    frame_times = np.arange(data.max_len) * data.tr
    # Create interpolated timeseries; retain only good frame times and timeseries data
    cubic_spline = CubicSpline(frame_times[data.sample_mask], timeseries[data.sample_mask, :], extrapolate=False)
    # Only replace the high motion volumes with the interpolated data
    timeseries[~data.sample_mask, :] = cubic_spline(frame_times)[~data.sample_mask, :]

    # Remove ends if not condition, condition already removes these volumes in the data.scans list
    if not data.condition:
        timeseries = np.delete(timeseries, data.censored_ends, axis=0) if data.censored_ends else timeseries

    return timeseries


# Only retain the condition indices that contain anchors/true on both ends if interpolation requested
def _get_contiguous_ends(data):
    # If all volumes are retained, then early return
    if np.all(data.sample_mask):
        return []

    # Example sample mask: [0,0,1,1,0,0,1,1,0,0]; 1 = kept; Diff array of sample mask: [0,1,0,-1,0,1,0,-1,0]
    # Indices not 0: [1,3,5,7]; Add + 1 to each element to obtain transition indxs from sample mask: [2,4,6,8]
    split_indices = np.where(np.diff(data.sample_mask, n=1) != 0)[0] + 1
    # Split into groups of contiguous indices: ([0,1], [2,3], [4,5], [6,7], [8,9])
    contiguous_indices = np.split(np.arange(data.max_len), split_indices)
    # Check if first index in sample mask is 0
    start_indices = contiguous_indices[0].tolist() if data.sample_mask[0] == 0 else []
    # Check if last index in sample mask is 0
    end_indices = contiguous_indices[-1].tolist() if data.sample_mask[-1] == 0 else []

    return start_indices + end_indices


# Interpolating, padding for future condition extraction, or deleting censored data from timeseries
def _process_censored_timeseries(timeseries, data):
    timeseries = _pad_timeseries(timeseries, data)

    if data.interpolate:
        timeseries = _interpolate_censored_volumes(timeseries, data)
    else:
        timeseries = np.delete(timeseries, data.censored_frames, axis=0) if not data.condition else timeseries

    return timeseries


# Check if indices valid or truncate indices if shift applied and causes indices to be out of bounds
def _check_indices(data, arr_shape, LG, warn=True):
    # Warn is false in instances were shift is used to prevent unnecessary logging, the warning is specifically
    # for instances related to misalignment such as potentially incorrect onsets and durations
    if warn:
        LG.warning(
            f"{data.head}" + f"[CONDITION: {data.condition}] Max scan index exceeds timeseries max index. "
            f"Max condition index is {max(data.scans)}, while max timeseries index is {arr_shape - 1}. Timing may "
            "be misaligned or specified repetition time incorrect. If intentional, ignore warning. Only "
            "indices for condition within the timeseries range will be extracted."
        )

    return [scan for scan in data.scans if scan in range(arr_shape)]
