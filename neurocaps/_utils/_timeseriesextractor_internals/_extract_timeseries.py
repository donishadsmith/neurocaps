"""Internal function to extract timeseries with or without multiprocessing"""
import copy, json, math, os, re
import numpy as np, pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import index_img, load_img
from .._logger import _logger

def _extract_timeseries(subj_id, nifti_files, mask_files, event_files, confound_files, confound_metadata_files,
                        run_list, tr, condition, parcel_approach, signal_clean_info, verbose, flush_print, task_info):

    # Logger inside function to give logger to each child process if parallel processing is done
    LG  = _logger(__name__, flush=flush_print)

    # Initialize subject dictionary
    subject_timeseries = {subj_id: {}}

    for run in run_list:
        # Initialize flag; This is for flagging runs where the percentage of volumes exceeding the "fd_threshold"
        # is greater than "outlier_percentage"
        flagged = False

        # Due to prior checking in _setup_extraction within TimeseriesExtractor, run_list = [None] is assumed to be a
        # single file without the run- description
        run_id = "run-0" if run is None else run
        run = run if run is not None else ""

        # Get files from specific run
        files = {}
        files["nifti"] = _grab(run, nifti_files)
        files["mask"] = _grab(run, mask_files)
        files["confound"] = _grab(run, confound_files)
        files["confound_meta"] = _grab(run, confound_metadata_files)
        files["event"] = _grab(run, event_files)

        confound_df = pd.read_csv(files["confound"], sep="\t") if signal_clean_info["use_confounds"] else None

        # Base message
        subject_header, sub_message = _header(task_info["session"], task_info["task"],
                                              run_id.split("-")[-1], files["nifti"], subj_id)

        if verbose:
            base_file = os.path.basename(files['nifti'])
            LG.info(f"{subject_header}" + f"Preparing for Timeseries Extraction using [FILE: {base_file}].")

        # Initialize variables
        censor_volumes, scan_list = [], []
        dummy_scans, outlier_limit = None, None

        # Check for non-steady_state if requested
        if signal_clean_info["dummy_scans"]:
            dummy_scans = _get_dummy(signal_clean_info["dummy_scans"], confound_df, verbose, subject_header,
                                     LG)

        if signal_clean_info["use_confounds"] and signal_clean_info["fd_threshold"]:
            if "framewise_displacement" in confound_df.columns:
                fd_array = confound_df["framewise_displacement"].fillna(0).values
                # Truncate fd_array if dummy scans;
                if dummy_scans: fd_array = fd_array[dummy_scans:]

                # Get censor volumes vector, outlier_limit, and threshold
                censor_volumes, outlier_limit, threshold = _censor(signal_clean_info["fd_threshold"], fd_array)

                if len(censor_volumes) == fd_array.shape[0]:
                    LG.warning(f"{subject_header}" + "Timeseries Extraction Skipped: Timeseries will be empty due to "
                               f"all volumes exceeding a framewise displacement of {threshold}.")
                    continue

                if outlier_limit and condition is None:
                    percentage, flagged = _mark(len(censor_volumes), len(fd_array), outlier_limit)

            else:
                LG.warning(f"{subject_header}" + "`fd_threshold` specified but 'framewise_displacement' column not "
                           "found in the confound tsv file. Removal of volumes after nuisance regression will not be "
                           "but timeseries extraction will continue."
                           )

        if files['event']:
            event_df = pd.read_csv(files['event'], sep="\t")
            # Get specific timing information for specific condition
            condition_df = event_df[event_df["trial_type"] == condition]

            if condition_df.empty:
                LG.warning(f"{subject_header}" + f"[CONDITION: {condition}] Timeseries Extraction Skipped: The "
                           "requested condition does not exist in the 'trial_type' column of the event file.")
                continue

            # Get condition indices
            scan_list = _get_condition(condition_df, tr, scan_list)

            # Adjust for dummy before censoring since censoring is dummy adjusted above
            if dummy_scans: scan_list = [scan - dummy_scans for scan in scan_list if scan not in range(0, dummy_scans)]

            if censor_volumes:
                scan_list, n = _filter_condition(scan_list, fd_array, condition, subject_header, censor_volumes, LG)
                if outlier_limit: percentage, flagged = _mark(n - len(scan_list), n, outlier_limit)

            if not scan_list:
                LG.warning(f"{subject_header}" + f"[CONDITION: {condition}] Timeseries Extraction Skipped: Timeseries "
                           "will be empty when filtered to only include volumes from the requested condition. Possibly "
                           "due to the TRs corresponding to the condition being removed by `dummy_scans` or filtered "
                           "due to exceeding threshold for `fd_threshold`.")
                continue

        if flagged:
            percentage_message = f"Percentage of volumes exceeding the threshold limit is {percentage*100}%"
            if condition: percentage_message = f"{percentage_message} for [CONDITION: {condition}]"
            LG.warning(f"{subject_header}" + f"Timeseries Extraction Skipped: Run flagged due to more than "
                       f"{outlier_limit*100}% of the volumes exceeding the framewise displacement threshold limit "
                       f" {threshold}. {percentage_message}")
            continue

        timeseries = _continue_extraction(files, confound_df=confound_df, condition=condition,
                                          signal_clean_info=signal_clean_info, dummy_scans=dummy_scans, tr=tr,
                                          parcel_approach=parcel_approach, LG=LG, verbose=verbose,
                                          censor_volumes=censor_volumes, scan_list=scan_list,
                                          subject_header=subject_header)

        if timeseries.shape[0] == 0:
            LG.warning(f"{subject_header}" + f"Timeseries is empty and will not be appended to the "
                       "`subject_timeseries` dictionary.")
        else:
            subject_timeseries[subj_id].update({run_id: timeseries})

    if not subject_timeseries[subj_id]:
        LG.warning(f"{sub_message.split(' | RUN:')[0]}] Timeseries Extraction Skipped: No runs were extracted.")
        subject_timeseries = None

    return subject_timeseries

def _continue_extraction(files, confound_df, signal_clean_info, dummy_scans, condition, tr, parcel_approach,
                         LG, verbose, censor_volumes, scan_list, subject_header):
    # Extract confound information of interest and ensure confound file does not contain NAs
    if signal_clean_info["use_confounds"]:

        confound_names = copy.deepcopy(signal_clean_info["confound_names"])

        # Extract first "n" numbers of specified WM and CSF components
        if files["confound_meta"]:
            confound_names = _acompcor(confound_names,
                                       files["confound_meta"],
                                       signal_clean_info["n_acompcor_separate"])

        # Get confounds
        confounds = _get_confounds(confound_names, confound_df, verbose, subject_header, LG)

    # Create the masker for extracting time series
    masker = NiftiLabelsMasker(
        mask_img=files["mask"],
        labels_img=parcel_approach[list(parcel_approach)[0]]["maps"],
        resampling_target='data',
        t_r=tr,
        **signal_clean_info["masker_init"]
    )

    # Load and discard volumes if needed
    nifti_img = load_img(files["nifti"])
    if dummy_scans:
        nifti_img = index_img(nifti_img, slice(dummy_scans, None))
        if signal_clean_info["use_confounds"]:
            confounds.drop(list(range(0, dummy_scans)), axis=0, inplace=True)

    # Extract timeseries
    if signal_clean_info["use_confounds"]: timeseries = masker.fit_transform(nifti_img, confounds=confounds)
    else: timeseries = masker.fit_transform(nifti_img)

    if censor_volumes and not condition: timeseries = np.delete(timeseries, censor_volumes, axis=0)

    if condition:
        # If any out of bound indices, remove them instead of getting indexing error.
        if max(scan_list) > timeseries.shape[0] - 1:
            scan_list = _check_indices(scan_list, timeseries.shape[0], condition, subject_header, LG)
        # Extract condition
        timeseries = timeseries[scan_list,:]

    return timeseries

# Grab files
def _grab(run, files):
    if files: return [file for file in files if run in os.path.basename(file)][0]
    else: return None

# Create message header
def _header(session, task, run_message, nifti_file, subj_id):
    if session:
        session_message = session
    else:
        base_filename = os.path.basename(nifti_file)
        session_id = re.search("ses-(\\S+?)[-_]", base_filename)[0][:-1] if "ses-" in base_filename else None
        session_message = session_id.split("-")[-1] if session_id else None
    sub_message = f'[SUBJECT: {subj_id} | SESSION: {session_message} | TASK: {task} | RUN: {run_message}]'

    return f"{sub_message} ", sub_message

# Get dummy scan number
def _get_dummy(info, confound_df, verbose, subject_header, LG):
    if isinstance(info, dict) and info["auto"] is True:
        n, flag = len([col for col in confound_df.columns if "non_steady_state" in col]), "auto"
        if "min" in info and n < info["min"]: n, flag = info["min"], "min"
        if "max" in info and n > info["max"]: n, flag = info["max"], "max"
        if n == 0: n = None

        if verbose:
            if flag == "auto":
                if n:
                    LG.info(f"{subject_header}" + "Number of dummy scans to be removed based on "
                            f"'non_steady_state_outlier_XX' columns: {n}.")
                else:
                    LG.info(f"{subject_header}" + "No 'non_steady_state_outlier_XX' columns were found so 0 "
                            "dummy scans will be removed.")
            else:
                LG.info(f"{subject_header}" + f"Default dummy scans set by '{flag}' will be used: {n}")

    return n if isinstance(info, dict) else info

# Create censor vector
def _censor(info, fd_array):
    outlier_limit, threshold = None, None

    if isinstance(info, dict):
        threshold = info["threshold"]
        if "outlier_percentage" in info: outlier_limit = info["outlier_percentage"]
    else:
        threshold = info

    censor_volumes = list(np.where(fd_array > threshold)[0])

    return censor_volumes, outlier_limit, threshold

# Determine if run is flagged; when condition is not specified
def _mark(n_censor, n, outlier_limit):
    percentage = n_censor/n
    flagged = True if percentage > outlier_limit else False

    return percentage, flagged

# Get event condition
def _get_condition(condition_df, tr, scan_list):
    # Convert times into scan numbers to obtain the scans taken when the participant was exposed to the
    # condition of interest; include partial scans
    for i in condition_df.index:
        onset_scan = int(condition_df.loc[i,"onset"]/tr)
        end_scan = math.ceil((condition_df.loc[i,"onset"] + condition_df.loc[i,"duration"])/tr)
        # Add one since range is not inclusive
        scan_list.extend(list(range(onset_scan, end_scan + 1)))

    # Get unique scans to not duplicate information
    scan_list = sorted(list(set(scan_list)))

    return scan_list

def _filter_condition(scan_list, fd_array, condition, subject_header, censor_volumes, LG):
    # Assess if any condition indices greater than fd array to not dilute outlier calculation
    if max(scan_list) > fd_array.shape[0] - 1:
        scan_list = _check_indices(scan_list, fd_array.shape[0], condition, subject_header, LG)
    # Get length of scan list prior to assess outliers if requested
    before_censor = len(scan_list)
    scan_list = [volume for volume in scan_list if volume not in censor_volumes]

    return scan_list, before_censor

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
def _get_confounds(confound_names, confound_df, verbose, subject_header, LG):
    valid_confounds = []
    invalid_confounds = []
    for confound_name in confound_names:
        if "*" in confound_name:
            prefix = confound_name.split("*")[0]
            confounds_list = [col for col in confound_df.columns if col.startswith(prefix)]
        else:
            confounds_list = [col for col in confound_df.columns if col == confound_name]

        if confounds_list: valid_confounds.extend(confounds_list)
        else: invalid_confounds.extend([confound_name])

    if valid_confounds: confounds = confound_df[valid_confounds].fillna(0)
    else: confounds = None

    if invalid_confounds and verbose:
        LG.info(f"{subject_header}" + f"The following confounds were not found: {', '.join(invalid_confounds)}.")

    if not confounds.empty and verbose:
        LG.info(f"{subject_header}" + "The following confounds will be used for nuisance regression: "
                f"{', '.join(list(confounds.columns))}.")

    return confounds

# Check if indices valid
def _check_indices(scan_list, arr_shape, condition, subject_header, LG):
    LG.warning(f"{subject_header}" + f"[CONDITION: {condition}] Max scan index for exceeds timeseries max index. "
               f"Max condition index is {max(scan_list)}, while max timeseries index is {arr_shape - 1}. Timing may "
               "be misaligned or specified repetition time incorrect. If intentional, ignore warning. Extracting "
               "indices for condition only within timeseries range.")

    return [scan for scan in scan_list if scan in range(0,arr_shape)]
