"""Internal function to extract timeseries with or without multiprocessing"""

import copy, json, math, os, textwrap, warnings
import numpy as np, pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import index_img, load_img

def _extract_timeseries(subj_id, nifti_files, mask_files, event_files, confound_files, confound_metadata_files,
                        run_list, tr, condition, parcel_approach, signal_clean_info, verbose, flush_print):

    # Initialize subject dictionary
    subject_timeseries = {subj_id: {}}

    for run in run_list:
        # Initialize flag; This is for flagging runs where the percentage of volumes exceeding the "fd_threshold" is greater than "outlier_percentage"
        flagged = False

        # Due to prior checking in _setup_extraction within TimeseriesExtractor, run_list = [None] is assumed to be a single file without the run- description
        run_id = "run-0" if run is None else run
        run = run if run is not None else ""

        # Get files from specific run
        nifti_file = [nifti_file for nifti_file in nifti_files if run in os.path.basename(nifti_file)]
        if len(mask_files) != 0:
            mask_file = [mask_file for mask_file in mask_files if run in os.path.basename(mask_file)][0]
        else:
            mask_file = None
        confound_file = [confound_file for confound_file in confound_files
                         if run in os.path.basename(confound_file)] if signal_clean_info["use_confounds"] else None

        if signal_clean_info["use_confounds"] and signal_clean_info["n_acompcor_separate"]:
            confound_metadata_file = [confound_metadata_file for confound_metadata_file
                                  in confound_metadata_files if run in os.path.basename(confound_metadata_file)]
        else: confound_metadata_file = None

        if verbose: print(f"Running subject: {subj_id}; run: {run_id}; \n {nifti_file}", flush=flush_print)

        confound_df = pd.read_csv(confound_file[0], sep="\t") if signal_clean_info["use_confounds"] else None

        event_file = None if len(event_files) == 0 else [event_file for event_file in event_files
                                                         if run in os.path.basename(event_file)]

        # Initialize variables
        scan_list, censor_volumes = [], []
        censor = False
        threshold, outlier_limit = None, None

        if signal_clean_info["use_confounds"] and signal_clean_info["fd_threshold"]:
            if "framewise_displacement" in confound_df.columns:
                censor = True
                fd_array = confound_df["framewise_displacement"].fillna(0).values
                # Truncate fd_array if dummy scans; done to only assess if the truncated array meets the "outlier_percentage" criteria
                if signal_clean_info["dummy_scans"]: fd_array = fd_array[signal_clean_info["dummy_scans"]:]

                # Check if float or dict
                if isinstance(signal_clean_info["fd_threshold"], dict):
                    threshold = signal_clean_info["fd_threshold"]["threshold"]
                    if "outlier_percentage" in signal_clean_info["fd_threshold"]:
                        outlier_limit = signal_clean_info["fd_threshold"]["outlier_percentage"]
                else:
                    threshold = signal_clean_info["fd_threshold"]

                censor_volumes.extend(list(np.where(fd_array > threshold)[0]))

                if len(censor_volumes) == fd_array.shape[0]:
                    warnings.warn(textwrap.dedent(f"""
                                                  For subject {subj_id} timeseries for {run} timeseries empty due to
                                                  the framewise displacement of all volumes exceeding {threshold}.
                                                  """))
                    continue

                if outlier_limit and condition is None:
                    # Determine if run is flagged; when condition is not specified
                    flagged = True if len(censor_volumes)/len(fd_array) > outlier_limit else False
            else:
                warnings.warn(textwrap.dedent(f"""
                                                For subject {subj_id}, `fd_threshold` specified but
                                                'framewise_displacement' is not a column in the confound dataframe so
                                                removal of volumes after nuisance regression will not be done.
                                                """))

        if event_file:
            event_df = pd.read_csv(event_file[0], sep="\t")
            # Get specific timing information for specific condition
            condition_df = event_df[event_df["trial_type"] == condition]

            if len(condition_df.index) == 0:
                warnings.warn(textwrap.dedent(f"""
                          For subject {subj_id} - {run}, condition: {condition} does not exist in the "trial_type"
                          column of the event file. Skipping timeseries extraction.
                          """))
                continue

            # Convert times into scan numbers to obtain the scans taken when the participant was exposed to the
            # condition of interest; include partial scans
            for i in condition_df.index:
                onset_scan = int(condition_df.loc[i,"onset"]/tr)
                duration_scan = math.ceil((condition_df.loc[i,"onset"] + condition_df.loc[i,"duration"])/tr)
                if signal_clean_info["dummy_scans"]:
                    scan_list.extend([scan - signal_clean_info["dummy_scans"]
                                      for scan in range(onset_scan, duration_scan + 1)
                                      if scan not in range(0, signal_clean_info["dummy_scans"])])
                else:
                    scan_list.extend(list(range(onset_scan, duration_scan + 1)))

            if censor:
                # Get length of scan list prior to assess outliers if requested
                before_censor = len(scan_list)
                scan_list = [volume for volume in scan_list if volume not in censor_volumes]
                if outlier_limit:
                    flagged = True if 1 - (len(scan_list)/before_censor) > outlier_limit else False

            if len(scan_list) == 0:
                warnings.warn(textwrap.dedent(f"""
                          Subject {subj_id} timeseries will be empty for {run} when filtered to only include volumes
                          from condition - {condition}. Most likely due to TRs corresponding to the condition being
                          removed by `dummy_scans`. Skipping timeseries extraction.
                          """))
                continue

        if flagged is True:
            warnings.warn(textwrap.dedent(f"""
                        Subject {subj_id} for {run} has been flagged because more than {outlier_limit*100}%
                        of the volumes exceeded the framewise displacement (FD) threshold limit of {threshold}.
                        Skipping timeseries extraction.
                        """))
            continue

        timeseries = _continue_extraction(subj_id=subj_id, run_id=run_id, nifti_file=nifti_file, mask_file=mask_file,
                                          confound_df=confound_df, confound_metadata_file=confound_metadata_file,
                                          condition=condition, signal_clean_info=signal_clean_info, tr=tr,
                                          parcel_approach=parcel_approach, flush_print=flush_print, verbose=verbose,
                                          censor_volumes=censor_volumes, scan_list=scan_list)

        if timeseries.shape[0] == 0:
            warnings.warn(textwrap.dedent(f"""
                          Subject {subj_id} timeseries is empty for {run}. Skipping appending run to the final
                          dictionary.
                          """))
        else:
            subject_timeseries[subj_id].update({run_id: timeseries})

    if len(subject_timeseries[subj_id]) == 0:
        warnings.warn(f"No runs for subject {subj_id} were extracted.")
        subject_timeseries = None

    return subject_timeseries

def _continue_extraction(subj_id, run_id, nifti_file, mask_file, confound_df, confound_metadata_file, signal_clean_info,
                         condition, tr, parcel_approach, flush_print, verbose, censor_volumes, scan_list):
    # Extract confound information of interest and ensure confound file does not contain NAs
    if signal_clean_info["use_confounds"]:
        # Extract first "n" numbers of specified WM and CSF components
        # acompcor extraction from fmriprepjson files implementation from Michael Riedel & JulioAPeraza at https://github.com/NBCLab/abcd_fmriprep-analysis/blob/5ffd3ba2d6e6d56344318abdd90c8f2f19a93101/analysis/rest/denoising.py#L111
        confound_names = copy.deepcopy(signal_clean_info["confound_names"])
        if confound_metadata_file:
            with open(confound_metadata_file[0], "r") as confounds_json:
                confound_metadata = json.load(confounds_json)

            acompcors = sorted([acompcor for acompcor in confound_metadata if "a_comp_cor" in acompcor])

            acompcors_CSF = [acompcor_CSF for acompcor_CSF in acompcors if
                                confound_metadata[acompcor_CSF]["Mask"] == "CSF"][0:signal_clean_info["n_acompcor_separate"]]
            acompcor_WM = [acompcor_WM for acompcor_WM in acompcors if
                            confound_metadata[acompcor_WM]["Mask"] == "WM"][0:signal_clean_info["n_acompcor_separate"]]

            confound_names.extend(acompcors_CSF + acompcor_WM)

        valid_confounds = []
        invalid_confounds = []
        for confound_name in confound_names:
            if "*" in confound_name:
                prefix = confound_name.split("*")[0]
                confounds_list = [col for col in confound_df.columns if col.startswith(prefix)]
            else:
                confounds_list = [col for col in confound_df.columns if col == confound_name]

            if len(confounds_list) > 0: valid_confounds.extend(confounds_list)
            else: invalid_confounds.extend([confound_name])

        if len(invalid_confounds) > 0:
            if verbose: print(f"Subject {subj_id} did not have the following confounds: {invalid_confounds}", flush=flush_print)

        confounds = confound_df[valid_confounds]
        confounds = confounds.fillna(0)
        if verbose: print(f"Confounds used for subject: {subj_id}; {run_id} - {confounds.columns}", flush=flush_print)

    # Create the masker for extracting time series
    masker = NiftiLabelsMasker(
        mask_img=mask_file,
        labels_img=parcel_approach[list(parcel_approach)[0]]["maps"],
        resampling_target='data',
        standardize=signal_clean_info["standardize"],
        detrend=signal_clean_info["detrend"],
        low_pass=signal_clean_info["low_pass"],
        high_pass=signal_clean_info["high_pass"],
        t_r=tr,
        smoothing_fwhm=signal_clean_info["fwhm"]
    )

    # Load and discard volumes if needed
    nifti_img = load_img(nifti_file[0])
    if signal_clean_info["dummy_scans"]:
        nifti_img = index_img(nifti_img, slice(signal_clean_info["dummy_scans"], None))
        if signal_clean_info["use_confounds"]:
            confounds.drop(list(range(0,signal_clean_info["dummy_scans"])),axis=0,inplace=True)

    # Extract timeseries
    if signal_clean_info["use_confounds"]: timeseries = masker.fit_transform(nifti_img, confounds=confounds)
    else: timeseries = masker.fit_transform(nifti_img)

    if len(censor_volumes) > 0 and condition is None:
        timeseries = np.delete(timeseries, censor_volumes, axis=0)

    if condition:
        # Extract specific condition from timeseries while removing any overlapping indices
        timeseries = timeseries[sorted(list(set(scan_list))),:]

    return timeseries
