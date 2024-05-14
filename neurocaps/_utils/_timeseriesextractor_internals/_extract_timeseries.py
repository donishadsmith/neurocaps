def _extract_timeseries(subj_id, nifti_files, mask_files, event_files, confound_files, confound_metadata_files, run_list, tr, condition, parcel_approach, signal_clean_info, verbose, flush_print):

    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.image import index_img, load_img
    import copy, json, math, numpy as np, pandas as pd, warnings

    # Intitialize subject dictionary
    subject_timeseries = {subj_id: {}}
    
    for run in run_list:

        run_id = "run-1" if run == None else run
        run = run if run != None else ""

        # Get files from specific run
        nifti_file = [nifti_file for nifti_file in nifti_files if run in nifti_file.split("/")[-1]]
        if len(mask_files) != 0:
            mask_file = [mask_file for mask_file in mask_files if run in mask_file.split("/")[-1]]
        else:
            mask_file = []
        confound_file = [confound_file for confound_file in confound_files if run in confound_file.split("/")[-1]] if signal_clean_info["use_confounds"] else None
        confound_metadata_file = [confound_metadata_file for confound_metadata_file in confound_metadata_files if run in confound_metadata_file.split("/")[-1]] if signal_clean_info["use_confounds"] and signal_clean_info["n_acompcor_separate"] else None

        if verbose: print(f"Running subject: {subj_id}; run: {run_id}; \n {nifti_file}", flush=flush_print)

        confound_df = pd.read_csv(confound_file[0], sep="\t") if signal_clean_info["use_confounds"] else None

        event_file = None if len(event_files) == 0 else [event_file for event_file in event_files if run in event_file.split("/")[-1]]

        # Extract confound information of interest and ensure confound file does not contain NAs
        if signal_clean_info["use_confounds"]:
            # Extract first "n" numbers of specified WM and CSF components
            confound_names = copy.deepcopy(signal_clean_info["confound_names"])
            if confound_metadata_file:
                with open(confound_metadata_file[0]) as foo:
                    confound_metadata = json.load(foo)
            
                acompcors = sorted([acompcor for acompcor in confound_metadata.keys() if "a_comp_cor" in acompcor])

                acompcors_CSF = [acompcor_CSF for acompcor_CSF in acompcors if confound_metadata[acompcor_CSF]["Mask"] == "CSF"][0:signal_clean_info["n_acompcor_separate"]]
                acompcor_WM = [acompcor_WM for acompcor_WM in acompcors if confound_metadata[acompcor_WM]["Mask"] == "WM"][0:signal_clean_info["n_acompcor_separate"]]
                
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

            if signal_clean_info["fd_threshold"]: 
                censor = True
                if "framewise_displacement" in confound_df.columns:
                    fd_array = confound_df["framewise_displacement"].fillna(0).values
                else:
                    censor = False
                    warnings.warn(f"For subject {subj_id}, `fd_threshold` specified but 'framewise_displacement' is not a column in the confound dataframe so removal of volumes after nuisance regression will not be done.")
            else:
                censor = False

            if verbose: print(f"Confounds used for subject: {subj_id}; run: {run_id} - {confounds.columns}", flush=flush_print)

        # Create the masker for extracting time series
        if len(mask_file) !=0:
            masker = NiftiLabelsMasker(
                mask_img=mask_file[0],
                labels_img=parcel_approach[list(parcel_approach.keys())[0]]["maps"], 
                resampling_target='data',
                standardize=signal_clean_info["standardize"],
                detrend=signal_clean_info["detrend"],
                low_pass=signal_clean_info["low_pass"],
                high_pass=signal_clean_info["high_pass"],
                t_r=tr,
                smoothing_fwhm=signal_clean_info["fwhm"]
            )
        else:
            masker = NiftiLabelsMasker(
                labels_img=parcel_approach[list(parcel_approach.keys())[0]]["maps"], 
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
                # Truncate the fd_array also
                if censor:
                    fd_array = fd_array[signal_clean_info["dummy_scans"]:]
            offset = signal_clean_info["dummy_scans"] 

        # Extract timeseries
        timeseries = masker.fit_transform(nifti_img, confounds=confounds) if signal_clean_info["use_confounds"] else masker.fit_transform(nifti_img)

        if censor:
            censor_volumes = list(np.where(fd_array > signal_clean_info["fd_threshold"])[0])

        if event_file:
            event_df = pd.read_csv(event_file[0], sep=None)
            # Get specific timing information for specific condition
            condition_df = event_df[event_df["trial_type"] == condition] 

            # Empty list for scans
            scan_list = []

            # Convert times into scan numbers to obtain the scans taken when the participant was exposed to the condition of interest; include partial scans
            for i in condition_df.index:
                onset_scan, duration_scan = int(condition_df.loc[i,"onset"]/tr), math.ceil((condition_df.loc[i,"onset"] + condition_df.loc[i,"duration"])/tr)
                if signal_clean_info["dummy_scans"]:
                    scan_list.extend([scan - offset for scan in range(onset_scan, duration_scan + 1) if scan not in range(0, signal_clean_info["dummy_scans"])])
                else:
                    scan_list.extend(list(range(onset_scan, duration_scan + 1)))
                if censor:
                    scan_list = [volume for volume in scan_list if volume not in censor_volumes]

            # Timeseries with the extracted scans corresponding to condition; set is used to remove overlapping TRs    
            timeseries = timeseries[sorted(list(set(scan_list))),:]
        else:
            if censor:
                timeseries = np.delete(timeseries, censor_volumes, axis=0)

        if timeseries.shape[0] == 0:
            warnings.warn(f"Subject {subj_id} timeseries is empty for {run}. Most likely due to condition not existing or TRs correspoonding to the condition being removed by `dummy_scans`.")
        else:
            subject_timeseries[subj_id].update({run_id: timeseries})
    
    if len(subject_timeseries[subj_id].keys()) == 0:
        subject_timeseries = None

    return subject_timeseries