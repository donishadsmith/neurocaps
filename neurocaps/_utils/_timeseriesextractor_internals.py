def _extract_timeseries(subj_id, nifti_files, mask_files, event_files, confound_files, confound_metadata_files, run_list, tr, condition, parcel_approach, signal_clean_info):

    from nilearn.maskers import NiftiLabelsMasker
    from nilearn.image import index_img, load_img
    import pandas as pd, warnings, json, math

    # Intitialize dictionary; Current subject will alway be the last subject in the subjects attribute
    subject_timeseries = {subj_id: {}}
    
    for run in run_list:

        # Get files from specific run
        nifti_file = [nifti_file for nifti_file in nifti_files if run in nifti_file]
        mask_file = [mask_file for mask_file in mask_files if run in mask_file]
        confound_file = [confound_file for confound_file in confound_files if run in confound_file] if signal_clean_info["use_confounds"] else None
        confound_metadata_file = [confound_metadata_file for confound_metadata_file in confound_metadata_files if run in confound_metadata_file] if signal_clean_info["use_confounds"] and signal_clean_info["n_acompcor_separate"] else None

        print(f"Running subject: {subj_id}; {run}; \n {nifti_file}")

        if len(nifti_file) == 0 or len(mask_file) == 0:
            warnings.warn(f"Skipping subject: {subj_id}; {run} do to missing nifti or mask file.")
            continue
        
        if signal_clean_info["use_confounds"]:
            if len(confound_file) == 0:
                warnings.warn(f"Skipping subject: {subj_id}; {run} do to missing confound file.")
                continue
            if len(confound_metadata_file) == 0 and signal_clean_info["n_acompcor_separate"]:
                warnings.warn(f"Skipping subject: {subj_id}; {run} do to missing confound metadata files to locate the first six components of the white-matter and cerobrospinal fluid masks seperately.")
                continue

        confound_df = pd.read_csv(confound_file[0], sep=None) if signal_clean_info["use_confounds"] else None

        event_file = None if len(event_files) == 0 else [event_file for event_file in event_files if run in event_file]

        # Extract confound information of interest and ensure confound file does not contain NAs
        if signal_clean_info["use_confounds"]:
            # Extract first "n" numbers of specified WM and CSF components
            if confound_metadata_file:
                with open(confound_metadata_file[0]) as foo:
                    confound_metadata = json.load(foo)
            
                acompcors = sorted([acompcor for acompcor in confound_metadata.keys() if "a_comp_cor" in acompcor])

                acompcors_CSF = [acompcor_CSF for acompcor_CSF in acompcors if confound_metadata[acompcor_CSF]["Mask"] == "CSF"][0:signal_clean_info["n_acompcor_separate"]]
                acompcor_WM = [acompcor_WM for acompcor_WM in acompcors if confound_metadata[acompcor_WM]["Mask"] == "WM"][0:signal_clean_info["n_acompcor_separate"]]
                
                signal_clean_info["confound_names"].extend(acompcors_CSF + acompcor_WM)

            valid_confounds = []
            invalid_confounds = []
            for confound_name in signal_clean_info["confound_names"]:
                if "*" in confound_name:
                    prefix = confound_name.split("*")[0]
                    confounds_list = [col for col in confound_df.columns if col.startswith(prefix)]
                else:
                    confounds_list = [col for col in confound_df.columns if col == confound_name] 
            
                if len(confounds_list) > 0: valid_confounds.extend(confounds_list)
                else: invalid_confounds.extend([confound_name])

            if len(invalid_confounds) > 0: print(f"Subject {subj_id} did not have the following confounds: {invalid_confounds}")

            confounds = confound_df[valid_confounds]
            confounds = confounds.fillna(0)

        # Create the masker for extracting time series
        masker = NiftiLabelsMasker(
            mask_img=mask_file[0],
            labels_img=parcel_approach[list(parcel_approach.keys())[0]]["maps"], 
            labels=parcel_approach[list(parcel_approach.keys())[0]]["labels"], 
            resampling_target='data',
            standardize=signal_clean_info["standardize"],
            detrend=signal_clean_info["detrend"],
            low_pass=signal_clean_info["low_pass"],
            high_pass=signal_clean_info["high_pass"],
            t_r=tr
        )
        # Load and discard volumes if needed
        nifti_img = load_img(nifti_file[0])
        if signal_clean_info["dummy_scans"]: 
            nifti_img = index_img(nifti_img, slice(signal_clean_info["dummy_scans"], None))
            if signal_clean_info["use_confounds"]: confounds.drop(list(range(0,signal_clean_info["dummy_scans"])),axis=0,inplace=True)

        # Extract timeseries
        timeseries = masker.fit_transform(nifti_img, confounds=confounds) if signal_clean_info["use_confounds"] else masker.fit_transform(nifti_img)

        if event_file:
            event_df = pd.read_csv(event_file[0], sep=None)
            # Get specific timing information for specific condition
            condition_df = event_df[event_df["trial_type"] == condition] if condition else event_df

            # Empty list for scans
            scan_list = []

            # Convert times into scan numbers to obtain the scans taken when the participant was exposed to the condition of interest; include partial scans
            for i in condition_df.index:
                onset_scan, duration_scan = int(condition_df.loc[i,"onset"]/tr), math.ceil((condition_df.loc[i,"onset"] + condition_df.loc[i,"duration"])/tr)
                scan_list.extend(range(onset_scan, duration_scan + 1))

            # Timeseries with the extracted scans corresponding to condition; set is used to remove overlapping TRs    
            timeseries = timeseries[sorted(list(set(scan_list))),:]

        subject_timeseries[subj_id].update({run: timeseries})

    return subject_timeseries