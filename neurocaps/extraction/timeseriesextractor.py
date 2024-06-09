import json, os, re, sys, warnings
from typing import Union, Optional, Dict, List
from pathlib import Path
from .._utils import _TimeseriesExtractorGetter, _check_kwargs, _check_confound_names, _check_parcel_approach, _extract_timeseries

class TimeseriesExtractor(_TimeseriesExtractorGetter):
    """
    Timeseries Extractor Class

    Initializes the TimeseriesExtractor class.

    Parameters
    ----------
        space: str, default="MNI152NLin2009cAsym"
            The standard template space that the preprocessed bold data is registered to.
        standardize: bool or str, default="zscore_sample"
            Determines whether to standardize the timeseries. Refer to Nilearn's NiftiLabelsMasker for available options.
        detrend: bool, default=True
            Detrends the timeseries during extraction.
        low_pass: float or None, default=None
            Filters out signals above the specified cutoff frequency.
        high_pass: float or None, default=None
            Filters out signals below the specified cutoff frequency.
        parcel_approach: Dict[str, Dict], default={"Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}}
            The approach to parcellate BOLD images. This should be a nested dictionary with the first key being the atlas name. Currently, only "Schaefer", "AAL", and "Custom" are supported.

            - For "Schaefer", available options include n_rois, yeo_networks, and resolution_mm. Refer to Nilearn's datasets.fetch_atlas_schaefer_2018 for valid inputs.
            - For "AAL", the only option is version. Refer to Nilearn's datasets.fetch_atlas_aal for valid inputs.
            - For "Custom", the key must include a subkey called maps specifying the directory location of the parcellation Nifti file (e.g., {"Custom": {"maps": "/location/to/parcellation.nii.gz"}}).
        use_confounds: bool, default=True
            Determines whether to use confounds when extracting timeseries.
        confound_names: List[str] or None, default=None
            Specifies the names of confounds to use from confound files. If None, default confounds are used. Note, an asterisk ("*") can be used to find confound names that start with the term preceding the asterisk.
            For instance, "cosine*" will find all confound names in the confound files starting with "cosine".
        fwhm: float or None, default=None
            Applies spatial smoothing to data (in millimeters). Note that using parcellations already averages voxels within parcel boundaries, which can improve signal-to-noise ratio (SNR) assuming Gaussian noise distribution.
            However, smoothing may also blur parcel boundaries.
        fd_threshold: float or None, default=None
            Sets a threshold to remove frames after nuisance regression and timeseries extraction. This requires a column named framewise_displacement in the confounds dataframe and use_confounds set to True.
            Additionally, `framewise_displacement` should not need be specified in confound_names if using this parameter.
        n_acompcor_separate: int or None, default=None
            Specifies the number of separate acompcor components derived from white-matter (WM) and cerebrospinal fluid (CSF) masks to use. For example, if set to 5, the first five components from the WM mask
            and the first five from the CSF mask will be used, totaling ten acompcor components. If this parameter is not None, any acompcor components listed in confound_names will be disregarded. To use acompcor
            components derived from combined masks (WM & CSF), leave this parameter as None and list the specific acompcors of interest in confound_names.
        dummy_scans: int or None, default=None
            Removes the first n volumes before extracting the timeseries.

    Notes for ``confounds_names``
    -----------------------------
    Additionally, default `confound_names` changes if `high_pass` is not None.

    If `high_pass` is not None, then the cosine parameters and acompcor parameters are not used because the cosine parameters are high-pass filters
    and the acompcor regressors are estimated on the high-pass filtered version of the data:
    ::

        confound_names = ["trans_x", "trans_x_derivative1","trans_y", "trans_y_derivative1", 
                          "trans_z", "trans_z_derivative1",  "rot_x", "rot_x_derivative1",
                          "rot_y", "rot_y_derivative1", "rot_z", "rot_z_derivative1"]

    If `high_pass` is None, then:
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

    Notes for ``parcel_approach``
    -----------------------------
    If using a "Custom" parcellation approach, ensure each node in your dataset includes both left (lh) and right (rh) hemisphere versions. This function assumes that the background label is "zero".
    Do not add a background label in the "nodes" or "networks" key; the zero index should correspond to the first ID that is not zero.

    Custom Key Structure:
    ---------------------
    - 'maps': Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g., .nii for NIfTI files). For plotting purposes, this key is not required.
    - 'nodes': List of all node labels used in your study, arranged in the exact order they correspond to indices in your parcellation files.
      Each label should match the parcellation index it represents. For example, if the parcellation label "1" corresponds to the left hemisphere
      visual cortex area 1, then "LH_Vis1" should occupy the 0th index in this list. This ensures that data extraction and analysis accurately reflect the anatomical regions intended.
      For timeseries extraction, this key is not required.
    - 'regions': Dictionary defining major brain regions. Each region should list node indices under "lh" and "rh" to specify left and right hemisphere nodes. For timeseries extraction, this key is not required.
    
    Example
    --------
    The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis) and hippocampus regions:
    ::

        parcel_approach = {"Custom": {"maps": "/location/to/parcellation.nii.gz",
                             "nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus", "RH_Vis1", "RH_Vis2", "RH_Hippocampus"],
                             "regions": {"Vis" : {"lh": [0,1],
                                                  "rh": [3,4]},
                                         "Hippocampus": {"lh": [2],
                                                         "rh": [5]}}}}
                                                         
    """
    def __init__(self, space: str = "MNI152NLin2009cAsym", standardize: Union[bool, str]="zscore_sample", detrend: bool=True, low_pass: Optional[float]=None, 
                 high_pass: Optional[float]=None, parcel_approach: Dict[str, Dict]={"Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}}, 
                 use_confounds: bool=True, confound_names: Optional[List[str]]=None, fwhm: Optional[float]=None, fd_threshold: Optional[float]=None, n_acompcor_separate: Optional[int]=None, 
                 dummy_scans: Optional[int]=None) -> None:
        
        self._space = space
        self._signal_clean_info = {"standardize": standardize, "detrend": detrend, "low_pass": low_pass, "high_pass": high_pass, "fwhm": fwhm, 
                                   "dummy_scans": dummy_scans, "use_confounds": use_confounds,  "n_acompcor_separate": n_acompcor_separate,
                                   "fd_threshold": None}   

        # Check parcel_apprach
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_approach)

        if self._signal_clean_info["use_confounds"]:
            self._signal_clean_info["confound_names"] = _check_confound_names(high_pass=high_pass, specified_confound_names=confound_names, n_acompcor_separate=n_acompcor_separate)
            self._signal_clean_info["fd_threshold"] = fd_threshold

    def get_bold(self, bids_dir: Path, task: str, session: Optional[Union[int,str]]=None, runs: Optional[List[int]]=None, condition: Optional[str]=None, tr: Optional[Union[int, float]]=None, 
                 run_subjects: Optional[List[str]]=None, exclude_subjects: Optional[List[str]]= None, pipeline_name: Optional[str]=None, n_cores: Optional[int]=None, verbose: bool=True, flush_print: bool=False,
                 exclude_niftis: Optional[List[str]]=None) -> None: 
        """Get Bold Data

        Collects files needed to extract timeseries data from NIfTI files for BIDS-compliant datasets containing a derivatives folder. This function assumes that your BOLD data was preprocessed using a standard 
        preprocessing pipeline such as fMRIPrep.

        Parameters
        ----------
        bids_dir: Path
            Path to a BIDS compliant directory. 
        task: str
            Name of task to process.
        session: int or None, default=None
            Session to extract timeseries from. Only a single session can be extracted at a time. 
        runs: List[int], default=None
            List of run numbers to extract timeseries data from. Extracts all runs if unspecified. For instance, if only "run-0" and "run-1" should be extracted, then 
            `runs=[0,1]`.
        condition: str or None, default=None
            Specific condition in the task to extract from. Only a single condition can be extracted at a time.
        tr: int or float or None, default=None
            Repetition time for task.
        run_subjects: List[str] or None, default=None
            List of subject IDs to process. Processes all subjects if None.
        exclude_subjects: List[str] or None, default=None
            List of subject IDs to exclude.  
        pipeline_name: str or None, default=None
            The name of the pipeline folder in the derivatives folder containing the preprocessed data. If None, BIDSLayout will use the name of dset_dir with derivatives=True. This parameter
            should be used if there are multiple pipelines in the derivatives folder.
        n_cores: int or None, default=None
            The number of CPU cores to use for multiprocessing with joblib.
        verbose: bool, default=True
            Print subject-specific information such as confounds being extracted, and id and run of subject being processed during timeseries extraction.
        flush_print: bool, default=False
            Flush the printed subject-specific infomation produced during the timeseries extraction process.
        exclude_niftis: List[str] or None, default=None
            List of specific preprocessed NIfTI files to exclude, preventing their timeseries from being extracted. Used if there are specific runs across differnt participants that need to be
            excluded.

        Note
        ----
        This function stores the extracted timeseries as a nested dictionary and stores it in `self.subject_timeseres`. The first level of the nested dictionary consists of the subject ID as a string, 
        the second level consists of the run numbers in the form of 'run-#' (where # is the corresponding number of the run), and the last level must consist of the timeseries 
        (as a numpy array) associated with that run.

        Additionally, if your files do not specify a run number due to your subjects only having a single run, the run id key for the second level of the nested dictionary defaults to "run-0".

        **This function cannot be used on Windows PCs.**
        """
        import bids
        from joblib import cpu_count, delayed, Parallel

        if sys.platform == "win32":
            raise SystemError("Cannot use this method on Windows devices since it relies on the `pybids` module which is only compatable with POSIX systems.")

        # Update attributes
        self._task_info = {"task": task, "condition": condition, "session": session, "runs": runs, "tr": tr}

        # Intiialize new attributes
        self._subject_ids = []
        self._subject_timeseries = {}
        self._subject_info = {}

        if pipeline_name:
            layout = bids.BIDSLayout(bids_dir, derivatives=os.path.join(bids_dir, "derivatives", pipeline_name))
        else:
            layout = bids.BIDSLayout(bids_dir, derivatives=True)

        print(f"Bids layout collected.", flush=True)
        print(layout, flush=True)
        subj_id_list = sorted(layout.get(return_type="id", target="subject", task=task, space=self._space, suffix="bold")) 

        if exclude_subjects: 
            exclude_subjects = [str(subj_id) if not isinstance(subj_id,str) else subj_id for subj_id in exclude_subjects]
            subj_id_list = sorted([subj_id for subj_id in subj_id_list if subj_id not in exclude_subjects])

        if run_subjects: 
            run_subjects = [str(subj_id) if not isinstance(subj_id,str) else subj_id for subj_id in run_subjects]
            subj_id_list = sorted([subj_id for subj_id in subj_id_list if subj_id in run_subjects])

        # Setup extraction
        self._setup_extraction(layout=layout, subj_id_list=subj_id_list, exclude_niftis=exclude_niftis)

        if n_cores: 
            if n_cores > cpu_count(): raise ValueError(f"More cores specified than available - Number of cores specified: {n_cores}; Max cores available: {cpu_count()}.")
            if isinstance(n_cores, int): self._n_cores = n_cores
            else: raise ValueError("`n_cores` must be an integer.")
            
            # Generate list of tuples for each subject
            args_list = [(subj_id, self._subject_info[subj_id]["nifti_files"],self._subject_info[subj_id]["mask_files"],self._subject_info[subj_id]["event_files"],
                          self._subject_info[subj_id]["confound_files"], self._subject_info[subj_id]["confound_metadata_files"], self._subject_info[subj_id]["run_list"],
                          self._subject_info[subj_id]["tr"], condition, self._parcel_approach, self._signal_clean_info, verbose, flush_print
                          ) for subj_id in self._subject_ids]
            
            with Parallel(n_jobs=self._n_cores) as parallel:
                outputs = parallel(delayed(_extract_timeseries)(*args) for args in args_list)
            
            for output in outputs:
                if isinstance(output, dict):
                    self._subject_timeseries.update(output)

        else:
            for subj_id in self._subject_ids:

                subject_timeseries=_extract_timeseries(subj_id=subj_id, nifti_files=self._subject_info[subj_id]["nifti_files"], mask_files=self._subject_info[subj_id]["mask_files"], 
                                                       event_files=self._subject_info[subj_id]["event_files"], confound_files=self._subject_info[subj_id]["confound_files"],
                                                       confound_metadata_files=self._subject_info[subj_id]["confound_metadata_files"], run_list=self._subject_info[subj_id]["run_list"], 
                                                       tr=self._subject_info[subj_id]["tr"], condition=condition, parcel_approach=self._parcel_approach, signal_clean_info=self._signal_clean_info,
                                                       verbose=verbose, flush_print=flush_print)
            
                # Aggregate new timeseries
                if isinstance(subject_timeseries, dict): self._subject_timeseries.update(subject_timeseries)
        
    # Get valid subjects to iterate through
    def _setup_extraction(self, layout, subj_id_list, exclude_niftis):
       for subj_id in subj_id_list:
            if self._task_info["session"]:
                nifti_files = sorted(layout.get(scope="derivatives", return_type="file",suffix="bold", task=self._task_info["task"], space=self._space, session=self._task_info["session"],extension = "nii.gz", subject=subj_id))
                bold_metadata_files = sorted(layout.get(scope="derivatives", return_type="file",suffix="bold", task=self._task_info["task"], space=self._space, session=self._task_info["session"], extension = "json", subject=subj_id))
                event_files = sorted([file for file in sorted(layout.get(return_type="filename",suffix="events", task=self._task_info["task"], session=self._task_info["session"],extension = "tsv", subject = subj_id))]) if self._task_info["condition"] else []
                confound_files = sorted(layout.get(scope='derivatives', return_type='file', desc='confounds', task=self._task_info["task"], session=self._task_info["session"],extension = "tsv", subject=subj_id))
                confound_metadata_files = sorted(layout.get(scope='derivatives', return_type='file', desc='confounds', task=self._task_info["task"], session=self._task_info["session"],extension = "json", subject=subj_id))
                mask_files = sorted(layout.get(scope='derivatives', return_type='file', suffix='mask', task=self._task_info["task"], space=self._space, session=self._task_info["session"], extension = "nii.gz", subject=subj_id))
            else:
                nifti_files = sorted(layout.get(scope="derivatives", return_type="file",suffix="bold", task=self._task_info["task"], space=self._space, extension = "nii.gz", subject=subj_id))
                bold_metadata_files = sorted(layout.get(scope="derivatives", return_type="file",suffix="bold", task=self._task_info["task"], space=self._space, extension = "json", subject=subj_id))
                event_files = sorted([file for file in sorted(layout.get(return_type="filename",suffix="events", task=self._task_info["task"], extension = "tsv", subject = subj_id))]) if self._task_info["condition"] else []
                confound_files = sorted(layout.get(scope='derivatives', return_type='file', desc='confounds', task=self._task_info["task"], extension = "tsv", subject=subj_id))
                confound_metadata_files = sorted(layout.get(scope='derivatives', return_type='file', desc='confounds', task=self._task_info["task"], extension = "json", subject=subj_id))
                mask_files = sorted(layout.get(scope='derivatives', return_type='file', suffix='mask', task=self._task_info["task"], space=self._space, extension = "nii.gz", subject=subj_id))
            
            # Remove excluded file from the nifti_files list, which will prevent it from being processed
            if exclude_niftis and len(nifti_files) != 0:
                nifti_files = [nifti_file for nifti_file in nifti_files if os.path.basename(nifti_file) not in exclude_niftis]

            # Generate a list of runs to iterate through based on runs in nifti_files
            if self._task_info["runs"]:
                check_runs = [f"run-{run}" for run in self._task_info["runs"]] 
            elif len(nifti_files) != 0:
                if "run-" in os.path.basename(nifti_files[0]):
                    check_runs = [re.search("run-(\\S+?)[-_]",os.path.basename(x))[0][:-1] for x in nifti_files]
                else:
                   check_runs = [] 
            else:
                check_runs = []

            # Generate a list of runs to iterate through based on runs in nifti_files
            if not self._task_info["session"] and len(nifti_files) != 0:
                if "ses-" in os.path.basename(nifti_files[0]):
                    check_sessions = [re.search("ses-(\\S+?)[-_]",os.path.basename(x))[0][:-1] for x in nifti_files]
                    if len(list(set(check_sessions))) > 1:
                        raise ValueError(f"`session` not specified but subject {subj_id} has more than one session : {sorted(list(set(check_sessions)))}. In order to continue timeseries extraction, the specific session to extract must be specified.")

            if len(nifti_files) == 0:
                warnings.warn(f"Skipping subject: {subj_id} due to missing nifti files.")
                continue
            
            if len(mask_files) == 0:
                warnings.warn(f"Subject: {subj_id} is missing mask file but timeseries extraction will continue.")
                
            if self._signal_clean_info["use_confounds"]:
                if len(confound_files) == 0:
                    warnings.warn(f"Skipping subject: {subj_id} due to missing confound files.")
                    continue
                if len(confound_metadata_files) == 0 and self._signal_clean_info["n_acompcor_separate"]:
                    warnings.warn(f"Skipping subject: {subj_id} due to missing confound metadata to locate the first six components of the white-matter and cerobrospinal fluid masks separately.")
                    continue
            
            if self._task_info["condition"] and len(event_files) == 0:
                warnings.warn(f"Skipping subject: {subj_id} due to having no event files.")
                continue
                
            if len(check_runs) != 0:
                run_list = []
                # Check if at least one run has all files present
                for run in check_runs:
                    curr_list = []
                    # Assess is any of these returns True
                    curr_list.append(any([run in file for file in nifti_files]))
                    if self._task_info["condition"]: curr_list.append(any([run in file for file in event_files]))
                    if self._signal_clean_info["use_confounds"]:
                        curr_list.append(any([run in file for file in confound_files]))
                        if self._signal_clean_info["n_acompcor_separate"]: curr_list.append(any([run in file for file in confound_metadata_files]))
                    if len(mask_files) != 0: curr_list.append(any([run in file for file in mask_files]))
                    # Append runs that contain all needed files
                    if all(curr_list): run_list.append(run)
                
                # Skip subject if no run has all needed files present
                if len(run_list) != len(check_runs) or len(run_list) == 0:
                    if len(run_list) == 0:
                        if self._task_info["condition"]: warnings.warn(f"Skipping subject: {subj_id} due to no nifti file, mask file, confound tsv file, confound json file being from the same run.")
                        else: warnings.warn(f"Skipping subject: {subj_id} due to no nifti file, mask file, event file, confound tsv file, confound json file being from the same run.")
                        continue
                    else: warnings.warn(f"Subject: {subj_id} only has the following runs available: {', '.join(run_list)}.")
            else:
                run_list = [None]
            # Add subject list to subject attribute. These are subjects that will be ran
            self._subject_ids.append(subj_id)

            # Get repetition time for the subject
            tr = self._task_info["tr"] if self._task_info["tr"] else json.load(open(bold_metadata_files[0]))["RepetitionTime"]

            # Store subject specific information
            self._subject_info[subj_id] = {"nifti_files": nifti_files, "event_files": event_files, "confound_files": confound_files, "confound_metadata_files": confound_metadata_files, "mask_files": mask_files,
                                           "tr": tr, "run_list": run_list}

    def timeseries_to_pickle(self, output_dir: Path, file_name: Optional[str]=None) -> None:
        """Save Bold Data

        Saves the timeseries dictionary obtained from running `get_bold()` as a pickle file.

        Parameters
        ----------
        output_dir: Path
            Directory to save to. The directory will be created if it does not exist.
        file_name: str or None, default=None
            Name of the file with or without the "pkl" extension.
        """
        import joblib

        if not hasattr(self, "_subject_timeseries"):
            raise AttributeError("Cannot save pickle file since `self._subject_timeseries` does not exist, either run `self.get_bold()` or assign a valid timeseries dictionary to `self.subject_timeseries`.")

        if output_dir:
            if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        if file_name == None: save_file_name = "subject_timeseries.pkl"
        else: save_file_name = f"{os.path.splitext(file_name.rstrip())[0].rstrip()}.pkl" 

        with open(os.path.join(output_dir,save_file_name), "wb") as f:
            joblib.dump(self._subject_timeseries,f)

    def visualize_bold(self, subj_id: Union[int,str], run: int, roi_indx: Optional[Union[int, List[int]]]=None, region: Optional[str]=None, 
                       show_figs: bool=True, output_dir: Optional[Path]=None, file_name: Optional[str]=None, **kwargs) -> None:
        """Plot Bold Data

        Collects files needed to extract timeseries data from NIfTI files for BIDS-compliant datasets.

        Parameters
        ----------
        subj_id: str pr int
            The subject ID.
        run: int
            The run to plot.
        roi_indx: int or List[int] or None, default=None
            The indices of the parcellation nodes to plot. See self.node_indices for valid node names and indices.
        region: str or None, default=None
            The region of the parcellation to plot. If not None, all nodes in the specified region will be averaged then plotted. See `regions` in self.parcel_approach 
            for valid regions names.
        show_figs: bool, default=True
            Whether to show the figures.
        output_dir: Path or None, default=None
            Directory to save to. The directory will be created if it does not exist.
        file_name: str or None, default=None
            Name of the file without the extension.
        **kwargs: Dict
            Keyword arguments used when saving figures. Valid keywords include:

            - "dpi": int, default=300
                Dots per inch for the figure. Default is 300 if `output_dir` is provided and `dpi` is not specified.
            - "figsize": tuple, default=(11, 5)
                Size of the figure in inches. Default is (11, 5) if "figsize" is not specified.

        Returns
        -------
        matplotlib.Figure
            An instance of a matplotlib figure.
        """
    
        import matplotlib.pyplot as plt, numpy as np

        if not hasattr(self, "_subject_timeseries"):
            raise AttributeError("Cannot plot bold data since `self._subject_timeseries` does not exist, either run `self.get_bold()` or assign a valid timeseries structure to self.subject_timeseries.")

        if isinstance(subj_id, int): subj_id = str(subj_id)

        if roi_indx !=None and region != None:
            raise ValueError("`roi_indx` and `region` can not be used simultaneously.")
        
        if file_name != None and output_dir == None: warnings.warn("`file_name` supplied but no `output_dir` specified. Files will not be saved.")
        
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # Defaults
        defaults = {"dpi": 300,"figsize": (11,5)}

        plot_dict = _check_kwargs(defaults, **kwargs)

        # Obtain the column indices associated with the rois; add logic for roi_indx == 0 since it would be recognized as False
        if roi_indx or roi_indx == 0:
            if type(roi_indx) == int:
                plot_indxs = roi_indx
            
            elif type(roi_indx) == str:
                # Check if parcellation_approach is custom
                if "Custom" in self._parcel_approach.keys() and "nodes" not in self._parcel_approach["Custom"].keys():
                    _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")
                plot_indxs = self._parcel_approach[self._parcel_approach.keys[0]]["nodes"].index(roi_indx)
            
            elif type(roi_indx) == list:
                if all([isinstance(indx,int) for indx in roi_indx]):
                    plot_indxs = np.array(roi_indx)
                elif all([isinstance(indx,str) for indx in roi_indx]):
                    # Check if parcellation_approach is custom
                    if "Custom" in self._parcel_approach.keys() and "nodes" not in self._parcel_approach["Custom"].keys():
                        _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")
                    plot_indxs = np.array([self._parcel_approach[self._parcel_approach.keys[0]]["nodes"].index(index) for index in roi_indx])
                else:
                    raise ValueError("All elements in `roi_indx` need to be all strings or all integers.")
                
        elif region:
            if "Custom" in self._parcel_approach.keys():
                if "regions" not in self._parcel_approach["Custom"].keys():
                    _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")
                else:
                    plot_indxs =  np.array(self._parcel_approach["Custom"]["regions"][region]["lh"] + self._parcel_approach["Custom"]["regions"][region]["rh"])
            else:
                plot_indxs = np.array([index for index, label in enumerate(self._parcel_approach[list(self._parcel_approach.keys())[0]]["nodes"]) if region in label])
        
        plt.figure(figsize=plot_dict["figsize"])

        if roi_indx or roi_indx == 0: 
            plt.plot(range(1, self._subject_timeseries[subj_id][f"run-{run}"].shape[0] + 1), self._subject_timeseries[subj_id][f"run-{run}"][:,plot_indxs])
        elif region:  
            plt.plot(range(1, self._subject_timeseries[subj_id][f"run-{run}"].shape[0] + 1), np.mean(self._subject_timeseries[subj_id][f"run-{run}"][:,plot_indxs], axis=1))
            plt.title(region)
        plt.xlabel("TR")

        if output_dir:
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            file_name = f"{os.path.splitext(file_name.rstrip())[0].rstrip()}.png" if file_name else f'subject-{subj_id}_run-{run}_timeseries.png'
            plt.savefig(os.path.join(output_dir,file_name), dpi=plot_dict["dpi"])

        if show_figs == False:
            plt.close()
        else:
            plt.show()