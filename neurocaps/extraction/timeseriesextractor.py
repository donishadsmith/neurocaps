import json, os, re, sys, warnings
from typing import Union
from .._utils import _TimeseriesExtractorGetter, _check_confound_names, _check_parcel_approach, _extract_timeseries

class TimeseriesExtractor(_TimeseriesExtractorGetter):
    def __init__(self, space: str="MNI152NLin2009cAsym", standardize: Union[bool,str]="zscore_sample", detrend: bool=False , low_pass: float=None, high_pass: float=None, 
                 parcel_approach : dict={"Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}}, use_confounds: bool=True, confound_names: list[str]=None, 
                 fwhm: float=None, fd_threshold: float=None, n_acompcor_separate: int=None, dummy_scans: int=None):
        """Timeseries Extractor Class
        
        Initializes the TimeseriesExtractor class to prepare for Co-activation Patterns (CAPs) analysis.

        Parameters
        ----------
        space: str, default="MNI152NLin2009cAsym"
            The brain template space data is in. 
        standardize: bool, default=True
            Determines whether to standardize the timeseries. Refer to Nilearn's NiftiLabelsMasker for available options. 
        detrend: bool, default=True
            Detrends timeseries during extraction.
        low_pass: bool, default=None
            Signals above cutoff frequency will be filtered out.
        high_pass: float, default=None
            Signals below cutoff frequency will be filtered out.
        parcel_approach: dict, default={"Schaefer": {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}}
            Approach to use to parcellate bold images. Should be in the form of a nested dictionary where the first key is the atlas.
            Currently only "Schaefer", "AAL", and "Custom" is supported. For the sub-dictionary for "Schaefer", available options includes "n_rois", "yeo_networks", and "resolution_mm".
            Please refer to the documentation for Nilearn's `datasets.fetch_atlas_schaefer_2018` for valid inputs. For the subdictionary for "AAL" only "version"
            is an option. Please refer to the documentation for Nilearn's `datasets.fetch_atlas_aal` for valid inputs. As of version 0.8.9, you can replace "Schaefer"
            and "AAL" with "Custom". At minimum, if "Custom" is specified, a subkey, called "maps" specifying the directory location of the parcellation as a Nifti (e.g .nii or .nii.gz)
            - {"Custom": {"maps": "/location/to/parcellation.nii.gz"}}.
        use_confounds: bool, default=True
            To use confounds when extracting timeseries.
        confound_names: List[str], default=None
            Names of confounds to use in confound files. If None, default confounds are used.
        fwhm: float, default=None
            Parameter for spatial smoothing.
        fd_threshold: float, default=None
            Threshold criteria to remove frames after nuisance regression and timeseries extraction. For this to work, a column named "framewise_displacement" must be
            in the confounds dataframe and `use_confounds` must be true. Additionally, 'framewise_displacemnt' does not need to be specified in the `confound_names` for this to work.
        n_acompcor_separate: int, default = None
            The number of separate acompcor components derived from the white-matter (WM) and cerebrospinal (CSF) masks to use. For instance if '5' is assigned to this parameter
            then the first five components derived from the WM mask and the first five components derived from the CSF mask will be used, resulting in ten acompcor components being
            used. If this parameter is not none any acompcor components listed in the confound names will be disregarded in order to locate the first 'n' components derived from the 
            masks. To use the acompcor components derived from the combined masks (WM & CSF) leave this parameter as 'None' and list the specific acompcors of interest in the 
            `confound_names` parameter.
        dummy_scans: float, default=None
            Removes the first `n` number of volumes before extracting the timeseries.
        
        Notes for `confounds_names`
        --------------------------
        For the `confound_names` parameter, an asterisk ("*") can be used to find the name of confounds that starts with the term preceding the asterisk.
        For instance, "cosine*" will find all confound names in the confound files starting with "cosine".

        Notes for `parcel_approach`
        ---------------------------
        If using a "Custom" parcellation approach, ensure each node in your dataset includes both left (lh) and right (rh) hemisphere versions. Also, this function assumes that the background label is "zero". Do not add a a background label, in the "nodes" or "networks" key,
        the zero index should correspond the first id that is not zero.

        Custom Key Structure:
        - maps: Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g., .nii for NIfTI files). For plotting purposes, this key is not required.
        - nodes:  list of all node labels used in your study, arranged in the exact order they correspond to indices in your parcellation files. 
          Each label should match the parcellation index it represents. For example, if the parcellation label "1" corresponds to the left hemisphere 
          visual cortex area 1, then "LH_Vis1" should occupy the 0th index in this list. This ensures that data extraction and analysis accurately reflect the anatomical regions intended.
          For timeseries extraction, this key is not required.
        - regions: Dictionary defining major brain regions. Each region should list node indices under "lh" and "rh" to specify left and right hemisphere nodes. For timeseries extraction, this key is not required.
        
        Example 
        The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis) and hippocampus regions:

        parcel_approach = {"Custom": {"maps": "/location/to/parcellation.nii.gz",
                             "nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus", "RH_Vis1", "RH_Vis2", "RH_Hippocampus"],
                             "regions": {"Vis" : {"lh": [0,1],
                                                  "rh": [3,4]},
                                         "Hippocampus": {"lh": [2],
                                                         "rh": [5]}}}}
        """
        self._space = space
        self._signal_clean_info = {"standardize": standardize, "detrend": detrend, "low_pass": low_pass, "high_pass": high_pass, "fwhm": fwhm, 
                                   "dummy_scans": dummy_scans, "use_confounds": use_confounds,  "n_acompcor_separate": n_acompcor_separate,
                                   "fd_threshold": None}   

        # Check parcel_apprach
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_approach)

        if self._signal_clean_info["use_confounds"]:
            self._signal_clean_info["confound_names"] = _check_confound_names(high_pass=high_pass, specified_confound_names=confound_names, n_acompcor_separate=n_acompcor_separate)
            self._signal_clean_info["fd_threshold"] = fd_threshold

    def get_bold(self, bids_dir: str, task: str, session: Union[int,str]=None, runs: list[int]=None, condition: str=None, tr: Union[int, float]=None, 
                 run_subjects: list[str]=None, exclude_subjects: list[str]= None, pipeline_name: str=None, n_cores: Union[bool, int]=None, verbose: bool=True, flush_print: bool=False,
                 exclude_niftis: list[str]=None) -> None: 
        """Get Bold Data

        Collects files needed to extract timeseries data from NIfTI files for BIDS-compliant datasets.

        Parameters
        ----------
        bids_dir: str
            Path to a BIDS compliant directory. 
        task: str
            Name of task to process.
        session: int, default=None
            Session to extract timeseries from. Only a single session can be extracted at a time. 
        runs: list[int], default=None
            Run number to extract timeseries data from. Extracts all runs if unspecified.
        condition: str, default=None
            Specific condition in the task to extract from. Only a single condition can be extracted at a time.
        tr: int or float, default=None
            Repetition time for task.
        run_subjects: list[str], default=None
            List of subject IDs to process. Processes all subjects if None.
        exclude_subjects: list[str], default=None
            List of subject IDs to exclude.  
        pipeline_name: str, default=None
            The name of the pipeline folder in the derivatives folder containing the preprocessed data. If None, BIDSLayout will use the name of dset_dir with derivatives=True. This parameter
            should be used if their are multiple pipelines in the derivatives folder.
        n_cores: bool or int, default=None
            The number of CPU cores to use for multiprocessing. If true, all available cores will be used.
        verbose: bool, default=True
            Print subject specific information such as confounds being extracted and id and run of subject being processed during timeseries extraction.
        flush_print: bool, default=False
            Flush the printed subject specific infomation produced during the timeseries extraction process.
        exclude_niftis: list[str], default=None
            Exclude certain preprocessed nifti files to prevent the timeseries of that file from being extracted. Used if there are specific runs across differnt participants that need to be
            excluded.
        """
        import bids, multiprocessing

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
            if n_cores == True:
                self._n_cores = multiprocessing.cpu_count()
            else:
                if n_cores > multiprocessing.cpu_count():
                    raise ValueError(f"More cores specified than available - Number of cores specified: {n_cores}; Max cores available: {multiprocessing.cpu_count()}.")
                else:
                    self._n_cores = n_cores
            
            # Generate list of tuples for each subject
            args_list = [(subj_id, self._subject_info[subj_id]["nifti_files"],self._subject_info[subj_id]["mask_files"],self._subject_info[subj_id]["event_files"],
                          self._subject_info[subj_id]["confound_files"], self._subject_info[subj_id]["confound_metadata_files"], self._subject_info[subj_id]["run_list"],
                          self._subject_info[subj_id]["tr"], condition, self._parcel_approach, self._signal_clean_info, verbose, flush_print
                          ) for subj_id in self._subject_ids]

            with multiprocessing.Pool(processes=self._n_cores) as pool:
                outputs = pool.starmap(_extract_timeseries, args_list)
            
            for output in outputs:
                if isinstance(output, dict):
                    self._subject_timeseries.update(output)

            # Ensure subjects are sorted
            self._subject_timeseries = dict(sorted(self._subject_timeseries.items()))

            # Ensure processes close
            pool.close()
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

    def timeseries_to_pickle(self, output_dir: str, file_name: str=None):
        """Save Bold Data

        Saves the timeseries dictionary obtained from running `get_bold()` as a pickle file.

        Parameters
        ----------
        output_dir: str
            Directory to save the file to. Will create the directory if it does not exist.
        file_name: str, default=None
            Name of the file without or without the "pkl" extension.
        """
        import pickle

        if not hasattr(self, "_subject_timeseries"):
            raise AttributeError("Cannot save pickle file since `self._subject_timeseries` does not exist, either run `self.get_bold()` or assign a valid timeseries dictionary to `self.subject_timeseries`.")

        if output_dir:
            if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        if file_name == None: save_file_name = "subject_timeseries.pkl"
        else: save_file_name = f"{os.path.splitext(file_name.rstrip())[0].rstrip()}.pkl" 

        with open(os.path.join(output_dir,save_file_name), "wb") as f:
            pickle.dump(self._subject_timeseries,f)

    def visualize_bold(self, subj_id: Union[int,str], run: int, roi_indx: Union[int, list[int]]=None, region: str=None, show_figs: bool=True, output_dir: str=None, file_name: str=None, **kwargs):
        """Plot Bold Data

        Collects files needed to extract timeseries data from NIfTI files for BIDS-compliant datasets.

        Parameters
        ----------
        subj_id: str
            Subject ID, as a string, to plot.
        run: int
            The run to plot.
        roi_indx: int or list[int], default=None
            The indices of the Schaefer nodes to plot. See self.node_indices for valid node names and indices.
        region: str, default=None
            The region of the parcellation to plot. If not None, all nodes in the specified region will be averaged then plotted. See `regions` in self.parcel_approach 
            for valid regions names.
        show_figs: bool, default=True
            Whether to show figires or not to show figures
        output_dir: str, default=None
            Directory to save the file to. Will create the directory if it does not exist.
        file_name: str, default=None
            Name of the file without the extension.
        **kwargs: dict
            Keyword arguments used when saving figures. Valid keywords include:
        
            - "dpi": int, default=300
                Dots per inch for the figure. Default is 300 if `output_dir` is provided and `dpi` is not specified.
            - "figsize": tuple, default=(11, 5)
                Size of the figure in inches. Default is (11, 5) if "figsize" is not specified.

        Notes
        -----
        If using a "Custom" parcellation approach, ensure each node in your dataset includes both left (lh) and right (rh) hemisphere versions. Also, this function assumes that the background label is "zero". Do not add a a background label, in the "nodes" or "networks" key,
        the zero index should correspond the first id that is not zero.

        Custom Key Structure:
        - maps: Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g., .nii for NIfTI files). For plotting purposes, this label is not required.
        - nodes: A list of all node labels used in your study, arranged in the exact order they correspond to indices in your parcellation files. 
          Each label should match the parcellation index it represents. For example, if the parcellation label "1" corresponds to the left hemisphere 
          visual cortex area 1, then "LH_Vis1" should occupy the 0th index in this list. This ensures that data extraction and analysis accurately reflect the anatomical regions intended.
        - regions: Dictionary defining major brain regions. Each region should list node indices under "lh" and "rh" to specify left and right hemisphere nodes.
        
        Example 
        The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis) and hippocampus regions:

        parcel_approach = {"Custom": {"maps": "/location/to/parcellation.nii.gz",
                             "nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus", "RH_Vis1", "RH_Vis2", "RH_Hippocampus"],
                             "regions": {"Vis" : {"lh": [0,1],
                                                  "rh": [3,4]},
                                         "Hippocampus": {"lh": [2],
                                                         "rh": [5]}}}}
        """
    
        import matplotlib.pyplot as plt, numpy as np

        if not hasattr(self, "_subject_timeseries"):
            raise AttributeError("Cannot plot bold data since `self._subject_timeseries` does not exist, either run `self.get_bold()` or assign a valid timeseries structure to self.subject_timeseries.")

        if isinstance(subj_id,int): subj_id = str(subj_id)

        if roi_indx !=None and region != None:
            raise ValueError("`roi_indx` and `region` can not be used simultaneously.")
        
        if file_name != None and output_dir == None: warnings.warn("`file_name` supplied but no `output_dir` specified. Files will not be saved.")
        
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        plot_dict = dict(dpi = kwargs["dpi"] if kwargs and "dpi" in kwargs.keys() else 300,
                         figsize = kwargs["figsize"] if kwargs and "figsize" in kwargs.keys() else (11, 5))
        
        if kwargs:
            invalid_kwargs = {key : value for key, value in kwargs.items() if key not in plot_dict.keys()}
            if len(invalid_kwargs.keys()) > 0:
                print(f"Invalid kwargs arguments used and will be ignored {invalid_kwargs}.")

        # Obtain the column indices associated with the rois; add logic for roi_indx == 0 since it would be recognized as False
        if roi_indx or roi_indx == 0:
            if type(roi_indx) == int:
                plot_indxs = roi_indx
            
            elif type(roi_indx) == str:
                # Check if parcellation_approach is custom
                if "Custom" in self.parcel_approach.keys() and "nodes" not in self.parcel_approach["Custom"].keys():
                    _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")
                plot_indxs = self._parcel_approach[self._parcel_approach.keys[0]]["nodes"].index(roi_indx)
            
            elif type(roi_indx) == list:
                if all([isinstance(indx,int) for indx in roi_indx]):
                    plot_indxs = np.array(roi_indx)
                elif all([isinstance(indx,str) for indx in roi_indx]):
                    # Check if parcellation_approach is custom
                    if "Custom" in self.parcel_approach.keys() and "nodes" not in self.parcel_approach["Custom"].keys():
                        _check_parcel_approach(parcel_approach=self._parcel_approach, call="visualize_bold")
                    plot_indxs = np.array([self._parcel_approach[self._parcel_approach.keys[0]]["nodes"].index(index) for index in roi_indx])
                else:
                    raise ValueError("All elements in `roi_indx` need to be all strings or all integers.")
                
        elif region:
            if "Custom" in self.parcel_approach.keys():
                if "regions" not in self.parcel_approach["Custom"].keys():
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