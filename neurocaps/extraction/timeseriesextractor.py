from typing import Union
from .._utils import _TimeseriesExtractorGetter, _check_parcel_approach, _extract_timeseries
import re, os, warnings, json

class TimeseriesExtractor(_TimeseriesExtractorGetter):
    def __init__(self, space: str="MNI152NLin2009cAsym", standardize: Union[bool,str]="zscore_sample", detrend: bool=False , low_pass: float=None, high_pass: float=None, 
                 parcel_approach : dict={"Schaefer": {"n_rois": 400, "yeo_networks": 7}}, use_confounds: bool=True, confound_names: list[str]=None, n_acompcor_separate: int=None, dummy_scans: int=None):
        """Timeseries Extractor Class
        
        Initializes a TimeseriesExtractor to prepare for Co-activation Patterns (CAPs) analysis.

        Parameters
        ----------
        space : str, default="MNI152NLin2009cAsym"
            The brain template space data is in. 
        standardize : bool, default=True
            Determines whether to standardize the timeseries. Refer to Nilearn's NiftiLabelsMasker for available options. 
        detrend : bool, default=True
            Detrends timeseries during extraction.
        low_pass : bool, default=None
            Signals above cutoff frequency will be filtered out.
        high_pass : float, default=None
            Signals below cutoff frequency will be filtered out.
        parcel_approach : dict, default={"Schaefer": {"n_rois": 400, "yeo_networks": 7}}
            Approach to use to parcellate bold images. Should be in the form of a nested dictionary where the first key is the atlas.
            Currently only "Schaefer" is important. For the sub-dcitionary, for "Schaefer", available options includes "n_rois" and "yeo_networks".
            Please refer to the documentation for Nilearn's `datasets.fetch_atlas_schaefer_2018` for valid inputs.
        use_confounds : bool, default=True
            To use confounds when extracting timeseries.
        confound_names : List[str], default=None
            Names of confounds to use in confound files. If None, default confounds are used.
        n_acompcor_separate : int, default = None
            The number of seperate acompcor components derived from the white-matter (WM) and cerebrospinal (CSF) masks to use. For instance if '5' is assigned to this parameter
            then the first five components derived from the WM mask and the first five components derived from the CSF mask will be used, resulting in ten acompcor components being
            used. If this parameter is not none any acompcor components listed in the confound names will be disregarded in order to locate the first 'n' components derived from the 
            masks. To use the acompcor components derived from the combined masks (WM & CSF) leave this parameter as 'None' and list the specific acompcors of interest in the 
            `confound_names` parameter.
        dummy_scans : float, default=None
            Remove the first `n` number of volumes before extracting timeseries.
        

        Notes
        -----
        For the `confound_names` parameter, an asterisk ("*") can be used to find the name of confounds that starts with the term preceding the asterisk.
        For instance, "cosine*" will find all confound names in the confound files starting with "cosine".
        """
        self._space = space
        self._signal_clean_info = {"standardize": standardize, "use_confounds": use_confounds, "detrend": detrend, 
                                   "low_pass": low_pass, "high_pass": high_pass, "dummy_scans": dummy_scans,"n_acompcor_separate": n_acompcor_separate}   

        # Check parcel_apprach
        self._parcel_approach = _check_parcel_approach(parcel_approach=parcel_approach)

        if self._signal_clean_info["use_confounds"]:
            if confound_names == None:
                if self._signal_clean_info["high_pass"]:
                    # Do not use cosine or acompcor regressor if high pass filtering is not None. Acompcor regressors are estimated on high pass filtered version 
                    # of data form fmriprep
                    self._signal_clean_info["confound_names"] = [
                        "trans_x", "trans_x_derivative1", "trans_x_power2", "trans_x_derivative1_power2",
                        "trans_y", "trans_y_derivative1", "trans_y_derivative1_power2", "trans_y_power2",
                        "trans_z", "trans_z_derivative1", "trans_z_power2", "trans_z_derivative1_power2",
                        "rot_x", "rot_x_derivative1", "rot_x_power2", "rot_x_derivative1_power2",
                        "rot_y", "rot_y_derivative1", "rot_y_power2", "rot_y_derivative1_power2",
                        "rot_z", "rot_z_derivative1", "rot_z_derivative1_power2", "rot_z_power2"
                    ]
                else:
                    self._signal_clean_info["confound_names"] = [
                        "cosine*",
                        "trans_x", "trans_x_derivative1", "trans_x_power2", "trans_x_derivative1_power2",
                        "trans_y", "trans_y_derivative1", "trans_y_derivative1_power2", "trans_y_power2",
                        "trans_z", "trans_z_derivative1", "trans_z_power2", "trans_z_derivative1_power2",
                        "rot_x", "rot_x_derivative1", "rot_x_power2", "rot_x_derivative1_power2",
                        "rot_y", "rot_y_derivative1", "rot_y_power2", "rot_y_derivative1_power2",
                        "rot_z", "rot_z_derivative1", "rot_z_derivative1_power2", "rot_z_power2", 
                        "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05", "a_comp_cor_06"
                    ]
            else:
                self._signal_clean_info["confound_names"] = confound_names
                assert type(self._signal_clean_info["confound_names"]) == list and len(self._signal_clean_info["confound_names"]) > 0 , "confound_names must be a non-empty list"

            if self._signal_clean_info["n_acompcor_separate"]:
                check_confounds = [confound for confound in self._signal_clean_info["confound_names"] if "a_comp_cor" not in confound]
                if len(self._signal_clean_info["confound_names"]) > len(check_confounds):
                    removed_confounds = [element for element in self._signal_clean_info["confound_names"] if element not in check_confounds]
                    warnings.warn(f"Since `n_acompcor_separate` has been specified, specified acompcor components in `confound_names` will be disregarded and replaced with the first {self._signal_clean_info['n_acompcor_separate']} components of the white matter and cerebrospinal fluid masks for each participant. The following components will not be used {removed_confounds}")
                    self._signal_clean_info["confound_names"] = check_confounds 

            print(f"List of confound regressors that will be used during timeseries extraction if available in confound dataframe: {self._signal_clean_info['confound_names']}")

            
    def get_bold(self, bids_dir: str, session: int, runs: list[int]=None, task: str="rest", condition: str=None, tr: Union[int, float]=None, run_subjects: list[str]=None, exclude_subjects: list[str]= None, pipeline_name: str=None, n_cores: Union[bool, int]=None) -> None: 
        """Get Bold Data

        Collects files needed to extract timeseries data from NIfTI files for BIDS-compliant datasets.

        Parameters
        ----------
        bids_dir : str
            Path to a BIDS compliant directory. 
        session : int
            Session number.
        run : list[int]
            Run number to extract timeseries data from. Extracts all runs if unspecified.
        task : str, default="rest"
            Task name.
        condition : str, default=None
            Specific condition in the task to extract from
        tr : int or float, default=None
            Repetition time.
        run_subjects : List[str], default=None
            List of subject IDs to process. Processes all subjects if None.
        exclude_subjects : List[str], default=None
            List of subject IDs to exclude.  
        pipeline_name: str, default=None
            The name of the pipeline folder in the derivatives folder containing the preprocessed data. If None, BIDSLayout will use the name of dset_dir with derivatives=True. This parameter
            should be used if their are multiple pipelines in the derivatives folder.
        n_cores: bool or int, default=None
            The number of CPU cores to use for multiprocessing. If true, all available cores will be used.
        """
        import bids, multiprocessing

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

        print(f"Bids layout collected.")

        subj_id_list = sorted(layout.get(return_type="id", target="subject", task=task, space=self._space, suffix="bold")) 

        if exclude_subjects: 
            exclude_subjects = [str(subj_id) if not isinstance(subj_id,str) else subj_id for subj_id in exclude_subjects]
            subj_id_list = sorted([subj_id for subj_id in subj_id_list if subj_id not in exclude_subjects])

        if run_subjects: 
            run_subjects = [str(subj_id) if not isinstance(subj_id,str) else subj_id for subj_id in run_subjects]
            subj_id_list = sorted([subj_id for subj_id in subj_id_list if subj_id in run_subjects])

        # Setup extraction
        self._setup_extraction(layout=layout, subj_id_list=subj_id_list)

        if n_cores:
            if n_cores == True:
                self._n_cores = multiprocessing.cpu_count()
            else:
                if n_cores > multiprocessing.cpu_count():
                    raise ValueError(f"More cores specified than available - Number of cores specified: {n_cores}; Max cores available: {multiprocessing.cpu_count()}")
                else:
                    self._n_cores = n_cores
            
            # Generate list of tuples for each subject
            args_list = [(subj_id, self._subject_info[subj_id]["nifti_files"],self._subject_info[subj_id]["mask_files"],self._subject_info[subj_id]["event_files"],
                          self._subject_info[subj_id]["confound_files"], self._subject_info[subj_id]["confound_metadata_files"], self._subject_info[subj_id]["run_list"],
                          self._subject_info[subj_id]["tr"], condition, self._parcel_approach, self._signal_clean_info
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
                                                       tr=self._subject_info[subj_id]["tr"], condition=condition, parcel_approach=self._parcel_approach, signal_clean_info=self._signal_clean_info)
            
                # Aggregate new timeseries
                self._subject_timeseries.update(subject_timeseries)
        
    
    # Get valid subjects to iterate through
    def _setup_extraction(self, layout, subj_id_list):
       for subj_id in subj_id_list:
            
            nifti_files = sorted(layout.get(scope="derivatives", return_type="file",suffix="bold", task=self._task_info["task"], space=self._space, session=self._task_info["session"],extension = "nii.gz", subject=subj_id))
            bold_metadata_files = sorted(layout.get(scope="derivatives", return_type="file",suffix="bold", task=self._task_info["task"], space=self._space, session=self._task_info["session"], extension = "json", subject=subj_id))
            event_files = sorted([file for file in sorted(layout.get(return_type="filename",suffix="events", task=self._task_info["task"], session=self._task_info["session"],extension = "tsv", subject = subj_id))])
            confound_files = sorted(layout.get(scope='derivatives', return_type='file', desc='confounds', task=self._task_info["task"], session=self._task_info["session"],extension = "tsv", subject=subj_id))
            confound_metadata_files = sorted(layout.get(scope='derivatives', return_type='file', desc='confounds', task=self._task_info["task"], session=self._task_info["session"],extension = "json", subject=subj_id))
            mask_files = sorted(layout.get(scope='derivatives', return_type='file', suffix='mask', task=self._task_info["task"], space=self._space, session=self._task_info["session"], extension = "nii.gz", subject=subj_id))
            # Generate a list of runs to iterate through based on runs in nifti_files
            run_list = [f"run-{run}" for run in self._task_info["runs"]] if self._task_info["runs"] else [re.search("run-(\d+)",x)[0] for x in nifti_files]

            if len(nifti_files) == 0 or len(mask_files) == 0:
                warnings.warn(f"Skipping subject: {subj_id} due to missing nifti or mask files.")
                continue

            if self._signal_clean_info["use_confounds"]:
                if len(confound_files) == 0:
                    warnings.warn(f"Skipping subject: {subj_id} due to missing confound files.")
                    continue
                if len(confound_metadata_files) == 0 and self._signal_clean_info["n_acompcor_separate"]:
                    warnings.warn(f"Skipping subject: {subj_id} due to missing confound metadata to locate the first six components of the white-matter and cerobrospinal fluid masks seperately.")
                    continue
            
            if self._task_info["task"] != "rest" and len(event_files) == 0:
                warnings.warn(f"Skipping subject: {subj_id} due to having no event files.")
                continue
            
            bool_list = []

            # Check if at least one run has all files present
            for run in run_list:
                curr_list = []
                # Assess is any of these returns True
                if self._task_info["task"] != "rest": curr_list.append(any([run in file for file in event_files]))
                if self._signal_clean_info["use_confounds"]:
                    curr_list.append(any([run in file for file in confound_files]))
                    if self._signal_clean_info["n_acompcor_separate"]: curr_list.append(any([run in file for file in confound_metadata_files]))
                curr_list.append(any([run in file for file in mask_files]))
                # Assess if all returns True for a specific run number
                bool_list.append(all(curr_list))
            
            # Skip subject if no run has all needed files present
            if True not in bool_list:
                if self._task_info["task"] != "rest": warnings.warn(f"Skipping subject: {subj_id} due to no nifti file, mask file, confound tsv file, confound json file being from the same run.")
                else: warnings.warn(f"Skipping subject: {subj_id} due to no nifti file, mask file, event file, confound tsv file, confound json file being from the same run.")
                continue
            
            # Add subject list to subject attribute. These are subjects that will be ran
            self._subject_ids.append(subj_id)

            # Get repetition time for the subject
            tr = tr if self._task_info["tr"] else json.load(open(bold_metadata_files[0]))["RepetitionTime"]

            # Store subject specific information
            self._subject_info[subj_id] = {"nifti_files": nifti_files, "event_files": event_files, "confound_files": confound_files, "confound_metadata_files": confound_metadata_files, "mask_files": mask_files,
                                           "tr": tr, "run_list": run_list}

    def timeseries_to_pickle(self, output_dir: str, file_name: str):
        """Save Bold Data

        Saves the timeseries dictionary as a pickle file where columns are the subject ID and indices are the runs. Each cell contains the timeseries array.

        Parameters
        ----------
        output_dir : str
            Directory to save the file to.
        file_name : str
            Name of the file without the "pkl" extension.
        """

        import pickle

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        with open(os.path.join(output_dir,file_name + ".pkl"), "wb") as f:
            pickle.dump(self._subject_timeseries,f)

    def visualize_bold(self, subj_id: Union[int,str], run: int, roi_indx: Union[int, list[int]]=None, network: str=None, show_figs: bool=True, output_dir: str=None, file_name: str=None, **kwargs):
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
        network: str, default=None
            The Schaefer network to plot. If not None, all nodes in the specified Schaefer network will be averaged then plotted. See self.atlas_networks for valid network names.
        show_figs: bool, defaults=True
            Whether to show figires or not to show figures
        output_dir : str
            Directory to save the file to.
        file_name : str
            Name of the file with the extension to signify the file type.
        kwargs: dict
            Keyword arguments used when saving figures. Valid keywords include "dpi" and "figsize". If output_dir is not None and no inputs for dpi and format are given,
            dpi defaults to 300. If "figsize" has no input, figure sizes defaults to (8,6).

        Raises
        ------
        ValueError
            If both `roi_indx` and `network` are specified.
        AssertionError
            If `file_name` does not contain an extension to signify the file type.
        """
    
        import matplotlib.pyplot as plt, numpy as np

        if isinstance(subj_id,int): subj_id = str(subj_id)

        if roi_indx !=None and network != None:
            raise ValueError("`roi_indx` and network can not be used simultaneously.")
        
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            assert "." in file_name, "`file_name` must be specified if `output_dir` is specified and it must contain an extension to signify the file type."

        plot_dict = dict(dpi = kwargs["dpi"] if kwargs and "dpi" in kwargs.keys() else 300,
                         figsize= kwargs["figsize"] if kwargs and "figsize" in kwargs.keys else (15, 5))

        # Obtain the column indices associated with the rois; add logic for roi_indx == 0 since it would be recognized as False
        if roi_indx or roi_indx == 0:
            if type(roi_indx) == int:
                plot_indxs = roi_indx
            
            elif type(roi_indx) == str:
                plot_indxs = self._parcel_approach[self._parcel_approach.keys[0]]["labels"].index(roi_indx)
            
            elif type(roi_indx) == list:
                if type(roi_indx[0]) == int:
                    plot_indxs = np.array(roi_indx)
                elif type(roi_indx[0]) == str:
                    plot_indxs = np.array([self._parcel_approach[self._parcel_approach.keys[0]]["labels"].index(index) for index in roi_indx])
        
        elif network:
            plot_indxs = np.array([index for index, label in enumerate(self._parcel_approach[list(self._parcel_approach.keys())[0]]["labels"]) if network in label])
        
        plt.figure(figsize=plot_dict["figsize"])

        if roi_indx or roi_indx == 0: 
            plt.plot(range(1, self._subject_timeseries[subj_id][f"run-{run}"].shape[0] + 1), self._subject_timeseries[subj_id][f"run-{run}"][:,plot_indxs])
        elif network:  
            plt.plot(range(1, self._subject_timeseries[subj_id][f"run-{run}"].shape[0] + 1), np.mean(self._subject_timeseries[subj_id][f"run-{run}"][:,plot_indxs], axis=1))
            plt.title(network)
        plt.xlabel("TR")

        if output_dir:
            plt.savefig(os.path.join(output_dir,file_name), dpi=plot_dict["dpi"])

        if show_figs == False:
            plt.close()



        
        
        
