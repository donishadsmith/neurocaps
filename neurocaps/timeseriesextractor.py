import re, os
from typing import Union
from .getters import _TimeseriesExtractorGetter

class TimeseriesExtractor(_TimeseriesExtractorGetter):
    def __init__(self, space: str="MNI152NLin2009cAsym", standardize: Union[bool,str]=False, detrend: bool=False , low_pass: float=None, high_pass: float=None, n_rois: int=400, n_networks: int=7, use_confounds: bool=True, confound_names: list[str]=None, discard_volumes: int=None):
        """Timeseries Extractor Class
        
        Initializes a TimeseriesExtractor to prepare for Co-activation Patterns (CAPs) analysis.

        Parameters
        ----------
        space : str, default="MNI152NLin2009cAsym"
            The brain template space data is in. 
        standardize : bool, default=False
            Determines whether to standardize the timeseries. 
        detrend : bool, default=True
            Detrends timeseries during extraction.
        low_pass : bool, default=None
            Signals above cutoff frequency will be filtered out.
        high_pass : float, default=None
            Signals below cutoff frequency will be filtered out.
        discard_volumes : float, default=None
            Remove the first `n` number of volumes before extracting timeseries.
        n_rois : int, default=400
            The number of regions of interest (ROIs) to use for Schaefer parcellation.
        use_confounds : bool, default=True
            To use confounds when extracting timeseries.
        confound_names : List[str], default=None
            Names of confounds to use in confound files. If None, default confounds are used.
        """
        self._space = space
        self._standardize = standardize
        self._n_rois = n_rois
        self._n_networks = n_networks
        self._use_confounds = use_confounds
        self._detrend = detrend
        self._low_pass = low_pass
        self._high_pass = high_pass
        self._discard_volumes = discard_volumes

        if self._use_confounds:
            if confound_names == None:
                # Hardcoded confound names
                self._confound_names = [
                    "cosine00", "cosine01", "cosine02",
                    "trans_x", "trans_x_derivative1", "trans_x_power2", "trans_x_derivative1_power2",
                    "trans_y", "trans_y_derivative1", "trans_y_derivative1_power2", "trans_y_power2",
                    "trans_z", "trans_z_derivative1", "trans_z_power2", "trans_z_derivative1_power2",
                    "rot_x", "rot_x_derivative1", "rot_x_power2", "rot_x_derivative1_power2",
                    "rot_y", "rot_y_derivative1", "rot_y_power2", "rot_y_derivative1_power2",
                    "rot_z", "rot_z_derivative1", "rot_z_derivative1_power2", "rot_z_power2",
                    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02", "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05", "a_comp_cor_06"
                ]
            else:
                self._confound_names = confound_names
        
        
        assert type(self._confound_names) == list and len(self._confound_names) > 0 , "confound_names must be a non-empty list"

        # Get atlas
        from nilearn import datasets
        self._atlas = datasets.fetch_atlas_schaefer_2018(n_rois=self._n_rois, yeo_networks=self._n_networks)
        self._atlas_labels = [label.decode().split("7Networks_")[-1]  for label in self._atlas.labels]

        # Get node networks
        self._atlas_networks = list(dict.fromkeys([re.split("LH_|RH_", node)[-1].split("_")[0] for node in self._atlas_labels]))

    def get_bold(self, bids_dir: str, session: int, runs: list[int]=None, task: str="rest", condition: str=None, tr: Union[int, float]=None, run_subjects: list[str]=None, pipeline_name: str=None) -> None: 
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
        pipeline_name:str, default=None
            The name of the pipeline folder in the derivatives folder containing the preprocessed data. If None, BIDSLayout will use the name of dset_dir with derivatives=True. This parameter
            should be used if their are multiple pipelines in the derivatives folder.
        """
        import bids, json

        # Update attributes
        self._task = task
        self._condition = condition
        self._session = session
        self._run = runs
        if tr:
            self._tr = tr

        # Intiialize new attributes
        self._subject_ids = []
        self._subject_timeseries = {}

        if pipeline_name:
            layout = bids.BIDSLayout(bids_dir, derivatives=os.path.join(bids_dir, "derivatives", pipeline_name))
        else:
            layout = bids.BIDSLayout(bids_dir, derivatives=True)

        print(f"Bids layout collected.")

        subj_id_list = sorted(layout.get(return_type="id", target="subject", task=task, space=self._space, suffix="bold")) 

        if run_subjects: subj_id_list = sorted([subj_id for subj_id in subj_id_list if subj_id in run_subjects])

        for subj_id in subj_id_list:

            nifti_files = sorted(layout.get(scope="derivatives", return_type="file",suffix="bold", task=task, space=self._space, session=session,extension = "nii.gz", subject=subj_id))
            json_files = sorted(layout.get(scope="derivatives", return_type="file",suffix="bold", task=task, space=self._space, session=session, extension = "json"))
            event_files = None if task == "rest" else sorted([file for file in sorted(layout.get(return_type="filename",suffix="events", task=task, session=session,extension = "tsv", subject = subj_id))])
            confound_files = sorted(layout.get(scope='derivatives', return_type='file', desc='confounds', task=task, session=session,extension = "tsv", subject=subj_id))
            mask_files = sorted(layout.get(scope='derivatives', return_type='file', suffix='mask', task=task, space=self._space, session=session, extension = "nii.gz", subject=subj_id))
            # Generate a list of runs to iterate through based on runs in nifti_files
            run_list = [f"run-{run}" for run in runs] if runs else [re.search("run-(\d+)",x)[0] for x in nifti_files]

            if len(nifti_files) == 0 or len(mask_files) == 0:
                print(f"Skipping subject: {subj_id} due to missing nifti or mask files.")
                continue

            if self._use_confounds:
                if  len(confound_files) == 0:
                    print(f"Skipping subject: {subj_id} due to missing confound files.")
                    continue
            
            if task != "rest" and len(event_files) == 0:
                print(f"Skipping subject: {subj_id} due to having no event files.")
                continue
            
            # Add subject list to subject attribute
            self._subject_ids.extend(subj_id)

            # Get repetition time for the subject
            tr = tr if tr else json.load(open(json_files[0]))["RepetitionTime"]

            self._extract_timeseries(subj_id=subj_id, mask_files=mask_files, nifti_files=nifti_files, event_files=event_files, confound_files=confound_files,
                                    run_list=run_list, condition=condition, tr=tr)
        
    def _extract_timeseries(self, subj_id, nifti_files, mask_files, event_files, confound_files, run_list, tr, condition=None):

        from nilearn.maskers import NiftiLabelsMasker
        from nilearn.image import index_img, load_img
        import pandas as pd 

        # Intitialize dictionary; Current subject will alway be the last subject in the subjects attribute
        subject_timeseries = {subj_id: {}}
        
        for run in run_list:

            # Get files from specific run
            nifti_file = [nifti_file for nifti_file in nifti_files if run in nifti_file]
            mask_file = [mask_file for mask_file in mask_files if run in mask_file]
            confound_file = [confound_file for confound_file in confound_files if run in confound_file] if self._use_confounds else None

            print(f"Running subject: {subj_id}; {run}; \n {nifti_file}")

            if len(nifti_file) == 0 or len(mask_file) == 0:
                print(f"Skipping subject: {subj_id}; {run} do to missing nifti or mask file.")
                continue
            
            if self._use_confounds:
                if len(confound_file) == 0:
                    print(f"Skipping subject: {subj_id}; {run} do to missing confound file.")
                    continue

            confound_df = pd.read_csv(confound_file[0], sep=None) if self._use_confounds else None

            event_file = None if event_files == None else [event_file for event_file in event_files if run in event_file]

            if event_file:
                if len(event_file) == 0:
                    print(f"Skipping subject: {subj_id}; {run} do to missing event file.")
                    continue
                else:
                    event_df = pd.read_csv(event_file[0], sep=None)

            # Extract confound information of interest and nsure confound file does not contain NAs
            if self._use_confounds:
                confounds = confound_df[[col for col in confound_df if col in self._confound_names]]
                confounds = confounds.fillna(0)
            
            # Create the masker for extracting time series
            masker = NiftiLabelsMasker(
                mask_img=mask_file[0],
                labels_img=self._atlas.maps, 
                labels=self._atlas.labels, 
                resampling_target='data',
                standardize=self._standardize,
                detrend=self._detrend,
                low_pass=self._low_pass,
                high_pass=self._high_pass,
                t_r=tr
            )
            # Load and discard volumes if needed
            nifti_img = load_img(nifti_file[0])
            if self._discard_volumes: 
                nifti_img = index_img(nifti_img, slice(self._discard_volumes, None))
                confounds.drop(list(range(0,self._discard_volumes)),axis=0,inplace=True)

            # Extract timeseries
            timeseries = masker.fit_transform(nifti_img, confounds=confounds) if self._use_confounds else masker.fit_transform(nifti_img)

            if event_file:
                # Get specific timing information for specific condition
                condition_df = event_df[event_df["trial_type"] == condition] if condition else event_df

                # Empty list for scans
                scan_list = []

                # Convert times into scan numbers to obtain the scans taken when the participant was exposed to the condition of interest; round to nearest whole number
                for i in condition_df.index:
                    onset_scan, duration_scan = int(condition_df.loc[i,"onset"]/tr), int((condition_df.loc[i,"onset"] + condition_df.loc[i,"duration"])/tr)
                    scan_list.extend(range(onset_scan, duration_scan + 1))

                # Timeseries with the extracted scans corresponding to condition; set is used to remove overlapping TRs    
                timeseries = timeseries[sorted(list(set(scan_list)))]
    

            subject_timeseries[subj_id].update({run: timeseries})


        # Aggregate new timeseries
        self._subject_timeseries.update(subject_timeseries)

    def timeseries_to_pickle(self, output_dir: str, file_name: str):
        """Save Bold Data

        Saves the timeseries dictionary as a pickle file where columns are the subject ID and indices are the runs. Each cell contains the timeseries array.

        Parameters
        ----------
        output_dir : str
            Directory to save the file to.
        file_name : str
            Name of the file without the extension the extension.
        """

        import pickle

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        with open(os.path.join(output_dir,file_name + ".pkl"), "wb") as f:
            pickle.dump(self._subject_timeseries,f)

    def visualize_bold(self, subj_id: str, run: int, roi_indx: Union[int, list[int]]=None, network: str=None, show_figs: bool=True, output_dir: str=None, file_name: str=None, **kwargs):
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

        if roi_indx !=None and network != None:
            raise ValueError("`roi_indx` and network can not be used simultaneously.")
        

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            assert "." in file_name, "`file_name` must be specified if `output_dir` is specified and it must contain an extension to signify the file type."

        if kwargs:
                dpi = kwargs["dpi"] if "dpi" in kwargs.keys() else 300
                figsize= kwargs["figsize"] if "figsize" in kwargs.keys else (15, 5)
        else:
            dpi = 300
            figsize = (15, 5)

        # Obtain the column indices associated with the rois; add logic for roi_indx == 0 since it would be recognizes as False
        if roi_indx or roi_indx == 0:
            if type(roi_indx) == int:
                plot_indxs = roi_indx
            
            elif type(roi_indx) == str:
                plot_indxs = self._atlas_labels.index(roi_indx)
            
            elif type(roi_indx) == list:
                if type(roi_indx[0]) == int:
                    plot_indxs = np.array(roi_indx)
                elif type(roi_indx[0]) == str:
                    plot_indxs = np.array([self._atlas_labels.index(index) for index in roi_indx])
        
        elif network:
            plot_indxs = np.array([index for index, node in enumerate(self._atlas_labels) if network in node])
        

        plt.figure(figsize=figsize)

        if roi_indx or roi_indx == 0: 
            plt.plot(range(1, self._subject_timeseries[subj_id][f"run-{run}"].shape[0] + 1), self._subject_timeseries[subj_id][f"run-{run}"][:,plot_indxs])
        elif network:  
            plt.plot(range(1, self._subject_timeseries[subj_id][f"run-{run}"].shape[0] + 1), np.mean(self._subject_timeseries[subj_id][f"run-{run}"][:,plot_indxs], axis=1))
            plt.title(network)
        plt.xlabel("TR")

        if output_dir:
            plt.savefig(os.path.join(output_dir,file_name), dpi=dpi)

        if show_figs == False:
            plt.close()



        
        
        
