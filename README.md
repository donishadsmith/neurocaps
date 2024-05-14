# neurocaps
This is a Python package to perform Co-activation Patterns (CAPs) analyses, which involves using kmeans clustering to group timepoints (TR's) into brain states, on both resting-state or task data. It is compatible with data preprocessed with **fMRIPrep** and assumes your directory is BIDS-compliant and contains a derivatives folder with a pipeline folder, such as fMRIPrep, containing preprocessed BOLD data.

**Still in beta but stable.**

# Installation

**Note**: The `.get_bold()` method in the `TimeseriesExtractor` class in this package uses pybids, which is only functional on POSIX operating system and Mac OS. Assuming you have a pickled timeseries dictionary in the proper nested form, this package can be used on windows to visualize the bold timeseries, use the `CAP` class, and the `merge_dicts()` fuction. 

To install, using your preferred terminal:

**Installation with pip:**

```bash

pip install neurocaps

```

**From source (Development version):**

```bash
pip install git+https://github.com/donishadsmith/neurocaps.git
```

or

```bash

git clone https://github.com/donishadsmith/neurocaps/
cd neurocaps
pip install -e .

```

# Usage
**This package contains two main classes - `TimeseriesExtractor`, for extracting the timeseries, and `CAP`, for performing the CAPs analysis.**

**Note**: When extracting the timeseries, **this package uses the Schaefer atlas, the Automated Anatomical Labeling (AAL) atlas, or a custom parcellation where all nodes has a left and right version**. The number of ROIs and networks for the Schaefer atlas can be modified with the `parcel_approach` parameter when initializing the main `TimeseriesExtractor` class. To modify it, you must use a nested dictionary, where the primary key is "Schaefer" and the sub-keys are "n_rois" and "yeo_networks". Example: `parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm: 1}}`. Similary, the version of the AAL atlas can be modified using `parcel_approach = {"AAL": {"version": "SPM12"}}`. 

If using a "Custom" parcellation approach, ensure each node in your dataset includes both left (lh) and right (rh) hemisphere versions. 

Custom Key Structure:
- maps: Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g., .nii for NIfTI files). For plotting purposes, this key is not required.
- nodes:  list of all node labels used in your study, arranged in the exact order they correspond to indices in your parcellation files. 
Each label should match the parcellation index it represents. For example, if the parcellation label "0" corresponds to the left hemisphere 
visual cortex area 1, then "LH_Vis1" should occupy the 0th index in this list. This ensures that data extraction and analysis accurately reflect the anatomical regions intended. For timeseries extraction, this key is not required.
- regions: Dictionary defining major brain regions. Each region should list node indices under "lh" and "rh" to specify left and right hemisphere nodes. For    timeseries extraction, this key is not required.
        
Example 
The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis) and hippocampus regions:

```Python
parcel_approach= {"Custom": {"maps": "/location/to/parcellation.nii.gz",
                             "nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus", "RH_Vis1", "RH_Vis2", "RH_Hippocampus"],
                             "regions": {"Vis" : {"lh": [0,1],
                                                  "rh": [3,4]},
                                         "Hippocampus": {"lh": [2],
                                                         "rh": [5]}}}}
 ```

**Main features for `TimeseriesExtractor` includes:**

- Timeseries extraction for resting state or task data and creating a nested dictionary containing the subject ID, run number, and associated timeseries. This is used as input for the `get_caps()` method in the `CAP` class.
- Saving the nested dictionary containing timeseries as a pickle file.
- Visualizing the timeseries of a Schaefer or AAL node or network subject's run. Also includes the ability to save plots.
- Ability to use parallel processing by specifiying the number of CPU cores to use in the `n_cores` parameter in the `get_bold()` method. Testing on an HPC using a loop with `TimeseriesExtractor.get_bold()` to extract the session 1 and 2 bold timeries from 105 subjects from resting-state data (single-run containing 360 volumes) and two task datasets (three-runs containing 200 volumes each and two-runs containing 200 volumes) reduced processing time from 5 hrs 48 mins to 1 hr 26 mins (using 10 cores).

**Main features for `CAP` includes:**

- Performing the silhouette or elbow method to identify the optimal cluster size. When the optimal cluster size is identified, the optimal model is saved as an attribute.
- Visualizing the CAPs identified as an outer product or regular heatmap. For outer products, you also have the ability to use subplots to reduce the number of individual plots. You can also save the plots and use them. Please refer to the docstring for the `visualize_caps()` method in the `CAP` class to see the list of available kwargs arguments to modify plots.
- Grouping feature to perform CAPs independently on groups of subject IDs. When grouping is specified, k-means clustering, silhouette and elbow methods, as well as plotting, are done for each independent group.
- Calculating CAP metrics as described in [Liu et al., 2018](https://doi.org/10.1016/j.neuroimage.2018.01.041)[^1] and [Yang et al., 2021](https://doi.org/10.1016/j.neuroimage.2021.118193)[^2], where *temporal fraction* is the proportion of total volumes spent in a single CAP over all volumes in a run, *persistence* is the average time spent in a single CAP before transitioning to another CAP (average consecutive/uninterrupted time), and *counts* is the frequency of each CAP observed in a run, and *transition frequency* is the number of switches between
different CAPs across the entire run.

**Additionally, the `neurocaps.analysis` submodule contains the `merge_dicts` function, which allows you to merge the subject_timeseries dictionaries (merged dictionary will be returned and can be saved as a pickle file) for overlapping subjects across tasks in order to identify similar CAPs across different tasks[^3].**

Please refer to [demo.ipynb](https://github.com/donishadsmith/neurocaps/blob/main/demo.ipynb) to see a more extensive demonstration of the features included in this package. 

Quick code example:

```python
# Examples use randomized data

from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP

"""If an asterisk '*' is after a name, all confounds starting with the 
term preceding the parameter will be used. in this case, all parameters 
starting with cosine will be used."""
confounds = ["cosine*", "trans_x", "trans_x_derivative1", "trans_y", 
             "trans_y_derivative1", "trans_z","trans_z_derivative1", 
             "rot_x", "rot_x_derivative1", "rot_y", "rot_y_derivative1", 
             "rot_z","rot_z_derivative1"]

"""If use_confounds is True but no confound_names provided, there are hardcoded 
confound names that will extract the data from the confound files outputted by fMRIPrep
`n_acompcor_separate` will use the first 'n' components derived from the separate 
white-matter (WM) and cerebrospinal fluid (CSF). To use the acompcor components from the 
combined mask, list them in the `confound_names` parameter"""
parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 2}}

extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                 use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01, 
                                 confound_names=confounds, n_acompcor_separate=6)

bids_dir = "/path/to/bids/dir"

# If there are multiple pipelines in the derivatives folder, you can specify a specific pipeline
pipeline_name = "fmriprep-1.4.0"

# Resting State
# extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

# Task; use parallel processing with `n_cores`
extractor.get_bold(bids_dir=bids_dir, task="emo", condition="positive", 
                   pipeline_name=pipeline_name, n_cores=10)

cap_analysis = CAP(parcel_approach=extractor.parcel_approach,
                    n_clusters=6)

cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, 
                      standardize = True)

# Visualize CAPs
cap_analysis.visualize_caps(visual_scope="regions", plot_options="outer product", 
                            task_title="- Positive Valence", ncol=3, sharey=True, 
                            subplots=True)

cap_analysis.visualize_caps(visual_scope="nodes", plot_options="outer product", 
                            task_title="- Positive Valence", ncol=3,sharey=True, 
                            subplots=True, xlabel_rotation=90, tight_layout=False, 
                            hspace = 0.4)

```
**Graph Outputs:**
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/4699bbd9-1f55-462b-9d9e-4ef17da79ad4)
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/506c5be5-540d-43a9-8a61-c02062f5c6f9)

```python

# Get CAP metrics
outputs = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, tr=2.0, 
                                         return_df=True, output_dir=output_dir,
                                         metrics=["temporal fraction", "persistence"],
                                         continuous_runs=True, file_name="All_Subjects_CAPs_metrics")

print(outputs["temporal fraction"])
```
**DataFrame Output:**
| Subject_ID | Group | Run | CAP-1 | CAP-2 | CAP-3 | CAP-4 | CAP-5 | CAP-6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | All Subjects | Continuous Runs | 0.14 | 0.17 | 0.14 | 0.2 | 0.15 | 0.19 |
| 2 | All Subjects | Continuous Runs | 0.17 | 0.17 | 0.16 | 0.16 | 0.15 | 0.19 |
| 3 | All Subjects | Continuous Runs | 0.15 | 0.2 | 0.14 | 0.18 | 0.17 | 0.17 |
| 4 | All Subjects | Continuous Runs | 0.17 | 0.21 | 0.18 | 0.17 | 0.1 | 0.16 |
| 5 | All Subjects | Continuous Runs | 0.14 | 0.19 | 0.14 | 0.16 | 0.2 | 0.18 |
| 6 | All Subjects | Continuous Runs | 0.16 | 0.21 | 0.16 | 0.18 | 0.16 | 0.13 |
| 7 | All Subjects | Continuous Runs | 0.16 | 0.16 | 0.17 | 0.15 | 0.19 | 0.17 |
| 8 | All Subjects | Continuous Runs | 0.17 | 0.21 | 0.13 | 0.14 | 0.17 | 0.18 |
| 9 | All Subjects | Continuous Runs | 0.18 | 0.1 | 0.17 | 0.18 | 0.16 | 0.2 |
| 10 | All Subjects | Continuous Runs | 0.14 | 0.19 | 0.14 | 0.17 | 0.19 | 0.16 |

```python
# If you experience coverage issues, usually smoothing helps to mitigate these issues
cap_analysis.caps2surf(fwhm=2)
```
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/46ea5174-0ded-4640-a1f9-c21e798e0459)

# Testing 
This package was tested using a closed dataset as well as a modified version of a single subject open dataset to test the TimeseriesExtractor function on Github Actions. Furthermore, the open dataset provided by [Laumann & Poldrack](https://openfmri.org/dataset/ds000031/) and used in [Laumann et al., 2015](https://doi.org/10.1016/j.neuron.2015.06.037)[^4]. Additionally, this data was obtained from the OpenfMRI database. Its accession number is ds000031. Modifications to the data consist of truncating the preprocessed bold data and confounds form 448 timepoints to 34 timepoints, only including session 002 data, adding a dataset_description.json file to the fmriprep folder, excluding the nii.gz file in the root bids folder, only retaining the mask, truncated preprocessed bold file, and truncated confounds file in the fmriprep folder, and slighly changing the naming style of the mask, preprocessed bold file, and confounds file in the fmriprep folder to conform with the naming conventions of modern fmriprep outputs.

Testing with custom parcellations was done with the HCPex parcellation, an extension of the HCP (Human Connectome Project) parcellation, which adds 66 subcortical areas [^5], [^6]. This original atlas can be downloaded from https://github.com/wayalan/HCPex.


# References
[^1]: Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals. NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

[^2]: Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia. NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

[^3]: Kupis, L., Romero, C., Dirks, B., Hoang, S., Parladé, M. V., Beaumont, A. L., Cardona, S. M., Alessandri, M., Chang, C., Nomi, J. S., & Uddin, L. Q. (2020). Evoked and intrinsic brain network dynamics in children with autism spectrum disorder. NeuroImage: Clinical, 28, 102396. https://doi.org/10.1016/j.nicl.2020.102396

[^4]: Laumann, T. O., Gordon, E. M., Adeyemo, B., Snyder, A. Z., Joo, S. J., Chen, M. Y., Gilmore, A. W., McDermott, K. B., Nelson, S. M., Dosenbach, N. U., Schlaggar, B. L., Mumford, J. A., Poldrack, R. A., & Petersen, S. E. (2015). Functional system and areal organization of a highly sampled individual human brain. Neuron, 87(3), 657–670. https://doi.org/10.1016/j.neuron.2015.06.037

[^5]: Huang CC, Rolls ET, Feng J, Lin CP. An extended Human Connectome Project multimodal parcellation atlas of the human cortex and subcortical areas. Brain Struct Funct. 2022 Apr;227(3):763-778. Epub 2021 Nov 17. doi: 10.1007/s00429-021-02421-6

[^6]: Huang CC, Rolls ET, Hsu CH, Feng J, Lin CP. Extensive Cortical Connectivity of the Human Hippocampal Memory System: Beyond the "What" and "Where" Dual Stream Model. Cerebral Cortex. 2021 May 19;bhab113. doi: 10.1093/cercor/bhab113.
