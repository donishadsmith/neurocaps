# neurocaps
[![Latest Version](https://img.shields.io/pypi/v/neurocaps.svg)](https://pypi.python.org/pypi/neurocaps/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.11642615-blue)](https://doi.org/10.5281/zenodo.12682514)
[![Test Status](https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This is a Python package designed to perform Co-activation Patterns (CAPs) analyses. It utilizes k-means clustering to group timepoints (TRs) into brain states, applicable to both resting-state and task-based fMRI data. The package is compatible with data preprocessed using **fMRIPrep** and assumes your directory is BIDS-compliant, containing a derivatives folder with a pipeline folder (such as fMRIPrep) that holds the preprocessed BOLD data.

# Installation

**Note**: The `get_bold()` method in the `TimeseriesExtractor` class relies on pybids, which is only functional on POSIX operating systems and macOS. If you have a pickled timeseries dictionary in the correct nested form, you can use this package on Windows to visualize the BOLD timeseries, the `CAP` class, as well as the `merge_dicts()` and `standardize()` functions in the in the `neurcaps.analysis` submodule.

To install, use your preferred terminal:

**Installation using pip:**

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
**Note, documentation of each function can be found at https://neurocaps.readthedocs.io/en/latest/api.html**

**This package contains two main classes: `TimeseriesExtractor` for extracting the timeseries, and `CAP` for performing the CAPs analysis.**

**Note:** When extracting the timeseries, this package uses either the Schaefer atlas, the Automated Anatomical Labeling (AAL) atlas, or a custom parcellation where all nodes have a left and right version (bilateral nodes). The number of ROIs and networks for the Schaefer atlas can be adjusted with the parcel_approach parameter when initializing the `TimeseriesExtractor` class.

To modify it, you must use a nested dictionary, where the primary key is "Schaefer" and the sub-keys are "n_rois" and "yeo_networks". For example:

```python
parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 1}}
```

Similarly, the version of the AAL atlas can be modified using:

```python
parcel_approach = {"AAL": {"version": "SPM12"}}
```

If using a "Custom" parcellation approach, ensure each region in your dataset includes both left (lh) and right (rh) hemisphere versions of nodes (bilateral nodes). 

Custom Key Structure:
- `"maps'`: Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g., .nii for NifTI files). For plotting purposes, this key is not required.
- `"nodes"`: List of all node labels used in your study, arranged in the exact order they correspond to indices in your parcellation files. 
Each label should match the parcellation index it represents. For example, if the parcellation label "0" corresponds to the left hemisphere 
visual cortex area 1, then "LH_Vis1" should occupy the 0th index in this list. This ensures that data extraction and analysis accurately reflect the anatomical regions intended. For timeseries extraction, this key is not required.
- `"regions"`: Dictionary defining major brain regions. Each region should list node indices under "lh" and "rh" to specify left and right hemisphere nodes. For timeseries extraction, this key is not required.
        
Example:
The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis) and hippocampus regions:

```Python
        parcel_approach = {
            "Custom": {
                "maps": "/location/to/parcellation.nii.gz",
                "nodes": [
                    "LH_Vis1",
                    "LH_Vis2",
                    "LH_Hippocampus",
                    "RH_Vis1",
                    "RH_Vis2",
                    "RH_Hippocampus"
                ],
                "regions": {
                    "Vis": {
                        "lh": [0, 1],
                        "rh": [3, 4]
                    },
                    "Hippocampus": {
                        "lh": [2],
                        "rh": [5]
                    }
                }
            }
        }
 ```

**Main features for `TimeseriesExtractor` includes:**

- **Timeseries Extraction:** Extract timeseries for resting-state or task data, creating a nested dictionary containing the subject ID, run number, and associated timeseries. This serves as input for the `get_caps()` method in the `CAP` class.
- **Saving Timeseries:** Save the nested dictionary containing timeseries as a pickle file.
- **Visualization:** Visualize the timeseries of a Schaefer, AAL, or Custom parcellation node or region/network in a specific subject's run, with options to save the plots.
- **Parallel Processing:** Use parallel processing by specifying the number of CPU cores in the `n_cores` parameter in the `get_bold()` method. Testing on an HPC using a loop with `TimeseriesExtractor.get_bold()` to extract session 1 and 2 BOLD timeseries from 105 subjects from resting-state data (single run containing 360 volumes) and two task datasets (three runs containing 200 volumes each and two runs containing 200 volumes) reduced processing time from 5 hours 48 minutes to 1 hour 26 minutes (using 10 cores). *Note:* If you are using an HPC, remember to allocate the appropriate amount of CPU cores with your workload manager. For instance in slurm use `#SBATCH --cpus-per-task=10` if you intend to use 10 cores.

**Main features for `CAP` includes:**
- **Optimal Cluster Size Identification:** Perform the Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions to identify the optimal cluster size, saving the optimal model as an attribute.
- **Parallel Processing:** Use parallel processing, when using the Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions , by specifying the number of CPU cores in the `n_cores` parameter in the `get_caps()` method. *Note:* If you are using an HPC, remember to allocate the appropriate amount of CPU cores with your workload manager. For instance in slurm use `#SBATCH --cpus-per-task=10` if you intend to use 10 cores.
- **Grouping:** Perform CAPs analysis for entire sample or groups of subject IDs (using the `groups` parameter when initializing the `CAP` class). K-means clustering, all cluster selection methods (Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions), and plotting are done for each group when specified.
- **CAP Visualization:** Visualize the CAPs as outer products or heatmaps, with options to use subplots to reduce the number of individual plots, as well as save. Refer to the [documentation](https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2plot) for the `caps2plot()` method in the `CAP` class for available `**kwargs` arguments and parameters to modify plots.
- **Save CAPs as NifTIs:** Convert the atlas used for parcellation to a stat map and saves them (`caps2niftis`). 
- **Surface Plot Visualization:** Convert the atlas used for parcellation to a stat map projected onto a surface plot with options to customize and save plots. Refer to the [documentation](https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2surf) for the `caps2surf()` method in the `CAP` class for available `**kwargs` arguments and parameters to modify plots. Also includes the option to save the NifTIs. There is also another a parameter in `caps2surf`, `fslr_giftis_dict`, which can be used if the CAPs NifTI files were converted to GifTI files using a tool such as Connectome Workbench, which may work better for converting your atlas to fslr space. This parameter allows plotting without re-running the analysis and only initializing the `CAP` class and using the `caps2surf` method is needed.
- **Correlation Matrix Creation:** Create a correlation matrix from CAPs with options to customize and save plots. Additionally can produce dataframes where each element contains its associated uncorrected p-value in parentheses that is accompanied by an asterisk using the following significance code `{"<0.05": "*", "<0.01": "**", "<0.001": "***"}`. Refer to the [documentation](https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2corr) for the `caps2corr()` method in the `CAP` class for available `**kwargs` arguments and parameters to modify plots.
- **CAP Metrics Calculation:** Calculate CAP metrics (`calculate_metrics()`) as described in [Liu et al., 2018](https://doi.org/10.1016/j.neuroimage.2018.01.041)[^1] and [Yang et al., 2021](https://doi.org/10.1016/j.neuroimage.2021.118193)[^2]:
    - *Temporal Fraction:* The proportion of total volumes spent in a single CAP over all volumes in a run.
        ```python
            predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
            target = 1
            temporal_fraction = 4/6
        ```
    - *Persistence:* The average time spent in a single CAP before transitioning to another CAP (average consecutive/uninterrupted time).
        ```python
            predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
            target = 1
            # Sequences for 1 are [1] and [1,1,1]
            persistence = (1 + 3)/2 # Average number of frames
            tr = 2
            if tr:
                persistence = ((1 + 3) * 2)/2 # Turns average frames into average time
        ```
    - *Counts:* The frequency of each CAP observed in a run.
        ```python
            predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
            target = 1
            counts = 4
        ```
    - *Transition Frequency:* The number of switches between different CAPs across the entire run.
        ```python
            predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
            # Transitions between unique CAPs occur at indices 0 -> 1, 1 -> 2, and 4 -> 5
            transition_frequency = 3
        ```
- **Cosine Similarity Radar Plots:** Create radar plots showing the cosine similarity between CAPs and regions. Especially useful as a quantitative method to categorize CAPs by determining the regions containing the most nodes demonstrating increased co-activation or decreased co-deactivation [^3]. Refer to the [documentation](https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2radar) in `caps2radar` in the `CAP` class for a more detailed explanation as well as available `**kwargs` arguments and parameters to modify plots. **Note**, the "Low Amplitude"are negative cosine similarity values. The absolute value of those cosine similarities are taken so that the radar plot starts at 0 and magnitude comparisons between the "High Amplitude" and "Low Amplitude" groups are easier to see. Below is an example of how the cosine similarity is calculated for this function.
    ```python
        import numpy as np
        # Nodes in order of their label ID, "LH_Vis1" is the 0th index in the parcellation
        # but has a label ID of 1, and RH_SomSot2 is in the 7th index but has a label ID
        # of 8 in the parcellation.
        nodes = ["LH_Vis1", "LH_Vis2", "LH_SomSot1", "LH_SomSot2",
                    "RH_Vis1", "RH_Vis2", "RH_SomSot1", "RH_SomSot2"]
        # Binary representation of the nodes in Vis, essentially acts as
        # a mask isolating the modes for for Vis
        binary_vector = [1,1,0,0,1,1,0,0]
        # Cluster centroid for CAP 1
        cap_1_cluster_centroid = [-0.3, 1.5, 2, -0.2, 0.7, 1.3, -0.5, 0.4]
        # Dot product is the sum of all the values here [-0.3, 1.5, 0, 0, 0.7, 1.3, 0, 0]
        dot_product = np.dot(cap_1_cluster_centroid, binary_vector)

        norm_cap_1_cluster_centroid = np.linalg.norm(cap_1_cluster_centroid)
        norm_binary_vector = np.linalg.norm(binary_vector)
        # Cosine similarity between CAP 1 and the visual network
        cosine_similarity = dot_product/(norm_cap_1_cluster_centroid * norm_binary_vector)
    ```

**Additionally, the `neurocaps.analysis` submodule contains two additional functions:**

- `merge_dicts`: Merge the subject_timeseries dictionaries for overlapping subjects across tasks to identify similar CAPs across different tasks. The merged dictionary can be saved as a pickle file.
- `standardize`: Standardizes each run independently for all subjects in the subject timeseries.
- `change_dtype`: Changes the dtype of all subjects in the subject timeseries to help with memory usage.

Please refer to [demo.ipynb](https://github.com/donishadsmith/neurocaps/blob/main/demo.ipynb) or https://neurocaps.readthedocs.io/en/latest/examples/examples.html for a more extensive demonstration of the features included in this package.

**Quick Code Examples (Examples use randomized data)**:

```python

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
extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

# Task; use parallel processing with `n_cores`
extractor.get_bold(bids_dir=bids_dir, task="emo", condition="positive", 
                   pipeline_name=pipeline_name, n_cores=10)

cap_analysis = CAP(parcel_approach=extractor.parcel_approach)

cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, 
                      n_clusters=6,
                      standardize = True)

# Visualize CAPs
# You can use seaborn's premade palettes as strings or generate your own custom palettes
# Using seaborn's diverging_palette function, matplotlib's LinearSegmentedColormap, 
# or other Classes or functions compatable with seaborn

cap_analysis.caps2plot(visual_scope="regions", plot_options="outer_product", 
                       suffix_title="- Positive Valence", ncol=3, sharey=True, 
                       subplots=True, cmap="coolwarm")
# Create the colormap
import seaborn as sns
palette = sns.diverging_palette(260, 10, s=80, l=55, n=256, as_cmap=True)

cap_analysis.caps2plot(visual_scope="nodes", plot_options="outer_product", 
                       suffix_title="- Positive Valence", ncol=3,sharey=True, 
                       subplots=True, xlabel_rotation=90, tight_layout=False, 
                       hspace = 0.4, cmap=palette)

```
**Plot Outputs:**
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/e1ab0f55-0c4c-4701-8f3a-838c2470d44d)
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/43e46a0a-8721-4df9-88fa-04758a34142e)

```python

# Get CAP metrics
outputs = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, tr=2.0, 
                                         return_df=True, output_dir=output_dir,
                                         metrics=["temporal_fraction", "persistence"],
                                         continuous_runs=True, file_name="All_Subjects_CAPs_metrics")

print(outputs["temporal_fraction"])
```
**DataFrame Output:**
| Subject_ID | Group | Run | CAP-1 | CAP-2 | CAP-3 | CAP-4 | CAP-5 | CAP-6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | All_Subjects | continuous_runs | 0.14 | 0.17 | 0.14 | 0.2 | 0.15 | 0.19 |
| 2 | All_Subjects | continuous_runs | 0.17 | 0.17 | 0.16 | 0.16 | 0.15 | 0.19 |
| 3 | All_Subjects | continuous_runs | 0.15 | 0.2 | 0.14 | 0.18 | 0.17 | 0.17 |
| 4 | All_Subjects | continuous_runs | 0.17 | 0.21 | 0.18 | 0.17 | 0.1 | 0.16 |
| 5 | All_Subjects | continuous_runs | 0.14 | 0.19 | 0.14 | 0.16 | 0.2 | 0.18 |
| 6 | All_Subjects | continuous_runs | 0.16 | 0.21 | 0.16 | 0.18 | 0.16 | 0.13 |
| 7 | All_Subjects | continuous_runs | 0.16 | 0.16 | 0.17 | 0.15 | 0.19 | 0.17 |
| 8 | All_Subjects | continuous_runs | 0.17 | 0.21 | 0.13 | 0.14 | 0.17 | 0.18 |
| 9 | All_Subjects | continuous_runs | 0.18 | 0.1 | 0.17 | 0.18 | 0.16 | 0.2 |
| 10 | All_Subjects | continuous_runs | 0.14 | 0.19 | 0.14 | 0.17 | 0.19 | 0.16 |

```python
# Create surface plots of CAPs; there will be as many plots as CAPs
# If you experience coverage issues, usually smoothing helps to mitigate these issues
cap_analysis.caps2surf(fwhm=2, cmap="cold_hot", layout="row",  size=(500, 100), zoom=1, cbar_location="bottom")

#You can also generate your own colormaps using matplotlib's LinearSegmentedColormap

# Create the colormap
from matplotlib.colors import LinearSegmentedColormap
colors = ["#1bfffe", "#00ccff", "#0099ff", "#0066ff", "#0033ff", "#c4c4c4",
          "#ff6666", "#ff3333", "#FF0000","#ffcc00","#FFFF00"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cold_hot", colors, N=256)
cap_analysis.caps2surf(fwhm=2, cmap=custom_cmap, size=(500, 100), layout="row")
```
**Partial Plot Outputs:** (*Note*: one image will be generated per CAP)
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/fadc946a-214b-4fbf-8316-2f32ab0b026e)
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/8207914a-6bf0-47a9-8be8-3504d0a56516)


```python
# Create correlation matrix
cap_analysis.caps2corr(annot=True ,figsize=(6,4),cmap="coolwarm")

# You can use seaborn's premade palettes as strings or generate your own custom palettes
# Using seaborn's diverging_palette function, matplotlib's LinearSegmentedColormap, 
# or other Classes or functions compatable with seaborn

# Create the colormap
import seaborn as sns
palette = sns.diverging_palette(260, 10, s=80, l=55, n=256, as_cmap=True)
cap_analysis.caps2corr(annot=True, figsize=(6,4), cmap=palette)
```
**Plot Output:**
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/57a2ce81-13d3-40d0-93e7-0ca910f7b0be)
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/9a8329df-65c7-4ad0-8b81-edc73f2d960d)

```python
# Create radar plots showing cosine similarity between region/networks and caps
radialaxis={"showline": True, "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)", "gridcolor": "rgba(0, 0, 0, 0.25)", "ticks": "outside" , "tickfont": {"size": 14, "color": "black"},
"range": [0,0.3], "tickvals": [0.1,0.2,0.3]}
cap_analysis.caps2radar(radialaxis=radialaxis, fill="toself")
```
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/5ab17b92-bac9-48a9-9f4c-25bb1a69bf1c)

# Testing 
This package was tested using a closed dataset as well as a modified version of a single-subject open dataset to test the `TimeseriesExtractor` function on GitHub Actions. The open dataset provided by [Laumann & Poldrack](https://openfmri.org/dataset/ds000031/) and used in [Laumann et al., 2015](https://doi.org/10.1016/j.neuron.2015.06.037)[^4]. was also utilized. This data was obtained from the OpenfMRI database, accession number ds000031. 

Modifications to the data included:

- Truncating the preprocessed BOLD data and confounds from 448 timepoints to 40 timepoints.
- Only including session 002 data.
- Adding a dataset_description.json file to the fmriprep folder.
- Excluding the nii.gz file in the root BIDS folder.
- Retaining only the mask, truncated preprocessed BOLD file, and truncated confounds file in the fmriprep folder.
- Slightly changing the naming style of the mask, preprocessed BOLD file, and confounds file in the fmriprep folder to conform with the naming conventions of modern fmriprep outputs.
- Testing with custom parcellations was done using the HCPex parcellation, an extension of the HCP (Human Connectome Project) parcellation, which adds 66 subcortical areas. This original atlas can be downloaded from.

Testing with custom parcellations was done with the HCPex parcellation, an extension of the HCP (Human Connectome Project) parcellation, which adds 66 subcortical areas [^5], [^6]. This original atlas can be downloaded from https://github.com/wayalan/HCPex.

# References
[^1]: Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals. NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

[^2]: Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia. NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

[^3]: Zhang, R., Yan, W., Manza, P., Shokri-Kojori, E., Demiral, S. B., Schwandt, M., Vines, L., Sotelo, D., Tomasi, D., Giddens, N. T., Wang, G., Diazgranados, N., Momenan, R., & Volkow, N. D. (2023). 
Disrupted brain state dynamics in opioid and alcohol use disorder: attenuation by nicotine use. Neuropsychopharmacology, 49(5), 876–884. https://doi.org/10.1038/s41386-023-01750-w      

[^4]: Laumann, T. O., Gordon, E. M., Adeyemo, B., Snyder, A. Z., Joo, S. J., Chen, M. Y., Gilmore, A. W., McDermott, K. B., Nelson, S. M., Dosenbach, N. U., Schlaggar, B. L., Mumford, J. A., Poldrack, R. A., & Petersen, S. E. (2015). Functional system and areal organization of a highly sampled individual human brain. Neuron, 87(3), 657–670. https://doi.org/10.1016/j.neuron.2015.06.037

[^5]: Huang CC, Rolls ET, Feng J, Lin CP. An extended Human Connectome Project multimodal parcellation atlas of the human cortex and subcortical areas. Brain Struct Funct. 2022 Apr;227(3):763-778. Epub 2021 Nov 17. doi: 10.1007/s00429-021-02421-6

[^6]: Huang CC, Rolls ET, Hsu CH, Feng J, Lin CP. Extensive Cortical Connectivity of the Human Hippocampal Memory System: Beyond the "What" and "Where" Dual Stream Model. Cerebral Cortex. 2021 May 19;bhab113. doi: 10.1093/cercor/bhab113.
