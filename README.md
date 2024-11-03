# neurocaps
[![Latest Version](https://img.shields.io/pypi/v/neurocaps.svg)](https://pypi.python.org/pypi/neurocaps/)
[![Python Versions](https://img.shields.io/pypi/pyversions/neurocaps.svg)](https://pypi.python.org/pypi/neurocaps/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.11642615-teal)](https://doi.org/10.5281/zenodo.14031867)
[![Github Repository](https://img.shields.io/badge/Source%20Code-neurocaps-purple)](https://github.com/donishadsmith/neurocaps)
[![Test Status](https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/github/donishadsmith/neurocaps/graph/badge.svg?token=WS2V7I16WF)](https://codecov.io/github/donishadsmith/neurocaps)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Platform Support](https://img.shields.io/badge/OS-Ubuntu%20|%20macOS%20|%20Windows-blue)

This is a Python package designed to perform Co-activation Patterns (CAPs) analyses. It utilizes k-means clustering to group timepoints (TRs) into brain states, applicable to both resting-state and task-based fMRI data. The package is compatible with data preprocessed using **fMRIPrep** and assumes your directory is BIDS-compliant, containing a derivatives folder with a pipeline folder (such as fMRIPrep) that holds the preprocessed BOLD data.

## Installation
To install neurocaps, follow the instructions below using your preferred terminal.

### For a standard installation from PyPi:
```bash

pip install neurocaps

```

**Windows Users**

To avoid installation errors related to long paths not being enabled, pybids will not be installed by default.
To include pybids in your installation, use:

```bash

pip install neurocaps[windows]

```

Alternatively, you can install pybids separately:

```bash

pip install pybids

```
### Installation from Source (Development Version)
To install the latest development version from the source, there are two options:

1. Install directly via pip:
```bash

pip install git+https://github.com/donishadsmith/neurocaps.git

```

2. Clone the repository and install locally:

```bash

git clone https://github.com/donishadsmith/neurocaps/
cd neurocaps
pip install -e .

```
**Windows Users**

To include pybids when installing the development version on Windows, use:

```bash

git clone https://github.com/donishadsmith/neurocaps/
cd neurocaps
pip install -e .[windows]
```

## Usage
**Note, documentation of each function can be found at https://neurocaps.readthedocs.io/en/latest/api.html**

**This package contains two main classes: `TimeseriesExtractor` for extracting the timeseries, and `CAP` for performing the CAPs analysis.**

**Main features for `TimeseriesExtractor` includes:**
- **Timeseries Extraction:** Extract timeseries for resting-state or task data using Schaefer, AAL, or a lateralized Custom parcellation for spatial dimensionality reduction.
- **Parallel Processing:** Use parallel processing to speed up timeseries extraction.
- **Saving Timeseries:** Save the nested dictionary containing timeseries as a pickle file.
- **Visualization:** Visualize the timeseries at the region or node level of the parcellation.

**Main features for `CAP` includes:**
- **Grouping:** Perform CAPs analysis for entire sample or groups of subject IDs.
- **Optimal Cluster Size Identification:** Perform the Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions to identify the optimal cluster size and automatically save the optimal model as an attribute.
- **Parallel Processing:** Use parallel processing to speed up optimal cluster size identification.
- **CAP Visualization:** Visualize the CAPs as outer products or heatmaps at either the region or node level of the parcellation. 
- **Save CAPs as NifTIs:** Convert the atlas used for parcellation to a statistical NifTI image.
- **Surface Plot Visualization:** Project CAPs onto a surface plot.
- **Correlation Matrix Creation:** Create a correlation matrix from CAPs.
- **CAP Metrics Calculation:** Calculate several CAP metrics as described in [Liu et al., 2018](https://doi.org/10.1016/j.neuroimage.2018.01.041)[^1] and [Yang et al., 2021](https://doi.org/10.1016/j.neuroimage.2021.118193)[^2]:
    - *Temporal Fraction:* The proportion of total volumes spent in a single CAP over all volumes in a run.
    - *Persistence:* The average time spent in a single CAP before transitioning to another CAP
    - *Counts:* The total number of initiations of a specific CAP across an entire run. An initiation is defined as the first occurrence of a CAP.
    - *Transition Frequency:* The number of transitions between different CAPs across the entire run.
    - *Transition Probability:* The probability of transitioning from one CAP to another CAP (or the same CAP). This is calculated as (Number of transitions from A to B)/(Total transitions from A).
- **Cosine Similarity Radar Plots:** Create radar plots showing the cosine similarity between positive and negative
activations of each CAP and each a-priori regions in a parcellation [^3] [^4].

**Additionally, the `neurocaps.analysis` submodule contains additional functions:**

- `merge_dicts`: Merge the subject_timeseries dictionaries for overlapping subjects across tasks to identify similar CAPs across different tasks. The merged dictionary can be saved as a pickle file.
- `standardize`: Standardizes each run independently for all subjects in the subject timeseries.
- `change_dtype`: Changes the dtype of all subjects in the subject timeseries to help with memory usage.
- `transition_matrix`: Uses the "transition_probability" output from ``CAP.calculate_metrics`` to generate and visualize the averaged transition probability matrix for all groups from the analysis.

Please refer to [demo.ipynb](https://github.com/donishadsmith/neurocaps/blob/main/demo.ipynb) or https://neurocaps.readthedocs.io/en/latest/examples/examples.html for a more extensive demonstration of the features included in this package.

**Quick Code Examples (CAP examples use randomized data to simulate multiple subjects)**:

```python

from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP

# Set specific confounds for nuisance regression
confounds = ['Cosine*', 'Rot*']

# Set parcellation
parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 2}}

# Initialize TimeseriesExtractor
extractor = TimeseriesExtractor(parcel_approach=parcel_approach,
                                standardize="zscore_sample",
                                use_confounds=True,
                                detrend=True,
                                low_pass=0.15,
                                high_pass=0.01,
                                confound_names=confounds,
                                n_acompcor_separate=2)

bids_dir = "/path/to/bids/dir"

# If there are multiple pipelines in the derivatives folder, you can specify a specific pipeline

# pipeline_name = "fmriprep-1.4.0"
pipeline_name = fmriprep_1.0.0/fmriprep/

# Extract timeseries for subjects in the BIDS directory
extractor.get_bold(bids_dir=bids_dir,
                   task="rest",
                   session='002',
                   pipeline_name=pipeline_name
                   verbose=True,
                   flush=True)
```
**Output:**
```
2024-11-02 20:35:47,797 neurocaps._utils.extraction.check_confound_names [INFO] Confound regressors to be used if available: Cosine*, aComp*, Rot*.
2024-11-02 20:35:48,898 neurocaps.extraction.timeseriesextractor [INFO] BIDS Layout: ...0.4_ses001-022\ds000031_R1.0.4 | Subjects: 1 | Sessions: 1 | Runs: 1
2024-11-02 20:35:48,941 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 01 | SESSION: 002 | TASK: rest | RUN: 001] Preparing for Timeseries Extraction using [FILE: sub-01_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz].
2024-11-02 20:35:48,946 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 01 | SESSION: 002 | TASK: rest | RUN: 001] The following confounds will be used for nuisance regression: Cosine00, Cosine01, Cosine02, Cosine03, Cosine04, Cosine05, Cosine06, aCompCor00, aCompCor01, aCompCor02, aCompCor03, aCompCor04, aCompCor05, RotX, RotY, RotZ.
```

```python
# Task; use parallel processing with `n_cores` with theoretical dataset containing multiple subjects
extractor.get_bold(bids_dir=bids_dir,
                   task="emo",
                   condition="positive", 
                   pipeline_name=pipeline_name,
                   n_cores=10)

cap_analysis = CAP(parcel_approach=extractor.parcel_approach)

cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, 
                      n_clusters=6,
                      standardize = True)

# Visualize CAPs
kwargs = {"sharey": True, "ncol": 3, "subplots": True, "cmap": "coolwarm"}

cap_analysis.caps2plot(visual_scope="regions",
                       plot_options="outer_product", 
                       suffix_title="- Positive Valence",
                       **kwargs)

# Create the colormap
import seaborn as sns

palette = sns.diverging_palette(260, 10, s=80, l=55, n=256, as_cmap=True)

kwargs["cmap"] = palette
kwargs.update({"xlabel_rotation": 90, "tight_layout": False, "hspace": 0.4})

cap_analysis.caps2plot(visual_scope="nodes",
                       plot_options="outer_product", 
                       suffix_title="- Positive Valence",
                       **kwargs)
```
**Plot Outputs:**
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/e1ab0f55-0c4c-4701-8f3a-838c2470d44d)
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/43e46a0a-8721-4df9-88fa-04758a34142e)

```python

# Get CAP metrics
outputs = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries,
                                         tr=2.0, 
                                         return_df=True,
                                         output_dir=output_dir,
                                         metrics=["temporal_fraction", "persistence"],
                                         continuous_runs=True,
                                         prefix_file_name="All_Subjects_CAPs_metrics")

print(outputs["temporal_fraction"])
```
**DataFrame Output:**
| Subject_ID | Group | Run | CAP-1 | CAP-2 | CAP-3 | CAP-4 | CAP-5 | CAP-6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | All Subjects | run-continuous | 0.14 | 0.17 | 0.14 | 0.2 | 0.15 | 0.19 |
| 2 | All Subjects | run-continuous | 0.17 | 0.17 | 0.16 | 0.16 | 0.15 | 0.19 |
| 3 | All Subjects | run-continuous | 0.15 | 0.2 | 0.14 | 0.18 | 0.17 | 0.17 |
| 4 | All Subjects | run-continuous | 0.17 | 0.21 | 0.18 | 0.17 | 0.1 | 0.16 |
| 5 | All Subjects | run-continuous | 0.14 | 0.19 | 0.14 | 0.16 | 0.2 | 0.18 |
| 6 | All Subjects | run-continuous | 0.16 | 0.21 | 0.16 | 0.18 | 0.16 | 0.13 |
| 7 | All Subjects | run-continuous | 0.16 | 0.16 | 0.17 | 0.15 | 0.19 | 0.17 |
| 8 | All Subjects | run-continuous | 0.17 | 0.21 | 0.13 | 0.14 | 0.17 | 0.18 |
| 9 | All Subjects | run-continuous | 0.18 | 0.1 | 0.17 | 0.18 | 0.16 | 0.2 |
| 10 | All Subjects | run-continuous | 0.14 | 0.19 | 0.14 | 0.17 | 0.19 | 0.16 |

```python
# Create surface plots of CAPs; there will be as many plots as CAPs
kwargs = {"cmap": "cold_hot", "layout": "row", "size": (500, 100), "zoom": 1,
          "cbar_location":"bottom"}

cap_analysis.caps2surf(fwhm=2, **kwargs)

#You can also generate your own colormaps using matplotlib's LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap

colors = ["#1bfffe", "#00ccff", "#0099ff", "#0066ff", "#0033ff", "#c4c4c4",
          "#ff6666", "#ff3333", "#FF0000","#ffcc00","#FFFF00"]

custom_cmap = LinearSegmentedColormap.from_list("custom_cold_hot", colors, N=256)

kwargs["cmap"] = custom_cmap

cap_analysis.caps2surf(**kwargs)
```
**Partial Plot Outputs:** (*Note*: one image will be generated per CAP)

<img src="https://github.com/donishadsmith/neurocaps/assets/112973674/fadc946a-214b-4fbf-8316-2f32ab0b026e" width=70% height=70%>
<img src="https://github.com/donishadsmith/neurocaps/assets/112973674/8207914a-6bf0-47a9-8be8-3504d0a56516" width=70% height=70%>

```python
# Create Pearson correlation matrix
kwargs = {"annot": True, "cmap": "coolwarm"}

cap_analysis.caps2corr(**kwargs)

# Create the colormap
import seaborn as sns

palette = sns.diverging_palette(260, 10, s=80, l=55, n=256, as_cmap=True)

kwargs["cmap"] = palette

cap_analysis.caps2corr(**kwargs)
```
**Plot Output:**

<img src="https://github.com/donishadsmith/neurocaps/assets/112973674/57a2ce81-13d3-40d0-93e7-0ca910f7b0be" width=70% height=70%>
<img src="https://github.com/donishadsmith/neurocaps/assets/112973674/9a8329df-65c7-4ad0-8b81-edc73f2d960d" width=70% height=70%>

```python
# Create radar plots showing cosine similarity between region/networks and caps
radialaxis={"showline": True, 
            "linewidth": 2, 
            "linecolor": "rgba(0, 0, 0, 0.25)", 
            "gridcolor": "rgba(0, 0, 0, 0.25)",
            "ticks": "outside" , 
            "tickfont": {"size": 14, "color": "black"}, 
            "range": [0,0.6],
            "tickvals": [0.1,"","",0.4, "","", 0.6]}

legend = {"yanchor": "top", 
        "y": 0.99, 
        "x": 0.99,
        "title_font_family": "Times New Roman", 
        "font": {"size": 12, "color": "black"}}

colors =  {"High Amplitude": "black", "Low Amplitude": "orange"}


kwargs = {"radialaxis": radial, "fill": "toself", "legend": legend,
"color_discrete_map": colors, "height": 400, "width": 600}

cap_analysis.caps2radar(output_dir=output_dir, **kwargs)
```
**Partial Plot Outputs:** (*Note*: one image will be generated per CAP)

<img src="https://github.com/user-attachments/assets/b190b209-a036-46a5-881f-3d40cffda1c0" width=70% height=70%>
<img src="https://github.com/user-attachments/assets/8bd56af5-fbe9-4d57-8f58-2c332af986f9" width=70% height=70%>
<img src="https://github.com/user-attachments/assets/81b739f4-bd7f-41bf-9b42-14d8376b5239" width=70% height=70%>

```python
# Get transition probabilities for all participants in a dataframe, then convert to an averaged matrix
from neurocaps.analysis import transition_matrix

# Optimal cluster sizes are saved automatically
cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, 
                      cluster_selection_method="davies_bouldin",
                      standardize=True,
                      n_clusters=range(2,6))

outputs = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, 
                                         return_df=True,
                                         metrics=["transition_probability"],
                                         continuous_runs=True,
                                         output_dir=output_dir,
                                         prefix_file_name="All_Subjects_CAPs_metrics")

print(outputs["transition_probability"]["All Subjects"])

kwargs = {"cmap": "viridis", "fmt": ".3f", "annot": True}

trans_outputs = transition_matrix(trans_dict=outputs["transition_probability"],
                                  show_figs=True,
                                  return_df=True,
                                  output_dir=output_dir.
                                  **kwargs)

print(trans_outputs["All Subjects"])
```
**Outputs:**
```
2024-11-02 21:04:23,067 neurocaps.analysis.cap [INFO] [GROUP: All Subjects | METHOD: davies_bouldin] Optimal cluster size is 3.
```
| Subject_ID | Group | Run | 1.1 | 1.2 | 1.3 | 2.1 | 2.2 | 2.3 | 3.1 | 3.2 | 3.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | All Subjects | run-continuous | 0.326 | 0.261 | 0.413 | 0.245 | 0.449 | 0.306 | 0.352 | 0.278 | 0.37 |
| 2 | All Subjects | run-continuous | 0.4 | 0.25 | 0.35 | 0.486 | 0.108 | 0.405 | 0.346 | 0.365 | 0.288 |
| 3 | All Subjects | run-continuous | 0.354 | 0.229 | 0.417 | 0.383 | 0.362 | 0.255 | 0.241 | 0.352 | 0.407 |
| 4 | All Subjects | run-continuous | 0.283 | 0.37 | 0.348 | 0.302 | 0.321 | 0.377 | 0.32 | 0.38 | 0.3 |
| 5 | All Subjects | run-continuous | 0.292 | 0.354 | 0.354 | 0.38 | 0.28 | 0.34 | 0.294 | 0.392 | 0.314 |
| 6 | All Subjects | run-continuous | 0.339 | 0.304 | 0.357 | 0.333 | 0.231 | 0.436 | 0.444 | 0.222 | 0.333 |
| 7 | All Subjects | run-continuous | 0.424 | 0.203 | 0.373 | 0.45 | 0.275 | 0.275 | 0.34 | 0.32 | 0.34 |
| 8 | All Subjects | run-continuous | 0.25 | 0.271 | 0.479 | 0.39 | 0.244 | 0.366 | 0.35 | 0.3 | 0.35 |
| 9 | All Subjects | run-continuous | 0.429 | 0.265 | 0.306 | 0.319 | 0.298 | 0.383 | 0.245 | 0.377 | 0.377 |
| 10 | All Subjects | run-continuous | 0.333 | 0.375 | 0.292 | 0.306 | 0.347 | 0.347 | 0.327 | 0.269 | 0.404 |

<img src="https://github.com/user-attachments/assets/3ab1d123-0c3e-47e3-b24c-52bfda13d3ef" width=70% height=70%>

| From/To | CAP-1 | CAP-2 | CAP-3 |
| --- | --- | --- | --- |
| CAP-1 | 0.343 | 0.288 | 0.369 |
| CAP-2 | 0.36 | 0.291 | 0.349 |
| CAP-3 | 0.326 | 0.326 | 0.348 |

## Testing 
This package was tested using a closed dataset as well as a modified version of a single-subject open dataset to test the `TimeseriesExtractor` function on GitHub Actions. The open dataset provided by [Laumann & Poldrack](https://openfmri.org/dataset/ds000031/) and used in [Laumann et al., 2015](https://doi.org/10.1016/j.neuron.2015.06.037)[^5]. was also utilized. This data was obtained from the OpenfMRI database, accession number ds000031. 

Modifications to the data included:

- Truncating the preprocessed BOLD data and confounds from 448 timepoints to 40 timepoints.
- Only including session 002 data.
- Adding a dataset_description.json file to the fmriprep folder.
- Excluding the nii.gz file in the root BIDS folder.
- Retaining only the mask, truncated preprocessed BOLD file, and truncated confounds file in the fmriprep folder.
- Slightly changing the naming style of the mask, preprocessed BOLD file, and confounds file in the fmriprep folder to conform with the naming conventions of modern fmriprep outputs.
- Testing with custom parcellations was done using the HCPex parcellation, an extension of the HCP (Human Connectome Project) parcellation, which adds 66 subcortical areas. This original atlas can be downloaded from.

Testing with custom parcellations was done with the HCPex parcellation, an extension of the HCP (Human Connectome Project) parcellation, which adds 66 subcortical areas [^6], [^7]. This original atlas can be downloaded from https://github.com/wayalan/HCPex.

## Contributing
Please refer the [contributing guidelines](https://github.com/donishadsmith/neurocaps/blob/test/CONTRIBUTING.md) on how to contribute to neurocaps.

## Acknowledgements
This package was initially inspired by a co-activation patterns implementation in mtorabi59's [pydfc](https://github.com/neurodatascience/dFC) package.

## References
[^1]: Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals. NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

[^2]: Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia. NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

[^3]: Zhang, R., Yan, W., Manza, P., Shokri-Kojori, E., Demiral, S. B., Schwandt, M., Vines, L., Sotelo, D., Tomasi, D., Giddens, N. T., Wang, G., Diazgranados, N., Momenan, R., & Volkow, N. D. (2023). 
Disrupted brain state dynamics in opioid and alcohol use disorder: attenuation by nicotine use. Neuropsychopharmacology, 49(5), 876–884. https://doi.org/10.1038/s41386-023-01750-w      

[^4]: Ingwersen, T., Mayer, C., Petersen, M., Frey, B. M., Fiehler, J., Hanning, U., Kühn, S., Gallinat, J., Twerenbold, R., Gerloff, C., Cheng, B., Thomalla, G., & Schlemm, E. (2024). Functional MRI brain state occupancy in the presence of cerebral small vessel disease — A pre-registered replication analysis of the Hamburg City Health Study. Imaging Neuroscience, 2, 1–17. https://doi.org/10.1162/imag_a_00122

[^5]: Laumann, T. O., Gordon, E. M., Adeyemo, B., Snyder, A. Z., Joo, S. J., Chen, M. Y., Gilmore, A. W., McDermott, K. B., Nelson, S. M., Dosenbach, N. U., Schlaggar, B. L., Mumford, J. A., Poldrack, R. A., & Petersen, S. E. (2015). Functional system and areal organization of a highly sampled individual human brain. Neuron, 87(3), 657–670. https://doi.org/10.1016/j.neuron.2015.06.037

[^6]: Huang CC, Rolls ET, Feng J, Lin CP. An extended Human Connectome Project multimodal parcellation atlas of the human cortex and subcortical areas. Brain Struct Funct. 2022 Apr;227(3):763-778. Epub 2021 Nov 17. doi: 10.1007/s00429-021-02421-6

[^7]: Huang CC, Rolls ET, Hsu CH, Feng J, Lin CP. Extensive Cortical Connectivity of the Human Hippocampal Memory System: Beyond the "What" and "Where" Dual Stream Model. Cerebral Cortex. 2021 May 19;bhab113. doi: 10.1093/cercor/bhab113.
