# neurocaps
[![Latest Version](https://img.shields.io/pypi/v/neurocaps.svg)](https://pypi.python.org/pypi/neurocaps/)
[![Python Versions](https://img.shields.io/pypi/pyversions/neurocaps.svg)](https://pypi.python.org/pypi/neurocaps/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.11642615-teal)](https://doi.org/10.5281/zenodo.14231418)
[![Github Repository](https://img.shields.io/badge/Source%20Code-neurocaps-purple)](https://github.com/donishadsmith/neurocaps)
[![Test Status](https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/github/donishadsmith/neurocaps/graph/badge.svg?token=WS2V7I16WF)](https://codecov.io/github/donishadsmith/neurocaps)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Platform Support](https://img.shields.io/badge/OS-Ubuntu%20|%20macOS%20|%20Windows-blue)

neurocaps is a Python package for performing Co-activation Patterns (CAPs) analyses on resting-state or task-based fMRI
data (resting-state & task-based). CAPs identifies recurring brain states through k-means clustering of BOLD timeseries
data [^1].

**neurocaps is most optimized for fMRI data preprocessed with fMRIPrep and assumes a BIDs compliant directory
such as the example directory structures below:**

Basic BIDS directory:
```

    bids_root/
    ├── dataset_description.json
    ├── sub-<subject_label>/
    │   └── func/
    │       └── *task-*_events.tsv
    ├── derivatives/
    │   └── fmriprep-<version_label>/
    │       ├── dataset_description.json
    │       └── sub-<subject_label>/
    │           └── func/
    │               ├── *confounds_timeseries.tsv
    │               ├── *brain_mask.nii.gz
    │               └── *preproc_bold.nii.gz
```

BIDS directory with session-level organization:
```

    bids_root/
    ├── dataset_description.json
    ├── sub-<subject_label>/
    │   └── ses-<session_label>/
    │       └── func/
    │           └── *task-*_events.tsv
    ├── derivatives/
    │   └── fmriprep-<version_label>/
    │       ├── dataset_description.json
    │       └── sub-<subject_label>/
    │           └── ses-<session_label>/
    │               └── func/
    │                   ├── *confounds_timeseries.tsv
    │                   ├── *brain_mask.nii.gz
    │                   └── *preproc_bold.nii.gz
```

*Note: Only the preprocessed BOLD file is required. Additional files such as the confounds tsv (needed for denoising),
mask, and task timing tsv file (needed for filtering a specific task condition) depend on the specific analyses.
The "dataset_description.json" is required in both the bids root and pipeline directories for querying with pybids*

## Installation
To install neurocaps, follow the instructions below using your preferred terminal.

### Standard Installation from PyPi
```bash

pip install neurocaps

```

**Windows Users**

To avoid installation errors related to long paths not being enabled, pybids will not be installed by default.
Refer to official [Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell)
to enable long paths.

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
**Note, documentation of each function can be found in the [API](https://neurocaps.readthedocs.io/en/latest/api.html)
section of the documentation homepage.**

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

- `merge_dicts`: Merge the subject_timeseries dictionaries for overlapping subjects across tasks to identify similar CAPs across different tasks [^5]. The merged dictionary can be saved as a pickle file.
- `standardize`: Standardizes each run independently for all subjects in the subject timeseries.
- `change_dtype`: Changes the dtype of all subjects in the subject timeseries to help with memory usage.
- `transition_matrix`: Uses the "transition_probability" output from ``CAP.calculate_metrics`` to generate and visualize the averaged transition probability matrix for all groups from the analysis.

Please refer to the [demos](https://github.com/donishadsmith/neurocaps/tree/main/demos) or
the [tutorials](https://neurocaps.readthedocs.io/en/latest/examples/examples.html) on the documentation website
for a more extensive demonstration of the features included in this package.

**Demonstration**:

Use dataset from OpenNeuro [^6]:
```python
# Download Sample Dataset from OpenNeuro, requires the openneuro-py package
# [Dataset] doi: doi:10.18112/openneuro.ds005381.v1.0.0
from openneuro import download

demo_dir = "neurocaps_demo"
os.makedirs(demo_dir)

# Include the run-1 and run-2 data from two tasks for two subjects
include = [
    "dataset_description.json",
    "sub-0004/ses-2/func/*run-[12]*events*",
    "sub-0006/ses-2/func/*run-[12]*events*",
    "derivatives/fmriprep/sub-0004/fmriprep/sub-0004/ses-2/func/*run-[12]*confounds_timeseries*",
    "derivatives/fmriprep/sub-0004/fmriprep/sub-0004/ses-2/func/*run-[12]_space-MNI152NLin*preproc_bold*",
    "derivatives/fmriprep/sub-0004/fmriprep/sub-0004/ses-2/func/*run-[12]_space-MNI152NLin*brain_mask*",
    "derivatives/fmriprep/sub-0006/fmriprep/sub-0006/ses-2/func/*run-[12]*confounds_timeseries*",
    "derivatives/fmriprep/sub-0006/fmriprep/sub-0006/ses-2/func/*run-[12]_space-MNI152NLin*preproc_bold*",
    "derivatives/fmriprep/sub-0006/fmriprep/sub-0006/ses-2/func/*run-[12]_space-MNI152NLin*brain_mask*",
    ]

download(dataset="ds005381", include=include, target_dir=demo_dir)

# Create a "dataset_description" file for the pipeline folder if needed
import json

desc = {
    "Name": "fMRIPrep - fMRI PREProcessing workflow",
    "BIDSVersion": "1.0.0",
    "DatasetType": "derivative",
    "GeneratedBy": [
        {
            "Name": "fMRIPrep",
            "Version": "20.2.0",
            "CodeURL": "https://github.com/nipreps/fmriprep"
        }
    ]
}

with open("neurocaps_demo/derivatives/fmriprep/dataset_description.json", 'w', encoding='utf-8') as f:
    json.dump(desc, f)
```

```python
from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP

# Set specific confounds for nuisance regression
confounds = [
    "cosine*",
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z"
]

# Set parcellation
parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 2}}

# Initialize TimeseriesExtractor
extractor = TimeseriesExtractor(space="MNI152NLin6Asym",
                                parcel_approach=parcel_approach,
                                standardize="zscore_sample",
                                use_confounds=True,
                                detrend=True,
                                low_pass=0.1,
                                high_pass=None,
                                n_acompcor_separate=2, # 2 acompcor from WM and CSF masks = 4 total
                                confound_names=confounds,
                                fd_threshold={"threshold": 0.35, "outlier_percentage": 0.20, "n_before": 2,
                                              "n_after": 1, "use_sample_mask": True})

# Extract timeseries for subjects in the BIDS directory; Subject 0006 run-1 will be flagged and skipped
extractor.get_bold(bids_dir="neurocaps_demo",
                   task="DET",
                   session="2",
                   n_cores=None,
                   pipeline_name="fmriprep", # Can specify if multiple pipelines exists in derivatives directory
                   verbose=True)
```
**Output:**
```
2024-11-24 23:45:02,378 neurocaps._utils.extraction.check_confound_names [INFO] Confound regressors to be used if available: cosine*, trans_x, trans_y, trans_z, rot_x, rot_y, rot_z.
2024-11-24 23:45:02,513 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 0004 | SESSION: 2 | TASK: DET | RUN: 1] Preparing for Timeseries Extraction using [FILE: sub-0004_ses-2_task-DET_run-1_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz].
2024-11-24 23:45:02,525 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 0004 | SESSION: 2 | TASK: DET | RUN: 1] The following confounds will be used for nuisance regression: cosine00, cosine01, cosine02, cosine03, trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, a_comp_cor_00, a_comp_cor_01, a_comp_cor_33, a_comp_cor_34.
2024-11-24 23:45:12,837 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 0004 | SESSION: 2 | TASK: DET | RUN: 2] Preparing for Timeseries Extraction using [FILE: sub-0004_ses-2_task-DET_run-2_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz].
2024-11-24 23:45:12,844 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 0004 | SESSION: 2 | TASK: DET | RUN: 2] The following confounds will be used for nuisance regression: cosine00, cosine01, cosine02, cosine03, trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, a_comp_cor_00, a_comp_cor_01, a_comp_cor_100, a_comp_cor_101.
2024-11-24 23:45:23,065 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 0006 | SESSION: 2 | TASK: DET | RUN: 1] Preparing for Timeseries Extraction using [FILE: sub-0006_ses-2_task-DET_run-1_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz].
2024-11-24 23:45:23,065 neurocaps._utils.extraction.extract_timeseries [WARNING] [SUBJECT: 0006 | SESSION: 2 | TASK: DET | RUN: 1] Timeseries Extraction Skipped: Run flagged due to more than 20.0% of the volumes exceeding the framewise displacement threshold of 0.35. Percentage of volumes exceeding the threshold limit is 26.785714285714285%.
2024-11-24 23:45:23,065 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 0006 | SESSION: 2 | TASK: DET | RUN: 2] Preparing for Timeseries Extraction using [FILE: sub-0006_ses-2_task-DET_run-2_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz].
2024-11-24 23:45:23,081 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 0006 | SESSION: 2 | TASK: DET | RUN: 2] The following confounds will be used for nuisance regression: cosine00, cosine01, cosine02, cosine03, trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, a_comp_cor_00, a_comp_cor_01, a_comp_cor_24, a_comp_cor_25.
```

```python
# Get CAPs
cap_analysis = CAP(parcel_approach=extractor.parcel_approach)

cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                      n_clusters=2,
                      standardize=True)

# `sharey` only applicable to outer product plots
kwargs = {"sharey": True, "ncol": 3, "subplots": True, "cmap": "coolwarm", "xticklabels_size": 10,
          "yticklabels_size": 10, "xlabel_rotation": 90, "cbarlabels_size": 10}

# Outer Product
cap_analysis.caps2plot(visual_scope="regions",
                       plot_options=["outer_product"],
                       suffix_title="- DET Task",
                       **kwargs)

# Heatmap
kwargs["xlabel_rotation"] = 0

cap_analysis.caps2plot(visual_scope="regions",
                       plot_options=["heatmap"],
                       suffix_title="- DET Task",
                       **kwargs)
```
**Plot Outputs:**

<img src="assets/outerproduct.png" width=70% height=70%>
<img src="assets/heatmap.png" width=70% height=70%>

```python

# Get CAP metrics
outputs = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries,
                                         tr=2.0, # TR to convert persistence to time units
                                         return_df=True,
                                         metrics=["temporal_fraction", "persistence"],
                                         continuous_runs=True)

# Subject 0006 only has run-2 data since run-1 was flagged during timeseries extraction
print(outputs["temporal_fraction"])
```
**DataFrame Output:**
| Subject_ID | Group | Run | CAP-1 | CAP-2 |
| --- | --- | --- | --- | --- |
| 0004 | All Subjects | run-continuous | 0.501529 | 0.498471 |
| 0006 | All Subjects | run-2 | 0.520000 | 0.480000 |

```python
# Create surface plots
kwargs = {"cmap": "cold_hot", "layout": "row", "size": (500, 200), "zoom": 1,
          "cbar_kws": {"location": "bottom"}, "color_range": (-1, 1)}

cap_analysis.caps2surf(**kwargs)
```
**Plot Outputs:**

<img src="assets/cap1.png" width=70% height=70%>
<img src="assets/cap2.png" width=70% height=70%>

```python
# Create Pearson correlation matrix
kwargs = {"annot": True, "cmap": "viridis", "xticklabels_size": 10,
          "yticklabels_size": 10, "cbarlabels_size": 10}

cap_analysis.caps2corr(**kwargs)
```
**Plot Output:**

<img src="assets/correlation.png" width=70% height=70%>

```python
# Create radar plots showing cosine similarity between region/networks and caps
radialaxis={"showline": True,
            "linewidth": 2,
            "linecolor": "rgba(0, 0, 0, 0.25)",
            "gridcolor": "rgba(0, 0, 0, 0.25)",
            "ticks": "outside" ,
            "tickfont": {"size": 14, "color": "black"},
            "range": [0, 0.6],
            "tickvals": [0.1, "", "", 0.4, "", "", 0.6]}

legend = {"yanchor": "top",
          "y": 0.99,
          "x": 0.99,
          "title_font_family": "Times New Roman",
          "font": {"size": 12, "color": "black"}}

colors =  {"High Amplitude": "black", "Low Amplitude": "orange"}

kwargs = {"radialaxis": radialaxis, "fill": "toself", "legend": legend, "color_discrete_map": colors,
          "height": 400, "width": 600}

cap_analysis.caps2radar(**kwargs)
```
**Plot Outputs:**

<img src="assets/cap1radar.png" width=70% height=70%>
<img src="assets/cap2radar.png" width=70% height=70%>

```python
# Get transition probabilities for all participants in a dataframe, then convert to an averaged matrix
from neurocaps.analysis import transition_matrix

# Optimal cluster sizes are saved automatically
cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                      cluster_selection_method="silhouette",
                      standardize=True,
                      show_figs=True,
                      n_clusters=range(2, 6))

outputs = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries,
                                         return_df=True,
                                         metrics=["transition_probability"],
                                         continuous_runs=True)

print(outputs["transition_probability"]["All Subjects"])

kwargs = {"cmap": "Blues", "fmt": ".3f", "annot": True, "vmin": 0, "vmax": 1, "xticklabels_size": 10,
          "yticklabels_size": 10, "cbarlabels_size": 10}

trans_outputs = transition_matrix(trans_dict=outputs["transition_probability"],
                                  show_figs=True,
                                  return_df=True,
                                  **kwargs)

print(trans_outputs["All Subjects"])
```
**Outputs:**
```
2024-11-24 23:58:39,470 neurocaps.analysis.cap [INFO] [GROUP: All Subjects | METHOD: silhouette] Optimal cluster size is 2.
```
| Subject_ID | Group | Run | 1.1 | 1.2 | 2.1 | 2.2 |
| --- | --- | --- | --- | --- | --- | --- |
| 0004 | All Subjects | run-continuous | 0.802395 | 0.197605 | 0.207547 | 0.792453 |
| 0006 | All Subjects | run-2 | 0.790123 | 0.209877 | 0.235294 | 0.764706 |

<img src="assets/silhouette.png" width=70% height=70%>
<img src="assets/transprob.png" width=70% height=70%>

| From/To | CAP-1 | CAP-2 |
| --- | --- | --- |
| CAP-1 | 0.796259 | 0.203741 |
| CAP-2 | 0.221421 | 0.778579 |

## Testing
This package was tested using a closed dataset as well as a modified version of a single-subject open dataset to test the `TimeseriesExtractor` function on GitHub Actions. The open dataset provided by [Laumann & Poldrack](https://openfmri.org/dataset/ds000031/) and used in [Laumann et al., 2015](https://doi.org/10.1016/j.neuron.2015.06.037)[^7]. was also utilized. This data was obtained from the OpenfMRI database, accession number ds000031.

Modifications to the data included:

- Truncating the preprocessed BOLD data and confounds from 448 timepoints to 40 timepoints.
- Only including session 002 data.
- Adding a dataset_description.json file to the fmriprep folder.
- Excluding the nii.gz file in the root BIDS folder.
- Retaining only the mask, truncated preprocessed BOLD file, and truncated confounds file in the fmriprep folder.
- Slightly changing the naming style of the mask, preprocessed BOLD file, and confounds file in the fmriprep folder to conform with the naming conventions of modern fmriprep outputs.
- Testing with custom parcellations was done using the HCPex parcellation, an extension of the HCP (Human Connectome Project) parcellation, which adds 66 subcortical areas. This original atlas can be downloaded from.

Testing with custom parcellations was done with the HCPex parcellation, an extension of the HCP (Human Connectome Project) parcellation, which adds 66 subcortical areas [^8], [^9]. This original atlas can be downloaded from https://github.com/wayalan/HCPex.

## Contributing
Please refer the [contributing guidelines](https://github.com/donishadsmith/neurocaps/blob/test/CONTRIBUTING.md) on how to contribute to neurocaps.

## Acknowledgements
Neurocaps relies on several popular data processing, machine learning, neuroimaging, and visualization
[packages](https://neurocaps.readthedocs.io/en/latest/#dependencies).

Additionally, some foundational concepts in this package take inspiration from features or design patterns implemented
in other neuroimaging Python packages, specically:

- mtorabi59's [pydfc](https://github.com/neurodatascience/dFC), a toolbox that allows comparisons among several popular
dynamic functionality methods.
- 62442katieb's [idconn](https://github.com/62442katieb/IDConn), a pipeline for assessing individual differences in
resting-state or task-based functional connectivity.

## References
[^1]: Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals. NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

[^2]: Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia. NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

[^3]: Zhang, R., Yan, W., Manza, P., Shokri-Kojori, E., Demiral, S. B., Schwandt, M., Vines, L., Sotelo, D., Tomasi, D., Giddens, N. T., Wang, G., Diazgranados, N., Momenan, R., & Volkow, N. D. (2023).
Disrupted brain state dynamics in opioid and alcohol use disorder: attenuation by nicotine use. Neuropsychopharmacology, 49(5), 876–884. https://doi.org/10.1038/s41386-023-01750-w

[^4]: Ingwersen, T., Mayer, C., Petersen, M., Frey, B. M., Fiehler, J., Hanning, U., Kühn, S., Gallinat, J., Twerenbold, R., Gerloff, C., Cheng, B., Thomalla, G., & Schlemm, E. (2024). Functional MRI brain state occupancy in the presence of cerebral small vessel disease — A pre-registered replication analysis of the Hamburg City Health Study. Imaging Neuroscience, 2, 1–17. https://doi.org/10.1162/imag_a_00122

[^5]: Kupis, L., Romero, C., Dirks, B., Hoang, S., Parladé, M. V., Beaumont, A. L., Cardona, S. M., Alessandri, M., Chang, C., Nomi, J. S., & Uddin, L. Q. (2020). Evoked and intrinsic brain network dynamics in children with autism spectrum disorder. NeuroImage: Clinical, 28, 102396. https://doi.org/10.1016/j.nicl.2020.102396

[^6]: Hyunwoo Gu and Joonwon Lee and Sungje Kim and Jaeseob Lim and Hyang-Jung Lee and Heeseung Lee and Minjin Choe and Dong-Gyu Yoo and Jun Hwan (Joshua) Ryu and Sukbin Lim and Sang-Hun Lee (2024). Discrimination-Estimation Task. OpenNeuro. [Dataset] doi: https://doi.org/10.18112/openneuro.ds005381.v1.0.0

[^7]: Laumann, T. O., Gordon, E. M., Adeyemo, B., Snyder, A. Z., Joo, S. J., Chen, M. Y., Gilmore, A. W., McDermott, K. B., Nelson, S. M., Dosenbach, N. U., Schlaggar, B. L., Mumford, J. A., Poldrack, R. A., & Petersen, S. E. (2015). Functional system and areal organization of a highly sampled individual human brain. Neuron, 87(3), 657–670. https://doi.org/10.1016/j.neuron.2015.06.037

[^8]: Huang, CC., Rolls, E.T., Feng, J. et al. An extended Human Connectome Project multimodal parcellation atlas of the human cortex and subcortical areas. Brain Struct Funct 227, 763–778 (2022). https://doi.org/10.1007/s00429-021-02421-6

[^9]: Huang, C.-C., Rolls, E. T., Hsu, C.-C. H., Feng, J., & Lin, C.-P. (2021). Extensive Cortical Connectivity of the Human Hippocampal Memory System: Beyond the “What” and “Where” Dual Stream Model. Cerebral Cortex, 31(10), 4652–4669. https://doi.org/10.1093/cercor/bhab113
