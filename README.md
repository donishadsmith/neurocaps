# neurocaps
[![Latest Version](https://img.shields.io/pypi/v/neurocaps.svg)](https://pypi.python.org/pypi/neurocaps/)
[![Python Versions](https://img.shields.io/pypi/pyversions/neurocaps.svg)](https://pypi.python.org/pypi/neurocaps/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.11642615-teal)](https://doi.org/10.5281/zenodo.13922595)
[![Github Repository](https://img.shields.io/badge/Source%20Code-neurocaps-purple)](https://github.com/donishadsmith/neurocaps)
[![Test Status](https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml)
[![codecov](https://codecov.io/github/donishadsmith/neurocaps/graph/badge.svg?token=WS2V7I16WF)](https://codecov.io/github/donishadsmith/neurocaps)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Platform Support](https://img.shields.io/badge/OS-Ubuntu%20|%20macOS%20|%20Windows-blue)

This is a Python package designed to perform Co-activation Patterns (CAPs) analyses. It utilizes k-means clustering to group timepoints (TRs) into brain states, applicable to both resting-state and task-based fMRI data. The package is compatible with data preprocessed using **fMRIPrep** and assumes your directory is BIDS-compliant, containing a derivatives folder with a pipeline folder (such as fMRIPrep) that holds the preprocessed BOLD data.

This package was initially inspired by a co-activation patterns implementation in mtorabi59's [pydfc](https://github.com/neurodatascience/dFC) package.

# Installation
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

# Usage
**Note, documentation of each function can be found at https://neurocaps.readthedocs.io/en/latest/api.html**

**This package contains two main classes: `TimeseriesExtractor` for extracting the timeseries, and `CAP` for performing the CAPs analysis.**

**Note:** When extracting the timeseries, this package uses either the Schaefer atlas, the Automated Anatomical Labeling (AAL) atlas, or a custom parcellation that is lateralized (where each region/network has nodes in the left and right hemispheres). The number of ROIs and networks for the Schaefer atlas can be adjusted with the parcel_approach parameter when initializing the `TimeseriesExtractor` class.

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
- `"maps"`: Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g., .nii for NifTI files). For plotting purposes, this key is not required.
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

- **Timeseries Extraction:** Extract timeseries for resting-state or task data, creating a nested dictionary containing the subject ID, run number, and associated timeseries. This serves as input for the `get_caps` method in the `CAP` class.
- **Saving Timeseries:** Save the nested dictionary containing timeseries as a pickle file.
- **Visualization:** Visualize the timeseries of a Schaefer, AAL, or Custom parcellation node or region/network in a specific subject's run, with options to save the plots.
- **Parallel Processing:** Use parallel processing by specifying the number of CPU cores in the `n_cores` parameter in the `get_bold` method. Testing on an HPC using a loop with `TimeseriesExtractor.get_bold` to extract session 1 and 2 BOLD timeseries from 105 subjects from resting-state data (single run containing 360 volumes) and two task datasets (three runs containing 200 volumes each and two runs containing 200 volumes) reduced processing time from 5 hours 48 minutes to 1 hour 26 minutes (using 10 cores). *Note:* If you are using an HPC, remember to allocate the appropriate amount of CPU cores with your workload manager. For instance in slurm use `#SBATCH --cpus-per-task=10` if you intend to use 10 cores.

**Main features for `CAP` includes:**
- **Optimal Cluster Size Identification:** Perform the Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions to identify the optimal cluster size, saving the optimal model as an attribute.
- **Parallel Processing:** Use parallel processing, when using the Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions , by specifying the number of CPU cores in the `n_cores` parameter in the `get_caps` method. *Note:* If you are using an HPC, remember to allocate the appropriate amount of CPU cores with your workload manager. For instance in slurm use `#SBATCH --cpus-per-task=10` if you intend to use 10 cores.
- **Grouping:** Perform CAPs analysis for entire sample or groups of subject IDs (using the `groups` parameter when initializing the `CAP` class). K-means clustering, all cluster selection methods (Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions), and plotting are done for each group when specified.
- **CAP Visualization:** Visualize the CAPs as outer products or heatmaps, with options to use subplots to reduce the number of individual plots, as well as save. Refer to the [documentation](https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2plot) for the `caps2plot` method in the `CAP` class for available `**kwargs` arguments and parameters to modify plots.
- **Save CAPs as NifTIs:** Convert the atlas used for parcellation to a stat map and saves them (`caps2niftis`). 
- **Surface Plot Visualization:** Convert the atlas used for parcellation to a stat map projected onto a surface plot with options to customize and save plots. Refer to the [documentation](https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2surf) for the `caps2surf` method in the `CAP` class for available `**kwargs` arguments and parameters to modify plots. Also includes the option to save the NifTIs. There is also another a parameter in `caps2surf`, `fslr_giftis_dict`, which can be used if the CAPs NifTI files were converted to GifTI files using a tool such as Connectome Workbench, which may work better for converting your atlas to fslr space. This parameter allows plotting without re-running the analysis and only initializing the `CAP` class and using the `caps2surf` method is needed.
- **Correlation Matrix Creation:** Create a correlation matrix from CAPs with options to customize and save plots. Additionally can produce dataframes where each element contains its associated uncorrected p-value in parentheses that is accompanied by an asterisk using the following significance code `{"<0.05": "*", "<0.01": "**", "<0.001": "***"}`. Refer to the [documentation](https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2corr) for the `caps2corr` method in the `CAP` class for available `**kwargs` arguments and parameters to modify plots.
- **CAP Metrics Calculation:** Calculate CAP metrics (`calculate_metrics`) as described in [Liu et al., 2018](https://doi.org/10.1016/j.neuroimage.2018.01.041)[^1] and [Yang et al., 2021](https://doi.org/10.1016/j.neuroimage.2021.118193)[^2]:
    - *Temporal Fraction:* The proportion of total volumes spent in a single CAP over all volumes in a run.
       Additionally, in the supplementary material of Yang et al., the stated relationship between
       temporal fraction, counts, and persistence is temporal fraction = (persistence*counts)/total volumes
       If persistence and temporal fraction is converted into time units, then temporal fraction = (persistence*counts)/(total volumes * TR)
        ```python
            predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
            target = 1
            temporal_fraction = 4/6
        ```
    - *Persistence:* The average time spent in a single CAP before transitioning to another CAP
      (average consecutive/uninterrupted time).
        ```python
            predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
            target = 1
            # Sequences for 1 are [1] and [1,1,1]
            persistence = (1 + 3)/2 # Average number of frames
            tr = 2
            if tr:
                persistence = ((1 + 3) * 2)/2 # Turns average frames into average time
        ```
    - *Counts:* The total number of initiations of a specific CAP across an entire run. An initiation is
           defined as the first occurrence of a CAP. If the same CAP is maintained in contiguous segment
           (indicating stability), it is still counted as a single initiation. 
        ```python
            predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
            target = 1
            # Initiations of CAP-1 occur at indices 0 and 2
            counts = 2
        ```
    - *Transition Frequency:* The number of transitions between different CAPs across the entire run.
        ```python
            predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
            # Transitions between unique CAPs occur at indices 0 -> 1, 1 -> 2, and 4 -> 5
            transition_frequency = 3
        ```

    - *Transition Probability:* The probability of transitioning from one CAP to another CAP (or the same CAP).
    This is calculated as (Number of transitions from A to B)/ (Total transitions from A). Note that the transition
    probability from CAP-A -> CAP-B is not the same as CAP-B -> CAP-A.
    ```python
        # Note last two numbers in the predicted timeseries are switched for this example
        predicted_subject_timeseries = [1, 2, 1, 1, 3, 1]
        # If three CAPs were identified in the analysis
        combinations = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)]
        target = (1,2) # Represents transition from CAP-1 -> CAP-2
        # There are 4 ones in the timeseries but only three transitions from 1; 1 -> 2, 1 -> 1, 1 -> 3
        n_transitions_from_1 = 3
        # There is only one 1 -> 2 transition.
        transition_probability = 1/3
        # 1 -> 1 has a probability of 1/3 and 1 -> 3 has a probability of 1/3
    ```
- **Cosine Similarity Radar Plots:** Create radar plots showing the cosine similarity between positive and negative
activations of each CAP and each a-priori regions in a parcellation [^3] [^4].
Refer to the [documentation](https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2radar)
in `caps2radar` in the `CAP` class for a more detailed explanation as well as available `**kwargs` arguments and parameters to modify plots.

    ```python
    import numpy as np
                            
    # Define nodes with their corresponding label IDs
    nodes = ["LH_Vis1", "LH_Vis2", "LH_SomSot1", "LH_SomSot2",
            "RH_Vis1", "RH_Vis2", "RH_SomSot1", "RH_SomSot2"]
    # Binary mask for the Visual Network (Vis)
    binary_vector = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    # Example cluster centroid for CAP 1
    cap_1_cluster_centroid = np.array([-0.3, 1.5, 2.0, -0.2, 0.7, 1.3, -0.5, 0.4])
    # Assign values less than 0 as 0 to isolate the high amplitude activations
    high_amp = np.where(cap_1_cluster_centroid > 0, cap_1_cluster_centroid, 0)
    # Assign values less than 0 as 0 to isolate the low amplitude activations; Also invert the sign
    low_amp = high_amp = np.where(cap_1_cluster_centroid < 0, -cap_1_cluster_centroid, 0)

    # Compute dot product between the binary vector with the positive and negative activations
    high_dot = np.dot(high_amp, binary_vector)
    low_dot = np.dot(low_amp, binary_vector)

    # Compute the norms
    high_norm = np.linalg.norm(high_amp)
    low_norm = np.linalg.norm(low_amp)
    bin_norm = np.linalg.norm(binary_vector)

    # Calculate cosine similarity
    high_cos = high_dot / (high_norm * bin_norm)
    low_cos = low_dot / (low_norm * bin_norm)
    ```

**Additionally, the `neurocaps.analysis` submodule contains two additional functions:**

- `merge_dicts`: Merge the subject_timeseries dictionaries for overlapping subjects across tasks to identify similar CAPs across different tasks. The merged dictionary can be saved as a pickle file.
- `standardize`: Standardizes each run independently for all subjects in the subject timeseries.
- `change_dtype`: Changes the dtype of all subjects in the subject timeseries to help with memory usage.
- `transition_matrix`: Uses the "transition_probability" output from ``CAP.calculate_metrics`` to generate and visualize the averaged transition probability matrix for all groups from the analysis.

Please refer to [demo.ipynb](https://github.com/donishadsmith/neurocaps/blob/main/demo.ipynb) or https://neurocaps.readthedocs.io/en/latest/examples/examples.html for a more extensive demonstration of the features included in this package.

**Quick Code Examples (Examples use randomized data)**:

```python

from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP

"""If an asterisk '*' is after a name, all confounds starting with the 
term preceding the parameter will be used. in this case, all parameters 
starting with cosine will be used."""

confounds = ['Cosine*', 'Rot*']

"""If use_confounds is True but no confound_names provided, there are hardcoded 
confound names that will extract the data from the confound files outputted by fMRIPrep
`n_acompcor_separate` will use the first 'n' components derived from the separate 
white-matter (WM) and cerebrospinal fluid (CSF). To use the acompcor components from the 
combined mask, list them in the `confound_names` parameter"""

parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 2}}

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

# Resting State
extractor.get_bold(bids_dir=bids_dir,
                   task="rest",
                   session='002',
                   pipeline_name=pipeline_name
                   verbose=True,
                   flush=True)
```
**Output:**
```
2024-09-16 00:17:11,689 [INFO] Confound regressors to be used if available: Cosine*, aComp*, Rot*.
2024-09-16 00:17:12,113 [INFO] BIDS Layout: ...0.4_ses001-022/ds000031_R1.0.4 | Subjects: 1 | Sessions: 1 | Runs: 1
2024-09-16 00:17:13,914 [INFO] [SUBJECT: 01 | SESSION: 002 | TASK: rest | RUN: 001] Preparing for timeseries extraction using [FILE: sub-01_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz].
2024-09-16 00:17:13,917 [INFO] [SUBJECT: 01 | SESSION: 002 | TASK: rest | RUN: 001] The following confounds will be used for nuisance regression: Cosine00, Cosine01, Cosine02, Cosine03, Cosine04, Cosine05, Cosine06, aCompCor00, aCompCor01, aCompCor02, aCompCor03, aCompCor04, aCompCor05, RotX, RotY, RotZ.
```

```python
# Task; use parallel processing with `n_cores`
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
# You can use seaborn's premade palettes as strings or generate your own custom palettes
# Using seaborn's diverging_palette function, matplotlib's LinearSegmentedColormap, 
# or other Classes or functions compatible with seaborn
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
kwargs = {"cmap": "cold_hot", "layout": "row", "size": (500, 100), "zoom": 1,
          "cbar_location":"bottom"}

cap_analysis.caps2surf(fwhm=2, **kwargs)

#You can also generate your own colormaps using matplotlib's LinearSegmentedColormap

# Create the colormap
from matplotlib.colors import LinearSegmentedColormap

colors = ["#1bfffe", "#00ccff", "#0099ff", "#0066ff", "#0033ff", "#c4c4c4",
          "#ff6666", "#ff3333", "#FF0000","#ffcc00","#FFFF00"]

custom_cmap = LinearSegmentedColormap.from_list("custom_cold_hot", colors, N=256)

kwargs["cmap"] = custom_cmap

cap_analysis.caps2surf(fwhm=2, **kwargs)
```
**Partial Plot Outputs:** (*Note*: one image will be generated per CAP)
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/fadc946a-214b-4fbf-8316-2f32ab0b026e)
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/8207914a-6bf0-47a9-8be8-3504d0a56516)

```python
# Create correlation matrix

kwargs = {"annot": True ,"figsize": (6,4), "cmap": "coolwarm"}

cap_analysis.caps2corr(**kwargs)

# You can use seaborn's premade palettes as strings or generate your own custom palettes
# Using seaborn's diverging_palette function, matplotlib's LinearSegmentedColormap, 
# or other Classes or functions compatable with seaborn

# Create the colormap
import seaborn as sns

palette = sns.diverging_palette(260, 10, s=80, l=55, n=256, as_cmap=True)

kwargs["cmap"] = palette

cap_analysis.caps2corr(**kwargs)
```
**Plot Output:**
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/57a2ce81-13d3-40d0-93e7-0ca910f7b0be)
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/9a8329df-65c7-4ad0-8b81-edc73f2d960d)

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
![All_Subjects_CAP-1_radar](https://github.com/user-attachments/assets/b190b209-a036-46a5-881f-3d40cffda1c0)
![All_Subjects_CAP-2_radar](https://github.com/user-attachments/assets/8bd56af5-fbe9-4d57-8f58-2c332af986f9)
![All_Subjects_CAP-3_radar](https://github.com/user-attachments/assets/81b739f4-bd7f-41bf-9b42-14d8376b5239)

```python
# Get transition probabilities for all participants in a dataframe, then convert to an averaged matrix
from neurocaps.analysis import transition_matrix

# Optimal cluster sizes are saved automatically
cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, 
                      cluster_selection_method="davies_bouldin",
                      standardize=True,
                      n_clusters=list(range(2,6)))

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
2024-09-16 00:09:54,273 [INFO] [GROUP: All Subjects | METHOD: davies_bouldin] Optimal cluster size is 3.
```
| Subject_ID | Group | Run | 1.1 | 1.2 | 1.3 | 2.1 | 2.2 | 2.3 | 3.1 | 3.2 | 3.3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | All Subjects | continuous_runs | 0.326 | 0.261 | 0.413 | 0.245 | 0.449 | 0.306 | 0.352 | 0.278 | 0.37 |
| 2 | All Subjects | continuous_runs | 0.4 | 0.25 | 0.35 | 0.486 | 0.108 | 0.405 | 0.346 | 0.365 | 0.288 |
| 3 | All Subjects | continuous_runs | 0.354 | 0.229 | 0.417 | 0.383 | 0.362 | 0.255 | 0.241 | 0.352 | 0.407 |
| 4 | All Subjects | continuous_runs | 0.283 | 0.37 | 0.348 | 0.302 | 0.321 | 0.377 | 0.32 | 0.38 | 0.3 |
| 5 | All Subjects | continuous_runs | 0.292 | 0.354 | 0.354 | 0.38 | 0.28 | 0.34 | 0.294 | 0.392 | 0.314 |
| 6 | All Subjects | continuous_runs | 0.339 | 0.304 | 0.357 | 0.333 | 0.231 | 0.436 | 0.444 | 0.222 | 0.333 |
| 7 | All Subjects | continuous_runs | 0.424 | 0.203 | 0.373 | 0.45 | 0.275 | 0.275 | 0.34 | 0.32 | 0.34 |
| 8 | All Subjects | continuous_runs | 0.25 | 0.271 | 0.479 | 0.39 | 0.244 | 0.366 | 0.35 | 0.3 | 0.35 |
| 9 | All Subjects | continuous_runs | 0.429 | 0.265 | 0.306 | 0.319 | 0.298 | 0.383 | 0.245 | 0.377 | 0.377 |
| 10 | All Subjects | continuous_runs | 0.333 | 0.375 | 0.292 | 0.306 | 0.347 | 0.347 | 0.327 | 0.269 | 0.404 |

![image](https://github.com/user-attachments/assets/3ab1d123-0c3e-47e3-b24c-52bfda13d3ef)

| | CAP-1 | CAP-2 | CAP-3 |
| --- | --- | --- | --- |
| CAP-1 | 0.343 | 0.288 | 0.369 |
| CAP-2 | 0.36 | 0.291 | 0.349 |
| CAP-3 | 0.326 | 0.326 | 0.348 |

# Testing 
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

# Contributing
Please refer the [contributing guidelines](https://github.com/donishadsmith/neurocaps/blob/test/CONTRIBUTING.md) on how to contribute to neurocaps.

# References
[^1]: Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals. NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

[^2]: Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia. NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

[^3]: Zhang, R., Yan, W., Manza, P., Shokri-Kojori, E., Demiral, S. B., Schwandt, M., Vines, L., Sotelo, D., Tomasi, D., Giddens, N. T., Wang, G., Diazgranados, N., Momenan, R., & Volkow, N. D. (2023). 
Disrupted brain state dynamics in opioid and alcohol use disorder: attenuation by nicotine use. Neuropsychopharmacology, 49(5), 876–884. https://doi.org/10.1038/s41386-023-01750-w      

[^4]: Ingwersen, T., Mayer, C., Petersen, M., Frey, B. M., Fiehler, J., Hanning, U., Kühn, S., Gallinat, J., Twerenbold, R., Gerloff, C., Cheng, B., Thomalla, G., & Schlemm, E. (2024). Functional MRI brain state occupancy in the presence of cerebral small vessel disease — A pre-registered replication analysis of the Hamburg City Health Study. Imaging Neuroscience, 2, 1–17. https://doi.org/10.1162/imag_a_00122

[^5]: Laumann, T. O., Gordon, E. M., Adeyemo, B., Snyder, A. Z., Joo, S. J., Chen, M. Y., Gilmore, A. W., McDermott, K. B., Nelson, S. M., Dosenbach, N. U., Schlaggar, B. L., Mumford, J. A., Poldrack, R. A., & Petersen, S. E. (2015). Functional system and areal organization of a highly sampled individual human brain. Neuron, 87(3), 657–670. https://doi.org/10.1016/j.neuron.2015.06.037

[^6]: Huang CC, Rolls ET, Feng J, Lin CP. An extended Human Connectome Project multimodal parcellation atlas of the human cortex and subcortical areas. Brain Struct Funct. 2022 Apr;227(3):763-778. Epub 2021 Nov 17. doi: 10.1007/s00429-021-02421-6

[^7]: Huang CC, Rolls ET, Hsu CH, Feng J, Lin CP. Extensive Cortical Connectivity of the Human Hippocampal Memory System: Beyond the "What" and "Where" Dual Stream Model. Cerebral Cortex. 2021 May 19;bhab113. doi: 10.1093/cercor/bhab113.
