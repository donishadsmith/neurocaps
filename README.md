# NeuroCAPs: Neuroimaging Co-Activation Patterns

[![Latest Version](https://img.shields.io/pypi/v/neurocaps.svg)](https://pypi.python.org/pypi/neurocaps/)
[![Python Versions](https://img.shields.io/pypi/pyversions/neurocaps.svg)](https://pypi.python.org/pypi/neurocaps/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.11642615-teal)](https://doi.org/10.5281/zenodo.16430050)
[![Github Repository](https://img.shields.io/badge/Source%20Code-neurocaps-purple)](https://github.com/donishadsmith/neurocaps)
[![Test Status](https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml/badge.svg)](https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml)
[![Documentation Status](https://readthedocs.org/projects/neurocaps/badge/?version=stable)](http://neurocaps.readthedocs.io/en/stable/?badge=stable)
[![Codecov](https://codecov.io/github/donishadsmith/neurocaps/graph/badge.svg?token=WS2V7I16WF)](https://codecov.io/github/donishadsmith/neurocaps)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Platform Support](https://img.shields.io/badge/OS-Ubuntu%20|%20macOS%20|%20Windows-blue)
[![Docker](https://img.shields.io/badge/docker-donishadsmith/neurocaps-darkblue.svg?logo=docker)](https://hub.docker.com/r/donishadsmith/neurocaps/tags/)
[![JOSS](https://joss.theoj.org/papers/0e5c44d5d82402fa0f28e6a8833428f0/status.svg)](https://joss.theoj.org/papers/0e5c44d5d82402fa0f28e6a8833428f0)

NeuroCAPs (**Neuro**imaging **C**o-**A**ctivation **P**attern**s**) is a Python package for performing Co-Activation
Patterns (CAPs) analyses on resting-state or task-based fMRI data. CAPs identifies recurring brain states by applying
k-means clustering on BOLD timeseries data [^1].

<img src="docs/assets/workflow.png">

## Installation
**NeuroCAPs requires Python 3.9-3.12.**

To install NeuroCAPs, follow the instructions below using your preferred terminal.

### Standard Installation from PyPi
```bash

pip install neurocaps

```

#### Windows Users
PyBIDS will not be installed by default due to installation errors that may occur if long paths
aren't enabled (Refer to official [Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell)
to enable this feature).

To include PyBIDS in your installation, use:

```bash

pip install neurocaps[windows]

```

Alternatively, you can install PyBIDS separately:

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

git clone --depth 1 https://github.com/donishadsmith/neurocaps/
cd neurocaps
pip install -e .
# Clone with submodules to include test dataset ~140 MB
git submodule update --init

```

#### Windows Users
To include PyBIDS when installing the development version on Windows, use:

```bash

git clone --depth 1 https://github.com/donishadsmith/neurocaps/
cd neurocaps
pip install -e .[windows]
# Clone with submodules to include test dataset ~140 MB
git submodule update --init
```

## Docker
If [Docker](https://docs.docker.com/) is available on your system, you can use the NeuroCAPs Docker
image, which includes the demos and configures a headless display for VTK.

To pull the Docker image:
```bash

docker pull donishadsmith/neurocaps && docker tag donishadsmith/neurocaps neurocaps
```

The image can be run as:

1. An interactive bash session (default):

```bash

docker run -it neurocaps
```

2. A Jupyter Notebook with port forwarding:

```bash

docker run -it -p 9999:9999 neurocaps notebook
```

## Features
NeuroCAPs is built around two main classes (``TimeseriesExtractor`` and ``CAP``) and includes several
features to perform the complete CAPs workflow from postprocessing to visualizations.
Notable features includes:

- Timeseries Extraction (``TimeseriesExtractor``):
    - extracts BOLD timeseries from resting-state or task-based fMRI data
    - supports deterministic parcellations such as the Schaefer and AAL, in addition to custom-defined deterministic parcellations
    - performs nuisance regression, motion scrubbing, and additional features
    - reports quality control information based on framewise displacement

    **Important**:
       NeuroCAPs is most optimized for fMRI data preprocessed with
       [fMRIPrep](https://fmriprep.org/en/stable/) and assumes the data is BIDs compliant.
       Refer to [NeuroCAPs' BIDS Structure and Entities Documentation](https://neurocaps.readthedocs.io/en/stable/bids.html)
       for additional information.

- CAPs Analysis (``CAP``):
    - performs k-means clustering on individuals or groups
    - identifies the optimal number of clusters using Silhouette, Elbow, Davies Bouldin, or Variance Ratio methods
    - computes several temporal dynamic metrics [^2] [^3]:
        - temporal fraction (fraction of time)
        - persistence (dwell time)
        - counts (state initiation)
        - transition frequency & probability
    - produces several visualizations:
        - heatmaps and outer product plots
        - surface plots
        - correlation matrices
        - cosine similarity radar plots [^4] [^5]

- Utilities:
  - plot transition matrices
  - merge timeseries data across tasks or session [^6]
  - generate the custom parcellation dictionary structure from the parcellation's metadata file
  - fetch preset custom parcellation approaches

Full details for every function and parameter are available in the
[API Documentation](https://neurocaps.readthedocs.io/en/stable/api.html).

## Quick Start
The following code demonstrates a high-level example using NeuroCAPs (with simulated data) to
perform the CAPs analysis. A variant of this example using real data from [OpenNeuro](https://openneuro.org/)
is available on the [readthedocs](https://neurocaps.readthedocs.io/en/stable/tutorials/tutorial-8.html).
Additional [tutorials]([demos](https://neurocaps.readthedocs.io/en/stable/tutorials/)) and
[interactive demonstrations](https://github.com/donishadsmith/neurocaps/tree/main/demos) are
also available.

1. Extract timeseries data
```python
import numpy as np

from neurocaps.extraction import TimeseriesExtractor
from neurocaps.utils import simulate_bids_dataset

# Set seed
np.random.seed(0)

# Generate a simulated BIDS directory with fMRIPrep derivatives
# or replace with a real BIDs dataset with fMRIPrep derivatives
bids_root = simulate_bids_dataset(
    n_subs=3, n_runs=1, n_volumes=100, task_name="rest"
)

# Using Schaefer, one of the default parcellation approaches
parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}

# List of fMRIPrep-derived confounds for nuisance regression
confound_names = [
    "cosine*",
    "trans*",
    "rot*",
    "a_comp_cor_00",
    "a_comp_cor_01",
    "a_comp_cor_02",
    "a_comp_cor_03",
    "a_comp_cor_04",
]

# Initialize extractor with signal cleaning parameters
extractor = TimeseriesExtractor(
    space="MNI152NLin2009cAsym",
    parcel_approach=parcel_approach,
    confound_names=confound_names,
    standardize=False,
    fd_threshold={
        "threshold": 0.90,
        "outlier_percentage": 0.30,
    },
)

# Extract BOLD data from preprocessed fMRIPrep data
# which should be located in the "derivatives" folder
# within the BIDS root directory
# The extracted timeseries data is automatically stored
extractor.get_bold(
    bids_dir=bids_root, task="rest", tr=2, n_cores=1, verbose=False
)

# Retrieve the dataframe containing QC information for each subject
# to use for downstream statistical analyses
qc_df = extractor.report_qc()
print(qc_df)
```

| Subject_ID | Run | Mean_FD | Std_FD | Frames_Scrubbed | ... |
|------------|-----|---------|--------|-----------------|-----|
| 0 | run-0 | 0.516349 | 0.289657 |  9 | ... |
| 1 | run-0 | 0.526343 | 0.297550 | 17 | ... |
| 2 | run-0 | 0.518041 | 0.273964 |  8 | ... |

2. Use k-means clustering to identify the optimal number of CAPs from the data using a heuristic
```python
from neurocaps.analysis import CAP

# Initialize CAP class
cap_analysis = CAP(parcel_approach=extractor.parcel_approach)

# Identify the optimal number of CAPs (clusters)
# using the silhouette method to test 2-20
# The optimal number of CAPs is automatically stored
cap_analysis.get_caps(
    subject_timeseries=extractor.subject_timeseries,
    n_clusters=range(2, 21),
    standardize=True,
    cluster_selection_method="silhouette",
    max_iter=500,
    n_init=10,
)
```

3. Compute temporal dynamic metrics for downstream statistical analyses
```python
# Calculate temporal fraction of each CAP for all subjects
metric_dict = cap_analysis.calculate_metrics(
    extractor.subject_timeseries, metrics=["temporal_fraction"]
)
print(metric_dict["temporal_fraction"])
```

| Subject_ID | Group | Run | CAP-1 | CAP-2 |
|------------|-------|-----|-------|-------|
| 0 | All Subjects | run-0 | 0.505495 | 0.494505 |
| 1 | All Subjects | run-0 | 0.530120 | 0.469880 |
| 2 | All Subjects | run-0 | 0.521739 | 0.478261 |

4. Visualize CAPs
```python
# Project CAPs onto surface plots
# and generate cosine similarity network alignment of CAPs
from neurocaps.utils import PlotDefaults

surface_kwargs = PlotDefaults.caps2surf()
surface_kwargs["layout"] = "row"
surface_kwargs["size"] = (500, 100)

radar_kwargs = PlotDefaults.caps2radar()
radar_kwargs["height"] = 400
radar_kwargs["width"] = 600

radialaxis = {
    "showline": True,
    "linewidth": 2,
    "linecolor": "rgba(0, 0, 0, 0.25)",
    "gridcolor": "rgba(0, 0, 0, 0.25)",
    "ticks": "outside",
    "tickfont": {"size": 14, "color": "black"},
    "range": [0, 0.4],
    "tickvals": [0.1, "", "", 0.4],
}

legend = {
    "yanchor": "top",
    "y": 0.99,
    "x": 0.99,
    "title_font_family": "Times New Roman",
    "font": {"size": 12, "color": "black"},
}

radar_kwargs["radialaxis"] = radialaxis
radar_kwargs["legend"] = legend

cap_analysis.caps2surf(**surface_kwargs).caps2radar(**radar_kwargs)
```

![CAP-1 Surface Image.](paper/cap_1_surface.png)

![CAP-2 Surface Image.](paper/cap_2_surface.png)

![CAP-1 Radar Image.](paper/cap_1_radar.png)

![CAP-2 Radar Image.](paper/cap_2_radar.png)

```python
import pandas as pd

df = pd.DataFrame(cap_analysis.cosine_similarity["All Subjects"]["CAP-1"])
# Note for "Low Amplitude" the absolute values of the
# negative cosine similarities are stored
df["Net"] = df["High Amplitude"] - df["Low Amplitude"]
df["Regions"] = cap_analysis.cosine_similarity["All Subjects"]["Regions"]
print(df)
```
| High Amplitude | Low Amplitude | Net | Regions |
|----------------|---------------|-----|---------|
| 0.340826 | 0.309850 | 0.030976 | Vis |
| 0.155592 | 0.318072 | -0.162480 | SomMot |
| 0.213348 | 0.181667 | 0.031681  | DorsAttn |
| 0.287179 | 0.113046 | 0.174133  | SalVentAttn |
| 0.027542 | 0.168325 | -0.140783 | Limbic |
| 0.236915 | 0.195235 | 0.041680  | Cont |
| 0.238242 | 0.208548 | 0.029694 | Default |

```python
df = pd.DataFrame(cap_analysis.cosine_similarity["All Subjects"]["CAP-2"])
df["Net"] = df["High Amplitude"] - df["Low Amplitude"]
df["Regions"] = cap_analysis.cosine_similarity["All Subjects"]["Regions"]
print(df)
```

| High Amplitude | Low Amplitude | Net | Regions |
|----------------|---------------|-----|---------|
| 0.309850 | 0.340826 | -0.030976 | Vis |
| 0.318072 | 0.155592 | 0.162480  | SomMot |
| 0.181667 | 0.213348 | -0.031681 | DorsAttn |
| 0.113046 | 0.287179 | -0.174133 | SalVentAttn |
| 0.168325 | 0.027542 | 0.140783  | Limbic |
| 0.195235 | 0.236915 | -0.041680 | Cont |
| 0.208548 | 0.230242 | -0.021694 | Default |


Note: For information about logging, refer to [NeuroCAPs' Logging Guide](https://neurocaps.readthedocs.io/en/stable/user_guide/logging.html).

## Acknowledgements
NeuroCAPs relies on several popular data processing, machine learning, neuroimaging, and visualization
[packages](https://neurocaps.readthedocs.io/en/stable/#dependencies).

Additionally, some foundational concepts in this package take inspiration from features or design
patterns implemented in other neuroimaging Python packages, specically:

- mtorabi59's [pydfc](https://github.com/neurodatascience/dFC), a toolbox that allows comparisons
among several popular dynamic functionality methods.
- 62442katieb's [IDConn](https://github.com/62442katieb/IDConn), a pipeline for assessing individual
differences in resting-state or task-based functional connectivity.

## Reporting Issues
Bug reports, feature requests, and documentation enhancements can be reported using the
templates offered when creating a new issue in the
[issue tracker](https://github.com/donishadsmith/neurocaps/issues).

## Contributing
Please refer the [contributing guidelines](https://neurocaps.readthedocs.io/en/stable/contributing.html)
on how to contribute to NeuroCAPs.

## References
[^1]: Liu, X., Chang, C., & Duyn, J. H. (2013). Decomposition of spontaneous brain activity into
distinct fMRI co-activation patterns. Frontiers in Systems Neuroscience, 7.
https://doi.org/10.3389/fnsys.2013.00101

[^2]: Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state
fMRI signals. NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

[^3]: Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible
coactivation patterns of functional brain networks reveal the aberrant dynamic state transition in
schizophrenia. NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

[^4]: Zhang, R., Yan, W., Manza, P., Shokri-Kojori, E., Demiral, S. B., Schwandt, M., Vines, L.,
Sotelo, D., Tomasi, D., Giddens, N. T., Wang, G., Diazgranados, N., Momenan, R., & Volkow, N. D. (2023).
Disrupted brain state dynamics in opioid and alcohol use disorder: attenuation by nicotine use.
Neuropsychopharmacology, 49(5), 876–884. https://doi.org/10.1038/s41386-023-01750-w

[^5]: Ingwersen, T., Mayer, C., Petersen, M., Frey, B. M., Fiehler, J., Hanning, U., Kühn, S.,
Gallinat, J., Twerenbold, R., Gerloff, C., Cheng, B., Thomalla, G., & Schlemm, E. (2024).
Functional MRI brain state occupancy in the presence of cerebral small vessel disease —
A pre-registered replication analysis of the Hamburg City Health Study. Imaging Neuroscience,
2, 1–17. https://doi.org/10.1162/imag_a_00122

[^6]: Kupis, L., Romero, C., Dirks, B., Hoang, S., Parladé, M. V., Beaumont, A. L., Cardona, S. M.,
Alessandri, M., Chang, C., Nomi, J. S., & Uddin, L. Q. (2020). Evoked and intrinsic brain network
dynamics in children with autism spectrum disorder. NeuroImage: Clinical, 28, 102396.
https://doi.org/10.1016/j.nicl.2020.102396

[^7]: Hyunwoo Gu and Joonwon Lee and Sungje Kim and Jaeseob Lim and Hyang-Jung Lee and Heeseung Lee
and Minjin Choe and Dong-Gyu Yoo and Jun Hwan (Joshua) Ryu and Sukbin Lim and Sang-Hun Lee (2024).
Discrimination-Estimation Task. OpenNeuro. [Dataset] doi: https://doi.org/10.18112/openneuro.ds005381.v1.0.0
