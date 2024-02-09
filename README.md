# neurocaps
This is an alpha Python package to perform Co-activation Patterns (CAPs) analyses, which involves using kmeans clustering to group timepoints (TR's) into brain states, on both resting-state or task data. It is compatabe with BIDS-compliant directory and assumes your directory contains a derivatives folder with a pipeline folder, such as fMRIPrep, containing preprocessed BOLD data.

**This package is still in development but is functional. Currently, it allows visualizations of CAPs but does not calculate CAPs metrics, which will be included in a future update. Has been tested using data preprocessed by fMRIPrep.**

# Installation

This package uses pybids, which is only functional on POSIX operating system and Mac OS. To install, using your preferred terminal:

Clone repository

```bash

git clone https://github.com/donishadsmith/neurocaps/

```

Change directory

```bash

cd neurocaps

```

Install 

```bash
pip install -e .

```

# Usage
 This package contains two main classes - `TimeseriesExtractor`, for extracting the timeseries, and `CAP`, for performing the cap analysis.

Note: When extracting the timeseries, this package uses the Schaefer atlas. The number of ROIs and networks for the Schaefer atlas can be modified with `n_rois`` and `n_networks` when initializing the main `TimeseriesExtractor` class.

Main features for `TimeseriesExtractor` include:

- Timeseries extraction for resting state or task data and creating a nested dictionary containing the subject ID, run number, and associated timeseries. This is used as input for the `get_caps()` method in the `CAP` class.
- Saving the nested dictionary containing timeseries as a pickle file.
- Visualizing the timeseries of a Schaefer node or network subject's run. Also includes the ability to save plots.

Main features for `CAP` include:

- Performing the silhouette or elbow method to identify the optimal cluster size. When the optimal cluster size is identified, the optimal model is saved as an attribute.
- Visualizing the CAPs identified as an outer product or regular heatmap. For outer products, you also have the ability to use subplots to reduce the number of individual plots. You can also save the plots and use them. Please refer to the docstring for the `visualize_caps()` method in the `CAP` class to see the list of available kwargs arguments to modify plots.
- Grouping feature to perform CAPs independently on groups of subject IDs. When grouping is specified, k-means clustering, silhouette and elbow methods, as well as plotting, are done for each independent group.
- Calculating CCAP metrics as described in [Yang et al., 2021](https://doi.org/10.1016/j.neuroimage.2021.118193) where `fraction of time` is the proportion of total volumes spent in a single CAP over all volumes in a run,
`persistence` is the average time spent in a single CAP before transitioning to another CAP (average consecutive/uninterrupted time), and `counts` is the frequency of each CAP observed in a run.

Please refer to [demo.ipynb](https://github.com/donishadsmith/neurocaps/blob/main/demo.ipynb) to see multiple examples of how to use this package.

Quick code example

```python

from neurocaps import TimeseriesExtractor, CAP

# If use_confounds is True but no confound_names provided, there are hardcoded confound names that will extract the data from the confound files outputted by fMRIPrep

extractor = TimeseriesExtractor(n_rois=100, standardize="zscore_sample", use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01)

bids_dir = "/path/to/bids/dir"

# If there are multiple pipelines in the derivatives folder

pipeline_name = "fmriprep-1.4.0"

# Resting State
# extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

# Task
extractor.get_bold(bids_dir=bids_dir, task="emo", condition="positive", pipeline_name=pipeline_name)

cap_analysis = CAP(node_labels=extractor.atlas_labels, n_clusters=6)

cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, standardize = True)

cap_analysis.visualize_caps(visual_scope="networks", plot_options="outer product", task_title="- Positive Valence", ncol=3, sharey=True, subplots=True)

cap_analysis.visualize_caps(visual_scope="nodes", plot_options="outer product", task_title="- Positive Valence", ncol=3, sharey=True, subplots=True, xlabel_rotation=90, tight_layout=False, hspace = 0.4)

cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True, output_dir=output_dir, continuous_runs=True, file_name="All_Subjects_CAPs_metrics")

```

![image](https://github.com/donishadsmith/neurocaps/assets/112973674/4699bbd9-1f55-462b-9d9e-4ef17da79ad4)
![image](https://github.com/donishadsmith/neurocaps/assets/112973674/506c5be5-540d-43a9-8a61-c02062f5c6f9)


<details>
  
  <summary>DataFrame</summary>

    | Subject_ID | Group | Run | Metric | CAP-1 | CAP-2 | CAP-3 | CAP-4 | CAP-5 | CAP-6 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | 1 | All Subjects | Continuous Runs | Fraction of Time | 0.16 | 0.14 | 0.18 | 0.19 | 0.16 | 0.17 |
    | 1 | All Subjects | Continuous Runs | Counts | 47.0 | 43.0 | 54.0 | 57.0 | 47.0 | 52.0 |
    | 1 | All Subjects | Continuous Runs | Persistence | 1.34 | 1.34 | 1.29 | 1.19 | 1.18 | 1.24 |
    | 2 | All Subjects | Continuous Runs | Fraction of Time | 0.17 | 0.1 | 0.15 | 0.2 | 0.21 | 0.18 |
    | 2 | All Subjects | Continuous Runs | Counts | 50.0 | 29.0 | 45.0 | 61.0 | 62.0 | 53.0 |
    | 2 | All Subjects | Continuous Runs | Persistence | 1.19 | 1.32 | 1.12 | 1.24 | 1.33 | 1.2 |
    | 3 | All Subjects | Continuous Runs | Fraction of Time | 0.14 | 0.16 | 0.14 | 0.18 | 0.22 | 0.16 |
    | 3 | All Subjects | Continuous Runs | Counts | 42.0 | 49.0 | 41.0 | 53.0 | 66.0 | 49.0 |
    | 3 | All Subjects | Continuous Runs | Persistence | 1.14 | 1.11 | 1.17 | 1.33 | 1.14 | 1.14 |
    | 4 | All Subjects | Continuous Runs | Fraction of Time | 0.17 | 0.16 | 0.15 | 0.17 | 0.19 | 0.16 |
    | 4 | All Subjects | Continuous Runs | Counts | 50.0 | 47.0 | 44.0 | 52.0 | 58.0 | 49.0 |
    | 4 | All Subjects | Continuous Runs | Persistence | 1.19 | 1.09 | 1.16 | 1.3 | 1.32 | 1.17 |
    | 5 | All Subjects | Continuous Runs | Fraction of Time | 0.18 | 0.2 | 0.19 | 0.15 | 0.14 | 0.15 |
    | 5 | All Subjects | Continuous Runs | Counts | 53.0 | 60.0 | 57.0 | 45.0 | 41.0 | 44.0 |
    | 5 | All Subjects | Continuous Runs | Persistence | 1.27 | 1.2 | 1.3 | 1.25 | 1.32 | 1.19 |
    | 6 | All Subjects | Continuous Runs | Fraction of Time | 0.15 | 0.16 | 0.18 | 0.17 | 0.16 | 0.18 |
    | 6 | All Subjects | Continuous Runs | Counts | 45.0 | 49.0 | 53.0 | 52.0 | 47.0 | 54.0 |
    | 6 | All Subjects | Continuous Runs | Persistence | 1.1 | 1.17 | 1.26 | 1.21 | 1.15 | 1.29 |
    | 7 | All Subjects | Continuous Runs | Fraction of Time | 0.21 | 0.14 | 0.17 | 0.18 | 0.15 | 0.15 |
    | 7 | All Subjects | Continuous Runs | Counts | 62.0 | 43.0 | 51.0 | 53.0 | 45.0 | 46.0 |
    | 7 | All Subjects | Continuous Runs | Persistence | 1.22 | 1.19 | 1.24 | 1.23 | 1.13 | 1.18 |
    | 8 | All Subjects | Continuous Runs | Fraction of Time | 0.18 | 0.18 | 0.13 | 0.18 | 0.16 | 0.17 |
    | 8 | All Subjects | Continuous Runs | Counts | 54.0 | 53.0 | 39.0 | 55.0 | 49.0 | 50.0 |
    | 8 | All Subjects | Continuous Runs | Persistence | 1.19 | 1.15 | 1.15 | 1.25 | 1.26 | 1.14 |
    | 9 | All Subjects | Continuous Runs | Fraction of Time | 0.14 | 0.17 | 0.17 | 0.22 | 0.18 | 0.13 |
    | 9 | All Subjects | Continuous Runs | Counts | 41.0 | 51.0 | 51.0 | 65.0 | 53.0 | 39.0 |
    | 9 | All Subjects | Continuous Runs | Persistence | 1.14 | 1.09 | 1.13 | 1.36 | 1.23 | 1.18 |
    | 10 | All Subjects | Continuous Runs | Fraction of Time | 0.13 | 0.2 | 0.16 | 0.18 | 0.15 | 0.18 |
    | 10 | All Subjects | Continuous Runs | Counts | 39.0 | 60.0 | 47.0 | 54.0 | 45.0 | 55.0 |
    | 10 | All Subjects | Continuous Runs | Persistence | 1.03 | 1.2 | 1.12 | 1.32 | 1.15 | 1.25 |

    
</details>

| Subject_ID | Group | Run | Metric | CAP-1 | CAP-2 | CAP-3 | CAP-4 | CAP-5 | CAP-6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | All Subjects | Continuous Runs | Fraction of Time | 0.16 | 0.14 | 0.18 | 0.19 | 0.16 | 0.17 |
| 1 | All Subjects | Continuous Runs | Counts | 47.0 | 43.0 | 54.0 | 57.0 | 47.0 | 52.0 |
| 1 | All Subjects | Continuous Runs | Persistence | 1.34 | 1.34 | 1.29 | 1.19 | 1.18 | 1.24 |
| 2 | All Subjects | Continuous Runs | Fraction of Time | 0.17 | 0.1 | 0.15 | 0.2 | 0.21 | 0.18 |
| 2 | All Subjects | Continuous Runs | Counts | 50.0 | 29.0 | 45.0 | 61.0 | 62.0 | 53.0 |
| 2 | All Subjects | Continuous Runs | Persistence | 1.19 | 1.32 | 1.12 | 1.24 | 1.33 | 1.2 |
| 3 | All Subjects | Continuous Runs | Fraction of Time | 0.14 | 0.16 | 0.14 | 0.18 | 0.22 | 0.16 |
| 3 | All Subjects | Continuous Runs | Counts | 42.0 | 49.0 | 41.0 | 53.0 | 66.0 | 49.0 |
| 3 | All Subjects | Continuous Runs | Persistence | 1.14 | 1.11 | 1.17 | 1.33 | 1.14 | 1.14 |
| 4 | All Subjects | Continuous Runs | Fraction of Time | 0.17 | 0.16 | 0.15 | 0.17 | 0.19 | 0.16 |
| 4 | All Subjects | Continuous Runs | Counts | 50.0 | 47.0 | 44.0 | 52.0 | 58.0 | 49.0 |
| 4 | All Subjects | Continuous Runs | Persistence | 1.19 | 1.09 | 1.16 | 1.3 | 1.32 | 1.17 |
| 5 | All Subjects | Continuous Runs | Fraction of Time | 0.18 | 0.2 | 0.19 | 0.15 | 0.14 | 0.15 |
| 5 | All Subjects | Continuous Runs | Counts | 53.0 | 60.0 | 57.0 | 45.0 | 41.0 | 44.0 |
| 5 | All Subjects | Continuous Runs | Persistence | 1.27 | 1.2 | 1.3 | 1.25 | 1.32 | 1.19 |
| 6 | All Subjects | Continuous Runs | Fraction of Time | 0.15 | 0.16 | 0.18 | 0.17 | 0.16 | 0.18 |
| 6 | All Subjects | Continuous Runs | Counts | 45.0 | 49.0 | 53.0 | 52.0 | 47.0 | 54.0 |
| 6 | All Subjects | Continuous Runs | Persistence | 1.1 | 1.17 | 1.26 | 1.21 | 1.15 | 1.29 |
| 7 | All Subjects | Continuous Runs | Fraction of Time | 0.21 | 0.14 | 0.17 | 0.18 | 0.15 | 0.15 |
| 7 | All Subjects | Continuous Runs | Counts | 62.0 | 43.0 | 51.0 | 53.0 | 45.0 | 46.0 |
| 7 | All Subjects | Continuous Runs | Persistence | 1.22 | 1.19 | 1.24 | 1.23 | 1.13 | 1.18 |
| 8 | All Subjects | Continuous Runs | Fraction of Time | 0.18 | 0.18 | 0.13 | 0.18 | 0.16 | 0.17 |
| 8 | All Subjects | Continuous Runs | Counts | 54.0 | 53.0 | 39.0 | 55.0 | 49.0 | 50.0 |
| 8 | All Subjects | Continuous Runs | Persistence | 1.19 | 1.15 | 1.15 | 1.25 | 1.26 | 1.14 |
| 9 | All Subjects | Continuous Runs | Fraction of Time | 0.14 | 0.17 | 0.17 | 0.22 | 0.18 | 0.13 |
| 9 | All Subjects | Continuous Runs | Counts | 41.0 | 51.0 | 51.0 | 65.0 | 53.0 | 39.0 |
| 9 | All Subjects | Continuous Runs | Persistence | 1.14 | 1.09 | 1.13 | 1.36 | 1.23 | 1.18 |
| 10 | All Subjects | Continuous Runs | Fraction of Time | 0.13 | 0.2 | 0.16 | 0.18 | 0.15 | 0.18 |
| 10 | All Subjects | Continuous Runs | Counts | 39.0 | 60.0 | 47.0 | 54.0 | 45.0 | 55.0 |
| 10 | All Subjects | Continuous Runs | Persistence | 1.03 | 1.2 | 1.12 | 1.32 | 1.15 | 1.25 |

# References

Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia. NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193
