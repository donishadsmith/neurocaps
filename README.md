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

Please refer to [demo.ipynb](https://github.com/donishadsmith/neurocaps/blob/main/demo.ipynb) to see multiple examples of how to use this package.

Quick code example

```python

from neurocaps import TimeseriesExtractor, CAP

# If use_confounds is True but no confound_names provided, there are hardcoded confound names that will extract the data from the confound files outputted by fMRIPrep

extractor = TimeseriesExtractor(n_rois=100, standardize=False, use_confounds=True)

bids_dir = "/path/to/bids/dir"

# If there are multiple pipelines in the derivatives folder

pipeline_name = "fmriprep-1.4.0"

# Resting State
# extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

# Task
extractor.get_bold(bids_dir=bids_dir, task="emo", condition="positive", pipeline_name=pipeline_name)

cap_analysis = CAP(node_labels=extractor.atlas_labels, n_clusters=6)

cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries)

cap_analysis.visualize_caps(visual_scope="networks", plot_options="outer_product", task_title="- Positive Valence", ncol=3, sharey=True, subplots=True)


```

![image](https://github.com/donishadsmith/neurocaps/assets/112973674/4699bbd9-1f55-462b-9d9e-4ef17da79ad4)


