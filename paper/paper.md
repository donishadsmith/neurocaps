---
title: 'NeuroCAPs: A Python Package for Performing Co-Activation Patterns Analyses on Resting-State
        and Task-Based fMRI Data'
tags:
    - Python
    - neuroimaging
    - fMRI
    - dynamic functional connectivity
    - co-activation patterns
date: 7 July 2025
output: pdf_document
authors:
    - name: Donisha Smith
      orchid: 0000-0001-7019-3520
      affiliation: 1

affiliations:
  - index: 1
    name: Department of Psychology, Florida International University

bibliography: paper.bib
---

# Summary
Co-Activation Patterns (CAPs) is a dynamic functional connectivity technique that clusters similar
spatial distributions of brain activity. To make this analytical technique more accessible to
neuroimaging researchers, NeuroCAPs, an open source Python package, was developed. This package
performs end-to-end CAPs analyses on preprocessed resting-state or task-based functional magnetic
resonance imaging (fMRI) data, and is most optimized for data preprocessed with fMRIPrep, a robust
preprocessing pipeline designed to minimize manual user input and enhance reproducibility
[@Esteban2019].

# Background
Numerous fMRI studies employ static functional connectivity (sFC) techniques to analyze correlative
activity within and between brain regions. However, these approaches operate under the assumption
that functional connectivity patterns, which change within seconds [@Jiang2022], remain stationary
throughout the entire data acquisition period [@Hutchison2013].

Unlike sFC approaches, dynamic functional connectivity (dFC) methods enable the analysis of dynamic
functional states, which are characterized by consistent, replicable, and distinct periods of
time-varying brain connectivity patterns [@Rabany2019]. Among these techniques, CAPs analysis
aggregates similar spatial distributions of brain activity using clustering techniques, typically
the k-means algorithm, to capture the dynamic nature of brain activity [@Liu2013; @Liu2018].

# Statement of Need
The typical CAPs workflow can be programmatically time-consuming to manually orchestrate as it
generally entails several steps:

1. implement spatial dimensionality reduction of timeseries data
2. perform nuisance regression and scrub high-motion volumes (excessive head motion)
3. concatenate the timeseries data from multiple subjects into a single matrix
4. apply k-means clustering to the concatenated data and select the optimal number of
   clusters (CAPs) using heuristics such as the elbow or silhouette methods
5. generate different visualizations to enhance the interpretability of the CAP

While other excellent CAPs toolboxes exist, they are often implemented in proprietary languages such
as MATLAB (which is the case for TbCAPs [@Bolton2020]), lack comprehensive end-to-end analytical
pipelines for both resting-state and task-based fMRI data with temporal dynamic metrics and
visualization capabilities (such as capcalc [@Frederick2022]), or are comprehensive, but generalized
toolboxes for evaluating and comparing different dFC methods (such as pydFC [@Torabi2024]).

NeuroCAPs addresses these limitations by providing an accessible Python package specifically
for performing end-to-end CAPs analyses, from post-processing of fMRI data to creation of temporal
metrics for downstream statistical analyses and visualizations to facilitate interpretations.
However, many of NeuroCAPs' post-processing functionalities assumes that fMRI data is organized in
a Brain Imaging Data Structure (BIDS) compliant directory and is most optimized for data
preprocessed with fMRIPrep [@Esteban2019] or preprocessing pipelines that generate similar
outputs (e.g. NiBabies [@Goncalves2025]). Furthermore, NeuroCAPs only supports the k-means
algorithm for clustering, which is the clustering algorithm typically employed when performing
the CAPs analysis [@Liu2013].

# Modules
The core functionalities of NeuroCAPs are concentrated in three modules:

1. `neurocaps.extraction`
Contains the `TimeseriesExtractor` class, which:

- collects preproccessed BOLD data from an BIDS-compliant dataset [@Yarkoni2019]
- leverages Nilearn's [@Nilearn] `NiftiLabelsMasker` to perform nuisance regression and spatial
dimensionality reduction using deterministic parcellations (e.g., Schaefer [@Schaefer2018],
AAL [@Tzourio-Mazoyer2002]).
- scrubs high-motion volumes using fMRIPrep-derived framewise displacement values.
- reports quality control information related to high-motion or non-steady state volumes.

2. `neurocaps.analysis`
Contains the CAP class for performing the main analysis, as well as several standalone
utility functions.

- The `CAP` class:
  - performs k-means clustering [@scikit-learn] to identify CAPs, supporting both single and
    optimized cluster selection with heuristics such as the silhouette and elbow method [Arvai2023].
  - computes subject-level temporal dynamics metrics (e.g., fractional occupancy, transition
    probabilities) for statistical analysis.
  - converts identified CAPs back into NIfTI statistical maps for spatial interpretation.
  - integrates multiple plotting libraries [@Hunter:2007; @Waskom2021; @plotly; @Gale2021] to
    provide a diverse range of visualization options.

- Standalone functions:
Provide tools for data standardization [@harris2020array], merging timeseries data across sessions
or tasks, and creating group-averaged transition matrices.

3. `neurocaps.utils`

Contains a utility function, `generate_custom_parcel_approach`, which automatically creates
the necessary data structures from a parcellation's metadata file.

# Workflow Example
The following code demonstrates a simple workflow example using NeuroCAPs to perform the CAPs
analysis.

1. Extract timeseries data
```python
from neurocaps.extraction import TimeseriesExtractor

# Using Schaefer, one of the default parcellation approaches
parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}

# List of fMRIPrep-derived confounds for nuisance regression
confound_names = [
    "cosine*",
    "trans_x",
    "trans_x_derivative1",
    "trans_y",
    "trans_y_derivative1",
    "trans_z",
    "trans_z_derivative1",
    "rot_x",
    "rot_x_derivative1",
    "rot_y",
    "rot_y_derivative1",
    "rot_z",
    "rot_z_derivative1",
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
        "threshold": 0.50,
        "outlier_percentage": 0.30,
    },
)

# Extract BOLD data from preprocessed fMRIPrep data
# which should be located in the "derivatives" folder
# within the BIDS root directory
# The extracted timeseries data is automatically stored
extractor.get_bold(
    bids_dir="path/to/bids/root",
    pipeline_name="fmriprep",
    session="1",
    task="rest",
    tr=2,
    verbose=False,
)

# Retrieve the dataframe containing QC information for each subject
# to use for downstream statistical analyses
qc_df = extractor.report_qc()
print(qc_df)
```

2. Use k-means clustering to identify the optimal number of CAPs from the data using a heuristic
```python
from neurocaps.analysis import CAP

# Initialize CAP class
cap_analysis = CAP(parcel_approach=extractor.parcel_approach)

# Identify the optimal number of CAPs (clusters)
# using the elbow method to test 2-20
# The optimal number of CAPs is automatically stored
cap_analysis.get_caps(
    subject_timeseries=extractor.subject_timeseries,
    n_clusters=range(2, 21),
    standardize=True,
    cluster_selection_method="elbow",
    max_iter=500,
    n_init=10,
    random_state=0,
)
```

3. Compute temporal dynamic metrics for downstream statistical analyses
```python
# Calculate temporal fraction and persistence of each CAP for all subjects
output = cap_analysis.calculate_metrics(
    extractor.subject_timeseries, metrics=["temporal_fraction", "persistence"]
)
print(output["temporal_fraction"])
```

4. Visualize CAPs
```python
# Project CAPs onto surface plots
# and generate cosine similarity network alignment of CAPs
cap_analysis.caps2surf().caps2radar()
```

# Documentation
Comprehensive documentations and interactive tutorials of NeuroCAPS can be found at
[https://neurocaps.readthedocs.io/](https://neurocaps.readthedocs.io/) and on its
[repository](https://github.com/donishadsmith/neurocaps).

# Research Utility
NeuroCAPs was originally developed (and later expanded and refined for broader use) to facilitate
the analysis in @Smith2025, which has been submitted for peer review by the same author.

# Acknowledgements
Funding provided by the Dissertation Year Fellowship (DYF) Program at Florida International
University (FIU) assisted in further refinement and expansion of NeuroCAPs.

# References
