---
title: 'NeuroCAPs: A Python Package for Performing Co-Activation Patterns Analyses on Resting-State and Task-Based fMRI Data'
tags:
    - Python
    - neuroimaging
    - fMRI
    - dynamic functional connectivity
    - co-activation patterns
date: 4 April 2025
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
Co-Activation Patterns (CAPs) is a dynamic functional connectivity technique that clusters similar spatial distributions
of brain activity. To make this analytical technique more accessible to neuroimaging researchers, NeuroCAPs,
an open source Python package, was developed. This package performs end-to-end CAPs analyses on preprocessed resting-state
or task-based functional magnetic resonance imaging (fMRI) data, and is most optimized for data preprocessed with
fMRIPrep, a robust preprocessing pipeline designed to minimize manual user input and enhance reproducibility
[@Esteban2019].

# Background
Numerous fMRI studies employ static functional connectivity (sFC) techniques to analyze correlative activity within and
between brain regions. However, these approaches operate under the assumption that functional connectivity patterns,
which change within seconds [@Jiang2022], remain stationary throughout the entire data acquisition period
[@Hutchison2013].

Unlike sFC approaches, dynamic functional connectivity (dFC) methods enable the analysis of dynamic functional states,
which are characterized by consistent, replicable, and distinct periods of time-varying brain connectivity patterns
[@Rabany2019]. Among these techniques, CAPs analysis aggregates similar spatial distributions of brain activity using
clustering techniques, such as the k-means algorithm, to capture the dynamic nature of brain activity [@Liu2013; @Liu2018].

# Statement of Need
Currently, performing an end-to-end CAPs analysis presents challenges due to the numerous steps required. Researchers must:

1. clean timeseries data through nuisance regression and censor frames with high framewise displacement (excessive head motion).
2. perform spatial dimensionality reduction.
3. concatenate timeseries data across multiple subjects for k-means clustering.
4. select an optimal cluster size using heuristics such as the elbow or silhouette methods.
5. implement various visualization techniques to enhance interpretability of the results.

While other excellent CAPs toolboxes exist, they are often implemented in proprietary languages such as MATLAB
(which is the case for TbCAPs [@Bolton2020]), lack comprehensive end-to-end analytical pipelines for both
resting-state and task-based fMRI data with temporal dynamic metrics and visualization capabilities (such as capcalc
[@Frederick2022]), or are comprehensive, but generalized toolboxes for evaluating and comparing different dFC
methods (such as pydFC [@Torabi2024]). NeuroCAPs addresses these limitations by providing an accessible Python package
specifically for performing end-to-end CAPs analyses, from post-processing of fMRI data to creation of temporal metrics
for downstream statistical analyses and visualizations to facilitate interpretations. However, many of NeuroCAPs'
post-processing functionalities assumes that fMRI data is organized in a Brain Imaging Data Structure (BIDS) compliant
directory and is most optimized for data preprocessed with fMRIPrep [@Esteban2019] or preprocessing pipelines
that generate similar outputs (e.g. NiBabies [@Goncalves2025]).

# Modules
NeuroCAPs consists of four modules, with core functionality primarily distributed between two main modules
(`neurocaps.extraction` and `neurocaps.analysis`) that handle the entire workflow, from post-processing to temporal
metric computation and visualization capabilities.

**neurocaps.exceptions**

This module contains custom exceptions. These include `BIDSQueryError`, which supports NeuroCAPs' integration with
PyBIDS [@Yarkoni2019] by providing guidance when issues arise with BIDS directories; `NoElbowDetectedError`, which
offers solutions when optimal cluster determination fails using the elbow method implemented via Kneed [@Arvai2023];
and ``UnsupportedFileExtensionError``, which handles cases when pickled inputs have unsupported file extensions.

**neurocaps.extraction**

This module contains the `TimeseriesExtractor` class, which:

- leverages extracts Nilearn's [@Nilearn] `NiftiLabelsMasker` to perform nuisance regression on resting-state and
task-based fMRI data and use deterministic parcellations (such as the Schaefer [@Schaefer2018], Automated Anatomical
Labeling [@Tzourio-Mazoyer2002], and Human Connectome Project extended [@Huang2022]) for spatial
dimensionality reduction.
- censors high-motion volumes using fMRIPrep-derived framewise displacement values and stores the extracted timeseries
information in a dictionary mapping subject IDs to run IDs and their associated timeseries.
- reports quality control information related to framewise displacement and dummy volumes.
- saves extracted timeseries data as a pickle file.
- visualizes timeseries data for a specific subject's run.

**neurocaps.analysis**

This module contains the `CAP` class, which:

- allows group-specific analyses or analyses on all subjects.
- performs k-means clustering (from Scikit-learn [@scikit-learn]) for CAP identification, while supporting a single
cluster size or a range of clusters in combination with various cluster selection methods to determine the optimal
cluster size.
- computes various subject-level temporal dynamics metrics for downstream statistical analyses.
- enables conversion of CAPs to NIfTI statistical maps.
- provides diverse visualization options, using Matplotlib [@Hunter:2007], Seaborn [@Waskom2021], Plotly [@plotly], and
Surfplot [@Gale2021], to facilitate scientific interpretations.

Additionally, the module provides standalone functions for:

- changing the data type [@harris2020array] and performing additional standardization of timeseries data produced by
`TimeseriesExtractor`.
- merging multiple timeseries data across different dictionaries produced by `TimeseriesExtractor` by identifying
similar subjects and concatenating their data, which facilitates analyses to identify CAPs across sessions or
tasks.
- creating averaged transition probability matrices from subject-level transition probabilities.

**neurocaps.typing**

This module provides custom type definitions compatible with static type checkers, enabling proper construction of
dictionary structures for parcellations or timeseries data when manual creation is necessary.

# Examples
Comprehensive demonstrations and tutorials for NeuroCAPs can be found on its repository at
[https://github.com/donishadsmith/neurocaps](https://github.com/donishadsmith/neurocaps)
and its documentation at [https://neurocaps.readthedocs.io/](https://neurocaps.readthedocs.io/).

# Research Utility
NeuroCAPs was originally developed (and later refined for broader use) to facilitate the analysis in @Smith2025,
which has been submitted for peer review by the same author.

# Acknowledgements
Funding provided by the Dissertation Year Fellowship (DYF) Program at Florida International University (FIU) assisted
in further refinement of NeuroCAPs.

# References
