---
title: 'NeuroCAPs: A Python Package for Performing Co-Activation Patterns Analyses on Resting-State and Task-Based fMRI Data'
tags:
    - Python
    - neuroimaging
date: "19 March 2025"
output: pdf_document
authors:
    - name: Donisha Smith
      orchid: 0000-0001-7019-3520
      affiliation: "1"

affiliations:
  - index: 1
    name: Department of Psychology, Florida International University

bibliography: paper.bib
---

# Summary
Co-Activation Patterns (CAPs) is a dynamic functional connectivity technique that clusters similar spatial distributions
of brain activity. To make this analytical technique more accessible to neuroimaging researchers, NeuroCAPs [@zenodo],
an open source Python package, was developed to perform end-to-end co-activation patterns (CAPs) analyses on either
resting-state or task-based fMRI data. Additionally, it is published under the MIT license.

# Background
Numerous functional magnetic resonance imaging (fMRI) studies employ static functional connectivity (sFC) techniques to
analyze connectivity within and between brain regions. These approaches operate under the assumption that functional
connectivity patterns, defined as correlation in activity between brain regions of interest, remain stationary throughout
the entire data acquisition period [@Hutchison2013]. However, substantial evidence suggests that functional connectivity
can change within seconds [@Jiang2022].

Unlike sFC approaches, dynamic functional connectivity (dFC) methods account for temporal variability within and between
brain regions. These dFC methods enable analysis of dynamic functional states, characterized by consistent, replicable,
and distinct periods of varying connectivity patterns [@Rabany2019]. Among these techniques, co-activation
patterns (CAPs) analysis aggregates similar spatial distributions of brain activity using clustering techniques,
typically k-means. This process not only clusters these patterns but also creates an average representation of brain
states assigned to the same cluster (i.e., CAP), thereby capturing the dynamic nature of brain activity during the
scanning session [@Liu2018]. Similar CAPs can be identified across both resting-state data (representing the brain's
fundamental architecture that spontaneously fluctuates without external stimuli) and task data (representing functional
coupling between brain regions in response to a specific stimuli or task) [@Kupis2020].

# Statement of Need
Currently, performing an end-to-end CAPs analysis presents challenges due to the numerous steps required. Researchers must:

1. clean timeseries data through nuisance regression and censor frames with high framewise displacement (excessive head motion).
2. perform spatial dimensionality reduction.
3. concatenate timeseries data across multiple subjects for k-means clustering.
4. select an optimal cluster size using heuristics such as the elbow or silhouette methods.
5. implement various visualization techniques to enhance interpretability of the results.

While other excellent CAPs toolboxes exist, they are often implemented in proprietary languages such as `MATLAB`
(which is the case for `TbCAPs` [@Bolton2020]), lack comprehensive end-to-end analytical pipelines for both
resting-state and task-based fMRI data with temporal dynamic metrics and visualization capabilities (such as `capcalc`
[@Frederick2022]), or are generalized toolboxes for assessing similarity between different dFC
methods (such as `PydFC` [@Torabi2024]). NeuroCAPs addresses these limitations by providing an accessible Python package
that leverages object-oriented programming design to create a flexible pipeline-like architecture specifically for
performing CAPs analyses. Furthermore, NeuroCAPs offers the ability to directly save certain outputs, such as the
extracted timeseries, and supports continuing work across sessions through pickle serialization of Python objects,
allowing an analysis to be completed incrementally. The package assumes fMRI data is organized in a BIDS-compliant
directory and is optimized for data preprocessed with `fMRIPrep` [@Esteban2019], a preprocessing workflow designed to
minimize manual user input and enhance reproducibility. However, preprocessed fMRI data organized in similar directory
formats as `fMRIPrep` outputs can still leverage NeuroCAPs' timeseries extraction functionalities. In addition, the
necessary data structure (a dictionary mapping subject IDs to run IDs and its associated timeseries data) can be manually
created to leverage many of NeuroCAPs' analytical capabilities.

# Modules
The package consists of four modules, with core functionality primarily distributed between two main modules
(`neurocaps.extraction` and `neurocaps.analysis`) that handle the entire workflow from postprocessing to
visualization, significantly streamlining the CAPs analysis process.

**neurocaps.exceptions**

This module contains custom exceptions, one of which is `BIDSQueryError`. Since NeuroCAPs utilizes PyBIDS
[@Yarkoni2019], a Python package for querying BIDS-compliant directories, this exception was created to guide users and
provide potential fixes when no subject IDs are detected in the specified BIDS directories. The other exception,
`NoElbowDetected`, was created to provide potential solutions in the event that the elbow method (implemented by
``KneeLocator`` from the Kneed package [@Arvai_2023]) could not identify the optimal cluster size for k-means.

**neurocaps.extraction**

This module contains the `TimeseriesExtractor` class, which:

- extracts both resting-state and task-based functional MRI data using lateralized brain parcellations
(such as the Schaefer [@Schaefer2018], Automated Anatomical Labeling [@Tzourio-Mazoyer2002], and Human Connectome
Project extended [@Huang2022] parcellations) for spatial dimensionality reduction.
- leverages Nilearn's [@Nilearn] `NiftiLabelsMasker` to perform nuisance regression, censors high-motion
volumes using fMRIPrep-derived regressors, and stores the extracted timeseries information in a dictionary mapping
subject IDs to run IDs and their associated timeseries.
- saves extracted timeseries data in a serialized pickle format.
- reports the number of censored frames, interpolated frames, and mean and standard deviation of continuous high motion segments for each subject.
- visualizes timeseries data for a specific subject's run.

**neurocaps.analysis**

This module contains the `CAP` class, which:

- allows group-specific analyses or analyses on all subjects (the default configuration).
- performs k-means clustering (from Scikit-learn [@scikit-learn]) for CAP identification, while supporting a single
cluster size or a range of clusters in combination with various cluster selection methods to determine the optimal
cluster size, including:
  - the elbow method (from Kneed package [@Arvai_2023])
  - the silhouette score, davies-bouldin index, and variance ratio methods (all from Scikit-learn [@scikit-learn])
- computes various temporal dynamics metrics (including counts/state initiation, temporal fraction, persistence,
transition frequency, and transition probability) at the subject-level and exports data to a CSV file for downstream
statistical analyses.
- enables conversion of CAPs to NIfTI statistical maps.
- provides multiple visualization options for CAPs including heatmaps, outer products, correlation matrices, and cosine
similarity radar plots (showing network correspondence to both positive and negative CAP activations).

Additionally, the module provides standalone functions for:

- changing the data type and performing additional standardization of timeseries data produced by `TimeseriesExtractor`.
- merging multiple timeseries data across different dictionaries produced by `TimeseriesExtractor` by identifying
similar subjects and concatenating their data, which facilitates analyses to identify CAPs across sessions or different
tasks.
- creating averaged transition probability matrices from subject-level transition probabilities.

**neurocaps.typing**

This module contains custom type definitions that can be used with static type checkers to build appropriate dictionary
structures for specifying parcellations or the timeseries data if these structures must be manually created.

# Documentation
Comprehensive documentation for NeuroCAPs, related to installation, usage, and the application programming interface
(API) can be found at neurocaps.readthedocs.io [@documentation], with Jupyter Notebook
demonstrations available on its Github repository [@repo].

# Example Application
NeuroCAPs was originally developed (and later refined for broader use) to facilitate the analysis in @Smith2025. In
this manuscript that was submitted for publication, NeuroCAPs was used to extract timeseries data, cluster and
identify CAPs using the elbow method, and produce visualizations for CAPs (i.e., heatmap, surface plots, correlation
matrix, and cosine similarity plots)

# Acknowledgements
Funding provided by the Dissertation Year Fellowship (DYF) Program at Florida International University (FIU),
assisted in further refinement of the NeuroCAPs package.

# References
