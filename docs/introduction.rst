**neurocaps**
=============
.. image:: https://img.shields.io/pypi/v/neurocaps.svg
   :target: https://pypi.python.org/pypi/neurocaps/
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/neurocaps.svg
   :target: https://pypi.python.org/pypi/neurocaps/
   :alt: Python Versions

.. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.11642615-teal
   :target: https://doi.org/10.5281/zenodo.14286989
   :alt: DOI

.. image:: https://img.shields.io/badge/Source%20Code-neurocaps-purple
   :target: https://github.com/donishadsmith/neurocaps
   :alt: GitHub Repository

.. image:: https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml/badge.svg
   :target: https://github.com/donishadsmith/neurocaps/actions/workflows/testing.yaml
   :alt: Test Status

.. image:: https://codecov.io/github/donishadsmith/neurocaps/graph/badge.svg?token=WS2V7I16WF
   :target: https://codecov.io/github/donishadsmith/neurocaps
   :alt: codecov

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/badge/OS-Ubuntu%20|%20macOS%20|%20Windows-blue
  :alt: Platform Support



neurocaps is a Python package for performing Co-activation Patterns (CAPs) analyses on resting-state or task-based fMRI
data (resting-state & task-based). CAPs identifies recurring brain states through k-means clustering of BOLD timeseries
data [1]_.

**neurocaps is most optimized for fMRI data preprocessed with fMRIPrep and assumes a BIDs compliant directory
such as the example directory structures below:**

Basic BIDS directory:

::

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

BIDS directory with session-level organization:

::

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

*Note: Only the preprocessed BOLD file is required. Additional files such as the confounds tsv (needed for denoising),
mask, and task timing tsv file (needed for filtering a specific task condition) depend on the specific analyses.
The "dataset_description.json" is required in both the bids root and pipeline directories for querying with pybids*

Citing
------
::

  Smith, D. (2024). neurocaps. Zenodo. https://doi.org/10.5281/zenodo.14286989

Usage
-----
This package contains two main classes: ``TimeseriesExtractor`` for extracting the timeseries, and ``CAP`` for performing the CAPs analysis.

Main features for ``TimeseriesExtractor`` includes:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Timeseries Extraction:** Extract timeseries for resting-state or task data using Schaefer, AAL, or a lateralized Custom parcellation for spatial dimensionality reduction (``self.get_bold``).
- **Parallel Processing:** Use parallel processing to speed up timeseries extraction.
- **Saving Timeseries:** Save the nested dictionary containing timeseries as a pickle file (``self.timeseries_to_pickle``).
- **Visualization:** Visualize the timeseries at the region or node level of the parcellation (``self.visualize_bold``).

Main features for ``CAP`` includes:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- **Grouping:** Perform CAPs analysis for entire sample or groups of subject IDs
- **Optimal Cluster Size Identification:** Perform the Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions to identify the optimal cluster size and automatically save the optimal model as an attribute (``self.get_caps``).
- **Parallel Processing:** Use parallel processing to speed up optimal cluster size identification.
- **CAP Visualization:** Visualize the CAPs as outer products or heatmaps at either the region or node level of the parcellation (``self.caps2plot``).
- **Save CAPs as NifTIs:** Convert the atlas used for parcellation to a statistical NifTI image (``self.caps2niftis``).
- **Surface Plot Visualization:** Project CAPs onto a surface plot (``self.caps2surf``).
- **Correlation Matrix Creation:** Create a correlation matrix from CAPs (``self.caps2corr``).
- **CAP Metrics Calculation:** Calculate several CAP metrics as described in `Liu et al., 2018 <https://doi.org/10.1016/j.neuroimage.2018.01.041>`_ [1]_ and `Yang et al., 2021 <https://doi.org/10.1016/j.neuroimage.2021.118193>`_ [2]_ (``self.calculate_metrics``):
    - *Temporal Fraction:* The proportion of total volumes spent in a single CAP over all volumes in a run.
    - *Persistence:* The average time spent in a single CAP before transitioning to another CAP (average consecutive/uninterrupted time).
    - *Counts:* The total number of initiations of a specific CAP across an entire run. An initiation is
      defined as the first occurrence of a CAP.
    - *Transition Frequency:* The number of transitions between different CAPs across the entire run.
    - *Transition Probability* : The probability of transitioning from one CAP to another CAP (or the same CAP). This is calculated as (Number of transitions from A to B)/ (Total transitions from A).
- **Cosine Similarity Radar Plots:** Create radar plots showing the cosine similarity between positive and negative activations of each CAP and each a-priori regions in a parcellation [3]_ [4]_ (``self.caps2radar``).

**Additionally, the neurocaps.analysis submodule contains additional functions:**

- ``merge_dicts``: Merge the subject_timeseries dictionaries for overlapping subjects across tasks to identify similar CAPs across different tasks [5]_. The merged dictionary can be saved as a pickle file.
- ``standardize``: Standardizes each run independently for all subjects in the subject timeseries.
- ``change_dtype``: Changes the dtype of all subjects in the subject timeseries to help with memory usage.
- ``transition_matrix``: Uses the "transition_probability" output from ``CAP.calculate_metrics`` to generate and visualize the averaged transition probability matrix for all groups from the analysis.

Please refer to the `demos <https://github.com/donishadsmith/neurocaps/tree/main/demos>`_ or `tutorials <https://neurocaps.readthedocs.io/en/latest/examples/examples.html>`_ for a more extensive demonstration of the features included in this package.

Dependencies
------------
Neurocaps relies on several packages:

::

   dependencies = ["numpy>=1.22.0",
                   "pandas>=2.0.0",
                   "joblib>=1.3.0",
                   "matplotlib>=3.6.0",
                   "seaborn>=0.11.0",
                   "kneed>=0.8.0",
                   "nibabel>=3.2.0",
                   "nilearn>=0.10.1, !=0.10.3",
                   "scikit-learn>=1.4.0",
                   "scipy>=1.6.0",
                   "brainspace>=0.1.16",
                   "surfplot>=0.2.0",
                   "neuromaps>=0.0.5",
                   "pybids>=0.16.2; platform_system != 'Windows'",
                   "plotly>=4.9",
                   "nbformat>=4.2.0",
                   "kaleido==0.1.0.post1; platform_system == 'Windows'",
                   "kaleido; platform_system != 'Windows'",
                   "setuptools; python_version>='3.12'",
                   "vtk<9.4.0"
                  ]

Acknowledgements
----------------
Some foundational concepts in neurocaps take inspiration from features or design patterns implemented in other
neuroimaging Python packages, specifically:

- mtorabi59's `pydfc <https://github.com/neurodatascience/dFC>`_, a toolbox that allows comparisons among several popular dynamic functionality methods.
- 62442katieb's `idconn <https://github.com/62442katieb/IDConn>`_, a pipeline for assessing individual differences in resting-state or task-based functional connectivity.

References
----------
.. [1] Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals. NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

.. [2] Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia. NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

.. [3] Zhang, R., Yan, W., Manza, P., Shokri-Kojori, E., Demiral, S. B., Schwandt, M., Vines, L., Sotelo, D., Tomasi, D., Giddens, N. T., Wang, G., Diazgranados, N., Momenan, R., & Volkow, N. D. (2023).
       Disrupted brain state dynamics in opioid and alcohol use disorder: attenuation by nicotine use. Neuropsychopharmacology, 49(5), 876–884. https://doi.org/10.1038/s41386-023-01750-w

.. [4] Ingwersen, T., Mayer, C., Petersen, M., Frey, B. M., Fiehler, J., Hanning, U., Kühn, S., Gallinat, J., Twerenbold, R., Gerloff, C., Cheng, B., Thomalla, G., & Schlemm, E. (2024).
       Functional MRI brain state occupancy in the presence of cerebral small vessel disease — A pre-registered replication analysis of the Hamburg City Health Study. Imaging Neuroscience, 2, 1–17. https://doi.org/10.1162/imag_a_00122

.. [5] Kupis, L., Romero, C., Dirks, B., Hoang, S., Parladé, M. V., Beaumont, A. L., Cardona, S. M., Alessandri, M., Chang, C., Nomi, J. S., & Uddin, L. Q. (2020). Evoked and intrinsic brain network dynamics in children with autism spectrum disorder. NeuroImage: Clinical, 28, 102396. https://doi.org/10.1016/j.nicl.2020.102396
