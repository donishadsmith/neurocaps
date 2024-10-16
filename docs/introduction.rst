**neurocaps**
=============
.. image:: https://img.shields.io/pypi/v/neurocaps.svg
   :target: https://pypi.python.org/pypi/neurocaps/
   :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/neurocaps.svg
   :target: https://pypi.python.org/pypi/neurocaps/
   :alt: Python Versions

.. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.11642615-teal
   :target: https://doi.org/10.5281/zenodo.13929160
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

This is a Python package designed to perform Co-activation Patterns (CAPs) analyses. It utilizes k-means clustering to group timepoints (TRs) into brain states, applicable to both resting-state and task-based fMRI data. 
The package is compatible with data preprocessed using **fMRIPrep** and assumes your directory is BIDS-compliant, containing a derivatives folder with a pipeline folder (such as fMRIPrep) that holds the preprocessed BOLD data.

This package was initially inspired by a co-activation patterns implementation in mtorabi59's `pydfc <https://github.com/neurodatascience/dFC>`_ package.

Citing
======
::
  
  Smith, D. (2024). neurocaps. Zenodo. https://doi.org/10.5281/zenodo.13929160

Usage
=====
This package contains two main classes: ``TimeseriesExtractor`` for extracting the timeseries, and ``CAP`` for performing the CAPs analysis.

**Note:** When extracting the timeseries, this package uses either the Schaefer atlas, the Automated Anatomical Labeling (AAL) atlas, or a custom parcellation that is lateralized (where each region/network has nodes in the left and right hemispheres). 
The number of ROIs and networks for the Schaefer atlas can be adjusted with the parcel_approach parameter when initializing the ``TimeseriesExtractor`` class.

To modify it, you must use a nested dictionary, where the primary key is "Schaefer" and the sub-keys are "n_rois" and "yeo_networks". For example:

.. code-block:: python

    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 1}}

Similarly, the version of the AAL atlas can be modified using:

.. code-block:: python

    parcel_approach = {"AAL": {"version": "SPM12"}}

If using a "Custom" parcellation approach, ensure each node in your dataset includes both left (lh) and right (rh) hemisphere versions (bilateral nodes). 

Custom Key Structure:
---------------------
- "maps": Directory path containing necessary parcellation files. Ensure files are in a supported format (e.g., .nii for NifTI files). For plotting purposes, this key is not required.
- "nodes":  List of all node labels used in your study, arranged in the exact order they correspond to indices in your parcellation files. Each label should match the parcellation index it represents. For example, if the parcellation label "0" corresponds to the left hemisphere visual cortex area 1, then "LH_Vis1" should occupy the 0th index in this list. This ensures that data extraction and analysis accurately reflect the anatomical regions intended. For timeseries extraction, this key is not required.
- "regions": Dictionary defining major brain regions. Each region should list node indices under "lh" and "rh" to specify left and right hemisphere nodes. For timeseries extraction, this key is not required.
        
Example:
--------
The provided example demonstrates setting up a custom parcellation containing nodes for the visual network (Vis) and hippocampus regions:

.. code-block:: python

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

Main features for ``TimeseriesExtractor`` includes:
---------------------------------------------------

- **Timeseries Extraction:** Extract timeseries for resting-state or task data, creating a nested dictionary containing the subject ID, run number, and associated timeseries. This serves as input for the ``get_caps`` method in the ``CAP`` class.
- **Saving Timeseries:** Save the nested dictionary containing timeseries as a pickle file.
- **Visualization:** Visualize the timeseries of a Schaefer, AAL, or Custom parcellation node or region/network in a specific subject's run, with options to save the plots.
- **Parallel Processing:** Use parallel processing by specifying the number of CPU cores in the ``n_cores`` parameter in the ``get_bold`` method. Testing on an HPC using a loop with ``TimeseriesExtractor.get_bold`` to extract session 1 and 2 
  BOLD timeseries from 105 subjects from resting-state data (single run containing 360 volumes) and two task datasets (three runs containing 200 volumes each and two runs containing 200 volumes) reduced processing time from 5 hours 48 minutes to 1 hour 26 minutes 
  (using 10 cores). *Note:* If you are using an HPC, remember to allocate the appropriate amount of CPU cores with your workload manager. For instance in slurm use ``#SBATCH --cpus-per-task=10`` if you intend to use 10 cores.

Main features for ``CAP`` includes:
-----------------------------------

- **Optimal Cluster Size Identification:** Perform the Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions to identify the optimal cluster size, saving the optimal model as an attribute.
- **Parallel Processing:** Use parallel processing, when using the Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions by specifying the number of CPU cores in the ``n_cores`` parameter in the ``get_caps`` method. 
  *Note:* If you are using an HPC, remember to allocate the appropriate amount of CPU cores with your workload manager. For instance in slurm use ``#SBATCH --cpus-per-task=10`` if you intend to use 10 cores.
- **Grouping:** Perform CAPs analysis for entire sample or groups of subject IDs (using the ``groups`` parameter when initializing the ``CAP`` class). K-means clustering, all cluster selection methods (Davies Bouldin, Silhouette, Elbow, or Variance Ratio criterions), and plotting are done for each group when specified.
- **CAP Visualization:** Visualize the CAPs as outer products or heatmaps, with options to use subplots to reduce the number of individual plots, as well as save. 
  Refer to the `documentation <https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2plot>`_ for the ``caps2plot`` method in the ``CAP`` class for available ``**kwargs`` arguments and parameters to modify plots.
- **Save CAPs as NifTIs:** Convert the atlas used for parcellation to a stat map and saves them (``caps2niftis``). 
- **Surface Plot Visualization:** Convert the atlas used for parcellation to a stat map projected onto a surface plot with options to customize and save plots. 
  Refer to the `documentation <https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2surf>`_ for the ``caps2surf`` method in the ``CAP`` class for available ``**kwargs`` arguments and parameters to modify plots. 
  Also includes the option to save the NifTIs. There is also another a parameter in ``caps2surf``, ``fslr_giftis_dict``, which can be used if the CAPs NifTI files were converted to GifTI files using a tool such as Connectome Workbench, which may work better for 
  converting your atlas to fslr space. This parameter allows plotting without re-running the analysis and only initializing the ``CAP`` class and using the ``caps2surf`` method is needed.
- **Correlation Matrix Creation:** Create a correlation matrix from CAPs with options to customize and save plots. Additionally can produce dataframes where each element contains its associated uncorrected p-value in parentheses that is accompanied by an asterisk using the following significance
  code ``{"<0.05": "*", "<0.01": "**", "<0.001": "***"}``. Refer to the `documentation <https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2corr>`_
  for the ``caps2corr`` method in the ``CAP`` class for available ``**kwargs`` arguments and parameters to modify plots.
- **CAP Metrics Calculation:** Calculate CAP metrics (``calculate_metrics``) as described in `Liu et al., 2018 <https://doi.org/10.1016/j.neuroimage.2018.01.041>`_ [1]_ and `Yang et al., 2021 <https://doi.org/10.1016/j.neuroimage.2021.118193>`_ [2]_:
    - *Temporal Fraction:* The proportion of total volumes spent in a single CAP over all volumes in a run.
      ::

          predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
          target = 1
          temporal_fraction = 4/6

    - *Persistence:* The average time spent in a single CAP before transitioning to another CAP (average consecutive/uninterrupted time).
      ::

          predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
          target = 1
          # Sequences for 1 are [1] and [1,1,1]
          persistence = (1 + 3)/2 # Average number of frames
          tr = 2
          if tr:
              persistence = ((1 + 3) * 2)/2 # Turns average frames into average time

    - *Counts:* The total number of initiations of a specific CAP across an entire run. An initiation is
      defined as the first occurrence of a CAP. If the same CAP is maintained in contiguous segment
      (indicating stability), it is still counted as a single initiation. 
      ::

          predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
          target = 1
          # Initiations of CAP-1 occur at indices 0 and 2
          counts = 2

    - *Transition Frequency:* The number of transitions between different CAPs across the entire run.
      ::

          predicted_subject_timeseries = [1, 2, 1, 1, 1, 3]
          # Transitions between unique CAPs occur at indices 0 -> 1, 1 -> 2, and 4 -> 5
          transition_frequency = 3

    - *Transition Probability* : The probability of transitioning from one CAP to another CAP (or the same CAP). This is calculated as (Number of transitions from A to B)/ (Total transitions from A). Note that the transition probability from CAP-A -> CAP-B is not the same as CAP-B -> CAP-A.
      ::

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

- **Cosine Similarity Radar Plots:** Create radar plots showing the cosine similarity between positive and negative activations of each CAP and each a-priori regions in a parcellation [3]_ [4]_. Refer to the `documentation <https://neurocaps.readthedocs.io/en/latest/generated/neurocaps.analysis.CAP.html#neurocaps.analysis.CAP.caps2radar>`_ in ``caps2radar`` in the ``CAP`` class for a more detailed explanation as well as available ``**kwargs`` arguments and parameters to modify plots.
  ::

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

**Additionally, the neurocaps.analysis submodule contains two additional functions:**

- ``merge_dicts``: Merge the subject_timeseries dictionaries for overlapping subjects across tasks to identify similar CAPs across different tasks. The merged dictionary can be saved as a pickle file.
- ``standardize``: Standardizes each run independently for all subjects in the subject timeseries.
- ``change_dtype``: Changes the dtype of all subjects in the subject timeseries to help with memory usage.
- ``transition_matrix``: Uses the "transition_probability" output from ``CAP.calculate_metrics`` to generate and visualize the averaged transition probability matrix for all groups from the analysis.

Please refer to `demo.ipynb <https://github.com/donishadsmith/neurocaps/blob/main/demo.ipynb>`_ for a more extensive demonstration of the features included in this package.

Dependencies
============

neurocaps relies on several packages:

:: 

    dependencies = ["numpy>=1.22.0, <2.0.0",
                    "pandas>=2.0.0",
                    "joblib>=1.3.0",
                    "matplotlib>=3.6.0",
                    "seaborn>=0.11.0",
                    "kneed>=0.8.0",
                    "nibabel>=3.2.0",
                    "nilearn>=0.10.1, !=0.10.3",
                    "scikit-learn>=1.4.0",
                    "scipy>=1.6.0",
                    "surfplot>=0.2.0",
                    "neuromaps>=0.0.5",
                    "pybids>=0.16.2; platform_system != 'Windows'",
                    "plotly>=4.9",
                    "nbformat>=4.2.0", # For plotly
                    "kaleido==0.1.0.post1; platform_system == 'Windows'", # Plotly saving seems to work best with this version for Windows
                    "kaleido; platform_system != 'Windows'",
                    "setuptools; python_version>='3.12'"
                   ]

References
==========

.. [1] Liu, X., Zhang, N., Chang, C., & Duyn, J. H. (2018). Co-activation patterns in resting-state fMRI signals. NeuroImage, 180, 485–494. https://doi.org/10.1016/j.neuroimage.2018.01.041

.. [2] Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation patterns of functional brain networks reveal the aberrant dynamic state transition in schizophrenia. NeuroImage, 237, 118193. https://doi.org/10.1016/j.neuroimage.2021.118193

.. [3] Zhang, R., Yan, W., Manza, P., Shokri-Kojori, E., Demiral, S. B., Schwandt, M., Vines, L., Sotelo, D., Tomasi, D., Giddens, N. T., Wang, G., Diazgranados, N., Momenan, R., & Volkow, N. D. (2023). 
       Disrupted brain state dynamics in opioid and alcohol use disorder: attenuation by nicotine use. Neuropsychopharmacology, 49(5), 876–884. https://doi.org/10.1038/s41386-023-01750-w

.. [4] Ingwersen, T., Mayer, C., Petersen, M., Frey, B. M., Fiehler, J., Hanning, U., Kühn, S., Gallinat, J., Twerenbold, R., Gerloff, C., Cheng, B., Thomalla, G., & Schlemm, E. (2024).
       Functional MRI brain state occupancy in the presence of cerebral small vessel disease — A pre-registered replication analysis of the Hamburg City Health Study. Imaging Neuroscience, 2, 1–17. https://doi.org/10.1162/imag_a_00122
  
