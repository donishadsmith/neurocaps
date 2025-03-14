Tutorial 1: Using ``TimeseriesExtractor``
=========================================
This module is designed to perform timeseries extraction, nuisance regression, and visualization. Additionally, it
generates the necessary dictionary structure required for ``CAP``. If the BOLD images have not been preprocessed using
fMRIPrep (or a similar pipeline), the dictionary structure can be manually created.

The output in the `Extracting Timeseries` section is generated from a test run using GitHub Actions. This test uses
a truncated version of the open dataset provided by `Laumann & Poldrack <https://openfmri.org/dataset/ds000031/>`_ [1]_
and was obtained from the OpenfMRI database, accession number ds000031.

Extracting Timeseries
---------------------
Note: when an asterisk (*) follows a name, all confounds that start with the preceding term will be automatically included.
For example, placing an asterisk after cosine (cosine*) will utilize all parameters that begin with cosine.

.. code-block:: python

    from neurocaps.extraction import TimeseriesExtractor

    dir = os.path.dirname(__file__)

    confounds = ["cosine*", "a_comp_cor*", "rot*"]

    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 2}}

    extractor = TimeseriesExtractor(
        space="MNI152NLin2009cAsym",
        parcel_approach=parcel_approach,
        standardize="zscore_sample",
        use_confounds=True,
        detrend=True,
        low_pass=0.15,
        high_pass=None,
        confound_names=confounds,
    )

    bids_dir = os.path.join(dir, "tests", "data", "dset")

    extractor.get_bold(
        bids_dir=bids_dir,
        session="002",
        task="rest",
        pipeline_name="fmriprep_1.0.0/fmriprep",
        tr=1.2,
        progress_bar=True,  # Parameter available in versions >= 0.21.5
    )

``print`` can be used to return a string representation of the ``TimeseriesExtractor`` class.

.. code-block:: python

    print(extractor)

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        Metadata:
        ===========================================================
        Preprocessed Bold Template Space                           : MNI152NLin2009cAsym
        Parcellation Approach                                      : Schaefer
        Signal Clean Information                                   : {'masker_init': {'standardize': 'zscore_sample', 'detrend': True, 'low_pass': 0.15, 'high_pass': None, 'smoothing_fwhm': None}, 'use_confounds': True, 'confound_names': ['cosine*', 'a_comp_cor*', 'rot*'],
        'n_acompcor_separate': None, 'dummy_scans': None, 'fd_threshold': None, 'dtype': None}
        Task Information                                           : {'task': 'rest', 'session': '002', 'runs': None, 'condition': None, 'condition_tr_shift': 0, 'tr': 1.2, 'slice_time_ref': 0.0}
        Number of Subjects                                         : 1
        CPU Cores Used for Timeseries Extraction (Multiprocessing) : None
        Subject Timeseries Byte Size                               : 16184 bytes

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        2025-03-09 08:24:10,432 neurocaps._utils.extraction.check_confound_names [INFO] Confound regressors to be used if available: cosine*, a_comp_cor*, rot*.
        2025-03-09 08:24:10,470 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 01 | SESSION: 002 | TASK: rest | RUN: 001] Preparing for Timeseries Extraction using [FILE: sub-01_ses-002_task-rest_run-001_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz].
        2025-03-09 08:24:10,482 neurocaps._utils.extraction.extract_timeseries [INFO] [SUBJECT: 01 | SESSION: 002 | TASK: rest | RUN: 001] The following confounds will be used for nuisance regression: cosine_00, cosine_01, cosine_02, cosine_03, cosine_04, cosine_05, cosine_06, a_comp_cor_00, a_comp_cor_01, a_comp_cor_02, a_comp_cor_03, a_comp_cor_04, a_comp_cor_05, rot_x, rot_y, rot_z.
        Processing Subjects: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.73s/it]

The extracted timeseries is stored as a nested dictionary and can be accessed using the ``subject_timeseries``
property. The ``TimeseriesExtractor`` class has several
`properties <https://neurocaps.readthedocs.io/en/stable/generated/neurocaps.extraction.TimeseriesExtractor.html#properties>`_.
**Some properties can also be used as setters.**

.. code-block:: python

    print(extraction.subject_timeseries)

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        {'01': {'run-001': array([[-0.12410211, -0.746016  , -0.9138416 , ...,  0.12293668,
        -0.3167036 , -0.4593077 ],
       [-1.0730965 ,  0.88747275,  0.83726895, ..., -0.9314818 ,
         0.5686499 ,  0.9783575 ],
       [-0.5288149 ,  0.62266237,  0.6349383 , ..., -0.5331197 ,
         0.5261529 ,  0.5858582 ],
       ...,
       [ 0.32443312, -0.42479128, -0.43596116, ...,  0.5425763 ,
        -0.2863486 , -0.31798226],
       [ 0.94420713, -0.7662241 , -0.6925075 , ...,  1.7636685 ,
        -0.4194046 , -0.5691561 ],
       [ 0.4901481 ,  0.33806482,  0.48850006, ..., -0.29197463,
        -0.08600576, -0.08736482]], dtype=float32)}}


Saving Timeseries
-----------------
.. code-block:: python

    extractor.timeseries_to_pickle(output_dir=dir, filename="rest_Schaefer.pkl")

Visualizing Timeseries
----------------------
.. code-block:: python

    # Visualizing a region
    extractor.visualize_bold(subj_id="01", region="Vis")

.. image:: embed/visualize_timeseries_regions.png
    :width: 1000

.. code-block:: python

    # Visualizing a several nodes
    extractor.visualize_bold(subj_id="01", run="001", roi_indx=[0, 1, 2])
    extractor.visualize_bold(subj_id="01", run="001", roi_indx=["LH_Vis_1", "LH_Vis_2", "LH_Vis_3"])

.. image:: embed/visualize_timeseries_nodes.png
    :width: 1000

==========

.. [1] Laumann, T. O., Gordon, E. M., Adeyemo, B., Snyder, A. Z., Joo, S. J., Chen, M. Y., Gilmore, A. W., McDermott, K. B., Nelson, S. M., Dosenbach, N. U., Schlaggar, B. L., Mumford, J. A., Poldrack, R. A., & Petersen, S. E. (2015). Functional system and areal organization of a highly sampled individual human brain. Neuron, 87(3), 657–670. https://doi.org/10.1016/j.neuron.2015.06.037
