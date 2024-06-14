Tutorial 1: Using ``TimeseriesExtractor``
=========================================

Extracting Timeseries
---------------------

.. code-block:: python

    from neurocaps.extraction import TimeseriesExtractor
    """If an asterisk '*' is after a name, all confounds starting with the 
    term preceding the parameter will be used. in this case, all parameters 
    starting with cosine will be used."""

    confounds = ["cosine*", "trans_x", "trans_x_derivative1", "trans_y", 
                "trans_y_derivative1", "trans_z","trans_z_derivative1", 
                "rot_x", "rot_x_derivative1", "rot_y", "rot_y_derivative1", 
                "rot_z","rot_z_derivative1"]

    """If use_confounds is True but no confound_names provided, there are hardcoded 
    confound names that will extract the data from the confound files outputted by fMRIPrep
    `n_acompcor_separate` will use the first 'n' components derived from the separate 
    white-matter (WM) and cerebrospinal fluid (CSF). To use the acompcor components from the 
    combined mask, list them in the `confound_names` parameter"""

    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 2}}

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01, 
                                    confound_names=confounds, n_acompcor_separate=6)

    bids_dir = "/path/to/bids/dir"

    # If there are multiple pipelines in the derivatives folder, you can specify a specific pipeline
    pipeline_name = "fmriprep-1.4.0"

    # Resting State
    extractor.get_bold(bids_dir=bids_dir, task="rest", pipeline_name=pipeline_name)

    # Task; use parallel processing with `n_cores`
    extractor.get_bold(bids_dir=bids_dir, task="emo", condition="positive", 
                    pipeline_name=pipeline_name, n_cores=10)

Saving Timeseries
-----------------
.. code-block:: python

    extractor.timeseries_to_pickle(output_dir="path/to/dir", filename="task-positive_Schaefer.pkl")

Visualizing Timeseries
----------------------
.. code-block:: python

    # Visualizing a region
    extractor.visualize_bold(subj_id="1", region="Vis")

.. image:: embed/visualize_timeseries_regions.png
    :width: 1000

.. code-block:: python

    # Visualizing a several nodes
    extractor.visualize_bold(subj_id="1",run=1, roi_indx=[0,1,2])
    # or
    extractor.visualize_bold(subj_id="1",run=1, roi_indx=["LH_Vis_1","LH_Vis_2","LH_Vis_3"])

.. image:: embed/visualize_timeseries_nodes.png
    :width: 1000


