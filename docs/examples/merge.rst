Tutorial 3: Merging Timeseries With ``neurocaps.analysis.merge_dicts``
======================================================================
The ``merge_dicts`` function allows you to combine timeseries data from different tasks, enabling analyses that identify
similar CAPs across these tasks. This is only useful when the tasks includes the same subjects. This function
produces a merged dictionary only containing subject IDs present across all input dictionaries. Additionally,
while the run IDS across task do not need to be similar, the timeseries of the same run-IDs across dictionaries
will be appended. Note that successful merging requires all dictionaries to contain the same number of columns/ROIs.

.. code-block:: python

    import numpy as np
    from neurocaps.analysis import merge_dicts

    # Using a random dictionary that would be generate by TimeseriesExtractor to demonstrate what merge_dicts does
    # Generating a subject_timeseries dict containing five subjects, each with three runs that have 10 timepoints
    # and 100 rois.
    subject_timeseries_1 = {str(x): {f"run-{y}": np.random.rand(10, 100) for y in range(1, 4)} for x in range(1, 6)}
    # Deleting run-2 for subject 2; situation where subject 2 only completed two runs of a task
    del subject_timeseries_1["2"]["run-3"]
    # Second dictionary contains 2 subjects, with a single run, each with 20 timepoints
    subject_timeseries_2 = {str(x): {f"run-{y}": np.random.rand(20, 100) for y in range(1, 2)} for x in range(1, 3)}

    # subject_timeseries_list also takes pickle files and can save the modified dictionaries as pickles too.
    subject_timeseries_merged = merge_dicts(
        subject_timeseries_list=[subject_timeseries_1, subject_timeseries_2],
        return_merged_dict=True,
        return_reduced_dicts=False,
    )

    for subj_id in subject_timeseries_merged["merged"]:
        for run_id in subject_timeseries_merged["merged"][subj_id]:
            timeseries = subject_timeseries_merged["merged"][subj_id][run_id]
            print(f"sub-{subj_id}; {run_id} shape is {timeseries.shape}")

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        sub-1; run-1 shape is (30, 100)
        sub-1; run-2 shape is (10, 100)
        sub-1; run-3 shape is (10, 100)
        sub-2; run-1 shape is (30, 100)
        sub-2; run-2 shape is (10, 100)

.. code-block:: python

    # The original dictionaries can also be returned too. The only modifications done is that the originals will
    # Only contain the subjects present across all dictionaries in the list
    merged_dicts = merge_dicts(
        subject_timeseries_list=[subject_timeseries_1, subject_timeseries_2],
        return_merged_dict=True,
        return_reduced_dicts=True,
    )

    for dict_id in merged_dicts:
        for subj_id in merged_dicts[dict_id]:
            for run_id in merged_dicts[dict_id][subj_id]:
                timeseries = merged_dicts[dict_id][subj_id][run_id]
                print(f"For {dict_id} sub-{subj_id}; {run_id} shape is {timeseries.shape}")

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        For dict_0 sub-1; run-1 shape is (10, 100)
        For dict_0 sub-1; run-2 shape is (10, 100)
        For dict_0 sub-1; run-3 shape is (10, 100)
        For dict_0 sub-2; run-1 shape is (10, 100)
        For dict_0 sub-2; run-2 shape is (10, 100)
        For dict_1 sub-1; run-1 shape is (20, 100)
        For dict_1 sub-2; run-1 shape is (20, 100)
        For merged sub-1; run-1 shape is (30, 100)
        For merged sub-1; run-2 shape is (10, 100)
        For merged sub-1; run-3 shape is (10, 100)
        For merged sub-2; run-1 shape is (30, 100)
        For merged sub-2; run-2 shape is (10, 100)
