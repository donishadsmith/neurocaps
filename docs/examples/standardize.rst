Tutorial 4: Standardizing Within Runs Using ``neurocaps.analysis.standardize``
==============================================================================
While standardizing the features/columns within runs can be done using the ``standardize`` parameter within the
``TimeseriesExtractor`` class, the ``neurocaps.analysis.standardize`` function can be used to perform
within-run standardization post-extraction.

.. code-block:: python

    import numpy as np
    from neurocaps.analysis import standardize

    subject_timeseries = {str(x): {f"run-{y}": np.random.rand(10, 100) for y in range(1, 4)} for x in range(1, 6)}

    # Getting mean and standard deviation for run 1 and 2 of subject 1
    mean_vec_1 = subject_timeseries["1"]["run-1"].mean(axis=0)
    std_vec_1 = subject_timeseries["1"]["run-1"].std(ddof=1, axis=0)
    mean_vec_2 = subject_timeseries["1"]["run-2"].mean(axis=0)
    std_vec_2 = subject_timeseries["1"]["run-2"].std(ddof=1, axis=0)

    # Avoid numerical stability issues
    std_vec_1[std_vec_1 < np.finfo(np.float64).eps] = 1.0
    std_vec_2[std_vec_2 < np.finfo(np.float64).eps] = 1.0

    standardized_subject_timeseries = standardize(subject_timeseries_list=[subject_timeseries])

    standardized_1 = (subject_timeseries["1"]["run-1"] - mean_vec_1) / std_vec_1
    standardized_2 = (subject_timeseries["1"]["run-2"] - mean_vec_2) / std_vec_2

    print(np.array_equal(standardized_subject_timeseries["dict_0"]["1"]["run-1"], standardized_1))
    print(np.array_equal(standardized_subject_timeseries["dict_0"]["1"]["run-2"], standardized_2))

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        True
        True
