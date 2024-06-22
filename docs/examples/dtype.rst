Tutorial 5: Changing Dtype With ``neurocaps.analysis.change_dtype``
===================================================================
Changes the dtype of all participant's numpy arrays to assist with memory usage.

.. code-block:: python

    import numpy as np
    from neurocaps.analysis import change_dtype

    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(50,100) for y in range(1,3)} for x in range(1,3)}
    converted_subject_timeseries = change_dtype(subject_timeseries=subject_timeseries, dtype=np.float32)
    for subj_id in subject_timeseries:
        for run in subject_timeseries[subj_id]:
            print(f"""
                  subj-{subj_id}; {run}:
                  dtype before conversion {subject_timeseries[subj_id][run].dtype}
                  dtype after conversion: {converted_subject_timeseries[subj_id][run].dtype}
                  """)

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

         subj-1; run-1:
         dtype before conversion float64
         dtype after conversion: float32
        

         subj-1; run-2:
         dtype before conversion float64
         dtype after conversion: float32
        

         subj-2; run-1:
         dtype before conversion float64
         dtype after conversion: float32
        

         subj-2; run-2:
         dtype before conversion float64
         dtype after conversion: float32
            
