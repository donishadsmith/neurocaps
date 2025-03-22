Tutorial 5: Changing Dtype With ``change_dtype``
===================================================================
The dtype of the all participant's NumPy arrays can be changed to assist with memory usage.

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/donishadsmith/neurocaps/blob/stable/docs/examples/notebooks/dtype.ipynb

|colab|

.. code-block:: python

    import numpy as np
    from neurocaps.analysis import change_dtype

    subject_timeseries = {str(x): {f"run-{y}": np.random.rand(50, 100) for y in range(1, 3)} for x in range(1, 3)}
    converted_subject_timeseries = change_dtype(subject_timeseries_list=[subject_timeseries], dtype=np.float32)
    for subj_id in subject_timeseries:
        for run in subject_timeseries[subj_id]:
            print(
                f"""
                  subj-{subj_id}; {run}:
                  dtype before conversion {subject_timeseries[subj_id][run].dtype}
                  dtype after conversion: {converted_subject_timeseries["dict_0"][subj_id][run].dtype}
                  """
            )

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


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter Notebook <notebooks/dtype.ipynb>`
