Tutorial 6: Generating Transition Probability Matrices ``neurocaps.analysis.transition_matrix``
===============================================================================================
``CAP.calculate_metrics``` can be used to calculate the transition probabilities for all subjects,
which can then be converted to matrix form and visualized with ``transition_matrix``.

.. code-block:: python

    import numpy as np
    from neurocaps.analysis import CAP, transition_matrix

    # Simulate `subject_timeseries` structure
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(10, 100) for y in range(1, 4)} for x in range(1, 11)}

    cap_analysis.get_caps(subject_timeseries=subject_timeseries,
                          cluster_selection_method="davies_bouldin",
                          standardize=True,
                          n_clusters=list(range(2, 6)))

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        2024-09-16 00:09:54,273 [INFO] [GROUP: All Subjects | METHOD: davies_bouldin] Optimal cluster size is 3.

.. code-block:: python

    outputs = cap_analysis.calculate_metrics(subject_timeseries=subject_timeseries,
                                             return_df=True,
                                             metrics=["transition_probability"],
                                             continuous_runs=True,
                                             output_dir=output_dir,
                                             prefix_filename="All_Subjects_CAPs_metrics")

    print(outputs["transition_probability"]["All Subjects"])

.. csv-table::
   :file: embed/transition_probability-All_Subjects.csv
   :header-rows: 1

.. code-block:: python

    kwargs = {"cmap": "viridis", "fmt": ".3f", "annot": True}

    trans_outputs = transition_matrix(trans_dict=outputs["transition_probability"],
                                      show_figs=True,
                                      return_df=True,
                                      **kwargs)

.. image:: embed/All_Subjects_CAPs_transition_probability_matrix.png
    :width: 600

.. code-block:: python

    print(trans_outputs["All Subjects"])

.. csv-table::
   :file: embed/All_Subjects_CAPs_transition_probability_matrix.csv
   :header-rows: 1
