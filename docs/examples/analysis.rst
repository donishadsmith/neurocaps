Tutorial 2: Using ``CAP``
=========================

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/donishadsmith/neurocaps/blob/stable/docs/examples/notebooks/analysis.ipynb

|colab|

The ``CAP`` class is designed to perform CAPs analyses (on all subjects or group of subjects). It offers the flexibility
to analyze data from all subjects or focus on specific groups, compute CAP-specific metrics, and generate visualizations
to aid in the interpretation of results.

Performing CAPs on All Subjects
-------------------------------
All information pertaining to CAPs (k-means models, activation vectors/cluster centroids, etc) are stored as attributes
in the ``CAP`` class and this information is used by all methods in the class. These attributes are accessible via
`properties <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.analysis.CAP.html#properties>`_.
**Some properties can also be used as setters.**

.. code-block:: python

    import numpy as np
    from neurocaps.analysis import CAP

    # Extracting timseries
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 2}}

    # Simulate data for example; Subject IDs will be sorted lexicographically
    sub_ids = [f"0{x}" if x < 10 else x for x in range(1, 11)]
    subject_timeseries = {
        str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in sub_ids
    }

    # Initialize CAP class
    cap_analysis = CAP(parcel_approach=parcel_approach)

    # Get CAPs
    cap_analysis.get_caps(
        subject_timeseries=subject_timeseries,
        n_clusters=range(2, 11),
        cluster_selection_method="elbow",
        show_figs=True,
        step=2,
        progress_bar=True,  # Available in versions >= 0.21.5
    )

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        2025-07-03 18:14:40,114 neurocaps.analysis.cap._internals.cluster [INFO] No groups specified. Using default group 'All Subjects' containing all subject IDs from `subject_timeseries`. The `self.groups` dictionary will remain fixed unless the `CAP` class is re-initialized or `self.clear_groups()` is used.
        Collecting Subject Timeseries Data [GROUP: All Subjects]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00, 668.15it/s]
        Concatenating Timeseries Data Per Group: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:01<00:00, 178.01it/s]
        Clustering [GROUP: All Subjects]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 20.38it/s]
        2025-07-03 18:15:07,367 neurocaps.analysis.cap._internals.cluster [INFO] [GROUP: All Subjects | METHOD: elbow] Optimal cluster size is 5.

.. image:: embed/All_Subjects_elbow.png
    :width: 600


``print`` can be used to return a string representation of the ``CAP`` class.

.. code-block:: python

    print(cap_analysis)

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        Current Object State:
        ================================================
        Parcellation Approach                                      : Schaefer
        Groups                                                     : All Subjects
        Number of Clusters                                         : [2, 3, 4, 5, 6, 7, 8, 9, 10]
        Cluster Selection Method                                   : elbow
        Optimal Number of Clusters (if Range of Clusters Provided) : {'All Subjects': np.int64(5)}
        CPU Cores Used for Clustering (Multiprocessing)            : None
        User-Specified Runs IDs Used for Clustering                : None
        Concatenated Timeseries Bytes                              : 2400184 bytes
        Standardized Concatenated Timeseries                       : True
        Co-Activation Patterns (CAPs)                              : {'All Subjects': 5}
        Variance Explained by Clustering                           : {'All Subjects': np.float64(0.02448526803307005)}

Performing CAPs on Groups
-------------------------
.. code-block:: python

    cap_analysis = CAP(
        groups={"A": ["01", "02", "03", "05"], "B": ["04", "06", "07", "08", "09", "10"]}
    )

    cap_analysis.get_caps(
        subject_timeseries=subject_timeseries,
        n_clusters=range(2, 21),
        cluster_selection_method="silhouette",
        show_figs=True,
        step=2,
        progress_bar=True,
    )

    # The concatenated data can be safely deleted since only the kmeans models and any
    # standardization parameters are used for computing temporal metrics.
    del cap_analysis.concatenated_timeseries

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        Collecting Subject Timeseries Data [GROUP: A]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00, 582.04it/s]
        Collecting Subject Timeseries Data [GROUP: B]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00, 706.37it/s]
        Concatenating Timeseries Data Per Group: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00, 308.08it/s]

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        Clustering [GROUP: A]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19/19 [00:01<00:00, 18.71it/s]
        2025-07-03 18:15:53,981 neurocaps.analysis.cap._internals.cluster [INFO] [GROUP: A | METHOD: silhouette] Optimal cluster size is 2.

.. image:: embed/A_silhouette.png
    :width: 600

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        Clustering [GROUP: B]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 19/19 [00:01<00:00, 12.48it/s]
        2025-07-03 18:15:55,236 neurocaps.analysis.cap._internals.cluster [INFO] [GROUP: B | METHOD: silhouette] Optimal cluster size is 2.

.. image:: embed/B_silhouette.png
    :width: 600

Calculate Metrics
-----------------
Note that if ``standardize`` was set to True in ``CAP.get_caps()``, then the column (ROI) means and standard deviations
computed from the concatenated data used to obtain the CAPs are also used to standardize each subject in the timeseries
data inputted into ``CAP.calculate_metrics()``. This ensures proper CAP assignments for each subjects frames.

.. code-block:: python

    df_dict = cap_analysis.calculate_metrics(
        subject_timeseries=subject_timeseries,
        return_df=True,
        metrics=["temporal_fraction", "counts", "transition_probability"],
        continuous_runs=True,
        progress_bar=True,
    )

    print(df_dict["temporal_fraction"])

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        Computing Metrics for Subjects: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 159.78it/s]

.. csv-table::
   :file: embed/temporal_fraction.csv
   :header-rows: 1

Plotting CAPs
-------------

.. code-block:: python

    import seaborn as sns

    cap_analysis = CAP(
        parcel_approach={"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 1}}
    )

    cap_analysis.get_caps(subject_timeseries=subject_timeseries, n_clusters=6)

    sns.diverging_palette(145, 300, s=60, as_cmap=True)

    palette = sns.diverging_palette(260, 10, s=80, l=55, n=256, as_cmap=True)

    kwargs = {
        "subplots": True,
        "fontsize": 14,
        "ncol": 3,
        "sharey": True,
        "tight_layout": False,
        "xlabel_rotation": 0,
        "hspace": 0.3,
        "cmap": palette,
    }

    cap_analysis.caps2plot(
        visual_scope="regions", plot_options="outer_product", show_figs=True, **kwargs
    )

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        2025-07-03 18:16:21,487 neurocaps.analysis.cap._internals.cluster [INFO] No groups specified. Using default group 'All Subjects' containing all subject IDs from `subject_timeseries`. The `self.groups` dictionary will remain fixed unless the `CAP` class is re-initialized.


.. image:: embed/All_Subjects_CAPs_outer_product_heatmap-regions.png
    :width: 1000


.. code-block:: python

    cap_analysis.caps2plot(
        visual_scope="nodes",
        plot_options="heatmap",
        xticklabels_size=7,
        yticklabels_size=7,
        show_figs=True,
    )

.. image:: embed/All_Subjects_CAPs_heatmap-nodes.png
    :width: 600

Generate Correlation Matrix
-----------------------------------
.. code-block:: python

    cap_analysis.caps2corr(method="pearson", annot=True, cmap="viridis", show_figs=True)

.. image:: embed/All_Subjects_CAPs_correlation_matrix.png
    :width: 600

.. code-block:: python

    corr_dict = cap_analysis.caps2corr(method="pearson", return_df=True)
    print(corr_dict["All Subjects"])

.. csv-table::
   :file: embed/All_Subjects_CAPs_correlation_matrix.csv
   :header-rows: 1

Creating Surface Plots
----------------------
.. code-block:: python

    from matplotlib.colors import LinearSegmentedColormap

    # Create the colormap
    colors = [
        "#1bfffe",
        "#00ccff",
        "#0099ff",
        "#0066ff",
        "#0033ff",
        "#c4c4c4",
        "#ff6666",
        "#ff3333",
        "#FF0000",
        "#ffcc00",
        "#FFFF00",
    ]

    custom_cmap = LinearSegmentedColormap.from_list("custom_cold_hot", colors, N=256)

    # Apply custom cmap to surface plots
    cap_analysis.caps2surf(progress_bar=True, cmap=custom_cmap, size=(500, 100), layout="row")

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        Generating Surface Plots [GROUP: All Subjects]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.91s/it]

.. image:: embed/All_Subjects_CAP-1_surface_plot.png
    :width: 800

.. image:: embed/All_Subjects_CAP-2_surface_plot.png
    :width: 800

Plotting CAPs to Radar
----------------------
.. code-block:: python

    radialaxis = {
        "showline": True,
        "linewidth": 2,
        "linecolor": "rgba(0, 0, 0, 0.25)",
        "gridcolor": "rgba(0, 0, 0, 0.25)",
        "ticks": "outside",
        "tickfont": {"size": 14, "color": "black"},
        "range": [0, 0.6],
        "tickvals": [0.1, "", "", 0.4, "", "", 0.6],
    }

    legend = {
        "yanchor": "top",
        "y": 0.99,
        "x": 0.99,
        "title_font_family": "Times New Roman",
        "font": {"size": 12, "color": "black"},
    }

    colors = {"High Amplitude": "red", "Low Amplitude": "blue"}

    kwargs = {
        "radialaxis": radialaxis,
        "fill": "toself",
        "legend": legend,
        "color_discrete_map": colors,
        "height": 400,
        "width": 600,
    }

    cap_analysis.caps2radar(**kwargs)

.. image:: embed/All_Subjects_CAP-1_radar.png
    :width: 800
.. image:: embed/All_Subjects_CAP-2_radar.png
    :width: 800

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter Notebook <notebooks/analysis.ipynb>`
