Tutorial 2: Using ``neurocaps.analysis.CAP``
============================================

Performing CAPs on All Subjects
-------------------------------
.. code-block:: python

    from neurocaps.extraction import TimeseriesExtractor
    from neurocaps.analysis import CAP

    # Extracting timseries
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7, "resolution_mm": 2}}

    extractor = TimeseriesExtractor(parcel_approach=parcel_approach, standardize="zscore_sample",
                                    use_confounds=True, detrend=True, low_pass=0.15, high_pass=0.01,
                                    confound_names=confounds, n_acompcor_separate=6)

    extractor.get_bold(bids_dir=bids_dir, task="emo", condition="positive",
                    pipeline_name=pipeline_name, n_cores=10)

    # Saving timeseries
    extractor.timeseries_to_pickle(output_dir="path/to/dir", filename="task-positive_Schaefer.pkl")

    # Initialize CAP class
    cap_analysis = CAP()

    # Get CAPs
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=list(range(2,11)),
                          cluster_selection_method="elbow", show_figs=True, step=2)

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        Optimal cluster size for All Subjects is 6.

.. image:: embed/All_Subjects_elbow.png
    :width: 600

Performing CAPs on Groups
-------------------------
.. code-block:: python

    cap_analysis = CAP(groups={"A": ["1","2","3","5"], "B": ["4","6","7","8","9","10"]})

    cap_analysis.get_caps(subject_timeseries="subject_timeseries.pkl",
                          n_clusters=list(range(2,21)),
                          cluster_selection_method="silhouette", show_figs=True, step=2)

.. rst-class:: sphx-glr-script-out

    .. code-block:: none

        Optimal cluster size for A is 2.
        Optimal cluster size for B is 2.

.. image:: embed/A_silhouette.png
    :width: 600

.. image:: embed/B_silhouette.png
    :width: 600

Calculate Metrics
-----------------
.. code-block:: python

    df_dict = cap_analysis.calculate_metrics(subject_timeseries="subject_timeseries.pkl", return_df=True,
                                             metrics = ["temporal_fraction", "counts"], continuous_runs=True)

    df_dict["temporal_fraction"]

.. csv-table::
   :file: embed/temporal_fraction.csv
   :header-rows: 1

Plotting CAPs
-------------

.. code-block:: python

    import seaborn as sns

    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, n_clusters=6)
    sns.diverging_palette(145, 300, s=60, as_cmap=True)
    palette = sns.diverging_palette(260, 10, s=80, l=55, n=256, as_cmap=True)
    cap_analysis.caps2plot(visual_scope="regions", plot_options="outer_product", subplots=True,
                           fontsize=14, ncol=3, sharey=False, tight_layout=False, xlabel_rotation=0,
                           hspace = 0.3, cmap=palette, show_figs =True)

.. image:: embed/All_Subjects_CAPs_outer_product_heatmap-regions.png
    :width: 1000


.. code-block:: python

    cap_analysis.caps2plot(visual_scope="nodes", plot_options="heatmap",
                           xticklabels_size = 7, yticklabels_size = 7,
                           show_figs=True)

.. image:: embed/All_Subjects_CAPs_heatmap-nodes.png
    :width: 600

Generate Correlation Matrix
---------------------------
.. code-block:: python

    cap_analysis.caps2corr(annot=True, cmap="viridis", show_figs=True)

.. image:: embed/All_Subjects_CAPs_correlation_matrix.png
    :width: 600

Creating Surface Plots
----------------------
.. code-block:: python

    from matplotlib.colors import LinearSegmentedColormap

    # Create the colormap
    colors = ["#1bfffe", "#00ccff", "#0099ff", "#0066ff", "#0033ff", "#c4c4c4", "#ff6666",
            "#ff3333", "#FF0000","#ffcc00","#FFFF00"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cold_hot", colors, N=256)

    # Apply custom cmap to surface plots
    cap_analysis.caps2surf(cmap=custom_cmap, size=(500, 100), layout="row")

.. rst-class:: sphx-glr-script-out
    
    .. code-block:: none

        Optimal cluster size for All Subjects is 2.

.. image:: embed/All_Subjects_CAP-1_surface_plot.png
    :width: 800
.. image:: embed/All_Subjects_CAP-2_surface_plot.png
    :width: 800

Plotting CAPs to Radar
----------------------
.. code-block:: python

    radialaxis={"showline": True, "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)",
                "gridcolor": "rgba(0, 0, 0, 0.25)", "ticks": "outside" ,
                "tickfont": {"size": 14, "color": "black"}, "range": [0,0.3],
                "tickvals": [0.1,0.2,0.3]}
    legend = {"yanchor": "top", "xanchor": "left", "y": 0.99, "x": 0.01,
              "title_font_family": "Times New Roman", "font": {"size": 16, "color": "black"}}

    cap_analysis.caps2radar(radialaxis=radialaxis, fill="toself", legend=legend)

.. image:: embed/All_Subjects_CAP-1_radar.png
    :width: 800
.. image:: embed/All_Subjects_CAP-2_radar.png
    :width: 800