import os, numpy as np, pandas as pd, pickle, pytest, warnings

from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP, change_dtype

def test_CAP_get_caps_no_groups():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,11)}
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, n_clusters=2)
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)

def test_CAP_get_caps_with_groups():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,11)}
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries)
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)

def test_CAP_get_caps_with_no_groups_cluster_selection():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,11)}
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")

    assert max(cap_analysis.silhouette_scores["All Subjects"], key=cap_analysis.silhouette_scores["All Subjects"].get) == cap_analysis.optimal_n_clusters["All Subjects"]

    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["All Subjects"].values())
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="variance_ratio")

    assert max(cap_analysis.variance_ratio["All Subjects"], key=cap_analysis.variance_ratio["All Subjects"].get) == cap_analysis.optimal_n_clusters["All Subjects"]

    assert all(elem >= 0 for elem in cap_analysis.variance_ratio["All Subjects"].values())

    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="davies_bouldin")

    assert min(cap_analysis.davies_bouldin["All Subjects"], key=cap_analysis.davies_bouldin["All Subjects"].get) == cap_analysis.optimal_n_clusters["All Subjects"]

    assert all(elem >= 0 for elem in cap_analysis.davies_bouldin["All Subjects"].values())

def test_CAP_get_caps_with_groups_and_cluster_selection():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,11)}
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)

    assert max(cap_analysis.silhouette_scores["A"], key=cap_analysis.silhouette_scores["A"].get) == cap_analysis.optimal_n_clusters["A"]
    assert max(cap_analysis.silhouette_scores["B"], key=cap_analysis.silhouette_scores["B"].get) == cap_analysis.optimal_n_clusters["B"]

    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="davies_bouldin")

    assert min(cap_analysis.davies_bouldin["A"], key=cap_analysis.davies_bouldin["A"].get) == cap_analysis.optimal_n_clusters["A"]
    assert min(cap_analysis.davies_bouldin["B"], key=cap_analysis.davies_bouldin["B"].get) == cap_analysis.optimal_n_clusters["B"]

    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="variance_ratio")

    assert max(cap_analysis.variance_ratio["A"], key=cap_analysis.variance_ratio["A"].get) == cap_analysis.optimal_n_clusters["A"]
    assert max(cap_analysis.variance_ratio["B"], key=cap_analysis.variance_ratio["B"].get) == cap_analysis.optimal_n_clusters["B"]


def test_CAP_get_caps_no_groups_pkl():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    with open("sample_timeseries.pkl", "rb") as f:
        subject_timeseries = pickle.load(f)
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=2)
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)

def test_CAP_get_caps_with_groups_pkl():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    with open("sample_timeseries.pkl", "rb") as f:
        subject_timeseries = pickle.load(f)
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, n_clusters=2)
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)

def test_CAP_get_caps_with_no_groups_and_silhouette_method_pkl():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    with open("sample_timeseries.pkl", "rb") as f:
        subject_timeseries = pickle.load(f)
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette", n_cores=1)
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["All Subjects"].values())

def test_CAP_get_caps_with_groups_and_silhouette_method_pkl():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    with open("sample_timeseries.pkl", "rb") as f:
        subject_timeseries = pickle.load(f)
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["A"].values())
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["A"].values())
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)

def test_CAP_get_caps_with_groups_and_silhouette_method_pkl():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    with open("sample_timeseries.pkl", "rb") as f:
        subject_timeseries = pickle.load(f)
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")

    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["A"].values())
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["A"].values())

    df_dict = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True)
    assert len(df_dict) == 4

    new_timeseries = change_dtype(extractor.subject_timeseries, dtype="float16")
    cap_analysis.calculate_metrics(subject_timeseries=new_timeseries, return_df=True)

    cap_analysis.get_caps(subject_timeseries=new_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette", runs=["1", "2"])

    cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True, runs=1)
    cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True, runs=1,
                                   continuous_runs=True)

    cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True, runs="1")


    cap_analysis.caps2plot(subplots=True, xlabel_rotation=90, sharey=True, borderwidths=10, show_figs=False)

    cap_analysis.caps2plot(subplots=False, yticklabels_size=5, wspace = 0.1, visual_scope="nodes", xlabel_rotation=90,
                        xticklabels_size = 5, hspace = 0.6, tight_layout = False, show_figs=False)

    df = cap_analysis.caps2corr(annot=True, show_figs=False, return_df=True)
    assert isinstance(df, dict)
    assert isinstance(df["A"], pd.DataFrame)
    assert len(list(df)) == 2

    radialaxis={"showline": True, "linewidth": 2, "linecolor": "rgba(0, 0, 0, 0.25)", "gridcolor": "rgba(0, 0, 0, 0.25)",
            "ticks": "outside" , "tickfont": {"size": 14, "color": "black"}, "range": [0,0.3],
            "tickvals": [0.1,0.2,0.3]}
    cap_analysis.caps2radar(radialaxis=radialaxis, fill="toself", scattersize=10, show_figs=False, as_html=True)
