import os, numpy as np, pickle, pytest, warnings

from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP

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

def test_CAP_get_caps_with_no_groups_and_silhouette_method():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,11)}
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["All Subjects"].values())

def test_CAP_get_caps_with_groups_and_silhouette_method():
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
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")
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