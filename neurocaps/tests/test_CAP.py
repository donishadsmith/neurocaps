import os, numpy as np, sys, pytest, warnings
dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(dirname)

from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP

def test_CAP_get_caps_no_groups():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    extractor._subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,400) for y in range(1,4)} for x in range(1,11)}
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, n_clusters=2)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries)
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (400,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (400,)
    
def test_CAP_get_caps_with_groups():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    extractor._subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,400) for y in range(1,4)} for x in range(1,11)}
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]}, n_clusters=2)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries)
    assert cap_analysis.caps["A"]["CAP-1"].shape == (400,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (400,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (400,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (400,)

def test_CAP_get_caps_with_no_groups_and_silhouette_method():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    extractor._subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,400) for y in range(1,4)} for x in range(1,11)}
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, n_clusters=[2,3,4,5], cluster_selection_method="silhouette")
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries)
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (400,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (400,)


def test_CAP_get_caps_with_groups_and_silhouette_method():
    warnings.simplefilter('ignore')
    parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
    extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
    extractor._subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,400) for y in range(1,4)} for x in range(1,11)}
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]}, n_clusters=[2,3,4,5], cluster_selection_method="silhouette")
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries)
    assert cap_analysis.caps["A"]["CAP-1"].shape == (400,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (400,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (400,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (400,)
    



 