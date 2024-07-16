import copy, numpy as np, pandas as pd, pickle, pytest, warnings
from kneed import KneeLocator
from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP, change_dtype

warnings.simplefilter('ignore')
parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}
extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
subject_timeseries = {str(x) : {f"run-{y}": np.random.rand(100,100) for y in range(1,4)} for x in range(1,11)}
extractor.subject_timeseries = subject_timeseries

# Similar to internal function used in CAP; function is _concatenated_timeseries
def concat_data(subject_table,standardize,runs=[1,2,3]):
    concatenated_timeseries = {group: None for group in set(subject_table.values())}
    std = {group: None for group in set(subject_table.values())}

    for sub, group in subject_table.items():
        for run in subject_timeseries[sub]:
            if int(run.split("run-")[-1]) in runs:
                if concatenated_timeseries[group] is None:
                    concatenated_timeseries[group] = subject_timeseries[sub][run]
                    # Sanity check to check mutability, both variables should point to same array
                    assert id(concatenated_timeseries[group]) == id(subject_timeseries[sub][run])
                    first_sub, first_run = sub, run
                    first_subject_timeseries = copy.deepcopy(subject_timeseries[sub][run])
                    # Deepcopy results in creation of new, independent array; no longer pointing to the same object in memory
                    # Any mutability issues for subject_timeseries[sub][run] will be caught by later assert
                    assert id(first_subject_timeseries) != id(subject_timeseries[sub][run])
                else:
                    concatenated_timeseries[group] = np.vstack([concatenated_timeseries[group],
                                                                subject_timeseries[sub][run]])
                    
    # Check to check mutability, array are unique but should be the same, another sanity check to ensure no mutability issues
    # since np.vstack would create a new array.
    assert np.array_equal(first_subject_timeseries, subject_timeseries[first_sub][first_run])

    if standardize:
        for _, group in subject_table.items():
            # Recalculating means and stdev, will cause minor floating point differences so np.allclose must be used. Done to
            # ensure the correct means and std are being calculated for each group if np.allclose is True this should be the case
            concatenated_timeseries[group] -= np.mean(concatenated_timeseries[group], axis=0)
            std[group] = np.std(concatenated_timeseries[group], ddof=1, axis=0)
            # Taken from nilearn pipeline, used for numerical stability purposes to avoid numpy division error
            std[group][std[group] < np.finfo(np.float64).eps] = 1.0
            concatenated_timeseries[group] /= std[group]

    return concatenated_timeseries

# Similar method to how labels are predicted in CAP.calculate_metrics, same labels are then used to calculate the metrics
def predict_labels(subject_timeseries, cap_analysis, standardize, group, runs=[1,2,3]):
    labels = None
    group_dict = cap_analysis.groups[group]
    for sub in subject_timeseries:
        for run in subject_timeseries[sub]:
            if int(run.split("run-")[-1]) in runs and sub in group_dict:
                new_timeseries = copy.deepcopy(subject_timeseries[sub][run])
                if standardize:
                    # .calculate_methods uses the previously calculated means and stdev conducted in the CAP.get_bold method,
                    # its done this way to avoid the minor numerical differences caused by floating point operations due to
                    # recalculating statistics so np.array_equal can be used to assess these predicted labels and the ones from the
                    # kmeans model
                    new_timeseries -= cap_analysis.means[group]
                    new_timeseries /= cap_analysis.stdev[group]

                if labels is None:
                    labels = cap_analysis.kmeans[group].predict(new_timeseries)
                else:
                    labels = np.hstack([labels, cap_analysis.kmeans[group].predict(new_timeseries)])
    return labels

@pytest.mark.parametrize("standardize", [True, False])
def test_no_groups_no_cluster_selection(standardize):
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, n_clusters=2, standardize=standardize)
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)

    concatenated_timeseries = concat_data(cap_analysis.subject_table, standardize=standardize)

    # Concatenated data used for kmeans
    assert cap_analysis.concatenated_timeseries["All Subjects"].shape == (3000,100)
    if standardize is False:
        assert np.array_equal(cap_analysis.concatenated_timeseries["All Subjects"], concatenated_timeseries["All Subjects"])
    else:
        # Floating point differences
        assert np.allclose(cap_analysis.concatenated_timeseries["All Subjects"], concatenated_timeseries["All Subjects"])
    
    # Get labels; Validate that predicted labels for individual runs and getting labels will get the same labels in .labels_
    # This validates the way labels are predicted in the CAP.calculate_metrics(), which iterates through the subject_timeseries and predict
    # labels for each run of each subject. This is a way to ensure that even dictionaries merge with the merge_dicts function
    # should have the same labels when predicting individual subsets of the entire dictionary
    labels = predict_labels(subject_timeseries, cap_analysis, standardize, "All Subjects")

    assert np.array_equal(labels, cap_analysis.kmeans["All Subjects"].labels_)

@pytest.mark.parametrize("standardize", [True, False])
def test_groups_no_cluster_selection(standardize):
    # Should ignore duplicate id
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,standardize=standardize)
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)
    # Concatenated data used in kmeans
    cap_analysis.concatenated_timeseries["A"].shape == (1200,100)
    cap_analysis.concatenated_timeseries["A"].shape == (1800,100)
    concatenated_timeseries = concat_data(cap_analysis.subject_table, standardize=standardize)
    if standardize is False:
        assert np.array_equal(cap_analysis.concatenated_timeseries["A"], concatenated_timeseries["A"])
        assert np.array_equal(cap_analysis.concatenated_timeseries["B"], concatenated_timeseries["B"])
    else:
        assert np.allclose(cap_analysis.concatenated_timeseries["A"], concatenated_timeseries["A"])
        assert np.allclose(cap_analysis.concatenated_timeseries["B"], concatenated_timeseries["B"])

    labels = predict_labels(subject_timeseries, cap_analysis, standardize, "A")
    assert np.array_equal(labels, cap_analysis.kmeans["A"].labels_)
    labels = predict_labels(subject_timeseries, cap_analysis, standardize, "B")
    assert np.array_equal(labels, cap_analysis.kmeans["B"].labels_)

    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, runs=[1,2],standardize=standardize)
    cap_analysis.concatenated_timeseries["A"].shape == (800,100)
    cap_analysis.concatenated_timeseries["A"].shape == (1200,100)
    concatenated_timeseries = concat_data(cap_analysis.subject_table, standardize=standardize, runs=[1,2])
    if standardize is False:
        assert np.array_equal(cap_analysis.concatenated_timeseries["A"], concatenated_timeseries["A"])
        assert np.array_equal(cap_analysis.concatenated_timeseries["B"], concatenated_timeseries["B"])
    else:
        assert np.allclose(cap_analysis.concatenated_timeseries["A"], concatenated_timeseries["A"])
        assert np.allclose(cap_analysis.concatenated_timeseries["B"], concatenated_timeseries["B"])

    labels = predict_labels(subject_timeseries, cap_analysis, standardize, "A", runs=[1,2])
    assert np.array_equal(labels, cap_analysis.kmeans["A"].labels_)
    labels = predict_labels(subject_timeseries, cap_analysis, standardize, "B", runs=[1,2])
    assert np.array_equal(labels, cap_analysis.kmeans["B"].labels_)

def test_no_groups_cluster_selection():
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)

    # Elbow sometimes does find the elbow with random data
    try:
        cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                            n_clusters=list(range(2,41)), cluster_selection_method="elbow")
        print(cap_analysis.optimal_n_clusters["All Subjects"])
        try:
            assert all(elem >= 0 for elem in cap_analysis.inertia["All Subjects"].values())
            kneedle = KneeLocator(x=list(cap_analysis.inertia["All Subjects"]),
                                        y=list(cap_analysis.inertia["All Subjects"].values()),
                                        curve="convex", direction="decreasing")
            assert kneedle.elbow == cap_analysis.optimal_n_clusters["All Subjects"]
            assert cap_analysis.inertia["All Subjects"][cap_analysis.optimal_n_clusters["All Subjects"]] == cap_analysis.kmeans["All Subjects"].inertia_
        except:
            raise ValueError("Different results for kneedle and optimal cluster or not all inertia values are positive.")
    except:
        warnings.warn("Elbow could not be found for random data.")
        pass

    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")
    # Maximum silhouette is the most optimal
    assert max(cap_analysis.silhouette_scores["All Subjects"], key=cap_analysis.silhouette_scores["All Subjects"].get) == cap_analysis.optimal_n_clusters["All Subjects"]

    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["All Subjects"].values())
    assert all(-1 <= elem <= 1 for elem in cap_analysis.silhouette_scores["All Subjects"].values())
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="variance_ratio")

    # Maximum variance ratio is the most optimal
    assert max(cap_analysis.variance_ratio["All Subjects"], key=cap_analysis.variance_ratio["All Subjects"].get) == cap_analysis.optimal_n_clusters["All Subjects"]

    # All values not negative
    assert all(elem >= 0 for elem in cap_analysis.variance_ratio["All Subjects"].values())

    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="davies_bouldin")

    # Mininimum davies_bouldin is the most optimal
    assert min(cap_analysis.davies_bouldin["All Subjects"], key=cap_analysis.davies_bouldin["All Subjects"].get) == cap_analysis.optimal_n_clusters["All Subjects"]

    # All values not negative
    assert all(elem >= 0 for elem in cap_analysis.davies_bouldin["All Subjects"].values())

def test_groups_and_cluster_selection():
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)
        # Elbow sometimes does find the elbow with random data, uses kneed to locate elbow

    # Elbow sometimes does find the elbow with random data
    try:
        cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
        cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                            n_clusters=list(range(2,41)), cluster_selection_method="elbow")
        try:
            assert all(elem >= 0 for elem in cap_analysis.inertia["A"].values())
            assert all(elem >= 0 for elem in cap_analysis.inertia["B"].values())
            kneedle = KneeLocator(x=list(cap_analysis.inertia["A"]),
                                        y=list(cap_analysis.inertia["A"].values()),
                                        curve="convex", direction="decreasing")
            assert kneedle.elbow == cap_analysis.optimal_n_clusters["A"]
            kneedle = KneeLocator(x=list(cap_analysis.inertia["B"]),
                                        y=list(cap_analysis.inertia["B"].values()),
                                        curve="convex", direction="decreasing")
            assert kneedle.elbow == cap_analysis.optimal_n_clusters["B"]
        except:
            raise ValueError("Different results for kneedle and optimal cluster or not all inertia values are positive.")
    except:
        warnings.warn("Elbow could not be found for random data.")
        pass

    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")
    
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

def test_no_groups_pkl():
    with open("sample_timeseries.pkl", "rb") as f:
        subject_timeseries = pickle.load(f)
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=2)
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)

def test_groups_pkl():
    with open("sample_timeseries.pkl", "rb") as f:
        subject_timeseries = pickle.load(f)
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, n_clusters=2)
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)

def test_no_groups_and_silhouette_method():
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette", n_cores=1)
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["All Subjects"].values())
    assert all(-1 <= elem <= 1 for elem in cap_analysis.silhouette_scores["All Subjects"].values())

def test_groups_and_silhouette_method():
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["A"].values())
    assert all(-1 <= elem <= 1 for elem in cap_analysis.silhouette_scores["A"].values())
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["B"].values())
    assert all(-1 <= elem <= 1 for elem in cap_analysis.silhouette_scores["B"].values())
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)

def test_multiple_methods():
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach, groups={"A": [1,2,3,5], "B": [4,6,7,8,9,10,7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries,
                          n_clusters=[2,3,4,5], cluster_selection_method="silhouette")

    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["A"].values())
    assert all(elem > 0  or elem < 0 for elem in cap_analysis.silhouette_scores["A"].values())

    df_dict = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True)
    assert len(df_dict) == 4

    new_timeseries = change_dtype(extractor.subject_timeseries, dtype="float16")

    cap_analysis.calculate_metrics(subject_timeseries=new_timeseries, return_df=True)

    # No crashing
    met1 = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True, runs=1)
    met2 = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True, runs=1,
                                   continuous_runs=True)
    
    # If only one run continuous_runs should not differ
    assert met1["persistence"].equals(met2["persistence"])

    met1 = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True)
    met2 = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True,continuous_runs=True)
    
    # Should differ
    assert not met1["persistence"].equals(met2["persistence"])

    # Continuous run should have 1/3 the number of rows since each subject in the randomized data has three runs
    assert met1["persistence"].shape[0]/3 == met2["persistence"].shape[0]

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
