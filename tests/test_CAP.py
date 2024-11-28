import copy, glob, logging, math, os, pickle, sys, tempfile
import nibabel as nib, numpy as np, pandas as pd, pytest
from kneed import KneeLocator
from neurocaps.extraction import TimeseriesExtractor
from neurocaps.analysis import CAP, change_dtype

LG = logging.getLogger(__name__)
LG.setLevel(logging.WARNING)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LG.addHandler(handler)

tmp_dir = tempfile.TemporaryDirectory()

parcel_approach = {"Schaefer": {"n_rois": 100, "yeo_networks": 7}}

with open(os.path.join(os.path.dirname(__file__), "data", "HCPex_parcel_approach.pkl"), "rb") as f:
    custom_parcel_approach = pickle.load(f)
    custom_parcel_approach["Custom"]["maps"] = os.path.join(os.path.dirname(__file__), "data", "HCPex.nii.gz")
    custom_subject_timeseries = {
        str(x): {f"run-{y}": np.random.rand(100, 426) for y in range(1, 4)} for x in range(1, 11)
    }

extractor = TimeseriesExtractor(parcel_approach=parcel_approach)
extractor2 = TimeseriesExtractor(parcel_approach={"AAL": {}})

aal_subject_timeseries = {str(x): {f"run-{y}": np.random.rand(100, 116) for y in range(1, 4)} for x in range(1, 11)}

subject_timeseries = {str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 11)}
extractor.subject_timeseries = subject_timeseries

# Similar to internal function used in CAP; function is _concatenated_timeseries
def concat_data(subject_table, standardize, runs=[1, 2, 3]):
    concatenated_timeseries = {group: None for group in set(subject_table.values())}
    std = {group: None for group in set(subject_table.values())}

    for sub, group in subject_table.items():
        for run in subject_timeseries[sub]:
            if int(run.split("run-")[-1]) in runs:
                if concatenated_timeseries[group] is None:
                    concatenated_timeseries[group] = subject_timeseries[sub][run]
                else:
                    concatenated_timeseries[group] = np.vstack(
                        [concatenated_timeseries[group], subject_timeseries[sub][run]]
                    )

    if standardize:
        for _, group in subject_table.items():
            # Recalculating means and stdev, will cause minor floating point differences so use np.allclose
            concatenated_timeseries[group] -= np.mean(concatenated_timeseries[group], axis=0)
            std[group] = np.std(concatenated_timeseries[group], ddof=1, axis=0)
            eps = np.finfo(std[group].dtype).eps
            # Taken from nilearn pipeline, used for numerical stability purposes to avoid numpy division error
            std[group][std[group] < eps] = 1.0
            concatenated_timeseries[group] /= std[group]

    return concatenated_timeseries


# Similar method to how labels are predicted in CAP.calculate_metrics, same labels are then used to calculate the metrics
def predict_labels(timeseries, cap_analysis, standardize, group, runs=[1, 2, 3]):
    labels = None
    group_dict = cap_analysis.groups[group]
    for sub in timeseries:
        for run in timeseries[sub]:
            if int(run.split("run-")[-1]) in runs and sub in group_dict:
                new_timeseries = copy.deepcopy(timeseries[sub][run])
                if standardize:
                    new_timeseries -= cap_analysis.means[group]
                    new_timeseries /= cap_analysis.stdev[group]
                if labels is None:
                    labels = cap_analysis.kmeans[group].predict(new_timeseries)
                else:
                    labels = np.hstack([labels, cap_analysis.kmeans[group].predict(new_timeseries)])

    return labels


# Get segments
def segments(target, timeseries):
    # Binary representation of numpy array - if [1,2,1,1,1,3] and target is 1, then it is [1,0,1,1,1,0]
    binary_arr = np.where(timeseries == target, 1, 0)
    # Get indices of values that equal 1; [0,2,3,4]
    target_indices = np.where(binary_arr == 1)[0]
    # Count the transitions, indices where diff > 1 is a transition; diff of indices = [2,1,1];
    # binary for diff > 1 = [1,0,0]; thus, segments = transitions + first_sequence(1) = 2
    segments = np.where(np.diff(target_indices, n=1) > 1, 1, 0).sum() + 1

    return binary_arr, segments


@pytest.fixture(autouse=False, scope="module")
def remove_files():
    yield
    png_files = glob.glob((os.path.join(tmp_dir.name, "*.png")))
    csv_files = glob.glob((os.path.join(tmp_dir.name, "*.csv")))
    html_files = glob.glob((os.path.join(tmp_dir.name, "*.html")))
    nii_files = glob.glob((os.path.join(tmp_dir.name, "*.nii.gz")))

    [os.remove(x) for x in png_files]
    [os.remove(x) for x in csv_files]
    [os.remove(x) for x in html_files]
    [os.remove(x) for x in nii_files]


@pytest.mark.parametrize("standardize", [True, False])
def test_no_groups_no_cluster_selection(standardize):
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, n_clusters=2, standardize=standardize)
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)
    assert len(cap_analysis.caps["All Subjects"]) == len(np.unique(cap_analysis.kmeans["All Subjects"].labels_))

    concatenated_timeseries = concat_data(cap_analysis.subject_table, standardize=standardize)

    # Concatenated data used for kmeans
    assert cap_analysis.concatenated_timeseries["All Subjects"].shape == (3000, 100)
    if standardize is False:
        assert np.array_equal(
            cap_analysis.concatenated_timeseries["All Subjects"], concatenated_timeseries["All Subjects"]
        )
    else:
        # Floating point differences
        assert np.allclose(
            cap_analysis.concatenated_timeseries["All Subjects"], concatenated_timeseries["All Subjects"]
        )

    # Validate labels
    labels = predict_labels(subject_timeseries, cap_analysis, standardize, "All Subjects")
    assert np.array_equal(labels, cap_analysis.kmeans["All Subjects"].labels_)
    # Quick check deleter
    del cap_analysis.concatenated_timeseries
    assert not cap_analysis.concatenated_timeseries


def test_skip_subject():
    subject_timeseries = copy.deepcopy(extractor.subject_timeseries)
    del subject_timeseries["2"]["run-1"]
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=subject_timeseries, runs=1, n_clusters=2)

    assert cap_analysis.concatenated_timeseries["All Subjects"].shape == (900, 100)

    df_dict = cap_analysis.calculate_metrics(subject_timeseries=subject_timeseries, runs=1, metrics="counts")
    assert "2" not in df_dict["counts"]["Subject_ID"].values


@pytest.mark.parametrize("standardize", [True, False])
def test_groups_no_cluster_selection(standardize):
    # Should ignore duplicate id
    cap_analysis = CAP(
        parcel_approach=extractor.parcel_approach, groups={"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]}
    )
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, standardize=standardize)
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)

    # Concatenated data used in kmeans
    cap_analysis.concatenated_timeseries["A"].shape == (1200, 100)
    cap_analysis.concatenated_timeseries["A"].shape == (1800, 100)
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

    cap_analysis = CAP(
        parcel_approach=extractor.parcel_approach, groups={"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]}
    )
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, runs=[1, 2], standardize=standardize)
    cap_analysis.concatenated_timeseries["A"].shape == (800, 100)
    cap_analysis.concatenated_timeseries["A"].shape == (1200, 100)
    concatenated_timeseries = concat_data(cap_analysis.subject_table, standardize=standardize, runs=[1, 2])

    if standardize is False:
        assert np.array_equal(cap_analysis.concatenated_timeseries["A"], concatenated_timeseries["A"])
        assert np.array_equal(cap_analysis.concatenated_timeseries["B"], concatenated_timeseries["B"])
    else:
        assert np.allclose(cap_analysis.concatenated_timeseries["A"], concatenated_timeseries["A"])
        assert np.allclose(cap_analysis.concatenated_timeseries["B"], concatenated_timeseries["B"])

    labels = predict_labels(subject_timeseries, cap_analysis, standardize, "A", runs=[1, 2])
    assert np.array_equal(labels, cap_analysis.kmeans["A"].labels_)

    labels = predict_labels(subject_timeseries, cap_analysis, standardize, "B", runs=[1, 2])
    assert np.array_equal(labels, cap_analysis.kmeans["B"].labels_)


@pytest.mark.parametrize("standardize, n_cores", [(False, 2), (True, None)])
def test_no_groups_cluster_selection(standardize, n_cores):
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)

    # Elbow sometimes does find the elbow with random data
    try:
        cap_analysis.get_caps(
            subject_timeseries=extractor.subject_timeseries,
            n_clusters=list(range(2, 41)),
            cluster_selection_method="elbow",
            standardize=standardize,
        )
        try:
            assert all(elem >= 0 for elem in cap_analysis.cluster_scores["Scores"]["All Subjects"].values())
            kneedle = KneeLocator(
                x=list(cap_analysis.cluster_scores["Scores"]["All Subjects"]),
                y=list(cap_analysis.cluster_scores["Scores"]["All Subjects"].values()),
                curve="convex",
                direction="decreasing",
            )
            assert kneedle.elbow == cap_analysis.optimal_n_clusters["All Subjects"]
            assert (
                cap_analysis.cluster_scores["Scores"]["All Subjects"][cap_analysis.optimal_n_clusters["All Subjects"]]
                == cap_analysis.kmeans["All Subjects"].inertia_
            )
            # Slightly redundant assertion
            assert (
                len(cap_analysis.caps["All Subjects"])
                == len(np.unique(cap_analysis.kmeans["All Subjects"].labels_))
                == cap_analysis.optimal_n_clusters["All Subjects"]
            )
        except:
            raise ValueError(
                "Different results for kneedle and optimal cluster or not all inertia values are positive."
            )
    except:
        LG.warning("Elbow could not be found for random data.")
        pass

    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="silhouette",
        standardize=standardize,
        n_cores=n_cores,
    )

    assert cap_analysis.variance_explained["All Subjects"] >= 0 or cap_analysis.variance_explained["All Subjects"] <= 1

    # Maximum silhouette is the most optimal
    assert (
        max(
            cap_analysis.cluster_scores["Scores"]["All Subjects"],
            key=cap_analysis.cluster_scores["Scores"]["All Subjects"].get,
        )
        == cap_analysis.optimal_n_clusters["All Subjects"]
    )

    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)
    assert all(elem > 0 or elem < 0 for elem in cap_analysis.cluster_scores["Scores"]["All Subjects"].values())
    assert all(-1 <= elem <= 1 for elem in cap_analysis.cluster_scores["Scores"]["All Subjects"].values())
    assert (
        len(cap_analysis.caps["All Subjects"])
        == len(np.unique(cap_analysis.kmeans["All Subjects"].labels_))
        == cap_analysis.optimal_n_clusters["All Subjects"]
    )

    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="variance_ratio",
        standardize=standardize,
    )

    # Maximum variance ratio is the most optimal
    assert (
        max(
            cap_analysis.cluster_scores["Scores"]["All Subjects"],
            key=cap_analysis.cluster_scores["Scores"]["All Subjects"].get,
        )
        == cap_analysis.optimal_n_clusters["All Subjects"]
    )
    assert (
        len(cap_analysis.caps["All Subjects"])
        == len(np.unique(cap_analysis.kmeans["All Subjects"].labels_))
        == cap_analysis.optimal_n_clusters["All Subjects"]
    )

    # All values not negative
    assert all(elem >= 0 for elem in cap_analysis.cluster_scores["Scores"]["All Subjects"].values())

    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="davies_bouldin",
        standardize=standardize,
    )

    # Mininimum davies_bouldin is the most optimal
    assert (
        min(
            cap_analysis.cluster_scores["Scores"]["All Subjects"],
            key=cap_analysis.cluster_scores["Scores"]["All Subjects"].get,
        )
        == cap_analysis.optimal_n_clusters["All Subjects"]
    )
    assert (
        len(cap_analysis.caps["All Subjects"])
        == len(np.unique(cap_analysis.kmeans["All Subjects"].labels_))
        == cap_analysis.optimal_n_clusters["All Subjects"]
    )

    # All values not negative
    assert all(elem >= 0 for elem in cap_analysis.cluster_scores["Scores"]["All Subjects"].values())

    assert cap_analysis.variance_explained["All Subjects"] >= 0 or cap_analysis.variance_explained["All Subjects"] <= 1


@pytest.mark.parametrize("standardize", [False, True])
def test_groups_and_cluster_selection(standardize):
    cap_analysis = CAP(
        parcel_approach=extractor.parcel_approach, groups={"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]}
    )
    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="silhouette",
        standardize=standardize,
    )
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)

    # Elbow sometimes does find the elbow with random data, uses kneed to locate elbow
    try:
        cap_analysis = CAP(
            parcel_approach=extractor.parcel_approach, groups={"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]}
        )
        cap_analysis.get_caps(
            subject_timeseries=extractor.subject_timeseries,
            n_clusters=list(range(2, 41)),
            cluster_selection_method="elbow",
            standardize=standardize,
        )
        try:
            assert all(elem >= 0 for elem in cap_analysis.cluster_scores["Scores"]["A"].values())
            assert all(elem >= 0 for elem in cap_analysis.cluster_scores["Scores"]["B"].values())
            kneedle = KneeLocator(
                x=list(cap_analysis.cluster_scores["Scores"]["A"]),
                y=list(cap_analysis.cluster_scores["Scores"]["A"].values()),
                curve="convex",
                direction="decreasing",
            )
            assert kneedle.elbow == cap_analysis.optimal_n_clusters["A"]
            kneedle = KneeLocator(
                x=list(cap_analysis.cluster_scores["Scores"]["B"]),
                y=list(cap_analysis.cluster_scores["Scores"]["B"].values()),
                curve="convex",
                direction="decreasing",
            )
            assert kneedle.elbow == cap_analysis.optimal_n_clusters["B"]
        except:
            raise ValueError(
                "Different results for kneedle and optimal cluster or not all inertia values are positive."
            )
    except:
        LG.warning("Elbow could not be found for random data.")
        pass

    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="silhouette",
        standardize=standardize,
    )
    assert (
        max(cap_analysis.cluster_scores["Scores"]["A"], key=cap_analysis.cluster_scores["Scores"]["A"].get)
        == cap_analysis.optimal_n_clusters["A"]
    )
    assert (
        max(cap_analysis.cluster_scores["Scores"]["B"], key=cap_analysis.cluster_scores["Scores"]["B"].get)
        == cap_analysis.optimal_n_clusters["B"]
    )

    assert cap_analysis.variance_explained["A"] >= 0 or cap_analysis.variance_explained["A"] <= 1
    assert cap_analysis.variance_explained["B"] >= 0 or cap_analysis.variance_explained["B"] <= 1

    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="davies_bouldin",
        standardize=standardize,
    )
    assert (
        min(cap_analysis.cluster_scores["Scores"]["A"], key=cap_analysis.cluster_scores["Scores"]["A"].get)
        == cap_analysis.optimal_n_clusters["A"]
    )
    assert (
        min(cap_analysis.cluster_scores["Scores"]["B"], key=cap_analysis.cluster_scores["Scores"]["B"].get)
        == cap_analysis.optimal_n_clusters["B"]
    )

    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="variance_ratio",
        standardize=standardize,
    )
    assert (
        max(cap_analysis.cluster_scores["Scores"]["A"], key=cap_analysis.cluster_scores["Scores"]["A"].get)
        == cap_analysis.optimal_n_clusters["A"]
    )
    assert (
        max(cap_analysis.cluster_scores["Scores"]["B"], key=cap_analysis.cluster_scores["Scores"]["B"].get)
        == cap_analysis.optimal_n_clusters["B"]
    )


def test_var_explained():
    timeseries = {str(x): {f"run-{y}": np.random.rand(10, 116) for y in range(1)} for x in range(1)}
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=10)
    assert cap_analysis.variance_explained["All Subjects"] == 1


def test_no_groups_pkl():
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(
        subject_timeseries=os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"), n_clusters=2
    )
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)


def test_groups_pkl():
    cap_analysis = CAP(
        parcel_approach=extractor.parcel_approach, groups={"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]}
    )
    cap_analysis.get_caps(
        subject_timeseries=os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"), n_clusters=2
    )
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)


def test_no_groups_and_silhouette_method():
    extractor.subject_timeseries = subject_timeseries
    cap_analysis = CAP(parcel_approach=extractor.parcel_approach)
    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries, n_clusters=[2, 3, 4, 5], cluster_selection_method="silhouette"
    )
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)
    assert all(elem > 0 or elem < 0 for elem in cap_analysis.cluster_scores["Scores"]["All Subjects"].values())
    assert all(-1 <= elem <= 1 for elem in cap_analysis.cluster_scores["Scores"]["All Subjects"].values())


def test_groups_and_silhouette_method():
    cap_analysis = CAP(
        parcel_approach=extractor.parcel_approach, groups={"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]}
    )
    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries, n_clusters=[2, 3, 4, 5], cluster_selection_method="silhouette"
    )
    assert all(elem > 0 or elem < 0 for elem in cap_analysis.cluster_scores["Scores"]["A"].values())
    assert all(-1 <= elem <= 1 for elem in cap_analysis.cluster_scores["Scores"]["A"].values())
    assert all(elem > 0 or elem < 0 for elem in cap_analysis.cluster_scores["Scores"]["B"].values())
    assert all(-1 <= elem <= 1 for elem in cap_analysis.cluster_scores["Scores"]["B"].values())
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)


def test_calculate_metrics():
    cap_analysis = CAP(
        parcel_approach=extractor.parcel_approach, groups={"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]}
    )
    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries, n_clusters=[2, 3, 4, 5], cluster_selection_method="silhouette"
    )
    assert all(elem > 0 or elem < 0 for elem in cap_analysis.cluster_scores["Scores"]["A"].values())
    assert all(elem > 0 or elem < 0 for elem in cap_analysis.cluster_scores["Scores"]["A"].values())

    df_dict = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True)
    assert len(df_dict) == 4

    for i in range(1, 4):
        met1 = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True, runs=i)
        met2 = cap_analysis.calculate_metrics(
            subject_timeseries=extractor.subject_timeseries, return_df=True, runs=i, continuous_runs=True
        )
        # If only one run continuous_runs should not differ
        assert met1["persistence"].equals(met2["persistence"])

    met1 = cap_analysis.calculate_metrics(subject_timeseries=extractor.subject_timeseries, return_df=True)
    met2 = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries, return_df=True, continuous_runs=True
    )
    # Should differ
    assert not met1["persistence"].equals(met2["persistence"])
    # Continuous run should have 1/3 the number of rows since each subject in the randomized data has three runs
    assert met1["persistence"].shape[0] / 3 == met2["persistence"].shape[0]

    # Based on the equation in the supplementary of Yang et al 2021; temporal fraction = (persistence*counts)/total
    cap_analysis.get_caps(
        subject_timeseries=extractor.subject_timeseries, n_clusters=[2, 3, 4, 5], cluster_selection_method="silhouette"
    )
    counts = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries, return_df=True, metrics="counts"
    )["counts"]
    persistence = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries, return_df=True, metrics="persistence"
    )["persistence"]
    temp = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries, return_df=True, metrics="temporal_fraction"
    )["temporal_fraction"]

    for cap in ["CAP-1", "CAP-2"]:
        for i in temp.index:
            assert math.isclose((counts.loc[i, cap] * persistence.loc[i, cap]) / 100, temp.loc[i, cap], abs_tol=0.01)

    # Check for continuous too
    counts = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries, return_df=True, metrics="counts", continuous_runs=True
    )["counts"]
    persistence = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries, return_df=True, metrics="persistence", continuous_runs=True
    )["persistence"]
    temp = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries,
        return_df=True,
        metrics="temporal_fraction",
        continuous_runs=True,
    )["temporal_fraction"]
    for cap in ["CAP-1", "CAP-2"]:
        for i in temp.index:
            assert math.isclose((counts.loc[i, cap] * persistence.loc[i, cap]) / 300, temp.loc[i, cap], abs_tol=0.01)

    # Check values of metrics; new methods
    counts_df = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries, return_df=True, metrics="counts", continuous_runs=False
    )["counts"]
    counts_df = counts_df[[x for x in counts_df.columns if x.startswith("CAP")]]

    temp_df = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries,
        return_df=True,
        metrics="temporal_fraction",
        continuous_runs=False,
    )["temporal_fraction"]
    temp_df = temp_df[[x for x in temp_df.columns if x.startswith("CAP")]]

    persistence_df = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries, return_df=True, metrics="persistence", continuous_runs=False
    )["persistence"]
    persistence_df = persistence_df[[x for x in persistence_df.columns if x.startswith("CAP")]]

    transition_frequency_df = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries,
        return_df=True,
        metrics="transition_frequency",
        continuous_runs=False,
    )["transition_frequency"]

    transition_probability_df = cap_analysis.calculate_metrics(
        subject_timeseries=extractor.subject_timeseries,
        return_df=True,
        metrics="transition_probability",
        continuous_runs=False,
    )["transition_probability"]
    # Get first subject
    first_subject_timeseries = {}
    first_subject_timeseries.update({"1": subject_timeseries["1"]})
    first_subject_labels = (
        predict_labels(first_subject_timeseries, cap_analysis, standardize=True, group="A", runs=[1]) + 1
    )
    n_caps = cap_analysis.optimal_n_clusters["A"]

    # Counts
    counts_dict = {}
    for target in range(1, n_caps + 1):
        _, counts = segments(target, first_subject_labels)
        counts_dict.update({target: counts})

    assert [x for x in list(counts_dict.values()) if not math.isnan(x)] == [
        x for x in counts_df.loc[0, :].values if not math.isnan(x)
    ]

    # Temporal Fraction
    sorted_frequency_dict = {num: np.where(first_subject_labels == num, 1, 0).sum() for num in range(1, n_caps + 1)}
    proportion_dict = {num: value / len(first_subject_labels) for num, value in sorted_frequency_dict.items()}
    assert [x for x in list(proportion_dict.values()) if not math.isnan(x)] == [
        x for x in temp_df.loc[0, :].values if not math.isnan(x)
    ]

    # Transition Frequency
    transition_frequency = np.where(np.diff(first_subject_labels, n=1) != 0, 1, 0).sum()
    assert transition_frequency_df.iloc[0, 3] == transition_frequency

    # Persistence
    tr = None
    persistence_dict = {}
    for target in range(1, n_caps + 1):
        binary, counts = segments(target, first_subject_labels)
        persistence_dict.update({target: (binary.sum() / counts) * (tr if tr else 1)})
    assert [x for x in list(persistence_dict.values()) if not math.isnan(x)] == [
        x for x in persistence_df.loc[0, :].values if not math.isnan(x)
    ]

    # Check that all transition probabilities sum to 1
    for group in cap_analysis.groups:
        df = transition_probability_df[group]
        for i in df.index:
            for cap in cap_analysis.caps[group]:
                target_cap = cap.split("-")[-1]
                columns = df.filter(regex=rf"^{target_cap}\.").columns.tolist()
                assert (
                    math.isclose(df.loc[i, columns].values.sum(), 1, rel_tol=0.01)
                    or df.loc[i, columns].values.sum() == 0
                )


def test_calculate_metrics_w_pickle():
    cap_analysis = CAP(parcel_approach=parcel_approach, groups={"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, n_clusters=2)
    metrics = ["temporal_fraction", "counts", "transition_frequency", "persistence", "transition_probability"]
    cap_analysis.calculate_metrics(
        subject_timeseries=os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
        metrics=metrics,
        output_dir=tmp_dir.name,
    )
    csv_files = glob.glob(os.path.join(tmp_dir.name, "*.csv"))
    assert len(csv_files) == 6
    expected_files = [f"{metric}.csv" for metric in metrics[:-1]]
    expected_files += ["transition_probability-A.csv", "transition_probability-B.csv"]
    assert sorted(expected_files) == sorted([os.path.basename(x) for x in csv_files])
    assert all(os.path.getsize(file) > 0 for file in csv_files)
    [os.remove(x) for x in csv_files]


def check_imgs(values_dict, plot_type="map"):
    if plot_type == "map":
        heatmap_files = glob.glob(os.path.join(tmp_dir.name, "*heatmap*.png"))
        assert any(["nodes" in x for x in heatmap_files])
        assert any(["regions" in x for x in heatmap_files])
        outer_files = glob.glob(os.path.join(tmp_dir.name, "*outer*.png"))
        assert any(["nodes" in x for x in outer_files])
        assert any(["regions" in x for x in outer_files])

        assert len(heatmap_files) == values_dict["heatmap"] and len(outer_files) == values_dict["outer"]
        [os.remove(file) for file in heatmap_files + outer_files]
    elif plot_type == "radar":
        if "html" in values_dict:
            radar_html = glob.glob(os.path.join(tmp_dir.name, "*radar*.html"))
            assert len(radar_html) == values_dict["html"]
            [os.remove(file) for file in radar_html]
        else:
            radar_png = glob.glob(os.path.join(tmp_dir.name, "*radar*.png"))
            assert len(radar_png) == values_dict["png"]
            [os.remove(file) for file in radar_png]
    elif plot_type == "nifti":
        nii_files = glob.glob(os.path.join(tmp_dir.name, "*.nii.gz"))
        assert len(nii_files) == values_dict["nii.gz"]
        [os.remove(file) for file in nii_files]
    else:
        surface_png = glob.glob(os.path.join(tmp_dir.name, "*surface*.png"))
        assert len(surface_png) == values_dict["png"]
        [os.remove(file) for file in surface_png]


@pytest.mark.parametrize(
    "current_timeseries, parcel_approach",
    [(extractor.subject_timeseries, extractor.parcel_approach), (custom_subject_timeseries, custom_parcel_approach)],
)
def test_plotting_functions(current_timeseries, parcel_approach):
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(
        subject_timeseries=current_timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="silhouette",
        output_dir=tmp_dir.name,
        step=2,
        show_figs=False,
    )
    assert all(elem > 0 or elem < 0 for elem in cap_analysis.cluster_scores["Scores"]["All Subjects"].values())

    files = glob.glob(os.path.join(tmp_dir.name, "*.png"))
    assert files
    [os.remove(file) for file in files]

    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=current_timeseries, n_clusters=2)

    # Plotting Functions with different configurations
    cap_analysis.caps2plot(
        subplots=True, xlabel_rotation=90, sharey=True, borderwidths=10, show_figs=False, output_dir=tmp_dir.name
    )

    files = glob.glob(os.path.join(tmp_dir.name, "*.png"))
    assert len(files) == 1
    [os.remove(file) for file in files]

    cap_analysis.caps2plot(
        subplots=False,
        yticklabels_size=5,
        wspace=0.1,
        visual_scope=["regions", "nodes"],
        xlabel_rotation=90,
        xticklabels_size=5,
        hspace=0.6,
        tight_layout=False,
        suffix_filename="suffix_name",
        show_figs=False,
        plot_options=["heatmap", "outer_product"],
        hemisphere_labels=False,
        output_dir=tmp_dir.name,
    )
    check_imgs(values_dict={"heatmap": 2, "outer": 4})

    cap_analysis.caps2plot(
        subplots=False,
        yticklabels_size=5,
        wspace=0.1,
        visual_scope=["regions", "nodes"],
        xlabel_rotation=90,
        xticklabels_size=5,
        hspace=0.6,
        tight_layout=False,
        show_figs=False,
        plot_options=["heatmap", "outer_product"],
        hemisphere_labels=True,
        invalid_kwarg=0,
        cbarlabels_size=8,
        output_dir=tmp_dir.name,
    )
    check_imgs(values_dict={"heatmap": 2, "outer": 4})

    cap_analysis.caps2plot(
        subplots=True,
        xlabel_rotation=90,
        sharey=True,
        borderwidths=10,
        show_figs=False,
        visual_scope=["regions", "nodes"],
        plot_options=["outer_product", "heatmap"],
        output_dir=tmp_dir.name,
    )
    check_imgs(values_dict={"heatmap": 2, "outer": 2})

    cap_analysis.caps2plot(
        subplots=True,
        xlabel_rotation=90,
        sharey=True,
        borderwidths=10,
        show_figs=False,
        visual_scope=["regions", "nodes"],
        plot_options=["outer_product", "heatmap"],
        hemisphere_labels=True,
        output_dir=tmp_dir.name,
    )
    check_imgs(values_dict={"heatmap": 2, "outer": 2})

    df = cap_analysis.caps2corr(
        annot=True,
        show_figs=False,
        return_df=True,
        suffix_filename="suffix_name_corr",
        output_dir=tmp_dir.name,
        save_df=True,
        cbarlabels_size=8,
    )
    assert isinstance(df, dict)
    assert isinstance(df["All Subjects"], pd.DataFrame)
    assert len(list(df)) == 1

    png_file = glob.glob(os.path.join(tmp_dir.name, "*correlation_matrix*.png"))
    csv_file = glob.glob(os.path.join(tmp_dir.name, "*correlation_matrix*.csv"))
    assert png_file
    assert csv_file
    assert os.path.getsize(csv_file[0]) > 0
    [os.remove(file) for file in png_file + csv_file]

    radialaxis = {
        "showline": True,
        "linewidth": 2,
        "linecolor": "rgba(0, 0, 0, 0.25)",
        "gridcolor": "rgba(0, 0, 0, 0.25)",
        "ticks": "outside",
        "tickfont": {"size": 14, "color": "black"},
        "range": [0, 0.3],
        "tickvals": [0.1, 0.2, 0.3],
    }

    # Radar plotting functions
    cap_analysis.caps2radar(output_dir=tmp_dir.name, show_figs=False, suffix_filename="suffix_name")
    check_imgs(plot_type="radar", values_dict={"png": 2})
    cap_analysis.caps2radar(
        radialaxis=radialaxis, fill="toself", show_figs=False, as_html=True, output_dir=tmp_dir.name
    )
    check_imgs(plot_type="radar", values_dict={"html": 2})

    cap_analysis.caps2radar(
        radialaxis=radialaxis, fill="toself", show_figs=False, as_html=False, output_dir=tmp_dir.name
    )
    check_imgs(plot_type="radar", values_dict={"png": 2})

    cap_analysis.caps2radar(
        radialaxis=radialaxis,
        fill="toself",
        use_scatterpolar=True,
        scattersize=10,
        show_figs=False,
        as_html=True,
        output_dir=tmp_dir.name,
    )
    check_imgs(plot_type="radar", values_dict={"html": 2})


@pytest.mark.skipif(sys.platform != "linux", reason="VTK action only works for Linux")
def test_caps2surf(remove_files):
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=subject_timeseries, n_clusters=2)

    cap_analysis.caps2surf(
        method="nearest",
        fwhm=1,
        save_stat_maps=True,
        suffix_filename="suffix_name",
        output_dir=tmp_dir.name,
        suffix_title="placeholder",
        show_figs=False,
    )
    check_imgs(plot_type="surface", values_dict={"png": 2})
    check_imgs(plot_type="nifti", values_dict={"nii.gz": 2})

    cap_analysis.caps2surf(
        method="linear", fwhm=1, save_stat_maps=False, output_dir=tmp_dir.name, as_outline=True, show_figs=False
    )
    check_imgs(plot_type="surface", values_dict={"png": 2})
    check_imgs(plot_type="nifti", values_dict={"nii.gz": 0})


@pytest.mark.parametrize(
    "current_timeseries, parcel_approach",
    [
        (extractor.subject_timeseries, extractor.parcel_approach),
        (aal_subject_timeseries, extractor2.parcel_approach),
        (custom_subject_timeseries, custom_parcel_approach),
    ],
)
def test_niftis(current_timeseries, parcel_approach):
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=current_timeseries, n_clusters=2)

    atlas_data = nib.load(parcel_approach[list(parcel_approach)[0]]["maps"]).get_fdata()
    labels = sorted(np.unique(atlas_data))[1:]

    cap_analysis.caps2niftis(output_dir=tmp_dir.name, suffix_filename="suffix_name")
    nifti_files = glob.glob(os.path.join(tmp_dir.name, "*.nii.gz"))

    # Check that elements of the cluster centroid are correctly assigned to their corresponding labels in atlas
    for indx, file in enumerate(nifti_files, start=1):
        act_values = []
        nifti_img = nib.load(file).get_fdata()
        for label in labels:
            # Extract first coordinate and append to activation list
            coords = list(zip(*np.where(atlas_data == label)))[0]
            act_value = nifti_img[coords[0], coords[1], coords[2]]
            act_values.append(act_value)

        # Assess if reconstructing 1D array from 3D nifti produces the same cluster centroid
        np.array_equal(cap_analysis.caps["All Subjects"][f"CAP-{indx}"], np.array(act_values))

    # Check files
    check_imgs(plot_type="nifti", values_dict={"nii.gz": 2})

    if "Custom" in parcel_approach:
        # Assess that knn interpolation works
        original_nifti = nib.load(cap_analysis.parcel_approach["Custom"]["maps"]).get_fdata()

        # Replace zeros in CAP 1D vectors to conduct assessment
        for i in range(1, 3):
            cap_analysis.caps["All Subjects"][f"CAP-{i}"] = np.where(
                cap_analysis.caps["All Subjects"][f"CAP-{i}"] == 0, 1, cap_analysis.caps["All Subjects"][f"CAP-{i}"]
            )
            # Assert no zeroes
            assert (
                sum(cap_analysis.caps["All Subjects"][f"CAP-{i}"][cap_analysis.caps["All Subjects"][f"CAP-{i}"] == 0])
                == 0
            )

        # Check knn interpolation with schaefer reference
        cap_analysis.caps2niftis(
            output_dir=tmp_dir.name,
            suffix_filename="Schaefer_ref",
            knn_dict={"k": 1, "resolution_mm": 1, "remove_labels": [50]},
        )

        # Check interpolation using Schaefer reference
        for i in glob.glob(os.path.join(tmp_dir.name, "*Schaefer_ref*")):
            interpolated_nifti = nib.load(i).get_fdata()
            assert not np.array_equal(original_nifti[original_nifti == 0], interpolated_nifti[original_nifti == 0])

        # Check files
        check_imgs(plot_type="nifti", values_dict={"nii.gz": 2})

        cap_analysis.caps2niftis(
            output_dir=tmp_dir.name, suffix_filename="AAL_ref", knn_dict={"k": 3, "reference_atlas": "AAL"}
        )

        # Check interpolation using AAL reference
        for i in glob.glob(os.path.join(tmp_dir.name, "*AAL_ref*")):
            interpolated_nifti = nib.load(i).get_fdata()
            assert not np.array_equal(original_nifti[original_nifti == 0], interpolated_nifti[original_nifti == 0])

        # Check files
        check_imgs(plot_type="nifti", values_dict={"nii.gz": 2})


def test_calculate_metrics_w_change_dtype():
    cap_analysis = CAP(parcel_approach=parcel_approach, groups={"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]})
    cap_analysis.get_caps(subject_timeseries=extractor.subject_timeseries, n_clusters=2)
    new_timeseries = change_dtype([extractor.subject_timeseries], dtype="float16")
    cap_analysis.calculate_metrics(
        subject_timeseries=new_timeseries["dict_0"],
        return_df=True,
        prefix_filename="prefixname",
        output_dir=tmp_dir.name,
    )
    assert glob.glob(os.path.join(tmp_dir.name, "*prefixname*"))
