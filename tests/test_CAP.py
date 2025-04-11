import copy, glob, math, os, re, sys

import nibabel as nib, numpy as np, pandas as pd, pytest

from kneed import KneeLocator

from neurocaps.analysis import CAP
from .utils import Parcellation, check_imgs, concat_data, get_first_subject, segments, segments_mirrored, predict_labels


@pytest.fixture(autouse=False, scope="module")
def remove_files(tmp_dir):
    """Cleans files in temporary directory."""
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
def test_without_groups_and_without_cluster_selection(standardize):
    """
    Test base use case of `CAP.get_caps()` without groups or a specific method for selecting optimal clusters.
    Assesses shape and order of the concatenated data for correctness and that the deleter property works. Also
    validates that the subject-level predicted labels (using and not using standardization) matches labels produced by
    kmeans.labels_ (which is done in `calculate_metrics`). Also checks that the deleter property is functional and the
    __str__ dunder method works.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP()

    # No error; Testing __str__
    print(cap_analysis)

    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=2, standardize=standardize)
    assert cap_analysis.standardize == standardize
    assert cap_analysis.n_clusters == 2
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)
    assert len(cap_analysis.caps["All Subjects"]) == len(np.unique(cap_analysis.kmeans["All Subjects"].labels_))

    # No error; Testing __str__
    print(cap_analysis)

    # All subjects in subject table
    assert all(i in timeseries for i in cap_analysis.subject_table)

    concatenated_timeseries = concat_data(timeseries, cap_analysis.subject_table, standardize=standardize)

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
    labels = predict_labels(timeseries, cap_analysis, standardize, "All Subjects")
    assert np.array_equal(labels, cap_analysis.kmeans["All Subjects"].labels_)

    # Quick check deleter
    del cap_analysis.concatenated_timeseries
    assert not cap_analysis.concatenated_timeseries


def test_subject_skipping():
    """
    Tests that a subject is properly skipped in both `get_caps` and `calculate_metrics` if they do not have any
    requested runs.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    subject_timeseries = copy.deepcopy(timeseries)
    del subject_timeseries["2"]["run-1"]
    cap_analysis = CAP()
    cap_analysis.get_caps(subject_timeseries=subject_timeseries, runs=1, n_clusters=2)

    assert cap_analysis.runs == [1]
    assert cap_analysis.concatenated_timeseries["All Subjects"].shape == (900, 100)

    df_dict = cap_analysis.calculate_metrics(subject_timeseries=subject_timeseries, runs=1, metrics="counts")
    assert "2" not in df_dict["counts"]["Subject_ID"].values


@pytest.mark.parametrize("standardize, runs", ([True, [1, 2]], [False, ["run-1", "run-2"]]))
def test_groups_without_cluster_selection(standardize, runs):
    """
    Tests case when the `group` parameter is used. Ensures duplicate IDs are ignored, the `subject_table`
    property which maps IDs to groups is properly constructed, the expected shape and order of the concatenated data
    is correct for both groups, and verifies subject-level prediction matches labels produced by kmeans.labels_ (
    which is done in `calculate_metrics`). Checks when run is specified and not specified.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    # Should ignore duplicate id
    groups = {"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]}
    cap_analysis = CAP(groups=groups)

    cap_analysis.get_caps(subject_timeseries=timeseries, standardize=standardize)

    # Subject table created correctly
    assert all(k in groups.get(v) for k, v in cap_analysis.subject_table.items())

    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)

    # Concatenated data used in kmeans
    cap_analysis.concatenated_timeseries["A"].shape == (1200, 100)
    cap_analysis.concatenated_timeseries["A"].shape == (1800, 100)
    concatenated_timeseries = concat_data(timeseries, cap_analysis.subject_table, standardize=standardize)

    if standardize is False:
        assert np.array_equal(cap_analysis.concatenated_timeseries["A"], concatenated_timeseries["A"])
        assert np.array_equal(cap_analysis.concatenated_timeseries["B"], concatenated_timeseries["B"])
    else:
        assert np.allclose(cap_analysis.concatenated_timeseries["A"], concatenated_timeseries["A"], atol=0.00001)
        assert np.allclose(cap_analysis.concatenated_timeseries["B"], concatenated_timeseries["B"], atol=0.00001)

    # Demonstrates that label assignment using `.predict` method and paired with proper subject level standardization
    # using the mean and std dev computed from the concatenated data produces the same labels stored in `.labels_`
    labels = predict_labels(timeseries, cap_analysis, standardize, "A")
    assert np.array_equal(labels, cap_analysis.kmeans["A"].labels_)

    labels = predict_labels(timeseries, cap_analysis, standardize, "B")
    assert np.array_equal(labels, cap_analysis.kmeans["B"].labels_)

    cap_analysis = CAP(groups={"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]})
    cap_analysis.get_caps(subject_timeseries=timeseries, runs=runs, standardize=standardize)
    cap_analysis.concatenated_timeseries["A"].shape == (800, 100)
    cap_analysis.concatenated_timeseries["A"].shape == (1200, 100)
    run_nums = [int(y) for y in [str(x).removeprefix("run-") for x in runs]]
    concatenated_timeseries = concat_data(
        timeseries, cap_analysis.subject_table, standardize=standardize, runs=run_nums
    )

    if standardize is False:
        assert np.array_equal(cap_analysis.concatenated_timeseries["A"], concatenated_timeseries["A"])
        assert np.array_equal(cap_analysis.concatenated_timeseries["B"], concatenated_timeseries["B"])
    else:
        assert np.allclose(cap_analysis.concatenated_timeseries["A"], concatenated_timeseries["A"], atol=0.00001)
        assert np.allclose(cap_analysis.concatenated_timeseries["B"], concatenated_timeseries["B"], atol=0.00001)

    labels = predict_labels(timeseries, cap_analysis, standardize, "A", runs=run_nums)
    assert np.array_equal(labels, cap_analysis.kmeans["A"].labels_)

    labels = predict_labels(timeseries, cap_analysis, standardize, "B", runs=run_nums)
    assert np.array_equal(labels, cap_analysis.kmeans["B"].labels_)


def test_no_mutability():
    """
    Ensure no mutability when only single timeseries data.
    """
    subject_timeseries = {str(x): {f"run-{y}": np.random.rand(50, 100) for y in range(1)} for x in range(1)}

    original_timeseries = copy.deepcopy(subject_timeseries)

    cap_analysis = CAP()

    cap_analysis.get_caps(subject_timeseries=subject_timeseries, standardize=True)

    assert np.array_equal(subject_timeseries["0"]["run-0"], original_timeseries["0"]["run-0"])

    cap_analysis.calculate_metrics(subject_timeseries=subject_timeseries)

    assert np.array_equal(subject_timeseries["0"]["run-0"], original_timeseries["0"]["run-0"])


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(
    "groups, n_cores", [(["All Subjects"], None), ({"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]}, 2)]
)
def test_elbow_method(groups, n_cores):
    """
    Tests the elbow method and that related information and the optimal cluster size is stored in properties. Sometimes
    elbow is not found for randomly generated data. Reruns if there is a failure as opposed to using a seed.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups=None if isinstance(groups, list) else groups)

    cap_analysis.get_caps(
        subject_timeseries=timeseries,
        n_clusters=list(range(2, 41)),
        cluster_selection_method="elbow",
        n_cores=n_cores,
    )

    for group in groups:
        assert all(elem >= 0 for elem in cap_analysis.cluster_scores["Scores"][group].values())

        kneedle = KneeLocator(
            x=list(cap_analysis.cluster_scores["Scores"][group]),
            y=list(cap_analysis.cluster_scores["Scores"][group].values()),
            curve="convex",
            direction="decreasing",
        )

        assert kneedle.elbow == cap_analysis.optimal_n_clusters[group]
        assert cap_analysis.cluster_selection_method == "elbow"
        assert cap_analysis.cluster_scores["Cluster_Selection_Method"] == "elbow"
        assert (
            cap_analysis.cluster_scores["Scores"][group][cap_analysis.optimal_n_clusters[group]]
            == cap_analysis.kmeans[group].inertia_
        )

        # Slightly redundant assertion
        assert (
            len(cap_analysis.caps[group])
            == len(np.unique(cap_analysis.kmeans[group].labels_))
            == cap_analysis.optimal_n_clusters[group]
        )


@pytest.mark.parametrize(
    "groups, n_cores", [(["All Subjects"], None), ({"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]}, 2)]
)
def test_silhouette_method(groups, n_cores):
    """
    Tests the silhouette method and that related information and the optimal cluster size, based on maximum value, is
    stored in properties.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups=None if isinstance(groups, list) else groups)
    cap_analysis.get_caps(
        subject_timeseries=timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="silhouette",
        n_cores=n_cores,
    )

    for group in groups:
        assert all(elem > 0 or elem < 0 for elem in cap_analysis.cluster_scores["Scores"][group].values())
        assert all(-1 <= elem <= 1 for elem in cap_analysis.cluster_scores["Scores"][group].values())
        assert cap_analysis.cluster_selection_method == "silhouette"
        assert cap_analysis.cluster_scores["Cluster_Selection_Method"] == "silhouette"
        # Maximum value most optimal
        assert (
            max(cap_analysis.cluster_scores["Scores"][group], key=cap_analysis.cluster_scores["Scores"][group].get)
            == cap_analysis.optimal_n_clusters[group]
        )
        assert (
            len(cap_analysis.caps[group])
            == len(np.unique(cap_analysis.kmeans[group].labels_))
            == cap_analysis.optimal_n_clusters[group]
        )


@pytest.mark.parametrize(
    "groups, n_cores", [(["All Subjects"], None), ({"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]}, 2)]
)
def test_davies_bouldin_method(groups, n_cores):
    """
    Tests the Davies Bouldin method and that related information and the optimal cluster size, based on minimum
    value, is stored in properties.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups=None if isinstance(groups, list) else groups)
    cap_analysis.get_caps(
        subject_timeseries=timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="davies_bouldin",
        n_cores=n_cores,
    )

    assert cap_analysis.n_cores == n_cores

    for group in groups:
        assert cap_analysis.cluster_selection_method == "davies_bouldin"
        assert cap_analysis.cluster_scores["Cluster_Selection_Method"] == "davies_bouldin"
        # Minimum value most optimal
        assert (
            min(cap_analysis.cluster_scores["Scores"][group], key=cap_analysis.cluster_scores["Scores"][group].get)
            == cap_analysis.optimal_n_clusters[group]
        )
        assert (
            len(cap_analysis.caps[group])
            == len(np.unique(cap_analysis.kmeans[group].labels_))
            == cap_analysis.optimal_n_clusters[group]
        )
        # All values not negative
        assert all(elem >= 0 for elem in cap_analysis.cluster_scores["Scores"][group].values())


@pytest.mark.parametrize(
    "groups, n_cores", [(["All Subjects"], None), ({"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]}, 2)]
)
def test_variance_ratio_method(groups, n_cores):
    """
    Tests the variance ratio method and that related information and the optimal cluster size, based on
    maximum value, is stored in properties.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups=None if isinstance(groups, list) else groups)
    cap_analysis.get_caps(
        subject_timeseries=timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="variance_ratio",
        n_cores=n_cores,
    )

    for group in groups:
        assert cap_analysis.cluster_selection_method == "variance_ratio"
        assert cap_analysis.cluster_scores["Cluster_Selection_Method"] == "variance_ratio"
        # Maximum value most optimal
        assert (
            max(cap_analysis.cluster_scores["Scores"][group], key=cap_analysis.cluster_scores["Scores"][group].get)
            == cap_analysis.optimal_n_clusters[group]
        )
        assert (
            len(cap_analysis.caps[group])
            == len(np.unique(cap_analysis.kmeans[group].labels_))
            == cap_analysis.optimal_n_clusters[group]
        )
        # All values not negative
        assert all(elem >= 0 for elem in cap_analysis.cluster_scores["Scores"][group].values())


@pytest.mark.parametrize("method", ["silhouette", "davies_bouldin", "variance_ratio"])
def test_methods_parallel_and_sequential_equivalence(method):
    """
    Tests that the same scores for each method are produced wheter using sequential or parallel processing.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP()
    cap_analysis.get_caps(
        subject_timeseries=timeseries,
        n_clusters=[2, 3, 4, 5],
        random_state=0,
        cluster_selection_method=method,
    )

    sequential = np.array(list(cap_analysis.cluster_scores["Scores"]["All Subjects"].values()))

    cap_analysis.get_caps(
        subject_timeseries=timeseries,
        n_clusters=[2, 3, 4, 5],
        random_state=0,
        cluster_selection_method=method,
        n_cores=2,
    )

    parallel = np.array(list(cap_analysis.cluster_scores["Scores"]["All Subjects"].values()))

    assert np.allclose(sequential, parallel, atol=0.0001)


def test_var_explained():
    """
    Tests variance explained by ensuring that variance equals 1 when all observations are in their own clusters,
    resulting in an inertia of 0.
    """
    timeseries = {str(x): {f"run-{y}": np.random.rand(10, 116) for y in range(1)} for x in range(1)}

    cap_analysis = CAP()
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=10)
    assert cap_analysis.variance_explained["All Subjects"] == 1


def test_no_groups_using_pickle():
    """
    Verifies that pickles can be used as input in `get_caps`.
    """
    cap_analysis = CAP()
    cap_analysis.get_caps(
        subject_timeseries=os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"), n_clusters=2
    )
    assert cap_analysis.caps["All Subjects"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["All Subjects"]["CAP-2"].shape == (100,)


def test_groups_using_pickle():
    """
    Verifies that pickles can be used as input in `get_caps`.
    """
    cap_analysis = CAP(groups={"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]})
    cap_analysis.get_caps(
        subject_timeseries=os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"), n_clusters=2
    )
    assert cap_analysis.caps["A"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["A"]["CAP-2"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-1"].shape == (100,)
    assert cap_analysis.caps["B"]["CAP-2"].shape == (100,)


def test_temporal_fraction():
    """
    Verifies temporal fraction is computed correctly.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups={"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]})
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=3)

    # Should ignore bad metric
    df_dict = cap_analysis.calculate_metrics(
        subject_timeseries=timeseries, return_df=True, metrics=["temporal_fraction", "incorrect"]
    )
    assert len(df_dict) == 1

    df = df_dict["temporal_fraction"]
    df = df[[x for x in df.columns if x.startswith("CAP")]]
    assert all(df.apply(lambda x: all(x.values <= 1) and all(x.values >= 0)).values)

    # Get first subject
    first_subject_labels = get_first_subject(timeseries, cap_analysis)

    sorted_frequency_dict = {num: np.where(first_subject_labels == num, 1, 0).sum() for num in range(1, 4)}
    proportion_dict = {num: value / len(first_subject_labels) for num, value in sorted_frequency_dict.items()}
    assert [x for x in list(proportion_dict.values()) if not math.isnan(x)] == [
        x for x in df.loc[0, :].values if not math.isnan(x)
    ]
    # Temporal fractions should sum to 1
    assert math.isclose(np.array(list(proportion_dict.values())).sum(), 1, abs_tol=0.0001)


def test_counts():
    """
    Verifies counts is computed correctly.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups={"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]})
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=3)

    df_dict = cap_analysis.calculate_metrics(subject_timeseries=timeseries, return_df=True, metrics=["counts"])

    df = df_dict["counts"]
    df = df[[x for x in df.columns if x.startswith("CAP")]]
    all(df.apply(lambda x: x.values.dtype == "int64" and all(x.values >= 0)).values)

    # Get first subject
    first_subject_labels = get_first_subject(timeseries, cap_analysis)

    counts_reimplemented_dict = {}
    for target in range(1, 4):
        _, counts = segments(target, first_subject_labels)
        counts_reimplemented_dict.update({target: counts})

    counts_dict = {}
    for target in range(1, 4):
        _, counts = segments_mirrored(target, first_subject_labels)
        counts = counts if target in first_subject_labels else 0
        counts_dict.update({target: counts})

    assert all([counts_reimplemented_dict[target] == counts_dict[target] for target in range(1, 4)])

    assert [x for x in list(counts_dict.values()) if not math.isnan(x)] == [
        x for x in df.loc[0, :].values if not math.isnan(x)
    ]


@pytest.mark.parametrize("tr", [None, 2])
def test_persistence(tr):
    """
    Verifies persistence is computed correctly.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups={"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]})
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=3)

    df_dict = cap_analysis.calculate_metrics(
        subject_timeseries=timeseries, return_df=True, metrics=["persistence"], tr=tr
    )
    assert len(df_dict) == 1

    df = df_dict["persistence"]
    df = df[[x for x in df.columns if x.startswith("CAP")]]
    assert all(df.apply(lambda x: all(x.values >= 0)).values)

    # Get first subject
    first_subject_labels = get_first_subject(timeseries, cap_analysis)

    tr = tr

    persistence_reimplemented_dict = {}
    for target in range(1, 4):
        seg_list, counts = segments(target, first_subject_labels)
        val = sum(seg_list) / len(seg_list) if counts != 0 else 0
        persistence_reimplemented_dict.update({target: val * (tr if tr else 1)})

    persistence_dict = {}
    for target in range(1, 4):
        binary, counts = segments_mirrored(target, first_subject_labels)
        persistence_dict.update({target: (binary.sum() / counts) * (tr if tr else 1)})
        assert all(x in [0, 1] for x in binary)

    arr1, arr2 = np.array(list(persistence_reimplemented_dict.values())), np.array(list(persistence_dict.values()))

    assert np.allclose(arr1, arr2, atol=0.0001)

    # Assert that the external calculation is the same as internal
    assert [x for x in list(persistence_dict.values()) if not math.isnan(x)] == [
        x for x in df.loc[0, :].values if not math.isnan(x)
    ]


def test_transition_frequency():
    """
    Verifies transition frequency is computed correctly.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups={"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]})
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=3)

    df_dict = cap_analysis.calculate_metrics(
        subject_timeseries=timeseries, return_df=True, metrics=["transition_frequency"]
    )
    assert len(df_dict) == 1

    df = df_dict["transition_frequency"]
    assert all(df.iloc[:, 3].values >= 0)

    # Get first subject
    first_subject_labels = get_first_subject(timeseries, cap_analysis)

    # Different from internal implementation
    transition_frequency = 0
    for indx in range(1, len(first_subject_labels)):
        if first_subject_labels[indx - 1] != first_subject_labels[indx]:
            transition_frequency += 1

    assert df.iloc[0, 3] == transition_frequency


def test_transition_probability():
    """
    Verifies transition probability is computed correctly.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups={"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]})
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=3)

    df_dict = cap_analysis.calculate_metrics(
        subject_timeseries=timeseries, return_df=True, metrics=["transition_probability"]
    )
    assert len(df_dict) == 1

    # Check that all transition probabilities sum to 1
    for group in cap_analysis.groups:
        df = df_dict["transition_probability"][group]
        df = df[[x for x in df.columns if x.startswith("CAP")]]

        assert all(df.apply(lambda x: all(x.values <= 1) and all(x.values >= 0)).values)

        for i in df.index:
            for cap in cap_analysis.caps[group]:
                target_cap = cap.split("-")[-1]
                columns = df.filter(regex=rf"^{target_cap}\.").columns.tolist()
                assert (
                    math.isclose(df.loc[i, columns].values.sum(), 1, abs_tol=0.0001)
                    or df.loc[i, columns].values.sum() == 0
                )


def test_runs():
    """
    Tests to ensure that using `continuous_runs` produces the expected shape.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP()
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=2)

    for i in range(1, 4):
        met1 = cap_analysis.calculate_metrics(subject_timeseries=timeseries, return_df=True, runs=i)
        met2 = cap_analysis.calculate_metrics(
            subject_timeseries=timeseries, return_df=True, runs=i, continuous_runs=True
        )
        # If only one run `continuous_runs`` should not differ
        assert met1["persistence"].equals(met2["persistence"])

    met1 = cap_analysis.calculate_metrics(subject_timeseries=timeseries, return_df=True)
    met2 = cap_analysis.calculate_metrics(subject_timeseries=timeseries, return_df=True, continuous_runs=True)
    # Should differ
    assert not met1["persistence"].equals(met2["persistence"])
    # Continuous run should have 1/3 the number of rows since each subject in the randomized data has three runs
    assert met1["persistence"].shape[0] / 3 == met2["persistence"].shape[0]


@pytest.mark.parametrize("continuous_runs", [False, True])
def test_metrics_mathematical_relationship(continuous_runs):
    """
    Verifies the mathematical relationship between temporal fraction, persistence, and counts stated in the supplementary
    Yang et al 2021; temporal fraction = (persistence*counts)/total. Does this verification for when `continuous_runs`
    is True and False.

    Reference
    ---------
    Yang, H., Zhang, H., Di, X., Wang, S., Meng, C., Tian, L., & Biswal, B. (2021). Reproducible coactivation patterns
    of functional brain networks reveal the aberrant dynamic state transition in schizophrenia. NeuroImage, 237,
    118193. https://doi.org/10.1016/j.neuroimage.2021.118193

    """
    timeseries = Parcellation.get_schaefer("timeseries")

    # Based on the equation in the supplementary of Yang et al 2021; temporal fraction = (persistence*counts)/total
    cap_analysis = CAP()
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=2)

    counts = cap_analysis.calculate_metrics(
        subject_timeseries=timeseries,
        return_df=True,
        metrics="counts",
        continuous_runs=continuous_runs,
    )["counts"]

    persistence = cap_analysis.calculate_metrics(
        subject_timeseries=timeseries,
        return_df=True,
        metrics="persistence",
        continuous_runs=continuous_runs,
    )["persistence"]

    temp = cap_analysis.calculate_metrics(
        subject_timeseries=timeseries,
        return_df=True,
        metrics=["temporal_fraction"],
        continuous_runs=continuous_runs,
    )["temporal_fraction"]

    N = 100 if not continuous_runs else 300

    for cap in ["CAP-1", "CAP-2"]:
        for i in temp.index:
            assert math.isclose((counts.loc[i, cap] * persistence.loc[i, cap]) / N, temp.loc[i, cap], abs_tol=0.0001)
            if persistence.loc[i, cap] == 0:
                assert counts.loc[i, cap] == 0 and temp.loc[i, cap] == 0


def test_calculate_metrics_using_pickle(tmp_dir):
    """
    Ensure that pickles can be used as input for `calculate_metrics`.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups={"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]})
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=2)

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


def test_subject_setter():
    """
    Tests the `subject_setter` setter property. Ensures that `calculate_metrics` respects when a subject is added
    to this property.
    """
    timeseries = Parcellation.get_schaefer("timeseries")

    cap_analysis = CAP(groups={"A": [0, 1, 2, 4], "B": [3, 5, 6, 7, 8, 9, 6]})
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=2)

    metrics = ["temporal_fraction"]

    df_shape = cap_analysis.calculate_metrics(
        subject_timeseries=os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
        metrics=metrics,
        continuous_runs=True,
    )["temporal_fraction"].shape

    new_timeseries = copy.deepcopy(timeseries)
    new_timeseries.update({"12": {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)}})

    subject_table = copy.deepcopy(cap_analysis.subject_table)
    subject_table.update({"12": "A"})

    # Set
    cap_analysis.subject_table = subject_table

    new_df_shape = cap_analysis.calculate_metrics(
        subject_timeseries=new_timeseries,
        metrics=metrics,
        continuous_runs=True,
    )["temporal_fraction"].shape

    assert df_shape[0] + 1 == new_df_shape[0]


def test_parcel_setter():
    """
    Tests the `parcel_approach` setter property.
    """
    parcel_approach = Parcellation.get_schaefer("parcellation")
    cap_analysis = CAP()

    assert not cap_analysis.parcel_approach

    # Set new parcel approach
    cap_analysis.parcel_approach = parcel_approach
    assert "Schaefer" in cap_analysis.parcel_approach


def test_get_caps_cluster_selection_plot(tmp_dir):
    """
    Ensures that `get_caps` produces a plot when requested.
    """
    timeseries = Parcellation.get_schaefer("timeseries")
    cap_analysis = CAP()

    cap_analysis.get_caps(
        subject_timeseries=timeseries,
        n_clusters=[2, 3, 4, 5],
        cluster_selection_method="silhouette",
        output_dir=tmp_dir.name,
        step=2,
        show_figs=False,
    )

    files = glob.glob(os.path.join(tmp_dir.name, "*.png"))
    assert files
    [os.remove(file) for file in files]


@pytest.mark.parametrize(
    "timeseries, parcel_approach",
    [
        (Parcellation.get_aal("timeseries", "SPM8"), Parcellation.get_aal("parcellation", "SPM8")),
        (Parcellation.get_aal("timeseries", "3v2"), Parcellation.get_aal("parcellation", "3v2")),
        (Parcellation.get_schaefer("timeseries"), Parcellation.get_schaefer("parcellation")),
        (Parcellation.get_custom("timeseries"), Parcellation.get_custom("parcellation")),
    ],
)
def test_caps2plot(tmp_dir, timeseries, parcel_approach):
    """
    Ensures `caps2plot` produces the expected number of plots and that properties, which store the region means and
    outer product information, have the expected shape based on the parcellation used. May fail on occasion due
    to  _tkinter.TclError.
    """
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=2)

    # Subplots set to False
    kwargs = dict(
        subplots=False,
        yticklabels_size=5,
        borderwidths=10,
        wspace=0.1,
        xlabel_rotation=90,
        xticklabels_size=5,
        hspace=0.6,
        tight_layout=True,
        cbarlabels_size=8,
        invalid_kwarg=0,
        suffix_filename="suffix_name",
        show_figs=False,
        visual_scope=["regions", "nodes"],
        plot_options=["heatmap", "outer_product"],
        output_dir=tmp_dir.name,
    )

    cap_analysis.caps2plot(**kwargs)
    check_imgs(tmp_dir, values_dict={"heatmap": 2, "outer": 4})

    # Assess hemisphere labels
    if "AAL" not in parcel_approach:
        kwargs["hemisphere_labels"] = True
        cap_analysis.caps2plot(**kwargs)
        check_imgs(tmp_dir, values_dict={"heatmap": 2, "outer": 4})

    # Subplots set to True
    kwargs["subplots"] = True
    kwargs["hemisphere_labels"] = False
    kwargs["share_y"] = True
    cap_analysis.caps2plot(**kwargs)
    check_imgs(tmp_dir, values_dict={"heatmap": 2, "outer": 2})

    if "AAL" not in parcel_approach:
        kwargs["hemisphere_labels"] = True
        cap_analysis.caps2plot(**kwargs)
        check_imgs(tmp_dir, values_dict={"heatmap": 2, "outer": 2})

    parcel_name = list(parcel_approach.keys())[0]
    if parcel_name != "Custom":
        parcel_dict = cap_analysis.parcel_approach
        nodes_dim = (len(parcel_dict[parcel_name]["nodes"]), len(parcel_dict[parcel_name]["nodes"]))
        regions_dim = (len(parcel_dict[parcel_name]["regions"]),)
    else:
        nodes_dim = (426, 426)
        regions_dim = (23,)

    assert cap_analysis.outer_products["All Subjects"]["CAP-1"].shape == nodes_dim
    assert cap_analysis.region_means["All Subjects"]["CAP-1"].shape == regions_dim


def test_caps2corr(tmp_dir):
    """
    Ensures `caps2corr` produces the expected number of plots and files.
    """
    timeseries = Parcellation.get_schaefer("timeseries")
    cap_analysis = CAP()
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=2)

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


@pytest.mark.parametrize(
    "timeseries, parcel_approach",
    [
        (Parcellation.get_schaefer("timeseries"), Parcellation.get_schaefer("parcellation")),
        (Parcellation.get_custom("timeseries"), Parcellation.get_custom("parcellation")),
        (Parcellation.get_aal("timeseries", "SPM8"), Parcellation.get_aal("parcellation", "SPM8")),
        (Parcellation.get_aal("timeseries", "3v2"), Parcellation.get_aal("parcellation", "3v2")),
    ],
)
def test_caps2radar(tmp_dir, timeseries, parcel_approach):
    """
    Ensures `caps2radar` produces the expected number of plots.
    """
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=2)

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
    check_imgs(tmp_dir, plot_type="radar", values_dict={"png": 2})
    cap_analysis.caps2radar(
        radialaxis=radialaxis, fill="toself", show_figs=False, as_html=True, output_dir=tmp_dir.name
    )
    check_imgs(tmp_dir, plot_type="radar", values_dict={"html": 2})

    cap_analysis.caps2radar(
        radialaxis=radialaxis, fill="toself", show_figs=False, as_html=False, output_dir=tmp_dir.name
    )
    check_imgs(tmp_dir, plot_type="radar", values_dict={"png": 2})

    cap_analysis.caps2radar(
        radialaxis=radialaxis,
        fill="toself",
        use_scatterpolar=True,
        scattersize=10,
        show_figs=False,
        as_html=True,
        output_dir=tmp_dir.name,
    )
    check_imgs(tmp_dir, plot_type="radar", values_dict={"html": 2})


@pytest.mark.parametrize(
    "timeseries, parcel_approach",
    [
        (Parcellation.get_schaefer("timeseries"), Parcellation.get_schaefer("parcellation")),
        (Parcellation.get_custom("timeseries"), Parcellation.get_custom("parcellation")),
        (Parcellation.get_aal("timeseries", "SPM8"), Parcellation.get_aal("parcellation", "SPM8")),
        (Parcellation.get_aal("timeseries", "3v2"), Parcellation.get_aal("parcellation", "3v2")),
    ],
)
def test_caps2niftis(tmp_dir, timeseries, parcel_approach):
    """
    Ensures `caps2niftis` produces the expected number of files, the KNN interpolation works, and that
    the 3D NifTI image can be used to reconstruct the 1D cluster centroid.
    """
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=2)

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
    check_imgs(tmp_dir, plot_type="nifti", values_dict={"nii.gz": 2})

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
        check_imgs(tmp_dir, plot_type="nifti", values_dict={"nii.gz": 2})

        cap_analysis.caps2niftis(
            output_dir=tmp_dir.name, suffix_filename="AAL_ref", knn_dict={"k": 3, "reference_atlas": "AAL"}
        )

        # Check interpolation using AAL reference
        for i in glob.glob(os.path.join(tmp_dir.name, "*AAL_ref*")):
            interpolated_nifti = nib.load(i).get_fdata()
            assert not np.array_equal(original_nifti[original_nifti == 0], interpolated_nifti[original_nifti == 0])

        # Check files
        check_imgs(tmp_dir, plot_type="nifti", values_dict={"nii.gz": 2})


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") and sys.platform != "linux", reason="VTK Github coaction only works for Linux"
)
@pytest.mark.parametrize(
    "timeseries, parcel_approach",
    [
        (Parcellation.get_schaefer("timeseries"), Parcellation.get_schaefer("parcellation")),
        (Parcellation.get_custom("timeseries"), Parcellation.get_custom("parcellation")),
        (Parcellation.get_aal("timeseries", "SPM8"), Parcellation.get_aal("parcellation", "SPM8")),
        (Parcellation.get_aal("timeseries", "3v2"), Parcellation.get_aal("parcellation", "3v2")),
    ],
)
def test_caps2surf(tmp_dir, remove_files, timeseries, parcel_approach):
    """
    Ensures `caps2surf` produces the expected number of plots.
    """
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(subject_timeseries=timeseries, n_clusters=2)

    cap_analysis.caps2surf(
        method="nearest",
        fwhm=1,
        save_stat_maps=True,
        suffix_filename="suffix_name",
        output_dir=tmp_dir.name,
        suffix_title="placeholder",
        show_figs=False,
    )
    check_imgs(tmp_dir, plot_type="surface", values_dict={"png": 2})
    check_imgs(tmp_dir, plot_type="nifti", values_dict={"nii.gz": 2})

    cap_analysis.caps2surf(
        method="linear", fwhm=1, save_stat_maps=False, output_dir=tmp_dir.name, as_outline=True, show_figs=False
    )
    check_imgs(tmp_dir, plot_type="surface", values_dict={"png": 2})
    check_imgs(tmp_dir, plot_type="nifti", values_dict={"nii.gz": 0})


def test_method_chaining(tmp_dir):
    """
    Tests if method chanining works.
    """
    parcel_approach = Parcellation.get_schaefer("parcellation")
    timeseries = Parcellation.get_schaefer("timeseries")

    a = {"show_figs": False}
    cap_analysis = CAP(parcel_approach=parcel_approach)
    cap_analysis.get_caps(timeseries, n_clusters=2).caps2plot(**a).caps2niftis(tmp_dir.name).caps2radar(**a)

    # Should not be None
    assert cap_analysis.cosine_similarity

    check_imgs(tmp_dir, plot_type="nifti", values_dict={"nii.gz": 2})


def test_check_raise_error():
    """
    Tests that the proper error messages are being produced.
    """
    error_msg = {
        "_caps": "Cannot plot caps since `self.caps` is None. Run `self.get_caps()` first.",
        "_parcel_approach": (
            "`self.parcel_approach` is None. Add `parcel_approach` using "
            "`self.parcel_approach=parcel_approach` to use this function."
        ),
        "_kmeans": ("Cannot calculate metrics since `self.kmeans` is None. Run " "`self.get_caps()` first."),
    }

    for i in ["_caps", "_parcel_approach", "_kmeans"]:
        with pytest.raises(AttributeError, match=re.escape(error_msg[i])):
            CAP._raise_error(i)


def test_raise_error_methods():
    """
    Tests that the proper error messages are being produced when using functions.
    """
    error_msg = {
        "_caps": ("Cannot plot caps since `self.caps` is None. Run `self.get_caps()` first."),
        "_parcel_approach": (
            "`self.parcel_approach` is None. Add `parcel_approach` using "
            "`self.parcel_approach=parcel_approach` to use this function."
        ),
        "_kmeans": ("Cannot calculate metrics since `self.kmeans` is None. Run `self.get_caps()` first."),
    }

    cap_analysis = CAP()

    with pytest.raises(AttributeError, match=re.escape(error_msg["_caps"])):
        cap_analysis.caps2corr()

    with pytest.raises(AttributeError, match=re.escape(error_msg["_parcel_approach"])):
        cap_analysis.caps2plot()

    with pytest.raises(AttributeError, match=re.escape(error_msg["_kmeans"])):
        cap_analysis.calculate_metrics(os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"))


def test_compute_cosine_similarity():
    """
    Tests cosine similarity computation.
    """
    amp, bin_vec = np.array([0.3, 0.3, 0.3]), np.array([0, 0, 0])

    assert np.isnan(CAP._compute_cosine_similarity(amp, bin_vec))
    assert math.isclose(CAP._compute_cosine_similarity(amp, np.where(bin_vec, 0, 1)), 1, abs_tol=0.0001)
