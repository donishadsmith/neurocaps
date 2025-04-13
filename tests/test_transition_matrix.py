import glob, os
import numpy as np, pytest

from neurocaps.analysis import CAP, transition_matrix

from .utils import check_outputs


@pytest.mark.parametrize(
    "group, suffix_title, suffix_filename",
    [
        (None, None, None),
        ({"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]}, None, "suffix_filename"),
        ({"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]}, "suffix_title", "suffix_filename"),
    ],
)
def test_transition_matrix(tmp_dir, group, suffix_title, suffix_filename):
    """
    Tests that the subject-level transition probabilities are used to generate the correct averages for
    transition probabilities at the group level. Also ensure that the proper files are produced.
    """

    subject_timeseries = {str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 11)}
    cap_analysis = CAP(groups=group)
    cap_analysis.get_caps(subject_timeseries=subject_timeseries, n_clusters=3)
    output = cap_analysis.calculate_metrics(subject_timeseries=subject_timeseries, metrics="transition_probability")
    trans_output = transition_matrix(
        output["transition_probability"],
        output_dir=tmp_dir.name,
        show_figs=False,
        borderwidths=2,
        suffix_filename=suffix_filename,
        suffix_title=suffix_title,
    )

    groups = list(trans_output)

    for group in groups:
        assert output["transition_probability"][group].loc[:, "1.1"].mean() == trans_output[group].loc["CAP-1", "CAP-1"]
        assert output["transition_probability"][group].loc[:, "1.2"].mean() == trans_output[group].loc["CAP-1", "CAP-2"]
        assert output["transition_probability"][group].loc[:, "2.1"].mean() == trans_output[group].loc["CAP-2", "CAP-1"]

    check_outputs(tmp_dir, {"csv": len(groups), "png": len(groups)}, plot_type="trans")

    trans_output = transition_matrix(
        output["transition_probability"], output_dir=tmp_dir.name, show_figs=False, save_df=False, as_pickle=True
    )

    check_outputs(tmp_dir, {"pkl": len(groups)}, plot_type="pickle")
