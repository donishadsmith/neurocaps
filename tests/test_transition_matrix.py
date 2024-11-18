import glob, os, tempfile
import numpy as np, pytest

from neurocaps.analysis import CAP, transition_matrix

tmp_dir = tempfile.TemporaryDirectory()


@pytest.mark.parametrize("group", [None, {"A": [1, 2, 3, 5], "B": [4, 6, 7, 8, 9, 10, 7]}])
def test_transition_matrix(group):
    subject_timeseries = {str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(1, 11)}
    cap_analysis = CAP(groups=group)
    cap_analysis.get_caps(subject_timeseries=subject_timeseries, n_clusters=3)
    output = cap_analysis.calculate_metrics(subject_timeseries=subject_timeseries, metrics="transition_probability")
    trans_output = transition_matrix(
        output["transition_probability"], output_dir=tmp_dir.name, show_figs=False, borderwidths=2
    )

    groups = list(trans_output)

    for group in groups:
        assert output["transition_probability"][group].loc[:, "1.1"].mean() == trans_output[group].loc["CAP-1", "CAP-1"]
        assert output["transition_probability"][group].loc[:, "1.2"].mean() == trans_output[group].loc["CAP-1", "CAP-2"]
        assert output["transition_probability"][group].loc[:, "2.1"].mean() == trans_output[group].loc["CAP-2", "CAP-1"]

    png_files = glob.glob(os.path.join(tmp_dir.name, "*transition_probability*.png"))
    assert len(png_files) == len(groups)
    csv_files = glob.glob(os.path.join(tmp_dir.name, "*transition_probability*.csv"))
    assert all(os.path.getsize(file) > 0 for file in csv_files)
    assert len(csv_files) == len(groups)

    [os.remove(x) for x in png_files + csv_files]
