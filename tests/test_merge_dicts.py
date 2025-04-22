import glob, os

import numpy as np, pytest

from neurocaps.analysis import merge_dicts


@pytest.mark.parametrize(
    "return_reduced_dicts, return_merged_dicts", [(True, True), (False, False), (True, False), (False, True)]
)
def test_merge_dicts(return_reduced_dicts, return_merged_dicts):
    """Ensures the expected shape of the merged dictionaries are produced."""
    subject_timeseries = {str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 4)} for x in range(10)}

    subject_timeseries_2 = {str(x): {f"run-{y}": np.random.rand(100, 100) for y in range(1, 3)} for x in range(8)}

    subject_timeseries_merged = merge_dicts(
        [subject_timeseries, subject_timeseries_2],
        return_reduced_dicts=return_reduced_dicts,
        return_merged_dict=return_merged_dicts,
    )
    if return_merged_dicts:
        assert subject_timeseries_merged["merged"]["0"]["run-1"].shape == (200, 100)
        assert subject_timeseries_merged["merged"]["0"]["run-2"].shape == (200, 100)
        assert subject_timeseries_merged["merged"]["0"]["run-3"].shape == (100, 100)
        assert len(subject_timeseries_merged["merged"]["1"].keys()) == 3
        assert list(subject_timeseries_merged["merged"]["1"].keys()) == ["run-1", "run-2", "run-3"]
        assert id(subject_timeseries_merged["merged"]["1"]["run-3"]) != id(subject_timeseries["1"]["run-3"])

    all_dicts = merge_dicts(
        [subject_timeseries, subject_timeseries_2],
        return_reduced_dicts=return_reduced_dicts,
        return_merged_dict=return_merged_dicts,
    )

    if return_reduced_dicts and return_merged_dicts:
        # Should equal minimum intersected subjects
        assert len(all_dicts["merged"]) == len(subject_timeseries_2)
        assert all_dicts["dict_0"].keys() == all_dicts["dict_1"].keys() == all_dicts["merged"].keys()
        assert not np.array_equal(all_dicts["dict_0"]["1"]["run-1"], all_dicts["dict_1"]["1"]["run-1"])

    all_dicts = merge_dicts(
        [subject_timeseries, subject_timeseries_2],
        return_reduced_dicts=return_reduced_dicts,
        return_merged_dict=return_merged_dicts,
    )

    if return_reduced_dicts:
        assert all_dicts["dict_0"].keys() == all_dicts["dict_1"].keys()
        assert len(all_dicts["dict_0"].keys()) == len(all_dicts["dict_1"].keys())


@pytest.mark.parametrize(
    "return_reduced_dicts, return_merged_dicts", [(True, True), (False, False), (True, False), (False, True)]
)
def test_merge_dicts_pkl(tmp_dir, return_reduced_dicts, return_merged_dicts):
    """
    Ensures the expected shape of the merged dictionaries are produced and proper files are saved. Assesses when
    pickles are used as input.
    """
    subject_timeseries_merged = merge_dicts(
        [
            os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
            os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
        ],
        return_reduced_dicts=return_reduced_dicts,
        return_merged_dict=return_merged_dicts,
    )

    if return_merged_dicts:
        assert subject_timeseries_merged["merged"]["1"]["run-1"].shape == (200, 100)
        assert subject_timeseries_merged["merged"]["1"]["run-2"].shape == (200, 100)
        assert subject_timeseries_merged["merged"]["1"]["run-3"].shape == (200, 100)
        assert len(subject_timeseries_merged["merged"]["1"].keys()) == 3
        assert list(subject_timeseries_merged["merged"]["1"].keys()) == ["run-1", "run-2", "run-3"]

    all_dicts = merge_dicts(
        [
            os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
            os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
        ],
        return_reduced_dicts=return_reduced_dicts,
        return_merged_dict=return_merged_dicts,
    )

    if return_reduced_dicts and return_merged_dicts:
        assert all_dicts["dict_0"].keys() == all_dicts["dict_1"].keys() == all_dicts["merged"].keys()

    all_dicts = merge_dicts(
        [
            os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
            os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
        ],
        return_reduced_dicts=return_reduced_dicts,
        return_merged_dict=return_merged_dicts,
    )
    if return_reduced_dicts:
        assert all_dicts["dict_0"].keys() == all_dicts["dict_1"].keys()

    # Test files
    if return_reduced_dicts:
        all_dicts = merge_dicts(
            [
                os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
                os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
            ],
            return_reduced_dicts=True,
            return_merged_dict=True,
            output_dir=tmp_dir.name,
            save_reduced_dicts=True,
        )

        files = glob.glob(os.path.join(tmp_dir.name, "*merged*")) + glob.glob(os.path.join(tmp_dir.name, "*reduced*"))
        assert len(files) == 3

        files_basename = [os.path.basename(file) for file in files]
        assert "subject_timeseries_0_reduced.pkl" in files_basename
        assert "subject_timeseries_1_reduced.pkl" in files_basename
        assert "merged_subject_timeseries.pkl" in files_basename
        assert all(os.path.getsize(file) > 0 for file in files)
        [os.remove(x) for x in files]

        all_dicts = merge_dicts(
            [
                os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
                os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
            ],
            return_reduced_dicts=False,
            return_merged_dict=False,
            output_dir=tmp_dir.name,
            filenames=["test_0_reduced.pkl", "test_1_reduced.pkl", "test_merged.pkl"],
            save_reduced_dicts=True,
        )

        files = glob.glob(os.path.join(tmp_dir.name, "*merged*")) + glob.glob(os.path.join(tmp_dir.name, "*reduced*"))
        assert len(files) == 3

        files_basename = [os.path.basename(file) for file in files]
        assert "test_0_reduced.pkl" in files_basename
        assert "test_1_reduced.pkl" in files_basename
        assert "test_merged.pkl" in files_basename
        assert all(os.path.getsize(file) > 0 for file in files)
        [os.remove(x) for x in files]

        if return_merged_dicts:
            all_dicts = merge_dicts(
                [
                    os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
                    os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
                ],
                return_reduced_dicts=False,
                return_merged_dict=True,
                filenames="test_merged.pkl",
                output_dir=tmp_dir.name,
                save_reduced_dicts=False,
            )

            files = glob.glob(os.path.join(tmp_dir.name, "*merged*")) + glob.glob(
                os.path.join(tmp_dir.name, "*reduced*")
            )
            assert len(files) == 1

            files_basename = [os.path.basename(file) for file in files]
            assert "test_merged.pkl" in files_basename
            assert all(os.path.getsize(file) > 0 for file in files)
            [os.remove(x) for x in files]

            # Use no name to check default name
            all_dicts = merge_dicts(
                [
                    os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
                    os.path.join(os.path.dirname(__file__), "data", "sample_timeseries.pkl"),
                ],
                return_reduced_dicts=False,
                return_merged_dict=True,
                output_dir=tmp_dir.name,
                save_reduced_dicts=False,
            )

            files = glob.glob(os.path.join(tmp_dir.name, "*merged*")) + glob.glob(
                os.path.join(tmp_dir.name, "*reduced*")
            )
            assert len(files) == 1

            files_basename = [os.path.basename(file) for file in files]
            assert "merged_subject_timeseries.pkl" in files_basename
            assert all(os.path.getsize(file) > 0 for file in files)
            [os.remove(x) for x in files]
