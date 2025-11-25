"""Module for creating simulated datasets."""

import json, os
from typing import Union

from joblib import Parallel, delayed
from numpy.typing import NDArray
from tqdm.auto import tqdm

import nibabel as nib
import numpy as np
import pandas as pd

from neurocaps.analysis.cap._internals.surface import save_nifti_img
from neurocaps.typing import SubjectTimeseries
from neurocaps.utils import _io as io_utils
from neurocaps.utils._logging import setup_logger

LG = setup_logger(__name__)


def simulate_bids_dataset(
    n_subs: int = 1,
    n_runs: int = 1,
    n_volumes: int = 100,
    task_name: str = "rest",
    output_dir: Union[str, None] = None,
    n_cores: Union[int, None] = None,
    progress_bar: bool = False,
) -> str:
    """
    Generate a Simulated BIDS Dataset with fMRIPrep Derivatives.

    Creates a minimal BIDS dataset structure with fMRIPrep derivatives, including:
        - BIDS root directory with only dataset description JSON file
        - Derivatives folder with fMRIPrep outputs:
            - Dataset description JSON file
            - One simulated NIfTI image per subject/run
            - One confounds TSV file per subject/run

    .. note::
       Returns ``output_dir`` if the path exists.

    .. versionadded:: 0.34.1

    Parameters
    ----------
    n_subs : :obj:`int`, default=1
        Number of subjects.

    n_runs : :obj:`int`, default=1
        Number of runs for each subject.

    n_volumes : :obj:`int`, default=100
        Number of volumes for the NIfTI images.

    task_name : :obj:`str`, default="rest"
        Name of task.

    output_dir : :obj:`str` or :obj:`None`, default=None
        Path to save the simulated BIDS directory to.

        .. important::
           If None, a directory named "simulated_bids_dir" will be created in the current working
           directory.

    n_cores : :obj:`int` or :obj:`None`, default=None
        The number of cores to use for multiprocessing with Joblib (over subjects). The "loky"
        backend is used.

        .. versionadded:: 0.34.3

    progress_bar : :obj:`bool`, default=False
        If True, displays a progress bar.

        .. versionadded:: 0.34.3

    Returns
    -------
    str
        Root of the simulated BIDs directory.
    """
    bids_root = output_dir
    if not bids_root:
        bids_root = os.path.join(os.getcwd(), "simulated_bids_dir")

    if os.path.exists(bids_root):
        LG.warning("`output_dir` already exists. Returning the `output_dir` string.")
        return bids_root

    # Create root directory with derivatives folder
    fmriprep_dir = os.path.join(bids_root, "derivatives", "fmriprep")
    io_utils.makedir(fmriprep_dir)

    # Create dataset description for root and fmriprep
    save_dataset_description(create_dataset_description("Mock"), bids_root)
    save_dataset_description(create_dataset_description("fMRIPrep", derivative=True), fmriprep_dir)

    # Generate list of tuples for each subject
    args_list = [(fmriprep_dir, subj_id, n_runs, task_name, n_volumes) for subj_id in range(n_subs)]

    parallel = Parallel(return_as="generator", n_jobs=n_cores, backend="loky")
    # generator needed for tqdm, iteration triggers side effects (file creation)
    list(
        tqdm(
            parallel(delayed(_create_sub_files)(*args) for args in args_list),
            desc="Creating Simulated Subjects",
            total=len(args_list),
            disable=not progress_bar,
        )
    )

    return bids_root


def _create_sub_files(
    fmriprep_dir: str, subj_id: int, n_runs: int, task_name: str, n_volumes: int
) -> None:
    """Iterates through each to create simulate data."""
    sub_dir = os.path.join(fmriprep_dir, f"sub-{subj_id}", "func")
    io_utils.makedir(sub_dir, suppress_msg=True)

    for run_id in range(n_runs):
        base_filename = f"sub-{subj_id}_task-{task_name}_run-{run_id}"

        confound_df = _simulate_confound_data(n_volumes)
        filename = base_filename + "_desc-confounds_timeseries.tsv"
        _save_confound_data(confound_df, sub_dir, filename)

        nifti_img = _simulate_nifti_image(n_volumes)
        filename = base_filename + "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        save_nifti_img(nifti_img, sub_dir, filename)

    return None


def create_dataset_description(
    dataset_name: str, bids_version: str = "1.0.0", derivative: bool = False
) -> dict:
    """
    Generate Dataset Description.

    Creates a dictionary containing the name and BIDs version of a dataset.

    .. versionadded:: 0.34.1

    Parameters
    ----------
    dataset_name : :obj:`str`
        Name of the dataset.

    bids_version : :obj:`str`, default="1.0.0"
        Version of the BIDS dataset.

    derivative : :obj:`bool`, default=False
        Determines if "GeneratedBy" key is added to dictionary.

    Returns
    -------
    dict
        The dataset description dictionary
    """
    dataset_description = {"Name": dataset_name, "BIDSVersion": bids_version}

    if derivative:
        dataset_description.update({"GeneratedBy": [{"Name": dataset_name}]})

    return dataset_description


def save_dataset_description(dataset_description: dict, output_dir: str) -> None:
    """
    Save Dataset Description.

    Saves the dataset description dictionary as a file named "dataset_description.json" to the
    directory specified by ``output_dir``.

    .. versionadded:: 0.34.1

    Parameters
    ----------
    dataset_description : :obj:`dict`
        The dataset description dictionary.

    output_dir : :obj:`str`
        Path to save the JSON file to.

    """
    with open(os.path.join(output_dir, "dataset_description.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_description, f)


def _simulate_confound_data(n_rows: int) -> pd.DataFrame:
    """Creates a simulate confound dataframe."""
    columns = [
        "global_signal",
        "framewise_displacement",
        "a_comp_cor_00",
        "a_comp_cor_01",
        "a_comp_cor_02",
        "a_comp_cor_03",
        "a_comp_cor_04",
        "a_comp_cor_05",
        "cosine_00",
        "cosine_01",
        "cosine_02",
        "cosine_03",
        "cosine_04",
        "cosine_05",
        "cosine_06",
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
    ]

    n_cols = len(columns)
    random_data = np.random.rand(n_rows, n_cols)

    return pd.DataFrame(random_data, columns=columns)


def _save_confound_data(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """Saves confound data as tsv file."""
    df.to_csv(os.path.join(output_dir, filename), sep="\t", index=None)


def _simulate_nifti_image(n_volumes: int) -> nib.Nifti1Image:
    """Simulate a NIfTI of shape (97, 115, 98, n_volumes)."""
    img_shape = (97, 115, 98, n_volumes)
    affine = _create_affine(xyz_diagonal_value=2, translate_vec=np.array([-96, -132, -78, 1]))

    return nib.Nifti1Image(np.random.rand(*img_shape), affine)


def _create_affine(xyz_diagonal_value: int, translate_vec: NDArray) -> NDArray:
    """Generate an 4x4 affine matrix."""
    affine = np.zeros((4, 4))
    np.fill_diagonal(affine[:3, :3], xyz_diagonal_value)
    affine[:, 3:] = translate_vec[:, np.newaxis]

    return affine


def simulate_subject_timeseries(
    n_subs: int = 8, n_runs: int = 3, shape: tuple = (400, 100)
) -> SubjectTimeseries:
    """
    Generate Simulated Subject Timeseries Data.

    Creates a subject timeseries dictionary with randomly generated values.

    .. versionadded:: 0.34.1

    Parameters
    ----------
    n_subs : :obj:`int`, default=8
        Number of subjects.

    n_runs : :obj:`int`, default=3
        Number of runs for each subject.

    shape : :obj:`tuple`, default=(400, 100)
        Shape of the generated timeseries data in the form (rows, columns).

    Returns
    -------
    SubjectTimeseries
        A dictionary mapping subject IDs to their run IDs and their associated timeseries
        (TRs x ROIs) as a NumPy array. Refer to documentation for ``SubjectTimeseries`` in the
        "See Also" section for an example structure.

    See Also
    --------
    :data:`neurocaps.typing.SubjectTimeseries`
        Type definition for the subject timeseries dictionary structure.
        (See: `SubjectTimeseries Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.SubjectTimeseries.html#neurocaps.typing.SubjectTimeseries>`_)
    """
    return {
        str(subj_id): {f"run-{run_id}": np.random.rand(*shape) for run_id in range(n_runs)}
        for subj_id in range(n_subs)
    }


__all__ = [
    "simulate_bids_dataset",
    "create_dataset_description",
    "save_dataset_description",
    "simulate_subject_timeseries",
]
