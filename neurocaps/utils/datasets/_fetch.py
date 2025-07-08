"""Module containing functions related to fetching the preset parcellation approaches."""

import os
from typing import Literal, Union

import numpy as np

try:
    from nilearn.datasets._utils import fetch_files as _fetch_files
except ImportError:
    from nilearn.datasets.utils import _fetch_files

from ._presets import (
    ATLAS_N_NODES,
    PRESET_ATLAS_NAME,
    PRESET_JSON_NAME,
    PRESET_METADATA,
    OSF_FILE_URLS,
    VALID_PRESETS,
)
from neurocaps.typing import CustomParcelApproach
from neurocaps.utils import _io as io_utils
from neurocaps.utils._helpers import list_to_str
from neurocaps.utils._logging import setup_logger

LG = setup_logger(__name__)


def is_valid_preset(name: str) -> None:
    """Checks if the requested preset name is valid."""
    if not name in VALID_PRESETS:
        # No period
        raise ValueError(
            f"'{name}' is not a supported parcellation. Only the following are supported: "
            f"{list_to_str(VALID_PRESETS)}"
        )


def get_data_dir() -> str:
    """
    Gets the full path for 'neurocaps_data' in the users home directory. If it does not
    exist, then the directory and the subfolders are created.
    """
    data_dir = os.path.expanduser(os.path.join("~", "neurocaps_data"))
    # Create manually instead of using `fetch_files` to notify user that a directory was created
    io_utils.makedir(data_dir)

    return data_dir


def get_preset_subdir_paths(root_dir) -> tuple[str, str]:
    """Returns the names of the subdirs."""
    return os.path.join(root_dir, "presets", "jsons"), os.path.join(root_dir, "presets", "niftis")


def check_n_nodes(name: str, n_nodes: Union[int, None]) -> Union[int, None]:
    """Checks if the number of nodes for the parcellation is valid."""
    has_node_info = name in ATLAS_N_NODES
    if not has_node_info:
        if n_nodes:
            LG.warning(
                f"`n_nodes` parameter is not valid for the {name} parcellation and will be ignored."
            )
        return None

    if not n_nodes:
        n_nodes = ATLAS_N_NODES[name]["default_n"]
        LG.warning(f"Defaulting to {n_nodes} nodes for the {name} parcellation.")
    elif n_nodes not in ATLAS_N_NODES[name]["valid_n"]:
        if not isinstance(n_nodes, int):
            raise ValueError("`n_nodes` must be an integer.")
        else:
            raise ValueError(
                f"{n_nodes} is not valid for the {name} parcellation. "
                f"Only the following are supported: {list_to_str(list(ATLAS_N_NODES.get(name)))}."
            )

    return n_nodes


def get_preset_filenames(name: str, n_nodes: int = None) -> tuple[str, str]:
    """Retrieve the preset filenames."""
    json_filename, nifti_filename = PRESET_JSON_NAME[name], PRESET_ATLAS_NAME[name]
    if name in ATLAS_N_NODES and n_nodes:
        json_filename = json_filename.format(n_nodes)
        nifti_filename = nifti_filename.format(n_nodes)

    return json_filename, nifti_filename


def get_osf_file_url(filename: str) -> str:
    """Retrieves the url for a specific file."""
    base_filename = os.path.basename(filename)

    return r"https://osf.io/{}/download".format(OSF_FILE_URLS[base_filename])


def fetch_custom_parcel_approach(
    name: str, n_nodes: Union[int, None], overwrite: bool = False
) -> dict[Literal["Custom"], CustomParcelApproach]:
    """Fetches the "Custom" parcellation approach. Current valid names are "HCPex" and "4S"."""
    is_valid_preset(name)
    n_nodes = check_n_nodes(name, n_nodes)

    data_dir = get_data_dir()
    json_path, nifti_path = get_preset_subdir_paths(data_dir)

    json_filename, nifti_filename = get_preset_filenames(name, n_nodes)
    filenames = [os.path.join(json_path, json_filename), os.path.join(nifti_path, nifti_filename)]
    fetch_files_from_osf(filenames, overwrite=overwrite)

    # Load in json
    custom_parcel_approach = io_utils.open_json_file(os.path.join(json_path, json_filename))
    # Add path to parcellation file and metadata
    custom_parcel_approach["Custom"]["maps"] = os.path.join(nifti_path, nifti_filename)
    custom_parcel_approach["Custom"]["metadata"] = PRESET_METADATA[name]
    if n_nodes:
        custom_parcel_approach["Custom"]["metadata"]["n_nodes"] = n_nodes

    return custom_parcel_approach


def fetch_files_from_osf(
    filenames: Union[list[str], None],
    overwrite: bool = False,
    resume: bool = True,
    verbose: Union[bool, int] = 1,
) -> None:
    """
    Fetches the files from OSF storage.
    """
    bool_list = [
        io_utils.check_file_exist(filename, raise_errors=False, return_flag=True)
        for filename in filenames
    ]
    if not overwrite:
        filenames = np.array(filenames)
        filenames = filenames[~np.array(bool_list)]
        filenames = filenames.tolist()

    base_filenames = [os.path.basename(filename) for filename in filenames]
    if base_filenames:
        # No period
        LG.info(f"Downloading the following files from OSF: {list_to_str(base_filenames)}")

    opts = {"overwrite": overwrite}
    data_dirs = [os.path.dirname(filename) for filename in filenames]
    download_contents = zip(data_dirs, base_filenames)
    for data_dir, base_filename in download_contents:
        opts["move"] = base_filename
        # (file_path, url, opts)
        files = [(base_filename, get_osf_file_url(base_filename), opts)]
        # Will create data_dir path
        _ = _fetch_files(
            data_dir=data_dir, files=files, resume=resume, verbose=verbose, session=None
        )
