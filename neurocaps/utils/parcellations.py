"""Public utility functions for parcellations."""

from typing import Any, Literal, Optional, Union

import pandas as pd

from . import _io as io_utils
from ._helpers import list_to_str
from ._logging import setup_logger
from neurocaps.typing import CustomParcelApproach
from neurocaps.utils.datasets import _fetch as fetch_utils

LG = setup_logger(__name__)


def fetch_preset_parcel_approach(
    name: str, n_nodes: Optional[int] = None
) -> dict[Literal["Custom"], CustomParcelApproach]:
    """
    Fetches a Preset "Custom" Parcellation Approach.

    Creates a directory in the user home directory named "neurocaps_data" and downloads
    the ``parcel_approach`` (a JSON file) and the associated NifTI image from the
    Open Science Framework (OSF) if the corresponding files are not present in the
    directory.

    .. versionadded:: 0.32.2

    Parameters
    ----------
    name: :obj:`str`
        Name of the preset "Custom" parcellation approach to fetch. Options are "HCPex", "4S",
        and "Gordon".

        .. versionadded:: 0.32.3
           Added "Gordon".

    n_nodes: :obj:`int` or :obj:`None`, default=None
        Currently only relevant to "4S". Options for the "4S" are: 156, 256, 356,
        456, 556, 656, 757, 956, 1056. Defaults to 456 if None.

        .. note::
           The 856 node version of "4S" is currently unavailable.

    Returns
    -------
    dict[Literal["Custom"], CustomParcelApproach]
        A dictionary representing the "Custom" parcellation approach.

    Note
    ----
    **Region Mapping**: The mapping of regions/networks corresponds to the indices in the "nodes"
    list not its label ID. So, the first non-background index will be 0.

    See Also
    --------
    CustomParcelApproach
       The type definition for the Custom parcellation approach.
       (See `CustomParcelApproach Documentation
       <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.CustomParcelApproach.html#neurocaps.typing.CustomParcelApproach>`_)

    ParcelApproach
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation
        approaches.
        (See `ParcelApproach Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.ParcelApproach.html#neurocaps.typing.ParcelApproach>`_)
    """
    return fetch_utils.fetch_custom_parcel_approach(name, n_nodes)


def generate_custom_parcel_approach(
    filepath_or_df: Union[pd.DataFrame, str],
    maps_path: str,
    column_map: dict[Literal["nodes", "regions", "hemispheres"], str],
    hemisphere_map: Optional[dict[Literal["lh", "rh"], list[str]]] = None,
    background_label: Optional[str] = "Background",
    metadata: Optional[dict[str, Any]] = None,
) -> dict[Literal["Custom"], CustomParcelApproach]:
    """
    Generate a "Custom" Parcellation Approach From a Tabular Metadata File.

    Constructs the nested dictionary for the "Custom" ``parcel_approach`` by reading a tabular
    file containing metadata about the parcellation.

    .. important:
        The labels in the dataframe are assumed to be in order (minimum -> maximum label).

    .. versionadded:: 0.32.0

    Parameters
    ----------
    filepath_or_df: :obj:`pd.Dataframe` or :obj:`str`
        Path to the parcellation description file (e.g., "Schaefer_400.tsv"). Supports files
        with the following extension: ".csv", ".tsv", ".txt", ".xlsx". Can also be
        a pandas Dataframe.

    maps_path: :obj:`str`
        Path to the corresponding NIfTI parcellation file (e.g., "Schaefer_400.nii.gz").

    column_map: :obj:`dict[Literal["nodes", "regions", "hemispheres"], str]`
        A dictionary mapping keys to column names in the metadata file. The following keys are
        valid:

        - "nodes" (Required): Column name for the ROI labels/names.
        - "regions" (Optional): Column name for the network/region names.
        - "hemispheres" (Optional): Column name for hemisphere labels.

        .. note::
           If a "regions" key is provided in ``column_map``, then the "regions" subkey will be
           created. Additionally, for lateralization, a "hemispheres" key must be provided in
           ``column_map``. If the "regions" key is not provided, then the output custom
           parcellation dictionary will only contain the "maps" and "nodes" subkeys.

    hemisphere_map: :obj:`dict[Literal["lh", "rh"], list[str]]` or :obj:`None`, default=None
        A dictionary mapping hemisphere values in the metadata column to "lh" and "rh"
        (e.g. {"lh": ["Left", "L", "l", "LH"], "rh": ["Right", "R", "r", "RH"]]}).

        .. important::
           An error will be raised for cases where a region/network is partially lateralized
           (where not all nodes in the network are lateralized). Note that the lateralization
           information is only used in a specific case in ``CAP.caps2plot`` when
           ``visual_scope`` is set to "nodes" and the ``add_custom_node_labels`` kwarg is True.

    background_label: :obj:`str` or :obj:`None`, default="Background"
        The label name for the background ROI to be excluded (e.g. "background", "Unknown", etc).

        .. important::
           Will be used to only check the **first** row in the metadata file. Since it is
           assumed that the first label, typically the 0 index is the background. This label
           will be removed and the first non-background label is assigned as the 0 index in the
           "regions" sub-dictionary, which will shift the remaining labels backwards. Note that
           the mapping of regions/networks corresponds to the indices in the "nodes" list
           not its label ID.

    metadata: :obj:`dict[str, Any]` or None, default=None
        Metadata information to add to the "metadata" key if not None.

        .. versionadded:: 0.32.2

    Returns
    -------
    dict[Literal["Custom"], CustomParcelApproach]
        A dictionary representing the "Custom" parcellation approach.

    See Also
    --------
    CustomParcelApproach
       The type definition for the Custom parcellation approach.
       (See `CustomParcelApproach Documentation
       <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.CustomParcelApproach.html#neurocaps.typing.CustomParcelApproach>`_)

    ParcelApproach
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation
        approaches.
        (See `ParcelApproach Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.ParcelApproach.html#neurocaps.typing.ParcelApproach>`_)

    Note
    ----
    **Dataframe Structure**: The following is an example dataframe containing the metadata of
    the parcellation. The specific column names for the columns containing the labels (nodes),
    networks (regions), and hemisphere are to be mapped using ``column_map``. Note that the
    column names do not have to be named "labels", "networks", and "hemispheres". In this case,
    the map would be ``{"nodes": "labels", "regions": "networks", "hemispheres": "hemisphere}``.

    The labels for the hemispheres can be mapped using ``hemisphere_map``. In this case, the
    map would be ``{"lh": ["L"], "rh": [R, "rh"]}``.

    +------------------------+----------+------------+-----+
    | labels                 | networks | hemisphere | ... |
    +========================+==========+============+=====+
    | SalVentAttn_TempOccPar | VAN      | L          | ... |
    +------------------------+----------+------------+-----+
    | SalVentAttn_FrOperIns  | VAN      | R          | ... |
    +------------------------+----------+------------+-----+
    | Limbic_OFC             | Limbic   | rh         | ... |
    +------------------------+----------+------------+-----+
    | ...                    | ...      | ...        | ... |
    +------------------------+----------+------------+-----+

    **Excel Files**: Pandas requires an optional dependency (openpyxl) to be installed for
    Excel files (".xlsx").
    """
    _validate_column_map(column_map)
    _validate_hemisphere_map(hemisphere_map)

    df = _open_tabular_data(filepath_or_df)
    _check_if_columns_exists(df, column_map)

    # Drop first row if background label
    if background_label and df.loc[0, column_map["nodes"]] == background_label:
        df = df.iloc[1:].copy()

    # Check for NaN values
    for key in column_map:
        if key in ["nodes", "regions"]:
            _check_column_validity(df[column_map[key]])

    df = df.reset_index(drop=True)

    # Create base `parcel_approach`
    custom_parcel_approach = {"maps": str(maps_path), "nodes": df[column_map["nodes"]].tolist()}

    if "regions" in column_map:
        custom_parcel_approach["regions"] = _construct_regions_dict(df, column_map, hemisphere_map)

    if metadata:
        custom_parcel_approach["metadata"] = metadata

    return {"Custom": custom_parcel_approach}


def _validate_column_map(column_map: dict[Literal["nodes", "regions", "hemispheres"], str]) -> None:
    """Checks the validity of ``column_map``."""
    if not isinstance(column_map, dict):
        raise TypeError("`column_map` must be a dictionary.")

    required_keys = ["nodes"]
    optional_keys = ["regions", "hemispheres"]
    # Use for loop for now; may extend in future
    for key in required_keys:
        if key not in column_map:
            raise ValueError(f"`column_map` must contain a key for '{key}'.")

    # Check if value is string
    for key in column_map:
        if key not in required_keys + optional_keys:
            continue

        val = column_map.get(key)
        if not isinstance(val, str):
            raise ValueError(
                f"In `column_map`, '{key}' must be a string value but is currently assigned: {val}."
            )

    invalid_keys = set(column_map).difference(list(required_keys + optional_keys))
    if invalid_keys:
        LG.warning(
            f"The following keys are invalid and will be ignored: {list_to_str(invalid_keys)}."
        )


def _open_tabular_data(filepath_or_df: Union[pd.DataFrame, str]) -> pd.DataFrame:
    """Opens a tabular dataset."""
    if isinstance(filepath_or_df, pd.DataFrame):
        return filepath_or_df

    ext = io_utils.validate_file(filepath_or_df, [".csv", ".tsv", ".txt", ".xlsx"], return_ext=True)

    if ext != ".xlsx":
        df = pd.read_csv(filepath_or_df, sep=None, engine="python")
    else:
        df = pd.read_excel(filepath_or_df)

    return df


def _check_if_columns_exists(
    df: pd.DataFrame,
    column_map: dict[Literal["nodes", "regions", "hemispheres"], str],
) -> None:
    """Checks if the columns in ``column_map`` exists in the dataframe."""
    invalid_cols = set(column_map.values()).difference(df.columns)
    if invalid_cols:
        raise KeyError(
            "The following columns were not found in the metadata file: "
            f"{list_to_str(invalid_cols)}."
        )


def _validate_hemisphere_map(
    hemisphere_map: Optional[dict[Literal["lh", "rh"], list[str]]],
) -> None:
    """Validates ``hemisphere_map``."""
    if not hemisphere_map:
        return None

    if not isinstance(hemisphere_map, dict):
        raise TypeError("`hemisphere_map` must be a dictionary.")

    required_keys = ["lh", "rh"]
    missing_keys = set(required_keys).difference(hemisphere_map)
    if missing_keys:
        raise KeyError(
            "The following required keys are missing in ``hemisphere_dict``: "
            f"{list_to_str(missing_keys)}."
        )


def _construct_regions_dict(
    df: pd.DataFrame,
    column_map: dict[Literal["nodes", "regions", "hemispheres"], str],
    hemisphere_map: Optional[dict[Literal["lh", "rh"], list[str]]],
) -> dict[str, Union[dict[Literal["lh", "rh"], list[int]], list[int]]]:
    """Construct the "regions" dictionary."""
    regions_dict = {}

    # Only attempt to use hemispheres if the user mapped the column.
    attempt_lateralization = "hemispheres" in column_map

    if attempt_lateralization and hemisphere_map is None:
        # Use a default map if none is provided by the user.
        hemisphere_map = {"lh": ["L", "LH", "Left", "left"], "rh": ["R", "RH", "Right", "right"]}

    # Iterate through grouped dataframe -> (region_name, subset of data containing region_name)
    for region_name, group_df in df.groupby(column_map["regions"]):
        regions_dict[region_name] = group_df.index.tolist()

        if attempt_lateralization:
            if _is_group_lateralized(group_df, column_map["hemispheres"], hemisphere_map):
                lh_indices, rh_indices = _get_lateralized_region_indices(
                    group_df, column_map["hemispheres"], hemisphere_map
                )
                lateralized_indices = lh_indices + rh_indices
                _check_partial_lateralization(
                    group_df,
                    region_name,
                    column_map["nodes"],
                    column_map["hemispheres"],
                    lateralized_indices,
                )

                regions_dict[region_name] = {"lh": lh_indices, "rh": rh_indices}

    return regions_dict


def _has_nans_in_series(series: pd.Series) -> bool:
    """Checks for NaNs in a specific column."""
    return series.isna().sum()


def _check_column_validity(series: pd.Series) -> None:
    """Check if column as NaNs and is invalid (use for "nodes" and "regions")"""
    if _has_nans_in_series(series):
        raise ValueError(f"Must assign values to NaNs in the following column: {series.name}.")


def _is_group_lateralized(
    group_df: pd.DataFrame,
    hemisphere_col: str,
    hemisphere_map: dict[Literal["lh", "rh"], list[str]],
) -> bool:
    """Check if subset of data has any lateralized nodes."""
    return (
        group_df[hemisphere_col].notna().any()
        and group_df[hemisphere_col].isin(hemisphere_map["lh"] + hemisphere_map["rh"]).any()
    )


def _get_lateralized_region_indices(
    group_df: pd.DataFrame,
    hemisphere_col: str,
    hemisphere_map: dict[Literal["lh", "rh"], list[str]],
) -> tuple[list[int], list[int]]:
    """Get the indices for the left and right hemisphere from a grouped dataframe."""
    lh_indices = group_df[group_df[hemisphere_col].isin(hemisphere_map["lh"])].index.tolist()
    rh_indices = group_df[group_df[hemisphere_col].isin(hemisphere_map["rh"])].index.tolist()

    return lh_indices, rh_indices


def _check_partial_lateralization(
    group_df: pd.DataFrame,
    region_name: str,
    node_col: str,
    hemisphere_col: str,
    lateralized_indices: list[int],
) -> None:
    """
    Some regions are only partially lateralized (e.g. Cole-Anticevic atlas). This will result
    in an error.
    """
    if len(lateralized_indices) != len(group_df):
        unmapped_nodes = group_df[~group_df.index.isin(lateralized_indices)][node_col].tolist()
        raise ValueError(
            f"Region '{region_name}' has unmappable hemisphere labels for nodes: "
            f"{list_to_str(unmapped_nodes)}. Recommended to replace the hemisphere "
            "information of all nodes in this region to NaN in the following column: "
            f"{hemisphere_col}."
        )


__all__ = ["fetch_preset_parcel_approach", "generate_custom_parcel_approach"]
