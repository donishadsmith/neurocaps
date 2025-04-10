"""Module containing custom types."""

from typing import Literal, TypedDict, Union
from typing_extensions import Required, NotRequired

from numpy import floating
from numpy.typing import NDArray

SubjectTimeseries = dict[str, dict[str, NDArray[floating]]]
"""
   Type Definition for the Subject Timeseries Dictionary Structure.

   A dictionary mapping subject IDs to their run IDs and their associated timeseries (TRs x ROIs) as a NumPy array.
   The structure is as follows:

    ::

        subject_timeseries = {
                "101": {
                    "run-0": np.array(shape=[TR, ROIs]),
                    "run-1": np.array(shape=[TR, ROIs]),
                    "run-2": np.array(shape=[TR, ROIs]),
                },
                "102": {
                    "run-0": np.array(shape=[TR, ROIs]),
                    "run-1": np.array(shape=[TR, ROIs]),
                }
            }

    .. important:: The run IDs must be in the form "run-{0}" (e.g. "run-0" or "run-zero").
"""


class SchaeferParcelConfig(TypedDict):
    """
    Type Definition for the Schaefer Parcellation Configurations.

    A ``TypedDict`` representing the available subkeys (second level keys for "Schaefer") for initializing the Schaefer
    parcellation in the ``TimeseriesExtractor`` or ``CAP`` classes. The structure is as follows:

    ::

        {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}

    Parameters
    ----------
    n_rois: :obj:`int`
        Number of ROIs (Default=400). Options are 100, 200, 300, 400, 500, 600, 700, 800, 900, or 1000.
    yeo_networks: :obj:`int`
        Number of Yeo networks (Default=7). Options are 7 or 17.
    resolution_mm: :obj:`int`
        Spatial resolution in millimeters (Default=1). Options are 1 or 2.

    See Also
    --------
    ParcelConfig
        Type definition representing the configuration options and structure for the Schaefer and AAL parcellations.

    Notes
    -----
    See `Nilearn's fetch Schaefer documentation\
    <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_schaefer_2018.html>`_  for more information.
    """

    n_rois: NotRequired[int]
    yeo_networks: NotRequired[int]
    resolution_mm: NotRequired[int]


class AALParcelConfig(TypedDict):
    """
    Type Definition for the AAL Parcellation Configurations.

    A ``TypedDict`` representing the available subkeys (second level keys for "AAL") for initializing the AAL
    parcellation in the ``TimeseriesExtractor`` or ``CAP`` classes. The structure is as follows:

    ::

        {"version": "SPM12"}

    Parameters
    ----------
    version: :obj:`str`
        AAL parcellation version to use (Default="SPM12" if ``{"AAL": {}}`` is given). Options are "SPM5", "SPM8",
        "SPM12", or "3v2".

    See Also
    --------
    ParcelConfig
        Type definition representing the configuration options and structure for the Schaefer and AAL parcellations.

    Notes
    -----
    See `Nilearn's fetch AAL documentation\
    <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_aal.html>`_ for more information.
    """

    version: NotRequired[str]


ParcelConfig = Union[dict[Literal["Schaefer"], SchaeferParcelConfig], dict[Literal["AAL"], AALParcelConfig]]
"""
   Type Definition for the Parcellation Configurations.

   A dictionary mapping the Schaefer or AAL parcellation to their associated configuration subkeys that are used
   by the ``TimeseriesExtractor`` and ``CAP`` class to create the processed ``ParcelApproach``. The structure is
   as follows:

    ::

        # Structure of Schaefer
        {"Schaefer": SchaeferParcelConfig}


        # Structure of AAL
        {"AAL": AALParcelConfig}

    See Also
    --------
    :class:`neurocaps.typing.SchaeferParcelConfig`
        Type definition representing configuration options for the Schaefer parcellation.
    :class:`neurocaps.typing.AALParcelConfig`
        Type definition representing configuration options for the AAL parcellation.
"""


# No doc string
class ParcelApproachBase(TypedDict):
    maps: NotRequired[str]
    nodes: NotRequired[list[str]]


class SchaeferParcelApproach(ParcelApproachBase):
    """
    Type Definition for the Schaefer Parcellation Approach.

    A ``TypedDict`` representing the subkeys (second level keys for "Schaefer") for the processed Schaefer parcellation
    produced by the ``TimeseriesExtractor`` or ``CAP`` classes. The structure is as follows:

    ::

        {
            "maps": "path/to/parcellation.nii.gz",
            "nodes": ["LH_Vis1", "LH_SomSot1", "RH_Vis1", "RH_Somsot1"],
            "regions": ["Vis", "SomSot"]
        }

    Parameters
    ----------
    maps: :obj:`str`
        Path to the Schaefer parcellation.
    nodes: :obj:`list[str]`
        List of nodes (ROIs) in the Schaefer parcellation. Ordered in ascending order of their label ID in the
        parcellation and must exclude "Background".
    regions: :obj:`list[str]`
        List of networks in the Schaefer parcellation. **Important**: For certain visualization methods, the ``in``
        operator is used to determine which nodes belong to which network. Therefore, network names must be contained
        within the corresponding node names (e.g., 'Vis' network should have nodes with 'Vis' in their names).

    See Also
    --------
    ParcelApproach
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation approaches.
    """

    regions: NotRequired[list[str]]


class AALParcelApproach(ParcelApproachBase):
    """
    Type Definition for the AAL Parcellation Approach.

    A ``TypedDict`` representing the subkeys (second level keys for "AAL") for the processed AAL parcellation produced
    by the ``TimeseriesExtractor`` or ``CAP`` classes. The structure is as follows:

    ::

        {
            "maps": "path/to/parcellation.nii.gz",
            "nodes": ["Precentral_L", "Precentral_R", "Frontal_Sup", "Frontal_Sup_R"],
            "regions": ["Precentral", "Frontal"]
        }

    Parameters
    ----------
    maps: :obj:`str`
        Path to the AAL parcellation.
    nodes: :obj:`list[str]`
        List of nodes (ROIs) in the AAL parcellation. Ordered in ascending order of their label ID in the parcellation
        and must exclude "Background".
    regions: :obj:`list[str]`
        List of networks in the AAL parcellation. **Important**: For certain visualization methods, the ``in``
        operator is used to determine which nodes belong to which network. Therefore, network names must be contained
        within the corresponding node names (e.g., 'Frontal' network should have nodes with 'Frontal' in their names).

    See Also
    --------
    ParcelApproach
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation approaches.
    """

    regions: NotRequired[list[str]]


class CustomRegionHemispheres(TypedDict):
    """
    Type Definition for Hemisphere Mapping in Custom Parcellation Regions.

    A ``TypedDict`` representing the mapping of the index position of the "nodes" to the left and right hemispheres.

    ::

        {"lh": [0, 1], "rh": [3, 4, 5]}

    Parameters
    ----------
    lh: :obj:`list[int] | range`
       List of integers or range representing the index positions of elements in the "nodes" list belonging to the
       left hemisphere of a specific region.
    rh: :obj:`list[int] | range`
       List of integers or range representing the index positions of elements in the "nodes" list belonging to the
       right hemisphere of a specific region.

    See Also
    --------
    CustomParcelApproach
       The type definition for the Custom parcellation approach.
    """

    lh: Required[Union[list[int], range]]
    rh: Required[Union[list[int], range]]


class CustomParcelApproach(ParcelApproachBase):
    """
    Type Definition for the Custom Parcellation Approach.

    A ``TypedDict`` representing the subkeys (second level keys for "Custom") for the user-defined Custom parcellation
    approach. The structure is as follows:

    ::

        {
            "maps": "path/to/parcellation.nii.gz",
            "nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus", "RH_Vis1", "RH_Vis2", "RH_Vis3", "RH_Hippocampus"],
            "regions": {
                "Visual": CustomRegionHemispheres
                "Hippocampus": CustomRegionHemispheres
            }
        }

    Parameters
    ----------
    maps: :obj:`str`
        Path to the Custom parcellation.
    nodes: :obj:`list[str]`
        List of nodes (ROIs) in the Custom parcellation. Ordered in ascending order of their label ID in the
        parcellation and must exclude "Background".
    regions: :obj:`dict[str, CustomRegionHemispheres]`
        Dictionary mapping the regions to their left and right hemispheres.

    See Also
    --------
    CustomRegionHemispheres
        Type definition of the Custom hemisphere dictionary for the "regions" subkeys.
    ParcelApproach
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation approaches.
    """

    regions: NotRequired[dict[str, CustomRegionHemispheres]]


ParcelApproach = Union[
    dict[Literal["Schaefer"], SchaeferParcelApproach],
    dict[Literal["AAL"], AALParcelApproach],
    dict[Literal["Custom"], CustomParcelApproach],
]
"""
  Type Definition for the Parcellation Approaches.

   A dictionary mapping the Schaefer, AAL, and Custom parcellation approaches to their associated subkeys:

    ::

        # Structure of Schaefer
        {"Schaefer": SchaeferParcelApproach}

        # Structure of AAL
        {"AAL": AALParcelApproach}

        # Structure of Custom
        {"Custom": CustomParcelApproach}

    See Also
    --------
    :class:`neurocaps.typing.SchaeferParcelApproach`
        Type definition representing the structure of the Schaefer parcellation approach.
    :class:`neurocaps.typing.AALParcelApproach`
        Type definition representing the structure of the AAL parcellation approach.
    :class:`neurocaps.typing.CustomParcelApproach`
        Type definition representing the structure of the Custom parcellation approach.
"""

__all__ = [
    "SubjectTimeseries",
    "SchaeferParcelConfig",
    "AALParcelConfig",
    "ParcelConfig",
    "SchaeferParcelApproach",
    "AALParcelApproach",
    "CustomParcelApproach",
    "CustomRegionHemispheres",
    "ParcelApproach",
]
