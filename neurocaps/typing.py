"""Module containing custom types."""

from typing import Any, Literal, TypedDict, Union
from typing_extensions import Required, NotRequired

from numpy import floating
from numpy.typing import NDArray

SubjectTimeseries = dict[str, dict[str, NDArray[floating]]]
"""
   Type Definition for the Subject Timeseries Dictionary Structure.

   A dictionary mapping subject IDs to their run IDs and their associated timeseries (TRs x ROIs)
   as a NumPy array. The structure is as follows:

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

    A ``TypedDict`` representing the available subkeys (second level keys for "Schaefer") for
    initializing the Schaefer parcellation in the ``TimeseriesExtractor`` or ``CAP`` classes. The
    structure is as follows:

    ::

        {"n_rois": 400, "yeo_networks": 7, "resolution_mm": 1}

    Parameters
    ----------
    n_rois: :obj:`int`
        Number of ROIs (Default=400). Options are 100, 200, 300, 400, 500, 600, 700, 800, 900, or
        1000.
    yeo_networks: :obj:`int`
        Number of Yeo networks (Default=7). Options are 7 or 17.
    resolution_mm: :obj:`int`
        Spatial resolution in millimeters (Default=1). Options are 1 or 2.

    See Also
    --------
    ParcelConfig
        Type definition representing the configuration options and structure for the Schaefer and
        AAL parcellations.
        (See `ParcelConfig Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.ParcelConfig.html#neurocaps.typing.ParcelConfig>`_)

    Notes
    -----
    See `Nilearn's fetch Schaefer documentation\
    <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_schaefer_2018.html>`_
    for more information.
    """

    n_rois: NotRequired[int]
    yeo_networks: NotRequired[int]
    resolution_mm: NotRequired[int]


class AALParcelConfig(TypedDict):
    """
    Type Definition for the AAL Parcellation Configurations.

    A ``TypedDict`` representing the available subkeys (second level keys for "AAL") for initializing
    the AAL parcellation in the ``TimeseriesExtractor`` or ``CAP`` classes. The structure is as follows:

    ::

        {"version": "SPM12"}

    Parameters
    ----------
    version: :obj:`str`
        AAL parcellation version to use (Default="SPM12" if ``{"AAL": {}}`` is given). Options are
        "SPM5", "SPM8", "SPM12", or "3v2".

    See Also
    --------
    ParcelConfig
        Type definition representing the configuration options and structure for the Schaefer and
        AAL parcellations.
        (See `ParcelConfig Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.ParcelConfig.html#neurocaps.typing.ParcelConfig>`_)

    Notes
    -----
    See `Nilearn's fetch AAL documentation\
    <https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_aal.html>`_
    for more information.
    """

    version: NotRequired[str]


ParcelConfig = Union[
    dict[Literal["Schaefer"], SchaeferParcelConfig], dict[Literal["AAL"], AALParcelConfig]
]
"""
   Type Definition for the Parcellation Configurations.

   A dictionary mapping the Schaefer or AAL parcellation to their associated configuration subkeys
   that are used by the ``TimeseriesExtractor`` and ``CAP`` class to create the processed
   ``ParcelApproach``. The structure is as follows:

    ::

        # Structure of Schaefer
        {"Schaefer": SchaeferParcelConfig}


        # Structure of AAL
        {"AAL": AALParcelConfig}

    See Also
    --------
    :class:`neurocaps.typing.SchaeferParcelConfig`
        Type definition representing configuration options for the Schaefer parcellation.
        (See `SchaeferParcelConfig Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.SchaeferParcelConfig.html#neurocaps.typing.SchaeferParcelConfig>`_)

    :class:`neurocaps.typing.AALParcelConfig`
        Type definition representing configuration options for the AAL parcellation.
        (See `AALParcelConfig Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.AALParcelConfig.html#neurocaps.typing.AALParcelConfig>`_)
"""


# No doc string
class ParcelApproachBase(TypedDict):
    maps: NotRequired[str]
    nodes: NotRequired[list[str]]
    metadata: NotRequired[dict[str, Any]]


class SchaeferParcelApproach(ParcelApproachBase):
    """
    Type Definition for the Schaefer Parcellation Approach.

    A ``TypedDict`` representing the subkeys (second level keys for "Schaefer") for the processed
    Schaefer parcellation produced by the ``TimeseriesExtractor`` or ``CAP`` classes. The structure
    is as follows:

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
        List of nodes (ROIs) in the Schaefer parcellation. Ordered in ascending order of their label
        ID in the parcellation and must exclude "Background".

    regions: :obj:`list[str]`
        List of networks in the Schaefer parcellation. **Important**: For certain visualization
        methods, the ``in`` operator is used to determine which nodes belong to which network.
        Therefore, network names must be contained within the corresponding node names (e.g., "Vis"
        network should have nodes with "Vis" in their names).

    metadata: :obj:`dict[str, Any]`
        Dictionary containing metadata information about the parcellation. This key is purely
        informational and can be removed, modified, or extended.

        .. versionadded:: 0.32.2

    See Also
    --------
    ParcelApproach
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation
        approaches.
        (See `ParcelApproach Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.ParcelApproach.html#neurocaps.typing.ParcelApproach>`_)
    """

    regions: NotRequired[list[str]]
    metadata: NotRequired[dict[str, Any]]


class AALParcelApproach(ParcelApproachBase):
    """
    Type Definition for the AAL Parcellation Approach.

    A ``TypedDict`` representing the subkeys (second level keys for "AAL") for the processed AAL
    parcellation produced by the ``TimeseriesExtractor`` or ``CAP`` classes. The structure is as
    follows:

    ::

        {
            "maps": "path/to/parcellation.nii.gz",
            "nodes": ["Precentral_L", "Precentral_R", "Frontal_Sup", "Frontal_Sup_R"],
            "regions": ["Precentral", "Frontal_Sup"]
        }

    Parameters
    ----------
    maps: :obj:`str`
        Path to the AAL parcellation.

    nodes: :obj:`list[str]`
        List of nodes (ROIs) in the AAL parcellation. Ordered in ascending order of their label ID
        in the parcellation and must exclude "Background".

    regions: :obj:`list[str]`
        List of networks in the AAL parcellation. **Important**: For certain visualization methods,
        the ``in`` operator is used to determine which nodes belong to which region. Therefore,
        region names must be contained within the corresponding node names (e.g.,
        "Frontal_Sup" region should have nodes with "Frontal_Sup" in their names).

    metadata: :obj:`dict[str, Any]`
        Dictionary containing metadata information about the parcellation. This key is purely
        informational and can be removed, modified, or extended.

        .. versionadded:: 0.32.2

    See Also
    --------
    ParcelApproach
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation
        approaches.
        (See `ParcelApproach Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.ParcelApproach.html#neurocaps.typing.ParcelApproach>`_)
    """

    regions: NotRequired[list[str]]


class CustomRegionHemispheres(TypedDict):
    """
    Type Definition for Hemisphere Mapping in Custom Parcellation Regions.

    A ``TypedDict`` representing the mapping of the index position of the "nodes" to the left and
    right hemispheres.

    ::

        {"lh": [0, 1], "rh": [3, 4, 5]}

    Parameters
    ----------
    lh: :obj:`list[int] | range`
       List of integers or range representing the index positions of elements in the "nodes" list
       belonging to the left hemisphere of a specific region.

    rh: :obj:`list[int] | range`
       List of integers or range representing the index positions of elements in the "nodes" list
       belonging to the right hemisphere of a specific region.

    See Also
    --------
    CustomParcelApproach
       The type definition for the Custom parcellation approach.
       (See `CustomParcelApproach Documentation
       <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.CustomParcelApproach.html#neurocaps.typing.CustomParcelApproach>`_)

    Note
    ----
    For additional information, refer to `NeuroCAPs Parcellation Documentation
    <https://neurocaps.readthedocs.io/en/stable/user_guide/parcellations.html>`_
    """

    lh: Required[Union[list[int], range]]
    rh: Required[Union[list[int], range]]


class CustomParcelApproach(ParcelApproachBase):
    """
    Type Definition for the Custom Parcellation Approach.

    A ``TypedDict`` representing the subkeys (second level keys for "Custom") for the user-defined
    deterministic "Custom" parcellation approach. The structure is as follows:

    ::

        # Lateralized regions
        {
            "maps": "path/to/parcellation.nii.gz",
            "nodes": ["LH_Vis1", "LH_Vis2", "LH_Hippocampus", "RH_Vis1", "RH_Vis2", "RH_Vis3", "RH_Hippocampus"],
            "regions": {
                "Visual": CustomRegionHemispheres
                "Hippocampus": CustomRegionHemispheres
            }
        }

        # Non-lateralized regions
        {
            "maps": "path/to/parcellation.nii.gz",
            "nodes": ["Vis1", "Vis2", "Vis3", "Hippocampus"],
            "regions": {
                "Visual": range(3)
                "Hippocampus": [3]
            }
        }

        # Mixture of lateralized and non-lateralized regions
        {
            "maps": "path/to/parcellation.nii.gz",
            "nodes": ["Vis1", "Vis2", "Vis3", "LH_Vis2", "LH_Hippocampus"],
            "regions": {
                "Visual": range(3)
                "Hippocampus": CustomRegionHemispheres
            }
        }

    Parameters
    ----------
    maps: :obj:`str`
        Path to the Custom parcellation.

    nodes: :obj:`list[str]`
        List of nodes (ROIs) in the Custom parcellation. Ordered in ascending order of their label
        ID in the parcellation and must exclude "Background".

    regions: :obj:`dict[str, list[int] | range]` or :obj:`dict[str, CustomRegionHemispheres]`
        Dictionary mapping the regions to a list integers (or range) representing the index
        positions of elements in the "nodes" list belonging to the region or a dictionary mapping
        the region to a dictionary.

        .. note::
           The use of ``CustomRegionHemispheres`` to define lateralized regions (i.e., with "lh"
           and "rh" keys) is only relevant when calling ``CAP.caps2plot`` with the
           ``add_custom_node_labels`` kwarg set to ``True``. This information allows for the
           creation of simplified axis labels that include hemisphere information. In all other
           methods, the lateralization structure is ignored.

        .. versionchanged:: 0.30.0
           "regions" subkey can now be of type `dict[str, list[int] | range]` for non-lateralized
           regions.

    metadata: :obj:`dict[str, Any]`
        Dictionary containing metadata information about the parcellation. This key is purely
        informational and can be removed, modified, or extended.

        .. versionadded:: 0.32.2

    See Also
    --------
    CustomRegionHemispheres
        Type definition of the Custom hemisphere dictionary for the "regions" subkeys.
        (See `CustomRegionHemispheres Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.CustomRegionHemispheres.html#neurocaps.typing.CustomRegionHemispheres>`_)

    ParcelApproach
        Type definition representing the structure of the Schaefer, AAL, and Custom parcellation
        approaches.
        (See `ParcelApproach Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.ParcelApproach.html#neurocaps.typing.ParcelApproach>`_)

    Note
    ----
    For additional information, refer to `NeuroCAPs Parcellation Documentation
    <https://neurocaps.readthedocs.io/en/stable/user_guide/parcellations.html>`_
    """

    regions: NotRequired[
        Union[dict[str, Union[list[int], range]], dict[str, CustomRegionHemispheres]]
    ]


ParcelApproach = Union[
    dict[Literal["Schaefer"], SchaeferParcelApproach],
    dict[Literal["AAL"], AALParcelApproach],
    dict[Literal["Custom"], CustomParcelApproach],
]
"""
  Type Definition for the Parcellation Approaches.

   A dictionary mapping the Schaefer, AAL, and Custom parcellation approaches to their associated
   subkeys:

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
        (See `SchaeferParcelApproach Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.SchaeferParcelApproach.html#neurocaps.typing.SchaeferParcelApproach>`_)

    :class:`neurocaps.typing.AALParcelApproach`
        Type definition representing the structure of the AAL parcellation approach.
        (See `AALParcelApproach Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.AALParcelApproach.html#neurocaps.typing.AALParcelApproach>`_)

    :class:`neurocaps.typing.CustomParcelApproach`
        Type definition representing the structure of the Custom parcellation approach.
        (See `CustomParcelApproach Documentation
        <https://neurocaps.readthedocs.io/en/stable/api/generated/neurocaps.typing.CustomParcelApproach.html#neurocaps.typing.CustomParcelApproach>`_)
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
