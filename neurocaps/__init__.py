"""
Co-Activation Patterns (CAPs) for resting-state and task-based fMRI data.
-------------------------------------------------------------------------
Documentation and comprehensive tutorials can be found at https://neurocaps.readthedocs.io.

Submodules
----------
extraction -- Contains ``TimeseriesExtractor`` class for timeseries extraction quality control, and BOLD visualization

analysis -- Class and functions for co-activation patterns analysis, merging timeseries ROI standardization,
and changing timeseries dtype

typing -- Type definitions for ``SubjectTimeseries``, ``ParcelConfig``, and ``ParcelApproach``

utils -- Tools for fetching preset parcellation approaches, creating parcellation approaches from
tabular metadata, data simulation, and plotting defaults
"""

from . import analysis, extraction, exceptions, utils

# Don't include "typing"
__all__ = ["analysis", "extraction", "exceptions", "utils"]

# Version in single place
__version__ = "0.37.1"
