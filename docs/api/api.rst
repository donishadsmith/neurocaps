API
===

Publicly available classes, functions, exceptions, types, and utility functions within NeuroCAPs.

.. toctree::
   :hidden:
   :maxdepth: 1

   exceptions
   extraction
   analysis
   typing
   utils

.. autosummary::
   :nosignatures:

   neurocaps.exceptions.BIDSQueryError
   neurocaps.exceptions.NoElbowDetectedError
   neurocaps.exceptions.UnsupportedFileExtensionError
   neurocaps.extraction.TimeseriesExtractor
   neurocaps.analysis.CAP
   neurocaps.analysis.change_dtype
   neurocaps.analysis.merge_dicts
   neurocaps.analysis.standardize
   neurocaps.analysis.transition_matrix
   neurocaps.typing.SubjectTimeseries
   neurocaps.typing.ParcelConfig
   neurocaps.typing.SchaeferParcelConfig
   neurocaps.typing.AALParcelConfig
   neurocaps.typing.ParcelApproach
   neurocaps.typing.SchaeferParcelApproach
   neurocaps.typing.AALParcelApproach
   neurocaps.typing.CustomParcelApproach
   neurocaps.utils.fetch_preset_parcel_approach
   neurocaps.utils.generate_custom_parcel_approach
