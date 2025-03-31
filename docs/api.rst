API
===

:mod:`neurocaps.exceptions` - Exceptions
----------------------------------------
Module containing custom exceptions.

.. versionadded:: 0.22.2

.. currentmodule:: neurocaps.exceptions

.. autosummary::
   :template: exceptions.rst
   :nosignatures:
   :toctree: generated/

   BIDSQueryError
   NoElbowDetectedError

:mod:`neurocaps.extraction` - Timeseries Extraction
---------------------------------------------------
Module containing the ``TimeseriesExtractor`` class for extracting timeseries data from preprocessed BIDS datasets
(using pipelines such as fMRIPrep), reporting quality control metrics (number of frames scrubbed or interpolated) as
well as for pickling (serializing) and visualizing the extracted data.

.. currentmodule:: neurocaps.extraction

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: generated/

   TimeseriesExtractor

:mod:`neurocaps.analysis` - Co-Activation Patterns (CAPs) Analysis
-------------------------------------------------------------------
Module containing the ``CAP`` class for performing and visualizing CAPs analyses, as well as functions for changing
dtype, merging, and standardizing timeseries data, and creating averaged transition probability matrices from the
participant-wise transition probabilities dataframes created by ``CAP.calculate_metrics()``.

.. currentmodule:: neurocaps.analysis

.. autosummary::
   :template: class.rst
   :nosignatures:
   :toctree: generated/

   CAP

.. autosummary::
   :template: function.rst
   :nosignatures:
   :toctree: generated/

   change_dtype
   merge_dicts
   standardize
   transition_matrix

:mod:`neurocaps.typing` - Types
--------------------------------
Module containing custom types.

.. versionadded:: 0.23.6

.. currentmodule:: neurocaps.typing

.. autosummary::
   :template: types.rst
   :nosignatures:
   :toctree: generated/

   SubjectTimeseries
   ParcelConfig
   SchaeferParcelConfig
   AALParcelConfig
   ParcelApproach
   SchaeferParcelApproach
   AALParcelApproach
   CustomParcelApproach
   CustomRegionHemispheres
