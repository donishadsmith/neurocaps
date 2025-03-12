API
===

:mod:`neurocaps.exceptions` - Exceptions
----------------------------------------
Module containing custom exceptions (available in versions >= 0.22.2).

.. currentmodule:: neurocaps.exceptions

.. autosummary::
   :template: exceptions.rst
   :toctree: generated/

   BIDSQueryError

:mod:`neurocaps.extraction` - Timeseries Extraction
---------------------------------------------------
Module containing the ``TimeseriesExtractor`` class for extracting timeseries data from preprocessed BIDS datasets
(using pipelines such as fMRIPrep), as well as for pickling (serializing) and visualizing the extracted data.

.. currentmodule:: neurocaps.extraction

.. autosummary::
   :template: class.rst
   :toctree: generated/

   TimeseriesExtractor

:mod:`neurocaps.analysis` - Co-Activation Patterns (CAPs) Analysis
-------------------------------------------------------------------
Module containing the ``CAP`` class for performing and visualizing CAPs analyses, as well as functions for changing
dtype, merging, and standardizing timeseries data, and creating averaged transition probability matrices from the
participant-wise *transition probabilities* dataframes created by ``CAP.calculate_metrics``.

.. currentmodule:: neurocaps.analysis

.. autosummary::
   :template: class.rst
   :toctree: generated/

   CAP

.. autosummary::
   :template: function.rst
   :toctree: generated/

   change_dtype
   merge_dicts
   standardize
   transition_matrix
