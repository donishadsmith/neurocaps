API
===

Publicly available modules in NeuroCAPs.

.. toctree::
   :hidden:
   :maxdepth: 1

   exceptions
   extraction
   analysis
   typing
   utils

.. list-table::

   * - Module
     - Description
   * - :doc:`exceptions`
     - Definitions for custom exceptions for BIDS querying error, elbow method error, unsupported
       file formats
   * - :doc:`extraction`
     - Timeseries extraction, quality control, and BOLD visualization
   * - :doc:`analysis`
     - Co-activation patterns analysis, merging timeseries, ROI standardization, and changing timeseries dtype
   * - :doc:`typing`
     - Type definitions for ``SubjectTimeseries``, ``ParcelConfig``, and ``ParcelApproach``
   * - :doc:`utils`
     - Fetching preset parcellation approaches, creating parcellation approaches from tabular
       metadata, data simulation, and plotting defaults
