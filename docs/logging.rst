Logging
=======

In neurocaps, all informational messages and warnings are managed using Python's logging module. By default, logs are
output to the console (``sys.stdout``) with a logging level of ``INFO``. Before importing the neurocaps package, you can
configure the root handler or specific module loggers to override these default settings. Additionally, the name of
each logger is unique to the module and corresponds to the module's name (``__name__.split(".")[-1]``).

Configuration (Without Parallel Processing)
-------------------------------------------
This configuration sets the root logger to output to the console and configures a specific module logger to use a
``FileHandler``.

.. code-block:: python

    import logging

    # Configure the root logger for all loggers to propagate to
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
        )

    # Configuring the logger for the internal function that does timeseries extraction
    extract_timeseries_logger = logging.getLogger('_extract_timeseries')
    extract_timeseries_logger.setLevel(logging.WARNING)
    file_handler = logging.FileHandler('neurocaps.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    # Import package
    from neurocaps.extraction import TimeseriesExtractor

    # ...code...


Logging Configuration (With Parallel Processing)
------------------------------------------------
When using joblib's loky backend for parallel processing, child processes do not inherit global logging configurations.
Consequently, internal logs are output to the console by default even when parallel processing is enabled. To redirect
these logs to a specific handler, you can set up a ``multiprocessing.Manager().Queue()`` (which is passed to
``QueueHandler`` internally) and ``QueueListener``. This approach allows logs produced by the
internal ``_extract_timeseries`` function to be redirected when parallel processing is enabled. Additionally, the names
of each logger are unique to the module and is the name of the file (``__name__.split(".")[-1]``).

.. code-block:: python

    import logging
    from logging.handlers import QueueListener
    from multiprocessing import Manager

    # Configure root with FileHandler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('neurocaps.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    root_logger.addHandler(file_handler)

    if __name__ == "__main__":
        # Import the TimeseriesExtractor
        from neurocaps.extraction import TimeseriesExtractor

        # Setup managed queue
        manager = Manager()
        queue = manager.Queue()

        # Set up the queue listener
        listener = QueueListener(queue, *root_logger.handlers)

        # Start listener
        listener.start()

        extractor = TimeseriesExtractor()
        
        # Use the `parallel_log_config` parameter to pass queue and the logging level 
        extractor.get_bold(bids_dir="path/to/bids/dir",
                        task="rest", 
                        tr=2,
                        n_cores=5,
                        parallel_log_config = {"queue": queue, "level": logging.WARNING})

        # Stop listener
        listener.stop()
