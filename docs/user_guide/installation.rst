Installation
============
To install neurocaps, follow the instructions below using your preferred terminal.

Standard Installation from PyPi
-------------------------------
.. code-block:: bash

    pip install neurocaps

Windows Users
^^^^^^^^^^^^^
To avoid installation errors related to long paths not being enabled, pybids will not be installed by default.
Refer to official `Microsoft documentation <https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell>`_
to enable long paths.

To include pybids in your installation, use:

.. code-block:: bash

    pip install neurocaps[windows]


Alternatively, you can install pybids separately:

.. code-block:: bash

    pip install pybids

Installation from Source (Development Version)
----------------------------------------------
To install the latest development version from the source, there are two options:

1. Install directly via pip:

.. code-block:: bash

    pip install git+https://github.com/donishadsmith/neurocaps.git


2. Clone the repository and install locally:

.. code-block:: bash

    git clone https://github.com/donishadsmith/neurocaps/
    cd neurocaps
    pip install -e .

Windows Users
^^^^^^^^^^^^^
To include pybids when installing the development version on Windows, use:

.. code-block:: bash

    git clone https://github.com/donishadsmith/neurocaps/
    cd neurocaps
    pip install -e .[windows]

Docker
------
If `Docker <https://docs.docker.com/>`_ is available on your system, you can use the neurocaps Docker image, which
includes the demos and configures a headless display for VTK.

To pull the Docker image:

.. code-block:: bash

    docker pull donishadsmith/neurocaps && docker tag donishadsmith/neurocaps neurocaps

The image can be run as:

1. An interactive bash session (default):

.. code-block:: bash

    docker run -it neurocaps

2. A Jupyter Notebook with port forwarding:

.. code-block:: bash

    docker run -it -p 9999:9999 neurocaps notebook
