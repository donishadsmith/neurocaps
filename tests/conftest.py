import logging, os, shutil, sys, tempfile

# Assign null handler to root to reduce logged output when using `pytest -s`
logging.getLogger().addHandler(logging.NullHandler())

import matplotlib, pytest

from .utils import get_paths


@pytest.fixture(autouse=False, scope="module")
def tmp_dir():
    """Create temporary directory for each test module."""
    temp_dir = tempfile.TemporaryDirectory()

    yield temp_dir

    temp_dir.cleanup()


@pytest.fixture(autouse=False, scope="function")
def data_dir(tmp_dir):
    """
    Copies the test data to the temporary directory, then removes the "data" folder, while leaving
    the temporary directory to minimize cross-test contamination.
    """
    work_dir = os.path.dirname(__file__)

    # Copy test data to temporary directory
    shutil.copytree(os.path.join(work_dir, "data"), os.path.join(tmp_dir.name, "data"))

    yield

    # Remove sub directory
    shutil.rmtree(os.path.join(tmp_dir.name, "data"))


@pytest.fixture(autouse=False, scope="module")
def get_vars(tmp_dir):
    """Set up test environment with required data."""
    # Get paths
    return get_paths(tmp_dir.name)


@pytest.fixture(autouse=False, scope="session")
def logger():
    """Logging fixture."""
    LG = logging.getLogger("Test_Monitor")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")

    formatter = logging.Formatter("%(asctime)s %(name)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    LG.addHandler(handler)

    yield LG


@pytest.fixture(autouse=True, scope="session")
def use_agg_backend():
    """Changes backend for matplotlib to prevent random tcl errors."""
    matplotlib.use("Agg")
    return


@pytest.fixture(autouse=True, scope="session")
def create_data_directories():
    """
    Function to copy nilearn, neuromaps, and neurocaps data to home directory to prevent
    file fetching from OSF or other websites.
    """
    curr_dir = os.path.join(os.path.dirname(__file__), "data")

    target_nilearn = os.path.expanduser("~/nilearn_data")
    if not os.path.isdir(target_nilearn):
        shutil.copytree(os.path.join(curr_dir, "nilearn_data"), target_nilearn)

    target_neuromaps = os.path.expanduser("~/neuromaps-data")
    if not os.path.isdir(target_neuromaps):
        shutil.copytree(os.path.join(curr_dir, "neuromaps-data"), target_neuromaps)

    target_neurocaps = os.path.expanduser("~/neurocaps_data")
    if not os.path.isdir(target_neurocaps):
        shutil.copytree(os.path.join(curr_dir, "neurocaps_data"), target_neurocaps)
