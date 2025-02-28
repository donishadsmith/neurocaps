import logging, os, shutil, sys, tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

import pytest

from .utils import get_paths


@pytest.fixture(scope="module")
def tmp_dir():
    """Create temporary directory for each test module."""
    temp_dir = tempfile.TemporaryDirectory()

    yield temp_dir

    temp_dir.cleanup()


@pytest.fixture(autouse=False, scope="module")
def data_dir(tmp_dir):
    """Copies the test dataset to the temporary directory."""
    work_dir = os.path.dirname(__file__)

    # Copy test data to temporary directory
    shutil.copytree(os.path.join(work_dir, "data", "dset"), os.path.join(tmp_dir.name, "data", "dset"))


@pytest.fixture(autouse=False, scope="module")
def get_vars(tmp_dir):
    """Set up test environment with required data."""
    # Get paths
    return get_paths(tmp_dir.name)


@pytest.fixture(autouse=True)
def logger():
    """Logging fixture."""
    yield logging.getLogger()


@pytest.fixture(autouse=True)
def set_logging_level(request):
    """Fixture to enable/disable logs."""
    # Checks if function has the "enable_logs" marker
    if not request.node.get_closest_marker("enable_logs"):
        # Only critical logs enabled when marker not present
        logging.getLogger().setLevel(logging.CRITICAL)
    else:
        logging.getLogger().setLevel(logging.INFO)

    yield
