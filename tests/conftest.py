import tempfile
import pytest


@pytest.fixture(scope="module")
def tmp_dir():
    """Create temporary directory for each test module."""
    temp_dir = tempfile.TemporaryDirectory()
    yield temp_dir
    temp_dir.cleanup()
