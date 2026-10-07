import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture()
def mock_tf():
    from mock_tokenfactory import MockServer

    with MockServer() as srv:
        yield srv
