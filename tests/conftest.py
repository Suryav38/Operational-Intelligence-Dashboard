import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))

@pytest.fixture(scope='session')
def data():
    import fab_data
    return fab_data.load()
