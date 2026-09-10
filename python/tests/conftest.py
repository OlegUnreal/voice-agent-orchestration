import pytest


@pytest.fixture(autouse=True)
def helix_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HELIX_DATA_DIR", str(tmp_path))
