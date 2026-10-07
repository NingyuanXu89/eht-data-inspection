"""ALIST filename selection and the existing EAT preprocessing pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from eat.io import hops

from eht_inspection.alist import load_alist


@pytest.fixture
def alist_reader(monkeypatch):
    raw = pd.DataFrame({
        "source": ["M87", "M87", "1921-293", "OTHER"],
        "baseline": ["JL", "JJ", "JP", "LX"],
        "polarization": ["RR"] * 4,
        "root_id": ["aaaaaa"] * 4,
        "snr": [10.] * 4, "amp": [1.] * 4,
        "expt_no": [3880] * 4, "ref_freq": [230000.] * 4,
        "year": [2026] * 4, "freq_code": ["W16"] * 4,
        "scan_id": ["No0001", "No0001", "No0002", "No0003"],
        "timetag": ["100-010203"] * 4,
    })
    paths = []

    def read(path):
        paths.append(Path(path))
        return raw.copy(deep=True)

    monkeypatch.setattr(hops, "read_alist", read)
    return paths, raw


@pytest.mark.parametrize("kwargs, relative_path", [
    ({}, "alist.v6"),
    ({"avg_time": 2}, "alist.v6.2s.avg"),
    ({"multi_freq": True}, "230GHz/alist.v6"),
    ({"multi_freq": True, "freq": 345, "avg_time": 10}, "345GHz/alist.v6.10s.avg"),
])
def test_alist_filenames_and_preprocessing(tmp_path, alist_reader, kwargs, relative_path):
    paths, raw = alist_reader
    before = raw.copy(deep=True)
    data = load_alist(data_dir=tmp_path / "stage3", **kwargs)
    assert paths == [tmp_path / "stage3" / relative_path]
    assert data.source.tolist() == ["M87"]
    assert data.baseline.tolist() == ["JL"]
    assert data.scan_no.tolist() == [0]
    np.testing.assert_allclose(data.days, [99 + 1/24 + 2/1440 + 3/86400])
    np.testing.assert_allclose(data.snr, [10])
    pd.testing.assert_frame_equal(raw, before)


def test_alist_default_directory_and_source_fix(monkeypatch, tmp_path, alist_reader):
    monkeypatch.chdir(tmp_path)
    paths, _ = alist_reader
    data = load_alist(source="J1924-2914")
    assert paths == [Path("alist.v6")]
    assert data.source.tolist() == ["J1924-2914"]
    assert data.baseline.tolist() == ["JP"]
    assert data.scan_no.tolist() == [1]


def test_alist_stage_directories_are_independent(tmp_path, alist_reader):
    paths, _ = alist_reader
    for stage in ("stage3", "stage5"):
        load_alist("M87", data_dir=tmp_path / stage)
    assert paths == [tmp_path / "stage3/alist.v6", tmp_path / "stage5/alist.v6"]
