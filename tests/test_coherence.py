import numpy as np
import pandas as pd

from eht_inspection.coherence import coherence_ratio, select_low_coherence_high_snr


def test_coherence_ratio():
    avg_scan = pd.DataFrame({"amp": [5.0, 2.0, 1.0]})
    avg_short = pd.DataFrame({"amp": [10.0, 2.0, 4.0]})

    np.testing.assert_allclose(coherence_ratio(avg_scan, avg_short), [0.5, 1.0, 0.25])


def test_select_low_coherence_high_snr_from_short_average():
    avg_scan = pd.DataFrame(
        {
            "amp": [5.0, 9.0, 1.0],
            "snr": [10.0, 20.0, 4.0],
            "baseline": ["AB", "AC", "AD"],
        }
    )
    avg_short = pd.DataFrame({"amp": [10.0, 10.0, 10.0]})

    out = select_low_coherence_high_snr(
        avg_scan,
        avg_short,
        threshold=0.7,
        snr_min=7.0,
    )

    assert out["baseline"].tolist() == ["AB"]


def test_select_low_coherence_high_snr_from_existing_column():
    df = pd.DataFrame({"C_2s": [0.6, 0.8], "snr": [8.0, 20.0]})
    out = select_low_coherence_high_snr(df, threshold=0.7, snr_min=7.0)
    assert out.index.tolist() == [0]

