import pandas as pd
import pytest

from eht_inspection.filters import combine_masks, equality_filters, filter_df, invert_mask


def test_filter_df_basic_operations():
    df = pd.DataFrame(
        {
            "source": ["M87", "M87", "3C273"],
            "baseline": ["AB", "AC", "AB"],
            "snr": [5.0, 12.0, 20.0],
        }
    )

    out = filter_df(
        df,
        {
            "source": ("==", "M87"),
            "snr": (">=", 10.0),
        },
    )

    assert out["baseline"].tolist() == ["AC"]


def test_filter_df_in_and_errors():
    df = pd.DataFrame({"baseline": ["AB", "AC", "BD"]})
    out = filter_df(df, {"baseline": ("in", ["AB", "BD"])})
    assert out["baseline"].tolist() == ["AB", "BD"]

    with pytest.raises(KeyError):
        filter_df(df, {"missing": ("==", 1)})
    with pytest.raises(ValueError):
        filter_df(df, {"baseline": ("contains", "A")})


def test_logical_filter_helpers():
    a = pd.Series([True, True, False])
    b = pd.Series([True, False, True])

    assert combine_masks(a, b, how="and").tolist() == [True, False, False]
    assert combine_masks(a, b, how="or").tolist() == [True, True, True]
    assert invert_mask(a).tolist() == [False, False, True]
    assert equality_filters(source="M87") == {"source": ("==", "M87")}

    with pytest.raises(ValueError):
        combine_masks(a, how="xor")

