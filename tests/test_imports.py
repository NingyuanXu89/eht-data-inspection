def test_package_imports():
    import eht_inspection
    assert eht_inspection.__version__


def test_lightweight_helpers_import():
    from eht_inspection.filters import combine_masks, equality_filters, filter_df
    from eht_inspection.utils import (
        get_baselines_from_station_list,
        get_subplot_grid,
        scan_ids_from_intervals,
        wrap_phase,
    )
    assert filter_df
    assert equality_filters(source="M87") == {"source": ("==", "M87")}
    assert get_subplot_grid(5) == (2, 3)
    assert get_baselines_from_station_list(["A", "B", "C"]) == ["A-B", "A-C", "B-C"]
    assert scan_ids_from_intervals([0.1, 0.9, 1.1], [(0, 1), (1, 2)]).tolist() == [0, 0, 1]
    assert wrap_phase(190) == -170
    assert round(wrap_phase(3.5, rad=True, return_degrees=True), 6) == -159.464772
    assert combine_masks(True, False, how="or")


def test_fringe_split_record_loaders_import():
    from eht_inspection.fringe import load_type120, load_type212
    assert load_type120
    assert load_type212


def test_template_alignment_imports():
    from eht_inspection.closure import (
        quadrangle_relation_label,
        triangle_names_from_station_list,
    )
    from eht_inspection.plotting import default_pol_marker_map, save_figure
    from eht_inspection.uvfits import visibility_arrays_to_dataframe

    assert triangle_names_from_station_list(["A", "B", "C"]) == [("A", "B", "C")]
    assert quadrangle_relation_label(("A", "B", "C", "D")) == "AB * CD / (AC * BD)"
    assert default_pol_marker_map()["RR"] == "x"
    assert save_figure
    assert visibility_arrays_to_dataframe
