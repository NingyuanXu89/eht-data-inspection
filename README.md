# EHT Data Inspection

Utilities for inspecting EHT/VLBI data products across HOPS pipeline stages.

The package is organized around three product types:

- HOPS/fourfit fringe files
- HOPS `alist` summary files
- UVFITS visibility files

The code in `src/eht_inspection/` is extracted from the original working
notebooks and helper modules while preserving the scientific calculations and
plot conventions.

See [DATA_DIAGNOSTICS.md](DATA_DIAGNOSTICS.md) for stage-by-stage guidance on
ALIST, UVFITS, and individual fringe-file diagnostics.

## Install

From the repository root:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Optional EHT/HOPS functionality still requires the external EAT/HOPS
environment used by the original notebooks.

`eat`, `ehtim`, and `astropy` are Python package dependencies. The external
HOPS tools and sourced HOPS runtime are still required only for mk4/fringe-file
operations such as `fplot`, `spectrum`, and type-120/type-212 access.

## Example Imports

```python
from eht_inspection.alist import load_alist, summarize_delay_outliers
from eht_inspection.coherence import coherence_ratio
from eht_inspection.plotting import (
    plot_scan_bandpass_all_baselines,
    plot_snr_across_stages,
    plot_uv_coverage,
)
from eht_inspection.uvfits import load_obs_uvfits, build_scan_coherency_matrix_from_uvfits
```

## Basic Usage

Load and compare alist stages:

```python
from eht_inspection.alist import load_alist
from eht_inspection.plotting import plot_snr_across_stages

stage3 = load_alist(stage=3, source="M87", multi_stage=True, data_dir="/path/to/alists")
stage5 = load_alist(stage=5, source="M87", multi_stage=True, data_dir="/path/to/alists")

fig = plot_snr_across_stages(
    {"stage 3": stage3, "stage 5": stage5},
    filters={"baseline": ("==", "LN"), "polarization": ("==", "LL")},
)
```

Compute coherence diagnostics:

```python
from eht_inspection.coherence import coherence_ratio, select_low_coherence_high_snr

c_2s = coherence_ratio(avg_scan_df, avg_2s_df)
outliers = select_low_coherence_high_snr(
    avg_scan_df,
    avg_2s_df,
    threshold=0.7,
    snr_min=7,
)
```

Check stage 3 independent fringe-fit consistency with triangle sums of MBD
and delay rate. These are troubleshooting diagnostics, not calibrated
visibility closure quantities. Stage 5 fixes fringe-search locations from
station-based delay and rate solutions, so its ALIST MBD/rate triangle sums
close by construction and do not test agreement among independent baseline
fits. Wrong station solutions can still give zero sums while reducing
coherence. Use UVFITS for closure phase and amplitude:

```python
from eht_inspection.alist import (
    compute_alist_closure_triangles,
    plot_alist_closure_vs_scan,
    summarize_alist_closure_outliers,
)

triangles = compute_alist_closure_triangles(stage3, snr_min=7)
fig, axs = plot_alist_closure_vs_scan(triangles, quantity="closure_mbdelay")
fig, axs = plot_alist_closure_vs_scan(triangles, quantity="closure_delay_rate")
flags = summarize_alist_closure_outliers(triangles)
flags["stations"]  # stations shared by flagged triangles
```

Inspect a bandpass scan from an unaveraged UVFITS file with multiple IFs
and one channel per IF:

```python
from eht_inspection.uvfits import build_scan_coherency_matrix_from_uvfits
from eht_inspection.plotting import plot_scan_bandpass_all_baselines

scan = build_scan_coherency_matrix_from_uvfits(
    "example_unaveraged.uvfits", scannum=0, unaveraged=True,
)

fig, axs = plot_scan_bandpass_all_baselines(
    scan,
    var="phase",
    scan_num=0,
    average_over_time=True,
    show_snr=True,
)
```

`unaveraged=True` is your declaration of the file's averaging history; it is
not inferred from the filename or headers. `show_snr=True` adds compact
subplot titles such as `AX-GL: 11, 9.5, 1.5, 1.2`. Values are median IF SNRs
of the coherently time-averaged visibility, in exactly the plotted label/legend
order (normally RR, RL, LR, LL). Missing estimates show `N/A`. Existing titles
remain unchanged when `show_snr=False`, the default. SNR titles require
`average_over_time=True` and do not change the plotted curves.

When original selected UVFITS weights are nonpositive or NaN, loading prints
counts and affected locations, for example:

```text
UVFITS flags (selected input; weights <= 0 or NaN):
  Flagged samples: 8/36 (22.22%)
  Affected records: 3/3 (100.00%)
  Dropped records: 1/3 (33.33%)
  AX-GL RL scan 0: 1/3 samples (33.33%); 1/1 affected records; IF 2; channel 0; time 0.100000-0.100000 h
```

A sample is one record/IF/channel/polarization cell; a record is one
baseline/time row. Counts use selected input data, before forced-polarization
changes, and exclude products absent from the file. Affected records may
remain usable; dropped records follow the existing parallel-hand retention
rule. Locations use zero-based scan/IF/channel indices and observation hours;
missing scan information is shown as `unknown`. These counts do not infer
observations absent from the file. See [UVFITS diagnostics](DATA_DIAGNOSTICS.md#uvfits-diagnostics)
for uncertainty assumptions and interpretation.

Both `load_obs_uvfits` and `build_scan_coherency_matrix_from_uvfits` accept
`print_flag_summary=True` (default). Keep it enabled for the initial load and
pass `print_flag_summary=False` to subsequent scan-loading calls to avoid
repeating the report. This controls only the flag summary, not other loading
messages or NumPy warnings, and does not change the returned data.

Export an `fplot` PDF for a fringe file:

```python
from eht_inspection.fringe import export_fplot_pdf

pdf_path = export_fplot_pdf("data/AX.B.17.43RHPD", outdir="pdf")
```

## Example Workflows

- `notebooks/examples/inspect_alist.ipynb`: alist loading, stage comparison,
  coherence histograms, station/polarization diagnostics, and delay outliers.
- `notebooks/examples/inspect_uvfits.ipynb`: UVFITS loading, uv coverage,
  coherency matrices, bandpass plots, time/scan plots, and closure quantities.
- `notebooks/examples/inspect_fringe.ipynb`: HOPS fringe-file metadata,
  type-212/type-120 access, adhoc correction, spectrum/timeseries, and `fplot`.

## Module Layout

- `fringe.py`: HOPS/fourfit fringe helpers and `fplot` PDF export.
- `alist.py`: alist loading, stage comparison, station/polarization diagnostics,
  and stage 3 MBD/delay-rate triangle checks.
- `uvfits.py`: UVFITS loading, coherency matrices, bandpass/time/scan plots,
  closure products, and visibility DataFrame views.
- `closure.py`: closure phase/amplitude entry points and naming helpers.
- `coherence.py`: `C_2s` calculations and low-coherence/high-SNR selection.
- `plotting.py`: shared plotting entry points and save/marker helpers.
- `filters.py`: reusable DataFrame filters and logical mask helpers.
- `utils.py`: general phase, baseline, triangle/quadrangle, scan, and grid helpers.

## Data

The repository may contain small sample data for examples under `data/sample/`.
Large science data products should generally stay outside package builds.

Default tests use only small mock arrays and DataFrames; they do not require
real EHT data files. Put tiny redistributable fixtures in `data/sample/`.
Keep full UVFITS files, HOPS fringe products, correlator files, and complete
pipeline alist dumps outside version control.

## Optional Dependencies

Python dependencies are listed in `pyproject.toml`. The package depends on
`eat`, `ehtim`, `astropy`, `numpy`, `pandas`, and `matplotlib`.

The external HOPS runtime is not installed by pip. Source the HOPS environment
before using mk4/fringe-file operations such as `fplot`, `spectrum`,
`timeseries`, `load_type212`, or `load_type120`.

## Current Limitations

- Fringe-file inspection depends on local HOPS/EAT configuration and available
  companion correlator files for some type-120 operations.
- Large science data are intentionally not part of the default package or test
  suite.
