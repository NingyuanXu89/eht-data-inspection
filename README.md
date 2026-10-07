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
    source="M87", obs_day="3888",
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

To summarize frequency scatter after the same time averaging, use
`summarize_scan_bandpass`. It returns one row per scan/baseline/polarization,
with real and imaginary std/p16/p50/p84, median IF SNR, RMS propagated thermal
uncertainty, and usable/total IF counts (for example, 17/32). It uses the same
baseline and polarization order as the plot, excluding autocorrelations by
default. SNR medians agree with the subplot titles, which round for display;
the table retains full precision. The function does not print or save.

This complete notebook example collects the scans of one unaveraged file,
displays the first rows, and exports a CSV using caller-chosen names:

```python
from pathlib import Path
import pandas as pd
from IPython.display import display
from eht_inspection.uvfits import (
    load_obs_uvfits,
    build_scan_coherency_matrix_from_uvfits,
    summarize_scan_bandpass,
)

filename = Path("example_unaveraged.uvfits")  # Multiple IFs, one channel per IF.
savedir = "bandpass_summary"
savename = "example_bandpass_summary.csv"
obs = load_obs_uvfits(filename, return_dict=True, include_scan_ids=True)
scan_ids = obs["scan_ids"]
tables = []
for scan_num in sorted(set(scan_ids[scan_ids >= 0])):
    scan = build_scan_coherency_matrix_from_uvfits(
        filename, scannum=scan_num, scan_ids=scan_ids,
        unaveraged=True, print_flag_summary=False,
    )
    tables.append(summarize_scan_bandpass(scan, obs_day=filename.name))
table = pd.concat(tables, ignore_index=True)
display(table.head())
output_path = Path(savedir) / savename
output_path.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(output_path, index=False)
```

This example uses file scan metadata; if unavailable, supply your scan IDs or
intervals to the existing loading workflow. `obs_day` is a caller-supplied
file/day label; omitted labels remain missing. For one loaded scan, simply
call `summarize_scan_bandpass(scan)` and display the returned DataFrame.

The real/imaginary statistics describe the time-averaged complex signals
across usable IFs, rather than amplitude/phase statistics or another frequency
average. Std uses `ddof=1`; percentiles use linear interpolation. All statistics
use the same usable IFs with finite means and finite positive uncertainties
for every contributing integration. No usable IFs give NaN statistics; one
usable IF gives NaN std but available percentiles, SNR, and thermal RMS.
Thermal RMS is `sqrt(mean(sigma_mean**2))`, a per-real/imaginary-component
noise reference assuming calibrated inverse-variance weights and independent
thermal noise. Bandpass structure and phase slopes can also increase scatter.
See [frequency-scatter interpretation](DATA_DIAGNOSTICS.md#bandpass-frequency-scatter-summary).

When original selected UVFITS weights are nonpositive or NaN, loading prints
counts and affected locations, for example:

```text
UVFITS flags (selected input; weights <= 0 or NaN):
  Flagged samples: 8/36 (22.22%)
  Affected records: 3/3 (100.00%)
  Dropped records: 1/3 (33.33%)
  AX-GL RL time 00:00:00-00:30:00: 1/3 samples (33.33%); 1/1 affected records; IF 2
```

A sample is one record/IF/channel/polarization cell; a record is one
baseline/time row. Counts use selected input data, before forced-polarization
changes, and exclude products absent from the file. Affected records may
remain usable; dropped records follow the existing parallel-hand retention
rule. Locations use full NX intervals rounded to `HH:MM:SS` and zero-based IF indices;
missing scan information is shown as `unknown`. These counts do not infer
observations absent from the file. See [UVFITS diagnostics](DATA_DIAGNOSTICS.md#uvfits-diagnostics)
for uncertainty assumptions and interpretation.

File-based scan assignments use each NX row's one-based inclusive `START VIS`
and `END VIS` ranges on the **original input records**, before filtering.
This prevents overlapping NX time windows from moving a baseline's records
into a neighboring scan. Scan IDs are zero-based NX row indices, shared across
baselines; they do not enumerate each baseline's scans separately. If record
ranges are missing, time intervals provide a fallback; invalid/overlapping
record ranges also fall back with a warning. Uncovered records remain unknown.

Request `include_scan_ids=True, return_dict=True` to get `uvdata["scan_ids"]`,
`uvdata["scan_start"]`, and `uvdata["scan_end"]` aligned with the retained
visibility rows. Times are rounded NX boundaries, without wrapping hours at 24.
The IDs index retained NX rows; they are not original observing-schedule scan
numbers when scans were omitted from the file. The scan-building wrapper uses
these IDs by default; explicitly supplied `scan_ids` or `scans` still override
its automatic assignments. `scantable` continues to contain the original NX
time windows. Calling `scan_ids_from_intervals(times, scantable)` can disagree
with NX record membership when those windows overlap. Similarly,
`ehtim`'s `obs.add_scans()` infers scans from time gaps and may merge adjacent
NX entries or give them different numbers. Use a consistent scan definition
when comparing bandpass summaries and flag contribution plots.

The wrapper attaches the selected records' shared `scan_start` and `scan_end`
to its result; custom selections spanning different NX intervals raise an error.
Visibility and closure plots require these known time labels. Titles use the
start time, and filenames use `HHMMSS`, alongside `source` and optional
caller-supplied `obs_day`, e.g. `M87_3888_013200_phase_vs_channel_all_baselines.png`.
For a direct array-based builder call, attach the labels before plotting:

```python
selected = scan_ids == scannum
result["scan_start"] = uvdata["scan_start"][selected][0]
result["scan_end"] = uvdata["scan_end"][selected][0]
```

The bandpass summary places these labels after `obs_day` and keeps the indexing
`scan_num` as its last column. Unknown metadata raises an error instead of
substituting surviving timestamps. Flag-contribution plots continue to use IDs.

Both `load_obs_uvfits` and `build_scan_coherency_matrix_from_uvfits` accept
`print_flag_summary=True` (default). Keep it enabled for the initial load and
pass `print_flag_summary=False` to subsequent scan-loading calls to avoid
repeating the report. This controls only the flag summary, not other loading
messages or NumPy warnings, and does not change the returned data.

To see which scans, baselines, and polarizations contribute most to flagging,
keep the original counts in an optional table and make two contribution plots:

```python
from eht_inspection.uvfits import load_obs_uvfits
from eht_inspection.plotting import plot_uvfits_flag_contributions

uvdata = load_obs_uvfits(
    filename, IF=all, return_dict=True, include_flag_summary=True,
)
flag_summary = uvdata["flag_summary"]
plots = plot_uvfits_flag_contributions(flag_summary)
fig_scan, axes_scan = plots["scan"]
fig_baseline, axes_baseline = plots["baseline"]
# Optional exports:
# fig_scan.savefig("flags_per_scan.png", dpi=150, bbox_inches="tight")
# fig_baseline.savefig("flags_per_baseline.png", dpi=150, bbox_inches="tight")
```

Each figure has one panel per polarization present in the input file. In the
scan figure, bars show flagged-sample fractions per scan, split by baseline;
in the baseline figure, bars show fractions per baseline, split by scan.
Every colored segment uses **all selected input samples in that bar and
polarization** as its denominator, so the segments add to the total flagged
fraction rather than to 100%. Labels give the sample count (`n`). Baselines
are ranked by overall flagged fraction; scans remain in chronological order,
with unknown scans last. The five largest contributors by flagged-sample
count are shown separately, with the rest combined as `Other`; change this
with `max_contributors`. All bars and counts remain included.

The table has one row per observed scan/baseline/polarization group, including
zero-flag groups and records dropped by loading. Its `flagged_fraction` column
is the **group's own rate** (0 to 1), useful for finding a small baseline with
a high rate even when its contribution to a larger scan is small:

```python
flag_summary.sort_values("flagged_fraction", ascending=False)[
    ["scan", "baseline", "polarization", "flagged_samples", "total_samples", "flagged_fraction"]
]
```

Flags are original selected weights `<= 0` or NaN, before polarization
forcing. Products absent from the file and observations never recorded are
excluded. These figures measure flagged samples, not affected records.
`include_flag_summary=True` requires `return_dict=True`; the default return
keys, tuple API, visibility filtering, and averaging remain unchanged.

The table also includes `scan_start`, `scan_end`, `finite_nonzero_samples`, and
`finite_nonzero_records`. The audit examines original selected flagged samples
before masking: both real and imaginary components must be finite and at least
one must be nonzero. With printing enabled, per-product audit totals follow
`Dropped records`; any such observations also get grouped location lines.
Use `print_flag_summary=False` to return the table without printing it.

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
