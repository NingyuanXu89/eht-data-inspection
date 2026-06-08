# EHT Data Inspection

Utilities for inspecting EHT/VLBI data products across HOPS pipeline stages.

The package is organized around three product types:

- HOPS/fourfit fringe files
- HOPS `alist` summary files
- UVFITS visibility files

The code in `src/eht_inspection/` is extracted from the original working
notebooks and helper modules while preserving the scientific calculations and
plot conventions.

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

Inspect a UVFITS scan:

```python
from eht_inspection.uvfits import build_scan_coherency_matrix_from_uvfits
from eht_inspection.plotting import plot_scan_bandpass_all_baselines

scan = build_scan_coherency_matrix_from_uvfits("example.uvfits", scannum=0)

fig, axs = plot_scan_bandpass_all_baselines(
    scan,
    var="phase",
    scan_num=0,
    average_over_time=True,
)
```

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
- `alist.py`: alist loading, stage comparison, station/polarization diagnostics.
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
