# EHT data diagnostics: ALIST, UVFITS, and fringe files

This guide combines the repository's inspection tools with a review of the
diagnostic checklist supplied by the user. It describes **leads for inspection**,
not pass/fail rules. Compare the same source, scan, baseline, polarization,
frequency setup, and averaging interval before attributing a difference to a
pipeline stage. The EHT-HOPS [pipeline description][ehthops-stages] and
[global fringe solution][global-fringe] provide the stage definitions below.

## Which product answers which question?

| Product | Best use | Main limitation |
| --- | --- | --- |
| Stage 3 ALIST | Survey fringe SNR, scan coherence, delay/rate outliers, and independently fitted baseline MBD/rate triangle sums. | Fit summaries, not calibrated complex visibilities. |
| Stage 5 ALIST | Survey SNR, amplitude, residual phase, and coherence after the global fringe solution. | Standard stage 5 MBD/rate triangle sums close by construction and cannot validate the imposed solution. |
| UVFITS | Inspect complex visibility amplitude/phase and source-sensitive closure phase/amplitude, ideally across time, frequency, and polarization. | Interpretation depends on conversion, calibration, polarization basis, SNR, and averaging. |
| Individual fringe file | Open its `fplot` to diagnose one scan × baseline × polarization fit. | One fit cannot show array-wide or source closure behavior. |

### Stage map

| Stage | Fourfit input and result | New solution derived for the next stage |
| --- | --- | --- |
| 1 | Default flags and search windows; baseline fringe fits. | Phase bandpass. |
| 2 | Fringe fits with phase bandpass. | Ad hoc atmospheric phase corrections. |
| 3 | Fringe fits with ad hoc phases; MBD and delay rate are still fitted on individual baselines. | R−L delay offsets. |
| 4 | Fringe fits with R−L delay offsets. | Global station delay and rate solution. |
| 5 | Final fringe fits with the global solution. In the standard implementation, station differences set zero-width MBD/rate search locations. | Fringe files for conversion. |
| 6 | Mk4 fringe files are converted to UVFITS. | — |
| 7 | UVFITS gets a priori amplitude calibration and field angle rotation. | — |
| 8 | UVFITS gets R/L gain-ratio calibration. | — |

The station model sets a baseline's stage 5 delay and rate to differences such
as `delay_B − delay_A`; it is not a new independent measurement of that
baseline's best delay/rate peak. If the model is inaccurate, coherent amplitude
and SNR can fall even while the reported stage 5 triangle sums remain zero.
An isolated weak stage 5 detection is also not evidence that a new peak was
selected inside a narrow window: a zero-width search evaluates the imposed
location with one effective trial. [Source][global-fringe]

## ALIST diagnostics

Implemented in [`alist.py`](src/eht_inspection/alist.py) and
[`coherence.py`](src/eht_inspection/coherence.py). ALIST `mbdelay` is in µs,
`delay_rate` is in ps/s, and the closure MBD plot converts µs to ns.
ALIST `amp` is a corrected correlation coefficient, not a flux density; do
not compare it numerically with calibrated UVFITS amplitude in Jy.
See the [HOPS units reference][hops-units].

| Plot or table | What to look for | Possible explanations and next check | Useful stages |
| --- | --- | --- | --- |
| `plot_snr_across_stages` | Matched-baseline SNR changes, missing fits, or a station-wide loss after applying a new correction. Ad hoc phase correction often raises scan coherence and SNR, but monotonic improvement is not a rule. | Check phase-bandpass/ad hoc solution, R−L delay, stage 5 imposed fringe location, flags, and the individual `fplot`. A low-SNR stage 5 measurement can be real because the global solution reduces search trials. | 1–5, especially 2→3 and 3/4→5 |
| `plot_coherence_hist`, `plot_coherence_diagnostics`, `coherence_ratio`, `select_low_coherence_high_snr` | `C_2s = A_scan / A_2s`. Low values on well-detected baselines suggest loss when averaging over the scan. Cluster by scan, baseline, and station. | Residual atmospheric phase, poor ad hoc correction, delay/rate error, or instrumental changes. Values above 1 can arise from noise, bias, weighting, or mismatched records; no single cause follows from the ratio. **Match rows explicitly before division**: the helpers divide scalar ALIST amplitudes by row/index and do not align records by scan/baseline keys. | 2–5, chiefly 3 and 5 |
| `plot_quantity_vs_scan_by_station(..., quantities="resid_phas")` and `quantities="amp"` | Phase/amp offsets, jumps, scatter, or station/polarization patterns. `resid_phas` is the baseline fringe phase after fitting, so it need not be zero or flat. | Compare a suspect baseline's `fplot`, then UVFITS phase/time and closure phase. Source structure can contribute to phase; gains and coherence affect amplitude. | 3–5 |
| `plot_quantity_vs_scan_by_station(..., quantities="mbdelay")` and `quantities="delay_rate"` | At stage 3, identify isolated fits that disagree with neighboring scans/polarizations or station-sharing baselines. | Low-SNR false fit, MBD ambiguity, clock/model change, bandpass or R−L delay problem, or source structure. Verify against SNR and `fplot`. Stage 5 values primarily reflect the imposed global solution. | **Stage 3 primary**; 4 for comparison |
| `plot_quantity_vs_scan_by_station(..., quantities="mbdelay - sbdelay")`, `summarize_delay_outliers` | Large or unusual SBD–MBD differences and rate outliers relative to the rest of the matched data. | Check ambiguity, frequency-dependent phase, fit quality, and the actual search panels in `fplot`. There is no universal requirement that SBD–MBD be zero or within a particular SBD resolution; the outlier helper uses empirical thresholds. | 3–5 |
| `add_pol_difference`, `add_rl_double_difference`, `compute_rl_double_difference_table` | RR−LL or cross-hand delay differences; `(RR−RL)−(LR−LL)` tests whether four polarization fits are mutually consistent under a station-based R/L model. | R−L signal-path delay, weak cross-hand fit, polarization labeling/basis, or polarized source effects. A nonzero double difference is a lead, not proof of a particular station fault or a universal failure at every stage. | Stage 3 for R−L investigation; later stages for residuals |
| `compute_alist_closure_triangles`, `plot_alist_closure_vs_scan`, `summarize_alist_closure_outliers` | For each matched stage 3 RR or LL triangle, inspect directed sums `q_AB + q_BC + q_CA` separately for MBD and rate. Start with high-SNR legs and compare scans/triangles. | Inconsistent baseline fits, ambiguity choice, low-SNR leg, baseline-dependent phase, or **source structure** can produce nonzero sums. A `closure_mbdelay_over_ambiguity` value near `+1` or `−1` means the sum is about one *ambiguity spacing* in magnitude, not a certainty of an ambiguity mistake. It is clearest when all three legs have the same spacing. The `stations` table counts flagged triangles containing each station; it does not identify the faulty station. | **Stage 3 only** |

The stage 3 ALIST MBD/rate sums are **fit-consistency diagnostics**, not the
UVFITS visibility closure quantities. Station-based errors cancel from an
ideal directed sum, but source structure can also create closure delay; hence
"nonzero = bad pipeline" and "source contributes almost nothing" are both
unsafe rules. The code uses separate RR/LL panels, excludes HOPS station `A`
(unconverted ALMA) by default, and marks triangles with co-located station
pairs as `trivial`. Its robust outlier score is relative to observed scatter,
not a measurement uncertainty. [Source-structure evidence][source-closure-delay]

## UVFITS diagnostics

Implemented in [`uvfits.py`](src/eht_inspection/uvfits.py) and
[`plotting.py`](src/eht_inspection/plotting.py). Build a coherency matrix with
`build_scan_coherency_matrix_from_uvfits` before using the per-scan plots.
The closure builder `build_closure_products_from_coherency` computes, for
separate RR and LL products,

```text
closure phase = arg(V_AB V_BC V_CA)
closure log-amplitude = log|V_AB| + log|V_CD| − log|V_AC| − log|V_BD|
closure amplitude = exp(closure log-amplitude)
```

Ideal station phase and gain terms cancel in these expressions. Source
structure can make closure phase nonzero and closure amplitude different from
1. The code plots log closure amplitude around a reference of 0 by default.

| Plot or table | What to look for | Possible explanations and next check | Input requirement |
| --- | --- | --- | --- |
| `plot_uv_coverage` | Missing expected scan/baseline samples relative to the schedule and array. | Dropout, flagging, missing fringe/conversion, or intentionally absent observations. Geometry and schedule determine the expected tracks. | `u`, `v` from UVFITS |
| `plot_amp_uvdist_whole_dataset` | Amplitude trends with baseline length and differences among RR, LL, RL, LR; check station-sharing tracks. | Source structure and polarization **or** gain/SEFD, opacity, pointing, coherence, R/L gain ratio, leakage, or mixed basis. RR=LL and tiny cross-hands are not universal requirements. This helper plots `abs(V)` without averaging. | Complex UVFITS visibilities |
| `plot_scan_bandpass_all_baselines` | Phase slope/step, channel-dependent amplitude, missing channels, or one polarization that differs. | Residual delay, phase/amplitude bandpass, flags, interference, or source effects. A raw bandpass need not be perfectly flat. The function can use any coherency result retaining channels; `average_over_time=True` coherently averages complex values across the scan. Channel-averaged data cannot reveal a channel spectrum. | Channel-resolved UVFITS |
| `plot_result_vs_time_all_baselines` | Phase drift/jump and amplitude changes within one scan for the selected channel. | Residual rate/ad hoc phase error, atmosphere, instrumental change, source evolution with uv position, or gain/coherence loss. Neither phase nor amplitude is universally flat. | Time-resolved UVFITS |
| `plot_results_vs_scan_all_baselines` | Scan-to-scan changes and polarization differences. | Station gain, field rotation/R-L calibration, source/uv geometry, flags, or poor coherence. The helper **coherently averages complex visibilities over time** within each scan before taking phase or amplitude. | Multiple scan coherency results |
| `plot_closure_phase_vs_time_all_triangles` | Stable or changing RR and LL triangle phases; isolated noisy triangles versus repeatable structure across scans. | Repeatable nonzero phase can be source structure; isolated jumps/scatter can be weak data or baseline-dependent errors. Do not demand zero or RR=LL without a source/polarization model. The plotted channel average is circular in phase. | Closure products from RR/LL data |
| `plot_closure_amp_vs_time_all_quadrangles` | Log-amplitude trends around the source's expected value; unusual scatter or one quadrangle that disagrees with related ones. | Source structure can produce nonzero log closure amplitude. Large scatter or offsets can also come from baseline-dependent coherence, flags, low-SNR amplitude bias, or residual bandpass. | Closure products from RR/LL data |

The UVFITS closure builder excludes `AA` and `ALMA` by default because ordinary
RR/LL formulas assume a common circular polarization basis. This matters for
the experimental data described in [`data/README.md`](data/README.md): ALMA
has not been polarization-converted despite circular UVFITS labels. Interpret
absolute flux and polarization plots in light of that file's calibration
caveats. These closure plots do not add uncertainty bars or an automatic SNR
cut; inspect the constituent baseline visibilities before interpreting a
single noisy point.

## Individual fringe files: `fplot`

[`fplot`](src/eht_inspection/fringe.py) and `export_fplot_pdf` wrap the EAT/HOPS
fringe plot. For a flagged scan × baseline × polarization, check:

| Check | Question it answers |
| --- | --- |
| File metadata, controls, polarization, SNR/quality | Is this the intended fit and a credible detection? Interpret quality codes in the applicable HOPS format; do not require one code or a universal SNR threshold. |
| Delay/rate search and SBD/MBD information, where shown | Is the fringe peak isolated, ambiguous, near a search boundary, or inconsistent with the imposed stage 5 location? |
| Phase/amplitude across frequency and time, where shown | Are there slopes, jumps, phase wander, band-edge behavior, or scan coherence loss? |
| Used data/flags and available fit statistics | Did one channel, short interval, or missing data drive the result? |

The exact `fplot` layout and fields depend on the HOPS version and fringe
file. Use the plot to investigate a candidate; do not infer a cause from one
panel alone.

## Practical diagnosis patterns

| Pattern | Most useful cross-check |
| --- | --- |
| Stage 2→3 SNR fails to improve and `C_2s` is low | Inspect ad hoc phase solution and time/frequency phase in `fplot`; compare time-resolved UVFITS if available. |
| Stage 3 has one large MBD/rate triangle sum | Check its three baseline fits, minimum-leg SNR, ambiguity spacings, polarization, and neighboring triangles; allow for source structure. |
| Stage 5 MBD/rate sums are zero, but amplitude or SNR falls | Compare independent stage 3/4 fringe locations with imposed station differences; inspect `fplot`, coherence, and dropouts. Zero sums do not certify the station model. |
| UVFITS closure is repeatably nonzero with good baseline SNR | Compare with source/polarization structure and related triangles or quadrangles before calling it a calibration fault. |
| Trouble appears only after UVFITS conversion/calibration | Inspect UVFITS polarization basis and labels, channel/flag handling, a priori gains, field angle rotation, and R/L calibration stage. |

## Corrections to the supplied checklist

- The implemented ALIST builder is `compute_alist_closure_triangles`, not
  `build_alist_closure_products`.
- Stage 5 ALIST MBD/rate closure is imposed by the station-based zero-width
  solution. A wrong solution primarily shows up through lost coherence/SNR or
  dropouts, not a predictable nonzero triangle sum or a newly selected peak.
- ALIST residual phase, SBD−MBD, UVFITS phase, bandpass amplitude, closure
  phase, and closure amplitude have no universal zero/flat target. Their
  interpretation depends on the source, polarization, calibration, and SNR.
- Channel-resolved UVFITS bandpass plotting is possible whenever channels
  remain in the input; the helper is not restricted to a particular “raw” file.
- Station counts from flagged ALIST triangles prioritize follow-up; they do
  not prove which station is responsible. Dataset-specific examples such as a
  particular IF jump or delay-control value require evidence from that run.

## References

- [EHT-HOPS pipeline components][ehthops-stages]
- [Blackburn et al., *EHT-HOPS pipeline*, global fringe fitting and coherence][global-fringe]
- [EHT-HOPS units in ALIST and other products][hops-units]
- [Kareinen et al., source structure and closure delay][source-closure-delay]

[ehthops-stages]: https://ehthops.readthedocs.io/en/latest/pipeline_components.html
[global-fringe]: https://arxiv.org/pdf/1903.08832
[hops-units]: https://ehthops.readthedocs.io/en/latest/units.html
[source-closure-delay]: https://link.springer.com/article/10.1007/s00190-024-01837-2
