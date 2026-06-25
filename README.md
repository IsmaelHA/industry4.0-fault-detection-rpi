# Early Fault Detection with Slope Entropy on a Raspberry Pi 5

Slope-Entropy-based monitoring pipeline for early bearing-fault detection on the NASA IMS Bearing Dataset, designed to run on an embedded Raspberry Pi 5.

## Setup (Raspberry Pi OS Bookworm)

```bash
sudo apt install -y python3-venv python3-full
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the NASA IMS Bearing Dataset from <https://data.nasa.gov/dataset/ims-bearings>, unpack it, and place the `2nd_test/` directory at `data/raw/2nd_test/`.

## Pi run sequence

Run from the repo root with the venv active.

```bash
python scripts/capture_environment.py     # -> results/environment.txt (Pi model, OS, versions)
python scripts/inspect_dataset.py data/raw # sanity-check files / channels / time span

python run_pipeline.py                     # Phases 1-6: F1-F4 + 3-stage + CSVs
python scripts/run_sensitivity.py          # F5 (m sweep + gamma sweep)
python scripts/run_perf.py                 # perf_log_{baseline,usecols,float32}.csv
python scripts/build_perf_figure.py        # F6 execution-time plot
```

## Method summary

- **Slope Entropy** with **fixed** thresholds (`delta`, `gamma` in g units), embedding `m`.
- **Segmentation**: non-overlapping windows of `window` samples.
- **Turning point**: 3-sigma control limit on the entropy timeline — first sustained
  excursion (≥ `min_consecutive` points outside the healthy `mean ± n_sigma*std` band).
- **Comparison**: healthy / transition / faulty stages, per-stage entropy summary table
  (quantitative) + raw-signal panels (qualitative); RMS reported as a classic baseline.
- **Optimisation**: `usecols` (parse only the bearing column) and `float32`.

## Outputs

`results/figures/`
- `F1_raw_healthy.png`, `F2_raw_degraded.png`, `F3_entropy_timeline.png`,
  `F4_turning_point.png`, `F5_sensitivity_m.png`, `F5_sensitivity_gamma.png`,
  `F6_execution_time.png`, `Fextra_3stage_raw.png`, `Fextra_3stage_entropy.png`

`results/`
- `entropy_timeline.csv`, `entropy_timeline_m{3,4,5}.csv`, `entropy_timeline_g{...}.csv`
- `control_limit_sweep.csv`, `indicator_change_points.csv`, `sensitivity_change_points.csv`
- `comparison_summary.csv`
- `perf_log_{baseline,usecols,float32}.csv`, `environment.txt`

## Layout

```
.
├── config.yaml           # single source of truth
├── requirements.txt
├── run_pipeline.py       # Phases 1-6 orchestrator
├── src/
│   ├── runtime.py        # config + RNG
│   ├── io_ims.py         # dataset I/O
│   ├── segmentation.py   # windowing
│   ├── slope_entropy.py  # Slope Entropy (pure NumPy)
│   ├── analysis.py       # 3-sigma control limit + RMS
│   ├── comparison.py     # 3-stage split + per-stage stats
│   ├── timeline.py       # per-file aggregation + CSV writer
│   ├── perf.py           # Stopwatch + environment capture
│   └── plots.py          # all matplotlib figures
├── scripts/
│   ├── inspect_dataset.py
│   ├── capture_environment.py
│   ├── run_sensitivity.py
│   ├── run_perf.py
│   └── build_perf_figure.py
├── data/raw/             # IMS dataset (not committed)
└── results/              # outputs (figures + CSVs)
```

## Configuration

All parameters live in `config.yaml`:
`seed`, `data_root`, `channel`, `segmentation.{window,stride}`,
`entropy.{m,delta,gamma,adaptive_thresholds}`, `sweep.sampling_stride`,
`control_limit.{healthy_fraction,n_sigma,min_consecutive,n_sigma_sweep}`,
`comparison.margin`, `sensitivity.{m_values,gamma_values}`.
