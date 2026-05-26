# Early Fault Detection with Slope Entropy on a Raspberry Pi 5

Slope-Entropy-based monitoring pipeline for early bearing-fault detection on the NASA IMS Bearing Dataset, designed to run on an embedded Raspberry Pi 5.

## Setup

```bash
pip install -r requirements.txt
```

Download the NASA IMS Bearing Dataset from <https://data.nasa.gov/dataset/ims-bearings>, unpack it, and place the `2nd_test/` directory at `data/raw/2nd_test/`.

## Run

```bash
python run_pipeline.py
```

Phase 1 prints descriptive statistics for the first (healthy) file and saves `results/figures/F1_raw_healthy.png`.

## Layout

```
.
├── config.yaml          # single source of truth for every parameter
├── requirements.txt
├── run_pipeline.py      # thin orchestrator
├── src/
│   ├── utils.py         # config + RNG helpers
│   ├── io_ims.py        # dataset I/O
│   └── plots.py         # matplotlib helpers
├── scripts/
│   └── inspect_dataset.py
├── tests/
├── report/
│   ├── paper.tex
│   └── references.bib
├── data/raw/            # IMS dataset (not committed)
└── results/
    └── figures/
```

## Configuration

All parameters live in `config.yaml`. Phase 1 uses `seed`, `data_root`, `channel`, `results_dir`, and `figure.{duration_ms,dpi}`.
