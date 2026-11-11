# Autonomous Driving Perception & Collision Risk Assessment

**Undergraduate Thesis · Work in Progress (WIP)**

A simulation-based research project exploring vehicle perception and collision-risk assessment in **MetaDrive**, combining **YOLO-based object detection**, **LiDAR ranging**, and temporal sensor measurements. The goal is to develop and evaluate a learning-based risk assessment system for semi-autonomous driving.

> **Status:** This repository documents the current implementation and ongoing experiments; the risk-prediction model and evaluation are not yet complete. It is a research prototype, not a production or safety-certified driving system.

## Project overview

The current codebase explores:

- **Simulation:** vehicle scenarios and trajectories in MetaDrive.
- **Camera perception:** object detection and tracking using Ultralytics YOLO and BoT-SORT.
- **Sensor fusion:** associating camera tracks with LiDAR clusters by angular overlap and estimated distance.
- **Motion estimation:** recording relative/closing speed, object distance, track IDs, and timestamps.
- **Data collection:** automated multi-episode simulation runs with per-frame vehicle measurements, ego-vehicle telemetry, and episode outcomes.
- **Risk research:** exploratory time-to-collision (TTC) analysis and preparation of temporally windowed features for a potential GRU-based classifier.

## Repository structure

```text
.
├── 14_speed_logging.py            # Interactive simulation and speed/distance logging
├── automatic_data_collection.py   # Automated multi-episode collector
├── botsort_metadrive_reid.yaml    # Tracker configuration
├── analysis/
│   ├── analyse_60.py              # Summary statistics for the latest automated run
│   ├── check_speed_log.py         # Relative-speed consistency checks
│   ├── collision_analysis.py      # TTC exploration around a recorded collision
│   ├── inspect_dataset.py         # Explore recorded episode data (see note below)
│   ├── prepare_gru_dataset.py     # Feature aggregation, labeling, GRU sequences (WIP)
│   └── trajectory_analysis.py     # Track segmentation and TTC exploration
├── data/
│   └── samples/                   # Small, illustrative CSV excerpts
├── docs/                           # Add thesis figures/screenshots here
├── requirements.txt
└── .gitignore
```

## Getting started

### Requirements

- Python 3.11 (development environment)
- An environment capable of running MetaDrive with camera rendering
- Python packages listed in `requirements.txt`
- YOLO model checkpoint named `yolo26n.pt` (not included in this repository)

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> Dependency versions are not pinned because this export does not include a verified environment lockfile. Compatibility may vary by MetaDrive/Ultralytics release. The scripts have **not been runtime-tested** as part of this repository packaging step.

### Run interactive simulation

From the repository root:

```bash
python 14_speed_logging.py
```

This script writes local `speed_measurements.csv` and `episode_events.csv` files next to the script. New generated logs are ignored by Git.

### Collect data from multiple episodes

```bash
python automatic_data_collection.py --episodes 10 --max-steps 400 --mode mixed
```

For visual preview:

```bash
python automatic_data_collection.py --episodes 5 --max-steps 400 --preview
```

Each collector run writes results to `auto_runs/<run-id>/`:

- `vehicle_measurements.csv` — matched camera/LiDAR object observations
- `ego_steps.csv` — ego-vehicle state for simulation steps
- `episode_events.csv` — episode-level outcomes
- `run_settings.json` — experiment settings

The `auto_runs/` directory is excluded from Git to avoid committing large generated datasets by accident.

### Analyze a collected run

After collecting at least one run, execute from the repository root:

```bash
python analysis/analyse_60.py
python analysis/prepare_gru_dataset.py
```

`prepare_gru_dataset.py` constructs 20-step windows and provisional collision-within-2-seconds labels. **Missing values in the feature sequences still need to be handled** before reliable GRU training. No trained GRU model or validated performance metrics are included here.

Other scripts under `analysis/` are exploratory and contain assumptions about particular tracks, CSV files, or run locations. They may require adapting their input paths or selecting a suitable run; they are not all standalone demonstrations.

## Sample data and results

The `data/samples/` directory contains a *small* set of existing CSV logs for inspecting the logged schema. These are not the full experiment dataset and should not be treated as a benchmark or published experimental result.

**Planned updates:** stable risk labels and feature handling; sequence-model training and evaluation; reproducible experiment configuration; visual demonstrations and selected plots.

## Acknowledgments / tools

- [MetaDrive](https://metadriverse.github.io/metadrive/) for simulated driving scenarios
- [Ultralytics](https://www.ultralytics.com/) for YOLO-based object perception and tracking

The project is an ongoing undergraduate thesis. If the thesis work involves other contributors or institution-specific distribution requirements, those should be credited/checked before publication.
