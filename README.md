# graphcast-africa

GraphCast operational weather forecasting pipeline tailored for African meteorological contexts.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## Overview

Deconstructs the [DeepMind GraphCast/GenCast demo notebook](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_demo_cloud_vm.ipynb) into a modular Python package. Runs the GraphCast Operational model and subsets output to the African continent.

## Quick Start

```bash
pip install -e ".[dev]"
python scripts/download_assets.py --assets ./assets
python scripts/run_forecast.py --source opendata --date 20240601 --time 0000 --lead-time 120
python scripts/benchmark_gpu.py --source opendata --date 20240601 --time 0000 --lead-time 120 --warmup-runs 1 --benchmark-runs 3 --json
```

## GPU Benchmarking

Use the dedicated benchmark runner to collect GPU speed and latency metrics for GraphCast runs.

```bash
graphcast-africa-benchmark-gpu --source opendata --date 20240601 --time 0000 --lead-time 120 --warmup-runs 1 --benchmark-runs 3 --json
```

The benchmark output includes:

- Data retrieval time
- Model load time
- First rollout latency (cold/warmup path)
- Benchmark rollout latency summary (`avg`, `min`, `max`, `p95`)
- Throughput in lead-hours per second
- GPU snapshot from `nvidia-smi` (when available)
=======
python scripts/run_forecast.py --source opendata --date 20260415--time 0000 --lead-time 120
>>>>>>> main
