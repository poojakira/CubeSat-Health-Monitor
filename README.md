# Orbit-Q — Distributed ML Satellite Telemetry Platform

**Multi-model ML anomaly detection for satellite telemetry — academic/personal project**

[![CI](https://github.com/poojakira/orbit-Q/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/orbit-Q/actions)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2)]()
[![Tests](https://img.shields.io/badge/tests-11%20passed-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]()

**Multi-Model Ensemble** · **Triton Fusion Kernels** · **MLflow Tracking** · **HMAC Auth** · **10-Page Dashboard**

---

## 1. Overview

Orbit-Q is a satellite telemetry anomaly detection platform built as a hands-on learning project in ML systems, distributed pipelines, MLOps, and CUDA kernel engineering. It provides an end-to-end pipeline from telemetry ingestion to multi-model ensemble detection and a Streamlit command & control dashboard.

### Key Features

- **Multi-Model Ensemble** — IsolationForest (global outliers), PyTorch Autoencoder (feature manifold), LSTM (temporal patterns)
- **Triton Fusion Kernel** — Custom CUDA kernel for nanosecond-level score combining across ensemble models
- **MLOps Lifecycle** — MLflow experiment tracking; automated drift detection and retraining at 0.1 KL-divergence threshold
- **HMAC-SHA256 Auth** — Stateless stream token authentication with TTL and audit trail logging
- **10-Page Dashboard** — Streamlit C2 interface for live telemetry, anomaly alerts, MLflow lineage, and endpoint health

---

## 2. Architecture

### Package Structure

| # | Module | Role |
|---|---|---|
| 1 | `cli.py` | Main entry point with 6 mission-critical commands |
| 2 | `engine/` | Core ML ensemble and custom CUDA kernels for score fusion |
| 3 | `ingestion/` | High-frequency telemetry entry point (REST/gRPC) |
| 4 | `orchestrator/` | Central rules engine and stream processing coordinator |
| 5 | `dashboard/` | Full-stack Streamlit C2 interface |
| 6 | `mlflow_tracking/` | Experiment lineage and automated model maintenance |
| 7 | `simulator/` | Fault-injection telemetry generators for testing |

---

## 3. ML Ensemble

| # | Model | Type | Role |
|---|---|---|---|
| 1 | **IsolationForest** | Tree ensemble | Global outlier detection |
| 2 | **PyTorch Autoencoder** | Neural network | Feature manifold learning |
| 3 | **LSTM** | Recurrent network | Temporal pattern modeling |
| 4 | **Triton Fusion Kernel** | CUDA kernel | Nanosecond-level score combining |

---

## 4. Quick Start

```bash
git clone https://github.com/poojakira/orbit-Q.git
cd orbit-Q
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -e .
```

```bash
ORBIT_Q_SIGNING_SECRET=your-secure-secret-key
MLFLOW_TRACKING_URI=sqlite:///mlruns/orbit_q.db
```

---

## 5. CLI Commands

| # | Command | Description |
|---|---|---|
| 1 | `orbit-q simulator` | Start a single-satellite mock telemetry stream |
| 2 | `orbit-q orchestrator` | Run the ML pipeline and rule-dispatch daemon |
| 3 | `orbit-q dashboard` | Launch the Streamlit command center (default :8501) |
| 4 | `orbit-q benchmark` | Execute a high-rate throughput and latency stress test |
| 5 | `orbit-q stress-test` | Simulate multiple concurrent satellite streams |
| 6 | `orbit-q retrain` | Manually trigger the ensemble retraining pipeline |

---

## 6. Operator Dashboard (10-Page Suite)

| # | Page | Description |
|---|---|---|
| 1 | Live Telemetry | High-frequency streaming charts for all satellite subsystems |
| 2 | Alert & Command | Real-time anomaly log with interactive operator intervention tools |
| 3 | Hardware Diagnostics | Deep-dive into thermal, electrical, and mechanical telemetry |
| 4 | Orbital Tracking | TLE-based position visualization and signal lock status |
| 5 | Raw Telemetry Logs | Searchable database of all historical telemetry packets |
| 6 | Performance Audit | MLOps compliance tracker; accuracy vs. contamination audit |
| 7 | Inference Latency | Microsecond-level tracking of GPU engine performance |
| 8 | MLflow Lineage | Full experiment lineage; tracks every mission pulse and model run |
| 9 | Model Retraining | Manual trigger interface for the ensemble retraining pipeline |
| 10 | Endpoint Health | Real-time status of the ingestion API and downstream services |

---

## 7. Security & Reliability

- **Auth**: Stateless HMAC-SHA256 stream tokens with defined TTL
- **Graceful Fallback**: Automatic CPU fallback if cuML/GPU components are unavailable
- **Resilient Data**: Handles missing packets, latency jitter, and corrupted (NaN) sensor inputs
- **Audit**: Every detected anomaly and system command recorded in an audit trail

---

## 8. Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

| Suite | Count |
|---|---|
| Tests passing | 11 |

---

## 9. References

- **IsolationForest**: Liu et al., Isolation Forest, ICDM 2008
- **Triton Kernels**: https://triton-lang.org/
- **MLflow**: https://mlflow.org/

---

## 10. Team Contributions

> This is an academic/personal project built to learn ML systems, CUDA kernel engineering, MLOps, and distributed pipelines. Neither contributor has professional industry experience — all work was done as self-directed learning.

### Pooja Kiran

| # | What I Worked On | What I Built / Learned | Outcome |
|---|---|---|---|
| 1 | Multi-Model ML Ensemble Engine | Designed and implemented 3-model ensemble: IsolationForest (global outliers), PyTorch Autoencoder (feature manifold), LSTM (temporal patterns) | 3 models fused; ensemble handles 200 Hz+ satellite telemetry streams |
| 2 | Custom Triton CUDA Fusion Kernel | Engineered CUDA kernel (`engine/`) for nanosecond-level score fusion across all 3 ensemble models | Nanosecond-level fusion latency; avoids Python GIL overhead at high frequency |
| 3 | cuML / CPU Graceful Fallback | Implemented automatic fallback from cuML GPU to sklearn CPU when CUDA unavailable | 100% portability across GPU and CPU environments |
| 4 | DDP Multi-GPU Training | Implemented PyTorch Distributed Data Parallel (DDP) via `mp.spawn` for SLURM-compatible multi-GPU training | SLURM-compatible; no Horovod/Ray dependency |
| 5 | MLflow Experiment Lineage | Built full MLflow tracking system for every model training run, drift event, and retraining trigger | Full lineage tracking; automated retraining at 0.1 KL-divergence drift threshold |
| 6 | Drift Detection & Auto-Retraining | Implemented statistical drift detection pipeline with KL-divergence monitoring and auto-retrain triggers | Prevents model staleness under changing sensor calibrations |
| 7 | HMAC-SHA256 Stream Authentication | Designed stateless HMAC stream token authentication with defined TTL and comprehensive audit logging | Stateless auth; every anomaly and command recorded in tamper-proof audit trail |

### Rhutvik Pachghare

| # | What I Worked On | What I Built / Learned | Outcome |
|---|---|---|---|
| 1 | Mission Simulation Engine | Built fault-injection telemetry simulator (`simulator/`) generating realistic satellite sensor streams with configurable anomaly injection | Supports multiple concurrent satellite streams via `orbit-q stress-test` |
| 2 | Distributed Orchestrator | Engineered the central rules engine and stream processing coordinator (`orchestrator/`) managing the ML pipeline daemon | Real-time dispatch; handles missing packets, latency jitter, and NaN sensor inputs |
| 3 | 10-Page Streamlit C2 Dashboard | Developed the full-stack Streamlit Command & Control interface across 10 specialized mission control modules | 10 pages: live telemetry, alert/command, diagnostics, orbital tracking, logs, audit, latency, MLflow, retraining, endpoint health |
| 4 | REST/gRPC Ingestion Layer | Implemented high-throughput telemetry ingestion endpoints (`ingestion/`) with automated event-schema mapping | Supports 200 Hz+ ingestion rate with gRPC buffering and REST load balancing fallback |
| 5 | Code formatting & CI | Applied Black formatting and standardized code style across the codebase; configured GitHub Actions CI | Uniform code style across all modules; CI runs on every push to main |

---

**License**: MIT | **Platform**: Python 3.9+ | **GPU**: CUDA 11.8+
