# Communication-Efficient Distributed Fraud Detection

A distributed machine learning system that simulates fraud detection across geographically distributed payment data centers using **LocalSGD**. The project evaluates the trade-off between **communication efficiency** and **model performance** while providing configurable network simulation, experiment automation, monitoring, and model serving.

---

## Features

- Centralized and LocalSGD training
- Multi-data center simulation (5 workers)
- Configurable synchronization interval (K = 1, 5, 10, 20, 50, 100)
- Network latency and bandwidth simulation
- Communication cost analysis
- Automated experiment runner
- System resource monitoring (CPU, RAM, GPU)
- FastAPI inference service
- Docker support
- YAML-based configuration
- Automatic report and plot generation

---

## Project Structure

```text
Phase8/
├── api/               # FastAPI model serving
├── analysis/          # Reports and visualizations
├── config/            # YAML configurations
├── experiments/       # Experiment runner
├── models/            # Fraud detection model
├── monitoring/        # System monitoring
├── network/           # Network simulation
├── training/          # Centralized & LocalSGD training
├── artifacts/         # Generated outputs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Experimental Results (Centralized Baseline)

| Metric | Value |
|---------|------:|
| Accuracy | **99.936%** |
| Precision | **98.38%** |
| Recall | **51.35%** |
| F1 Score | **0.6747** |
| ROC-AUC | **0.9902** |
| PR-AUC | **0.7655** |

---

## Training

### Centralized Training

```bash
python training/centralized_baseline.py
```

### LocalSGD Training

```bash
python training/local_sgd.py
```

### Run Complete Experiment Suite

```bash
python experiments/experiment_runner.py
```

---

## Network Simulation

The simulator supports multiple network topologies for evaluating distributed training performance.

- Fully Connected
- Ring
- Star
- Tree
- Custom

Collected metrics include:

- Communication latency
- Synchronization time
- Communication cost
- Bytes transferred
- Bandwidth utilization

---

## Generated Outputs

Each experiment automatically generates:

- Model checkpoints
- Training metrics
- Network metrics
- Communication logs
- JSON reports
- CSV metrics
- Training plots
- Network visualizations
- CPU, RAM and GPU utilization statistics

---

## REST API

Start the API server

```bash
uvicorn api.main:app --reload
```

Available endpoints

| Method | Endpoint |
|---------|----------|
| GET | `/health` |
| GET | `/metrics` |
| GET | `/model/info` |
| POST | `/predict` |
| POST | `/predict/batch` |
| POST | `/model/reload` |

Swagger UI

```
http://localhost:8000/docs
```

---

## Docker

Build

```bash
docker build -t fraud-api .
```

Run

```bash
docker run -p 8000:8000 fraud-api
```

---

## Configuration

All experiments are configurable through YAML files.

Configuration options include:

- Learning rate
- Batch size
- Optimizer
- Local epochs
- Synchronization interval
- Network topology
- Model architecture
- Training parameters

---

## Technology Stack

- Python
- PyTorch
- FastAPI
- Uvicorn
- Docker
- Docker Compose
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- PyYAML
- PSUtil

---

## Future Work

- Adaptive LocalSGD
- Gradient compression
- Federated learning support
- Multi-GPU distributed training
- Kubernetes deployment
- Production monitoring

---
