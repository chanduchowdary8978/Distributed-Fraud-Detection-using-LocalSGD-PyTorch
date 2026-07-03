# Communication-Efficient Distributed Fraud Detection using LocalSGD

A distributed machine learning system that simulates fraud detection across geographically distributed payment data centers using **LocalSGD**. The project evaluates the trade-off between **communication efficiency** and **model performance** by training local models independently and periodically synchronizing them through configurable communication rounds.

---

## Features

- Centralized and LocalSGD training
- Multi-data center simulation (5 workers)
- Configurable synchronization intervals (K = 1, 5, 10, 20, 50, 100)
- Network latency and bandwidth simulation
- Communication cost analysis
- Automated experiment runner
- System resource monitoring (CPU, RAM and GPU)
- FastAPI model serving
- Dockerized deployment
- YAML-based configuration
- Automatic metrics, reports and plot generation

---

## Architecture

The system simulates distributed fraud detection across five geographically distributed payment data centers. Each worker trains a local model independently using LocalSGD while a central coordinator periodically synchronizes model parameters. A configurable network simulation layer measures communication latency, bandwidth utilization, synchronization overhead and communication cost during distributed training.

---

## Project Structure

```text
.
├── api/                  # FastAPI model serving
├── analysis/             # Reports and visualizations
├── config/               # YAML configurations
├── experiments/          # Experiment runner
├── models/               # Fraud detection model
├── monitoring/           # System monitoring
├── network/              # Network simulation
├── training/             # Centralized & LocalSGD training
├── artifacts/            # Generated outputs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
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

### Centralized Baseline

```bash
python training/centralized_baseline.py
```

### LocalSGD Training

```bash
python training/local_sgd.py
```

### Run All Experiments

```bash
python experiments/experiment_runner.py
```

---

## Network Simulation

The project supports multiple network topologies for evaluating distributed training performance.

Supported topologies:

- Fully Connected
- Ring
- Star
- Tree
- Custom

Collected network metrics:

- Communication latency
- Synchronization time
- Communication cost
- Bytes transferred
- Bandwidth utilization

---

## Generated Outputs

Each experiment automatically generates:

### Model Outputs

- Model checkpoints
- Training metrics
- Validation metrics

### Network Metrics

- Communication logs
- Synchronization statistics
- Network performance metrics

### System Monitoring

- CPU utilization
- RAM utilization
- GPU utilization
- Experiment duration

### Reports

- JSON reports
- CSV metrics
- Training plots
- Network visualizations

---

## REST API

Start the API server:

```bash
uvicorn api.main:app --reload
```

Available endpoints:

| Method | Endpoint |
|---------|----------|
| GET | `/health` |
| GET | `/metrics` |
| GET | `/model/info` |
| POST | `/predict` |
| POST | `/predict/batch` |
| POST | `/model/reload` |

Swagger Documentation:

```
http://localhost:8000/docs
```

---

## Docker

### Build Image

```bash
docker build -t fraud-api .
```

### Run Container

```bash
docker run -p 8000:8000 fraud-api
```

### Pull Prebuilt Image

```bash
docker pull chanduchowdary8978/fraud-api:latest
```

### Run Prebuilt Image

```bash
docker run -p 8000:8000 chanduchowdary8978/fraud-api:latest
```

---

## Configuration

All experiments are configurable using YAML files.

Supported configuration options include:

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

- Python 3.11
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
- Prometheus and Grafana integration
- Distributed hyperparameter optimization

---

## License

This project is licensed under the MIT License. See the **LICENSE** file for details.
