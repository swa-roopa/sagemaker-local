# sagemaker-local

Day 3 of the AIOps stream: an incident-detection model taken from raw data to a live endpoint. Everything runs on a laptop. MLflow tracks and versions the models, and SageMaker **local mode** runs AWS's real SageMaker training and serving containers in Docker, so no SageMaker access or AWS bill is needed.

## What it shows

1. **Data:** 265 snapshots of a web service's health (one every 5 seconds), with 6 injected incidents.
2. **Train and track:** four models trained, scored and logged to MLflow.
3. **Register:** the best model saved as `ops-incident-detector` with version numbers.
4. **Honest number:** the same model scored again with the ambiguous "transition" rows kept (F1 1.00 → about 0.57).
5. **SageMaker training job:** `train.py` runs inside AWS's scikit-learn container.
6. **Endpoint:** the trained model is served at `http://localhost:8085`.
7. **Predict:** four health numbers in, `0` (normal) or `1` (incident) out.
8. **Clean up:** the endpoint and containers are deleted.

## Folder layout

```
sagemaker-local/
├── README.md
├── requirements.txt          pinned Python libraries
├── docs/flow.md              every command, in order
├── data/
│   ├── make_golden.py        generates the dataset
│   └── golden/incident_dataset.csv
├── ml/
│   └── train_models.py       Lab 3.4: train 4 models, log to MLflow, register the best
└── sagemaker/
    ├── train.py              the SageMaker training script (also runs on its own)
    └── local_mode_demo.py    train, deploy, predict and clean up in local mode
```

Created when you run it (not shared): `.venv/`, `mlflow.db`, `mlruns/`, `data/incident_dataset.csv`.

## The data

| Column | Meaning |
|---|---|
| `req_rate` | Requests per second |
| `err_rate` | Share of requests that failed |
| `p95_latency` | 95th-percentile response time, in seconds |
| `cpu` | CPU usage, 0 to 1 |
| `is_incident` | Label: 0 normal, 1 incident |
| `incident_type` | normal, latency, errors or cpu |
| `transition` | True where the 1-minute metric window and the instant label disagree |

The dataset is a synthetic stand-in for the Day 1 golden export, with the same columns, counts and story.

## Requirements

- Python 3.11
- Docker Desktop, running
- For the first local-mode run only: AWS credentials that can pull from Amazon ECR, to download the SageMaker scikit-learn image (about 5 GB). No SageMaker permissions are needed. Later runs work offline.

## Setup

```bash
cd sagemaker-local
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/make_golden.py
cp data/golden/incident_dataset.csv data/incident_dataset.csv
```

## Run

```bash
python ml/train_models.py                        # train, track, register (v1)
KEEP_TRANSITIONS=1 python ml/train_models.py     # the honest number (v2)
python ml/train_models.py                        # clean again (v3)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

export SAGEMAKER_SUPPRESS_V2_WARNING=1
python sagemaker/local_mode_demo.py --keep       # train + deploy; Ctrl+C cleans up
```

While the endpoint is running, in a second terminal:

```bash
curl http://localhost:8085/ping
curl -X POST http://localhost:8085/invocations -H "Content-Type: text/csv" -d "6.0,0.002,1.4,0.1"
```

The four numbers are, in order: requests per second, failed share, p95 latency in seconds, CPU share. The full sequence is in [docs/flow.md](docs/flow.md).

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it says running |
| Port 8085 already in use | `SERVING_PORT=8090 python sagemaker/local_mode_demo.py --keep`, then use 8090 in curl |
| curl: `Couldn't connect` | The demo isn't waiting any more; run it again with `--keep` |
| MLflow UI shows no runs | Start it from this folder, with the same `sqlite:///mlflow.db` |
| `Can not safely convert int64 to float64` | Write numbers with decimals (`15.0`, not `15`) |
| Leftover containers after a crash | `docker ps -aq --filter "name=-algo-1-" \| xargs docker rm -f` |

## Notes

- Local mode uses the same SageMaker SDK code as the cloud. Only `instance_type="local"` differs. Console pages, Model Registry approval and Pipelines need a real AWS account.
- The demo serves on port 8085 because 8080 is often taken by Jenkins or Tomcat.
- `--keep` leaves the endpoint up until Ctrl+C; without it the script cleans up by itself.
