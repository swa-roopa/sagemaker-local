# Day 3 demo: commands in order

## First-time setup (once per machine)

Needs Python 3.11 and Docker Desktop. Run from the `sagemaker-local` folder.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first local-mode run downloads AWS's SageMaker scikit-learn image (about 5 GB) and needs AWS credentials that can pull from ECR. After that it runs offline.

## 0. Before the session

```bash
open -a Docker
cd sagemaker-local
source .venv/bin/activate
export SAGEMAKER_SUPPRESS_V2_WARNING=1      # hide the SDK v2 notice
docker images | grep sagemaker-scikit-learn
lsof -nP -iTCP:8085 -sTCP:LISTEN            # check nothing is using port 8085
```

## 1. Dataset

```bash
python data/make_golden.py
cp data/golden/incident_dataset.csv data/incident_dataset.csv
```

## 2. MLflow: train, track, register (Lab 3.4)

```bash
rm -rf mlflow.db mlruns
python ml/train_models.py
```

Second terminal:

```bash
cd sagemaker-local
source .venv/bin/activate
# mlflow.db is MLflow's memory: a small SQLite database in one file
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open http://localhost:5001

## 3. Load the registered model

```bash
python -c '
import mlflow.pyfunc, pandas as pd
mlflow.set_tracking_uri("sqlite:///mlflow.db")
m = mlflow.pyfunc.load_model("models:/ops-incident-detector/1")
print(m.predict(pd.DataFrame([
    {"req_rate": 15.0, "err_rate": 0.0, "p95_latency": 0.08, "cpu": 0.1},
    {"req_rate": 6.0,  "err_rate": 0.0, "p95_latency": 1.4,  "cpu": 0.1}])))
'
```

## 4. The honest number

```bash
KEEP_TRANSITIONS=1 python ml/train_models.py
python ml/train_models.py
```

## 5. Test the SageMaker script locally (Lab 4.2)

```bash
mkdir -p /tmp/sm/train /tmp/sm/model
cp data/incident_dataset.csv /tmp/sm/train/
python sagemaker/train.py --train /tmp/sm/train --model-dir /tmp/sm/model
ls /tmp/sm/model
python sagemaker/train.py --train /tmp/sm/train --model-dir /tmp/sm/model --n-estimators 50
python sagemaker/train.py --train /tmp/sm/train --model-dir /tmp/sm/model --n-estimators 200
```

## 6. SageMaker local mode: train, deploy, predict

Terminal 1:

```bash
python sagemaker/local_mode_demo.py --keep
```

## 7. Call the endpoint

Terminal 2:

```bash
curl http://localhost:8085/ping
curl -X POST http://localhost:8085/invocations -H "Content-Type: text/csv" -d "15.0,0.002,0.08,0.1"
curl -X POST http://localhost:8085/invocations -H "Content-Type: text/csv" -d "6.0,0.002,1.4,0.1"
curl -X POST http://localhost:8085/invocations -H "Content-Type: text/csv" -d "15.0,0.3,0.08,0.1"
curl -X POST http://localhost:8085/invocations -H "Content-Type: text/csv" -d "15.0,0.002,0.08,0.9"
```

## 8. Clean up

Terminal 1: press `Ctrl+C`

```bash
docker ps -a --filter "name=-algo-1-"
lsof -nP -iTCP:8085 -sTCP:LISTEN
```

## If needed

```bash
SERVING_PORT=8090 python sagemaker/local_mode_demo.py --keep
python sagemaker/local_mode_demo.py
pkill -INT -f "local_mode_demo.py --keep"
docker ps -aq --filter "name=-algo-1-" | xargs docker rm -f
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002
```
