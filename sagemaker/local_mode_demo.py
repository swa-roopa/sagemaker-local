"""SageMaker local mode: the real SageMaker training job + endpoint flow, run in Docker.

Same SKLearn estimator and train.py as a cloud training job, but instance_type="local",
so the SageMaker service is never called. Needs Docker running and AWS credentials that
can pull the SageMaker scikit-learn image from ECR (no SageMaker permissions needed).

Run from the lab folder:  python sagemaker/local_mode_demo.py
Keep the endpoint up to call it yourself (Ctrl+C cleans up):  python sagemaker/local_mode_demo.py --keep
"""
import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import time

import boto3
import numpy as np
import sagemaker.local.entities as local_entities
from sagemaker.local import LocalSession
from sagemaker.sklearn.estimator import SKLearn

# The first /ping attempts are refused until the container is listening; that's expected.
logging.getLogger("urllib3").setLevel(logging.ERROR)

# The SageMaker images are x86; on Apple Silicon they run under emulation, so give the
# endpoint more than the SDK's default 120 s to pass its health check.
local_entities.HEALTH_CHECK_TIMEOUT_LIMIT = 300
# Port 8080 is often taken on trainer laptops (Jenkins, Tomcat); serve on another port.
SERVING_PORT = int(os.environ.get("SERVING_PORT", "8085"))

KEEP = argparse.ArgumentParser()
KEEP.add_argument("--keep", action="store_true", help="leave the endpoint running until Ctrl+C")
KEEP = KEEP.parse_args().keep

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
# Local mode never assumes this role; any well-formed ARN works.
ROLE = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-local"

session = LocalSession(boto_session=boto3.Session(region_name=REGION))
session.config = {"local": {"local_code": True, "serving_port": SERVING_PORT}}  # no S3, free port
# deploy() looks up a default bucket name via STS even though local_code never uploads;
# naming one up front keeps the demo working offline once the image is cached.
session._default_bucket = "sagemaker-local-demo"

work = tempfile.mkdtemp(prefix="sm-local-")
train_dir = os.path.join(work, "train")
os.makedirs(train_dir)
shutil.copy("data/incident_dataset.csv", train_dir)  # the `train` channel -> SM_CHANNEL_TRAIN
# The container pip-installs the source folder and writes build files into it, so give it a copy.
src_dir = os.path.join(work, "src")
os.makedirs(src_dir)
shutil.copy("sagemaker/train.py", src_dir)

estimator = SKLearn(
    entry_point="train.py",
    source_dir=src_dir,
    framework_version="1.2-1",
    instance_type="local",            # "ml.m5.large" in the cloud
    instance_count=1,
    role=ROLE,
    sagemaker_session=session,
    output_path=f"file://{work}/output",
    hyperparameters={"n-estimators": 100},
    metric_definitions=[
        {"Name": "f1", "Regex": r"f1=([0-9.]+)"},
        {"Name": "recall", "Regex": r"recall=([0-9.]+)"},
    ],
)

print("== training job (local container) ==")
estimator.fit({"train": f"file://{train_dir}"})
print("model artifact:", estimator.model_data)

print("== deploying local endpoint ==")
# One web-server worker is enough for a demo. wait=False skips the SDK's 30 s status poll:
# in local mode deploy() already blocks until the container passes its /ping health check.
predictor = estimator.deploy(initial_instance_count=1, instance_type="local", model_server_workers=1, wait=False)
try:
    # Averages of each incident type: normal, latency, errors
    rows = np.array([
        [15.0, 0.002, 0.08, 0.10],
        [6.0, 0.002, 1.40, 0.10],
        [15.0, 0.300, 0.08, 0.10],
    ])
    t0 = time.time()
    preds = predictor.predict(rows)
    cold = time.time() - t0
    for label, pred in zip(["normal", "latency", "errors"], preds):
        print(f"{label} -> {int(pred)}")
    t0 = time.time()
    predictor.predict(rows)
    print(f"first call (cold start): {cold:.1f}s | second call (warm): {time.time() - t0:.2f}s")
    if KEEP:
        print(f"""
== endpoint is running at http://localhost:{SERVING_PORT} ==
In another terminal:
  curl http://localhost:{SERVING_PORT}/ping
  curl -X POST http://localhost:{SERVING_PORT}/invocations -H "Content-Type: text/csv" -d "6.0,0.002,1.4,0.1"
Columns: req_rate,err_rate,p95_latency,cpu  ->  answer 0 = normal, 1 = incident
Press Ctrl+C here to clean up.""", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print()
finally:
    print("== cleanup: deleting local endpoint and model ==")
    predictor.delete_model()      # needs the endpoint config, so it goes first
    predictor.delete_endpoint()
    shutil.rmtree(work, ignore_errors=True)
    # Remove the stopped SageMaker job/endpoint containers (named *-algo-1-*), nothing else.
    ids = subprocess.run(["docker", "ps", "-aq", "--filter", "name=-algo-1-"], capture_output=True, text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    print("your endpoint still present: no")
