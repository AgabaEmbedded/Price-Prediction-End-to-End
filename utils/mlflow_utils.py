"""
utils/mlflow_utils.py
─────────────────────
MLflow + S3 setup helpers.

Two modes depending on config:
  - "local"  → sqlite DB locally, artifacts stored on S3
  - URL      → remote MLflow server (e.g. EC2), artifacts stored on S3

Your AWS credentials must be available via one of:
  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  - ~/.aws/credentials file
  - IAM role (if on EC2)
"""

import os
import logging
import yaml
import mlflow
import dagshub

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
DAGSHUB_KEY = os.getenv("DAGSHUB_KEY")

log = logging.getLogger(__name__)


def load_config(path="configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_mlflow(cfg: dict | None = None) -> mlflow.MlflowClient:
    """
    Configure MLflow tracking URI and return an MlflowClient.

    Call this once at the start of any pipeline script.
    """
    if cfg is None:
        cfg = load_config()

    mc = cfg["mlflow"]

    if mc["location"] == "local":
        # Local sqlite-backed tracking, artifacts on S3
        db_path = Path(mc["local_db_path"]).resolve()
        tracking_uri = f"sqlite:///{db_path}"
        log.info(f"MLflow tracking: local SQLite @ {db_path}")
        mlflow.set_tracking_uri(tracking_uri)
    elif mc["location"] == "aws":
        tracking_uri = mc["tracking_uri"]
        log.info(f"MLflow tracking: remote server @ {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        os.environ["MLFLOW_ARTIFACT_ROOT"] = mc["s3_artifact_uri"]
        
    elif mc["location"] == "dagshub":
        dagshub.auth.add_app_token(DAGSHUB_KEY)
        dagshub.init(repo_owner=mc['repo_owner'], repo_name=mc['repo_name'], mlflow=True)
        # Use DagsHub's default location instead of forcing an S3 URI
        artifact_loc = None 
    else:
        artifact_loc = mc["s3_artifact_uri"]

    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        # If artifact_loc is None, MLflow uses the tracking server's default
        exp_id = mlflow.create_experiment(name=exp_name, artifact_location=artifact_loc)
    

    # Create or get experiment
    exp_name = mc["experiment_name"]
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        # If artifact_loc is None, MLflow uses the tracking server's default
        exp_id = mlflow.create_experiment(name=exp_name)
        log.info(f"Created MLflow experiment '{exp_name}'  (id={exp_id})")
    else:
        log.info(f"Using existing MLflow experiment '{exp_name}'  (id={exp.experiment_id})")

    mlflow.set_experiment(exp_name)
    return mlflow.MlflowClient()


def log_dataset_info(df, split: str, cfg: dict):
    """Log dataset statistics as MLflow tags/params."""
    mlflow.set_tag(f"data.{split}.rows", len(df))
    mlflow.set_tag(f"data.{split}.start", str(df.index[0].date()))
    mlflow.set_tag(f"data.{split}.end", str(df.index[-1].date()))
    label_dist = df["label"].value_counts().to_dict()
    mlflow.set_tag(f"data.{split}.label_dist", str(label_dist))


def get_best_run(experiment_name: str, metric: str = "val_f1_macro") -> dict:
    """Retrieve the best run from an experiment by a given metric."""
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")

    best = runs[0]
    log.info(
        f"Best run: {best.info.run_id}  "
        f"model={best.data.tags.get('model_name', '?')}  "
        f"{metric}={best.data.metrics.get(metric, '?'):.4f}"
    )
    return {
        "run_id":     best.info.run_id,
        "model_name": best.data.tags.get("model_name"),
        "metrics":    best.data.metrics,
        "params":     best.data.params,
    }
