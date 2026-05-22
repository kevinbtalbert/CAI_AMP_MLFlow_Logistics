# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
# ###########################################################################
"""
CML Model entrypoint for trained sklearn classifiers logged with MLflow.

The ``predict`` function is the handler registered when deploying this file
as a CML Model.  It expects a JSON payload with a ``features`` key containing
a list of 20 numeric values (matching the dataset produced by
``scripts/data.py``) and returns the predicted class and class probabilities.

Environment variables (set on the CML Model deployment)
--------------------------------------------------------
``MLFLOW_MODEL_NAME``   Load the latest version of a registered model from the
                        CML Model Registry (e.g. ``kneighbors-classifier`` or
                        ``random-forest-classifier``).  Takes priority over
                        ``MLFLOW_RUN_ID`` when both are set.
``MLFLOW_RUN_ID``       Load a specific training run artifact.  Used when
                        ``MLFLOW_MODEL_NAME`` is not set.
If neither variable is set the API falls back to the most-recent MLflow run.

Example request body
--------------------
{"features": [0.1, -0.5, 1.2, 0.0, 0.3, -1.1, 0.8, 0.2, -0.4, 0.6,
               0.9, -0.2, 1.5, -0.7, 0.4, 0.1, -0.3, 0.8, -0.9, 0.5]}

Example response
----------------
{"prediction": 1, "probabilities": [0.23, 0.77], "run_id": "<mlflow-run-id>"}
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

try:
    import cml.models_v1 as cml_models
    import cml.metrics_v1 as cml_metrics

    CML_AVAILABLE = True
except ImportError:
    # Running outside CML — stub the decorators so the file still imports cleanly.
    class _Models:
        def cml_model(self, **kwargs):
            def deco(fn):
                return fn
            return deco

    class _Metrics:
        def track_metric(self, key, value):
            pass

    cml_models = _Models()
    cml_metrics = _Metrics()
    CML_AVAILABLE = False

_model = None
_run_id: str | None = None


def _load_model() -> None:
    """Load the MLflow sklearn model into the module-level ``_model`` variable.

    Resolution order:
    1. ``MLFLOW_MODEL_NAME`` — latest version from the CML Model Registry.
    2. ``MLFLOW_RUN_ID``     — specific run artifact.
    3. Fallback              — most-recent run across all experiments.
    """
    global _model, _run_id

    model_name = os.getenv("MLFLOW_MODEL_NAME", "").strip()
    run_id = os.getenv("MLFLOW_RUN_ID", "").strip()

    if model_name:
        # Load the latest production-ready version from the CML Model Registry.
        uri = f"models:/{model_name}/latest"
        _model = mlflow.sklearn.load_model(uri)
        _run_id = f"registry:{model_name}@latest"
    elif run_id:
        _run_id = run_id
        _model = mlflow.sklearn.load_model(f"runs:/{_run_id}/models")
    else:
        # Fall back to the most-recent run across all experiments.
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=["0"],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError(
                "No MLflow runs found. Train a model first, or set "
                "MLFLOW_MODEL_NAME / MLFLOW_RUN_ID."
            )
        _run_id = runs[0].info.run_id
        _model = mlflow.sklearn.load_model(f"runs:/{_run_id}/models")


# Load on module import — CML initialises the module once before serving requests.
_load_model()


@cml_models.cml_model(metrics=True)
def predict(args: Dict[str, Any]) -> Dict[str, Any]:
    """CML model handler.

    Parameters
    ----------
    args:
        JSON-decoded request body.  Must contain a ``features`` key with a
        list of 20 numeric values.

    Returns
    -------
    dict with keys ``prediction``, ``probabilities``, and ``run_id``.
    """
    try:
        features: List[float] = args["features"]

        if len(features) != 20:
            return {"error": f"Expected 20 features, got {len(features)}."}

        prediction = int(_model.predict([features])[0])
        probabilities: List[float] = _model.predict_proba([features])[0].tolist()

        cml_metrics.track_metric("prediction", prediction)

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "run_id": _run_id,
        }

    except KeyError:
        return {"error": "Request body must contain a 'features' key."}
    except Exception as exc:
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
