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
Hyperparameter sweep using GridSearchCV.

Instead of training once with fixed hyperparameters (as in train_kneighbors.py
and train_random_forest.py), this script tries every combination of
hyperparameter values defined in the search grids below.  sklearn's
GridSearchCV runs k-fold cross-validation for each combination and picks the
one with the best average validation accuracy.

The whole sweep is one MLflow run.  The best hyperparameters, the best
cross-validation score, and the held-out test accuracy are logged, and only
the winning model is saved and registered — so the Model Registry always
ends up with the best result rather than every trial.
"""

import argparse

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.data import X_train, X_test, y_train, y_test

# ---------------------------------------------------------------------------
# Hyperparameter grids
# Each key maps to a list of values to try.  GridSearchCV will test every
# possible combination — so 3 values × 2 values = 6 total fits per fold.
# ---------------------------------------------------------------------------

KNN_GRID = {
    "n_neighbors": [3, 5, 7, 11, 15],
    "weights": ["uniform", "distance"],
}

RF_GRID = {
    "max_depth": [2, 3, 5, None],       # None = grow until leaves are pure
    "n_estimators": [50, 100, 200],
    "min_samples_split": [2, 5],
}

# ---------------------------------------------------------------------------
# CLI — choose which model family to sweep
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["knn", "rf"],
    default="knn",
    help="Which classifier to sweep: 'knn' (k-nearest neighbors) or 'rf' (random forest)",
)
parser.add_argument(
    "--cv",
    type=int,
    default=5,
    help="Number of cross-validation folds (default: 5)",
)
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Build the estimator and search grid based on the chosen model
# ---------------------------------------------------------------------------

if args.model == "knn":
    experiment_name = "kneighbors-sweep"
    registered_model_name = "kneighbors-classifier"

    # Wrap the classifier in a StandardScaler pipeline so features are
    # normalised before distance calculations — important for KNN.
    base_estimator = make_pipeline(StandardScaler(), KNeighborsClassifier())

    # GridSearchCV needs parameter names prefixed with the step name when
    # using a Pipeline.  "kneighborsclassifier__" is the auto-generated name.
    param_grid = {
        f"kneighborsclassifier__{k}": v for k, v in KNN_GRID.items()
    }

else:
    experiment_name = "random-forest-sweep"
    registered_model_name = "random-forest-classifier"

    base_estimator = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))

    param_grid = {
        f"randomforestclassifier__{k}": v for k, v in RF_GRID.items()
    }

# ---------------------------------------------------------------------------
# Run the sweep inside a single MLflow run
# ---------------------------------------------------------------------------

mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=f"{args.model}-grid-search"):

    # GridSearchCV tries every combination in param_grid.
    # cv=args.cv means each combination is evaluated on `cv` different
    # train/validation splits of the training data.
    # refit=True (the default) re-trains the winner on the full training set.
    search = GridSearchCV(
        estimator=base_estimator,
        param_grid=param_grid,
        cv=args.cv,
        scoring="accuracy",
        refit=True,         # keep the best model fitted and ready to use
        n_jobs=-1,          # use all available CPU cores
        verbose=1,
    )

    print(f"Starting grid search over {len(search.param_grid)} combinations "  # type: ignore[attr-defined]
          f"with {args.cv}-fold CV...")
    search.fit(X_train, y_train)

    # The best combination found by cross-validation.
    best_params = search.best_params_
    best_cv_score = search.best_score_      # mean CV accuracy across folds
    best_estimator = search.best_estimator_ # already re-fitted on full X_train

    # Evaluate the winner on the held-out test set (data it has never seen).
    test_accuracy = best_estimator.score(X_test, y_test)

    print(f"\nBest params : {best_params}")
    print(f"Best CV acc : {best_cv_score:.4f}")
    print(f"Test acc    : {test_accuracy:.4f}")

    # Log the winning hyperparameters — strip the pipeline prefix for readability.
    clean_params = {k.split("__", 1)[-1]: v for k, v in best_params.items()}
    mlflow.log_params(clean_params)

    # Log the number of CV folds so the run is fully reproducible.
    mlflow.log_param("cv_folds", args.cv)

    # Log the key metrics for this run.
    mlflow.log_metrics({
        "best_cv_accuracy": best_cv_score,
        "test_accuracy": test_accuracy,
    })

    # Save the best model as a run artifact.
    mlflow.sklearn.log_model(best_estimator, artifact_path="models")

    print(f"\nRun complete. Model artifact saved. Run ID: {mlflow.active_run().info.run_id}")
