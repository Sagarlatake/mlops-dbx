"""
Model training utilities for the Iris MLOps project.
"""

from typing import Dict, Tuple

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def create_model(
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> RandomForestClassifier:

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )

    return model


def calculate_metrics(
    y_true,
    y_pred,
) -> Dict[str, float]:

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics


def train_model(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:

    model = create_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_train)

    metrics = calculate_metrics(
        y_train,
        predictions
    )

    return model, metrics