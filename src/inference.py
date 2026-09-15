"""
Inference utilities for the Iris MLOps project.
"""

from typing import List

import pandas as pd


def validate_inference_columns(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> None:
    """
    Validate that all required inference feature columns exist.
    """

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing inference columns: {missing_columns}"
        )

    print("Inference column validation: PASSED")


def validate_inference_data(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> None:
    """
    Validate inference data before prediction.
    """

    validate_inference_columns(df, feature_columns)

    if df.empty:
        raise ValueError("Inference dataset is empty")

    null_counts = df[feature_columns].isnull().sum()

    columns_with_nulls = (
        null_counts[null_counts > 0].to_dict()
    )

    if columns_with_nulls:
        raise ValueError(
            f"Inference data contains null values: "
            f"{columns_with_nulls}"
        )

    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Inference column '{column}' "
                f"must be numeric"
            )

    print("Inference data validation: PASSED")


def generate_predictions(
    model,
    df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Generate predictions and prediction probabilities.
    """

    validate_inference_data(
        df,
        feature_columns,
    )

    X = df[feature_columns]

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    result = df.copy()

    result["prediction"] = predictions

    result["probability_setosa"] = probabilities[:, 0]
    result["probability_versicolor"] = probabilities[:, 1]
    result["probability_virginica"] = probabilities[:, 2]

    return result


def validate_predictions(
    df: pd.DataFrame,
) -> None:
    """
    Validate generated predictions.
    """

    required_columns = [
        "prediction",
        "probability_setosa",
        "probability_versicolor",
        "probability_virginica",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {missing_columns}"
        )

    if df["prediction"].isnull().any():
        raise ValueError(
            "Prediction contains null values"
        )

    probability_columns = [
        "probability_setosa",
        "probability_versicolor",
        "probability_virginica",
    ]

    for column in probability_columns:

        if df[column].isnull().any():
            raise ValueError(
                f"{column} contains null values"
            )

        if ((df[column] < 0) | (df[column] > 1)).any():
            raise ValueError(
                f"{column} contains invalid probability values"
            )

    probability_sum = df[probability_columns].sum(axis=1)

    if not ((probability_sum - 1.0).abs() < 0.0001).all():
        raise ValueError(
            "Prediction probabilities do not sum to 1"
        )

    print("Prediction validation: PASSED")