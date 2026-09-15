"""
Feature engineering utilities for the Iris MLOps project.
"""

from typing import List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "petal_to_sepal_length_ratio",
    "petal_to_sepal_width_ratio",
]

KEY_COLUMNS = [
    "record_id",
]


def validate_input_columns(df: DataFrame) -> None:
    """
    Validate that all required input columns exist.
    """

    required_columns = [
        "record_id",
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
        "species",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Input column validation: PASSED")


def validate_key(df: DataFrame) -> None:
    """
    Validate record_id.
    """

    null_count = (
        df.filter(F.col("record_id").isNull())
        .count()
    )

    if null_count > 0:
        raise ValueError(
            f"record_id contains {null_count} null values"
        )

    duplicate_count = (
        df.groupBy("record_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"record_id contains {duplicate_count} duplicate keys"
        )

    print("Key validation: PASSED")


def create_features(df: DataFrame) -> DataFrame:
    """
    Create derived Iris features.
    """

    validate_input_columns(df)
    validate_key(df)

    feature_df = (
        df
        .withColumn(
            "petal_to_sepal_length_ratio",
            F.col("petal_length")
            / F.col("sepal_length")
        )
        .withColumn(
            "petal_to_sepal_width_ratio",
            F.col("petal_length")
            / F.col("sepal_width")
        )
    )

    return feature_df


def validate_features(df: DataFrame) -> None:
    """
    Validate engineered features.
    """

    required_features = (
        KEY_COLUMNS
        + FEATURE_COLUMNS
        + ["target", "species"]
    )

    missing_columns = [
        column
        for column in required_features
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    null_counts = (
        df.select(required_features)
        .select(
            [
                F.sum(
                    F.col(column).isNull().cast("int")
                ).alias(column)
                for column in required_features
            ]
        )
    )

    null_row = null_counts.collect()[0]

    total_nulls = sum(
        value or 0
        for value in null_row
    )

    if total_nulls > 0:
        raise ValueError(
            f"Feature table contains {total_nulls} null values"
        )

    invalid_ratio_count = (
        df.filter(
            (F.col("petal_to_sepal_length_ratio") <= 0)
            |
            (F.col("petal_to_sepal_width_ratio") <= 0)
        )
        .count()
    )

    if invalid_ratio_count > 0:
        raise ValueError(
            f"Invalid ratio values found: "
            f"{invalid_ratio_count}"
        )

    duplicate_key_count = (
        df.groupBy("record_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    if duplicate_key_count > 0:
        raise ValueError(
            f"Duplicate record_id values: "
            f"{duplicate_key_count}"
        )

    print("Feature validation: PASSED")


def get_feature_columns() -> List[str]:
    """
    Return feature columns used by the model.
    """

    return FEATURE_COLUMNS.copy()