"""
Data preparation utilities for the Iris MLOps project.

Responsibilities:
    1. Load Iris dataset
    2. Standardize column names
    3. Add deterministic record ID
    4. Validate schema
    5. Validate data types
    6. Validate null values
    7. Detect and remove exact duplicates
    8. Validate target values
    9. Validate feature values
    10. Split data into train/test sets
"""

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "target"

SPECIES_COLUMN = "species"

REQUIRED_COLUMNS = [
    "record_id",
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "target",
    "species",
]


# ---------------------------------------------------------------------
# 1. Load Iris dataset
# ---------------------------------------------------------------------

def load_iris_data() -> pd.DataFrame:
    """
    Load the sklearn Iris dataset and standardize column names.

    Returns:
        pandas.DataFrame
    """

    iris = load_iris(as_frame=True)

    pdf = iris.frame.copy()

    pdf = pdf.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )

    pdf["species"] = pdf["target"].map(
        {
            0: "setosa",
            1: "versicolor",
            2: "virginica",
        }
    )

    # Create deterministic record ID.
    #
    # We deliberately do NOT use monotonically_increasing_id()
    # because that would be generated later by Spark and could
    # change between executions.
    pdf["record_id"] = range(1, len(pdf) + 1)

    # Put record_id first
    pdf = pdf[
        [
            "record_id",
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "target",
            "species",
        ]
    ]

    return pdf


# ---------------------------------------------------------------------
# 2. Validate required columns
# ---------------------------------------------------------------------

def validate_columns(pdf: pd.DataFrame) -> None:
    """
    Validate that all required columns exist.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in pdf.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Column validation: PASSED")


# ---------------------------------------------------------------------
# 3. Validate data types
# ---------------------------------------------------------------------

def validate_data_types(pdf: pd.DataFrame) -> None:
    """
    Validate numeric and categorical columns.
    """

    numeric_columns = [
        "record_id",
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]

    categorical_columns = [
        "species"
    ]

    for column in numeric_columns:

        if not pd.api.types.is_numeric_dtype(
            pdf[column]
        ):
            raise TypeError(
                f"Column '{column}' must be numeric. "
                f"Actual type: {pdf[column].dtype}"
            )

    for column in categorical_columns:

        if not (
            pd.api.types.is_object_dtype(pdf[column])
            or pd.api.types.is_string_dtype(pdf[column])
        ):
            raise TypeError(
                f"Column '{column}' must be string/object. "
                f"Actual type: {pdf[column].dtype}"
            )

    print("Data type validation: PASSED")


# ---------------------------------------------------------------------
# 4. Validate null values
# ---------------------------------------------------------------------

def validate_nulls(pdf: pd.DataFrame) -> None:
    """
    Validate that required columns contain no null values.
    """

    null_counts = (
        pdf[REQUIRED_COLUMNS]
        .isnull()
        .sum()
    )

    total_nulls = int(
        null_counts.sum()
    )

    if total_nulls > 0:

        print("Null counts:")
        print(
            null_counts[
                null_counts > 0
            ]
        )

        raise ValueError(
            f"Null values detected: {total_nulls}"
        )

    print("Null validation: PASSED")


# ---------------------------------------------------------------------
# 5. Detect exact duplicates
# ---------------------------------------------------------------------

def get_duplicate_count(
    pdf: pd.DataFrame
) -> int:
    """
    Return number of exact duplicate rows.

    Important:
    This function does NOT fail the pipeline.
    Duplicate records are handled separately.
    """

    duplicate_count = int(
        pdf.duplicated().sum()
    )

    return duplicate_count


# ---------------------------------------------------------------------
# 6. Remove exact duplicates
# ---------------------------------------------------------------------

def remove_duplicates(
    pdf: pd.DataFrame
) -> Tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate records.

    Returns:
        cleaned DataFrame
        number of removed records
    """

    before_count = len(pdf)

    duplicate_count = get_duplicate_count(pdf)

    print(
        f"Duplicate records detected: {duplicate_count}"
    )

    if duplicate_count == 0:

        print(
            "No duplicate records found."
        )

        return pdf.copy(), 0

    cleaned_pdf = (
        pdf
        .drop_duplicates()
        .reset_index(drop=True)
    )

    after_count = len(cleaned_pdf)

    removed_count = (
        before_count - after_count
    )

    print(
        f"Rows before duplicate removal: "
        f"{before_count}"
    )

    print(
        f"Rows after duplicate removal: "
        f"{after_count}"
    )

    print(
        f"Duplicate rows removed: "
        f"{removed_count}"
    )

    return cleaned_pdf, removed_count


# ---------------------------------------------------------------------
# 7. Validate target
# ---------------------------------------------------------------------

def validate_target(
    pdf: pd.DataFrame
) -> None:
    """
    Validate target values and species mapping.
    """

    valid_targets = {
        0,
        1,
        2
    }

    actual_targets = set(
        pdf[TARGET_COLUMN]
        .unique()
    )

    invalid_targets = (
        actual_targets - valid_targets
    )

    if invalid_targets:

        raise ValueError(
            f"Invalid target values: "
            f"{invalid_targets}"
        )

    expected_species = {
        0: "setosa",
        1: "versicolor",
        2: "virginica",
    }

    for target, species in expected_species.items():

        invalid_mapping = (
            pdf[
                pdf[TARGET_COLUMN] == target
            ][SPECIES_COLUMN]
            != species
        ).any()

        if invalid_mapping:

            raise ValueError(
                f"Invalid species mapping "
                f"for target {target}"
            )

    print("Target validation: PASSED")


# ---------------------------------------------------------------------
# 8. Validate feature ranges
# ---------------------------------------------------------------------

def validate_feature_ranges(
    pdf: pd.DataFrame
) -> None:
    """
    Validate that Iris measurements are positive.

    This is a basic domain validation rule.
    """

    for column in FEATURE_COLUMNS:

        invalid_count = int(
            (
                pdf[column] <= 0
            ).sum()
        )

        if invalid_count > 0:

            raise ValueError(
                f"Invalid values found in "
                f"{column}: {invalid_count}"
            )

    print("Feature range validation: PASSED")


# ---------------------------------------------------------------------
# 9. Validate record ID
# ---------------------------------------------------------------------

def validate_record_id(
    pdf: pd.DataFrame
) -> None:
    """
    Validate record_id uniqueness.
    """

    duplicate_ids = int(
        pdf["record_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids > 0:

        raise ValueError(
            f"Duplicate record_id values: "
            f"{duplicate_ids}"
        )

    null_ids = int(
        pdf["record_id"]
        .isnull()
        .sum()
    )

    if null_ids > 0:

        raise ValueError(
            f"Null record_id values: "
            f"{null_ids}"
        )

    print(
        "Record ID validation: PASSED"
    )


# ---------------------------------------------------------------------
# 10. Complete validation
# ---------------------------------------------------------------------

def validate_data(
    pdf: pd.DataFrame
) -> None:
    """
    Execute all data-quality validations.

    This function DOES NOT remove duplicates.
    Duplicate handling is explicitly performed
    by remove_duplicates().
    """

    print("=" * 60)
    print("STARTING DATA VALIDATION")
    print("=" * 60)

    validate_columns(pdf)

    validate_data_types(pdf)

    validate_nulls(pdf)

    validate_record_id(pdf)

    validate_target(pdf)

    validate_feature_ranges(pdf)

    print("=" * 60)
    print("DATA VALIDATION COMPLETED")
    print("=" * 60)


# ---------------------------------------------------------------------
# 11. Train/Test Split
# ---------------------------------------------------------------------

def split_data(
    pdf: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Stratification is used to maintain class distribution.
    """

    if not 0 < test_size < 1:

        raise ValueError(
            "test_size must be between 0 and 1"
        )

    train_pdf, test_pdf = train_test_split(
        pdf,
        test_size=test_size,
        random_state=random_state,
        stratify=pdf[TARGET_COLUMN],
    )

    train_pdf = (
        train_pdf
        .reset_index(drop=True)
    )

    test_pdf = (
        test_pdf
        .reset_index(drop=True)
    )

    return train_pdf, test_pdf