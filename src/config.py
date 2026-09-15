"""
Central configuration for the Iris MLOps project.
"""

# ============================================================
# Unity Catalog
# ============================================================

CATALOG = "mlops_demo"

SCHEMA = "iris"


# ============================================================
# Unity Catalog Tables
# ============================================================

RAW_TABLE = (
    f"{CATALOG}.{SCHEMA}.iris_raw"
)

TRAIN_TABLE = (
    f"{CATALOG}.{SCHEMA}.iris_train"
)

TEST_TABLE = (
    f"{CATALOG}.{SCHEMA}.iris_test"
)

FEATURE_TABLE = (
    f"{CATALOG}.{SCHEMA}.iris_features"
)

PREDICTION_TABLE = (
    f"{CATALOG}.{SCHEMA}.iris_predictions"
)

MONITORING_TABLE = (
    f"{CATALOG}.{SCHEMA}.iris_monitoring"
)


# ============================================================
# MLflow / Model Registry
# ============================================================

EXPERIMENT_NAME = (
    "/Shared/iris-mlops"
)

REGISTERED_MODEL = (
    f"{CATALOG}.{SCHEMA}.iris_classifier"
)


# ============================================================
# Machine Learning Configuration
# ============================================================

TARGET_COLUMN = "target"

RANDOM_STATE = 42

TEST_SIZE = 0.2