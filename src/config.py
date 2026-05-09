from __future__ import annotations

from pathlib import Path

# Project root, used to build the rest of the paths from one place.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data and source locations.
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "adult.csv"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SRC_DIR = ROOT_DIR / "src"

# Output folders for figures, tables, and saved metrics.
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Rename raw columns once so the rest of the code can use cleaner names.
RAW_TO_CLEAN_COLUMN_MAP = {
    "age": "age",
    "workclass": "workclass",
    "fnlwgt": "fnlwgt",
    "education": "education",
    "educational-num": "education_num",
    "marital-status": "marital_status",
    "occupation": "occupation",
    "relationship": "relationship",
    "race": "race",
    "gender": "gender",
    "capital-gain": "capital_gain",
    "capital-loss": "capital_loss",
    "hours-per-week": "hours_per_week",
    "native-country": "native_country",
    "income": "income",
}

# Columns dropped before modeling.
RAW_DROP_COLUMNS = ["education", "fnlwgt"]

# Shared id/label settings.
ROW_ID_COLUMN = "row_id"
LABEL_SOURCE_COLUMN = "income"
LABEL_COLUMN = "income_gt_50k"
POSITIVE_LABEL = ">50K"

# Raw feature groups used in the audit and EDA sections.
NUMERIC_RAW_FEATURES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_RAW_FEATURES = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native_country",
]

# Main model inputs after light feature engineering.
NUMERIC_MODEL_FEATURES = [
    "age",
    "education_num",
    "hours_per_week",
    "capital_gain_log1p",
    "capital_loss_log1p",
]

CATEGORICAL_MODEL_FEATURES = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native_country_grouped",
]

MODEL_BASE_FEATURES = NUMERIC_MODEL_FEATURES + CATEGORICAL_MODEL_FEATURES

# Shared split and reproducibility settings.
MAIN_SPLIT_SEED = 42
ROBUSTNESS_SEEDS = [7, 42, 99]
TEST_SIZE = 0.20

# Logistic regression tuning grid.
CV_FOLDS = 5
SVM_CV_FOLDS = 3

LOGISTIC_C_GRID = [1e-3, 3.162e-3, 1e-2, 3.162e-2, 1e-1, 3.162e-1, 1.0, 3.162, 10.0]

# Interaction-screening limits.
MAX_INTERACTIONS = 6
HIGH_DEPENDENCY_TOP_K = 4

# SVM tuning settings.
SVM_TUNING_SUBSET = 15000
LINEAR_SVM_C_GRID = [0.1, 1.0, 10.0]
POLY_SVM_PARAM_GRID = [
    {"C": 0.3, "degree": 2, "gamma": "scale", "coef0": 0.0},
    {"C": 1.0, "degree": 2, "gamma": "scale", "coef0": 1.0},
    {"C": 1.0, "degree": 3, "gamma": "scale", "coef0": 0.0},
    {"C": 3.0, "degree": 3, "gamma": "scale", "coef0": 1.0},
]
RBF_SVM_PARAM_GRID = [
    {"C": 0.3, "gamma": "scale"},
    {"C": 1.0, "gamma": "scale"},
    {"C": 3.0, "gamma": "scale"},
    {"C": 10.0, "gamma": "scale"},
    {"C": 3.0, "gamma": 0.05},
    {"C": 10.0, "gamma": 0.05},
]

# Baseline discretization used for mutual information.
MI_BASELINE_BINS = {
    "age": {"kind": "quantile", "bins": 8},
    "education_num": {"kind": "quantile", "bins": 8},
    "hours_per_week": {"kind": "quantile", "bins": 8},
    "capital_gain": {"kind": "zero_plus_quantile", "positive_bins": 4},
    "capital_loss": {"kind": "zero_plus_quantile", "positive_bins": 4},
}

# Coarser bins used for the sensitivity check.
MI_SENSITIVITY_BINS = {
    "age": {"kind": "quantile", "bins": 5},
    "education_num": {"kind": "quantile", "bins": 5},
    "hours_per_week": {"kind": "quantile", "bins": 5},
    "capital_gain": {"kind": "zero_plus_quantile", "positive_bins": 2},
    "capital_loss": {"kind": "zero_plus_quantile", "positive_bins": 2},
}

# Shared plotting defaults.
PLOT_DPI = 180
FIGURE_SIZE_TALL = (10, 12)
