from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def collapse_native_country(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    if value == "United-States":
        return "United-States"
    return "Non-United-States"


def build_analysis_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    analysis_df = df.copy()
    analysis_df[config.LABEL_COLUMN] = (
        analysis_df[config.LABEL_SOURCE_COLUMN] == config.POSITIVE_LABEL
    ).astype(int)
    analysis_df = analysis_df.drop(columns=config.RAW_DROP_COLUMNS + [config.LABEL_SOURCE_COLUMN])

    for column in config.CATEGORICAL_RAW_FEATURES:
        if column == "native_country":
            continue
        analysis_df[column] = analysis_df[column].fillna("Missing")

    analysis_df["native_country_grouped"] = analysis_df["native_country"].apply(collapse_native_country)
    analysis_df["native_country"] = analysis_df["native_country"].fillna("Missing")

    analysis_df["capital_gain_log1p"] = np.log1p(analysis_df["capital_gain"])
    analysis_df["capital_loss_log1p"] = np.log1p(analysis_df["capital_loss"])
    return analysis_df


def retained_features_table() -> pd.DataFrame:
    rows = []
    for feature in config.NUMERIC_RAW_FEATURES:
        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "used_in_analysis": True,
                "used_in_modeling": feature in {"age", "education_num", "capital_gain", "capital_loss", "hours_per_week"},
                "notes": "raw numeric feature",
            }
        )
    for feature in config.CATEGORICAL_RAW_FEATURES:
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "used_in_analysis": True,
                "used_in_modeling": True,
                "notes": "native_country is collapsed for main modeling" if feature == "native_country" else "raw categorical feature",
            }
        )
    rows.extend(
        [
            {
                "feature": "capital_gain_log1p",
                "feature_type": "derived_numeric",
                "used_in_analysis": False,
                "used_in_modeling": True,
                "notes": "log1p transform for modeling",
            },
            {
                "feature": "capital_loss_log1p",
                "feature_type": "derived_numeric",
                "used_in_analysis": False,
                "used_in_modeling": True,
                "notes": "log1p transform for modeling",
            },
            {
                "feature": "native_country_grouped",
                "feature_type": "derived_categorical",
                "used_in_analysis": True,
                "used_in_modeling": True,
                "notes": "United-States vs Non-United-States vs Missing",
            },
            {
                "feature": config.LABEL_COLUMN,
                "feature_type": "label",
                "used_in_analysis": True,
                "used_in_modeling": True,
                "notes": "binary target derived from income",
            },
        ]
    )
    return pd.DataFrame(rows)


def add_interaction_terms(df: pd.DataFrame, interaction_pairs: Iterable[tuple[str, str]]) -> tuple[pd.DataFrame, list[str]]:
    output_df = df.copy()
    created_columns: list[str] = []
    for feature_a, feature_b in interaction_pairs:
        column_name = f"interaction__{feature_a}__x__{feature_b}"
        output_df[column_name] = output_df[feature_a] * output_df[feature_b]
        created_columns.append(column_name)
    return output_df, created_columns


def build_preprocessor(interaction_columns: Iterable[str] | None = None) -> ColumnTransformer:
    interaction_columns = list(interaction_columns or [])
    numeric_features = config.NUMERIC_MODEL_FEATURES + interaction_columns

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, config.CATEGORICAL_MODEL_FEATURES),
        ],
        sparse_threshold=0.3,
    )


def feature_group_from_encoded_name(feature_name: str) -> str:
    if feature_name.startswith("num__"):
        cleaned = feature_name.replace("num__", "", 1)
        if cleaned.startswith("interaction__"):
            return "interaction"
        return cleaned
    if feature_name.startswith("cat__"):
        cleaned = feature_name.replace("cat__", "", 1)
        for column in sorted(config.CATEGORICAL_MODEL_FEATURES, key=len, reverse=True):
            prefix = f"{column}_"
            if cleaned.startswith(prefix):
                return column
        return cleaned
    return feature_name
