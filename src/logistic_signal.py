from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src import config
from src.metrics import compute_classification_metrics, score_to_prediction
from src.plotting import save_figure, set_plot_style
from src.preprocess import build_preprocessor, feature_group_from_encoded_name

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

warnings.filterwarnings("ignore", message=".*penalty.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*", category=UserWarning)


def make_logistic_pipeline(c_value: float, interaction_columns: list[str] | None = None, seed: int = config.MAIN_SPLIT_SEED):
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(interaction_columns=interaction_columns)),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    l1_ratio=1.0,
                    solver="saga",
                    C=c_value,
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def cross_validate_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    c_grid: list[float] | None = None,
    interaction_columns: list[str] | None = None,
    seed: int = config.MAIN_SPLIT_SEED,
) -> pd.DataFrame:
    c_grid = c_grid or config.LOGISTIC_C_GRID
    splitter = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    rows = []
    for c_value in c_grid:
        pipeline = make_logistic_pipeline(c_value=c_value, interaction_columns=interaction_columns, seed=seed)
        scores = cross_validate(pipeline, X_train, y_train, cv=splitter, scoring=SCORING, n_jobs=-1)
        row = {"C": c_value}
        for metric_name in SCORING:
            row[f"mean_cv_{metric_name}"] = float(np.mean(scores[f"test_{metric_name}"]))
            row[f"std_cv_{metric_name}"] = float(np.std(scores[f"test_{metric_name}"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("C").reset_index(drop=True)


def fit_logistic_and_extract_coefficients(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    c_grid: list[float] | None = None,
    interaction_columns: list[str] | None = None,
    seed: int = config.MAIN_SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c_grid = c_grid or config.LOGISTIC_C_GRID
    coefficient_rows = []
    group_rows = []
    for c_value in c_grid:
        pipeline = make_logistic_pipeline(c_value=c_value, interaction_columns=interaction_columns, seed=seed)
        pipeline.fit(X_train, y_train)
        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        coefficients = model.coef_[0]
        current_rows = []
        for feature_name, coefficient in zip(feature_names, coefficients):
            feature_group = feature_group_from_encoded_name(feature_name)
            current_rows.append(
                {
                    "C": c_value,
                    "encoded_feature": feature_name,
                    "feature_group": feature_group,
                    "coefficient": float(coefficient),
                    "abs_coefficient": float(abs(coefficient)),
                    "nonzero": bool(abs(coefficient) > 1e-9),
                }
            )
        coefficient_rows.extend(current_rows)

        coef_df = pd.DataFrame(current_rows)
        grouped = (
            coef_df.groupby("feature_group")
            .agg(
                any_nonzero=("nonzero", "max"),
                total_abs_coefficient=("abs_coefficient", "sum"),
                signed_total_coefficient=("coefficient", "sum"),
            )
            .reset_index()
        )
        grouped["C"] = c_value
        group_rows.extend(grouped.to_dict(orient="records"))

    coefficient_df = pd.DataFrame(coefficient_rows)
    group_df = pd.DataFrame(group_rows)
    return coefficient_df, group_df


def feature_group_stability_table(group_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        group_df.groupby("feature_group")
        .agg(
            nonzero_count=("any_nonzero", "sum"),
            nonzero_share=("any_nonzero", "mean"),
            mean_total_abs_coefficient=("total_abs_coefficient", "mean"),
            max_total_abs_coefficient=("total_abs_coefficient", "max"),
        )
        .reset_index()
        .sort_values(["nonzero_share", "mean_total_abs_coefficient"], ascending=[False, False])
    )
    return summary


def choose_best_c(cv_summary_df: pd.DataFrame) -> float:
    best_row = cv_summary_df.sort_values(["mean_cv_roc_auc", "C"], ascending=[False, True]).iloc[0]
    return float(best_row["C"])


def evaluate_logistic_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    c_value: float,
    interaction_columns: list[str] | None = None,
    seed: int = config.MAIN_SPLIT_SEED,
) -> tuple[dict, Pipeline]:
    pipeline = make_logistic_pipeline(c_value=c_value, interaction_columns=interaction_columns, seed=seed)
    pipeline.fit(X_train, y_train)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    y_pred = score_to_prediction(y_score)
    metrics = compute_classification_metrics(y_test, y_pred, y_score)
    metrics["C"] = c_value
    return metrics, pipeline


def plot_coefficient_paths(coefficient_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    top_features = (
        coefficient_df.groupby("encoded_feature")["abs_coefficient"].max().sort_values(ascending=False).head(15).index.tolist()
    )
    plot_df = coefficient_df[coefficient_df["encoded_feature"].isin(top_features)].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature_name in top_features:
        subset = plot_df[plot_df["encoded_feature"] == feature_name].sort_values("C")
        ax.plot(np.log10(subset["C"]), subset["coefficient"], marker="o", label=feature_name)
    ax.set_title("L1 Logistic Regression Coefficient Paths")
    ax.set_xlabel("log10(C)")
    ax.set_ylabel("Coefficient")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save_figure(fig, output_path)
