from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from src import config
from src.logistic_signal import evaluate_logistic_model
from src.metrics import compute_classification_metrics, score_to_prediction
from src.plotting import save_figure, set_plot_style
from src.preprocess import build_preprocessor, feature_group_from_encoded_name
from src.split import apply_split, load_or_create_split


def _fit_logistic_feature_groups(train_df: pd.DataFrame, best_c: float) -> list[str]:
    X_train = train_df[config.MODEL_BASE_FEATURES]
    y_train = train_df[config.LABEL_COLUMN]
    _, pipeline = evaluate_logistic_model(X_train, y_train, X_train, y_train, c_value=best_c)
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "coefficient": coefficients,
            "feature_group": [feature_group_from_encoded_name(name) for name in feature_names],
            "abs_coefficient": np.abs(coefficients),
        }
    )
    grouped = coef_df.groupby("feature_group")["abs_coefficient"].sum().sort_values(ascending=False)
    return grouped.head(5).index.tolist()


def _evaluate_svm_seed(train_df: pd.DataFrame, test_df: pd.DataFrame, family: str, params: dict) -> dict:
    X_train = train_df[config.MODEL_BASE_FEATURES]
    y_train = train_df[config.LABEL_COLUMN]
    X_test = test_df[config.MODEL_BASE_FEATURES]
    y_test = test_df[config.LABEL_COLUMN]
    if family == "linear":
        model = LinearSVC(max_iter=10000, random_state=config.MAIN_SPLIT_SEED, **params)
    else:
        model = SVC(kernel=family, **params)
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_score = pipeline.decision_function(X_test)
    y_pred = score_to_prediction(y_score)
    return compute_classification_metrics(y_test, y_pred, y_score)


def run_robustness_checks(
    analysis_df: pd.DataFrame,
    baseline_best_c: float,
    linear_params: dict,
    nonlinear_family: str,
    nonlinear_params: dict,
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    rows = []
    top_feature_groups: dict[int, list[str]] = {}

    for seed in config.ROBUSTNESS_SEEDS:
        manifest = load_or_create_split(analysis_df, seed)
        train_df, test_df = apply_split(analysis_df, manifest)

        X_train = train_df[config.MODEL_BASE_FEATURES]
        y_train = train_df[config.LABEL_COLUMN]
        X_test = test_df[config.MODEL_BASE_FEATURES]
        y_test = test_df[config.LABEL_COLUMN]

        logistic_metrics, _ = evaluate_logistic_model(X_train, y_train, X_test, y_test, c_value=baseline_best_c, seed=seed)
        rows.append({"seed": seed, "model": "logistic_l1", **logistic_metrics})
        top_feature_groups[seed] = _fit_logistic_feature_groups(train_df, baseline_best_c)

        linear_metrics = _evaluate_svm_seed(train_df, test_df, family="linear", params=linear_params)
        rows.append({"seed": seed, "model": "svm_linear", **linear_metrics})

        nonlinear_metrics = _evaluate_svm_seed(train_df, test_df, family=nonlinear_family, params=nonlinear_params)
        rows.append({"seed": seed, "model": f"svm_{nonlinear_family}", **nonlinear_metrics})

    return pd.DataFrame(rows), top_feature_groups


def plot_robustness_ranges(robustness_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    for model_name, subset in robustness_df.groupby("model"):
        ax.plot(subset["seed"], subset["roc_auc"], marker="o", label=model_name)
    ax.set_title("ROC-AUC Stability Across Random Seeds")
    ax.set_xlabel("Split Seed")
    ax.set_ylabel("ROC-AUC")
    ax.legend()
    save_figure(fig, output_path)
