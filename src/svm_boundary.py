from __future__ import annotations

import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from src import config
from src.metrics import compute_classification_metrics, score_to_prediction
from src.plotting import save_figure, set_plot_style
from src.preprocess import build_preprocessor


def _subsample_training_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= config.SVM_TUNING_SUBSET:
        return X, y
    X_subset, _, y_subset, _ = train_test_split(
        X,
        y,
        train_size=config.SVM_TUNING_SUBSET,
        stratify=y,
        random_state=config.MAIN_SPLIT_SEED,
    )
    return X_subset, y_subset


def _evaluate_cv_scores(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    splitter = StratifiedKFold(n_splits=config.SVM_CV_FOLDS, shuffle=True, random_state=config.MAIN_SPLIT_SEED)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring={"roc_auc": "roc_auc", "f1": "f1"},
        n_jobs=-1,
    )
    return {
        "mean_cv_roc_auc": float(np.mean(scores["test_roc_auc"])),
        "std_cv_roc_auc": float(np.std(scores["test_roc_auc"])),
        "mean_cv_f1": float(np.mean(scores["test_f1"])),
        "std_cv_f1": float(np.std(scores["test_f1"])),
    }


def tune_linear_svm(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, dict]:
    X_subset, y_subset = _subsample_training_data(X_train, y_train)
    rows = []
    for c_value in config.LINEAR_SVM_C_GRID:
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", LinearSVC(C=c_value, max_iter=10000, random_state=config.MAIN_SPLIT_SEED)),
            ]
        )
        summary = _evaluate_cv_scores(pipeline, X_subset, y_subset)
        rows.append({"family": "linear", "params": str({"C": c_value}), **summary})
    tuning_df = pd.DataFrame(rows).sort_values("mean_cv_roc_auc", ascending=False).reset_index(drop=True)
    best_params = ast.literal_eval(tuning_df.iloc[0]["params"])
    return tuning_df, best_params


def tune_kernel_svm(X_train: pd.DataFrame, y_train: pd.Series, family: str) -> tuple[pd.DataFrame, dict]:
    X_subset, y_subset = _subsample_training_data(X_train, y_train)
    param_grid = config.POLY_SVM_PARAM_GRID if family == "poly" else config.RBF_SVM_PARAM_GRID
    rows = []
    for params in param_grid:
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", SVC(kernel=family, **params)),
            ]
        )
        summary = _evaluate_cv_scores(pipeline, X_subset, y_subset)
        rows.append({"family": family, "params": str(params), **summary})
    tuning_df = pd.DataFrame(rows).sort_values("mean_cv_roc_auc", ascending=False).reset_index(drop=True)
    best_params = ast.literal_eval(tuning_df.iloc[0]["params"])
    return tuning_df, best_params


def _fit_and_evaluate_svm(
    family: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
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
    metrics = compute_classification_metrics(y_test, y_pred, y_score)
    return metrics


def tune_and_evaluate_svm_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_df[config.MODEL_BASE_FEATURES]
    y_train = train_df[config.LABEL_COLUMN]
    X_test = test_df[config.MODEL_BASE_FEATURES]
    y_test = test_df[config.LABEL_COLUMN]

    linear_tuning, linear_params = tune_linear_svm(X_train, y_train)
    poly_tuning, poly_params = tune_kernel_svm(X_train, y_train, family="poly")
    rbf_tuning, rbf_params = tune_kernel_svm(X_train, y_train, family="rbf")

    tuning_df = pd.concat([linear_tuning, poly_tuning, rbf_tuning], ignore_index=True)
    comparison_rows = []
    for family, params, tuning_table in [
        ("linear", linear_params, linear_tuning),
        ("poly", poly_params, poly_tuning),
        ("rbf", rbf_params, rbf_tuning),
    ]:
        metrics = _fit_and_evaluate_svm(family, params, X_train, y_train, X_test, y_test)
        comparison_rows.append(
            {
                "family": family,
                "params": str(params),
                "best_cv_roc_auc": float(tuning_table.iloc[0]["mean_cv_roc_auc"]),
                **metrics,
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return tuning_df, comparison_df


def build_final_model_comparison(
    logistic_metrics_df: pd.DataFrame,
    interaction_metrics_df: pd.DataFrame,
    svm_comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in logistic_metrics_df.iterrows():
        rows.append({"model": "logistic_baseline", **row.to_dict()})
    for _, row in interaction_metrics_df.iterrows():
        rows.append({"model": row["model"], **row.to_dict()})
    for _, row in svm_comparison_df.iterrows():
        rows.append({"model": f"svm_{row['family']}", **row.to_dict()})
    final_df = pd.DataFrame(rows)
    if "family" in final_df.columns:
        final_df = final_df.drop(columns=["family"])
    return final_df


def plot_svm_kernel_comparison(comparison_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    plot_df = comparison_df.sort_values("roc_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(plot_df["family"], plot_df["roc_auc"], color=["#4c78a8", "#72b7b2", "#f58518"])
    ax.set_title("Held-Out ROC-AUC by SVM Family")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("SVM Family")
    save_figure(fig, output_path)
