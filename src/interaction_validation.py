from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.logistic_signal import cross_validate_logistic, evaluate_logistic_model, fit_logistic_and_extract_coefficients
from src.plotting import save_figure, set_plot_style
from src.preprocess import add_interaction_terms


def _classify_interaction(
    high_dependency: bool,
    delta_cv_auc: float,
    delta_test_auc: float,
    nonzero_share: float,
    sign_consistency: float,
) -> str:
    predictive = delta_cv_auc >= 0.002 and delta_test_auc >= -0.002 and nonzero_share >= 0.5
    unstable = sign_consistency < 0.75 or (0.1 <= nonzero_share < 0.5) or abs(delta_cv_auc) < 0.001

    if predictive and high_dependency:
        return "predictive_high_dependency"
    if predictive and not high_dependency:
        return "predictive_low_dependency"
    if unstable:
        return "unstable"
    if high_dependency:
        return "high_dependency_weak_predictive"
    return "unstable"


def evaluate_interaction_candidates(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    baseline_best_c: float,
    baseline_cv_auc: float,
    baseline_test_auc: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]], float, dict]:
    X_train = train_df[config.MODEL_BASE_FEATURES]
    y_train = train_df[config.LABEL_COLUMN]
    X_test = test_df[config.MODEL_BASE_FEATURES]
    y_test = test_df[config.LABEL_COLUMN]

    tested_rows = []
    survival_rows = []

    for _, candidate in candidate_df.iterrows():
        interaction_pair = [(candidate["model_feature_a"], candidate["model_feature_b"])]
        X_train_aug, interaction_columns = add_interaction_terms(X_train, interaction_pair)
        X_test_aug, _ = add_interaction_terms(X_test, interaction_pair)

        cv_summary = cross_validate_logistic(
            X_train_aug,
            y_train,
            c_grid=[baseline_best_c],
            interaction_columns=interaction_columns,
            seed=config.MAIN_SPLIT_SEED,
        )
        candidate_cv_auc = float(cv_summary.iloc[0]["mean_cv_roc_auc"])
        delta_cv_auc = candidate_cv_auc - baseline_cv_auc

        coefficient_df, _ = fit_logistic_and_extract_coefficients(
            X_train_aug,
            y_train,
            interaction_columns=interaction_columns,
            seed=config.MAIN_SPLIT_SEED,
        )
        interaction_feature_name = f"num__{interaction_columns[0]}"
        interaction_coefficients = coefficient_df[coefficient_df["encoded_feature"] == interaction_feature_name].copy()
        nonzero_share = float(interaction_coefficients["nonzero"].mean()) if not interaction_coefficients.empty else 0.0
        nonzero_coefficients = interaction_coefficients.loc[interaction_coefficients["nonzero"], "coefficient"]
        sign_consistency = (
            float(max((nonzero_coefficients > 0).mean(), (nonzero_coefficients < 0).mean()))
            if not nonzero_coefficients.empty
            else 0.0
        )
        mean_abs_coefficient = float(interaction_coefficients["abs_coefficient"].mean()) if not interaction_coefficients.empty else 0.0

        metrics, _ = evaluate_logistic_model(
            X_train_aug,
            y_train,
            X_test_aug,
            y_test,
            c_value=baseline_best_c,
            interaction_columns=interaction_columns,
            seed=config.MAIN_SPLIT_SEED,
        )
        delta_test_auc = metrics["roc_auc"] - baseline_test_auc
        high_dependency = bool(candidate["candidate_rank"] <= config.HIGH_DEPENDENCY_TOP_K)
        outcome = _classify_interaction(high_dependency, delta_cv_auc, delta_test_auc, nonzero_share, sign_consistency)

        tested_rows.append(
            {
                **candidate.to_dict(),
                "interaction_name": interaction_columns[0],
                "cv_roc_auc": candidate_cv_auc,
                "delta_cv_roc_auc": delta_cv_auc,
                "test_roc_auc": metrics["roc_auc"],
                "delta_test_roc_auc": delta_test_auc,
                "test_f1": metrics["f1"],
                "high_dependency": high_dependency,
                "evidence_code": outcome,
            }
        )
        survival_rows.append(
            {
                "interaction_name": interaction_columns[0],
                "feature_a": candidate["feature_a"],
                "feature_b": candidate["feature_b"],
                "nonzero_share": nonzero_share,
                "sign_consistency": sign_consistency,
                "mean_abs_coefficient": mean_abs_coefficient,
                "evidence_code": outcome,
            }
        )

    tested_df = pd.DataFrame(tested_rows).sort_values("candidate_rank").reset_index(drop=True)
    survival_df = pd.DataFrame(survival_rows).sort_values("nonzero_share", ascending=False).reset_index(drop=True)

    survivors = tested_df[
        tested_df["evidence_code"].isin({"predictive_high_dependency", "predictive_low_dependency"})
    ].sort_values(["delta_cv_roc_auc", "delta_test_roc_auc"], ascending=False)
    survivor_pairs = [(row["model_feature_a"], row["model_feature_b"]) for _, row in survivors.head(3).iterrows()]

    if survivor_pairs:
        X_train_combined, interaction_columns = add_interaction_terms(X_train, survivor_pairs)
        X_test_combined, _ = add_interaction_terms(X_test, survivor_pairs)
        combined_cv_summary = cross_validate_logistic(
            X_train_combined,
            y_train,
            interaction_columns=interaction_columns,
            seed=config.MAIN_SPLIT_SEED,
        )
        combined_best_c = float(combined_cv_summary.sort_values(["mean_cv_roc_auc", "C"], ascending=[False, True]).iloc[0]["C"])
        combined_metrics, _ = evaluate_logistic_model(
            X_train_combined,
            y_train,
            X_test_combined,
            y_test,
            c_value=combined_best_c,
            interaction_columns=interaction_columns,
            seed=config.MAIN_SPLIT_SEED,
        )
    else:
        combined_best_c = baseline_best_c
        combined_metrics = {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
            "C": combined_best_c,
        }

    return tested_df, survival_df, survivor_pairs, combined_best_c, combined_metrics


def interaction_model_metrics_table(
    baseline_metrics: dict,
    combined_metrics: dict,
    survivor_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    rows = [{"model": "baseline_main_effects", **baseline_metrics, "interaction_count": 0, "selected_interactions": ""}]
    rows.append(
        {
            "model": "interaction_augmented",
            **combined_metrics,
            "interaction_count": len(survivor_pairs),
            "selected_interactions": ", ".join(f"{a} x {b}" for a, b in survivor_pairs),
        }
    )
    return pd.DataFrame(rows)


def plot_interaction_delta_auc(tested_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    plot_df = tested_df.copy()
    plot_df["label"] = plot_df["feature_a"] + " x " + plot_df["feature_b"]
    plot_df = plot_df.sort_values("delta_cv_roc_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#f58518" if value > 0 else "#bdbdbd" for value in plot_df["delta_cv_roc_auc"]]
    ax.barh(plot_df["label"], plot_df["delta_cv_roc_auc"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Interaction Contribution Relative to Baseline Logistic Model")
    ax.set_xlabel("Delta CV ROC-AUC")
    save_figure(fig, output_path)
