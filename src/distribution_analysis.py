from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from src import config
from src.plotting import save_figure, set_plot_style

LABEL_MAP = {0: "<=50K", 1: ">50K"}


def _cohens_d(class_zero: np.ndarray, class_one: np.ndarray) -> float:
    n0 = len(class_zero)
    n1 = len(class_one)
    var0 = np.var(class_zero, ddof=1)
    var1 = np.var(class_one, ddof=1)
    pooled_std = math.sqrt(((n0 - 1) * var0 + (n1 - 1) * var1) / max(n0 + n1 - 2, 1))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(class_one) - np.mean(class_zero)) / pooled_std)


def _single_feature_auc(y_true: pd.Series, feature_values: pd.Series) -> float:
    auc = roc_auc_score(y_true, feature_values)
    return float(max(auc, 1.0 - auc))


def gaussian_fit_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in config.NUMERIC_RAW_FEATURES:
        class_zero = train_df.loc[train_df[config.LABEL_COLUMN] == 0, feature].to_numpy()
        class_one = train_df.loc[train_df[config.LABEL_COLUMN] == 1, feature].to_numpy()
        mean_zero = float(np.mean(class_zero))
        mean_one = float(np.mean(class_one))
        var_zero = float(np.var(class_zero, ddof=0))
        var_one = float(np.var(class_one, ddof=0))
        ks_stat = float(stats.ks_2samp(class_zero, class_one).statistic)
        auc = _single_feature_auc(train_df[config.LABEL_COLUMN], train_df[feature])
        zero_share = float((train_df[feature] == 0).mean())
        skewness = float(train_df[feature].skew())
        non_gaussian_flag = bool(zero_share >= 0.75 or abs(skewness) >= 1.0)
        if non_gaussian_flag and auc >= 0.65:
            shape_category = "non_gaussian_informative"
        elif non_gaussian_flag:
            shape_category = "non_gaussian_weak"
        else:
            shape_category = "roughly_symmetric"

        rows.append(
            {
                "feature": feature,
                "mean_<=50k": mean_zero,
                "variance_<=50k": var_zero,
                "mean_>50k": mean_one,
                "variance_>50k": var_one,
                "mean_difference_gt50k_minus_le50k": mean_one - mean_zero,
                "cohens_d": _cohens_d(class_zero, class_one),
                "ks_statistic": ks_stat,
                "single_feature_roc_auc": auc,
                "zero_share": zero_share,
                "skewness": skewness,
                "non_gaussian_flag": non_gaussian_flag,
                "shape_category": shape_category,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["single_feature_roc_auc", "ks_statistic", "cohens_d"], ascending=[False, False, False]
    )


def plot_gaussian_overlays(train_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=config.FIGURE_SIZE_TALL)
    axes = axes.flatten()
    palette = {0: "#4c78a8", 1: "#f58518"}

    for ax, feature in zip(axes, config.NUMERIC_RAW_FEATURES):
        x_min = float(train_df[feature].min())
        x_max = float(train_df[feature].max())
        x_grid = np.linspace(x_min, x_max, 400)

        for label_value in [0, 1]:
            values = train_df.loc[train_df[config.LABEL_COLUMN] == label_value, feature]
            mean = values.mean()
            std = values.std(ddof=0)
            ax.hist(values, bins=30, density=True, alpha=0.35, color=palette[label_value], label=LABEL_MAP[label_value])
            if std > 0:
                ax.plot(x_grid, stats.norm.pdf(x_grid, loc=mean, scale=std), color=palette[label_value], linewidth=2)
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel("Density")

    axes[-1].axis("off")
    axes[0].legend()
    save_figure(fig, output_path)


def plot_continuous_separation_ranking(summary_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    plot_df = summary_df.sort_values("single_feature_roc_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(plot_df["feature"], plot_df["single_feature_roc_auc"], color="#4c78a8")
    ax.set_title("Continuous Feature Separation Ranking")
    ax.set_xlabel("Single-Feature ROC-AUC")
    ax.set_ylabel("Feature")
    save_figure(fig, output_path)
