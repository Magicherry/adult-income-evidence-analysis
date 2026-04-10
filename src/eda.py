from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from src import config
from src.plotting import save_figure, set_plot_style

LABEL_MAP = {0: "<=50K", 1: ">50K"}


def class_balance_table(train_df: pd.DataFrame) -> pd.DataFrame:
    counts = train_df[config.LABEL_COLUMN].value_counts().sort_index()
    proportions = train_df[config.LABEL_COLUMN].value_counts(normalize=True).sort_index()
    return pd.DataFrame(
        {
            "income_class": [LABEL_MAP[idx] for idx in counts.index],
            "count": counts.values,
            "proportion": proportions.values,
        }
    )


def continuous_feature_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    grouped = train_df.groupby(config.LABEL_COLUMN)[config.NUMERIC_RAW_FEATURES]
    summary = grouped.agg(["mean", "std", "median", "min", "max"])
    summary.columns = ["_".join(part for part in column if part) for column in summary.columns]
    summary = summary.reset_index().rename(columns={config.LABEL_COLUMN: "income_class"})
    summary["income_class"] = summary["income_class"].map(LABEL_MAP)
    return summary


def categorical_frequency_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    selected_features = [
        "workclass",
        "occupation",
        "marital_status",
        "relationship",
        "gender",
        "native_country_grouped",
    ]
    rows = []
    for feature in selected_features:
        counts = train_df.groupby(feature).size()
        positive_rate = train_df.groupby(feature)[config.LABEL_COLUMN].mean()
        for category, count in counts.sort_values(ascending=False).items():
            rows.append(
                {
                    "feature": feature,
                    "category": category,
                    "count": int(count),
                    "share": float(count / len(train_df)),
                    "positive_rate": float(positive_rate.loc[category]),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "count"], ascending=[True, False])


def continuous_flag_table(train_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in config.NUMERIC_RAW_FEATURES:
        series = train_df[feature]
        skewness = float(series.skew())
        zero_share = float((series == 0).mean())
        dominant_value_share = float(series.value_counts(normalize=True, dropna=False).iloc[0])

        if zero_share >= 0.75:
            flag_code = "zero_inflated"
        elif abs(skewness) >= 1.0:
            flag_code = "high_skew"
        elif dominant_value_share >= 0.35:
            flag_code = "dominant_value"
        else:
            flag_code = "none"

        rows.append(
            {
                "feature": feature,
                "skewness": skewness,
                "zero_share": zero_share,
                "dominant_value_share": dominant_value_share,
                "flag_code": flag_code,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["flag_code", "zero_share", "dominant_value_share", "feature"],
        ascending=[True, False, False, True],
    )


def plot_class_balance(class_balance_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(class_balance_df["income_class"], class_balance_df["count"], color=["#4c78a8", "#f58518"])
    ax.set_title("Training Split Class Balance")
    ax.set_ylabel("Count")
    for bar, proportion in zip(bars, class_balance_df["proportion"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{proportion:.1%}", ha="center", va="bottom")
    save_figure(fig, output_path)


def plot_continuous_by_income(train_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=config.FIGURE_SIZE_TALL)
    axes = axes.flatten()
    palette = {0: "#4c78a8", 1: "#f58518"}

    for ax, feature in zip(axes, config.NUMERIC_RAW_FEATURES):
        for label_value in [0, 1]:
            values = train_df.loc[train_df[config.LABEL_COLUMN] == label_value, feature]
            ax.hist(
                values,
                bins=30,
                alpha=0.5,
                density=True,
                color=palette[label_value],
                label=LABEL_MAP[label_value],
            )
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel("Density")
    axes[-1].axis("off")
    axes[0].legend()
    save_figure(fig, output_path)


def plot_categorical_frequency_grid(categorical_summary_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    features = [
        "workclass",
        "occupation",
        "marital_status",
        "relationship",
        "gender",
        "native_country_grouped",
    ]
    top_k = {
        "workclass": 8,
        "occupation": 8,
        "marital_status": 7,
        "relationship": 6,
        "gender": 2,
        "native_country_grouped": 3,
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, feature in zip(axes, features):
        subset = categorical_summary_df[categorical_summary_df["feature"] == feature].head(top_k[feature]).copy()
        subset = subset.sort_values("count")
        ax.barh(subset["category"], subset["count"], color="#4c78a8")
        for _, row in subset.iterrows():
            ax.text(row["count"], row["category"], f"  {row['positive_rate']:.1%} >50K", va="center")
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("Count")
    save_figure(fig, output_path)

