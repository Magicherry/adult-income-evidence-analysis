from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from src import config
from src.plotting import save_figure, set_plot_style

MI_FEATURES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native_country_grouped",
]

NUMERIC_MI_FEATURES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

DOMAIN_PLAUSIBLE_RESERVES = [
    ("education_num", "hours_per_week"),
    ("age", "hours_per_week"),
    ("age", "education_num"),
    ("education_num", "capital_gain"),
    ("hours_per_week", "capital_gain"),
]


@dataclass
class BinningRule:
    feature: str
    kind: str
    edges: list[float]


def _quantile_edges(series: pd.Series, bins: int) -> list[float]:
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(series.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        edges = np.array([series.min(), series.max() + 1e-9])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.tolist()


def fit_binning_rules(train_df: pd.DataFrame, spec: dict) -> dict[str, BinningRule]:
    rules: dict[str, BinningRule] = {}
    for feature, rule_spec in spec.items():
        if rule_spec["kind"] == "quantile":
            edges = _quantile_edges(train_df[feature], rule_spec["bins"])
            rules[feature] = BinningRule(feature=feature, kind="quantile", edges=edges)
        elif rule_spec["kind"] == "zero_plus_quantile":
            positive_values = train_df.loc[train_df[feature] > 0, feature]
            if positive_values.empty:
                edges = [0.0, np.inf]
            else:
                edges = _quantile_edges(positive_values, rule_spec["positive_bins"])
            rules[feature] = BinningRule(feature=feature, kind="zero_plus_quantile", edges=edges)
        else:
            raise ValueError(f"Unsupported binning kind: {rule_spec['kind']}")
    return rules


def apply_binning_rules(df: pd.DataFrame, rules: dict[str, BinningRule]) -> pd.DataFrame:
    discrete = pd.DataFrame(index=df.index)
    for feature in MI_FEATURES:
        if feature in rules:
            rule = rules[feature]
            if rule.kind == "quantile":
                discrete[feature] = pd.cut(
                    df[feature],
                    bins=rule.edges,
                    include_lowest=True,
                    duplicates="drop",
                ).astype(str)
            else:
                output = pd.Series(index=df.index, dtype=object)
                output[df[feature] == 0] = "zero"
                positive_mask = df[feature] > 0
                if positive_mask.any():
                    binned = pd.cut(
                        df.loc[positive_mask, feature],
                        bins=rule.edges,
                        include_lowest=True,
                        duplicates="drop",
                    ).astype(str)
                    output.loc[positive_mask] = binned
                output[df[feature] < 0] = "negative"
                discrete[feature] = output.fillna("missing").astype(str)
        else:
            discrete[feature] = df[feature].fillna("Missing").astype(str)
    return discrete


def compute_mi_outputs(discrete_df: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mi_matrix = pd.DataFrame(index=MI_FEATURES, columns=MI_FEATURES, dtype=float)
    pair_rows = []
    label_rows = []

    for feature in MI_FEATURES:
        label_rows.append(
            {"feature": feature, "feature_label_mi": float(mutual_info_score(discrete_df[feature], y))}
        )

    for i, feature_a in enumerate(MI_FEATURES):
        for j, feature_b in enumerate(MI_FEATURES):
            if j < i:
                continue
            mi_value = float(mutual_info_score(discrete_df[feature_a], discrete_df[feature_b]))
            mi_matrix.loc[feature_a, feature_b] = mi_value
            mi_matrix.loc[feature_b, feature_a] = mi_value
            if feature_a != feature_b:
                pair_rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "pairwise_mi": mi_value,
                    }
                )

    label_df = pd.DataFrame(label_rows).sort_values("feature_label_mi", ascending=False).reset_index(drop=True)
    pairs_df = pd.DataFrame(pair_rows).sort_values("pairwise_mi", ascending=False).reset_index(drop=True)
    pairs_df["pair_rank"] = range(1, len(pairs_df) + 1)
    return mi_matrix, label_df, pairs_df


def _model_feature_name(feature: str) -> str:
    if feature == "capital_gain":
        return "capital_gain_log1p"
    if feature == "capital_loss":
        return "capital_loss_log1p"
    return feature


def select_candidate_interactions(pairs_df: pd.DataFrame, label_df: pd.DataFrame) -> pd.DataFrame:
    label_lookup = label_df.set_index("feature")["feature_label_mi"].to_dict()
    numeric_pairs = pairs_df[
        pairs_df["feature_a"].isin(NUMERIC_MI_FEATURES) & pairs_df["feature_b"].isin(NUMERIC_MI_FEATURES)
    ].copy()
    numeric_pairs["dependency_rank"] = range(1, len(numeric_pairs) + 1)
    numeric_pairs["combined_label_mi"] = numeric_pairs["feature_a"].map(label_lookup) + numeric_pairs["feature_b"].map(
        label_lookup
    )

    top_dependency = numeric_pairs.head(config.HIGH_DEPENDENCY_TOP_K).copy()
    top_dependency["selection_source"] = "mi_top_pair"

    selected_pairs = {tuple(sorted((row["feature_a"], row["feature_b"]))) for _, row in top_dependency.iterrows()}
    reserve_rows = []
    for feature_a, feature_b in DOMAIN_PLAUSIBLE_RESERVES:
        key = tuple(sorted((feature_a, feature_b)))
        if key in selected_pairs:
            continue
        mask = (
            ((numeric_pairs["feature_a"] == feature_a) & (numeric_pairs["feature_b"] == feature_b))
            | ((numeric_pairs["feature_a"] == feature_b) & (numeric_pairs["feature_b"] == feature_a))
        )
        if mask.any():
            row = numeric_pairs.loc[mask].iloc[0].copy()
            row["selection_source"] = "domain_plausibility_reserve"
            reserve_rows.append(row)
        if len(top_dependency) + len(reserve_rows) >= config.MAX_INTERACTIONS:
            break

    candidate_df = pd.concat([top_dependency, pd.DataFrame(reserve_rows)], ignore_index=True)
    candidate_df = candidate_df.drop_duplicates(subset=["feature_a", "feature_b"]).copy()
    candidate_df["candidate_rank"] = range(1, len(candidate_df) + 1)
    candidate_df["feature_a_label_mi"] = candidate_df["feature_a"].map(label_lookup)
    candidate_df["feature_b_label_mi"] = candidate_df["feature_b"].map(label_lookup)
    candidate_df["model_feature_a"] = candidate_df["feature_a"].map(_model_feature_name)
    candidate_df["model_feature_b"] = candidate_df["feature_b"].map(_model_feature_name)
    return candidate_df


def plot_mi_heatmap(mi_matrix: pd.DataFrame, output_path) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(mi_matrix.values.astype(float), cmap="viridis")
    ax.set_xticks(range(len(mi_matrix.columns)))
    ax.set_yticks(range(len(mi_matrix.index)))
    ax.set_xticklabels(mi_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(mi_matrix.index)
    ax.set_title("Pairwise Mutual Information Heatmap")
    fig.colorbar(image, ax=ax, shrink=0.8)
    save_figure(fig, output_path)


def plot_top_pairs(pairs_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    top_pairs = pairs_df.head(10).copy()
    top_pairs["pair"] = top_pairs["feature_a"] + " x " + top_pairs["feature_b"]
    top_pairs = top_pairs.sort_values("pairwise_mi", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_pairs["pair"], top_pairs["pairwise_mi"], color="#4c78a8")
    ax.set_title("Top Feature-Feature Mutual Information Pairs")
    ax.set_xlabel("Mutual Information")
    save_figure(fig, output_path)


def plot_feature_label_ranking(label_df: pd.DataFrame, output_path) -> None:
    set_plot_style()
    plot_df = label_df.sort_values("feature_label_mi", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df["feature_label_mi"], color="#f58518")
    ax.set_title("Feature-Label Mutual Information Ranking")
    ax.set_xlabel("Mutual Information with Income Label")
    save_figure(fig, output_path)


def mi_sensitivity_overlap(
    baseline_pairs: pd.DataFrame,
    sensitivity_pairs: pd.DataFrame,
    baseline_candidates: pd.DataFrame,
    sensitivity_candidates: pd.DataFrame,
) -> dict:
    baseline_top = {tuple(sorted((row["feature_a"], row["feature_b"]))) for _, row in baseline_pairs.head(10).iterrows()}
    sensitivity_top = {
        tuple(sorted((row["feature_a"], row["feature_b"]))) for _, row in sensitivity_pairs.head(10).iterrows()
    }
    baseline_candidate_set = {
        tuple(sorted((row["feature_a"], row["feature_b"]))) for _, row in baseline_candidates.iterrows()
    }
    sensitivity_candidate_set = {
        tuple(sorted((row["feature_a"], row["feature_b"]))) for _, row in sensitivity_candidates.iterrows()
    }

    top_overlap = baseline_top & sensitivity_top
    candidate_overlap = baseline_candidate_set & sensitivity_candidate_set
    top_union = baseline_top | sensitivity_top
    candidate_union = baseline_candidate_set | sensitivity_candidate_set
    return {
        "top_overlap_count": len(top_overlap),
        "top_union_count": len(top_union),
        "candidate_overlap_count": len(candidate_overlap),
        "candidate_union_count": len(candidate_union),
        "top_overlap_pairs": [f"{a} x {b}" for a, b in sorted(top_overlap)],
        "candidate_overlap_pairs": [f"{a} x {b}" for a, b in sorted(candidate_overlap)],
    }
