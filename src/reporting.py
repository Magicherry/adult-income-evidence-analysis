from __future__ import annotations

import pandas as pd


def _format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"
    columns = [str(column) for column in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_format_cell(row[column]) for column in df.columns) + " |")
    return "\n".join([header, separator, *rows])


def markdown_section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n"
