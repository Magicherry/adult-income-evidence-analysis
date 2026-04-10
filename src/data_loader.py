from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src import config


def _strip_object_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
            df.loc[df[column].isin({"nan", "None"}), column] = np.nan
    return df


def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(config.RAW_DATA_PATH)


def load_standardized_data() -> pd.DataFrame:
    df = load_raw_data().rename(columns=config.RAW_TO_CLEAN_COLUMN_MAP).copy()
    df.insert(0, config.ROW_ID_COLUMN, range(len(df)))
    object_columns = [column for column in df.columns if df[column].dtype == "object"]
    df = _strip_object_columns(df, object_columns)
    for column in object_columns:
        df.loc[df[column] == "?", column] = np.nan
    return df


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "nunique": int(series.nunique(dropna=True)),
                "example_values": " | ".join(series.dropna().astype(str).head(5).tolist()),
            }
        )
    return pd.DataFrame(rows)
