from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.utils import load_json, save_json


def split_manifest_path(seed: int) -> Path:
    return config.SPLITS_DIR / f"split_seed_{seed}.json"


def create_stratified_split(df: pd.DataFrame, seed: int) -> dict:
    train_ids, test_ids = train_test_split(
        df[config.ROW_ID_COLUMN],
        test_size=config.TEST_SIZE,
        random_state=seed,
        stratify=df[config.LABEL_COLUMN],
    )
    manifest = {
        "seed": seed,
        "test_size": config.TEST_SIZE,
        "train_row_ids": sorted(int(value) for value in train_ids.tolist()),
        "test_row_ids": sorted(int(value) for value in test_ids.tolist()),
        "train_size": int(len(train_ids)),
        "test_size_rows": int(len(test_ids)),
    }
    save_json(manifest, split_manifest_path(seed))
    return manifest


def load_or_create_split(df: pd.DataFrame, seed: int) -> dict:
    path = split_manifest_path(seed)
    if path.exists():
        return load_json(path)
    return create_stratified_split(df, seed)


def apply_split(df: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids = set(manifest["train_row_ids"])
    test_ids = set(manifest["test_row_ids"])
    train_df = df[df[config.ROW_ID_COLUMN].isin(train_ids)].copy()
    test_df = df[df[config.ROW_ID_COLUMN].isin(test_ids)].copy()
    return train_df, test_df
