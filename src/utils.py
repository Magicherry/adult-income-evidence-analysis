from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src import config


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_directories() -> None:
    for path in [
        config.DATA_DIR,
        config.PROCESSED_DIR,
        config.SPLITS_DIR,
        config.NOTEBOOKS_DIR,
        config.SRC_DIR,
        config.OUTPUTS_DIR,
        config.FIGURES_DIR,
        config.TABLES_DIR,
        config.METRICS_DIR,
    ]:
        ensure_directory(path)


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_directory(path.parent)
    df.to_csv(path, index=index)


def save_json(payload: Any, path: Path) -> None:
    import json

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> Any:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
