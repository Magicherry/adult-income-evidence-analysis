from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src import config
from src.utils import ensure_directory


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = config.PLOT_DPI
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_directory(path.parent)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=config.PLOT_DPI)
    plt.close(fig)
