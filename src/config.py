"""Shared paths and helpers for generated outputs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Wage.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


def save_text(filename: str, content: str) -> None:
    """Write a UTF-8 text file in the output directory."""
    output_path = OUTPUT_DIR / filename
    output_path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _format_value(value: Any, float_digits: int) -> str:
    """Format one table cell for plain-text export."""
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if number == 0:
            return "0"
        if abs(number) < 1e-4 or abs(number) >= 1e6:
            return f"{number:.6e}"
        if number.is_integer():
            return str(int(number))
        return f"{number:.{float_digits}f}"
    return str(value)


def save_dataframe(
    filename: str, dataframe: pd.DataFrame, float_digits: int = 6
) -> None:
    """Export a dataframe as a clean plain-text table."""
    formatted = dataframe.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(
            lambda value: _format_value(value, float_digits)
        )
    save_text(filename, formatted.to_string(index=False))
