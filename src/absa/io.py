from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_data_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}. Put your dataset at data/raw/data.csv (see README.md)."
        )
    return pd.read_csv(path)

