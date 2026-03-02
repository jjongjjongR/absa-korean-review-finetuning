from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Step1Config:
    text_col: str = "text"
    label_col: str = "label"
    train_size_per_class: int = 2000
    random_state: int = 16


def prepare_binary_dataset(df: pd.DataFrame, cfg: Step1Config) -> pd.DataFrame:
    """
    Notebook logic:
    - group by text, mean label
    - threshold >= 0.5 -> 1 else 0
    - sample balanced 0/1
    """

    if cfg.text_col not in df.columns or cfg.label_col not in df.columns:
        raise KeyError(f"Expected columns: {cfg.text_col}, {cfg.label_col}. Got: {list(df.columns)}")

    df_gb = df.groupby(cfg.text_col)[cfg.label_col].mean().reset_index()
    df_gb[cfg.label_col] = np.where(df_gb[cfg.label_col] >= 0.5, 1, 0)

    df_0 = df_gb[df_gb[cfg.label_col] == 0].sample(cfg.train_size_per_class, random_state=cfg.random_state)
    df_1 = df_gb[df_gb[cfg.label_col] == 1].sample(cfg.train_size_per_class, random_state=cfg.random_state)

    out = pd.concat([df_0, df_1], axis=0).sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)
    return out

