from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Step2Config:
    text_col: str = "text"
    aspect_col: str = "aspect"
    label_col: str = "label"


def to_multilabel_table(df: pd.DataFrame, cfg: Step2Config) -> pd.DataFrame:
    """
    Match the notebook transformation:
    - one-hot encode `aspect`
    - aggregate per review text with max
    - keep one row per text with 0/1 aspect columns
    """

    for col in (cfg.text_col, cfg.aspect_col, cfg.label_col):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}'. Got: {list(df.columns)}")

    df_oe = pd.get_dummies(
        df[[cfg.text_col, cfg.aspect_col, cfg.label_col]],
        columns=[cfg.aspect_col],
        dtype=int,
        prefix="",
        prefix_sep="",
    )
    aspect_cols = [c for c in df_oe.columns if c not in {cfg.text_col, cfg.label_col}]
    if not aspect_cols:
        raise ValueError("No aspect columns were generated from the input data.")

    for col in aspect_cols:
        df_oe[col] = df_oe[col] * df_oe[cfg.label_col].astype(int)

    wide = df_oe.groupby(cfg.text_col, as_index=False)[aspect_cols].max()
    return wide

