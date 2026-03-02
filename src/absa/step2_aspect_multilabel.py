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
    Convert (text, aspect, label) long table into wide multi-label format:
      - one row per text
      - one column per aspect (0/1)
    """

    for col in (cfg.text_col, cfg.aspect_col, cfg.label_col):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}'. Got: {list(df.columns)}")

    df = df[[cfg.text_col, cfg.aspect_col, cfg.label_col]].drop_duplicates([cfg.text_col, cfg.aspect_col])

    wide = (
        df.pivot_table(index=cfg.text_col, columns=cfg.aspect_col, values=cfg.label_col, fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return wide

