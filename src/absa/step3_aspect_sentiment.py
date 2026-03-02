from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Step3Config:
    text_col: str = "text"
    aspect_col: str = "aspect"
    label_col: str = "label"


def build_input_texts(df: pd.DataFrame, cfg: Step3Config, fmt: str = "case1") -> list[str]:
    for col in (cfg.text_col, cfg.aspect_col, cfg.label_col):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}'. Got: {list(df.columns)}")

    if fmt == "case1":
        return [f"[ASPECT] {a} [SEP] {t}" for a, t in zip(df[cfg.aspect_col], df[cfg.text_col])]
    if fmt == "case2":
        return [f"{a} : {t}" for a, t in zip(df[cfg.aspect_col], df[cfg.text_col])]
    raise ValueError("fmt must be 'case1' or 'case2'")

