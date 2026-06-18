from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "dataset_uyu_brl_2005_2026.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"

EXPECTED_COLUMNS = [
    "fecha",
    "brl_por_usd",
    "uyu_por_usd",
    "brl_por_uyu",
    "uyu_por_brl",
    "retorno_diario_brl_por_uyu",
    "log_retorno_brl_por_uyu",
]


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el CSV esperado: {path}")
    df = pd.read_csv(path)
    validate_columns(df, EXPECTED_COLUMNS)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.sort_values("fecha").reset_index(drop=True)
    if "retorno_diario_uyu_por_brl" not in df.columns:
        df["retorno_diario_uyu_por_brl"] = df["uyu_por_brl"].pct_change()
    if "log_retorno_uyu_por_brl" not in df.columns:
        positive = pd.to_numeric(df["uyu_por_brl"], errors="coerce")
        df["log_retorno_uyu_por_brl"] = np.log(positive).diff()
    return df


def validate_columns(df: pd.DataFrame, expected_columns: list[str]) -> None:
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def iqr_outlier_bounds(series: pd.Series) -> tuple[float, float, int]:
    x = clean_numeric(series)
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((x < lower) | (x > upper)).sum())
    return float(lower), float(upper), count


def format_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"
