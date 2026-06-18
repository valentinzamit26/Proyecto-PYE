from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "2005-01-01"
END_DATE = "2026-06-01"
ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = ROOT / "data" / "raw" / "dataset_uyu_brl_2005_2026.csv"

TICKERS = {
    "uyu_por_usd": "USDUYU=X",
    "brl_por_usd": "USDBRL=X",
}


def descargar_datos() -> pd.DataFrame:
    data = yf.download(
        list(TICKERS.values()),
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise ValueError("No se descargaron datos. Revisar conexion o tickers.")

    close = data["Close"].rename(
        columns={
            "USDUYU=X": "uyu_por_usd",
            "USDBRL=X": "brl_por_usd",
        }
    )
    return close.dropna().copy()


def crear_dataset(close: pd.DataFrame) -> pd.DataFrame:
    df = close.copy()
    df["brl_por_uyu"] = df["brl_por_usd"] / df["uyu_por_usd"]
    df["uyu_por_brl"] = df["uyu_por_usd"] / df["brl_por_usd"]
    df["retorno_diario_brl_por_uyu"] = df["brl_por_uyu"].pct_change()
    df["log_retorno_brl_por_uyu"] = np.log(df["brl_por_uyu"]).diff()
    df["retorno_diario_uyu_por_brl"] = df["uyu_por_brl"].pct_change()
    df["log_retorno_uyu_por_brl"] = np.log(df["uyu_por_brl"]).diff()
    df = df.reset_index().rename(columns={"Date": "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def main() -> None:
    close = descargar_datos()
    df = crear_dataset(close)
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    print(f"Archivo creado: {CSV_FILE}")


if __name__ == "__main__":
    main()

