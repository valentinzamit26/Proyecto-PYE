from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from utils import DATA_PATH, FIGURES_DIR


VARIABLE = "uyu_por_brl"


def cargar_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values("fecha")


def main() -> None:
    df = cargar_dataset()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df["fecha"], df[VARIABLE], color="#1f77b4", linewidth=1.5)
    plt.title("Evolucion diaria del tipo de cambio UYU por BRL")
    plt.xlabel("Fecha")
    plt.ylabel("UYU por 1 BRL")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "grafico_basico_evolucion_uyu_por_brl.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(df[VARIABLE].dropna(), bins=40, color="#2ca02c", edgecolor="black", alpha=0.75)
    plt.title("Histograma del tipo de cambio UYU por BRL")
    plt.xlabel("UYU por 1 BRL")
    plt.ylabel("Frecuencia")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "grafico_basico_histograma_uyu_por_brl.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()

