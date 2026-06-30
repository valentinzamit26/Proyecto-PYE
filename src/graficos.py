from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils import FIGURES_DIR, ensure_output_dirs, load_dataset


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def generate_figures() -> list[str]:
    ensure_output_dirs()
    df = load_dataset()
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")
    palette = "#1f77b4"
    generated: list[str] = []

    plt.figure(figsize=(8, 4))
    plt.plot(df["fecha"], df["uyu_por_brl"], color=palette, linewidth=1)
    plt.title("Evolucion diaria del tipo de cambio UYU/BRL")
    plt.xlabel("Fecha")
    plt.ylabel("UYU por BRL")
    savefig("serie_uyu_por_brl.png")
    generated.append("serie_uyu_por_brl.png")

    plt.figure(figsize=(7, 4))
    plt.hist(df["uyu_por_brl"].dropna(), bins=40, color=palette, alpha=0.78, edgecolor="white")
    plt.title("Distribucion empirica del tipo de cambio UYU/BRL")
    plt.xlabel("UYU por BRL")
    plt.ylabel("Frecuencia")
    savefig("hist_uyu_por_brl.png")
    generated.append("hist_uyu_por_brl.png")

    plt.figure(figsize=(6, 3))
    plt.boxplot(df["uyu_por_brl"].dropna(), vert=False, patch_artist=True, boxprops={"facecolor": "#9ecae1"})
    plt.title("Boxplot del tipo de cambio UYU/BRL")
    plt.xlabel("UYU por BRL")
    savefig("boxplot_uyu_por_brl.png")
    generated.append("boxplot_uyu_por_brl.png")

    lag_data = df[["uyu_por_brl"]].copy()
    lag_data["lag1"] = lag_data["uyu_por_brl"].shift(1)
    lag_data["lag2"] = lag_data["uyu_por_brl"].shift(2)

    plt.figure(figsize=(5.5, 5))
    plt.scatter(lag_data["lag1"], lag_data["uyu_por_brl"], s=10, alpha=0.45, color="#1f77b4", edgecolors="none")
    plt.title("Diagrama de dispersion: X_t vs X_{t-1}")
    plt.xlabel("UYU/BRL en t-1")
    plt.ylabel("UYU/BRL en t")
    savefig("lag1_uyu_por_brl.png")
    generated.append("lag1_uyu_por_brl.png")

    plt.figure(figsize=(5.5, 5))
    plt.scatter(lag_data["lag2"], lag_data["uyu_por_brl"], s=10, alpha=0.45, color="#1f77b4", edgecolors="none")
    plt.title("Diagrama de dispersion: X_t vs X_{t-2}")
    plt.xlabel("UYU/BRL en t-2")
    plt.ylabel("UYU/BRL en t")
    savefig("lag2_uyu_por_brl.png")
    generated.append("lag2_uyu_por_brl.png")

    plt.figure(figsize=(5, 5))
    stats.probplot(df["uyu_por_brl"].dropna(), dist="norm", plot=plt)
    plt.title("Q-Q plot de UYU/BRL frente a Normal")
    savefig("qq_uyu_por_brl.png")
    generated.append("qq_uyu_por_brl.png")

    x_level = df["uyu_por_brl"].dropna().to_numpy()
    x_level = x_level - x_level.mean()
    denom_level = np.dot(x_level, x_level)
    lags_level = np.arange(31)
    acf_level = [1.0] + [float(np.dot(x_level[:-lag], x_level[lag:]) / denom_level) for lag in range(1, 31)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(lags_level, acf_level, color="#4c78a8")
    ax.set_title("Autocorrelacion de la cotizacion UYU/BRL")
    ax.set_xlabel("Rezago")
    ax.set_ylabel("Autocorrelacion")
    ax.set_ylim(0, 1.05)
    savefig("acf_uyu_por_brl.png")
    generated.append("acf_uyu_por_brl.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(df["brl_por_usd"], df["uyu_por_usd"], s=12, alpha=0.45, color="#9467bd", edgecolors="none")
    plt.title("Relacion entre BRL/USD y UYU/USD")
    plt.xlabel("BRL por USD")
    plt.ylabel("UYU por USD")
    savefig("scatter_brl_usd_vs_uyu_usd.png")
    generated.append("scatter_brl_usd_vs_uyu_usd.png")

    plt.figure(figsize=(6, 5))
    corr = df.drop(columns=["fecha"]).corr(numeric_only=True)
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Matriz de correlacion de variables numericas")
    savefig("matriz_correlacion.png")
    generated.append("matriz_correlacion.png")

    return generated


if __name__ == "__main__":
    generate_figures()
