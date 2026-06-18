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

    plt.figure(figsize=(5, 5))
    stats.probplot(df["uyu_por_brl"].dropna(), dist="norm", plot=plt)
    plt.title("Q-Q plot de UYU/BRL frente a Normal")
    savefig("qq_uyu_por_brl.png")
    generated.append("qq_uyu_por_brl.png")

    plt.figure(figsize=(7, 4))
    plt.hist(df["log_retorno_brl_por_uyu"].dropna(), bins=60, color="#2ca02c", alpha=0.78, edgecolor="white")
    plt.title("Distribucion empirica de log-retornos diarios")
    plt.xlabel("Log-retorno diario BRL/UYU")
    plt.ylabel("Frecuencia")
    savefig("hist_log_retornos.png")
    generated.append("hist_log_retornos.png")

    plt.figure(figsize=(5, 5))
    stats.probplot(df["log_retorno_brl_por_uyu"].dropna(), dist="norm", plot=plt)
    plt.title("Q-Q plot de log-retornos frente a Normal")
    savefig("qq_log_retornos.png")
    generated.append("qq_log_retornos.png")

    x = df["log_retorno_brl_por_uyu"].dropna().to_numpy()
    x = x - x.mean()
    denom = np.dot(x, x)
    lags = np.arange(31)
    acf_values = [1.0] + [float(np.dot(x[:-lag], x[lag:]) / denom) for lag in range(1, 31)]
    conf = 1.96 / np.sqrt(len(x))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(lags, acf_values, color="#4c78a8")
    ax.axhline(conf, color="#d62728", linestyle="--", linewidth=1)
    ax.axhline(-conf, color="#d62728", linestyle="--", linewidth=1)
    ax.set_title("Autocorrelacion de log-retornos diarios")
    ax.set_xlabel("Rezago")
    ax.set_ylabel("Autocorrelacion")
    savefig("acf_log_retornos.png")
    generated.append("acf_log_retornos.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(df["brl_por_usd"], df["uyu_por_usd"], s=12, alpha=0.45, color="#9467bd", edgecolors="none")
    plt.title("Relacion entre BRL/USD y UYU/USD")
    plt.xlabel("BRL por USD")
    plt.ylabel("UYU por USD")
    savefig("scatter_brl_usd_vs_uyu_usd.png")
    generated.append("scatter_brl_usd_vs_uyu_usd.png")

    rolling = df.set_index("fecha")["log_retorno_brl_por_uyu"].rolling(60).std()
    plt.figure(figsize=(8, 4))
    rolling.plot(color="#d62728", linewidth=1)
    plt.title("Volatilidad movil de 60 ruedas - log-retornos")
    plt.xlabel("Fecha")
    plt.ylabel("Desvio estandar movil")
    savefig("volatilidad_movil_60d.png")
    generated.append("volatilidad_movil_60d.png")

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
