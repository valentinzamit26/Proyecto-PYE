from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from utils import TABLES_DIR, clean_numeric, ensure_output_dirs, load_dataset


CONF_LEVELS = [0.90, 0.95, 0.99]


def mean_confidence_intervals(series: pd.Series) -> pd.DataFrame:
    x = clean_numeric(series)
    n = len(x)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    rows = []
    for level in CONF_LEVELS:
        alpha = 1 - level
        crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        rows.append(
            {
                "nivel_confianza": level,
                "media": mean,
                "limite_inferior": mean - crit * se,
                "limite_superior": mean + crit * se,
                "amplitud": 2 * crit * se,
                "metodo": "t de Student",
            }
        )
    return pd.DataFrame(rows)


def variance_confidence_intervals(series: pd.Series) -> pd.DataFrame:
    x = clean_numeric(series)
    n = len(x)
    s2 = x.var(ddof=1)
    rows = []
    for level in CONF_LEVELS:
        alpha = 1 - level
        lower = (n - 1) * s2 / stats.chi2.ppf(1 - alpha / 2, df=n - 1)
        upper = (n - 1) * s2 / stats.chi2.ppf(alpha / 2, df=n - 1)
        rows.append(
            {
                "nivel_confianza": level,
                "varianza_muestral": s2,
                "limite_inferior": lower,
                "limite_superior": upper,
                "amplitud": upper - lower,
                "metodo": "chi-cuadrado",
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(series: pd.Series, level: float = 0.95, n_boot: int = 5000, seed: int = 42) -> pd.DataFrame:
    x = clean_numeric(series).to_numpy()
    rng = np.random.default_rng(seed)
    boot = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    alpha = 1 - level
    return pd.DataFrame(
        [
            {
                "nivel_confianza": level,
                "media_bootstrap": boot.mean(),
                "limite_inferior": np.quantile(boot, alpha / 2),
                "limite_superior": np.quantile(boot, 1 - alpha / 2),
                "amplitud": np.quantile(boot, 1 - alpha / 2) - np.quantile(boot, alpha / 2),
                "metodo": "bootstrap percentil",
                "repeticiones": n_boot,
                "semilla": seed,
            }
        ]
    )


def normality_tests(series: pd.Series, name: str) -> pd.DataFrame:
    x = clean_numeric(series)
    # D'Agostino funciona bien en muestras grandes. Shapiro se limita por tamano.
    dag = stats.normaltest(x)
    ks = stats.kstest((x - x.mean()) / x.std(ddof=1), "norm")
    return pd.DataFrame(
        [
            {"variable": name, "prueba": "D'Agostino-Pearson", "estadistico": dag.statistic, "p_value": dag.pvalue},
            {"variable": name, "prueba": "Kolmogorov-Smirnov estandarizado", "estadistico": ks.statistic, "p_value": ks.pvalue},
        ]
    )


def distribution_fit(series: pd.Series) -> pd.DataFrame:
    x = clean_numeric(series)
    rows = []

    candidates = {
        "normal": stats.norm.fit(x),
        "lognormal": stats.lognorm.fit(x, floc=0) if (x > 0).all() else None,
        "gamma": stats.gamma.fit(x, floc=0) if (x > 0).all() else None,
    }

    for name, params in candidates.items():
        if params is None:
            continue
        dist = {"normal": stats.norm, "lognormal": stats.lognorm, "gamma": stats.gamma}[name]
        ks = stats.kstest(x, dist.cdf, args=params)
        rows.append(
            {
                "distribucion": name,
                "parametros": "; ".join(f"{p:.6g}" for p in params),
                "ks_estadistico": ks.statistic,
                "p_value": ks.pvalue,
            }
        )
    return pd.DataFrame(rows).sort_values("ks_estadistico")


def chi_square_distribution_fit(series: pd.Series, bins: int = 12) -> pd.DataFrame:
    x = clean_numeric(series)
    n = len(x)
    candidates = {
        "normal": (stats.norm, stats.norm.fit(x), 2),
        "lognormal": (stats.lognorm, stats.lognorm.fit(x, floc=0), 3) if (x > 0).all() else None,
        "gamma": (stats.gamma, stats.gamma.fit(x, floc=0), 3) if (x > 0).all() else None,
    }
    rows = []
    for name, item in candidates.items():
        if item is None:
            continue
        dist, params, estimated_params = item
        probabilities = np.linspace(0, 1, bins + 1)
        edges = dist.ppf(probabilities, *params)
        edges[0] = -np.inf
        edges[-1] = np.inf
        observed, _ = np.histogram(x, bins=edges)
        expected = np.full(bins, n / bins)
        statistic = float(((observed - expected) ** 2 / expected).sum())
        dof = max(bins - 1 - estimated_params, 1)
        rows.append(
            {
                "distribucion": name,
                "clases": bins,
                "frecuencia_esperada_minima": expected.min(),
                "grados_libertad": dof,
                "chi2_estadistico": statistic,
                "p_value": stats.chi2.sf(statistic, dof),
            }
        )
    return pd.DataFrame(rows).sort_values("chi2_estadistico")


def correlation_tests(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("brl_por_usd", "uyu_por_usd"),
        ("brl_por_usd", "uyu_por_brl"),
        ("uyu_por_usd", "uyu_por_brl"),
        ("brl_por_uyu", "uyu_por_brl"),
    ]
    rows = []
    for a, b in pairs:
        clean = df[[a, b]].dropna()
        r, p = stats.pearsonr(clean[a], clean[b])
        rows.append({"variable_x": a, "variable_y": b, "pearson_r": r, "p_value": p, "n": len(clean)})
    return pd.DataFrame(rows)


def autocorrelation_table(series: pd.Series, nlags: int = 20) -> pd.DataFrame:
    x = clean_numeric(series).to_numpy()
    x = x - x.mean()
    denom = np.dot(x, x)
    values = [1.0]
    for lag in range(1, nlags + 1):
        values.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return pd.DataFrame({"lag": range(nlags + 1), "autocorrelacion": values})


def generate_inference_tables() -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    df = load_dataset()
    cambio = df["uyu_por_brl"]
    log_ret = df["log_retorno_brl_por_uyu"]

    tables = {
        "ic_media_uyu_por_brl": mean_confidence_intervals(cambio),
        "ic_varianza_uyu_por_brl": variance_confidence_intervals(cambio),
        "ic_media_log_retornos": mean_confidence_intervals(log_ret),
        "ic_varianza_log_retornos": variance_confidence_intervals(log_ret),
        "bootstrap_media_uyu_por_brl": bootstrap_mean_ci(cambio),
        "normalidad": pd.concat(
            [
                normality_tests(cambio, "uyu_por_brl"),
                normality_tests(log_ret, "log_retorno_brl_por_uyu"),
            ],
            ignore_index=True,
        ),
        "bondad_ajuste_uyu_por_brl": distribution_fit(cambio),
        "bondad_ajuste_chi2_uyu_por_brl": chi_square_distribution_fit(cambio),
        "correlaciones": correlation_tests(df),
        "autocorrelacion_log_retornos": autocorrelation_table(log_ret),
    }

    for name, table in tables.items():
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False)

    return tables


if __name__ == "__main__":
    generate_inference_tables()
