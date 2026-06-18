from __future__ import annotations

import pandas as pd

from utils import TABLES_DIR, clean_numeric, ensure_output_dirs, iqr_outlier_bounds, load_dataset


def descriptive_metrics(series: pd.Series) -> pd.DataFrame:
    x = clean_numeric(series)
    lower, upper, outliers = iqr_outlier_bounds(x)
    mode = x.mode()
    rows = [
        ("n", x.shape[0]),
        ("media", x.mean()),
        ("mediana", x.median()),
        ("moda", mode.iloc[0] if not mode.empty else pd.NA),
        ("q1", x.quantile(0.25)),
        ("q3", x.quantile(0.75)),
        ("iqr", x.quantile(0.75) - x.quantile(0.25)),
        ("varianza_muestral", x.var(ddof=1)),
        ("desvio_estandar_muestral", x.std(ddof=1)),
        ("minimo", x.min()),
        ("maximo", x.max()),
        ("rango", x.max() - x.min()),
        ("coeficiente_variacion", x.std(ddof=1) / x.mean()),
        ("asimetria", x.skew()),
        ("curtosis", x.kurt()),
        ("limite_outlier_inferior", lower),
        ("limite_outlier_superior", upper),
        ("cantidad_outliers_iqr", outliers),
    ]
    return pd.DataFrame(rows, columns=["medida", "valor"])


def generate_tables() -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    df = load_dataset()

    resumen = pd.DataFrame(
        [
            ("filas", len(df)),
            ("variables", df.shape[1]),
            ("fecha_minima", df["fecha"].min().date().isoformat()),
            ("fecha_maxima", df["fecha"].max().date().isoformat()),
            ("nulos_totales", int(df.isna().sum().sum())),
            ("duplicados_fecha", int(df["fecha"].duplicated().sum())),
            ("max_error_producto_inverso", float((df["uyu_por_brl"] * df["brl_por_uyu"] - 1).abs().max())),
        ],
        columns=["indicador", "valor"],
    )
    nulos = df.isna().sum().reset_index()
    nulos.columns = ["variable", "nulos"]

    descriptivo_cambio = descriptive_metrics(df["uyu_por_brl"])
    descriptivo_retornos = descriptive_metrics(df["log_retorno_brl_por_uyu"])
    descriptivo_retornos_uyu = descriptive_metrics(df["log_retorno_uyu_por_brl"])

    anual = (
        df.assign(anio=df["fecha"].dt.year)
        .groupby("anio", as_index=False)
        .agg(
            media_uyu_por_brl=("uyu_por_brl", "mean"),
            desvio_uyu_por_brl=("uyu_por_brl", "std"),
            media_log_retorno=("log_retorno_brl_por_uyu", "mean"),
            desvio_log_retorno=("log_retorno_brl_por_uyu", "std"),
            observaciones=("uyu_por_brl", "count"),
        )
    )

    for name, table in {
        "resumen_dataset": resumen,
        "nulos": nulos,
        "descriptivo_uyu_por_brl": descriptivo_cambio,
        "descriptivo_log_retornos": descriptivo_retornos,
        "descriptivo_log_retornos_uyu_por_brl": descriptivo_retornos_uyu,
        "resumen_anual": anual,
    }.items():
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False)

    return {
        "resumen": resumen,
        "nulos": nulos,
        "descriptivo_cambio": descriptivo_cambio,
        "descriptivo_retornos": descriptivo_retornos,
        "descriptivo_retornos_uyu": descriptivo_retornos_uyu,
        "anual": anual,
    }


if __name__ == "__main__":
    generate_tables()
