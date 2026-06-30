from __future__ import annotations

from pathlib import Path

import pandas as pd

from analisis_descriptivo import generate_tables
from graficos import generate_figures
from inferencia import generate_inference_tables
from utils import TABLES_DIR, format_float, load_dataset


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def metric_value(table: pd.DataFrame, key: str) -> float:
    return float(table.loc[table["medida"] == key, "valor"].iloc[0])


def exact_modes(series: pd.Series) -> tuple[list[float], int]:
    counts = series.dropna().value_counts()
    max_count = int(counts.max())
    values = sorted(float(v) for v in counts[counts == max_count].index)
    return values, max_count


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, max_rows: int | None = None) -> None:
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: format_float(x, 4))
    headers = [tex_escape(col) for col in view.columns]
    rows = []
    for record in view.astype(str).itertuples(index=False, name=None):
        rows.append(" & ".join(tex_escape(value) for value in record) + r" \\")
    align = "l" * len(headers)
    latex = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            rf"\caption{{{tex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{{align}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(latex, encoding="utf-8")


def write_result_macros(descriptive: dict[str, pd.DataFrame], inference: dict[str, pd.DataFrame], figures: list[str]) -> None:
    df = load_dataset()
    cambio = descriptive["descriptivo_cambio"]
    boot = inference["bootstrap_media_uyu_por_brl"].iloc[0]
    corr = inference["correlaciones"]
    corr_brl_uyu = corr[(corr["variable_x"] == "brl_por_usd") & (corr["variable_y"] == "uyu_por_usd")].iloc[0]
    chi2_table = inference["bondad_ajuste_chi2_uyu_por_brl"]
    chi2_fit = chi2_table.iloc[0]
    chi2_second = chi2_table.iloc[1]
    acf = inference["autocorrelacion_uyu_por_brl"].set_index("lag")["autocorrelacion"]
    resumen_lookup = dict(zip(descriptive["resumen"]["indicador"], descriptive["resumen"]["valor"]))
    modes, mode_freq = exact_modes(df["uyu_por_brl"])

    macros = {
        "TotalRegistros": len(df),
        "TotalVariables": df.shape[1],
        "FechaMinima": df["fecha"].min().date().isoformat(),
        "FechaMaxima": df["fecha"].max().date().isoformat(),
        "NulosTotales": int(df.isna().sum().sum()),
        "DuplicadosFecha": int(float(resumen_lookup.get("duplicados_fecha", 0))),
        "MaxErrorProductoInverso": format_float(float(resumen_lookup.get("max_error_producto_inverso", 0)), 8),
        "MediaCambio": format_float(metric_value(cambio, "media"), 4),
        "MedianaCambio": format_float(metric_value(cambio, "mediana"), 4),
        "ModaCambioUno": format_float(modes[0], 4),
        "ModaCambioDos": format_float(modes[1], 4) if len(modes) > 1 else format_float(modes[0], 4),
        "FrecuenciaModaCambio": mode_freq,
        "DesvioCambio": format_float(metric_value(cambio, "desvio_estandar_muestral"), 4),
        "VarianzaCambio": format_float(metric_value(cambio, "varianza_muestral"), 4),
        "QunoCambio": format_float(metric_value(cambio, "q1"), 4),
        "QtresCambio": format_float(metric_value(cambio, "q3"), 4),
        "IqrCambio": format_float(metric_value(cambio, "iqr"), 4),
        "AsimetriaCambio": format_float(metric_value(cambio, "asimetria"), 4),
        "CurtosisCambio": format_float(metric_value(cambio, "curtosis"), 4),
        "OutliersCambio": int(metric_value(cambio, "cantidad_outliers_iqr")),
        "BootstrapMediaInf": format_float(boot["limite_inferior"], 4),
        "BootstrapMediaSup": format_float(boot["limite_superior"], 4),
        "CorrBrlUsdUyuUsd": format_float(corr_brl_uyu["pearson_r"], 4),
        "MejorDistribucionChi": tex_escape(chi2_fit["distribucion"]),
        "MejorDistribucionChiEst": format_float(chi2_fit["chi2_estadistico"], 2),
        "MejorDistribucionChiGl": int(chi2_fit["grados_libertad"]),
        "MejorDistribucionChiP": "p<0.001" if chi2_fit["p_value"] < 0.001 else format_float(chi2_fit["p_value"], 4),
        "SegundaDistribucionChi": tex_escape(chi2_second["distribucion"]),
        "SegundaDistribucionChiEst": format_float(chi2_second["chi2_estadistico"], 2),
        "SegundaDistribucionChiGl": int(chi2_second["grados_libertad"]),
        "AcfLagUno": format_float(acf.loc[1], 4),
        "AcfLagDos": format_float(acf.loc[2], 4),
        "AcfLagCinco": format_float(acf.loc[5], 4),
        "AcfLagDiez": format_float(acf.loc[10], 4),
        "AcfLagVeinte": format_float(acf.loc[20], 4),
        "AcfLagTreinta": format_float(acf.loc[30], 4),
        "CantidadFiguras": len(figures),
    }

    lines = ["% Archivo generado automaticamente por src/run_all.py\n"]
    for key, value in macros.items():
        lines.append(f"\\newcommand{{\\{key}}}{{{value}}}\n")
    (TABLES_DIR / "resultados.tex").write_text("".join(lines), encoding="utf-8")


def write_latex_fragments(descriptive: dict[str, pd.DataFrame], inference: dict[str, pd.DataFrame]) -> None:
    write_latex_table(
        descriptive["descriptivo_cambio"],
        TABLES_DIR / "descriptivo_uyu_por_brl.tex",
        "Medidas descriptivas de la cotizacion UYU/BRL.",
        "tab:descriptivo-cambio",
    )
    write_latex_table(
        inference["ic_media_uyu_por_brl"],
        TABLES_DIR / "ic_media_uyu_por_brl.tex",
        "Intervalos de confianza clasicos para la media de UYU/BRL.",
        "tab:ic-media-cambio",
    )
    write_latex_table(
        inference["ic_varianza_uyu_por_brl"],
        TABLES_DIR / "ic_varianza_uyu_por_brl.tex",
        "Intervalos de confianza clasicos para la varianza de UYU/BRL.",
        "tab:ic-varianza-cambio",
    )
    write_latex_table(
        inference["bondad_ajuste_chi2_uyu_por_brl"],
        TABLES_DIR / "bondad_ajuste_chi2_uyu_por_brl.tex",
        "Comparacion de distribuciones candidatas mediante prueba chi-cuadrado.",
        "tab:bondad-ajuste-chi2",
    )
    write_latex_table(
        inference["bondad_ajuste_chi2_clases_lognormal"],
        TABLES_DIR / "bondad_ajuste_chi2_clases_lognormal.tex",
        "Frecuencias observadas y esperadas para el ajuste Log-normal.",
        "tab:bondad-ajuste-clases-lognormal",
    )
    write_latex_table(
        inference["correlaciones"],
        TABLES_DIR / "correlaciones.tex",
        "Correlaciones de Pearson entre variables cambiarias.",
        "tab:correlaciones",
    )
    write_latex_table(
        inference["autocorrelacion_uyu_por_brl"],
        TABLES_DIR / "autocorrelacion_uyu_por_brl.tex",
        "Autocorrelaciones seleccionadas de la cotizacion UYU/BRL.",
        "tab:autocorrelacion-cambio",
    )


def main() -> None:
    descriptive = generate_tables()
    inference = generate_inference_tables()
    figures = generate_figures()
    write_result_macros(descriptive, inference, figures)
    write_latex_fragments(descriptive, inference)
    print(f"Tablas generadas en {TABLES_DIR}")
    print(f"Figuras generadas: {len(figures)}")


if __name__ == "__main__":
    main()
