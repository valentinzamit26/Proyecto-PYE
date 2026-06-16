# Codigo Proyecto-PYE
import numpy as np
import pandas as pd
import yfinance as yf


START_DATE = "2005-01-01"
END_DATE = "2026-06-01"  # yfinance no incluye la fecha final

CSV_FILE = "dataset_uyu_brl_2005_2026.csv"

TICKERS = {
    "uyu_por_usd": "USDUYU=X",  # pesos uruguayos por 1 dolar
    "brl_por_usd": "USDBRL=X",  # reales brasilenos por 1 dolar
}


def descargar_datos():
    data = yf.download(
        list(TICKERS.values()),
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError("No se descargaron datos. Revisa la conexion o los tickers.")

    close = data["Close"].rename(
        columns={
            "USDUYU=X": "uyu_por_usd",
            "USDBRL=X": "brl_por_usd",
        }
    )

    return close.dropna().copy()


def crear_dataset(close):
    df = close.copy()

    # Formula del proyecto:
    # BRL por UYU = (BRL por USD) / (UYU por USD)
    df["brl_por_uyu"] = df["brl_por_usd"] / df["uyu_por_usd"]

    # Columna inversa, util para interpretar: cuantos pesos uruguayos vale 1 real.
    df["uyu_por_brl"] = df["uyu_por_usd"] / df["brl_por_usd"]

    df["retorno_diario_brl_por_uyu"] = df["brl_por_uyu"].pct_change()
    df["log_retorno_brl_por_uyu"] = np.log(df["brl_por_uyu"]).diff()

    df = df.reset_index().rename(columns={"Date": "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"])

    return df


def crear_resumen(df):
    columnas = ["brl_por_uyu", "uyu_por_brl"]
    medidas = {
        "cantidad": "count",
        "media": "mean",
        "mediana": "median",
        "cuartil_1": lambda x: x.quantile(0.25),
        "cuartil_3": lambda x: x.quantile(0.75),
        "varianza": "var",
        "desvio_estandar": "std",
        "minimo": "min",
        "maximo": "max",
    }

    filas = []
    for nombre, funcion in medidas.items():
        fila = {"medida": nombre}
        for columna in columnas:
            serie = df[columna].dropna()
            fila[columna] = getattr(serie, funcion)() if isinstance(funcion, str) else funcion(serie)
        filas.append(fila)

    return pd.DataFrame(filas)


def main():
    close = descargar_datos()
    df = crear_dataset(close)
    resumen = crear_resumen(df)

    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")

    print("\nPrimeras filas del dataset:")
    print(df.head(10).to_string(index=False))

    print("\nResumen estadistico:")
    print(resumen.to_string(index=False))

    print(f"\nArchivo creado:\n- {CSV_FILE}")


if __name__ == "__main__":
