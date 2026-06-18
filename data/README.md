# Datos

Colocar en `data/raw/` el archivo:

```text
dataset_uyu_brl_2005_2026.csv
```

El archivo utilizado en este proyecto contiene cotizaciones historicas para construir el tipo de cambio UYU/BRL. La fuente declarada es Yahoo Finance, accedida mediante `yfinance`, usando las series:

- `USDUYU=X`: pesos uruguayos por dolar estadounidense.
- `USDBRL=X`: reales brasilenos por dolar estadounidense.

La variable principal se calcula como:

```text
uyu_por_brl = uyu_por_usd / brl_por_usd
```

No guardar credenciales, tokens ni claves de APIs en esta carpeta.

