# Proyecto final - Probabilidad y Estadistica

## Tema

Analisis estadistico del tipo de cambio entre el peso uruguayo y el real brasileno.

El proyecto estudia la cotizacion `UYU/BRL`, entendida como la cantidad de pesos uruguayos necesarios para comprar un real brasileno. La variable se construye de forma indirecta a partir de las cotizaciones `USDUYU=X` y `USDBRL=X`, usando el dolar estadounidense como moneda de referencia.

## Objetivo

Analizar estadisticamente el comportamiento del tipo de cambio UYU/BRL mediante estadistica descriptiva, analisis exploratorio, ajuste de distribuciones, intervalos de confianza y pruebas de hipotesis adecuadas para una serie temporal financiera.

## Fuente de datos

- Fuente: Yahoo Finance, descargada mediante Python y `yfinance`.
- Archivo esperado: `data/raw/dataset_uyu_brl_2005_2026.csv`.
- Periodo observado en el archivo actual: se valida automaticamente con los scripts.

## Integrantes

- Luis Farias
- Cristian Fernandez
- Alejandro Pereira

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecutar el analisis

Desde la raiz del proyecto:

```powershell
python src/run_all.py
```

Esto genera tablas en `outputs/tables/` y figuras en `outputs/figures/`.

## Usar Google Colab

El notebook principal para trabajar en Colab es:

```text
notebooks/analisis_uyu_brl_colab.ipynb
```

Ese notebook carga el CSV desde GitHub, valida la variable principal `uyu_por_brl`, calcula estadisticas y genera graficos. Es la version mas amigable para mostrar el codigo y el proceso de analisis.

## Compilar el informe LaTeX

Desde la carpeta `report/`:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

El PDF resultante queda como `report/main.pdf`.

Si no se quiere instalar LaTeX localmente, usar Overleaf. Ver:

```text
docs/guia_latex_google_docs.md
```

## Guardar una version del PDF

Luego de compilar:

```powershell
Copy-Item report\main.pdf versions\informe_v0.20.pdf
```

## Proximo commit sugerido

```text
docs: redacta primera version del informe academico
```
