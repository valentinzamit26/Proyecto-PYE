# Guia rapida: LaTeX, Overleaf y Google Docs

## Opcion recomendada: Overleaf

Overleaf permite compilar LaTeX desde el navegador, sin instalar nada.

Pasos:

1. Entrar a <https://www.overleaf.com/>.
2. Crear un proyecto nuevo.
3. Subir la carpeta `report/`.
4. Subir tambien `outputs/figures/` y `outputs/tables/`, respetando la misma estructura del repo.
5. Abrir `report/main.tex`.
6. Compilar con `pdfLaTeX`.
7. Descargar el PDF y guardarlo como `versions/informe_v0.20.pdf`.

Ventaja: es lo mas simple para entregar un PDF formal.

## Opcion local: instalar LaTeX

En Windows se puede instalar MiKTeX:

<https://miktex.org/download>

Luego, desde PowerShell:

```powershell
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
Copy-Item main.pdf ..\versions\informe_v0.20.pdf
```

Si aparece una ventana pidiendo instalar paquetes faltantes, aceptar.

## Alternativa: Google Docs

Google Docs es mas facil para redactar colaborativamente, pero no maneja tan bien:

- referencias cruzadas automaticas;
- tablas generadas desde CSV;
- ecuaciones largas;
- control exacto de figuras;
- bibliografia reproducible.

Uso recomendado:

1. Mantener LaTeX como version final para entregar.
2. Usar Google Docs para discutir redaccion entre integrantes.
3. Copiar cambios aprobados desde Google Docs hacia los archivos `.tex`.
4. Compilar el PDF final desde Overleaf.

Si el equipo prefiere trabajar solo en Google Docs, se puede crear una version `.docx` o Google Docs del informe, pero habria que revisar manualmente figuras, tablas, formulas y paginado.

## Como pensar LaTeX sin sufrir

- `report/main.tex` es el archivo principal.
- `report/sections/` contiene el texto por secciones.
- Las figuras viven en `outputs/figures/`.
- Las tablas viven en `outputs/tables/`.
- Para insertar una figura se usa `\includegraphics`.
- Para citar una figura o tabla se usa `\ref{...}`.

La regla practica: no editar todo en `main.tex`; editar solo la seccion correspondiente.

