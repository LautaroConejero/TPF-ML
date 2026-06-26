# TPF-ML

Trabajo final de Machine Learning sobre NASA C-MAPSS para prediccion de Remaining Useful Life (RUL).

## Cierre final

El proyecto queda organizado por subset:

- FD001: LightGBM quantile, `temporal_w50`, `rul_cap=125`.
- FD002: XGBoost `condition_fault_sensitive`, `window_size=50`.
- FD003: LightGBM quantile `fault_sensitive`, `window_size=50`.
- FD004: XGBoost `condition_fault_sensitive`, `window_size=70`.

Los artefactos finales estan en `conclusion/` y las explicaciones completas estan en `notas/`.

## Modelo final ejecutable

El script `predict_final.py` entrena el modelo final de cada subset con el archivo `train_FD00X.txt` y genera predicciones para el ultimo ciclo disponible de cada motor en `test_FD00X.txt`. No requiere usar `RUL_FD00X.txt` para predecir; esos archivos solo se usan si se pide el diagnostico opcional.

```powershell
python predict_final.py --subset all --output-dir predictions\final_executable
```

Para reproducir tambien las metricas oficiales cuando los archivos `RUL_FD00X.txt` estan disponibles:

```powershell
python predict_final.py --subset all --output-dir predictions\final_executable --include-diagnostics
```

Salidas principales:

- `predictions/final_executable/all_final_predictions.csv`
- `predictions/final_executable/fd001_final_predictions.csv`
- `predictions/final_executable/fd002_final_predictions.csv`
- `predictions/final_executable/fd003_final_predictions.csv`
- `predictions/final_executable/fd004_final_predictions.csv`

## Informe final

El informe esta en `informe/informe_final.tex`, con formato IEEE Conference. En esta maquina no hay compilador LaTeX instalado; con una distribucion LaTeX disponible se compila con:

```powershell
pdflatex -interaction=nonstopmode -output-directory informe informe\informe_final.tex
pdflatex -interaction=nonstopmode -output-directory informe informe\informe_final.tex
```

Antes de entregar, completar autores y email en el encabezado del `.tex`.

Lecturas recomendadas:

- `notas/resumen_final_tp.txt`
- `notas/guia_informe_final.txt`
- `notas/indice_notas_final.txt`
- `notebooks/conclusion/01_conclusion_final.ipynb`
- `conclusion/README.md`

Para regenerar el cierre:

```powershell
python conclusion\build_conclusion_artifacts.py
```
