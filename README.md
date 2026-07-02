# TPF-ML

Proyecto final de Machine Learning sobre NASA C-MAPSS para prediccion de Remaining Useful Life (RUL) en motores turbofan.

El cierre final usa modelos separados por subset, porque FD001, FD002, FD003 y FD004 tienen distinta combinacion de condiciones operativas y modos de falla. Ademas de las predicciones de RUL, el proyecto incluye una extension operativa para priorizacion de mantenimiento y una extension fisico-operativa para interpretar sensores, areas del motor y sensibilidad por perturbacion.

## Modelos Finales

- FD001: LightGBM quantile, features temporales, ventana 50, RUL cap 125.
- FD002: XGBoost condition_fault_sensitive, ventana 50, RUL cap 125.
- FD003: LightGBM fault_sensitive, ventana 50, RUL cap 125.
- FD004: XGBoost condition_fault_sensitive, ventana 70, RUL cap 125.

La seleccion de modelos, hiperparametros, predicciones y metricas finales no deben modificarse para la entrega.

## Datos Esperados

El repo espera los archivos C-MAPSS en `CMAPSSData/`:

- `CMAPSSData/train_FD001.txt`, `CMAPSSData/test_FD001.txt`, `CMAPSSData/RUL_FD001.txt`
- `CMAPSSData/train_FD002.txt`, `CMAPSSData/test_FD002.txt`, `CMAPSSData/RUL_FD002.txt`
- `CMAPSSData/train_FD003.txt`, `CMAPSSData/test_FD003.txt`, `CMAPSSData/RUL_FD003.txt`
- `CMAPSSData/train_FD004.txt`, `CMAPSSData/test_FD004.txt`, `CMAPSSData/RUL_FD004.txt`

Los archivos `RUL_FD00X.txt` se usan para diagnostico y evaluacion final. El ejecutable de prediccion puede generar predicciones sin RUL real si no se solicita `--include-diagnostics`.

## Reproducir Predicciones Finales

```powershell
python predict_final.py --subset all --output-dir predictions/final_executable --include-diagnostics
```

Salidas principales:

- `predictions/final_executable/all_final_predictions.csv`
- `predictions/final_executable/all_final_metrics.csv`
- `predictions/final_executable/fd001_final_predictions.csv`
- `predictions/final_executable/fd002_final_predictions.csv`
- `predictions/final_executable/fd003_final_predictions.csv`
- `predictions/final_executable/fd004_final_predictions.csv`

## Regenerar Artefactos De Conclusion

```powershell
python conclusion/build_conclusion_artifacts.py
```

Genera resumen de modelos, metricas finales, metricas por rango de RUL, ranking de prioridad de mantenimiento y payload de conclusion.

## Regenerar Analisis Fisico-Operativo

```powershell
python conclusion/build_physical_operational_artifacts.py
python conclusion/build_physical_area_summary.py
```

El primer script reconstruye la interpretacion por sensor fisico, importancia agrupada y sensibilidad por permutacion. El segundo agrega la lectura por area del motor para informe/poster.

## Archivos Finales Importantes

- `predictions/final_executable/all_final_predictions.csv`
- `conclusion/final_model_summary.csv`
- `conclusion/final_metric_summary.csv`
- `conclusion/final_rul_bin_metrics.csv`
- `conclusion/maintenance_priority_ranking.csv`
- `conclusion/maintenance_decision_summary.csv`
- `conclusion/final_conclusion_payload.json`
- `conclusion/physical_sensor_dictionary.csv`
- `conclusion/physical_feature_importance_overall.csv`
- `conclusion/physical_perturbation_sensitivity.csv`
- `conclusion/physical_importance_by_engine_area.csv`
- `conclusion/physical_sensitivity_by_engine_area.csv`
- `conclusion/physical_operational_payload.json`
- `conclusion/final_deliverable_manifest.csv`
- `conclusion/final_deliverable_manifest.json`

## Notebooks

Los notebooks son evidencia del desarrollo experimental: EDA, validacion, busqueda de modelos, calibraciones y cierre. Sus outputs se conservan para evitar recalcular experimentos pesados. No se deben limpiar automaticamente ni reescribir con herramientas como `nbstripout` o `nbconvert --clear-output`.

La entrega ejecutable final esta concentrada en `predict_final.py`, `conclusion/build_conclusion_artifacts.py`, `conclusion/build_physical_operational_artifacts.py`, `conclusion/build_physical_area_summary.py` y los artefactos de `conclusion/`.

Indice de notebooks relevantes para entrega: `notebooks/README_final_results.md`.

## Smoke Test

```powershell
python scripts/smoke_test_final.py
```

Este test valida existencia de scripts/configs/artefactos principales, lectura de CSV finales, imports basicos de `src/` y compilacion de scripts finales. No ejecuta notebooks ni busquedas de hiperparametros.
