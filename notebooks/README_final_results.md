# Indice de notebooks para entrega final

Los notebooks se conservan como evidencia del proceso experimental. Sus outputs no deben limpiarse ni regenerarse automaticamente: varios experimentos son costosos y los resultados visibles forman parte de la trazabilidad del proyecto.

La entrega reproducible se concentra en los scripts finales (`predict_final.py`, `conclusion/build_conclusion_artifacts.py`, `conclusion/build_physical_operational_artifacts.py`) y en los artefactos de `conclusion/`.

## Notebooks de conclusion

- `notebooks/conclusion/01_conclusion_final.ipynb`: cierre principal, lectura de metricas finales y comparacion por subset.
- `notebooks/conclusion/02_interpretacion_fisica_operativa.ipynb`: reporte compacto del analisis fisico-operativo generado desde CSV finales.

## Archivo historico de conclusion

Estos notebooks fueron movidos a `notebooks/conclusion/archive/` para no confundir versiones historicas con el cierre vigente. No son fuente vigente de metricas finales y pueden contener resultados previos; se preservan solo por trazabilidad.

- `notebooks/conclusion/archive/notebook_conclusion_final_v03.ipynb`: version historica del cierre final v03.
- `notebooks/conclusion/archive/notebook_conclusion_final_actualizado_backup_pre_v03.ipynb`: respaldo historico anterior al cierre v03.
- `notebooks/conclusion/archive/01_conclusion_final_backup_before_calibration_v01.ipynb`: respaldo historico previo a calibraciones finales.

## FD001

- `notebooks/FD001/modeling/13_fd001_final_validation_selection.ipynb`: seleccion final por validacion artificial.
- `notebooks/FD001/modeling/14_fd001_official_test_final_once.ipynb`: evaluacion oficial una vez cerrado el modelo.
- `notebooks/FD001/modeling/15_fd001_validation_report_assets.ipynb`: recursos de reporte y diagnostico de validacion.
- `notebooks/FD001/modeling/18_fd001_lgbm_final_search_and_robustness.ipynb`: busqueda final LightGBM y robustez que llevo al modelo quantile final.

## FD002

- `notebooks/FD002/modeling/01_fd002_model_selection_and_hyperparam_search.ipynb`: seleccion de familias y busqueda inicial para FD002.
- `notebooks/FD002/modeling/02_fd002_feature_engineering_improvement.ipynb`: features condition-aware y fault-sensitive.
- `notebooks/FD002/modeling/03_fd002_robustness_probe.ipynb`: prueba de robustez del candidato final.
- `notebooks/FD002/28_fd002_multiprefix_candidate_lock_v02.ipynb`: cierre de candidato multiprefix y diagnosticos asociados.

## FD003

- `notebooks/FD003/modeling/19_fd003_transfer_fd001_pipeline_comparison.ipynb`: transferencia desde FD001.
- `notebooks/FD003/error_analysis/20_fd003_cluster_error_analysis.ipynb`: analisis de clusters y patrones latentes.
- `notebooks/FD003/modeling/21_fd003_short_tuning.ipynb`: ajuste acotado de modelo.
- `notebooks/FD003/modeling/22_fd003_fault_sensitive_features.ipynb`: features sensibles a degradacion que sostienen el modelo final.
- `notebooks/FD003/modeling/23_fd003_pseudo_cluster_experiments.ipynb`: pseudo-clusters descartados como mejora final.
- `notebooks/FD003/modeling/25_fd003_internal_validation_evaluation_report.ipynb`: evaluacion interna final.

## FD004

- `notebooks/EDA/FD004/05_eda_fd004_conditions_fault_patterns.ipynb`: conditions y patrones residuales.
- `notebooks/FD004/modeling/01_fd004_modeling_final.ipynb`: modelado final FD004.
- `notebooks/FD004/26_fd004_high_rul_condition_cluster_calibration_v01.ipynb`: calibracion y analisis por condition/cluster.
- `notebooks/FD004/28_fd004_multiprefix_late_stage_guard_v02.ipynb`: diagnosticos multiprefix y guardas tardias.

## EDA de soporte

- `notebooks/EDA/FD001/01_eda_fd001.ipynb`: estructura del caso base.
- `notebooks/EDA/general/02_eda_subsets_comparison.ipynb`: comparacion general FD001-FD004.
- `notebooks/EDA/FD002/04_eda_fd002_conditions.ipynb`: conditions en FD002.
- `notebooks/EDA/FD003/03_eda_fd003_patrones_degradacion.ipynb`: patrones de degradacion FD003.
