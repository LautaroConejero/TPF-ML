# Actualizacion del notebook final de conclusion v01

## Alcance

- Notebook oficial actualizado: `notebooks/conclusion/01_conclusion_final.ipynb`.
- Backup previo creado: `notebooks/conclusion/01_conclusion_final_backup_before_calibration_v01.ipynb`.
- No se modificaron notebooks historicos de modeling ni artefactos oficiales previos de `conclusion/`.

## Artefactos cargados por la nueva seccion

- Resumen global corregido: `results/final_calibration_summary_v02.csv`.
- Seleccion final corregida: `results/final_model_selection_after_calibration_v02.csv`.
- Metricas por rango RUL: `results/FD002/fd002_rul_bin_metrics_v01.csv`, `results/FD003/fd003_rul_bin_metrics_v01.csv`, `results/FD004/fd004_rul_bin_metrics_v01.csv`.
- Auditorias por condicion: `results/FD002/fd002_condition_error_analysis_v01.csv`, `results/FD004/fd004_condition_error_analysis_v01.csv`.
- Auditoria de clusters FD004: `results/FD004/fd004_cluster_error_analysis_v01.csv`.

## Decision post-calibracion

- FD001 queda congelado; no se hicieron experimentos nuevos.
- FD002 retiene `fd002_baseline_xgb_condition_fault_sensitive_mid_guard`.
- FD003 retiene `fd003_offset_plus_0`.
- FD004 cambia a `fd004_high_rul_thr120_off2`.

Los archivos `_v01` contienen las metricas detalladas generadas en la corrida inicial. Los archivos globales `_v02` corrigen solamente la marca de seleccion final y la razon de seleccion sin sobrescribir los `_v01`.
