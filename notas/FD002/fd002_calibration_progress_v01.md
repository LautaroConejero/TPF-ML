# Calibration progress v01

Created at: 2026-06-26T04:36:41

## Initial diagnosis
- Notebook folders read: `notebooks/FD001`, `notebooks/FD002`, `notebooks/FD003`, `notebooks/FD004`, `notebooks/conclusion`.
- Results folders read: `results/FD001`, `results/FD002`, `results/FD003`, `results/FD004`.
- Config folders read: `configs/FD001`, `configs/FD002`, `configs/FD003`, `configs/FD004`.
- Notes folders read: `notas`, including `notas/hallazgos` and existing dataset notes.
- Existing utilities used: `src.data`, `src.preprocessed_FD001`, `src.fd001_modeling`, `src.fd002_modeling`, `src.fd003_improvement_utils`, `src.fd004_modeling`.
- C-MAPSS implementation used: standard project-compatible asymmetric penalty, `exp(-d/13)-1` for conservative errors and `exp(d/10)-1` for late/dangerous errors.
- Test final was not used. All calibration metrics use existing internal validation/artificial-cutoff prediction files.

## Existing conclusion notebook inputs
- `notebooks/conclusion/01_conclusion_final.ipynb` currently reads `conclusion/final_model_summary.csv`, `conclusion/final_metric_summary.csv`, `conclusion/final_rul_bin_metrics.csv`, `conclusion/maintenance_decision_summary.csv`, and `conclusion/maintenance_priority_ranking.csv`.

## Current final candidates
- FD002: `xgb_condition_fault_sensitive_mid_guard`.
- FD003: `fd003_lgbm_w50_cap125_quantile_a04_none_fault_sensitive`.
- FD004: `fd004_xgb_fs_bin_weights_w70`.
## FD002 experiment decision
- Base model: `xgb_condition_fault_sensitive_mid_guard`.
- Secondary model for ensemble: `xgb_squarederror_condition_normalized_weighted`.
- Experiments run: baseline, simple ensembles, high-RUL piecewise calibration, ensemble plus calibration for near-best ensembles, and condition error analysis.
- High-cap retraining was not run because this pass uses existing internal validation predictions and avoids retraining expensive historical pipelines.
- Selected row: `fd002_baseline_xgb_condition_fault_sensitive_mid_guard`.
- Decision: nan
- Existing configs and historical notebooks were not modified.
