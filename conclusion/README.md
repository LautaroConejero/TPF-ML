# Conclusion - C-MAPSS RUL

Esta carpeta consolida las implementaciones finales del proyecto: modelos seleccionados, metricas finales, ranking de prioridad de mantenimiento y reglas simples de decision operativa.

Criterio metodologico:
- Los modelos se seleccionaron con validacion artificial por motores completos.
- El test oficial se usa solo para reporte final, no para buscar hiperparametros.
- Las decisiones operativas se basan en RUL predicho y se complementan con flags de error peligroso/conservador cuando hay etiqueta real disponible.

Modelos finales:
- FD001: candidate_03_B_quantile_a040_search_14 (LightGBM), feature_set=temporal, window=50, cap=125.
- FD002: xgb_condition_fault_sensitive_mid_guard (xgboost), feature_set=condition_fault_sensitive, window=50, cap=125.
- FD003: fd003_lgbm_w50_cap125_quantile_a04_none_fault_sensitive (LightGBM), feature_set=fault_sensitive, window=50, cap=125.
- FD004: fd004_xgb_fs_bin_weights_w70 (xgboost), feature_set=condition_fault_sensitive, window=70, cap=125.

Metricas oficiales finales:
- FD001: RMSE 13.949, C-MAPSS 283.898, dangerous 8.00%.
- FD002: RMSE 25.280, C-MAPSS 5349.925, dangerous 5.02%.
- FD003: RMSE 14.555, C-MAPSS 394.763, dangerous 9.00%.
- FD004: RMSE 26.071, C-MAPSS 4600.217, dangerous 6.45%.

Resumen de decision:
- FD001: 22 urgentes, 16 programar pronto, 13 monitoreo cercano.
- FD002: 61 urgentes, 37 programar pronto, 47 monitoreo cercano.
- FD003: 21 urgentes, 17 programar pronto, 16 monitoreo cercano.
- FD004: 55 urgentes, 30 programar pronto, 39 monitoreo cercano.

Archivos:
- final_model_summary.csv
- final_metric_summary.csv
- final_rul_bin_metrics.csv
- maintenance_priority_ranking.csv
- maintenance_decision_summary.csv
- final_conclusion_payload.json
