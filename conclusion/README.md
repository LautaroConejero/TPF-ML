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

Lectura por rangos de RUL:
- Los modelos son mas precisos cerca de falla (0-30 ciclos), donde el costo operativo de sobreestimar RUL es mayor.
- La zona 60-90 concentra varios dangerous errors y conviene discutirla explicitamente en el informe.
- En 90+ los errores suben por el RUL cap y por la menor prioridad operativa de distinguir vidas remanentes largas.

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
- final_deliverable_manifest.csv
- final_deliverable_manifest.json

Lectura final:
- Los modelos finales quedan cerrados por subset y mantienen validacion por motores completos antes del reporte oficial.
- Las metricas oficiales se consolidan en `final_metric_summary.csv` y los rangos de RUL en `final_rul_bin_metrics.csv`.
- La priorizacion de mantenimiento queda materializada en `maintenance_priority_ranking.csv` y `maintenance_decision_summary.csv`.
- La interpretacion fisico-operativa agrega trazabilidad entre sensores, areas del motor, importancia agrupada y sensibilidad por permutacion.
- El manifiesto final registra scripts, configs, predicciones, notebooks de evidencia y materiales de reporte sin mover ni reescribir notebooks.

Interpretacion fisico-operativa:
- Se agrego una auditoria fisico-operativa que no cambia los modelos finales, sino que interpreta sus senales por sensor base.
- Top sensores globales por importancia agrupada: Ps30 (sensor_11), T50 (sensor_4), NRf (sensor_13).
- Sensores con mayor sensibilidad por permutacion: FD004:Ps30 (sensor_11), FD002:Ps30 (sensor_11), FD002:BPR (sensor_15).
- FD003 se conecta con patrones latentes en core speed y presion HPC; FD004 agrega la complejidad de condiciones operativas y senales fuel/HPC/bypass.
- Las permutaciones agrupan todas las columnas derivadas de cada sensor para evitar interpretar estadisticos temporales aislados.

Archivos fisico-operativos:
- physical_sensor_dictionary.csv
- physical_feature_importance_by_dataset.csv
- physical_feature_importance_overall.csv
- physical_perturbation_sensitivity.csv
- physical_pattern_links.csv
- physical_operational_payload.json
- physical_importance_by_engine_area.csv
- physical_sensitivity_by_engine_area.csv
- figures/physical_feature_importance_top_sensors.png
- figures/physical_perturbation_sensitivity_top_sensors.png
- figures/physical_importance_by_engine_area.png
