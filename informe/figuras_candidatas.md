# Figuras candidatas para el informe

Inventario inicial de graficos ya exportados. No implica que todos deban insertarse en el informe final; la seleccion deberia priorizar figuras compactas, legibles y directamente vinculadas con metricas finales, errores por rango de RUL, mantenimiento e interpretacion fisico-operativa.

| Ruta | Que muestra | Posible uso en informe | Prioridad | Observacion |
|---|---|---|---|---|
| `conclusion/figures/physical_feature_importance_top_sensors.png` | Importancia fisico-operativa por sensores principales. | Interpretacion post-hoc de variables relevantes. | Alta | Buena candidata para discutir sensores base. |
| `conclusion/figures/physical_perturbation_sensitivity_top_sensors.png` | Sensibilidad por perturbacion de sensores principales. | Robustez y efecto operativo de cambios en senales. | Alta | Complementa la importancia por sensor. |
| `conclusion/figures/physical_importance_by_engine_area.png` | Importancia agrupada por area del motor. | Discusion fisico-operativa resumida. | Alta | Figura compacta para cierre de resultados. |
| `figures/fd001_decision/official_test_predicted_vs_true.png` | Predicho vs real en test oficial de FD001. | Resultados finales por subset. | Alta | Verificar legibilidad antes de insertar. |
| `figures/fd001_decision/official_test_error_vs_true_rul.png` | Error contra RUL real en test oficial de FD001. | Analisis de sesgo por zona de RUL. | Alta | Util para discutir riesgo operacional. |
| `figures/fd001_decision/validation_mae_by_rul_bin.png` | MAE por rango de RUL en validacion FD001. | Desempeno por tramos de vida remanente. | Alta | Vincular con `final_rul_bin_metrics.csv`. |
| `figures/fd001_validation_final/candidate_error_distribution.png` | Distribucion de errores de candidato FD001. | Discusion de dispersion y errores extremos. | Media | Usar solo si aporta sobre las figuras oficiales. |
| `figures/fd001_validation_final/candidate_dangerous_vs_conservative.png` | Balance de errores peligrosos y conservadores. | Evaluacion operacional de riesgo. | Media | Puede servir si hay espacio. |
| `figures/FD002/validation_best_model/predicted_vs_true.png` | Predicho vs real del mejor modelo FD002. | Resultados finales por subset. | Alta | Subset con multiples condiciones. |
| `figures/FD002/validation_best_model/error_vs_true_rul.png` | Error contra RUL real en FD002. | Analisis de errores por zona de RUL. | Alta | Priorizar si se discute condicion operativa. |
| `figures/FD002/validation_best_model/mae_by_rul_bin.png` | MAE por rango de RUL en FD002. | Comparacion de desempeno por bins. | Alta | Vincular con metricas por rango. |
| `figures/FD002/validation_best_model/worst_cases_abs_error.png` | Peores casos por error absoluto en FD002. | Analisis de fallos del modelo. | Media | Usar si se discuten casos limite. |
| `figures/FD003/fd003_internal_validation_pred_vs_true.png` | Predicho vs real en validacion FD003. | Resultados finales por subset. | Alta | Subset con multiples modos de falla. |
| `figures/FD003/fd003_internal_validation_error_vs_true_rul.png` | Error contra RUL real en FD003. | Analisis de errores por zona de RUL. | Alta | Figura util para discusion de degradacion. |
| `figures/FD003/fd003_internal_validation_metrics_by_rul_bin.png` | Metricas por rango de RUL en FD003. | Desempeno por tramos de RUL. | Alta | Buena candidata para bins de RUL. |
| `figures/FD003/fd003_internal_validation_maintenance_ranking.png` | Ranking de mantenimiento en FD003. | Priorizacion operacional. | Alta | Puede apoyar la seccion de mantenimiento. |
| `figures/FD003/fd003_internal_validation_cmapss_penalty_distribution.png` | Distribucion de penalizacion C-MAPSS en FD003. | Explicar penalizacion asimetrica. | Media | Usar si queda espacio para score C-MAPSS. |
| `figures/FD003/fd003_fault_sensitive_pred_vs_true_comparison.png` | Comparacion predicho vs real para variantes fault-sensitive. | Discusion de features sensibles a falla. | Media | No usar como resultado final si confunde con seleccion cerrada. |
| `figures/FD003/fd003_fault_sensitive_cmapss_by_rul_bin.png` | C-MAPSS por rango de RUL en variantes FD003. | Analisis complementario por bins. | Media | Revisar coherencia con metricas finales. |
| `figures/FD003/fd003_fault_sensitive_dangerous_by_rul_bin.png` | Error peligroso por rango de RUL en FD003. | Riesgo operacional por zona de RUL. | Media | Candidata si se enfatiza seguridad. |
| `figures/FD004/validation_best_model/predicted_vs_true.png` | Predicho vs real del mejor modelo FD004. | Resultados finales por subset. | Alta | Subset mas complejo por condicion y falla. |
| `figures/FD004/validation_best_model/error_vs_true_rul.png` | Error contra RUL real en FD004. | Analisis de errores por zona de RUL. | Alta | Prioritaria para discutir dificultad. |
| `figures/FD004/validation_best_model/mae_by_rul_bin.png` | MAE por rango de RUL en FD004. | Comparacion de desempeno por bins. | Alta | Vincular con `final_rul_bin_metrics.csv`. |
| `figures/FD004/validation_best_model/worst_cases_abs_error.png` | Peores casos por error absoluto en FD004. | Analisis de casos extremos. | Media | Usar solo si aporta una conclusion concreta. |
| `outputs/figures/cmapss_subset_sizes.png` | Cantidad de motores por subset. | Contexto del conjunto de datos. | Media | Puede acompañar la seccion II si hay espacio. |
| `outputs/figures/cmapss_cycles_by_subset.png` | Distribucion de ciclos por subset. | Contexto de longitudes de trayectoria. | Media | Util para explicar heterogeneidad. |
| `outputs/figures/cmapss_rul_by_subset.png` | Distribucion de RUL por subset. | Contexto del objetivo. | Media | Verificar que no duplique texto de seccion II. |
| `outputs/figures/cmapss_rul_cap_effect.png` | Efecto del RUL cap. | Metodologia o apendice. | Media | Puede justificar cap de 125 ciclos. |
| `outputs/figures/cmapss_sensor_rul_corr_heatmap.png` | Correlaciones sensor-RUL. | Exploracion de senales relevantes. | Media | Usar con cautela por mezcla de condiciones. |
| `outputs/figures/fd002_condition_distribution.png` | Distribucion de condiciones en FD002. | Explicar complejidad de multiples condiciones. | Media | Contexto para FD002. |
| `outputs/figures/fd004_condition_distribution.png` | Distribucion de condiciones en FD004. | Explicar complejidad de multiples condiciones. | Media | Contexto para FD004. |
| `outputs/figures/fd002_sensor_rul_corr_by_condition_heatmap.png` | Correlacion sensor-RUL por condicion en FD002. | Justificar normalizacion por condicion. | Media | Candidata tecnica, revisar legibilidad. |
| `outputs/figures/fd004_sensor_rul_corr_by_condition_heatmap.png` | Correlacion sensor-RUL por condicion en FD004. | Justificar features condition-sensitive. | Media | Candidata tecnica, revisar legibilidad. |
