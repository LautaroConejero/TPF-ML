# TPF-ML

Trabajo final de Machine Learning sobre NASA C-MAPSS para prediccion de Remaining Useful Life (RUL).

## Cierre final

El proyecto queda organizado por subset:

- FD001: LightGBM quantile, `temporal_w50`, `rul_cap=125`.
- FD002: XGBoost `condition_fault_sensitive`, `window_size=50`.
- FD003: LightGBM quantile `fault_sensitive`, `window_size=50`.
- FD004: XGBoost `condition_fault_sensitive`, `window_size=70`.

Los artefactos finales estan en `conclusion/` y las explicaciones completas estan en `notas/`.

Lecturas recomendadas:

- `notas/resumen_final_tp.txt`
- `notas/indice_notas_final.txt`
- `conclusion/README.md`

Para regenerar el cierre:

```powershell
python conclusion\build_conclusion_artifacts.py
```
