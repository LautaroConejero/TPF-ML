Predicciones finales - C-MAPSS RUL

Archivos principales de predicción:
- GarciaConejero_Needleman_FD001_predictions.csv
- GarciaConejero_Needleman_FD002_predictions.csv
- GarciaConejero_Needleman_FD003_predictions.csv
- GarciaConejero_Needleman_FD004_predictions.csv

Cada archivo contiene una predicción por motor para el último ciclo observado del subconjunto correspondiente.

Columnas:
- unit: identificador del motor.
- cycle: último ciclo observado.
- predicted_RUL: vida útil remanente estimada en ciclos.
- dataset: subconjunto C-MAPSS.
- model_name: modelo final usado.
- representation: representación usada por el modelo.
- window_size_used: cantidad de ciclos usada en la ventana temporal final.

La carpeta final_executable/ contiene salidas generadas por el ejecutable final, incluyendo métricas y archivos con diagnósticos cuando está disponible el RUL real. Esos archivos se incluyen para trazabilidad y reproducción del análisis, no como formato principal de predicción.

El dataset usado para reproducir y evaluar está en:
../informe_codigo/CMAPSSData/

El informe y las tablas de resultados están en:
../informe_codigo/informe_final.pdf
../informe_codigo/conclusion/