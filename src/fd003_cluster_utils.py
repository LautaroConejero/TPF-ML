from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.fd003_transfer_utils import (
    evaluate_fd003_transfer_split,
    load_fd003_train,
    prepare_fd003_temporal_validation_only,
)


DEFAULT_CLUSTER_SENSORS = ["sensor_7", "sensor_9", "sensor_12", "sensor_14", "sensor_15"]


def _linear_slope(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, deg=1)[0])


def build_fd003_unit_cluster_features(train_df, sensors=None, edge_cycles=10):
    sensors = [sensor for sensor in (sensors or DEFAULT_CLUSTER_SENSORS) if sensor in train_df.columns]
    rows = []
    for unit, group in train_df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        first = group.head(edge_cycles)
        last = group.tail(edge_cycles)
        row = {
            "unit_number": int(unit),
            "total_life": int(group["cycle"].max()),
            "n_cycles": int(len(group)),
            "cluster_sensors": ",".join(sensors),
        }
        for sensor in sensors:
            values = group[sensor].to_numpy(dtype=float)
            row[f"{sensor}_initial_mean"] = float(first[sensor].mean())
            row[f"{sensor}_final_mean"] = float(last[sensor].mean())
            row[f"{sensor}_delta_final_initial"] = row[f"{sensor}_final_mean"] - row[f"{sensor}_initial_mean"]
            row[f"{sensor}_mean"] = float(group[sensor].mean())
            row[f"{sensor}_std"] = float(group[sensor].std(ddof=0))
            row[f"{sensor}_slope"] = _linear_slope(values)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_fd003_unit_clusters(train_df, n_clusters=2, random_state=42):
    features = build_fd003_unit_cluster_features(train_df)
    metadata_cols = ["unit_number", "total_life", "n_cycles", "cluster_sensors"]
    feature_cols = [column for column in features.columns if column not in metadata_cols]
    scaler = StandardScaler()
    X = scaler.fit_transform(features[feature_cols])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    result = features.copy()
    result["cluster_id"] = kmeans.fit_predict(X)
    result["cluster_method"] = "kmeans"
    result["n_clusters"] = n_clusters
    result["cluster_random_state"] = random_state
    result["cluster_feature_columns"] = ",".join(feature_cols)
    return result, feature_cols


def find_existing_fd003_cluster_file(project_root):
    candidates = [
        Path(project_root) / "results" / "FD003" / "fd003_unit_clusters.csv",
        Path(project_root) / "results" / "fd003_unit_clusters.csv",
    ]
    candidates.extend(Path(project_root).glob("results/**/*fd003*cluster*.csv"))
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def cluster_summary(clusters):
    return (
        clusters.groupby("cluster_id", as_index=False)
        .agg(
            n_units=("unit_number", "nunique"),
            mean_total_life=("total_life", "mean"),
            median_total_life=("total_life", "median"),
            min_total_life=("total_life", "min"),
            max_total_life=("total_life", "max"),
        )
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )


def cmapss_penalty(error):
    error = np.asarray(error, dtype=float)
    return np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)


def normalize_fd003_predictions_for_cluster_analysis(predictions):
    result = predictions.copy()
    result["unit_number"] = result["unit"].astype(int)
    result["cutoff_cycle"] = result["cycle"].astype(int)
    result["true_rul"] = result["y_true_rul_raw"].astype(float)
    result["pred_rul"] = result["y_pred_rul"].astype(float)
    result["squared_error"] = result["error"] ** 2
    result["cmapss_penalty"] = cmapss_penalty(result["error"])
    columns = [
        "random_state",
        "unit_number",
        "cutoff_cycle",
        "true_rul",
        "pred_rul",
        "error",
        "abs_error",
        "squared_error",
        "dangerous_error",
        "conservative_error",
        "cmapss_penalty",
    ]
    return result[columns].copy()


def rerun_fd003_transfer_predictions(config, data_dir, dropped_columns, random_states, results_dir):
    rows = []
    prediction_tables = []
    for state in random_states:
        prepared = prepare_fd003_temporal_validation_only(
            data_dir=data_dir,
            eval_size=0.2,
            random_state=state,
            max_rul=config["rul_cap"],
            cut_ruls=(20, 50, 80, 110, 140),
            window_size=config["window_size"],
            drop_columns=dropped_columns,
        )
        row, predictions = evaluate_fd003_transfer_split(prepared, config, random_state=state)
        rows.append(row)
        prediction_tables.append(normalize_fd003_predictions_for_cluster_analysis(predictions))
    detail = pd.DataFrame(rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    return detail, predictions


def add_rul_bins(predictions):
    result = predictions.copy()
    result["rul_bin"] = pd.cut(
        result["true_rul"],
        bins=[0, 31, 61, 101, np.inf],
        labels=["0-30", "31-60", "61-100", "101+"],
        right=False,
        include_lowest=True,
    )
    return result


def grouped_error_metrics(predictions, group_cols):
    rows = []
    for keys, group in predictions.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        y_true = group["true_rul"].to_numpy(dtype=float)
        y_pred = group["pred_rul"].to_numpy(dtype=float)
        row.update(
            {
                "n_predictions": int(len(group)),
                "n_units": int(group["unit_number"].nunique()),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
                "cmapss_score": float(group["cmapss_penalty"].sum()),
                "cmapss_score_mean": float(group["cmapss_penalty"].mean()),
                "dangerous_error_pct": float(group["dangerous_error"].mean() * 100.0),
                "conservative_error_pct": float(group["conservative_error"].mean() * 100.0),
                "bias_mean": float(group["error"].mean()),
                "median_error": float(group["error"].median()),
                "abs_error_p90": float(group["abs_error"].quantile(0.90)),
                "abs_error_p95": float(group["abs_error"].quantile(0.95)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_cluster_error_interpretation(cluster_summary_df, metrics_by_cluster, metrics_by_bin):
    worse_rmse = metrics_by_cluster.sort_values("rmse", ascending=False).iloc[0]
    worse_cmapss = metrics_by_cluster.sort_values("cmapss_score", ascending=False).iloc[0]
    worse_dangerous = metrics_by_cluster.sort_values("dangerous_error_pct", ascending=False).iloc[0]
    near_failure = metrics_by_bin.loc[metrics_by_bin["rul_bin"].astype(str) == "0-30"].copy()
    if near_failure.empty:
        near_failure_text = "No hubo filas suficientes en el bin 0-30 para comparar errores cerca de la falla."
    else:
        nf = near_failure.sort_values("rmse", ascending=False).iloc[0]
        near_failure_text = (
            f"Cerca de la falla (RUL 0-30), el cluster {int(nf['cluster_id'])} muestra el mayor RMSE "
            f"({nf['rmse']:.3f}) dentro de los bins disponibles."
        )

    rmse_gap = metrics_by_cluster["rmse"].max() - metrics_by_cluster["rmse"].min()
    cmapss_gap = metrics_by_cluster["cmapss_score"].max() - metrics_by_cluster["cmapss_score"].min()
    dangerous_gap = metrics_by_cluster["dangerous_error_pct"].max() - metrics_by_cluster["dangerous_error_pct"].min()
    if rmse_gap > 2 or cmapss_gap > 100 or dangerous_gap > 3:
        conclusion = (
            "Los resultados sugieren que parte del deterioro en FD003 puede estar asociado a patrones "
            "latentes de degradacion con distinto comportamiento predictivo."
        )
    else:
        conclusion = (
            "Los clusters no explican de forma fuerte los errores del modelo; el deterioro frente a FD001 parece mas distribuido."
        )

    lines = [
        "FD003 - Analisis de errores por clusters latentes",
        "",
        "Advertencia metodologica:",
        "Los clusters no son etiquetas reales de fault mode. Se interpretan como patrones latentes de degradacion y se usan solo para analisis de error.",
        "No se usaron como feature del modelo, no se uso test oficial y no se selecciono modelo con esta informacion.",
        "",
        "Resumen de clusters:",
    ]
    for _, row in cluster_summary_df.iterrows():
        lines.append(
            f"- Cluster {int(row['cluster_id'])}: {int(row['n_units'])} motores, vida media {row['mean_total_life']:.2f}, "
            f"mediana {row['median_total_life']:.2f}, min {int(row['min_total_life'])}, max {int(row['max_total_life'])}."
        )
    lines.extend(
        [
            "",
            "Metricas por cluster:",
        ]
    )
    for _, row in metrics_by_cluster.iterrows():
        lines.append(
            f"- Cluster {int(row['cluster_id'])}: MAE {row['mae']:.3f}, RMSE {row['rmse']:.3f}, "
            f"C-MAPSS {row['cmapss_score']:.3f}, dangerous {row['dangerous_error_pct']:.2f}%, "
            f"conservative {row['conservative_error_pct']:.2f}%, bias {row['bias_mean']:.3f}."
        )
    lines.extend(
        [
            "",
            f"Mayor RMSE: cluster {int(worse_rmse['cluster_id'])} ({worse_rmse['rmse']:.3f}).",
            f"Mayor C-MAPSS total: cluster {int(worse_cmapss['cluster_id'])} ({worse_cmapss['cmapss_score']:.3f}).",
            f"Mayor dangerous error %: cluster {int(worse_dangerous['cluster_id'])} ({worse_dangerous['dangerous_error_pct']:.2f}%).",
            near_failure_text,
            "",
            conclusion,
        ]
    )
    return "\n".join(lines) + "\n"
