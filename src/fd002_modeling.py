from __future__ import annotations

from itertools import product
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data import (
    FEATURE_COLUMNS,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    add_train_rul,
    last_cycle_rows,
    load_cmapss_subset,
)
from src.data_splitting import split_units
from src.fd001_experiment_utils import weights_from_scheme
from src.fd001_modeling import (
    metrics_by_model,
    metrics_by_rul_bin,
    plot_validation_diagnostics,
    prediction_frame,
)
from src.preprocessed_FD001 import make_fd001_artificial_cutoffs


DEFAULT_RUL_CAP = 125
DEFAULT_CUT_RULS = (20, 50, 80, 110, 140)
DEFAULT_WINDOW_SIZE = 50
CONDITION_COUNT = 6
DANGEROUS_ERROR_THRESHOLD = 20.0
FD002_FAULT_SENSITIVE_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
]


def resolve_data_dir(data_dir):
    data_dir = Path(data_dir)
    if data_dir.is_absolute() or data_dir.exists():
        return data_dir
    project_data_dir = PROJECT_ROOT / data_dir
    if project_data_dir.exists():
        return project_data_dir
    return data_dir


def fd002_output_paths(project_root=PROJECT_ROOT):
    project_root = Path(project_root)
    paths = {
        "results": project_root / "results" / "FD002",
        "configs": project_root / "configs" / "FD002",
        "figures": project_root / "figures" / "FD002",
        "predictions": project_root / "predictions",
        "notes": project_root / "notas" / "hallazgos" / "FD002",
        "checkpoints": project_root / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def fd002_train_with_rul(data_dir="CMAPSSData", rul_cap=DEFAULT_RUL_CAP):
    data = load_cmapss_subset("FD002", data_dir=resolve_data_dir(data_dir))
    train = add_train_rul(data.train, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    if rul_cap is not None:
        train["RUL"] = train["RUL"].clip(upper=rul_cap)
    return train


def fit_condition_preprocessor(train_df, random_state=42):
    settings_scaler = StandardScaler()
    settings_x = settings_scaler.fit_transform(train_df[SETTING_COLUMNS])
    condition_model = KMeans(
        n_clusters=CONDITION_COUNT,
        random_state=random_state,
        n_init=50,
    )
    raw_labels = condition_model.fit_predict(settings_x)

    centers = pd.DataFrame(
        settings_scaler.inverse_transform(condition_model.cluster_centers_),
        columns=SETTING_COLUMNS,
    )
    centers["raw_condition"] = range(len(centers))
    centers = centers.sort_values(SETTING_COLUMNS).reset_index(drop=True)
    centers["condition_id"] = range(1, CONDITION_COUNT + 1)
    raw_to_condition = dict(zip(centers["raw_condition"], centers["condition_id"]))

    train_cond = train_df.copy()
    train_cond["condition_id"] = pd.Series(raw_labels).map(raw_to_condition).to_numpy()
    condition_stats = {}
    for sensor in SENSOR_COLUMNS:
        stats = train_cond.groupby("condition_id")[sensor].agg(["mean", "std"])
        stats["std"] = stats["std"].replace(0.0, np.nan)
        condition_stats[sensor] = stats

    return {
        "settings_scaler": settings_scaler,
        "condition_model": condition_model,
        "condition_centers": centers,
        "raw_to_condition": raw_to_condition,
        "condition_stats": condition_stats,
    }


def add_fd002_condition_features(df, preprocessor):
    result = df.copy()
    settings_x = preprocessor["settings_scaler"].transform(result[SETTING_COLUMNS])
    raw_labels = preprocessor["condition_model"].predict(settings_x)
    result["condition_id"] = pd.Series(raw_labels).map(preprocessor["raw_to_condition"]).to_numpy()

    for condition_id in range(1, CONDITION_COUNT + 1):
        result[f"condition_{condition_id}"] = (result["condition_id"] == condition_id).astype(float)

    for sensor in SENSOR_COLUMNS:
        stats = preprocessor["condition_stats"][sensor]
        means = result["condition_id"].map(stats["mean"])
        stds = result["condition_id"].map(stats["std"])
        z_values = (result[sensor] - means) / stds
        result[f"{sensor}_cond_z"] = z_values.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return result


def base_columns_for_feature_set(feature_set):
    condition_cols = ["condition_id"] + [f"condition_{i}" for i in range(1, CONDITION_COUNT + 1)]
    sensor_z_cols = [f"{sensor}_cond_z" for sensor in SENSOR_COLUMNS]
    if feature_set == "raw":
        return FEATURE_COLUMNS.copy()
    if feature_set == "raw_plus_condition":
        return FEATURE_COLUMNS + condition_cols
    if feature_set in {"condition_normalized", "condition_fault_sensitive"}:
        return FEATURE_COLUMNS + condition_cols + sensor_z_cols
    raise ValueError(f"Unknown FD002 feature_set: {feature_set!r}")


def temporal_columns_for_base(base_columns, stats=None):
    if stats is None:
        stats = ("last", "mean", "std", "min", "max", "delta", "slope")
    return [f"{column}_{stat}" for column in base_columns for stat in stats]


def _window_sums(values, window_size):
    values = np.asarray(values, dtype=float)
    n = len(values)
    pos = np.arange(n)
    starts = np.maximum(pos - int(window_size) + 1, 0)
    csum = np.cumsum(values)
    csum_sq = np.cumsum(values * values)
    csum_pos = np.cumsum(pos * values)
    prev = starts - 1

    sum_y = csum[pos] - np.where(prev >= 0, csum[prev], 0.0)
    sum_y2 = csum_sq[pos] - np.where(prev >= 0, csum_sq[prev], 0.0)
    sum_pos_y = csum_pos[pos] - np.where(prev >= 0, csum_pos[prev], 0.0)
    count = pos - starts + 1
    return starts, count.astype(float), sum_y, sum_y2, sum_pos_y


def _rolling_slope(values, window_size):
    starts, count, sum_y, _, sum_pos_y = _window_sums(values, window_size)
    sum_x = count * (count - 1.0) / 2.0
    sum_x2 = (count - 1.0) * count * (2.0 * count - 1.0) / 6.0
    sum_xy = sum_pos_y - starts * sum_y
    denom = count * sum_x2 - sum_x * sum_x
    slope = np.divide(
        count * sum_xy - sum_x * sum_y,
        denom,
        out=np.zeros_like(sum_y, dtype=float),
        where=denom != 0,
    )
    return slope


def _rolling_delta(values, window_size):
    values = np.asarray(values, dtype=float)
    starts = np.maximum(np.arange(len(values)) - int(window_size) + 1, 0)
    return values - values[starts]


def _rolling_std(values, window_size):
    _, count, sum_y, sum_y2, _ = _window_sums(values, window_size)
    mean = sum_y / count
    variance = np.maximum(sum_y2 / count - mean * mean, 0.0)
    return np.sqrt(variance)


def _safe_divide_array(numerator, denominator, eps=1e-6):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=np.abs(denominator) > eps,
    )


def make_temporal_features_fast(source_df, endpoints_df=None, feature_columns=None, window_size=DEFAULT_WINDOW_SIZE):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    source = source_df.sort_values(["unit", "cycle"]).copy()
    endpoints = None if endpoints_df is None else endpoints_df.sort_values(["unit", "cycle"]).copy()
    temporal_columns = temporal_columns_for_base(feature_columns)
    frames = []

    for _, group in source.groupby("unit", sort=True):
        group = group.reset_index(drop=True)
        n = len(group)
        pos = np.arange(n)
        starts = np.maximum(pos - int(window_size) + 1, 0)

        feature_data = {"window_size_used": np.minimum(pos + 1, int(window_size))}
        for column in feature_columns:
            values = group[column].to_numpy(dtype=float)
            _, count, sum_y, sum_y2, _ = _window_sums(values, window_size)
            mean = sum_y / count
            variance = np.maximum(sum_y2 / count - mean * mean, 0.0)
            rolling = pd.Series(values).rolling(window=int(window_size), min_periods=1)
            first_values = values[starts]

            feature_data[f"{column}_last"] = values
            feature_data[f"{column}_mean"] = mean
            feature_data[f"{column}_std"] = np.sqrt(variance)
            feature_data[f"{column}_min"] = rolling.min().to_numpy()
            feature_data[f"{column}_max"] = rolling.max().to_numpy()
            feature_data[f"{column}_delta"] = values - first_values
            feature_data[f"{column}_slope"] = _rolling_slope(values, window_size)

        frames.append(pd.concat([group, pd.DataFrame(feature_data)], axis=1))

    all_features = pd.concat(frames, ignore_index=True)
    if endpoints is None:
        return all_features.reset_index(drop=True)

    feature_lookup = all_features[["unit", "cycle", "window_size_used"] + temporal_columns]
    return endpoints.merge(feature_lookup, on=["unit", "cycle"], how="left").reset_index(drop=True)


def make_fd002_fault_sensitive_features(
    source_df,
    endpoints_df=None,
    sensors=None,
    window_size=DEFAULT_WINDOW_SIZE,
    value_suffix="_cond_z",
):
    sensors = list(FD002_FAULT_SENSITIVE_SENSORS if sensors is None else sensors)
    source = source_df.sort_values(["unit", "cycle"]).copy()
    endpoints = None if endpoints_df is None else endpoints_df.sort_values(["unit", "cycle"]).copy()
    frames = []

    for _, group in source.groupby("unit", sort=True):
        group = group.reset_index(drop=True)
        feature_data = {
            "unit": group["unit"].to_numpy(),
            "cycle": group["cycle"].to_numpy(),
        }
        sensor_cache = {}

        for sensor in sensors:
            column = f"{sensor}{value_suffix}"
            values = group[column].to_numpy(dtype=float)
            slope_main = _rolling_slope(values, window_size)
            slope_30 = _rolling_slope(values, 30)
            slope_10 = _rolling_slope(values, 10)
            delta_30 = _rolling_delta(values, 30)
            delta_10 = _rolling_delta(values, 10)

            prefix = f"{sensor}_cond_fs"
            feature_data[f"{prefix}_slope_10"] = slope_10
            feature_data[f"{prefix}_slope_30"] = slope_30
            feature_data[f"{prefix}_slope_main"] = slope_main
            feature_data[f"{prefix}_delta_10"] = delta_10
            feature_data[f"{prefix}_delta_30"] = delta_30
            feature_data[f"{prefix}_volatility_10"] = _rolling_std(values, 10)
            feature_data[f"{prefix}_volatility_30"] = _rolling_std(values, 30)
            feature_data[f"{prefix}_slope_diff_10_main"] = slope_10 - slope_main
            feature_data[f"{prefix}_slope_ratio_10_main"] = _safe_divide_array(slope_10, slope_main)
            feature_data[f"{prefix}_acceleration_10_30"] = slope_10 - slope_30
            sensor_cache[sensor] = {
                "slope_10": slope_10,
                "delta_30": delta_30,
            }

        if {"sensor_9", "sensor_7"}.issubset(sensor_cache):
            feature_data["sensor_9_minus_sensor_7_cond_fs_slope_10"] = (
                sensor_cache["sensor_9"]["slope_10"] - sensor_cache["sensor_7"]["slope_10"]
            )
        if {"sensor_14", "sensor_7"}.issubset(sensor_cache):
            feature_data["sensor_14_minus_sensor_7_cond_fs_slope_10"] = (
                sensor_cache["sensor_14"]["slope_10"] - sensor_cache["sensor_7"]["slope_10"]
            )
        if {"sensor_9", "sensor_14"}.issubset(sensor_cache):
            feature_data["sensor_9_minus_sensor_14_cond_fs_delta_30"] = (
                sensor_cache["sensor_9"]["delta_30"] - sensor_cache["sensor_14"]["delta_30"]
            )
        if {"sensor_4", "sensor_11"}.issubset(sensor_cache):
            feature_data["sensor_4_minus_sensor_11_cond_fs_slope_10"] = (
                sensor_cache["sensor_4"]["slope_10"] - sensor_cache["sensor_11"]["slope_10"]
            )
        if {"sensor_15", "sensor_2"}.issubset(sensor_cache):
            feature_data["sensor_15_minus_sensor_2_cond_fs_delta_30"] = (
                sensor_cache["sensor_15"]["delta_30"] - sensor_cache["sensor_2"]["delta_30"]
            )

        frames.append(pd.DataFrame(feature_data))

    all_features = pd.concat(frames, ignore_index=True)
    feature_columns = [column for column in all_features.columns if column not in {"unit", "cycle"}]

    if endpoints is None:
        return all_features, feature_columns

    endpoint_features = endpoints[["unit", "cycle"]].merge(
        all_features[["unit", "cycle"] + feature_columns],
        on=["unit", "cycle"],
        how="left",
    )
    endpoint_features[feature_columns] = endpoint_features[feature_columns].fillna(0.0)
    return endpoint_features, feature_columns


def add_fd002_extra_features_for_feature_set(
    temporal_df,
    source_aug,
    endpoints_aug,
    feature_set,
    window_size=DEFAULT_WINDOW_SIZE,
):
    if feature_set != "condition_fault_sensitive":
        return temporal_df, []

    extra_df, extra_columns = make_fd002_fault_sensitive_features(
        source_aug,
        endpoints_df=endpoints_aug,
        window_size=window_size,
    )
    result = temporal_df.merge(extra_df, on=["unit", "cycle"], how="left")
    result[extra_columns] = result[extra_columns].fillna(0.0)
    return result, extra_columns


def scale_temporal_frames(train_temporal, eval_temporal, feature_columns):
    scaler = StandardScaler()
    x_train = pd.DataFrame(
        scaler.fit_transform(train_temporal[feature_columns]),
        columns=feature_columns,
        index=train_temporal.index,
    )
    x_eval = pd.DataFrame(
        scaler.transform(eval_temporal[feature_columns]),
        columns=feature_columns,
        index=eval_temporal.index,
    )
    return scaler, x_train, x_eval


def prepare_fd002_temporal_validation(
    data_dir="CMAPSSData",
    eval_size=0.2,
    random_state=42,
    rul_cap=DEFAULT_RUL_CAP,
    cut_ruls=DEFAULT_CUT_RULS,
    window_size=DEFAULT_WINDOW_SIZE,
    feature_set="condition_normalized",
):
    train = fd002_train_with_rul(data_dir=data_dir, rul_cap=rul_cap)
    train_units, eval_units = split_units(
        train,
        unit_col="unit",
        test_size=eval_size,
        random_state=random_state,
    )
    train_source = train.loc[train["unit"].isin(train_units)].copy()
    eval_source = train.loc[train["unit"].isin(eval_units)].copy()
    eval_cutoffs = make_fd001_artificial_cutoffs(
        eval_source,
        cut_ruls=cut_ruls,
        max_rul=rul_cap,
    )

    condition_preprocessor = fit_condition_preprocessor(train_source, random_state=random_state)
    train_aug = add_fd002_condition_features(train_source, condition_preprocessor)
    eval_source_aug = add_fd002_condition_features(eval_source, condition_preprocessor)
    eval_cutoffs_aug = add_fd002_condition_features(eval_cutoffs, condition_preprocessor)

    base_columns = base_columns_for_feature_set(feature_set)
    train_temporal = make_temporal_features_fast(
        train_aug,
        endpoints_df=train_aug,
        feature_columns=base_columns,
        window_size=window_size,
    )
    eval_temporal = make_temporal_features_fast(
        eval_source_aug,
        endpoints_df=eval_cutoffs_aug,
        feature_columns=base_columns,
        window_size=window_size,
    )
    temporal_columns = temporal_columns_for_base(base_columns)
    train_temporal, train_extra_columns = add_fd002_extra_features_for_feature_set(
        train_temporal,
        train_aug,
        train_aug,
        feature_set,
        window_size=window_size,
    )
    eval_temporal, eval_extra_columns = add_fd002_extra_features_for_feature_set(
        eval_temporal,
        eval_source_aug,
        eval_cutoffs_aug,
        feature_set,
        window_size=window_size,
    )
    extra_columns = train_extra_columns or eval_extra_columns
    feature_columns = temporal_columns + extra_columns
    scaler, x_train, x_eval = scale_temporal_frames(train_temporal, eval_temporal, feature_columns)

    return {
        "dataset": "FD002",
        "feature_set": feature_set,
        "base_feature_columns": base_columns,
        "feature_columns": feature_columns,
        "extra_feature_columns": extra_columns,
        "fault_sensitive_sensors": list(FD002_FAULT_SENSITIVE_SENSORS)
        if feature_set == "condition_fault_sensitive"
        else [],
        "window_size": int(window_size),
        "rul_cap": rul_cap,
        "cut_ruls": tuple(cut_ruls),
        "random_state": int(random_state),
        "train_units": train_units,
        "eval_units": eval_units,
        "condition_preprocessor": condition_preprocessor,
        "scaler": scaler,
        "train_source_df": train_source,
        "eval_source_df": eval_source,
        "eval_cutoff_df": eval_cutoffs,
        "train_df": train_temporal,
        "eval_df": eval_temporal,
        "X_train": x_train,
        "y_train": train_temporal["RUL"].copy(),
        "y_train_raw": train_temporal["RUL_raw"].copy(),
        "X_eval": x_eval,
        "y_eval": eval_temporal["RUL"].copy(),
        "y_eval_raw": eval_temporal["RUL_raw"].copy(),
    }


def prepare_fd002_temporal_full_train_for_test(
    data_dir="CMAPSSData",
    rul_cap=DEFAULT_RUL_CAP,
    window_size=DEFAULT_WINDOW_SIZE,
    feature_set="condition_normalized",
    random_state=42,
):
    data = load_cmapss_subset("FD002", data_dir=resolve_data_dir(data_dir))
    train = add_train_rul(data.train, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    if rul_cap is not None:
        train["RUL"] = train["RUL"].clip(upper=rul_cap)

    condition_preprocessor = fit_condition_preprocessor(train, random_state=random_state)
    train_aug = add_fd002_condition_features(train, condition_preprocessor)

    test_last = last_cycle_rows(data.test).merge(data.rul, on="unit", how="left")
    test_last = test_last.rename(columns={"final_rul": "RUL_raw"})
    test_last["RUL"] = test_last["RUL_raw"]
    if rul_cap is not None:
        test_last["RUL"] = test_last["RUL"].clip(upper=rul_cap)

    test_source_aug = add_fd002_condition_features(data.test, condition_preprocessor)
    test_last_aug = add_fd002_condition_features(test_last, condition_preprocessor)

    base_columns = base_columns_for_feature_set(feature_set)
    train_temporal = make_temporal_features_fast(
        train_aug,
        endpoints_df=train_aug,
        feature_columns=base_columns,
        window_size=window_size,
    )
    test_temporal = make_temporal_features_fast(
        test_source_aug,
        endpoints_df=test_last_aug,
        feature_columns=base_columns,
        window_size=window_size,
    )
    temporal_columns = temporal_columns_for_base(base_columns)
    train_temporal, train_extra_columns = add_fd002_extra_features_for_feature_set(
        train_temporal,
        train_aug,
        train_aug,
        feature_set,
        window_size=window_size,
    )
    test_temporal, test_extra_columns = add_fd002_extra_features_for_feature_set(
        test_temporal,
        test_source_aug,
        test_last_aug,
        feature_set,
        window_size=window_size,
    )
    extra_columns = train_extra_columns or test_extra_columns
    feature_columns = temporal_columns + extra_columns
    scaler = StandardScaler()
    x_train = pd.DataFrame(
        scaler.fit_transform(train_temporal[feature_columns]),
        columns=feature_columns,
        index=train_temporal.index,
    )
    x_test = pd.DataFrame(
        scaler.transform(test_temporal[feature_columns]),
        columns=feature_columns,
        index=test_temporal.index,
    )

    return {
        "dataset": "FD002",
        "feature_set": feature_set,
        "base_feature_columns": base_columns,
        "feature_columns": feature_columns,
        "extra_feature_columns": extra_columns,
        "fault_sensitive_sensors": list(FD002_FAULT_SENSITIVE_SENSORS)
        if feature_set == "condition_fault_sensitive"
        else [],
        "window_size": int(window_size),
        "rul_cap": rul_cap,
        "condition_preprocessor": condition_preprocessor,
        "scaler": scaler,
        "train_df": train_temporal,
        "test_last_df": test_temporal,
        "X_train": x_train,
        "y_train": train_temporal["RUL"].copy(),
        "y_train_raw": train_temporal["RUL_raw"].copy(),
        "X_test_last": x_test,
        "y_test_last": test_temporal["RUL"].copy(),
        "y_test_last_raw": test_temporal["RUL_raw"].copy(),
    }


def make_model(config, random_state=42):
    model_type = config["model_type"]
    params = dict(config.get("params", {}))
    if model_type == "ridge":
        return Ridge(**params)
    if model_type == "random_forest":
        params.setdefault("random_state", random_state)
        params.setdefault("n_jobs", 1)
        return RandomForestRegressor(**params)
    if model_type == "extra_trees":
        params.setdefault("random_state", random_state)
        params.setdefault("n_jobs", 1)
        return ExtraTreesRegressor(**params)
    if model_type == "hist_gradient_boosting":
        params.setdefault("random_state", random_state)
        return HistGradientBoostingRegressor(**params)
    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor

        params.setdefault("random_state", random_state)
        params.setdefault("n_jobs", 1)
        params.setdefault("verbose", -1)
        return LGBMRegressor(**params)
    if model_type == "xgboost":
        from xgboost import XGBRegressor

        params.setdefault("random_state", random_state)
        params.setdefault("n_jobs", 1)
        params.setdefault("tree_method", "hist")
        params.setdefault("verbosity", 0)
        return XGBRegressor(**params)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def fd002_weights_from_scheme(y_raw, scheme):
    if scheme == "mid_rul_guard":
        y_raw = np.asarray(y_raw, dtype=float)
        return np.select(
            [y_raw <= 30, (y_raw > 30) & (y_raw <= 60), (y_raw > 60) & (y_raw <= 90)],
            [4.0, 3.0, 3.0],
            default=1.0,
        )
    if scheme == "mid_rul_guard_soft":
        y_raw = np.asarray(y_raw, dtype=float)
        return np.select(
            [y_raw <= 30, (y_raw > 30) & (y_raw <= 60), (y_raw > 60) & (y_raw <= 90)],
            [3.0, 2.0, 2.0],
            default=1.0,
        )
    return weights_from_scheme(y_raw, scheme)


def fit_predict_config(prepared, config, random_state=42):
    model = make_model(config, random_state=random_state)
    weights = fd002_weights_from_scheme(prepared["y_train_raw"], config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(prepared["X_train"], prepared["y_train"])
    else:
        model.fit(prepared["X_train"], prepared["y_train"], sample_weight=weights)
    y_pred = model.predict(prepared["X_eval"])
    predictions = prediction_frame(
        prepared["eval_df"],
        y_pred,
        model_name=config["candidate_label"],
        representation=config["representation"],
    )
    predictions["feature_set"] = prepared["feature_set"]
    predictions["window_size"] = prepared["window_size"]
    predictions["rul_cap"] = prepared["rul_cap"]
    predictions["sample_weight_scheme"] = config.get("sample_weight_scheme", "none")
    predictions["model_type"] = config["model_type"]
    return model, predictions


def metric_row(predictions, config, prepared, extra=None):
    row = metrics_by_model(predictions).iloc[0].to_dict()
    row.update(
        {
            "candidate_label": config["candidate_label"],
            "model_type": config["model_type"],
            "feature_set": prepared["feature_set"],
            "window_size": prepared["window_size"],
            "rul_cap": prepared["rul_cap"],
            "sample_weight_scheme": config.get("sample_weight_scheme", "none"),
            "n_features": len(prepared["feature_columns"]),
            "params": params_to_json(config.get("params", {})),
            "selection_note": config.get("selection_note", ""),
        }
    )
    if extra:
        row.update(extra)
    return row


def selection_sort(df):
    return df.sort_values(
        ["cmapss_score", "rmse", "dangerous_error_pct", "mae"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def params_to_json(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def model_comparison_configs():
    return [
        {
            "candidate_label": "fd001_like_random_forest_raw",
            "model_type": "random_forest",
            "feature_set": "raw",
            "window_size": 50,
            "representation": "temporal_w50_raw",
            "sample_weight_scheme": "none",
            "selection_note": "FD001-style temporal model without explicit condition handling.",
            "params": {
                "n_estimators": 80,
                "max_depth": 14,
                "min_samples_leaf": 3,
            },
        },
        {
            "candidate_label": "hist_gb_raw",
            "model_type": "hist_gradient_boosting",
            "feature_set": "raw",
            "window_size": 50,
            "representation": "temporal_w50_raw",
            "sample_weight_scheme": "none",
            "selection_note": "Fast boosting baseline over raw temporal features.",
            "params": {
                "loss": "squared_error",
                "learning_rate": 0.05,
                "max_iter": 250,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 15,
            },
        },
        {
            "candidate_label": "hist_gb_raw_plus_condition",
            "model_type": "hist_gradient_boosting",
            "feature_set": "raw_plus_condition",
            "window_size": 50,
            "representation": "temporal_w50_condition_id",
            "sample_weight_scheme": "none",
            "selection_note": "Raw temporal features plus inferred condition_id and condition mix.",
            "params": {
                "loss": "squared_error",
                "learning_rate": 0.05,
                "max_iter": 250,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 15,
            },
        },
        {
            "candidate_label": "hist_gb_condition_normalized",
            "model_type": "hist_gradient_boosting",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized",
            "sample_weight_scheme": "none",
            "selection_note": "Condition-aware representation based on FD002 EDA.",
            "params": {
                "loss": "squared_error",
                "learning_rate": 0.05,
                "max_iter": 250,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 15,
            },
        },
        {
            "candidate_label": "hist_gb_condition_normalized_weighted",
            "model_type": "hist_gradient_boosting",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized_weighted",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "Condition-normalized features with near-failure sample weighting.",
            "params": {
                "loss": "squared_error",
                "learning_rate": 0.05,
                "max_iter": 250,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 15,
            },
        },
        {
            "candidate_label": "extra_trees_condition_normalized",
            "model_type": "extra_trees",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized",
            "sample_weight_scheme": "none",
            "selection_note": "Nonlinear tree ensemble fallback with condition-normalized features.",
            "params": {
                "n_estimators": 120,
                "max_depth": 18,
                "min_samples_leaf": 2,
            },
        },
    ]


def sample_hgb_search_configs(n_configs=12, random_state=42):
    grid = {
        "window_size": [30, 50, 70],
        "loss": ["squared_error", "absolute_error", "quantile"],
        "quantile": [0.35, 0.45],
        "learning_rate": [0.04, 0.06, 0.08],
        "max_iter": [180, 250, 350],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_features": [0.8, 1.0],
        "sample_weight_scheme": ["none", "bin_weights", "soft"],
    }
    keys = list(grid)
    all_configs = []
    for values in product(*(grid[key] for key in keys)):
        item = dict(zip(keys, values))
        if item["loss"] != "quantile":
            item["quantile"] = None
        all_configs.append(item)

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(all_configs), size=int(n_configs), replace=False)
    configs = []
    for idx, grid_item_idx in enumerate(indices):
        item = all_configs[int(grid_item_idx)]
        params = {
            "loss": item["loss"],
            "learning_rate": item["learning_rate"],
            "max_iter": item["max_iter"],
            "max_leaf_nodes": item["max_leaf_nodes"],
            "min_samples_leaf": item["min_samples_leaf"],
            "l2_regularization": item["l2_regularization"],
            "max_features": item["max_features"],
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 15,
        }
        if item["loss"] == "quantile":
            params["quantile"] = item["quantile"]
        configs.append(
            {
                "candidate_label": f"hgb_search_{idx:02d}_{item['loss']}_w{item['window_size']}",
                "model_type": "hist_gradient_boosting",
                "feature_set": "condition_normalized",
                "window_size": int(item["window_size"]),
                "representation": f"temporal_w{item['window_size']}_condition_normalized",
                "sample_weight_scheme": item["sample_weight_scheme"],
                "selection_note": "FD002 condition-normalized HistGradientBoosting hyperparameter search.",
                "params": params,
            }
        )
    return configs


def fd001_fd003_lgbm_reference_params():
    return {
        "objective": "quantile",
        "alpha": 0.4,
        "learning_rate": 0.03,
        "n_estimators": 1300,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 10.0,
    }


def lgbm_xgb_model_comparison_configs():
    reference_params = fd001_fd003_lgbm_reference_params()
    return [
        {
            "candidate_label": "fd001_fd003_lgbm_reference_raw",
            "model_type": "lightgbm",
            "feature_set": "raw",
            "window_size": 50,
            "representation": "temporal_w50_raw",
            "sample_weight_scheme": "none",
            "selection_note": "FD001/FD003 LightGBM recipe transferred without FD002 condition-aware features.",
            "params": reference_params,
        },
        {
            "candidate_label": "fd001_fd003_lgbm_reference_condition_normalized",
            "model_type": "lightgbm",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized",
            "sample_weight_scheme": "none",
            "selection_note": "Same LightGBM recipe as FD001/FD003, adapted only through FD002 condition-normalized features.",
            "params": reference_params,
        },
        {
            "candidate_label": "lgbm_regression_condition_normalized_weighted",
            "model_type": "lightgbm",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "LightGBM regression with FD002 condition-aware features and near-failure weights.",
            "params": {
                "objective": "regression",
                "learning_rate": 0.04,
                "n_estimators": 900,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
            },
        },
        {
            "candidate_label": "lgbm_quantile_a04_condition_normalized_weighted",
            "model_type": "lightgbm",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "LightGBM quantile alpha=0.4 with FD002 condition-aware features and near-failure weights.",
            "params": {
                "objective": "quantile",
                "alpha": 0.4,
                "learning_rate": 0.03,
                "n_estimators": 1100,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.85,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
            },
        },
        {
            "candidate_label": "xgb_squarederror_condition_normalized",
            "model_type": "xgboost",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized",
            "sample_weight_scheme": "none",
            "selection_note": "XGBoost baseline with FD002 condition-aware features.",
            "params": {
                "objective": "reg:squarederror",
                "n_estimators": 650,
                "max_depth": 3,
                "learning_rate": 0.04,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "candidate_label": "xgb_squarederror_condition_normalized_weighted",
            "model_type": "xgboost",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "XGBoost baseline with FD002 condition-aware features and near-failure weights.",
            "params": {
                "objective": "reg:squarederror",
                "n_estimators": 650,
                "max_depth": 3,
                "learning_rate": 0.04,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 5.0,
            },
        },
    ]


def sample_lgbm_xgb_search_configs(n_lgbm=12, n_xgb=4, random_state=42):
    lgbm_grid = {
        "window_size": [30, 50, 70],
        "objective": ["regression", "quantile"],
        "alpha": [0.35, 0.4, 0.45],
        "learning_rate": [0.02, 0.03, 0.05],
        "n_estimators": [700, 1000, 1300],
        "num_leaves": [15, 31, 47],
        "min_child_samples": [10, 20, 40],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "sample_weight_scheme": ["none", "bin_weights", "soft"],
    }
    xgb_grid = {
        "window_size": [50, 70],
        "n_estimators": [400, 650, 900],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0, 5.0, 10.0],
        "sample_weight_scheme": ["none", "bin_weights"],
    }

    rng = np.random.default_rng(random_state)
    configs = []

    lgbm_keys = list(lgbm_grid)
    lgbm_all = [dict(zip(lgbm_keys, values)) for values in product(*(lgbm_grid[key] for key in lgbm_keys))]
    for idx, grid_idx in enumerate(rng.choice(len(lgbm_all), size=int(n_lgbm), replace=False)):
        item = lgbm_all[int(grid_idx)]
        params = {
            "objective": item["objective"],
            "learning_rate": item["learning_rate"],
            "n_estimators": item["n_estimators"],
            "num_leaves": item["num_leaves"],
            "max_depth": -1,
            "min_child_samples": item["min_child_samples"],
            "subsample": item["subsample"],
            "colsample_bytree": item["colsample_bytree"],
            "reg_alpha": item["reg_alpha"],
            "reg_lambda": item["reg_lambda"],
        }
        if item["objective"] == "quantile":
            params["alpha"] = item["alpha"]
        configs.append(
            {
                "candidate_label": f"lgbm_search_{idx:02d}_{item['objective']}_w{item['window_size']}",
                "model_type": "lightgbm",
                "feature_set": "condition_normalized",
                "window_size": int(item["window_size"]),
                "representation": f"temporal_w{item['window_size']}_condition_normalized",
                "sample_weight_scheme": item["sample_weight_scheme"],
                "selection_note": "FD002 condition-normalized LightGBM hyperparameter search.",
                "params": params,
            }
        )

    xgb_keys = list(xgb_grid)
    xgb_all = [dict(zip(xgb_keys, values)) for values in product(*(xgb_grid[key] for key in xgb_keys))]
    for idx, grid_idx in enumerate(rng.choice(len(xgb_all), size=int(n_xgb), replace=False)):
        item = xgb_all[int(grid_idx)]
        configs.append(
            {
                "candidate_label": f"xgb_search_{idx:02d}_squarederror_w{item['window_size']}",
                "model_type": "xgboost",
                "feature_set": "condition_normalized",
                "window_size": int(item["window_size"]),
                "representation": f"temporal_w{item['window_size']}_condition_normalized",
                "sample_weight_scheme": item["sample_weight_scheme"],
                "selection_note": "FD002 condition-normalized XGBoost hyperparameter search.",
                "params": {
                    "objective": "reg:squarederror",
                    "n_estimators": item["n_estimators"],
                    "max_depth": item["max_depth"],
                    "learning_rate": item["learning_rate"],
                    "subsample": item["subsample"],
                    "colsample_bytree": item["colsample_bytree"],
                    "reg_alpha": item["reg_alpha"],
                    "reg_lambda": item["reg_lambda"],
                },
            }
        )
    return configs


def feature_engineering_search_configs():
    reference_params = fd001_fd003_lgbm_reference_params()
    return [
        {
            "candidate_label": "xgb_condition_fault_sensitive_weighted",
            "model_type": "xgboost",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive",
            "sample_weight_scheme": "bin_weights",
            "selection_note": (
                "Current best FD002 XGBoost recipe plus condition-normalized fault-sensitive features."
            ),
            "params": {
                "objective": "reg:squarederror",
                "n_estimators": 650,
                "max_depth": 3,
                "learning_rate": 0.04,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "candidate_label": "xgb_condition_fault_sensitive_mid_guard",
            "model_type": "xgboost",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive_mid_guard",
            "sample_weight_scheme": "mid_rul_guard",
            "selection_note": (
                "Fault-sensitive XGBoost with extra weight on the 30-90 RUL band where validation errors were riskier."
            ),
            "params": {
                "objective": "reg:squarederror",
                "n_estimators": 650,
                "max_depth": 3,
                "learning_rate": 0.04,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "candidate_label": "xgb_condition_fault_sensitive_shallow_w50",
            "model_type": "xgboost",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "Shallower fault-sensitive XGBoost to test whether extra features need stronger regularization.",
            "params": {
                "objective": "reg:squarederror",
                "n_estimators": 900,
                "max_depth": 2,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 10.0,
            },
        },
        {
            "candidate_label": "xgb_condition_fault_sensitive_w70",
            "model_type": "xgboost",
            "feature_set": "condition_fault_sensitive",
            "window_size": 70,
            "representation": "temporal_w70_condition_fault_sensitive",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "Fault-sensitive XGBoost with a wider temporal context.",
            "params": {
                "objective": "reg:squarederror",
                "n_estimators": 650,
                "max_depth": 3,
                "learning_rate": 0.04,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 5.0,
            },
        },
        {
            "candidate_label": "lgbm_reference_condition_fault_sensitive",
            "model_type": "lightgbm",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive",
            "sample_weight_scheme": "none",
            "selection_note": "FD001/FD003 LightGBM recipe with FD002 condition-normalized fault-sensitive features.",
            "params": reference_params,
        },
        {
            "candidate_label": "lgbm_quantile_a04_condition_fault_sensitive_weighted",
            "model_type": "lightgbm",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "Current FD002 LightGBM quantile recipe plus fault-sensitive features.",
            "params": {
                "objective": "quantile",
                "alpha": 0.4,
                "learning_rate": 0.03,
                "n_estimators": 1100,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.85,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
            },
        },
        {
            "candidate_label": "lgbm_quantile_a035_condition_fault_sensitive_w70",
            "model_type": "lightgbm",
            "feature_set": "condition_fault_sensitive",
            "window_size": 70,
            "representation": "temporal_w70_condition_fault_sensitive",
            "sample_weight_scheme": "none",
            "selection_note": "Best previous FD002 LGBM search shape retested with fault-sensitive features.",
            "params": {
                "objective": "quantile",
                "alpha": 0.35,
                "learning_rate": 0.03,
                "n_estimators": 1000,
                "num_leaves": 15,
                "max_depth": -1,
                "min_child_samples": 10,
                "subsample": 1.0,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 10.0,
            },
        },
        {
            "candidate_label": "lgbm_regression_condition_fault_sensitive_mid_guard",
            "model_type": "lightgbm",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive_mid_guard",
            "sample_weight_scheme": "mid_rul_guard_soft",
            "selection_note": "Fault-sensitive LightGBM regression with moderated extra weight on 30-90 RUL.",
            "params": {
                "objective": "regression",
                "learning_rate": 0.04,
                "n_estimators": 900,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
            },
        },
    ]


def evaluate_configs(configs, data_dir="CMAPSSData", random_state=42, rul_cap=DEFAULT_RUL_CAP, cut_ruls=DEFAULT_CUT_RULS):
    prepared_cache = {}
    metric_rows = []
    prediction_tables = []
    fitted_models = {}

    for config in configs:
        cache_key = (config["feature_set"], int(config["window_size"]), int(random_state))
        if cache_key not in prepared_cache:
            prepared_cache[cache_key] = prepare_fd002_temporal_validation(
                data_dir=data_dir,
                random_state=random_state,
                rul_cap=rul_cap,
                cut_ruls=cut_ruls,
                window_size=int(config["window_size"]),
                feature_set=config["feature_set"],
            )
        prepared = prepared_cache[cache_key]
        model, predictions = fit_predict_config(prepared, config, random_state=random_state)
        metric_rows.append(metric_row(predictions, config, prepared, extra={"random_state": random_state}))
        prediction_tables.append(predictions)
        fitted_models[config["candidate_label"]] = model

    metrics = selection_sort(pd.DataFrame(metric_rows))
    predictions = pd.concat(prediction_tables, ignore_index=True)
    return metrics, predictions, fitted_models


def official_test_predictions(prepared, model, config):
    y_pred = model.predict(prepared["X_test_last"])
    predictions = prediction_frame(
        prepared["test_last_df"],
        y_pred,
        model_name=config["candidate_label"],
        representation=config["representation"],
    )
    predictions["feature_set"] = prepared["feature_set"]
    predictions["window_size"] = prepared["window_size"]
    predictions["rul_cap"] = prepared["rul_cap"]
    predictions["sample_weight_scheme"] = config.get("sample_weight_scheme", "none")
    predictions["model_type"] = config["model_type"]
    return predictions


def fit_final_model_and_predict(data_dir, best_config, random_state=42, rul_cap=DEFAULT_RUL_CAP):
    prepared = prepare_fd002_temporal_full_train_for_test(
        data_dir=data_dir,
        rul_cap=rul_cap,
        window_size=int(best_config["window_size"]),
        feature_set=best_config["feature_set"],
        random_state=random_state,
    )
    model = make_model(best_config, random_state=random_state)
    weights = fd002_weights_from_scheme(prepared["y_train_raw"], best_config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(prepared["X_train"], prepared["y_train"])
    else:
        model.fit(prepared["X_train"], prepared["y_train"], sample_weight=weights)
    predictions = official_test_predictions(prepared, model, best_config)
    return prepared, model, predictions


def prediction_metrics_single(predictions, n_col="n_test"):
    row = metrics_by_model(predictions).iloc[0].to_dict()
    row[n_col] = row.pop("n_eval")
    return pd.DataFrame([row])


def config_from_metric_row(row):
    params = json.loads(row["params"])
    return {
        "candidate_label": row["candidate_label"],
        "model_type": row["model_type"],
        "feature_set": row["feature_set"],
        "window_size": int(row["window_size"]),
        "representation": row["representation"],
        "sample_weight_scheme": row.get("sample_weight_scheme", "none"),
        "selection_note": row.get("selection_note", ""),
        "params": params,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=False)


def build_best_model_config(best_row, validation_metrics, official_metrics, prepared, paths):
    best_config = config_from_metric_row(best_row)
    return {
        "dataset": "FD002",
        "task": "Remaining Useful Life regression",
        "selection_policy": (
            "Best model selected by lowest C-MAPSS score on artificial-cutoff validation. "
            "Official FD002 test is used only after selection for final reporting."
        ),
        "problem_analysis": {
            "fd001_reference": "FD001 has one operating condition and one fault mode.",
            "fd002_difference": (
                "FD002 has six operating conditions and one fault mode; conditions change inside each engine trajectory."
            ),
            "chosen_strategy": (
                "Use a condition-aware temporal tabular model: infer condition_id from settings, add condition one-hot "
                "features, add sensor z-scores normalized within each condition, and test a fault-sensitive feature "
                "set inspired by the FD003 improvement."
            ),
            "fd003_transfer_lesson": (
                "For FD003, extra degradation-oriented temporal features improved the model more than ordinary tuning; "
                "FD002 therefore tests the same idea after correcting for its six operating conditions."
            ),
        },
        "preprocessing": {
            "rul_raw_definition": "max_cycle - cycle for train; RUL_FD002.txt for official test last cycle",
            "training_target": "RUL capped",
            "metric_target": "RUL_raw uncapped",
            "rul_cap": int(prepared["rul_cap"]),
            "cut_ruls": list(DEFAULT_CUT_RULS),
            "condition_count": CONDITION_COUNT,
            "condition_method": "KMeans over setting_1, setting_2 and setting_3; model fit only on training split",
            "feature_set": best_config["feature_set"],
            "window_size": int(best_config["window_size"]),
            "n_features": int(len(prepared["feature_columns"])),
            "base_feature_columns": list(prepared["base_feature_columns"]),
            "feature_columns": list(prepared["feature_columns"]),
            "extra_feature_columns": list(prepared.get("extra_feature_columns", [])),
            "fault_sensitive_sensors": list(prepared.get("fault_sensitive_sensors", [])),
        },
        "validation": {
            "split_policy": "complete engine units, never individual rows",
            "cutoff_rule": "cut_cycle = max_cycle - cut_rul; skip if cut_cycle < 1",
            "random_state": int(best_row["random_state"]),
        },
        "final_model": {
            "candidate_label": best_config["candidate_label"],
            "model_type": best_config["model_type"],
            "representation": best_config["representation"],
            "sample_weight_scheme": best_config["sample_weight_scheme"],
            "hyperparameters": best_config["params"],
        },
        "validation_metrics": validation_metrics,
        "official_test_metrics": official_metrics,
        "artifacts": {
            "model_comparison": str(paths["results"] / "fd002_model_family_comparison.csv"),
            "external_model_comparison": str(paths["results"] / "fd002_lgbm_xgb_model_comparison.csv"),
            "hyperparameter_search": str(paths["results"] / "fd002_lgbm_xgb_hyperparam_search.csv"),
            "feature_engineering_search": str(paths["results"] / "fd002_feature_engineering_search.csv"),
            "final_candidate_ranking": str(paths["results"] / "fd002_final_candidate_ranking.csv"),
            "validation_predictions": str(paths["results"] / "fd002_best_validation_predictions.csv"),
            "official_test_predictions": str(paths["predictions"] / "fd002_best_model_predictions.csv"),
            "official_test_metrics": str(paths["results"] / "fd002_official_test_metrics.csv"),
        },
    }


def write_interpretation_note(path, model_comparison, search_results, best_config_payload):
    best_search = search_results.iloc[0]
    lines = [
        "FD002 - seleccion de modelo y busqueda de hiperparametros",
        "",
        "Resumen del trabajo:",
        "- FD001 se habia modelado con features temporales, RUL capped, validacion por unidades y cortes artificiales.",
        "- FD002 no debe tratarse como FD001 puro: tiene seis condiciones operativas que cambian dentro de cada motor.",
        "- El EDA mostro que los settings separan seis conditions y que normalizar sensores por condition recupera senal de RUL.",
        "",
        "Decision de modelado:",
        "- Se comparo un baseline estilo FD001 sin condiciones contra modelos temporales condition-aware.",
        "- Se eligio una representacion condition-normalized: settings, sensores crudos, condition_id/one-hot y sensores z-score por condition.",
        "- Con LightGBM y XGBoost disponibles, la busqueda final compara transferencia LGBM FD001/FD003, LGBM adaptado a FD002 y XGBoost.",
        "- Ademas se probo una representacion condition_fault_sensitive inspirada en FD003: pendientes, deltas, volatilidad y aceleracion sobre sensores normalizados por condition.",
        "- La aceptacion del modelo final sigue usando C-MAPSS de validacion artificial; el test oficial se reserva para reporte.",
        "",
        "Mejor candidato interno:",
        (
            f"- {best_search['candidate_label']}: C-MAPSS {best_search['cmapss_score']:.3f}, "
            f"RMSE {best_search['rmse']:.3f}, dangerous error {best_search['dangerous_error_pct']:.2f}%."
        ),
        (
            f"- feature_set={best_search['feature_set']}, modelo={best_search['model_type']}, "
            f"pesos={best_search['sample_weight_scheme']}."
        ),
        "",
        "Comparacion principal:",
    ]
    for _, row in model_comparison.head(6).iterrows():
        lines.append(
            f"- {row['candidate_label']}: C-MAPSS {row['cmapss_score']:.3f}, "
            f"RMSE {row['rmse']:.3f}, feature_set={row['feature_set']}."
        )
    lines.extend(
        [
            "",
            "Config final guardado en:",
            "- configs/FD002/fd002_best_model_config.json",
            "",
            "Busqueda feature engineering:",
            "- results/FD002/fd002_feature_engineering_search.csv",
            "",
            "Predicciones oficiales guardadas en:",
            "- predictions/fd002_best_model_predictions.csv",
            "",
            "Nota metodologica:",
            "- El test oficial no se uso para elegir hiperparametros; solo para reportar el candidato final.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_fd002_search_summary(search_results, model_comparison, figures_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    top = search_results.head(12).copy()
    sns.barplot(data=top, y="candidate_label", x="cmapss_score", color="#4C78A8")
    plt.xlabel("C-MAPSS score")
    plt.ylabel("")
    plt.title("FD002: mejores candidatos de busqueda")
    plt.tight_layout()
    plt.savefig(figures_dir / "fd002_model_search_top_models_cmapss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=search_results,
        x="rmse",
        y="cmapss_score",
        hue="sample_weight_scheme",
        style="window_size",
        s=70,
    )
    plt.title("FD002: busqueda RMSE vs C-MAPSS")
    plt.tight_layout()
    plt.savefig(figures_dir / "fd002_model_search_rmse_vs_cmapss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    top_cmp = model_comparison.head(8).copy()
    sns.barplot(data=top_cmp, y="candidate_label", x="cmapss_score", color="#72B7B2")
    plt.xlabel("C-MAPSS score")
    plt.ylabel("")
    plt.title("FD002: comparacion de representaciones y modelos")
    plt.tight_layout()
    plt.savefig(figures_dir / "fd002_model_family_comparison_cmapss.png", dpi=150)
    plt.close()


def run_fd002_modeling_workflow(
    project_root=PROJECT_ROOT,
    data_dir="CMAPSSData",
    random_state=42,
    search_size=12,
):
    paths = fd002_output_paths(project_root)

    comparison_configs = model_comparison_configs()
    comparison_metrics, comparison_predictions, _ = evaluate_configs(
        comparison_configs,
        data_dir=data_dir,
        random_state=random_state,
    )
    comparison_metrics.to_csv(paths["results"] / "fd002_model_family_comparison.csv", index=False)
    comparison_predictions.to_csv(paths["results"] / "fd002_model_family_comparison_predictions.csv", index=False)

    search_configs = sample_hgb_search_configs(n_configs=search_size, random_state=random_state)
    search_metrics, search_predictions, _ = evaluate_configs(
        search_configs,
        data_dir=data_dir,
        random_state=random_state,
    )
    search_metrics.to_csv(paths["results"] / "fd002_hgb_hyperparam_search.csv", index=False)
    search_predictions.to_csv(paths["results"] / "fd002_hgb_hyperparam_search_predictions.csv", index=False)

    external_configs = lgbm_xgb_model_comparison_configs()
    external_metrics, external_predictions, _ = evaluate_configs(
        external_configs,
        data_dir=data_dir,
        random_state=random_state,
    )
    external_metrics.to_csv(paths["results"] / "fd002_lgbm_xgb_model_comparison.csv", index=False)
    external_predictions.to_csv(paths["results"] / "fd002_lgbm_xgb_model_comparison_predictions.csv", index=False)

    external_search_configs = sample_lgbm_xgb_search_configs(
        n_lgbm=search_size,
        n_xgb=max(4, search_size // 3),
        random_state=random_state,
    )
    external_search_metrics, external_search_predictions, _ = evaluate_configs(
        external_search_configs,
        data_dir=data_dir,
        random_state=random_state,
    )
    external_search_metrics.to_csv(paths["results"] / "fd002_lgbm_xgb_hyperparam_search.csv", index=False)
    external_search_predictions.to_csv(
        paths["results"] / "fd002_lgbm_xgb_hyperparam_search_predictions.csv",
        index=False,
    )

    feature_configs = feature_engineering_search_configs()
    feature_metrics, feature_predictions, _ = evaluate_configs(
        feature_configs,
        data_dir=data_dir,
        random_state=random_state,
    )
    feature_metrics.to_csv(paths["results"] / "fd002_feature_engineering_search.csv", index=False)
    feature_predictions.to_csv(paths["results"] / "fd002_feature_engineering_search_predictions.csv", index=False)

    ranking_parts = [
        comparison_metrics.assign(experiment_stage="baseline_model_family"),
        search_metrics.assign(experiment_stage="hgb_search"),
        external_metrics.assign(experiment_stage="lgbm_xgb_comparison"),
        external_search_metrics.assign(experiment_stage="lgbm_xgb_search"),
        feature_metrics.assign(experiment_stage="feature_engineering_search"),
    ]
    final_ranking = selection_sort(pd.concat(ranking_parts, ignore_index=True))
    final_ranking.to_csv(paths["results"] / "fd002_final_candidate_ranking.csv", index=False)

    best_row = final_ranking.iloc[0]
    best_config = config_from_metric_row(best_row)

    validation_prepared = prepare_fd002_temporal_validation(
        data_dir=data_dir,
        random_state=random_state,
        window_size=int(best_config["window_size"]),
        feature_set=best_config["feature_set"],
    )
    validation_model, validation_predictions = fit_predict_config(
        validation_prepared,
        best_config,
        random_state=random_state,
    )
    validation_metrics = metrics_by_model(validation_predictions)
    validation_bin_metrics = metrics_by_rul_bin(validation_predictions)
    validation_predictions.to_csv(paths["results"] / "fd002_best_validation_predictions.csv", index=False)
    validation_metrics.to_csv(paths["results"] / "fd002_best_validation_metrics.csv", index=False)
    validation_bin_metrics.to_csv(paths["results"] / "fd002_best_validation_metrics_by_rul_bin.csv", index=False)

    full_prepared, final_model, official_predictions = fit_final_model_and_predict(
        data_dir=data_dir,
        best_config=best_config,
        random_state=random_state,
    )
    official_metrics = prediction_metrics_single(official_predictions, n_col="n_test")
    official_bin_metrics = metrics_by_rul_bin(official_predictions)
    official_predictions.to_csv(paths["predictions"] / "fd002_best_model_predictions.csv", index=False)
    official_predictions.to_csv(paths["results"] / "fd002_official_test_predictions.csv", index=False)
    official_metrics.to_csv(paths["results"] / "fd002_official_test_metrics.csv", index=False)
    official_bin_metrics.to_csv(paths["results"] / "fd002_official_test_metrics_by_rul_bin.csv", index=False)

    plot_validation_diagnostics(
        validation_predictions,
        paths["figures"] / "validation_best_model",
        "FD002 best validation model",
    )
    plot_fd002_search_summary(final_ranking, comparison_metrics, paths["figures"])

    try:
        import joblib

        joblib.dump(
            {
                "model": final_model,
                "condition_preprocessor": full_prepared["condition_preprocessor"],
                "scaler": full_prepared["scaler"],
                "feature_columns": full_prepared["feature_columns"],
                "base_feature_columns": full_prepared["base_feature_columns"],
                "config": best_config,
            },
            paths["checkpoints"] / "fd002_best_model.joblib",
        )
    except Exception as exc:
        import warnings

        warnings.warn(
            f"Could not save FD002 checkpoint: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    config_payload = build_best_model_config(
        best_row=best_row,
        validation_metrics=validation_metrics.iloc[0].to_dict(),
        official_metrics=official_metrics.iloc[0].to_dict(),
        prepared=full_prepared,
        paths=paths,
    )
    write_json(paths["configs"] / "fd002_best_model_config.json", config_payload)
    write_interpretation_note(
        paths["notes"] / "fd002_model_selection_interpretation.txt",
        final_ranking,
        final_ranking,
        config_payload,
    )

    return {
        "model_comparison": comparison_metrics,
        "hgb_hyperparameter_search": search_metrics,
        "lgbm_xgb_model_comparison": external_metrics,
        "lgbm_xgb_hyperparameter_search": external_search_metrics,
        "feature_engineering_search": feature_metrics,
        "final_candidate_ranking": final_ranking,
        "validation_metrics": validation_metrics,
        "official_metrics": official_metrics,
        "best_config": config_payload,
        "paths": paths,
    }
