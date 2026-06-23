from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data_splitting import split_units
from src.fd003_cluster_utils import (
    build_fd003_unit_cluster_features,
    cmapss_penalty,
    fit_fd003_unit_clusters,
)
from src.fd003_transfer_utils import (
    fd003_feature_columns,
    load_fd003_train,
    make_lgbm_from_fd001_config,
)
from src.preprocessed_FD001 import (
    make_fd001_artificial_cutoffs,
    make_temporal_features,
    temporal_feature_columns,
)
from src.data import add_train_rul


FD003_RANDOM_STATES = [0, 1, 2, 3, 4]
FD003_CUT_RULS = (20, 50, 80, 110, 140)
FD003_RELEVANT_SENSORS = ["sensor_7", "sensor_9", "sensor_12", "sensor_14", "sensor_15"]
FD001_FINAL_LGBM_PARAMS = {
    "colsample_bytree": 0.8,
    "learning_rate": 0.03,
    "max_depth": -1,
    "min_child_samples": 10,
    "n_estimators": 1300,
    "num_leaves": 15,
    "reg_alpha": 0.5,
    "reg_lambda": 10.0,
    "subsample": 0.9,
}
FD003_LGBM_REFINEMENT_GRID = {
    "num_leaves": [7, 15, 31],
    "min_child_samples": [10, 20, 40],
    "reg_lambda": [5.0, 10.0, 20.0],
    "reg_alpha": [0.0, 0.5, 1.0],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}


def ensure_fd003_dirs(project_root):
    paths = {
        "results": Path(project_root) / "results" / "FD003",
        "figures": Path(project_root) / "figures" / "FD003",
        "configs": Path(project_root) / "configs" / "FD003",
        "notes": Path(project_root) / "notas" / "FD003",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def fd003_train_with_rul(data_dir, rul_cap=None):
    train = load_fd003_train(data_dir)
    train = add_train_rul(train, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    if rul_cap is not None:
        train["RUL_capped"] = train["RUL"].clip(upper=rul_cap)
        train["RUL"] = train["RUL_capped"]
    return train


def _linear_slope(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, deg=1)[0])


def _safe_ratio(a, b, eps=1e-6):
    return float(a / b) if abs(b) > eps else 0.0


def make_fault_sensitive_feature_frame(source_df, endpoints_df, sensors, window_size):
    source = source_df.sort_values(["unit", "cycle"]).copy()
    endpoints = endpoints_df.sort_values(["unit", "cycle"]).copy()
    history_by_unit = {
        unit: group.reset_index(drop=True)
        for unit, group in source.groupby("unit", sort=True)
    }

    rows = []
    for _, endpoint in endpoints.iterrows():
        unit = endpoint["unit"]
        cycle = endpoint["cycle"]
        history = history_by_unit[unit]
        available = history.loc[history["cycle"] <= cycle]
        row = {"unit": int(unit), "cycle": int(cycle)}
        sensor_cache = {}
        for sensor in sensors:
            values_main = available[sensor].tail(window_size).to_numpy(dtype=float)
            values_30 = available[sensor].tail(30).to_numpy(dtype=float)
            values_10 = available[sensor].tail(10).to_numpy(dtype=float)
            slope_main = _linear_slope(values_main)
            slope_30 = _linear_slope(values_30)
            slope_10 = _linear_slope(values_10)
            delta_30 = float(values_30[-1] - values_30[0]) if len(values_30) else 0.0
            delta_10 = float(values_10[-1] - values_10[0]) if len(values_10) else 0.0
            row[f"{sensor}_fs_slope_10"] = slope_10
            row[f"{sensor}_fs_slope_30"] = slope_30
            row[f"{sensor}_fs_slope_main"] = slope_main
            row[f"{sensor}_fs_delta_10"] = delta_10
            row[f"{sensor}_fs_delta_30"] = delta_30
            row[f"{sensor}_fs_volatility_10"] = float(np.std(values_10, ddof=0)) if len(values_10) else 0.0
            row[f"{sensor}_fs_volatility_30"] = float(np.std(values_30, ddof=0)) if len(values_30) else 0.0
            row[f"{sensor}_fs_slope_diff_10_main"] = slope_10 - slope_main
            row[f"{sensor}_fs_slope_ratio_10_main"] = _safe_ratio(slope_10, slope_main)
            row[f"{sensor}_fs_acceleration_10_30"] = slope_10 - slope_30
            sensor_cache[sensor] = {
                "slope_10": slope_10,
                "delta_30": delta_30,
            }
        if {"sensor_9", "sensor_7"}.issubset(sensor_cache):
            row["sensor_9_minus_sensor_7_slope_short"] = sensor_cache["sensor_9"]["slope_10"] - sensor_cache["sensor_7"]["slope_10"]
        if {"sensor_14", "sensor_7"}.issubset(sensor_cache):
            row["sensor_14_minus_sensor_7_slope_short"] = sensor_cache["sensor_14"]["slope_10"] - sensor_cache["sensor_7"]["slope_10"]
        if {"sensor_9", "sensor_14"}.issubset(sensor_cache):
            row["sensor_9_minus_sensor_14_delta_30"] = sensor_cache["sensor_9"]["delta_30"] - sensor_cache["sensor_14"]["delta_30"]
        rows.append(row)
    return pd.DataFrame(rows)


def make_fd003_temporal_dataset(
    data_dir,
    random_state,
    window_size,
    rul_cap,
    feature_set="base",
    drop_columns=None,
    eval_size=0.2,
    cut_ruls=FD003_CUT_RULS,
):
    train = fd003_train_with_rul(data_dir, rul_cap=rul_cap)
    base_feature_columns, dropped_columns = fd003_feature_columns(train, drop_columns=drop_columns)
    train_units, eval_units = split_units(
        train,
        unit_col="unit",
        test_size=eval_size,
        random_state=random_state,
    )
    train_source_df = train.loc[train["unit"].isin(train_units)].copy()
    eval_source_df = train.loc[train["unit"].isin(eval_units)].copy()
    eval_cutoff_df = make_fd001_artificial_cutoffs(
        eval_source_df,
        cut_ruls=cut_ruls,
        max_rul=rul_cap,
    )

    train_temporal = make_temporal_features(
        train_source_df,
        endpoints_df=train_source_df,
        feature_columns=base_feature_columns,
        window_size=window_size,
    )
    eval_temporal = make_temporal_features(
        eval_source_df,
        endpoints_df=eval_cutoff_df,
        feature_columns=base_feature_columns,
        window_size=window_size,
    )
    feature_columns = temporal_feature_columns(base_feature_columns)

    extra_feature_columns = []
    if feature_set == "fault_sensitive":
        sensors = [sensor for sensor in FD003_RELEVANT_SENSORS if sensor in base_feature_columns]
        train_extra = make_fault_sensitive_feature_frame(train_source_df, train_source_df, sensors, window_size)
        eval_extra = make_fault_sensitive_feature_frame(eval_source_df, eval_cutoff_df, sensors, window_size)
        extra_feature_columns = [column for column in train_extra.columns if column not in {"unit", "cycle"}]
        train_temporal = train_temporal.merge(train_extra, on=["unit", "cycle"], how="left")
        eval_temporal = eval_temporal.merge(eval_extra, on=["unit", "cycle"], how="left")
        feature_columns = feature_columns + extra_feature_columns
    elif feature_set != "base":
        raise ValueError(f"Unknown feature_set: {feature_set}")

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train_temporal[feature_columns]),
        columns=feature_columns,
        index=train_temporal.index,
    )
    X_eval = pd.DataFrame(
        scaler.transform(eval_temporal[feature_columns]),
        columns=feature_columns,
        index=eval_temporal.index,
    )
    return {
        "train_raw": train,
        "train_source_df": train_source_df,
        "eval_source_df": eval_source_df,
        "eval_df": eval_temporal,
        "train_df": train_temporal,
        "eval_cutoff_df": eval_cutoff_df,
        "X_train": X_train,
        "X_eval": X_eval,
        "y_train": train_temporal["RUL"].copy(),
        "y_train_raw": train_temporal["RUL_raw"].copy(),
        "y_eval_raw": eval_temporal["RUL_raw"].copy(),
        "feature_columns": feature_columns,
        "base_feature_columns": base_feature_columns,
        "extra_feature_columns": extra_feature_columns,
        "dropped_columns": dropped_columns,
        "train_units": train_units,
        "eval_units": eval_units,
        "scaler": scaler,
        "feature_set": feature_set,
        "window_size": window_size,
        "rul_cap": rul_cap,
    }


def make_lgbm_regressor_from_config(config, random_state):
    from lightgbm import LGBMRegressor

    params = dict(config.get("hyperparameters", FD001_FINAL_LGBM_PARAMS))
    params.update(
        {
            "objective": config["objective"],
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    if config["objective"] == "quantile":
        params["alpha"] = float(config["alpha"])
    return LGBMRegressor(**params)


def low_rul_sample_weights(y_raw):
    y = np.asarray(y_raw, dtype=float)
    return np.select(
        [y <= 30, (y > 30) & (y <= 60), (y > 60) & (y <= 100)],
        [2.0, 1.5, 1.2],
        default=1.0,
    )


def sample_weights_for_scheme(y_raw, scheme):
    if scheme == "none":
        return None
    if scheme == "low_rul_weight":
        return low_rul_sample_weights(y_raw)
    raise ValueError(f"Unknown sample_weight_scheme: {scheme}")


def normalized_predictions_from_prepared(prepared, y_pred, config, random_state, variant=None):
    result = pd.DataFrame(
        {
            "random_state": random_state,
            "unit_number": prepared["eval_df"]["unit"].to_numpy(dtype=int),
            "cutoff_cycle": prepared["eval_df"]["cycle"].to_numpy(dtype=int),
            "true_rul": prepared["eval_df"]["RUL_raw"].to_numpy(dtype=float),
            "pred_rul": np.clip(np.asarray(y_pred, dtype=float), 0.0, None),
        }
    )
    result["error"] = result["pred_rul"] - result["true_rul"]
    result["abs_error"] = result["error"].abs()
    result["squared_error"] = result["error"] ** 2
    result["dangerous_error"] = result["error"] > 20.0
    result["conservative_error"] = result["error"] < -20.0
    result["cmapss_penalty"] = cmapss_penalty(result["error"])
    result["model_name"] = config.get("model_name", "")
    result["approach"] = config.get("approach", "")
    result["window_size"] = config["window_size"]
    result["rul_cap"] = config["rul_cap"]
    result["objective"] = config["objective"]
    result["alpha"] = config.get("alpha")
    result["sample_weight_scheme"] = config.get("sample_weight_scheme", "none")
    result["feature_set"] = config.get("feature_set", "base")
    if variant is not None:
        result["variant"] = variant
    return result


def metrics_from_predictions(predictions):
    y_true = predictions["true_rul"].to_numpy(dtype=float)
    y_pred = predictions["pred_rul"].to_numpy(dtype=float)
    return {
        "n_predictions": int(len(predictions)),
        "n_units": int(predictions["unit_number"].nunique()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "cmapss_score": float(predictions["cmapss_penalty"].sum()),
        "cmapss_score_mean": float(predictions["cmapss_penalty"].mean()),
        "dangerous_error_pct": float(predictions["dangerous_error"].mean() * 100.0),
        "conservative_error_pct": float(predictions["conservative_error"].mean() * 100.0),
        "bias_mean": float(predictions["error"].mean()),
        "abs_error_p90": float(predictions["abs_error"].quantile(0.90)),
        "abs_error_p95": float(predictions["abs_error"].quantile(0.95)),
    }


def evaluate_fd003_config_split(config, data_dir, random_state, drop_columns=None):
    prepared = make_fd003_temporal_dataset(
        data_dir=data_dir,
        random_state=random_state,
        window_size=int(config["window_size"]),
        rul_cap=int(config["rul_cap"]),
        feature_set=config.get("feature_set", "base"),
        drop_columns=drop_columns,
    )
    model = make_lgbm_regressor_from_config(config, random_state=random_state)
    weights = sample_weights_for_scheme(prepared["y_train_raw"], config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(prepared["X_train"], prepared["y_train"])
    else:
        model.fit(prepared["X_train"], prepared["y_train"], sample_weight=weights)
    predictions = normalized_predictions_from_prepared(
        prepared,
        model.predict(prepared["X_eval"]),
        config,
        random_state,
    )
    metrics = metrics_from_predictions(predictions)
    metrics.update(
        {
            "model_name": config.get("model_name", ""),
            "approach": config.get("approach", ""),
            "window_size": config["window_size"],
            "rul_cap": config["rul_cap"],
            "objective": config["objective"],
            "alpha": config.get("alpha"),
            "sample_weight_scheme": config.get("sample_weight_scheme", "none"),
            "feature_set": config.get("feature_set", "base"),
            "random_state": random_state,
            "n_features": len(prepared["feature_columns"]),
            "dropped_columns": ",".join(prepared["dropped_columns"]),
        }
    )
    return metrics, predictions, prepared


def summarize_detail(detail, group_cols):
    metric_cols = [
        "mae",
        "rmse",
        "r2",
        "cmapss_score",
        "dangerous_error_pct",
        "conservative_error_pct",
        "bias_mean",
        "abs_error_p90",
        "abs_error_p95",
    ]
    grouped = detail.groupby(group_cols, dropna=False)
    summary = grouped[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    worst = grouped.agg(
        worst_rmse=("rmse", "max"),
        worst_cmapss_score=("cmapss_score", "max"),
        worst_dangerous_error_pct=("dangerous_error_pct", "max"),
    ).reset_index()
    result = summary.merge(worst, on=group_cols, how="left")
    rename = {}
    for metric in metric_cols:
        rename[f"{metric}_mean"] = f"mean_{metric}"
        rename[f"{metric}_std"] = f"std_{metric}"
    return result.rename(columns=rename).reset_index(drop=True)


def select_best_summary_row(summary):
    return (
        summary.sort_values(
            ["mean_cmapss_score", "mean_dangerous_error_pct", "mean_rmse", "std_cmapss_score"],
            ascending=[True, True, True, True],
        )
        .iloc[0]
        .to_dict()
    )


def fd003_short_tuning_configs():
    # Stratified short search: cover each requested axis without running the
    # full Cartesian grid, which is too large for this stage.
    candidates = []
    for window_size in [30, 50, 70]:
        candidates.append((window_size, 125, "quantile", 0.4, "none"))
        candidates.append((window_size, 125, "quantile", 0.4, "low_rul_weight"))
        candidates.append((window_size, 125, "regression", None, "none"))
    for rul_cap in [100, 125, 150]:
        candidates.append((50, rul_cap, "quantile", 0.4, "none"))
        candidates.append((50, rul_cap, "quantile", 0.4, "low_rul_weight"))
        candidates.append((50, rul_cap, "regression", None, "none"))
    for alpha in [0.35, 0.4, 0.5]:
        candidates.append((50, 125, "quantile", alpha, "none"))
        candidates.append((50, 125, "quantile", alpha, "low_rul_weight"))
    for scheme in ["none", "low_rul_weight"]:
        candidates.append((50, 125, "regression", None, scheme))

    unique = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)

    configs = []
    for window_size, rul_cap, objective, alpha, scheme in unique:
        configs.append(
            {
                "model_name": (
                    f"fd003_lgbm_w{window_size}_cap{rul_cap}_{objective}"
                    f"{'' if alpha is None else '_a' + str(alpha).replace('.', '')}_{scheme}"
                ),
                "approach": "short_tuning",
                "model_family": "LightGBM",
                "window_size": window_size,
                "rul_cap": rul_cap,
                "objective": objective,
                "alpha": alpha,
                "sample_weight_scheme": scheme,
                "feature_set": "base",
                "hyperparameters": FD001_FINAL_LGBM_PARAMS,
                "low_rul_weight_formula": "RUL<=30:2.0, 30<RUL<=60:1.5, 60<RUL<=100:1.2, else:1.0",
            }
        )
    return configs


def fd003_lgbm_hyperparam_refinement_configs(base_config, n_configs=24, random_state=42):
    from itertools import product

    fixed = {
        "learning_rate": 0.03,
        "n_estimators": 1300,
        "max_depth": -1,
    }
    base_params = dict(base_config.get("hyperparameters", FD001_FINAL_LGBM_PARAMS))
    base_params.update(fixed)

    keys = list(FD003_LGBM_REFINEMENT_GRID)
    all_params = []
    for values in product(*(FD003_LGBM_REFINEMENT_GRID[key] for key in keys)):
        params = dict(zip(keys, values))
        params.update(fixed)
        all_params.append(params)

    def param_signature(params):
        return tuple((key, params[key]) for key in sorted(params))

    baseline_signature = param_signature(base_params)
    candidates = [params for params in all_params if param_signature(params) != baseline_signature]
    rng = np.random.default_rng(random_state)
    sample_size = max(0, min(n_configs - 1, len(candidates)))
    sampled_indices = rng.choice(len(candidates), size=sample_size, replace=False)

    selected_params = [base_params] + [candidates[int(index)] for index in sampled_indices]
    configs = []
    for idx, params in enumerate(selected_params):
        label = "baseline_current" if idx == 0 else f"search_{idx:02d}"
        config = dict(base_config)
        config["model_name"] = f"fd003_lgbm_refinement_{label}"
        config["approach"] = "lgbm_hyperparam_refinement"
        config["hyperparameters"] = params
        config["hyperparameter_search_label"] = label
        config["refinement_fixed_fields"] = {
            "window_size": config["window_size"],
            "rul_cap": config["rul_cap"],
            "objective": config["objective"],
            "alpha": config.get("alpha"),
            "sample_weight_scheme": config.get("sample_weight_scheme", "none"),
            "feature_set": config.get("feature_set", "fault_sensitive"),
        }
        configs.append(config)
    return configs


def is_defensible_fd003_refinement(best_row, current_row, min_cmapss_delta=5.0):
    cmapss_delta = current_row["mean_cmapss_score"] - best_row["mean_cmapss_score"]
    dangerous_delta = best_row["mean_dangerous_error_pct"] - current_row["mean_dangerous_error_pct"]
    worst_delta = best_row["worst_cmapss_score"] - current_row["worst_cmapss_score"]
    rmse_delta = best_row["mean_rmse"] - current_row["mean_rmse"]
    return bool(
        cmapss_delta >= min_cmapss_delta
        and dangerous_delta <= 0.5
        and worst_delta <= 0.0
        and rmse_delta <= 0.75
    )


def config_from_summary_row(row, hyperparameters=None):
    return {
        "model_name": row["model_name"],
        "approach": row.get("approach", ""),
        "model_family": "LightGBM",
        "window_size": int(row["window_size"]),
        "rul_cap": int(row["rul_cap"]),
        "objective": row["objective"],
        "alpha": None if pd.isna(row.get("alpha")) else float(row.get("alpha")),
        "sample_weight_scheme": row.get("sample_weight_scheme", "none"),
        "feature_set": row.get("feature_set", "base"),
        "hyperparameters": hyperparameters or FD001_FINAL_LGBM_PARAMS,
        "selected_by": "artificial_validation_multi_split",
    }


def add_rul_bins_to_predictions(predictions):
    result = predictions.copy()
    result["rul_bin"] = pd.cut(
        result["true_rul"],
        bins=[0, 31, 61, 101, np.inf],
        labels=["0-30", "31-60", "61-100", "101+"],
        right=False,
        include_lowest=True,
    )
    return result


def metrics_by_group(predictions, group_cols):
    rows = []
    for keys, group in predictions.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(metrics_from_predictions(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def save_json(path, data):
    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            if np.isnan(value):
                return None
            return float(value)
        if isinstance(value, np.ndarray):
            return convert(value.tolist())
        return value

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(convert(data), file, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def comparison_row_from_summary(row, approach, notes):
    return {
        "model_name": row.get("model_name", approach),
        "approach": approach,
        "window_size": row.get("window_size"),
        "rul_cap": row.get("rul_cap"),
        "objective": row.get("objective"),
        "alpha": row.get("alpha"),
        "sample_weight_scheme": row.get("sample_weight_scheme"),
        "mean_MAE": row.get("mean_mae"),
        "std_MAE": row.get("std_mae"),
        "mean_RMSE": row.get("mean_rmse"),
        "std_RMSE": row.get("std_rmse"),
        "mean_R2": row.get("mean_r2"),
        "std_R2": row.get("std_r2"),
        "mean_CMAPSS": row.get("mean_cmapss_score"),
        "std_CMAPSS": row.get("std_cmapss_score"),
        "mean_dangerous_error_pct": row.get("mean_dangerous_error_pct"),
        "mean_conservative_error_pct": row.get("mean_conservative_error_pct"),
        "mean_bias": row.get("mean_bias_mean"),
        "worst_RMSE": row.get("worst_rmse"),
        "worst_CMAPSS": row.get("worst_cmapss_score"),
        "notes": notes,
    }


def train_cluster_classifier_with_oof(X_train, units, cluster_labels, random_state):
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    groups = np.asarray(units)
    y = np.asarray(cluster_labels, dtype=int)
    proba = np.zeros((len(y), 2), dtype=float)
    accuracies = []
    unique_groups = np.unique(groups)
    n_splits = min(3, len(unique_groups))
    if n_splits >= 2:
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in splitter.split(X_train, y, groups):
            fold_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1,
            )
            fold_model.fit(X_train.iloc[train_idx], y[train_idx])
            fold_proba = predict_cluster_proba_two_columns(fold_model, X_train.iloc[val_idx])
            proba[val_idx] = fold_proba
            accuracies.append(float(accuracy_score(y[val_idx], fold_model.predict(X_train.iloc[val_idx]))))
    else:
        proba[:] = 0.5
    classifier.fit(X_train, y)
    return classifier, proba, float(np.mean(accuracies)) if accuracies else np.nan


def predict_cluster_proba_two_columns(classifier, X):
    raw = classifier.predict_proba(X)
    proba = np.zeros((len(X), 2), dtype=float)
    for idx, klass in enumerate(classifier.classes_):
        if int(klass) in (0, 1):
            proba[:, int(klass)] = raw[:, idx]
    row_sums = proba.sum(axis=1)
    empty = row_sums == 0
    if empty.any():
        proba[empty] = 0.5
        row_sums = proba.sum(axis=1)
    proba = proba / row_sums[:, None]
    return proba


def fit_split_train_clusters(train_raw, train_units, random_state):
    train_subset = train_raw.loc[train_raw["unit"].isin(train_units)].copy()
    cluster_features = build_fd003_unit_cluster_features(train_subset)
    feature_cols = [
        column
        for column in cluster_features.columns
        if column not in {"unit_number", "total_life", "n_cycles", "cluster_sensors"}
    ]
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_features[feature_cols])
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=20)
    cluster_features["cluster_id_train"] = kmeans.fit_predict(X)
    return cluster_features[["unit_number", "cluster_id_train"]]


def _fit_lgbm_predict(config, X_train, y_train, y_train_raw, X_eval, random_state):
    model = make_lgbm_regressor_from_config(config, random_state=random_state)
    weights = sample_weights_for_scheme(y_train_raw, config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=weights)
    return model.predict(X_eval)


def _with_cluster_probability_columns(X, proba):
    result = X.copy()
    result["cluster_prob_0"] = proba[:, 0]
    result["cluster_prob_1"] = proba[:, 1]
    return result


def _pseudo_variant_config(base_config, variant, approach):
    config = dict(base_config)
    config["model_name"] = f"{base_config.get('model_name', 'fd003_lgbm')}_{variant}"
    config["approach"] = approach
    return config


def evaluate_fd003_pseudo_cluster_split(config, data_dir, random_state, drop_columns=None):
    prepared = make_fd003_temporal_dataset(
        data_dir=data_dir,
        random_state=random_state,
        window_size=int(config["window_size"]),
        rul_cap=int(config["rul_cap"]),
        feature_set=config.get("feature_set", "base"),
        drop_columns=drop_columns,
    )

    train_clusters = fit_split_train_clusters(
        prepared["train_raw"],
        prepared["train_units"],
        random_state=random_state,
    )
    cluster_map = train_clusters.set_index("unit_number")["cluster_id_train"]
    train_cluster_labels = prepared["train_df"]["unit"].map(cluster_map).to_numpy(dtype=int)

    classifier, train_oof_proba, classifier_oof_accuracy = train_cluster_classifier_with_oof(
        prepared["X_train"],
        prepared["train_df"]["unit"].to_numpy(dtype=int),
        train_cluster_labels,
        random_state=random_state,
    )
    eval_proba = predict_cluster_proba_two_columns(classifier, prepared["X_eval"])

    prediction_tables = []
    metric_rows = []

    base_variant_config = _pseudo_variant_config(config, "global_baseline", "pseudo_cluster_baseline")
    base_pred = _fit_lgbm_predict(
        config,
        prepared["X_train"],
        prepared["y_train"],
        prepared["y_train_raw"],
        prepared["X_eval"],
        random_state=random_state,
    )
    base_predictions = normalized_predictions_from_prepared(
        prepared,
        base_pred,
        base_variant_config,
        random_state,
        variant="global_baseline",
    )
    base_predictions["cluster_prob_0"] = eval_proba[:, 0]
    base_predictions["cluster_prob_1"] = eval_proba[:, 1]
    prediction_tables.append(base_predictions)

    metrics = metrics_from_predictions(base_predictions)
    metrics.update(
        {
            "random_state": random_state,
            "variant": "global_baseline",
            "model_name": base_variant_config["model_name"],
            "approach": base_variant_config["approach"],
            "window_size": config["window_size"],
            "rul_cap": config["rul_cap"],
            "objective": config["objective"],
            "alpha": config.get("alpha"),
            "sample_weight_scheme": config.get("sample_weight_scheme", "none"),
            "feature_set": config.get("feature_set", "base"),
            "n_features": len(prepared["feature_columns"]),
            "uses_predicted_cluster_probabilities": False,
        }
    )
    metric_rows.append(metrics)

    prob_variant_config = _pseudo_variant_config(config, "global_with_cluster_probs", "pseudo_cluster_prob_features")
    X_train_prob = _with_cluster_probability_columns(prepared["X_train"], train_oof_proba)
    X_eval_prob = _with_cluster_probability_columns(prepared["X_eval"], eval_proba)
    prob_pred = _fit_lgbm_predict(
        config,
        X_train_prob,
        prepared["y_train"],
        prepared["y_train_raw"],
        X_eval_prob,
        random_state=random_state,
    )
    prob_predictions = normalized_predictions_from_prepared(
        prepared,
        prob_pred,
        prob_variant_config,
        random_state,
        variant="global_with_cluster_probs",
    )
    prob_predictions["cluster_prob_0"] = eval_proba[:, 0]
    prob_predictions["cluster_prob_1"] = eval_proba[:, 1]
    prediction_tables.append(prob_predictions)

    metrics = metrics_from_predictions(prob_predictions)
    metrics.update(
        {
            "random_state": random_state,
            "variant": "global_with_cluster_probs",
            "model_name": prob_variant_config["model_name"],
            "approach": prob_variant_config["approach"],
            "window_size": config["window_size"],
            "rul_cap": config["rul_cap"],
            "objective": config["objective"],
            "alpha": config.get("alpha"),
            "sample_weight_scheme": config.get("sample_weight_scheme", "none"),
            "feature_set": config.get("feature_set", "base"),
            "n_features": len(prepared["feature_columns"]) + 2,
            "uses_predicted_cluster_probabilities": True,
        }
    )
    metric_rows.append(metrics)

    mixture_variant_config = _pseudo_variant_config(config, "mixture_of_experts_soft", "pseudo_cluster_soft_mixture")
    expert_predictions = []
    mixture_fallback = False
    for cluster_id in [0, 1]:
        mask = train_cluster_labels == cluster_id
        n_units = prepared["train_df"].loc[mask, "unit"].nunique()
        if mask.sum() < 50 or n_units < 5:
            mixture_fallback = True
            expert_predictions.append(base_pred)
            continue
        expert_pred = _fit_lgbm_predict(
            config,
            prepared["X_train"].loc[mask],
            prepared["y_train"].loc[mask],
            prepared["y_train_raw"].loc[mask],
            prepared["X_eval"],
            random_state=random_state + cluster_id + 11,
        )
        expert_predictions.append(expert_pred)
    mixture_pred = eval_proba[:, 0] * expert_predictions[0] + eval_proba[:, 1] * expert_predictions[1]
    mixture_predictions = normalized_predictions_from_prepared(
        prepared,
        mixture_pred,
        mixture_variant_config,
        random_state,
        variant="mixture_of_experts_soft",
    )
    mixture_predictions["cluster_prob_0"] = eval_proba[:, 0]
    mixture_predictions["cluster_prob_1"] = eval_proba[:, 1]
    prediction_tables.append(mixture_predictions)

    metrics = metrics_from_predictions(mixture_predictions)
    metrics.update(
        {
            "random_state": random_state,
            "variant": "mixture_of_experts_soft",
            "model_name": mixture_variant_config["model_name"],
            "approach": mixture_variant_config["approach"],
            "window_size": config["window_size"],
            "rul_cap": config["rul_cap"],
            "objective": config["objective"],
            "alpha": config.get("alpha"),
            "sample_weight_scheme": config.get("sample_weight_scheme", "none"),
            "feature_set": config.get("feature_set", "base"),
            "n_features": len(prepared["feature_columns"]),
            "uses_predicted_cluster_probabilities": True,
            "mixture_fallback": mixture_fallback,
        }
    )
    metric_rows.append(metrics)

    diagnostics = {
        "random_state": random_state,
        "cluster_classifier_oof_accuracy": classifier_oof_accuracy,
        "n_train_units_cluster_0": int((train_clusters["cluster_id_train"] == 0).sum()),
        "n_train_units_cluster_1": int((train_clusters["cluster_id_train"] == 1).sum()),
        "n_train_rows_cluster_0": int((train_cluster_labels == 0).sum()),
        "n_train_rows_cluster_1": int((train_cluster_labels == 1).sum()),
        "mean_eval_cluster_prob_0": float(eval_proba[:, 0].mean()),
        "mean_eval_cluster_prob_1": float(eval_proba[:, 1].mean()),
        "mixture_fallback": bool(mixture_fallback),
    }
    return pd.DataFrame(metric_rows), pd.concat(prediction_tables, ignore_index=True), diagnostics


def run_fd003_pseudo_cluster_experiments(config, data_dir, random_states=FD003_RANDOM_STATES, drop_columns=None):
    detail_tables = []
    prediction_tables = []
    diagnostics_rows = []
    for state in random_states:
        detail, predictions, diagnostics = evaluate_fd003_pseudo_cluster_split(
            config=config,
            data_dir=data_dir,
            random_state=state,
            drop_columns=drop_columns,
        )
        detail_tables.append(detail)
        prediction_tables.append(predictions)
        diagnostics_rows.append(diagnostics)
    return (
        pd.concat(detail_tables, ignore_index=True),
        pd.concat(prediction_tables, ignore_index=True),
        pd.DataFrame(diagnostics_rows),
    )
