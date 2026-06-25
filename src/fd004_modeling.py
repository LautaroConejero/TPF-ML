from __future__ import annotations

import argparse
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

import pandas as pd

from src.data import add_train_rul, last_cycle_rows, load_cmapss_subset
from src.data_splitting import split_units
from src.fd001_modeling import (
    metrics_by_model,
    metrics_by_rul_bin,
    plot_validation_diagnostics,
    prediction_frame,
)
from src.fd002_modeling import (
    CONDITION_COUNT,
    DEFAULT_CUT_RULS,
    DEFAULT_RUL_CAP,
    DEFAULT_WINDOW_SIZE,
    add_fd002_condition_features,
    base_columns_for_feature_set,
    fd001_fd003_lgbm_reference_params,
    fd002_weights_from_scheme,
    fit_condition_preprocessor,
    make_fd002_fault_sensitive_features,
    make_model,
    make_temporal_features_fast,
    metric_row,
    resolve_data_dir,
    scale_temporal_frames,
    selection_sort,
    temporal_columns_for_base,
    write_json,
)
from src.preprocessed_FD001 import make_fd001_artificial_cutoffs


FD004_FAULT_SENSITIVE_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_14",
    "sensor_15",
    "sensor_17",
]


def fd004_output_paths(project_root=PROJECT_ROOT):
    project_root = Path(project_root)
    paths = {
        "results": project_root / "results" / "FD004",
        "configs": project_root / "configs" / "FD004",
        "figures": project_root / "figures" / "FD004",
        "predictions": project_root / "predictions",
        "notes": project_root / "notas" / "hallazgos" / "FD004",
        "checkpoints": project_root / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def fd004_train_with_rul(data_dir="CMAPSSData", rul_cap=DEFAULT_RUL_CAP):
    data = load_cmapss_subset("FD004", data_dir=resolve_data_dir(data_dir))
    train = add_train_rul(data.train, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    if rul_cap is not None:
        train["RUL"] = train["RUL"].clip(upper=rul_cap)
    return train


def add_fd004_extra_features_for_feature_set(
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
        sensors=FD004_FAULT_SENSITIVE_SENSORS,
        window_size=window_size,
    )
    result = temporal_df.merge(extra_df, on=["unit", "cycle"], how="left")
    result[extra_columns] = result[extra_columns].fillna(0.0)
    return result, extra_columns


def prepare_fd004_temporal_validation(
    data_dir="CMAPSSData",
    eval_size=0.2,
    random_state=42,
    rul_cap=DEFAULT_RUL_CAP,
    cut_ruls=DEFAULT_CUT_RULS,
    window_size=DEFAULT_WINDOW_SIZE,
    feature_set="condition_fault_sensitive",
):
    train = fd004_train_with_rul(data_dir=data_dir, rul_cap=rul_cap)
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
    train_temporal, train_extra_columns = add_fd004_extra_features_for_feature_set(
        train_temporal,
        train_aug,
        train_aug,
        feature_set,
        window_size=window_size,
    )
    eval_temporal, eval_extra_columns = add_fd004_extra_features_for_feature_set(
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
        "dataset": "FD004",
        "feature_set": feature_set,
        "base_feature_columns": base_columns,
        "feature_columns": feature_columns,
        "extra_feature_columns": extra_columns,
        "fault_sensitive_sensors": list(FD004_FAULT_SENSITIVE_SENSORS)
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


def prepare_fd004_temporal_full_train_for_test(
    data_dir="CMAPSSData",
    rul_cap=DEFAULT_RUL_CAP,
    window_size=DEFAULT_WINDOW_SIZE,
    feature_set="condition_fault_sensitive",
    random_state=42,
):
    data = load_cmapss_subset("FD004", data_dir=resolve_data_dir(data_dir))
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
    train_temporal, train_extra_columns = add_fd004_extra_features_for_feature_set(
        train_temporal,
        train_aug,
        train_aug,
        feature_set,
        window_size=window_size,
    )
    test_temporal, test_extra_columns = add_fd004_extra_features_for_feature_set(
        test_temporal,
        test_source_aug,
        test_last_aug,
        feature_set,
        window_size=window_size,
    )
    extra_columns = train_extra_columns or test_extra_columns
    feature_columns = temporal_columns + extra_columns
    scaler, x_train, x_test = scale_temporal_frames(train_temporal, test_temporal, feature_columns)

    return {
        "dataset": "FD004",
        "feature_set": feature_set,
        "base_feature_columns": base_columns,
        "feature_columns": feature_columns,
        "extra_feature_columns": extra_columns,
        "fault_sensitive_sensors": list(FD004_FAULT_SENSITIVE_SENSORS)
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


def fit_predict_config(prepared, config, random_state=42):
    model = make_model(config, random_state=random_state)
    weights = fd002_weights_from_scheme(prepared["y_train_raw"], config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(prepared["X_train"], prepared["y_train"])
    else:
        model.fit(prepared["X_train"], prepared["y_train"], sample_weight=weights)
    predictions = prediction_frame(
        prepared["eval_df"],
        model.predict(prepared["X_eval"]),
        model_name=config["candidate_label"],
        representation=config["representation"],
    )
    predictions["dataset"] = "FD004"
    predictions["feature_set"] = prepared["feature_set"]
    predictions["window_size"] = prepared["window_size"]
    predictions["rul_cap"] = prepared["rul_cap"]
    predictions["sample_weight_scheme"] = config.get("sample_weight_scheme", "none")
    predictions["model_type"] = config["model_type"]
    return model, predictions


def evaluate_configs(configs, data_dir="CMAPSSData", random_state=42, rul_cap=DEFAULT_RUL_CAP):
    prepared_cache = {}
    metric_rows = []
    prediction_tables = []
    fitted_models = {}

    for config in configs:
        cache_key = (config["feature_set"], int(config["window_size"]), int(random_state))
        if cache_key not in prepared_cache:
            prepared_cache[cache_key] = prepare_fd004_temporal_validation(
                data_dir=data_dir,
                random_state=random_state,
                rul_cap=rul_cap,
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


def summarize_multi_split(detail):
    group_cols = [
        "candidate_label",
        "model_type",
        "feature_set",
        "window_size",
        "rul_cap",
        "sample_weight_scheme",
        "n_features",
        "params",
        "selection_note",
    ]
    metric_cols = ["mae", "rmse", "r2", "cmapss_score", "dangerous_error_pct"]
    grouped = detail.groupby(group_cols, dropna=False)
    summary = grouped[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    worst = grouped.agg(
        worst_cmapss_score=("cmapss_score", "max"),
        worst_rmse=("rmse", "max"),
        worst_dangerous_error_pct=("dangerous_error_pct", "max"),
    ).reset_index()
    summary = summary.merge(worst, on=group_cols, how="left")
    rename = {}
    for metric in metric_cols:
        rename[f"{metric}_mean"] = f"mean_{metric}"
        rename[f"{metric}_std"] = f"std_{metric}"
    summary = summary.rename(columns=rename)
    return summary.sort_values(
        ["mean_cmapss_score", "mean_dangerous_error_pct", "mean_rmse", "std_cmapss_score"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def evaluate_configs_multi_split(
    configs,
    data_dir="CMAPSSData",
    random_states=(0, 1, 2),
    rul_cap=DEFAULT_RUL_CAP,
):
    details = []
    predictions = []
    for state in random_states:
        metrics, preds, _ = evaluate_configs(
            configs,
            data_dir=data_dir,
            random_state=int(state),
            rul_cap=rul_cap,
        )
        metrics["eval_random_state"] = int(state)
        preds["eval_random_state"] = int(state)
        details.append(metrics)
        predictions.append(preds)
    detail = pd.concat(details, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    return detail, prediction_table, summarize_multi_split(detail)


def fd004_reference_configs():
    lgbm_reference = fd001_fd003_lgbm_reference_params()
    xgb_fd002_params = {
        "objective": "reg:squarederror",
        "n_estimators": 650,
        "max_depth": 3,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 5.0,
    }
    return [
        {
            "candidate_label": "fd003_lgbm_reference_raw",
            "model_type": "lightgbm",
            "feature_set": "raw",
            "window_size": 50,
            "representation": "temporal_w50_raw",
            "sample_weight_scheme": "none",
            "selection_note": "FD003 final recipe transferred to FD004 without explicit condition handling.",
            "params": lgbm_reference,
        },
        {
            "candidate_label": "fd003_lgbm_reference_condition_normalized",
            "model_type": "lightgbm",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized",
            "sample_weight_scheme": "none",
            "selection_note": "FD003 final recipe plus FD002-style condition normalization.",
            "params": lgbm_reference,
        },
        {
            "candidate_label": "fd003_lgbm_reference_condition_fault_sensitive",
            "model_type": "lightgbm",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive",
            "sample_weight_scheme": "none",
            "selection_note": "FD003 final recipe plus FD002 condition control and FD004 fault-sensitive features.",
            "params": lgbm_reference,
        },
        {
            "candidate_label": "fd002_xgb_condition_normalized_weighted",
            "model_type": "xgboost",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "FD002 XGBoost condition-normalized recipe transferred to FD004.",
            "params": xgb_fd002_params,
        },
        {
            "candidate_label": "fd002_xgb_condition_fault_sensitive_mid_guard",
            "model_type": "xgboost",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive_mid_guard",
            "sample_weight_scheme": "mid_rul_guard",
            "selection_note": "FD002 final XGBoost recipe transferred with FD004 fault-sensitive sensors.",
            "params": xgb_fd002_params,
        },
        {
            "candidate_label": "fd002_lgbm_quantile_condition_normalized_weighted",
            "model_type": "lightgbm",
            "feature_set": "condition_normalized",
            "window_size": 50,
            "representation": "temporal_w50_condition_normalized",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "FD002 LightGBM quantile recipe transferred to FD004.",
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
    ]


def fd004_hyperparam_search_configs():
    return [
        {
            "candidate_label": "fd004_lgbm_fs_quantile_a04_leaves15",
            "model_type": "lightgbm",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive",
            "sample_weight_scheme": "none",
            "selection_note": "FD003 final LightGBM shape over FD004 condition-fault-sensitive features.",
            "params": fd001_fd003_lgbm_reference_params(),
        },
        {
            "candidate_label": "fd004_lgbm_fs_quantile_a035_w50",
            "model_type": "lightgbm",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive",
            "sample_weight_scheme": "none",
            "selection_note": "Slightly more conservative LightGBM quantile objective for FD004.",
            "params": {
                "objective": "quantile",
                "alpha": 0.35,
                "learning_rate": 0.03,
                "n_estimators": 1100,
                "num_leaves": 15,
                "max_depth": -1,
                "min_child_samples": 10,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.5,
                "reg_lambda": 10.0,
            },
        },
        {
            "candidate_label": "fd004_lgbm_fs_quantile_a04_weighted",
            "model_type": "lightgbm",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive_weighted",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "FD004 fault-sensitive LightGBM with near-failure weights.",
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
            "candidate_label": "fd004_lgbm_fs_regression_mid_guard_soft",
            "model_type": "lightgbm",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive_mid_guard_soft",
            "sample_weight_scheme": "mid_rul_guard_soft",
            "selection_note": "LightGBM regression with moderated extra weight on 30-90 RUL.",
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
            "candidate_label": "fd004_xgb_fs_mid_guard_depth3",
            "model_type": "xgboost",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive_mid_guard",
            "sample_weight_scheme": "mid_rul_guard",
            "selection_note": "FD002 final XGBoost settings over FD004 fault-sensitive features.",
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
            "candidate_label": "fd004_xgb_fs_bin_weights_depth3",
            "model_type": "xgboost",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "FD004 XGBoost fault-sensitive model with standard near-failure weights.",
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
            "candidate_label": "fd004_xgb_fs_mid_guard_depth2_reg10",
            "model_type": "xgboost",
            "feature_set": "condition_fault_sensitive",
            "window_size": 50,
            "representation": "temporal_w50_condition_fault_sensitive_mid_guard",
            "sample_weight_scheme": "mid_rul_guard",
            "selection_note": "More regularized shallow XGBoost for FD004.",
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
            "candidate_label": "fd004_xgb_fs_bin_weights_w70",
            "model_type": "xgboost",
            "feature_set": "condition_fault_sensitive",
            "window_size": 70,
            "representation": "temporal_w70_condition_fault_sensitive",
            "sample_weight_scheme": "bin_weights",
            "selection_note": "Wider temporal context for FD004 XGBoost fault-sensitive model.",
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


def official_test_predictions(prepared, model, config):
    predictions = prediction_frame(
        prepared["test_last_df"],
        model.predict(prepared["X_test_last"]),
        model_name=config["candidate_label"],
        representation=config["representation"],
    )
    predictions["dataset"] = "FD004"
    predictions["feature_set"] = prepared["feature_set"]
    predictions["window_size"] = prepared["window_size"]
    predictions["rul_cap"] = prepared["rul_cap"]
    predictions["sample_weight_scheme"] = config.get("sample_weight_scheme", "none")
    predictions["model_type"] = config["model_type"]
    return predictions


def fit_final_model_and_predict(data_dir, best_config, random_state=42, rul_cap=DEFAULT_RUL_CAP):
    prepared = prepare_fd004_temporal_full_train_for_test(
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


def single_prediction_metrics(predictions, n_col="n_test"):
    row = metrics_by_model(predictions).iloc[0].to_dict()
    row[n_col] = row.pop("n_eval")
    return pd.DataFrame([row])


def build_best_model_config(
    best_config,
    validation_summary,
    validation_single_split_metrics,
    official_metrics,
    prepared,
    paths,
    random_states,
):
    return {
        "dataset": "FD004",
        "task": "Remaining Useful Life regression",
        "selection_policy": (
            "Candidates are selected by lowest mean C-MAPSS on artificial-cutoff validation over held-out "
            "engine units. Official FD004 test is used only after selection for final reporting."
        ),
        "problem_analysis": {
            "fd002_lesson": (
                "FD002 has six operating conditions and one fault mode; explicit condition clustering and "
                "sensor normalization by condition are required before temporal modeling."
            ),
            "fd003_lesson": (
                "FD003 has one operating condition and two degradation patterns; the best improvement came "
                "from fault-sensitive slopes, deltas, volatility and sensor interactions rather than cluster labels."
            ),
            "fd004_hypothesis": (
                "FD004 combines both difficulties, so the final representation controls operating condition first "
                "and then adds fault-sensitive temporal features on condition-normalized sensors."
            ),
        },
        "preprocessing": {
            "rul_raw_definition": "max_cycle - cycle for train; RUL_FD004.txt for official test last cycle",
            "training_target": "RUL capped",
            "metric_target": "RUL_raw uncapped",
            "rul_cap": int(prepared["rul_cap"]),
            "cut_ruls": list(DEFAULT_CUT_RULS),
            "condition_count": CONDITION_COUNT,
            "condition_method": "KMeans over settings fit only on the training split",
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
            "random_states": [int(state) for state in random_states],
        },
        "final_model": {
            "candidate_label": best_config["candidate_label"],
            "model_type": best_config["model_type"],
            "representation": best_config["representation"],
            "sample_weight_scheme": best_config.get("sample_weight_scheme", "none"),
            "hyperparameters": best_config["params"],
        },
        "validation_summary": validation_summary,
        "validation_single_split_metrics": validation_single_split_metrics,
        "official_test_metrics": official_metrics,
        "artifacts": {
            "model_family_comparison": str(paths["results"] / "fd004_model_family_comparison.csv"),
            "hyperparameter_search": str(paths["results"] / "fd004_hyperparam_search.csv"),
            "finalist_summary": str(paths["results"] / "fd004_finalist_multisplit_summary.csv"),
            "validation_predictions": str(paths["results"] / "fd004_best_validation_predictions.csv"),
            "official_test_predictions": str(paths["predictions"] / "fd004_best_model_predictions.csv"),
            "official_test_metrics": str(paths["results"] / "fd004_official_test_metrics.csv"),
        },
    }


def write_interpretation_note(path, comparison, search, finalist_summary, config_payload):
    best = finalist_summary.iloc[0]
    lines = [
        "FD004 - modelo final interno",
        "",
        "Lectura previa:",
        "- FD004 mezcla la dificultad de FD002 (seis condiciones operativas que cambian dentro del motor) con la de FD003 (dos patrones de degradacion sin etiqueta por unidad).",
        "- Por eso no conviene copiar FD002 o FD003 en crudo. Primero se controla la condicion operativa y despues se agregan features de degradacion calculadas hasta el ciclo observado.",
        "- Los clusters residuales del EDA no se usan como target ni feature directa para evitar leakage de trayectoria completa.",
        "",
        "Comparacion FD002/FD003 transferida:",
    ]
    for _, row in comparison.head(6).iterrows():
        lines.append(
            f"- {row['candidate_label']}: C-MAPSS {row['cmapss_score']:.3f}, RMSE {row['rmse']:.3f}, "
            f"dangerous {row['dangerous_error_pct']:.2f}%, feature_set={row['feature_set']}."
        )
    lines.extend(["", "Busqueda de hiperparametros acotada:"])
    for _, row in search.head(6).iterrows():
        lines.append(
            f"- {row['candidate_label']}: C-MAPSS {row['cmapss_score']:.3f}, RMSE {row['rmse']:.3f}, "
            f"dangerous {row['dangerous_error_pct']:.2f}%."
        )
    lines.extend(
        [
            "",
            "Seleccion multi-split:",
            (
                f"- {best['candidate_label']}: mean C-MAPSS {best['mean_cmapss_score']:.3f}, "
                f"mean RMSE {best['mean_rmse']:.3f}, mean dangerous {best['mean_dangerous_error_pct']:.2f}%, "
                f"worst C-MAPSS {best['worst_cmapss_score']:.3f}."
            ),
            "",
            "Modelo final:",
            f"- {config_payload['final_model']['candidate_label']}",
            f"- feature_set={config_payload['preprocessing']['feature_set']}, window_size={config_payload['preprocessing']['window_size']}, n_features={config_payload['preprocessing']['n_features']}.",
            "",
            "Artefactos:",
            "- src/fd004_modeling.py",
            "- results/FD004/fd004_model_family_comparison.csv",
            "- results/FD004/fd004_hyperparam_search.csv",
            "- results/FD004/fd004_finalist_multisplit_summary.csv",
            "- configs/FD004/fd004_best_model_config.json",
            "- predictions/fd004_best_model_predictions.csv",
            "- checkpoints/fd004_best_model.joblib",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fd004_modeling_workflow(
    project_root=PROJECT_ROOT,
    data_dir="CMAPSSData",
    random_state=42,
    finalist_random_states=(0, 1, 2),
    finalist_count=3,
):
    paths = fd004_output_paths(project_root)

    comparison_configs = fd004_reference_configs()
    comparison_metrics, comparison_predictions, _ = evaluate_configs(
        comparison_configs,
        data_dir=data_dir,
        random_state=random_state,
    )
    comparison_metrics.to_csv(paths["results"] / "fd004_model_family_comparison.csv", index=False)
    comparison_predictions.to_csv(paths["results"] / "fd004_model_family_comparison_predictions.csv", index=False)

    search_configs = fd004_hyperparam_search_configs()
    search_metrics, search_predictions, _ = evaluate_configs(
        search_configs,
        data_dir=data_dir,
        random_state=random_state,
    )
    search_metrics.to_csv(paths["results"] / "fd004_hyperparam_search.csv", index=False)
    search_predictions.to_csv(paths["results"] / "fd004_hyperparam_search_predictions.csv", index=False)

    all_configs = comparison_configs + search_configs
    config_by_label = {config["candidate_label"]: config for config in all_configs}
    initial_ranking = selection_sort(
        pd.concat(
            [
                comparison_metrics.assign(experiment_stage="reference_transfer"),
                search_metrics.assign(experiment_stage="hyperparam_search"),
            ],
            ignore_index=True,
        )
    )
    initial_ranking.to_csv(paths["results"] / "fd004_initial_candidate_ranking.csv", index=False)

    finalist_labels = list(dict.fromkeys(initial_ranking["candidate_label"].head(int(finalist_count)).tolist()))
    finalist_configs = [config_by_label[label] for label in finalist_labels]
    finalist_detail, finalist_predictions, finalist_summary = evaluate_configs_multi_split(
        finalist_configs,
        data_dir=data_dir,
        random_states=finalist_random_states,
    )
    finalist_detail.to_csv(paths["results"] / "fd004_finalist_multisplit_detail.csv", index=False)
    finalist_predictions.to_csv(paths["results"] / "fd004_finalist_multisplit_predictions.csv", index=False)
    finalist_summary.to_csv(paths["results"] / "fd004_finalist_multisplit_summary.csv", index=False)

    best_label = finalist_summary.iloc[0]["candidate_label"]
    best_config = config_by_label[best_label]

    validation_prepared = prepare_fd004_temporal_validation(
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
    validation_predictions.to_csv(paths["results"] / "fd004_best_validation_predictions.csv", index=False)
    validation_metrics.to_csv(paths["results"] / "fd004_best_validation_metrics.csv", index=False)
    validation_bin_metrics.to_csv(paths["results"] / "fd004_best_validation_metrics_by_rul_bin.csv", index=False)

    full_prepared, final_model, official_predictions = fit_final_model_and_predict(
        data_dir=data_dir,
        best_config=best_config,
        random_state=random_state,
    )
    official_metrics = single_prediction_metrics(official_predictions, n_col="n_test")
    official_bin_metrics = metrics_by_rul_bin(official_predictions)
    official_predictions.to_csv(paths["predictions"] / "fd004_best_model_predictions.csv", index=False)
    official_predictions.to_csv(paths["results"] / "fd004_official_test_predictions.csv", index=False)
    official_metrics.to_csv(paths["results"] / "fd004_official_test_metrics.csv", index=False)
    official_bin_metrics.to_csv(paths["results"] / "fd004_official_test_metrics_by_rul_bin.csv", index=False)

    plot_validation_diagnostics(
        validation_predictions,
        paths["figures"] / "validation_best_model",
        "FD004 best validation model",
    )

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
            paths["checkpoints"] / "fd004_best_model.joblib",
        )
    except Exception as exc:
        import warnings

        warnings.warn(
            f"Could not save FD004 checkpoint: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    config_payload = build_best_model_config(
        best_config=best_config,
        validation_summary=finalist_summary.iloc[0].to_dict(),
        validation_single_split_metrics=validation_metrics.iloc[0].to_dict(),
        official_metrics=official_metrics.iloc[0].to_dict(),
        prepared=full_prepared,
        paths=paths,
        random_states=finalist_random_states,
    )
    write_json(paths["configs"] / "fd004_best_model_config.json", config_payload)
    write_interpretation_note(
        paths["notes"] / "fd004_modeling_interpretation.txt",
        comparison_metrics,
        search_metrics,
        finalist_summary,
        config_payload,
    )

    return {
        "model_family_comparison": comparison_metrics,
        "hyperparam_search": search_metrics,
        "initial_ranking": initial_ranking,
        "finalist_detail": finalist_detail,
        "finalist_summary": finalist_summary,
        "validation_metrics": validation_metrics,
        "official_metrics": official_metrics,
        "best_config": config_payload,
        "paths": paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the FD004 RUL modeling workflow.")
    parser.add_argument("--data-dir", default="CMAPSSData")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--finalist-count", type=int, default=3)
    parser.add_argument("--finalist-random-states", default="0,1,2")
    args = parser.parse_args()
    finalist_random_states = tuple(
        int(value.strip()) for value in args.finalist_random_states.split(",") if value.strip()
    )
    result = run_fd004_modeling_workflow(
        data_dir=args.data_dir,
        random_state=args.random_state,
        finalist_random_states=finalist_random_states,
        finalist_count=args.finalist_count,
    )
    print(result["finalist_summary"].head(5).to_string(index=False))
    print(result["official_metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
