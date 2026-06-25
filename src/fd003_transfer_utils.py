from pathlib import Path
import json

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import FEATURE_COLUMNS, add_train_rul, read_cmapss_file
from src.data_splitting import split_units
from src.fd001_modeling import metrics_by_model, prediction_frame
from src.preprocessed_FD001 import (
    DEFAULT_CUT_RULS,
    DEFAULT_RUL_CAP,
    make_fd001_artificial_cutoffs,
    make_temporal_features,
    temporal_feature_columns,
)


def load_fd003_train(data_dir="CMAPSSData"):
    return read_cmapss_file(Path(data_dir) / "train_FD003.txt")


def fd003_zero_variance_feature_columns(train_df, feature_columns=None):
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    variances = train_df[feature_columns].var(axis=0, ddof=0)
    return variances.loc[variances == 0].index.tolist()


def fd003_feature_columns(train_df, drop_columns=None):
    drop_columns = fd003_zero_variance_feature_columns(train_df) if drop_columns is None else list(drop_columns)
    drop_set = set(drop_columns)
    return [column for column in FEATURE_COLUMNS if column not in drop_set], drop_columns


def fd003_train_overview(train_df):
    train_cycles = train_df.groupby("unit")["cycle"].agg(["min", "max", "count"])
    return {
        "train_shape": train_df.shape,
        "train_units": int(train_df["unit"].nunique()),
        "train_nulls": int(train_df.isna().sum().sum()),
        "train_cycle_min": int(train_cycles["max"].min()),
        "train_cycle_median": float(train_cycles["max"].median()),
        "train_cycle_max": int(train_cycles["max"].max()),
    }


def prepare_fd003_temporal_validation_only(
    data_dir="CMAPSSData",
    eval_size=0.2,
    random_state=42,
    max_rul=DEFAULT_RUL_CAP,
    cut_ruls=DEFAULT_CUT_RULS,
    window_size=50,
    drop_columns=None,
):
    train_raw = load_fd003_train(data_dir=data_dir)
    train = add_train_rul(train_raw, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    train["RUL_capped"] = train["RUL"].clip(upper=max_rul) if max_rul is not None else train["RUL"]
    train["RUL"] = train["RUL_capped"]

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
        max_rul=max_rul,
    )

    train_temporal_df = make_temporal_features(
        train_source_df,
        endpoints_df=train_source_df,
        feature_columns=base_feature_columns,
        window_size=window_size,
    )
    eval_temporal_df = make_temporal_features(
        eval_source_df,
        endpoints_df=eval_cutoff_df,
        feature_columns=base_feature_columns,
        window_size=window_size,
    )
    temporal_columns = temporal_feature_columns(base_feature_columns)

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train_temporal_df[temporal_columns]),
        columns=temporal_columns,
        index=train_temporal_df.index,
    )
    X_eval = pd.DataFrame(
        scaler.transform(eval_temporal_df[temporal_columns]),
        columns=temporal_columns,
        index=eval_temporal_df.index,
    )

    return {
        "train_raw": train_raw,
        "feature_columns": temporal_columns,
        "base_feature_columns": base_feature_columns,
        "dropped_columns": dropped_columns,
        "train_units": train_units,
        "eval_units": eval_units,
        "scaler": scaler,
        "train_df": train_temporal_df,
        "train_source_df": train_source_df,
        "eval_source_df": eval_source_df,
        "eval_df": eval_temporal_df,
        "eval_cutoff_df": eval_cutoff_df,
        "X_train": X_train,
        "y_train": train_temporal_df["RUL"].copy(),
        "y_train_raw": train_temporal_df["RUL_raw"].copy(),
        "X_eval": X_eval,
        "y_eval": eval_temporal_df["RUL"].copy(),
        "y_eval_raw": eval_temporal_df["RUL_raw"].copy(),
    }


def load_fd001_final_quantile_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def make_lgbm_from_fd001_config(config, random_state=42):
    from lightgbm import LGBMRegressor

    params = dict(config["hyperparameters"])
    params.update(
        {
            "objective": config["objective"],
            "alpha": config["alpha"],
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    return LGBMRegressor(**params)


def evaluate_fd003_transfer_split(prepared, config, random_state=42):
    model = make_lgbm_from_fd001_config(config, random_state=random_state)
    model.fit(prepared["X_train"], prepared["y_train"])
    predictions = prediction_frame(
        prepared["eval_df"],
        model.predict(prepared["X_eval"]),
        model_name=config["model_name"],
        representation=f"temporal_w{config['window_size']}",
    )
    predictions["dataset"] = "FD003"
    predictions["random_state"] = random_state
    predictions["sample_weight_scheme"] = config["sample_weight_scheme"]
    predictions["objective"] = config["objective"]
    predictions["alpha"] = config["alpha"]
    predictions["window_size"] = config["window_size"]
    predictions["rul_cap"] = config["rul_cap"]

    row = metrics_by_model(predictions).iloc[0].to_dict()
    row.update(
        {
            "dataset": "FD003",
            "model_name": config["model_name"],
            "model_family": config["model_family"],
            "objective": config["objective"],
            "alpha": config["alpha"],
            "sample_weight_scheme": config["sample_weight_scheme"],
            "window_size": config["window_size"],
            "rul_cap": config["rul_cap"],
            "random_state": random_state,
            "n_train_units": len(prepared["train_units"]),
            "n_eval_units": len(prepared["eval_units"]),
            "n_features": len(prepared["feature_columns"]),
            "dropped_columns": ",".join(prepared["dropped_columns"]),
            "conservative_error_pct": float(predictions["conservative_error"].mean() * 100.0),
            "bias_mean": float(predictions["error"].mean()),
        }
    )
    return row, predictions


def summarize_fd003_transfer(detail):
    group_cols = ["dataset", "model_name", "model_family", "objective", "alpha", "sample_weight_scheme", "window_size", "rul_cap"]
    metrics = [
        "mae",
        "rmse",
        "r2",
        "cmapss_score",
        "dangerous_error_pct",
        "conservative_error_pct",
        "bias_mean",
    ]
    grouped = detail.groupby(group_cols, dropna=False)
    summary = grouped[metrics].agg(["mean", "std"]).reset_index()
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
    for metric in metrics:
        rename[f"{metric}_mean"] = f"mean_{metric}"
        rename[f"{metric}_std"] = f"std_{metric}"
    return result.rename(columns=rename).reset_index(drop=True)


def fd001_config_to_comparison_row(config):
    summary = config["validation_summary"]
    return {
        "dataset": "FD001",
        "model": config["model_name"],
        "window": config["window_size"],
        "cap": config["rul_cap"],
        "objective": config["objective"],
        "alpha": config["alpha"],
        "mean_mae": summary["mean_mae"],
        "mean_rmse": summary["mean_rmse"],
        "mean_r2": summary["mean_r2"],
        "mean_cmapss_score": summary["mean_cmapss_score"],
        "mean_dangerous_error_pct": summary["mean_dangerous_error_pct"],
        "mean_conservative_error_pct": summary["mean_conservative_error_pct"],
        "mean_bias": summary["mean_bias_mean"],
        "worst_rmse": summary["worst_rmse"],
        "worst_cmapss_score": summary["worst_cmapss_score"],
    }


def fd003_summary_to_comparison_row(summary):
    row = summary.iloc[0]
    return {
        "dataset": "FD003",
        "model": row["model_name"],
        "window": row["window_size"],
        "cap": row["rul_cap"],
        "objective": row["objective"],
        "alpha": row["alpha"],
        "mean_mae": row["mean_mae"],
        "mean_rmse": row["mean_rmse"],
        "mean_r2": row["mean_r2"],
        "mean_cmapss_score": row["mean_cmapss_score"],
        "mean_dangerous_error_pct": row["mean_dangerous_error_pct"],
        "mean_conservative_error_pct": row["mean_conservative_error_pct"],
        "mean_bias": row["mean_bias_mean"],
        "worst_rmse": row["worst_rmse"],
        "worst_cmapss_score": row["worst_cmapss_score"],
    }
