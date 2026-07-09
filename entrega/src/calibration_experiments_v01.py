from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import add_train_rul, load_cmapss_subset
from src.data_splitting import split_units
from src.fd002_modeling import add_fd002_condition_features, fit_condition_preprocessor
from src.preprocessed_FD001 import make_fd001_artificial_cutoffs


CREATED_AT = datetime.now().isoformat(timespec="seconds")
RUL_BINS = [0, 25, 50, 75, 100, 125, math.inf]
RUL_BIN_LABELS = ["0-25", "25-50", "50-75", "75-100", "100-125", "125+"]
THRESHOLDS = [70, 80, 90, 100, 110, 120]
OFFSETS = [0, 2, 4, 6, 8, 10, 12, 15, 20]
FD002_ENSEMBLE_WEIGHTS = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.4, 0.6)]
FD003_OFFSETS = [0, 1, 2, 3, 4]


def ensure_dirs() -> None:
    for path in [
        "notas/FD002",
        "notas/FD003",
        "notas/FD004",
        "configs/FD002",
        "configs/FD003",
        "configs/FD004",
        "results/FD002",
        "results/FD003",
        "results/FD004",
        "notebooks/FD002",
        "notebooks/FD003",
        "notebooks/FD004",
    ]:
        (PROJECT_ROOT / path).mkdir(parents=True, exist_ok=True)


def safe_write_text(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def selected_final_frame(selected_row: pd.Series, reason: str) -> pd.DataFrame:
    row = selected_row.to_dict()
    row["selected_final_candidate"] = True
    row["reason_selected"] = reason
    return pd.DataFrame([row])


def write_yaml(path: Path, payload: dict) -> None:
    def render(value, indent=0):
        prefix = " " * indent
        if isinstance(value, dict):
            lines = []
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(render(item, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {render(item, 0)}")
            return "\n".join(lines)
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    lines.append(render(item, indent + 2))
                else:
                    lines.append(f"{prefix}- {render(item, 0)}")
            return "\n".join(lines)
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (int, float)):
            return str(value)
        text = str(value).replace('"', '\\"')
        return f'"{text}"'

    if path.exists():
        return
    safe_write_text(path, render(payload) + "\n")


def cmapss_penalty(error: pd.Series | np.ndarray) -> np.ndarray:
    error = np.asarray(error, dtype=float)
    return np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)


def add_error_columns(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    out = df.copy()
    out["error"] = out[pred_col] - out["true_RUL"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    out["cmapss_penalty"] = cmapss_penalty(out["error"])
    out["dangerous_any"] = out["error"] > 0
    out["dangerous_10"] = out["error"] > 10
    out["dangerous_20"] = out["error"] > 20
    out["conservative"] = out["error"] < 0
    out["rul_bin_v01"] = pd.cut(
        out["true_RUL"],
        bins=RUL_BINS,
        labels=RUL_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    return out


def r2_score_manual(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom == 0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


def metrics_for_predictions(
    df: pd.DataFrame,
    dataset: str,
    candidate_name: str,
    model_type: str,
    calibration_type: str,
    config_file: str,
    notebook_file: str,
    predictions_file: str,
    metrics_file: str,
    selected: bool = False,
    reason: str = "",
) -> dict:
    scored = add_error_columns(df, "pred")
    high = scored["true_RUL"] > 125
    low = scored["true_RUL"] <= 50
    total_score = float(scored["cmapss_penalty"].sum())
    return {
        "dataset": dataset,
        "candidate_name": candidate_name,
        "model_type": model_type,
        "calibration_type": calibration_type,
        "config_file": config_file,
        "notebook_file": notebook_file,
        "predictions_file": predictions_file,
        "metrics_file": metrics_file,
        "n_motors": int(len(scored)),
        "MAE": float(scored["abs_error"].mean()),
        "RMSE": float(np.sqrt(scored["squared_error"].mean())),
        "R2": r2_score_manual(scored["true_RUL"], scored["pred"]),
        "CMAPSS_total": total_score,
        "CMAPSS_mean": float(scored["cmapss_penalty"].mean()),
        "dangerous_any_pct": float(scored["dangerous_any"].mean() * 100.0),
        "dangerous_10_pct": float(scored["dangerous_10"].mean() * 100.0),
        "dangerous_20_pct": float(scored["dangerous_20"].mean() * 100.0),
        "conservative_pct": float(scored["conservative"].mean() * 100.0),
        "bias": float(scored["error"].mean()),
        "MAE_RUL_le_50": float(scored.loc[low, "abs_error"].mean()) if low.any() else float("nan"),
        "MAE_RUL_gt_125": float(scored.loc[high, "abs_error"].mean()) if high.any() else float("nan"),
        "score_share_RUL_gt_125": float(scored.loc[high, "cmapss_penalty"].sum() / total_score) if total_score else float("nan"),
        "pred_max": float(scored["pred"].max()),
        "true_RUL_max": float(scored["true_RUL"].max()),
        "selected_final_candidate": bool(selected),
        "reason_selected": reason,
    }


def bin_metrics(df: pd.DataFrame, dataset: str, candidate_name: str) -> pd.DataFrame:
    scored = add_error_columns(df, "pred")
    rows = []
    total_score = float(scored["cmapss_penalty"].sum())
    for label, group in scored.groupby("rul_bin_v01", observed=False):
        if group.empty:
            continue
        rows.append(
            {
                "dataset": dataset,
                "candidate_name": candidate_name,
                "rul_bin": str(label),
                "n_motors": int(len(group)),
                "MAE": float(group["abs_error"].mean()),
                "RMSE": float(np.sqrt(group["squared_error"].mean())),
                "CMAPSS_total": float(group["cmapss_penalty"].sum()),
                "CMAPSS_mean": float(group["cmapss_penalty"].mean()),
                "CMAPSS_share": float(group["cmapss_penalty"].sum() / total_score) if total_score else float("nan"),
                "dangerous_any_pct": float(group["dangerous_any"].mean() * 100.0),
                "dangerous_10_pct": float(group["dangerous_10"].mean() * 100.0),
                "dangerous_20_pct": float(group["dangerous_20"].mean() * 100.0),
                "conservative_pct": float(group["conservative"].mean() * 100.0),
                "bias": float(group["error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def top10_cmapss(df: pd.DataFrame, dataset: str, candidate_name: str) -> pd.DataFrame:
    scored = add_error_columns(df, "pred")
    cols = ["unit_id", "true_RUL", "pred", "error", "abs_error", "cmapss_penalty"]
    for optional in ["condition_dominant", "cluster_id", "split_id"]:
        if optional in scored.columns:
            cols.append(optional)
    result = scored.sort_values("cmapss_penalty", ascending=False).head(10)[cols].copy()
    result.insert(0, "dataset", dataset)
    result.insert(1, "candidate_name", candidate_name)
    return result


def condition_metrics(df: pd.DataFrame, dataset: str, candidate_name: str) -> pd.DataFrame:
    if "condition_dominant" not in df.columns:
        return pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "candidate_name": candidate_name,
                    "condition_dominant": "not_available",
                    "note": "condition_dominant was not available for this prediction table",
                }
            ]
        )
    scored = add_error_columns(df, "pred")
    rows = []
    for condition, group in scored.groupby("condition_dominant", dropna=False):
        rows.append(
            {
                "dataset": dataset,
                "candidate_name": candidate_name,
                "condition_dominant": condition,
                "n_motors": int(len(group)),
                "MAE": float(group["abs_error"].mean()),
                "RMSE": float(np.sqrt(group["squared_error"].mean())),
                "R2": r2_score_manual(group["true_RUL"], group["pred"]),
                "CMAPSS_total": float(group["cmapss_penalty"].sum()),
                "CMAPSS_mean": float(group["cmapss_penalty"].mean()),
                "dangerous_any_pct": float(group["dangerous_any"].mean() * 100.0),
                "dangerous_10_pct": float(group["dangerous_10"].mean() * 100.0),
                "dangerous_20_pct": float(group["dangerous_20"].mean() * 100.0),
                "conservative_pct": float(group["conservative"].mean() * 100.0),
                "bias": float(group["error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def normalize_project_predictions(df: pd.DataFrame, candidate_name: str | None = None) -> pd.DataFrame:
    table = df.copy()
    if candidate_name is not None and "model_name" in table.columns:
        table = table.loc[table["model_name"] == candidate_name].copy()
    rename = {
        "unit": "unit_id",
        "unit_number": "unit_id",
        "y_true_rul_raw": "true_RUL",
        "true_rul": "true_RUL",
        "y_pred_rul": "pred",
        "pred_rul": "pred",
        "eval_random_state": "split_id",
        "split_seed": "split_id",
    }
    table = table.rename(columns={k: v for k, v in rename.items() if k in table.columns})
    required = ["unit_id", "true_RUL", "pred"]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(f"Missing columns after normalization: {missing}")
    if "cycle" not in table.columns and "cutoff_cycle" in table.columns:
        table["cycle"] = table["cutoff_cycle"]
    return table


def fd002_condition_lookup() -> pd.DataFrame:
    data = load_cmapss_subset("FD002", data_dir=PROJECT_ROOT / "CMAPSSData")
    train = add_train_rul(data.train)
    train_units, eval_units = split_units(train, random_state=42)
    train_source = train.loc[train["unit"].isin(train_units)].copy()
    eval_source = train.loc[train["unit"].isin(eval_units)].copy()
    eval_cutoffs = make_fd001_artificial_cutoffs(eval_source, cut_ruls=[20, 50, 80, 110, 140])
    condition_preprocessor = fit_condition_preprocessor(train_source, random_state=42)
    augmented = add_fd002_condition_features(eval_cutoffs, condition_preprocessor)
    return augmented[["unit", "cycle", "condition_id"]].rename(
        columns={"unit": "unit_id", "condition_id": "condition_dominant"}
    )


def fd004_condition_lookup(split_id: int) -> pd.DataFrame:
    data = load_cmapss_subset("FD004", data_dir=PROJECT_ROOT / "CMAPSSData")
    train = add_train_rul(data.train)
    train_units, eval_units = split_units(train, random_state=split_id)
    train_source = train.loc[train["unit"].isin(train_units)].copy()
    eval_source = train.loc[train["unit"].isin(eval_units)].copy()
    eval_cutoffs = make_fd001_artificial_cutoffs(eval_source, cut_ruls=[20, 50, 80, 110, 140])
    condition_preprocessor = fit_condition_preprocessor(train_source, random_state=split_id)
    augmented = add_fd002_condition_features(eval_cutoffs, condition_preprocessor)
    result = augmented[["unit", "cycle", "condition_id"]].rename(
        columns={"unit": "unit_id", "condition_id": "condition_dominant"}
    )
    result["split_id"] = split_id
    return result


def merge_conditions(predictions: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    keys = ["unit_id", "cycle"]
    if "split_id" in predictions.columns and "split_id" in lookup.columns:
        keys.append("split_id")
    out = predictions.merge(lookup, on=keys, how="left")
    return out


def piecewise_calibrate(df: pd.DataFrame, threshold: float, offset: float) -> pd.DataFrame:
    out = df.copy()
    out["pred"] = np.where(out["pred"] > threshold, out["pred"] + offset, out["pred"])
    return out


def prediction_export(
    baseline: pd.DataFrame,
    calibrated: pd.DataFrame,
    dataset: str,
    candidate_name: str,
    config_id: str,
) -> pd.DataFrame:
    left = baseline.copy().rename(columns={"pred": "pred_baseline"})
    right = calibrated.copy().rename(columns={"pred": "pred_calibrated"})
    keys = ["unit_id"]
    for optional in ["cycle", "split_id"]:
        if optional in left.columns and optional in right.columns:
            keys.append(optional)
    cols_right = keys + ["pred_calibrated"]
    merged = left.merge(right[cols_right], on=keys, how="left")
    merged["true_RUL"] = merged["true_RUL"].astype(float)
    merged["error_baseline"] = merged["pred_baseline"] - merged["true_RUL"]
    merged["error_calibrated"] = merged["pred_calibrated"] - merged["true_RUL"]
    merged["cmapss_baseline"] = cmapss_penalty(merged["error_baseline"])
    merged["cmapss_calibrated"] = cmapss_penalty(merged["error_calibrated"])
    merged["candidate_name"] = candidate_name
    merged["config_id"] = config_id
    merged["dataset"] = dataset
    keep = [
        "dataset",
        "unit_id",
        "cycle",
        "true_RUL",
        "pred_baseline",
        "pred_calibrated",
        "error_baseline",
        "error_calibrated",
        "cmapss_baseline",
        "cmapss_calibrated",
        "condition_dominant",
        "cluster_id",
        "split_id",
        "candidate_name",
        "config_id",
    ]
    for col in keep:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged[keep]


def select_fd_candidate(metrics: pd.DataFrame, baseline_name: str) -> tuple[pd.Series, str]:
    baseline = metrics.loc[metrics["candidate_name"] == baseline_name].iloc[0]
    eligible = metrics.loc[
        (metrics["RMSE"] <= baseline["RMSE"] + 1e-9)
        & (metrics["dangerous_20_pct"] <= baseline["dangerous_20_pct"] + 2.0)
        & (metrics["MAE_RUL_le_50"] <= baseline["MAE_RUL_le_50"] + 1e-9)
    ].copy()
    if eligible.empty:
        return baseline, "No calibrated candidate met RMSE, dangerous_20, and low-RUL constraints; baseline retained."
    eligible = eligible.sort_values(["CMAPSS_mean", "RMSE", "dangerous_20_pct"])
    best = eligible.iloc[0]
    if best["CMAPSS_mean"] < baseline["CMAPSS_mean"]:
        return best, "Selected because it reduced CMAPSS mean without worsening RMSE, dangerous_20, or RUL<=50 MAE."
    return baseline, "No calibrated candidate clearly improved CMAPSS mean under constraints; baseline retained."


def select_fd003(metrics: pd.DataFrame, baseline_name: str) -> tuple[pd.Series, str]:
    baseline = metrics.loc[metrics["candidate_name"] == baseline_name].iloc[0]
    eligible = metrics.loc[
        (metrics["CMAPSS_mean"] < baseline["CMAPSS_mean"])
        & (metrics["RMSE"] <= baseline["RMSE"] + 1e-9)
        & (metrics["dangerous_20_pct"] <= baseline["dangerous_20_pct"] + 1.0)
    ].copy()
    if eligible.empty:
        return baseline, "FD003 calibration did not clearly improve CMAPSS while keeping RMSE and dangerous errors controlled; baseline retained."
    best = eligible.sort_values(["CMAPSS_mean", "RMSE", "dangerous_20_pct"]).iloc[0]
    return best, "Selected because small bias calibration improved CMAPSS without RMSE or dangerous-error degradation."


def write_configs() -> None:
    common = {
        "split_strategy": "complete engine units with artificial RUL cutoffs",
        "metrics": [
            "MAE",
            "RMSE",
            "R2",
            "bias",
            "CMAPSS_total",
            "CMAPSS_mean",
            "dangerous_any_pct",
            "dangerous_10_pct",
            "dangerous_20_pct",
            "conservative_pct",
        ],
        "created_at": CREATED_AT,
        "official_test_usage": "no se uso test final",
    }
    write_yaml(
        PROJECT_ROOT / "configs/FD002/fd002_high_rul_calibration_grid_v01.yaml",
        {
            **common,
            "dataset": "FD002",
            "base_model": "xgb_condition_fault_sensitive_mid_guard",
            "input_files": ["results/FD002/fd002_best_validation_predictions.csv"],
            "output_files": [
                "results/FD002/fd002_high_rul_calibration_results_v01.csv",
                "results/FD002/fd002_predictions_calibrated_v01.csv",
                "results/FD002/fd002_final_candidate_after_calibration_v01.csv",
            ],
            "thresholds": THRESHOLDS,
            "offsets": OFFSETS,
            "selection_criterion": "lowest CMAPSS_mean subject to RMSE and dangerous_20 constraints",
            "seed": 42,
        },
    )
    write_yaml(
        PROJECT_ROOT / "configs/FD002/fd002_ensemble_calibration_v01.yaml",
        {
            **common,
            "dataset": "FD002",
            "base_models": [
                "xgb_condition_fault_sensitive_mid_guard",
                "xgb_squarederror_condition_normalized_weighted",
            ],
            "input_files": [
                "results/FD002/fd002_best_validation_predictions.csv",
                "results/FD002/fd002_lgbm_xgb_model_comparison_predictions.csv",
            ],
            "ensemble_weights": FD002_ENSEMBLE_WEIGHTS,
            "output_files": [
                "results/FD002/fd002_ensemble_results_v01.csv",
            ],
            "selection_criterion": "ensemble can replace baseline only if CMAPSS_mean improves without RMSE or dangerous_20 degradation",
            "seed": 42,
        },
    )
    write_yaml(
        PROJECT_ROOT / "configs/FD003/fd003_small_offset_calibration_v01.yaml",
        {
            **common,
            "dataset": "FD003",
            "base_model": "fd003_lgbm_w50_cap125_quantile_a04_none_fault_sensitive",
            "input_files": ["results/FD003/fd003_internal_validation_final_model_predictions.csv"],
            "offsets": FD003_OFFSETS,
            "output_files": [
                "results/FD003/fd003_small_offset_calibration_results_v01.csv",
                "results/FD003/fd003_predictions_calibrated_v01.csv",
                "results/FD003/fd003_final_candidate_after_calibration_v01.csv",
            ],
            "selection_criterion": "accept only if CMAPSS improves while RMSE and dangerous_20 remain controlled",
            "seed": "existing internal validation splits",
        },
    )
    write_yaml(
        PROJECT_ROOT / "configs/FD004/fd004_high_rul_calibration_grid_v01.yaml",
        {
            **common,
            "dataset": "FD004",
            "base_model": "fd004_xgb_fs_bin_weights_w70",
            "input_files": ["results/FD004/fd004_finalist_multisplit_predictions.csv"],
            "thresholds": THRESHOLDS,
            "offsets": OFFSETS,
            "output_files": [
                "results/FD004/fd004_high_rul_calibration_results_v01.csv",
                "results/FD004/fd004_predictions_calibrated_v01.csv",
                "results/FD004/fd004_final_candidate_after_calibration_v01.csv",
            ],
            "selection_criterion": "lowest CMAPSS_mean subject to RMSE and dangerous_20 constraints",
            "seed": "existing FD004 finalist multisplit",
        },
    )
    write_yaml(
        PROJECT_ROOT / "configs/FD004/fd004_condition_cluster_calibration_v01.yaml",
        {
            **common,
            "dataset": "FD004",
            "base_model": "fd004_xgb_fs_bin_weights_w70",
            "condition_analysis": "dominant condition at artificial cutoff cycle",
            "cluster_analysis": "not executed unless residual cluster labels are available",
            "input_files": ["results/FD004/fd004_finalist_multisplit_predictions.csv"],
            "output_files": [
                "results/FD004/fd004_condition_error_analysis_v01.csv",
                "results/FD004/fd004_cluster_error_analysis_v01.csv",
            ],
            "seed": "existing FD004 finalist multisplit",
        },
    )


def run_fd002() -> tuple[pd.DataFrame, pd.DataFrame]:
    notebook = "notebooks/FD002/26_fd002_high_rul_calibration_and_ensemble_v01.ipynb"
    config_high = "configs/FD002/fd002_high_rul_calibration_grid_v01.yaml"
    config_ensemble = "configs/FD002/fd002_ensemble_calibration_v01.yaml"
    baseline = normalize_project_predictions(
        pd.read_csv(PROJECT_ROOT / "results/FD002/fd002_best_validation_predictions.csv")
    )
    condition_lookup = fd002_condition_lookup()
    baseline = merge_conditions(baseline, condition_lookup)
    baseline["candidate_source"] = "baseline_final"

    second = normalize_project_predictions(
        pd.read_csv(PROJECT_ROOT / "results/FD002/fd002_lgbm_xgb_model_comparison_predictions.csv"),
        "xgb_squarederror_condition_normalized_weighted",
    )
    second = merge_conditions(second, condition_lookup)
    merge_keys = ["unit_id", "cycle", "true_RUL"]
    second_aligned = baseline[merge_keys].merge(second[merge_keys + ["pred"]], on=merge_keys, how="left")
    if second_aligned["pred"].isna().any():
        raise ValueError("FD002 ensemble candidate could not be aligned to baseline predictions.")

    candidates = []
    ensemble_rows = []
    baseline_metrics = metrics_for_predictions(
        baseline,
        "FD002",
        "fd002_baseline_xgb_condition_fault_sensitive_mid_guard",
        "xgboost",
        "none",
        config_high,
        notebook,
        "results/FD002/fd002_best_validation_predictions.csv",
        "results/FD002/fd002_high_rul_calibration_results_v01.csv",
    )
    candidates.append((baseline_metrics, baseline, baseline))
    for w1, w2 in FD002_ENSEMBLE_WEIGHTS:
        ens = baseline.copy()
        ens["pred"] = w1 * baseline["pred"].to_numpy(dtype=float) + w2 * second_aligned["pred"].to_numpy(dtype=float)
        name = f"fd002_ensemble_{w1:.1f}_{w2:.1f}".replace(".", "p")
        row = metrics_for_predictions(
            ens,
            "FD002",
            name,
            "ensemble",
            f"ensemble_{w1:.1f}_{w2:.1f}",
            config_ensemble,
            notebook,
            "results/FD002/fd002_predictions_calibrated_v01.csv",
            "results/FD002/fd002_ensemble_results_v01.csv",
        )
        ensemble_rows.append(row)
        candidates.append((row, ens, ens))

    calibration_rows = [baseline_metrics]
    ensemble_frame = pd.DataFrame(ensemble_rows).sort_values(["CMAPSS_mean", "RMSE"])
    near_ensemble_names = set(ensemble_frame.head(2)["candidate_name"])
    for base_metrics, base_pred, _ in list(candidates):
        if base_metrics["candidate_name"].startswith("fd002_ensemble") and base_metrics["candidate_name"] not in near_ensemble_names:
            continue
        for threshold in THRESHOLDS:
            for offset in OFFSETS:
                calibrated = piecewise_calibrate(base_pred, threshold, offset)
                name = f"{base_metrics['candidate_name']}_thr{threshold}_off{offset}"
                row = metrics_for_predictions(
                    calibrated,
                    "FD002",
                    name,
                    base_metrics["model_type"],
                    f"piecewise_high_rul_threshold_{threshold}_offset_{offset}",
                    config_high,
                    notebook,
                    "results/FD002/fd002_predictions_calibrated_v01.csv",
                    "results/FD002/fd002_high_rul_calibration_results_v01.csv",
                )
                row["threshold"] = threshold
                row["offset"] = offset
                row["base_candidate"] = base_metrics["candidate_name"]
                calibration_rows.append(row)
                candidates.append((row, calibrated, base_pred))

    metrics = pd.DataFrame(calibration_rows).sort_values(["CMAPSS_mean", "RMSE", "dangerous_20_pct"])
    selected_row, reason = select_fd_candidate(metrics, "fd002_baseline_xgb_condition_fault_sensitive_mid_guard")
    metrics.loc[metrics["candidate_name"] == selected_row["candidate_name"], "selected_final_candidate"] = True
    metrics.loc[metrics["candidate_name"] == selected_row["candidate_name"], "reason_selected"] = reason
    selected_pred = next(pred for row, pred, _ in candidates if row["candidate_name"] == selected_row["candidate_name"])
    selected_base = next(base for row, _, base in candidates if row["candidate_name"] == selected_row["candidate_name"])

    safe_to_csv(pd.DataFrame(ensemble_rows), PROJECT_ROOT / "results/FD002/fd002_ensemble_results_v01.csv")
    safe_to_csv(metrics, PROJECT_ROOT / "results/FD002/fd002_high_rul_calibration_results_v01.csv")
    safe_to_csv(
        condition_metrics(selected_pred, "FD002", selected_row["candidate_name"]),
        PROJECT_ROOT / "results/FD002/fd002_condition_error_analysis_v01.csv",
    )
    safe_to_csv(
        prediction_export(selected_base, selected_pred, "FD002", selected_row["candidate_name"], "fd002_calibration_v01"),
        PROJECT_ROOT / "results/FD002/fd002_predictions_calibrated_v01.csv",
    )
    final = selected_final_frame(selected_row, reason)
    final["previous_final_model"] = "xgb_condition_fault_sensitive_mid_guard"
    final["changed_vs_previous"] = final["candidate_name"] != "fd002_baseline_xgb_condition_fault_sensitive_mid_guard"
    safe_to_csv(final, PROJECT_ROOT / "results/FD002/fd002_final_candidate_after_calibration_v01.csv")
    safe_to_csv(
        bin_metrics(selected_pred, "FD002", selected_row["candidate_name"]),
        PROJECT_ROOT / "results/FD002/fd002_rul_bin_metrics_v01.csv",
    )
    safe_to_csv(
        top10_cmapss(selected_pred, "FD002", selected_row["candidate_name"]),
        PROJECT_ROOT / "results/FD002/fd002_top10_cmapss_v01.csv",
    )
    return metrics, final


def run_fd003() -> tuple[pd.DataFrame, pd.DataFrame]:
    notebook = "notebooks/FD003/26_fd003_small_bias_calibration_v01.ipynb"
    config = "configs/FD003/fd003_small_offset_calibration_v01.yaml"
    baseline = normalize_project_predictions(
        pd.read_csv(PROJECT_ROOT / "results/FD003/fd003_internal_validation_final_model_predictions.csv")
    )
    baseline["candidate_source"] = "baseline_final_internal_validation"
    rows = []
    candidates = []
    for offset in FD003_OFFSETS:
        pred = baseline.copy()
        pred["pred"] = pred["pred"] + offset
        name = f"fd003_offset_plus_{offset}"
        row = metrics_for_predictions(
            pred,
            "FD003",
            name,
            "LightGBM",
            f"global_offset_plus_{offset}",
            config,
            notebook,
            "results/FD003/fd003_predictions_calibrated_v01.csv",
            "results/FD003/fd003_small_offset_calibration_results_v01.csv",
        )
        row["offset"] = offset
        rows.append(row)
        candidates.append((row, pred))
    metrics = pd.DataFrame(rows).sort_values(["CMAPSS_mean", "RMSE", "dangerous_20_pct"])
    selected_row, reason = select_fd003(metrics, "fd003_offset_plus_0")
    metrics.loc[metrics["candidate_name"] == selected_row["candidate_name"], "selected_final_candidate"] = True
    metrics.loc[metrics["candidate_name"] == selected_row["candidate_name"], "reason_selected"] = reason
    selected_pred = next(pred for row, pred in candidates if row["candidate_name"] == selected_row["candidate_name"])
    base_pred = next(pred for row, pred in candidates if row["candidate_name"] == "fd003_offset_plus_0")
    safe_to_csv(metrics, PROJECT_ROOT / "results/FD003/fd003_small_offset_calibration_results_v01.csv")
    safe_to_csv(
        prediction_export(base_pred, selected_pred, "FD003", selected_row["candidate_name"], "fd003_calibration_v01"),
        PROJECT_ROOT / "results/FD003/fd003_predictions_calibrated_v01.csv",
    )
    final = selected_final_frame(selected_row, reason)
    final["previous_final_model"] = "fd003_lgbm_w50_cap125_quantile_a04_none_fault_sensitive"
    final["changed_vs_previous"] = final["candidate_name"] != "fd003_offset_plus_0"
    safe_to_csv(final, PROJECT_ROOT / "results/FD003/fd003_final_candidate_after_calibration_v01.csv")
    safe_to_csv(
        bin_metrics(selected_pred, "FD003", selected_row["candidate_name"]),
        PROJECT_ROOT / "results/FD003/fd003_rul_bin_metrics_v01.csv",
    )
    safe_to_csv(
        top10_cmapss(selected_pred, "FD003", selected_row["candidate_name"]),
        PROJECT_ROOT / "results/FD003/fd003_top10_cmapss_v01.csv",
    )
    return metrics, final


def run_fd004() -> tuple[pd.DataFrame, pd.DataFrame]:
    notebook = "notebooks/FD004/26_fd004_high_rul_condition_cluster_calibration_v01.ipynb"
    config = "configs/FD004/fd004_high_rul_calibration_grid_v01.yaml"
    raw = pd.read_csv(PROJECT_ROOT / "results/FD004/fd004_finalist_multisplit_predictions.csv")
    baseline = normalize_project_predictions(raw, "fd004_xgb_fs_bin_weights_w70")
    lookups = pd.concat([fd004_condition_lookup(seed) for seed in sorted(baseline["split_id"].dropna().unique())])
    baseline = merge_conditions(baseline, lookups)
    baseline["candidate_source"] = "baseline_final_multisplit"
    rows = []
    candidates = []
    baseline_row = metrics_for_predictions(
        baseline,
        "FD004",
        "fd004_baseline_xgb_fs_bin_weights_w70",
        "xgboost",
        "none",
        config,
        notebook,
        "results/FD004/fd004_finalist_multisplit_predictions.csv",
        "results/FD004/fd004_high_rul_calibration_results_v01.csv",
    )
    rows.append(baseline_row)
    candidates.append((baseline_row, baseline, baseline))
    for threshold in THRESHOLDS:
        for offset in OFFSETS:
            pred = piecewise_calibrate(baseline, threshold, offset)
            name = f"fd004_high_rul_thr{threshold}_off{offset}"
            row = metrics_for_predictions(
                pred,
                "FD004",
                name,
                "xgboost",
                f"piecewise_high_rul_threshold_{threshold}_offset_{offset}",
                config,
                notebook,
                "results/FD004/fd004_predictions_calibrated_v01.csv",
                "results/FD004/fd004_high_rul_calibration_results_v01.csv",
            )
            row["threshold"] = threshold
            row["offset"] = offset
            rows.append(row)
            candidates.append((row, pred, baseline))
    metrics = pd.DataFrame(rows).sort_values(["CMAPSS_mean", "RMSE", "dangerous_20_pct"])
    selected_row, reason = select_fd_candidate(metrics, "fd004_baseline_xgb_fs_bin_weights_w70")
    metrics.loc[metrics["candidate_name"] == selected_row["candidate_name"], "selected_final_candidate"] = True
    metrics.loc[metrics["candidate_name"] == selected_row["candidate_name"], "reason_selected"] = reason
    selected_pred = next(pred for row, pred, _ in candidates if row["candidate_name"] == selected_row["candidate_name"])
    selected_base = next(base for row, _, base in candidates if row["candidate_name"] == selected_row["candidate_name"])
    safe_to_csv(metrics, PROJECT_ROOT / "results/FD004/fd004_high_rul_calibration_results_v01.csv")
    safe_to_csv(
        condition_metrics(selected_pred, "FD004", selected_row["candidate_name"]),
        PROJECT_ROOT / "results/FD004/fd004_condition_error_analysis_v01.csv",
    )
    safe_to_csv(
        pd.DataFrame(
            [
                {
                    "dataset": "FD004",
                    "candidate_name": selected_row["candidate_name"],
                    "cluster_id": "not_available",
                    "n_motors": 0,
                    "note": "No residual cluster label file was available in the repo; cluster calibration was not executed.",
                }
            ]
        ),
        PROJECT_ROOT / "results/FD004/fd004_cluster_error_analysis_v01.csv",
    )
    safe_to_csv(
        prediction_export(selected_base, selected_pred, "FD004", selected_row["candidate_name"], "fd004_calibration_v01"),
        PROJECT_ROOT / "results/FD004/fd004_predictions_calibrated_v01.csv",
    )
    final = selected_final_frame(selected_row, reason)
    final["previous_final_model"] = "fd004_xgb_fs_bin_weights_w70"
    final["changed_vs_previous"] = final["candidate_name"] != "fd004_baseline_xgb_fs_bin_weights_w70"
    safe_to_csv(final, PROJECT_ROOT / "results/FD004/fd004_final_candidate_after_calibration_v01.csv")
    safe_to_csv(
        bin_metrics(selected_pred, "FD004", selected_row["candidate_name"]),
        PROJECT_ROOT / "results/FD004/fd004_rul_bin_metrics_v01.csv",
    )
    safe_to_csv(
        top10_cmapss(selected_pred, "FD004", selected_row["candidate_name"]),
        PROJECT_ROOT / "results/FD004/fd004_top10_cmapss_v01.csv",
    )
    return metrics, final


def build_global_summary(fd002_final: pd.DataFrame, fd003_final: pd.DataFrame, fd004_final: pd.DataFrame) -> pd.DataFrame:
    fd001 = pd.DataFrame(
        [
            {
                "dataset": "FD001",
                "candidate_name": "candidate_03_B_quantile_a040_search_14",
                "model_type": "LightGBM",
                "calibration_type": "frozen_no_new_experiment",
                "config_file": "configs/FD001/fd001_best_model_config.json",
                "notebook_file": "notebooks/conclusion/01_conclusion_final.ipynb",
                "predictions_file": "results/FD001/fd001_lgbm_final_candidate_robustness.csv",
                "metrics_file": "results/FD001/fd001_lgbm_final_candidate_robustness_summary.csv",
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "CMAPSS_mean": np.nan,
                "CMAPSS_total": np.nan,
                "dangerous_any_pct": np.nan,
                "dangerous_10_pct": np.nan,
                "dangerous_20_pct": np.nan,
                "conservative_pct": np.nan,
                "bias": np.nan,
                "MAE_RUL_le_50": np.nan,
                "MAE_RUL_gt_125": np.nan,
                "score_share_RUL_gt_125": np.nan,
                "selected_final_candidate": True,
                "reason_selected": "FD001 frozen by request; no calibration experiment run.",
            }
        ]
    )
    combined = pd.concat([fd001, fd002_final, fd003_final, fd004_final], ignore_index=True, sort=False)
    return combined


def build_selection_summary(global_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in global_summary.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "recommended_final_model_after_calibration": row["candidate_name"],
                "changed_vs_previous": bool(row.get("changed_vs_previous", False)),
                "selected_final_candidate": bool(row.get("selected_final_candidate", False)),
                "reason_selected": row.get("reason_selected", ""),
                "config_file": row.get("config_file", ""),
                "predictions_file": row.get("predictions_file", ""),
                "metrics_file": row.get("metrics_file", ""),
                "CMAPSS_mean": row.get("CMAPSS_mean", np.nan),
                "RMSE": row.get("RMSE", np.nan),
                "dangerous_20_pct": row.get("dangerous_20_pct", np.nan),
                "bias": row.get("bias", np.nan),
            }
        )
    return pd.DataFrame(rows)


def write_progress_notes(fd002_final: pd.DataFrame, fd003_final: pd.DataFrame, fd004_final: pd.DataFrame) -> None:
    diagnosis = dedent(
        f"""
        # Calibration progress v01

        Created at: {CREATED_AT}

        ## Initial diagnosis
        - Notebook folders read: `notebooks/FD001`, `notebooks/FD002`, `notebooks/FD003`, `notebooks/FD004`, `notebooks/conclusion`.
        - Results folders read: `results/FD001`, `results/FD002`, `results/FD003`, `results/FD004`.
        - Config folders read: `configs/FD001`, `configs/FD002`, `configs/FD003`, `configs/FD004`.
        - Notes folders read: `notas`, including `notas/hallazgos` and existing dataset notes.
        - Existing utilities used: `src.data`, `src.preprocessed_FD001`, `src.fd001_modeling`, `src.fd002_modeling`, `src.fd003_improvement_utils`, `src.fd004_modeling`.
        - C-MAPSS implementation used: standard project-compatible asymmetric penalty, `exp(-d/13)-1` for conservative errors and `exp(d/10)-1` for late/dangerous errors.
        - Test final was not used. All calibration metrics use existing internal validation/artificial-cutoff prediction files.

        ## Existing conclusion notebook inputs
        - `notebooks/conclusion/01_conclusion_final.ipynb` currently reads `conclusion/final_model_summary.csv`, `conclusion/final_metric_summary.csv`, `conclusion/final_rul_bin_metrics.csv`, `conclusion/maintenance_decision_summary.csv`, and `conclusion/maintenance_priority_ranking.csv`.

        ## Current final candidates
        - FD002: `xgb_condition_fault_sensitive_mid_guard`.
        - FD003: `fd003_lgbm_w50_cap125_quantile_a04_none_fault_sensitive`.
        - FD004: `fd004_xgb_fs_bin_weights_w70`.
        """
    ).strip()
    safe_write_text(PROJECT_ROOT / "notas/FD002/fd002_calibration_progress_v01.md", diagnosis + "\n")
    safe_write_text(
        PROJECT_ROOT / "notas/FD003/fd003_calibration_progress_v01.md",
        diagnosis
        + dedent(
            f"""

            ## FD003 experiment decision
            - Base model: `fd003_lgbm_w50_cap125_quantile_a04_none_fault_sensitive`.
            - Experiment run: global offsets +0, +1, +2, +3, +4 on internal validation predictions.
            - Alpha probe was not run because the request allowed it only if quick/practical; this pass kept FD003 to calibration-only evaluation.
            - Selected row: `{fd003_final.iloc[0]['candidate_name']}`.
            - Decision: {fd003_final.iloc[0]['reason_selected']}
            - New files: `results/FD003/fd003_small_offset_calibration_results_v01.csv`, `results/FD003/fd003_predictions_calibrated_v01.csv`, `results/FD003/fd003_final_candidate_after_calibration_v01.csv`.
            - Existing configs and historical notebooks were not modified.
            """
        ).strip()
        + "\n",
    )
    safe_write_text(
        PROJECT_ROOT / "notas/FD004/fd004_calibration_progress_v01.md",
        diagnosis
        + dedent(
            f"""

            ## FD004 experiment decision
            - Base model: `fd004_xgb_fs_bin_weights_w70`.
            - Experiment run: high-RUL piecewise calibration on existing finalist multisplit internal validation predictions.
            - Condition analysis was generated from dominant condition at artificial cutoff cycles.
            - Cluster analysis was not executed because no FD004 residual cluster label CSV was available.
            - Window 80/90 probe was not run to avoid retraining and because the request prioritized controlled calibration.
            - Selected row: `{fd004_final.iloc[0]['candidate_name']}`.
            - Decision: {fd004_final.iloc[0]['reason_selected']}
            - Existing configs and historical notebooks were not modified.
            """
        ).strip()
        + "\n",
    )
    # Append FD002-specific details to its note after creating the shared diagnosis.
    fd002_note = PROJECT_ROOT / "notas/FD002/fd002_calibration_progress_v01.md"
    fd002_note.write_text(
        fd002_note.read_text(encoding="utf-8")
        + dedent(
            f"""

            ## FD002 experiment decision
            - Base model: `xgb_condition_fault_sensitive_mid_guard`.
            - Secondary model for ensemble: `xgb_squarederror_condition_normalized_weighted`.
            - Experiments run: baseline, simple ensembles, high-RUL piecewise calibration, ensemble plus calibration for near-best ensembles, and condition error analysis.
            - High-cap retraining was not run because this pass uses existing internal validation predictions and avoids retraining expensive historical pipelines.
            - Selected row: `{fd002_final.iloc[0]['candidate_name']}`.
            - Decision: {fd002_final.iloc[0]['reason_selected']}
            - Existing configs and historical notebooks were not modified.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def notebook_json(title: str, dataset: str, result_files: list[str], extra_code: str) -> dict:
    markdown = f"# {title}\n\nThis notebook summarizes the v01 calibration artifacts generated from internal validation predictions only. No final test data is used."
    setup = "\n".join(
        [
            "from pathlib import Path",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "PROJECT_ROOT = Path.cwd()",
            f"DATASET = {dataset!r}",
        ]
    )
    load_lines = []
    for i, file in enumerate(result_files):
        var = f"table_{i}"
        load_lines.append(f"{var} = pd.read_csv(PROJECT_ROOT / {file!r})")
        load_lines.append(f"print({file!r}, {var}.shape)")
        load_lines.append(f"display({var}.head())")
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": (setup + "\n").splitlines(True)},
        {"cell_type": "markdown", "metadata": {}, "source": ["## Generated CSVs\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ("\n".join(load_lines) + "\n").splitlines(True)},
        {"cell_type": "markdown", "metadata": {}, "source": ["## Required diagnostic plots\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": extra_code.splitlines(True)},
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebooks() -> None:
    common_code = dedent(
        """
        pred = table_1.copy()
        metric = table_0.copy()
        if {'true_RUL', 'pred_calibrated'}.issubset(pred.columns):
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(pred['true_RUL'], pred['pred_baseline'], s=12, alpha=0.5, label='baseline')
            ax.scatter(pred['true_RUL'], pred['pred_calibrated'], s=12, alpha=0.5, label='selected/calibrated')
            ax.plot([pred['true_RUL'].min(), pred['true_RUL'].max()], [pred['true_RUL'].min(), pred['true_RUL'].max()], color='black', linewidth=1)
            ax.set_xlabel('True RUL')
            ax.set_ylabel('Predicted RUL')
            ax.legend()
            plt.show()

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(pred['true_RUL'], pred['error_calibrated'], s=12, alpha=0.6)
            ax.axhline(0, color='black', linewidth=1)
            ax.set_xlabel('True RUL')
            ax.set_ylabel('Error = pred - true')
            plt.show()
        if 'rul_bin' in table_2.columns:
            table_2.plot.bar(x='rul_bin', y='MAE', figsize=(6, 4), title='MAE by RUL bin')
            plt.show()
            table_2.plot.bar(x='rul_bin', y='CMAPSS_total', figsize=(6, 4), title='CMAPSS contribution by RUL bin')
            plt.show()
        display(table_3.head(10))
        """
    )
    safe_write_text(
        PROJECT_ROOT / "notebooks/FD002/26_fd002_high_rul_calibration_and_ensemble_v01.ipynb",
        json.dumps(
            notebook_json(
                "FD002 high-RUL calibration and ensemble v01",
                "FD002",
                [
                    "results/FD002/fd002_high_rul_calibration_results_v01.csv",
                    "results/FD002/fd002_predictions_calibrated_v01.csv",
                    "results/FD002/fd002_rul_bin_metrics_v01.csv",
                    "results/FD002/fd002_top10_cmapss_v01.csv",
                    "results/FD002/fd002_condition_error_analysis_v01.csv",
                ],
                common_code
                + "\nprint('Condition metrics')\ndisplay(table_4)\n",
            ),
            indent=2,
        ),
    )
    safe_write_text(
        PROJECT_ROOT / "notebooks/FD003/26_fd003_small_bias_calibration_v01.ipynb",
        json.dumps(
            notebook_json(
                "FD003 small bias calibration v01",
                "FD003",
                [
                    "results/FD003/fd003_small_offset_calibration_results_v01.csv",
                    "results/FD003/fd003_predictions_calibrated_v01.csv",
                    "results/FD003/fd003_rul_bin_metrics_v01.csv",
                    "results/FD003/fd003_top10_cmapss_v01.csv",
                ],
                common_code,
            ),
            indent=2,
        ),
    )
    safe_write_text(
        PROJECT_ROOT / "notebooks/FD004/26_fd004_high_rul_condition_cluster_calibration_v01.ipynb",
        json.dumps(
            notebook_json(
                "FD004 high-RUL condition and cluster calibration v01",
                "FD004",
                [
                    "results/FD004/fd004_high_rul_calibration_results_v01.csv",
                    "results/FD004/fd004_predictions_calibrated_v01.csv",
                    "results/FD004/fd004_rul_bin_metrics_v01.csv",
                    "results/FD004/fd004_top10_cmapss_v01.csv",
                    "results/FD004/fd004_condition_error_analysis_v01.csv",
                    "results/FD004/fd004_cluster_error_analysis_v01.csv",
                ],
                common_code
                + "\nprint('Condition metrics')\ndisplay(table_4)\nprint('Cluster diagnostics')\ndisplay(table_5)\n",
            ),
            indent=2,
        ),
    )


def main() -> None:
    ensure_dirs()
    write_configs()
    fd002_metrics, fd002_final = run_fd002()
    fd003_metrics, fd003_final = run_fd003()
    fd004_metrics, fd004_final = run_fd004()
    global_summary = build_global_summary(fd002_final, fd003_final, fd004_final)
    safe_to_csv(global_summary, PROJECT_ROOT / "results/final_calibration_summary_v01.csv")
    safe_to_csv(
        build_selection_summary(global_summary),
        PROJECT_ROOT / "results/final_model_selection_after_calibration_v01.csv",
    )
    write_progress_notes(fd002_final, fd003_final, fd004_final)
    write_notebooks()
    print("Calibration v01 artifacts created.")


if __name__ == "__main__":
    main()
