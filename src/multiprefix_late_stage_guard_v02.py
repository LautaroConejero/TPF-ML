from __future__ import annotations

import json
import math
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUL_BINS = [0, 50, 75, 100, 125, math.inf]
RUL_BIN_LABELS = ["0-50", "50-75", "75-100", "100-125", "125+"]
TOLERANCE_MAE_LOW_RUL = 0.10


def safe_to_csv(df: pd.DataFrame, rel_path: str) -> None:
    path = PROJECT_ROOT / rel_path
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_write_text(rel_path: str, content: str) -> None:
    path = PROJECT_ROOT / rel_path
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def cmapss_penalty(error: pd.Series) -> pd.Series:
    values = error.astype(float).to_numpy()
    return pd.Series(
        np.where(values < 0, np.exp(-values / 13.0) - 1.0, np.exp(values / 10.0) - 1.0),
        index=error.index,
    )


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["pred_RUL"] = result["pred_RUL"].clip(lower=0)
    result["error"] = result["pred_RUL"] - result["true_RUL"]
    result["abs_error"] = result["error"].abs()
    result["squared_error"] = result["error"] ** 2
    result["cmapss_penalty"] = cmapss_penalty(result["error"])
    result["dangerous_any"] = result["error"] > 0
    result["dangerous_10"] = result["error"] > 10
    result["dangerous_20"] = result["error"] > 20
    result["conservative"] = result["error"] < 0
    result["rul_bin"] = pd.cut(result["true_RUL"], bins=RUL_BINS, labels=RUL_BIN_LABELS, include_lowest=True, right=True)
    return result


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def bin_mae(group: pd.DataFrame, label: str) -> float:
    values = group.loc[group["rul_bin"].astype(str).eq(label), "abs_error"]
    return float(values.mean()) if len(values) else float("nan")


def metrics_for_candidate(group: pd.DataFrame) -> dict:
    total_score = float(group["cmapss_penalty"].sum())
    low = group["true_RUL"] <= 50
    guarded = group.get("guard_applied", pd.Series(False, index=group.index)).astype(bool)
    return {
        "dataset": str(group["dataset"].iloc[0]),
        "candidate_name": str(group["candidate_name"].iloc[0]),
        "n_eval_points": int(len(group)),
        "n_unique_units": int(group["unit_id"].nunique()),
        "MAE": float(group["abs_error"].mean()),
        "RMSE": float(np.sqrt(group["squared_error"].mean())),
        "R2": r2_score(group["true_RUL"], group["pred_RUL"]),
        "bias": float(group["error"].mean()),
        "CMAPSS_total": total_score,
        "CMAPSS_mean": float(group["cmapss_penalty"].mean()),
        "dangerous_any_pct": float(group["dangerous_any"].mean() * 100.0),
        "dangerous_10_pct": float(group["dangerous_10"].mean() * 100.0),
        "dangerous_20_pct": float(group["dangerous_20"].mean() * 100.0),
        "conservative_rate_pct": float(group["conservative"].mean() * 100.0),
        "MAE_RUL_le_50": bin_mae(group, "0-50"),
        "MAE_RUL_50_75": bin_mae(group, "50-75"),
        "MAE_RUL_75_100": bin_mae(group, "75-100"),
        "MAE_RUL_100_125": bin_mae(group, "100-125"),
        "MAE_RUL_gt_125": bin_mae(group, "125+"),
        "pred_min": float(group["pred_RUL"].min()),
        "pred_max": float(group["pred_RUL"].max()),
        "true_RUL_max": float(group["true_RUL"].max()),
        "n_guarded": int(guarded.sum()),
        "pct_guarded": float(guarded.mean() * 100.0),
        "n_guarded_RUL_le_50": int((guarded & low).sum()),
        "pct_guarded_RUL_le_50": float((guarded & low).sum() / max(int(low.sum()), 1) * 100.0),
        "mean_delta_vs_eol": float(group["delta_vs_eol"].mean()) if "delta_vs_eol" in group else 0.0,
        "mean_delta_vs_baseline": float(group["delta_vs_baseline"].mean()) if "delta_vs_baseline" in group else 0.0,
        "dangerous_20_RUL_le_50": float(group.loc[low, "dangerous_20"].mean() * 100.0) if low.any() else float("nan"),
        "CMAPSS_mean_RUL_le_50": float(group.loc[low, "cmapss_penalty"].mean()) if low.any() else float("nan"),
    }


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics_for_candidate(group) for _, group in predictions.groupby("candidate_name", sort=False)]
    return pd.DataFrame(rows)


def metrics_by_bin(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (dataset, candidate), candidate_group in predictions.groupby(["dataset", "candidate_name"], sort=False):
        total_score = float(candidate_group["cmapss_penalty"].sum())
        for rul_bin, group in candidate_group.groupby("rul_bin", observed=False):
            if group.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_name": candidate,
                    "rul_bin": str(rul_bin),
                    "n_eval_points": int(len(group)),
                    "n_unique_units": int(group["unit_id"].nunique()),
                    "MAE": float(group["abs_error"].mean()),
                    "RMSE": float(np.sqrt(group["squared_error"].mean())),
                    "CMAPSS_total": float(group["cmapss_penalty"].sum()),
                    "CMAPSS_mean": float(group["cmapss_penalty"].mean()),
                    "CMAPSS_share": float(group["cmapss_penalty"].sum() / total_score) if total_score else float("nan"),
                    "dangerous_any_pct": float(group["dangerous_any"].mean() * 100.0),
                    "dangerous_10_pct": float(group["dangerous_10"].mean() * 100.0),
                    "dangerous_20_pct": float(group["dangerous_20"].mean() * 100.0),
                    "bias": float(group["error"].mean()),
                }
            )
    return pd.DataFrame(rows)


def fd002_lock() -> None:
    results = pd.read_csv(PROJECT_ROOT / "results/FD002/fd002_multiprefix_eol_smoothing_results_v01.csv")
    bins = pd.read_csv(PROJECT_ROOT / "results/FD002/fd002_multiprefix_metrics_by_rul_bin_v01.csv")
    diagnostics = pd.read_csv(PROJECT_ROOT / "results/FD002/fd002_multiprefix_diagnostics_v01.csv")
    baseline = results.loc[results["candidate_name"].eq("baseline")].iloc[0]
    eol_mean = results.loc[results["candidate_name"].eq("eol_mean")].iloc[0]
    best = results.sort_values(["CMAPSS_mean", "dangerous_20_pct", "RMSE"]).iloc[0]

    adopted = bool(
        best["candidate_name"] == "eol_mean"
        and eol_mean["CMAPSS_mean"] < baseline["CMAPSS_mean"] * 0.995
        and eol_mean["dangerous_20_pct"] < baseline["dangerous_20_pct"]
        and eol_mean["RMSE"] <= baseline["RMSE"]
        and eol_mean["MAE_RUL_le_50"] <= baseline["MAE_RUL_le_50"] + TOLERANCE_MAE_LOW_RUL
        and bool(diagnostics["no_future_prefixes_used"].iloc[0])
    )
    reason = (
        "Adopted: eol_mean is the best CMAPSS candidate and improves dangerous_20, RMSE, and low-RUL MAE without future-prefix leakage."
        if adopted
        else "Not adopted: one or more consolidation guards failed."
    )

    final = pd.DataFrame(
        [
            {
                "dataset": "FD002",
                "selected_candidate": "eol_mean",
                "baseline_candidate": "baseline",
                "MAE": eol_mean["MAE"],
                "RMSE": eol_mean["RMSE"],
                "R2": eol_mean["R2"],
                "bias": eol_mean["bias"],
                "CMAPSS_total": eol_mean["CMAPSS_total"],
                "CMAPSS_mean": eol_mean["CMAPSS_mean"],
                "dangerous_any": eol_mean["dangerous_any_pct"],
                "dangerous_10": eol_mean["dangerous_10_pct"],
                "dangerous_20": eol_mean["dangerous_20_pct"],
                "conservative_rate": eol_mean["conservative_rate_pct"],
                "MAE_RUL_le_50": eol_mean["MAE_RUL_le_50"],
                "MAE_RUL_50_75": eol_mean["MAE_RUL_50_75"],
                "MAE_RUL_75_100": eol_mean["MAE_RUL_75_100"],
                "MAE_RUL_100_125": eol_mean["MAE_RUL_100_125"],
                "MAE_RUL_gt_125": eol_mean["MAE_RUL_gt_125"],
                "pred_min": eol_mean["pred_min"],
                "pred_max": eol_mean["pred_max"],
                "true_RUL_max": eol_mean["true_RUL_max"],
                "n_eval_points": eol_mean["n_eval_points"],
                "n_unique_units": eol_mean["n_unique_units"],
                "adopted_flag": adopted,
                "adoption_reason": reason,
            }
        ]
    )
    diag_out = pd.DataFrame(
        [
            {
                "dataset": "FD002",
                "baseline_candidate": "baseline",
                "selected_candidate": "eol_mean",
                "best_by_CMAPSS_mean": best["candidate_name"],
                "adopted_flag": adopted,
                "adoption_reason": reason,
                "uses_final_test": False,
                "no_future_prefixes_used": bool(diagnostics["no_future_prefixes_used"].iloc[0]),
                "prefix_spacing_mode": diagnostics["prefix_spacing_mode"].iloc[0],
                "baseline_CMAPSS_mean": baseline["CMAPSS_mean"],
                "selected_CMAPSS_mean": eol_mean["CMAPSS_mean"],
                "delta_CMAPSS_mean": eol_mean["CMAPSS_mean"] - baseline["CMAPSS_mean"],
                "baseline_dangerous_20": baseline["dangerous_20_pct"],
                "selected_dangerous_20": eol_mean["dangerous_20_pct"],
                "baseline_RMSE": baseline["RMSE"],
                "selected_RMSE": eol_mean["RMSE"],
                "baseline_MAE_RUL_le_50": baseline["MAE_RUL_le_50"],
                "selected_MAE_RUL_le_50": eol_mean["MAE_RUL_le_50"],
            }
        ]
    )
    selected_bins = bins.loc[bins["candidate_name"].isin(["baseline", "eol_mean"])].copy()
    safe_to_csv(final, "results/FD002/FD002_final_candidate_multiprefix_v02.csv")
    safe_to_csv(selected_bins, "results/FD002/FD002_final_candidate_multiprefix_metrics_by_bin_v02.csv")
    safe_to_csv(diag_out, "results/FD002/FD002_final_candidate_multiprefix_diagnostics_v02.csv")
    safe_write_text(
        "notebooks/FD002/28_fd002_multiprefix_candidate_lock_v02.ipynb",
        json.dumps(fd002_notebook_json(), indent=2, ensure_ascii=False) + "\n",
    )


def load_fd004_base() -> pd.DataFrame:
    preds = pd.read_csv(PROJECT_ROOT / "results/FD004/fd004_multiprefix_predictions_v01.csv")
    keys = ["unit_id", "split_id", "cycle", "true_RUL"]
    baseline = preds.loc[preds["candidate_name"].eq("baseline"), keys + ["pred_RUL", "n_prefixes_used", "prefix_cycles_used", "max_prefix_le_target"]].rename(
        columns={"pred_RUL": "baseline_pred"}
    )
    eol = preds.loc[preds["candidate_name"].eq("eol_mean"), keys + ["pred_RUL", "n_prefixes_used", "prefix_cycles_used", "max_prefix_le_target"]].rename(
        columns={"pred_RUL": "eol_mean_pred"}
    )
    merged = baseline.merge(eol, on=keys, suffixes=("_baseline", "_eol"), validate="one_to_one")
    if not merged["max_prefix_le_target_baseline"].all() or not merged["max_prefix_le_target_eol"].all():
        raise ValueError("Future-prefix leakage check failed in FD004 v01 predictions.")
    return merged


def build_candidate(base: pd.DataFrame, name: str, pred: pd.Series, guard_applied: pd.Series | bool, rule: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "dataset": "FD004",
            "unit_id": base["unit_id"].astype(int),
            "split_id": base["split_id"].astype(int),
            "cycle": base["cycle"].astype(int),
            "true_RUL": base["true_RUL"].astype(float),
            "candidate_name": name,
            "pred_RUL": pred.astype(float),
            "baseline_pred": base["baseline_pred"].astype(float),
            "eol_mean_pred": base["eol_mean_pred"].astype(float),
            "guard_applied": guard_applied if isinstance(guard_applied, pd.Series) else bool(guard_applied),
            "guard_rule": rule,
            "n_prefixes_used": base["n_prefixes_used_eol"].astype(int),
            "prefix_cycles_used": base["prefix_cycles_used_eol"].astype(str),
            "max_prefix_le_target": base["max_prefix_le_target_eol"].astype(bool),
        }
    )
    out["delta_vs_eol"] = out["pred_RUL"] - out["eol_mean_pred"]
    out["delta_vs_baseline"] = out["pred_RUL"] - out["baseline_pred"]
    return add_error_columns(out)


def fd004_guard_candidates() -> pd.DataFrame:
    base = load_fd004_base()
    b = base["baseline_pred"].astype(float)
    e = base["eol_mean_pred"].astype(float)
    candidates = [
        build_candidate(base, "baseline", b, False, "current fd004_high_rul_thr120_off2 baseline"),
        build_candidate(base, "eol_mean", e, False, "plain eol_mean from multiprefix v01"),
    ]
    rules = {
        "eol_mean_guard_b50_floor0": (np.maximum(e, b), b <= 50, "if baseline_pred <= 50, max(eol_mean_pred, baseline_pred)"),
        "eol_mean_guard_b50_floor2": (np.maximum(e, b - 2), b <= 50, "if baseline_pred <= 50, max(eol_mean_pred, baseline_pred - 2)"),
        "eol_mean_guard_b50_blend50": (np.where(b <= 50, 0.5 * b + 0.5 * e, e), b <= 50, "if baseline_pred <= 50, blend baseline/eol 50-50"),
        "eol_mean_guard_b60_floor2": (np.maximum(e, b - 2), b <= 60, "if baseline_pred <= 60, max(eol_mean_pred, baseline_pred - 2)"),
        "eol_mean_guard_b60_blend50": (np.where(b <= 60, 0.5 * b + 0.5 * e, e), b <= 60, "if baseline_pred <= 60, blend baseline/eol 50-50"),
        "eol_mean_guard_b50_only_if_lower_floor2": (
            np.where((b <= 50) & (e < b), np.maximum(e, b - 2), e),
            (b <= 50) & (e < b),
            "if baseline_pred <= 50 and eol_mean_pred < baseline_pred, floor at baseline_pred - 2",
        ),
        "eol_mean_guard_b60_only_if_lower_floor2": (
            np.where((b <= 60) & (e < b), np.maximum(e, b - 2), e),
            (b <= 60) & (e < b),
            "if baseline_pred <= 60 and eol_mean_pred < baseline_pred, floor at baseline_pred - 2",
        ),
    }
    for name, (pred, mask, rule) in rules.items():
        pred_series = pd.Series(pred, index=base.index).astype(float)
        guard_applied = pd.Series(mask, index=base.index).astype(bool) & (pred_series.round(12) != e.round(12))
        candidates.append(build_candidate(base, name, pred_series, guard_applied, rule))
    return pd.concat(candidates, ignore_index=True)


def apply_fd004_acceptance(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    baseline = out.loc[out["candidate_name"].eq("baseline")].iloc[0]
    out["acceptable"] = (
        (out["CMAPSS_mean"] < baseline["CMAPSS_mean"] * 0.995)
        & (out["dangerous_20_pct"] <= baseline["dangerous_20_pct"])
        & (out["RMSE"] <= baseline["RMSE"] * 1.01)
        & (out["MAE_RUL_le_50"] <= baseline["MAE_RUL_le_50"] + TOLERANCE_MAE_LOW_RUL)
        & (out["dangerous_20_RUL_le_50"] <= baseline["dangerous_20_RUL_le_50"])
        & (out["n_unique_units"] >= max(10, int(baseline["n_unique_units"] * 0.9)))
    )
    out.loc[out["candidate_name"].eq("baseline"), "acceptable"] = False
    complexity = {
        "eol_mean": 0,
        "eol_mean_guard_b50_floor0": 1,
        "eol_mean_guard_b50_floor2": 1,
        "eol_mean_guard_b50_only_if_lower_floor2": 2,
        "eol_mean_guard_b50_blend50": 2,
        "eol_mean_guard_b60_floor2": 2,
        "eol_mean_guard_b60_only_if_lower_floor2": 3,
        "eol_mean_guard_b60_blend50": 3,
        "baseline": 9,
    }
    out["rule_complexity_rank"] = out["candidate_name"].map(complexity).fillna(5)
    acceptable = out.loc[out["acceptable"]].sort_values(["CMAPSS_mean", "dangerous_20_pct", "MAE_RUL_le_50", "RMSE", "rule_complexity_rank"])
    if acceptable.empty:
        selected_name = "baseline"
        reason = "No v02 candidate passed all acceptance guards; retain baseline."
    else:
        selected_name = acceptable.iloc[0]["candidate_name"]
        if selected_name == "eol_mean":
            reason = "Adopted: plain eol_mean already satisfies the low-RUL tolerance, so no guard is forced."
        else:
            reason = "Adopted: selected guard satisfies acceptance constraints with best CMAPSS priority."
    out["selected_best_candidate"] = out["candidate_name"].eq(selected_name)
    out["selection_reason"] = ""
    out.loc[out["selected_best_candidate"], "selection_reason"] = reason
    return out


def fd004_guard() -> None:
    predictions = fd004_guard_candidates()
    metrics = apply_fd004_acceptance(metrics_table(predictions))
    selected_name = metrics.loc[metrics["selected_best_candidate"], "candidate_name"].iloc[0]
    selected_reason = metrics.loc[metrics["selected_best_candidate"], "selection_reason"].iloc[0]
    selected_predictions = predictions.loc[predictions["candidate_name"].eq(selected_name)]
    eol_metrics = metrics.loc[metrics["candidate_name"].eq("eol_mean")].iloc[0]
    baseline = metrics.loc[metrics["candidate_name"].eq("baseline")].iloc[0]
    selected = metrics.loc[metrics["candidate_name"].eq(selected_name)].iloc[0]
    diagnostics = pd.DataFrame(
        [
            {
                "dataset": "FD004",
                "baseline_candidate": "baseline",
                "plain_eol_candidate": "eol_mean",
                "selected_candidate": selected_name,
                "accepted": bool(selected["acceptable"]),
                "selection_reason": selected_reason,
                "uses_final_test": False,
                "no_future_prefixes_used": bool(predictions["max_prefix_le_target"].all()),
                "baseline_CMAPSS_mean": baseline["CMAPSS_mean"],
                "eol_mean_CMAPSS_mean": eol_metrics["CMAPSS_mean"],
                "selected_CMAPSS_mean": selected["CMAPSS_mean"],
                "baseline_dangerous_20": baseline["dangerous_20_pct"],
                "eol_mean_dangerous_20": eol_metrics["dangerous_20_pct"],
                "selected_dangerous_20": selected["dangerous_20_pct"],
                "baseline_MAE_RUL_le_50": baseline["MAE_RUL_le_50"],
                "eol_mean_MAE_RUL_le_50": eol_metrics["MAE_RUL_le_50"],
                "selected_MAE_RUL_le_50": selected["MAE_RUL_le_50"],
                "low_rul_tolerance_abs": TOLERANCE_MAE_LOW_RUL,
                "plain_eol_low_rul_within_tolerance": bool(eol_metrics["MAE_RUL_le_50"] <= baseline["MAE_RUL_le_50"] + TOLERANCE_MAE_LOW_RUL),
                "n_guarded_selected": int(selected["n_guarded"]),
                "pct_guarded_selected": selected["pct_guarded"],
                "n_eval_points": int(selected["n_eval_points"]),
                "n_unique_units": int(selected["n_unique_units"]),
            }
        ]
    )
    safe_to_csv(metrics.sort_values(["CMAPSS_mean", "dangerous_20_pct", "MAE_RUL_le_50"]), "results/FD004/FD004_multiprefix_late_stage_guard_results_v02.csv")
    safe_to_csv(predictions, "results/FD004/FD004_multiprefix_late_stage_guard_predictions_v02.csv")
    safe_to_csv(metrics.loc[metrics["candidate_name"].eq(selected_name)].copy(), "results/FD004/FD004_multiprefix_late_stage_guard_best_candidate_v02.csv")
    safe_to_csv(metrics_by_bin(predictions), "results/FD004/FD004_multiprefix_late_stage_guard_metrics_by_bin_v02.csv")
    safe_to_csv(diagnostics, "results/FD004/FD004_multiprefix_late_stage_guard_diagnostics_v02.csv")
    safe_write_text(
        "notebooks/FD004/28_fd004_multiprefix_late_stage_guard_v02.ipynb",
        json.dumps(fd004_notebook_json(), indent=2, ensure_ascii=False) + "\n",
    )


def notebook_base_cells(title: str, objective: str, setup_code: str, analysis_code: str, final_markdown: str) -> list[dict]:
    return [
        {"cell_type": "markdown", "metadata": {}, "source": f"# {title}\n\n{objective}\n".splitlines(True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": setup_code.strip().splitlines(True)},
        {"cell_type": "markdown", "metadata": {}, "source": ["## Results\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": analysis_code.strip().splitlines(True)},
        {"cell_type": "markdown", "metadata": {}, "source": final_markdown.strip().splitlines(True)},
    ]


def fd002_notebook_json() -> dict:
    setup = """
from pathlib import Path
import pandas as pd
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd()
for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    if (candidate / "results" / "FD002" / "FD002_final_candidate_multiprefix_v02.csv").exists():
        PROJECT_ROOT = candidate
        break

final = pd.read_csv(PROJECT_ROOT / "results/FD002/FD002_final_candidate_multiprefix_v02.csv")
bins = pd.read_csv(PROJECT_ROOT / "results/FD002/FD002_final_candidate_multiprefix_metrics_by_bin_v02.csv")
diagnostics = pd.read_csv(PROJECT_ROOT / "results/FD002/FD002_final_candidate_multiprefix_diagnostics_v02.csv")
v01_results = pd.read_csv(PROJECT_ROOT / "results/FD002/fd002_multiprefix_eol_smoothing_results_v01.csv")
"""
    analysis = """
display(v01_results.sort_values("CMAPSS_mean"))
display(final)
display(diagnostics)
display(bins)
"""
    final_md = """## Decision final v02

- Candidato consolidado: `eol_mean`.
- Se adopta para FD002 porque confirma mejora clara de CMAPSS mean, dangerous_20, RMSE y MAE en RUL <= 50 contra baseline.
- No se uso test final.
- No se usaron prefijos futuros: la consolidacion hereda el check `no_future_prefixes_used=True` de la corrida multi-prefix v01.
- No se siguio sobreajustando: esta v02 solo bloquea el candidato recomendado con las metricas ya generadas."""
    return {
        "cells": notebook_base_cells(
            "FD002 multi-prefix candidate lock v02",
            "Consolidate eol_mean as the FD002 recommended multi-prefix candidate without further tuning.",
            setup,
            analysis,
            final_md,
        ),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def fd004_notebook_json() -> dict:
    setup = """
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd()
for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    if (candidate / "results" / "FD004" / "FD004_multiprefix_late_stage_guard_results_v02.csv").exists():
        PROJECT_ROOT = candidate
        break

results = pd.read_csv(PROJECT_ROOT / "results/FD004/FD004_multiprefix_late_stage_guard_results_v02.csv")
predictions = pd.read_csv(PROJECT_ROOT / "results/FD004/FD004_multiprefix_late_stage_guard_predictions_v02.csv")
best = pd.read_csv(PROJECT_ROOT / "results/FD004/FD004_multiprefix_late_stage_guard_best_candidate_v02.csv")
bins = pd.read_csv(PROJECT_ROOT / "results/FD004/FD004_multiprefix_late_stage_guard_metrics_by_bin_v02.csv")
diagnostics = pd.read_csv(PROJECT_ROOT / "results/FD004/FD004_multiprefix_late_stage_guard_diagnostics_v02.csv")
"""
    analysis = """
display(results.sort_values("CMAPSS_mean"))
display(results[results["acceptable"]].sort_values(["CMAPSS_mean", "dangerous_20_pct", "MAE_RUL_le_50", "RMSE"]))
display(diagnostics)

best_name = best.loc[0, "candidate_name"]
base = predictions[predictions["candidate_name"].eq("baseline")].copy()
eol = predictions[predictions["candidate_name"].eq("eol_mean")].copy()
guard = predictions[predictions["candidate_name"].eq(best_name)].copy()

fig, ax = plt.subplots(figsize=(5, 5))
for label, frame in [("baseline", base), ("eol_mean", eol), (best_name, guard)]:
    ax.scatter(frame["true_RUL"], frame["pred_RUL"], s=12, alpha=0.45, label=label)
lim = max(predictions["true_RUL"].max(), predictions["pred_RUL"].max())
ax.plot([0, lim], [0, lim], color="black", linewidth=1)
ax.set_xlabel("True RUL")
ax.set_ylabel("Predicted RUL")
ax.set_title("True vs predicted RUL")
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
for label, frame in [("baseline", base), ("eol_mean", eol), (best_name, guard)]:
    ax.scatter(frame["true_RUL"], frame["error"], s=12, alpha=0.45, label=label)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("True RUL")
ax.set_ylabel("Pred - true")
ax.set_title("Residuals by true RUL")
ax.legend()
plt.show()

plot_bins = bins[bins["candidate_name"].isin(["baseline", "eol_mean", best_name])].copy()
for value_col, title in [
    ("MAE", "MAE by RUL bin"),
    ("dangerous_20_pct", "Dangerous_20 by RUL bin"),
    ("CMAPSS_share", "CMAPSS share by RUL bin"),
]:
    pivot = plot_bins.pivot(index="rul_bin", columns="candidate_name", values=value_col)
    display(pivot)
    pivot.plot(kind="bar", figsize=(7, 4), title=title)
    plt.tight_layout()
    plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
guard["delta_vs_eol"].hist(ax=axes[0], bins=30)
axes[0].set_title(f"{best_name}: guard_pred - eol_mean_pred")
guard["delta_vs_baseline"].hist(ax=axes[1], bins=30)
axes[1].set_title(f"{best_name}: guard_pred - baseline_pred")
plt.tight_layout()
plt.show()

for label, frame in [("baseline", base), ("eol_mean", eol), (best_name, guard)]:
    display(Markdown(f"### Top 10 CMAPSS - {label}"))
    display(frame.sort_values("cmapss_penalty", ascending=False).head(10)[["unit_id", "split_id", "cycle", "true_RUL", "pred_RUL", "error", "cmapss_penalty", "guard_applied"]])

changed = guard[guard["guard_applied"]].sort_values("cycle").head(8)
display(Markdown("### Examples where selected guard changes prediction"))
display(changed[["unit_id", "split_id", "cycle", "true_RUL", "baseline_pred", "eol_mean_pred", "pred_RUL", "delta_vs_eol", "delta_vs_baseline"]])
for _, row in changed[["unit_id", "split_id"]].drop_duplicates().head(4).iterrows():
    mask = guard["unit_id"].eq(row["unit_id"]) & guard["split_id"].eq(row["split_id"])
    unit_guard = guard[mask].sort_values("cycle")
    unit_base = base[mask].sort_values("cycle")
    unit_eol = eol[mask].sort_values("cycle")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(unit_base["cycle"], unit_base["true_RUL"], marker="o", label="true RUL")
    ax.plot(unit_base["cycle"], unit_base["pred_RUL"], marker="o", label="baseline")
    ax.plot(unit_eol["cycle"], unit_eol["pred_RUL"], marker="o", label="eol_mean")
    ax.plot(unit_guard["cycle"], unit_guard["pred_RUL"], marker="o", label=best_name)
    ax.set_title(f"Unit {int(row['unit_id'])} split {int(row['split_id'])}")
    ax.set_xlabel("cycle")
    ax.set_ylabel("RUL")
    ax.legend()
    plt.tight_layout()
    plt.show()
"""
    final_md = """## Decision final v02

- Mejor candidato encontrado: leer `FD004_multiprefix_late_stage_guard_best_candidate_v02.csv`.
- Comparar contra baseline actual `fd004_high_rul_thr120_off2` y contra `eol_mean` sin guard usando las tablas de arriba.
- Si `eol_mean` ya cumple la tolerancia de MAE_RUL_le_50 (+0.10 absoluto), no se fuerza un guard.
- No se uso test final.
- No se usaron prefijos futuros: el notebook valida `max_prefix_le_target` heredado de v01.
- Riesgo metodologico: los guards son reglas post-smoothing sobre predicciones internas; todavia no deben promoverse a archivos globales finales hasta decidir el cierre."""
    return {
        "cells": notebook_base_cells(
            "FD004 multi-prefix late-stage guard v02",
            "Test simple late-stage guards on existing multi-prefix predictions without retraining or test-final usage.",
            setup,
            analysis,
            final_md,
        ),
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    fd002_lock()
    fd004_guard()
    print("multiprefix late-stage guard v02 artifacts created")


if __name__ == "__main__":
    main()
