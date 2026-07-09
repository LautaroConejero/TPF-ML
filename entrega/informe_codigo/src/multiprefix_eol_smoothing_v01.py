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
CANDIDATES = ["baseline", "eol_mean", "eol_median", "eol_weighted_mean", "eol_last3_mean", "eol_last3_median"]


def ensure_dirs() -> None:
    for rel in ["results/FD002", "results/FD004", "notebooks/FD002", "notebooks/FD004"]:
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)


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


def normalize_source(dataset: str) -> pd.DataFrame:
    if dataset == "FD002":
        raw = pd.read_csv(PROJECT_ROOT / "results/FD002/fd002_best_validation_predictions.csv")
        raw = raw.loc[raw["model_name"].eq("xgb_condition_fault_sensitive_mid_guard")].copy()
        raw["split_id"] = 0
        raw["pred_prefix"] = raw["y_pred_rul"].astype(float)
        source_file = "results/FD002/fd002_best_validation_predictions.csv"
        current_candidate = "xgb_condition_fault_sensitive_mid_guard"
        calibration_rule = "none"
    elif dataset == "FD004":
        raw = pd.read_csv(PROJECT_ROOT / "results/FD004/fd004_finalist_multisplit_predictions.csv")
        raw = raw.loc[raw["model_name"].eq("fd004_xgb_fs_bin_weights_w70")].copy()
        raw["split_id"] = raw["eval_random_state"].astype(int)
        raw["pred_prefix"] = np.where(raw["y_pred_rul"].astype(float) > 120.0, raw["y_pred_rul"].astype(float) + 2.0, raw["y_pred_rul"].astype(float))
        source_file = "results/FD004/fd004_finalist_multisplit_predictions.csv"
        current_candidate = "fd004_high_rul_thr120_off2"
        calibration_rule = "if raw_pred_RUL > 120, add +2 before EOL aggregation"
    else:
        raise ValueError(dataset)

    table = pd.DataFrame(
        {
            "dataset": dataset,
            "unit_id": raw["unit"].astype(int),
            "split_id": raw["split_id"].astype(int),
            "cycle": raw["cycle"].astype(int),
            "true_RUL": raw["y_true_rul_raw"].astype(float),
            "pred_prefix": raw["pred_prefix"].astype(float),
            "raw_pred_prefix": raw["y_pred_rul"].astype(float),
            "source_model_name": raw["model_name"].astype(str),
            "source_file": source_file,
            "current_candidate": current_candidate,
            "calibration_rule": calibration_rule,
            "window_size_used": raw.get("window_size_used", pd.Series(np.nan, index=raw.index)),
            "max_cycle": raw.get("max_cycle", pd.Series(np.nan, index=raw.index)),
        }
    )
    table = table.sort_values(["split_id", "unit_id", "cycle"]).reset_index(drop=True)
    return table


def weighted_mean(values: np.ndarray, distances: np.ndarray) -> float:
    weights = 1.0 / (1.0 + distances.astype(float))
    return float(np.average(values, weights=weights))


def smoothed_predictions(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["split_id", "unit_id"]
    for (split_id, unit_id), group in source.groupby(group_cols, sort=False):
        group = group.sort_values("cycle").reset_index(drop=True)
        for _, target in group.iterrows():
            prefixes = group.loc[group["cycle"] <= target["cycle"]].copy()
            if prefixes.empty:
                raise RuntimeError("Internal error: each target should include itself as a prefix.")
            eol = prefixes["cycle"].astype(float).to_numpy() + prefixes["pred_prefix"].astype(float).to_numpy()
            distances = float(target["cycle"]) - prefixes["cycle"].astype(float).to_numpy()
            last3_eol = eol[-3:]

            values = {
                "baseline": float(target["pred_prefix"]),
                "eol_mean": float(np.mean(eol) - target["cycle"]),
                "eol_median": float(np.median(eol) - target["cycle"]),
                "eol_weighted_mean": weighted_mean(eol, distances) - float(target["cycle"]),
                "eol_last3_mean": float(np.mean(last3_eol) - target["cycle"]),
                "eol_last3_median": float(np.median(last3_eol) - target["cycle"]),
            }
            values = {key: max(0.0, value) for key, value in values.items()}

            eol_baseline = float(target["cycle"] + target["pred_prefix"])
            for candidate, pred in values.items():
                eol_smoothed = float(target["cycle"] + pred)
                rows.append(
                    {
                        "dataset": target["dataset"],
                        "unit_id": int(unit_id),
                        "split_id": int(split_id),
                        "cycle": int(target["cycle"]),
                        "true_RUL": float(target["true_RUL"]),
                        "candidate_name": candidate,
                        "pred_RUL": pred,
                        "baseline_pred_RUL": float(target["pred_prefix"]),
                        "raw_pred_RUL": float(target["raw_pred_prefix"]),
                        "eol_baseline": eol_baseline,
                        "eol_smoothed": eol_smoothed,
                        "n_prefixes_used": int(len(prefixes)),
                        "prefix_cycles_used": "|".join(str(int(cycle)) for cycle in prefixes["cycle"].tolist()),
                        "min_prefix_cycle": int(prefixes["cycle"].min()),
                        "max_prefix_cycle": int(prefixes["cycle"].max()),
                        "max_prefix_le_target": bool(prefixes["cycle"].max() <= target["cycle"]),
                        "prefix_cycle_spacing_mode": float(prefixes["cycle"].sort_values().diff().dropna().mode().iloc[0])
                        if len(prefixes) > 1 and not prefixes["cycle"].sort_values().diff().dropna().mode().empty
                        else np.nan,
                        "source_model_name": target["source_model_name"],
                        "source_file": target["source_file"],
                        "current_candidate": target["current_candidate"],
                        "calibration_rule": target["calibration_rule"],
                    }
                )
    result = pd.DataFrame(rows)
    result["error"] = result["pred_RUL"] - result["true_RUL"]
    result["abs_error"] = result["error"].abs()
    result["squared_error"] = result["error"] ** 2
    result["cmapss_penalty"] = cmapss_penalty(result["error"])
    result["dangerous_any"] = result["error"] > 0
    result["dangerous_10"] = result["error"] > 10
    result["dangerous_20"] = result["error"] > 20
    result["conservative"] = result["error"] < 0
    result["rul_bin"] = pd.cut(result["true_RUL"], bins=RUL_BINS, labels=RUL_BIN_LABELS, right=True, include_lowest=True)
    return result


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def metrics_for_candidate(group: pd.DataFrame) -> dict:
    total_score = float(group["cmapss_penalty"].sum())

    def bin_mae(label: str) -> float:
        subset = group.loc[group["rul_bin"].astype(str).eq(label), "abs_error"]
        return float(subset.mean()) if len(subset) else float("nan")

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
        "MAE_RUL_le_50": bin_mae("0-50"),
        "MAE_RUL_50_75": bin_mae("50-75"),
        "MAE_RUL_75_100": bin_mae("75-100"),
        "MAE_RUL_100_125": bin_mae("100-125"),
        "MAE_RUL_gt_125": bin_mae("125+"),
        "pred_min": float(group["pred_RUL"].min()),
        "pred_max": float(group["pred_RUL"].max()),
        "true_RUL_max": float(group["true_RUL"].max()),
        "mean_prefixes_used": float(group["n_prefixes_used"].mean()),
        "min_prefixes_used": int(group["n_prefixes_used"].min()),
        "max_prefixes_used": int(group["n_prefixes_used"].max()),
    }


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics_for_candidate(group) for _, group in predictions.groupby("candidate_name", sort=False)]
    table = pd.DataFrame(rows)
    order = {name: idx for idx, name in enumerate(CANDIDATES)}
    table["candidate_order"] = table["candidate_name"].map(order)
    return table.sort_values(["CMAPSS_mean", "dangerous_20_pct", "RMSE", "candidate_order"]).drop(columns=["candidate_order"])


def bin_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
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


def choose_best(metrics: pd.DataFrame) -> tuple[pd.Series, str, bool]:
    baseline = metrics.loc[metrics["candidate_name"].eq("baseline")].iloc[0]
    ranked = metrics.sort_values(["CMAPSS_mean", "dangerous_20_pct", "RMSE"]).reset_index(drop=True)
    best = ranked.iloc[0]
    if best["candidate_name"] == "baseline":
        return best, "Baseline remains best by CMAPSS_mean; smoothing is not recommended.", False
    cmaps_improves = best["CMAPSS_mean"] < baseline["CMAPSS_mean"] * 0.995
    danger_ok = best["dangerous_20_pct"] <= baseline["dangerous_20_pct"]
    rmse_ok = best["RMSE"] <= baseline["RMSE"] * 1.01
    low_rul_ok = best["MAE_RUL_le_50"] <= baseline["MAE_RUL_le_50"]
    enough_units = best["n_unique_units"] >= max(10, int(baseline["n_unique_units"] * 0.9))
    accepted = bool(cmaps_improves and danger_ok and rmse_ok and low_rul_ok and enough_units)
    if accepted:
        reason = "Accepted: CMAPSS_mean improves, dangerous_20 does not increase, RMSE and low-RUL MAE remain controlled."
    else:
        reason = (
            "Not accepted: best smoothing did not satisfy all adoption guards "
            f"(CMAPSS_improves={cmaps_improves}, danger_ok={danger_ok}, rmse_ok={rmse_ok}, "
            f"low_rul_ok={low_rul_ok}, enough_units={enough_units})."
        )
    return best, reason, accepted


def diagnostics(source: pd.DataFrame, predictions: pd.DataFrame, best_name: str, accepted: bool, reason: str) -> pd.DataFrame:
    baseline = predictions.loc[predictions["candidate_name"].eq("baseline")]
    best = predictions.loc[predictions["candidate_name"].eq(best_name)]
    merged = baseline[
        ["unit_id", "split_id", "cycle", "true_RUL", "pred_RUL", "eol_baseline", "cmapss_penalty"]
    ].merge(
        best[["unit_id", "split_id", "cycle", "pred_RUL", "eol_smoothed", "cmapss_penalty", "n_prefixes_used", "max_prefix_le_target"]],
        on=["unit_id", "split_id", "cycle"],
        suffixes=("_baseline", "_best"),
    )
    return pd.DataFrame(
        [
            {
                "dataset": source["dataset"].iloc[0],
                "best_candidate": best_name,
                "accepted": accepted,
                "reason": reason,
                "source_file": source["source_file"].iloc[0],
                "source_model_name": source["source_model_name"].iloc[0],
                "current_candidate": source["current_candidate"].iloc[0],
                "calibration_rule": source["calibration_rule"].iloc[0],
                "uses_final_test": False,
                "uses_only_existing_internal_validation_predictions": True,
                "no_future_prefixes_used": bool(predictions["max_prefix_le_target"].all()),
                "prefix_rule_requested": "t, t-20, t-40, t-60, t-80, t-100 when enough history",
                "prefix_rule_available": "existing artificial-cutoff prediction rows, observed spacing is 30 cycles",
                "prefix_spacing_mode": float(source.sort_values(["split_id", "unit_id", "cycle"]).groupby(["split_id", "unit_id"])["cycle"].diff().dropna().mode().iloc[0]),
                "n_eval_points": int(baseline.shape[0]),
                "n_unique_units": int(baseline["unit_id"].nunique()),
                "mean_prefixes_used": float(best["n_prefixes_used"].mean()),
                "min_prefixes_used": int(best["n_prefixes_used"].min()),
                "max_prefixes_used": int(best["n_prefixes_used"].max()),
                "n_points_with_prefix_history": int((best["n_prefixes_used"] > 1).sum()),
                "baseline_cmapss_total": float(merged["cmapss_penalty_baseline"].sum()),
                "best_cmapss_total": float(merged["cmapss_penalty_best"].sum()),
                "cmapss_total_delta": float(merged["cmapss_penalty_best"].sum() - merged["cmapss_penalty_baseline"].sum()),
                "eol_baseline_std": float(merged["eol_baseline"].std()),
                "eol_smoothed_std": float(merged["eol_smoothed"].std()),
            }
        ]
    )


def prediction_export(predictions: pd.DataFrame, best_name: str) -> pd.DataFrame:
    keep = predictions.loc[predictions["candidate_name"].isin(["baseline", best_name])].copy()
    return keep[
        [
            "dataset",
            "unit_id",
            "split_id",
            "cycle",
            "true_RUL",
            "candidate_name",
            "pred_RUL",
            "baseline_pred_RUL",
            "raw_pred_RUL",
            "error",
            "abs_error",
            "cmapss_penalty",
            "dangerous_any",
            "dangerous_10",
            "dangerous_20",
            "conservative",
            "rul_bin",
            "eol_baseline",
            "eol_smoothed",
            "n_prefixes_used",
            "prefix_cycles_used",
            "min_prefix_cycle",
            "max_prefix_cycle",
            "max_prefix_le_target",
            "source_model_name",
            "current_candidate",
            "calibration_rule",
        ]
    ]


def notebook_json(dataset: str) -> dict:
    lower = dataset.lower()
    title = f"{dataset} multi-prefix EOL smoothing v01"
    setup = dedent(
        f"""
        from pathlib import Path
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import Markdown, display

        PROJECT_ROOT = Path.cwd()
        for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
            if (candidate / "results" / "{dataset}" / "{lower}_multiprefix_eol_smoothing_results_v01.csv").exists():
                PROJECT_ROOT = candidate
                break

        results = pd.read_csv(PROJECT_ROOT / "results/{dataset}/{lower}_multiprefix_eol_smoothing_results_v01.csv")
        predictions = pd.read_csv(PROJECT_ROOT / "results/{dataset}/{lower}_multiprefix_predictions_v01.csv")
        best = pd.read_csv(PROJECT_ROOT / "results/{dataset}/{lower}_multiprefix_best_candidate_v01.csv")
        bins = pd.read_csv(PROJECT_ROOT / "results/{dataset}/{lower}_multiprefix_metrics_by_rul_bin_v01.csv")
        diagnostics = pd.read_csv(PROJECT_ROOT / "results/{dataset}/{lower}_multiprefix_diagnostics_v01.csv")
        """
    ).strip()
    plot_code = dedent(
        """
        display(results.sort_values("CMAPSS_mean"))
        display(diagnostics)

        best_name = best.loc[0, "candidate_name"]
        baseline_pred = predictions[predictions["candidate_name"].eq("baseline")].copy()
        best_pred = predictions[predictions["candidate_name"].eq(best_name)].copy()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(baseline_pred["true_RUL"], baseline_pred["pred_RUL"], s=14, alpha=0.5, label="baseline")
        ax.scatter(best_pred["true_RUL"], best_pred["pred_RUL"], s=14, alpha=0.5, label=best_name)
        lim = max(baseline_pred["true_RUL"].max(), baseline_pred["pred_RUL"].max(), best_pred["pred_RUL"].max())
        ax.plot([0, lim], [0, lim], color="black", linewidth=1)
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title("True RUL vs predicted RUL")
        ax.legend()
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(baseline_pred["true_RUL"], baseline_pred["error"], s=14, alpha=0.45, label="baseline")
        ax.scatter(best_pred["true_RUL"], best_pred["error"], s=14, alpha=0.45, label=best_name)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Pred - true")
        ax.set_title("Residual by true RUL")
        ax.legend()
        plt.show()

        compare_bins = bins[bins["candidate_name"].isin(["baseline", best_name])].copy()
        pivot_mae = compare_bins.pivot(index="rul_bin", columns="candidate_name", values="MAE")
        display(pivot_mae)
        pivot_mae.plot(kind="bar", figsize=(7, 4), title="MAE by RUL bin")
        plt.ylabel("MAE")
        plt.tight_layout()
        plt.show()

        pivot_share = compare_bins.pivot(index="rul_bin", columns="candidate_name", values="CMAPSS_share")
        display(pivot_share)
        pivot_share.plot(kind="bar", figsize=(7, 4), title="CMAPSS share by RUL bin")
        plt.ylabel("Share")
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 4))
        baseline_pred["eol_baseline"].hist(ax=ax, bins=25, alpha=0.45, label="baseline EOL")
        best_pred["eol_smoothed"].hist(ax=ax, bins=25, alpha=0.45, label=f"{best_name} smoothed EOL")
        ax.set_title("Estimated failure-cycle distribution")
        ax.set_xlabel("Estimated failure cycle")
        ax.legend()
        plt.show()

        example_units = best_pred.sort_values("cmapss_penalty", ascending=False)[["split_id", "unit_id"]].drop_duplicates().head(4)
        for _, row in example_units.iterrows():
            mask = best_pred["split_id"].eq(row["split_id"]) & best_pred["unit_id"].eq(row["unit_id"])
            bmask = baseline_pred["split_id"].eq(row["split_id"]) & baseline_pred["unit_id"].eq(row["unit_id"])
            unit_best = best_pred.loc[mask].sort_values("cycle")
            unit_base = baseline_pred.loc[bmask].sort_values("cycle")
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(unit_base["cycle"], unit_base["true_RUL"], marker="o", label="true RUL")
            ax1.plot(unit_base["cycle"], unit_base["pred_RUL"], marker="o", label="baseline pred")
            ax1.plot(unit_best["cycle"], unit_best["pred_RUL"], marker="o", label=f"{best_name} pred")
            ax1.set_xlabel("cycle")
            ax1.set_ylabel("RUL")
            ax2 = ax1.twinx()
            ax2.plot(unit_best["cycle"], unit_best["eol_smoothed"], linestyle="--", color="tab:red", label="smoothed EOL")
            ax2.set_ylabel("Estimated failure cycle")
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="best")
            ax1.set_title(f"Unit {int(row['unit_id'])} split {int(row['split_id'])}")
            plt.tight_layout()
            plt.show()

        top_baseline = baseline_pred.sort_values("cmapss_penalty", ascending=False).head(10)
        top_best = best_pred.sort_values("cmapss_penalty", ascending=False).head(10)
        display(Markdown("### Top 10 CMAPSS baseline"))
        display(top_baseline[["unit_id", "split_id", "cycle", "true_RUL", "pred_RUL", "error", "cmapss_penalty", "n_prefixes_used"]])
        display(Markdown(f"### Top 10 CMAPSS {best_name}"))
        display(top_best[["unit_id", "split_id", "cycle", "true_RUL", "pred_RUL", "error", "cmapss_penalty", "n_prefixes_used"]])
        """
    ).strip()
    conclusion = dedent(
        """
        row = best.iloc[0]
        diag = diagnostics.iloc[0]
        baseline = results[results["candidate_name"].eq("baseline")].iloc[0]
        display(Markdown(
            f\"\"\"### Conclusion v01

            - Mejor candidato encontrado: `{row['candidate_name']}`.
            - Comparacion contra baseline: CMAPSS_mean {baseline['CMAPSS_mean']:.4f} -> {row['CMAPSS_mean']:.4f}; dangerous_20 {baseline['dangerous_20_pct']:.2f}% -> {row['dangerous_20_pct']:.2f}%; RMSE {baseline['RMSE']:.4f} -> {row['RMSE']:.4f}.
            - Recomendacion: {'adoptar' if bool(row['accepted']) else 'no adoptar todavia'}.
            - Riesgos metodologicos: esta version usa prefijos ya disponibles en cortes artificiales internos, con spacing observado de 30 ciclos, no una grilla exacta t-20.
            - Confirmacion: no se uso test final.
            - Confirmacion: para cada ciclo t solo se usaron prefijos del mismo motor/split con ciclo <= t. Check no_future_prefixes_used={bool(diag['no_future_prefixes_used'])}.
            \"\"\"
        ))
        """
    ).strip()
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                f"# {title}\n\n"
                "This notebook evaluates multi-prefix / EOL smoothing using existing internal-validation artificial-cutoff predictions only. "
                "It does not use final test data and does not train a new large model.\n"
            ).splitlines(True),
        },
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": (setup + "\n").splitlines(True)},
        {"cell_type": "markdown", "metadata": {}, "source": ["## Candidate comparison and diagnostics\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": (plot_code + "\n").splitlines(True)},
        {"cell_type": "markdown", "metadata": {}, "source": ["## Final decision cell\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": (conclusion + "\n").splitlines(True)},
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


def run_dataset(dataset: str) -> None:
    lower = dataset.lower()
    source = normalize_source(dataset)
    predictions = smoothed_predictions(source)
    metrics = metrics_table(predictions)
    best, reason, accepted = choose_best(metrics)
    metrics["selected_best_candidate"] = metrics["candidate_name"].eq(best["candidate_name"])
    metrics["accepted_for_adoption"] = False
    metrics.loc[metrics["candidate_name"].eq(best["candidate_name"]), "accepted_for_adoption"] = accepted
    metrics["selection_reason"] = ""
    metrics.loc[metrics["candidate_name"].eq(best["candidate_name"]), "selection_reason"] = reason

    best_frame = metrics.loc[metrics["candidate_name"].eq(best["candidate_name"])].copy()
    best_frame["baseline_candidate"] = "baseline"
    best_frame["baseline_CMAPSS_mean"] = float(metrics.loc[metrics["candidate_name"].eq("baseline"), "CMAPSS_mean"].iloc[0])
    best_frame["baseline_RMSE"] = float(metrics.loc[metrics["candidate_name"].eq("baseline"), "RMSE"].iloc[0])
    best_frame["baseline_dangerous_20_pct"] = float(metrics.loc[metrics["candidate_name"].eq("baseline"), "dangerous_20_pct"].iloc[0])
    best_frame["accepted"] = accepted
    best_frame["reason"] = reason

    safe_to_csv(metrics, f"results/{dataset}/{lower}_multiprefix_eol_smoothing_results_v01.csv")
    safe_to_csv(prediction_export(predictions, str(best["candidate_name"])), f"results/{dataset}/{lower}_multiprefix_predictions_v01.csv")
    safe_to_csv(best_frame, f"results/{dataset}/{lower}_multiprefix_best_candidate_v01.csv")
    safe_to_csv(bin_metrics(predictions), f"results/{dataset}/{lower}_multiprefix_metrics_by_rul_bin_v01.csv")
    safe_to_csv(
        diagnostics(source, predictions, str(best["candidate_name"]), accepted, reason),
        f"results/{dataset}/{lower}_multiprefix_diagnostics_v01.csv",
    )
    safe_write_text(
        f"notebooks/{dataset}/27_{lower}_multiprefix_eol_smoothing_v01.ipynb",
        json.dumps(notebook_json(dataset), indent=2, ensure_ascii=False) + "\n",
    )


def main() -> None:
    ensure_dirs()
    run_dataset("FD002")
    run_dataset("FD004")
    print("Multi-prefix EOL smoothing v01 artifacts created.")


if __name__ == "__main__":
    main()
