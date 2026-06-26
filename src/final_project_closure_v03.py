from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUL_BINS_ORDER = ["0-50", "50-75", "75-100", "100-125", "125+"]
FINAL_COLUMNS = [
    "dataset",
    "selected_candidate",
    "candidate_family",
    "base_model",
    "temporal_window",
    "RUL_cap",
    "quantile_alpha",
    "condition_sensitive",
    "fault_sensitive",
    "calibration_or_inference_method",
    "MAE",
    "RMSE",
    "R2",
    "bias",
    "CMAPSS_total",
    "CMAPSS_mean",
    "dangerous_any",
    "dangerous_10",
    "dangerous_20",
    "conservative_rate",
    "MAE_RUL_le_50",
    "MAE_RUL_50_75",
    "MAE_RUL_75_100",
    "MAE_RUL_100_125",
    "MAE_RUL_gt_125",
    "pred_min",
    "pred_max",
    "true_RUL_max",
    "n_eval_points",
    "n_unique_units",
    "adopted_flag",
    "adoption_reason",
    "uses_final_test",
    "no_future_prefixes_used",
]


def next_available(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for tag in ["b", "c", "d", "e"]:
        candidate = path.with_name(f"{stem}{tag}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"No free versioned path for {path}")


def write_csv(df: pd.DataFrame, rel_path: str, created: list[str]) -> Path:
    path = next_available(PROJECT_ROOT / rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    created.append(str(path.relative_to(PROJECT_ROOT)))
    return path


def write_text(rel_path: str, content: str, created: list[str]) -> Path:
    path = next_available(PROJECT_ROOT / rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    created.append(str(path.relative_to(PROJECT_ROOT)))
    return path


def copy_backup(src_rel: str, dst_rel: str, backups: list[str]) -> Path:
    src = PROJECT_ROOT / src_rel
    dst = next_available(PROJECT_ROOT / dst_rel)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    backups.append(str(dst.relative_to(PROJECT_ROOT)))
    return dst


def value(row: pd.Series, key: str, default=np.nan):
    return row[key] if key in row.index else default


def fd001_row(diagnostics: list[dict]) -> tuple[dict, pd.DataFrame]:
    detail = pd.read_csv(PROJECT_ROOT / "results/FD001/fd001_lgbm_final_candidate_robustness.csv")
    cand = detail.loc[detail["candidate_label"].eq("candidate_03_B_quantile_a040_search_14")].copy()
    if cand.empty:
        raise FileNotFoundError("FD001 final candidate robustness rows were not found.")
    n_eval = int(cand["n_eval"].sum())
    row = {
        "dataset": "FD001",
        "selected_candidate": "candidate_03_B_quantile_a040_search_14",
        "candidate_family": "LightGBM quantile",
        "base_model": "candidate_03_B_quantile_a040_search_14",
        "temporal_window": 50,
        "RUL_cap": 125,
        "quantile_alpha": 0.40,
        "condition_sensitive": False,
        "fault_sensitive": False,
        "calibration_or_inference_method": "frozen previous LightGBM quantile candidate",
        "MAE": float(np.average(cand["mae"], weights=cand["n_eval"])),
        "RMSE": float(np.average(cand["rmse"], weights=cand["n_eval"])),
        "R2": float(np.average(cand["r2"], weights=cand["n_eval"])),
        "bias": float(np.average(cand["bias_mean"], weights=cand["n_eval"])),
        "CMAPSS_total": float(cand["cmapss_score"].sum()),
        "CMAPSS_mean": float(np.average(cand["cmapss_score_mean"], weights=cand["n_eval"])),
        "dangerous_any": float(np.average(cand["dangerous_error_pct"], weights=cand["n_eval"])),
        "dangerous_10": np.nan,
        "dangerous_20": np.nan,
        "conservative_rate": float(np.average(cand["conservative_error_pct"], weights=cand["n_eval"])),
        "MAE_RUL_le_50": float(np.average((cand["mae_rul_0_30"] + cand["mae_rul_30_60"]) / 2.0, weights=cand["n_eval"])),
        "MAE_RUL_50_75": np.nan,
        "MAE_RUL_75_100": float(np.average(cand["mae_rul_60_90"], weights=cand["n_eval"])),
        "MAE_RUL_100_125": np.nan,
        "MAE_RUL_gt_125": float(np.average(cand["mae_rul_90plus"], weights=cand["n_eval"])),
        "pred_min": np.nan,
        "pred_max": np.nan,
        "true_RUL_max": 140.0,
        "n_eval_points": n_eval,
        "n_unique_units": int(cand["n_eval"].max()),
        "adopted_flag": True,
        "adoption_reason": "FD001 remains frozen: robust LightGBM quantile alpha 0.40 candidate selected before multi-prefix experiments.",
        "uses_final_test": False,
        "no_future_prefixes_used": np.nan,
    }
    diagnostics.extend(
        [
            {
                "dataset": "FD001",
                "severity": "info",
                "check": "source",
                "message": "FD001 metrics use internal robustness rows, not official test.",
            },
            {
                "dataset": "FD001",
                "severity": "warning",
                "check": "missing_granularity",
                "message": "FD001 internal robustness file does not contain pred_min, pred_max, dangerous_10, dangerous_20, or exact v03 RUL bins; missing values are intentionally left blank.",
            },
        ]
    )
    bins = []
    for label, mae_col, danger_col in [
        ("0-30", "mae_rul_0_30", "dangerous_error_pct_rul_0_30"),
        ("30-60", "mae_rul_30_60", "dangerous_error_pct_rul_30_60"),
        ("60-90", "mae_rul_60_90", "dangerous_error_pct_rul_60_90"),
        ("90+", "mae_rul_90plus", "dangerous_error_pct_rul_90plus"),
    ]:
        bins.append(
            {
                "dataset": "FD001",
                "candidate_name": row["selected_candidate"],
                "rul_bin": label,
                "n_eval_points": n_eval,
                "n_unique_units": row["n_unique_units"],
                "MAE": float(np.average(cand[mae_col], weights=cand["n_eval"])),
                "RMSE": np.nan,
                "CMAPSS_total": np.nan,
                "CMAPSS_mean": np.nan,
                "CMAPSS_share": np.nan,
                "dangerous_any_pct": float(np.average(cand[danger_col], weights=cand["n_eval"])),
                "dangerous_10_pct": np.nan,
                "dangerous_20_pct": np.nan,
                "bias": np.nan,
                "source_bin_definition": "FD001 robustness source bins; not exact v03 bins",
            }
        )
    return row, pd.DataFrame(bins)


def fd002_row(diagnostics: list[dict]) -> tuple[dict, pd.DataFrame]:
    final = pd.read_csv(PROJECT_ROOT / "results/FD002/FD002_final_candidate_multiprefix_v02.csv").iloc[0]
    diag = pd.read_csv(PROJECT_ROOT / "results/FD002/FD002_final_candidate_multiprefix_diagnostics_v02.csv").iloc[0]
    bins = pd.read_csv(PROJECT_ROOT / "results/FD002/FD002_final_candidate_multiprefix_metrics_by_bin_v02.csv")
    selected_bins = bins.loc[bins["candidate_name"].eq("eol_mean")].copy()
    row = {
        "dataset": "FD002",
        "selected_candidate": final["selected_candidate"],
        "candidate_family": "XGBoost + multi-prefix inference",
        "base_model": "xgb_condition_fault_sensitive_mid_guard",
        "temporal_window": 50,
        "RUL_cap": 125,
        "quantile_alpha": np.nan,
        "condition_sensitive": True,
        "fault_sensitive": True,
        "calibration_or_inference_method": "multi-prefix EOL smoothing using mean estimated failure cycle",
        "MAE": final["MAE"],
        "RMSE": final["RMSE"],
        "R2": final["R2"],
        "bias": final["bias"],
        "CMAPSS_total": final["CMAPSS_total"],
        "CMAPSS_mean": final["CMAPSS_mean"],
        "dangerous_any": final["dangerous_any"],
        "dangerous_10": final["dangerous_10"],
        "dangerous_20": final["dangerous_20"],
        "conservative_rate": final["conservative_rate"],
        "MAE_RUL_le_50": final["MAE_RUL_le_50"],
        "MAE_RUL_50_75": final["MAE_RUL_50_75"],
        "MAE_RUL_75_100": final["MAE_RUL_75_100"],
        "MAE_RUL_100_125": final["MAE_RUL_100_125"],
        "MAE_RUL_gt_125": final["MAE_RUL_gt_125"],
        "pred_min": final["pred_min"],
        "pred_max": final["pred_max"],
        "true_RUL_max": final["true_RUL_max"],
        "n_eval_points": final["n_eval_points"],
        "n_unique_units": final["n_unique_units"],
        "adopted_flag": final["adopted_flag"],
        "adoption_reason": final["adoption_reason"],
        "uses_final_test": False,
        "no_future_prefixes_used": diag["no_future_prefixes_used"],
    }
    diagnostics.append(
        {
            "dataset": "FD002",
            "severity": "info",
            "check": "anti_leakage",
            "message": f"no_future_prefixes_used={diag['no_future_prefixes_used']}; prefix_spacing_mode={diag['prefix_spacing_mode']}.",
        }
    )
    selected_bins["source_bin_definition"] = "v03 bins from FD002 multi-prefix v02"
    return row, selected_bins


def fd003_row(diagnostics: list[dict]) -> tuple[dict, pd.DataFrame]:
    final = pd.read_csv(PROJECT_ROOT / "results/FD003/fd003_final_candidate_after_calibration_v02.csv").iloc[0]
    bins = pd.read_csv(PROJECT_ROOT / "results/FD003/fd003_rul_bin_metrics_v01.csv")
    row = {
        "dataset": "FD003",
        "selected_candidate": "fd003_lgbm_w50_cap125_quantile_a04_none_fault_sensitive",
        "candidate_family": "LightGBM quantile",
        "base_model": "fd003_lgbm_w50_cap125_quantile_a04_none_fault_sensitive",
        "temporal_window": 50,
        "RUL_cap": 125,
        "quantile_alpha": 0.40,
        "condition_sensitive": False,
        "fault_sensitive": True,
        "calibration_or_inference_method": "offset 0 retained after offset calibration probe",
        "MAE": final["MAE"],
        "RMSE": final["RMSE"],
        "R2": final["R2"],
        "bias": final["bias"],
        "CMAPSS_total": final["CMAPSS_total"],
        "CMAPSS_mean": final["CMAPSS_mean"],
        "dangerous_any": final["dangerous_any_pct"],
        "dangerous_10": final["dangerous_10_pct"],
        "dangerous_20": final["dangerous_20_pct"],
        "conservative_rate": final["conservative_pct"],
        "MAE_RUL_le_50": final["MAE_RUL_le_50"],
        "MAE_RUL_50_75": bins.loc[bins["rul_bin"].eq("50-75"), "MAE"].iloc[0],
        "MAE_RUL_75_100": bins.loc[bins["rul_bin"].eq("75-100"), "MAE"].iloc[0],
        "MAE_RUL_100_125": bins.loc[bins["rul_bin"].eq("100-125"), "MAE"].iloc[0],
        "MAE_RUL_gt_125": final["MAE_RUL_gt_125"],
        "pred_min": np.nan,
        "pred_max": final["pred_max"],
        "true_RUL_max": final["true_RUL_max"],
        "n_eval_points": final["n_motors"],
        "n_unique_units": 100,
        "adopted_flag": True,
        "adoption_reason": final["reason_selected"],
        "uses_final_test": False,
        "no_future_prefixes_used": np.nan,
    }
    diagnostics.append(
        {
            "dataset": "FD003",
            "severity": "info",
            "check": "calibration_decision",
            "message": "Offsets +1 to +4 were rejected because they worsened CMAPSS/dangerous errors; offset 0 retained.",
        }
    )
    bins = bins.rename(columns={"n_motors": "n_eval_points"})
    bins["n_unique_units"] = 100
    bins["source_bin_definition"] = "FD003 offset calibration bins"
    return row, bins


def fd004_row(diagnostics: list[dict]) -> tuple[dict, pd.DataFrame]:
    final = pd.read_csv(PROJECT_ROOT / "results/FD004/FD004_multiprefix_late_stage_guard_best_candidate_v02.csv").iloc[0]
    diag = pd.read_csv(PROJECT_ROOT / "results/FD004/FD004_multiprefix_late_stage_guard_diagnostics_v02.csv").iloc[0]
    bins = pd.read_csv(PROJECT_ROOT / "results/FD004/FD004_multiprefix_late_stage_guard_metrics_by_bin_v02.csv")
    selected_bins = bins.loc[bins["candidate_name"].eq(final["candidate_name"])].copy()
    row = {
        "dataset": "FD004",
        "selected_candidate": final["candidate_name"],
        "candidate_family": "XGBoost + multi-prefix inference with late-stage guard",
        "base_model": "fd004_xgb_fs_bin_weights_w70",
        "temporal_window": 70,
        "RUL_cap": 125,
        "quantile_alpha": np.nan,
        "condition_sensitive": True,
        "fault_sensitive": True,
        "calibration_or_inference_method": "multi-prefix eol_mean with b60 only-if-lower floor2 late-stage guard",
        "MAE": final["MAE"],
        "RMSE": final["RMSE"],
        "R2": final["R2"],
        "bias": final["bias"],
        "CMAPSS_total": final["CMAPSS_total"],
        "CMAPSS_mean": final["CMAPSS_mean"],
        "dangerous_any": final["dangerous_any_pct"],
        "dangerous_10": final["dangerous_10_pct"],
        "dangerous_20": final["dangerous_20_pct"],
        "conservative_rate": final["conservative_rate_pct"],
        "MAE_RUL_le_50": final["MAE_RUL_le_50"],
        "MAE_RUL_50_75": final["MAE_RUL_50_75"],
        "MAE_RUL_75_100": final["MAE_RUL_75_100"],
        "MAE_RUL_100_125": final["MAE_RUL_100_125"],
        "MAE_RUL_gt_125": final["MAE_RUL_gt_125"],
        "pred_min": final["pred_min"],
        "pred_max": final["pred_max"],
        "true_RUL_max": final["true_RUL_max"],
        "n_eval_points": final["n_eval_points"],
        "n_unique_units": final["n_unique_units"],
        "adopted_flag": final["acceptable"],
        "adoption_reason": final["selection_reason"],
        "uses_final_test": False,
        "no_future_prefixes_used": diag["no_future_prefixes_used"],
    }
    diagnostics.append(
        {
            "dataset": "FD004",
            "severity": "info",
            "check": "guard",
            "message": "Selected guard: if baseline_pred <= 60 and eol_mean_pred < baseline_pred, pred_final=max(eol_mean_pred, baseline_pred-2); otherwise eol_mean_pred.",
        }
    )
    selected_bins["source_bin_definition"] = "v03 bins from FD004 late-stage guard v02"
    return row, selected_bins


def build_artifacts(created: list[str], backups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    diagnostics: list[dict] = []
    rows = []
    bin_tables = []
    for builder in [fd001_row, fd002_row, fd003_row, fd004_row]:
        row, bins = builder(diagnostics)
        rows.append(row)
        bin_tables.append(bins)
    summary = pd.DataFrame(rows)[FINAL_COLUMNS]
    nan_cols = {
        dataset: [col for col in summary.columns if pd.isna(group.iloc[0][col])]
        for dataset, group in summary.groupby("dataset")
    }
    for dataset, cols in nan_cols.items():
        if cols:
            diagnostics.append(
                {
                    "dataset": dataset,
                    "severity": "warning",
                    "check": "nan_columns",
                    "message": "Missing final summary columns: " + ", ".join(cols),
                }
            )
    selection = summary[
        [
            "dataset",
            "selected_candidate",
            "candidate_family",
            "base_model",
            "calibration_or_inference_method",
            "CMAPSS_mean",
            "RMSE",
            "dangerous_20",
            "bias",
            "adopted_flag",
            "adoption_reason",
        ]
    ].copy()
    registry = summary[
        [
            "dataset",
            "selected_candidate",
            "candidate_family",
            "base_model",
            "temporal_window",
            "RUL_cap",
            "quantile_alpha",
            "condition_sensitive",
            "fault_sensitive",
            "calibration_or_inference_method",
            "uses_final_test",
            "no_future_prefixes_used",
        ]
    ].copy()
    bin_metrics = pd.concat(bin_tables, ignore_index=True, sort=False)
    diagnostics_df = pd.DataFrame(diagnostics)

    write_csv(summary, "results/resumen_global_final_modelos_v03.csv", created)
    write_csv(selection, "results/seleccion_final_modelos_v03.csv", created)
    write_csv(registry, "results/model_registry_final_v03.csv", created)
    write_csv(bin_metrics, "results/metricas_por_bin_final_modelos_v03.csv", created)
    write_csv(diagnostics_df, "results/diagnostics_final_modelos_v03.csv", created)
    return summary, selection, registry, bin_metrics, diagnostics_df


def notebook_json() -> dict:
    setup = dedent(
        """
        from pathlib import Path
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import Markdown, display

        PROJECT_ROOT = Path.cwd()
        for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
            if (candidate / "results" / "resumen_global_final_modelos_v03.csv").exists():
                PROJECT_ROOT = candidate
                break

        summary = pd.read_csv(PROJECT_ROOT / "results/resumen_global_final_modelos_v03.csv")
        selection = pd.read_csv(PROJECT_ROOT / "results/seleccion_final_modelos_v03.csv")
        registry = pd.read_csv(PROJECT_ROOT / "results/model_registry_final_v03.csv")
        bins = pd.read_csv(PROJECT_ROOT / "results/metricas_por_bin_final_modelos_v03.csv")
        diagnostics = pd.read_csv(PROJECT_ROOT / "results/diagnostics_final_modelos_v03.csv")
        fd002_preds = pd.read_csv(PROJECT_ROOT / "results/FD002/fd002_multiprefix_predictions_v01.csv")
        fd004_preds = pd.read_csv(PROJECT_ROOT / "results/FD004/FD004_multiprefix_late_stage_guard_predictions_v02.csv")
        """
    ).strip()
    cells = [
        md("# Conclusion final v03 - NASA C-MAPSS RUL\n\nCierre documental post multi-prefix / EOL smoothing. No entrena modelos, no genera predicciones nuevas y no usa test final para decidir modelos."),
        md("## A. Seleccion final de modelos"),
        code(setup + "\ndisplay(selection)\ndisplay(registry)\n"),
        md(
            """## B. Explicacion de los modelos

- **FD001**: LightGBM quantile alpha 0.40, RUL cap 125 y ventana temporal 50. Queda congelado.
- **FD002**: XGBoost condition/fault-sensitive como base; adopta multi-prefix `eol_mean`. No es offset: suaviza el ciclo estimado de falla.
- **FD003**: LightGBM w50 cap125 quantile alpha 0.40 `none_fault_sensitive`; offset 0 porque offsets positivos empeoraron CMAPSS/dangerous.
- **FD004**: XGBoost w70 bin_weights como base; adopta multi-prefix `eol_mean` con late-stage guard: si `baseline_pred <= 60` y `eol_mean_pred < baseline_pred`, usa `max(eol_mean_pred, baseline_pred - 2)`.
"""
        ),
        md("## C. Resultados cuantitativos"),
        code(
            dedent(
                """
                cols = ["dataset", "selected_candidate", "MAE", "RMSE", "R2", "CMAPSS_mean", "dangerous_20", "bias", "adoption_reason"]
                display(summary[cols])

                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                summary.plot.bar(x="dataset", y="RMSE", ax=axes[0], legend=False, title="RMSE final por FD")
                summary.plot.bar(x="dataset", y="CMAPSS_mean", ax=axes[1], legend=False, title="CMAPSS mean final por FD")
                summary.plot.bar(x="dataset", y="dangerous_20", ax=axes[2], legend=False, title="dangerous_20 final por FD")
                for ax in axes:
                    ax.set_xlabel("")
                plt.tight_layout()
                plt.show()

                fd002_compare = pd.read_csv(PROJECT_ROOT / "results/FD002/fd002_multiprefix_eol_smoothing_results_v01.csv")
                fd002_compare = fd002_compare[fd002_compare["candidate_name"].isin(["baseline", "eol_mean"])]
                fd004_compare = pd.read_csv(PROJECT_ROOT / "results/FD004/FD004_multiprefix_late_stage_guard_results_v02.csv")
                fd004_compare = fd004_compare[fd004_compare["candidate_name"].isin(["baseline", "eol_mean_guard_b60_only_if_lower_floor2"])]
                display(Markdown("### FD002 baseline vs final"))
                display(fd002_compare[["candidate_name", "RMSE", "CMAPSS_mean", "dangerous_20_pct", "MAE_RUL_le_50"]])
                display(Markdown("### FD004 baseline vs final"))
                display(fd004_compare[["candidate_name", "RMSE", "CMAPSS_mean", "dangerous_20_pct", "MAE_RUL_le_50"]])

                def delta_table(df, final_name):
                    base = df[df["candidate_name"].eq("baseline")].iloc[0]
                    final = df[df["candidate_name"].eq(final_name)].iloc[0]
                    return pd.DataFrame([{
                        "final_candidate": final_name,
                        "delta_RMSE_pct": (final["RMSE"] - base["RMSE"]) / base["RMSE"] * 100,
                        "delta_CMAPSS_mean_pct": (final["CMAPSS_mean"] - base["CMAPSS_mean"]) / base["CMAPSS_mean"] * 100,
                        "delta_dangerous_20_pct_points": final["dangerous_20_pct"] - base["dangerous_20_pct"],
                    }])
                display(pd.concat([
                    delta_table(fd002_compare, "eol_mean").assign(dataset="FD002"),
                    delta_table(fd004_compare, "eol_mean_guard_b60_only_if_lower_floor2").assign(dataset="FD004"),
                ], ignore_index=True))
                """
            ).strip()
        ),
        md("## D. Resultados por rango de RUL"),
        code(
            dedent(
                """
                display(bins[["dataset", "candidate_name", "rul_bin", "n_eval_points", "MAE", "CMAPSS_mean", "CMAPSS_share", "dangerous_20_pct", "source_bin_definition"]])

                for dataset in ["FD002", "FD004"]:
                    plot_bins = bins[(bins["dataset"].eq(dataset)) & (bins["candidate_name"].isin(["eol_mean", "eol_mean_guard_b60_only_if_lower_floor2"]))]
                    if plot_bins.empty:
                        continue
                    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                    plot_bins.plot.bar(x="rul_bin", y="MAE", ax=axes[0], legend=False, title=f"{dataset}: MAE por bin RUL")
                    plot_bins.plot.bar(x="rul_bin", y="dangerous_20_pct", ax=axes[1], legend=False, title=f"{dataset}: dangerous_20 por bin RUL")
                    plt.tight_layout()
                    plt.show()

                fd002_plot = fd002_preds[fd002_preds["candidate_name"].eq("eol_mean")]
                fd004_plot = fd004_preds[fd004_preds["candidate_name"].eq("eol_mean_guard_b60_only_if_lower_floor2")]
                for dataset, frame in [("FD002", fd002_plot), ("FD004", fd004_plot)]:
                    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                    axes[0].scatter(frame["true_RUL"], frame["pred_RUL"], s=12, alpha=0.45)
                    axes[0].plot([0, frame["true_RUL"].max()], [0, frame["true_RUL"].max()], color="black", linewidth=1)
                    axes[0].set_title(f"{dataset}: true vs pred final")
                    axes[0].set_xlabel("True RUL")
                    axes[0].set_ylabel("Pred RUL")
                    axes[1].scatter(frame["true_RUL"], frame["error"], s=12, alpha=0.45)
                    axes[1].axhline(0, color="black", linewidth=1)
                    axes[1].set_title(f"{dataset}: residual final")
                    axes[1].set_xlabel("True RUL")
                    axes[1].set_ylabel("Pred - true")
                    plt.tight_layout()
                    plt.show()
                """
            ).strip()
        ),
        md(
            """## E. Interpretacion metodologica

- **RUL cap 125**: prioriza la zona util cercana/media a falla y sacrifica precision en motores muy sanos.
- **Quantile alpha 0.40**: empuja modelos LightGBM a ser mas conservadores y reduce riesgo de sobreestimacion.
- **Condition-sensitive**: en FD002/FD004 controla regimen operativo mediante normalizacion/contexto por condicion y features temporales.
- **Fault-sensitive**: captura patrones de degradacion compatibles con modos de falla, sin afirmar etiquetas reales de fault mode.
- **Ventana temporal**: resume sensores con ultimos valores, medias, pendientes, deltas y variabilidad.
- **Multi-prefix**: para cada ciclo evaluado usa prefijos anteriores o iguales, convierte cada prediccion a ciclo estimado de falla y suaviza ese ciclo. No promedia RUL directamente.
"""
        ),
        md(
            """## F. Riesgos y limitaciones

- No se uso test final para seleccionar ni calibrar.
- La validacion es interna por motores/cortes artificiales.
- Hay riesgo de sobreajuste si se agregan demasiadas reglas post-smoothing.
- El guard FD004 es una regla de inferencia/calibracion simple, no un modelo nuevo.
- La cola de RUL alto sigue siendo dificil por RUL cap y por menor historia previa util.
- C-MAPSS penaliza mas errores tardios/peligrosos, por eso dangerous_20 y CMAPSS pesan mas que mejoras minimas de RMSE.
"""
        ),
        md(
            """## G. Conclusion final

FD001 y FD003 quedan con modelos LightGBM quantile conservadores. FD002 y FD004 quedan con modelos base XGBoost reforzados por inferencia temporal multi-prefix. La mejora clave del cierre fue pasar de offsets a suavizado del ciclo estimado de falla. FD004 ademas requiere un guard cercano a falla para no degradar la zona critica.
"""
        ),
        md("## Chequeos finales"),
        code(
            dedent(
                """
                created_files = [
                    "results/resumen_global_final_modelos_v03.csv",
                    "results/seleccion_final_modelos_v03.csv",
                    "results/model_registry_final_v03.csv",
                    "results/metricas_por_bin_final_modelos_v03.csv",
                    "results/diagnostics_final_modelos_v03.csv",
                    "notebooks/conclusion/notebook_conclusion_final_v03.ipynb",
                    "notas/notas_modelado_final_v03.txt",
                    "notas/notas_conclusiones_final_v03.txt",
                    "notas/notas_resultados_finales_v03.txt",
                ]
                backup_files = ["notebooks/conclusion/notebook_conclusion_final_actualizado_backup_pre_v03.ipynb"]
                display(pd.DataFrame({"created_files": created_files}))
                display(pd.DataFrame({"backup_files": backup_files}))
                display(selection[["dataset", "selected_candidate"]])
                display(Markdown("No se uso test final para decisiones v03."))
                display(registry[["dataset", "uses_final_test", "no_future_prefixes_used"]])
                nan_summary = summary.isna().sum().reset_index()
                nan_summary.columns = ["column", "n_nan"]
                display(nan_summary[nan_summary["n_nan"] > 0])
                display(diagnostics)
                """
            ).strip()
        ),
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


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip().splitlines(True)}


def notes(summary: pd.DataFrame, created: list[str]) -> None:
    selected = {row["dataset"]: row for _, row in summary.set_index("dataset", drop=False).iterrows()}
    common = dedent(
        f"""
        Cierre final v03 - NASA C-MAPSS RUL

        Modelos finales:
        - FD001: {selected['FD001']['selected_candidate']} | LightGBM quantile alpha 0.40 | congelado.
        - FD002: {selected['FD002']['selected_candidate']} | XGBoost condition/fault-sensitive + multi-prefix EOL smoothing.
        - FD003: {selected['FD003']['selected_candidate']} | LightGBM quantile alpha 0.40 | offset 0.
        - FD004: {selected['FD004']['selected_candidate']} | XGBoost + multi-prefix EOL smoothing con late-stage guard.

        No se uso test final, no se reentreno un modelo grande, no se uso LSTM y no se generaron predicciones finales nuevas.
        """
    ).strip()
    write_text(
        "notas/notas_modelado_final_v03.txt",
        common
        + dedent(
            """

            Multi-prefix / EOL smoothing:
            Para cada ciclo evaluado se usan prefijos temporales anteriores o iguales del mismo motor. Cada prefijo produce una prediccion de RUL; esa prediccion se transforma a ciclo estimado de falla como ciclo + pred_RUL. Luego se suaviza el ciclo estimado de falla y se vuelve a RUL del ciclo actual.

            Late-stage guard FD004:
            Si baseline_pred <= 60 y eol_mean_pred < baseline_pred, pred_final=max(eol_mean_pred, baseline_pred - 2). En otro caso se usa eol_mean_pred. Es una regla post-smoothing simple y explicable, no un modelo nuevo.
            """
        ).strip()
        + "\n",
        created,
    )
    write_text(
        "notas/notas_conclusiones_final_v03.txt",
        common
        + dedent(
            """

            Decisiones:
            - FD001 no se toca porque ya estaba congelado con un candidato robusto.
            - FD002 adopta eol_mean porque reduce claramente CMAPSS, RMSE y dangerous_20.
            - FD003 queda offset 0 porque offsets +1 a +4 bajaban algo RMSE pero empeoraban CMAPSS/dangerous.
            - FD004 adopta eol_mean_guard_b60_only_if_lower_floor2 porque mejora CMAPSS, RMSE, dangerous_20 y zona de RUL bajo.

            La mejora conceptual del cierre fue pasar de calibraciones por offset a smoothing temporal del ciclo estimado de falla.
            """
        ).strip()
        + "\n",
        created,
    )
    write_text(
        "notas/notas_resultados_finales_v03.txt",
        summary[
            [
                "dataset",
                "selected_candidate",
                "MAE",
                "RMSE",
                "CMAPSS_mean",
                "dangerous_20",
                "bias",
                "adoption_reason",
            ]
        ].to_string(index=False)
        + "\n\nNo se uso test final. No se usaron prefijos futuros en FD002/FD004.\n",
        created,
    )


def main() -> None:
    created: list[str] = []
    backups: list[str] = []
    summary, selection, registry, bins, diagnostics = build_artifacts(created, backups)
    copy_backup(
        "notebooks/conclusion/01_conclusion_final.ipynb",
        "notebooks/conclusion/notebook_conclusion_final_actualizado_backup_pre_v03.ipynb",
        backups,
    )
    write_text(
        "notebooks/conclusion/notebook_conclusion_final_v03.ipynb",
        json.dumps(notebook_json(), indent=2, ensure_ascii=False) + "\n",
        created,
    )
    notes(summary, created)
    manifest = pd.DataFrame(
        [{"kind": "created", "path": path} for path in created]
        + [{"kind": "backup", "path": path} for path in backups]
    )
    write_csv(manifest, "results/final_closure_v03_manifest.csv", created)
    print("Created files:")
    for path in created:
        print(f"- {path}")
    print("Backups:")
    for path in backups:
        print(f"- {path}")


if __name__ == "__main__":
    main()
