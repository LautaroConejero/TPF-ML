from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import add_train_rul, last_cycle_rows, load_cmapss_subset
from src.fd001_experiment_utils import make_lgbm_from_search_params, weights_from_scheme
from src.fd001_modeling import metrics_by_model, metrics_by_rul_bin, prediction_frame, regression_metrics
from src.fd003_improvement_utils import (
    FD003_RELEVANT_SENSORS,
    make_fault_sensitive_feature_frame,
    make_lgbm_regressor_from_config,
    sample_weights_for_scheme,
)
from src.fd003_transfer_utils import fd003_feature_columns
from src.preprocessed_FD001 import (
    make_temporal_features,
    prepare_fd001_temporal_full_train_for_test,
    temporal_feature_columns,
)


CONCLUSION_DIR = PROJECT_ROOT / "conclusion"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=False)


def normalize_metric_row(dataset: str, scope: str, metrics: pd.DataFrame) -> dict:
    row = metrics.iloc[0].to_dict()
    return {
        "dataset": dataset,
        "evaluation_scope": scope,
        "model_name": row.get("model_name", ""),
        "representation": row.get("representation", ""),
        "n": int(row.get("n_test", row.get("n_eval", row.get("n_predictions", 0)))),
        "mae": row.get("mae", row.get("mean_MAE", np.nan)),
        "rmse": row.get("rmse", row.get("mean_RMSE", np.nan)),
        "r2": row.get("r2", row.get("mean_R2", np.nan)),
        "cmapss_score": row.get(
            "cmapss_score",
            row.get("mean_CMAPSS", row.get("cmapss_total_all_predictions", np.nan)),
        ),
        "cmapss_score_mean": row.get(
            "cmapss_score_mean",
            row.get("cmapss_mean_penalty", np.nan),
        ),
        "dangerous_error_pct": row.get("dangerous_error_pct", row.get("mean_dangerous_error_pct", np.nan)),
    }


def generate_fd001_final_official(random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = read_json(CONFIGS_DIR / "FD001" / "fd001_final_quantile_candidate_notebook18_config.json")
    prepared = prepare_fd001_temporal_full_train_for_test(
        data_dir="CMAPSSData",
        max_rul=int(config["rul_cap"]),
        window_size=int(config["window_size"]),
    )
    model_config = {
        "objective": config["objective"],
        "alpha": config.get("alpha"),
        "sample_weight_scheme": config.get("sample_weight_scheme", "none"),
        "params": config["hyperparameters"],
    }
    model = make_lgbm_from_search_params(model_config, random_state=random_state)
    weights = weights_from_scheme(prepared["y_train_raw"], config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(prepared["X_train"], prepared["y_train"])
    else:
        model.fit(prepared["X_train"], prepared["y_train"], sample_weight=weights)

    predictions = prediction_frame(
        prepared["test_last_df"],
        model.predict(prepared["X_test_last"]),
        model_name=config["model_name"],
        representation=f"temporal_w{config['window_size']}",
    )
    predictions["dataset"] = "FD001"
    predictions["model_family"] = config["model_family"]
    predictions["sample_weight_scheme"] = config.get("sample_weight_scheme", "none")
    predictions["official_test_used_for_selection"] = False
    metrics = metrics_by_model(predictions)
    metrics.insert(0, "dataset", "FD001")
    metrics["n_test"] = metrics.pop("n_eval")

    out_dir = RESULTS_DIR / "FD001"
    predictions.to_csv(out_dir / "fd001_final_quantile_official_test_predictions.csv", index=False)
    metrics.to_csv(out_dir / "fd001_final_quantile_official_test_metrics.csv", index=False)
    metrics_by_rul_bin(predictions).to_csv(
        out_dir / "fd001_final_quantile_official_test_metrics_by_rul_bin.csv",
        index=False,
    )
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_DIR / "fd001_final_quantile_best_model_predictions.csv", index=False)
    predictions.to_csv(PREDICTIONS_DIR / "fd001_best_model_predictions.csv", index=False)
    metrics.to_csv(out_dir / "fd001_official_test_metrics.csv", index=False)
    write_json(
        CONFIGS_DIR / "FD001" / "fd001_best_model_config.json",
        {
            "dataset": "FD001",
            "task": "Remaining Useful Life regression",
            "selection_policy": config["selection_basis"],
            "official_test_used_for_selection": False,
            "preprocessing": {
                "rul_raw_definition": "max_cycle - cycle for train; RUL_FD001.txt for official test last cycle",
                "training_target": "RUL capped",
                "metric_target": "RUL_raw uncapped",
                "rul_cap": int(config["rul_cap"]),
                "representation": "temporal",
                "window_size": int(config["window_size"]),
                "dropped_columns": list(prepared["dropped_columns"]),
                "n_features": int(len(prepared["feature_columns"])),
                "feature_columns": list(prepared["feature_columns"]),
            },
            "validation": {
                "protocol": config["validation_protocol"],
                "robustness_random_states": list(config["robustness_random_states"]),
                "validation_summary": config["validation_summary"],
            },
            "final_model": {
                "model_name": config["model_name"],
                "model_family": config["model_family"],
                "objective": config["objective"],
                "alpha": config.get("alpha"),
                "sample_weight_scheme": config.get("sample_weight_scheme", "none"),
                "hyperparameters": config["hyperparameters"],
            },
            "official_test_metrics": metrics.iloc[0].to_dict(),
            "artifacts": {
                "predictions": "predictions/fd001_best_model_predictions.csv",
                "explicit_final_predictions": "predictions/fd001_final_quantile_best_model_predictions.csv",
                "official_test_metrics": "results/FD001/fd001_official_test_metrics.csv",
                "explicit_final_metrics": "results/FD001/fd001_final_quantile_official_test_metrics.csv",
            },
        },
    )
    return predictions, metrics


def prepare_fd003_temporal_full_train_for_test(
    data_dir: str = "CMAPSSData",
    rul_cap: int = 125,
    window_size: int = 50,
):
    data = load_cmapss_subset("FD003", data_dir=PROJECT_ROOT / data_dir)
    train = add_train_rul(data.train, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    train["RUL"] = train["RUL"].clip(upper=rul_cap)

    base_feature_columns, dropped_columns = fd003_feature_columns(train)
    train_temporal = make_temporal_features(
        train,
        endpoints_df=train,
        feature_columns=base_feature_columns,
        window_size=window_size,
    )
    test_last = last_cycle_rows(data.test).merge(data.rul, on="unit", how="left")
    test_last = test_last.rename(columns={"final_rul": "RUL_raw"})
    test_last["RUL"] = test_last["RUL_raw"].clip(upper=rul_cap)
    test_temporal = make_temporal_features(
        data.test,
        endpoints_df=test_last,
        feature_columns=base_feature_columns,
        window_size=window_size,
    )
    feature_columns = temporal_feature_columns(base_feature_columns)

    train_extra = make_fault_sensitive_feature_frame(
        train,
        train,
        [sensor for sensor in FD003_RELEVANT_SENSORS if sensor in base_feature_columns],
        window_size,
    )
    test_extra = make_fault_sensitive_feature_frame(
        data.test,
        test_last,
        [sensor for sensor in FD003_RELEVANT_SENSORS if sensor in base_feature_columns],
        window_size,
    )
    extra_columns = [column for column in train_extra.columns if column not in {"unit", "cycle"}]
    train_temporal = train_temporal.merge(train_extra, on=["unit", "cycle"], how="left")
    test_temporal = test_temporal.merge(test_extra, on=["unit", "cycle"], how="left")
    feature_columns = feature_columns + extra_columns

    from sklearn.preprocessing import StandardScaler

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
        "base_feature_columns": base_feature_columns,
        "dropped_columns": dropped_columns,
        "feature_columns": feature_columns,
        "extra_feature_columns": extra_columns,
        "train_df": train_temporal,
        "test_last_df": test_temporal,
        "X_train": x_train,
        "y_train": train_temporal["RUL"],
        "y_train_raw": train_temporal["RUL_raw"],
        "X_test_last": x_test,
    }


def generate_fd003_final_official(random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = read_json(CONFIGS_DIR / "FD003" / "fd003_final_candidate_config.json")
    prepared = prepare_fd003_temporal_full_train_for_test(
        rul_cap=int(config["rul_cap"]),
        window_size=int(config["window_size"]),
    )
    model = make_lgbm_regressor_from_config(config, random_state=random_state)
    weights = sample_weights_for_scheme(prepared["y_train_raw"], config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(prepared["X_train"], prepared["y_train"])
    else:
        model.fit(prepared["X_train"], prepared["y_train"], sample_weight=weights)

    predictions = prediction_frame(
        prepared["test_last_df"],
        model.predict(prepared["X_test_last"]),
        model_name=config["model_name"],
        representation=f"temporal_w{config['window_size']}_{config['feature_set']}",
    )
    predictions["dataset"] = "FD003"
    predictions["model_family"] = config["model_family"]
    predictions["feature_set"] = config["feature_set"]
    predictions["sample_weight_scheme"] = config.get("sample_weight_scheme", "none")
    predictions["official_test_used_for_selection"] = False
    metrics = metrics_by_model(predictions)
    metrics.insert(0, "dataset", "FD003")
    metrics["n_test"] = metrics.pop("n_eval")

    out_dir = RESULTS_DIR / "FD003"
    predictions.to_csv(out_dir / "fd003_final_official_test_predictions.csv", index=False)
    metrics.to_csv(out_dir / "fd003_final_official_test_metrics.csv", index=False)
    metrics_by_rul_bin(predictions).to_csv(
        out_dir / "fd003_final_official_test_metrics_by_rul_bin.csv",
        index=False,
    )
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_DIR / "fd003_best_model_predictions.csv", index=False)
    return predictions, metrics


def load_final_prediction_tables() -> dict[str, tuple[str, pd.DataFrame]]:
    fd001_path = RESULTS_DIR / "FD001" / "fd001_final_quantile_official_test_predictions.csv"
    fd003_path = RESULTS_DIR / "FD003" / "fd003_final_official_test_predictions.csv"
    if not fd001_path.exists():
        generate_fd001_final_official()
    if not fd003_path.exists():
        generate_fd003_final_official()

    tables = {
        "FD001": ("official_test", pd.read_csv(fd001_path)),
        "FD002": ("official_test", pd.read_csv(PREDICTIONS_DIR / "fd002_best_model_predictions.csv")),
        "FD003": ("official_test", pd.read_csv(fd003_path)),
        "FD004": ("official_test", pd.read_csv(PREDICTIONS_DIR / "fd004_best_model_predictions.csv")),
    }
    return tables


def normalize_predictions(dataset: str, scope: str, predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    rename = {
        "unit_number": "unit",
        "true_rul": "y_true_rul_raw",
        "pred_rul": "y_pred_rul",
        "cutoff_cycle": "cycle",
    }
    result = result.rename(columns={key: value for key, value in rename.items() if key in result.columns})
    result["dataset"] = dataset
    result["evaluation_scope"] = scope
    if "error" not in result.columns:
        result["error"] = result["y_pred_rul"] - result["y_true_rul_raw"]
    if "abs_error" not in result.columns:
        result["abs_error"] = result["error"].abs()
    result["dangerous_error"] = result["error"] > 20.0
    result["conservative_error"] = result["error"] < -20.0
    return result


def add_decision_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    y_pred = result["y_pred_rul"].astype(float)
    result["maintenance_band"] = np.select(
        [y_pred <= 30, (y_pred > 30) & (y_pred <= 60), (y_pred > 60) & (y_pred <= 90)],
        ["urgent_replace", "schedule_soon", "monitor_close"],
        default="continue_monitoring",
    )
    result["decision_recommendation"] = np.select(
        [y_pred <= 30, (y_pred > 30) & (y_pred <= 60), (y_pred > 60) & (y_pred <= 90)],
        [
            "priorizar intervencion inmediata",
            "programar mantenimiento preventivo",
            "monitorear de cerca",
        ],
        default="seguir operando con monitoreo normal",
    )
    result["risk_flag"] = np.select(
        [result["dangerous_error"], result["conservative_error"]],
        ["dangerous_overestimate", "conservative_underestimate"],
        default="within_20_cycles",
    )
    return result


def build_priority_rankings(prediction_tables: dict[str, tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for dataset, (scope, table) in prediction_tables.items():
        normalized = normalize_predictions(dataset, scope, table)
        decision = add_decision_columns(normalized)
        sort_cols = ["y_pred_rul", "dangerous_error", "abs_error"]
        ascending = [True, False, False]
        ranked = decision.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        ranked.insert(0, "priority_rank", np.arange(1, len(ranked) + 1))
        keep = [
            "priority_rank",
            "dataset",
            "evaluation_scope",
            "unit",
            "cycle",
            "y_true_rul_raw",
            "y_pred_rul",
            "error",
            "abs_error",
            "maintenance_band",
            "decision_recommendation",
            "risk_flag",
        ]
        rows.append(ranked[[column for column in keep if column in ranked.columns]])
    return pd.concat(rows, ignore_index=True)


def build_decision_summary(priority: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in priority.groupby("dataset", sort=True):
        metrics = regression_metrics(group["y_true_rul_raw"], group["y_pred_rul"])
        row = {
            "dataset": dataset,
            "n_cases": len(group),
            "urgent_replace": int((group["maintenance_band"] == "urgent_replace").sum()),
            "schedule_soon": int((group["maintenance_band"] == "schedule_soon").sum()),
            "monitor_close": int((group["maintenance_band"] == "monitor_close").sum()),
            "continue_monitoring": int((group["maintenance_band"] == "continue_monitoring").sum()),
            "dangerous_cases": int((group["risk_flag"] == "dangerous_overestimate").sum()),
            "conservative_cases": int((group["risk_flag"] == "conservative_underestimate").sum()),
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def build_model_summary() -> pd.DataFrame:
    fd001 = read_json(CONFIGS_DIR / "FD001" / "fd001_final_quantile_candidate_notebook18_config.json")
    fd002 = read_json(CONFIGS_DIR / "FD002" / "fd002_best_model_config.json")
    fd003 = read_json(CONFIGS_DIR / "FD003" / "fd003_final_candidate_config.json")
    fd004 = read_json(CONFIGS_DIR / "FD004" / "fd004_best_model_config.json")
    return pd.DataFrame(
        [
            {
                "dataset": "FD001",
                "final_model": fd001["model_name"],
                "model_family": fd001["model_family"],
                "feature_set": "temporal",
                "window_size": fd001["window_size"],
                "rul_cap": fd001["rul_cap"],
                "sample_weight_scheme": fd001["sample_weight_scheme"],
                "selection_basis": fd001["selection_basis"],
            },
            {
                "dataset": "FD002",
                "final_model": fd002["final_model"]["candidate_label"],
                "model_family": fd002["final_model"]["model_type"],
                "feature_set": fd002["preprocessing"]["feature_set"],
                "window_size": fd002["preprocessing"]["window_size"],
                "rul_cap": fd002["preprocessing"]["rul_cap"],
                "sample_weight_scheme": fd002["final_model"]["sample_weight_scheme"],
                "selection_basis": fd002["selection_policy"],
            },
            {
                "dataset": "FD003",
                "final_model": fd003["model_name"],
                "model_family": fd003["model_family"],
                "feature_set": fd003["feature_set"],
                "window_size": fd003["window_size"],
                "rul_cap": fd003["rul_cap"],
                "sample_weight_scheme": fd003["sample_weight_scheme"],
                "selection_basis": fd003["selection_basis"],
            },
            {
                "dataset": "FD004",
                "final_model": fd004["final_model"]["candidate_label"],
                "model_family": fd004["final_model"]["model_type"],
                "feature_set": fd004["preprocessing"]["feature_set"],
                "window_size": fd004["preprocessing"]["window_size"],
                "rul_cap": fd004["preprocessing"]["rul_cap"],
                "sample_weight_scheme": fd004["final_model"]["sample_weight_scheme"],
                "selection_basis": fd004["selection_policy"],
            },
        ]
    )


def build_metric_summary() -> pd.DataFrame:
    fd001_metrics = pd.read_csv(RESULTS_DIR / "FD001" / "fd001_final_quantile_official_test_metrics.csv")
    fd002_metrics = pd.read_csv(RESULTS_DIR / "FD002" / "fd002_official_test_metrics.csv")
    fd003_metrics = pd.read_csv(RESULTS_DIR / "FD003" / "fd003_final_official_test_metrics.csv")
    fd004_metrics = pd.read_csv(RESULTS_DIR / "FD004" / "fd004_official_test_metrics.csv")
    rows = [
        normalize_metric_row("FD001", "official_test_final_config", fd001_metrics),
        normalize_metric_row("FD002", "official_test_final_config", fd002_metrics),
        normalize_metric_row("FD003", "official_test_final_config", fd003_metrics),
        normalize_metric_row("FD004", "official_test_final_config", fd004_metrics),
    ]
    internal_fd003 = pd.read_csv(RESULTS_DIR / "FD003" / "fd003_internal_validation_final_metrics.csv")
    rows.append(normalize_metric_row("FD003", "internal_validation_multisplit", internal_fd003))
    return pd.DataFrame(rows)


def build_bin_summary() -> pd.DataFrame:
    sources = [
        ("FD001", "official_test_final_config", RESULTS_DIR / "FD001" / "fd001_final_quantile_official_test_metrics_by_rul_bin.csv"),
        ("FD002", "official_test_final_config", RESULTS_DIR / "FD002" / "fd002_official_test_metrics_by_rul_bin.csv"),
        ("FD003", "official_test_final_config", RESULTS_DIR / "FD003" / "fd003_final_official_test_metrics_by_rul_bin.csv"),
        ("FD004", "official_test_final_config", RESULTS_DIR / "FD004" / "fd004_official_test_metrics_by_rul_bin.csv"),
        ("FD003", "internal_validation_multisplit", RESULTS_DIR / "FD003" / "fd003_internal_validation_final_metrics_by_rul_bin.csv"),
    ]
    frames = []
    for dataset, scope, path in sources:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "evaluation_scope", scope)
        frame.insert(0, "dataset", dataset)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def write_readme(model_summary: pd.DataFrame, metric_summary: pd.DataFrame, decision_summary: pd.DataFrame) -> None:
    lines = [
        "# Conclusion - C-MAPSS RUL",
        "",
        "Esta carpeta consolida las implementaciones finales del proyecto: modelos seleccionados, metricas finales, ranking de prioridad de mantenimiento y reglas simples de decision operativa.",
        "",
        "Criterio metodologico:",
        "- Los modelos se seleccionaron con validacion artificial por motores completos.",
        "- El test oficial se usa solo para reporte final, no para buscar hiperparametros.",
        "- Las decisiones operativas se basan en RUL predicho y se complementan con flags de error peligroso/conservador cuando hay etiqueta real disponible.",
        "",
        "Modelos finales:",
    ]
    for _, row in model_summary.iterrows():
        lines.append(
            f"- {row['dataset']}: {row['final_model']} ({row['model_family']}), "
            f"feature_set={row['feature_set']}, window={row['window_size']}, cap={row['rul_cap']}."
        )
    lines.extend(["", "Metricas oficiales finales:"])
    official = metric_summary.loc[metric_summary["evaluation_scope"].str.startswith("official_test")]
    for _, row in official.iterrows():
        lines.append(
            f"- {row['dataset']}: RMSE {row['rmse']:.3f}, C-MAPSS {row['cmapss_score']:.3f}, "
            f"dangerous {row['dangerous_error_pct']:.2f}%."
        )
    lines.extend(
        [
            "",
            "Lectura por rangos de RUL:",
            "- Los modelos son mas precisos cerca de falla (0-30 ciclos), donde el costo operativo de sobreestimar RUL es mayor.",
            "- La zona 60-90 concentra varios dangerous errors y conviene discutirla explicitamente en el informe.",
            "- En 90+ los errores suben por el RUL cap y por la menor prioridad operativa de distinguir vidas remanentes largas.",
        ]
    )
    lines.extend(["", "Resumen de decision:"])
    for _, row in decision_summary.iterrows():
        lines.append(
            f"- {row['dataset']}: {int(row['urgent_replace'])} urgentes, "
            f"{int(row['schedule_soon'])} programar pronto, {int(row['monitor_close'])} monitoreo cercano."
        )
    lines.extend(
        [
            "",
            "Archivos:",
            "- final_model_summary.csv",
            "- final_metric_summary.csv",
            "- final_rul_bin_metrics.csv",
            "- maintenance_priority_ranking.csv",
            "- maintenance_decision_summary.csv",
            "- final_conclusion_payload.json",
            "- final_deliverable_manifest.csv",
            "- final_deliverable_manifest.json",
            "",
            "Lectura final:",
            "- Los modelos finales quedan cerrados por subset y mantienen validacion por motores completos antes del reporte oficial.",
            "- Las metricas finales oficiales estan en `final_metric_summary.csv`.",
            "- Las predicciones ejecutables finales estan en `predictions/final_executable/`.",
            "- Las metricas por rango de RUL estan en `final_rul_bin_metrics.csv`.",
            "- La priorizacion de mantenimiento queda materializada en `maintenance_priority_ranking.csv` y `maintenance_decision_summary.csv`.",
            "- Los notebooks historicos de conclusion estan en `notebooks/conclusion/archive/` y no son fuente de metricas finales.",
            "- El manifiesto final registra scripts, configs, predicciones, notebooks de evidencia y materiales de reporte sin mover ni reescribir notebooks.",
            "",
            "Interpretacion fisico-operativa:",
            "- Se agrega una auditoria post-hoc que no cambia los modelos finales ni sus metricas oficiales.",
            "- El analisis interpreta senales por sensor base y por area fisica del motor.",
            "- Las importancias se agrupan por variable base para no interpretar cada estadistico temporal por separado.",
            "- Las permutaciones agrupan todas las columnas derivadas del mismo sensor para medir sensibilidad fisica de forma consistente.",
            "- Los artefactos se regeneran con `python conclusion/build_physical_operational_artifacts.py` y `python conclusion/build_physical_area_summary.py`.",
            "",
            "Archivos fisico-operativos:",
            "- physical_sensor_dictionary.csv",
            "- physical_feature_importance_by_dataset.csv",
            "- physical_feature_importance_overall.csv",
            "- physical_perturbation_sensitivity.csv",
            "- physical_pattern_links.csv",
            "- physical_operational_payload.json",
            "- physical_importance_by_engine_area.csv",
            "- physical_sensitivity_by_engine_area.csv",
            "- figures/physical_feature_importance_top_sensors.png",
            "- figures/physical_perturbation_sensitivity_top_sensors.png",
            "- figures/physical_importance_by_engine_area.png",
        ]
    )
    (CONCLUSION_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final conclusion artifacts.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    parse_args(argv)
    CONCLUSION_DIR.mkdir(parents=True, exist_ok=True)
    generate_fd001_final_official()
    generate_fd003_final_official()
    prediction_tables = load_final_prediction_tables()

    model_summary = build_model_summary()
    metric_summary = build_metric_summary()
    bin_summary = build_bin_summary()
    priority = build_priority_rankings(prediction_tables)
    decision_summary = build_decision_summary(priority)

    model_summary.to_csv(CONCLUSION_DIR / "final_model_summary.csv", index=False)
    metric_summary.to_csv(CONCLUSION_DIR / "final_metric_summary.csv", index=False)
    bin_summary.to_csv(CONCLUSION_DIR / "final_rul_bin_metrics.csv", index=False)
    priority.to_csv(CONCLUSION_DIR / "maintenance_priority_ranking.csv", index=False)
    decision_summary.to_csv(CONCLUSION_DIR / "maintenance_decision_summary.csv", index=False)

    payload = {
        "methodology": {
            "selection": "artificial validation by complete held-out units",
            "official_test": "used only after model selection for final reporting",
            "decision_thresholds": {
                "urgent_replace": "predicted RUL <= 30",
                "schedule_soon": "30 < predicted RUL <= 60",
                "monitor_close": "60 < predicted RUL <= 90",
                "continue_monitoring": "predicted RUL > 90",
                "dangerous_error": "predicted RUL - true RUL > 20",
            },
        },
        "artifacts": {
            "final_model_summary": "conclusion/final_model_summary.csv",
            "final_metric_summary": "conclusion/final_metric_summary.csv",
            "final_rul_bin_metrics": "conclusion/final_rul_bin_metrics.csv",
            "maintenance_priority_ranking": "conclusion/maintenance_priority_ranking.csv",
            "maintenance_decision_summary": "conclusion/maintenance_decision_summary.csv",
        },
    }
    write_json(CONCLUSION_DIR / "final_conclusion_payload.json", payload)
    write_readme(model_summary, metric_summary, decision_summary)

    print("Wrote conclusion artifacts:")
    for path in [
        "README.md",
        "final_model_summary.csv",
        "final_metric_summary.csv",
        "final_rul_bin_metrics.csv",
        "maintenance_priority_ranking.csv",
        "maintenance_decision_summary.csv",
        "final_conclusion_payload.json",
    ]:
        print(f"- conclusion/{path}")


if __name__ == "__main__":
    main()
