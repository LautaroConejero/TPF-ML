from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "predict_final.py",
    "conclusion/build_conclusion_artifacts.py",
    "conclusion/build_physical_operational_artifacts.py",
    "conclusion/build_physical_area_summary.py",
    "configs/FD001/fd001_best_model_config.json",
    "configs/FD002/fd002_best_model_config.json",
    "configs/FD003/fd003_final_candidate_config.json",
    "configs/FD004/fd004_best_model_config.json",
    "conclusion/final_model_summary.csv",
    "conclusion/final_metric_summary.csv",
    "conclusion/final_rul_bin_metrics.csv",
    "conclusion/maintenance_priority_ranking.csv",
    "conclusion/maintenance_decision_summary.csv",
    "conclusion/final_conclusion_payload.json",
    "conclusion/physical_sensor_dictionary.csv",
    "conclusion/physical_feature_importance_overall.csv",
    "conclusion/physical_perturbation_sensitivity.csv",
    "conclusion/physical_operational_payload.json",
    "conclusion/physical_importance_by_engine_area.csv",
    "conclusion/physical_sensitivity_by_engine_area.csv",
    "conclusion/final_deliverable_manifest.csv",
    "conclusion/final_deliverable_manifest.json",
]

CSV_FILES = [
    "conclusion/final_model_summary.csv",
    "conclusion/final_metric_summary.csv",
    "conclusion/final_rul_bin_metrics.csv",
    "conclusion/maintenance_priority_ranking.csv",
    "conclusion/maintenance_decision_summary.csv",
    "conclusion/physical_sensor_dictionary.csv",
    "conclusion/physical_feature_importance_overall.csv",
    "conclusion/physical_perturbation_sensitivity.csv",
    "conclusion/physical_importance_by_engine_area.csv",
    "conclusion/physical_sensitivity_by_engine_area.csv",
    "conclusion/final_deliverable_manifest.csv",
    "predictions/final_executable/all_final_predictions.csv",
    "predictions/final_executable/all_final_metrics.csv",
]

MODULES = [
    "src.data",
    "src.data_splitting",
    "src.preprocessed_FD001",
    "src.fd001_experiment_utils",
    "src.fd001_modeling",
    "src.fd002_modeling",
    "src.fd003_improvement_utils",
    "src.fd004_modeling",
]

SCRIPTS_TO_COMPILE = [
    "predict_final.py",
    "conclusion/build_conclusion_artifacts.py",
    "conclusion/build_physical_operational_artifacts.py",
    "conclusion/build_physical_area_summary.py",
    "scripts/smoke_test_final.py",
]


def check_exists(errors: list[str]) -> None:
    for rel_path in REQUIRED_FILES:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            errors.append(f"Missing required file: {rel_path}")


def check_csv_readable(errors: list[str]) -> None:
    for rel_path in CSV_FILES:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            errors.append(f"Missing CSV: {rel_path}")
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            errors.append(f"Could not read CSV {rel_path}: {exc}")
            continue
        if frame.empty:
            errors.append(f"CSV is empty: {rel_path}")


def check_imports(errors: list[str]) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"Could not import {module_name}: {exc}")


def check_compilation(errors: list[str]) -> None:
    for rel_path in SCRIPTS_TO_COMPILE:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            errors.append(f"Missing script for compile check: {rel_path}")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            errors.append(f"Could not compile {rel_path}: {exc}")


def main() -> int:
    errors: list[str] = []
    check_exists(errors)
    check_csv_readable(errors)
    check_imports(errors)
    check_compilation(errors)

    if errors:
        print("Final smoke test failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Final smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
