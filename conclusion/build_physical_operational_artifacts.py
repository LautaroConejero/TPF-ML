from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import add_train_rul, last_cycle_rows, read_cmapss_file, read_rul_file
from src.fd001_experiment_utils import make_lgbm_from_search_params, weights_from_scheme
from src.fd001_modeling import regression_metrics
from src.fd002_modeling import (
    add_fd002_condition_features,
    add_fd002_extra_features_for_feature_set,
    base_columns_for_feature_set,
    fd002_weights_from_scheme,
    fit_condition_preprocessor,
    make_model,
    make_temporal_features_fast,
    scale_temporal_frames,
    temporal_columns_for_base,
)
from src.fd003_improvement_utils import (
    FD003_RELEVANT_SENSORS,
    make_fault_sensitive_feature_frame,
    make_lgbm_regressor_from_config,
    sample_weights_for_scheme,
)
from src.fd003_transfer_utils import fd003_feature_columns
from src.fd004_modeling import add_fd004_extra_features_for_feature_set
from src.preprocessed_FD001 import (
    fd001_feature_columns,
    make_temporal_features,
    temporal_feature_columns,
)


CONCLUSION_DIR = PROJECT_ROOT / "conclusion"
FIGURES_DIR = CONCLUSION_DIR / "figures"
NOTES_DIR = PROJECT_ROOT / "notas"
NOTE_PATH = NOTES_DIR / "interpretacion_fisica_operativa.txt"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "conclusion" / "02_interpretacion_fisica_operativa.ipynb"
DATA_DIR = PROJECT_ROOT / "CMAPSSData"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RANDOM_STATE = 42

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
PHYSICALLY_RELEVANT_SENSORS = [
    "sensor_11",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_14",
    "sensor_12",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]


@dataclass
class FinalRun:
    dataset: str
    model: object
    feature_columns: list[str]
    x_test: pd.DataFrame
    y_test_raw: pd.Series
    baseline_pred: np.ndarray
    model_name: str
    representation: str
    feature_set: str
    window_size: int
    rul_cap: int


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict) -> None:
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
            return None if np.isnan(value) else float(value)
        if isinstance(value, np.ndarray):
            return convert(value.tolist())
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(convert(payload), file, indent=2, ensure_ascii=False)


def load_train_test(subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        read_cmapss_file(DATA_DIR / f"train_{subset}.txt"),
        read_cmapss_file(DATA_DIR / f"test_{subset}.txt"),
    )


def attach_test_rul(subset: str, test_last: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    result = test_last.copy()
    rul_path = DATA_DIR / f"RUL_{subset}.txt"
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing official RUL file: {rul_path}")
    result = result.merge(read_rul_file(rul_path), on="unit", how="left")
    result = result.rename(columns={"final_rul": "RUL_raw"})
    result["RUL"] = result["RUL_raw"].clip(upper=rul_cap)
    return result


def sensor_dictionary() -> pd.DataFrame:
    rows = [
        ("sensor_1", "T2", "Total temperature at fan inlet.", "inlet/fan", "Context signal for inlet thermodynamic state."),
        ("sensor_2", "T24", "Total temperature at LPC outlet.", "LPC", "Temperature changes can reflect compression efficiency and degradation."),
        ("sensor_3", "T30", "Total temperature at HPC outlet.", "HPC", "HPC outlet temperature is a degradation-sensitive thermodynamic signal."),
        ("sensor_4", "T50", "Total temperature at LPT outlet.", "turbine/LPT", "Turbine outlet temperature can shift as efficiency and flow path health change."),
        ("sensor_5", "P2", "Pressure at fan inlet.", "inlet/fan", "Context signal for inlet pressure and operating regime."),
        ("sensor_6", "P15", "Total pressure in bypass duct.", "bypass", "Bypass pressure can reflect fan/bypass operating behavior."),
        ("sensor_7", "P30", "Total pressure at HPC outlet.", "HPC", "HPC outlet pressure is directly tied to compressor health and loading."),
        ("sensor_8", "Nf", "Physical fan speed.", "fan/spool", "Fan speed variation helps separate regime and degradation effects."),
        ("sensor_9", "Nc", "Physical core speed.", "core spool", "Core speed dynamics are informative for degradation patterns."),
        ("sensor_10", "epr", "Engine pressure ratio.", "global engine pressure", "Global pressure ratio summarizes engine pressure behavior."),
        ("sensor_11", "Ps30", "Static pressure at HPC outlet.", "HPC", "Static HPC pressure is often informative for RUL and dangerous overestimation."),
        ("sensor_12", "phi", "Ratio of fuel flow to Ps30.", "fuel/HPC interaction", "Fuel-to-pressure ratio links control demand with compressor state."),
        ("sensor_13", "NRf", "Corrected fan speed.", "fan/spool", "Corrected fan speed helps compare operation across regimes."),
        ("sensor_14", "NRc", "Corrected core speed.", "core spool", "Corrected core speed captures core-spool degradation signals."),
        ("sensor_15", "BPR", "Bypass ratio.", "bypass/fan", "Bypass ratio can separate flow-path and fan/bypass degradation behavior."),
        ("sensor_16", "farB", "Burner fuel-air ratio.", "combustor", "Combustor fuel-air ratio can reflect control and thermal state."),
        ("sensor_17", "htBleed", "Bleed enthalpy.", "bleed system", "Bleed behavior is connected to thermal management and degradation."),
        ("sensor_18", "Nf_dmd", "Demanded fan speed.", "control system", "Control demand signal for fan-speed command."),
        ("sensor_19", "PCNfR_dmd", "Demanded corrected fan speed.", "control system", "Control demand signal for corrected fan-speed command."),
        ("sensor_20", "W31", "HPT coolant bleed.", "HPT cooling/bleed", "Cooling bleed can indicate thermal protection and turbine health demands."),
        ("sensor_21", "W32", "LPT coolant bleed.", "LPT cooling/bleed", "Cooling bleed can indicate turbine thermal-management behavior."),
        ("setting_1", "setting_1", "Operational setting associated with flight regime.", "operating condition", "Separates operating regimes from degradation signal."),
        ("setting_2", "setting_2", "Operational setting associated with flight regime.", "operating condition", "Separates operating regimes from degradation signal."),
        ("setting_3", "setting_3", "Operational setting associated with flight regime.", "operating condition", "Separates operating regimes from degradation signal."),
        ("condition", "condition", "Inferred operating condition from settings.", "operating condition", "Groups condition_id and one-hot features that control regime effects."),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "raw_variable",
            "physical_symbol",
            "physical_description",
            "engine_area",
            "interpretation_for_rul",
        ],
    )


def fit_fd001_final() -> FinalRun:
    payload = read_json(CONFIGS_DIR / "FD001" / "fd001_best_model_config.json")
    preprocessing = payload["preprocessing"]
    config = payload["final_model"]
    train_raw, test_raw = load_train_test("FD001")
    rul_cap = int(preprocessing["rul_cap"])
    window_size = int(preprocessing["window_size"])

    train = add_train_rul(train_raw, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    train["RUL"] = train["RUL"].clip(upper=rul_cap)
    base_columns = fd001_feature_columns(drop_columns=preprocessing.get("dropped_columns"))
    train_temporal = make_temporal_features(train, train, base_columns, window_size)
    test_last = attach_test_rul("FD001", last_cycle_rows(test_raw), rul_cap)
    test_temporal = make_temporal_features(test_raw, test_last, base_columns, window_size)
    feature_columns = temporal_feature_columns(base_columns)

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
    model_config = {
        "objective": config["objective"],
        "alpha": config.get("alpha"),
        "sample_weight_scheme": config.get("sample_weight_scheme", "none"),
        "params": config["hyperparameters"],
    }
    model = make_lgbm_from_search_params(model_config, random_state=RANDOM_STATE)
    weights = weights_from_scheme(train_temporal["RUL_raw"], config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(x_train, train_temporal["RUL"])
    else:
        model.fit(x_train, train_temporal["RUL"], sample_weight=weights)

    pred = np.clip(model.predict(x_test), 0.0, None)
    return FinalRun(
        dataset="FD001",
        model=model,
        feature_columns=feature_columns,
        x_test=x_test,
        y_test_raw=test_temporal["RUL_raw"].copy(),
        baseline_pred=pred,
        model_name=config["model_name"],
        representation=f"temporal_w{window_size}",
        feature_set="temporal",
        window_size=window_size,
        rul_cap=rul_cap,
    )


def fit_fd002_like_final(subset: str) -> FinalRun:
    payload = read_json(CONFIGS_DIR / subset / f"{subset.lower()}_best_model_config.json")
    preprocessing = payload["preprocessing"]
    final_model = payload["final_model"]
    model_config = {
        "candidate_label": final_model["candidate_label"],
        "model_type": final_model["model_type"],
        "feature_set": preprocessing["feature_set"],
        "window_size": int(preprocessing["window_size"]),
        "rul_cap": int(preprocessing["rul_cap"]),
        "representation": final_model["representation"],
        "sample_weight_scheme": final_model.get("sample_weight_scheme", "none"),
        "params": final_model["hyperparameters"],
    }
    train_raw, test_raw = load_train_test(subset)
    train = add_train_rul(train_raw, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    train["RUL"] = train["RUL"].clip(upper=model_config["rul_cap"])

    condition_preprocessor = fit_condition_preprocessor(train, random_state=RANDOM_STATE)
    train_aug = add_fd002_condition_features(train, condition_preprocessor)
    test_source_aug = add_fd002_condition_features(test_raw, condition_preprocessor)
    test_last = attach_test_rul(subset, last_cycle_rows(test_raw), model_config["rul_cap"])
    test_last_aug = add_fd002_condition_features(test_last, condition_preprocessor)

    base_columns = base_columns_for_feature_set(model_config["feature_set"])
    train_temporal = make_temporal_features_fast(
        train_aug,
        endpoints_df=train_aug,
        feature_columns=base_columns,
        window_size=model_config["window_size"],
    )
    test_temporal = make_temporal_features_fast(
        test_source_aug,
        endpoints_df=test_last_aug,
        feature_columns=base_columns,
        window_size=model_config["window_size"],
    )
    temporal_columns = temporal_columns_for_base(base_columns)
    if subset == "FD002":
        train_temporal, train_extra_columns = add_fd002_extra_features_for_feature_set(
            train_temporal,
            train_aug,
            train_aug,
            model_config["feature_set"],
            window_size=model_config["window_size"],
        )
        test_temporal, test_extra_columns = add_fd002_extra_features_for_feature_set(
            test_temporal,
            test_source_aug,
            test_last_aug,
            model_config["feature_set"],
            window_size=model_config["window_size"],
        )
    else:
        train_temporal, train_extra_columns = add_fd004_extra_features_for_feature_set(
            train_temporal,
            train_aug,
            train_aug,
            model_config["feature_set"],
            window_size=model_config["window_size"],
        )
        test_temporal, test_extra_columns = add_fd004_extra_features_for_feature_set(
            test_temporal,
            test_source_aug,
            test_last_aug,
            model_config["feature_set"],
            window_size=model_config["window_size"],
        )
    feature_columns = temporal_columns + (train_extra_columns or test_extra_columns)
    _, x_train, x_test = scale_temporal_frames(train_temporal, test_temporal, feature_columns)

    model = make_model(model_config, random_state=RANDOM_STATE)
    weights = fd002_weights_from_scheme(train_temporal["RUL_raw"], model_config["sample_weight_scheme"])
    if weights is None:
        model.fit(x_train, train_temporal["RUL"])
    else:
        model.fit(x_train, train_temporal["RUL"], sample_weight=weights)

    pred = np.clip(model.predict(x_test), 0.0, None)
    return FinalRun(
        dataset=subset,
        model=model,
        feature_columns=feature_columns,
        x_test=x_test,
        y_test_raw=test_temporal["RUL_raw"].copy(),
        baseline_pred=pred,
        model_name=model_config["candidate_label"],
        representation=model_config["representation"],
        feature_set=model_config["feature_set"],
        window_size=model_config["window_size"],
        rul_cap=model_config["rul_cap"],
    )


def fit_fd003_final() -> FinalRun:
    config = read_json(CONFIGS_DIR / "FD003" / "fd003_final_candidate_config.json")
    train_raw, test_raw = load_train_test("FD003")
    rul_cap = int(config["rul_cap"])
    window_size = int(config["window_size"])
    train = add_train_rul(train_raw, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    train["RUL"] = train["RUL"].clip(upper=rul_cap)

    base_columns, _ = fd003_feature_columns(train)
    train_temporal = make_temporal_features(train, train, base_columns, window_size)
    test_last = attach_test_rul("FD003", last_cycle_rows(test_raw), rul_cap)
    test_temporal = make_temporal_features(test_raw, test_last, base_columns, window_size)
    relevant = [sensor for sensor in FD003_RELEVANT_SENSORS if sensor in base_columns]
    train_extra = make_fault_sensitive_feature_frame(train, train, relevant, window_size)
    test_extra = make_fault_sensitive_feature_frame(test_raw, test_last, relevant, window_size)
    extra_columns = [column for column in train_extra.columns if column not in {"unit", "cycle"}]
    train_temporal = train_temporal.merge(train_extra, on=["unit", "cycle"], how="left")
    test_temporal = test_temporal.merge(test_extra, on=["unit", "cycle"], how="left")
    feature_columns = temporal_feature_columns(base_columns) + extra_columns

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
    model = make_lgbm_regressor_from_config(config, random_state=RANDOM_STATE)
    weights = sample_weights_for_scheme(train_temporal["RUL_raw"], config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(x_train, train_temporal["RUL"])
    else:
        model.fit(x_train, train_temporal["RUL"], sample_weight=weights)

    pred = np.clip(model.predict(x_test), 0.0, None)
    return FinalRun(
        dataset="FD003",
        model=model,
        feature_columns=feature_columns,
        x_test=x_test,
        y_test_raw=test_temporal["RUL_raw"].copy(),
        baseline_pred=pred,
        model_name=config["model_name"],
        representation=f"temporal_w{window_size}_{config['feature_set']}",
        feature_set=config["feature_set"],
        window_size=window_size,
        rul_cap=rul_cap,
    )


def fit_all_final_models() -> dict[str, FinalRun]:
    return {
        "FD001": fit_fd001_final(),
        "FD002": fit_fd002_like_final("FD002"),
        "FD003": fit_fd003_final(),
        "FD004": fit_fd002_like_final("FD004"),
    }


def base_variables_from_feature(feature: str) -> list[str]:
    sensors = sorted(set(re.findall(r"sensor_\d+", feature)), key=lambda value: int(value.split("_")[1]))
    if sensors:
        return sensors
    settings = sorted(set(re.findall(r"setting_\d+", feature)), key=lambda value: int(value.split("_")[1]))
    if settings:
        return settings
    if feature.startswith("condition_") or feature == "condition_id":
        return ["condition"]
    return ["other"]


def dictionary_lookup(dictionary: pd.DataFrame) -> pd.DataFrame:
    return dictionary.set_index("raw_variable")


def attach_physical_columns(frame: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    lookup = dictionary_lookup(dictionary)
    result = frame.copy()
    for column in ["physical_symbol", "physical_description", "engine_area", "interpretation_for_rul"]:
        result[column] = result["base_variable"].map(lookup[column])
    result["physical_symbol"] = result["physical_symbol"].fillna(result["base_variable"])
    result["physical_description"] = result["physical_description"].fillna("Derived or aggregate feature not tied to one raw sensor.")
    result["engine_area"] = result["engine_area"].fillna("derived")
    result["interpretation_for_rul"] = result["interpretation_for_rul"].fillna("Used as a derived modeling signal.")
    return result


def build_importance_tables(runs: dict[str, FinalRun], dictionary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset, run in runs.items():
        importances = getattr(run.model, "feature_importances_", None)
        if importances is None:
            continue
        feature_rows = []
        for feature, importance in zip(run.feature_columns, np.asarray(importances, dtype=float)):
            bases = base_variables_from_feature(feature)
            share = float(importance) / len(bases)
            for base in bases:
                feature_rows.append({"base_variable": base, "feature": feature, "importance": share})
        feature_frame = pd.DataFrame(feature_rows)
        grouped = (
            feature_frame.groupby("base_variable", as_index=False)
            .agg(
                importance_sum=("importance", "sum"),
                n_derived_features=("feature", "nunique"),
            )
            .sort_values("importance_sum", ascending=False)
            .reset_index(drop=True)
        )
        total = grouped["importance_sum"].sum()
        grouped["importance_normalized"] = np.where(total > 0, grouped["importance_sum"] / total, 0.0)
        grouped["rank_in_dataset"] = grouped["importance_normalized"].rank(
            method="dense",
            ascending=False,
        ).astype(int)
        grouped.insert(0, "dataset", dataset)
        rows.append(grouped)
    by_dataset = pd.concat(rows, ignore_index=True)
    by_dataset = attach_physical_columns(by_dataset, dictionary)
    by_dataset = by_dataset[
        [
            "dataset",
            "base_variable",
            "physical_symbol",
            "physical_description",
            "engine_area",
            "importance_sum",
            "importance_normalized",
            "rank_in_dataset",
            "n_derived_features",
        ]
    ].sort_values(["dataset", "rank_in_dataset", "base_variable"])

    overall = (
        by_dataset.groupby("base_variable", as_index=False)
        .agg(
            mean_importance_normalized=("importance_normalized", "mean"),
            max_importance_normalized=("importance_normalized", "max"),
            datasets_where_present=("dataset", lambda values: ",".join(sorted(set(values)))),
        )
        .sort_values(["mean_importance_normalized", "max_importance_normalized"], ascending=False)
        .reset_index(drop=True)
    )
    overall["overall_rank"] = np.arange(1, len(overall) + 1)
    overall = attach_physical_columns(overall, dictionary)
    overall = overall[
        [
            "base_variable",
            "physical_symbol",
            "physical_description",
            "engine_area",
            "mean_importance_normalized",
            "max_importance_normalized",
            "datasets_where_present",
            "overall_rank",
        ]
    ]
    return by_dataset.reset_index(drop=True), overall.reset_index(drop=True)


def feature_label(row: pd.Series) -> str:
    symbol = row.get("physical_symbol", row["base_variable"])
    if symbol == row["base_variable"]:
        return str(row["base_variable"])
    return f"{symbol} ({row['base_variable']})"


def plot_importance(overall: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = overall.loc[~overall["base_variable"].isin(["condition", "other"])].head(12).copy()
    top["label"] = top.apply(feature_label, axis=1)
    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=top.sort_values("mean_importance_normalized", ascending=True),
        x="mean_importance_normalized",
        y="label",
        color="#4C78A8",
    )
    plt.xlabel("Mean normalized grouped importance")
    plt.ylabel("")
    plt.title("Top physical sensors by grouped model importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "physical_feature_importance_top_sensors.png", dpi=160)
    plt.close()


def metrics_for_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return regression_metrics(y_true, np.clip(y_pred, 0.0, None))


def columns_for_base(feature_columns: list[str], base_variable: str) -> list[str]:
    return [column for column in feature_columns if base_variable in base_variables_from_feature(column)]


def candidate_sensors_for_run(dataset: str, by_dataset: pd.DataFrame, run: FinalRun) -> list[str]:
    ranked = (
        by_dataset.loc[
            (by_dataset["dataset"] == dataset)
            & by_dataset["base_variable"].str.startswith("sensor_")
        ]
        .sort_values("rank_in_dataset")["base_variable"]
        .head(8)
        .tolist()
    )
    candidates = list(dict.fromkeys(ranked + PHYSICALLY_RELEVANT_SENSORS))
    return [
        sensor
        for sensor in candidates
        if columns_for_base(run.feature_columns, sensor)
    ][:12]


def build_sensitivity_table(
    runs: dict[str, FinalRun],
    by_dataset: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for dataset, run in runs.items():
        baseline_metrics = metrics_for_predictions(run.y_test_raw, run.baseline_pred)
        for base in candidate_sensors_for_run(dataset, by_dataset, run):
            columns = columns_for_base(run.feature_columns, base)
            perturbed = run.x_test.copy()
            permutation = rng.permutation(len(perturbed))
            perturbed.loc[:, columns] = perturbed.loc[:, columns].to_numpy()[permutation]
            perturbed_pred = np.clip(run.model.predict(perturbed), 0.0, None)
            perturbed_metrics = metrics_for_predictions(run.y_test_raw, perturbed_pred)
            row = {
                "dataset": dataset,
                "base_variable": base,
                "baseline_mae": baseline_metrics["mae"],
                "perturbed_mae": perturbed_metrics["mae"],
                "delta_mae": perturbed_metrics["mae"] - baseline_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "perturbed_rmse": perturbed_metrics["rmse"],
                "delta_rmse": perturbed_metrics["rmse"] - baseline_metrics["rmse"],
                "baseline_r2": baseline_metrics["r2"],
                "perturbed_r2": perturbed_metrics["r2"],
                "delta_r2": perturbed_metrics["r2"] - baseline_metrics["r2"],
                "baseline_cmapss_score": baseline_metrics["cmapss_score"],
                "perturbed_cmapss_score": perturbed_metrics["cmapss_score"],
                "delta_cmapss_score": perturbed_metrics["cmapss_score"] - baseline_metrics["cmapss_score"],
                "baseline_dangerous_error_pct": baseline_metrics["dangerous_error_pct"],
                "perturbed_dangerous_error_pct": perturbed_metrics["dangerous_error_pct"],
                "delta_dangerous_error_pct": (
                    perturbed_metrics["dangerous_error_pct"] - baseline_metrics["dangerous_error_pct"]
                ),
                "mean_abs_prediction_shift": float(np.mean(np.abs(perturbed_pred - run.baseline_pred))),
                "n_perturbed_features": len(columns),
            }
            rows.append(row)
    result = attach_physical_columns(pd.DataFrame(rows), dictionary)
    ordered = [
        "dataset",
        "base_variable",
        "physical_symbol",
        "physical_description",
        "engine_area",
        "baseline_mae",
        "perturbed_mae",
        "delta_mae",
        "baseline_rmse",
        "perturbed_rmse",
        "delta_rmse",
        "baseline_r2",
        "perturbed_r2",
        "delta_r2",
        "baseline_cmapss_score",
        "perturbed_cmapss_score",
        "delta_cmapss_score",
        "baseline_dangerous_error_pct",
        "perturbed_dangerous_error_pct",
        "delta_dangerous_error_pct",
        "mean_abs_prediction_shift",
        "n_perturbed_features",
    ]
    return result[ordered].sort_values(
        ["dataset", "delta_cmapss_score", "mean_abs_prediction_shift"],
        ascending=[True, False, False],
    )


def plot_sensitivity(sensitivity: pd.DataFrame) -> None:
    plot_frame = (
        sensitivity.assign(
            sensitivity_score=lambda df: df["delta_cmapss_score"].clip(lower=0)
            + 50.0 * df["mean_abs_prediction_shift"]
        )
        .sort_values("sensitivity_score", ascending=False)
        .head(12)
        .copy()
    )
    plot_frame["label"] = plot_frame.apply(
        lambda row: f"{row['dataset']} - {feature_label(row)}",
        axis=1,
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_frame.sort_values("sensitivity_score", ascending=True),
        x="sensitivity_score",
        y="label",
        color="#F58518",
    )
    plt.xlabel("Sensitivity score: positive delta C-MAPSS + 50 x prediction shift")
    plt.ylabel("")
    plt.title("Sensors with strongest perturbation sensitivity")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "physical_perturbation_sensitivity_top_sensors.png", dpi=160)
    plt.close()


def build_pattern_links(dictionary: pd.DataFrame) -> pd.DataFrame:
    sources = {
        "FD003": [
            (
                "notas/hallazgos/FD003/hallazgo_clusters_fd003.txt",
                ["sensor_9", "sensor_14", "sensor_7"],
                "Clusters exploratorios con una sola condicion operativa; diferencias en core speed, corrected core speed y presion HPC sugieren patrones de degradacion distintos.",
            ),
            (
                "notas/FD003/fd003_internal_validation_evaluation_interpretation.txt",
                ["sensor_7", "sensor_9", "sensor_12", "sensor_14", "sensor_15"],
                "Features fault-sensitive calculadas sin mirar el futuro; se usan como senales de dinamica de degradacion, no como etiquetas de falla.",
            ),
        ],
        "FD004": [
            (
                "notas/hallazgos/FD004/hallazgo_fd004_conditions_fault_patterns.txt",
                ["sensor_11", "sensor_4", "sensor_17", "sensor_3", "sensor_2", "sensor_9"],
                "Al normalizar por condicion reaparecen sensores informativos ligados a presion/temperatura HPC, turbina, bleed y core speed.",
            ),
            (
                "notas/hallazgos/FD004/hallazgo_fd004_conditions_fault_patterns.txt",
                ["sensor_12", "sensor_15", "sensor_7"],
                "Clusters residuales de FD004 muestran trayectorias distintas en fuel/HPC, bypass ratio y presion HPC; lectura exploratoria, no etiqueta supervisada.",
            ),
        ],
    }
    rows = []
    existing_text = {}
    for _, source_rows in sources.items():
        for source, _, _ in source_rows:
            path = PROJECT_ROOT / source
            existing_text[source] = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    for dataset, source_rows in sources.items():
        for source, sensors, interpretation in source_rows:
            text = existing_text.get(source, "")
            for sensor in sensors:
                if text and sensor not in text:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "pattern_source": source,
                        "base_variable": sensor,
                        "interpretation": interpretation,
                    }
                )
    return attach_physical_columns(pd.DataFrame(rows), dictionary)[
        [
            "dataset",
            "pattern_source",
            "base_variable",
            "physical_symbol",
            "physical_description",
            "engine_area",
            "interpretation",
        ]
    ]


def write_interpretation_note() -> None:
    if NOTE_PATH.exists():
        return
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Interpretacion fisico-operativa complementaria",
        "",
        "Se incorporo una lectura fisico-operativa del cierre final del proyecto para conectar los modelos de RUL con variables fisicas del turbofan simulado. Esta capa no modifica la seleccion final de modelos: funciona como auditoria e interpretacion sobre los candidatos ya cerrados para FD001, FD002, FD003 y FD004.",
        "",
        "La primera decision metodologica fue construir un diccionario de sensores C-MAPSS. Cada sensor crudo se vinculo con su simbolo fisico, una descripcion operativa y un area del motor. De esta forma, los resultados dejan de leerse solo como columnas numericas y pasan a expresarse como senales de temperatura, presion, velocidades de spool, bypass, combustible, bleed o settings de condicion operativa.",
        "",
        "La importancia de variables se agrupo por sensor base. Esto evita interpretar por separado cada estadistico temporal, por ejemplo last, mean, slope, delta o z-score por condicion, cuando todos provienen de la misma magnitud fisica. La lectura agregada es mas defendible para el informe: un sensor aparece como relevante si el conjunto de sus transformaciones aporta al modelo.",
        "",
        "Ademas se agrego una sensibilidad por permutacion. Para cada subset se perturbaron en conjunto las columnas derivadas de sensores relevantes y se recalcularon MAE, RMSE, R2, C-MAPSS, dangerous error y desplazamiento medio de prediccion. Si la permutacion empeora las metricas o mueve mucho las predicciones, el modelo depende operativamente de esa senal. Esta prueba no prueba causalidad fisica, pero ayuda a auditar robustez y dependencia del modelo.",
        "",
        "Finalmente, los patrones latentes de FD003 y FD004 se reinterpretaron con el diccionario fisico. En FD003, los clusters exploratorios se conectan principalmente con variables de core speed y presion HPC, como Nc, NRc y P30. En FD004, la lectura es mas cautelosa porque conviven condiciones operativas y posibles modos de falla: despues de controlar condicion, aparecen senales asociadas a fuel/HPC, bypass y presion HPC. Estos patrones se tratan como evidencia exploratoria de dinamicas de degradacion distintas, no como etiquetas supervisadas ni como verdad de fault mode.",
        "",
        "En sintesis, el analisis fisico-operativo agrega trazabilidad entre features, sensores y comportamiento predictivo. Su aporte principal es hacer explicable por que ciertas familias de senales influyen en el RUL estimado y donde conviene mirar cuando se auditan errores peligrosos o diferencias entre subsets.",
    ]
    NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_subsets(runs: dict[str, FinalRun], sensitivity: pd.DataFrame) -> list[dict]:
    rows = []
    for dataset, run in runs.items():
        metrics = metrics_for_predictions(run.y_test_raw, run.baseline_pred)
        top_sens = (
            sensitivity.loc[sensitivity["dataset"] == dataset]
            .sort_values(["delta_cmapss_score", "mean_abs_prediction_shift"], ascending=False)
            .head(3)
        )
        rows.append(
            {
                "dataset": dataset,
                "model_name": run.model_name,
                "representation": run.representation,
                "feature_set": run.feature_set,
                "window_size": run.window_size,
                "rul_cap": run.rul_cap,
                "official_metrics": metrics,
                "top_sensitivity_sensors": top_sens[["base_variable", "physical_symbol"]].to_dict("records"),
            }
        )
    return rows


def write_payload(
    dictionary: pd.DataFrame,
    overall: pd.DataFrame,
    sensitivity: pd.DataFrame,
    runs: dict[str, FinalRun],
) -> None:
    top_sensitivity = (
        sensitivity.assign(
            sensitivity_score=lambda df: df["delta_cmapss_score"].clip(lower=0)
            + 50.0 * df["mean_abs_prediction_shift"]
        )
        .sort_values("sensitivity_score", ascending=False)
        .head(10)
    )
    payload = {
        "artifacts": {
            "physical_sensor_dictionary": "conclusion/physical_sensor_dictionary.csv",
            "physical_feature_importance_by_dataset": "conclusion/physical_feature_importance_by_dataset.csv",
            "physical_feature_importance_overall": "conclusion/physical_feature_importance_overall.csv",
            "physical_perturbation_sensitivity": "conclusion/physical_perturbation_sensitivity.csv",
            "physical_pattern_links": "conclusion/physical_pattern_links.csv",
            "physical_operational_payload": "conclusion/physical_operational_payload.json",
            "feature_importance_figure": "conclusion/figures/physical_feature_importance_top_sensors.png",
            "perturbation_sensitivity_figure": "conclusion/figures/physical_perturbation_sensitivity_top_sensors.png",
            "report_note": "notas/interpretacion_fisica_operativa.txt",
            "notebook": "notebooks/conclusion/02_interpretacion_fisica_operativa.ipynb",
        },
        "physical_sensors_considered": dictionary.loc[
            dictionary["raw_variable"].str.startswith("sensor_"),
            ["raw_variable", "physical_symbol", "engine_area"],
        ].to_dict("records"),
        "top_global_importance_sensors": overall.loc[
            overall["base_variable"].str.startswith("sensor_")
        ].head(10).to_dict("records"),
        "top_sensitivity_sensors": top_sensitivity[
            [
                "dataset",
                "base_variable",
                "physical_symbol",
                "delta_cmapss_score",
                "mean_abs_prediction_shift",
                "sensitivity_score",
            ]
        ].to_dict("records"),
        "subset_summary": summarize_subsets(runs, sensitivity),
        "methodology": {
            "importance_grouping": "Feature importances are split across detected raw sensors and then normalized within each dataset.",
            "perturbation": "All derived columns for a sensor are permuted together with random_state=42 on official test last-cycle features.",
            "selection_policy": "The final model selection is not changed by this script.",
        },
    }
    write_json(CONCLUSION_DIR / "physical_operational_payload.json", payload)


def update_readme(overall: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    readme_path = CONCLUSION_DIR / "README.md"
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Conclusion - C-MAPSS RUL\n"
    marker = "\nInterpretacion fisico-operativa:\n"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"

    top_importance = overall.loc[overall["base_variable"].str.startswith("sensor_")].head(3)
    top_sensitivity = (
        sensitivity.assign(
            sensitivity_score=lambda df: df["delta_cmapss_score"].clip(lower=0)
            + 50.0 * df["mean_abs_prediction_shift"]
        )
        .sort_values("sensitivity_score", ascending=False)
        .head(3)
    )
    lines = [
        "",
        "Interpretacion fisico-operativa:",
        "- Se agrego una auditoria fisico-operativa que no cambia los modelos finales, sino que interpreta sus senales por sensor base.",
        "- Top sensores globales por importancia agrupada: "
        + ", ".join(f"{row.physical_symbol} ({row.base_variable})" for row in top_importance.itertuples())
        + ".",
        "- Sensores con mayor sensibilidad por permutacion: "
        + ", ".join(f"{row.dataset}:{row.physical_symbol} ({row.base_variable})" for row in top_sensitivity.itertuples())
        + ".",
        "- FD003 se conecta con patrones latentes en core speed y presion HPC; FD004 agrega la complejidad de condiciones operativas y senales fuel/HPC/bypass.",
        "- Las permutaciones agrupan todas las columnas derivadas de cada sensor para evitar interpretar estadisticos temporales aislados.",
        "",
        "Archivos fisico-operativos:",
        "- physical_sensor_dictionary.csv",
        "- physical_feature_importance_by_dataset.csv",
        "- physical_feature_importance_overall.csv",
        "- physical_perturbation_sensitivity.csv",
        "- physical_pattern_links.csv",
        "- physical_operational_payload.json",
        "- figures/physical_feature_importance_top_sensors.png",
        "- figures/physical_perturbation_sensitivity_top_sensors.png",
    ]
    readme_path.write_text(text.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def write_notebook() -> None:
    if NOTEBOOK_PATH.exists():
        return
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Interpretacion fisico-operativa\n", "\n", "Reporte compacto de los artefactos generados por `conclusion/build_physical_operational_artifacts.py`."],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "from IPython.display import Image, display\n",
                "import pandas as pd\n",
                "\n",
                "PROJECT_ROOT = Path.cwd()\n",
                "root_marker = Path('conclusion') / 'build_conclusion_artifacts.py'\n",
                "if not (PROJECT_ROOT / root_marker).exists():\n",
                "    for parent in PROJECT_ROOT.parents:\n",
                "        if (parent / root_marker).exists():\n",
                "            PROJECT_ROOT = parent\n",
                "            break\n",
                "if not (PROJECT_ROOT / root_marker).exists():\n",
                "    raise RuntimeError(\n",
                "        f'No se encontro la carpeta conclusion/ desde {Path.cwd()}. '\n",
                "        'Abri el notebook desde la raiz del repo TPF-ML o ejecuta primero: '\n",
                "        'python conclusion/build_physical_operational_artifacts.py'\n",
                "    )\n",
                "CONCLUSION_DIR = PROJECT_ROOT / 'conclusion'\n",
                "FIGURES_DIR = CONCLUSION_DIR / 'figures'\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dictionary = pd.read_csv(CONCLUSION_DIR / 'physical_sensor_dictionary.csv')\n",
                "importance_overall = pd.read_csv(CONCLUSION_DIR / 'physical_feature_importance_overall.csv')\n",
                "importance_by_dataset = pd.read_csv(CONCLUSION_DIR / 'physical_feature_importance_by_dataset.csv')\n",
                "sensitivity = pd.read_csv(CONCLUSION_DIR / 'physical_perturbation_sensitivity.csv')\n",
                "pattern_links = pd.read_csv(CONCLUSION_DIR / 'physical_pattern_links.csv')\n",
            ],
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["## Diccionario fisico\n"]},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["dictionary.head(24)\n"],
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["## Ranking global de importancia fisica\n"]},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "importance_overall.loc[importance_overall['base_variable'].str.startswith('sensor_')].head(12)\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Image(filename=str(FIGURES_DIR / 'physical_feature_importance_top_sensors.png')))\n"
            ],
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["## Ranking por subset\n"]},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "importance_by_dataset.sort_values(['dataset', 'rank_in_dataset']).groupby('dataset').head(8)\n"
            ],
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["## Sensibilidad por perturbacion\n"]},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sensitivity.sort_values(['delta_cmapss_score', 'mean_abs_prediction_shift'], ascending=False).head(15)\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Image(filename=str(FIGURES_DIR / 'physical_perturbation_sensitivity_top_sensors.png')))\n"
            ],
        },
        {"cell_type": "markdown", "metadata": {}, "source": ["## Conexion con patrones FD003/FD004\n"]},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["pattern_links\n"],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build physical-operational conclusion artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    CONCLUSION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dictionary = sensor_dictionary()
    dictionary.to_csv(CONCLUSION_DIR / "physical_sensor_dictionary.csv", index=False)

    runs = fit_all_final_models()
    by_dataset, overall = build_importance_tables(runs, dictionary)
    by_dataset.to_csv(CONCLUSION_DIR / "physical_feature_importance_by_dataset.csv", index=False)
    overall.to_csv(CONCLUSION_DIR / "physical_feature_importance_overall.csv", index=False)
    plot_importance(overall)

    sensitivity = build_sensitivity_table(runs, by_dataset, dictionary)
    sensitivity.to_csv(CONCLUSION_DIR / "physical_perturbation_sensitivity.csv", index=False)
    plot_sensitivity(sensitivity)

    pattern_links = build_pattern_links(dictionary)
    pattern_links.to_csv(CONCLUSION_DIR / "physical_pattern_links.csv", index=False)

    write_interpretation_note()
    write_payload(dictionary, overall, sensitivity, runs)
    update_readme(overall, sensitivity)
    write_notebook()

    print("Physical-operational artifacts available:")
    for path in [
        "conclusion/physical_sensor_dictionary.csv",
        "conclusion/physical_feature_importance_by_dataset.csv",
        "conclusion/physical_feature_importance_overall.csv",
        "conclusion/physical_perturbation_sensitivity.csv",
        "conclusion/physical_pattern_links.csv",
        "conclusion/physical_operational_payload.json",
        "conclusion/figures/physical_feature_importance_top_sensors.png",
        "conclusion/figures/physical_perturbation_sensitivity_top_sensors.png",
        "notas/interpretacion_fisica_operativa.txt",
        "notebooks/conclusion/02_interpretacion_fisica_operativa.ipynb",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
