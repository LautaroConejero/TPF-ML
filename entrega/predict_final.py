from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import add_train_rul, last_cycle_rows, read_cmapss_file, read_rul_file
from src.fd001_experiment_utils import make_lgbm_from_search_params, weights_from_scheme
from src.fd001_modeling import metrics_by_model
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


SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def normalize_subset(subset: str) -> str:
    subset = subset.strip().upper()
    if subset == "ALL":
        return subset
    if not subset.startswith("FD"):
        subset = f"FD{subset}"
    if subset not in SUBSETS:
        raise ValueError(f"Subset invalido: {subset}. Use FD001, FD002, FD003, FD004 o all.")
    return subset


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_train_test(data_dir: Path, subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    if not train_path.exists():
        raise FileNotFoundError(f"No existe {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"No existe {test_path}")
    return read_cmapss_file(train_path), read_cmapss_file(test_path)


def maybe_attach_true_rul(data_dir: Path, subset: str, test_last: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    rul_path = data_dir / f"RUL_{subset}.txt"
    result = test_last.copy()
    if rul_path.exists():
        result = result.merge(read_rul_file(rul_path), on="unit", how="left")
        result = result.rename(columns={"final_rul": "RUL_raw"})
        result["RUL"] = result["RUL_raw"].clip(upper=rul_cap)
    return result


def make_clean_predictions(
    test_last: pd.DataFrame,
    y_pred: np.ndarray,
    subset: str,
    model_name: str,
    representation: str,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "unit": test_last["unit"].to_numpy(dtype=int),
            "cycle": test_last["cycle"].to_numpy(dtype=int),
            "predicted_RUL": np.clip(np.asarray(y_pred, dtype=float), 0.0, None),
            "dataset": subset,
            "model_name": model_name,
            "representation": representation,
        }
    )
    if "window_size_used" in test_last.columns:
        result["window_size_used"] = test_last["window_size_used"].to_numpy(dtype=int)
    return result


def add_optional_metrics_columns(predictions: pd.DataFrame, test_last: pd.DataFrame) -> pd.DataFrame:
    if "RUL_raw" not in test_last.columns:
        return predictions
    result = predictions.copy()
    result["y_true_rul_raw"] = test_last["RUL_raw"].to_numpy(dtype=float)
    result["error"] = result["predicted_RUL"] - result["y_true_rul_raw"]
    result["abs_error"] = result["error"].abs()
    result["dangerous_error"] = result["error"] > 20.0
    result["conservative_error"] = result["error"] < -20.0
    return result


def fd001_predict(data_dir: Path, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    payload = read_json(PROJECT_ROOT / "configs" / "FD001" / "fd001_best_model_config.json")
    config = payload["final_model"]
    preprocessing = payload["preprocessing"]
    train_raw, test_raw = load_train_test(data_dir, "FD001")

    rul_cap = int(preprocessing["rul_cap"])
    window_size = int(preprocessing["window_size"])
    train = add_train_rul(train_raw, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    train["RUL"] = train["RUL"].clip(upper=rul_cap)

    base_columns = fd001_feature_columns(drop_columns=preprocessing.get("dropped_columns"))
    train_temporal = make_temporal_features(
        train,
        endpoints_df=train,
        feature_columns=base_columns,
        window_size=window_size,
    )
    test_last = maybe_attach_true_rul(data_dir, "FD001", last_cycle_rows(test_raw), rul_cap)
    test_temporal = make_temporal_features(
        test_raw,
        endpoints_df=test_last,
        feature_columns=base_columns,
        window_size=window_size,
    )
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
    model = make_lgbm_from_search_params(model_config, random_state=random_state)
    weights = weights_from_scheme(train_temporal["RUL_raw"], config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(x_train, train_temporal["RUL"])
    else:
        model.fit(x_train, train_temporal["RUL"], sample_weight=weights)

    representation = f"temporal_w{window_size}"
    predictions = make_clean_predictions(
        test_temporal,
        model.predict(x_test),
        "FD001",
        config["model_name"],
        representation,
    )
    predictions = add_optional_metrics_columns(predictions, test_temporal)
    metrics = metrics_from_clean(predictions, config["model_name"], representation)
    return predictions, metrics


def fd002_like_predict(
    data_dir: Path,
    subset: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    payload = read_json(PROJECT_ROOT / "configs" / subset / f"{subset.lower()}_best_model_config.json")
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
    train_raw, test_raw = load_train_test(data_dir, subset)
    train = add_train_rul(train_raw, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    train["RUL"] = train["RUL"].clip(upper=model_config["rul_cap"])

    condition_preprocessor = fit_condition_preprocessor(train, random_state=random_state)
    train_aug = add_fd002_condition_features(train, condition_preprocessor)
    test_source_aug = add_fd002_condition_features(test_raw, condition_preprocessor)
    test_last = maybe_attach_true_rul(data_dir, subset, last_cycle_rows(test_raw), model_config["rul_cap"])
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

    extra_columns = train_extra_columns or test_extra_columns
    feature_columns = temporal_columns + extra_columns
    _, x_train, x_test = scale_temporal_frames(train_temporal, test_temporal, feature_columns)

    model = make_model(model_config, random_state=random_state)
    weights = fd002_weights_from_scheme(train_temporal["RUL_raw"], model_config["sample_weight_scheme"])
    if weights is None:
        model.fit(x_train, train_temporal["RUL"])
    else:
        model.fit(x_train, train_temporal["RUL"], sample_weight=weights)

    predictions = make_clean_predictions(
        test_temporal,
        model.predict(x_test),
        subset,
        model_config["candidate_label"],
        model_config["representation"],
    )
    predictions = add_optional_metrics_columns(predictions, test_temporal)
    metrics = metrics_from_clean(predictions, model_config["candidate_label"], model_config["representation"])
    return predictions, metrics


def fd003_predict(data_dir: Path, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    config = read_json(PROJECT_ROOT / "configs" / "FD003" / "fd003_final_candidate_config.json")
    train_raw, test_raw = load_train_test(data_dir, "FD003")
    rul_cap = int(config["rul_cap"])
    window_size = int(config["window_size"])

    train = add_train_rul(train_raw, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    train["RUL"] = train["RUL"].clip(upper=rul_cap)

    base_feature_columns, _ = fd003_feature_columns(train)
    train_temporal = make_temporal_features(
        train,
        endpoints_df=train,
        feature_columns=base_feature_columns,
        window_size=window_size,
    )
    test_last = maybe_attach_true_rul(data_dir, "FD003", last_cycle_rows(test_raw), rul_cap)
    test_temporal = make_temporal_features(
        test_raw,
        endpoints_df=test_last,
        feature_columns=base_feature_columns,
        window_size=window_size,
    )

    relevant = [sensor for sensor in FD003_RELEVANT_SENSORS if sensor in base_feature_columns]
    train_extra = make_fault_sensitive_feature_frame(train, train, relevant, window_size)
    test_extra = make_fault_sensitive_feature_frame(test_raw, test_last, relevant, window_size)
    extra_columns = [column for column in train_extra.columns if column not in {"unit", "cycle"}]
    train_temporal = train_temporal.merge(train_extra, on=["unit", "cycle"], how="left")
    test_temporal = test_temporal.merge(test_extra, on=["unit", "cycle"], how="left")
    feature_columns = temporal_feature_columns(base_feature_columns) + extra_columns

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

    model = make_lgbm_regressor_from_config(config, random_state=random_state)
    weights = sample_weights_for_scheme(train_temporal["RUL_raw"], config.get("sample_weight_scheme", "none"))
    if weights is None:
        model.fit(x_train, train_temporal["RUL"])
    else:
        model.fit(x_train, train_temporal["RUL"], sample_weight=weights)

    representation = f"temporal_w{window_size}_{config['feature_set']}"
    predictions = make_clean_predictions(
        test_temporal,
        model.predict(x_test),
        "FD003",
        config["model_name"],
        representation,
    )
    predictions = add_optional_metrics_columns(predictions, test_temporal)
    metrics = metrics_from_clean(predictions, config["model_name"], representation)
    return predictions, metrics


def metrics_from_clean(predictions: pd.DataFrame, model_name: str, representation: str) -> pd.DataFrame | None:
    if "y_true_rul_raw" not in predictions.columns:
        return None
    frame = predictions.rename(columns={"predicted_RUL": "y_pred_rul"}).copy()
    frame["model_name"] = model_name
    frame["representation"] = representation
    frame["RUL_raw"] = frame["y_true_rul_raw"]
    return metrics_by_model(frame)


def predict_subset(data_dir: Path, subset: str, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if subset == "FD001":
        return fd001_predict(data_dir, random_state)
    if subset == "FD002":
        return fd002_like_predict(data_dir, "FD002", random_state)
    if subset == "FD003":
        return fd003_predict(data_dir, random_state)
    if subset == "FD004":
        return fd002_like_predict(data_dir, "FD004", random_state)
    raise ValueError(subset)


def write_subset_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame | None,
    output_dir: Path,
    subset: str,
    include_diagnostics: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_columns = ["unit", "cycle", "predicted_RUL", "dataset", "model_name", "representation"]
    if "window_size_used" in predictions.columns:
        clean_columns.append("window_size_used")
    clean = predictions[clean_columns].copy()
    clean.to_csv(output_dir / f"{subset.lower()}_final_predictions.csv", index=False)

    if include_diagnostics:
        predictions.to_csv(output_dir / f"{subset.lower()}_final_predictions_with_diagnostics.csv", index=False)
    if metrics is not None:
        metrics.to_csv(output_dir / f"{subset.lower()}_final_metrics.csv", index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena el modelo final de C-MAPSS para uno o todos los subsets y genera "
            "predicciones limpias por motor. No requiere archivos RUL_FD00X.txt."
        )
    )
    parser.add_argument("--subset", default="all", help="FD001, FD002, FD003, FD004 o all.")
    parser.add_argument("--data-dir", default="CMAPSSData", help="Directorio con train_FD00X.txt y test_FD00X.txt.")
    parser.add_argument("--output-dir", default="predictions/final_executable", help="Directorio de salida.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Guarda columnas de verdad/error si RUL_FD00X.txt esta disponible.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    subset_arg = normalize_subset(args.subset)
    subsets = SUBSETS if subset_arg == "ALL" else (subset_arg,)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    all_predictions = []
    all_metrics = []
    for subset in subsets:
        predictions, metrics = predict_subset(data_dir, subset, random_state=args.random_state)
        write_subset_outputs(predictions, metrics, output_dir, subset, args.include_diagnostics)
        all_predictions.append(predictions)
        if metrics is not None:
            metric_row = metrics.copy()
            metric_row.insert(0, "dataset", subset)
            all_metrics.append(metric_row)
        print(f"{subset}: wrote {len(predictions)} predictions")

    if len(all_predictions) > 1:
        combined = pd.concat(
            [
                frame[["unit", "cycle", "predicted_RUL", "dataset", "model_name", "representation"]]
                for frame in all_predictions
            ],
            ignore_index=True,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_dir / "all_final_predictions.csv", index=False)
    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(output_dir / "all_final_metrics.csv", index=False)


if __name__ == "__main__":
    main()
