from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import FEATURE_COLUMNS, add_train_rul, last_cycle_rows, load_cmapss_subset
from src.data_splitting import split_units


FD001_CONSTANT_COLUMNS = [
    "setting_3",
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

DEFAULT_RUL_CAP = 125
DEFAULT_CUT_RULS = (20, 50, 80, 110, 140)
DEFAULT_TEMPORAL_WINDOW = 30


def _resolve_data_dir(data_dir):
    data_dir = Path(data_dir)
    if data_dir.is_absolute() or data_dir.exists():
        return data_dir

    project_data_dir = PROJECT_ROOT / data_dir
    if project_data_dir.exists():
        return project_data_dir
    return data_dir


def fd001_feature_columns(drop_columns=None):
    """Return FD001 feature columns after removing constant columns."""
    if drop_columns is None:
        drop_columns = FD001_CONSTANT_COLUMNS

    drop_set = set(drop_columns)
    return [column for column in FEATURE_COLUMNS if column not in drop_set]


def prepare_fd001_current_cycle(
    data_dir="CMAPSSData",
    eval_size=0.2,
    random_state=42,
    max_rul=DEFAULT_RUL_CAP,
    drop_columns=None,
):
    """Prepare FD001 current-cycle data for baseline models.

    Split is done by complete engine units. The scaler is fit only on train.
    Test uses the last observed row of each engine, matching C-MAPSS labels.
    """
    data = load_cmapss_subset("FD001", data_dir=_resolve_data_dir(data_dir))
    feature_columns = fd001_feature_columns(drop_columns)
    dropped_columns = list(drop_columns or FD001_CONSTANT_COLUMNS)

    train = add_train_rul(data.train, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    if max_rul is not None:
        train["RUL"] = train["RUL"].clip(upper=max_rul)

    train_units, eval_units = split_units(
        train,
        unit_col="unit",
        test_size=eval_size,
        random_state=random_state,
    )
    train_df = train.loc[train["unit"].isin(train_units)].copy()
    eval_df = train.loc[train["unit"].isin(eval_units)].copy()

    test_last_df = last_cycle_rows(data.test).merge(data.rul, on="unit", how="left")
    test_last_df = test_last_df.rename(columns={"final_rul": "RUL_raw"})
    test_last_df["RUL"] = test_last_df["RUL_raw"]
    if max_rul is not None:
        test_last_df["RUL"] = test_last_df["RUL"].clip(upper=max_rul)

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train_df[feature_columns]),
        columns=feature_columns,
        index=train_df.index,
    )
    X_eval = pd.DataFrame(
        scaler.transform(eval_df[feature_columns]),
        columns=feature_columns,
        index=eval_df.index,
    )
    X_test_last = pd.DataFrame(
        scaler.transform(test_last_df[feature_columns]),
        columns=feature_columns,
        index=test_last_df.index,
    )

    return {
        "feature_columns": feature_columns,
        "dropped_columns": dropped_columns,
        "train_units": train_units,
        "eval_units": eval_units,
        "scaler": scaler,
        "train_df": train_df,
        "eval_df": eval_df,
        "test_last_df": test_last_df,
        "X_train": X_train,
        "y_train": train_df["RUL"].copy(),
        "X_eval": X_eval,
        "y_eval": eval_df["RUL"].copy(),
        "X_test_last": X_test_last,
        "y_test_last": test_last_df["RUL"].copy(),
    }


def make_fd001_artificial_cutoffs(
    df: pd.DataFrame,
    cut_ruls=DEFAULT_CUT_RULS,
    max_rul=DEFAULT_RUL_CAP,
    units=None,
) -> pd.DataFrame:
    """Create test-like validation rows by cutting run-to-failure trajectories.

    For each selected unit and each requested RUL, the cutoff cycle is
    ``max_cycle - cut_rul``. Rows whose cutoff would be before cycle 1 are
    skipped. The target used for final metrics is always ``RUL_raw``.
    """
    source = df.sort_values(["unit", "cycle"]).copy()
    if units is not None:
        unit_set = set(units)
        source = source.loc[source["unit"].isin(unit_set)].copy()

    rows = []
    for unit, group in source.groupby("unit", sort=True):
        max_cycle = int(group["cycle"].max())
        for cut_rul in cut_ruls:
            cut_rul = int(cut_rul)
            cut_cycle = max_cycle - cut_rul
            if cut_cycle < 1:
                continue

            cutoff_row = group.loc[group["cycle"] <= cut_cycle].tail(1)
            if cutoff_row.empty:
                continue

            row = cutoff_row.iloc[0].copy()
            row["cut_rul"] = cut_rul
            row["cut_cycle"] = int(row["cycle"])
            row["max_cycle"] = max_cycle
            row["RUL_raw"] = float(cut_rul)
            row["RUL"] = min(float(cut_rul), float(max_rul)) if max_rul is not None else float(cut_rul)
            row["is_artificial_cutoff"] = True
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(source.columns) + ["cut_rul", "cut_cycle", "max_cycle", "RUL_raw", "RUL", "is_artificial_cutoff"])

    result = pd.DataFrame(rows).sort_values(["unit", "cut_rul"]).reset_index(drop=True)
    for column in ["unit", "cycle", "cut_rul", "cut_cycle", "max_cycle"]:
        result[column] = result[column].astype(int)
    return result


def prepare_fd001_test_last(
    data_dir="CMAPSSData",
    feature_columns=None,
    scaler: StandardScaler | None = None,
    max_rul=DEFAULT_RUL_CAP,
):
    """Prepare the official FD001 test set at the last observed cycle per unit."""
    data = load_cmapss_subset("FD001", data_dir=_resolve_data_dir(data_dir))
    if feature_columns is None:
        feature_columns = fd001_feature_columns()

    test_last_df = last_cycle_rows(data.test).merge(data.rul, on="unit", how="left")
    test_last_df = test_last_df.rename(columns={"final_rul": "RUL_raw"})
    test_last_df["RUL"] = test_last_df["RUL_raw"]
    if max_rul is not None:
        test_last_df["RUL"] = test_last_df["RUL"].clip(upper=max_rul)

    X_test_last = test_last_df[feature_columns].copy()
    if scaler is not None:
        X_test_last = pd.DataFrame(
            scaler.transform(X_test_last),
            columns=feature_columns,
            index=test_last_df.index,
        )

    return {
        "test_df": data.test,
        "test_last_df": test_last_df,
        "X_test_last": X_test_last,
        "y_test_last": test_last_df["RUL"].copy(),
        "y_test_last_raw": test_last_df["RUL_raw"].copy(),
    }


def prepare_fd001_current_cycle_with_cutoff_eval(
    data_dir="CMAPSSData",
    eval_size=0.2,
    random_state=42,
    max_rul=DEFAULT_RUL_CAP,
    cut_ruls=DEFAULT_CUT_RULS,
    drop_columns=None,
):
    """Prepare current-cycle FD001 data with artificial cutoff validation.

    Training examples come only from train units. Evaluation examples come from
    held-out units cut at test-like RUL horizons, avoiding the invalid
    "last-cycle only" validation where every RUL is zero.
    """
    data = load_cmapss_subset("FD001", data_dir=_resolve_data_dir(data_dir))
    feature_columns = fd001_feature_columns(drop_columns)
    dropped_columns = list(drop_columns or FD001_CONSTANT_COLUMNS)

    train = add_train_rul(data.train, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    if max_rul is not None:
        train["RUL"] = train["RUL"].clip(upper=max_rul)

    train_units, eval_units = split_units(
        train,
        unit_col="unit",
        test_size=eval_size,
        random_state=random_state,
    )
    train_df = train.loc[train["unit"].isin(train_units)].copy()
    eval_source_df = train.loc[train["unit"].isin(eval_units)].copy()
    eval_cutoff_df = make_fd001_artificial_cutoffs(
        eval_source_df,
        cut_ruls=cut_ruls,
        max_rul=max_rul,
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train_df[feature_columns]),
        columns=feature_columns,
        index=train_df.index,
    )
    X_eval = pd.DataFrame(
        scaler.transform(eval_cutoff_df[feature_columns]),
        columns=feature_columns,
        index=eval_cutoff_df.index,
    )

    test_prepared = prepare_fd001_test_last(
        data_dir=data_dir,
        feature_columns=feature_columns,
        scaler=scaler,
        max_rul=max_rul,
    )

    return {
        "feature_columns": feature_columns,
        "dropped_columns": dropped_columns,
        "train_units": train_units,
        "eval_units": eval_units,
        "scaler": scaler,
        "train_df": train_df,
        "eval_source_df": eval_source_df,
        "eval_df": eval_cutoff_df,
        "eval_cutoff_df": eval_cutoff_df,
        "test_last_df": test_prepared["test_last_df"],
        "X_train": X_train,
        "y_train": train_df["RUL"].copy(),
        "y_train_raw": train_df["RUL_raw"].copy(),
        "X_eval": X_eval,
        "y_eval": eval_cutoff_df["RUL"].copy(),
        "y_eval_raw": eval_cutoff_df["RUL_raw"].copy(),
        "X_test_last": test_prepared["X_test_last"],
        "y_test_last": test_prepared["y_test_last"],
        "y_test_last_raw": test_prepared["y_test_last_raw"],
    }


def _linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values.astype(float), deg=1)[0])


def make_temporal_features(
    source_df: pd.DataFrame,
    endpoints_df: pd.DataFrame | None = None,
    feature_columns=None,
    window_size=DEFAULT_TEMPORAL_WINDOW,
) -> pd.DataFrame:
    """Build window features ending at each endpoint row.

    The returned frame keeps endpoint metadata and appends seven temporal
    summaries for each feature: last, mean, std, min, max, delta and slope.
    """
    if feature_columns is None:
        feature_columns = fd001_feature_columns()

    source = source_df.sort_values(["unit", "cycle"]).copy()
    endpoints = source if endpoints_df is None else endpoints_df.sort_values(["unit", "cycle"]).copy()

    history_by_unit = {
        unit: group.reset_index(drop=True)
        for unit, group in source.groupby("unit", sort=True)
    }

    rows = []
    for _, endpoint in endpoints.iterrows():
        unit = endpoint["unit"]
        cycle = endpoint["cycle"]
        history = history_by_unit[unit]
        window = history.loc[history["cycle"] <= cycle].tail(window_size)
        if window.empty:
            continue

        row = endpoint.to_dict()
        row["window_size_used"] = len(window)
        for column in feature_columns:
            values = window[column].to_numpy(dtype=float)
            row[f"{column}_last"] = values[-1]
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
            row[f"{column}_delta"] = float(values[-1] - values[0])
            row[f"{column}_slope"] = _linear_slope(values)

        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


def temporal_feature_columns(feature_columns, stats=None):
    """Return deterministic temporal feature names for base FD001 columns."""
    if stats is None:
        stats = ("last", "mean", "std", "min", "max", "delta", "slope")
    return [f"{column}_{stat}" for column in feature_columns for stat in stats]


def prepare_fd001_temporal_with_cutoff_eval(
    data_dir="CMAPSSData",
    eval_size=0.2,
    random_state=42,
    max_rul=DEFAULT_RUL_CAP,
    cut_ruls=DEFAULT_CUT_RULS,
    window_size=DEFAULT_TEMPORAL_WINDOW,
    drop_columns=None,
):
    """Prepare FD001 temporal-window features with artificial cutoff validation."""
    data = load_cmapss_subset("FD001", data_dir=_resolve_data_dir(data_dir))
    base_feature_columns = fd001_feature_columns(drop_columns)
    dropped_columns = list(drop_columns or FD001_CONSTANT_COLUMNS)

    train = add_train_rul(data.train, max_rul=None)
    train["RUL_raw"] = train["RUL"]
    if max_rul is not None:
        train["RUL"] = train["RUL"].clip(upper=max_rul)

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

    test_prepared = prepare_fd001_test_last(
        data_dir=data_dir,
        feature_columns=base_feature_columns,
        scaler=None,
        max_rul=max_rul,
    )
    test_temporal_df = make_temporal_features(
        test_prepared["test_df"],
        endpoints_df=test_prepared["test_last_df"],
        feature_columns=base_feature_columns,
        window_size=window_size,
    )
    X_test_last = pd.DataFrame(
        scaler.transform(test_temporal_df[temporal_columns]),
        columns=temporal_columns,
        index=test_temporal_df.index,
    )

    return {
        "feature_columns": temporal_columns,
        "base_feature_columns": base_feature_columns,
        "dropped_columns": dropped_columns,
        "window_size": window_size,
        "train_units": train_units,
        "eval_units": eval_units,
        "scaler": scaler,
        "train_df": train_temporal_df,
        "train_source_df": train_source_df,
        "eval_source_df": eval_source_df,
        "eval_df": eval_temporal_df,
        "eval_cutoff_df": eval_cutoff_df,
        "test_last_df": test_temporal_df,
        "X_train": X_train,
        "y_train": train_temporal_df["RUL"].copy(),
        "y_train_raw": train_temporal_df["RUL_raw"].copy(),
        "X_eval": X_eval,
        "y_eval": eval_temporal_df["RUL"].copy(),
        "y_eval_raw": eval_temporal_df["RUL_raw"].copy(),
        "X_test_last": X_test_last,
        "y_test_last": test_temporal_df["RUL"].copy(),
        "y_test_last_raw": test_temporal_df["RUL_raw"].copy(),
    }


def prepare_fd001_current_cycle_full_train_for_test(
    data_dir="CMAPSSData",
    max_rul=DEFAULT_RUL_CAP,
    drop_columns=None,
):
    """Prepare current-cycle features using all FD001 train rows for final test."""
    data = load_cmapss_subset("FD001", data_dir=_resolve_data_dir(data_dir))
    feature_columns = fd001_feature_columns(drop_columns)
    dropped_columns = list(drop_columns or FD001_CONSTANT_COLUMNS)

    train_df = add_train_rul(data.train, max_rul=None)
    train_df["RUL_raw"] = train_df["RUL"]
    if max_rul is not None:
        train_df["RUL"] = train_df["RUL"].clip(upper=max_rul)

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train_df[feature_columns]),
        columns=feature_columns,
        index=train_df.index,
    )

    test_prepared = prepare_fd001_test_last(
        data_dir=data_dir,
        feature_columns=feature_columns,
        scaler=scaler,
        max_rul=max_rul,
    )

    train_units = sorted(train_df["unit"].unique())
    return {
        "feature_columns": feature_columns,
        "dropped_columns": dropped_columns,
        "train_units": train_units,
        "eval_units": [],
        "scaler": scaler,
        "train_df": train_df,
        "eval_df": pd.DataFrame(),
        "test_last_df": test_prepared["test_last_df"],
        "X_train": X_train,
        "y_train": train_df["RUL"].copy(),
        "y_train_raw": train_df["RUL_raw"].copy(),
        "X_test_last": test_prepared["X_test_last"],
        "y_test_last": test_prepared["y_test_last"],
        "y_test_last_raw": test_prepared["y_test_last_raw"],
    }


def prepare_fd001_temporal_full_train_for_test(
    data_dir="CMAPSSData",
    max_rul=DEFAULT_RUL_CAP,
    window_size=DEFAULT_TEMPORAL_WINDOW,
    drop_columns=None,
):
    """Prepare temporal-window features using all FD001 train rows for final test."""
    data = load_cmapss_subset("FD001", data_dir=_resolve_data_dir(data_dir))
    base_feature_columns = fd001_feature_columns(drop_columns)
    dropped_columns = list(drop_columns or FD001_CONSTANT_COLUMNS)

    train_df = add_train_rul(data.train, max_rul=None)
    train_df["RUL_raw"] = train_df["RUL"]
    if max_rul is not None:
        train_df["RUL"] = train_df["RUL"].clip(upper=max_rul)

    train_temporal_df = make_temporal_features(
        train_df,
        endpoints_df=train_df,
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

    test_prepared = prepare_fd001_test_last(
        data_dir=data_dir,
        feature_columns=base_feature_columns,
        scaler=None,
        max_rul=max_rul,
    )
    test_temporal_df = make_temporal_features(
        test_prepared["test_df"],
        endpoints_df=test_prepared["test_last_df"],
        feature_columns=base_feature_columns,
        window_size=window_size,
    )
    X_test_last = pd.DataFrame(
        scaler.transform(test_temporal_df[temporal_columns]),
        columns=temporal_columns,
        index=test_temporal_df.index,
    )

    train_units = sorted(train_df["unit"].unique())
    return {
        "feature_columns": temporal_columns,
        "base_feature_columns": base_feature_columns,
        "dropped_columns": dropped_columns,
        "window_size": window_size,
        "train_units": train_units,
        "eval_units": [],
        "scaler": scaler,
        "train_df": train_temporal_df,
        "eval_df": pd.DataFrame(),
        "test_last_df": test_temporal_df,
        "X_train": X_train,
        "y_train": train_temporal_df["RUL"].copy(),
        "y_train_raw": train_temporal_df["RUL_raw"].copy(),
        "X_test_last": X_test_last,
        "y_test_last": test_temporal_df["RUL"].copy(),
        "y_test_last_raw": test_temporal_df["RUL_raw"].copy(),
    }


def preprocessing_summary(preprocessed):
    """Build a compact summary for notebooks and quick checks."""
    rows = [
        {
            "split": "train",
            "rows": len(preprocessed["train_df"]),
            "units": len(preprocessed["train_units"]),
            "features": len(preprocessed["feature_columns"]),
            "target_mean": preprocessed["y_train"].mean(),
            "target_min": preprocessed["y_train"].min(),
            "target_max": preprocessed["y_train"].max(),
        }
    ]

    y_eval_key = "y_eval_raw" if "y_eval_raw" in preprocessed else "y_eval"
    if y_eval_key in preprocessed and "eval_df" in preprocessed:
        rows.append(
            {
                "split": "eval",
                "rows": len(preprocessed["eval_df"]),
                "units": len(preprocessed["eval_units"]),
                "features": len(preprocessed["feature_columns"]),
                "target_mean": preprocessed[y_eval_key].mean(),
                "target_min": preprocessed[y_eval_key].min(),
                "target_max": preprocessed[y_eval_key].max(),
            }
        )

    y_test_key = "y_test_last_raw" if "y_test_last_raw" in preprocessed else "y_test_last"
    if y_test_key in preprocessed and "test_last_df" in preprocessed:
        rows.append(
            {
                "split": "test_last",
                "rows": len(preprocessed["test_last_df"]),
                "units": preprocessed["test_last_df"]["unit"].nunique(),
                "features": len(preprocessed["feature_columns"]),
                "target_mean": preprocessed[y_test_key].mean(),
                "target_min": preprocessed[y_test_key].min(),
                "target_max": preprocessed[y_test_key].max(),
            }
        )

    return pd.DataFrame(rows)
