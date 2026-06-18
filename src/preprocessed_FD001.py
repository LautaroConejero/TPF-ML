from pathlib import Path
import sys

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
    data = load_cmapss_subset("FD001", data_dir=data_dir)
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


def preprocessing_summary(preprocessed):
    """Build a compact summary for notebooks and quick checks."""
    return pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(preprocessed["train_df"]),
                "units": len(preprocessed["train_units"]),
                "features": len(preprocessed["feature_columns"]),
                "target_mean": preprocessed["y_train"].mean(),
                "target_min": preprocessed["y_train"].min(),
                "target_max": preprocessed["y_train"].max(),
            },
            {
                "split": "eval",
                "rows": len(preprocessed["eval_df"]),
                "units": len(preprocessed["eval_units"]),
                "features": len(preprocessed["feature_columns"]),
                "target_mean": preprocessed["y_eval"].mean(),
                "target_min": preprocessed["y_eval"].min(),
                "target_max": preprocessed["y_eval"].max(),
            },
            {
                "split": "test_last",
                "rows": len(preprocessed["test_last_df"]),
                "units": preprocessed["test_last_df"]["unit"].nunique(),
                "features": len(preprocessed["feature_columns"]),
                "target_mean": preprocessed["y_test_last"].mean(),
                "target_min": preprocessed["y_test_last"].min(),
                "target_max": preprocessed["y_test_last"].max(),
            },
        ]
    )
