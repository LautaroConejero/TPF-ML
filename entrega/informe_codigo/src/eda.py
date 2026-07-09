from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS, SENSOR_COLUMNS, SETTING_COLUMNS, unit_last_cycles


def dataset_overview(
    train: pd.DataFrame, test: pd.DataFrame, rul: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build a compact overview of train, test and optional RUL files."""
    rows = [
        _frame_overview("train", train),
        _frame_overview("test", test),
    ]
    if rul is not None:
        rows.append(_frame_overview("rul", rul))
    return pd.DataFrame(rows)


def _frame_overview(name: str, df: pd.DataFrame) -> dict[str, int | str]:
    return {
        "dataset": name,
        "rows": len(df),
        "columns": df.shape[1],
        "units": df["unit"].nunique() if "unit" in df.columns else np.nan,
        "missing_values": int(df.isna().sum().sum()),
        "duplicated_rows": int(df.duplicated().sum()),
    }


def cycle_summary(df: pd.DataFrame) -> pd.Series:
    """Describe the number of observed cycles per unit."""
    return unit_last_cycles(df)["last_cycle"].describe()


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns with missing values, sorted by missing percentage."""
    total = df.isna().sum()
    summary = pd.DataFrame(
        {
            "missing_count": total,
            "missing_pct": total / len(df) * 100,
        }
    )
    return summary.loc[summary["missing_count"] > 0].sort_values(
        ["missing_pct", "missing_count"], ascending=False
    )


def numeric_summary(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Summarize numeric columns for EDA tables."""
    if columns is None:
        columns = list(df.select_dtypes(include="number").columns)

    summary = df[columns].agg(["mean", "std", "min", "max"]).T
    quantiles = df[columns].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).T
    quantiles.columns = ["p01", "p25", "p50", "p75", "p99"]
    nunique = df[columns].nunique().rename("n_unique")

    return pd.concat([summary, quantiles, nunique], axis=1)


def constant_columns(
    df: pd.DataFrame, columns: list[str] | None = None
) -> list[str]:
    """Find columns with a single observed value."""
    if columns is None:
        columns = FEATURE_COLUMNS
    return [column for column in columns if df[column].nunique(dropna=False) <= 1]


def low_variance_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 1e-10,
) -> pd.DataFrame:
    """Find columns whose variance is at or below a threshold."""
    if columns is None:
        columns = FEATURE_COLUMNS

    variances = df[columns].var(numeric_only=True).sort_values()
    low_variance = variances[variances <= threshold]
    result = low_variance.rename("variance").reset_index()
    result.columns = ["column", "variance"]
    return result


def correlation_with_target(
    df: pd.DataFrame,
    target: str = "RUL",
    columns: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute feature correlations against a target column."""
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} is not present.")
    if columns is None:
        columns = FEATURE_COLUMNS

    correlations = df[columns + [target]].corr(method=method)[target].drop(target)
    result = correlations.rename("correlation").to_frame()
    result["abs_correlation"] = result["correlation"].abs()
    return result.sort_values("abs_correlation", ascending=False)


def train_test_distribution_shift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compare train/test distributions with simple, interpretable statistics."""
    if columns is None:
        columns = FEATURE_COLUMNS

    rows = []
    for column in columns:
        train_values = train[column].dropna()
        test_values = test[column].dropna()
        pooled_std = pd.concat([train_values, test_values]).std()
        mean_diff = test_values.mean() - train_values.mean()
        rows.append(
            {
                "column": column,
                "train_mean": train_values.mean(),
                "test_mean": test_values.mean(),
                "mean_diff": mean_diff,
                "standardized_mean_diff": mean_diff / pooled_std
                if pooled_std != 0
                else np.nan,
                "train_std": train_values.std(),
                "test_std": test_values.std(),
                "train_min": train_values.min(),
                "test_min": test_values.min(),
                "train_max": train_values.max(),
                "test_max": test_values.max(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "standardized_mean_diff", key=lambda s: s.abs(), ascending=False
    )


def sensor_columns() -> list[str]:
    return SENSOR_COLUMNS.copy()


def setting_columns() -> list[str]:
    return SETTING_COLUMNS.copy()
