import pandas as pd
from sklearn.model_selection import train_test_split


def split_units(
    df: pd.DataFrame,
    unit_col: str = "unit",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """Split unit ids while keeping complete engines together."""
    if unit_col not in df.columns:
        raise ValueError(f"Column {unit_col!r} is not present.")

    units = sorted(df[unit_col].unique())
    train_units, eval_units = train_test_split(
        units,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return sorted(train_units), sorted(eval_units)


def train_eval_split_by_unit(
    df: pd.DataFrame,
    unit_col: str = "unit",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/eval DataFrames without sharing engines between splits."""
    train_units, eval_units = split_units(
        df=df,
        unit_col=unit_col,
        test_size=test_size,
        random_state=random_state,
    )

    train_df = df.loc[df[unit_col].isin(train_units)].copy()
    eval_df = df.loc[df[unit_col].isin(eval_units)].copy()

    return train_df, eval_df


def split_summary(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    unit_col: str = "unit",
) -> pd.DataFrame:
    """Summarize row and unit counts for a train/eval split."""
    return pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_df),
                "units": train_df[unit_col].nunique(),
            },
            {
                "split": "eval",
                "rows": len(eval_df),
                "units": eval_df[unit_col].nunique(),
            },
        ]
    )
