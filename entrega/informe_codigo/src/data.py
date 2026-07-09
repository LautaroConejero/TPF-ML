from pathlib import Path

import pandas as pd


SUBSETS = ("FD001", "FD002", "FD003", "FD004")

INDEX_COLUMNS = ["unit", "cycle"]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS
COLUMN_NAMES = INDEX_COLUMNS + FEATURE_COLUMNS


class CmapssData:
    """Container for one C-MAPSS subset."""

    def __init__(
        self,
        subset: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        rul: pd.DataFrame,
    ):
        self.subset = subset
        self.train = train
        self.test = test
        self.rul = rul


def normalize_subset_name(subset: str) -> str:
    subset = subset.upper().strip()
    if not subset.startswith("FD"):
        subset = f"FD{subset}"
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}. Expected one of {SUBSETS}.")
    return subset


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Read a C-MAPSS train/test text file with stable column names."""
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} columns in {path}, found {df.shape[1]}."
        )
    return df


def read_rul_file(path: str | Path) -> pd.DataFrame:
    """Read the test RUL file and attach unit ids in file order."""
    path = Path(path)
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["final_rul"])
    rul.insert(0, "unit", range(1, len(rul) + 1))
    return rul


def load_cmapss_subset(
    subset: str = "FD001", data_dir: str | Path = "CMAPSSData"
) -> CmapssData:
    """Load train, test and RUL files for a C-MAPSS subset."""
    subset = normalize_subset_name(subset)
    data_dir = Path(data_dir)

    train = read_cmapss_file(data_dir / f"train_{subset}.txt")
    test = read_cmapss_file(data_dir / f"test_{subset}.txt")
    rul = read_rul_file(data_dir / f"RUL_{subset}.txt")

    return CmapssData(subset=subset, train=train, test=test, rul=rul)


def add_train_rul(
    train: pd.DataFrame, max_rul: int | float | None = None
) -> pd.DataFrame:
    """Add per-row remaining useful life for run-to-failure training data."""
    df = train.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    if max_rul is not None:
        df["RUL"] = df["RUL"].clip(upper=max_rul)
    return df


def unit_last_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unit with its observed number of cycles."""
    return (
        df.groupby("unit", as_index=False)["cycle"]
        .max()
        .rename(columns={"cycle": "last_cycle"})
    )


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the last observed row for each unit."""
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit").reset_index(drop=True)
