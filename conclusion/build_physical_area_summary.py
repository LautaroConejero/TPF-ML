from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONCLUSION_DIR = PROJECT_ROOT / "conclusion"
FIGURES_DIR = CONCLUSION_DIR / "figures"


def sensor_label(row: pd.Series) -> str:
    symbol = row.get("physical_symbol", row["base_variable"])
    if symbol == row["base_variable"]:
        return str(row["base_variable"])
    return f"{symbol} ({row['base_variable']})"


def top_sensors(group: pd.DataFrame, value_col: str, limit: int = 4) -> str:
    ranked = group.sort_values(value_col, ascending=False).head(limit)
    return "; ".join(sensor_label(row) for _, row in ranked.iterrows())


def build_importance_by_area(importance: pd.DataFrame) -> pd.DataFrame:
    sensor_rows = importance.loc[
        importance["base_variable"].str.startswith("sensor_")
        & importance["engine_area"].notna()
    ].copy()
    rows = []
    for area, group in sensor_rows.groupby("engine_area", sort=False):
        rows.append(
            {
                "engine_area": area,
                "mean_importance_normalized": float(group["mean_importance_normalized"].sum()),
                "max_importance_normalized": float(group["max_importance_normalized"].max()),
                "top_sensors": top_sensors(group, "mean_importance_normalized"),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["mean_importance_normalized", "max_importance_normalized"],
        ascending=False,
    )
    result["rank"] = range(1, len(result) + 1)
    return result.reset_index(drop=True)


def build_sensitivity_by_area(sensitivity: pd.DataFrame) -> pd.DataFrame:
    sensor_rows = sensitivity.loc[
        sensitivity["base_variable"].str.startswith("sensor_")
        & sensitivity["engine_area"].notna()
    ].copy()
    rows = []
    for area, group in sensor_rows.groupby("engine_area", sort=False):
        top = group.sort_values(
            ["delta_cmapss_score", "mean_abs_prediction_shift"],
            ascending=False,
        ).iloc[0]
        rows.append(
            {
                "engine_area": area,
                "mean_delta_cmapss_score": float(group["delta_cmapss_score"].mean()),
                "max_delta_cmapss_score": float(group["delta_cmapss_score"].max()),
                "mean_abs_prediction_shift": float(group["mean_abs_prediction_shift"].mean()),
                "top_dataset_sensor": f"{top['dataset']} - {sensor_label(top)}",
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["mean_delta_cmapss_score", "max_delta_cmapss_score", "mean_abs_prediction_shift"],
        ascending=False,
    )
    result["rank"] = range(1, len(result) + 1)
    return result.reset_index(drop=True)


def plot_importance_by_area(importance_by_area: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_frame = importance_by_area.sort_values("mean_importance_normalized", ascending=True)
    plt.figure(figsize=(9, 5.5))
    sns.barplot(
        data=plot_frame,
        x="mean_importance_normalized",
        y="engine_area",
        color="#4C78A8",
    )
    plt.xlabel("Aggregated mean normalized importance")
    plt.ylabel("")
    plt.title("Physical importance by engine area")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "physical_importance_by_engine_area.png", dpi=160)
    plt.close()


def update_readme() -> None:
    readme_path = CONCLUSION_DIR / "README.md"
    if not readme_path.exists():
        return
    text = readme_path.read_text(encoding="utf-8")
    additions = [
        "- physical_importance_by_engine_area.csv",
        "- physical_sensitivity_by_engine_area.csv",
        "- figures/physical_importance_by_engine_area.png",
    ]
    lines = text.rstrip().splitlines()
    for item in additions:
        if item not in lines:
            lines.append(item)
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build physical summaries by engine area.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    importance_path = CONCLUSION_DIR / "physical_feature_importance_overall.csv"
    sensitivity_path = CONCLUSION_DIR / "physical_perturbation_sensitivity.csv"
    if not importance_path.exists():
        raise FileNotFoundError(
            "Missing physical_feature_importance_overall.csv. Run "
            "python conclusion/build_physical_operational_artifacts.py first."
        )
    if not sensitivity_path.exists():
        raise FileNotFoundError(
            "Missing physical_perturbation_sensitivity.csv. Run "
            "python conclusion/build_physical_operational_artifacts.py first."
        )

    importance = pd.read_csv(importance_path)
    sensitivity = pd.read_csv(sensitivity_path)
    importance_by_area = build_importance_by_area(importance)
    sensitivity_by_area = build_sensitivity_by_area(sensitivity)

    CONCLUSION_DIR.mkdir(parents=True, exist_ok=True)
    importance_by_area.to_csv(CONCLUSION_DIR / "physical_importance_by_engine_area.csv", index=False)
    sensitivity_by_area.to_csv(CONCLUSION_DIR / "physical_sensitivity_by_engine_area.csv", index=False)
    plot_importance_by_area(importance_by_area)
    update_readme()

    print("Wrote physical area summary artifacts:")
    print("- conclusion/physical_importance_by_engine_area.csv")
    print("- conclusion/physical_sensitivity_by_engine_area.csv")
    print("- conclusion/figures/physical_importance_by_engine_area.png")


if __name__ == "__main__":
    main()
