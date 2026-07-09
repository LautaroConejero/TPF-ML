from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


RUL_BIN_EDGES = [0, 30, 60, 90, np.inf]
RUL_BIN_LABELS = ["0-30", "30-60", "60-90", "90+"]


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for deterministic-enough notebook runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cmapss_asymmetric_score(y_true, y_pred) -> float:
    """Compute the C-MAPSS-style asymmetric score.

    Positive errors overestimate RUL and are penalized more heavily.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    penalties = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalties))


def dangerous_error_rate(y_true, y_pred, threshold: float = 20.0) -> float:
    """Percentage of predictions that overestimate true RUL by more than threshold."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) > threshold) * 100.0)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the metrics used throughout the FD001 notebooks."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    score = cmapss_asymmetric_score(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
        "cmapss_score": score,
        "cmapss_score_mean": float(score / len(y_true)),
        "dangerous_error_pct": dangerous_error_rate(y_true, y_pred),
    }


def add_rul_bins(df: pd.DataFrame, true_col: str = "y_true_rul_raw") -> pd.DataFrame:
    """Attach interpretable RUL ranges for error analysis."""
    result = df.copy()
    result["rul_bin"] = pd.cut(
        result[true_col],
        bins=RUL_BIN_EDGES,
        labels=RUL_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    return result


def prediction_frame(
    metadata_df: pd.DataFrame,
    y_pred,
    model_name: str,
    representation: str,
    true_col: str = "RUL_raw",
) -> pd.DataFrame:
    """Create a common prediction table using uncapped RUL as ground truth."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    result = pd.DataFrame(
        {
            "unit": metadata_df["unit"].to_numpy(dtype=int),
            "cycle": metadata_df["cycle"].to_numpy(dtype=int),
            "y_true_rul_raw": metadata_df[true_col].to_numpy(dtype=float),
            "y_pred_rul": y_pred,
            "model_name": model_name,
            "representation": representation,
        }
    )
    for column in ["cut_rul", "cut_cycle", "max_cycle", "window_size_used"]:
        if column in metadata_df.columns:
            result[column] = metadata_df[column].to_numpy(dtype=int)

    result["error"] = result["y_pred_rul"] - result["y_true_rul_raw"]
    result["abs_error"] = result["error"].abs()
    result["dangerous_error"] = result["error"] > 20.0
    result["conservative_error"] = result["error"] < -20.0
    return add_rul_bins(result)


def metrics_by_model(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate global metrics by model and representation."""
    rows = []
    group_cols = ["representation", "model_name"]
    for keys, group in predictions.groupby(group_cols, sort=False):
        representation, model_name = keys
        row = {
            "representation": representation,
            "model_name": model_name,
            "n_eval": len(group),
        }
        row.update(regression_metrics(group["y_true_rul_raw"], group["y_pred_rul"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)


def metrics_by_rul_bin(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by RUL bin for each model."""
    rows = []
    group_cols = ["representation", "model_name", "rul_bin"]
    for keys, group in predictions.groupby(group_cols, observed=True, sort=False):
        representation, model_name, rul_bin = keys
        row = {
            "representation": representation,
            "model_name": model_name,
            "rul_bin": str(rul_bin),
            "n_eval": len(group),
        }
        row.update(regression_metrics(group["y_true_rul_raw"], group["y_pred_rul"]))
        rows.append(row)
    return pd.DataFrame(rows)


class _RULMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class TorchMLPRegressor:
    """Small sklearn-like PyTorch regressor for FD001 baselines."""

    def __init__(
        self,
        random_state: int = 42,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 250,
        patience: int = 25,
        validation_fraction: float = 0.15,
        device: str | None = None,
    ):
        self.random_state = random_state
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: _RULMLP | None = None
        self.history_: list[dict[str, float]] = []

    def fit(self, X, y):
        set_global_seed(self.random_state)
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        indices = np.arange(len(X_np))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            shuffle=True,
        )

        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X_np[val_idx], dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_np[val_idx], dtype=torch.float32).to(self.device)

        loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        self.model_ = _RULMLP(input_dim=X_np.shape[1], dropout=self.dropout).to(self.device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()

        best_loss = np.inf
        best_state = None
        epochs_without_improvement = 0
        self.history_ = []

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            train_losses = []
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(X_val), y_val).detach().cpu())

            self.history_.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(train_losses)),
                    "val_loss": val_loss,
                }
            )

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("The MLP must be fitted before calling predict().")
        X_np = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(torch.tensor(X_np, dtype=torch.float32).to(self.device))
        return preds.detach().cpu().numpy().reshape(-1)


def fd001_model_factories(random_state: int = 42, include_dummy: bool = True):
    """Return reproducible model factories used by the FD001 notebooks."""
    factories = OrderedDict()
    if include_dummy:
        factories["DummyRegressor"] = lambda: DummyRegressor(strategy="mean")
    factories["Ridge"] = lambda: Ridge(alpha=10.0)
    factories["RandomForestRegressor"] = lambda: RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=1,
    )
    factories["MLP"] = lambda: TorchMLPRegressor(random_state=random_state)
    return factories


def train_and_predict_models(
    prepared: dict,
    representation: str,
    model_names: list[str] | None = None,
    include_dummy: bool = True,
    random_state: int = 42,
):
    """Fit selected models and return metrics, validation predictions and estimators."""
    set_global_seed(random_state)
    factories = fd001_model_factories(random_state=random_state, include_dummy=include_dummy)
    if model_names is not None:
        factories = OrderedDict((name, factories[name]) for name in model_names)

    predictions = []
    fitted_models = {}
    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_eval = prepared["X_eval"]

    for model_name, factory in factories.items():
        model = factory()
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        y_pred = model.predict(X_eval)
        predictions.append(
            prediction_frame(
                prepared["eval_df"],
                y_pred,
                model_name=model_name,
                representation=representation,
            )
        )

    prediction_table = pd.concat(predictions, ignore_index=True)
    metrics = metrics_by_model(prediction_table)
    metrics.insert(2, "n_features", len(prepared["feature_columns"]))
    metrics.insert(3, "target_used_for_training", "RUL_capped")
    return metrics, prediction_table, fitted_models


def plot_validation_diagnostics(
    predictions: pd.DataFrame,
    figure_dir: str | Path,
    prefix: str,
) -> None:
    """Save the common validation plots requested for the FD001 pipeline."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=predictions,
        x="y_true_rul_raw",
        y="y_pred_rul",
        hue="model_name",
        alpha=0.75,
    )
    max_axis = max(predictions["y_true_rul_raw"].max(), predictions["y_pred_rul"].max())
    plt.plot([0, max_axis], [0, max_axis], color="black", linestyle="--", linewidth=1)
    plt.xlabel("RUL real sin cap")
    plt.ylabel("RUL predicho")
    plt.title(f"{prefix}: predicho vs real")
    plt.tight_layout()
    plt.savefig(figure_dir / "predicted_vs_true.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=predictions,
        x="y_true_rul_raw",
        y="error",
        hue="model_name",
        alpha=0.75,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.axhline(20, color="tab:red", linestyle="--", linewidth=1, label="error peligroso")
    plt.xlabel("RUL real sin cap")
    plt.ylabel("Prediccion - RUL real")
    plt.title(f"{prefix}: error vs RUL real")
    plt.tight_layout()
    plt.savefig(figure_dir / "error_vs_true_rul.png", dpi=150)
    plt.close()

    bin_metrics = metrics_by_rul_bin(predictions)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=bin_metrics, x="rul_bin", y="mae", hue="model_name")
    plt.xlabel("Rango de RUL real")
    plt.ylabel("MAE")
    plt.title(f"{prefix}: MAE por rango de RUL")
    plt.tight_layout()
    plt.savefig(figure_dir / "mae_by_rul_bin.png", dpi=150)
    plt.close()

    worst_cases = (
        predictions.sort_values("abs_error", ascending=False)
        .head(20)
        .assign(case=lambda df: df["model_name"] + " | u" + df["unit"].astype(str) + " c" + df["cycle"].astype(str))
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=worst_cases, y="case", x="abs_error", hue="model_name", dodge=False)
    plt.xlabel("Error absoluto")
    plt.ylabel("Caso")
    plt.title(f"{prefix}: peores casos por error absoluto")
    plt.tight_layout()
    plt.savefig(figure_dir / "worst_cases_abs_error.png", dpi=150)
    plt.close()

    bin_metrics.to_csv(figure_dir / "metrics_by_rul_bin.csv", index=False)


def official_test_prediction_frame(prepared: dict, model, model_name: str, representation: str) -> pd.DataFrame:
    """Predict on official FD001 test last rows and return the requested columns."""
    y_pred = model.predict(prepared["X_test_last"])
    predictions = prediction_frame(
        prepared["test_last_df"],
        y_pred,
        model_name=model_name,
        representation=representation,
    )
    return predictions[["unit", "y_true_rul_raw", "y_pred_rul", "error", "abs_error"]]
