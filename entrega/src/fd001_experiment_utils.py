from collections import OrderedDict
from itertools import product
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def add_bin_metric_columns(metrics, bin_metrics):
    result = metrics.copy()
    for label in ["0-30", "30-60", "60-90", "90+"]:
        safe_label = label.replace("-", "_").replace("+", "plus")
        subset = bin_metrics.loc[bin_metrics["rul_bin"].astype(str) == label]
        for _, row in subset.iterrows():
            mask = (
                (result["representation"] == row["representation"])
                & (result["model_name"] == row["model_name"])
            )
            result.loc[mask, f"mae_rul_{safe_label}"] = row["mae"]
            result.loc[mask, f"dangerous_error_pct_rul_{safe_label}"] = row["dangerous_error_pct"]
    return result


def metric_row_from_predictions(predictions, extra=None):
    from src.fd001_modeling import metrics_by_model, metrics_by_rul_bin

    row = metrics_by_model(predictions).iloc[0].to_dict()
    bins = metrics_by_rul_bin(predictions)
    row = add_bin_metric_columns(pd.DataFrame([row]), bins).iloc[0].to_dict()
    if extra:
        row.update(extra)
    return row, bins


def selection_sort(df):
    return df.sort_values(
        ["cmapss_score", "rmse", "dangerous_error_pct", "mae"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def available_boosting_factories(random_state=42):
    factories = OrderedDict()
    notes = []
    has_external = False
    factories["RandomForestRegressor"] = lambda: RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )

    try:
        from xgboost import XGBRegressor

        factories["XGBRegressor"] = lambda: XGBRegressor(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        has_external = True
        notes.append("XGBoost disponible: se incluye XGBRegressor.")
    except Exception as exc:
        notes.append(f"XGBoost no disponible: {type(exc).__name__}.")

    try:
        from lightgbm import LGBMRegressor

        factories["LGBMRegressor"] = lambda: LGBMRegressor(
            n_estimators=220,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        has_external = True
        notes.append("LightGBM disponible: se incluye LGBMRegressor.")
    except Exception as exc:
        notes.append(f"LightGBM no disponible: {type(exc).__name__}.")

    if not has_external:
        notes.append("Sin XGBoost/LightGBM: se usan fallbacks sklearn.")
        factories["HistGradientBoostingRegressor"] = lambda: HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=random_state,
        )
        factories["GradientBoostingRegressor"] = lambda: GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            random_state=random_state,
        )
        factories["ExtraTreesRegressor"] = lambda: ExtraTreesRegressor(
            n_estimators=220,
            max_depth=16,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    return factories, notes


def lgbm_factory(random_state=42):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=220,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def rul_priority_weights(y_true):
    y_true = np.asarray(y_true, dtype=float)
    return np.select(
        [y_true <= 30, (y_true > 30) & (y_true <= 60), (y_true > 60) & (y_true <= 90)],
        [4.0, 2.0, 1.5],
        default=1.0,
    )


def cmapss_weighted_objective(y_true, y_pred):
    """LightGBM objective approximating C-MAPSS with extra weight near failure."""
    error = np.clip(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float), -50.0, 50.0)
    weights = rul_priority_weights(y_true)

    early = error < 0
    grad = np.where(
        early,
        -(1.0 / 13.0) * np.exp(-error / 13.0),
        (1.0 / 10.0) * np.exp(error / 10.0),
    )
    hess = np.where(
        early,
        (1.0 / (13.0 ** 2)) * np.exp(-error / 13.0),
        (1.0 / (10.0 ** 2)) * np.exp(error / 10.0),
    )
    return weights * grad, weights * hess


def lgbm_cmapss_objective_factory(random_state=42):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective=cmapss_weighted_objective,
        n_estimators=220,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def make_rul_priority_weights(y_true, low_weight=2.0, mid_weight=1.4, high_weight=1.15):
    y_true = np.asarray(y_true, dtype=float)
    return np.select(
        [y_true <= 30, (y_true > 30) & (y_true <= 60), (y_true > 60) & (y_true <= 90)],
        [low_weight, mid_weight, high_weight],
        default=1.0,
    )


def make_cmapss_like_objective(
    over_scale=14.0,
    under_scale=18.0,
    low_weight=2.0,
    mid_weight=1.4,
    high_weight=1.15,
    clip_error=45.0,
):
    def objective(y_true, y_pred):
        error = np.clip(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float), -clip_error, clip_error)
        weights = make_rul_priority_weights(
            y_true,
            low_weight=low_weight,
            mid_weight=mid_weight,
            high_weight=high_weight,
        )
        over = error >= 0
        grad = np.where(
            over,
            (1.0 / over_scale) * np.exp(error / over_scale),
            -(1.0 / under_scale) * np.exp(-error / under_scale),
        )
        hess = np.where(
            over,
            (1.0 / (over_scale ** 2)) * np.exp(error / over_scale),
            (1.0 / (under_scale ** 2)) * np.exp(-error / under_scale),
        )
        return weights * grad, weights * hess

    return objective


def lgbm_asymmetric_objective_factory(
    random_state=42,
    over_scale=14.0,
    under_scale=18.0,
    low_weight=2.0,
    mid_weight=1.4,
    high_weight=1.15,
):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective=make_cmapss_like_objective(
            over_scale=over_scale,
            under_scale=under_scale,
            low_weight=low_weight,
            mid_weight=mid_weight,
            high_weight=high_weight,
        ),
        n_estimators=220,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def lgbm_quantile_factory(alpha=0.45, random_state=42):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=240,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def lgbm_huber_factory(alpha=0.9, random_state=42):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="huber",
        alpha=alpha,
        n_estimators=240,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def shifted_prediction_table(predictions, offset, model_name):
    from src.fd001_modeling import add_rul_bins

    result = predictions.copy()
    result["y_pred_rul"] = np.clip(result["y_pred_rul"].to_numpy(dtype=float) - float(offset), 0.0, None)
    result["model_name"] = model_name
    result["error"] = result["y_pred_rul"] - result["y_true_rul_raw"]
    result["abs_error"] = result["error"].abs()
    result["dangerous_error"] = result["error"] > 20.0
    result["conservative_error"] = result["error"] < -20.0
    if "rul_bin" in result.columns:
        result = result.drop(columns=["rul_bin"])
    return add_rul_bins(result)


def add_operational_error_columns(metrics, predictions):
    result = metrics.copy()
    extras = []
    for _, row in result.iterrows():
        mask = (
            (predictions["model_name"].astype(str) == str(row["model_name"]))
            & (predictions["representation"].astype(str) == str(row["representation"]))
        )
        for key in ["sample_weight_scheme", "training_objective"]:
            if key in result.columns and key in predictions.columns and pd.notna(row.get(key)):
                mask &= predictions[key].astype(str) == str(row[key])
        group = predictions.loc[mask]
        extras.append(
            {
                "mean_error": group["error"].mean(),
                "median_error": group["error"].median(),
                "conservative_error_pct": group["conservative_error"].mean() * 100.0,
                "p95_overestimate": group["error"].clip(lower=0).quantile(0.95),
            }
        )
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(extras)], axis=1)


def fd001_lgbm_param_space():
    return OrderedDict(
        [
            ("learning_rate", [0.02, 0.03, 0.05, 0.08]),
            ("n_estimators", [400, 700, 1000, 1300]),
            ("num_leaves", [15, 31, 47, 63]),
            ("max_depth", [4, 6, 8, -1]),
            ("min_child_samples", [10, 20, 40, 60]),
            ("subsample", [0.8, 0.9, 1.0]),
            ("colsample_bytree", [0.8, 0.9, 1.0]),
            ("reg_alpha", [0.0, 0.1, 0.5, 1.0]),
            ("reg_lambda", [0.0, 1.0, 5.0, 10.0]),
        ]
    )


def sample_lgbm_param_configs(n_configs=30, random_state=42):
    space = fd001_lgbm_param_space()
    keys = list(space)
    all_configs = [dict(zip(keys, values)) for values in product(*(space[key] for key in keys))]
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(all_configs), size=int(n_configs), replace=False)
    return [all_configs[int(idx)] for idx in indices]


def base_lgbm_bin_weights_params():
    return {
        "learning_rate": 0.035,
        "n_estimators": 220,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 0.1,
    }


def make_lgbm_from_search_params(config, random_state=42):
    from lightgbm import LGBMRegressor

    params = dict(config["params"])
    params.update(
        {
            "objective": config["objective"],
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    if config["objective"] == "quantile":
        params["alpha"] = float(config["alpha"])
    return LGBMRegressor(**params)


def make_lgbm_search_configs(n_family_a=30, n_family_b=30, random_state=42, quantile_alphas=(0.35, 0.40, 0.45)):
    configs = []
    for i, params in enumerate(sample_lgbm_param_configs(n_family_a, random_state=random_state)):
        configs.append(
            {
                "candidate_label": f"A_regression_bin_weights_search_{i:02d}",
                "family": "A_regression_bin_weights",
                "objective": "regression",
                "alpha": np.nan,
                "sample_weight_scheme": "bin_weights",
                "params": params,
            }
        )

    per_alpha = [n_family_b // len(quantile_alphas)] * len(quantile_alphas)
    for idx in range(n_family_b % len(quantile_alphas)):
        per_alpha[idx] += 1
    offset = 1000
    counter = 0
    for alpha, n_configs in zip(quantile_alphas, per_alpha):
        sampled = sample_lgbm_param_configs(n_configs, random_state=random_state + offset + int(alpha * 100))
        for params in sampled:
            configs.append(
                {
                    "candidate_label": f"B_quantile_a{int(alpha * 100):03d}_search_{counter:02d}",
                    "family": "B_quantile_no_weights",
                    "objective": "quantile",
                    "alpha": float(alpha),
                    "sample_weight_scheme": "none",
                    "params": params,
                }
            )
            counter += 1
    return configs


def params_to_json(params):
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def config_identity(config):
    alpha = "" if pd.isna(config.get("alpha", np.nan)) else f"{float(config['alpha']):.2f}"
    return "|".join(
        [
            str(config["family"]),
            str(config["objective"]),
            alpha,
            str(config["sample_weight_scheme"]),
            params_to_json(config["params"]),
        ]
    )


def evaluate_lgbm_search_config(prepared, config, random_state=42, window_size=50, rul_cap=125):
    model = make_lgbm_from_search_params(config, random_state=random_state)
    weights = weights_from_scheme(prepared["y_train_raw"], config["sample_weight_scheme"])
    predictions = fit_predict_model(
        prepared,
        model,
        model_name=config["candidate_label"],
        representation=f"temporal_w{window_size}",
        sample_weight=weights,
    )
    row, _ = metric_row_from_predictions(
        predictions,
        extra={
            "candidate_label": config["candidate_label"],
            "candidate_id": config_identity(config),
            "family": config["family"],
            "objective": config["objective"],
            "alpha": config["alpha"],
            "sample_weight_scheme": config["sample_weight_scheme"],
            "window_size": window_size,
            "rul_cap": rul_cap,
            "random_state": random_state,
            "params": params_to_json(config["params"]),
        },
    )
    row["conservative_error_pct"] = float(predictions["conservative_error"].mean() * 100.0)
    row["bias_mean"] = float(predictions["error"].mean())
    return row


def base_lgbm_candidate_config():
    return {
        "candidate_label": "base_regression_bin_weights",
        "family": "A_regression_bin_weights",
        "objective": "regression",
        "alpha": np.nan,
        "sample_weight_scheme": "bin_weights",
        "params": base_lgbm_bin_weights_params(),
    }


def configs_from_search_rows(rows):
    configs = []
    for _, row in rows.iterrows():
        configs.append(
            {
                "candidate_label": row["candidate_label"],
                "family": row["family"],
                "objective": row["objective"],
                "alpha": row["alpha"],
                "sample_weight_scheme": row["sample_weight_scheme"],
                "params": json.loads(row["params"]),
            }
        )
    return configs


def select_lgbm_robustness_candidates(search_results):
    selected_rows = []
    selected_rows.append(search_results.sort_values("cmapss_score", ascending=True).iloc[0])
    selected_rows.append(search_results.sort_values("rmse", ascending=True).iloc[0])
    selected_rows.append(search_results.sort_values(["dangerous_error_pct", "rmse"], ascending=[True, True]).iloc[0])
    selected_rows.append(
        search_results.loc[search_results["family"] == "A_regression_bin_weights"]
        .sort_values("cmapss_score", ascending=True)
        .iloc[0]
    )
    selected_rows.append(
        search_results.loc[search_results["family"] == "B_quantile_no_weights"]
        .sort_values("cmapss_score", ascending=True)
        .iloc[0]
    )

    configs = configs_from_search_rows(pd.DataFrame(selected_rows))
    configs.append(base_lgbm_candidate_config())

    unique = OrderedDict()
    for config in configs:
        unique[config_identity(config)] = config
    result = list(unique.values())
    for idx, config in enumerate(result):
        config["candidate_label"] = f"candidate_{idx:02d}_{config['candidate_label']}"
    return result


def summarize_lgbm_robustness(robustness):
    group_cols = ["candidate_id", "candidate_label", "family", "objective", "alpha", "sample_weight_scheme", "params"]
    metrics = [
        "mae",
        "rmse",
        "r2",
        "cmapss_score",
        "dangerous_error_pct",
        "conservative_error_pct",
        "bias_mean",
    ]
    grouped = robustness.groupby(group_cols, dropna=False)
    summary = grouped[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    worst = grouped.agg(
        worst_rmse=("rmse", "max"),
        worst_cmapss_score=("cmapss_score", "max"),
        worst_dangerous_error_pct=("dangerous_error_pct", "max"),
    ).reset_index()
    result = summary.merge(worst, on=group_cols, how="left")
    rename = {}
    for metric in metrics:
        rename[f"{metric}_mean"] = f"mean_{metric}"
        rename[f"{metric}_std"] = f"std_{metric}"
    result = result.rename(columns=rename)
    return result.sort_values(["mean_cmapss_score", "mean_rmse"]).reset_index(drop=True)


def fit_predict_model(prepared, model, model_name, representation, sample_weight=None):
    from src.fd001_modeling import prediction_frame

    if sample_weight is None:
        model.fit(prepared["X_train"], prepared["y_train"])
    else:
        model.fit(prepared["X_train"], prepared["y_train"], sample_weight=sample_weight)
    preds = model.predict(prepared["X_eval"])
    return prediction_frame(
        prepared["eval_df"],
        preds,
        model_name=model_name,
        representation=representation,
    )


class FlexibleMLPRegressor:
    def __init__(
        self,
        hidden_layers,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=250,
        patience=25,
        validation_fraction=0.15,
        random_state=42,
        device=None,
    ):
        import torch

        self.hidden_layers = list(hidden_layers)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None
        self.history_ = []

    def _build_model(self, input_dim):
        from torch import nn

        layers = []
        prev = input_dim
        for width in self.hidden_layers:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            prev = int(width)
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import random
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        idx = np.arange(len(X_np))
        train_idx, val_idx = train_test_split(
            idx,
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

        self.model_ = self._build_model(X_np.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = nn.MSELoss()
        best_loss = np.inf
        best_state = None
        patience_left = self.patience
        self.history_ = []

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            losses = []
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(X_val), y_val).detach().cpu())
            self.history_.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
            if patience_left <= 0:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch

        if self.model_ is None:
            raise RuntimeError("La MLP debe entrenarse antes de predecir.")
        self.model_.eval()
        X_np = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            preds = self.model_(torch.tensor(X_np, dtype=torch.float32).to(self.device))
        return np.clip(preds.detach().cpu().numpy().reshape(-1), 0.0, None)


def weights_from_scheme(y_raw, scheme):
    if scheme is None:
        return None
    if isinstance(scheme, str):
        schemes = {
            "none": None,
            "bin_weights": {0: 4.0, 30: 2.0, 60: 1.5, 90: 1.0},
            "aggressive": {0: 6.0, 30: 3.0, 60: 1.5, 90: 1.0},
            "soft": {0: 2.0, 30: 1.5, 60: 1.2, 90: 1.0},
        }
        scheme = schemes.get(scheme)
        if scheme is None:
            return None

    y_raw = np.asarray(y_raw, dtype=float)
    return np.select(
        [y_raw <= 30, (y_raw > 30) & (y_raw <= 60), (y_raw > 60) & (y_raw <= 90)],
        [scheme[0], scheme[30], scheme[60]],
        default=scheme[90],
    )


def parse_rul_cap(value):
    if pd.isna(value) or str(value) == "None":
        return None
    return int(float(value))


def parse_window_size(row, default_window_size=30):
    if pd.notna(row.get("window_size", np.nan)):
        return int(float(row["window_size"]))
    representation = str(row.get("representation", f"temporal_w{default_window_size}"))
    if "temporal_w" in representation:
        return int(representation.split("temporal_w")[-1].split("_")[0])
    return default_window_size


def final_model_from_row(row, random_state=42):
    model_name = str(row["model_name"])
    if model_name == "LGBMRegressor":
        return lgbm_factory(random_state=random_state)
    if model_name == "XGBRegressor":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "Ridge":
        return Ridge(alpha=10.0)
    if model_name == "MLP_tabular_cfg_03":
        return FlexibleMLPRegressor(
            hidden_layers=[128, 64, 32],
            dropout=0.1,
            weight_decay=1e-4,
            lr=1e-3,
            batch_size=256,
            max_epochs=250,
            patience=25,
            random_state=random_state,
        )
    raise ValueError(f"Modelo final no soportado: {model_name}")


def infer_window_size(row):
    if "window_size" in row and pd.notna(row["window_size"]):
        return int(float(row["window_size"]))
    rep = str(row.get("representation", ""))
    if "temporal_w" in rep:
        return int(rep.split("temporal_w")[-1].split("_")[0])
    return np.nan


def normalize_rows(df, source, selection_notes, sample_weight_scheme="none", base_rul_cap=125):
    metric_columns = ["mae", "rmse", "r2", "cmapss_score", "cmapss_score_mean", "dangerous_error_pct"]
    rows = []
    for _, row in df.iterrows():
        item = {
            "source": source,
            "model_name": row.get("model_name"),
            "representation": row.get("representation", ""),
            "window_size": infer_window_size(row),
            "rul_cap": row.get("rul_cap", base_rul_cap),
            "sample_weight_scheme": row.get("sample_weight_scheme", sample_weight_scheme),
            "selection_notes": selection_notes,
        }
        for col in metric_columns:
            item[col] = row.get(col, np.nan)
        rows.append(item)
    return rows


def load_if_exists(results_dir, name):
    path = results_dir / name
    if not path.exists():
        print(f"Falta {name}; se omite.")
        return None
    return pd.read_csv(path)


def mlp_seed_summary_as_row(summary):
    metric_columns = ["mae", "rmse", "r2", "cmapss_score", "cmapss_score_mean", "dangerous_error_pct"]
    wide = {
        "source": "mlp_cfg03_seed_summary",
        "model_name": "MLP_tabular_cfg_03",
        "representation": "temporal_w30",
        "window_size": 30,
        "rul_cap": 125,
        "sample_weight_scheme": "none",
        "selection_notes": "MLP cfg03 seed stability mean",
    }
    for _, row in summary.iterrows():
        metric = row["metric"]
        if metric in metric_columns:
            wide[metric] = row["mean"]
    return wide


def first_match(df, **filters):
    mask = pd.Series(True, index=df.index)
    for col, value in filters.items():
        if value is None:
            mask &= df[col].isna()
        else:
            mask &= df[col].astype(str) == str(value)
    result = df.loc[mask].copy()
    if result.empty:
        raise ValueError(f"No hay fila para filtros: {filters}")
    return result.iloc[0]


def row_from_series(label, row, note, sample_weight_scheme=None):
    return {
        "model_label": label,
        "representation": row.get("representation", ""),
        "window_size": row.get("window_size", np.nan),
        "rul_cap": row.get("rul_cap", np.nan),
        "sample_weight_scheme": sample_weight_scheme if sample_weight_scheme is not None else row.get("sample_weight_scheme", "none"),
        "mae": row["mae"],
        "rmse": row["rmse"],
        "r2": row["r2"],
        "cmapss_score": row["cmapss_score"],
        "dangerous_error_pct": row["dangerous_error_pct"],
        "note": note,
    }


def mlp_summary_value(summary, metric):
    row = summary.loc[summary["metric"] == metric].iloc[0]
    return row["mean"]


def save_bar(summary_plot, metric, output_path, xlabel, plt, sns):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=summary_plot, y="model_label", x=metric, color="#4C78A8")
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()
