from __future__ import annotations
import numpy as np
import pandas as pd
import optuna
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

@dataclass
class ForecastResult:
    coin_id: str
    vs_currency: str
    horizon_days: int
    last_date: str
    last_price: float
    predictions: List[Dict[str, float]]
    diagnostics: Dict[str, float]

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _build_xgb_pipeline(cat_cols: List[str], num_cols: List[str], params: dict) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    model = xgb.XGBRegressor(
        **params,
        n_estimators=params.get("n_estimators", 2000),
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )
    return Pipeline([("pre", pre), ("model", model)])

def tune_xgb_heavy(df: pd.DataFrame, feature_cols: List[str], y_col: str = "y",
                   trials: int = 60, cv_splits: int = 5) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Compute-heavy tuning using Optuna + walk-forward CV.
    Objective: minimize RMSE on next-day returns.
    """
    X = df[feature_cols].copy()
    y = df[y_col].astype(float).values

    cat_cols = [c for c in feature_cols if c in ("dow", "dom", "month")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "n_estimators": trial.suggest_int("n_estimators", 800, 3500),
        }

        pipe = _build_xgb_pipeline(cat_cols, num_cols, params)
        rmses = []
        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            rmses.append(_rmse(yte, pred))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best_params = study.best_params
    pipe = _build_xgb_pipeline(cat_cols, num_cols, best_params)
    pipe.fit(X, y)

    return pipe, {"xgb_cv_rmse": float(study.best_value)}

def fit_sarimax(df: pd.DataFrame, y_col: str = "y", exog_cols: List[str] | None = None) -> Tuple[object, Dict[str, float]]:
    """
    SARIMAX on returns with exogenous regressors. Uses a strong but sane default order.
    (You can grid-search orders too, but it gets extremely expensive.)
    """
    y = df[y_col].astype(float).values
    exog = df[exog_cols].astype(float).values if exog_cols else None

    # default order tuned for noisy returns: small ARMA + no seasonal by default
    model = SARIMAX(
        y,
        exog=exog,
        order=(2, 0, 2),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return model, {"sarimax_aic": float(model.aic)}

def iterative_forecast(df_feat: pd.DataFrame, feature_cols: List[str], xgb_pipe: Pipeline,
                       sarimax_model: object, exog_cols_for_sarimax: List[str] | None,
                       horizon_days: int) -> List[Dict[str, float]]:
    """
    Predict next-day returns iteratively, convert to price path.
    Ensemble: average of XGB return forecast + SARIMAX return forecast.
    """
    df = df_feat.copy()
    last_row = df.iloc[-1].copy()
    last_date = pd.to_datetime(last_row["date"])
    last_price = float(np.exp(last_row["log_price"])) if "log_price" in df.columns else float(df_feat.iloc[-1]["price"])

    # We'll maintain a small rolling window dataframe to update engineered lagged features.
    # For simplicity: append synthetic rows with predicted price, then recompute features would be expensive.
    # Instead, approximate by updating only the fields used (log_price, ret1, rolling stats) minimally.
    # This is already compute-heavy overall due to tuning; you can make this even heavier by recomputing features each step.

    # For better accuracy at compute expense: recompute features each step using a helper
    # is ideal, but requires carrying raw daily frame. We'll do lightweight update.
    history_rets = df["ret1"].astype(float).values.tolist()
    history_logp = df["log_price"].astype(float).values.tolist()

    preds = []
    current_logp = history_logp[-1]

    for i in range(1, horizon_days + 1):
        next_date = (last_date + pd.Timedelta(days=i))

        # Build a feature row from the last known row + updated rolling windows
        feat_row = last_row.copy()
        feat_row["date"] = next_date
        feat_row["dow"] = next_date.dayofweek
        feat_row["dom"] = next_date.day
        feat_row["month"] = next_date.month

        # Update rolling features based on history arrays
        def roll_mean(arr, w): return float(np.mean(arr[-w:])) if len(arr) >= w else float(np.mean(arr))
        def roll_std(arr, w): return float(np.std(arr[-w:], ddof=1)) if len(arr) >= w else float(np.std(arr, ddof=1))

        for w in (3, 7, 14, 30, 60):
            feat_row[f"ret_mean_{w}"] = roll_mean(history_rets, w)
            feat_row[f"ret_std_{w}"] = roll_std(history_rets, w)
            if len(history_logp) > w:
                feat_row[f"mom_{w}"] = float(current_logp - history_logp[-(w+1)])
            else:
                feat_row[f"mom_{w}"] = float(current_logp - history_logp[0])

        # XGB predicts next-day return
        X_feat = pd.DataFrame([feat_row[feature_cols].to_dict()])
        xgb_ret = float(xgb_pipe.predict(X_feat)[0])

        # SARIMAX predicts next-day return too
        if exog_cols_for_sarimax:
            ex = np.array([feat_row[c] for c in exog_cols_for_sarimax], dtype=float).reshape(1, -1)
            sar_ret = float(sarimax_model.forecast(steps=1, exog=ex)[0])
        else:
            sar_ret = float(sarimax_model.forecast(steps=1)[0])

        ens_ret = 0.5 * xgb_ret + 0.5 * sar_ret

        # Update log price
        current_logp = float(current_logp + ens_ret)
        price = float(np.exp(current_logp))

        preds.append({"date": next_date.strftime("%Y-%m-%d"), "predicted_price": price, "predicted_return": ens_ret})

        # Update history arrays
        history_rets.append(ens_ret)
        history_logp.append(current_logp)

        # carry forward row (so categorical/calendar already updated next step)
        last_row = feat_row

    return preds

def run_forecast(df_feat: pd.DataFrame, coin_id: str, vs_currency: str, horizon_days: int,
                 optuna_trials: int = 60) -> ForecastResult:
    # Pick features (exclude target + obvious non-features)
    drop = {"y", "date", "price", "mcap", "volume"}
    feature_cols = [c for c in df_feat.columns if c not in drop]

    # Train XGB (heavy tuning)
    xgb_pipe, diag_xgb = tune_xgb_heavy(df_feat, feature_cols, trials=optuna_trials, cv_splits=5)

    # SARIMAX exog: use a few strong continuous predictors
    exog_cols = [c for c in ("ret_mean_7", "ret_std_14", "mom_14", "log_vol", "log_mcap", "vol_mcap_ratio", "btc_ret1", "eth_ret1") if c in df_feat.columns]
    sar_model, diag_sar = fit_sarimax(df_feat, exog_cols=exog_cols if exog_cols else None)

    preds = iterative_forecast(df_feat, feature_cols, xgb_pipe, sar_model, exog_cols if exog_cols else None, horizon_days)

    last = df_feat.iloc[-1]
    return ForecastResult(
        coin_id=coin_id,
        vs_currency=vs_currency,
        horizon_days=horizon_days,
        last_date=str(pd.to_datetime(last["date"]).date()),
        last_price=float(last["price"]),
        predictions=preds,
        diagnostics={**diag_xgb, **diag_sar, "rows_used": float(len(df_feat))},
    )

