import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from coingecko_client import CoinGeckoClient
from features import build_daily_frame, add_features
from forecast import run_forecast, ForecastResult

app = FastAPI(title="FUNToken Price Prediction API", version="1.0.0")

COIN_ID = os.getenv("COIN_ID", "funtoken")  # FUNToken coin id on CoinGecko
DEFAULT_VS = os.getenv("VS_CURRENCY", "usd")

# Compute knobs (accuracy vs latency)
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "60"))     # increase for more accuracy (slower)
HISTORY_DAYS = os.getenv("HISTORY_DAYS", "max")           # "max" or number
INCLUDE_BTC_ETH = os.getenv("INCLUDE_BTC_ETH", "1") == "1"

cg = CoinGeckoClient()

class PredictResponse(BaseModel):
    coin_id: str
    vs_currency: str
    horizon_days: int
    last_date: str
    last_price: float
    predictions: list[dict]
    diagnostics: Dict[str, float]
    disclaimer: str

@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/price/live")
async def live_price(vs_currency: str = Query(DEFAULT_VS)):
    try:
        data = await cg.simple_price(ids=COIN_ID, vs_currencies=vs_currency)
        return {"coin_id": COIN_ID, "vs_currency": vs_currency, "price": data.get(COIN_ID, {}).get(vs_currency)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.get("/predict", response_model=PredictResponse)
async def predict(
    days: int = Query(7, ge=1, le=365, description="How many days forward to predict"),
    vs_currency: str = Query(DEFAULT_VS),
    optuna_trials: Optional[int] = Query(None, ge=10, le=500, description="Override Optuna trials (more = slower, potentially better)"),
):
    """
    Predict FUNToken price path days forward.
    Very compute-heavy by design (Optuna tuning + CV).
    """
    trials = optuna_trials or OPTUNA_TRIALS

    try:
        fun_json = await cg.market_chart(COIN_ID, vs_currency=vs_currency, days=HISTORY_DAYS, interval="daily", precision="full")
        fun_df = build_daily_frame(fun_json)

        btc_df = eth_df = None
        if INCLUDE_BTC_ETH:
            btc_json = await cg.market_chart("bitcoin", vs_currency=vs_currency, days=HISTORY_DAYS, interval="daily", precision="full")
            eth_json = await cg.market_chart("ethereum", vs_currency=vs_currency, days=HISTORY_DAYS, interval="daily", precision="full")
            btc_df = build_daily_frame(btc_json)
            eth_df = build_daily_frame(eth_json)

        df_feat = add_features(fun_df, btc_df=btc_df, eth_df=eth_df)

        # Safety: need enough history
        if len(df_feat) < 200:
            raise HTTPException(status_code=400, detail=f"Not enough usable history after feature engineering: {len(df_feat)} rows")

        result: ForecastResult = run_forecast(df_feat, coin_id=COIN_ID, vs_currency=vs_currency, horizon_days=days, optuna_trials=trials)

        return PredictResponse(
            **result.__dict__,
            disclaimer="Forecasts are statistical estimates from historical data, not financial advice, and can be very wrong—especially during regime changes/news-driven moves."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

