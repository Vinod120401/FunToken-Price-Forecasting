# FUNToken Internal Price Prediction Service

**Audience:** FUNToken engineering & data team
**Purpose:** Internal forecasting, experimentation, dashboards, and scenario analysis
**Status:** Internal-only (NOT financial advice, NOT user-facing)

---

## Overview

This service provides a **compute-heavy price forecasting API** for **FUNToken** using historical market data from CoinGecko.
It is optimized for **accuracy over latency**, using:

- CoinGecko historical price, volume, and market cap data
- Optional BTC & ETH exogenous signals
- Extensive feature engineering (returns, momentum, volatility, ratios)
- **Optuna-tuned XGBoost** (walk-forward CV)
- **SARIMAX** on returns
- **Ensemble averaging** (XGB + SARIMAX)
- Iterative multi-day forecasting

This is intended for **internal research, analytics, and tooling** only.

---

## Key Endpoints

### Health Check
```
GET /health
```

### Live Spot Price (Reference Only)
```
GET /price/live?vs_currency=usd
```

### Forecast
```
GET /predict?days=7&vs_currency=usd
```

**Query Parameters**
| Param | Description | Notes |
|-----|------------|------|
| `days` | Days forward to forecast | 1–365 |
| `vs_currency` | Quote currency | default: `usd` |
| `optuna_trials` | Override tuning iterations | higher = slower & potentially better |

---

## Environment Variables

```bash
# Core
COIN_ID=funtoken
VS_CURRENCY=usd

# CoinGecko
COINGECKO_API_KEY=your_key_here
COINGECKO_KEY_HEADER=x-cg-demo-api-key   # or x-cg-pro-api-key

# Modeling / Accuracy Controls
OPTUNA_TRIALS=80        # Increase for better tuning (slow)
HISTORY_DAYS=max        # Use full history
INCLUDE_BTC_ETH=1       # Use BTC/ETH as exogenous regressors
```

**Accuracy vs Compute Guidance**
| Setting | Effect |
|------|-------|
| OPTUNA_TRIALS ↑ | Better fit, exponential runtime |
| INCLUDE_BTC_ETH=1 | Better regime awareness |
| HISTORY_DAYS=max | More stable long-term trends |

---

## Architecture Summary

```
CoinGecko
   ↓
Daily Market Frame
   ↓
Feature Engineering
   ↓
┌──────────────────────────────┐
│  XGBoost (Optuna + CV)        │
│  SARIMAX (returns + exog)     │
└──────────────┬───────────────┘
               ↓
         Ensemble Forecast
               ↓
        Iterative Price Path
```

---

## What the Model Predicts (Important)

- The model **predicts next-day log returns**, not absolute prices
- Prices are reconstructed iteratively
- Error compounds with time → short horizons are more reliable

**Rule of Thumb**
| Horizon | Reliability |
|------|-------------|
| 1–3 days | High (relative) |
| 7–14 days | Medium |
| 30+ days | Low (trend-biased) |

---

## Internal Usage Recommendations

### ✅ Good Uses
- Internal dashboards
- Scenario comparison
- Volatility & regime analysis
- Research & experimentation
- Tokenomics modeling inputs

### ❌ Not Recommended
- Trading automation without additional signals
- User-facing predictions
- Marketing claims
- Legal/financial disclosures

---

## Performance Expectations

| Setting | Typical Runtime |
|------|----------------|
| 40 trials | ~30–60 sec |
| 80 trials | ~2–4 min |
| 200+ trials | 5–15+ min |

**Recommendation:**
Run behind a **job queue** or **cron-based cache** (hourly/daily), not synchronous user requests.

---

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8080
```

---

## Internal Disclaimer

This system provides **statistical forecasts only**, derived from historical data.
It does **not account for**:
- News
- Exchange outages
- Regulatory events
- Market manipulation
- Liquidity shocks

