import os
import httpx
from typing import Any, Dict, Optional

COINGECKO_BASE_URL = os.getenv("COINGECKO_BASE_URL", "https://api.coingecko.com/api/v3")

# CoinGecko uses different headers depending on plan (demo/pro).
# - Docs show `x-cg-demo-api-key` for public/demo endpoints in v3.0.1 docs
# - Pro uses `x-cg-pro-api-key`
DEFAULT_KEY_HEADER = os.getenv("COINGECKO_KEY_HEADER", "x-cg-demo-api-key")
API_KEY = os.getenv("COINGECKO_API_KEY", "").strip()

class CoinGeckoClient:
    def __init__(self, timeout_s: float = 30.0):
        self._timeout = timeout_s

    def _headers(self) -> Dict[str, str]:
        if not API_KEY:
            return {}
        return {DEFAULT_KEY_HEADER: API_KEY}

    async def market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: str = "max",
        interval: Optional[str] = "daily",
        precision: Optional[str] = "full",
    ) -> Dict[str, Any]:
        """
        GET /coins/{id}/market_chart
        Returns: { prices: [[ts, price], ...], market_caps: [...], total_volumes: [...] }
        """
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
        params: Dict[str, Any] = {"vs_currency": vs_currency, "days": days}
        # interval/precision are optional parameters commonly supported (docs examples include interval daily, precision full)
        if interval:
            params["interval"] = interval
        if precision:
            params["precision"] = precision

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(url, params=params, headers=self._headers())
            r.raise_for_status()
            return r.json()

    async def simple_price(self, ids: str, vs_currencies: str = "usd") -> Dict[str, Any]:
        """
        GET /simple/price
        """
        url = f"{COINGECKO_BASE_URL}/simple/price"
        params = {"ids": ids, "vs_currencies": vs_currencies}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.get(url, params=params, headers=self._headers())
            r.raise_for_status()
            return r.json()

