"""Read-only Hyperliquid market data access (candles, order book, open interest, funding).

No authentication is required for these endpoints, so this client works the
same whether the eventual trading mode is paper or live.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from hyperliquid.info import Info
from hyperliquid.utils import constants


@dataclass
class MarketSnapshot:
    coin: str
    mid_price: float
    mark_price: float
    oracle_price: float
    premium: float                # (mark - oracle) / oracle, from Hyperliquid directly
    funding: float
    open_interest: float
    prev_day_price: float
    day_notional_volume: float
    candles: list[dict]          # oldest -> newest, keys: t,o,h,l,c,v
    bid_levels: list[dict]       # [{px, sz}, ...] best first
    ask_levels: list[dict]
    sz_decimals: int             # exchange's required size precision for this coin


class HyperliquidClient:
    def __init__(self, network: str = "mainnet"):
        base_url = constants.TESTNET_API_URL if network == "testnet" else constants.MAINNET_API_URL
        self.info = Info(base_url, skip_ws=True)

    def get_candles(self, coin: str, timeframe: str, lookback_hours: int,
                     end_time_ms: int | None = None) -> list[dict]:
        """Raw historical candles, oldest -> newest. Factored out of
        get_snapshot so callers that need an OLDER window (e.g. multi-regime
        backtest screening - see backtest/engine.py, main.py) can pass an
        explicit `end_time_ms` instead of always ending "now"."""
        end = end_time_ms if end_time_ms is not None else int(time.time() * 1000)
        start = end - lookback_hours * 60 * 60 * 1000
        raw_candles = self.info.candles_snapshot(coin, timeframe, start, end)
        return [
            {
                "t": c["t"],
                "o": float(c["o"]),
                "h": float(c["h"]),
                "l": float(c["l"]),
                "c": float(c["c"]),
                "v": float(c["v"]),
            }
            for c in raw_candles
        ]

    def get_snapshot(self, coin: str, timeframe: str, candle_lookback_hours: int = 72) -> MarketSnapshot:
        meta, ctxs = self.info.meta_and_asset_ctxs()
        universe = meta["universe"]
        idx = next((i for i, a in enumerate(universe) if a["name"] == coin), None)
        if idx is None:
            raise ValueError(f"Unknown coin on Hyperliquid: {coin}")
        ctx = ctxs[idx]

        candles = self.get_candles(coin, timeframe, candle_lookback_hours)

        book = self.info.l2_snapshot(coin)
        bid_levels = [{"px": float(lvl["px"]), "sz": float(lvl["sz"])} for lvl in book["levels"][0]]
        ask_levels = [{"px": float(lvl["px"]), "sz": float(lvl["sz"])} for lvl in book["levels"][1]]

        return MarketSnapshot(
            coin=coin,
            mid_price=float(ctx["midPx"]) if ctx.get("midPx") else float(ctx["markPx"]),
            mark_price=float(ctx["markPx"]),
            oracle_price=float(ctx["oraclePx"]),
            premium=float(ctx["premium"]) if ctx.get("premium") is not None else 0.0,
            funding=float(ctx["funding"]),
            open_interest=float(ctx["openInterest"]),
            prev_day_price=float(ctx["prevDayPx"]),
            day_notional_volume=float(ctx["dayNtlVlm"]),
            candles=candles,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            sz_decimals=int(universe[idx]["szDecimals"]),
        )

    def is_valid_coin(self, coin: str) -> bool:
        meta, _ = self.info.meta_and_asset_ctxs()
        return any(a["name"] == coin for a in meta["universe"])

    def get_funding_history(self, coin: str, lookback_hours: int) -> list[tuple[int, float, float]]:
        """(timestamp_ms, funding, premium) tuples, oldest -> newest. Used
        by backtest/engine.py - this is real historical data (unlike order
        book / open interest, which are snapshot-only)."""
        end = int(time.time() * 1000)
        start = end - lookback_hours * 60 * 60 * 1000
        raw = self.info.funding_history(coin, start, end)
        return [(int(f["time"]), float(f["fundingRate"]), float(f["premium"])) for f in raw]
