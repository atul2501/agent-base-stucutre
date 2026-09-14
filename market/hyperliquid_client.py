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
    funding: float
    open_interest: float
    candles: list[dict]          # oldest -> newest, keys: t,o,h,l,c,v
    bid_levels: list[dict]       # [{px, sz}, ...] best first
    ask_levels: list[dict]


class HyperliquidClient:
    def __init__(self, network: str = "mainnet"):
        base_url = constants.TESTNET_API_URL if network == "testnet" else constants.MAINNET_API_URL
        self.info = Info(base_url, skip_ws=True)

    def get_snapshot(self, coin: str, timeframe: str, candle_lookback_hours: int = 72) -> MarketSnapshot:
        meta, ctxs = self.info.meta_and_asset_ctxs()
        universe = meta["universe"]
        idx = next((i for i, a in enumerate(universe) if a["name"] == coin), None)
        if idx is None:
            raise ValueError(f"Unknown coin on Hyperliquid: {coin}")
        ctx = ctxs[idx]

        end = int(time.time() * 1000)
        start = end - candle_lookback_hours * 60 * 60 * 1000
        raw_candles = self.info.candles_snapshot(coin, timeframe, start, end)
        candles = [
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

        book = self.info.l2_snapshot(coin)
        bid_levels = [{"px": float(l["px"]), "sz": float(l["sz"])} for l in book["levels"][0]]
        ask_levels = [{"px": float(l["px"]), "sz": float(l["sz"])} for l in book["levels"][1]]

        return MarketSnapshot(
            coin=coin,
            mid_price=float(ctx["midPx"]) if ctx.get("midPx") else float(ctx["markPx"]),
            mark_price=float(ctx["markPx"]),
            funding=float(ctx["funding"]),
            open_interest=float(ctx["openInterest"]),
            candles=candles,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
        )
