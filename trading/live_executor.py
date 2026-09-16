"""Real order placement on Hyperliquid. Only ever constructed when
Config.is_live() is True - see config.py. Runs against whichever HL_NETWORK
is configured: testnet (real testnet orders, fake funds - a safe way to
test this code path) or mainnet (real money - additionally requires
LIVE_TRADING_CONFIRMED).

Important honesty note: a Hyperliquid account holds ONE net position per
(account, coin). There is no way to give N agents N independent real
positions in the same token on the same wallet - the exchange nets them.
So "top N agents trade live" is implemented as: the real position tracks the
net long/short consensus of the current live-eligible agents' paper
positions, sized in fixed `per_slot_notional` increments and hard-capped at
`live_max_total_notional_usd` in total. See engine/orchestrator.py
`_sync_live_exposure` and README.md.
"""
from __future__ import annotations

import logging

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from config import Config

log = logging.getLogger(__name__)


class LiveExecutor:
    def __init__(self, config: Config):
        if not config.hl_private_key:
            raise ValueError("HYPERLIQUID_PRIVATE_KEY is required for live trading")
        self.config = config
        base_url = constants.TESTNET_API_URL if config.hl_network == "testnet" else constants.MAINNET_API_URL
        wallet = Account.from_key(config.hl_private_key)
        self.address = config.hl_wallet_address or wallet.address
        self.exchange = Exchange(wallet, base_url, account_address=self.address)
        self.info = Info(base_url, skip_ws=True)
        self._leverage_set_for: set[str] = set()

    def ensure_leverage(self, coin: str) -> None:
        if coin in self._leverage_set_for:
            return
        self.exchange.update_leverage(self.config.live_max_leverage, coin, is_cross=True)
        self._leverage_set_for.add(coin)
        log.info("Live leverage set to %dx cross for %s", self.config.live_max_leverage, coin)

    def get_account_equity(self) -> float | None:
        """Ground-truth total account value from the exchange (margin
        summary), used by the live circuit breaker to measure drawdown
        against - see engine/orchestrator.py. Returns None if the API call
        fails, so a transient network hiccup can't be mistaken for a real
        drawdown."""
        try:
            state = self.info.user_state(self.address)
            return float(state["marginSummary"]["accountValue"])
        except Exception as e:
            log.warning("Failed to fetch live account equity: %s", e)
            return None

    def get_actual_position(self, coin: str) -> tuple[str | None, float]:
        """Ground truth from the exchange itself - always reconcile against
        this rather than trusting our own bookkeeping, in case of manual
        trades on the same wallet or a missed update."""
        state = self.info.user_state(self.address)
        for ap in state.get("assetPositions", []):
            pos = ap["position"]
            if pos["coin"] == coin:
                szi = float(pos["szi"])
                if abs(szi) < 1e-9:
                    return None, 0.0
                return ("long" if szi > 0 else "short"), abs(szi)
        return None, 0.0

    def _round_size(self, size: float, sz_decimals: int) -> float:
        return round(size, sz_decimals)

    def adjust_to(self, coin: str, target_side: str | None, target_size: float, sz_decimals: int) -> list[dict]:
        """Move the real position toward (target_side, target_size). Returns a
        list of {action, side, notional, fill_price, status, detail} records,
        one per real order placed (a side flip takes two)."""
        self.ensure_leverage(coin)
        current_side, current_size = self.get_actual_position(coin)
        target_size = self._round_size(max(0.0, target_size), sz_decimals)
        results: list[dict] = []

        def place_open(is_buy: bool, sz: float, action: str, side: str) -> None:
            sz = self._round_size(sz, sz_decimals)
            if sz <= 0:
                return
            try:
                resp = self.exchange.market_open(coin, is_buy, sz)
                fill_px = _extract_fill_price(resp)
                results.append({"action": action, "side": side, "notional": None,
                                 "fill_price": fill_px, "status": "filled", "detail": str(resp)[:300]})
            except Exception as e:
                log.exception("Live market_open failed for %s", coin)
                results.append({"action": action, "side": side, "notional": None,
                                 "fill_price": None, "status": "error", "detail": str(e)[:300]})

        def place_close(sz: float | None, action: str, side: str) -> None:
            sz = self._round_size(sz, sz_decimals) if sz is not None else None
            if sz is not None and sz <= 0:
                return
            try:
                resp = self.exchange.market_close(coin, sz=sz)
                fill_px = _extract_fill_price(resp)
                results.append({"action": action, "side": side, "notional": None,
                                 "fill_price": fill_px, "status": "filled", "detail": str(resp)[:300]})
            except Exception as e:
                log.exception("Live market_close failed for %s", coin)
                results.append({"action": action, "side": side, "notional": None,
                                 "fill_price": None, "status": "error", "detail": str(e)[:300]})

        if current_side is None and target_side is None:
            return results  # flat and staying flat

        if current_side is None:
            place_open(target_side == "long", target_size, "open", target_side)
        elif target_side is None:
            place_close(None, "close", current_side)  # full close
        elif current_side == target_side:
            if target_size > current_size:
                place_open(target_side == "long", target_size - current_size, "increase", target_side)
            elif target_size < current_size:
                place_close(current_size - target_size, "decrease", current_side)
        else:
            place_close(None, "flip_close", current_side)
            place_open(target_side == "long", target_size, "flip_open", target_side)

        return results


def _extract_fill_price(resp: dict) -> float | None:
    try:
        statuses = resp["response"]["data"]["statuses"]
        for s in statuses:
            if "filled" in s:
                return float(s["filled"]["avgPx"])
    except Exception:
        pass
    return None
