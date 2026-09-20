"""Direct unit coverage for trading/live_executor.py's real-order-placement
logic - previously untested (only exercised indirectly via FakeLive stand-ins
in tests/test_circuit_breaker.py etc, which never touch this module's actual
adjust_to()/flip logic).

Constructs LiveExecutor via object.__new__ to skip __init__'s real Exchange/
Info wiring (which requires a live private key and hits the network), then
substitutes fake exchange/info objects that mimic just the SDK surface
adjust_to()/get_actual_position()/ensure_leverage() actually call."""
from __future__ import annotations

from trading.live_executor import LiveExecutor


class FakeExchange:
    def __init__(self):
        self.calls = []
        self.leverage_calls = []
        self.open_fail_coins = set()
        self.close_fail_coins = set()

    def update_leverage(self, leverage, coin, is_cross=True):
        self.leverage_calls.append((leverage, coin, is_cross))

    def market_open(self, coin, is_buy, sz):
        self.calls.append(("open", coin, is_buy, sz))
        if coin in self.open_fail_coins:
            raise RuntimeError("simulated open failure")
        return {"response": {"data": {"statuses": [{"filled": {"avgPx": "100.0"}}]}}}

    def market_close(self, coin, sz=None):
        self.calls.append(("close", coin, sz))
        if coin in self.close_fail_coins:
            raise RuntimeError("simulated close failure")
        return {"response": {"data": {"statuses": [{"filled": {"avgPx": "100.0"}}]}}}


class FakeInfo:
    def __init__(self, position=None):
        # position: (side, size) or None (flat)
        self._position = position

    def user_state(self, address):
        if self._position is None:
            return {"assetPositions": []}
        side, size = self._position
        szi = size if side == "long" else -size
        return {"assetPositions": [{"position": {"coin": "SOL", "szi": str(szi)}}]}


class FakeConfig:
    live_max_leverage = 5


def _executor(position=None) -> tuple[LiveExecutor, FakeExchange]:
    live = object.__new__(LiveExecutor)
    live.config = FakeConfig()
    live.address = "0xtest"
    live.exchange = FakeExchange()
    live.info = FakeInfo(position)
    live._leverage_set_for = set()
    return live, live.exchange


def test_open_from_flat():
    live, fake = _executor(position=None)
    results = live.adjust_to("SOL", "long", 1.0, sz_decimals=2)
    assert len(results) == 1
    assert results[0]["action"] == "open"
    assert results[0]["status"] == "filled"
    assert fake.calls == [("open", "SOL", True, 1.0)]


def test_close_to_flat():
    live, fake = _executor(position=("long", 2.0))
    results = live.adjust_to("SOL", None, 0.0, sz_decimals=2)
    assert len(results) == 1
    assert results[0]["action"] == "close"
    assert fake.calls == [("close", "SOL", None)]


def test_increase_same_side():
    live, fake = _executor(position=("long", 1.0))
    results = live.adjust_to("SOL", "long", 2.5, sz_decimals=2)
    assert results[0]["action"] == "increase"
    assert fake.calls == [("open", "SOL", True, 1.5)]


def test_decrease_same_side():
    live, fake = _executor(position=("long", 2.0))
    results = live.adjust_to("SOL", "long", 0.5, sz_decimals=2)
    assert results[0]["action"] == "decrease"
    assert fake.calls == [("close", "SOL", 1.5)]


def test_flip_side_both_orders_succeed():
    live, fake = _executor(position=("long", 1.0))
    results = live.adjust_to("SOL", "short", 1.0, sz_decimals=2)
    assert [r["action"] for r in results] == ["flip_close", "flip_open"]
    assert all(r["status"] == "filled" for r in results)
    assert fake.calls == [("close", "SOL", None), ("open", "SOL", False, 1.0)]


def test_flip_side_open_leg_fails_after_close_leg_succeeds():
    """The scenario engine/orchestrator.py's live-position reconciliation
    fix depends on: a flip's close leg fills but the open leg errors.
    adjust_to must report the open leg as status='error' (not silently
    succeed or raise) so the caller can reconcile against the real
    (now-flat) exchange position instead of trusting the originally-desired
    side."""
    live, fake = _executor(position=("long", 1.0))
    fake.open_fail_coins.add("SOL")
    results = live.adjust_to("SOL", "short", 1.0, sz_decimals=2)
    assert results[0]["action"] == "flip_close"
    assert results[0]["status"] == "filled"
    assert results[1]["action"] == "flip_open"
    assert results[1]["status"] == "error"


def test_flat_to_flat_places_no_orders():
    live, fake = _executor(position=None)
    results = live.adjust_to("SOL", None, 0.0, sz_decimals=2)
    assert results == []
    assert fake.calls == []


def test_zero_size_open_target_places_no_order():
    live, fake = _executor(position=None)
    # target_side is not None while current_side is None -> place_open is
    # called with sz=0, which its size<=0 guard turns into a no-op.
    results = live.adjust_to("SOL", "long", 0.0, sz_decimals=2)
    assert results == []
    assert fake.calls == []


def test_get_actual_position_flat():
    live, _ = _executor(position=None)
    side, size = live.get_actual_position("SOL")
    assert side is None
    assert size == 0.0


def test_get_actual_position_short():
    live, _ = _executor(position=("short", 3.25))
    side, size = live.get_actual_position("SOL")
    assert side == "short"
    assert size == 3.25


def test_ensure_leverage_only_called_once_per_coin():
    live, fake = _executor(position=None)
    live.ensure_leverage("SOL")
    live.ensure_leverage("SOL")
    assert len(fake.leverage_calls) == 1
    assert fake.leverage_calls[0] == (5, "SOL", True)
