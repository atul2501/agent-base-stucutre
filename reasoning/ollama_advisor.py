"""Ollama fallback: consulted only when the rule-based signal is ambiguous.

Uses Ollama's hosted cloud API by default (OLLAMA_HOST=https://ollama.com)
so no local model download / RAM is required - just an API key from
https://ollama.com/settings/keys. Any failure (missing key, rate limit,
network error, bad JSON) falls back to holding; the trading loop must never
crash because an LLM call failed.
"""
from __future__ import annotations

import json
import logging
import threading
import time

# Import config first so OLLAMA_HOST / OLLAMA_API_KEY are in os.environ
# before the ollama client builds its default Client() at import time.
from config import CONFIG

import ollama

from strategy.genome import Genome
from strategy.signals import Features, Signal

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a disciplined swing-trading risk advisor for a crypto perpetuals "
    "agent. You will be given a candidate trade whose rule-based signals are "
    "mixed. Decide whether to take the trade or hold. Reply ONLY as JSON: "
    '{"action": "long"|"short"|"hold", "confidence": 0.0-1.0, "rationale": "<=20 words"}. '
    "Be conservative - prefer hold when signals genuinely conflict."
)

# consult_batch() sends every ambiguous candidate from ONE trading cycle in
# a single request instead of one independent call per agent - see
# consult_batch's docstring for why.
BATCH_SYSTEM_PROMPT = (
    "You are a disciplined swing-trading risk advisor for a crypto perpetuals "
    "agent. You will be given a JSON array of candidate trades (each with its "
    "own \"id\"), all from the same market snapshot but from different trading "
    "agents whose rule-based signals are mixed. Decide whether to take EACH "
    "trade or hold, independently. Reply ONLY as JSON: "
    '{"decisions": [{"id": <int>, "action": "long"|"short"|"hold", '
    '"confidence": 0.0-1.0, "rationale": "<=20 words"}, ...]}, '
    "with exactly one decision per candidate id given. Be conservative - "
    "prefer hold when signals genuinely conflict."
)

# consult_exit_batch() - the advisory, tighten-only exit-council escalation
# (see engine/orchestrator.py::_process_exits and
# strategy/signals.py::council_oppose_position). Only ever asked whether to
# TIGHTEN an existing stop on an already-open position - it can never widen a
# stop or force a close, so a conservative/failed answer is simply "false".
EXIT_BATCH_SYSTEM_PROMPT = (
    "You are a disciplined swing-trading risk advisor for a crypto perpetuals "
    "agent. You will be given a JSON array of OPEN positions (each with its "
    "own \"id\"), all from the same market snapshot, whose own rule-based "
    "council could not clearly agree the market has turned against them. For "
    "EACH position, decide only whether its stop-loss should be pulled "
    "tighter right now (true) or left alone (false) - you cannot close the "
    "position or move the stop the other way. Reply ONLY as JSON: "
    '{"decisions": [{"id": <int>, "tighten": true|false, "rationale": "<=20 words"}, ...]}, '
    "with exactly one decision per id given. Be conservative - prefer false "
    "when signals genuinely conflict."
)

# Explicit timeout - consult() runs synchronously inside the orchestrator's
# main trading cycle (engine/orchestrator.py::_process_entries). Without a
# timeout, a hung/slow cloud endpoint would stall that entire cycle instead
# of failing fast into the "hold" fallback like every other failure mode
# here already does.
_client = ollama.Client(host=CONFIG.ollama_host, timeout=CONFIG.ollama_timeout_seconds)

_status_lock = threading.Lock()
_status = {"state": "unknown", "detail": "", "checked_at": None}


def _set_status(state: str, detail: str = "") -> None:
    with _status_lock:
        _status["state"] = state
        _status["detail"] = detail
        _status["checked_at"] = time.time()


def get_status() -> dict:
    with _status_lock:
        return dict(_status)


def check_health_now() -> None:
    """Synchronous health probe - cheap enough to call at startup and periodically."""
    if not CONFIG.ollama_enabled:
        _set_status("disabled")
        return
    if CONFIG.is_using_cloud_ollama() and not CONFIG.ollama_api_key:
        _set_status("not_configured", "OLLAMA_API_KEY is empty - get one at https://ollama.com/settings/keys")
        return
    try:
        _client.chat(
            model=CONFIG.ollama_model,
            messages=[{"role": "user", "content": "reply with the single word: ok"}],
            options={"num_predict": 5},
        )
        _set_status("ready", f"model={CONFIG.ollama_model} host={CONFIG.ollama_host}")
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "unauthorized" in msg or "authoriz" in msg:
            _set_status("invalid_api_key", str(e)[:200])
        elif "429" in msg or "rate" in msg or "quota" in msg or "credit" in msg:
            _set_status("rate_limited", str(e)[:200])
        elif "404" in msg or "not found" in msg:
            _set_status("model_unavailable", f"'{CONFIG.ollama_model}' not found on {CONFIG.ollama_host}: {e}"[:200])
        else:
            _set_status("unreachable", str(e)[:200])


def check_health_async() -> None:
    threading.Thread(target=check_health_now, daemon=True).start()


def consult(genome: Genome, features: Features, candidate: Signal) -> Signal:
    status = get_status()
    if status["state"] not in ("ready", "unknown"):
        # Already known to be broken (no key, rate limited, unreachable) - don't
        # waste a request/latency, just fall back immediately.
        return Signal("hold", 0.0, candidate.score, candidate.reasons + [f"ollama {status['state']}, held"], ambiguous=False)

    payload = {
        "coin": genome.coin,
        "candidate_action": candidate.action,
        "rule_based_reasons": candidate.reasons,
        "trend_up": features.trend_up,
        "rsi": round(features.rsi_value, 1),
        "order_book_bid_ask_ratio": round(features.ob_imbalance, 2),
        "open_interest_change_pct": features.oi_change_pct,
        "funding_rate": features.funding,
    }

    try:
        response = _client.chat(
            model=CONFIG.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            format="json",
            options={"temperature": 0.2},
        )
        content = response["message"]["content"]
        parsed = json.loads(content)
        action = str(parsed.get("action", "hold")).lower()
        if action not in ("long", "short", "hold"):
            action = "hold"
        confidence = float(parsed.get("confidence", 0.3))
        rationale = str(parsed.get("rationale", ""))[:200]
        _set_status("ready", f"model={CONFIG.ollama_model} host={CONFIG.ollama_host}")
        return Signal(
            action=action,
            confidence=max(0.0, min(1.0, confidence)),
            score=candidate.score,
            reasons=candidate.reasons + [f"ollama: {rationale}"],
            ambiguous=False,
        )
    except Exception as e:
        log.warning("Ollama advisor call failed (%s) - defaulting to hold", e)
        check_health_async()  # re-probe so status reflects the new failure
        return Signal("hold", 0.0, candidate.score, candidate.reasons + ["ollama unavailable, held"], ambiguous=False)


def consult_batch(items: list[tuple[Genome, Features, Signal]]) -> list[Signal]:
    """Batched form of consult(): resolves every ambiguous signal from ONE
    trading cycle in a SINGLE Ollama request, instead of one independent
    call per agent. Without this, a cycle where many agents are ambiguous
    at once (all against the same market snapshot) issued one sequential
    blocking HTTP round-trip per agent - each up to ollama_timeout_seconds -
    which could stall that cycle badly and burn through rate limits fast
    at scale. Order-preserving: returns exactly one Signal per input item,
    in the same order. Falls back to "hold" for the whole batch (or just
    one malformed entry within it - a bad response for one candidate
    doesn't have to cost every other candidate in the same batch its real
    decision) on any failure, same fail-safe philosophy as consult()."""
    if not items:
        return []

    status = get_status()
    if status["state"] not in ("ready", "unknown"):
        return [Signal("hold", 0.0, candidate.score, candidate.reasons + [f"ollama {status['state']}, held"], ambiguous=False)
                for _genome, _features, candidate in items]

    payload = [
        {
            "id": i,
            "coin": genome.coin,
            "candidate_action": candidate.action,
            "rule_based_reasons": candidate.reasons,
            "trend_up": features.trend_up,
            "rsi": round(features.rsi_value, 1),
            "order_book_bid_ask_ratio": round(features.ob_imbalance, 2),
            "open_interest_change_pct": features.oi_change_pct,
            "funding_rate": features.funding,
        }
        for i, (genome, features, candidate) in enumerate(items)
    ]

    try:
        response = _client.chat(
            model=CONFIG.ollama_model,
            messages=[
                {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            format="json",
            options={"temperature": 0.2},
        )
        content = response["message"]["content"]
        parsed = json.loads(content)
        decisions = parsed.get("decisions", []) if isinstance(parsed, dict) else parsed
        by_id = {d["id"]: d for d in decisions if isinstance(d, dict) and "id" in d} if isinstance(decisions, list) else {}
        _set_status("ready", f"model={CONFIG.ollama_model} host={CONFIG.ollama_host}")

        results = []
        for i, (_genome, _features, candidate) in enumerate(items):
            decision = by_id.get(i)
            try:
                if decision is None:
                    raise ValueError("no decision returned for this id")
                action = str(decision.get("action", "hold")).lower()
                if action not in ("long", "short", "hold"):
                    action = "hold"
                confidence = max(0.0, min(1.0, float(decision.get("confidence", 0.3))))
                rationale = str(decision.get("rationale", ""))[:200]
                results.append(Signal(action=action, confidence=confidence, score=candidate.score,
                                       reasons=candidate.reasons + [f"ollama: {rationale}"], ambiguous=False))
            except Exception:
                # Only THIS candidate degrades to hold - one malformed
                # entry in the batch doesn't cost every other candidate
                # its real decision.
                results.append(Signal("hold", 0.0, candidate.score,
                                       candidate.reasons + ["ollama batch: malformed decision, held"], ambiguous=False))
        return results
    except Exception as e:
        log.warning("Ollama batch advisor call failed (%s) - defaulting all %d candidate(s) to hold", e, len(items))
        check_health_async()
        return [Signal("hold", 0.0, candidate.score, candidate.reasons + ["ollama unavailable, held"], ambiguous=False)
                for _genome, _features, candidate in items]


def consult_exit_batch(items: list[tuple[Genome, Features, str]]) -> list[bool]:
    """Batched tighten-only advisory for OPEN positions whose own genome
    council (council_oppose_position) couldn't reach quorum either way this
    cycle. Each item is (genome, features, side) for one open position.
    Order-preserving: returns exactly one bool per input item, in the same
    order - True means "pull this position's stop tighter", False is the
    safe no-op default. Same fail-safe philosophy as consult_batch(): any
    failure (disabled/unreachable/malformed), or a malformed entry within an
    otherwise-successful batch, degrades to False rather than raising, since
    this function can never widen a stop or close a position - the worst a
    wrong answer here does is leave the deterministic stop exactly where it
    already was."""
    if not items:
        return []

    status = get_status()
    if status["state"] not in ("ready", "unknown"):
        return [False] * len(items)

    payload = [
        {
            "id": i,
            "coin": genome.coin,
            "side": side,
            "trend_up": features.trend_up,
            "rsi": round(features.rsi_value, 1),
            "order_book_bid_ask_ratio": round(features.ob_imbalance, 2),
            "open_interest_change_pct": features.oi_change_pct,
            "funding_rate": features.funding,
        }
        for i, (genome, features, side) in enumerate(items)
    ]

    try:
        response = _client.chat(
            model=CONFIG.ollama_model,
            messages=[
                {"role": "system", "content": EXIT_BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)},
            ],
            format="json",
            options={"temperature": 0.2},
        )
        content = response["message"]["content"]
        parsed = json.loads(content)
        decisions = parsed.get("decisions", []) if isinstance(parsed, dict) else parsed
        by_id = {d["id"]: d for d in decisions if isinstance(d, dict) and "id" in d} if isinstance(decisions, list) else {}
        _set_status("ready", f"model={CONFIG.ollama_model} host={CONFIG.ollama_host}")

        results = []
        for i in range(len(items)):
            decision = by_id.get(i)
            try:
                if decision is None:
                    raise ValueError("no decision returned for this id")
                results.append(bool(decision.get("tighten", False)))
            except Exception:
                # Only THIS candidate degrades to no-op - one malformed
                # entry in the batch doesn't cost every other candidate.
                results.append(False)
        return results
    except Exception as e:
        log.warning("Ollama exit-batch advisor call failed (%s) - defaulting all %d candidate(s) to no-op", e, len(items))
        check_health_async()
        return [False] * len(items)
