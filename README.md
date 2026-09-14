# agent-base-stucutre

An evolving population of swing-trading agents on Hyperliquid, trading
**exactly one token at a time**. Trades are decided from real Hyperliquid
market data. A local dashboard shows population health and profitability
live in your browser while it runs.

## The do-or-die lifecycle

- **Win a trade -> survive + multiply.** The agent keeps its balance (plus
  the profit) and spawns `CHILDREN_PER_WIN` (default 2) mutated copies of
  itself into the population.
- **Lose a trade -> die.** The agent is removed immediately. No second
  chances.
- **Both children winning -> promoted.** If an agent (`A`) spawns `A1` and
  `A2` and *both* go on to win at least one trade of their own, `A`, `A1`
  and `A2` are promoted to the `professional` tier - a proven, self
  sustaining lineage.
- **8-win streak -> shared.** An agent riding an 8-win streak
  (`WIN_STREAK_SHARE_THRESHOLD`) has its genome recorded in
  `strategy_shares`. When the population needs fresh agents, half the time
  a new one is cloned + mutated from a shared genome instead of pure
  random, so proven strategies propagate.
- **Population cap.** At most `POPULATION_CAP` (default 500) agents are
  alive at once; the lowest-fitness agents with at least one completed
  trade are culled first when wins push the count over the cap (newborns
  get a grace period).
- **Active traders.** Only the top `ACTIVE_TRADER_COUNT` (default 50) alive
  agents by fitness are allowed to open new (paper) positions each cycle.
- **Population floor.** If deaths shrink the population below
  `MIN_POPULATION_FLOOR`, fresh agents are spawned to keep the system
  self-sustaining.

## One token at a time, on purpose

Every agent in a given database trades the same `TOKEN` (default `SOL`).
Mixing tokens in one population would blur what each agent actually
learned, so switching assets is a deliberate, explicit action:

```bash
# change TOKEN in .env from SOL to, say, BNB, then:
python3 main.py --reset
```

`--reset` wipes every agent, trade, and shared strategy - "step 0" - and
starts learning the new token from scratch. Without `--reset`, starting
with a different `TOKEN` than the database was built for is refused with a
clear error rather than silently mixing strategies.

## The dashboard

`main.py` starts a local web dashboard automatically at
`http://127.0.0.1:8000` (configurable via `DASHBOARD_HOST`/`DASHBOARD_PORT`,
or disable with `--no-dashboard`). It shows, refreshing every 5s:

- Network (testnet/mainnet) and mode (paper/live) badges, plus Ollama's
  live health status.
- Population stats: alive agents, active traders, professional-tier count,
  total realized PnL, overall win rate, max generation reached.
- A chart of the best agent's total PnL by cycle.
- A leaderboard of the top 30 agents (tier, generation, wins/losses, streak,
  balance, total PnL).
- Recent trades, and - once live trading is armed - the real position and a
  log of every real order placed.

## How an agent decides to trade

Every signal dimension Hyperliquid's public data supports is in play, and
every agent has its own thresholds for all of it (32 genome parameters
total) - that diversity is what lets the population race toward what
actually works rather than everyone converging on one hand-picked setup.

1. Pull the token's latest candles, L2 order book, open interest, funding
   rate, mark/oracle premium, and 24h stats from Hyperliquid
   (`market/hyperliquid_client.py`).
2. Two **hard pre-filters** gate everything before anything else is even
   considered: bid/ask spread (skip if too illiquid) and ATR volatility
   regime (skip if the market's too dead or too chaotic for this agent's
   comfort zone).
3. The **trigger**: an EMA trend read + an RSI pullback/rally condition
   (buy dips in an uptrend, sell rallies in a downtrend).
4. The trigger is then scored against every other available signal as a
   confirmation or contradiction: order book imbalance, open-interest
   change, funding-rate crowding, mark/oracle premium, volume conviction
   (current candle vs its own recent average), VWAP deviation, MACD
   momentum, Bollinger %B (price's position within its bands), Stochastic
   RSI (a faster, more sensitive oversold/overbought read), and 24h macro
   momentum (`strategy/indicators.py`, `strategy/signals.py`).
5. A clearly confirmed setup fires the trade; a clearly contradicted one
   holds. A genuinely mixed read is sent to Ollama (`reasoning/ollama_advisor.py`)
   for a judgment call instead of guessing - the "self-understanding"
   fallback. Any failure (no API key, rate limited, unreachable) falls back
   to holding rather than crashing.
6. Exits are rule-based: stop loss / take profit / max hold time (all
   genome parameters).

Existing agents from before this indicator set existed load fine -
`Genome.from_dict` fills in any fields an older genome is missing with
that field's bounds midpoint, so a trained population never needs a
`--reset` just because the strategy space grew.

## Backtest pre-screening (why new agents aren't purely random)

Measured directly against real SOL data: a genuinely selective RSI+trend
trigger fires anywhere from 0 to ~7 times over a 72-hour window depending
on threshold, so waiting on live trades alone to tell a good genome from a
bad one is slow. Instead, every time a new agent is about to be born
(initial seeding, floor refill, or a winner's children after a live win),
`agents/population.py` generates several candidate genomes (`BACKTEST_CANDIDATES`,
default 5) and replays each against real historical data
(`backtest/engine.py`) before committing to the best-scoring one. Measured
result: agents chosen this way score ~6x better on average than a blind
random draw would.

This **only changes what genome an agent is born with** - the live
do-or-die mechanic is completely untouched. A backtested-promising agent
still has to win its first real trade to survive, exactly as before.

Honesty note: Hyperliquid's public API has no historical series for order
book depth or open interest (point-in-time snapshots only), so those two
confirmations are neutral during backtesting. Funding and mark/oracle
premium DO have real historical series (`Info.funding_history` includes
both) and are used for real. Hyperliquid also caps a single candle request
at ~5000 bars (~17 days of 5m data) - `BACKTEST_LOOKBACK_HOURS` defaults to
360 (15 days) to stay safely under that. Disable entirely with
`BACKTEST_ENABLED=false` if you'd rather agents stay purely random/mutated.

### Ollama runs in the cloud by default - no local model needed

Since running a local model needs RAM/disk you may not have to spare,
`OLLAMA_HOST` defaults to `https://ollama.com` (Ollama's hosted API) with
`OLLAMA_MODEL=gpt-oss:20b`. Get a free key at
https://ollama.com/settings/keys and put it in `.env` as `OLLAMA_API_KEY`.
The dashboard's Ollama badge reflects the real state: `not_configured` (no
key yet), `ready`, `rate_limited`, `invalid_api_key`, or `unreachable` -
trading keeps working in every state, just without the LLM tie-breaker
when it isn't `ready`. To use a local model instead, set
`OLLAMA_HOST=http://localhost:11434` and `ollama pull <model>` first.

## Trading mode + network

`TRADING_MODE` is the one switch that decides paper vs. real orders.
`HL_NETWORK` is a separate choice of which Hyperliquid environment to use:

- `TRADING_MODE=paper` (default) -> always simulated fills, never touches
  the real exchange, no matter what `HL_NETWORK` is.
- `TRADING_MODE=live` -> places real orders on whichever `HL_NETWORK` you're
  pointed at:
  - `HL_NETWORK=testnet` -> real orders with **fake testnet funds** - a safe
    way to test the live order-placement code itself before trusting it
    with money.
  - `HL_NETWORK=mainnet` -> **real money**. Additionally requires
    `LIVE_TRADING_CONFIRMED` in `.env` to be set to the exact phrase shown
    in `.env.example` - this extra gate only applies to mainnet, since
    testnet has nothing real to lose.

When live:

- Hard caps enforced in code, not just convention: `LIVE_MAX_TOTAL_NOTIONAL_USD`
  (total real dollars ever deployed at once - set this deliberately, it
  defaults low), `LIVE_ACTIVE_TRADER_COUNT` (how many top agents count
  toward the real position), and `LIVE_MAX_LEVERAGE`.
- **Important honesty note:** a Hyperliquid account holds *one net position
  per (account, coin)*. There is no way to give N agents N independent real
  positions in the same single token on one wallet - the exchange nets
  them. So "top 20 agents trade live" is implemented as: the real position
  tracks the **net long/short consensus** of the current top
  `LIVE_ACTIVE_TRADER_COUNT` agents' paper positions, in fixed-size slots
  (`LIVE_MAX_TOTAL_NOTIONAL_USD / LIVE_ACTIVE_TRADER_COUNT` each), capped in
  total. It is reconciled against the actual exchange position
  (`user_state`) before every adjustment rather than trusted from local
  bookkeeping. See `trading/live_executor.py` and
  `engine/orchestrator.py:_sync_live_exposure`.
- The paper simulation always keeps driving the evolutionary lifecycle
  (win/die/spawn) - live orders are a real-money mirror of that, not a
  second independent decision loop, so evolution never depends on exchange
  fills or latency.
- Every real order attempt (filled or failed) is logged to `live_orders`
  and shown on the dashboard.

## Storage

SQLite (`trading_agents.db` by default, WAL mode so the dashboard can read
while the loop writes) tracks: `agents` (genome, lineage, status, tier,
balance, stats), `trades`, `strategy_shares`, `population_cycles`,
`live_position` / `live_orders` (real-money audit trail), and `meta`
(current token guard).

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set TOKEN, get an OLLAMA_API_KEY, etc.
python3 main.py        # dashboard at http://127.0.0.1:8000
```

Runs forever, one cycle every `CYCLE_SECONDS` (default 900s / 15 min).
Ctrl+C to stop. Use `--reset` after changing `TOKEN`, `--no-dashboard` to
skip the web UI.

## Scope

- Swing-trading cadence (candles + a 15-minute default loop), not HFT.
- Open-interest % change is computed against the previous cycle's snapshot
  held in memory - it resets on restart (first cycle after a restart just
  skips that one confirmation).
- Going live is a deliberate two-flag decision (network + confirmation
  phrase) with hard notional/leverage caps - read the "Network + live
  trading" section above fully before setting them.
