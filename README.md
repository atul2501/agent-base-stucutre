# agent-base-stucutre

An evolving population of swing-trading agents on Hyperliquid. Every agent
is a set of strategy parameters ("genome"). Trades are decided from real
Hyperliquid market data and settled against a **paper** (simulated) account
- no real orders are placed by this code.

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
  `strategy_shares`. When the population needs fresh agents (see below),
  half the time a new agent is cloned + mutated from a shared genome
  instead of generated randomly, so proven strategies propagate.
- **Population cap.** At most `POPULATION_CAP` (default 500) agents are
  alive at once. When wins push the count over the cap, the lowest-fitness
  agents that have completed at least one trade are culled first (brand new
  agents get a grace period so they aren't killed before they've had a
  chance to trade).
- **Active traders.** Only the top `ACTIVE_TRADER_COUNT` (default 50) alive
  agents, ranked by fitness (realized PnL, win rate as tie-break), are
  allowed to open new positions each cycle. The rest keep tracking and
  waiting for a better rank.
- **Population floor.** If deaths shrink the population below
  `MIN_POPULATION_FLOOR`, fresh agents are spawned to keep the system
  self-sustaining.

## How an agent decides to trade

1. Pull the agent's coin's latest candles, L2 order book, open interest and
   funding rate from Hyperliquid (`market/hyperliquid_client.py`).
2. Compute an EMA trend + RSI pullback/rally trigger
   (`strategy/indicators.py`, `strategy/signals.py`).
3. Score the trigger against order book imbalance, open-interest change,
   and funding-rate crowding as confirmations/contradictions.
4. A clearly confirmed setup fires the trade; a clearly contradicted one
   holds. A genuinely mixed read is sent to a local **Ollama** model
   (`reasoning/ollama_advisor.py`) for a judgment call instead of guessing -
   this is the "self-understanding" fallback. If Ollama is unavailable, the
   agent defaults to holding rather than crashing.
5. Exits are rule-based: stop loss / take profit / max hold time
   (all genome parameters).

## Storage

SQLite (`trading_agents.db` by default) tracks:

- `agents` - genome, lineage (`parent_id`, `generation`), status
  (`alive`/`dead`), tier (`standard`/`professional`), balance, win/loss
  stats, win streak.
- `trades` - every open/closed paper trade per agent.
- `strategy_shares` - genomes shared after an 8-win streak.
- `population_cycles` - a log line per cycle (alive count, active traders,
  professional count, best performer) for tracking the population's health
  over time.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # adjust symbols / population size / etc. if desired
ollama pull qwen2.5:7b # only needed if OLLAMA_ENABLED=true
python3 main.py
```

It runs forever, one cycle every `CYCLE_SECONDS` (default 900s / 15 min,
matched to swing-trading timeframes rather than tick-by-tick). Ctrl+C to
stop.

## Scope / what this is not

- **Paper trading only.** `TRADING_MODE=live` is rejected at startup - live
  order placement against Hyperliquid was deliberately not implemented in
  this pass. Wiring it up later would mean adding an executor that uses the
  SDK's `Exchange` class with the wallet credentials in `.env`, and should
  be done deliberately with its own risk limits, not as a flag flip.
- **Single process, in-memory OI diffing.** Open-interest % change is
  computed against the previous cycle's snapshot, held in memory - it
  resets on restart (first cycle after a restart just skips the OI
  confirmation for one round).
- Not HFT: this is a swing-trading cadence (candles + a 15-minute default
  loop), not a low-latency market maker.
