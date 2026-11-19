# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Reki120/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Reki120/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A Node 22 forecaster for GTA VI trophy earn rates on Exophase at the end of week one: five quantiles per trophy, from the house's recorded reads of the game's trophy page. Built by an AI agent for make.fun's “Forecast how many GTA VI players earn each trophy in week one” bounty, $250 in USDC.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Method

The program reads every case in the input directory (`/input/<id>.in`, or
`$MAKEFUN_INPUT_DIR`) and writes one forecast per case (`/output/<id>.out`):
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`, one row per trophy,
values in percent, non-decreasing, deterministic.

1. **Distinct states.** Consecutive reads with identical players-tracked and
   rates collapse into one state — Exophase refreshes on its own schedule, so
   most reads repeat the last refresh. The anchor is each trophy's last
   non-null rate.
2. **Flat history (one distinct state).** The page has not refreshed since
   launch night, and a refresh pulls rates down because the newly tracked
   players hold fewer trophies. The median logit is shifted down by a refresh
   prior, `−2.0·(1 − exp(−horizonHours/1.2))`, with a 0.45 jump term in the
   spread.
3. **Rich history.** Three views blended in logit space, fitted on
   `x = ln(1 + hours since first read)`, anchored at the last state and
   extrapolated to the target: (a) a counts model — power-law growth of
   players-tracked and of per-trophy earners (rate × players / 100), each
   earner slope shrunk toward the pooled median slope; (b) a direct trend —
   per-trophy OLS slope of logit(rate) on x, shrunk the same way; (c)
   persistence at the last rate. Weights (counts/trend/persistence) are
   0.45/0.30/0.25 for 10+ states, 0.40/0.25/0.35 for 5–9, 0.30/0.20/0.50 for
   3–4, and 0.20/0.15/0.65 for 2 states (plus half the refresh prior). The
   final median logit is clamped to ±1.6 of the last rate's logit.
4. **Spread.** A per-day logit volatility from the median state-to-state
   squared logit increment, grown over the horizon and combined with a jump
   term (0.45 flat / 0.15 rich), clamped to [0.30, 2.0]. When the median shift
   is below −0.25 the spread widens asymmetrically (downside ×1.15, upside
   ×0.9). Quantiles use z = [−1.645, −0.674, 0, 0.674, 1.645], computed from
   the case's own levels, rounded to 4 decimals.

Node 22 standard library only; no network, no model call. Two runs on the
same case write byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=rwf799wr&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
