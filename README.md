# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ajengayukusu/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ajengayukusu/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one. A trophy's earn rate is a share of the players Exophase tracks, so it moves with the tracked-player count: launch night proved it — while the count sat at 638 every rate sat still, and when Exophase refreshed to 3,868 tracked players every rate but the one at 100% fell, the hardcore early crowd diluted by a broader one. Per trophy, the program fits logit(rate) against log10(players tracked) over the last 36 hours of reads, forecasts the player count at the target from its own log-log growth (capped at twenty times the last count), and reads the line there for the median. The spread combines the fit's residual and prediction variance with a random walk over the horizon, floored at 0.35 and capped at 3.0 logit units, so a flat history still carries real uncertainty about a week of growth.

Tested locally on both practice cases: valid shape (55 rows, five non-decreasing quantiles each, 0–100), and two runs byte-identical.

## Run it

```sh
python3 main.py
```

It reads every `*.in` case in `/input` and writes one `*.out` forecast per case to `/output` (`MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` stand in for them outside the grader). Deterministic: two runs write the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=50v8qgc6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
