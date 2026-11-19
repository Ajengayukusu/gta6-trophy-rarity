"""Forecast GTA VI trophy earn rates on Exophase at the case's target.

Idea
----
A trophy's earn rate is a share of the players Exophase tracks, so it moves
with two things the case file records: how many players are tracked, and how
far into launch week the game is. Launch night proved the point: while the
tracked-player count sat at 638 every rate sat still, and when Exophase
refreshed to 3,868 tracked players every rate but the one at 100% fell -- the
early crowd was the hardcore, and the broader crowd dilutes most rates.

So per trophy I fit, on the logit scale, a straight line against the log of
the tracked-player count:

    logit(rate) = a + b * log10(players_tracked)

The slope is the trophy's own story: negative for trophies the broadening
crowd dilutes, positive for early-story trophies more players keep reaching.
The fit uses the last 36 hours of reads before the checkpoint (the recent
regime; the whole history when that window holds too few distinct player
counts), with reads deduplicated by player count so the flat launch-night
plateau cannot outvote the rest.

The player count at the target is forecast the same way: log10(players)
against log10(hours since the first read + 1), extrapolated to the target
and capped at twenty times the last observed count, so a runaway fit cannot
invent players that do not exist. The median is the line read at that
player count.

The spread is honest about a week of unknown growth: the fit's residual
variance, the usual prediction variance of the line at the extrapolated
player count, and a random-walk term that grows with the horizon
(0.5 logit units per square-root day, the house baseline's own rate).
It is floored at 0.35 and capped at 3.0 logit units, so a trophy whose
history is perfectly flat still carries real uncertainty, and no trophy
carries an absurd one. The five quantiles are the median shifted by the
standard normal's quantiles times that spread, back through the logistic.

Deterministic: pure arithmetic, no randomness, no clock, no network, so two
runs write the same bytes. Standard library only.

How the grader runs it: `python3 /bot/main.py` (Python 3.13) in a sandbox
with no network. It reads every `*.in` case in /input and writes one
`*.out` forecast per case to /output. Outside the grader,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for them:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Only reads this recent before the checkpoint feed the per-trophy fit;
# the whole history is the fallback when the window is too thin.
FIT_WINDOW_HOURS = 36.0
# The random-walk spread, in logit units per square-root day.
SIGMA_PER_SQRT_DAY = 0.5
# Spread floor and cap, in logit units.
SIGMA_MIN = 0.35
SIGMA_MAX = 3.0
# The forecast player count never exceeds this multiple of the last one.
PLAYERS_CAP_MULTIPLE = 20.0


def clamp_rate(pct: float) -> float:
    """A rate as the logit sees it: the page prints 0.00 to 100.00, and
    0% and 100% have no logit, so clamp to 0.01..99.99 first."""
    return min(99.99, max(0.01, pct))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it. A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, s2, mean_x, sxx) where s2
    is the residual variance (0 when the fit is exact or degenerate) and
    sxx the spread of x around its mean. With fewer than 2 distinct x
    values the slope is 0 and the line is the mean of y."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if n < 2 or sxx <= 0:
        return mean_y, 0.0, 0.0, mean_x, 0.0
    b = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    a = mean_y - b * mean_x
    if n > 2:
        s2 = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    else:
        s2 = 0.0
    return a, b, s2, mean_x, sxx


def forecast_players(history, target_at, first_at):
    """Forecast log10(players tracked) at the target from the reads' own
    growth: log10(players) against log10(hours since the first read + 1).
    The result is capped at twenty times the last observed count."""
    xs = []
    ys = []
    for read in history:
        hours = (parse_time(read["at"]) - first_at).total_seconds() / 3600.0
        xs.append(math.log10(max(hours, 0.0) + 1.0))
        ys.append(math.log10(max(read["players_tracked"], 1)))
    a, b, _s2, _mx, _sxx = ols(xs, ys)
    target_hours = (target_at - first_at).total_seconds() / 3600.0
    x_target = math.log10(max(target_hours, 0.0) + 1.0)
    log_players = a + b * x_target
    last_log = ys[-1]
    log_players = min(log_players, last_log + math.log10(PLAYERS_CAP_MULTIPLE))
    log_players = max(log_players, last_log)
    return log_players


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    first_at = parse_time(history[0]["at"])
    horizon_days = case["horizon_minutes"] / 1440.0

    # One player-count forecast shared by every trophy.
    target_log_players = forecast_players(history, target, first_at)

    # The recent window for the per-trophy fits.
    window_start = checkpoint.timestamp() - FIT_WINDOW_HOURS * 3600.0
    recent = [r for r in history if parse_time(r["at"]).timestamp() >= window_start]

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # Reads that name this trophy, deduplicated by player count so a
        # long flat plateau counts once per distinct crowd size.
        points = {}
        for read in history:
            rate = read["rates"][i]
            if rate is None or read["players_tracked"] <= 0:
                continue
            points[read["players_tracked"]] = rate
        window_points = {}
        for read in recent:
            rate = read["rates"][i]
            if rate is None or read["players_tracked"] <= 0:
                continue
            window_points[read["players_tracked"]] = rate
        use = window_points if len(window_points) >= 3 else points
        xs = [math.log10(max(p, 1)) for p in use]
        ys = [logit(r) for r in use.values()]
        a, b, s2, mean_x, sxx = ols(xs, ys)

        median_logit = a + b * target_log_players
        # Prediction variance of the line at the extrapolated crowd, plus
        # the random walk over the horizon. The residual variance is
        # floored so a flat history still spreads.
        dx = target_log_players - mean_x
        if len(xs) > 2 and sxx > 0:
            pred_var = s2 * (1.0 + 1.0 / len(xs) + dx * dx / sxx)
        else:
            pred_var = s2
        sigma = math.sqrt(
            max(pred_var, 0.04) + (SIGMA_PER_SQRT_DAY ** 2) * horizon_days
        )
        sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
        rows.append(
            {
                "id": trophy["id"],
                "q": [
                    round4(logistic(median_logit + z * sigma)) for z in Z
                ],
            }
        )
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
