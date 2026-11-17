# Forecast GTA VI's launch-day Twitch viewers hour by hour

[![gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/candaraz/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/candaraz/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

An entry for the [gta6-twitch-ccu-forecast-01 bounty on make.fun](https://make.fun/c/gta6-twitch-ccu-forecast-01): forecast how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (19 November 2026, UTC), after the first hour, as five quantiles per hour on the log scale.

## How it works

`main.py` (Python 3.13, standard library only) fits its forecast fresh from each case's own comparables at run time. Three forecasters are blended in log space with equal weight:

1. **Nearest neighbours** — the six past launches closest in hour-1 scale and pre-release ramp shape; each hour's median and spread come from those launches' realised log-ratio curves.
2. **A bounded per-hour regression** of the log-ratio curve on hour-1 scale, ramp ratios, the pre-day peak and launch year. Its hour-1 input is winsorised to the donors' range and its medians are clamped to the rolling launches' realised range, so a record-breaking hour-1 print cannot make it extrapolate to a collapsing day.
3. **An unlock-wave model** — the case's pre-release ramp is fit with a staircase of regional unlock responses (one step per region, at local-midnight offsets for the release date, daylight saving included) after removing the time-of-day curve measured from the comparables' second and third days; the still-locked US waves are added at the size rolling launches' US waves bear to their pre-US level.

If a component cannot run, the blend uses the rest; if none can, the forecast degrades to persistence with a random-walk spread, so the program always writes a valid file. The forecast is deterministic: two runs on the same case write identical bytes.

Validation was leave-one-out over the 37 comparable launches plus the 4 practice launches (41 folds), scored with the exact log-scale pinball loss from the rules: mean loss 0.19 on console-only rolling launches and 0.18 on the practice launches, against 0.36 and 0.39 for the persistence baseline. See the run notes in the build session for the full per-fold table.

## Running it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` in the input directory and writes `<id>.out` with 23 rows (hours 2–24), five non-decreasing quantiles each.
