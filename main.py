"""Forecast GTA VI's launch-day Twitch viewers, hours 2..24, as five log-scale
quantiles per hour.

Model (fit fresh from each case's own comparables at run time; nothing is
learned from the future, and every step is deterministic):

  The shape of a launch day in log space is mostly three things: a stable
  time-of-day curve (Twitch viewership peaks around 00Z and 19-20Z and
  troughs around 08-09Z, measured here from the comparables' second and
  third days, when every region is unlocked), a staircase of regional
  unlock waves (New Zealand first, then Australia, Japan/Korea, central
  Europe, the UK, and the two US coasts, each at its local midnight), and
  the launch's scale, read off hour 1's print and the pre-release ramp.

  Three forecasters are blended in log space with equal weight:

  1. Nearest neighbours: the six past launches closest in hour-1 scale
     and pre-release ramp shape (log ratios of hours -1, -5, -13, -24 and
     the pre-day peak to hour 1); each hour's median and spread come from
     those neighbours' realised log-ratio curves.
  2. A bounded per-hour regression of the log-ratio curve on hour-1 scale,
     the ramp ratios, the pre-day peak and the launch year. Its hour-1
     input is winsorised to the donors' range and its medians are clamped
     to the rolling launches' realised range, so a record-breaking hour-1
     print cannot make it extrapolate to a collapsing day.
  3. A wave model: the case's pre-release ramp is fit with a staircase of
     unlock responses (one step per region, at the local-midnight offsets
     for the release date, daylight saving included) after removing the
     time-of-day curve; the still-locked US waves are then added at the
     size the rolling launches' US waves bear to their pre-US level.

  The blend's spread is the blended quantile width narrowed by a factor
  calibrated by leave-one-out validation over the comparables. If any
  component cannot run (too few donors, a sparse ramp), the blend uses
  the rest; if none can run, the forecast degrades to persistence with a
  random-walk spread, so the program always writes a valid file.

Standard library only. Reads every <id>.in in the input directory and
writes <id>.out: {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]}
for hours 2..24. Outside the grader, MAKEFUN_INPUT_DIR and
MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Hours of the release day to forecast.
FORECAST_HOURS = list(range(2, 25))
# Pre-release offsets (hours before day_start) used as ramp features.
RAMP_OFFSETS = [-1, -5, -13, -24]
# Nearest-neighbour count and the blend's spread calibration (leave-one-out).
KNN_K = 6
SPREAD_FACTOR = 0.85
SD_FLOOR = 0.08
SD_CAP = 1.5
# Fallback persistence spread, log units per square-root hour.
FALLBACK_SIGMA = 0.15
# Wave model's fixed spread for its own quantiles.
WAVE_SD = 0.4

# Fallback time-of-day curve (log units, mean zero), used only when no
# comparable carries a usable second or third day. The primary path always
# measures it from the case's own comparables.
FALLBACK_DIURNAL = [
    0.31, 0.18, 0.13, -0.01, -0.08, -0.19, -0.26, -0.31,
    -0.37, -0.37, -0.33, -0.31, -0.17, -0.05, 0.08, 0.15,
    0.15, 0.20, 0.24, 0.28, 0.30, 0.26, 0.20, 0.11,
]


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def pstdev(values):
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    return math.sqrt(sum((v - mu) ** 2 for v in values) / n)


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting; None when singular."""
    n = len(rhs)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for row in range(n):
            if row != col:
                factor = aug[row][col] / aug[col][col]
                if factor:
                    for c in range(col, n + 1):
                        aug[row][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def parse_time(text):
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None


def nth_weekday(year, month, weekday, n):
    """Date of the nth `weekday` (Monday 0) of a month."""
    first = datetime(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    return datetime(year, month, 1 + delta + 7 * (n - 1))


def last_weekday(year, month, weekday):
    if month == 12:
        nxt = datetime(year + 1, 1, 1)
    else:
        nxt = datetime(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def utc_offset(region, day):
    """UTC offset in hours of a region on a date (northern/southern DST rules)."""
    year = day.year
    if region == "NZ":
        start = last_weekday(year, 9, 6)
        end = nth_weekday(year, 4, 6, 1)
        return 13 if (day >= start or day < end) else 12
    if region == "AU":
        start = nth_weekday(year, 10, 6, 1)
        end = nth_weekday(year, 4, 6, 1)
        return 11 if (day >= start or day < end) else 10
    if region == "JP":
        return 9
    if region == "CET":
        start = last_weekday(year, 3, 6)
        end = last_weekday(year, 10, 6)
        return 2 if start <= day < end else 1
    if region == "UK":
        start = last_weekday(year, 3, 6)
        end = last_weekday(year, 10, 6)
        return 1 if start <= day < end else 0
    if region == "USE":
        start = nth_weekday(year, 3, 6, 2)
        end = nth_weekday(year, 11, 6, 1)
        return -4 if start <= day < end else -5
    if region == "USP":
        start = nth_weekday(year, 3, 6, 2)
        end = nth_weekday(year, 11, 6, 1)
        return -7 if start <= day < end else -8
    return 0


REGIONS = ["NZ", "AU", "JP", "CET", "UK", "USE", "USP"]
PRE_US_REGIONS = ["NZ", "AU", "JP", "CET", "UK"]


def unlock_offsets(day_start):
    """Hour offsets from day_start at which each region's local midnight falls."""
    day = parse_time(day_start) or datetime(2026, 11, 19)
    return {region: -utc_offset(region, day) for region in REGIONS}


def unlock_response(tau, length=1.0):
    """A region's unlock response tau hours after its midnight: 0 before, a
    short ramp as streams start, then 1."""
    if tau < 0:
        return 0.0
    if tau >= length:
        return 1.0
    return tau / length


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def hourly_curve(points, day_start, from_key="from"):
    """Collapse timestamped points to {hour offset from day_start: viewers},
    averaging points that share an hour. Non-positive prints are dropped:
    they carry no log-scale information."""
    start = parse_time(day_start)
    if start is None:
        return {}
    buckets = {}
    for point in points or []:
        stamp = parse_time(point.get(from_key)) if isinstance(point, dict) else None
        viewers = finite_number(point.get("viewers")) if isinstance(point, dict) else None
        if stamp is None or viewers is None or viewers <= 0:
            continue
        offset = int((stamp - start).total_seconds() // 3600)
        buckets.setdefault(offset, []).append(viewers)
    return {off: sum(vals) / len(vals) for off, vals in buckets.items()}


def case_hourly_history(case):
    """The case's own pre-release hourly curve (offsets <= 0). Prefers the
    hourly history; falls back to averaging the minute reads by hour."""
    day_start = case.get("day_start")
    curve = hourly_curve(case.get("history_hourly"), day_start)
    if curve:
        return {h: v for h, v in curve.items() if h <= 0}
    # Minute reads carry `at` instead of `from`.
    start = parse_time(day_start)
    if start is None:
        return {}
    buckets = {}
    for point in case.get("history") or []:
        if not isinstance(point, dict):
            continue
        stamp = parse_time(point.get("at"))
        viewers = finite_number(point.get("viewers"))
        if stamp is None or viewers is None or viewers <= 0:
            continue
        offset = int((stamp - start).total_seconds() // 3600)
        if offset <= 0:
            buckets.setdefault(offset, []).append(viewers)
    return {off: sum(vals) / len(vals) for off, vals in buckets.items()}


def hour_one_print(case, history_curve):
    """Hour 1's print: the `known` row for hour 1, else the history's hour-0
    point, else the last pre-release point, else 1. A print below 1 counts
    as 1."""
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == 1:
            viewers = finite_number(row.get("viewers"))
            if viewers is not None:
                return max(1.0, viewers)
    if 0 in history_curve:
        return max(1.0, history_curve[0])
    if history_curve:
        latest = max(history_curve)
        return max(1.0, history_curve[latest])
    return 1.0


def is_launch_curve(comp):
    """The GTA VI category's own history is not a launch and must not be a
    donor; launch comparables carry full release-day curves."""
    text = " ".join(str(comp.get(key) or "") for key in ("name", "unlock")).lower()
    return "not a launch" not in text


def classify_unlock(comp):
    """Coarse unlock pattern from the research notes: rolling midnight-local,
    simultaneous global, early-access contaminated, or unknown."""
    text = str(comp.get("unlock") or "").lower()
    early = "early-access" in text or "early access" in text
    simultaneous = (
        "simultaneous" in text or "single global" in text or "single time" in text
    ) and "rolling" not in text and "midnight local" not in text
    rolling = ("midnight local" in text or "rolling" in text) and not simultaneous
    if early:
        return "EA"
    if simultaneous:
        return "SIM"
    if rolling:
        return "ROLL"
    return "UNKN"


def is_console_only(comp):
    platforms = str(comp.get("platforms") or "").lower()
    return not any(word in platforms for word in ("pc", "steam", "windows"))


def load_donors(case):
    """Parse every usable comparable into a donor record. A donor needs a
    positive hour-1 print; the day curve and ramp features are whatever the
    series carries."""
    donors = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict) or not is_launch_curve(comp):
            continue
        curve = hourly_curve(comp.get("hourly"), comp.get("day_start"))
        if 0 not in curve:
            continue
        day = parse_time(comp.get("day_start"))
        donors.append({
            "name": str(comp.get("name") or ""),
            "day_start": comp.get("day_start"),
            "year": float(day.year) if day else 2026.0,
            "tag": classify_unlock(comp),
            "console": is_console_only(comp),
            "curve": curve,
        })
    return donors


def ramp_features(curve):
    """Log-scale ramp features from data at or before hour 1: hour-1 scale,
    log ratios of hours -1/-5/-13/-24 to hour 1, and the pre-day peak's log
    ratio. Missing hours stay None and are imputed from the donors."""
    if 0 not in curve:
        return None
    first = curve[0]
    feats = {"ln_h1": math.log(first)}
    for offset in RAMP_OFFSETS:
        key = "r%d" % offset
        feats[key] = math.log(curve[offset] / first) if offset in curve else None
    pre = [curve[h] for h in range(-24, 0) if h in curve]
    feats["prepeak"] = math.log(max(pre) / first) if pre else None
    return feats


def impute_features(rows, keys):
    """Fill missing features with the donors' medians; returns the filled
    rows and the medians used (for the case's own row)."""
    meds = {}
    for key in keys:
        vals = [row[key] for row in rows if row.get(key) is not None]
        meds[key] = median(vals) if vals else 0.0
    filled = [[row.get(key) if row.get(key) is not None else meds[key] for key in keys]
              for row in rows]
    return filled, meds


def donor_day_curves(donors):
    """Donors with a complete release day: log-ratio curves for hours 1..23
    (offsets 1..23) plus their ramp features."""
    usable = []
    for donor in donors:
        curve = donor["curve"]
        if any(h not in curve for h in range(0, 24)):
            continue
        feats = ramp_features(curve)
        if feats is None:
            continue
        feats["year"] = donor["year"]
        ratios = [math.log(curve[h] / curve[0]) for h in range(1, 24)]
        usable.append({"feats": feats, "ratios": ratios, "tag": donor["tag"],
                       "console": donor["console"]})
    return usable


KNN_KEYS = ["ln_h1", "r-1", "r-5", "r-13", "r-24", "prepeak"]
OLS_KEYS = ["ln_h1", "r-1", "r-5", "prepeak", "year"]


def knn_forecast(donor_rows, case_feats):
    """Median and spread of the K nearest donors' log-ratio curves, in
    standardised feature space. Returns 23 rows of five quantile ratios."""
    if len(donor_rows) < 3:
        return None
    train, meds = impute_features([d["feats"] for d in donor_rows], KNN_KEYS)
    point = [case_feats.get(k) if case_feats.get(k) is not None else meds[k] for k in KNN_KEYS]
    width = len(KNN_KEYS)
    means = [sum(row[j] for row in train) / len(train) for j in range(width)]
    scales = [pstdev([row[j] for row in train]) or 1.0 for j in range(width)]

    def distance(row):
        return math.sqrt(sum(((row[j] - point[j]) / scales[j]) ** 2 for j in range(width)))

    order = sorted(range(len(train)), key=lambda i: (distance(train[i]), i))
    neighbours = order[:KNN_K]
    rows = []
    for h in range(23):
        vals = sorted(donor_rows[i]["ratios"][h] for i in neighbours)
        centre = median(vals)
        spread = max(SD_FLOOR, pstdev(vals))
        rows.append([math.exp(centre + z * spread) for z in Z])
    return rows


def fit_regression(train, targets):
    """Per-hour ordinary least squares with a tiny ridge for stability.
    Returns (coefficients, residual sds) per hour."""
    n = len(train)
    width = len(train[0])
    design = [[1.0] + row for row in train]
    gram = [[sum(design[k][i] * design[k][j] for k in range(n)) for j in range(width + 1)]
            for i in range(width + 1)]
    for i in range(1, width + 1):
        gram[i][i] += 1e-6
    coefs, sds = [], []
    for h in range(23):
        target = [row[h] for row in targets]
        rhs = [sum(design[k][i] * target[k] for k in range(n)) for i in range(width + 1)]
        beta = solve_linear(gram, rhs)
        if beta is None:
            beta = [median(target)] + [0.0] * width
        fitted = [sum(beta[j] * design[k][j] for j in range(width + 1)) for k in range(n)]
        resid = [target[k] - fitted[k] for k in range(n)]
        coefs.append(beta)
        sds.append(max(SD_FLOOR, pstdev(resid)))
    return coefs, sds


def regression_forecast(donor_rows, case_feats):
    """Bounded per-hour regression. The hour-1 input is winsorised to the
    donors' range, and each hour's median is clamped to the rolling and
    console-only donors' realised range, so record scale cannot produce a
    collapsing day."""
    if len(donor_rows) < 8:
        return None
    train, meds = impute_features([d["feats"] for d in donor_rows], OLS_KEYS)
    point = [case_feats.get(k) if case_feats.get(k) is not None else meds[k] for k in OLS_KEYS]
    low = min(row[0] for row in train)
    high = max(row[0] for row in train)
    point[0] = max(low, min(high, point[0]))
    coefs, sds = fit_regression(train, [d["ratios"] for d in donor_rows])
    clamp_rows = [d for d in donor_rows if d["tag"] == "ROLL" or d["console"]]
    if len(clamp_rows) < 5:
        clamp_rows = donor_rows
    rows = []
    for h in range(23):
        beta = coefs[h]
        centre = beta[0] + sum(beta[j + 1] * point[j] for j in range(len(point)))
        vals = [d["ratios"][h] for d in clamp_rows]
        centre = max(min(vals) - 0.1, min(max(vals) + 0.1, centre))
        rows.append([math.exp(centre + z * sds[h]) for z in Z])
    return rows


def estimate_diurnal(donors):
    """The time-of-day curve in log units (mean zero): for each hour of the
    day, the median across donors of that hour's log viewers minus the
    day's mean log viewers, over the second and third days, when every
    region is unlocked."""
    curve = []
    for hour_of_day in range(24):
        vals = []
        for donor in donors:
            series = donor["curve"]
            for base in (24, 48):
                day = [series.get(base + k) for k in range(24)]
                if all(v is not None and v > 0 for v in day):
                    mean_log = sum(math.log(v) for v in day) / 24
                    vals.append(math.log(day[hour_of_day]) - mean_log)
        curve.append(median(vals) if vals else FALLBACK_DIURNAL[hour_of_day])
    mean_curve = sum(curve) / 24
    return [v - mean_curve for v in curve]


def fit_steps(offsets, levels, unlocks, regions):
    """Least squares for levels[h] = c + sum of region jumps x unlock
    responses. Jumps are clipped at zero: an unlock never removes viewers."""
    basis = [[1.0] + [unlock_response(h - unlocks[r]) for r in regions] for h in offsets]
    width = len(regions) + 1
    n = len(offsets)
    gram = [[sum(basis[k][i] * basis[k][j] for k in range(n)) for j in range(width)]
            for i in range(width)]
    for i in range(1, width):
        gram[i][i] += 1e-6
    rhs = [sum(basis[k][i] * levels[k] for k in range(n)) for i in range(width)]
    beta = solve_linear(gram, rhs)
    if beta is None:
        return None
    return beta[0], {r: max(0.0, beta[i + 1]) for i, r in enumerate(regions)}


def wave_forecast(case_curve, case_day_start, donors):
    """Unlock-wave forecast. The case's de-diurnalised pre-release ramp is
    fit with the pre-US unlock steps; the US steps are added at the size
    rolling launches' US steps bear to their pre-US level; the time-of-day
    curve is then re-applied. Returns 23 median log ratios, or None."""
    diurnal = estimate_diurnal(donors)
    if 0 not in case_curve:
        return None
    base = math.log(case_curve[0]) - diurnal[0]
    obs_offsets, obs_levels = [], []
    for h in range(-24, 1):
        if h in case_curve:
            obs_offsets.append(h)
            obs_levels.append(math.log(case_curve[h]) - diurnal[h % 24] - base)
    if len(obs_offsets) < 8:
        return None
    unlocks = unlock_offsets(case_day_start)
    fitted = fit_steps(obs_offsets, obs_levels, unlocks, PRE_US_REGIONS)
    if fitted is None:
        return None
    intercept, pre_jumps = fitted
    pre_level = intercept + sum(pre_jumps.values())
    # Rolling donors' US-to-pre-US jump ratios, from full-curve step fits.
    ratios, splits = [], []
    for donor in donors:
        if donor["tag"] != "ROLL":
            continue
        series = donor["curve"]
        if 0 not in series:
            continue
        donor_unlocks = unlock_offsets(donor["day_start"])
        donor_base = math.log(series[0]) - diurnal[0]
        hours, levels = [], []
        for h in range(-24, 24):
            if h in series:
                hours.append(h)
                levels.append(math.log(series[h]) - diurnal[h % 24] - donor_base)
        if len(hours) < 30:
            continue
        donor_fit = fit_steps(hours, levels, donor_unlocks, REGIONS)
        if donor_fit is None:
            continue
        donor_c, donor_jumps = donor_fit
        pre = donor_c + sum(donor_jumps[r] for r in PRE_US_REGIONS)
        us_jumps = donor_jumps["USE"] + donor_jumps["USP"]
        if pre > 0.05 and us_jumps >= 0:
            ratios.append(us_jumps / pre)
            splits.append(donor_jumps["USE"] / us_jumps if us_jumps > 0 else 0.5)
    if not ratios:
        # No rolling donor fit: fall back to every donor's ratio.
        for donor in donors:
            series = donor["curve"]
            if 0 not in series:
                continue
            donor_unlocks = unlock_offsets(donor["day_start"])
            donor_base = math.log(series[0]) - diurnal[0]
            hours, levels = [], []
            for h in range(-24, 24):
                if h in series:
                    hours.append(h)
                    levels.append(math.log(series[h]) - diurnal[h % 24] - donor_base)
            if len(hours) < 30:
                continue
            donor_fit = fit_steps(hours, levels, donor_unlocks, REGIONS)
            if donor_fit is None:
                continue
            donor_c, donor_jumps = donor_fit
            pre = donor_c + sum(donor_jumps[r] for r in PRE_US_REGIONS)
            us_jumps = donor_jumps["USE"] + donor_jumps["USP"]
            if pre > 0.05 and us_jumps >= 0:
                ratios.append(us_jumps / pre)
                splits.append(donor_jumps["USE"] / us_jumps if us_jumps > 0 else 0.5)
    if not ratios:
        return None
    ratio = median(ratios)
    split = median(splits) if splits else 0.5
    jumps = dict(pre_jumps)
    jumps["USE"] = pre_level * ratio * split
    jumps["USP"] = pre_level * ratio * (1.0 - split)
    level_at_zero = intercept + sum(
        jumps[r] * unlock_response(0 - unlocks[r]) for r in REGIONS)
    preds = []
    for h in range(1, 24):
        level = intercept + sum(jumps[r] * unlock_response(h - unlocks[r]) for r in REGIONS)
        preds.append(level + diurnal[h % 24] - diurnal[0] - level_at_zero)
    return [[math.exp(p + z * WAVE_SD) for z in Z] for p in preds]


def blend_forecasts(component_rows):
    """Equal-weight blend in log space, then a recalibrated symmetric spread
    around the blended median."""
    parts = [rows for rows in component_rows if rows is not None]
    if not parts:
        return None
    weight = 1.0 / len(parts)
    rows = []
    for h in range(23):
        blended = []
        for qi in range(5):
            log_val = sum(weight * math.log(max(1e-12, part[h][qi])) for part in parts)
            blended.append(math.exp(log_val))
        centre = math.log(blended[2])
        width = (math.log(blended[4]) - math.log(blended[0])) / (2 * 1.645)
        spread = max(SD_FLOOR, min(SD_CAP, width * SPREAD_FACTOR))
        rows.append([math.exp(centre + z * spread) for z in Z])
    return rows


def persistence_rows(first_print):
    """The degrade-to fallback: hour 1's print for every hour, with a
    random-walk spread widening in log space."""
    rows = []
    for h in FORECAST_HOURS:
        spread = FALLBACK_SIGMA * math.sqrt(h - 1)
        rows.append([first_print * math.exp(z * spread) for z in Z])
    return rows


def round_half_up(value):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100
    return int(rounded) if float(rounded).is_integer() else rounded


def clean_quantiles(raw_rows, first_print):
    """Clamp to (0, 1e9], enforce non-decreasing order, round half up, and
    re-enforce order after rounding. Always returns 23 valid rows."""
    rows = []
    for h, raw in zip(FORECAST_HOURS, raw_rows):
        vals = []
        for q in raw:
            try:
                number = float(q)
            except (TypeError, ValueError):
                number = first_print
            if not math.isfinite(number) or number <= 0:
                number = first_print
            vals.append(min(1e9, number))
        ordered = []
        running = 0.0
        for v in vals:
            running = max(running, v)
            ordered.append(running)
        rounded = [round_half_up(v) for v in ordered]
        final = []
        running = 0.0
        for v in rounded:
            running = max(running, float(v))
            final.append(running if running > 0 else 0.01)
        # Rounding can push a value to 0 only when the print itself is
        # tiny; keep every quantile strictly positive and ordered.
        fixed = []
        running = 0.0
        for v in final:
            running = max(running, v)
            if running <= 0:
                running = 0.01
            fixed.append(int(running) if float(running).is_integer() else running)
        rows.append({"hour": h, "q": fixed})
    return rows


def forecast_case(case):
    """Forecast one case. Any failure inside the model degrades to the
    persistence fallback; this function never raises."""
    try:
        history_curve = case_hourly_history(case)
        first_print = hour_one_print(case, history_curve)
        history_curve = dict(history_curve)
        history_curve[0] = first_print
        donors = load_donors(case)
        day_rows = donor_day_curves(donors)
        case_feats = ramp_features(history_curve) or {"ln_h1": math.log(first_print)}
        day = parse_time(case.get("day_start"))
        case_feats["year"] = float(day.year) if day else 2026.0
        components = [
            knn_forecast(day_rows, case_feats),
            regression_forecast(day_rows, case_feats),
            wave_forecast(history_curve, case.get("day_start"), donors),
        ]
        blended = blend_forecasts(components)
        if blended is None:
            blended = persistence_rows(first_print)
        else:
            # The components forecast ratios to hour 1's print; scale back
            # to viewers before the contract cleanup.
            blended = [[first_print * q for q in row] for row in blended]
        return {"hours": clean_quantiles(blended, first_print)}
    except Exception:
        try:
            first_print = hour_one_print(case, case_hourly_history(case))
        except Exception:
            first_print = 1.0
        return {"hours": clean_quantiles(persistence_rows(first_print), first_print)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(case, dict):
                raise ValueError("case is not an object")
            forecast = forecast_case(case)
        except Exception:
            forecast = {"hours": clean_quantiles(persistence_rows(1.0), 1.0)}
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
