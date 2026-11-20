"""gta6-trophy-rarity-forecast-01 entry: an earners/players decomposition forecaster.

Idea: an earn rate is a ratio, earners / players_tracked, and the two sides move for
different reasons. New tracked players dilute every rate the moment Exophase counts
them; earners only grow as the people already tracked keep playing. So this program
forecasts the two sides separately and divides at the target:

  * players_tracked: a blend of a saturating logistic in time and a power law in
    log-time, fitted to the distinct page states in the case's history;
  * earners of each trophy (rate x players / 100), forecast two ways and blended:
    a recency-weighted power law in log-time with its slope shrunk toward the
    pooled slope of its tier, and a lag-ratio model -- earners are a share of the
    players who had arrived one maturation lag earlier, the lag chosen per trophy
    for the share's stability, the share extrapolated with a damped trend. The
    blend is floored at the last observed count and capped by the player forecast;
  * the median rate is 100 x earners / players at the target, and the five quantiles
    are logit-normal around it, with a spread built from the fits' residuals, the two
    player models' disagreement, a random-walk diffusion over the horizon, and a
    floor for the refresh risk (the answer is the page as of Exophase's last refresh
    before the target, which nobody can time).

That decomposition is blended with a direct trend model -- per trophy, a
recency-weighted line of logit(rate) on ln(1 + hours since release), slope shrunk
toward its tier's pool, its move capped -- because the two fail differently: the
trend follows a local ramp but knows nothing of saturation or dilution, the
decomposition knows both but smooths the ramp. The blend weights are not fixed:
both models are refit on the first ~70% of the case's states and scored on the
rest, and the one that tracked this case's own dynamics better leads at the target.

A history that shows one unchanging page (the launch-night pattern: Exophase has not
refreshed yet) takes a separate branch that expects the refresh: players multiply,
earners lag, rates fall, and the spread is honestly wide.

Run as the grader runs it: python3 -P -s /bot/main.py, standard library only, no
network. Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output). Deterministic: no randomness, no clock, fixed
iteration order, so two runs write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc).timestamp()

# Tuning constants (see the model notes in the docstring).
DIFFUSION_PER_SQRT_DAY = 0.34   # logit spread of pure drift over the horizon
SPREAD_FLOOR = 0.15             # logit spread nobody gets below
SPREAD_CAP = 2.60
SIGMA_SCALE = 0.75   # global calibration of the final spread (synthetic-eval tuned)
LOGISTIC_MULTS = (1.15, 1.3, 1.6, 2.0, 2.6, 3.5, 5.0, 7.5, 11.0, 17.0, 26.0, 40.0, 65.0, 110.0)


def parse_hours(text):
    """ISO 8601 UTC timestamp -> hours since the game's release (2026-11-19T00:00Z)."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.timestamp() - RELEASE) / 3600.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers stay whole in the JSON."""
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(v) if float(v).is_integer() else v


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, residual_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    dof = max(n - 2, 1)
    sig = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / dof)
    return a, b, sig


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (b, x_bar, y_bar, residual_std,
    sxx) with the bars the weighted centroids the fitted line passes through and
    sxx the weighted spread of x (the slope's standard error is sig / sqrt(sxx))."""
    wsum = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / wsum
    my = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return 0.0, mx, my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    dof = max(sum(1 for w in ws if w > 0.05) - 2, 1)
    sig = math.sqrt(sum(w * (y - (my + b * (x - mx))) ** 2
                        for w, x, y in zip(ws, xs, ys)) / dof)
    return b, mx, my, sig, sxx


# --------------------------------------------------------------------------
# History handling
# --------------------------------------------------------------------------

def read_states(case):
    """Compress the read history into distinct page states.

    Exophase refreshes its counts in steps, so long runs of reads show one identical
    page; each run is one observation of the underlying process, timed at its first
    read. Returns (states, at_hours, target_hours) with states a list of
    {"t": hours, "players": float, "rates": [...]}.
    """
    states = []
    prev_key = None
    for read in case["history"]:
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        try:
            t = parse_hours(read["at"])
        except Exception:
            t = states[-1]["t"] if states else 0.0
        states.append({"t": t, "players": float(players) if players else 0.0,
                       "rates": list(rates)})
    at_h = parse_hours(case["at"])
    return states, at_h, at_h + case["horizon_minutes"] / 60.0


def last_rate(states, i):
    """A trophy's rate in the newest state that lists it (None if never listed)."""
    for st in reversed(states):
        if i < len(st["rates"]) and st["rates"][i] is not None:
            return st["rates"][i]
    return None


def logp_interp(pts):
    """log players as a function of time, log-linear between observed points,
    flat outside them. pts is a list of (hours, players)."""
    def f(t):
        if t <= pts[0][0]:
            return math.log(pts[0][1])
        for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                if tb <= ta:
                    return math.log(pb)
                g = (t - ta) / (tb - ta)
                return math.log(pa) + g * (math.log(pb) - math.log(pa))
        return math.log(pts[-1][1])
    return f


# --------------------------------------------------------------------------
# Players forecast
# --------------------------------------------------------------------------

def forecast_players(states, target_h):
    """Forecast players: returns (predict_logp, sigma_log_players).

    predict_logp(t) gives log players at any time t: the observed series itself
    (log-linearly interpolated) inside the history, and outside it a blend of
    saturating logistic curves over a grid of ceilings with a power law in
    log-time, weighted by fit quality. Earners are a lagged shadow of the player
    curve, so the earners fit needs the whole curve, not just its endpoint.
    """
    pts = [(st["t"], st["players"]) for st in states if st["players"] > 0]
    p_last = pts[-1][1]
    t_last = pts[-1][0]
    observed = logp_interp(pts)

    if len(pts) < 2 or all(p == p_last for _, p in pts):
        return (lambda t: observed(t) if t <= t_last else math.log(p_last)), 0.55
    t0 = pts[0][0]
    p_max = max(p for _, p in pts)

    def log_sse(preds):
        return sum((math.log(p) - pr) ** 2 for (_, p), pr in zip(pts, preds))

    candidates = []  # (predict_logp_fn, log_sse, residual_sigma)
    for mult in LOGISTIC_MULTS:
        ceil = p_max * mult
        if ceil <= p_max * 1.05:
            continue
        ys = [math.log((p / ceil) / (1.0 - p / ceil)) for _, p in pts]
        a, b, _ = ols([t for t, _ in pts], ys)
        if b <= 0:
            continue

        def make_logistic(a=a, b=b, ceil=ceil):
            def f(t):
                z = clamp(a + b * t, -30.0, 30.0)
                return math.log(ceil / (1.0 + math.exp(-z)))
            return f

        fn = make_logistic()
        sse = log_sse([fn(t) for t, _ in pts])
        candidates.append((fn, sse, math.sqrt(sse / max(len(pts) - 2, 1))))
    xs = [math.log(1.0 + (t - t0)) for t, _ in pts]
    a, b, sig_pw = ols(xs, [math.log(p) for _, p in pts])
    if b > 0:
        def power_fn(t, a=a, b=b, t0=t0):
            return a + b * math.log(1.0 + max(t - t0, 0.0))

        candidates.append((power_fn, log_sse([power_fn(t) for t, _ in pts]), sig_pw))

    if not candidates:
        return (lambda t: observed(t) if t <= t_last else math.log(p_last)), 0.55
    weights = [1.0 / (sse + 0.02) for _, sse, _ in candidates]
    wsum = sum(weights)
    log_cap = math.log(p_last) + math.log(80.0)

    def predict(t):
        if t <= t_last:
            return observed(t)
        v = sum(w * fn(t) for w, (fn, _, _) in zip(weights, candidates)) / wsum
        return clamp(v, math.log(p_last), log_cap)

    sig = sum(w * c[2] for w, c in zip(weights, candidates)) / wsum
    spread = max(fn(target_h) for fn, _, _ in candidates) - min(fn(target_h) for fn, _, _ in candidates)
    sigma = clamp(sig + 0.30 * spread + 0.10, 0.12, 1.10)
    return predict, sigma


# --------------------------------------------------------------------------
# Earners forecast
# --------------------------------------------------------------------------

# Earners accumulate as cohorts of players mature, so their growth in log-time is
# convex early: a line through the whole history averages the flat first hours with
# the recent ramp and misses both. The earner fit therefore weights recent states
# exponentially more (half-life EARNER_TAU_H hours) and is anchored at the weighted
# centroid of the recent data.
EARNER_TAU_H = 30.0


def forecast_earners(states, i, t0, target_h, p_target):
    """Fit one trophy's earners as a power law in log-time, recency-weighted.

    Returns a fit dict (slope, weighted centroid, residual sigma, last earners,
    count) or None when the trophy never appears.
    """
    xs, ys, ws = [], [], []
    t_last = states[-1]["t"]
    for st in states:
        if i < len(st["rates"]) and st["rates"][i] is not None and st["players"] > 0:
            earners = st["rates"][i] * st["players"] / 100.0
            xs.append(math.log(1.0 + (st["t"] - t0)))
            ys.append(math.log(earners + 0.5))
            ws.append(math.exp(-(t_last - st["t"]) / EARNER_TAU_H))
    if not xs:
        return None
    b, mx, my, sig, _sxx = wls(xs, ys, ws)
    e_last = math.exp(ys[-1]) - 0.5
    return {"b": b, "mx": mx, "my": my, "sig": sig, "e_last": e_last,
            "n": len(xs)}


def shrink_and_predict(fit, pooled_slope, target_h, t0, p_target):
    """Shrink a trophy's earner slope toward the pool and predict its earners."""
    x_t = math.log(1.0 + (target_h - t0))
    info = min(fit["n"], 12) * (fit["e_last"] + 1.0) / (fit["e_last"] + 25.0)
    lam = info / (info + 5.0)
    b = lam * fit["b"] + (1.0 - lam) * pooled_slope
    pred = math.exp(fit["my"] + b * (x_t - fit["mx"])) - 0.5
    pred = max(pred, fit["e_last"])            # earners do not un-earn
    pred = min(pred, 0.999 * p_target)         # and cannot outnumber players
    slope_gap = abs(fit["b"] - pooled_slope) * max(x_t - fit["mx"], 0.0)
    sig = math.sqrt(fit["sig"] ** 2 + (0.5 * slope_gap) ** 2)
    return pred, sig, b


LAG_GRID_H = (0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
LAG_WEIGHT = 0.55   # weight of the lag-ratio forecast in the earners blend


def forecast_earners_lag(states, i, predict_logp, target_h):
    """Earners as a lagged shadow of the player curve.

    A trophy's earners at time t are roughly a fixed share u of the players who
    had arrived one maturation lag earlier: E(t) ~ u * P(t - lag). The lag that
    makes that share most stable over the recent states is the trophy's; the
    share itself drifts as cohorts mature, so it is extrapolated with a damped
    trend. Returns (earners_at_target, sigma_log) or None.
    """
    series = [(st["t"], st["rates"][i] * st["players"] / 100.0)
              for st in states
              if i < len(st["rates"]) and st["rates"][i] is not None
              and st["players"] > 0]
    if len(series) < 3:
        return None
    pts = [(st["t"], st["players"]) for st in states if st["players"] > 0]
    observed = logp_interp(pts)
    recent = series[-8:]
    t_last = series[-1][0]
    best = None
    for lag in LAG_GRID_H:
        logs, ws = [], []
        for t, e in recent:
            rho = e / math.exp(observed(t - lag))
            if rho <= 0:
                continue
            logs.append(math.log(rho))
            ws.append(math.exp(-(t_last - t) / EARNER_TAU_H))
        if len(logs) < 3:
            continue
        wsum = sum(ws)
        mean = sum(w * v for w, v in zip(ws, logs)) / wsum
        var = sum(w * (v - mean) ** 2 for w, v in zip(ws, logs)) / wsum
        if best is None or var < best[0]:
            best = (var, lag, logs, ws)
    if best is None:
        return None
    _, lag, logs, ws = best
    ts = [t for t, _ in recent][-len(logs):]
    b, mx, my, _, _sxx2 = wls(ts, logs, ws)
    drift = clamp(0.5 * b * (target_h - mx), -0.7, 0.7)
    log_u = clamp(my + drift, math.log(1e-6), math.log(0.98))
    # The share should not collapse below its recent level nor explode past it.
    log_u = clamp(log_u, my - 0.4, my + 1.4)
    e_pred = math.exp(log_u) * math.exp(predict_logp(target_h - lag))
    sig = math.sqrt(best[0]) + 0.30
    return e_pred, sig


# --------------------------------------------------------------------------
# The unrefreshed-page branch (launch night: history shows one flat page)
# --------------------------------------------------------------------------

def forecast_flat(case, states, at_h, target_h):
    """The history never changed: Exophase has not refreshed inside it.

    A refresh is owed, and on launch night it multiplied the tracked players about
    sixfold while earners lagged (the seed cohort kept earning; the flood of new
    players held almost nothing), so rates fall hard. The one trophy every tracked
    player holds stays at the top. Spreads are wide: the refresh's timing and size
    are both unknown.
    """
    st = states[-1]
    players = st["players"] if st["players"] > 0 else 1.0
    age_h = max(at_h, 0.0)
    if age_h <= 24.0:
        g_players, sig_p = 6.0, 0.75
    else:
        g_players, sig_p = 2.2, 0.60
    g_earners, sig_e = 1.35, 0.60
    horizon_days = max(case["horizon_minutes"], 1) / 1440.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = st["rates"][i] if i < len(st["rates"]) else None
        if rate is None:
            rate = 0.05
        if rate >= 99.99:
            med, sig = 99.90, 0.35
        else:
            earners = rate * players / 100.0
            med = 100.0 * (earners * g_earners) / (players * g_players)
            med = clamp(med, 0.01, 99.9)
            sig_log = math.sqrt(sig_p ** 2 + sig_e ** 2)
            sig = math.sqrt((sig_log / max(1.0 - med / 100.0, 0.35)) ** 2
                            + (DIFFUSION_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                            + SPREAD_FLOOR ** 2)
            sig = clamp(sig, 0.85, SPREAD_CAP)
        rows.append(quantile_row(trophy["id"], med, sig))
    return {"trophies": rows}


def quantile_row(tid, median_pct, sigma_logit):
    med = clamp(median_pct, 0.01, 99.99)
    centre = logit_pct(med)
    qs = []
    prev = 0.0
    for z in Z:
        q = inv_logit_pct(centre + z * sigma_logit)
        q = clamp(round4(q), 0.001, 99.999)
        q = max(q, prev)
        qs.append(q)
        prev = q
    return {"id": tid, "q": qs}


# --------------------------------------------------------------------------
# The main forecast
# --------------------------------------------------------------------------

def decomp_forecast(states, trophies, target_h):
    """The earners/players decomposition: per trophy (median rate, logit sigma)
    or None where the trophy never appears in the history."""
    t0 = states[0]["t"]
    predict_logp, sig_p = forecast_players(states, target_h)
    p_target = math.exp(predict_logp(target_h))

    fits = [forecast_earners(states, i, t0, target_h, p_target)
            for i in range(len(trophies))]

    def pooled(idxs):
        num = den = 0.0
        for j in idxs:
            f = fits[j]
            if f is not None:
                w = f["e_last"] + 2.0
                num += w * f["b"]
                den += w
        return num / den if den > 0 else 0.0

    all_idx = list(range(len(trophies)))
    slope_all = pooled(all_idx)
    tiers = {}
    for j, tr in enumerate(trophies):
        tiers.setdefault(tr.get("tier", "?"), []).append(j)
    slope_tier = {k: (pooled(v) if len(v) >= 3 else slope_all)
                  for k, v in tiers.items()}

    sig_num = sig_den = 0.0
    for f in fits:
        if f is not None:
            w = f["e_last"] + 2.0
            sig_num += w * f["sig"]
            sig_den += w
    sig_pool = sig_num / sig_den if sig_den > 0 else 0.30

    out = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        if f is None:
            out.append(None)
            continue
        pool_slope = slope_tier.get(trophy.get("tier", "?"), slope_all)
        e_pow, sig_pow, _ = shrink_and_predict(f, pool_slope, target_h, t0, p_target)
        lag = forecast_earners_lag(states, i, predict_logp, target_h)
        if lag is not None and lag[0] > 0:
            e_lag, sig_lag = lag
            le_pow = math.log(max(e_pow, 0.05))
            le_lag = math.log(max(e_lag, 0.05))
            e_pred = math.exp(LAG_WEIGHT * le_lag + (1.0 - LAG_WEIGHT) * le_pow)
            sig_e = math.sqrt(LAG_WEIGHT * sig_lag ** 2
                              + (1.0 - LAG_WEIGHT) * sig_pow ** 2
                              + LAG_WEIGHT * (1.0 - LAG_WEIGHT) * (le_lag - le_pow) ** 2)
            e_pred = clamp(e_pred, f["e_last"], 0.999 * p_target)
        else:
            e_pred, sig_e = e_pow, sig_pow
        med = clamp(100.0 * max(e_pred, 0.35) / p_target, 0.01, 99.95)
        sig_log = math.sqrt(sig_e ** 2 + sig_p ** 2
                            + (0.5 * (f["sig"] + sig_pool)) ** 2 * 0.25)
        out.append((med, sig_log / max(1.0 - med / 100.0, 0.35)))
    return out


TREND_TAU_H = 30.0    # recency half-life of the rate-trend fit, hours
TREND_MAX_MOVE = 3.2  # the trend may move a rate at most this far on the logit scale


def trend_forecast(states, trophies, target_h):
    """A direct trend model: per trophy, a recency-weighted line of logit(rate)
    on ln(1 + hours since release), its slope shrunk toward the pooled slope of
    its tier, the move from the last observed logit capped. Per trophy
    (median rate, logit sigma) or None."""
    t_last = states[-1]["t"]
    x_t = math.log(1.0 + max(target_h, 0.0))
    fits = []
    for i in range(len(trophies)):
        fit = None
        # A rate's trend is a local thing: fit the last day of states, widening
        # the window only where a trophy's recent record is too thin to fit.
        for window in (24.0, 48.0, None):
            xs, ys, ws = [], [], []
            e_last = 0.0
            for st in states:
                if window is not None and st["t"] < t_last - window:
                    continue
                if i < len(st["rates"]) and st["rates"][i] is not None:
                    xs.append(math.log(1.0 + max(st["t"], 0.0)))
                    ys.append(logit_pct(st["rates"][i]))
                    ws.append(math.exp(-(t_last - st["t"]) / TREND_TAU_H))
                    if st["players"] > 0:
                        e_last = st["rates"][i] * st["players"] / 100.0
            if len(xs) >= 3:
                b, mx, my, sig, sxx = wls(xs, ys, ws)
                fit = {"b": b, "mx": mx, "my": my, "sig": sig, "sxx": sxx,
                       "n": len(xs), "e_last": e_last, "last_logit": ys[-1]}
                break
        fits.append(fit)

    def pooled(idxs):
        num = den = 0.0
        for j in idxs:
            f = fits[j]
            if f is not None:
                w = f["e_last"] + 2.0
                num += w * f["b"]
                den += w
        return num / den if den > 0 else 0.0

    all_idx = list(range(len(trophies)))
    slope_all = pooled(all_idx)
    tiers = {}
    for j, tr in enumerate(trophies):
        tiers.setdefault(tr.get("tier", "?"), []).append(j)
    slope_tier = {k: (pooled(v) if len(v) >= 3 else slope_all)
                  for k, v in tiers.items()}

    out = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        if f is None:
            out.append(None)
            continue
        pool_slope = slope_tier.get(trophy.get("tier", "?"), slope_all)
        info = min(f["n"], 12) * (f["e_last"] + 1.0) / (f["e_last"] + 25.0)
        lam = info / (info + 5.0)
        b = lam * f["b"] + (1.0 - lam) * pool_slope
        pred = f["my"] + b * (x_t - f["mx"])
        pred = clamp(pred, f["last_logit"] - TREND_MAX_MOVE,
                     f["last_logit"] + TREND_MAX_MOVE)
        dx = max(x_t - f["mx"], 0.0)
        se_b = f["sig"] / math.sqrt(f["sxx"]) if f["sxx"] > 1e-9 else 0.0
        sig = math.sqrt(f["sig"] ** 2 + (se_b * dx) ** 2
                        + (0.35 * abs(f["b"] - pool_slope) * dx) ** 2)
        out.append((inv_logit_pct(pred), sig))
    return out


def backtest_weights(states, trophies):
    """Fit both models on the first ~70% of states and score their medians on
    the rest; the model that tracked this case's own dynamics better earns the
    larger blend weight at the target. Returns (w_trend, w_decomp)."""
    if len(states) < 7:
        return 0.55, 0.45
    split = max(3, int(round(0.7 * len(states))))
    head, tail = states[:split], states[split:]
    errs = {}
    for name, fn in (("trend", trend_forecast), ("dec", decomp_forecast)):
        tot = cnt = 0.0
        for st in tail:
            preds = fn(head, trophies, st["t"])
            for i in range(len(trophies)):
                if preds[i] is None or i >= len(st["rates"]) or st["rates"][i] is None:
                    continue
                tot += abs(logit_pct(preds[i][0]) - logit_pct(st["rates"][i]))
                cnt += 1
        errs[name] = tot / cnt if cnt else None
    if errs["trend"] is None or errs["dec"] is None:
        return 0.55, 0.45
    w_t = 1.0 / (errs["trend"] + 0.15) ** 2
    w_d = 1.0 / (errs["dec"] + 0.15) ** 2
    w_t = clamp(w_t / (w_t + w_d), 0.15, 0.85)
    return w_t, 1.0 - w_t


def forecast_case(case):
    states, at_h, target_h = read_states(case)
    trophies = case["trophies"]
    if not states:
        states = [{"t": at_h, "players": 0.0, "rates": [None] * len(trophies)}]
    horizon_days = max(case["horizon_minutes"], 1) / 1440.0

    # One flat page: the refresh has not happened yet in view.
    if len(states) == 1:
        return forecast_flat(case, states, at_h, target_h)

    dec = decomp_forecast(states, trophies, target_h)
    trd = trend_forecast(states, trophies, target_h)
    w_trend, w_dec = backtest_weights(states, trophies)

    rows = []
    for i, trophy in enumerate(trophies):
        rate_last = last_rate(states, i)
        # Universal trophies (every tracked player holds it in every state) stay.
        universal = all(
            i < len(st["rates"]) and st["rates"][i] is not None
            and st["rates"][i] >= 99.99 for st in states)
        if universal:
            rows.append(quantile_row(trophy["id"], 99.90, 0.35))
            continue
        parts = []
        if dec[i] is not None:
            parts.append((w_dec, dec[i]))
        if trd[i] is not None:
            parts.append((w_trend, trd[i]))
        if not parts:
            med = rate_last if rate_last is not None else 0.05
            rows.append(quantile_row(trophy["id"], med, 1.20))
            continue
        wsum = sum(w for w, _ in parts)
        med_logit = sum(w * logit_pct(m) for w, (m, _) in parts) / wsum
        var = sum(w * (s ** 2 + (logit_pct(m) - med_logit) ** 2)
                  for w, (m, s) in parts) / wsum
        sig = SIGMA_SCALE * math.sqrt(var
                        + (DIFFUSION_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                        + SPREAD_FLOOR ** 2)
        med = inv_logit_pct(med_logit)
        if rate_last is not None and rate_last >= 99.99:
            med = max(med, 99.50)
            sig = min(sig, 0.50)
        rows.append(quantile_row(trophy["id"], med, clamp(sig, 0.30, SPREAD_CAP)))
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case.get('levels')}, not {LEVELS}")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
