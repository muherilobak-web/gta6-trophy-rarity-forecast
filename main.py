"""gta6-trophy-rarity-forecast-01: hierarchical partial-pooling forecaster.

Forecasts, for each of GTA VI's 55 PlayStation trophies, the percent of the
players Exophase tracks who will hold it at the case's target, as five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95). Python 3.13, standard library only,
no network, deterministic: two runs write the same bytes.

The idea (Concept B): a trophy's earn rate is a ratio E/N, and on launch week
it is N, the tracked-player pool, that moves the wildest. Exophase refreshes
its counts every few hours, and at each refresh the pool jumps (launch night:
638 -> 3,868 players in one refresh) while the newcomers hold far fewer
trophies than the players already tracked, so most rates FALL at a refresh.
Fitting a curve to the rate alone misses half the process. This program
therefore forecasts E (earners) and N (players tracked) separately and
divides, and treats the answer as the last refresh before the target, not
the target itself:

  1. History is split into epochs: consecutive reads with the same
     players_tracked are one epoch, because N moves only at a refresh. The
     rates drift continuously, so they must NOT key the epochs -- keying on
     them would make every read its own epoch and collapse the refresh
     interval to the read cadence. Each epoch keeps its first read (the
     state just after the refresh that opened it) and its last read (the
     freshest state before the next refresh): the pair brackets a refresh
     with at most one read interval of progression in between.
  2. N(t): log-linear fit of log N on ln(1 + hours since release) over the
     epoch representatives. A power law, so its growth rate declines as the
     launch surge saturates; its ratios give the future refresh multipliers
     (holding the observed median multiplier constant would compound it for
     every refresh to the target and over-dilute wildly).
  3. Progression: within an epoch N is flat, so a trophy's logit moves only
     by progression. The slope of that move, pooled over epochs and shrunk
     toward its (tier x content) group mean, is the progression rate; a fit
     across epoch starts would mix it with the refresh crashes and bias it
     toward zero. The level is anchored at the last read.
  4. Refreshes: the median gap between epoch starts is the refresh interval
     (default 4 h when the history is flat); the expected refreshes between
     the checkpoint and the target are counted, each multiplying N by the
     smooth-fit ratio (prior 6.0 for the first refresh of a flat history --
     the first refresh of this launch, as the kit documents it -- and 2.0
     after) and adding newcomers who earn each trophy at a fraction f of the
     current rate. f is estimated per content group from the refresh
     transitions the history itself shows, bracketing each refresh with the
     previous epoch's last read and the next epoch's first read, and falls
     back to a prior per group when no refresh has been observed yet.
  4. Progression: per trophy, a least-squares line of logit(rate) on
     ln(1 + hours) over the epoch representatives. Slopes are shrunk toward
     their content-group mean with empirical-Bayes weights from the fit's
     standard error (a flat history carries no slope information, so it
     shrinks hard; a trophy with a long, moving history barely shrinks).
     Intercepts are NOT shrunk: a trophy's level is directly observed, and
     shrinking it would drag a 100% trophy toward its group's ~1%.
  5. Median: the E/N decomposition and the direct logit fit, blended in
     logit space, weighted toward the decomposition for rare trophies (where
     the ratio is noisy and dilution dominates) and toward the direct fit
     for common ones. A trophy at ~100% stays ~100%: everyone tracked holds
     it, so a refresh cannot move it.
  6. Spread: random walk + fit residual + shrinkage variance + staleness
     variance (the uncertainty in how many refreshes land before the target
     and how big they are) + a floor that never lets a band get narrower
     than ~0.7 logit for a trophy under 5% -- one refresh can move a rare
     trophy's logit by more than a unit.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim,
standard library only. Outside the grader's container, MAKEFUN_INPUT_DIR
and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Prior refresh interval (minutes) when the history shows a single epoch:
# Exophase refreshed this game's counts about every 4 hours on launch night.
DEFAULT_REFRESH_MIN = 240.0
# Prior N multiplier of the first refresh of a flat history (the kit
# documents the first refresh of this launch: 638 -> 3,868 players, x6.06),
# and of later refreshes when no transition has been observed yet.
PRIOR_FIRST_MULT = 6.0
PRIOR_LATER_MULT = 2.0
# Prior newcomer earn fraction per content group (share of the current rate
# at which players who join at a refresh hold the trophy). Calibrated from
# the totals the kit documents for this launch's first refresh: players
# tracked 638 -> 3,868, trophies earned 943 -> 4,276. Nearly all of the
# newcomers' 3,333 trophies are the one at 100% (Gone Fishing, the first
# activity); outside it the newcomers earned ~0.03-0.13 trophies per player
# against the first cohort's 0.48, so f is small everywhere and near zero
# for late game, grind and 100% completion. Early story is reachable in a
# few hours, so it keeps the highest f of the set.
PRIOR_F = {
    "chapter": 0.30,
    "mission": 0.15,
    "final": 0.05,
    "completion": 0.01,
    "stunt": 0.10,
    "collection": 0.05,
    "crime": 0.08,
    "money": 0.05,
    "misc": 0.08,
}


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as the starter rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t: datetime) -> float:
    return (t - RELEASE).total_seconds() / 3600.0


def ols(xs, ys):
    """Least squares line y = a + b*x. Returns (a, b, resid_var, se_b, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    if n == 1:
        return ys[0], 0.0, 0.0, 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    dof = n - 2
    resid_var = rss / dof if dof > 0 else 0.0
    # A perfect fit over a flat history carries no information about the
    # slope: floor the residual sd at 0.5 logit so the slope standard error
    # stays honest and the shrinkage below can act on it.
    resid_sd = max(math.sqrt(resid_var), 0.5)
    se_b = resid_sd / math.sqrt(sxx)
    return a, b, resid_var, se_b, n


def content_group(description: str) -> str:
    """Content type of a trophy from its official description."""
    d = description.strip().lower()
    if d.startswith("complete chapter"):
        return "chapter"
    if d == "complete the final mission.":
        return "final"
    if "attain 100% completion" in d:
        return "completion"
    if "collectible card" in d or "waink" in d or "meditate" in d \
            or "unique animal" in d or "explore all" in d \
            or "garage properties" in d or "discover 100 vehicles" in d \
            or "register 10 vehicles" in d:
        return "collection"
    if "stunt" in d or "shooting range" in d or "drone race" in d \
            or "races" in d or "fishing challenge" in d \
            or "arcade game" in d or "challenges" in d \
            or "combat focus" in d or "drive-by" in d \
            or "vcia" in d or "rhinos" in d:
        return "stunt"
    if d.startswith("rob ") or "wanted level" in d or "showroom" in d:
        return "crime"
    if "laundering" in d or "spend $" in d:
        return "money"
    if d.startswith("complete"):
        return "mission"
    return "misc"


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_min = float(case["horizon_minutes"])
    target = parse_time(case["target"])
    at = parse_time(case["at"])

    # --- epochs: consecutive reads with the same players_tracked ---
    # N moves only at a refresh (that is what an epoch is); the rates drift
    # continuously as players earn, so they must NOT key the epochs -- keying
    # on them would make every read its own epoch and collapse the refresh
    # interval to the read cadence. Each epoch keeps its first read (the
    # state just after the refresh that opened it) and its last read (the
    # freshest state before the next refresh): the pair brackets a refresh
    # with at most one read interval of progression in between, which is
    # what the newcomer-fraction estimate below needs.
    epochs = []  # each: {"start", "end", "n", "rates_first", "rates"}
    for read in history:
        rates = read["rates"]
        if epochs and epochs[-1]["n"] == read["players_tracked"]:
            epochs[-1]["end"] = parse_time(read["at"])
            epochs[-1]["count"] += 1
            epochs[-1]["rates"] = rates
        else:
            epochs.append({
                "start": parse_time(read["at"]),
                "end": parse_time(read["at"]),
                "count": 1,
                "n": read["players_tracked"],
                "rates_first": rates,
                "rates": rates,
            })
    reps = epochs  # one representative read per epoch: its last read

    # --- refresh interval and multiplier, from observed transitions ---
    gaps = []
    mults = []
    for prev, nxt in zip(epochs, epochs[1:]):
        gaps.append((nxt["start"] - prev["start"]).total_seconds() / 60.0)
        if prev["n"] > 0:
            mults.append(nxt["n"] / prev["n"])
    interval = sorted(gaps)[len(gaps) // 2] if gaps else DEFAULT_REFRESH_MIN
    interval = max(30.0, min(1440.0, interval))
    mult_obs = sorted(mults)[len(mults) // 2] if mults else None

    # --- newcomer earn fraction f per content group, from transitions ---
    # f_i = (E_new - E_old) / ((N_new - N_old) * rate_old): the share of the
    # old rate at which the newcomers hold the trophy. The two reads bracket
    # the refresh (prev epoch's last read, next epoch's first read), so the
    # existing players' progression between them is at most one read
    # interval and cannot inflate f.
    f_obs = {}  # group -> list of f values
    for prev, nxt in zip(epochs, epochs[1:]):
        dn = nxt["n"] - prev["n"]
        if dn <= 0:
            continue
        for i, trophy in enumerate(trophies):
            r_old = prev["rates"][i]
            r_new = nxt["rates_first"][i]
            if r_old is None or r_new is None or r_old <= 0:
                continue
            e_old = r_old / 100.0 * prev["n"]
            e_new = r_new / 100.0 * nxt["n"]
            f = (e_new - e_old) / (dn * r_old / 100.0)
            grp = content_group(trophy["description"])
            f_obs.setdefault(grp, []).append(min(1.5, max(0.0, f)))

    def group_f(grp: str) -> float:
        vals = f_obs.get(grp)
        if vals:
            vals = sorted(vals)
            return vals[len(vals) // 2]
        return PRIOR_F.get(grp, 0.15)

    # --- N(t): log-linear fit over epoch representatives ---
    # log N = a_n + b_n * ln(1 + hours): a power law, so its growth rate
    # declines over time -- the launch surge saturates. Its ratios give the
    # future refresh multipliers: holding the observed median multiplier
    # constant would compound it for every refresh to the target and
    # over-dilute wildly (a 1.12x median over 27 refreshes is 21x).
    xs_n = [math.log1p(max(0.0, hours_since_release(e["start"]))) for e in reps]
    ys_n = [math.log(max(1, e["n"])) for e in reps]
    a_n, b_n, _, _, _ = ols(xs_n, ys_n)

    def n_smooth(t: datetime) -> float:
        x = math.log1p(max(0.0, hours_since_release(t)))
        return math.exp(a_n + b_n * x)

    # --- refresh times between the checkpoint and the target ---
    last_epoch = epochs[-1]
    target_ts = target.timestamp()
    next_refresh = last_epoch["start"].timestamp() + interval * 60.0
    at_ts = at.timestamp()
    while next_refresh <= at_ts:
        next_refresh += interval * 60.0
    refresh_times = []
    t_ref = next_refresh
    while t_ref <= target_ts:
        refresh_times.append(t_ref)
        t_ref += interval * 60.0
    n_refresh = len(refresh_times)
    lag_min = interval  # the answer is the last refresh before the target

    # --- per-trophy progression: within-epoch slopes ---
    # Between refreshes N is flat, so a trophy's logit moves only by
    # progression. The slope of that move, pooled over epochs, is the
    # progression rate; a fit across epoch starts would mix it with the
    # refresh crashes and bias it toward zero (the starts are the troughs
    # right after each refresh). The level is anchored at the last read.
    x_last_read = math.log1p(max(0.0, hours_since_release(parse_time(history[-1]["at"]))))
    fits = []
    for i, trophy in enumerate(trophies):
        slopes = []
        for e in epochs:
            r0 = e["rates_first"][i]
            r1 = e["rates"][i]
            if r0 is None or r1 is None:
                continue
            x0 = math.log1p(max(0.0, hours_since_release(e["start"])))
            x1 = math.log1p(max(0.0, hours_since_release(e["end"])))
            if x1 - x0 < 1e-9:
                continue
            slopes.append((logit(r1) - logit(r0)) / (x1 - x0))
        if slopes:
            s_sorted = sorted(slopes)
            b = s_sorted[len(s_sorted) // 2]
            if len(slopes) >= 2:
                mean_s = sum(slopes) / len(slopes)
                var_s = sum((s - mean_s) ** 2 for s in slopes) / len(slopes)
                se_b = max(math.sqrt(var_s) / math.sqrt(len(slopes)), 0.05)
            else:
                se_b = 0.5  # one epoch's slope is weakly identified
        else:
            b, se_b = 0.0, 0.5
        last_rate, last_time = None, None
        for e in reversed(epochs):
            if e["rates"][i] is not None:
                last_rate, last_time = e["rates"][i], e["end"]
                break
        if last_rate is None:
            last_rate, last_time = 1.0, parse_time(history[-1]["at"])
        x_anchor = math.log1p(max(0.0, hours_since_release(last_time)))
        a = logit(last_rate) - b * x_anchor
        # Residual variance: epoch endpoints against the fitted line.
        res = []
        for e in epochs:
            for r, tt in ((e["rates_first"][i], e["start"]), (e["rates"][i], e["end"])):
                if r is None:
                    continue
                x = math.log1p(max(0.0, hours_since_release(tt)))
                res.append((logit(r) - (a + b * x)) ** 2)
        resid_var = sum(res) / len(res) if res else 0.0
        fits.append({
            "a": a, "b": b, "resid_var": resid_var, "se_b": se_b, "n": len(slopes),
            "last": last_rate,
            "group": (trophy["tier"], content_group(trophy["description"])),
        })

    # --- empirical-Bayes shrinkage of slopes toward the group mean ---
    groups = {}
    for i, f in enumerate(fits):
        groups.setdefault(f["group"], []).append(i)
    for grp, idxs in groups.items():
        slopes = [fits[i]["b"] for i in idxs]
        mean_b = sum(slopes) / len(slopes)
        # Group variance of true slopes (method of moments).
        var_obs = sum((s - mean_b) ** 2 for s in slopes) / len(slopes) if slopes else 0.0
        mean_se2 = sum(fits[i]["se_b"] ** 2 for i in idxs) / len(idxs)
        tau2 = max(0.0, var_obs - mean_se2)
        for i in idxs:
            se2 = fits[i]["se_b"] ** 2
            if tau2 <= 0:
                lam = 0.0 if se2 > 0 else 1.0
            else:
                lam = tau2 / (tau2 + se2) if se2 > 0 else 1.0
            # lam = weight on the trophy's own slope; 1-lam on the group mean.
            fits[i]["b_star"] = lam * fits[i]["b"] + (1.0 - lam) * mean_b
            fits[i]["shrink_var"] = (1.0 - lam) * tau2
            fits[i]["lam"] = lam

    # --- horizon for the random-walk term ---
    x_target = math.log1p(max(0.0, hours_since_release(target)))
    days = max(horizon_min, 0.0) / 1440.0

    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        last = f["last"]

        # The ~100% trophy: everyone tracked holds it, so a refresh cannot
        # move it. Keep it pinned near 100 with a narrow band.
        if last >= 99.5:
            med = 99.99
            sd = 0.05
            rows.append({"id": trophy["id"], "q": [
                round4(inv_logit(logit(med) + z * sd)) for z in Z]})
            continue

        # Direct fit median at the target (progression of the players
        # already tracked, no refresh).
        med_direct = inv_logit(f["a"] + f["b_star"] * x_target)

        # E/N decomposition, simulated forward from the checkpoint:
        # between refreshes N is flat and earners progress along the fitted
        # line; at each refresh N jumps and the newcomers join holding the
        # trophy at the group's fraction f of the pre-refresh rate. The
        # jump size comes from the smooth N fit's ratios when the history
        # shows real growth (they decline as the surge saturates); a flat
        # history falls back to the priors (6x first refresh, 2x after).
        n_cur = float(last_epoch["n"])
        e_cur = last / 100.0 * n_cur
        grp = content_group(trophy["description"])
        f_new = group_f(grp)
        r_list = []
        t_prev = last_epoch["start"]
        for j, t_r in enumerate(refresh_times):
            t_dt = datetime.fromtimestamp(t_r, tz=timezone.utc)
            x_r = math.log1p(max(0.0, hours_since_release(t_dt)))
            rate_pre = inv_logit(f["a"] + f["b_star"] * x_r)
            e_cur = rate_pre / 100.0 * n_cur
            if len(epochs) >= 2:
                r_mult = n_smooth(t_dt) / n_smooth(t_prev)
                r_mult = max(1.0, min(3.0, r_mult))
            elif j == 0:
                r_mult = PRIOR_FIRST_MULT
            else:
                r_mult = PRIOR_LATER_MULT
            r_list.append(r_mult)
            t_prev = t_dt
            n_new = n_cur * r_mult
            e_cur = e_cur + f_new * (rate_pre / 100.0) * (n_new - n_cur)
            n_cur = n_new
        # After the last refresh, earners progress to the target.
        med_en = inv_logit(f["a"] + f["b_star"] * x_target) if n_refresh == 0 \
            else 100.0 * e_cur / n_cur
        if n_refresh > 0:
            # Progression of the post-refresh pool up to the target: the
            # pool is fixed, so the rate follows the fitted line from the
            # last refresh to the target.
            t_last_ref = datetime.fromtimestamp(refresh_times[-1], tz=timezone.utc)
            x_lr = math.log1p(max(0.0, hours_since_release(t_last_ref)))
            rate_lr = inv_logit(f["a"] + f["b_star"] * x_lr)
            rate_tgt = inv_logit(f["a"] + f["b_star"] * x_target)
            med_en = med_en * (rate_tgt / rate_lr) if rate_lr > 0 else med_en

        # Blend in logit space: rare trophies lean on the decomposition
        # (dilution dominates, the ratio is noisy), common ones on the
        # direct fit (progression dominates).
        w_en = 0.9 - 0.05 * min(med_direct, 10.0)
        w_en = max(0.4, min(0.9, w_en))
        med_logit = w_en * logit(med_en) + (1.0 - w_en) * logit(med_direct)
        med = inv_logit(med_logit)

        # Spread.
        floor = 0.9 if med < 1.0 else (0.7 if med < 5.0 else (0.5 if med < 25.0 else 0.4))
        # Staleness: each expected refresh moves a rare trophy's logit by
        # about |log((1 + f*(R-1))/R)|; the median already accounts for the
        # expected move, so only half of it counts as uncertainty.
        r_first = r_list[0] if r_list else (mult_obs if mult_obs is not None else PRIOR_FIRST_MULT)
        jump = abs(math.log((1.0 + f_new * (r_first - 1.0)) / r_first)) if r_first > 1 else 0.0
        staleness_var = n_refresh * (0.5 * jump) ** 2
        sd2 = (0.5 * math.sqrt(days)) ** 2 + f["resid_var"] \
            + f["shrink_var"] + staleness_var + floor * floor
        sd = math.sqrt(max(sd2, 0.04))

        qs = []
        for z in Z:
            q = inv_logit(med_logit + z * sd)
            qs.append(round4(min(99.99, max(0.01, q))))
        # Rounding is monotone, but enforce non-decreasing defensively.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
