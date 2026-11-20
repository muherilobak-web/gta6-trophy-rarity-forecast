# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/muherilobak-web/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/muherilobak-web/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for each of GTA VI's 55 PlayStation trophies, the percent of the
players Exophase tracks who will hold it at the case's target (2026-11-26
00:00 UTC for the final), as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95).

## The idea

A trophy's earn rate is a ratio E/N — and on launch week it is **N**, the
tracked-player pool, that moves the wildest. Exophase refreshes its counts
every few hours, and at each refresh the pool jumps (launch night: 638 →
3,868 players in one refresh) while the newcomers hold far fewer trophies
than the players already tracked, so most rates **fall** at a refresh.
Fitting a curve to the rate alone misses half the process.

So `main.py` forecasts **E (earners) and N (players tracked) separately and
divides**:

1. **Epochs.** History is split into epochs: consecutive reads with the same
   `players_tracked` are one epoch, because N moves only at a refresh. The
   rates drift continuously, so they must not key the epochs. Each epoch
   keeps its first read (just after the refresh that opened it) and its last
   read (freshest state before the next refresh) — the pair brackets a
   refresh with at most one read interval of progression in between.
2. **N(t).** Log-linear fit of log N on ln(1 + hours since release). A power
   law, so its growth rate declines as the launch surge saturates; its
   ratios give the future refresh multipliers (holding the observed median
   multiplier constant would compound it for every refresh to the target
   and over-dilute wildly).
3. **Progression.** Within an epoch N is flat, so a trophy's logit moves
   only by progression. The slope of that move, pooled over epochs and
   shrunk toward its (tier × content) group mean, is the progression rate;
   a fit across epoch starts would mix it with the refresh crashes and
   bias it toward zero. The level is anchored at the last read.
4. **Refreshes.** The median gap between epoch starts is the refresh
   interval (default 4 h when the history is flat). The expected refreshes
   between the checkpoint and the target are counted; each multiplies N by
   the smooth-fit ratio (prior 6.0 for the first refresh of a flat history
   — the first refresh of this launch, as the kit documents it — and 2.0
   after) and adds newcomers who earn each trophy at a fraction *f* of the
   current rate. *f* is estimated per content group from the refresh
   transitions the history itself shows, and falls back to a prior per
   group when no refresh has been observed yet.
5. **Spread.** The median blends the direct fit with the E/N simulation
   (weight to E/N for rare trophies, to the direct fit for common ones).
   The spread adds a random walk, the fit residual, the shrinkage
   uncertainty, and a staleness term from the next refresh's expected jump
   — so the bands are honest about the refreshes, not just the fit.

The 100% trophy ("Just Like Paradise") is pinned near 100%.

## Run it

```bash
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output \
  python3 -P -s main.py
```

Python 3.13, standard library only, no network. Deterministic: two runs
write the same bytes. ~0.1 s per case (cap 8 s).

## Local checks

Practice cases (p00/p01) run clean: 55 trophies, ids 1–55, 5 quantiles each,
non-decreasing, 2,790 bytes per output. On a refresh simulation built from
the kit's documented first refresh (638 → 3,868 players, earned 943 →
4,276), the model scores mean pinball loss **0.29–0.46** against the
persistence baseline's 0.78–1.69 across the plausible scenarios, with 0.98
interval coverage (baseline: 0.00). The real score comes from the server
dev grade after submit — the kit ships no practice answers, so local
scoring is impossible.

## Referral

Built on [make.fun](https://make.fun/r/bt57jd3y) — referral code `bt57jd3y`.

## License

MIT — see [LICENSE](LICENSE).
