# Day Curve — GTA VI launch-day Twitch viewers, hour by hour

![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sherrypubg128-blip/gta6-twitch-ccu-forecast-01/rank.svg)
![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sherrypubg128-blip/gta6-twitch-ccu-forecast-01/reward.svg)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), after the first hour's print, as
five quantiles per hour. Built for
[make.fun's GTA VI Twitch CCU forecast bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## Approach

A launch day's shape is set by who can play when. GTA VI unlocks at
midnight local time, country by country, so hour 1 holds New Zealand,
Australia, Asia and Europe with the UK just starting, while the US only
unlocks at 05:00Z (East) and 08:00Z (West). Viewership should climb through
the day and peak in the evening UTC — and past big launches show exactly
this shape, but only the ones that unlocked the same way.

`main.ts` learns the day curve from the case's own comparables (38 past big
launches in a practice case, 42 in the final), weighting each launch by how
much its unlock resembles GTA VI's (console-only and rolling-midnight count
more; early-access, simultaneous global unlocks and free-to-play count less),
and adapts to the target with the hour-1 channel count: across past
launches, the log of hour-1 channels strongly predicts how much a launch
day grows, because a streamer base already saturated at hour 1 has little
headroom left. For each hour the forecast is log-normal around hour 1's
print, with the centre blending the similarity-weighted median log ratio and
a weighted regression of the log ratio on log hour-1 channels, and a robust
spread (1.4826 × the weighted median absolute deviation of the log ratios),
smoothed across hours.

Leave-one-out cross-validation of this exact program on the 37 complete
comparable launches: mean pinball loss 0.63 over all 37, 0.26 over launches
with a real hour-1 audience (≥ 20k viewers), 0.22 over big launches (≥ 50k)
and 0.19 over console-only launches — against the house baseline's 0.93,
0.32, 0.25 and 0.32 on the same targets. 90% coverage 0.91–0.97 on the
launch populations that resemble GTA VI.

Deterministic: no clock, no randomness, no network. Node 22, standard
library only. Run locally with:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```
