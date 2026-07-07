# SPEC — World Cup 2026 model

## Goal

Predict the result of each remaining World Cup 2026 knockout fixture and publish the
predictions on a static site, alongside a running track record of how the model's past
predictions did against actual results. The site re-runs itself before every matchday.

- **Public research / analytics project — not betting advice.** No wagering guidance; no
  edge claimed. It is a small single-tournament sample: a live demo and calibration
  check, not a backtested trading system.
- **No betting odds.** The model is trained purely on match results and international
  ratings. There is no market/odds comparison anywhere in this project.
- **Free tools only.**

## Data

Two free, public-domain (CC0) GitHub sources, mirrored into `data/raw/` by `wc_fetch.py`
and validated by `wc_validate.py`:

1. **FIFA-World-Cup-2026-Dataset** (`data/raw/wc2026/`, updated daily) — relational
   current-tournament CSVs. `matches.csv` (104 = full schedule; Completed vs Scheduled),
   `teams.csv` (carries `elo_rating` and `fifa_ranking_pre_tournament`), `match_events`
   (used to rebuild 90-minute scores), plus venues/referees/stats/lineups adjuncts.
2. **martj42/international_results** (`data/raw/intl/results.csv`) — every international
   match 1872-present, with `tournament` type and a `neutral` flag; the basis for the
   rolling international Elo and the coefficient fit.

## Model

A **Dixon-Coles Poisson goal-rate model** (`src/wc_dixon_coles.py`). Home/away goals are
Poisson with the standard Dixon-Coles low-score (`ρ`) correction. Team attack/defence are
**tied to an international-Elo strength z-score**, not fit from this tournament's small
sample; only five global coefficients are estimated: `μ` (baseline), `γ` (home),
`a`/`d` (strength→scoring/conceding slopes), `ρ`.

- **Strength** (`src/wc_intl_elo.py`): a rolling Elo over the full international history
  (importance-weighted by match type, neutral-aware). It agrees with the CC0 dataset's
  own `elo_rating` at r ≈ 0.91. Each World Cup team's strength is its rating as of the
  tournament's opening date (pre-tournament — no lookahead).
- **Coefficients** (`src/wc_global.py`): fit once on ~25k internationals from 2000
  onward, weighted by importance × recency, using **zero** World Cup 2026 matches.
- **Regulation scores** (`src/wc_data.py`): a 90-minute Poisson model needs 90-minute
  results, so knockout matches decided in extra time are reduced to their score at 90'
  (rebuilt from `match_events`, goals with base minute ≤ 90).
- **Progression:** a level score at 90' → extra time (rates scaled to 30 min, same `ρ`)
  → penalties (50/50); each knockout fixture reports a per-team progression probability
  summing to 1.
- **Neutral venues:** knockout ties are predicted neutral; host-nation group-stage home
  advantage is not modelled.

### Validation

- `src/wc_backtest.py` — walk-forward, strictly chronological (refit on earlier-dated
  matches only): ~74 predictions, log loss 0.837 vs 1.121 base-rate, accuracy 0.689.
- `src/wc_global.py` — the all-history fit; on the same held-out matches it is within
  noise of the in-tournament fit (log loss 0.861 vs 0.837) but gives more credible,
  better-calibrated coefficients and needs no in-tournament warm-up, so it is the model
  the site uses.
- Unit tests: `tests/test_wc_dixon_coles.py`, `tests/test_wc_intl.py`.

## Publish

- `src/build_results_log.py` → `data/processed/results_log.csv`: one row per fixture with
  known teams — W/D/L probabilities, the confident pick, knockout progression, and (for
  completed matches) the actual 90' result and whether the pick was right. Regenerated
  from source each run (idempotent).
- `src/build_site.py` → `site/index.html`: a self-contained, theme-aware static page —
  confident predictions for upcoming fixtures + a track-record scoreboard. No backend, no
  external assets.
- `.github/workflows/update.yml`: every 3 hours (and on demand) re-pull data, rebuild,
  and deploy to GitHub Pages.

## Not yet done / possible next steps

- Roll the Elo through completed World Cup matches so late-round strengths reflect
  in-tournament form (currently static at tournament start).
- The 6 later-round fixtures (SFs, 3rd-place, Final) predict once their brackets fill.
- Corners/cards models — data is present (`match_team_stats`, `match_events`,
  `referees.avg_cards_per_game`) but out of scope for now.

## Provenance

This model was extracted into its own repository from a shared working directory; it is
independent of, and shares no data with, the author's separate English domestic-league
study.
