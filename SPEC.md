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

Three free, public-domain GitHub sources, mirrored into `data/raw/` by `wc_fetch.py` and
validated by `wc_validate.py`:

1. **FIFA-World-Cup-2026-Dataset** (`data/raw/wc2026/`, CC0, updated daily) — relational
   current-tournament CSVs. `matches.csv` (104 = full schedule; Completed vs Scheduled),
   `match_events` (used to rebuild 90-minute scores), plus teams/venues/referees/stats.
2. **martj42/international_results** (`data/raw/intl/results.csv`, CC0) — every
   international match 1872-present, with `tournament` type and a `neutral` flag; the
   model's training pool (`src/wc_train.py`).
3. **jfjelstul/worldcup** (`data/raw/fjelstul/matches.csv`, CC-BY-SA 4.0) — the Fjelstul
   World Cup Database: all knockout ties 1930-2022 with extra-time/penalty flags and who
   advanced; used to calibrate the knockout conversion (`src/wc_knockout.py`).

## Model

A **per-team Dixon-Coles Poisson goal model** (`src/wc_dixon_coles.py`). Unlike the
previous version — which collapsed each team onto a single Elo strength scalar — every
team now gets its own **attack** and **defence** parameter:

```text
log(λ_home) = μ + home_adv·(non-neutral) + attack[home] − defence[away]
log(λ_away) = μ − home_def·(non-neutral) + attack[away] − defence[home]
```

Fit by weighted maximum-likelihood (analytic gradient, L-BFGS-B) with a ridge prior on
attack/defence and the standard Dixon-Coles low-score (`ρ`) correction.

- **Training set** (`src/wc_train.py`): ~17k international matches, each weighted by
  **recency (2-year half-life) × tournament tier** (live World Cup 1.5, other majors 1.0,
  qualifiers / Nations League 0.6, friendlies 0.25). The pool includes the 2026 WC matches
  themselves, so form updates as the tournament plays out.
- **Two-sided home advantage**: `home_adv` (home scores more) + `home_def` (away scores
  less) are jointly estimated on the `neutral` flag; a one-sided spec under-rates home
  sides. World Cup ties are predicted neutral (only a host at home gets the bump, read from
  the data via `fixture_neutral`).
- **Regulation scores** (`src/wc_data.py`): a 90-minute Poisson model needs 90-minute
  results, so extra-time knockout matches are reduced to their score at 90' (rebuilt from
  `match_events`, goals with base minute ≤ 90).
- **Knockout progression** (`src/wc_knockout.py`): P(each team advances) layers extra
  time / penalties on the 90' output using an **empirically-calibrated favorite
  conversion** — `P(fav advances | level at 90') = logistic(γ·logit(q))`, `q` the
  regulation win-share, `γ ≈ 0.472` fit on 92 years of WC knockout history (favorites
  advance ~62% of drawn ties; a flat 50/50 shootout is wrong). Reproduced by
  `fit_from_data()` and pinned; a regression test asserts they still agree.
- **No lookahead**: `build_results_log.py` predicts each completed match walk-forward,
  refitting on only internationals strictly before that matchday.

### Validation

- `src/wc_backtest.py` — the walk-forward predictions: **94 matches, log loss 0.854 vs
  1.065 base-rate, accuracy 0.649, Brier 0.503, ECE ≈ 0.082.** (The prior Elo-scalar model
  scored log loss 0.902 / accuracy 0.628 / ECE ≈ 0.09 — every metric improved.)
- Unit tests: `tests/test_wc_dixon_coles.py` (per-team fit + pure goal math),
  `tests/test_wc_train.py` (weighting, no-lookahead, reconciliation),
  `tests/test_wc_knockout.py` (conversion properties + γ-refit regression).

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

- The 6 later-round fixtures (SFs, 3rd-place, Final) predict once their brackets fill.
- Corners/cards models — data is present (`match_team_stats`, `match_events`,
  `referees.avg_cards_per_game`) but out of scope for now.
- Tournament-wide Monte-Carlo bracket simulation (reach-final / win-tournament odds).

## Provenance

The core Dixon-Coles model, the weighted-training scheme, and the empirical knockout
conversion were adapted from the author's separate football betting-model codebase and
rebuilt here as an odds-free predictor. This repository is independent of, and shares no
data with, the author's separate English domestic-league study.
