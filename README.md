# World Cup 2026 — Model Predictions

**A per-team Dixon-Coles Poisson goal model — each team fit with its own attack and
defence rating on international match history — predicts the remaining World Cup 2026
knockout results**, and keeps a public track record against what actually happens. It
re-runs itself before every matchday — pulling fresh data, re-predicting the upcoming
fixtures, and updating the scoreboard automatically.

🔗 **Live site:** <https://anishkhetani.github.io/worldcup-2026-model/>

> **Public research / analytics project — not betting advice.** These are model
> probabilities published for interest and to hold the model honest against real
> results. No wagering guidance is given or implied, and no edge is claimed. This is a
> small single-tournament sample: a live demo and calibration check, not a backtested
> trading system.

Method summary below; full detail in [`SPEC.md`](SPEC.md).

## The model

A **per-team Dixon-Coles Poisson goal model**. Each team gets its own **attack** and
**defence** rating, fit by weighted maximum-likelihood on ~17k international matches, with
a two-sided home-advantage term and the standard Dixon-Coles low-scoring-draw (`ρ`)
correction:

```text
log(λ_home) = μ + home_adv·(non-neutral) + attack[home] − defence[away]
log(λ_away) = μ − home_def·(non-neutral) + attack[away] − defence[home]
```

- **Per-team attack/defence** (not a single strength number) distinguishes a
  high-scoring-but-leaky side from a defensive one of equal net strength.
- **Weighted training**: each match is weighted by recency (2-year half-life) × tournament
  importance (live World Cup 1.5, other majors 1.0, qualifiers/Nations League 0.6,
  friendlies 0.25).
- **Two-sided home advantage**: home teams both score more *and* concede less; both halves
  are jointly estimated. World Cup ties are predicted at neutral venues (only a host
  nation at home gets the bump).
- **Knockout progression** (extra time / penalties) uses a favorite-conversion
  **empirically calibrated on 92 years of World Cup knockout history** — favorites advance
  ~62% of drawn ties, not a 50/50 shootout, but far below their regulation edge.
- **No lookahead**: every completed match is predicted walk-forward, refit on only
  earlier-dated internationals.

## Track record (walk-forward, no lookahead)

On the matches already played, comparing each pre-match pick to the actual 90-minute
result:

| | correct | log loss | base-rate log loss | Brier |
|---|---:|---:|---:|---:|
| model | **61 / 94 (65%)** | **0.854** | 1.065 | 0.503 |

"Log loss" rewards being confident *and* right; the base-rate line is a naive predictor
(the completed-match H/D/A frequencies) for comparison. Numbers refresh automatically as
results land — see the live site for the current figure and the full match-by-match table.

## Automation

[`.github/workflows/update.yml`](.github/workflows/update.yml) runs once daily at 07:00
UTC (and on demand): re-pull the data, validate it, rebuild the predictions and site, and
deploy to GitHub Pages — so the live site refreshes after each matchday's results land.

## Run locally

```bash
pip install -r requirements.txt
cd src
python wc_fetch.py       # mirror the three CC0 / public-domain datasets -> data/raw/
python wc_validate.py    # validate the data
python wc_predict.py     # per-team Dixon-Coles predictions for the remaining fixtures
python wc_backtest.py    # walk-forward backtest (calibration + log loss)
python build_site.py     # regenerate results_log.csv + the static site/ (what CI deploys)
python -m pytest ../tests -q
```

## Data & licensing

Three free, public-domain sources — all GitHub. **No betting-odds source is used**; the
model is trained purely on match results.

- **[FIFA-World-Cup-2026-Dataset](https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset)**
  (current-tournament results, fixtures, venues, referees) — **CC0-1.0**, updated daily.
- **[martj42/international_results](https://github.com/martj42/international_results)**
  (all international matches 1872-present, the model's training pool) — **CC0**.
- **[jfjelstul/worldcup](https://github.com/jfjelstul/worldcup)** (Fjelstul World Cup
  Database — knockout history used to calibrate the extra-time/penalty conversion) —
  **CC-BY-SA 4.0**.

Public research project. Not affiliated with FIFA. Not betting advice.
