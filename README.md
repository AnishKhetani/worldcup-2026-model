# World Cup 2026 — Model Predictions

**A Dixon-Coles Poisson goal model, with team strength from a rolling international Elo
rating, predicts the remaining World Cup 2026 knockout results** — and keeps a public
track record against what actually happens. It re-runs itself before every matchday —
pulling fresh data, re-predicting the upcoming fixtures, and updating the scoreboard
automatically.

🔗 **Live site:** <https://anishkhetani.github.io/worldcup-2026-model/>

> **Public research / analytics project — not betting advice.** These are model
> probabilities published for interest and to hold the model honest against real
> results. No wagering guidance is given or implied, and no edge is claimed. This is a
> small single-tournament sample: a live demo and calibration check, not a backtested
> trading system.

Method summary below; full detail in [`SPEC.md`](SPEC.md).

## The model

A **Dixon-Coles Poisson goal model**. Home and away goals are Poisson with the standard
Dixon-Coles low-scoring correction (the 0-0/1-0/0-1/1-1 `ρ` adjustment). Team strength is
**not** fit from this tournament's tiny sample — it comes from a **rolling Elo over ~49k
international matches (1872–present)**, and only five global coefficients are fit:

```text
log(λ_home) = μ + γ·home + a·s_home − d·s_away
log(λ_away) = μ          + a·s_away − d·s_home       (s = international-Elo strength)
```

- **Progression** (knockouts have no draw): a level score at 90' goes to extra time
  (same rates scaled to 30 min) and then penalties (50/50), so each fixture reports a
  **"progresses" probability per team** that sums to 100%.
- **No lookahead:** coefficients are fit on ~25k *pre-tournament* internationals and each
  team's strength is its rating as of the tournament's start, so every prediction uses
  only information available before the match.
- **Neutral venues:** knockout ties are predicted neutral (no home bump); host-nation
  home advantage in the group stage is not modelled.

## Track record (walk-forward, no lookahead)

On the matches already played, comparing each pre-match pick to the actual 90-minute
result:

| | correct | log loss | base-rate log loss | Brier |
|---|---:|---:|---:|---:|
| model | **59 / 94 (63%)** | **0.902** | 1.065 | 0.536 |

"Log loss" rewards being confident *and* right; the base-rate line is a naive predictor
(the completed-match H/D/A frequencies) for comparison. Numbers refresh automatically as
results land — see the live site for the current figure and the full match-by-match table.

## Automation

[`.github/workflows/update.yml`](.github/workflows/update.yml) runs every 3 hours (and on
demand): re-pull the data, rebuild the predictions and site, and deploy to GitHub Pages —
so the live site is always current before the next matchday.

## Run locally

```bash
pip install -r requirements.txt
cd src
python wc_fetch.py       # mirror the two CC0 datasets -> data/raw/
python wc_validate.py    # validate the data
python wc_backtest.py    # walk-forward backtest (calibration + log loss)
python wc_global.py      # fit coefficients on all international history + predict
python build_site.py     # regenerate results_log.csv + the static site/ (what CI deploys)
python -m pytest ../tests -q
```

## Data & licensing

Two free, public-domain sources — both GitHub, both CC0. **No betting-odds source is
used**; the model is trained purely on match results and international ratings.

- **[FIFA-World-Cup-2026-Dataset](https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset)**
  (current-tournament results, fixtures, `elo_rating` / `fifa_ranking_pre_tournament`
  priors) — **CC0-1.0**, updated daily.
- **[martj42/international_results](https://github.com/martj42/international_results)**
  (all international matches 1872-present) — **CC0**, public domain.

Public research project. Not affiliated with FIFA. Not betting advice.
