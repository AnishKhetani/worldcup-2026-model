"""All-international match history + a rolling Elo, to fit the goal model's structure.

Motivation: the in-tournament model fits its five goal-rate coefficients (mu, gamma, a,
d, rho) from only ~94 World Cup matches. Those coefficients describe international
football in general, not this tournament, so they are better estimated from the FULL
history of international matches. This module supplies that:

  1. Mirror the public results set (martj42/international_results, ~49.5k matches,
     1872-present, CC0) into data/raw/intl/.
  2. Run a rolling Elo (World-Football-Elo style: home advantage unless neutral,
     tournament-importance K, goal-difference multiplier). Because Elo is walk-forward
     by construction, each match's PRE-match ratings use only earlier results.
  3. Emit a fit table (pre-match strengths + 90'-equivalent goals + neutral flag + an
     importance x recency weight) and each team's pre-tournament rating for prediction.

Team strength for the goal model is the pre-match Elo, standardised with ONE fixed
(centre, scale) so the historical fit and the World Cup prediction share a scale.
Name differences between the WC dataset and the results set are resolved by an explicit
alias map; anything unmatched is reported, never silently fuzzy-matched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc_config import DATA_DIR

INTL_DIR = DATA_DIR / "raw" / "intl"
RESULTS_CSV = INTL_DIR / "results.csv"
RESULTS_URL = ("https://raw.githubusercontent.com/martj42/international_results/"
               "master/results.csv")

TOURNAMENT_START = pd.Timestamp("2026-06-11")   # WC 2026 opening match
ELO_START = 1500.0
ELO_HOME_ADV = 100.0        # World-Football-Elo home advantage (skipped if neutral)
FIT_FROM = pd.Timestamp("2000-01-01")   # recency weighting makes older data ~irrelevant
RECENCY_HALFLIFE_YEARS = 8.0

# WC-2026 team_name -> results-set name.
NAME_ALIAS = {
    "Czechia": "Czech Republic", "USA": "United States", "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran", "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
}


def importance_index(tournament: str) -> int:
    """World-Football-Elo weight index by match type (also reused as the fit weight)."""
    t = str(tournament).lower()
    if "quali" in t:
        return 40
    if "world cup" in t:
        return 60
    if "nations league" in t:
        return 40
    if "friendly" in t:
        return 20
    for kw in ("euro", "copa am", "african cup", "asian cup", "gold cup",
               "concacaf", "confederations", "oceania nations", "nations cup"):
        if kw in t:
            return 50
    return 30


def _goal_diff_multiplier(gd: int) -> float:
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def download(force: bool = False) -> None:
    if RESULTS_CSV.exists() and not force:
        return
    import requests
    INTL_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(RESULTS_URL, headers={"User-Agent": "Mozilla/5.0 (research)"},
                     timeout=90)
    r.raise_for_status()
    RESULTS_CSV.write_bytes(r.content)


def _load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    return df.sort_values("date").reset_index(drop=True)


@dataclass
class IntlElo:
    fit_matches: pd.DataFrame      # standardised strengths + goals + neutral + weight
    ratings: dict                  # results-set team name -> current Elo
    pre_tournament: dict           # results-set name -> Elo as of WC 2026 kickoff
    centre: float                  # strength standardisation constants (shared)
    scale: float

    def strength(self, elo: float) -> float:
        return (elo - self.centre) / self.scale

    def wc_strength(self, wc_team_name: str) -> float | None:
        """Pre-tournament standardised strength for a WC-2026 team, or None if
        the team never appears in the results set (reported, not guessed)."""
        name = NAME_ALIAS.get(wc_team_name, wc_team_name)
        elo = self.pre_tournament.get(name)
        return None if elo is None else self.strength(elo)


def build(force_download: bool = False) -> IntlElo:
    download(force=force_download)
    df = _load_results()

    ratings: dict[str, float] = {}
    pre_tournament: dict[str, float] = {}
    captured = False
    rec = []  # per-match pre-Elo snapshots (all matches; filtered for fitting later)

    for row in df.itertuples():
        # snapshot ratings as of the tournament's opening date, once
        if not captured and row.date >= TOURNAMENT_START:
            pre_tournament = dict(ratings)
            captured = True
        rh = ratings.get(row.home_team, ELO_START)
        ra = ratings.get(row.away_team, ELO_START)
        adv = 0.0 if row.neutral else ELO_HOME_ADV
        exp_h = 1.0 / (1.0 + 10.0 ** (-((rh + adv) - ra) / 400.0))
        gd = row.home_score - row.away_score
        w_home = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        idx = importance_index(row.tournament)
        k = idx * _goal_diff_multiplier(gd)
        delta = k * (w_home - exp_h)
        rec.append((row.date, rh, ra, row.home_score, row.away_score,
                    bool(row.neutral), idx))
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta

    if not captured:      # tournament start is in the future relative to the data
        pre_tournament = dict(ratings)

    snap = pd.DataFrame(rec, columns=["date", "elo_home", "elo_away", "hg90", "ag90",
                                      "neutral", "importance"])

    # fitting sample: recent, pre-tournament (no lookahead into the WC being predicted)
    fit = snap[(snap["date"] >= FIT_FROM) & (snap["date"] < TOURNAMENT_START)].copy()
    centre = float(pd.concat([fit["elo_home"], fit["elo_away"]]).mean())
    scale = float(pd.concat([fit["elo_home"], fit["elo_away"]]).std(ddof=0)) or 1.0
    fit["home_strength"] = (fit["elo_home"] - centre) / scale
    fit["away_strength"] = (fit["elo_away"] - centre) / scale
    age_years = (TOURNAMENT_START - fit["date"]).dt.days / 365.25
    recency = np.exp(-age_years / RECENCY_HALFLIFE_YEARS)
    fit["weight"] = (fit["importance"] / 60.0) * recency

    return IntlElo(fit_matches=fit.reset_index(drop=True), ratings=ratings,
                   pre_tournament=pre_tournament, centre=centre, scale=scale)
