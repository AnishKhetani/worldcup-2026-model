"""Empirical favorite-conversion in drawn World Cup knockout ties (change #1).

For a knockout fixture the model needs P(each team advances) = wins the tie including
extra time and penalties, not just the 90-minute result. The naive way (what this repo
used before) is an extra-time Poisson plus a hard 50/50 shootout. That is wrong: it
treats the shootout as a coin flip regardless of who is favored, and it lets the
favorite's regulation edge flow into extra time largely unchanged.

This module answers it EMPIRICALLY, from 92 years of World Cup knockout history
(Fjelstul World Cup Database, all knockout ties 1930-2022), independently of any
betting market:

  Conditional on a tie being level at 90', the favorite advances with
      c(q) = logistic(gamma * logit(q)),   q = p_fav / (p_fav + p_dog),
  where q is the favorite's regulation win-share and gamma is fit through the origin
  (so evenly-matched sides are exactly 50/50). The favorite's regulation strength q is
  computed the same way in the historical fit (from a walk-forward Elo over martj42) and
  at prediction time (from the Dixon-Coles 90' probabilities), so the relationship
  transfers cleanly.

FINDINGS (reproduced by fit_from_data / tests/test_wc_knockout.py):
  * Favorites advance ~62% of drawn knockout ties (n~82), significantly above a coin
    flip -- so a flat 50/50 is wrong.
  * But far below the favorite's regulation edge implies: fitted gamma ~ 0.47, heavy
    compression toward the coin flip, because "still level at 90'" is itself strong
    evidence the tie is close.

GAMMA is pinned so the runtime never depends on re-reading the raw files; a regression
test re-fits from the local data and asserts they still agree.

Ported from the author's separate betting-model codebase (`wc26bet` knockout_conversion
+ advance), adapted to this repo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wc_config import DATA_DIR, HTTP_HEADERS, HTTP_TIMEOUT
from wc_train import RESULTS_CSV

# Fitted on the local Fjelstul + martj42 data (see fit_from_data). Pinned for a
# file-free runtime; the regression test re-fits and checks agreement.
GAMMA: float = 0.472

FJELSTUL_DIR = DATA_DIR / "raw" / "fjelstul"
FJELSTUL_CSV = FJELSTUL_DIR / "matches.csv"
FJELSTUL_URL = ("https://raw.githubusercontent.com/jfjelstul/worldcup/master/"
                "data-csv/matches.csv")

# martj42 spellings for the two historical entities Fjelstul names differently
# (confirmed by exact date+score matches).
FJELSTUL_TO_MARTJ42 = {"West Germany": "Germany", "Soviet Union": "Russia"}

# Walk-forward Elo params for the historical strength proxy (used only in the fit).
ELO_INIT, ELO_K, ELO_HA = 1500.0, 40.0, 65.0


def download_fjelstul(force: bool = False) -> None:
    """Mirror the Fjelstul World Cup Database matches.csv (CC-BY-SA 4.0)."""
    if FJELSTUL_CSV.exists() and not force:
        return
    import requests
    FJELSTUL_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(FJELSTUL_URL, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    body = r.content
    # Don't let a truncated download clobber good committed calibration data.
    if FJELSTUL_CSV.exists() and len(body) < 0.5 * FJELSTUL_CSV.stat().st_size:
        print(f"  matches.csv kept: new {len(body):,}B < 50% of existing "
              f"{FJELSTUL_CSV.stat().st_size:,}B — suspected truncation")
        return
    FJELSTUL_CSV.write_bytes(body)


# --- runtime conversion (pure, pinned) --------------------------------------
def conditional_conversion(q: float, gamma: float = GAMMA) -> float:
    """P(favorite advances | tie level at 90') for a favorite whose regulation
    win-share is q in [0.5, 1]. Odds-power form through the origin: q=0.5 -> 0.5."""
    q = min(max(float(q), 1e-6), 1 - 1e-6)
    if q <= 0.5:
        return 0.5
    lo = gamma * math.log(q / (1.0 - q))
    return 1.0 / (1.0 + math.exp(-lo))


def win_share(p_fav_win: float, p_dog_win: float) -> float:
    """Regulation win-share q = p_fav / (p_fav + p_dog); 0.5 if both ~0."""
    denom = p_fav_win + p_dog_win
    return 0.5 if denom <= 0 else p_fav_win / denom


@dataclass
class AdvanceBreakdown:
    p_home_advance: float
    p_away_advance: float
    fav_side: str          # "home" | "away" | "even"
    q_fav: float
    cond_home: float       # P(home advances | level at 90')
    cond_away: float


def p_advance(p_home_90: float, p_draw_90: float, p_away_90: float,
              *, gamma: float = GAMMA) -> AdvanceBreakdown:
    """Map a 90' Dixon-Coles prediction to P(advance) per team. Only the three 90'
    outcome probabilities are needed; the two advance probs sum to 1 by construction."""
    fav_home = p_home_90 >= p_away_90
    p_fav = max(p_home_90, p_away_90)
    p_dog = min(p_home_90, p_away_90)
    q_fav = win_share(p_fav, p_dog)
    c_fav = conditional_conversion(q_fav, gamma=gamma)
    if fav_home:
        cond_home, cond_away = c_fav, 1.0 - c_fav
        fav_side = "home" if p_home_90 > p_away_90 else "even"
    else:
        cond_home, cond_away = 1.0 - c_fav, c_fav
        fav_side = "away"
    p_home_adv = p_home_90 + p_draw_90 * cond_home
    p_away_adv = p_away_90 + p_draw_90 * cond_away
    return AdvanceBreakdown(p_home_adv, p_away_adv, fav_side, q_fav, cond_home, cond_away)


# --- reproducible empirical fit (analysis + regression test) ----------------
def _goal_mult(gd: int) -> float:
    gd = abs(int(gd))
    return 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)


def _load_martj42() -> pd.DataFrame:
    m = pd.read_csv(RESULTS_CSV)
    m["dt"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.dropna(subset=["home_score", "away_score", "dt"]).sort_values("dt").reset_index(drop=True)
    m["neutral"] = m["neutral"].astype(str).str.upper().isin(["TRUE", "1"])
    return m


def _build_elo(mart: pd.DataFrame):
    rating: dict[str, float] = {}
    hist: dict[str, list] = {}
    for r in mart.itertuples(index=False):
        rh = rating.get(r.home_team, ELO_INIT)
        ra = rating.get(r.away_team, ELO_INIT)
        ha = 0.0 if r.neutral else ELO_HA
        eh = 1.0 / (1.0 + 10 ** (-(rh + ha - ra) / 400.0))
        gd = int(r.home_score - r.away_score)
        sh = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        delta = ELO_K * _goal_mult(gd) * (sh - eh)
        rating[r.home_team] = rh + delta
        rating[r.away_team] = ra - delta
        hist.setdefault(r.home_team, []).append((r.dt, rating[r.home_team]))
        hist.setdefault(r.away_team, []).append((r.dt, rating[r.away_team]))
    return hist


def _elo_asof(hist, team, dt) -> float | None:
    lst = hist.get(team)
    if not lst:
        return None
    best = None
    for d, rr in lst:
        if d < dt:
            best = rr
        else:
            break
    return best if best is not None else ELO_INIT


def _fit_q_of_gap(mart: pd.DataFrame) -> float:
    from scipy.optimize import minimize
    rating: dict[str, float] = {}
    diffs, ywin = [], []
    for r in mart.itertuples(index=False):
        rh = rating.get(r.home_team, ELO_INIT)
        ra = rating.get(r.away_team, ELO_INIT)
        ha = 0.0 if r.neutral else ELO_HA
        diff = (rh + ha) - ra
        gd = int(r.home_score - r.away_score)
        if gd != 0:
            diffs.append(diff)
            ywin.append(1.0 if gd > 0 else 0.0)
        eh = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        sh = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        delta = ELO_K * _goal_mult(gd) * (sh - eh)
        rating[r.home_team] = rh + delta
        rating[r.away_team] = ra - delta
    x = np.asarray(diffs) / 400.0
    y = np.asarray(ywin)

    def nll(b):
        p = np.clip(1.0 / (1.0 + np.exp(-b[0] * x)), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    return float(minimize(nll, [1.0], method="Nelder-Mead").x[0])


def fit_from_data() -> dict:
    """Reproduce the empirical study from the local Fjelstul + martj42 files."""
    from scipy.optimize import minimize
    from scipy.stats import binomtest

    mart = _load_martj42()
    hist = _build_elo(mart)
    beta_q = _fit_q_of_gap(mart)

    def q_of_gap(gap):
        return 1.0 / (1.0 + math.exp(-beta_q * abs(gap) / 400.0))

    fj = pd.read_csv(FJELSTUL_CSV)
    fj["dt"] = pd.to_datetime(fj["match_date"], errors="coerce")
    ko = fj[fj.knockout_stage == 1]
    lvl = ko[(ko.extra_time == 1) | (ko.penalty_shootout == 1)]
    lvl = lvl[(lvl.home_team_win == 1) | (lvl.away_team_win == 1)]  # exclude replayed draws

    recs = []
    for r in lvl.itertuples(index=False):
        h = FJELSTUL_TO_MARTJ42.get(r.home_team_name, r.home_team_name)
        a = FJELSTUL_TO_MARTJ42.get(r.away_team_name, r.away_team_name)
        rh, ra = _elo_asof(hist, h, r.dt), _elo_asof(hist, a, r.dt)
        if rh is None or ra is None:
            continue
        eff_h, eff_a = rh, ra
        if str(r.country_name) == str(r.home_team_name):
            eff_h += ELO_HA
        if str(r.country_name) == str(r.away_team_name):
            eff_a += ELO_HA
        fav_home = eff_h >= eff_a
        gap = abs(eff_h - eff_a)
        fav_adv = (r.home_team_win == 1) if fav_home else (r.away_team_win == 1)
        recs.append(dict(gap=gap, q=q_of_gap(gap), fav_adv=bool(fav_adv),
                         is_pen=bool(r.penalty_shootout == 1)))
    d = pd.DataFrame(recs)

    lq = np.log(d.q / (1 - d.q)).to_numpy()
    yv = d.fav_adv.astype(float).to_numpy()

    def nll_gamma(g):
        p = np.clip(1.0 / (1.0 + np.exp(-g[0] * lq)), 1e-9, 1 - 1e-9)
        return -(yv * np.log(p) + (1 - yv) * np.log(1 - p)).sum()

    gamma = float(minimize(nll_gamma, [1.0], method="Nelder-Mead").x[0])
    bt = binomtest(int(d.fav_adv.sum()), len(d), 0.5)
    return {"gamma": gamma, "n": len(d), "fav_adv": float(d.fav_adv.mean()),
            "binom_p": float(bt.pvalue),
            "shootout_rate": float(d[d.is_pen].fav_adv.mean()),
            "extra_time_rate": float(d[~d.is_pen].fav_adv.mean())}
