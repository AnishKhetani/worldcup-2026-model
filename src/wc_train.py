"""Weighted international training set for the per-team Dixon-Coles model.

Source: martj42/international_results ONLY (all internationals 1872-present, CC0). It
already contains the 2026 World Cup matches, so the model learns from group-stage and
completed-knockout form as the tournament progresses -- no separate WC table to stack
(which would double-count).

Each match gets a weight = time_decay x tournament_tier (change #3):
  * time_decay: the Dixon-Coles (1997) exponential downweight exp(-xi * days), set by a
    2-year half-life -- appropriate for international football, where squads and form
    turn over fast (the previous model used an 8-year half-life, over-weighting stale
    results).
  * tournament_tier: the World Cup edition live/most-recent as of the reference date is
    weighted highest (1.5); other major finals 1.0; qualifiers / Nations League 0.6;
    friendlies 0.25. The "live WC" boost keys off the match's edition + reference date
    (not a hardcoded 2026), so it generalises for walk-forward backtests.

Everything is walk-forward safe: build_training_set(reference_date) keeps only matches
on-or-before that date.
"""
from __future__ import annotations

import math

import pandas as pd

from wc_config import DATA_DIR, HTTP_HEADERS, HTTP_TIMEOUT, WC_RAW_DIR

INTL_DIR = DATA_DIR / "raw" / "intl"
RESULTS_CSV = INTL_DIR / "results.csv"
RESULTS_URL = ("https://raw.githubusercontent.com/martj42/international_results/"
               "master/results.csv")

# martj42 spelling -> our canonical WC26 spelling (the 7 that differ; verified by exact
# team identity). Applied so historical results join onto the World Cup fixtures.
ALIAS_TO_CANONICAL = {
    "United States": "USA",
    "Ivory Coast": "Côte d'Ivoire",
    "Iran": "IR Iran",
    "Turkey": "Türkiye",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Czech Republic": "Czechia",
}

# Tournament-importance multipliers.
TIER_LIVE_WC = 1.5      # the World Cup edition current/most-recent at the ref date
TIER_MAJOR = 1.0        # other World Cups / continental finals
TIER_COMPETITIVE = 0.6  # qualifiers, Nations League
TIER_FRIENDLY = 0.25
TIER_OTHER = 0.5

_MAJORS = ("fifa world cup", "uefa euro", "copa américa", "copa america",
           "african cup of nations", "afc asian cup", "gold cup",
           "concacaf championship", "confederations cup")

DEFAULT_HALF_LIFE_DAYS = 730.0  # 2-year half-life (international-appropriate)


def download(force: bool = False) -> None:
    """Mirror martj42 results.csv into data/raw/intl/."""
    if RESULTS_CSV.exists() and not force:
        return
    import requests
    INTL_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(RESULTS_URL, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    RESULTS_CSV.write_bytes(r.content)


def tournament_tier(tournament: str, year: int,
                    current_major_year: int | None = None) -> float:
    """Tournament-importance multiplier for one match (see module docstring)."""
    t = str(tournament).lower()
    if t == "fifa world cup":
        return TIER_LIVE_WC if (current_major_year is not None
                                and year == current_major_year) else TIER_MAJOR
    if "qualif" in t:
        return TIER_COMPETITIVE
    if "nations league" in t:
        return TIER_COMPETITIVE
    if t == "friendly":
        return TIER_FRIENDLY
    if any(m in t for m in _MAJORS):
        return TIER_MAJOR
    return TIER_OTHER


def load_martj42() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df["home"] = df["home_team"].replace(ALIAS_TO_CANONICAL)
    df["away"] = df["away_team"].replace(ALIAS_TO_CANONICAL)
    df["match_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["neutral"] = df["neutral"].astype(str).str.upper().isin(["TRUE", "1"])
    return df


def _wc_edition_starts(m: pd.DataFrame) -> pd.Series:
    wc = m[m["tournament"] == "FIFA World Cup"].dropna(subset=["match_dt"])
    return wc.groupby(wc["match_dt"].dt.year)["match_dt"].min()


def current_major_year(reference_date, edition_starts: pd.Series) -> int | None:
    """The World Cup edition year live/most-recent at reference_date (latest edition
    whose first match is on-or-before that date). None if none has begun."""
    ref = pd.to_datetime(reference_date)
    eligible = [int(y) for y, d in edition_starts.items() if pd.notna(d) and d <= ref]
    return max(eligible) if eligible else None


def build_training_set(reference_date=None,
                       half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
                       min_weight: float = 1e-3,
                       m: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weighted match table for fitting: home/away/home_score/away_score/neutral/weight.

    Only matches on-or-before `reference_date` are kept (no lookahead), weighted by
    recency x tournament tier, then pruned to weight >= min_weight (ancient matches
    contribute ~nothing anyway).
    """
    if m is None:
        m = load_martj42()
    m = m.dropna(subset=["home_score", "away_score", "match_dt"]).copy()
    ref = (pd.to_datetime(reference_date) if reference_date is not None
           else m["match_dt"].max())
    m = m[m["match_dt"] <= ref].copy()

    xi = math.log(2.0) / half_life_days
    days = (ref - m["match_dt"]).dt.days.clip(lower=0)
    m["time_weight"] = (-xi * days).apply(math.exp)
    cmy = current_major_year(ref, _wc_edition_starts(m))
    m["tier"] = [tournament_tier(t, dt.year, cmy)
                 for t, dt in zip(m["tournament"], m["match_dt"])]
    m["weight"] = m["time_weight"] * m["tier"]
    m = m[m["weight"] >= min_weight].copy()
    return m[["match_dt", "date", "home", "away", "home_score", "away_score",
              "neutral", "tournament", "tier", "weight"]].reset_index(drop=True)


def fixture_neutral(home: str, away: str, m: pd.DataFrame | None = None) -> bool:
    """Neutral-venue flag for a fixture, from martj42's 2026 WC rows (authoritative).
    Falls back to True (neutral) if not found -- safe for knockout venues, which are
    neutral unless a host nation is playing at home."""
    if m is None:
        m = load_martj42()
    wc = m[(m["match_dt"] >= "2026-06-01") & (m["tournament"] == "FIFA World Cup")]
    hit = wc[((wc["home"] == home) & (wc["away"] == away))
             | ((wc["home"] == away) & (wc["away"] == home))]
    if len(hit):
        return bool(hit.iloc[0]["neutral"])
    return True


def wc_team_names() -> list[str]:
    """Canonical World Cup 2026 team names (from the CC0 teams.csv)."""
    return pd.read_csv(WC_RAW_DIR / "teams.csv")["team_name"].tolist()


def reconciliation(m: pd.DataFrame | None = None) -> dict:
    """Which WC26 teams are present in martj42 (after aliasing)? No silent drops."""
    if m is None:
        m = load_martj42()
    present = set(m["home"]) | set(m["away"])
    wc = set(wc_team_names())
    return {"matched": sorted(wc & present), "unmatched": sorted(wc - present),
            "n_matched": len(wc & present), "n_total": len(wc)}
