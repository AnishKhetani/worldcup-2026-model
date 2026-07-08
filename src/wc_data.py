"""Load + shape the World Cup 2026 modelling table from the mirrored CC0 CSVs.

Produces the tidy per-match frame the goal-rate model consumes, with three jobs the
raw CSVs don't do for you:

  1. **Strength prior.** Each team's pre-tournament `elo_rating` is standardised to a
     z-score `strength` (mean 0, sd 1 across the 48 teams). This is the *prior* the
     Dixon-Coles model leans on instead of fitting 96 attack/defence parameters from a
     3-games-each sample. `fifa_ranking_pre_tournament` is kept as a fallback strength
     for any team missing an Elo (none are, currently) and as a cross-check.
  2. **Regulation (90-minute) scores.** A Poisson goal model describes NORMAL time, so
     knockout matches decided in extra time must be reduced to their score at 90'.
     Group + `Regular` knockout matches already store the 90' score; `AET`/`Penalties`
     matches are rebuilt from `match_events` (Goal events with base minute <= 90 — this
     dataset writes 2nd-half stoppage as "90+x" but extra-time as raw 91..120, so the
     <=90 cut is exact). Verified on matches 81/87 (both level at 90).
  3. **Chronological order.** Rows are sorted by (date, match_id) so the walk-forward
     backtest can expand its training window without lookahead.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from wc_config import WC_COMPLETED_STATUSES, WC_RAW_DIR

HOST_NATIONS = {"USA", "United States", "Canada", "Mexico"}


def _completed_mask(status: pd.Series) -> pd.Series:
    """Boolean mask of played matches, tolerant of upstream status casing/spelling
    (see WC_COMPLETED_STATUSES). A brittle exact-string match here would silently zero
    the entire track record if the upstream ever relabelled 'Completed'."""
    return status.astype(str).str.strip().str.lower().isin(WC_COMPLETED_STATUSES)


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(WC_RAW_DIR / name)


def team_table() -> pd.DataFrame:
    """team_id -> name, elo, fifa rank, standardised strength z-score."""
    t = _csv("teams.csv").copy()
    elo = t["elo_rating"].astype(float)
    mu, sd = elo.mean(), elo.std(ddof=0)
    t["strength"] = (elo - mu) / (sd or 1.0)
    # Fallback for a missing Elo: map FIFA rank onto the same z-scale via -log(rank).
    if elo.isna().any():
        r = -np.log(t["fifa_ranking_pre_tournament"].astype(float))
        rz = (r - r.mean()) / (r.std(ddof=0) or 1.0)
        t.loc[elo.isna(), "strength"] = rz[elo.isna()]
    t["is_host"] = t["team_name"].isin(HOST_NATIONS)
    return t[["team_id", "team_name", "elo_rating",
              "fifa_ranking_pre_tournament", "strength", "is_host"]]


def _base_minute(s) -> int | None:
    m = re.match(r"\s*(\d+)", str(s))
    return int(m.group(1)) if m else None


def _regulation_scores() -> dict[int, tuple[int, int]]:
    """match_id -> (home_goals_90, away_goals_90) for AET/Penalties matches, rebuilt
    from Goal events at base minute <= 90. Other matches use their stored score."""
    matches = _csv("matches.csv")
    ev = _csv("match_events.csv")
    ev = ev[ev["event_type"] == "Goal"].copy()
    # Coerce to numeric so an unparseable minute becomes NaN (excluded) rather than
    # silently poisoning the comparison; warn so it doesn't pass unnoticed.
    ev["bm"] = pd.to_numeric(ev["minute"].apply(_base_minute), errors="coerce")
    unparsed = int(ev["bm"].isna().sum())
    if unparsed:
        print(f"  [warn] {unparsed} Goal event(s) with unparseable minute — "
              f"excluded from 90' reconstruction")
    reg = ev[ev["bm"] <= 90]
    out: dict[int, tuple[int, int]] = {}
    et_ids = matches.loc[matches["result_type"].isin(["AET", "Penalties"]), "match_id"]
    for mid in et_ids:
        row = matches.loc[matches["match_id"] == mid].iloc[0]
        g = reg[reg["match_id"] == mid]
        hg = int((g["team_id"] == row["home_team_id"]).sum())
        ag = int((g["team_id"] == row["away_team_id"]).sum())
        out[int(mid)] = (hg, ag)
    return out


def _result(hg: float, ag: float) -> str:
    return "H" if hg > ag else ("A" if hg < ag else "D")


def _advanced(row) -> str:
    """Which side actually advanced from a completed knockout tie: 'H', 'A', or ''.
    Penalties are decided by the shootout score; AET/regulation ties by the final score
    (which for AET already reflects extra time). Used to score the model's progression
    call, so a shootout win isn't counted as a miss just because 90' was level."""
    if str(row.get("result_type", "")).strip().lower() == "penalties":
        h, a = row.get("home_penalty_score"), row.get("away_penalty_score")
    else:
        h, a = row.get("home_score"), row.get("away_score")
    if pd.isna(h) or pd.isna(a):
        return ""
    return "H" if h > a else ("A" if a > h else "")


def _assemble() -> pd.DataFrame:
    matches = _csv("matches.csv").copy()
    stages = _csv("tournament_stages.csv").set_index("stage_id")
    teams = team_table().set_index("team_id")

    matches["stage"] = matches["stage_id"].map(stages["stage_name"])
    matches["is_knockout"] = matches["stage_id"].map(stages["is_knockout"]).astype(bool)

    reg = _regulation_scores()

    def hg90(r):
        return reg.get(int(r["match_id"]), (r["home_score"], r["away_score"]))[0]

    def ag90(r):
        return reg.get(int(r["match_id"]), (r["home_score"], r["away_score"]))[1]

    matches["hg90"] = matches.apply(hg90, axis=1)
    matches["ag90"] = matches.apply(ag90, axis=1)

    def strength(tid):
        return teams["strength"].get(tid, np.nan) if pd.notna(tid) else np.nan

    def name(tid):
        return teams["team_name"].get(tid) if pd.notna(tid) else None

    def host(tid):
        return bool(teams["is_host"].get(tid, False)) if pd.notna(tid) else False

    matches["home_team"] = matches["home_team_id"].map(name)
    matches["away_team"] = matches["away_team_id"].map(name)
    matches["home_strength"] = matches["home_team_id"].map(strength)
    matches["away_strength"] = matches["away_team_id"].map(strength)
    matches["home_is_host"] = matches["home_team_id"].map(host)
    matches["away_is_host"] = matches["away_team_id"].map(host)
    matches["date"] = pd.to_datetime(matches["date"])
    return matches.sort_values(["date", "match_id"]).reset_index(drop=True)


def load_played_matches() -> pd.DataFrame:
    """Completed matches with both teams and a 90' score, chronologically ordered."""
    m = _assemble()
    m = m[_completed_mask(m["status"])
          & m["home_strength"].notna() & m["away_strength"].notna()].copy()
    m["result90"] = [_result(h, a) for h, a in zip(m["hg90"], m["ag90"])]
    m["was_et"] = m["result_type"].isin(["AET", "Penalties"])
    m["advanced"] = m.apply(lambda r: _advanced(r) if r["is_knockout"] else "", axis=1)
    cols = ["match_id", "date", "stage", "is_knockout", "home_team", "away_team",
            "home_team_id", "away_team_id", "home_strength", "away_strength",
            "hg90", "ag90", "result90", "was_et", "advanced"]
    return m[cols].reset_index(drop=True)


def load_remaining_fixtures() -> pd.DataFrame:
    """Scheduled knockout matches whose two teams are already known (bracket filled)."""
    m = _assemble()
    m = m[~_completed_mask(m["status"])
          & m["home_strength"].notna() & m["away_strength"].notna()].copy()
    cols = ["match_id", "date", "stage", "home_team", "away_team",
            "home_team_id", "away_team_id", "home_strength", "away_strength",
            "home_is_host", "away_is_host"]
    return m[cols].reset_index(drop=True)


def count_pending_tbd() -> int:
    """Scheduled matches whose teams are not yet determined (await earlier results)."""
    m = _assemble()
    return int((~_completed_mask(m["status"])
                & (m["home_strength"].isna() | m["away_strength"].isna())).sum())
