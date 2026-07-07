"""Fit the goal-rate coefficients on ALL international history, then check whether that
beats fitting them on this tournament alone -- and re-predict the remaining fixtures.

Two models, same Dixon-Coles machinery, differing only in where the five coefficients
and the team strengths come from:

  * IN-TOURNAMENT (wc_backtest): strengths = teams.csv Elo; coefficients fit walk-forward
    on the WC matches played so far.
  * GLOBAL (here): strengths = a rolling Elo over ~49k internationals (pre-tournament,
    so no lookahead); coefficients fit ONCE on 2000-present internationals, weighted by
    importance x recency, using ZERO World Cup 2026 matches.

The global model therefore predicts the WC purely from prior international history. We
score both on the same held-out WC matches (log loss / Brier / accuracy / ECE) to see
whether the broader sample actually helps.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from wc_backtest import CLASSES, _brier, _logloss, reliability_table, run_backtest
from wc_config import PROCESSED_DIR
from wc_data import load_played_matches, load_remaining_fixtures, team_table
from wc_dixon_coles import DixonColesStrength
from wc_intl_elo import NAME_ALIAS, build

_IDX = {c: i for i, c in enumerate(CLASSES)}


def _elo_crosscheck(intl) -> None:
    t = team_table()
    mine, theirs = [], []
    for _, r in t.iterrows():
        elo = intl.pre_tournament.get(NAME_ALIAS.get(r["team_name"], r["team_name"]))
        if elo is not None:
            mine.append(elo)
            theirs.append(r["elo_rating"])
    corr = float(np.corrcoef(mine, theirs)[0, 1])
    print(f"Elo cross-check: rolling-Elo vs teams.csv elo_rating over "
          f"{len(mine)} WC teams -> correlation {corr:.3f}")


def _global_predictions(model, intl, played) -> pd.DataFrame:
    """Global-model normal-time H/D/A for the WC matches, using pre-tournament
    rolling-Elo strengths (designated-home gamma applied, matching the backtest)."""
    rows = []
    for _, m in played.iterrows():
        sh = intl.wc_strength(m["home_team"])
        sa = intl.wc_strength(m["away_team"])
        if sh is None or sa is None:
            continue
        p_h, p_d, p_a = model.predict_outcome(sh, sa, neutral=False)
        rows.append({"match_id": m["match_id"], "result90": m["result90"],
                     "p_home": p_h, "p_draw": p_d, "p_away": p_a})
    return pd.DataFrame(rows)


def _metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    y = df["result90"].map(_IDX).to_numpy()
    P = df[["p_home", "p_draw", "p_away"]].to_numpy()
    _, ece = reliability_table(P, y)
    return _logloss(P, y), _brier(P, y), float((P.argmax(1) == y).mean()), ece


def main() -> int:
    intl = build()
    print("=" * 74)
    print("WORLD CUP 2026 - does fitting on ALL international history help?")
    print("=" * 74)
    print(f"International fit sample: {len(intl.fit_matches):,} matches "
          f"(2000-01 .. tournament start), weighted by importance x recency.")
    _elo_crosscheck(intl)

    gmodel = DixonColesStrength().fit(intl.fit_matches)   # weight/neutral cols auto-used
    print(f"\nGLOBAL coefficients (history-fit): mu={gmodel.mu:.3f} "
          f"gamma={gmodel.gamma:.3f} a={gmodel.a:.3f} d={gmodel.d:.3f} "
          f"rho={gmodel.rho:.3f}")

    # --- head-to-head on the SAME held-out WC matches -----------------------
    played = load_played_matches()
    bt = run_backtest()                     # in-tournament walk-forward predictions
    gpred = _global_predictions(gmodel, intl, played)
    shared = sorted(set(bt["match_id"]) & set(gpred["match_id"]))
    A = bt[bt["match_id"].isin(shared)].sort_values("match_id").reset_index(drop=True)
    G = gpred[gpred["match_id"].isin(shared)].sort_values("match_id").reset_index(drop=True)

    print(f"\nHead-to-head on {len(shared)} shared held-out WC matches:")
    print(f"  {'model':<26}{'log loss':>10}{'brier':>9}{'accuracy':>10}{'ECE':>8}")
    for name, d in [("in-tournament (WC-only)", A), ("global (all history)", G)]:
        ll, br, ac, ece = _metrics(d)
        print(f"  {name:<26}{ll:>10.4f}{br:>9.4f}{ac:>10.3f}{ece:>8.3f}")
    # the global model can also score every played match (no warm-up needed)
    llf, brf, acf, ecef = _metrics(gpred)
    print(f"  {'global on all '+str(len(gpred))+' matches':<26}"
          f"{llf:>10.4f}{brf:>9.4f}{acf:>10.3f}{ecef:>8.3f}")

    # --- re-predict the remaining fixtures with the global model ------------
    print("\nRemaining fixtures - global (all-history) model, neutral venue:")
    fixtures = load_remaining_fixtures()
    out_rows = []
    for _, f in fixtures.iterrows():
        sh, sa = intl.wc_strength(f["home_team"]), intl.wc_strength(f["away_team"])
        if sh is None or sa is None:
            print(f"  {f['home_team']} vs {f['away_team']}: strength unavailable")
            continue
        p_h, p_d, p_a = gmodel.predict_outcome(sh, sa, neutral=True)
        prog_h, prog_a = gmodel.predict_progression(sh, sa, neutral=True)
        print(f"  {f['stage']:<15} {f['home_team']} vs {f['away_team']}: "
              f"W/D/L {p_h:.0%}/{p_d:.0%}/{p_a:.0%}   "
              f"progress {f['home_team']} {prog_h:.0%} / {f['away_team']} {prog_a:.0%}")
        out_rows.append({
            "match_id": f["match_id"], "stage": f["stage"],
            "home_team": f["home_team"], "away_team": f["away_team"],
            "p_home_win90": p_h, "p_draw90": p_d, "p_away_win90": p_a,
            "p_home_progress": prog_h, "p_away_progress": prog_a})

    if out_rows:
        out = PROCESSED_DIR / "wc_remaining_predictions_global.csv"
        pd.DataFrame(out_rows).to_csv(out, index=False)
        print(f"\nSaved: {out}")
    print("\nSmall-sample caveat still applies: a held-out check across a few dozen "
          "matches,\nnot a trading system. No edge is claimed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
