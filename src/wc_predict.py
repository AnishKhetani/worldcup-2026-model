"""Predict the remaining World Cup 2026 knockout fixtures.

Fits the strength-prior Dixon-Coles model on ALL matches played so far, then for each
upcoming fixture whose two teams are already known, reports:
  * normal-time win / draw / loss probabilities (what the goal model directly gives);
  * a "progresses" probability per team -- draws don't stand in a knockout, so this
    carries the tie through extra time and penalties (see wc_dixon_coles).

Predictions are NEUTRAL-venue: the knockout "home" team is an administrative bracket
label, not a real home tie, so neither side gets the fitted home bump. Any fixture
involving a host nation (USA/Canada/Mexico) is flagged, since a genuine home venue would
be a real effect this neutral prediction omits.

Fixtures whose teams are not yet decided (they await earlier knockout results) are
listed as pending -- their matchups don't exist yet, so there is nothing to predict.
"""
from __future__ import annotations

import sys

import pandas as pd

from wc_config import PROCESSED_DIR
from wc_data import count_pending_tbd, load_played_matches, load_remaining_fixtures
from wc_dixon_coles import DixonColesStrength


def main() -> int:
    played = load_played_matches()
    model = DixonColesStrength().fit(played)

    print("=" * 74)
    print("WORLD CUP 2026 - remaining knockout fixtures (strength-prior Dixon-Coles)")
    print("=" * 74)
    print(f"Fit on {len(played)} played matches. Model coefficients: "
          f"mu={model.mu:.3f} gamma(home)={model.gamma:.3f} "
          f"a(attack)={model.a:.3f} d(defence)={model.d:.3f} rho={model.rho:.3f}")
    print("Predictions are neutral-venue; win/draw/loss are normal-time (90').\n")

    fixtures = load_remaining_fixtures()
    if fixtures.empty:
        print("No fixtures with known teams yet.")
    out_rows = []
    for _, f in fixtures.iterrows():
        p_h, p_d, p_a = model.predict_outcome(
            f["home_strength"], f["away_strength"], neutral=True)
        prog_h, prog_a = model.predict_progression(
            f["home_strength"], f["away_strength"], neutral=True)
        host = " [host nation involved]" if (f["home_is_host"] or f["away_is_host"]) else ""
        print(f"{f['stage']} - {f['date'].date()}{host}")
        print(f"  {f['home_team']} vs {f['away_team']}")
        print(f"    normal-time :  {f['home_team']} {p_h:.1%}   "
              f"draw {p_d:.1%}   {f['away_team']} {p_a:.1%}")
        print(f"    progresses  :  {f['home_team']} {prog_h:.1%}   "
              f"{f['away_team']} {prog_a:.1%}\n")
        out_rows.append({
            "match_id": f["match_id"], "date": f["date"], "stage": f["stage"],
            "home_team": f["home_team"], "away_team": f["away_team"],
            "p_home_win90": p_h, "p_draw90": p_d, "p_away_win90": p_a,
            "p_home_progress": prog_h, "p_away_progress": prog_a,
        })

    pending = count_pending_tbd()
    if pending:
        print(f"({pending} further scheduled match(es) have undecided matchups - "
              f"teams TBD until earlier rounds finish; nothing to predict yet.)")

    if out_rows:
        out = PROCESSED_DIR / "wc_remaining_predictions.csv"
        pd.DataFrame(out_rows).to_csv(out, index=False)
        print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
