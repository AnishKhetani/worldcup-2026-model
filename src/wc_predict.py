"""Predict the remaining World Cup 2026 knockout fixtures (CLI report).

Fits the per-team Dixon-Coles model on the full weighted international pool as of now
(so it reflects group-stage + completed-knockout form), then for each upcoming fixture
whose teams are known reports the normal-time W/D/L and each team's progression
probability (via the empirical extra-time/penalty conversion in wc_knockout).

Fixtures whose teams aren't decided yet (await earlier results) are listed as pending.
"""
from __future__ import annotations

import sys

import pandas as pd

from wc_config import PROCESSED_DIR
from wc_data import count_pending_tbd, load_remaining_fixtures
from wc_dixon_coles import DixonColesModel
from wc_knockout import p_advance
from wc_train import build_training_set, fixture_neutral, load_martj42


def main() -> int:
    m = load_martj42()
    model = DixonColesModel().fit(build_training_set(m=m))

    print("=" * 74)
    print("WORLD CUP 2026 - remaining knockout fixtures (per-team Dixon-Coles)")
    print("=" * 74)
    print(f"Fit on {model.n_train:,} weighted international matches ({len(model.teams)} "
          f"teams). Coeffs: mu={model.mu:.3f} home_adv={model.home_adv:.3f} "
          f"home_def={model.home_def:.3f} rho={model.rho:.3f}")
    print("Win/draw/loss are normal-time (90'); progression includes extra time + "
          "penalties.\n")

    fixtures = load_remaining_fixtures()
    out_rows = []
    for _, f in fixtures.iterrows():
        neutral = fixture_neutral(f["home_team"], f["away_team"], m)
        o = model.predict_outcome(f["home_team"], f["away_team"], neutral)
        adv = p_advance(o["p_home"], o["p_draw"], o["p_away"])
        host = " [host nation]" if (f.get("home_is_host") or f.get("away_is_host")) else ""
        print(f"{f['stage']} - {f['date'].date()}{host}")
        print(f"  {f['home_team']} vs {f['away_team']}  "
              f"(xG {o['exp_goals_home']:.2f}-{o['exp_goals_away']:.2f})")
        print(f"    normal-time :  {f['home_team']} {o['p_home']:.1%}   "
              f"draw {o['p_draw']:.1%}   {f['away_team']} {o['p_away']:.1%}")
        print(f"    progresses  :  {f['home_team']} {adv.p_home_advance:.1%}   "
              f"{f['away_team']} {adv.p_away_advance:.1%}\n")
        out_rows.append({
            "match_id": f["match_id"], "date": f["date"], "stage": f["stage"],
            "home_team": f["home_team"], "away_team": f["away_team"],
            "p_home_win90": o["p_home"], "p_draw90": o["p_draw"], "p_away_win90": o["p_away"],
            "p_home_progress": adv.p_home_advance, "p_away_progress": adv.p_away_advance})

    pending = count_pending_tbd()
    if pending:
        print(f"({pending} further scheduled match(es) have undecided matchups - "
              f"teams TBD until earlier rounds finish.)")
    if out_rows:
        out = PROCESSED_DIR / "wc_remaining_predictions.csv"
        pd.DataFrame(out_rows).to_csv(out, index=False)
        print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
