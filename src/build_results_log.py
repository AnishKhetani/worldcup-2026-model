"""Generate results_log.csv: the model's World Cup 2026 predictions + track record.

Model: a per-team Dixon-Coles (wc_dixon_coles) fit on the weighted international pool
(wc_train). Predictions are strictly walk-forward and lookahead-free:

  * Completed matches are predicted PRE-match -- for each matchday the model is refit on
    only the internationals strictly BEFORE that day, then predicts that day's fixtures.
    So late-round predictions benefit from group-stage form, but never from the match
    being predicted (nor same-day matches). This drives the honest track record.
  * Upcoming fixtures use the model fit on all data available now.

Knockout progression (P(each team advances) through extra time / penalties) comes from
the empirically-calibrated favorite conversion in wc_knockout -- not a 50/50 shootout.

Regenerated from source each run (idempotent), so the scheduled job always reflects the
latest data with no stateful drift.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from wc_config import PROCESSED_DIR
from wc_data import _assemble, load_played_matches, load_remaining_fixtures, count_pending_tbd
from wc_dixon_coles import DixonColesModel
from wc_knockout import p_advance
from wc_train import build_training_set, fixture_neutral, load_martj42

CLASSES = ["H", "D", "A"]


def _fit_asof(reference_date, m) -> DixonColesModel:
    return DixonColesModel().fit(build_training_set(reference_date=reference_date, m=m))


def _row(mt, o, is_ko, m):
    """Assemble one results_log row from a fixture record `mt` and a 90' prediction `o`."""
    p_h, p_d, p_a = o["p_home"], o["p_draw"], o["p_away"]
    pick = CLASSES[int(np.argmax([p_h, p_d, p_a]))]
    row = {
        "match_id": int(mt["match_id"]),
        "date": pd.to_datetime(mt["date"]).date().isoformat(),
        "kickoff_utc": mt.get("kickoff_time_utc", ""),
        "stage": mt["stage"], "is_knockout": bool(is_ko),
        "home_team": mt["home_team"], "away_team": mt["away_team"],
        "p_home": round(p_h, 4), "p_draw": round(p_d, 4), "p_away": round(p_a, 4),
        "pick": pick, "pick_conf": round(max(p_h, p_d, p_a), 4),
        "home_progress": "", "away_progress": "", "progress_pick": "", "progress_conf": "",
        "advanced": "",
    }
    if is_ko:
        adv = p_advance(p_h, p_d, p_a)
        row["home_progress"] = round(adv.p_home_advance, 4)
        row["away_progress"] = round(adv.p_away_advance, 4)
        row["progress_pick"] = (mt["home_team"] if adv.p_home_advance >= adv.p_away_advance
                                else mt["away_team"])
        row["progress_conf"] = round(max(adv.p_home_advance, adv.p_away_advance), 4)
    return row, pick


def build_log() -> pd.DataFrame:
    m = load_martj42()
    kickoff = _assemble().set_index("match_id")["kickoff_time_utc"].to_dict()
    played = load_played_matches()
    played["kickoff_time_utc"] = played["match_id"].map(kickoff)

    rows = []
    # --- completed: walk-forward, one refit per matchday (strictly earlier data) ---
    for date, grp in played.groupby("date"):
        model = _fit_asof(pd.Timestamp(date) - pd.Timedelta(days=1), m)
        for _, mt in grp.iterrows():
            neutral = fixture_neutral(mt["home_team"], mt["away_team"], m)
            o = model.predict_outcome(mt["home_team"], mt["away_team"], neutral)
            row, pick = _row(mt, o, mt["is_knockout"], m)
            row["completed"] = True
            row["actual"] = mt["result90"]
            row["actual_score"] = f"{int(mt['hg90'])}-{int(mt['ag90'])}"
            # Knockout ties are scored on the model's progression call vs who actually
            # advanced (extra time / penalties), not the 90' H/D/A -- otherwise every
            # shootout the model called correctly would still read as a miss. Group
            # matches (and any knockout with an unknown advancer) score on the 90' result.
            if mt["is_knockout"] and mt["advanced"] in ("H", "A") and row["progress_pick"]:
                adv_team = mt["home_team"] if mt["advanced"] == "H" else mt["away_team"]
                row["advanced"] = adv_team
                row["correct"] = bool(row["progress_pick"] == adv_team)
            else:
                row["correct"] = bool(pick == mt["result90"])
            rows.append(row)

    # --- upcoming: model fit on everything available now ---
    fixtures = load_remaining_fixtures()
    if len(fixtures):
        cur = _fit_asof(None, m)
        stages = _assemble().set_index("match_id")
        fixtures["kickoff_time_utc"] = fixtures["match_id"].map(kickoff)
        for _, mt in fixtures.iterrows():
            is_ko = bool(stages.loc[mt["match_id"], "is_knockout"])
            neutral = fixture_neutral(mt["home_team"], mt["away_team"], m)
            o = cur.predict_outcome(mt["home_team"], mt["away_team"], neutral)
            row, _ = _row(mt, o, is_ko, m)
            row["completed"] = False
            row["actual"] = ""
            row["actual_score"] = ""
            row["correct"] = ""
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["completed", "date", "match_id"],
                                        ascending=[True, True, True])
    return df.reset_index(drop=True)


def track_record(df: pd.DataFrame) -> dict:
    done = df[df["completed"] == True]  # noqa: E712
    if done.empty:
        return {"n": 0}
    y = done["actual"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    P = done[["p_home", "p_draw", "p_away"]].to_numpy(float)
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))))
    brier = float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))
    acc = float(done["correct"].mean())
    base = np.array([np.mean(y == i) for i in range(3)])
    base_ll = float(-np.mean(np.log(np.clip(base[y], 1e-15, 1))))
    return {"n": int(len(done)), "accuracy": acc, "log_loss": ll, "brier": brier,
            "base_log_loss": base_ll, "correct": int(done["correct"].sum())}


def main() -> int:
    df = build_log()
    out = PROCESSED_DIR / "results_log.csv"
    df.to_csv(out, index=False)
    tr = track_record(df)
    n_up = int((df["completed"] == False).sum())  # noqa: E712
    print(f"Wrote {out}: {len(df)} matches ({tr.get('n', 0)} completed, {n_up} upcoming).")
    if tr.get("n"):
        print(f"Track record: {tr['correct']}/{tr['n']} correct "
              f"({tr['accuracy']:.1%}), log loss {tr['log_loss']:.3f} "
              f"(base-rate {tr['base_log_loss']:.3f}), Brier {tr['brier']:.3f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
