"""Generate results_log.csv: the model's World Cup 2026 predictions + track record.

One row per match with both teams known. The model is the all-history global
Dixon-Coles (wc_global / wc_intl_elo): coefficients fit on ~25k pre-tournament
internationals, team strength from a rolling international Elo as of the tournament
start -- so every prediction uses ZERO information from the match it predicts (no
lookahead), for completed and upcoming matches alike.

For each match we record the normal-time W/D/L probabilities, the confident pick
(argmax), and -- for knockouts -- each team's progression probability (draws resolve
via extra time / penalties). Completed matches also carry the actual 90' result and
whether the pick was right, which drives the track-record scoreboard on the site.

The file is regenerated from source each run (idempotent), so the scheduled job always
reflects the latest data with no stateful drift.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from wc_config import PROCESSED_DIR
from wc_data import _assemble, load_played_matches  # noqa: F401  (_assemble reused)
from wc_dixon_coles import DixonColesStrength
from wc_intl_elo import build

CLASSES = ["H", "D", "A"]
PICK_LABEL = {"H": "home", "D": "draw", "A": "away"}


def _all_known_matches() -> pd.DataFrame:
    """Every match (completed or scheduled) whose two teams are already known."""
    m = _assemble()
    m = m[m["home_strength"].notna() & m["away_strength"].notna()].copy()
    m["completed"] = m["status"].astype(str).str.lower().eq("completed")
    return m


def build_log() -> pd.DataFrame:
    intl = build()
    model = DixonColesStrength().fit(intl.fit_matches)   # weight/neutral cols auto-used

    played = load_played_matches().set_index("match_id")
    rows = []
    for _, m in _all_known_matches().iterrows():
        sh = intl.wc_strength(m["home_team"])
        sa = intl.wc_strength(m["away_team"])
        if sh is None or sa is None:
            continue
        # neutral-venue: WC knockouts are neutral, and treating the group stage the same
        # keeps one consistent basis (host-nation home edge is not modelled).
        p_h, p_d, p_a = model.predict_outcome(sh, sa, neutral=True)
        prog_h, prog_a = model.predict_progression(sh, sa, neutral=True)
        pick = CLASSES[int(np.argmax([p_h, p_d, p_a]))]
        is_ko = bool(m["is_knockout"])
        row = {
            "match_id": int(m["match_id"]),
            "date": pd.to_datetime(m["date"]).date().isoformat(),
            "kickoff_utc": m.get("kickoff_time_utc"),
            "stage": m["stage"], "is_knockout": is_ko,
            "home_team": m["home_team"], "away_team": m["away_team"],
            "completed": bool(m["completed"]),
            "p_home": round(p_h, 4), "p_draw": round(p_d, 4), "p_away": round(p_a, 4),
            "pick": pick, "pick_conf": round(max(p_h, p_d, p_a), 4),
            "home_progress": round(prog_h, 4) if is_ko else "",
            "away_progress": round(prog_a, 4) if is_ko else "",
            # progression favourite (the confident knockout call)
            "progress_pick": (m["home_team"] if prog_h >= prog_a else m["away_team"]) if is_ko else "",
            "progress_conf": round(max(prog_h, prog_a), 4) if is_ko else "",
        }
        if m["completed"] and m["match_id"] in played.index:
            actual = played.loc[m["match_id"], "result90"]
            row["actual"] = actual
            row["actual_score"] = f"{int(played.loc[m['match_id'],'hg90'])}-" \
                                  f"{int(played.loc[m['match_id'],'ag90'])}"
            row["correct"] = bool(pick == actual)
        else:
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
    # base-rate baseline (naive: the completed-match H/D/A frequencies)
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
