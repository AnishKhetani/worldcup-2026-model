"""Load + validate the World Cup data sources and print a per-table summary.

Checks:
  * row counts for every CC0 table;
  * matches.csv match_id is unique (primary key);
  * schema check -- every expected column present (extras are reported, not an error:
    the CC0 upstream updates daily and may add fields);
  * flag any COMPLETED match with a missing score (Scheduled/unplayed rows are expected
    to be null and are NOT flagged);
  * light referential integrity (team_id foreign keys resolve);
  * the international results history is present and well-formed.

Read-only: it never re-pulls or mutates the mirror. Non-zero exit on a hard invariant
failure (missing file, duplicate match_id, completed-but-scoreless match).
"""
from __future__ import annotations

import sys

import pandas as pd

from wc_config import (WC_COMPLETED_STATUSES, WC_CSV_FILES, WC_EXPECTED_SCHEMA,
                       WC_RAW_DIR)

PASS, WARN, FAIL = "  [ok]", "  [warn]", "  [FAIL]"


def _is_completed(status) -> bool:
    return isinstance(status, str) and status.strip().lower() in WC_COMPLETED_STATUSES


def validate_wc_csvs(problems: list[str]) -> pd.DataFrame | None:
    print("=" * 70)
    print("SOURCE 1 -- FIFA-World-Cup-2026-Dataset (CC0-1.0)")
    print("=" * 70)
    frames = {}
    for fname in WC_CSV_FILES:
        path = WC_RAW_DIR / fname
        if not path.exists():
            print(f"{FAIL} {fname}: file missing (run wc_fetch.py)")
            problems.append(f"{fname} missing")
            continue
        df = pd.read_csv(path)
        frames[fname] = df
        print(f"\n{fname}  ({len(df)} rows, {df.shape[1]} cols)")

        # schema check: expected columns must all be present; extras are informational
        expected = WC_EXPECTED_SCHEMA.get(fname)
        if expected:
            missing = [c for c in expected if c not in df.columns]
            extra = [c for c in df.columns if c not in expected]
            if missing:
                print(f"{FAIL} missing expected columns: {missing}")
                problems.append(f"{fname} missing columns {missing}")
            else:
                print(f"{PASS} all {len(expected)} expected columns present")
            if extra:
                print(f"{WARN} extra columns (kept): {extra}")

    # matches.csv-specific invariants
    m = frames.get("matches.csv")
    if m is not None:
        print("\n--- matches.csv invariants ---")
        if m["match_id"].is_unique:
            print(f"{PASS} match_id unique ({len(m)} matches)")
        else:
            dupes = m["match_id"][m["match_id"].duplicated()].tolist()
            print(f"{FAIL} duplicate match_id: {dupes}")
            problems.append(f"duplicate match_id {dupes}")

        counts = m["status"].value_counts(dropna=False)
        print(f"  status: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

        completed = m[m["status"].apply(_is_completed)]
        scoreless = completed[completed["home_score"].isna()
                              | completed["away_score"].isna()]
        if len(scoreless):
            print(f"{FAIL} {len(scoreless)} COMPLETED match(es) missing a score: "
                  f"{scoreless['match_id'].tolist()}")
            problems.append(f"{len(scoreless)} completed matches missing score")
        else:
            print(f"{PASS} all {len(completed)} completed matches have both scores")

        unplayed = m[~m["status"].apply(_is_completed)]
        print(f"  remaining (unplayed): {len(unplayed)} matches -- the prediction targets")

        teams = frames.get("teams.csv")
        if teams is not None:
            tid = set(teams["team_id"])
            bad = m[(~m["home_team_id"].isin(tid) & m["home_team_id"].notna())
                    | (~m["away_team_id"].isin(tid) & m["away_team_id"].notna())]
            tag = PASS if bad.empty else WARN
            print(f"{tag} team_id FKs resolve for {len(m) - len(bad)}/{len(m)} matches")

    mts = frames.get("match_team_stats.csv")
    if m is not None and mts is not None:
        n_completed = int(m["status"].apply(_is_completed).sum())
        exp = n_completed * 2
        tag = PASS if len(mts) == exp else WARN
        print(f"\n{tag} match_team_stats: {len(mts)} rows "
              f"(expected ~{exp} = {n_completed} completed x 2 teams)")

    return m


def validate_intl(problems: list[str]) -> None:
    from wc_intl_elo import RESULTS_CSV
    print("\n" + "=" * 70)
    print("SOURCE 2 -- international results history (martj42, CC0)")
    print("=" * 70)
    if not RESULTS_CSV.exists():
        print(f"{FAIL} results.csv missing (run wc_fetch.py)")
        problems.append("intl results.csv missing")
        return
    df = pd.read_csv(RESULTS_CSV)
    need = {"date", "home_team", "away_team", "home_score", "away_score",
            "tournament", "neutral"}
    missing = need - set(df.columns)
    if missing:
        print(f"{FAIL} missing expected columns: {sorted(missing)}")
        problems.append(f"intl missing columns {sorted(missing)}")
    else:
        print(f"{PASS} {len(df):,} matches, all {len(need)} expected columns present "
              f"({df['date'].min()} .. {df['date'].max()})")


def main() -> int:
    problems: list[str] = []
    validate_wc_csvs(problems)
    validate_intl(problems)

    print("\n" + "=" * 70)
    if problems:
        print(f"VALIDATION: {len(problems)} issue(s) flagged:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("VALIDATION: all checks passed.")
    print("=" * 70)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
