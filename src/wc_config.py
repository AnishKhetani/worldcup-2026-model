"""World Cup 2026 data-pipeline configuration.

The model is trained purely on match results -- no betting odds. Free, public-domain
data sources:
  1. FIFA-World-Cup-2026-Dataset (CC0, updated daily) -- relational current-tournament
     stats, mirrored into data/raw/wc2026/ (see wc_fetch.py).
  2. martj42/international_results (CC0, 1872-present) -- all international matches, the
     per-team Dixon-Coles training pool (see wc_train.py), mirrored into data/raw/intl/.
  3. jfjelstul/worldcup (CC-BY-SA 4.0) -- Fjelstul WC DB, for the knockout-conversion
     calibration (see wc_knockout.py), mirrored into data/raw/fjelstul/.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WC_RAW_DIR = DATA_DIR / "raw" / "wc2026"             # CC0 relational CSVs
PROCESSED_DIR = DATA_DIR / "processed"

for _d in (WC_RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- FIFA-World-Cup-2026-Dataset (CC0-1.0) ----------------------------------
# Raw GitHub content, main branch. Files are small; we mirror the whole relational
# set so joins are reproducible offline.
WC_REPO = "mominullptr/FIFA-World-Cup-2026-Dataset"
WC_BASE_URL = f"https://raw.githubusercontent.com/{WC_REPO}/main"

# CSV files to mirror. The first group is the core relational set; the rest are useful
# adjuncts (readable denormalized matches, lineups, players).
WC_CSV_FILES = [
    "teams.csv",
    "venues.csv",
    "tournament_stages.csv",
    "referees.csv",
    "matches.csv",
    "match_events.csv",
    "match_team_stats.csv",
    # adjuncts:
    "matches_detailed.csv",
    "match_lineups.csv",
    "squads_and_players.csv",
    "player_stats.csv",
]

# Expected columns per file (for schema validation). Validation REPORTS drift rather
# than failing hard -- the upstream dataset updates daily and may add columns.
WC_EXPECTED_SCHEMA = {
    "teams.csv": ["team_id", "team_name", "fifa_code", "group_letter",
                  "confederation", "fifa_ranking_pre_tournament", "elo_rating",
                  "manager_name"],
    "venues.csv": ["venue_id", "stadium_name", "city", "country", "capacity",
                   "latitude", "longitude", "elevation_meters"],
    "tournament_stages.csv": ["stage_id", "stage_name", "is_knockout"],
    "referees.csv": ["referee_id", "name", "country", "avg_cards_per_game"],
    "matches.csv": ["match_id", "date", "kickoff_time_utc", "stage_id", "venue_id",
                    "home_team_id", "away_team_id", "home_score", "away_score",
                    "status", "home_xg", "away_xg", "referee_id"],
    "match_events.csv": ["event_id", "match_id", "minute", "event_type", "team_id",
                         "player_id"],
    "match_team_stats.csv": ["match_id", "team_id", "possession_pct", "total_shots",
                             "shots_on_target", "corners", "fouls", "offsides",
                             "saves"],
}

# Statuses in matches.csv that mean the match has been played and MUST carry a score.
WC_COMPLETED_STATUSES = {"completed", "finished", "full-time", "ft", "played"}

# HTTP settings shared by the fetchers.
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (worldcup-2026-model research)"}
HTTP_TIMEOUT = 60
