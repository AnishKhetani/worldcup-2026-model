"""Tests for the weighted international training set (change #3)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import wc_train  # noqa: E402
from wc_train import RESULTS_CSV, tournament_tier  # noqa: E402


def test_tournament_tier_mapping():
    assert tournament_tier("FIFA World Cup", 2026, current_major_year=2026) == 1.5
    assert tournament_tier("FIFA World Cup", 2014, current_major_year=2026) == 1.0
    assert tournament_tier("FIFA World Cup qualification", 2025, 2026) == 0.6
    assert tournament_tier("UEFA Nations League", 2025, 2026) == 0.6
    assert tournament_tier("Friendly", 2025, 2026) == 0.25
    assert tournament_tier("UEFA Euro", 2024, 2026) == 1.0
    assert tournament_tier("Some Local Cup", 2025, 2026) == 0.5


@pytest.mark.skipif(not RESULTS_CSV.exists(), reason="martj42 results not downloaded")
def test_build_training_set_weighted_and_no_lookahead():
    ref = "2026-06-11"
    tr = wc_train.build_training_set(reference_date=ref)
    assert len(tr) > 5000
    assert (tr["match_dt"] <= ref).all()                 # no lookahead
    assert (tr["weight"] > 0).all() and tr["weight"].max() <= 1.5 + 1e-9
    assert {"home", "away", "home_score", "away_score", "neutral", "weight"} <= set(tr)
    # the live-WC boost is present in the sample
    assert (tr["tier"] == 1.5).any()


@pytest.mark.skipif(not RESULTS_CSV.exists(), reason="martj42 results not downloaded")
def test_all_wc_teams_reconcile_and_neutral_flag():
    rec = wc_train.reconciliation()
    assert rec["n_matched"] == rec["n_total"] and rec["unmatched"] == []
    assert isinstance(wc_train.fixture_neutral("Argentina", "Egypt"), bool)
