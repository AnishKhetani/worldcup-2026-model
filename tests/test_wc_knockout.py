"""Tests for the empirical knockout favorite-conversion (change #1)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import wc_knockout as kc  # noqa: E402
from wc_knockout import (FJELSTUL_CSV, conditional_conversion,  # noqa: E402
                         p_advance, win_share)
from wc_train import RESULTS_CSV  # noqa: E402


def test_conversion_boundary_and_monotonic():
    assert conditional_conversion(0.5) == 0.5          # even tie -> coin flip
    assert conditional_conversion(0.49) == 0.5         # dog side clamped
    vals = [conditional_conversion(q) for q in (0.55, 0.7, 0.85, 0.95)]
    assert all(0.5 < v < 1.0 for v in vals)
    assert vals == sorted(vals)                        # bigger favorite -> higher
    # heavy compression: a q=0.85 favorite converts well below 0.85
    assert conditional_conversion(0.85) < 0.75


def test_win_share():
    assert abs(win_share(0.6, 0.2) - 0.75) < 1e-9
    assert win_share(0.0, 0.0) == 0.5


def test_p_advance_sums_to_one_and_favours_stronger():
    adv = p_advance(0.60, 0.25, 0.15)
    assert abs((adv.p_home_advance + adv.p_away_advance) - 1.0) < 1e-9
    assert adv.p_home_advance > adv.p_away_advance
    even = p_advance(0.40, 0.20, 0.40)                 # symmetric fixture
    assert abs(even.p_home_advance - 0.5) < 1e-9


@pytest.mark.skipif(not (FJELSTUL_CSV.exists() and RESULTS_CSV.exists()),
                    reason="Fjelstul / martj42 data not downloaded")
def test_gamma_refit_reproduces_pinned():
    r = kc.fit_from_data()
    assert abs(r["gamma"] - kc.GAMMA) < 0.03           # pinned constant still holds
    assert r["n"] > 70 and 0.55 < r["fav_adv"] < 0.70  # ~62% of drawn ties
