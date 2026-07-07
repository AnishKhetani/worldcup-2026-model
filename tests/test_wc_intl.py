"""Tests for the all-international-history layer: match-importance weighting, the Elo
goal-difference multiplier, the weighted/neutral fit path, and (if the cached results
set is present) that the rolling Elo is sane and every WC team matches.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wc_dixon_coles import DixonColesStrength  # noqa: E402
from wc_intl_elo import (RESULTS_CSV, _goal_diff_multiplier,  # noqa: E402
                         importance_index)


def test_importance_index_mapping():
    assert importance_index("Friendly") == 20
    assert importance_index("FIFA World Cup") == 60
    assert importance_index("FIFA World Cup qualification") == 40   # quali beats "world cup"
    assert importance_index("UEFA Nations League") == 40
    assert importance_index("UEFA Euro") == 50
    assert importance_index("Copa América") == 50
    assert importance_index("Some Local Cup") == 30


def test_goal_diff_multiplier():
    assert _goal_diff_multiplier(0) == 1.0
    assert _goal_diff_multiplier(1) == 1.0
    assert _goal_diff_multiplier(2) == 1.5
    assert _goal_diff_multiplier(3) == (11 + 3) / 8
    assert _goal_diff_multiplier(-4) == (11 + 4) / 8   # uses |gd|


def test_weighted_neutral_fit_recovers_home_edge():
    # Give non-neutral matches a genuine home scoring edge; the neutral-aware weighted
    # fit should recover gamma > 0, so the home rate exceeds the neutral rate.
    rng = np.random.default_rng(1)
    n = 400
    neutral = rng.integers(0, 2, n).astype(bool)
    sh = rng.normal(0.3, 0.5, n)
    sa = rng.normal(-0.3, 0.5, n)
    home_mean = 1.3 + 0.6 * (~neutral)          # +0.6 goals at a real home venue
    df = pd.DataFrame({
        "home_strength": sh, "away_strength": sa,
        "hg90": rng.poisson(home_mean), "ag90": rng.poisson(1.1, n),
        "neutral": neutral, "weight": rng.uniform(0.2, 1.0, n),
    })
    m = DixonColesStrength().fit(df)  # auto-uses the neutral + weight columns
    assert m.fitted and m.gamma > 0.0
    lam_home, _ = m.rates(0.0, 0.0, neutral=False)
    lam_neutral, _ = m.rates(0.0, 0.0, neutral=True)
    assert lam_home > lam_neutral


def test_weights_change_the_fit():
    rng = np.random.default_rng(2)
    n = 200
    base = pd.DataFrame({
        "home_strength": rng.normal(0, 1, n), "away_strength": rng.normal(0, 1, n),
        "hg90": rng.poisson(1.5, n), "ag90": rng.poisson(1.2, n),
    })
    m_flat = DixonColesStrength().fit(base, weights=np.ones(n))
    w = np.ones(n); w[: n // 2] = 5.0            # up-weight the first half
    m_wt = DixonColesStrength().fit(base, weights=w)
    assert abs(m_flat.a - m_wt.a) + abs(m_flat.mu - m_wt.mu) > 1e-6


@pytest.mark.skipif(not RESULTS_CSV.exists(),
                    reason="international results set not downloaded")
def test_build_is_sane_and_matches_all_wc_teams():
    from wc_data import team_table
    from wc_intl_elo import NAME_ALIAS, build
    intl = build()
    # every WC 2026 team resolves to a pre-tournament strength
    missing = [r["team_name"] for _, r in team_table().iterrows()
               if intl.wc_strength(r["team_name"]) is None]
    assert missing == []
    # fit weights are positive and bounded; neutral/weight columns exist
    assert {"weight", "neutral", "home_strength", "away_strength"} <= set(intl.fit_matches)
    assert (intl.fit_matches["weight"] > 0).all()
    assert intl.fit_matches["weight"].max() <= 1.0 + 1e-9
    # rolling Elo agrees strongly with the dataset's own pre-tournament Elo
    t = team_table()
    mine = [intl.pre_tournament[NAME_ALIAS.get(n, n)] for n in t["team_name"]]
    corr = np.corrcoef(mine, t["elo_rating"])[0, 1]
    assert corr > 0.8
