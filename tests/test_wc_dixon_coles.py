"""Tests for the per-team Dixon-Coles model + its pure goal-math helpers."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wc_dixon_coles import (DixonColesModel, dc_tau, outcome_probs,  # noqa: E402
                            score_matrix)

TOL = 1e-9


def test_dc_tau_matches_definition():
    lam, mu, rho = 1.3, 1.1, 0.1
    assert dc_tau(0, 0, lam, mu, rho) == 1.0 - lam * mu * rho
    assert dc_tau(0, 1, lam, mu, rho) == 1.0 + lam * rho
    assert dc_tau(1, 0, lam, mu, rho) == 1.0 + mu * rho
    assert dc_tau(1, 1, lam, mu, rho) == 1.0 - rho
    assert dc_tau(2, 3, lam, mu, rho) == 1.0


def test_score_matrix_and_outcomes_simplex():
    m = score_matrix(1.6, 1.1, 0.08)
    assert abs(m.sum() - 1.0) < TOL and (m >= 0).all()
    p_h, p_d, p_a = outcome_probs(1.6, 1.1, 0.08)
    assert abs((p_h + p_d + p_a) - 1.0) < TOL and min(p_h, p_d, p_a) >= 0.0


def test_higher_rate_side_favoured_and_rho_moves_draw():
    p_h, _, p_a = outcome_probs(2.0, 0.8, 0.05)
    assert p_h > p_a
    assert abs(outcome_probs(1.2, 1.2, 0.0)[1] - outcome_probs(1.2, 1.2, 0.15)[1]) > 1e-4


def _synthetic(n=400, seed=0, home_edge=0.0):
    """n matches among 6 teams of graded strength; optional real home edge."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(6)]
    strength = {t: s for t, s in zip(teams, np.linspace(0.9, 0.2, 6))}
    rows = []
    for _ in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        neutral = bool(rng.integers(0, 2))
        lh = strength[h] * (1.0 + (home_edge if not neutral else 0.0))
        la = strength[a]
        rows.append(dict(home=h, away=a, home_score=rng.poisson(lh * 2.2),
                         away_score=rng.poisson(la * 2.2), neutral=neutral, weight=1.0))
    return pd.DataFrame(rows)


def test_fit_converges_and_orders_strength():
    m = DixonColesModel().fit(_synthetic())
    assert m.converged and len(m.teams) == 6
    tf = m.team_frame().reset_index(drop=True)
    # strongest synthetic team (T0) should rank above the weakest (T5)
    rank = {t: i for i, t in enumerate(tf["team"])}
    assert rank["T0"] < rank["T5"]
    p = m.predict_proba("T0", "T5", neutral=True)
    assert p[0] > p[2]                       # strong home beats weak away
    assert not m.knows("Nowhere United")     # unseen team flagged


def test_two_sided_home_advantage_recovered():
    m = DixonColesModel().fit(_synthetic(n=800, seed=3, home_edge=0.5))
    assert m.home_adv > 0.0                   # home scores more at a real venue
    lam_home, _ = m.expected_goals("T2", "T3", neutral=False)
    lam_neutral, _ = m.expected_goals("T2", "T3", neutral=True)
    assert lam_home > lam_neutral
