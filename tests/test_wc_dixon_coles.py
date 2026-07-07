"""Sanity tests for the strength-prior Dixon-Coles goal-rate model (WC 2026).

Run: cd src && python -m pytest ../tests -q

Guards the invariants the backtest and predictions rely on: the low-score correction
matches the Dixon-Coles definition, score/outcome probabilities form a valid simplex,
stronger teams are favoured, the home bump does something, and progression is a proper
no-draw redistribution (the two teams' progression probabilities sum to 1, and are 50/50
for an even neutral tie).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wc_dixon_coles import (DixonColesStrength, dc_tau, outcome_probs,  # noqa: E402
                            progression_probs, score_matrix)

TOL = 1e-9


def test_dc_tau_matches_definition():
    lam, mu, rho = 1.3, 1.1, 0.1
    assert dc_tau(0, 0, lam, mu, rho) == 1.0 - lam * mu * rho
    assert dc_tau(0, 1, lam, mu, rho) == 1.0 + lam * rho
    assert dc_tau(1, 0, lam, mu, rho) == 1.0 + mu * rho
    assert dc_tau(1, 1, lam, mu, rho) == 1.0 - rho
    # cells outside the 2x2 low-score block are untouched
    assert dc_tau(2, 0, lam, mu, rho) == 1.0
    assert dc_tau(3, 2, lam, mu, rho) == 1.0


def test_score_matrix_and_outcomes_are_simplex():
    m = score_matrix(1.6, 1.2, 0.08)
    assert abs(m.sum() - 1.0) < TOL
    assert (m >= 0).all()
    p_h, p_d, p_a = outcome_probs(1.6, 1.2, 0.08)
    assert abs((p_h + p_d + p_a) - 1.0) < TOL
    assert min(p_h, p_d, p_a) >= 0.0


def test_rho_zero_recovers_independent_poisson_draw():
    # With rho=0 the draw prob equals the independent-Poisson diagonal.
    lam, mu = 1.4, 1.1
    _, p_d, _ = outcome_probs(lam, mu, 0.0)
    ks = np.arange(11)
    from math import exp, factorial
    pk = lambda k, r: exp(-r) * r ** k / factorial(k)
    indep_diag = sum(pk(k, lam) * pk(k, mu) for k in ks)
    assert abs(p_d - indep_diag) < 1e-6


def test_low_score_correction_moves_draw_prob():
    # A non-zero rho should change the outcome split vs independence.
    base = outcome_probs(1.2, 1.2, 0.0)
    bumped = outcome_probs(1.2, 1.2, 0.15)
    assert abs(base[1] - bumped[1]) > 1e-4


def test_higher_rate_side_is_favoured():
    p_h, _, p_a = outcome_probs(2.0, 0.9, 0.05)
    assert p_h > p_a


def test_progression_sums_to_one_and_no_draw():
    ph, pa = progression_probs(1.7, 1.0, 0.05)
    assert abs((ph + pa) - 1.0) < 1e-9
    assert ph > pa  # the higher-rate side is more likely to go through


def test_even_neutral_tie_progresses_50_50():
    # Equal rates -> symmetric -> a coin flip to advance.
    ph, pa = progression_probs(1.3, 1.3, 0.05)
    assert abs(ph - 0.5) < 1e-9
    assert abs(pa - 0.5) < 1e-9


def test_model_rates_and_home_bump():
    m = DixonColesStrength(mu=0.1, gamma=0.25, a=0.3, d=0.3, rho=0.0)
    lam_home, _ = m.rates(0.0, 0.0, neutral=False)
    lam_neutral, _ = m.rates(0.0, 0.0, neutral=True)
    assert lam_home > lam_neutral  # home bump raises the home rate
    # stronger team (higher strength) scores more, concedes less
    lam_s, mu_s = m.rates(1.5, -1.5, neutral=True)
    assert lam_s > mu_s


def test_fit_recovers_strength_direction():
    # Synthetic: home always much stronger and wins big -> fitted model favours strength.
    rng = np.random.default_rng(0)
    n = 60
    sh = rng.normal(1.0, 0.3, n)
    sa = rng.normal(-1.0, 0.3, n)
    df = pd.DataFrame({
        "home_strength": sh, "away_strength": sa,
        "hg90": rng.poisson(2.5, n), "ag90": rng.poisson(0.6, n),
    })
    m = DixonColesStrength().fit(df)
    assert m.fitted
    p_h, _, p_a = m.predict_outcome(1.2, -1.2, neutral=True)
    assert p_h > p_a
