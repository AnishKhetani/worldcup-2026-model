"""Strength-prior Dixon-Coles goal-rate model for World Cup 2026 knockout matches.

Why this shape (and not a textbook per-team Dixon-Coles):
  A classic Dixon-Coles fits an attack and a defence parameter for every team. With 48
  teams and ~3 games each, 96 free team parameters are hopelessly under-identified. So
  here the team parameters are NOT free: a team's attack and defence are TIED to its
  pre-tournament Elo strength `s` (a z-score; see wc_data). Only FIVE global
  coefficients are fit from the played matches:

      log(lambda_home) = mu + gamma*home + a*s_home - d*s_away        (home goal rate)
      log(lambda_away) = mu          + a*s_away - d*s_home            (away goal rate)

    mu    baseline log goal-rate      a  how much own strength lifts scoring
    gamma home-field bump             d  how much opponent strength suppresses scoring
    rho   Dixon-Coles low-score correction (the standard 0-0/1-0/0-1/1-1 adjustment)

  This is a genuine Poisson goal-rate model with the standard low-scoring-draw
  correction; it just sources team strength from the Elo prior instead of estimating it
  from this tournament's small sample, exactly as required. If a == d it collapses to a
  pure supremacy model; letting them differ lets the played matches say whether strong
  teams win more by scoring more or by conceding less.

Home advantage: `gamma` is fit from the played matches (group games at host venues do
carry a real edge). Remaining knockout ties are predicted NEUTRAL (home=0 for both) —
the "home" side of a neutral-venue knockout is an administrative bracket label, not a
venue. Callers pass neutral=True for those.

Progression (knockouts have no draw): a tie at 90' goes to extra time — modelled as the
same rates scaled to 30 minutes (lambda/3) with the same rho — and, if still level, to
penalties (50/50). So P(team progresses) is strength-tilted through extra time rather
than a flat coin flip, and P(home progresses) + P(away progresses) = 1 by construction.

The math functions are pure and unit-tested (tests/test_wc_dixon_coles.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

MAX_GOALS = 10          # score-matrix truncation for normal time
MAX_GOALS_ET = 6        # extra time is low-scoring; a smaller grid is plenty
ET_FRACTION = 1.0 / 3.0  # 30 added minutes ~ a third of a 90' match's goal expectation


# --- pure Poisson / Dixon-Coles math -----------------------------------------
def _poisson_pmf(k: np.ndarray, lam: float) -> np.ndarray:
    from math import lgamma
    k = np.asarray(k, float)
    logp = -lam + k * np.log(max(lam, 1e-12)) - np.array([lgamma(int(i) + 1) for i in k])
    return np.exp(logp)


def dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score dependence factor for the four adjusted cells."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam: float, mu: float, rho: float,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    """P(home=x, away=y) grid with the DC correction applied and renormalised."""
    ks = np.arange(max_goals + 1)
    ph = _poisson_pmf(ks, lam)
    pa = _poisson_pmf(ks, mu)
    mat = np.outer(ph, pa)
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= dc_tau(x, y, lam, mu, rho)
    s = mat.sum()
    return mat / s if s > 0 else mat


def outcome_probs(lam: float, mu: float, rho: float,
                  max_goals: int = MAX_GOALS) -> tuple[float, float, float]:
    """(P home win, P draw, P away win) in normal time."""
    m = score_matrix(lam, mu, rho, max_goals)
    p_home = np.tril(m, -1).sum()   # x > y
    p_away = np.triu(m, 1).sum()    # x < y
    p_draw = np.trace(m)            # x == y
    return float(p_home), float(p_draw), float(p_away)


def progression_probs(lam: float, mu: float, rho: float) -> tuple[float, float]:
    """(P home progresses, P away progresses) through 90' -> ET -> penalties."""
    p_h, p_d, p_a = outcome_probs(lam, mu, rho)
    # extra time: same rates scaled to 30 minutes, same rho
    eh, ed, ea = outcome_probs(lam * ET_FRACTION, mu * ET_FRACTION, rho, MAX_GOALS_ET)
    home_tiebreak = eh + ed * 0.5   # win in ET, or level-then-penalties (coin flip)
    away_tiebreak = ea + ed * 0.5
    p_home_prog = p_h + p_d * home_tiebreak
    p_away_prog = p_a + p_d * away_tiebreak
    return float(p_home_prog), float(p_away_prog)


# --- fittable strength-prior model -------------------------------------------
@dataclass
class DixonColesStrength:
    mu: float = 0.0
    gamma: float = 0.2
    a: float = 0.3
    d: float = 0.3
    rho: float = 0.0
    fitted: bool = False

    def rates(self, s_home: float, s_away: float,
              neutral: bool = False) -> tuple[float, float]:
        """Expected (home, away) normal-time goals for a strength matchup."""
        home = 0.0 if neutral else 1.0
        lam = np.exp(self.mu + self.gamma * home + self.a * s_home - self.d * s_away)
        mu = np.exp(self.mu + self.a * s_away - self.d * s_home)
        return float(lam), float(mu)

    # -- likelihood (vectorised across matches) ------------------------------
    @staticmethod
    def _nll(params, sh, sa, hg, ag, home_ind, w) -> float:
        """Weighted negative log-likelihood. `home_ind` is 1 for the home side at a
        real venue and 0 at a neutral one (so gamma only helps a genuine home team);
        `w` weights each match (e.g. importance x recency for the historical fit)."""
        from scipy.special import gammaln
        mu0, gamma, a, d, rho = params
        lam = np.exp(mu0 + gamma * home_ind + a * sh - d * sa)
        mu = np.exp(mu0 + a * sa - d * sh)
        log_ph = -lam + hg * np.log(lam) - gammaln(hg + 1.0)
        log_pa = -mu + ag * np.log(mu) - gammaln(ag + 1.0)
        # Dixon-Coles correction, vectorised over the four adjusted cells
        tau = np.ones_like(lam)
        m00 = (hg == 0) & (ag == 0); tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        m01 = (hg == 0) & (ag == 1); tau[m01] = 1.0 + lam[m01] * rho
        m10 = (hg == 1) & (ag == 0); tau[m10] = 1.0 + mu[m10] * rho
        m11 = (hg == 1) & (ag == 1); tau[m11] = 1.0 - rho
        logp = w * (log_ph + log_pa + np.log(np.clip(tau, 1e-12, None)))
        return -float(np.sum(logp))

    def fit(self, matches, weights=None, neutral=None) -> "DixonColesStrength":
        """MLE on a matches frame (needs home/away_strength, hg90, ag90).

        `weights` (per-match) enables the importance/recency-weighted historical fit;
        `neutral` (per-match bool, or a 'neutral' column) zeroes the home bump for
        neutral-venue matches. Both default to the plain unweighted, all-home case so
        existing callers are unchanged.
        """
        sh = matches["home_strength"].to_numpy(float)
        sa = matches["away_strength"].to_numpy(float)
        hg = matches["hg90"].to_numpy(float)
        ag = matches["ag90"].to_numpy(float)
        if neutral is None:
            neutral = matches["neutral"] if "neutral" in matches else np.zeros(len(sh))
        home_ind = 1.0 - np.asarray(neutral, float)
        if weights is None:
            weights = matches["weight"] if "weight" in matches else np.ones(len(sh))
        w = np.asarray(weights, float)
        init = np.array([self.mu or 0.0, self.gamma, self.a, self.d, self.rho])
        res = minimize(self._nll, init, args=(sh, sa, hg, ag, home_ind, w),
                       method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 6000})
        self.mu, self.gamma, self.a, self.d, self.rho = (float(v) for v in res.x)
        self.fitted = True
        return self

    # -- prediction ----------------------------------------------------------
    def predict_outcome(self, s_home, s_away, neutral=False):
        lam, mu = self.rates(s_home, s_away, neutral)
        return outcome_probs(lam, mu, self.rho)

    def predict_progression(self, s_home, s_away, neutral=True):
        lam, mu = self.rates(s_home, s_away, neutral)
        return progression_probs(lam, mu, self.rho)
