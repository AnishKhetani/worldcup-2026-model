"""Per-team Dixon-Coles match-outcome model (change #2).

Each team gets its OWN attack and defence parameter, fit by weighted maximum-likelihood
on the full international pool (wc_train.build_training_set) -- not collapsed onto a
single Elo strength scalar. This separates a high-scoring-but-leaky side from a solid
defensive one of equal net strength, which a one-number strength model cannot.

Goal model (home team i vs away team j):
    log lambda_home = mu + home_adv*(non-neutral) + neutral_edge*(neutral)
                         + attack[i] - defense[j]
    log lambda_away = mu - home_def*(non-neutral)
                         + attack[j] - defense[i]
with the Dixon-Coles low-score correction tau(x, y; rho).

Venue advantage is TWO-SIDED and jointly estimated, keyed by the `neutral` flag:
  * home_adv    -- home-scoring boost on non-neutral matches.
  * home_def    -- away-scoring SUPPRESSION on non-neutral matches (the half a
                   one-sided spec misses; "away sides create less at a hostile venue"
                   is what actually lifts the home win probability into calibration).
  * neutral_edge -- a small, ridge-shrunk seed edge for the nominally-listed home team
                   on a neutral venue (World Cup ties are neutral, so this is ~0).

Each match carries a `weight` (Dixon-Coles time decay x tournament tier; see wc_train).
A light ridge (Gaussian prior on attack/defence) stabilises sparse teams. The negative
log-posterior and its gradient are analytic, so walk-forward refits are fast.

Ported from the author's separate betting-model codebase (the `wc26bet` Dixon-Coles),
adapted to this repo and stripped of any odds/provenance coupling. Progression (extra
time / penalties) is layered on top in wc_knockout.py, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import poisson

OUTCOMES = ("H", "D", "A")
MAX_GOALS = 12


# --- pure Dixon-Coles math (import-safe, unit-tested) ------------------------
def dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam_h: float, lam_a: float, rho: float,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    px = poisson.pmf(np.arange(max_goals + 1), lam_h)
    py = poisson.pmf(np.arange(max_goals + 1), lam_a)
    mat = np.outer(px, py)
    mat[0, 0] *= 1.0 - lam_h * lam_a * rho
    mat[0, 1] *= 1.0 + lam_h * rho
    mat[1, 0] *= 1.0 + lam_a * rho
    mat[1, 1] *= 1.0 - rho
    mat = np.clip(mat, 0.0, None)
    s = mat.sum()
    return mat / s if s > 0 else mat


def outcome_probs(lam_h: float, lam_a: float, rho: float) -> tuple[float, float, float]:
    m = score_matrix(lam_h, lam_a, rho)
    return (float(np.tril(m, -1).sum()),   # home win (x > y)
            float(np.trace(m)),            # draw
            float(np.triu(m, 1).sum()))    # away win (x < y)


# --- fittable per-team model -------------------------------------------------
@dataclass
class DixonColesModel:
    prior_sd: float = 1.0               # ridge on attack/defence (data carries signal)
    seed_prior_sd: float = 0.15         # tighter ridge on the neutral seed-edge term
    rho_bounds: tuple[float, float] = (-0.2, 0.2)
    max_goals: int = MAX_GOALS
    uncertainty_threshold: int = 3

    teams: list = field(default_factory=list)
    _index: dict = field(default_factory=dict)
    attack: np.ndarray | None = None
    defense: np.ndarray | None = None
    mu: float = 0.0
    home_adv: float = 0.0
    home_def: float = 0.0
    neutral_edge: float = 0.0
    rho: float = 0.0
    matches_played: dict = field(default_factory=dict)
    n_train: int = 0
    converged: bool = False

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "home" not in df.columns and "home_team_id" in df.columns:
            df = df.rename(columns={"home_team_id": "home", "away_team_id": "away"})
        df = df.dropna(subset=["home", "away", "home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        if "neutral" not in df.columns:
            df["neutral"] = False
        if "weight" not in df.columns:
            df["weight"] = 1.0
        return df

    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        df = self._prep(df)
        teams = sorted(set(df["home"]) | set(df["away"]), key=str)
        self.teams = teams
        self._index = {t: k for k, t in enumerate(teams)}
        n = len(teams)
        self.n_train = len(df)

        hi = df["home"].map(self._index).to_numpy()
        ai = df["away"].map(self._index).to_numpy()
        x = df["home_score"].to_numpy(float)
        y = df["away_score"].to_numpy(float)
        w = df["weight"].to_numpy(float)
        nn = (~df["neutral"].to_numpy(bool)).astype(float)   # 1 = home-adv applies
        nz = 1.0 - nn                                        # 1 = neutral -> seed edge
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        ridge = 1.0 / (self.prior_sd ** 2)
        seed_ridge = 1.0 / (self.seed_prior_sd ** 2)

        # param layout: [mu, home_adv, home_def, seed_edge, rho, attack(n), defense(n)]
        def nll_and_grad(p):
            mu, hadv, hdef, seed, rho = p[0], p[1], p[2], p[3], p[4]
            att = p[5:5 + n]
            dfn = p[5 + n:5 + 2 * n]
            loglh = mu + hadv * nn + seed * nz + att[hi] - dfn[ai]
            logla = mu - hdef * nn + att[ai] - dfn[hi]
            lh = np.exp(loglh)
            la = np.exp(logla)
            ll = np.sum(w * (x * loglh - lh + y * logla - la))
            tau = np.ones_like(lh)
            tau[m00] = 1.0 - lh[m00] * la[m00] * rho
            tau[m01] = 1.0 + lh[m01] * rho
            tau[m10] = 1.0 + la[m10] * rho
            tau[m11] = 1.0 - rho
            tau = np.clip(tau, 1e-9, None)
            ll += np.sum(w * np.log(tau))
            penalty = 0.5 * ridge * (att @ att + dfn @ dfn) + 0.5 * seed_ridge * seed ** 2
            nll = -ll + penalty

            g_lh = w * (x - lh)
            g_la = w * (y - la)
            t_lh = np.zeros_like(lh)
            t_lh[m00] = -lh[m00] * la[m00] * rho / tau[m00]
            t_lh[m01] = lh[m01] * rho / tau[m01]
            t_la = np.zeros_like(la)
            t_la[m00] = -lh[m00] * la[m00] * rho / tau[m00]
            t_la[m10] = la[m10] * rho / tau[m10]
            g_lh = g_lh + w * t_lh
            g_la = g_la + w * t_la

            G_att = np.bincount(hi, g_lh, n) + np.bincount(ai, g_la, n)
            G_dfn = np.bincount(ai, g_lh, n) + np.bincount(hi, g_la, n)
            grad = np.empty_like(p)
            grad[0] = -np.sum(g_lh + g_la)
            grad[1] = -np.sum(g_lh * nn)
            grad[2] = np.sum(g_la * nn)
            grad[3] = -np.sum(g_lh * nz) + seed_ridge * seed
            drho = np.zeros_like(lh)
            drho[m00] = -lh[m00] * la[m00] / tau[m00]
            drho[m01] = lh[m01] / tau[m01]
            drho[m10] = la[m10] / tau[m10]
            drho[m11] = -1.0 / tau[m11]
            grad[4] = -np.sum(w * drho)
            grad[5:5 + n] = -G_att + ridge * att
            grad[5 + n:5 + 2 * n] = G_dfn + ridge * dfn
            return nll, grad

        avg = max((x * w).sum() / w.sum() if w.sum() else 1.0, 0.2)
        p0 = np.zeros(5 + 2 * n)
        p0[0] = np.log(avg)
        p0[1] = 0.25
        bounds = ([(None, None)] * 4 + [self.rho_bounds] + [(None, None)] * (2 * n))
        res = optimize.minimize(nll_and_grad, p0, jac=True, method="L-BFGS-B",
                                bounds=bounds, options={"maxiter": 1000, "ftol": 1e-10})
        self.converged = bool(res.success)
        p = res.x
        self.mu, self.home_adv, self.home_def = float(p[0]), float(p[1]), float(p[2])
        self.neutral_edge, self.rho = float(p[3]), float(p[4])
        self.attack = p[5:5 + n].copy()
        self.defense = p[5 + n:5 + 2 * n].copy()
        counts = pd.concat([df["home"], df["away"]]).value_counts()
        self.matches_played = {t: int(counts.get(t, 0)) for t in teams}
        return self

    # -- prediction ----------------------------------------------------------
    def _strength(self, team) -> tuple[float, float]:
        k = self._index.get(team)
        if k is None or self.attack is None:
            return 0.0, 0.0
        return float(self.attack[k]), float(self.defense[k])

    def knows(self, team) -> bool:
        return team in self._index

    def expected_goals(self, home, away, neutral: bool = False) -> tuple[float, float]:
        ah, dh = self._strength(home)
        aa, da = self._strength(away)
        boost = self.neutral_edge if neutral else self.home_adv
        suppress = 0.0 if neutral else self.home_def
        lam_h = float(np.exp(self.mu + boost + ah - da))
        lam_a = float(np.exp(self.mu - suppress + aa - dh))
        return lam_h, lam_a

    def predict_outcome(self, home, away, neutral: bool = False) -> dict:
        lh, la = self.expected_goals(home, away, neutral)
        p_h, p_d, p_a = outcome_probs(lh, la, self.rho)
        mat = score_matrix(lh, la, self.rho, self.max_goals)
        ij = np.unravel_index(np.argmax(mat), mat.shape)
        return {"home": home, "away": away, "neutral": neutral,
                "p_home": p_h, "p_draw": p_d, "p_away": p_a,
                "exp_goals_home": lh, "exp_goals_away": la,
                "most_likely_score": (int(ij[0]), int(ij[1])),
                "home_matches": self.matches_played.get(home, 0),
                "away_matches": self.matches_played.get(away, 0)}

    def predict_proba(self, home, away, neutral: bool = False) -> np.ndarray:
        o = self.predict_outcome(home, away, neutral)
        return np.array([o["p_home"], o["p_draw"], o["p_away"]])

    def is_high_uncertainty(self, team) -> bool:
        return self.matches_played.get(team, 0) < self.uncertainty_threshold

    def team_frame(self) -> pd.DataFrame:
        rows = []
        for t in self.teams:
            a, d = self._strength(t)
            rows.append({"team": t, "matches": self.matches_played.get(t, 0),
                         "attack": round(a, 4), "defense": round(d, 4),
                         "net_strength": round(a + d, 4)})
        return (pd.DataFrame(rows).sort_values("net_strength", ascending=False)
                .reset_index(drop=True))


def fit_dixon_coles(df: pd.DataFrame, **kw) -> DixonColesModel:
    return DixonColesModel(**kw).fit(df)
