"""Walk-forward, strictly chronological backtest of the strength-prior Dixon-Coles
model on World Cup 2026 matches already played.

For each target match (in date order), the model is refit on ONLY the matches played on
an earlier date, then predicts that match's normal-time H/D/A. The pre-tournament Elo
prior is known before kickoff, so the only thing learned walk-forward is the handful of
global coefficients — no lookahead. Matches share dates, so training on strictly-earlier
DATES (not just earlier rows) avoids same-day leakage.

Reports, on the held-out predictions: multiclass log loss, Brier, accuracy, and a
one-vs-rest reliability table (predicted vs actual outcome frequency) with ECE — versus
a base-rate baseline that predicts the training H/D/A frequencies.

NOTE ON SAMPLE SIZE: this is a few dozen predictions from one tournament. It is a
calibration check and a live demo, NOT a backtested trading system — the 1,000+
settled-bet bar for claiming an edge does not apply here and no edge is claimed.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from wc_config import PROCESSED_DIR
from wc_data import load_played_matches
from wc_dixon_coles import DixonColesStrength

MIN_TRAIN = 20          # need a couple of matchdays before the first prediction
CLASSES = ["H", "D", "A"]


def _logloss(probs: np.ndarray, y_idx: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def _brier(probs: np.ndarray, y_idx: np.ndarray) -> float:
    onehot = np.eye(3)[y_idx]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def run_backtest() -> pd.DataFrame:
    df = load_played_matches()
    rows = []
    for i in range(len(df)):
        target = df.iloc[i]
        train = df[df["date"] < target["date"]]
        if len(train) < MIN_TRAIN:
            continue
        model = DixonColesStrength().fit(train)
        # matches were actually played with a designated home side -> neutral=False
        p_h, p_d, p_a = model.predict_outcome(
            target["home_strength"], target["away_strength"], neutral=False)
        base = train["result90"].value_counts(normalize=True)
        rows.append({
            "match_id": target["match_id"], "date": target["date"],
            "stage": target["stage"], "home_team": target["home_team"],
            "away_team": target["away_team"], "result90": target["result90"],
            "p_home": p_h, "p_draw": p_d, "p_away": p_a,
            "n_train": len(train),
            "base_H": base.get("H", 1 / 3), "base_D": base.get("D", 1 / 3),
            "base_A": base.get("A", 1 / 3),
        })
    return pd.DataFrame(rows)


def reliability_table(probs: np.ndarray, y_idx: np.ndarray, n_bins: int = 5):
    """One-vs-rest reliability: pool all class probabilities, bin, compare to observed."""
    onehot = np.eye(3)[y_idx]
    p_flat = probs.reshape(-1)
    o_flat = onehot.reshape(-1)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece, n = [], 0.0, len(p_flat)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (p_flat >= lo) & (p_flat < hi if b < n_bins - 1 else p_flat <= hi)
        if not sel.any():
            continue
        conf, acc, cnt = p_flat[sel].mean(), o_flat[sel].mean(), int(sel.sum())
        ece += cnt / n * abs(acc - conf)
        rows.append((f"[{lo:.1f},{hi:.1f}]", cnt, conf, acc))
    return rows, ece


def main() -> int:
    bt = run_backtest()
    if bt.empty:
        print("Not enough matches to backtest.")
        return 1
    y = bt["result90"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    P = bt[["p_home", "p_draw", "p_away"]].to_numpy()
    B = bt[["base_H", "base_D", "base_A"]].to_numpy()

    print("=" * 70)
    print("WORLD CUP 2026 - walk-forward Dixon-Coles backtest (normal-time H/D/A)")
    print("=" * 70)
    print(f"Predictions: {len(bt)} matches "
          f"({bt['date'].min().date()} .. {bt['date'].max().date()}), "
          f"expanding train window from {MIN_TRAIN}+ matches.\n")

    print(f"{'model':<18}{'log loss':>10}{'brier':>9}{'accuracy':>10}")
    acc = float((P.argmax(1) == y).mean())
    base_acc = float((B.argmax(1) == y).mean())
    print(f"{'Dixon-Coles':<18}{_logloss(P, y):>10.4f}{_brier(P, y):>9.4f}{acc:>10.3f}")
    print(f"{'base-rate':<18}{_logloss(B, y):>10.4f}{_brier(B, y):>9.4f}{base_acc:>10.3f}")

    print("\nActual vs mean-predicted outcome frequency:")
    for c, i in zip(CLASSES, range(3)):
        print(f"  {c}: actual {np.mean(y == i):.3f}   predicted {P[:, i].mean():.3f}")

    print("\nReliability (one-vs-rest, pooled H/D/A):")
    print(f"  {'bin':<12}{'n':>5}{'pred':>9}{'actual':>9}")
    rows, ece = reliability_table(P, y)
    for label, cnt, conf, obs in rows:
        print(f"  {label:<12}{cnt:>5}{conf:>9.3f}{obs:>9.3f}")
    print(f"  ECE = {ece:.4f}")

    out = PROCESSED_DIR / "wc_backtest_predictions.csv"
    bt.to_csv(out, index=False)
    print(f"\nSaved per-match predictions: {out}")
    print("\nNOTE: small single-tournament sample - a calibration check / live demo, "
          "not a\nbacktested trading system. No edge is claimed at this sample size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
