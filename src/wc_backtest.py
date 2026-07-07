"""Walk-forward backtest report for the per-team Dixon-Coles model.

Reuses the honest, lookahead-free walk-forward predictions from build_results_log (each
completed match predicted from only earlier-dated internationals), and reports the
model's normal-time H/D/A calibration: multiclass log loss, Brier, accuracy, and a
one-vs-rest reliability table with ECE, against a base-rate baseline.

NOTE ON SAMPLE SIZE: this is a few dozen predictions from one tournament -- a
calibration check / live demo, not a backtested trading system. No edge is claimed.
"""
from __future__ import annotations

import sys

import numpy as np

from build_results_log import build_log, track_record

CLASSES = ["H", "D", "A"]


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
    df = build_log()
    done = df[df["completed"] == True]  # noqa: E712
    if done.empty:
        print("No completed matches to backtest.")
        return 1
    tr = track_record(df)
    y = done["actual"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    P = done[["p_home", "p_draw", "p_away"]].to_numpy(float)

    print("=" * 70)
    print("WORLD CUP 2026 - walk-forward per-team Dixon-Coles backtest (90' H/D/A)")
    print("=" * 70)
    print(f"Predictions: {tr['n']} completed matches, each refit on only earlier data.\n")

    print(f"{'model':<18}{'log loss':>10}{'brier':>9}{'accuracy':>10}")
    print(f"{'Dixon-Coles':<18}{tr['log_loss']:>10.4f}{tr['brier']:>9.4f}{tr['accuracy']:>10.3f}")
    print(f"{'base-rate':<18}{tr['base_log_loss']:>10.4f}{'':>9}{'':>10}")

    print("\nActual vs mean-predicted outcome frequency:")
    for c, i in zip(CLASSES, range(3)):
        print(f"  {c}: actual {np.mean(y == i):.3f}   predicted {P[:, i].mean():.3f}")

    print("\nReliability (one-vs-rest, pooled H/D/A):")
    print(f"  {'bin':<12}{'n':>5}{'pred':>9}{'actual':>9}")
    rows, ece = reliability_table(P, y)
    for label, cnt, conf, obs in rows:
        print(f"  {label:<12}{cnt:>5}{conf:>9.3f}{obs:>9.3f}")
    print(f"  ECE = {ece:.4f}")
    print(f"\nCorrect calls: {tr['correct']}/{tr['n']} ({tr['accuracy']:.1%}).")
    print("\nNOTE: small single-tournament sample - a calibration check / live demo, "
          "not a\nbacktested trading system. No edge is claimed at this sample size.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
