"""Generate the 1280x630 social-preview.png (Open Graph / Twitter card + repo card).

Evergreen by design -- no stats that age. Written to site/ (so it deploys to Pages at
/social-preview.png, matching the og:image URL in build_site.py) and to .github/ (the
source for GitHub's Settings -> Social preview upload).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
BG = "#0e1116"
CARD = "#161b22"
INK = "#e6edf3"
MUTED = "#9aa4b2"
ACCENT = "#3b82f6"
LINE = "#232a33"

W, H = 1280, 630
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# top accent strip
ax.add_patch(Rectangle((0, H - 12), W, 12, color=ACCENT, zorder=5))

# a minimal knockout-bracket motif on the right
bx = 980
for i, (y0, y1) in enumerate([(150, 250), (330, 430)]):
    ym = (y0 + y1) / 2
    ax.plot([bx, bx + 55], [y0, y0], color=LINE, lw=3)
    ax.plot([bx, bx + 55], [y1, y1], color=LINE, lw=3)
    ax.plot([bx + 55, bx + 55], [y0, y1], color=LINE, lw=3)
    ax.plot([bx + 55, bx + 110], [ym, ym], color=LINE, lw=3)
ax.plot([bx + 110, bx + 110], [200, 380], color=LINE, lw=3)
ax.plot([bx + 110, bx + 175], [290, 290], color=ACCENT, lw=4)
ax.scatter([bx + 190], [290], s=260, color=ACCENT, zorder=6)

# eyebrow
ax.text(90, 500, "FIFA WORLD CUP 2026", color=ACCENT, fontsize=21,
        fontweight="bold", family="DejaVu Sans", alpha=0.95)
# title
ax.text(88, 430, "Model Predictions", color=INK, fontsize=76,
        fontweight="bold", family="DejaVu Sans")
# subtitle
ax.text(90, 335, "Per-team Dixon-Coles Poisson goal model",
        color=INK, fontsize=27, family="DejaVu Sans")
ax.text(90, 288, "Live, auto-updating public track record  ·  no betting odds",
        color=MUTED, fontsize=23, family="DejaVu Sans")

# footer chip with URL
chip = FancyBboxPatch((90, 150), 640, 62, boxstyle="round,pad=8,rounding_size=14",
                      linewidth=1.4, edgecolor=LINE, facecolor=CARD, zorder=4)
ax.add_patch(chip)
ax.text(118, 181, "anishkhetani.github.io/worldcup-2026-model", color=INK,
        fontsize=24, family="DejaVu Sans", va="center", zorder=5)

for dest in (ROOT / "site" / "social-preview.png",
             ROOT / ".github" / "social-preview.png"):
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, facecolor=BG, dpi=100)
    print(f"wrote {dest} ({dest.stat().st_size:,} bytes)")
plt.close(fig)
