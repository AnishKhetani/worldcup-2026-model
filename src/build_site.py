"""Render the static site (site/index.html) from results_log.csv.

No backend, no external assets: one self-contained, theme-aware HTML file that GitHub
Pages can serve directly. Two sections: the model's confident predictions for upcoming
fixtures, and a track record of how its past predictions did against actual results.
The predictions CSV is copied alongside for download / transparency.
"""
from __future__ import annotations

import datetime as dt
import html
import shutil
import sys
from pathlib import Path

import pandas as pd

from build_results_log import build_log, track_record
from wc_config import PROCESSED_DIR

SITE_DIR = Path(__file__).resolve().parents[1] / "site"
REPO_URL = "https://github.com/AnishKhetani/worldcup-2026-model"

CSS = """
:root{--bg:#f7f8fa;--card:#fff;--ink:#111418;--muted:#5a6472;--line:#e4e7ec;
--home:#2563eb;--draw:#94a3b8;--away:#e2683c;--ok:#16a34a;--no:#dc2626;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#161b22;--ink:#e6edf3;
--muted:#9aa4b2;--line:#232a33;--home:#3b82f6;--draw:#64748b;--away:#f97316}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px}
.sub{color:var(--muted);margin:0 0 18px}
.disc{background:color-mix(in srgb,var(--accent) 8%,transparent);
border:1px solid color-mix(in srgb,var(--accent) 25%,transparent);border-radius:10px;
padding:10px 14px;font-size:13px;color:var(--muted);margin:14px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:8px 0 4px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.tile .n{font-size:24px;font-weight:650}.tile .l{font-size:12px;color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:14px 16px;margin:10px 0}
.mt{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap}
.mt .stage{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.team{font-weight:650;font-size:17px}
.call{margin:8px 0 4px;font-size:14px}.call b{color:var(--accent)}
.bar{display:flex;height:22px;border-radius:6px;overflow:hidden;margin:8px 0 4px;font-size:11px}
.bar div{display:flex;align-items:center;justify-content:center;color:#fff;white-space:nowrap;min-width:0}
.bh{background:var(--home)}.bd{background:var(--draw)}.ba{background:var(--away)}
.lg{font-size:11px;color:var(--muted);display:flex;gap:14px;margin-top:2px}
.lg span::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:middle}
.lg .h::before{background:var(--home)}.lg .d::before{background:var(--draw)}.lg .a::before{background:var(--away)}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px}
th,td{text-align:left;padding:8px 8px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.c{text-align:center}.ok{color:var(--ok);font-weight:700}.no{color:var(--no);font-weight:700}
.foot{color:var(--muted);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:16px}
a{color:var(--accent)}.scroll{overflow-x:auto}
"""

_FULLNAME = {"H": "home win", "D": "draw", "A": "away win"}


def _pct(x) -> str:
    return f"{round(float(x) * 100)}%"


def _bar(ph, pd_, pa) -> str:
    seg = [("bh", ph), ("bd", pd_), ("ba", pa)]
    out = ['<div class="bar">']
    for cls, p in seg:
        w = max(float(p) * 100, 0)
        label = f"{round(w)}%" if w >= 9 else ""
        out.append(f'<div class="{cls}" style="width:{w:.1f}%">{label}</div>')
    out.append("</div>")
    return "".join(out)


def _upcoming_card(r) -> str:
    home, away = html.escape(r["home_team"]), html.escape(r["away_team"])
    call = ""
    if r["is_knockout"] and r["progress_pick"]:
        call = (f'<div class="call">Model call: <b>{html.escape(str(r["progress_pick"]))}'
                f'</b> to progress &middot; {_pct(r["progress_conf"])}</div>')
    else:
        pk = _FULLNAME.get(r["pick"], r["pick"])
        who = home if r["pick"] == "H" else (away if r["pick"] == "A" else "draw")
        call = (f'<div class="call">Model call: <b>{html.escape(who)}</b>'
                f'{"" if r["pick"]=="D" else " ("+pk+")"} &middot; {_pct(r["pick_conf"])}</div>')
    prog = ""
    if r["is_knockout"] and r["home_progress"] != "":
        prog = (f'<div class="lg" style="margin-top:6px">Progression: '
                f'{home} {_pct(r["home_progress"])} &middot; {away} {_pct(r["away_progress"])}</div>')
    return f"""<div class="card">
<div class="mt"><span class="team">{home} <span style="color:var(--muted)">v</span> {away}</span>
<span class="stage">{html.escape(str(r["stage"]))} &middot; {r["date"]}</span></div>
{call}
{_bar(r["p_home"], r["p_draw"], r["p_away"])}
<div class="lg"><span class="h">{home} {_pct(r["p_home"])}</span>
<span class="d">draw {_pct(r["p_draw"])}</span>
<span class="a">{away} {_pct(r["p_away"])}</span></div>
{prog}</div>"""


def _record_rows(done: pd.DataFrame) -> str:
    out = []
    for _, r in done.sort_values(["date", "match_id"], ascending=[False, False]).iterrows():
        home, away = html.escape(r["home_team"]), html.escape(r["away_team"])
        pick_team = home if r["pick"] == "H" else (away if r["pick"] == "A" else "Draw")
        act_team = home if r["actual"] == "H" else (away if r["actual"] == "A" else "Draw")
        mark = '<span class="ok">&#10003;</span>' if r["correct"] else '<span class="no">&#10007;</span>'
        out.append(
            f"<tr><td>{r['date']}</td><td>{html.escape(str(r['stage']))}</td>"
            f"<td>{home} v {away}</td><td>{pick_team} ({_pct(r['pick_conf'])})</td>"
            f"<td>{act_team} <span style='color:var(--muted)'>{html.escape(str(r['actual_score']))}</span></td>"
            f"<td class='c'>{mark}</td></tr>")
    return "".join(out)


def render(df: pd.DataFrame, tr: dict, updated: str) -> str:
    upcoming = df[df["completed"] == False]  # noqa: E712
    done = df[df["completed"] == True]       # noqa: E712

    up_html = ("".join(_upcoming_card(r) for _, r in upcoming.iterrows())
               if len(upcoming) else
               '<div class="card">No upcoming fixtures with confirmed teams right now — '
               'the next round fills in as results land.</div>')

    if tr.get("n"):
        tiles = f"""<div class="tiles">
<div class="tile"><div class="n">{tr['accuracy']*100:.0f}%</div><div class="l">correct calls ({tr['correct']}/{tr['n']})</div></div>
<div class="tile"><div class="n">{tr['log_loss']:.3f}</div><div class="l">log loss</div></div>
<div class="tile"><div class="n">{tr['base_log_loss']:.3f}</div><div class="l">base-rate log loss</div></div>
<div class="tile"><div class="n">{tr['brier']:.3f}</div><div class="l">Brier score</div></div></div>"""
        record = f"""{tiles}
<div class="scroll"><table><thead><tr><th>Date</th><th>Stage</th><th>Match</th>
<th>Model pick</th><th>Actual</th><th class="c">Hit</th></tr></thead>
<tbody>{_record_rows(done)}</tbody></table></div>"""
    else:
        record = '<div class="card">No completed matches scored yet.</div>'

    css = "<style>" + CSS + "</style>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>World Cup 2026 — Model Predictions</title>{css}</head><body><div class="wrap">
<h1>World Cup 2026 — Model Predictions</h1>
<p class="sub">A statistical model's result predictions for the remaining knockout
fixtures, and its running track record on matches already played.</p>
<div class="disc"><b>Research / analytics project — not betting advice.</b> These are
model probabilities published for interest and to keep the model honest against real
results. No wagering guidance is given or implied.</div>

<h2>Upcoming fixtures</h2>
{up_html}

<h2>Track record</h2>
<p class="sub">How the model's pre-match pick has fared on the {tr.get('n', 0)} matches
played so far. "Log loss" rewards being confident <i>and</i> right; the base-rate line
is a naive predictor for comparison.</p>
{record}

<h2>How it works</h2>
<p class="sub">A Dixon-Coles Poisson goal model. Team strength comes from a rolling Elo
over ~49k international matches (1872–present); the goal-rate coefficients are fit on
~25k pre-tournament internationals. Every prediction uses only information available
<i>before</i> the match — no lookahead. Full method: <a href="{REPO_URL}/blob/main/SPEC.md">SPEC.md</a>.</p>

<div class="foot">
Updated {updated} UTC · Data: <a href="https://github.com/mominullptr/FIFA-World-Cup-2026-Dataset">FIFA World Cup 2026 Dataset</a>
(CC0) &amp; <a href="https://github.com/martj42/international_results">international_results</a> (CC0).
· <a href="{REPO_URL}">Source &amp; method</a> · <a href="results_log.csv">predictions CSV</a><br>
Public research project. Not affiliated with FIFA. Not betting advice.
</div></div></body></html>"""


def main() -> int:
    df = build_log()
    df.to_csv(PROCESSED_DIR / "results_log.csv", index=False)
    tr = track_record(df)
    updated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "index.html").write_text(render(df, tr, updated), encoding="utf-8")
    shutil.copy(PROCESSED_DIR / "results_log.csv", SITE_DIR / "results_log.csv")
    print(f"Built {SITE_DIR/'index.html'} "
          f"({int((df['completed']==False).sum())} upcoming, {tr.get('n',0)} scored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
