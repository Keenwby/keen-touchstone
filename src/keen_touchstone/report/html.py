"""Self-contained HTML report: decay chart (SVG) + tables + honesty footer.

No external assets (opens from file://, mailable, committable). Palette is the
dataviz reference instance, validated for both modes (CVD ΔE ≈ 70; the aqua
series is sub-3:1 on the light surface, so the relief rule applies: direct
end-labels + the curve data table below the chart).
"""

from __future__ import annotations

import html as html_mod
import json

from keen_touchstone import __version__
from keen_touchstone.aggregate import SuiteResult

from .emit import RunMeta

# geometry
W, H = 760, 400
ML, MR, MT, MB = 52, 130, 18, 40


def _x(k: int, k_max: int) -> float:
    span = max(k_max - 1, 1)
    return ML + (k - 1) / span * (W - ML - MR)


def _y(v: float) -> float:
    return MT + (1 - v) * (H - MT - MB)


def _fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.1%}"


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def render_html(result: SuiteResult, meta: RunMeta) -> str:
    suite = result.suite
    curve = suite.reliability_decay_curve
    k_max = curve[-1].k
    hk = suite.headline_k or k_max
    pass1 = curve[0].pass_hat_k

    hat_pts = [(_x(p.k, k_max), _y(p.pass_hat_k)) for p in curve]
    at_pts = [(_x(p.k, k_max), _y(p.pass_at_k)) for p in curve if p.pass_at_k is not None]
    band_pts = [(_x(p.k, k_max), _y(p.ci_high)) for p in curve if p.ci_high is not None] + [
        (_x(p.k, k_max), _y(p.ci_low)) for p in reversed(curve) if p.ci_low is not None
    ]

    grid = "".join(
        f'<line x1="{ML}" y1="{_y(v):.1f}" x2="{W - MR}" y2="{_y(v):.1f}" class="grid"/>'
        f'<text x="{ML - 8}" y="{_y(v) + 4:.1f}" class="tick" text-anchor="end">{v:.0%}</text>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    k_step = max(1, (k_max - 1) // 12 + (1 if (k_max - 1) % 12 else 0)) if k_max > 13 else 1
    x_ticks = "".join(
        f'<text x="{_x(k, k_max):.1f}" y="{H - MB + 18}" class="tick" text-anchor="middle">{k}</text>'
        for k in range(1, k_max + 1, k_step)
    )
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" class="dot-hat"/>' for x, y in hat_pts
    )
    headline_x = _x(hk, k_max)

    hover_data = [
        {
            "k": p.k,
            "hat": _fmt_pct(p.pass_hat_k),
            "ci": f"[{_fmt_pct(p.ci_low)}, {_fmt_pct(p.ci_high)}]"
            if p.ci_low is not None
            else "—",
            "at": _fmt_pct(p.pass_at_k),
            "x": round(_x(p.k, k_max), 1),
        }
        for p in curve
    ]

    task_rows = "".join(
        f"<tr><td>{html_mod.escape(t.task_key)}</td><td>{t.n_rollouts}</td>"
        f"<td>{_fmt_pct(t.pass_rate)}</td><td>{_fmt_pct(t.pass_hat_k)}</td>"
        f"<td>[{_fmt_pct(t.pass_hat_k_ci_low)}, {_fmt_pct(t.pass_hat_k_ci_high)}]</td>"
        f"<td>{t.token_mean:,.0f}</td></tr>"
        if t.token_mean is not None
        else f"<tr><td>{html_mod.escape(t.task_key)}</td><td>{t.n_rollouts}</td>"
        f"<td>{_fmt_pct(t.pass_rate)}</td><td>{_fmt_pct(t.pass_hat_k)}</td>"
        f"<td>[{_fmt_pct(t.pass_hat_k_ci_low)}, {_fmt_pct(t.pass_hat_k_ci_high)}]</td><td>—</td></tr>"
        for t in result.tasks
    )
    curve_rows = "".join(
        f"<tr><td>{p.k}</td><td>{_fmt_pct(p.pass_hat_k)}</td>"
        f"<td>[{_fmt_pct(p.ci_low)}, {_fmt_pct(p.ci_high)}]</td><td>{_fmt_pct(p.pass_at_k)}</td></tr>"
        for p in curve
    )

    lever_text = {
        "more_rollouts_per_task": "most remaining noise is within-task sampling — more trials per task tightens the estimate faster than more tasks",
        "more_tasks": "tasks genuinely differ in difficulty — adding tasks improves the estimate faster than rerunning existing ones",
    }.get(result.lever or "")

    warn_html = "".join(
        f'<li><span class="warn-chip">⚠︎ warning</span> {html_mod.escape(w)}</li>'
        for w in result.warnings
    )
    lever_html = (
        f'<li><span class="lever-chip">lever</span> {html_mod.escape(lever_text)}</li>'
        if lever_text
        else ""
    )
    power_html = (
        f'<li><span class="warn-chip">⚠︎ power</span> {suite.power_status} — the headline CI is too wide to act on</li>'
        if suite.power_status
        else ""
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KeenTouchstone — reliability report</title>
<style>
:root {{
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,.10);
  --hat: #2a78d6; --at: #1baf7a; --warn: #fab219;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,.10);
    --hat: #3987e5; --at: #199e70;
  }}
}}
* {{ box-sizing: border-box; margin: 0; }}
body {{ background: var(--page); color: var(--ink); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; padding: 32px 16px 56px; }}
main {{ max-width: 860px; margin: 0 auto; display: grid; gap: 20px; }}
h1 {{ font-size: 21px; }} h2 {{ font-size: 15px; color: var(--ink-2); font-weight: 600; margin-bottom: 10px; }}
.sub {{ color: var(--ink-2); font-size: 13px; }}
.card {{ background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 18px 20px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
.tile .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
.tile .value {{ font-size: 34px; font-weight: 700; margin-top: 2px; }}
.tile .detail {{ font-size: 12.5px; color: var(--ink-2); }}
.legend {{ display: flex; gap: 18px; font-size: 13px; color: var(--ink-2); margin-bottom: 6px; }}
.chip {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: baseline; }}
.chart-wrap {{ position: relative; overflow-x: auto; }}
svg text.tick {{ font: 11.5px system-ui, sans-serif; fill: var(--muted); }}
svg text.endlabel {{ font: 12.5px system-ui, sans-serif; fill: var(--ink-2); }}
svg .grid {{ stroke: var(--grid); stroke-width: 1; }}
svg .axis {{ stroke: var(--axis); stroke-width: 1; }}
svg .band {{ fill: var(--hat); opacity: .15; }}
svg .line-hat {{ stroke: var(--hat); stroke-width: 2; fill: none; }}
svg .line-at {{ stroke: var(--at); stroke-width: 2; fill: none; stroke-dasharray: 6 5; }}
svg .dot-hat {{ fill: var(--hat); }}
svg .headline-mark {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 3; }}
svg .xhair {{ stroke: var(--muted); stroke-width: 1; visibility: hidden; }}
#tip {{ position: absolute; left: 0; top: 0; pointer-events: none; visibility: hidden; background: var(--surface); border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px; font-size: 12.5px; box-shadow: 0 4px 14px rgba(0,0,0,.14); white-space: nowrap; }}
#tip .row {{ display: flex; gap: 8px; justify-content: space-between; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th {{ text-align: left; color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }}
th, td {{ padding: 7px 10px 7px 0; border-bottom: 1px solid var(--grid); }}
td:not(:first-child), th:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
ul.notes {{ list-style: none; display: grid; gap: 8px; font-size: 13.5px; color: var(--ink-2); }}
.warn-chip {{ color: #7a5600; background: color-mix(in srgb, var(--warn) 28%, transparent); border-radius: 5px; padding: 1px 7px; font-size: 12px; font-weight: 600; }}
@media (prefers-color-scheme: dark) {{ .warn-chip {{ color: var(--warn); }} }}
.lever-chip {{ color: var(--ink-2); border: 1px solid var(--ring); border-radius: 5px; padding: 1px 7px; font-size: 12px; font-weight: 600; }}
footer {{ font-size: 12.5px; color: var(--muted); line-height: 1.7; }}
details summary {{ cursor: pointer; color: var(--ink-2); font-size: 13.5px; }}
code {{ font-size: .92em; }}
</style></head><body><main>

<header>
  <h1>KeenTouchstone — reliability report</h1>
  <div class="sub">{html_mod.escape(meta.source)} · model <code>{html_mod.escape(meta.model)}</code> · context {suite.context} · scorer {html_mod.escape(meta.scorer or "—")}</div>
</header>

<section class="tiles">
  <div class="card tile">
    <div class="label">pass@1 · capability</div>
    <div class="value">{_fmt_pct(pass1)}</div>
    <div class="detail">"can it do it at least once?"</div>
  </div>
  <div class="card tile">
    <div class="label">pass^{hk} · reliability</div>
    <div class="value">{_fmt_pct(suite.pass_hat_k)}</div>
    <div class="detail">95% CI [{_fmt_pct(suite.pass_hat_k_ci_low)}, {_fmt_pct(suite.pass_hat_k_ci_high)}] · {suite.ci_method}</div>
  </div>
  <div class="card tile">
    <div class="label">evidence</div>
    <div class="value">{suite.n_rollouts}</div>
    <div class="detail">rollouts · {len(result.tasks)} tasks{f" · ~{suite.token_mean:,.0f} tok/trial" if suite.token_mean else ""}</div>
  </div>
</section>

<section class="card">
  <h2>Reliability decay — pass^k with 95% CI band, k = 1…{k_max}</h2>
  <div class="legend">
    <span><span class="chip" style="background:var(--hat)"></span>pass^k — all k attempts succeed</span>
    <span><span class="chip" style="background:var(--at)"></span>pass@k — at least one succeeds</span>
  </div>
  <div class="chart-wrap" id="wrap">
  <svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Reliability decay curve: pass-hat-k with confidence band and pass-at-k for k from 1 to {k_max}">
    {grid}
    <line x1="{ML}" y1="{_y(0):.1f}" x2="{W - MR}" y2="{_y(0):.1f}" class="axis"/>
    {x_ticks}
    <text x="{(ML + W - MR) / 2:.0f}" y="{H - 4}" class="tick" text-anchor="middle">k (consecutive successes required)</text>
    <line x1="{headline_x:.1f}" y1="{MT}" x2="{headline_x:.1f}" y2="{_y(0):.1f}" class="headline-mark"/>
    <polygon points="{_polyline(band_pts)}" class="band"/>
    <polyline points="{_polyline(at_pts)}" class="line-at"/>
    <polyline points="{_polyline(hat_pts)}" class="line-hat"/>
    {dots}
    <text x="{at_pts[-1][0] + 8:.1f}" y="{at_pts[-1][1] + 4:.1f}" class="endlabel">pass@k {_fmt_pct(curve[-1].pass_at_k)}</text>
    <text x="{hat_pts[-1][0] + 8:.1f}" y="{hat_pts[-1][1] + 4:.1f}" class="endlabel">pass^k {_fmt_pct(curve[-1].pass_hat_k)}</text>
    <text x="{headline_x + 4:.1f}" y="{MT + 10}" class="tick">headline k={hk}</text>
    <line id="xhair" x1="0" y1="{MT}" x2="0" y2="{_y(0):.1f}" class="xhair"/>
  </svg>
  <div id="tip"></div>
  </div>
  <details><summary>curve data (table view)</summary>
    <table><thead><tr><th>k</th><th>pass^k</th><th>95% CI</th><th>pass@k</th></tr></thead>
    <tbody>{curve_rows}</tbody></table>
  </details>
</section>

<section class="card">
  <h2>Per-task reliability at k = {hk} (Beta-Binomial 95% CI, Jeffreys prior)</h2>
  <table><thead><tr><th>task</th><th>n</th><th>pass rate</th><th>pass^{hk}</th><th>95% CI</th><th>mean tokens</th></tr></thead>
  <tbody>{task_rows}</tbody></table>
</section>

{f'<section class="card"><h2>Read this before acting on the numbers</h2><ul class="notes">{lever_html}{power_html}{warn_html}</ul></section>' if (warn_html or lever_html or power_html) else ""}

<footer class="card">
  <strong>How these numbers were computed (the honesty footer).</strong><br>
  pass^k per task = C(c,k)/C(n,k) — the τ-bench unbiased estimator, identical to Inspect AI's <code>pass_k</code> reducer; suite = mean over tasks.
  Suite CI band = envelope of a cluster bootstrap resampling <em>tasks</em> (never pooled attempts) and a Jeffreys-posterior band on the same resamples ({suite.ci_method}, seed-reproducible) — coherent (monotone) along k, never narrower than either component; where the bootstrap alone would collapse to zero width near k = n, the posterior supplies the honest width.
  Curve range k ≤ min(nᵢ) so the task set is constant across k. Headline k = {hk} (defaults to half the curve depth: near k = n the per-task estimator degenerates to 0/1).
  Per-task CIs: Jeffreys Beta posterior, endpoints raised to the k-th power; the per-task <em>point</em> is the UMVUE, which is exactly 0 when c &lt; k — at that boundary it can sit marginally below its own (Bayesian) interval. Two correct estimators, one edge; shown unreconciled rather than fudged.
  A wide interval is the honest product — never suppressed.<br>
  keen-touchstone {__version__} · {html_mod.escape(meta.task_name)} · agent_config {html_mod.escape(suite.agent_config_hash)}
</footer>

</main>
<script>
const DATA = {json.dumps(hover_data)};
const wrap = document.getElementById("wrap"), tip = document.getElementById("tip"),
      xhair = document.getElementById("xhair"), svg = wrap.querySelector("svg");
svg.addEventListener("mousemove", (ev) => {{
  const r = svg.getBoundingClientRect(), sx = r.width / {W};
  const mx = (ev.clientX - r.left) / sx;
  let best = DATA[0];
  for (const d of DATA) if (Math.abs(d.x - mx) < Math.abs(best.x - mx)) best = d;
  xhair.setAttribute("x1", best.x); xhair.setAttribute("x2", best.x);
  xhair.style.visibility = "visible";
  tip.innerHTML = `<div class="row"><strong>k = ${{best.k}}</strong></div>` +
    `<div class="row"><span>pass^k</span><strong>${{best.hat}}</strong></div>` +
    `<div class="row"><span>95% CI</span><span>${{best.ci}}</span></div>` +
    `<div class="row"><span>pass@k</span><span>${{best.at}}</span></div>`;
  tip.style.visibility = "visible";
  const wr = wrap.getBoundingClientRect();
  let left = best.x * sx + 14; if (left + tip.offsetWidth > wr.width - 8) left = best.x * sx - tip.offsetWidth - 14;
  tip.style.left = left + "px"; tip.style.top = (ev.clientY - wr.top + 10) + "px";
}});
svg.addEventListener("mouseleave", () => {{ tip.style.visibility = "hidden"; xhair.style.visibility = "hidden"; }});
</script>
</body></html>
"""
