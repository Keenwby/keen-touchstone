"""Judge exam outputs: license.json + judge-report.html + terminal summary."""

from __future__ import annotations

import html as html_mod
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from keen_touchstone import __version__
from keen_touchstone.artifacts import JudgeCalibration
from keen_touchstone.judge.alt_test import AltTestOutcome
from keen_touchstone.judge.kappa import ExamResult


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.1%}"


def _num(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def emit_license(
    calibration: JudgeCalibration,
    exam: ExamResult,
    alt: AltTestOutcome | None,
    out_dir: Path,
    console: Console | None = None,
) -> Path:
    console = console or Console()
    out_dir.mkdir(parents=True, exist_ok=True)

    license_path = out_dir / "license.json"
    license_path.write_text(json.dumps(calibration.to_schema_dict(), indent=2) + "\n")
    html_path = out_dir / "judge-report.html"
    html_path.write_text(render_judge_html(calibration, exam, alt))

    _print_summary(calibration, exam, console)
    console.print(f"\n[dim]wrote[/dim] {license_path}")
    console.print(f"[dim]wrote[/dim] {html_path}")
    return license_path


def _print_summary(cal: JudgeCalibration, exam: ExamResult, console: Console) -> None:
    console.print()
    console.rule("[bold]KeenTouchstone — judge exam[/bold]")
    if cal.status == "JUDGE_LICENSED":
        console.print(f"  [bold green]✓ JUDGE_LICENSED[/bold green]  {cal.judge_id}  [dim]{cal.calibration_id}[/dim]")
    else:
        console.print(f"  [bold red]✗ NEEDS_HUMAN[/bold red]  {cal.judge_id}  [dim]{cal.calibration_id}[/dim]")

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim", min_width=22)
    table.add_column()
    kappa = (
        f"{_num(cal.kappa)}  [dim]\\[{_num(cal.kappa_ci_low)}, {_num(cal.kappa_ci_high)}] 95% CI[/dim]"
        if cal.kappa is not None
        else "[red]withheld / undefined[/red]"
    )
    table.add_row("Cohen's κ vs humans", kappa)
    table.add_row("raw agreement", _pct(cal.raw_agreement))
    table.add_row("TPR (catches real passes)", f"{_pct(cal.tpr)}  [dim]\\[{_pct(cal.tpr_ci_low)}, {_pct(cal.tpr_ci_high)}][/dim]")
    table.add_row("FPR (rubber-stamps fails)", f"{_pct(cal.fpr)}  [dim]\\[{_pct(cal.fpr_ci_low)}, {_pct(cal.fpr_ci_high)}][/dim]")
    table.add_row("anchor set", f"{cal.anchor_n_items} items · pass base rate {_pct(cal.prevalence)} · abstained {_pct(cal.abstention_rate)}")
    if cal.alt_test is not None:
        if cal.alt_test.applicable:
            verdict = "[green]PASSED[/green]" if cal.alt_test.passed else "[red]FAILED[/red]"
            table.add_row(
                "alt-test (replace humans?)",
                f"{verdict}  ω={_num(cal.alt_test.omega)} · ρ̄={_num(cal.alt_test.avg_advantage_probability)} · ε={cal.alt_test.epsilon}",
            )
        else:
            table.add_row("alt-test (replace humans?)", f"[dim]not applicable — {cal.alt_test.reason}[/dim]")
    console.print(table)
    for r in cal.reasons:
        console.print(f"  [yellow]•[/yellow] {r}")


def render_judge_html(cal: JudgeCalibration, exam: ExamResult, alt: AltTestOutcome | None) -> str:
    licensed = cal.status == "JUDGE_LICENSED"
    c = exam.confusion
    reasons_html = "".join(f"<li>{html_mod.escape(r)}</li>" for r in cal.reasons) or "<li>none — clean exam</li>"

    alt_html = ""
    if cal.alt_test is not None:
        if cal.alt_test.applicable and alt is not None:
            rows = "".join(
                f"<tr><td>{html_mod.escape(a.annotator)}</td><td>{a.n_items}</td>"
                f"<td>{a.rho_llm:.3f}</td><td>{a.rho_human:.3f}</td>"
                f"<td>{a.p_value:.4f}</td><td>{a.test}</td>"
                f"<td>{'✓ judge wins' if a.rejected else '—'}</td></tr>"
                for a in alt.annotators
            )
            verdict = "PASSED" if cal.alt_test.passed else "FAILED"
            alt_html = f"""
<section class="card">
  <h2>alt-test — may this judge replace your annotators? <strong class="{'ok' if cal.alt_test.passed else 'bad'}">{verdict}</strong></h2>
  <p class="sub">Leave-one-annotator-out (ACL 2025): ω = {cal.alt_test.omega:.2f} of annotators are out-represented by the judge (need ≥ 0.5) · ρ̄ = {cal.alt_test.avg_advantage_probability:.2f} = P(judge ≥ a random annotator) · ε = {cal.alt_test.epsilon}, Benjamini–Yekutieli q = {cal.alt_test.q}</p>
  <table><thead><tr><th>annotator</th><th>items</th><th>ρ judge</th><th>ρ human</th><th>p</th><th>test</th><th>after FDR</th></tr></thead><tbody>{rows}</tbody></table>
</section>"""
        else:
            alt_html = f"""
<section class="card"><h2>alt-test</h2>
  <p class="sub">Not applicable: {html_mod.escape(cal.alt_test.reason or "")} — reported honestly, never computed anyway.</p>
</section>"""

    kappa_tile = (
        f'<div class="value">{cal.kappa:.3f}</div><div class="detail">95% CI [{_num(cal.kappa_ci_low)}, {_num(cal.kappa_ci_high)}] · threshold {cal.thresholds.kappa_licensed} on {cal.thresholds.gate_on}</div>'
        if cal.kappa is not None
        else '<div class="value">withheld</div><div class="detail">too few items for an honest point estimate</div>'
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KeenTouchstone — judge exam</title>
<style>
:root {{ --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --ring:rgba(11,11,11,.10); --ok:#0ca30c; --bad:#d03b3b; --warn:#fab219; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --page:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --ring:rgba(255,255,255,.10); }} }}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--page); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; padding:32px 16px 56px; }}
main {{ max-width:820px; margin:0 auto; display:grid; gap:18px; }}
h1 {{ font-size:21px; }} h2 {{ font-size:15px; color:var(--ink-2); font-weight:600; margin-bottom:10px; }}
.sub {{ color:var(--ink-2); font-size:13px; margin-bottom:8px; }}
.card {{ background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:18px 20px; }}
.banner {{ display:flex; align-items:center; gap:12px; font-size:18px; font-weight:700; }}
.banner .dot {{ width:14px; height:14px; border-radius:50%; }}
.ok {{ color:var(--ok); }} .bad {{ color:var(--bad); }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }}
.tile .label {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
.tile .value {{ font-size:30px; font-weight:700; margin-top:2px; }}
.tile .detail {{ font-size:12.5px; color:var(--ink-2); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th {{ text-align:left; color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; }}
th,td {{ padding:7px 10px 7px 0; border-bottom:1px solid var(--grid); }}
td:not(:first-child),th:not(:first-child) {{ text-align:right; font-variant-numeric:tabular-nums; }}
ul {{ padding-left:20px; font-size:13.5px; color:var(--ink-2); display:grid; gap:6px; }}
footer {{ font-size:12.5px; color:var(--muted); line-height:1.7; }}
code {{ font-size:.92em; }}
</style></head><body><main>

<header>
  <h1>KeenTouchstone — judge exam（裁判体检单）</h1>
  <div class="sub">judge <code>{html_mod.escape(cal.judge_id)}</code>{f" · model <code>{html_mod.escape(cal.judge_model)}</code>" if cal.judge_model else ""} · anchors {html_mod.escape(cal.anchor_set_ref or "—")} · {html_mod.escape(cal.created_at)}</div>
</header>

<section class="card banner">
  <span class="dot" style="background:var({'--ok' if licensed else '--bad'})"></span>
  <span class="{'ok' if licensed else 'bad'}">{cal.status}</span>
  <span class="sub" style="margin:0">license <code>{html_mod.escape(cal.calibration_id)}</code></span>
</section>

<section class="tiles">
  <div class="card tile"><div class="label">Cohen's κ vs humans</div>{kappa_tile}</div>
  <div class="card tile"><div class="label">raw agreement</div><div class="value">{_pct(cal.raw_agreement)}</div><div class="detail">uncorrected for chance — κ is the honest number</div></div>
  <div class="card tile"><div class="label">TPR · catches real passes</div><div class="value">{_pct(cal.tpr)}</div><div class="detail">95% CI [{_pct(cal.tpr_ci_low)}, {_pct(cal.tpr_ci_high)}]</div></div>
  <div class="card tile"><div class="label">FPR · rubber-stamps fails</div><div class="value">{_pct(cal.fpr)}</div><div class="detail">95% CI [{_pct(cal.fpr_ci_low)}, {_pct(cal.fpr_ci_high)}] — the dangerous direction</div></div>
</section>

<section class="card">
  <h2>exam sheet — {cal.anchor_n_items} human-labeled items · pass base rate {_pct(cal.prevalence)} · abstained {_pct(cal.abstention_rate)}</h2>
  <table>
    <thead><tr><th></th><th>human: pass</th><th>human: fail</th></tr></thead>
    <tbody>
      <tr><td>judge: pass</td><td>{c["tp"]}</td><td>{c["fp"]} ← false pass</td></tr>
      <tr><td>judge: fail</td><td>{c["fn"]}</td><td>{c["tn"]}</td></tr>
    </tbody>
  </table>
</section>

{alt_html}

<section class="card">
  <h2>grounds &amp; caveats</h2>
  <ul>{reasons_html}</ul>
</section>

<footer class="card">
  <strong>How this exam works.</strong>
  κ = (observed − chance agreement)/(1 − chance agreement), computed ONLY against human-confirmed labels (anti-circularity: auto-labels may expand an eval set but never certify the judge). CI = item-bootstrap.
  TPR/FPR carry Jeffreys Beta CIs. Abstentions are excluded and counted, never coerced.
  Gate: κ {cal.thresholds.gate_on} ≥ {cal.thresholds.kappa_licensed} (Landis–Koch "acceptable"; 0.8 = strong), ≥ {cal.thresholds.min_items} items, abstention ≤ {cal.thresholds.max_abstention:.0%}{", alt-test must not fail" if cal.alt_test and cal.alt_test.applicable else ""}.
  A NEEDS_HUMAN judge's scores are not evidence — that is the point of the gate.<br>
  keen-touchstone {__version__}
</footer>

</main></body></html>
"""
