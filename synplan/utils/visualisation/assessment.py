"""Offline route-assessment tables for the existing chemical route report."""

from html import escape

ASSESSMENT_CSS = """
[hidden]{display:none!important}
.assessment-overview{padding:20px;margin:20px 0}
.assessment-overview h2{margin:0 0 12px;font-size:20px}
.assessment-note{color:#526171;line-height:1.6}
.assessment-scroll{overflow-x:auto}
.assessment-table{border-collapse:collapse;width:100%;font-size:13px}
.assessment-table th,.assessment-table td{padding:10px;text-align:left;
border-bottom:1px solid #e6e8eb;white-space:nowrap}
.assessment-table th{background:#f4f7fa}
.assessment-table a{color:#194c7e}
.assessment-step-table{margin:16px 18px}
.assessment-step-table caption{text-align:left;padding:0 0 10px;font-weight:600}
.forward-match,.forward-zero,.forward-assumed{display:inline-block;border-radius:4px;
padding:3px 8px;font-size:12px;font-weight:600}
.forward-match{background:#eaf5ef;color:#216145}
.forward-zero{background:#fceeed;color:#a62f36}
.forward-assumed{background:#eaf1fc;color:#285d9a}
.assessment-filters{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}
.assessment-filters button{padding:8px 12px;border:1px solid #cdd6df;
border-radius:4px;background:white;cursor:pointer}
.assessment-filters button[aria-pressed=true]{background:#163e5b;color:white}
.assessment-strings summary{padding:12px 18px;cursor:pointer}
"""

ASSESSMENT_JS = """
function filterAssessment(group) {
  document.querySelectorAll('[data-assessment-group]').forEach(node => {
    node.hidden = group !== 'all' && node.dataset.assessmentGroup !== group;
  });
  document.querySelectorAll('[data-assessment-filter]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.assessmentFilter === group));
  });
}
document.querySelectorAll('[data-assessment-filter]').forEach(button => {
  button.addEventListener('click', () => filterAssessment(button.dataset.assessmentFilter));
});
document.querySelectorAll('a[href^="#assessment-route-"]').forEach(link => {
  link.addEventListener('click', () => {
    const card = document.getElementById(link.hash.slice(1)).closest('.route');
    if (card.hidden) filterAssessment('all');
  });
});
document.querySelectorAll('.route[data-assessment-group]').forEach(card => {
  const rows = card.querySelectorAll('[data-step-color]');
  card.querySelectorAll('.draw .sp-num').forEach(label => {
    const row = rows[Number(label.textContent) - 1];
    if (row) label.previousElementSibling.setAttribute('fill', row.dataset.stepColor);
  });
});
"""


def assessment_group(route):
    if route.assessment is None:
        return "unassessed"
    return "failed" if route.assessment["zero_score_steps"] else "positive"


def _number(value):
    return "—" if value is None else f"{value:.6g}"


def _step_display(annotation):
    if annotation["score"] == 0:
        return "zero", "No top-10 match", "#bd3f3f"
    if annotation["rank"] is None:
        return "assumed", "Assumed · 1 reactant", "#285d9a"
    return "match", str(annotation["rank"]), "#267154"


def _badge(annotation):
    style, rank, _ = _step_display(annotation)
    return (
        f'<span class="forward-{style}" title="{rank}">{annotation["score"]:.3f}</span>'
    )


def assessment_steps(route, index):
    if route.assessment is None:
        return ""
    rows = []
    for number, step in enumerate(route, 1):
        annotation = step.forward_prediction
        _, rank, colour = _step_display(annotation)
        rows.append(
            f'<tr id="assessment-route-{index}-step-{number}" data-step-color="{colour}">'
            f"<td>{number}</td><td>{rank}</td><td>{_badge(annotation)}</td>"
            f"<td>{'Yes' if annotation['alert'] else 'No'}</td>"
            f'<td class="mono">{escape(annotation["product_inchikey"])}</td></tr>'
        )
    score = route.assessment
    combined = (
        f" · Combined: {_number(score['combined_score'])}"
        if "combined_score" in score
        else ""
    )
    return (
        '<div class="assessment-step-table assessment-scroll">'
        '<table class="assessment-table">'
        f"<caption>Forward route score: {_number(score['forward_score'])}{combined}"
        " · step numbers match the drawing</caption>"
        "<thead><tr><th>Step</th><th>Top-k rank / assumption</th><th>Step score</th>"
        "<th>Alert</th><th>Product InChIKey · stereo-free</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        '<a href="#assessment-overview">Back to score overview ↑</a></div>'
    )


def assessment_overview(routes, labels):
    combined = any(
        r.assessment is not None and "combined_score" in r.assessment for r in routes
    )
    groups = [assessment_group(route) for route in routes]
    rows = []
    for index, (route, label, group) in enumerate(zip(routes, labels, groups), 1):
        score = route.assessment or {}
        likelihood = (
            None if route.provenance is None else route.provenance.policy_log_likelihood
        )
        vector = " ".join(
            f'<a href="#assessment-route-{index}-step-{number}">'
            f"{_badge(step.forward_prediction)}</a>"
            for number, step in enumerate(route, 1)
            if route.assessment is not None
        )
        rows.append(
            f'<tr data-assessment-group="{group}">'
            f'<td><a href="#assessment-route-{index}">{escape(str(label))}</a></td>'
            f"<td>{_number(score.get('forward_score'))}</td>"
            f"<td>{_number(score.get('search_score'))}</td><td>{_number(likelihood)}</td>"
            + (f"<td>{_number(score.get('combined_score'))}</td>" if combined else "")
            + f"<td>{score.get('zero_score_steps', '—')} / {len(route)}</td>"
            + f"<td>{score.get('assumed_one_reactant_steps', '—')}</td>"
            + f"<td>{vector}</td></tr>"
        )
    buttons = "".join(
        f'<button data-assessment-filter="{group}" '
        f'aria-pressed="{str(group == "all").lower()}">{label} · {count}</button>'
        for group, label, count in (
            ("all", "All routes", len(routes)),
            ("positive", "No zero-score steps", groups.count("positive")),
            ("failed", "At least one zero-score step", groups.count("failed")),
        )
    )
    return (
        '<section id="assessment-overview" class="assessment-overview card">'
        "<h2>Forward route assessment</h2>"
        '<p class="assessment-note">The route score is the geometric mean of all step '
        "scores, counting each step once across every branch. A zero step makes the "
        "route score zero. Assumed one-reactant scores of 1.0 are included: "
        "[0.25, 1.0] gives 0.5. These are ranking heuristics, not probabilities of "
        "experimental success or measures of stereochemical correctness. "
        "Search scores and policy log likelihoods retain their original scale.</p>"
        + (
            '<p class="assessment-note">Combined score = search score &times; forward route '
            "score; explicitly enabled for this report.</p>"
            if combined
            else ""
        )
        + '<p><span class="forward-match">Model match</span> '
        '<span class="forward-zero">No top-10 match</span> '
        '<span class="forward-assumed">Assumed · 1 reactant</span></p>'
        f'<div class="assessment-filters">{buttons}</div>'
        '<div class="assessment-scroll"><table class="assessment-table">'
        "<thead><tr><th>Route ID</th><th>Forward score</th><th>Search score</th>"
        "<th>Policy log likelihood</th>"
        + ("<th>Combined score</th>" if combined else "")
        + "<th>Zero / total steps</th><th>Assumed steps</th>"
        "<th>Step scores · 1 → n</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )
