"""Independent, compact thermochemistry reports for completed planning routes.

Rendering reads scientific annotations only; it never loads a calculator or
examines forward-validation annotations. Full molecular records stay in JSON.
"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from html import escape
from math import isfinite
from pathlib import Path
from typing import Any

from synplan.chem.reaction.routes.representation.depiction import (
    _temporary_render_config,
)
from synplan.chem.reaction.score.aimnet.config import FILTER_REASONS
from synplan.utils.routedraw import ARROW_DEFS, ROUTE_CSS, drawable_copy, molecule_svg
from synplan.utils.svgslim import Doc, hidden_defs

ENERGIES = ("delta_e", "delta_h", "delta_g")
STATUS_LABELS = {
    "processed": "Processed",
    "filtered_out": "Filtered out",
    "errored": "Errored",
}


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _valid_variant(variant):
    return (
        variant.get("status") == "ok"
        and all(_finite(variant.get(k)) for k in ("delta_h", "delta_g"))
        and (
            _finite(variant.get("delta_e"))
            or (
                variant.get("delta_e") is None
                and bool(variant.get("species_corrections"))
            )
        )
    )


def display_status(annotation):
    """Map new and legacy annotations conservatively, without parsing messages."""
    if not annotation:
        return "errored"
    variants = annotation.get("variants", [])
    if (
        annotation.get("status") == "ok"
        and variants
        and all(map(_valid_variant, variants))
    ):
        return "processed"
    failure = annotation.get("failure") or {}
    if not variants and (
        failure.get("category") == "filtered_out"
        or (failure.get("category") is None and failure.get("reason") in FILTER_REASONS)
    ):
        return "filtered_out"
    return "errored"


def _number(value):
    return f"{value:.3f}" if _finite(value) else "—"


def _energy_range(variants, key):
    values = [v[key] for v in variants if _valid_variant(v) and _finite(v.get(key))]
    if not values:
        return "—"
    if len(variants) == 1:
        return _number(values[0])
    return f'<span class="range-label">min-max</span> {_number(min(values))} … {_number(max(values))}'


def lightweight_log(annotation, route_id, step_number):
    """Small grouped diagnostics, excluding model payloads and configuration."""
    annotation = annotation or {}
    entries: list[tuple[Any, dict[str, Any]]] = []
    if annotation.get("failure"):
        entries.append((None, annotation["failure"]))
    for variant in annotation.get("variants", []):
        if not _valid_variant(variant):
            failures = variant.get("failures") or [
                {
                    "stage": "result",
                    "reason": "unavailable_values",
                    "detail": "No valid thermochemistry is available.",
                }
            ]
            entries.extend((variant.get("id"), failure) for failure in failures)
    if not entries:
        entries = [
            (
                None,
                {
                    "stage": "assessment",
                    "reason": "missing_assessment",
                    "detail": "No complete assessment is available.",
                },
            )
        ]
    grouped = {}
    for variant_id, failure in entries:
        detail = str(failure.get("detail", ""))[:600]
        key = (
            failure.get("stage", "unknown"),
            failure.get("reason", "unknown"),
            failure.get("smiles"),
            detail,
        )
        group = grouped.setdefault(
            key,
            {
                "stage": key[0],
                "reason": key[1],
                "detail": detail,
                "species": key[2],
                "variant_ids": [],
                "conformer_failures": [],
            },
        )
        if variant_id is not None and variant_id not in group["variant_ids"]:
            group["variant_ids"].append(variant_id)
        # A cached molecular failure can occur in many variants. Count its
        # rejected conformers once, rather than inflating the molecular log.
        counts = Counter(
            (f.get("stage", "unknown"), f.get("reason", "unknown"))
            for f in failure.get("conformer_failures", [])
        )
        if not group["conformer_failures"]:
            group["conformer_failures"] = [
                {"stage": stage, "reason": reason, "count": count}
                for (stage, reason), count in sorted(counts.items())
            ]
    return {
        "route_id": str(route_id),
        "step": step_number,
        "failures": list(grouped.values()),
    }


def _equation_details(annotation):
    return (
        '<details class="equations"><summary>Equation</summary>'
        f"<p>Original: <code>{escape(str(annotation.get('original_reaction', 'Unavailable')))}</code></p>"
        f"<p>Balanced: <code>{escape(str(annotation.get('balanced_reaction') or 'Unavailable'))}</code></p>"
        f"<p>Inferred species: <code>{escape(json.dumps(annotation.get('inferred_species', {}), ensure_ascii=False))}</code></p>"
        "</details>"
    )


def _variant_details(variants):
    if len(variants) <= 1:
        return ""
    rows = []
    for variant in variants:
        rows.append(
            f"<tr><td>{escape(str(variant.get('id', '—')))}</td>"
            f"<td><code>{escape(str(variant.get('reaction', '')))}</code></td>"
            + "".join(
                f"<td>{_number(variant.get(k)) if _valid_variant(variant) else '—'}</td>"
                for k in ENERGIES
            )
            + f"<td>{'Processed' if _valid_variant(variant) else 'Errored'}</td></tr>"
        )
    return (
        '<details class="variants"><summary>Variants</summary><div class="thermo-scroll">'
        "<table><thead><tr><th>Variant</th><th>Equation</th><th>ΔE</th><th>ΔH</th><th>ΔG</th><th>Status</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></details>"
    )


def _step_table(route, route_id):
    rows = []
    for index, step in enumerate(route, 1):
        annotation = step.thermochemistry or {}
        variants = annotation.get("variants", [])
        status = display_status(annotation)
        label = STATUS_LABELS[status]
        if (annotation.get("failure") or {}).get("reason") == "run_aborted":
            label += " — run aborted"
        details = _equation_details(annotation) + _variant_details(variants)
        note = ""
        if status == "filtered_out":
            failure = annotation.get("failure", {})
            note = f'<div class="filter-reason">{escape(str(failure.get("reason", "filtered_out")))}: {escape(str(failure.get("detail", "")))}</div>'
        elif status == "errored":
            log = json.dumps(
                lightweight_log(annotation, route_id, index),
                indent=2,
                ensure_ascii=False,
            )
            details += (
                f'<details class="thermo-log" data-log-name="{escape(str(route_id), quote=True)}-step-{index}">'
                "<summary>Log</summary>"
                '<div class="log-actions"><button type="button" data-copy-log>Copy log</button>'
                '<button type="button" data-download-log>Download log</button>'
                '<span class="log-feedback" role="status"></span></div>'
                f"<pre>{escape(log)}</pre></details>"
            )
        count = sum(_valid_variant(v) for v in variants)
        rows.append(
            f'<tr class="thermo-step" data-step="{index}" data-status="{status}">'
            f'<th scope="row">{index}</th><td><span class="status {status}">{label}</span>{note}</td>'
            + "".join(
                f'<td class="energy">{_energy_range(variants, key)}</td>'
                for key in ENERGIES
            )
            + f"<td>{_number(annotation.get('balancing_confidence'))}</td>"
            + f'<td>{count} / {len(variants)}</td><td class="step-details">{details}</td></tr>'
        )
    return (
        '<div class="thermo-scroll"><table class="thermo-steps">'
        "<caption>Step thermochemistry · kcal/mol · products minus reactants</caption>"
        "<thead><tr><th>Step</th><th>Status</th><th>ΔE</th><th>ΔH</th><th>ΔG</th>"
        "<th>Balance confidence</th><th>Calculated / total variants</th><th>Details</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _manifest(routes, supplied):
    # Read the actual calculation provenance, not defaults from today's config.
    recipes = {}
    for route in routes:
        for step in route:
            calculation = (step.thermochemistry or {}).get("calculation") or {}
            if not calculation:
                continue
            identity = calculation.get("identity") or {}
            recipe = {
                k: calculation[k]
                for k in ("config", "method", "reference_conventions")
                if k in calculation
            }
            recipe["calculator"] = identity.get("calculator")
            key = json.dumps(recipe, sort_keys=True)
            recipes[key] = recipe
    if not recipes and supplied:
        recipes["run"] = {
            k: supplied[k] for k in ("config", "calculator") if k in supplied
        }
    context = deepcopy(supplied or {})
    for key in ("config", "calculator", "calculations"):
        context.pop(key, None)
    return {**context, "calculations": list(recipes.values())}


def _conditions(recipes):
    configs = {
        json.dumps(r.get("config", {}), sort_keys=True): r.get("config", {})
        for r in recipes
    }
    if not configs:
        return "Calculation conditions unavailable."
    if len(configs) > 1:
        return "Multiple calculation settings are present; see the Manifest for each set of conditions."
    config = next(iter(configs.values()))
    overrides = config.get("symmetry_numbers", {})
    concentration = config.get("concentration_mol_l")
    standard_state = (
        f"concentration: {escape(str(concentration))} mol/L"
        if concentration is not None
        else f"pressure: {escape(str(config.get('pressure', '—')))} Pa"
    )
    return (
        f"Temperature: {escape(str(config.get('temperature', '—')))} K; "
        f"{standard_state}. "
        "Rotational symmetry number defaults to 1 unless overridden per species."
        + (
            f" {len(overrides)} per-species overrides are recorded in the Manifest."
            if overrides
            else ""
        )
    )


THERMO_CSS = """
[hidden]{display:none!important}.thermo-title{display:flex;align-items:center;justify-content:space-between;gap:20px}
.manifest-button{padding:9px 14px;cursor:pointer;font:inherit;border:1px solid var(--rule);background:white;border-radius:4px}
.thermo-overview{margin-top:18px;color:var(--ink2)}.thermo-conditions{color:var(--ink2)}
.thermo-scroll{overflow-x:auto;padding:0 18px 16px}.thermo-steps, .variants table{width:100%;border-collapse:collapse;font-size:13px}
.thermo-steps caption{text-align:left;padding:14px 0;font-weight:600}.thermo-steps td,.thermo-steps th,.variants td,.variants th{padding:10px 8px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:top}
.thermo-steps thead{background:#f4f7fa}.energy{white-space:nowrap}.range-label{display:block;font-size:10px;color:var(--ink3)}
.status{font-weight:600;white-space:nowrap}.processed{color:#216145}.filtered_out{color:#80621c}.errored{color:#a62f36}
.filter-reason{font-size:12px;max-width:260px;overflow-wrap:anywhere;margin-top:4px}.step-details{min-width:120px;max-width:450px}
.step-details details{margin:0 0 6px}.step-details summary{cursor:pointer;color:#194c7e}.step-details code{overflow-wrap:anywhere;white-space:normal}
.thermo-log pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:340px;overflow:auto;min-width:250px}
.log-actions{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.log-actions button{cursor:pointer}.log-feedback{font-size:11px}
#thermo-manifest{width:min(850px,92vw);max-height:85vh;border:1px solid var(--rule);border-radius:6px;padding:24px}
#thermo-manifest::backdrop{background:rgba(15,20,25,.4)}#thermo-manifest pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
.manifest-head{display:flex;justify-content:space-between;align-items:center;gap:16px}.manifest-head h2{margin:0}
.run-aborted{padding:14px;background:#fceeed;color:#a62f36;border:1px solid #a62f36;margin-top:16px}
"""

THERMO_JS = r"""
(() => {
  const opener = document.getElementById('open-manifest');
  const dialog = document.getElementById('thermo-manifest');
  opener.addEventListener('click', () => dialog.showModal());
  document.getElementById('close-manifest').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => opener.focus());
  document.addEventListener('click', async event => {
    const button = event.target.closest('[data-copy-log],[data-download-log]');
    if (!button) return;
    const panel = button.closest('.thermo-log');
    const text = panel.querySelector('pre').textContent;
    if (button.hasAttribute('data-download-log')) {
      save('route-' + panel.dataset.logName.replace(/[^a-z0-9_-]/gi, '_') + '.json',
           new Blob([text], {type: 'application/json'}));
    } else {
      const feedback = panel.querySelector('.log-feedback');
      try {
        await navigator.clipboard.writeText(text);
        feedback.textContent = 'Copied';
      } catch (_) {
        const range = document.createRange();
        range.selectNodeContents(panel.querySelector('pre'));
        const selection = window.getSelection();
        selection.removeAllRanges(); selection.addRange(range);
        feedback.textContent = 'Log selected; use Copy or Download log.';
      }
    }
  });
})();
"""


def thermochemistry_report_html(
    routes,
    html_path=None,
    *,
    manifest=None,
    route_labels=None,
    show_policy_log_likelihood=False,
    phase=None,
):
    """Render all routes in input order, with compact tables below each drawing.

    The report is self-contained and never loads model dependencies. Manifest
    metadata supplies run/input information; actual settings are collected from
    step annotations. Forward annotations are retained in Route JSON only.
    """
    from synplan.utils.visualisation.assets import REPORT_CSS, REPORT_JS

    routes = list(routes)
    if phase is not None:
        if phase not in {"gas", "cpcm_water"}:
            raise ValueError("Unknown thermochemistry phase")

        def selected(step):
            record = step.thermochemistry or {}
            primary = (
                record.get("calculation", {}).get("config", {}).get("phase", "gas")
            )
            return (
                step
                if primary == phase
                else replace(
                    step, thermochemistry=record.get("additional_phases", {}).get(phase)
                )
            )

        routes = [
            replace(route, steps=tuple(selected(step) for step in route.steps))
            for route in routes
        ]
    labels = (
        list(route_labels)
        if route_labels is not None
        else [
            str(
                r.provenance.tree_node_id
                if r.provenance and r.provenance.tree_node_id is not None
                else i
            )
            for i, r in enumerate(routes, 1)
        ]
    )
    if len(labels) != len(routes):
        raise ValueError("Route labels must match the number of routes")
    provenance = _manifest(routes, manifest)
    phases = {
        r.get("config", {}).get("phase", "gas") for r in provenance["calculations"]
    }
    phase_label = (
        "Gas-phase"
        if phases == {"gas"}
        else "CPCM-water"
        if phases == {"cpcm_water"}
        else "Phase-specific"
    )
    counts = Counter(display_status(s.thermochemistry) for r in routes for s in r)
    with _temporary_render_config(mapping=False):
        doc, layouts = Doc(), {}
        cards = []
        for index, (route, label) in enumerate(zip(routes, labels, strict=True), 1):
            score = route.provenance.search_score if route.provenance else None
            policy = (
                f'<div class="kv"><div class="eyebrow">Policy log likelihood</div><div class="v">'
                f"{_number(route.provenance.policy_log_likelihood if route.provenance else None)}</div></div>"
                if show_policy_log_likelihood
                else ""
            )
            cards.append(
                f'<section class="route card" id="route-{index}" data-route-id="{escape(str(label), quote=True)}">'
                '<div class="rhead"><div class="kv"><div class="eyebrow">Route ID</div>'
                f'<div class="v id">{escape(str(label))}</div></div>'
                f'<div class="kv"><div class="eyebrow">Search score</div><div class="v">{_number(score)}</div></div>'
                f'<div class="kv"><div class="eyebrow">Steps</div><div class="v">{len(route)}</div></div>'
                + policy
                + '<div class="acts"><button class="act" data-act="zoom">Zoom</button>'
                '<button class="act" data-act="svg">SVG</button><button class="act" data-act="png">PNG</button></div></div>'
                + f'<div class="draw">{doc.route(route.svg(standalone=False, layouts=layouts))}</div>'
                + _step_table(route, label)
                + "</section>"
            )
        target = (
            molecule_svg(drawable_copy(routes[0].target, layouts)) if routes else ""
        )
        banner = (
            '<p class="run-aborted">Run aborted. Unprocessed steps are marked Errored — run aborted; see their logs.</p>'
            if provenance.get("run_status") == "aborted"
            else ""
        )
        page = (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>SynPlanner thermochemistry report</title>"
            f"<style>{ROUTE_CSS}{REPORT_CSS}{THERMO_CSS}</style></head><body>"
            + hidden_defs(ARROW_DEFS, doc.defs())
            + '<div class="wrap"><header class="page card">'
            '<div class="thermo-title"><h1>Route thermochemistry</h1>'
            '<button class="manifest-button" id="open-manifest" aria-haspopup="dialog" aria-controls="thermo-manifest">Manifest</button></div>'
            f'<div class="target"><div class="tile">{target}</div></div>'
            f'<p class="thermo-overview">{len(routes)} routes · {sum(counts.values())} steps · '
            f"{counts['processed']} processed · {counts['filtered_out']} filtered out · {counts['errored']} errored</p>"
            f'<p class="thermo-conditions">{_conditions(provenance["calculations"])}</p>'
            f"<p>{phase_label}, single-conformer RRHO values for each balanced equation. "
            "Min-max ranges cover calculated stereochemical variants; expand Variants for individual results. "
            "These values do not establish selectivity or reaction barriers.</p>"
            + banner
            + "</header>"
            + "".join(cards)
            + "</div>"
            '<dialog id="thermo-manifest" aria-labelledby="manifest-title"><div class="manifest-head">'
            '<h2 id="manifest-title">Calculation manifest</h2><button id="close-manifest" autofocus>Close</button></div>'
            + "<pre>"
            + escape(json.dumps(provenance, indent=2, ensure_ascii=False))
            + "</pre></dialog>"
            + "<script>"
            + REPORT_JS
            + THERMO_JS
            + "</script></body></html>"
        )
    if html_path is None:
        return page
    Path(html_path).write_text(page, encoding="utf-8")
    return None
