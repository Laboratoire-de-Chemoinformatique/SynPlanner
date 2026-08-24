"""The disconnection rules and a fragmentation DAG as depictable tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from chython import smarts, synthon_smiles

from synplan.chem.synthon.config import SynthonConfig, load_data
from synplan.chem.utils import safe_canonicalization
from synplan.utils.frames import ChemFrame


def _rule_reaction(record: Mapping[str, Any]) -> tuple[str, Any]:
    """The reaction SMARTS a record is best read as, and its depictable form.

    A ring record ships a hand-authored ``retro_smarts`` naming the real reagents; its raw
    ``smarts`` shows the two cut bonds instead, which reads as the wrong chemistry.
    """
    text = (
        record["retro_smarts"]
        if record["ring"] and record["retro_smarts"]
        else record["smarts"]
    )
    reaction = smarts(text)
    reaction.clean2d()
    return text, reaction


def _kind(record: Mapping[str, Any]) -> str:
    """Which of the three disconnection families a record belongs to."""
    if record["macro"]:
        return "macro"
    return "ring" if record["ring"] else "acyclic"


def rules_frame(
    rules: Iterable[Any] | None = None, *, config: SynthonConfig | None = None
) -> ChemFrame:
    """One row per synthon disconnection rule, the rule itself depicted.

    :param rules: Loaded rule objects carrying ``rule_id``, as returned by
        :func:`~synplan.chem.reaction.rules.synthon.synthon_priority_rules`. Restricts and
        orders the frame to what was actually loaded; ``None`` gives every shipped record,
        macrocyclic half included, which is what a kind/provenance census wants.
    :param config: Supplies the ``rules.json`` path.
    :return: A frame of ``id, name, kind, provenance, reaction_name, forms, reagents,
        rule, name, kind, provenance, reaction_name, forms, reagents,
    supersedes, smarts``, where ``kind`` is ``macro``/``ring``/``acyclic``.
    """
    records = load_data((config or SynthonConfig()).rules_path)["disconnections"]
    if rules is not None:
        by_id = {record["id"]: record for record in records}
        records = [by_id[rule.rule_id] for rule in rules]
    rows = []
    for record in records:
        text, reaction = _rule_reaction(record)
        rows.append(
            {
                "id": record["id"],
                "rule": reaction,
                "name": record["name"],
                "kind": _kind(record),
                "provenance": record["provenance"],
                "reaction_name": record["reaction_name"],
                "forms": record["forms"],
                "reagents": record["reagents"],
                "supersedes": record["supersedes"],
                "smarts": text,
            }
        )
    return ChemFrame(rows, depict_columns=["rule"])


def synthons_frame(dag: Any, stock: Mapping[str, Any] | None = None) -> ChemFrame:
    """One row per synthon of each fragmentation pathway, most available pathway first.

    :param dag: A :class:`~synplan.chem.synthon.fragment.DisconnectionDAG`.
    :param stock: The synthon stock the pathways were scored against. Without it ``in_stock``
        is ``None`` rather than a uniform ``False``, which would read as "nothing is stocked".
    :return: A frame of ``pathway, rules, depth, availability, synthon, labels, in_stock,
        smiles``, where ``pathway`` numbers the pathways in ``best_available`` order.
    """
    rows = []
    for index, pathway in enumerate(dag.best_available()):
        for smi in pathway.key:
            synthon = safe_canonicalization(synthon_smiles(smi))
            synthon.clean2d()
            rows.append(
                {
                    "pathway": index,
                    "rules": "|".join(pathway.rules),
                    "depth": pathway.depth,
                    "availability": pathway.availability,
                    "synthon": synthon,
                    "labels": "+".join(sorted(synthon.synthon_labels.values())),
                    "in_stock": None if stock is None else smi in stock,
                    "smiles": smi,
                }
            )
    return ChemFrame(rows, depict_columns=["synthon"])


def demo() -> None:
    from synplan.chem.reaction.rules.synthon import (
        SYNTHON_SOURCE_NAME,
        synthon_priority_rules,
    )
    from synplan.chem.synthon.fragment import fragment_smiles

    shipped = rules_frame()
    assert set(shipped.df["kind"]) == {"acyclic", "ring", "macro"}, (
        "every kind is shipped"
    )
    assert "<svg" in shipped.head(1)._repr_html_(), "the rule column depicts"
    loaded = synthon_priority_rules()[SYNTHON_SOURCE_NAME]
    selected = rules_frame(loaded)
    assert list(selected.df["id"]) == [rule.rule_id for rule in loaded], "order is kept"
    assert len(selected) < len(shipped), "the default selection is a subset"
    ring = shipped.df[shipped.df["kind"] == "ring"].iloc[0]
    assert "_" not in ring["smarts"].split(">>")[1], (
        "a ring rule shows its reagent form"
    )

    dag = fragment_smiles("CC(=O)NCc1ccccc1")
    synthons = synthons_frame(dag)
    assert len(synthons) == sum(len(p.key) for p in dag.pathways.values()), (
        "row per synthon"
    )
    assert set(synthons.df["labels"]) <= {"elec", "nuc", "elec+nuc"}, (
        "labels are tokens"
    )
    assert synthons.df["in_stock"].isna().all(), "no stock means unknown, not absent"
    assert "<svg" in synthons._repr_html_()
    stocked = next(iter(dag.pathways.values())).key[0]
    assert list(synthons_frame(dag, {stocked: set()}).df["in_stock"]).count(True) == 1
    print("synthon frames demo ok")


__all__ = ["rules_frame", "synthons_frame"]
