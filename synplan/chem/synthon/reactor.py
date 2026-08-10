"""Rule application for synthons: a Transformer that keeps the atom family and stamps the tokens."""

from collections.abc import Iterable, Iterator

from chython import smarts
from chython.containers import MoleculeContainer, QueryContainer, SynthonContainer
from chython.containers.synthon import restore_synthons
from chython.reactor import Transformer


class SynthonRuleError(ValueError):
    """A rule the loader refuses: a label where it would be silently inert."""


def query_labels(template: QueryContainer) -> dict[int, str]:
    """{atom-map number: token} for every labelled atom of a parsed template."""
    return {
        n: a._label
        for n, a in template.atoms()
        if getattr(a, "_label", None) is not None
    }


class SynthonTransformer(Transformer):
    """Transformer that keeps the synthon atom family and stamps the rule's labels.

    The labels are read off the REPLACEMENT template, which carries them inline. There is no
    synthon_labels argument and no slots table: the SMARTS string is the rule.
    """

    def __init__(
        self, pattern: QueryContainer, replacement: QueryContainer, **kwargs
    ) -> None:
        if bad := query_labels(pattern):
            # QueryElement.__eq__ never consults the label, so a reactant-side token matches
            # exactly what the bare bracket matches - silently inert rather than wrong-but-loud.
            raise SynthonRuleError(
                f"synthon label on the reactant side is not a query constraint: {bad}"
            )
        super().__init__(pattern, replacement, **kwargs)
        self._synthon_labels = query_labels(replacement)

    @classmethod
    def from_smarts(cls, rule_smarts: str, **kwargs) -> "SynthonTransformer":
        """Build from one reaction-SMARTS string. Never rebuild a rule from str(Reactor): the
        SMARTS writer drops the token, so the source string is the artefact."""
        left, right = rule_smarts.split(">>", 1)
        return cls(smarts(left.strip()), smarts(right.strip()), **kwargs)

    def __call__(self, structure: MoleculeContainer) -> Iterator[SynthonContainer]:
        for mapping in self._pattern.get_mapping(
            structure, automorphism_filter=self._automorphism_filter
        ):
            transformed = self._patcher(structure, mapping)
            yield restore_synthons(
                transformed,
                {
                    mapping[n]: token
                    for n, token in self._synthon_labels.items()
                    if n in mapping
                },
            )


def load_rules(records: Iterable[dict]) -> list[tuple[dict, SynthonTransformer]]:
    """(record, transformer) per rule record, so callers keep the id/name/flags beside the rule."""
    return [(r, SynthonTransformer.from_smarts(r["smarts"])) for r in records]


__all__ = ["SynthonRuleError", "SynthonTransformer", "load_rules", "query_labels"]
