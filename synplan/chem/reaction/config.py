"""Configuration for reaction-rule application."""

from pydantic import ConfigDict

from synplan.utils.config import BaseConfigModel


class ReactorConfig(BaseConfigModel):
    """Configuration for chython Reactor instances.

    Controls how Reactor objects are created when loading reaction rules.
    Intended for programmatic use, not config files.

    :param automorphism_filter: Baseline Chython match-deduplication setting.
        If True, duplicate matches are filtered for ordinary rules, while
        :func:`synplan.utils.loading.load_reaction_rules` may selectively
        override it for symmetry-breaking rules. If False, filtering is
        disabled for every rule.
    :param delete_atoms: If True, atoms in reactants but not in products are removed.
    :param one_shot: If True, do only single reaction center per application.

    .. note::
        ``fix_aromatic_rings`` and ``fix_tautomers`` are intentionally not
        exposed. SynPlanner's
        :class:`~synplan.chem.reaction.CanonicalRetroReactor` forces
        ``fix_aromatic_rings=False`` and runs the full kekule + thiele +
        tautomer-fix canonicalize pipeline inline in its ``_patcher``;
        tautomer fixing inside that inline call relies on chython's default
        ``fix_tautomers=True``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    automorphism_filter: bool = True
    delete_atoms: bool = False
    one_shot: bool = True

    def to_reactor_kwargs(self) -> dict:
        """Convert to kwargs dict for Reactor constructor."""
        return self.model_dump()
