"""One place spells a molecule, and this test is what keeps it that way.

``chython`` hands out ``canonicalize()`` as a method, so the cheapest way to
clean a molecule is always to call it on the spot. Doing that has cost this
project real bugs: the building-block catalogue is written with
``safe_canonicalization``, and a lookup spelled any other way silently misses
molecules the catalogue holds. The upstream tutorial teaches composing the
primitives by hand, which is why the habit keeps coming back.

Why this test is failing on you
-------------------------------

**You added a direct** ``molecule.canonicalize()``. The usual answer is to call
``synplan.chem.utils.safe_canonicalization`` instead; it is the same sequence
plus a guard for ``InvalidAromaticRing``, and it agrees with the catalogue. Note
it returns a *new* molecule and hands back the one it was given when chython
refuses, so ``result is molecule`` is how you detect that.

**Your call really is different.** Some are: the featuriser needs a kekulised
graph, the rebalancer must keep stereo the shared canonicalisers strip. Add the
file to ``ALLOWED`` with a sentence saying what makes it different. The list is
the argument, not a way around the test -- if the sentence is hard to write, the
call probably is not different.

**You removed the last direct call from an ``ALLOWED`` file.** The second test
fires: drop the entry. An exemption nobody needs is one that will silently cover
the next offender in that file.

**You moved or renamed a file.** ``HOME`` and the ``ALLOWED`` keys are literal
paths relative to ``synplan/``. Moving ``chem/utils.py`` or an exempt module
makes them stale, and the second test says which one.

**You called** ``canonicalize()`` **on something that is not a molecule.** The
check reads the AST and matches the method name, not the type, so any object
with a method of that name trips it. That is a genuine false positive -- say so
in ``ALLOWED``.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "synplan"

#: The one module allowed to canonicalise, because it is where the spelling is
#: decided: ``clean_molecule``, ``safe_canonicalization``,
#: ``validate_and_canonicalize``.
HOME = "chem/utils.py"

#: Direct calls that were examined and kept, with what makes them different.
ALLOWED = {
    "ml/featurization/molecules.py": (
        "the featuriser needs a kekulised graph and orders the steps itself; "
        "safe_canonicalization hands back an aromatic molecule"
    ),
    "chem/reaction/curation/rebalancing.py": (
        "keeps stereo on an imputed fragment, which every shared canonicaliser "
        "strips; already guarded by suppress(Exception)"
    ),
    "chem/reaction/routes/quality/protection/strategy.py": (
        "an example reagent parsed from the protection library, canonicalised "
        "inside the try that already covers parsing it"
    ),
}


def _calls_canonicalize(path: Path) -> bool:
    """True when the file calls ``.canonicalize()`` on something."""

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "canonicalize"
        ):
            return True
    return False


def test_only_chem_utils_canonicalises_molecules():
    offenders = sorted(
        relative
        for path in PACKAGE.rglob("*.py")
        for relative in [path.relative_to(PACKAGE).as_posix()]
        if relative != HOME and relative not in ALLOWED and _calls_canonicalize(path)
    )
    assert not offenders, (
        "call synplan.chem.utils.safe_canonicalization instead of "
        f"molecule.canonicalize() in: {', '.join(offenders)}. It is what the "
        "building-block catalogue is written with, so it is what a lookup has "
        "to be written with. If the call is genuinely different, add it to "
        "ALLOWED in this file with the reason."
    )


def test_every_allowed_exception_still_canonicalises():
    """An exemption nobody needs is an exemption that hides the next offender."""

    stale = sorted(
        relative for relative in ALLOWED if not _calls_canonicalize(PACKAGE / relative)
    )
    assert not stale, f"no longer calls canonicalize(), drop from ALLOWED: {stale}"
