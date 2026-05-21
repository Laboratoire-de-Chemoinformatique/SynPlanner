from synplan.chem.reaction import CanonicalRetroReactor, apply_reaction_rule
from synplan.chem.utils import is_useful_symmetric_reaction_rule, mol_from_smiles


def test_useful_symmetric_rule_requires_automorphism_filter_off():
    rule_smarts = (
        "[c:1]:[c:2](:[c:3])-[c:5](:[c:4]):[c:6]>>"
        "[c:4]:[c:5](:[c:6])-[B:8](-[O:9])-[O:10]."
        "[c:1]:[c:2](:[c:3])-[I:7]"
    )
    assert is_useful_symmetric_reaction_rule(rule_smarts)

    x_molecule = mol_from_smiles("C1=CC(C2C=CC=NC=2)=CC(C)=C1")
    reactor = CanonicalRetroReactor.from_smarts(
        rule_smarts, delete_atoms=False, automorphism_filter=False
    )

    assert len(list(apply_reaction_rule(x_molecule, reactor))) == 2

    reactor._automorphism_filter = True
    assert len(list(apply_reaction_rule(x_molecule, reactor))) == 1
