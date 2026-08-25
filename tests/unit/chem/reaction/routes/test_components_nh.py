"""An N-H heterocycle must survive the RouteCGR projection or no catalogue lookup matches."""

from chython import smiles

from synplan.chem.reaction.routes.representation.components import (
    route_cgr_pseudo_reactants_by_role,
)


def test_aromatic_nh_survives_pseudo_reactant_projection():
    reaction = smiles(
        "Clc1ncnc2[nH]ccc12.OB(O)c1ccccc1>>c1ccccc1-c1ncnc2[nH]ccc12"
    )
    reaction.canonicalize()
    azole = smiles("Clc1ncnc2[nH]ccc12")
    azole.canonicalize()

    roles = route_cgr_pseudo_reactants_by_role(~reaction)
    pieces = {str(m) for m in roles["real_bb"] + roles["supporting"]}

    assert str(azole) in pieces, (
        "decompose() drops the implicit H on aromatic N; without recanonicalisation "
        f"the azole comes back unmatchable. got {sorted(pieces)}"
    )
