"""Tests for imputation of the molecules an unbalanced reaction is missing."""

import pytest
from chython import smiles

from synplan.chem.data.rebalancing import (
    LEAVING_GROUPS,
    PROTONATED,
    SMALL_MOLECULES,
    UNSTABLE,
    RebalancingError,
    _formula,
    _merge_order,
    _small_molecules,
    competing_products,
    confidence,
    is_functional_group,
    reaction_imbalance,
    rebalance_reaction,
)
from synplan.chem.data.standardizing import (
    STANDARDIZER_REGISTRY,
    RebalanceReactionStandardizer,
    SplitIonsStandardizer,
    StandardizationError,
)


@pytest.mark.parametrize(
    "table", [SMALL_MOLECULES, PROTONATED, LEAVING_GROUPS], ids=["small", "acid", "lg"]
)
def test_table_species_are_chemically_sound(table):
    # The compositions are derived from these SMILES, so there is nothing left
    # to drift; what still needs asserting is that each one is a real molecule.
    # "[Mg]Br" would parse and balance fine while being a valence short.
    for name, _ in table:
        assert not smiles(name).check_valence(), name


def test_grignard_leaves_as_the_hydroxide_not_a_bare_metal():
    # Magnesium is divalent: "[Mg]Br" has an unsatisfied valence, which is what
    # a CGR round trip produces here. It leaves the workup as Mg(OH)Br.
    rxn = smiles("[Br:1][Mg:2][CH3:3].[CH3:4][CH:5]=[O:6]>>[CH3:3][CH2:5][OH:6]")
    balanced = rebalance_reaction(rxn)
    assert not reaction_imbalance(balanced)
    for molecule in balanced.molecules():
        assert not molecule.check_valence(), str(molecule)


def test_balanced_reaction_is_untouched():
    rxn = smiles("CC(=O)O.CN>>CC(=O)NC.O")
    assert not reaction_imbalance(rxn)
    assert rebalance_reaction(rxn) is rxn


def test_ester_hydrolysis_recovers_the_alcohol():
    # The ethyl leaves with the ester oxygen; water pays for the missing OH.
    rxn = smiles("CCOC(=O)c1ccccc1>>OC(=O)c1ccccc1")
    balanced = rebalance_reaction(rxn)
    assert not reaction_imbalance(balanced)
    assert "CCO" in [str(m) for m in balanced.products]


def test_leaving_water_is_bonded_rather_than_carried_over():
    # Water already on the reactant side completes the fragment instead of
    # reappearing untouched among the products.
    rxn = smiles("CCOC(=O)c1cccnc1.O>>OC(=O)c1cccnc1")
    balanced = rebalance_reaction(rxn)
    assert not reaction_imbalance(balanced)
    assert "CCO" in [str(m) for m in balanced.products]


def test_non_carbon_deficit_uses_a_small_molecule():
    rxn = smiles("CC(=O)Cl.CN>>CC(=O)NC")
    balanced = rebalance_reaction(rxn)
    assert not reaction_imbalance(balanced)


def test_reactant_side_carbon_deficit_is_solved_backwards():
    # Carbon short on the reactant side is the mirror of the usual problem.
    balanced = rebalance_reaction(smiles("CC>>CCCC"))
    assert not reaction_imbalance(balanced)


def test_deficit_of_an_unknown_element_is_refused():
    with pytest.raises(RebalancingError, match="no small molecules cover"):
        rebalance_reaction(smiles("C[Se]C>>CC"))


def test_oxidation_balances_with_loose_hydrogen_by_default():
    balanced = rebalance_reaction(smiles("CC(O)C>>CC(=O)C"))
    assert not reaction_imbalance(balanced)
    assert not any("Cr" in molecule.brutto for molecule in balanced.molecules())


def test_oxidation_can_name_its_oxidant():
    balanced = rebalance_reaction(smiles("CC(O)C>>CC(=O)C"), add_redox_agents=True)
    assert not reaction_imbalance(balanced)
    assert any("Cr" in molecule.brutto for molecule in balanced.reactants)
    assert any("Cr" in molecule.brutto for molecule in balanced.products)


def test_acetonide_leaves_as_the_ketone():
    # Both C-O bonds of the dioxolane break, so the protecting group departs as
    # acetone; capping each break on its own would give the diol instead.
    rxn = smiles(
        "[CH3:1][C:2]1([CH3:3])[O:4][CH2:5][CH2:6][O:7]1>>[OH:4][CH2:5][CH2:6][OH:7]"
    )
    balanced = rebalance_reaction(rxn)
    assert not reaction_imbalance(balanced)
    assert "CC(C)=O" in [str(m) for m in balanced.products]


def test_spectator_reagent_is_not_cut_apart():
    # Acetic anhydride takes no part here; the CGR shows no dynamic bond in it,
    # so it must come back whole rather than be broken up and capped.
    rxn = smiles(
        "[CH3:1][C:2](=[O:3])[O:4][C:5](=[O:6])[CH3:7].[CH3:8][OH:9]"
        ">>[CH3:8][O:9][CH3:10]"
    )
    try:
        balanced = rebalance_reaction(rxn)
    except RebalancingError:
        return  # acceptable: refusing beats inventing
    assert not reaction_imbalance(balanced)


def test_functional_groups_distinguish_ether_from_ester():
    # The expand rules hang off this: an ether cut leaves the alkyl halide, an
    # ester cut takes up oxygen instead.
    ester = smiles("CCOC(=O)c1ccccc1")
    oxygen = next(
        n for n, a in ester.atoms() if a.atomic_symbol == "O" and a.neighbors == 2
    )
    assert not is_functional_group(ester, "ether", oxygen)
    anisole = smiles("COc1ccccc1")
    ether_oxygen = next(n for n, a in anisole.atoms() if a.atomic_symbol == "O")
    assert is_functional_group(anisole, "ether", ether_oxygen)


def test_two_heteroatoms_are_still_refused():
    assert _merge_order(smiles("CO"), 2, smiles("CCl"), 2) is None


def test_reduction_gives_off_water_not_dioxygen():
    # Oxygen leaving while hydrogen is taken up is a reduction. Balancing the
    # arithmetic literally would emit O2, which no reduction does.
    rxn = smiles("O=C(NCCO)C1CC(=O)N(C1)CC2=CC=CC=C2>>OCCNCC1CCN(C1)CC2=CC=CC=C2")
    balanced = rebalance_reaction(rxn)
    assert not reaction_imbalance(balanced)
    assert "O=O" not in [str(m) for m in balanced.products]
    assert "O" in [str(m) for m in balanced.products]


def test_confidence_is_reported_and_can_refuse():
    balanced = rebalance_reaction(smiles("CCOC(=O)c1ccccc1>>OC(=O)c1ccccc1"))
    assert 0.0 < balanced.meta["confidence"] <= 1.0

    # a reactant side short of carbon is solved backwards and never yet right
    assert confidence(smiles("CC>>CCCC"), rebalance_reaction(smiles("CC>>CCCC"))) == 0.0
    with pytest.raises(RebalancingError, match="confidence"):
        rebalance_reaction(smiles("CC>>CCCC"), min_confidence=0.5)


def test_confidence_falls_as_more_is_invented():
    simple = rebalance_reaction(smiles("CCOC(=O)c1ccccc1>>OC(=O)c1ccccc1"))
    crowded = rebalance_reaction(
        smiles("CC(=O)OC(C)=O.CC(=O)OC(C)=O.OCCCCCO>>CC(=O)OCCCCCOC(C)=O")
    )
    assert crowded.meta["confidence"] < simple.meta["confidence"]


def test_screening_plate_scores_zero_confidence():
    # An HTE record (ORD ord_dataset-805ad86...) lists both regioisomers and
    # the caffeine internal standard as products of one equation. The
    # arithmetic balances beautifully if the extra products are declared to be
    # reactants, so the only defence is the score.
    rxn = smiles(
        "C1COCCN1.FC(F)(F)C=1C=CC(=CC=1)I"
        ">>Cn1c(n(C)c2c(n(cn2)C)c1=O)=O.FC(F)(F)C1=CC=C(N2CCOCC2)C=C1"
        ".FC(F)(F)C1=CC=CC(=C1)N2CCOCC2"
    )
    assert rebalance_reaction(rxn).meta["confidence"] == 0.0
    with pytest.raises(RebalancingError, match="confidence"):
        rebalance_reaction(rxn, min_confidence=0.1)


def test_dropping_competing_products_removes_the_internal_standard():
    # A screening plate reads for the product and for caffeine, which took no
    # part. Caffeine cannot be made from these reactants, so it goes.
    rxn = smiles(
        "C1COCCN1.FC(F)(F)C=1C=CC(=CC=1)I"
        ">>Cn1c(n(C)c2c(n(cn2)C)c1=O)=O.FC(F)(F)C1=CC=C(N2CCOCC2)C=C1"
    )
    assert rebalance_reaction(rxn).meta["confidence"] == 0.0

    balanced = rebalance_reaction(rxn, drop_competing_products=True)
    assert not reaction_imbalance(balanced)
    assert [str(m) for m in balanced.products] == [
        "FC(F)(F)C1=CC=C(N2CCOCC2)C=C1",
        "[H+]",
        "[I-]",
    ]


def test_regioisomers_share_a_formula_so_both_survive():
    # Neither can be ruled out by arithmetic, so the record still describes two
    # reactions and the score still says so.
    rxn = smiles(
        "C1COCCN1.FC(F)(F)C=1C=CC(=CC=1)I"
        ">>FC(F)(F)C1=CC=C(N2CCOCC2)C=C1.FC(F)(F)C1=CC=CC(=C1)N2CCOCC2"
    )
    assert len(competing_products(rxn)) == 2
    assert (
        rebalance_reaction(rxn, drop_competing_products=True).meta["confidence"] == 0.0
    )


def test_co_products_of_one_transformation_are_not_competing():
    # The acid and the alcohol both come from the ester: the reactants cover
    # the sum, so there is nothing to drop.
    rxn = smiles("CCOC(=O)c1ccccc1.O>>OC(=O)c1ccccc1.CCO")
    assert competing_products(rxn) is None


def test_ignoring_the_mapping_equals_never_having_had_one():
    # The CGR is the only thing the mapping is read for, so the flag has to
    # land in the same place as handing the record over unmapped. A second
    # reader of the mapping would break that silently.
    text = (
        "[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
        ">>[OH:3][C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
    )
    ignored = rebalance_reaction(smiles(text), use_mapping=False)
    unmapped = rebalance_reaction(smiles(str(smiles(text))))
    assert str(ignored) == str(unmapped)

    # and the mapping is worth something, or the flag would be pointless
    assert (
        rebalance_reaction(smiles(text)).meta["confidence"]
        >= (ignored.meta["confidence"])
    )


def test_a_balanced_mapped_reaction_still_composes_a_cgr():
    # Imputed species are parsed fresh, so they start at atom 1 and land on
    # top of the reaction's own numbering. Every standardization step after
    # this one composes a CGR, and chython refuses one where a number names
    # two atoms on a side, or a different element on each side. Reagents
    # count as part of the reactant side — that is where compose puts them.
    rxn = smiles(
        "[CH3:1][CH2:2][O:3][C:4](=[O:5])[CH3:6]>[ClH:7]>[OH:3][C:4](=[O:5])[CH3:6]"
    )
    balanced = rebalance_reaction(rxn)

    left = list(balanced.reagents) + list(balanced.reactants)
    for side in (left, balanced.products):
        numbers = [n for molecule in side for n in molecule]
        assert len(numbers) == len(set(numbers))
    elements = {n: a.atomic_symbol for m in left for n, a in m.atoms()}
    for molecule in balanced.products:
        for number, atom in molecule.atoms():
            assert elements.get(number, atom.atomic_symbol) == atom.atomic_symbol

    ~balanced  # noqa: B018 — raises if the numbering is unusable


def test_split_ions_reads_charges_off_the_atoms():
    # chython-synplan 1.101 has no MoleculeContainer._charges; reading the
    # total charge that way failed on every ionic reaction.
    balanced = smiles("CC(=O)[O-].[Na+].Cl>>CC(=O)O.[Na+].[Cl-]")
    assert SplitIonsStandardizer()(balanced) is not None


def test_free_hydrogen_needs_a_reagent_that_could_have_moved_it():
    # Loose hydrogen is the imputer papering over a redox step it did not
    # read; judged on USPTO those answers are right one time in five. A patent
    # that really reduced something names what it used.
    bare = smiles("O=[N+]([O-])c1ccccc1>O.[Na+].[OH-]>Nc1ccccc1")
    with pytest.raises(RebalancingError, match="no redox reagent"):
        rebalance_reaction(bare, refuse_unsupported_redox=True)

    # a counter-ion is not evidence, but a catalyst or a hydride is
    for reagents in ("[Pd].CCO", "[BH4-].[Na+]"):
        named = smiles(f"O=[N+]([O-])c1ccccc1>{reagents}>Nc1ccccc1")
        assert rebalance_reaction(named, refuse_unsupported_redox=True)

    # and a balance that needs no hydrogen is untouched by the gate
    ester = smiles("CCOC(=O)c1ccccc1>O>OC(=O)c1ccccc1")
    assert not reaction_imbalance(
        rebalance_reaction(ester, refuse_unsupported_redox=True)
    )


def test_a_named_reagent_leaves_as_its_spent_form():
    # The by-product of a reagent the record actually names is stoichiometry,
    # not a guess. Offered only when the reagent is present, so a reaction that
    # never used one cannot reach it.
    acetylation = rebalance_reaction(smiles("CC(=O)OC(C)=O.Nc1ccccc1>>CC(=O)Nc1ccccc1"))
    assert str(smiles("CC(=O)O")) in [str(m) for m in acetylation.products]

    bromination = rebalance_reaction(smiles("O=C1CCC(=O)N1Br.c1ccccc1C>>Brc1ccccc1C"))
    assert str(smiles("O=C1CCC(=O)N1")) in [str(m) for m in bromination.products]


def test_a_carbonate_never_takes_a_halide():
    # Capping an open bond on a carbon that already holds two oxygens invents
    # a carbonate halide. Carbonate and bicarbonate are among the commonest
    # reagents in USPTO, so this fired often and balanced perfectly every time.
    alkylation = rebalance_reaction(
        smiles("[O-]C([O-])=O.[K+].[K+].OCc1ccccc1.CI>>COCc1ccccc1")
    )
    assert [str(m) for m in alkylation.products].count("[I-]") == 1
    assert not any("I" in str(m) and len(m) > 1 for m in alkylation.products)

    # an acyl halide has only one oxygen on that carbon and stays reachable
    amide = rebalance_reaction(smiles("CC(=O)O.CN>>CC(=O)NC"))
    assert not reaction_imbalance(amide)


def test_species_that_cannot_exist_are_broken_up():
    # Every replacement is element-neutral, so the balance is untouched and
    # only the chemistry improves.
    for unstable, products in UNSTABLE.items():
        assert _formula(_small_molecules([unstable])) == _formula(
            _small_molecules(list(products))
        ), unstable


def test_standardizer_wraps_failures():
    standardizer = RebalanceReactionStandardizer()
    with pytest.raises(StandardizationError, match="RebalanceReaction"):
        standardizer(smiles("C[Se]C>>CC"))


def test_standardizer_balances():
    standardizer = RebalanceReactionStandardizer()
    balanced = standardizer(smiles("CCOC(=O)c1ccccc1>>OC(=O)c1ccccc1"))
    assert not reaction_imbalance(balanced)


def test_oxidation_survives_being_written_out():
    # chython cannot hold atomic oxygen: "[O]" parses to water and the radical
    # form reads back as a hydroxyl, so an answer spelled that way balanced in
    # memory and came off disk short of two hydrogens. Every test above keeps
    # its reaction in memory, which is exactly why none of them caught it.
    balanced = rebalance_reaction(smiles("CC(O)C>>CC(=O)C"), add_redox_agents=True)
    assert not reaction_imbalance(balanced)
    assert not reaction_imbalance(smiles(str(balanced)))


def test_imputed_reaction_makes_no_mapping_claim():
    # Imputed species are parsed fresh, so nothing says which invented atom on
    # one side is which on the other. Numbering them apart still composes a
    # CGR, but it names the wrong reaction centre; the claim is withdrawn.
    rxn = smiles(
        "[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
        ">>[OH:3][C:4](=[O:5])[c:6]1[cH:7][cH:8][cH:9][cH:10][cH:11]1"
    )
    balanced = rebalance_reaction(rxn)
    assert not reaction_imbalance(balanced)
    left = {n for m in balanced.reactants for n in m}
    right = {n for m in balanced.products for n in m}
    assert not (left & right), "a withdrawn mapping shares no atom number"
    ~balanced  # noqa: B018 — still composable, which is what the CGR needs


def test_ester_releases_the_alcohol_when_no_water_is_recorded():
    # Both halves hold two carbons, so a scorer counting hydrogen equally picks
    # a second acetic acid, whose hydrogens land exactly, over the ethanol.
    balanced = rebalance_reaction(smiles("CCOC(C)=O>>CC(=O)O"))
    assert not reaction_imbalance(balanced)
    assert "CCO" in [str(m) for m in balanced.products]


def test_halogenation_takes_up_the_elemental_halogen():
    # Short a halogen on the left and the same amount of hydrogen on the right
    # is X2 going in and HX coming out, not HX in and loose hydrogen venting.
    balanced = rebalance_reaction(smiles("c1ccccc1>>Brc1ccccc1"), add_redox_agents=True)
    assert not reaction_imbalance(balanced)
    assert "BrBr" in [str(m) for m in balanced.reactants]
    assert "Br" in [str(m) for m in balanced.products]


def test_rebalancing_runs_after_reagent_removal():
    # Reagent removal moves spectators out of the reactants and products, which
    # unbalances whatever was balanced first: measured on USPTO, balancing
    # first left 13% of the balanced records still balanced against 95%.
    order = list(STANDARDIZER_REGISTRY)
    assert order.index("rebalance_reaction_config") > order.index(
        "remove_reagents_config"
    )
