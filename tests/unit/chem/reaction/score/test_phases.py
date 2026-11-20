"""Independent standard states, reference cycles, phase availability and CSV resumes."""


def test_specified_stereo_and_mirror_policy():
    from synplan.chem.reaction.score.aimnet.geometry import matches_stereo
    from synplan.chem.reaction.score.aimnet.species import parse_species

    observed = parse_species("F[C@H](Cl)Br")
    assert matches_stereo(observed, parse_species("FC(Cl)Br"), True, False)
    assert not matches_stereo(observed, parse_species("F[C@@H](Cl)Br"), True, False)
    assert matches_stereo(observed, parse_species("F[C@@H](Cl)Br"), True, True)
    assert not matches_stereo(observed, parse_species("F[13C@H](Cl)Br"), True, True)
