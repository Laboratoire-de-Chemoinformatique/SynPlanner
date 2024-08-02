"""Module containing hardcoded decomposition reaction rules."""

from CGRtools import QueryContainer, ReactionContainer
from CGRtools.periodictable import ListElement

rules = []


def prepare():
    """Creates and returns three query containers and appends a reaction container to
    the "rules" list."""
    q_ = QueryContainer()
    p1_ = QueryContainer()
    p2_ = QueryContainer()
    rules.append(ReactionContainer((q_,), (p1_, p2_)))

    return q_, p1_, p2_


# R-amide/ester formation
# [C](-[N,O;D23;Zs])(-[C])=[O]>>[A].[C]-[C](-[O])=[O]
q, p1, p2 = prepare()
q.add_atom("C")
q.add_atom("C")
q.add_atom("O")
q.add_atom(ListElement(["N", "O"]), hybridization=1, neighbors=(2, 3))
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 2)
q.add_bond(2, 4, 1)

p1.add_atom("C")
p1.add_atom("C")
p1.add_atom("O")
p1.add_atom("O", _map=5)
p1.add_bond(1, 2, 1)
p1.add_bond(2, 3, 2)
p1.add_bond(2, 5, 1)

p2.add_atom("A", _map=4)

# acyl group addition with aromatic carbon's case (Friedel-Crafts)
# [C;Za]-[C](-[C])=[O]>>[C].[C]-[C](-[Cl])=[O]
q, p1, p2 = prepare()
q.add_atom("C")
q.add_atom("C")
q.add_atom("O")
q.add_atom("C", hybridization=4)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 2)
q.add_bond(2, 4, 1)

p1.add_atom("C")
p1.add_atom("C")
p1.add_atom("O")
p1.add_atom("Cl", _map=5)
p1.add_bond(1, 2, 1)
p1.add_bond(2, 3, 2)
p1.add_bond(2, 5, 1)

p2.add_atom("C", _map=4)

# Williamson reaction
# [C;Za]-[O]-[C;Zs;W0]>>[C]-[Br].[C]-[O]
q, p1, p2 = prepare()
q.add_atom("C", hybridization=4)
q.add_atom("O")
q.add_atom("C", hybridization=1, heteroatoms=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)

p1.add_atom("C")
p1.add_atom("O")
p1.add_bond(1, 2, 1)

p2.add_atom("C", _map=3)
p2.add_atom("Br")
p2.add_bond(3, 4, 1)

# Buchwald-Hartwig amination
# [N;D23;Zs;W0]-[C;Za]>>[C]-[Br].[N]
q, p1, p2 = prepare()
q.add_atom("N", heteroatoms=0, hybridization=1, neighbors=(2, 3))
q.add_atom("C", hybridization=4)
q.add_bond(1, 2, 1)

p1.add_atom("C", _map=2)
p1.add_atom("Br")
p1.add_bond(2, 3, 1)

p2.add_atom("N")

# imidazole imine atom's alkylation
# [C;r5](:[N;r5]-[C;Zs;W1]):[N;D2;r5]>>[C]-[Br].[N]:[C]:[N]
q, p1, p2 = prepare()
q.add_atom("N", rings_sizes=5)
q.add_atom("C", rings_sizes=5)
q.add_atom("N", rings_sizes=5, neighbors=2)
q.add_atom("C", hybridization=1, heteroatoms=(1, 2))
q.add_bond(1, 2, 4)
q.add_bond(2, 3, 4)
q.add_bond(1, 4, 1)

p1.add_atom("N")
p1.add_atom("C")
p1.add_atom("N")
p1.add_bond(1, 2, 4)
p1.add_bond(2, 3, 4)

p2.add_atom("C", _map=4)
p2.add_atom("Br")
p2.add_bond(4, 5, 1)

# Knoevenagel condensation (nitryl and carboxyl case)
# [C]=[C](-[C]#[N])-[C](-[O])=[O]>>[C]=[O].[C](-[C]#[N])-[C](-[O])=[O]
q, p1, p2 = prepare()
q.add_atom("C")
q.add_atom("C")
q.add_atom("C")
q.add_atom("N")
q.add_atom("C")
q.add_atom("O")
q.add_atom("O")
q.add_bond(1, 2, 2)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 3)
q.add_bond(2, 5, 1)
q.add_bond(5, 6, 2)
q.add_bond(5, 7, 1)

p1.add_atom("C", _map=2)
p1.add_atom("C")
p1.add_atom("N")
p1.add_atom("C")
p1.add_atom("O")
p1.add_atom("O")
p1.add_bond(2, 3, 1)
p1.add_bond(3, 4, 3)
p1.add_bond(2, 5, 1)
p1.add_bond(5, 6, 2)
p1.add_bond(5, 7, 1)

p2.add_atom("C", _map=1)
p2.add_atom("O", _map=8)
p2.add_bond(1, 8, 2)

# Knoevenagel condensation (double nitryl case)
# [C]=[C](-[C]#[N])-[C]#[N]>>[C]=[O].[C](-[C]#[N])-[C]#[N]
q, p1, p2 = prepare()
q.add_atom("C")
q.add_atom("C")
q.add_atom("C")
q.add_atom("N")
q.add_atom("C")
q.add_atom("N")
q.add_bond(1, 2, 2)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 3)
q.add_bond(2, 5, 1)
q.add_bond(5, 6, 3)

p1.add_atom("C", _map=2)
p1.add_atom("C")
p1.add_atom("N")
p1.add_atom("C")
p1.add_atom("N")
p1.add_bond(2, 3, 1)
p1.add_bond(3, 4, 3)
p1.add_bond(2, 5, 1)
p1.add_bond(5, 6, 3)

p2.add_atom("C", _map=1)
p2.add_atom("O", _map=8)
p2.add_bond(1, 8, 2)

# Knoevenagel condensation (double carboxyl case)
# [C]=[C](-[C](-[O])=[O])-[C](-[O])=[O]>>[C]=[O].[C](-[C](-[O])=[O])-[C](-[O])=[O]
q, p1, p2 = prepare()
q.add_atom("C")
q.add_atom("C")
q.add_atom("C")
q.add_atom("O")
q.add_atom("O")
q.add_atom("C")
q.add_atom("O")
q.add_atom("O")
q.add_bond(1, 2, 2)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 2)
q.add_bond(3, 5, 1)
q.add_bond(2, 6, 1)
q.add_bond(6, 7, 2)
q.add_bond(6, 8, 1)

p1.add_atom("C", _map=2)
p1.add_atom("C")
p1.add_atom("O")
p1.add_atom("O")
p1.add_atom("C")
p1.add_atom("O")
p1.add_atom("O")
p1.add_bond(2, 3, 1)
p1.add_bond(3, 4, 2)
p1.add_bond(3, 5, 1)
p1.add_bond(2, 6, 1)
p1.add_bond(6, 7, 2)
p1.add_bond(6, 8, 1)

p2.add_atom("C", _map=1)
p2.add_atom("O", _map=9)
p2.add_bond(1, 9, 2)

# heterocyclization with guanidine
# [c]((-[N;W0;Zs])@[n]@[c](-[N;D1])@[c;W0])@[n]@[c]-[O; D1]>>[C](-[N])(=[N])-[N].[C](#[N])-[C]-[C](-[O])=[O]
q, p1, p2 = prepare()
q.add_atom("C")
q.add_atom("N", heteroatoms=0, hybridization=1)
q.add_atom("N")
q.add_atom("C")
q.add_atom("N", neighbors=1)
q.add_atom("C", heteroatoms=0)
q.add_atom("N")
q.add_atom("C")
q.add_atom("O", neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(1, 3, 4)
q.add_bond(3, 4, 4)
q.add_bond(4, 5, 1)
q.add_bond(4, 6, 4)
q.add_bond(1, 7, 4)
q.add_bond(7, 8, 4)
q.add_bond(8, 9, 1)

p1.add_atom("C")
p1.add_atom("N")
p1.add_atom("N")
p1.add_atom("N", _map=7)
p1.add_bond(1, 2, 1)
p1.add_bond(1, 3, 2)
p1.add_bond(1, 7, 1)

p2.add_atom("C", _map=4)
p2.add_atom("N")
p2.add_atom("C")
p2.add_atom("C", _map=8)
p2.add_atom("O", _map=9)
p2.add_atom("O")
p2.add_bond(4, 5, 3)
p2.add_bond(4, 6, 1)
p2.add_bond(6, 8, 1)
p2.add_bond(8, 9, 2)
p2.add_bond(8, 10, 1)

# alkylation of amine
# [C]-[N]-[C]>>[C]-[N].[C]-[Br]
q, p1, p2 = prepare()
q.add_atom("C")
q.add_atom("N")
q.add_atom("C")
q.add_atom("C")
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)
q.add_bond(2, 4, 1)

p1.add_atom("C")
p1.add_atom("N")
p1.add_atom("C")
p1.add_bond(1, 2, 1)
p1.add_bond(2, 3, 1)

p2.add_atom("C", _map=4)
p2.add_atom("Cl")
p2.add_bond(4, 5, 1)

# Synthesis of guanidines
#
q, p1, p2 = prepare()
q.add_atom("N")
q.add_atom("C")
q.add_atom("N", hybridization=1)
q.add_atom("N", hybridization=1)
q.add_bond(1, 2, 2)
q.add_bond(2, 3, 1)
q.add_bond(2, 4, 1)

p1.add_atom("N")
p1.add_atom("C")
p1.add_atom("N")
p1.add_bond(1, 2, 3)
p1.add_bond(2, 3, 1)

p2.add_atom("N", _map=4)

# Grignard reaction with nitrile
#
q, p1, p2 = prepare()
q.add_atom("C")
q.add_atom("C")
q.add_atom("O")
q.add_atom("C")
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 2)
q.add_bond(2, 4, 1)

p1.add_atom("C")
p1.add_atom("C")
p1.add_atom("N")
p1.add_bond(1, 2, 1)
p1.add_bond(2, 3, 3)

p2.add_atom("C", _map=4)
p2.add_atom("Br")
p2.add_bond(4, 5, 1)

# Alkylation of alpha-carbon atom of nitrile
#
q, p1, p2 = prepare()
q.add_atom("N")
q.add_atom("C")
q.add_atom("C", neighbors=(3, 4))
q.add_atom("C", hybridization=1)
q.add_bond(1, 2, 3)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 1)

p1.add_atom("N")
p1.add_atom("C")
p1.add_atom("C")
p1.add_bond(1, 2, 3)
p1.add_bond(2, 3, 1)

p2.add_atom("C", _map=4)
p2.add_atom("Cl")
p2.add_bond(4, 5, 1)

# Gomberg-Bachmann reaction
#
q, p1, p2 = prepare()
q.add_atom("C", hybridization=4, heteroatoms=0)
q.add_atom("C", hybridization=4, heteroatoms=0)
q.add_bond(1, 2, 1)

p1.add_atom("C")
p1.add_atom("N", _map=3)
p1.add_bond(1, 3, 1)

p2.add_atom("C", _map=2)

# Cyclocondensation
#
q, p1, p2 = prepare()
q.add_atom("N", neighbors=2)
q.add_atom("C")
q.add_atom("C")
q.add_atom("C")
q.add_atom("N")
q.add_atom("C")
q.add_atom("C")
q.add_atom("O", neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 1)
q.add_bond(4, 5, 2)
q.add_bond(5, 6, 1)
q.add_bond(6, 7, 1)
q.add_bond(7, 8, 2)
q.add_bond(1, 7, 1)

p1.add_atom("N")
p1.add_atom("C")
p1.add_atom("C")
p1.add_atom("C")
p1.add_atom("O", _map=9)
p1.add_bond(1, 2, 1)
p1.add_bond(2, 3, 1)
p1.add_bond(3, 4, 1)
p1.add_bond(4, 9, 2)

p2.add_atom("N", _map=5)
p2.add_atom("C")
p2.add_atom("C")
p2.add_atom("O")
p2.add_atom("O", _map=10)
p2.add_bond(5, 6, 1)
p2.add_bond(6, 7, 1)
p2.add_bond(7, 8, 2)
p2.add_bond(7, 10, 1)

# heterocyclization dicarboxylic acids
#
q, p1, p2 = prepare()
q.add_atom("C", rings_sizes=(5, 6))
q.add_atom("O")
q.add_atom(ListElement(["O", "N"]))
q.add_atom("C", rings_sizes=(5, 6))
q.add_atom("O")
q.add_bond(1, 2, 2)
q.add_bond(1, 3, 1)
q.add_bond(3, 4, 1)
q.add_bond(4, 5, 2)

p1.add_atom("C")
p1.add_atom("O")
p1.add_atom("O", _map=6)
p1.add_bond(1, 2, 2)
p1.add_bond(1, 6, 1)

p2.add_atom("C", _map=4)
p2.add_atom("O")
p2.add_atom("O", _map=7)
p2.add_bond(4, 5, 2)
p2.add_bond(4, 7, 1)

__all__ = ["rules"]
