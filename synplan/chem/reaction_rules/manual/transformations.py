"""Module containing hardcoded transformation reaction rules."""

from CGRtools import QueryContainer, ReactionContainer
from CGRtools.periodictable import ListElement

rules = []


def prepare():
    """Creates and returns three query containers and appends a reaction container to
    the "rules" list."""
    q_ = QueryContainer()
    p_ = QueryContainer()
    rules.append(ReactionContainer((q_,), (p_,)))
    return q_, p_


# aryl nitro reduction
# [C;Za;W1]-[N;D1]>>[O-]-[N+](-[C])=[O]
q, p = prepare()
q.add_atom("N", neighbors=1)
q.add_atom("C", hybridization=4, heteroatoms=1)
q.add_bond(1, 2, 1)

p.add_atom("N", charge=1)
p.add_atom("C")
p.add_atom("O", charge=-1)
p.add_atom("O")
p.add_bond(1, 2, 1)
p.add_bond(1, 3, 1)
p.add_bond(1, 4, 2)

# aryl nitration
# [O-]-[N+](=[O])-[C;Za;W12]>>[C]
q, p = prepare()
q.add_atom("N", charge=1)
q.add_atom("C", hybridization=4, heteroatoms=(1, 2))
q.add_atom("O", charge=-1)
q.add_atom("O")
q.add_bond(1, 2, 1)
q.add_bond(1, 3, 1)
q.add_bond(1, 4, 2)

p.add_atom("C", _map=2)

# Beckmann rearrangement (oxime -> amide)
# [C]-[N;D2]-[C]=[O]>>[O]-[N]=[C]-[C]
q, p = prepare()
q.add_atom("C")
q.add_atom("N", neighbors=2)
q.add_atom("O")
q.add_atom("C")
q.add_bond(1, 2, 1)
q.add_bond(1, 3, 2)
q.add_bond(2, 4, 1)

p.add_atom("C")
p.add_atom("N")
p.add_atom("O")
p.add_atom("C")
p.add_bond(1, 2, 2)
p.add_bond(2, 3, 1)
p.add_bond(1, 4, 1)

# aldehydes or ketones into oxime/imine reaction
# [C;Zd;W1]=[N]>>[C]=[O]
q, p = prepare()
q.add_atom("C", hybridization=2, heteroatoms=1)
q.add_atom("N")
q.add_bond(1, 2, 2)

p.add_atom("C")
p.add_atom("O", _map=3)
p.add_bond(1, 3, 2)

# addition of halogen atom into phenol ring (orto)
# [C](-[Cl,F,Br,I;D1]):[C]-[O,N;Zs]>>[C](-[A]):[C]
q, p = prepare()
q.add_atom(ListElement(["O", "N"]), hybridization=1)
q.add_atom("C")
q.add_atom("C")
q.add_atom(ListElement(["Cl", "F", "Br", "I"]), neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 4)
q.add_bond(3, 4, 1)

p.add_atom("A")
p.add_atom("C")
p.add_atom("C")
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 4)

# addition of halogen atom into phenol ring (para)
# [C](:[C]:[C]:[C]-[O,N;Zs])-[Cl,F,Br,I;D1]>>[A]-[C]:[C]:[C]:[C]
q, p = prepare()
q.add_atom(ListElement(["O", "N"]), hybridization=1)
q.add_atom("C")
q.add_atom("C")
q.add_atom("C")
q.add_atom("C")
q.add_atom(ListElement(["Cl", "F", "Br", "I"]), neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 4)
q.add_bond(3, 4, 4)
q.add_bond(4, 5, 4)
q.add_bond(5, 6, 1)

p.add_atom("A")
p.add_atom("C")
p.add_atom("C")
p.add_atom("C")
p.add_atom("C")
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 4)
p.add_bond(3, 4, 4)
p.add_bond(4, 5, 4)

# hard reduction of Ar-ketones
# [C;Za]-[C;D2;Zs;W0]>>[C]-[C]=[O]
q, p = prepare()
q.add_atom("C", hybridization=4)
q.add_atom("C", hybridization=1, neighbors=2, heteroatoms=0)
q.add_bond(1, 2, 1)

p.add_atom("C")
p.add_atom("C")
p.add_atom("O")
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 2)

# reduction of alpha-hydroxy pyridine
# [C;W1]:[N;H0;r6]>>[C](:[N])-[O]
q, p = prepare()
q.add_atom("C", heteroatoms=1)
q.add_atom("N", rings_sizes=6, hydrogens=0)
q.add_bond(1, 2, 4)

p.add_atom("C")
p.add_atom("N")
p.add_atom("O")
p.add_bond(1, 2, 4)
p.add_bond(1, 3, 1)

# Reduction of alkene
# [C]-[C;D23;Zs;W0]-[C;D123;Zs;W0]>>[C](-[C])=[C]
q, p = prepare()
q.add_atom("C")
q.add_atom("C", heteroatoms=0, neighbors=(2, 3), hybridization=1)
q.add_atom("C", heteroatoms=0, neighbors=(1, 2, 3), hybridization=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)

p.add_atom("C")
p.add_atom("C")
p.add_atom("C")
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 2)

# Kolbe-Schmitt reaction
# [C](:[C]-[O;D1])-[C](=[O])-[O;D1]>>[C](-[O]):[C]
q, p = prepare()
q.add_atom("O", neighbors=1)
q.add_atom("C")
q.add_atom("C")
q.add_atom("C")
q.add_atom("O", neighbors=1)
q.add_atom("O")
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 4)
q.add_bond(3, 4, 1)
q.add_bond(4, 5, 1)
q.add_bond(4, 6, 2)

p.add_atom("O")
p.add_atom("C")
p.add_atom("C")
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 4)

# reduction of carboxylic acid
# [O;D1]-[C;D2]-[C]>>[C]-[C](-[O])=[O]
q, p = prepare()
q.add_atom("C")
q.add_atom("C", neighbors=2)
q.add_atom("O", neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)

p.add_atom("C")
p.add_atom("C")
p.add_atom("O")
p.add_atom("O")
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 1)
p.add_bond(2, 4, 2)

# halogenation of alcohols
# [C;Zs]-[Cl,Br;D1]>>[C]-[O]
q, p = prepare()
q.add_atom("C", hybridization=1, heteroatoms=1)
q.add_atom(ListElement(["Cl", "Br"]), neighbors=1)
q.add_bond(1, 2, 1)

p.add_atom("C")
p.add_atom("O", _map=3)
p.add_bond(1, 3, 1)

# Kolbe nitrilation
# [N]#[C]-[C;Zs;W0]>>[Br]-[C]
q, p = prepare()
q.add_atom("C", heteroatoms=0, hybridization=1)
q.add_atom("C")
q.add_atom("N")
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 3)

p.add_atom("C")
p.add_atom("Br", _map=4)
p.add_bond(1, 4, 1)

# Nitrile hydrolysis
# [O;D1]-[C]=[O]>>[N]#[C]
q, p = prepare()
q.add_atom("C")
q.add_atom("O", neighbors=1)
q.add_atom("O")
q.add_bond(1, 2, 1)
q.add_bond(1, 3, 2)

p.add_atom("C")
p.add_atom("N", _map=4)
p.add_bond(1, 4, 3)

# sulfamidation
# [c]-[S](=[O])(=[O])-[N]>>[c]
q, p = prepare()
q.add_atom("C", hybridization=4)
q.add_atom("S")
q.add_atom("O")
q.add_atom("O")
q.add_atom("N", neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 2)
q.add_bond(2, 4, 2)
q.add_bond(2, 5, 1)

p.add_atom("C")

# Ring expansion rearrangement
#
q, p = prepare()
q.add_atom("C")
q.add_atom("N")
q.add_atom("C", rings_sizes=6)
q.add_atom("C")
q.add_atom("O")
q.add_atom("C")
q.add_atom("C")
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 1)
q.add_bond(4, 5, 2)
q.add_bond(3, 6, 1)
q.add_bond(4, 7, 1)

p.add_atom("C")
p.add_atom("N")
p.add_atom("C")
p.add_atom("C")
p.add_atom("O")
p.add_atom("C")
p.add_atom("C")
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 2)
p.add_bond(3, 4, 1)
p.add_bond(4, 5, 1)
p.add_bond(4, 6, 1)
p.add_bond(4, 7, 1)

# hydrolysis of bromide alkyl
#
q, p = prepare()
q.add_atom("C", hybridization=1)
q.add_atom("O", neighbors=1)
q.add_bond(1, 2, 1)

p.add_atom("C")
p.add_atom("Br")
p.add_bond(1, 2, 1)

# Condensation of ketones/aldehydes and amines into imines
#
q, p = prepare()
q.add_atom("N", neighbors=(1, 2))
q.add_atom("C", neighbors=(2, 3), heteroatoms=1)
q.add_bond(1, 2, 2)

p.add_atom("C", _map=2)
p.add_atom("O")
p.add_bond(2, 3, 2)

# Halogenation of alkanes
#
q, p = prepare()
q.add_atom("C", hybridization=1)
q.add_atom(ListElement(["F", "Cl", "Br"]))
q.add_bond(1, 2, 1)

p.add_atom("C")

# heterocyclization
#
q, p = prepare()
q.add_atom("N", heteroatoms=0, hybridization=1, neighbors=(2, 3))
q.add_atom("C", heteroatoms=2)
q.add_atom("N", heteroatoms=0, neighbors=2)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 2)

p.add_atom("N")
p.add_atom("C")
p.add_atom("N")
p.add_atom("O")
p.add_bond(1, 2, 1)
p.add_bond(2, 4, 2)

# Reduction of nitrile
#
q, p = prepare()
q.add_atom("N", neighbors=1)
q.add_atom("C")
q.add_atom("C", hybridization=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)

p.add_atom("N")
p.add_atom("C")
p.add_atom("C")
p.add_bond(1, 2, 3)
p.add_bond(2, 3, 1)

# SPECIAL CASE
# Reduction of nitrile into methylamine
#
q, p = prepare()
q.add_atom("C", neighbors=1)
q.add_atom("N", neighbors=2)
q.add_atom("C")
q.add_atom("C", hybridization=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 1)

p.add_atom("N", _map=2)
p.add_atom("C")
p.add_atom("C")
p.add_bond(2, 3, 3)
p.add_bond(3, 4, 1)

# methylation of amides
#
q, p = prepare()
q.add_atom("O")
q.add_atom("C")
q.add_atom("N")
q.add_atom("C", neighbors=1)
q.add_bond(1, 2, 2)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 1)

p.add_atom("O")
p.add_atom("C")
p.add_atom("N")
p.add_bond(1, 2, 2)
p.add_bond(2, 3, 1)

# hydrocyanation of alkenes
#
q, p = prepare()
q.add_atom("C", hybridization=1)
q.add_atom("C")
q.add_atom("C")
q.add_atom("N")
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 3)

p.add_atom("C")
p.add_atom("C")
p.add_bond(1, 2, 2)

# decarbocylation (alpha atom of nitrile)
#
q, p = prepare()
q.add_atom("N")
q.add_atom("C")
q.add_atom("C", neighbors=2)
q.add_bond(1, 2, 3)
q.add_bond(2, 3, 1)

p.add_atom("N")
p.add_atom("C")
p.add_atom("C")
p.add_atom("C")
p.add_atom("O")
p.add_atom("O")
p.add_bond(1, 2, 3)
p.add_bond(2, 3, 1)
p.add_bond(3, 4, 1)
p.add_bond(4, 5, 2)
p.add_bond(4, 6, 1)

# Bichler-Napieralski reaction
#
q, p = prepare()
q.add_atom("C", rings_sizes=(6,))
q.add_atom("C", rings_sizes=(6,))
q.add_atom("N", rings_sizes=(6,), neighbors=2)
q.add_atom("C")
q.add_atom("C")
q.add_atom("C")
q.add_atom("O")
q.add_atom("O")
q.add_atom("C")
q.add_atom("O", neighbors=1)
q.add_bond(1, 2, 4)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 1)
q.add_bond(4, 5, 2)
q.add_bond(5, 6, 1)
q.add_bond(6, 7, 2)
q.add_bond(6, 8, 1)
q.add_bond(5, 9, 4)
q.add_bond(9, 10, 1)
q.add_bond(1, 9, 1)

p.add_atom("C")
p.add_atom("C")
p.add_atom("N")
p.add_atom("C")
p.add_atom("C")
p.add_atom("C")
p.add_atom("O")
p.add_atom("O")
p.add_atom("C")
p.add_atom("O")
p.add_atom("O")
p.add_bond(1, 2, 4)
p.add_bond(2, 3, 1)
p.add_bond(3, 4, 1)
p.add_bond(4, 5, 2)
p.add_bond(5, 6, 1)
p.add_bond(6, 7, 2)
p.add_bond(6, 8, 1)
p.add_bond(5, 9, 1)
p.add_bond(9, 10, 2)
p.add_bond(9, 11, 1)

# heterocyclization in Prins reaction
#
q, p = prepare()
q.add_atom("C")
q.add_atom("O")
q.add_atom("C")
q.add_atom(ListElement(["N", "O"]), neighbors=2)
q.add_atom("C")
q.add_atom("C")
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 1)
q.add_bond(4, 5, 1)
q.add_bond(5, 6, 1)
q.add_bond(1, 6, 1)

p.add_atom("C")
p.add_atom("C", _map=5)
p.add_bond(1, 5, 2)

# recyclization of tetrahydropyran through an opening the ring and dehydration
#
q, p = prepare()
q.add_atom("C")
q.add_atom("C")
q.add_atom("C")
q.add_atom(ListElement(["N", "O"]))
q.add_atom("C")
q.add_atom("C")
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)
q.add_bond(3, 4, 1)
q.add_bond(4, 5, 1)
q.add_bond(5, 6, 1)
q.add_bond(1, 6, 2)

p.add_atom("C")
p.add_atom("C")
p.add_atom("C")
p.add_atom("A")
p.add_atom("C")
p.add_atom("C")
p.add_atom("O")
p.add_bond(1, 2, 1)
p.add_bond(1, 7, 1)
p.add_bond(3, 7, 1)
p.add_bond(3, 4, 1)
p.add_bond(4, 5, 1)
p.add_bond(5, 6, 1)
p.add_bond(1, 6, 1)

# alkenes + h2o/hHal
#
q, p = prepare()
q.add_atom("C", hybridization=1)
q.add_atom("C", hybridization=1)
q.add_atom(ListElement(["O", "F", "Cl", "Br", "I"]), neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)

p.add_atom("C")
p.add_atom("C")
p.add_bond(1, 2, 2)

# methylation of dimethylamines
#
q, p = prepare()
q.add_atom("C", neighbors=1)
q.add_atom("N", neighbors=3)
q.add_bond(1, 2, 1)

p.add_atom("N", _map=2)

__all__ = ["rules"]
