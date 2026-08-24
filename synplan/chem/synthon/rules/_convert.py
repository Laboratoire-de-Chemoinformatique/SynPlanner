"""Offline converter: Synt-On's XML/JSON knowledge base -> the chython-dialect JSON that ships.

Run by hand; the output is committed and reviewed in the diff. Nothing translates at import time.
This is the only place Synt-On's integer reaction-centre codes exist — everything downstream is
the eight token strings.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ElementTree
from collections import defaultdict
from pathlib import Path

from chython import smarts, synthon_smiles
from chython.exceptions import InvalidAromaticRing
from chython.periodictable import AnyElement

from synplan.chem.synthon.rules._dialect import DialectError, to_chython
from synplan.chem.synthon.rules.validate import shifted_labels
from synplan.chem.synthon.transformer import SynthonTransformer, query_labels
from synplan.chem.utils import safe_canonicalization

# the one-time migration. Code 11 ("electrophilic nitrogen") collapses into 'elec': marksCombinations
# has no N:10 key, so on nitrogen "electrophile" already has exactly one meaning.
PAPER_CODE_TO_LABEL = {
    10: "elec",
    11: "elec",
    20: "nuc",
    21: "elecB",
    30: "elec2",
    40: "nuc2",
    50: "neut2",
    60: "elec*",
    70: "nuc*",
}

# upstream __marksCombinations (SyntOn.py), as (element, aromatic, code) pairs. Read verbatim, then
# migrated. The relation is symmetric upstream and stays symmetric here.
MARKS_COMBINATIONS = {
    "C:10": ["N:20", "O:20", "C:20", "c:20", "n:20", "S:20"],
    "c:10": ["N:20", "O:20", "C:20", "c:20", "n:20", "S:20"],
    "c:20": ["N:11", "C:10", "c:10"],
    "C:20": ["C:10", "c:10"],
    "c:21": ["N:20", "O:20", "n:20"],
    "C:21": ["N:20", "n:20"],
    "N:20": ["C:10", "c:10", "C:21", "c:21", "S:10"],
    "N:11": ["c:20"],
    "n:20": ["C:10", "c:10", "C:21", "c:21"],
    "O:20": ["C:10", "c:10", "c:21"],
    "S:20": ["C:10", "c:10"],
    "S:10": ["N:20"],
    "C:30": ["C:40", "N:40"],
    "C:40": ["C:30"],
    "C:50": ["C:50"],
    "C:70": ["C:60", "c:60"],
    "c:60": ["C:70"],
    "C:60": ["C:70"],
    "N:40": ["C:30"],
}
# F7: c:70 is produced by BB synthonisation (Boronics_BF3andMIDA, Bifunctional, SulfonesSulfinates)
# but has no upstream row, so any stocked aryl-BF3/MIDA/aryl-sulfinate synthon raised KeyError.
F7_ADDITIONS = {"c:70": ["C:60", "c:60"]}
# F18: the Suzuki pairing. Upstream's table is only a whole-molecule pre-filter for its seed walk -
# the bond is formed by each rule's ReconstructionReaction - so the table never had to list the
# boronate partner. Here the table IS the join, so without these rows R12.1/12.2/12.6 disconnect a
# biaryl and then reassemble to nothing. Read straight off those SMIRKS ([#23] = code 10,
# [#108] = code 21); note that none of them licenses C:10 + c:21, so neither do we. The plan's own
# Appendix A states this pairing ("in Suzuki the boronate is the nucleophile") while Appendix C
# omits it - this row set resolves that contradiction in favour of Appendix A.
F18_ADDITIONS = {"c:10": ["c:21", "C:21"], "C:10": ["C:21"]}

# upstream forbiddenMarks (SyntOn.py). Four entries are dead — [c:70] x3 and [c:11] x1 are never
# emitted by any of the 39 rules — and {N:11, N:11} collapses to a one-element frozenset that bans
# every mono-functional umpolung-nitrogen synthon, killing R3.3 hierarchically (F5).
FORBIDDEN_MARKS = [
    {"N:11", "c:10"},
    {"N:11", "c:20"},
    {"N:11", "O:20"},
    {"N:11", "C:30"},
    {"N:11", "C:10"},
    {"N:11"},
    {"N:11", "c:70"},
    {"N:11", "C:40"},
    {"N:11", "C:50"},
    {"N:11", "S:10"},
    {"c:11", "C:20"},
    {"c:70", "c:20"},
    {"c:21", "C:60"},
    {"c:21", "N:11"},
    {"c:20", "c:21"},
    {"c:70", "C:21"},
    {"c:20", "C:21"},
]

# Heterocyclisation: authored here, not in the upstream XML, which has no ring-forming rule at all.
# A ring synthon is an ordinary H-capped labelled fragment of the PRODUCT carrying two labels — the
# synthons of a triazole are a benzyl triazene and a styrene, not an azide and an alkyne — so the
# record format, the SMARTS dialect and the token set are unchanged; only `ring` is new.
# The fourth field is the target `check()` fires the rule on. chython's `D` counts HEAVY
# neighbours, so `[n;D3]` never matches an N-H azole and each such family needs an a/b twin pair.
#
# Three dialect rules, each found by a rule that silently misfired:
#  - the RHS bond orders are load-bearing: `[#7:2]=[#7:3]` de-aromatises, omitting the `=` gives a
#    saturated fragment;
#  - every mapped atom whose bonds must survive has to appear on the RHS, adjacent to its partner,
#    or that bond is lost;
#  - a ring-fusion atom needs `[c;R2:n]` to pin the traversal direction. `r5`/`r6` do not behave as
#    Daylight would suggest here.
# And one chemistry rule: inside a tautomerisable amide / amidine / thioamide / enol triad, label
# both ends or neither. chython moves the proton and leaves the label where it was, which is
# silently wrong unless the two atoms are the same element carrying the same token - then they are
# interchangeable and nothing moved. `check()` enforces exactly that through `validate`.
RING_RULES = [
    (
        "R16.1a",
        "1,2,3-triazole, N1-substituted / azide + alkyne (CuAAC, RuAAC, SPAAC, thermal Huisgen)",
        "[n;D3;+0;!$(n-[#7]);!$(n-[#8]);!$([n][#6]=[#8]):1]1[n;+0;D2:2][n;+0;D2:3][c;!$(c(:a)(:a):a):4][c;!$(c(:a)(:a):a):5]1"
        ">>[#7_elec:1][#7:2]=[#7_nuc:3].[#6_elec:4]=[#6_nuc:5]",
        "c1ccccc1Cn1cc(-c2ccccc2)nn1",
    ),
    (
        "R16.1b",
        "1,2,3-triazole, N-unsubstituted (1H) / azide (NaN3, TMS-N3) + alkyne [n;h1] twin of R16.1a",
        "[n;h1;+0:1]1[n;+0;D2:2][n;+0;D2:3][c;!$(c(:a)(:a):a):4][c;!$(c(:a)(:a):a):5]1"
        ">>[#7_elec:1][#7:2]=[#7_nuc:3].[#6_elec:4]=[#6_nuc:5]",
        "c1ccccc1-c1cn[nH]n1",
    ),
    (
        "R16.2a",
        "tetrazole, 1,5-disubstituted / organic azide + nitrile",
        "[n;D3;+0;!$(n-[#7]);!$(n-[#8]);!$([n][#6]=[#8]);!$([n]S(=O)=O):1]1[n;+0;D2:2][n;+0;D2:3][n;+0;D2:4][c;!$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][#7:2]=[#7_elec:3].[#7_nuc:4]=[#6_elec:5]",
        "Cn1nnnc1-c1ccccc1",
    ),
    (
        "R16.2b",
        "tetrazole, 5-substituted N-unsubstituted (1H) / NaN3 + nitrile, [n;h1] twin of R16.2a",
        "[n;h1;+0:1]1[n;+0;D2:2][n;+0;D2:3][n;+0;D2:4][c;!$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][#7:2]=[#7_elec:3].[#7_nuc:4]=[#6_elec:5]",
        "c1ccccc1-c1nnn[nH]1",
    ),
    (
        "R16.3a",
        "pyrazole, N1-substituted / Knorr (substituted hydrazine + 1,3-dicarbonyl, enaminone or ynone)",
        "[n;D3;+0;!$(n-[#7]);!$(n-[#8]);!$([n][#6]=[#8]);!$([n]S(=O)=O):1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][c;!$(c(:a)(:a):a):4][c;!$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][#7_nuc2:2].[#6_elec2:3][#6:4]=[#6_elec:5]",
        "O=S(=O)(N)c1ccc(cc1)-n1nc(cc1-c1ccc(C)cc1)C(F)(F)F",
    ),
    (
        "R16.3b",
        "pyrazole, N-unsubstituted (1H) / Knorr with hydrazine hydrate, [n;h1] twin of R16.3a",
        "[n;h1;+0:1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][c;!$(c(:a)(:a):a):4][c;!$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][#7_nuc2:2].[#6_elec2:3][#6:4]=[#6_elec:5]",
        "Cc1cc(-c2ccccc2)[nH]n1",
    ),
    (
        "R16.4b",
        "imidazole, N-unsubstituted (1H) / amidine + alpha-halo ketone",
        "[n;h1;+0;R1:1]1[c;R1;+0:2][n;D2;h0;+0:3][c;R1;+0:4][c;R1;+0:5]1"
        ">>[#7_nuc:1][#6:2]=[#7_nuc:3].[#6_elec:4]=[#6_elec:5]",
        "CCCCc1nc(Cl)c(CO)[nH]1",
    ),
    (
        "R16.5",
        "isoxazole / hydroxylamine + 1,3-dicarbonyl (Claisen-type condensation)",
        "[o;+0:1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][c;!$(c(:a)(:a):a):4][c;!$(c(:a)(:a):a):5]1"
        ">>[#8_nuc:1][#7_nuc2:2].[#6_elec2:3][#6:4]=[#6_elec:5]",
        "Cc1onc(-c2ccccc2)c1-c1ccc(cc1)S(N)(=O)=O",
    ),
    (
        "R16.6a",
        "pyridine / Kroehnke-Bohlmann-Rahtz (enamine + enone), orientation A",
        "[n;D2;+0:1]1[c;x2;!$(c[#7,#8,#16]):2][c;x2:3][c;x2;!$(c!@[OH]):4][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):5][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):6]1"
        ">>[#7_nuc:1]=[#6:2][#6_nuc2:3].[#6_elec2:4][#6:5]=[#6_elec:6]",
        "Cc1cccnc1",
    ),
    (
        "R16.6b",
        "pyridine / Kroehnke-Bohlmann-Rahtz, orientation B (mirror twin of R16.6a)",
        "[n;D2;+0:1]1[c;x2;!$(c[#8;h1]);!$(c[#16;h1]):2][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):3][c;x2;!$(c!@[OH]):4][c;x2:5][c;x2;!$(c[#7,#8,#16]):6]1"
        ">>[#7_nuc:1]=[#6:6][#6_nuc2:5].[#6_elec2:4][#6:3]=[#6_elec:2]",
        "Cc1cccnc1",
    ),
    (
        "R16.7",
        "pyridazine / hydrazine + 1,4-dicarbonyl; also phthalazine from an ortho-diacylarene",
        "[n;D2;+0:1]1[n;D2;+0:2][c;x2;!$(c!@[OH]):3][c;!$(c[#8;h1]);!$(c[#16;h1]):4][c;!$(c[#8;h1]);!$(c[#16;h1]):5][c;x2;!$(c!@[OH]):6]1"
        ">>[#7_nuc2:1][#7_nuc2:2].[#6_elec2:3][#6:4]=[#6:5][#6_elec2:6]",
        "Cc1cccnn1",
    ),
    (
        "R16.8",
        "indole / Fischer (arylhydrazine + ketone)",
        "[n;+0;!$([n][#6]=[#8]);!$([n]S(=O)=O);!$(n-[#7]);!$(n-[#8]):1]1[c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):3][c;R2;$(c(:a)(:a):a);$(c1ccccc1):4][c;R2;$(c(:a)(:a):a);$(c1ccccc1):5]1"
        ">>[#6_elec:2]=[#6_elec:3].[c_nuc:4][c:5][#7_nuc:1]",
        "Cc1cc2ccccc2[nH]1",
    ),
    (
        "R16.9",
        "quinoline / Friedlaender (2-aminoaryl ketone + ketone)",
        "[n;D2;+0:1]1[c;x2;!$(c!@[OH]):2][c;x2:3][c;x2;!$(c!@[OH]):4][c;R2;!$(c[A]):5][c;R2;!$(c[A]):6]1"
        ">>[#7_nuc2:1][c:6][c:5][#6_elec2:4].[#6_elec2:2][#6_nuc2:3]",
        "Cc1cc(C)c2ccccc2n1",
    ),
    (
        "R17.1",
        "Paal-Knorr pyrrole (1,4-dicarbonyl + NH3/RNH2)",
        "[n;+0;!$([n][#6]=[#8]);!$([n]S(=O)=O);!$(n-[#7]);!$(n-[#8]);R1:1]1[c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][c;!$(c(:a)(:a):a):3][c;!$(c(:a)(:a):a):4][c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):5]1"
        ">>[#7_nuc:1][#6:2]=[#6:3][#6:4]=[#6_elec:5]",
        "Cn1c(CC(=O)O)ccc1C(=O)c1ccc(C)cc1",
    ),
    (
        "R17.2",
        "Paal-Knorr thiophene (1,4-dicarbonyl + P4S10/Lawesson; also SH- + 1,3-diyne)",
        "[s;+0;R1:1]1[c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][c;!$(c(:a)(:a):a):3][c;!$(c(:a)(:a):a):4][c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):5]1"
        ">>[#16_nuc:1][#6:2]=[#6:3][#6:4]=[#6_elec:5]",
        "Cc1ccc(C)s1",
    ),
    (
        "R17.9",
        "Hinsberg thiophene (thiodiglycolate + 1,2-dicarbonyl)",
        "[s;+0:1]1[c;!$(c(:a)(:a):a);$(c!@[#6](=[#8])[#8]):2][c;!$(c(:a)(:a):a):3][c;!$(c(:a)(:a):a):4][c;!$(c(:a)(:a):a);$(c!@[#6](=[#8])[#8]):5]1"
        ">>[#6_nuc2:2][#16:1][#6_nuc2:5].[#6_elec2:3][#6_elec2:4]",
        "CCOC(=O)c1sc(C(=O)OCC)c(-c2ccccc2)c1-c1ccccc1",
    ),
    (
        "R17.12",
        "indole / Leimgruber-Batcho (covers Cadogan-Sundberg)",
        "[n;h1;+0;!$([n][#6]=[#8]);!$([n]S(=O)=O):1]1[c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][c;!$(c(:a)(:a):a):3][c;R2;$(c(:a)(:a):a);$(c1ccccc1):4][c;R2;$(c(:a)(:a):a);$(c1ccccc1):5]1"
        ">>[#7_nuc:1][c:5][c:4][#6:3]=[#6_elec:2]",
        "CNS(=O)(=O)Cc1ccc2[nH]cc(CCN(C)C)c2c1",
    ),
    (
        "R17.13",
        "indole / Larock heteroannulation (Bartoli maps to the same two bonds)",
        "[n;+0;!$(n-[#7]);!$(n-[#8]):1]1[c;h0;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][c;h0;!$(c(:a)(:a):a):3][c;R2;$(c(:a)(:a):a):4][c;R2;$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][c:5][c_elec:4].[#6_nuc:3]=[#6_elec:2]",
        "Cc1[nH]c2ccccc2c1C",
    ),
    (
        "R17.14",
        "indole / Madelung (and the Houlihan and Smith variants)",
        "[n;+0;!$([n][#6]=[#8]);!$([n]S(=O)=O);!$(n-[#7]);!$(n-[#8]):1]1[c;h0;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][c;!$(c(:a)(:a):a):3][c;R2;$(c(:a)(:a):a);$(c1ccccc1):4][c;R2;$(c(:a)(:a):a);$(c1ccccc1):5]1"
        ">>[#6_elec2:2][#7:1][c:5][c:4][#6_nuc2:3]",
        "c1ccc(-c2cc3ccccc3[nH]2)cc1",
    ),
    (
        "R17.16",
        "benzofuran / 5-endo-dig cycloisomerisation of a 2-alkynylphenol",
        "[o;+0:1]1[c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][c;!$(c(:a)(:a):a):3][c;R2;$(c(:a)(:a):a):4][c;R2;$(c(:a)(:a):a):5]1"
        ">>[#8_nuc:1][c:5][c:4][#6:3]=[#6_elec:2]",
        "CCCCc1oc2ccccc2c1C(=O)c1ccccc1",
    ),
    (
        "R17.17",
        "benzothiophene / 5-endo-dig cycloisomerisation of a 2-alkynylthiophenol",
        "[s;+0:1]1[c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][c;!$(c(:a)(:a):a):3][c;R2;$(c(:a)(:a):a):4][c;R2;$(c(:a)(:a):a):5]1"
        ">>[#16_nuc:1][c:5][c:4][#6:3]=[#6_elec:2]",
        "CCc1sc2ccc(O)cc2c1-c1ccc(O)cc1",
    ),
    (
        "R17.18",
        "carbazole / Cadogan (2-nitrobiphenyl + P(OEt)3) or Buchwald C-N of a 2-aminobiphenyl",
        "[n;+0;!$(n-[#7]);!$(n-[#8]):1]1[c;R2;$(c(:a)(:a):a):2][c;R2;$(c(:a)(:a):a):3][c;R2;$(c(:a)(:a):a):4][c;R2;$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][c:5][c:4][c:3][c_elec:2]",
        "COc1ccccc1OCCNCC(O)COc1cccc2c1c1ccccc1[nH]2",
    ),
    (
        "R17.19",
        "oxindole (indolin-2-one) / intramolecular lactamisation of a 2-aminophenylacetic acid",
        "[N;+0;!$([N]!@[#6]=[#8]);!$([N]!@S(=O)=O):1]1[C;+0:2](=[O:6])[C;+0:3][c;R2:4][c;R2:5]1"
        ">>[#7_nuc:1][c:5][c:4][#6:3][#6_elec:2]=[O:6]",
        "O=C1Nc2ccc(F)cc2C1=Cc1ccccc1",
    ),
    (
        "R17.20",
        "oxindole / Stolle intramolecular Friedel-Crafts of a 2-chloroacetanilide",
        "[N;+0;!$([N]!@[#6]=[#8]);!$([N]!@S(=O)=O):1]1[C;+0:2](=[O:6])[C;+0:3][c;R2:4][c;R2:5]1"
        ">>[#6_elec:3][#6:2](=[O:6])[#7:1][c:5][c_nuc:4]",
        "CC1C(=O)Nc2ccccc21",
    ),
    (
        "R17.30",
        "pyrazole, N1-substituted / nitrile imine (hydrazonoyl halide) + alkyne or alkene [3+2]",
        "[n;D3;+0;!$(n-[#7]);!$(n-[#8]);!$([n][#6]=[#8]);!$([n]S(=O)=O):1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][c;R1;!$(c(:a)(:a):a):4][c;R1;!$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][#7:2]=[#6_elec:3].[#6_nuc:4]=[#6_elec:5]",
        "Cn1nc(-c2ccccc2)cc1C",
    ),
    (
        "R17.31",
        "isoxazole / nitrile oxide + alkyne (or alkene) [3+2]",
        "[o;+0:1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][c;R1;!$(c(:a)(:a):a):4][c;R1;!$(c(:a)(:a):a):5]1"
        ">>[#8_nuc:1][#7:2]=[#6_elec:3].[#6_nuc:4]=[#6_elec:5]",
        "Cc1cc(-c2ccccc2)on1",
    ),
    (
        "R17.32a",
        "1,2,4-triazole, 4H / N4-substituted / Pellizzari (acylhydrazide + amidine or amide)",
        "[n;+0;D2:1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][n;D3;+0;!$(n-[#8]);!$([n][#6]=[#8]);!$([n]S(=O)=O):4][c;!$(c(:a)(:a):a):5]1"
        ">>[#7_nuc2:1][#7:2]=[#6_elec:3].[#7_nuc:4][#6_elec2:5]",
        "Cc1nnc(-c2ccccc2)n1C",
    ),
    (
        "R17.32b",
        "1,2,4-triazole, 4H, N-unsubstituted / Pellizzari, [n;h1] twin of R17.32a",
        "[n;+0;D2:1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][n;h1;+0:4][c;!$(c(:a)(:a):a):5]1"
        ">>[#7_nuc2:1][#7:2]=[#6_elec:3].[#7_nuc:4][#6_elec2:5]",
        "Cc1nc(-c2ccccc2)[nH]n1",
    ),
    (
        "R17.33",
        "1,2,4-triazole, 1H / N1-substituted / nitrile imine + nitrile [3+2]",
        "[n;D3;+0;!$(n-[#7]);!$(n-[#8]);!$([n][#6]=[#8]);!$([n]S(=O)=O):1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][n;+0;D2:4][c;!$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][#7:2]=[#6_elec:3].[#7_nuc:4]=[#6_elec:5]",
        "Cn1c(C)nc(-c2ccccc2)n1",
    ),
    (
        "R17.34",
        "1,2,4-oxadiazole / nitrile oxide + nitrile [3+2]",
        "[o;+0:1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][n;+0;D2:4][c;!$(c(:a)(:a):a):5]1"
        ">>[#8_nuc:1][#7:2]=[#6_elec:3].[#7_nuc:4]=[#6_elec:5]",
        "Cc1noc(-c2ccccc2)n1",
    ),
    (
        "R17.35",
        "1,2,4-thiadiazole / nitrile sulfide + nitrile [3+2]",
        "[s;+0:1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][n;+0;D2:4][c;!$(c(:a)(:a):a):5]1"
        ">>[#16_nuc:1][#7:2]=[#6_elec:3].[#7_nuc:4]=[#6_elec:5]",
        "Cc1nsc(-c2ccccc2)n1",
    ),
    (
        "R17.36",
        "1,3,4-thiadiazole / nitrile imine + C=S dipolarophile (dithioester, thioamide) [3+2] with beta-elimination",
        "[s;+0:1]1[c;!$(c(:a)(:a):a):2][n;+0;D2:3][n;+0;D2:4][c;!$(c(:a)(:a):a):5]1"
        ">>[#6_elec:2]=[#7:3][#7_nuc2:4].[#16_nuc:1][#6_elec2:5]",
        "Cc1nnc(-c2ccccc2)s1",
    ),
    (
        "R17.40",
        "Hantzsch thiazole, ring-opened S-vinyl thioimidate form (N3-C4)",
        "[s:1]1[c;R1;+0;!$([c]!@[#7,#8,#16]):2][n;D2;+0:3][c;R1;+0:4][c;R1;+0:5]1"
        ">>[#7_nuc:3]=[#6:2][#16:1][#6:5]=[#6_elec:4]",
        "Cc1csc(C)n1",
    ),
    (
        "R17.41",
        "van Leusen oxazole (TosMIC + aldehyde)",
        "[o:1]1[c;R1;+0;h1:2][n;D2;+0:3][c;R1;+0:4][c;R1;+0:5]1"
        ">>[#6_elec:2]=[#7:3][#6_nuc2:4].[#6_elec2:5][#8_nuc:1]",
        "Cc1cnco1",
    ),
    (
        "R17.42",
        "van Leusen imidazole (TosMIC + aldimine)",
        "[n;D3;+0;R1;!$([n][#6]=[#8]);!$([n]S(=O)=O);!$(n-[#7]);!$(n-[#8]):1]1[c;R1;+0;h1:2][n;D2;+0:3][c;R1;+0:4][c;R1;+0:5]1"
        ">>[#6_elec:2]=[#7:3][#6_nuc2:4].[#6_elec2:5][#7_nuc:1]",
        "Cn1cncc1-c1ccccc1",
    ),
    (
        "R17.43",
        "van Leusen thiazole (TosMIC + isothiocyanate or thiocarbonyl)",
        "[s:1]1[c;R1;+0;h1:2][n;D2;+0:3][c;R1;+0:4][c;R1;+0:5]1"
        ">>[#6_elec:2]=[#7:3][#6_nuc2:4].[#6_elec2:5][#16_nuc:1]",
        "c1ccccc1-c1cncs1",
    ),
    (
        "R17.44",
        "Cook-Heilbron 5-aminothiazole (alpha-aminonitrile + CS2 / isothiocyanate / dithioester)",
        "[s:1]1[c;R1;+0:2][n;D2;+0:3][c;R1;+0:4][c;R1;+0;$([c]!@[#7;+0;X3]):5]1"
        ">>[#16_nuc:1][#6_elec2:2].[#7_nuc2:3][#6:4]=[#6_elec:5]",
        "Nc1cnc(-c2ccccc2)s1",
    ),
    (
        "R17.45",
        "oxazole C2-O1 cyclodehydration (acyloin + nitrile / imidoyl electrophile) — NOT Robinson-Gabriel",
        "[o:1]1[c;R1;+0;!$([c]!@[#7,#8,#16]):2][n;D2;+0:3][c;R1;+0:4][c;R1;+0:5]1"
        ">>[#6_elec:2]=[#7:3][#6:4]=[#6:5][#8_nuc:1]",
        "Cc1oc(C)nc1C",
    ),
    (
        "R17.46",
        "oxazole, ring-opened O-vinyl imidate form (N3-C4)",
        "[o:1]1[c;R1;+0;!$([c]!@[#7,#8,#16]):2][n;D2;+0:3][c;R1;+0:4][c;R1;+0:5]1"
        ">>[#7_nuc:3]=[#6:2][#8:1][#6:5]=[#6_elec:4]",
        "Cc1coc(C)n1",
    ),
    (
        "R17.47",
        "2-oxazoline (4,5-dihydro-1,3-oxazole) from a 1,2-amino alcohol, O1-C2",
        "[O;+0;R1:1]1[C;+0;R1:2]=[N;+0;R1:3][C;X4;+0;R1:4][C;X4;+0;R1:5]1"
        ">>[#8_nuc:1][#6:5][#6:4][#7:3]=[#6_elec:2]",
        "c1ccccc1C1COC(C)=N1",
    ),
    (
        "R17.48",
        "2-thiazoline (4,5-dihydro-1,3-thiazole) from a 1,2-amino thiol, S1-C2",
        "[S;+0;R1:1]1[C;+0;R1:2]=[N;+0;R1:3][C;X4;+0;R1:4][C;X4;+0;R1:5]1"
        ">>[#16_nuc:1][#6:5][#6:4][#7:3]=[#6_elec:2]",
        "c1ccccc1C1=NCCS1",
    ),
    (
        "R17.49",
        "2-imidazoline (4,5-dihydro-1H-imidazole) from a 1,2-diamine, N1-C2",
        "[N;+0;R1;!$([N]!@[#6]=[O,S]):1]1[C;+0;R1:2]=[N;+0;R1:3][C;X4;+0;R1:4][C;X4;+0;R1:5]1"
        ">>[#7_nuc:1][#6:5][#6:4][#7:3]=[#6_elec:2]",
        "Clc1cccc(Cl)c1NC1=NCCN1",
    ),
    (
        "R17.50",
        "2-imidazoline from an amidine + 1,2-dielectrophile, N1-C5 + N3-C4",
        "[N;+0;R1;h1;!$([N]!@[#6]=[O,S]):1]1[C;+0;R1:2]=[N;+0;R1:3][C;X4;+0;R1:4][C;X4;+0;R1:5]1"
        ">>[#7_nuc:1][#6:2]=[#7_nuc:3].[#6_elec:4][#6_elec:5]",
        "CC1=NCCN1",
    ),
    (
        "R17.55a",
        "pyrazine / self-condensation of alpha-aminoketones (Staedel-Rugheimer, Gutknecht, Gastaldi), orientation A",
        "[n;D2;+0:1]1[c;x2;!$(c[#7,#8,#16]):2][c;x2:3][n;D2;+0:4][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):5][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):6]1"
        ">>[#7_nuc:1]=[#6:2][#6_elec2:3].[#7_nuc2:4][#6:5]=[#6_elec:6]",
        "NC(=O)c1cnccn1",
    ),
    (
        "R17.55b",
        "pyrazine / alpha-aminoketone self-condensation, orientation B (mirror twin of R16.1a0a)",
        "[n;D2;+0:1]1[c;x2;!$(c[#8;h1]);!$(c[#16;h1]):2][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):3][n;D2;+0:4][c;x2:5][c;x2;!$(c[#7,#8,#16]):6]1"
        ">>[#7_nuc:1]=[#6:6][#6_elec2:5].[#7_nuc2:4][#6:3]=[#6_elec:2]",
        "NC(=O)c1cnccn1",
    ),
    (
        "R17.56",
        "quinoxaline / o-phenylenediamine + 1,2-dicarbonyl (and pteridine by the Isay route)",
        "[n;D2;+0:1]1[c;x2:2][c;x2:3][n;D2;+0:4][c;R2;!$(c[A]):5][c;R2;!$(c[A]):6]1"
        ">>[#7_nuc2:1][c:6][c:5][#7_nuc2:4].[#6_elec2:2][#6_elec2:3]",
        "c1ccc(cc1)-c1nc2ccccc2nc1-c1ccccc1",
    ),
    (
        "R17.57a",
        "quinazoline / quinazolin-4(3H)-one, N3-H / Niementowski (anthranilic acid + amide)",
        "[n;D2;+0:1]1[c;x2:2][n;D2;+0:3][c;x2:4][c;R2;!$(c[A]):5][c;R2;!$(c[A]):6]1"
        ">>[#7_nuc2:1][c:6][c:5][#6_elec2:4].[#6_elec2:2][#7_nuc2:3]",
        "Cc1nc2ccccc2c(=O)[nH]1",
    ),
    (
        "R17.57b",
        "quinazolin-4(3H)-one, N3-substituted / Niementowski via acylanthranil",
        "[#7;A;+0:1]1=[#6:2][#7;A;D3;+0:3][#6:4](=[#8:7])[c;R2:5][c;R2:6]1"
        ">>[#7_nuc2:1][c:6][c:5][#6_elec:4]=[#8:7].[#6_elec2:2][#7_nuc:3]",
        "Cc1nc2ccccc2c(=O)n1-c1ccccc1C",
    ),
    (
        "R17.58",
        "quinoline / Combes (aniline + 1,3-diketone); also Skraup, Doebner-von Miller and Knorr",
        "[n;D2;+0:1]1[c;x2;!$(c!@[OH]):2][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):3][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):4][c;R2;!$(c[A]):5][c;R2;!$(c[A]):6]1"
        ">>[#7_nuc2:1][c:6][c_nuc:5].[#6_elec2:2][#6:3]=[#6_elec:4]",
        "Cc1cc(C)c2ccccc2n1",
    ),
    (
        "R17.59",
        "isoquinoline / Bischler-Napieralski, aromatic product (one-bond cut, C1-C8a)",
        "[c;x2;!$(c[#8;h1]);!$(c[#16;h1]):1]1[n;D2;+0:2][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):3][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):4][c;R2;!$(c[A]):5][c;R2;!$(c[A]):6]1"
        ">>[#6_elec:1]=[#7:2][#6:3]=[#6:4][c:5][c_nuc:6]",
        "COc1cc2ccnc(C)c2cc1OC",
    ),
    (
        "R17.60",
        "3,4-dihydroisoquinoline / Bischler-Napieralski, the literal one-step product",
        "[#6;A;!$([#6](=[#7])[#7]):1]1=[#7;A;+0:2][#6;A;X4:3][#6;A;X4:4][c;R2:5][c;R2:6]1"
        ">>[#6_elec:1]=[#7:2][#6:3][#6:4][c:5][c_nuc:6]",
        "COc1cc2CCN=C(C)c2cc1OC",
    ),
    (
        "R17.61",
        "isoquinoline / Pomeranz-Fritsch (aryl aldehyde + aminoacetaldehyde acetal)",
        "[c;x2;!$(c[#8;h1]);!$(c[#16;h1]):1]1[n;D2;+0:2][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):3][c;x2;!$(c[#8;h1]);!$(c[#16;h1]):4][c;R2;!$(c[A]):5][c;R2;!$(c[A]):6]1"
        ">>[#6_elec:4]=[#6:3][#7:2]=[#6:1][c:6][c_nuc:5]",
        "COc1cc2ccncc2cc1OC",
    ),
    (
        "R17.62",
        "1,2,3,4-tetrahydroisoquinoline / one-bond N-acyliminium cyclisation (Pictet-Spengler surrogate)",
        "[#6;A;X4;!$([#6]([#7,#8,#16])[#7]):1]1[#7;A;X3;+0:2][#6;A;X4:3][#6;A;X4:4][c;R2:5][c;R2:6]1"
        ">>[#6_elec:1][#7:2][#6:3][#6:4][c:5][c_nuc:6]",
        "COc1cc2CCNC(C)c2cc1OC",
    ),
    (
        "R17.70",
        "benzimidazole / Phillips (o-phenylenediamine + aldehyde or carboxylic acid), one-bond N1-C2 cut",
        "[n;+0;!$([n][#6]=[#8]);!$([n]S(=O)=O);!$(n-[#7]);!$(n-[#8]):1]1[c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][n;+0;h0;D2:3][c;R2;$(c(:a)(:a):a):4][c;R2;$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][c:5][c:4][#7:3]=[#6_elec:2]",
        "c1ccc(-c2nc3ccccc3[nH]2)cc1",
    ),
    (
        "R17.71",
        "benzimidazole / amidine + 1,2-dihaloarene (benzo analogue of the shipped R16.4b)",
        "[n;+0;!$([n][#6]=[#8]);!$([n]S(=O)=O);!$(n-[#7]);!$(n-[#8]):1]1[c;!$(c(:a)(:a):a):2][n;+0;h0;D2:3][c;R2;$(c(:a)(:a):a):4][c;R2;$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][#6:2]=[#7_nuc:3].[c_elec:4][c_elec:5]",
        "c1ccc(-c2nc3ccccc3[nH]2)cc1",
    ),
    (
        "R17.72",
        "benzoxazole / o-aminophenol + aldehyde or carboxylic acid, one-bond O1-C2 cut",
        "[o;+0:1]1[c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][n;+0;h0;D2:3][c;R2;$(c(:a)(:a):a):4][c;R2;$(c(:a)(:a):a):5]1"
        ">>[#8_nuc:1][c:5][c:4][#7:3]=[#6_elec:2]",
        "OC(=O)c1ccc2nc(-c3cc(Cl)cc(Cl)c3)oc2c1",
    ),
    (
        "R17.73",
        "benzothiazole / o-aminothiophenol + aldehyde or carboxylic acid, one-bond S1-C2 cut",
        "[s;+0:1]1[c;!$(c(:a)(:a):a);!$(c!@[#7,#8,#9,#16,#17,#35,#53]):2][n;+0;h0;D2:3][c;R2;$(c(:a)(:a):a):4][c;R2;$(c(:a)(:a):a):5]1"
        ">>[#16_nuc:1][c:5][c:4][#7:3]=[#6_elec:2]",
        "c1ccc(-c2nc3ccccc3s2)cc1",
    ),
    (
        "R17.74",
        "1H-indazole / o-fluoroaryl aldehyde or ketone + hydrazine",
        "[n;+0;!$([n][#6]=[#8]);!$([n]S(=O)=O);!$(n-[#7]);!$(n-[#8]):1]1[n;+0;D2:2][c;!$(c(:a)(:a):a):3][c;R2;$(c(:a)(:a):a):4][c;R2;$(c(:a)(:a):a):5]1"
        ">>[#7_nuc:1][#7_nuc2:2].[#6_elec2:3][c:4][c_elec:5]",
        "Nc1ccc2[nH]nc(-c3ccccc3)c2c1",
    ),
    (
        "R17.80",
        "piperidine / intramolecular C-N ring closure (reductive amination, N-alkylation, aza-Michael, Mitsunobu, hydroamination)",
        "[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);$([N]1[C;z1][C;z1][C;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]"
        ">>[#7_nuc:1].[#6_elec:2]",
        "c1ccccc1C1CCCNC1",
    ),
    (
        "R17.81",
        "pyrrolidine / intramolecular C-N ring closure",
        "[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);$([N]1[C;z1][C;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]"
        ">>[#7_nuc:1].[#6_elec:2]",
        "CN1CCCC1c1cccnc1",
    ),
    (
        "R17.82a",
        "morpholine / C-O ring closure (diethanolamine cyclodehydration, haloether or epoxide closure)",
        "[O;z1;+0;$([O]1[C;z1][C;z1][N;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]"
        ">>[#8_nuc:1].[#6_elec:2]",
        "c1ccccc1C1CNCCO1",
    ),
    (
        "R17.82b",
        "morpholine / C-N ring closure",
        "[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);$([N]1[C;z1][C;z1][O;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]"
        ">>[#7_nuc:1].[#6_elec:2]",
        "c1ccccc1C1CNCCO1",
    ),
    (
        "R17.83",
        "morpholine / 1,2-amino alcohol + 1,2-bis-electrophile (two bonds, on two DIFFERENT heteroatoms)",
        "[O;z1;+0;$([O]1[C;z1][C;z1][N;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]-@[C;z1;!D4:3]-@[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);R1:4]"
        ">>[#8_nuc:1].[#6_elec:2][#6_elec:3].[#7_nuc:4]",
        "c1ccccc1C1CNCCO1",
    ),
    (
        "R17.84a",
        "piperazine / 1,2-diamine + 1,2-bis-electrophile (two bonds, on two DIFFERENT nitrogens)",
        "[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);$([N]1[C;z1][C;z1][N;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]-@[C;z1;!D4:3]-@[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);R1:4]"
        ">>[#7_nuc:1].[#6_elec:2][#6_elec:3].[#7_nuc:4]",
        "COc1ccccc1N1CCNCC1",
    ),
    (
        "R17.84b",
        "piperazine / intramolecular C-N ring closure (one bond)",
        "[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);$([N]1[C;z1][C;z1][N;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]"
        ">>[#7_nuc:1].[#6_elec:2]",
        "COc1ccccc1N1CCNCC1",
    ),
    (
        "R17.85a",
        "thiomorpholine / 1,2-amino thiol + 1,2-bis-electrophile (two bonds)",
        "[S;z1;+0;$([S]1[C;z1][C;z1][N;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]-@[C;z1;!D4:3]-@[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);R1:4]"
        ">>[#16_nuc:1].[#6_elec:2][#6_elec:3].[#7_nuc:4]",
        "c1ccccc1C1CNCCS1",
    ),
    (
        "R17.85b",
        "thiomorpholine / C-S ring closure (one bond)",
        "[S;z1;+0;$([S]1[C;z1][C;z1][N;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]"
        ">>[#16_nuc:1].[#6_elec:2]",
        "c1ccccc1C1CNCCS1",
    ),
    (
        "R17.86a",
        "1,4-diazepane (homopiperazine) / 1,2-diamine + 1,3-bis-electrophile (two bonds)",
        "[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);$([N]1[C;z1][C;z1][N;z1][C;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:5]-@[C;z1:6]-@[C;z1;!D4:7]-@[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);R1:4]"
        ">>[#7_nuc:1].[#6_elec:5][#6:6][#6_elec:7].[#7_nuc:4]",
        "C1CN(CCNC1)S(=O)(=O)c1cccc2cnccc12",
    ),
    (
        "R17.86b",
        "1,4-diazepane / intramolecular C-N ring closure (one bond)",
        "[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);$([N]1[C;z1][C;z1][N;z1][C;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]"
        ">>[#7_nuc:1].[#6_elec:2]",
        "C1CN(CCNC1)S(=O)(=O)c1cccc2cnccc12",
    ),
    (
        "R17.87",
        "delta-lactam (piperidin-2-one) / amino acid or amino ester lactamisation",
        "[N;z1;+0;!$([#7][#7]);$([N]1[C;z2](=[O;z2])[C;z1][C;z1][C;z1][C;z1]1);R1:1]-@[C;z2;$([C]=[O]):2]"
        ">>[#7_nuc:1].[#6_elec:2]",
        "O=C1CCCCN1Cc1ccccc1",
    ),
    (
        "R17.88",
        "gamma-lactam (pyrrolidin-2-one) / amino acid or amino ester lactamisation",
        "[N;z1;+0;!$([#7][#7]);$([N]1[C;z2](=[O;z2])[C;z1][C;z1][C;z1]1);R1:1]-@[C;z2;$([C]=[O]):2]"
        ">>[#7_nuc:1].[#6_elec:2]",
        "COc1ccc(cc1OC1CCCC1)C1CNC(=O)C1",
    ),
    (
        "R17.89",
        "1,2,3,6-tetrahydropyridine / ring-closing metathesis of an N-tethered diene",
        "[C;z2;$([C]1=[C][C;z1][N;z1;+0][C;z1][C;z1]1):1]=@[C;z2:2]"
        ">>[#6_neut2:1].[#6_neut2:2]",
        "c1ccccc1C1=CCNCC1",
    ),
    (
        "R17.90",
        "morpholin-3-one / 1,2-amino alcohol + haloacetate (the reducible morpholine precursor)",
        "[N;z1;+0;!$([#7][#7]);$([N]1[C;z2](=[O;z2])[C;z1][O;z1][C;z1][C;z1]1);R1:1]-@[C;z2;$([C]=[O]):2]"
        ">>[#7_nuc:1].[#6_elec:2]",
        "O=C1COC(c2ccccc2)CN1",
    ),
    (
        "R17.91",
        "piperazin-2-one / 1,2-diamine + haloacetate (the reducible piperazine precursor)",
        "[N;z1;+0;!$([#7][#7]);$([N]1[C;z2](=[O;z2])[C;z1][N;z1][C;z1][C;z1]1);R1:1]-@[C;z2;$([C]=[O]):2]"
        ">>[#7_nuc:1].[#6_elec:2]",
        "O=C1CN(Cc2ccccc2)CCN1",
    ),
    (
        "R17.92",
        "piperidine / aza-annulation: amine-bearing C-nucleophile + 1,3-bis-electrophile (two bonds, N-C and C-C)",
        "[N;z1;+0;!$([#7][#6]=[#8]);!$([#7][#7]);$([N]1[C;z1][C;z1][C;z1][C;z1][C;z1]1);R1:1]-@[C;z1;!D4:2]-@[C;z1:3]-@[C;z1;!D4:4]-@[C;z1;$([#6][c]),$([#6][#6]=[#8]),$([#6][#6]#[#7]),$([#6][#7](=[#8])[#8]),$([#6][#16](=[#8])=[#8]):5]"
        ">>[#7_nuc:1].[#6_elec:2][#6:3][#6_elec:4].[#6_nuc:5]",
        "c1ccccc1C1CCCNC1",
    ),
    (
        "R17.93",
        "GENERIC saturated heterocycle / one-bond heteroatom-carbon ring closure (the declared residual)",
        "[N,O,S;z1;R;+0;!$([N,O,S][C;z1][#7,#8,#16]);!$([#7][#6]=[#8]);!$([#7][#7]);!$([#8][#6]=[#6,#7,#8,#15,#16,#34]);!$([N;z1]1[C;z1][C;z1][C;z1][C;z1]1);!$([N;z1]1[C;z1][C;z1][C,N,O,S;z1][C;z1][C;z1]1);!$([N;z1]1[C;z1][C;z1][N;z1][C;z1][C;z1][C;z1]1);!$([O,S;z1]1[C;z1][C;z1][N;z1][C;z1][C;z1]1):1]-@[C;z1;R;!D4;!$([C]([#7,#8,#16])[#7,#8,#16]);!r3;!r4:2]"
        ">>[*_nuc:1].[#6_elec:2]",
        "C1CCCNCC1",
    ),
]

# rows legal ONLY as the closing bond of a ring, so they cannot leak into acyclic `join`. Aliphatic
# C:nuc + N:elec is the C5-N1 bond of every 1,2,3-triazole; the shipped `pairs` restricts N:elec to
# AROMATIC C:nuc because its one source (R3.3, umpolung cross-coupling) is aryl chemistry, and
# putting these in `pairs` would claim an alkyl nucleophile aminates.
RING_PAIRS = [
    ["C", False, "nuc", "N", False, "elec"],
    ["N", False, "nuc", "N", False, "elec"],
]

# the leaving group a planner caps an attachment point with to recover an orderable reagent.
# Upstream has none: enumeration is synthon -> synthon -> molecule and the reagent is recovered by
# lookup in the synthon->BB file.
# ponytail: one representative LG per (element, aromatic, token); add a class-aware variant if a
# planner needs the exact stocked form.
LEAVING_GROUPS = {
    "C:elec": "Cl",
    "c:elec": "Br",
    "S:elec": "Cl",
    "N:elec": "OC(=O)c1ccccc1",
    "C:nuc": "[Mg]",
    "c:nuc": "[Mg]",
    "N:nuc": "H",
    "n:nuc": "H",
    "O:nuc": "H",
    "S:nuc": "H",
    "C:elecB": "B(O)O",
    "c:elecB": "B(O)O",
    "C:elec2": "O",
    "C:nuc2": "H",
    "N:nuc2": "H",
    "C:neut2": "C",
    "C:elec*": "H",
    "c:elec*": "H",
    "C:nuc*": "[B-](F)(F)F",
    "c:nuc*": "[B-](F)(F)F",
}

# dispatch selectors, read verbatim from __synthonsAssignement (SyntOn_BBs.py)
PG_BIFUNCTIONAL = [
    "Bifunctional_Acid_Ester",
    "Bifunctional_Acid_Nitro",
    "Bifunctional_Aldehyde_Ester",
    "Bifunctional_Amine_Ester",
    "Bifunctional_Ester_Isocyanates",
    "Bifunctional_Ester_SO2X",
    "Bifunctional_Aldehyde_Nitro",
    "Bifunctional_NbocAmino_Acid",
    "Bifunctional_NcbzAmino_Acid",
    "Bifunctional_Isothiocyanates_Acid",
    "Bifunctional_NfmocAmino_Acid",
    "Bifunctional_Aldehyde_Nboc",
    "Bifunctional_NTFAcAmino_Acid",
    "Bifunctional_Boronics_Ncbz",
    "Bifunctional_Boronics_Nfmoc",
    "Bifunctional_NbnDi_Amines",
    "Bifunctional_NbocDi_Amines",
    "Bifunctional_NcbzDi_Amines",
    "Bifunctional_NfmocDi_Amines",
    "Bifunctional_NTFAcDi_Amines",
    "Bifunctional_Di_Amines_NotherCarbamates",
    "Trifunctional_Acid_Aldehyde_Nitro",
    "Trifunctional_Acid_ArylHalide_Ester",
    "Trifunctional_Acid_ArylHalide_Nitro",
    "Trifunctional_Amines_ArylHalide_Nitro",
    "Trifunctional_NbocAmino_Acid_AlkyneCH",
    "Trifunctional_NbocAmino_Acid_ArylHalide",
    "Trifunctional_NfmocAmino_Acid_AlkyneCH",
    "Trifunctional_NfmocAmino_Acid_ArylHalide",
]
TWO_PG_TRIFUNCTIONAL = [
    "Trifunctional_Acid_Ester_Nitro",
    "Trifunctional_NbocAmino_Acid_Ester",
    "Trifunctional_NbocAmino_Acid_Nitro",
    "Trifunctional_Amines_Nboc_Ester",
    "Trifunctional_Nboc_NCbz_Amino_Acid",
    "Trifunctional_Nboc_Nfmoc_Amino_Acid",
    "Trifunctional_NfmocAmino_Acid_Ester",
    "Trifunctional_NfmocAmino_Acid_Nitro",
    "Trifunctional_Di_Esters_Amino",
]
FIRST_AS_PREP = [
    "Bifunctional_Acid_Aldehyde",
    "Bifunctional_Aldehyde_ArylHalide",
    "Bifunctional_Aldehyde_SO2X",
    "Bifunctional_Boronics_Acid",
    "Bifunctional_Boronics_Aldehyde",
    "Bifunctional_Hydroxy_Aldehyde",
    "Trifunctional_Acid_Aldehyde_ArylHalide",
    "Trifunctional_Acid_Aldehyde_Acetylenes",
    "Trifunctional_Acid_Aldehyde_Nitro",
    "Trifunctional_Amines_ArylHalide_Nitro",
    "Trifunctional_NbocAmino_Acid_AlkyneCH",
    "Trifunctional_NfmocAmino_Acid_AlkyneCH",
    "Trifunctional_Di_Esters_Amino",
]
POLYMER_REAGENTS = [
    "Reagents_PoliOxiranes",
    "Esters_PoliEsters",
    "Reagents_PoliIsocyanates",
    "SulfonylHalides_Poli_Sulfonylhalides",
]
ADDITIONAL_BIFUNCTIONAL = [
    "Aminoacids_N-AliphaticAmino_Acid",
    "Aminoacids_N-AromaticAmino_Acid",
    "Reagents_DiAmines",
]
# nitro -> amine is not really a protecting group, so the protected form is kept regardless of
# keepPG. Upstream compares this SMIRKS as an exact string twice; any whitespace change in the XML
# would have silently disabled it.
NITRO_REDUCTION = (
    "[N;+0,+1;$([N+](=O)([#6])[O-]),$(N(=O)([#6])=O):1](=[O:2])"
    "=,-[O;+0,-1:3]>>[NH2,+0:1]"
)

CLASS_KEYS = ("ShouldContainAtLeastOne", "ShouldAlsoContain", "shouldNotContain")

_BRANCH_STUB = re.compile(r"\([-=#:~]?(?:\[\*\]|\*)\)")
# the lookbehind matters: `_elec*` and `_nuc*` end in a star that is NOT a stub
_STUB = re.compile(r"[-=#:~]?(?:\[\*\]|(?<![A-Za-z])\*)")
_VANADIUM = re.compile(r"\[V\]")
# `*[V]c->c:10`: sigil, optional vanadium disambiguator, optional bond, element spelling, code
_LABEL_ALT = re.compile(
    r"^\*(?P<v>\[V\])?(?P<bond>=)?(?P<lhs>.+?)->(?P<iso>[0-9]*)(?P<el>[A-Za-z]+):(?P<code>[0-9]+)$"
)


class ConversionError(RuntimeError):
    """The upstream data said something this converter refuses to guess about."""


def _top_level_brackets(text: str) -> list[tuple[int, int]]:
    spans, depth, start = [], 0, None
    for i, c in enumerate(text):
        if c == "[":
            if not depth:
                start = i
            depth += 1
        elif c == "]":
            depth -= 1
            if not depth:
                spans.append((start, i))
    return spans


def _insert_token(template: str, atom_map: int, token: str) -> str:
    """Write `_token` into the bracket carrying `:atom_map`, just before the map."""
    suffix = f":{atom_map}"
    for start, end in _top_level_brackets(template):
        if template[start + 1 : end].endswith(suffix):
            cut = end - len(suffix)
            return f"{template[:cut]}_{token}{template[cut:]}"
    raise ConversionError(f"no bracket with map {atom_map} in {template!r}")


def _strip_stubs(template: str) -> str:
    """Drop the `[*]`/`*` attachment stubs and the `[V]` product-slot disambiguator (D15, W2/W4)."""
    return _STUB.sub("", _VANADIUM.sub("", _BRANCH_STUB.sub("", template)))


def _parse_label_options(labels: str) -> list[dict]:
    """A whole Labels string -> the distinct (element, aromatic, code, via_v, bond) it can assign.

    Isotope spellings collapse: they exist only because the reference matched the label as TEXT
    against a SMILES, and upstream's own enumerator KeyErrors on their output.
    """
    seen = set()
    for group in labels.split(";"):
        for alternative in group.split(","):
            match = _LABEL_ALT.match(alternative.strip())
            if match is None:
                raise ConversionError(f"unparsable Labels alternative {alternative!r}")
            seen.add(
                (
                    match["el"].upper(),
                    match["el"].islower(),
                    int(match["code"]),
                    bool(match["v"]),
                    2 if match["bond"] else 1,
                )
            )
    return [
        {"element": e, "aromatic": a, "code": c, "via_v": v, "bond": b}
        for e, a, c, v, b in sorted(seen)
    ]


def _fits(option: dict, target: dict) -> bool:
    """Aromaticity is deliberately NOT a filter here: the token does not encode it, the atom does,
    and a product template routinely spells an aromatic nitrogen `[N:2]`. It is a tie-breaker."""
    return (
        option["element"] == target["element"].upper()
        and option["via_v"] == target["via_v"]
        and option["bond"] == target["bond"]
    )


def _codes_for(options: list[dict], target: dict) -> list[int]:
    fits = [o for o in options if _fits(o, target)]
    codes = sorted({o["code"] for o in fits})
    if len(codes) > 1 and target["aromatic"] is not None:
        narrowed = sorted(
            {o["code"] for o in fits if o["aromatic"] == target["aromatic"]}
        )
        if narrowed:
            return narrowed
    return codes


def _component_targets(component: str) -> list[dict]:
    """Where each attachment stub of a product template sits: the mapped atom it labels."""
    query = smarts(to_chython(component))
    # an OR of mixed primitives translates to a mapped AnyElement, so "AnyElement" alone does not
    # identify a stub — an unmapped one does.
    mapped = {int(n) for n in re.findall(r":([1-9][0-9]*)\]", component)}
    targets = []
    for n, atom in query.atoms():
        if not isinstance(atom, AnyElement) or n in mapped:
            continue
        neighbours = list(query._bonds[n])
        if len(neighbours) != 1:
            raise ConversionError(
                f"stub with {len(neighbours)} neighbours in {component!r}"
            )
        (attached,) = neighbours
        bond = int(query._bonds[n][attached].order[0])
        via_v = query.atom(attached).atomic_symbol == "V"
        if via_v:
            rest = [m for m in query._bonds[attached] if m != n]
            if len(rest) != 1:
                raise ConversionError(
                    f"[V] with {len(rest)} real neighbours in {component!r}"
                )
            attached = rest[0]
        targets.append(
            {
                "map": attached,
                "element": query.atom(attached).atomic_symbol,
                "via_v": via_v,
                "bond": bond,
                "aromatic": _aromaticity(query)[attached],
            }
        )
    return targets


def _build_rule(
    reactant: str, product: str, labels: str | None, where: str
) -> list[str]:
    """One upstream (SMARTS, Labels) pair -> the rule SMARTS variants, tokens written inline."""
    left = to_chython(reactant)
    components = product.split(".")
    if labels in (None, "No"):
        return [f"{left}>>{_strip_stubs(to_chython(product))}"]

    options = _parse_label_options(labels)
    reactant_aromatic = _aromaticity(smarts(left))
    targets = []
    for index, component in enumerate(components):
        for target in _component_targets(component):
            if target["aromatic"] is None:
                target["aromatic"] = reactant_aromatic.get(target["map"])
            targets.append((index, target))

    # a stub matched by several distinct codes is several PRODUCTIVE labellings, not an ambiguity:
    # aniline emits both [NH2_nuc] and [NH2_nuc2].
    codes = []
    for _, target in targets:
        fits = _codes_for(options, target)
        if not fits:
            raise ConversionError(
                f"{where}: no Labels alternative fits the stub on atom {target['map']}"
            )
        codes.append(fits)
    branching = [i for i, c in enumerate(codes) if len(c) > 1]
    if len(branching) > 1:
        raise ConversionError(
            f"{where}: {len(branching)} stubs are each multiply labelled"
        )

    variants = []
    for choice in codes[branching[0]] if branching else [None]:
        assignment = {}
        for i, (_, target) in enumerate(targets):
            assignment[target["map"]] = choice if i in branching else codes[i][0]
        right = []
        for index, component in enumerate(components):
            # strip BEFORE stamping: the radical tokens end in a star of their own
            text = _strip_stubs(to_chython(component))
            for j, target in targets:
                if j == index:
                    text = _insert_token(
                        text,
                        target["map"],
                        PAPER_CODE_TO_LABEL[assignment[target["map"]]],
                    )
            right.append(text)
        variants.append(f"{left}>>{'.'.join(right)}")
    return variants


def _aromaticity(query) -> dict[int, bool | None]:
    out = {}
    for n, atom in query.atoms():
        hybridization = getattr(atom, "hybridization", ())
        out[n] = (
            True
            if hybridization == (4,)
            else (False if hybridization and 4 not in hybridization else None)
        )
    return out


def _split_label_sets(labels: str) -> list[str]:
    """`|` separates whole alternative labellings. Upstream keeps only the last (F8); we ship both."""
    return labels.split("|")


def convert_classes(path: Path) -> list[dict]:
    """SMARTSLibNew.json -> the ordered 147-class list. Order is load-bearing; never sort."""
    library = json.loads(path.read_text())
    out = []
    for big, subclasses in library.items():
        for sub, record in subclasses.items():
            unknown = set(record) - set(CLASS_KEYS)
            if unknown:
                raise ConversionError(f"{big}_{sub}: unknown keys {unknown}")
            out.append(
                {
                    "name": f"{big}_{sub}",
                    "at_least_one": [
                        to_chython(p) for p in record.get(CLASS_KEYS[0], ())
                    ],
                    "also": [to_chython(p) for p in record.get(CLASS_KEYS[1], ())],
                    "not": [to_chython(p) for p in record.get(CLASS_KEYS[2], ())],
                }
            )
    return out


def _strategy_and_func(name: str) -> tuple[str, int]:
    if name in TWO_PG_TRIFUNCTIONAL or "Trifunctional" in name:
        func = 3
    elif "Bifunctional" in name or name in ADDITIONAL_BIFUNCTIONAL:
        func = 2
    else:
        func = 1
    # the reference's own precedence chain
    if name in POLYMER_REAGENTS:
        strategy = "polymer"
    elif name in PG_BIFUNCTIONAL or name in TWO_PG_TRIFUNCTIONAL:
        strategy = "protecting_group"
    elif name in FIRST_AS_PREP:
        strategy = "first_as_prep"
    else:
        strategy = "normal"
    return strategy, func


def convert_marks(path: Path) -> list[dict]:
    """BB_Marks.xml -> 147 rule programs, one record per classifier subclass."""
    root = ElementTree.parse(path).getroot()
    out = []
    for big in root:
        for sub in big:
            if not sub.get("SMARTS"):
                continue
            name = f"{big.tag}_{sub.tag}"
            steps_smarts = sub.get("SMARTS").split("|")
            steps_labels = sub.get("Labels").split("|")
            if len(steps_smarts) != len(steps_labels):
                raise ConversionError(
                    f"{name}: {len(steps_smarts)} steps vs {len(steps_labels)} labels"
                )
            strategy, func = _strategy_and_func(name)
            steps = []
            for i, (step, label) in enumerate(zip(steps_smarts, steps_labels)):
                reactant, product = step.split(">>")
                variants = _build_rule(reactant, product, label, f"{name}[{i}]")
                steps.append(
                    {
                        "variants": variants,
                        # `|No|` is the section boundary and nothing else - upstream splits on it
                        # unconditionally (SyntOn_BBs.py `LabelsLIST.split("|No|")`).
                        "is_pg_removal": label == "No",
                        # nitro -> amine is a reduction, not a deprotection, so the "protected"
                        # form survives it at both keepPG values.
                        "keeps_protected": step == NITRO_REDUCTION,
                    }
                )
            out.append(
                {
                    "name": name,
                    "strategy": strategy,
                    "func": func,
                    "first_as_prep": name in FIRST_AS_PREP,
                    "steps": steps,
                }
            )
    return out


def _rule_records(path: Path, macro: bool) -> list[dict]:
    root = ElementTree.parse(path).getroot()
    out = []
    for group in root.find("AvailableReactions"):
        for rule in group:
            if rule.get("SMARTS") == "None":  # the R14.1 / MR14.1 range sentinel
                continue
            reactant, product = rule.get("SMARTS").split(">>")
            label_sets = _split_label_sets(rule.get("Labels"))
            suffixes = (
                ("",) if len(label_sets) == 1 else tuple("abcdefg"[: len(label_sets)])
            )
            for suffix, labels in zip(suffixes, label_sets):
                variants = _build_rule(reactant, product, labels, f"{rule.tag}{suffix}")
                if len(variants) != 1:
                    raise ConversionError(
                        f"{rule.tag}: {len(variants)} labelling variants"
                    )
                out.append(
                    {
                        "id": f"{rule.tag}{suffix}",
                        "name": rule.get("name") or group.get("name"),
                        "macro": macro,
                        "ring": False,
                        "smarts": variants[0],
                        "single_product": macro,
                    }
                )
    return out


def _macro_reactants(path: Path) -> dict[str, str]:
    root = ElementTree.parse(path).getroot()
    out = {}
    for group in root.find("AvailableReactions"):
        for rule in group:
            if rule.get("SMARTS") != "None":  # MR14.1 is the range sentinel
                out[rule.tag] = rule.get("SMARTS")
    return out


def _macro_twins(rules: list[dict], path: Path) -> list[dict]:
    """The MR set: the upstream macrocyclic REACTANT with the R rule's labelled product.

    The reactant is authored — it says which bond is the ring bond and carries the `!r3..!r11`
    guard, which the fork now expresses as a real excluded-ring-sizes field. The product comes from
    the R twin, which is why the macro file's own defects (aliphatic C mapped onto aromatic c in
    MR10.1/10.2/12.2, the 60/70 swap in MR13.1) are not inherited.
    """
    upstream = _macro_reactants(path)
    out = []
    for rule in rules:
        base = rule["id"][:-1] if rule["id"][-1].isalpha() else rule["id"]
        raw = upstream.get(f"M{base}")
        if raw is None:
            raise ConversionError(f"no macrocyclic twin for {rule['id']}")
        left = to_chython(raw.split(">>")[0])
        right = rule["smarts"].split(">>", 1)[1]
        reactant_maps = set(smarts(left))
        product_maps = {n for n, _ in smarts(right).atoms()}
        if not product_maps <= reactant_maps:
            raise ConversionError(
                f"M{rule['id']}: product maps {product_maps - reactant_maps} "
                "are absent from the macrocyclic reactant"
            )
        out.append(
            {
                "id": f"M{rule['id']}",
                "name": f"{rule['name']} (macrocyclic)",
                "macro": True,
                "ring": False,
                "smarts": f"{left}>>{right}",
                "single_product": True,
            }
        )
    return out


def _partner_pairs() -> list[list]:
    combos = {k: list(v) for k, v in MARKS_COMBINATIONS.items()}
    for additions in (F7_ADDITIONS, F18_ADDITIONS):
        for key, partners in additions.items():
            combos.setdefault(key, []).extend(partners)
    for key, partners in list(combos.items()):
        for partner in partners:
            if key not in combos.setdefault(partner, []):
                combos[partner].append(key)
    pairs = set()
    for key, partners in combos.items():
        for partner in partners:
            pairs.add(tuple(sorted((_migrate_key(key), _migrate_key(partner)))))
    return sorted([list(a) + list(b) for a, b in pairs])


def _migrate_key(key: str) -> tuple[str, bool, str]:
    element, code = key.split(":")
    return element.upper(), element.islower(), PAPER_CODE_TO_LABEL[int(code)]


def _upstream_produced(path: Path) -> set[str]:
    """The `element:code` keys the disconnection rules actually emit, in upstream spelling.

    Deadness has to be decided BEFORE the migration: collapsing code 11 into `elec` would
    otherwise turn upstream's never-emitted `c:11` into a live `c:elec` entry and ban a
    perfectly ordinary aryl-electrophile / alkyl-nucleophile fragment.
    """
    root = ElementTree.parse(path).getroot()
    keys = set()
    for group in root.find("AvailableReactions"):
        for rule in group:
            if rule.get("SMARTS") == "None":
                continue
            for label_set in _split_label_sets(rule.get("Labels")):
                for option in _parse_label_options(label_set):
                    symbol = (
                        option["element"].lower()
                        if option["aromatic"]
                        else option["element"]
                    )
                    keys.add(f"{symbol}:{option['code']}")
    return keys


def _forbidden_marks(emitted: set[str]) -> list[list]:
    """The 12 live entries, keyed on tokens. Four are dead upstream and one collapses (F5)."""
    out = []
    for entry in FORBIDDEN_MARKS:
        if (
            len(entry) == 1
        ):  # F5: {N:11, N:11} collapses and bans every mono-N:elec synthon
            continue
        if any(k not in emitted for k in entry):
            continue
        out.append(sorted([list(_migrate_key(k)) for k in entry]))
    return sorted(out)


def build(config_dir: Path) -> dict[str, object]:
    classes = convert_classes(config_dir / "SMARTSLibNew.json")
    marks = convert_marks(config_dir / "BB_Marks.xml")
    disconnections = _rule_records(config_dir / "Setup.xml", macro=False)
    # the ring rules go last: they have no macrocyclic twin, and `_select` slices the ORDERED list
    rules = {
        "disconnections": disconnections
        + _macro_twins(disconnections, config_dir / "SetupForMacrocycles.xml")
        + [
            {
                "id": rule_id,
                "name": name,
                "macro": False,
                "ring": True,
                "smarts": smarts_text,
                "single_product": False,
            }
            for rule_id, name, smarts_text, _target in RING_RULES
        ],
        "pairs": _partner_pairs(),
        "ring_pairs": RING_PAIRS,
        "leaving_groups": LEAVING_GROUPS,
        "forbidden_marks": _forbidden_marks(
            _upstream_produced(config_dir / "Setup.xml")
        ),
    }
    return {"bb_classes": classes, "bb_marks": marks, "rules": rules}


def _ring_labels_survive() -> list[str]:
    """Every ring rule fires on its own target and keeps its labels where it wrote them.

    A proton that moves BETWEEN two labelled atoms of the same element carrying the same token
    leaves the fragment chemically identical, so the blanket `shifted_labels` verdict is narrowed by
    that exemption - without it every N-H amidine rule is a false positive.
    """
    problems = []
    # ponytail: one authored target per rule; widen to the curation over-firing panel if a guard
    # edit ever needs more than "it still fires and still spells what it wrote"
    for rule_id, _name, smarts_text, target in RING_RULES:
        molecule = safe_canonicalization(synthon_smiles(target))
        cuts = list(SynthonTransformer.from_smarts(smarts_text)(molecule))
        if not cuts:
            problems.append(f"{rule_id} does not fire on its own target {target}")
            continue
        for part in next(iter(cuts)).split():
            shifted = shifted_labels(part)
            if shifted and sorted(b for b, _ in shifted.values()) != sorted(
                a for _, a in shifted.values()
            ):
                problems.append(f"{rule_id} labels move on canonicalisation: {shifted}")
    return problems


def _base_id(rule_id: str) -> str:
    return rule_id[:-1] if rule_id[-1].isalpha() else rule_id


def _no_duplicate_disconnections() -> list[str]:
    """No two ring rule families may hand back the same synthons for the same target.

    The panel is every rule's own authored target, so a rule added later is tried against every
    rule already shipped on a retron each of them was written for - which is where a duplicate
    shows. Measured against the ten lane-A/lane-E duplicate pairs the curation found, this panel
    catches nine; the tenth pair has no shipped winner to collide with.

    `a`/`b` twins are exempt by base id. A mirror pair (`R16.6a`/`R16.6b`) exists to emit both
    directions of one disconnection and coincides on a symmetric target by construction.
    """
    problems = []
    # ponytail: 76 authored targets, ~0.2 s. The wider check is the drug-like over-firing sweep in
    # research/synthon/curation/overfire_harness/sweep_now.py - run that when a guard edit lands.
    compiled = [
        (rule_id, SynthonTransformer.from_smarts(smarts_text))
        for rule_id, _name, smarts_text, _target in RING_RULES
    ]
    for _rule_id, _name, _smarts_text, target in RING_RULES:
        molecule = safe_canonicalization(synthon_smiles(target))
        owners = defaultdict(set)
        for rule_id, rule in compiled:
            try:
                cuts = list(rule(molecule))
            except InvalidAromaticRing:
                continue  # `Fragmenter._cut` skips such a rule for such a target too
            for cut in cuts:
                synthons = tuple(
                    sorted(str(safe_canonicalization(p)) for p in cut.split())
                )
                owners[synthons].add(rule_id)
        for synthons, ids in owners.items():
            if len({_base_id(i) for i in ids}) > 1:
                problems.append(
                    f"{sorted(ids)} give the same synthons on {target}: {list(synthons)}"
                )
    return problems


def check(built: dict, config_dir: Path) -> list[str]:
    """Assert every property the plan pins. Returns the failures, empty when the build is good."""
    problems = []
    classes, marks, rules = built["bb_classes"], built["bb_marks"], built["rules"]

    if len(classes) != 147:
        problems.append(f"{len(classes)} classes, expected 147")
    patterns = sum(len(c[k]) for c in classes for k in ("at_least_one", "also", "not"))
    if patterns != 2401:
        problems.append(f"{patterns} classifier patterns, expected 2401")
    for record in classes:
        for key in ("at_least_one", "also", "not"):
            for pattern in record[key]:
                try:
                    smarts(pattern)
                except Exception as exc:
                    problems.append(f"{record['name']}/{key}: {pattern!r} {exc}")

    if len(marks) != 147:
        problems.append(f"{len(marks)} rule programs, expected 147")
    if {c["name"] for c in classes} != {m["name"] for m in marks}:
        problems.append("bb_classes and bb_marks names are not in 1:1 correspondence")
    steps = sum(len(m["steps"]) for m in marks)
    if steps != 389:
        problems.append(f"{steps} rule steps, expected 389")
    strategies = {"polymer": 0, "protecting_group": 0, "first_as_prep": 0, "normal": 0}
    for record in marks:
        strategies[record["strategy"]] += 1
    if strategies != {
        "polymer": 4,
        "protecting_group": 38,
        "first_as_prep": 8,
        "normal": 97,
    }:
        problems.append(f"strategy dispatch counts {strategies}")
    # a protecting_group program with no boundary loses its whole deprotection stage silently
    sectionless = [
        m["name"]
        for m in marks
        if m["strategy"] == "protecting_group"
        and not any(s["is_pg_removal"] for s in m["steps"])
    ]
    if sectionless:
        problems.append(
            f"protecting-group programs with no `No` boundary: {sectionless}"
        )

    disconnections = [
        r for r in rules["disconnections"] if not r["macro"] and not r["ring"]
    ]
    macro = [r for r in rules["disconnections"] if r["macro"]]
    ring = [r for r in rules["disconnections"] if r["ring"]]
    if len(disconnections) != 39:
        problems.append(
            f"{len(disconnections)} disconnection rules, expected 39 (R12.3 splits)"
        )
    if len(macro) != 39:
        problems.append(f"{len(macro)} macro twins, expected 39")
    if len(ring) != 76:
        problems.append(f"{len(ring)} heterocyclisation rules, expected 76")
    if any(r["macro"] for r in ring):
        problems.append("a heterocyclisation rule is marked macrocyclic")
    if len({r["id"] for r in rules["disconnections"]}) != len(rules["disconnections"]):
        problems.append("duplicate rule id - `_positions()` would select both")
    problems.extend(_ring_labels_survive())
    problems.extend(_no_duplicate_disconnections())

    for record in rules["disconnections"] + [s for m in marks for s in m["steps"]]:
        for text in [record["smarts"]] if "smarts" in record else record["variants"]:
            left, _right = text.split(">>")
            try:
                if query_labels(smarts(left)):
                    problems.append(f"reactant-side token in {text!r}")
                transformer = SynthonTransformer.from_smarts(text)
            except Exception as exc:
                problems.append(f"{text!r} {type(exc).__name__}: {exc}")
                continue
            for token in transformer._synthon_labels.values():
                if token not in PAPER_CODE_TO_LABEL.values():
                    problems.append(f"{text!r} emits unknown token {token!r}")
            if '"slots"' in json.dumps(record):
                problems.append(f"{text!r} carries a slots key")

    if len(rules["pairs"]) != 29:
        problems.append(
            f"{len(rules['pairs'])} partner pairs, expected 29 (24 upstream + 2 for F7 + 3 for F18)"
        )
    if len(rules["ring_pairs"]) != 2:
        problems.append(
            f"{len(rules['ring_pairs'])} ring-only pairs, expected 2 (the two cycloadditions)"
        )
    if len(rules["forbidden_marks"]) != 12:
        problems.append(
            f"{len(rules['forbidden_marks'])} forbidden-mark entries, expected 12"
        )

    problems.extend(_macro_derivation_diff(config_dir, macro))
    return problems


def _macro_derivation_diff(config_dir: Path, generated: list[dict]) -> list[str]:
    """Build each MR rule from the macro file alone and diff it against the R twin's labelling."""
    notes = []
    root = ElementTree.parse(config_dir / "SetupForMacrocycles.xml").getroot()
    upstream = {
        r.tag: r
        for g in root.find("AvailableReactions")
        for r in g
        if r.get("SMARTS") != "None"
    }
    for record in generated:
        base = record["id"][:-1] if record["id"][-1].isalpha() else record["id"]
        rule = upstream.get(base)
        if rule is None:
            notes.append(f"note: {base} absent from SetupForMacrocycles.xml")
            continue
        reactant, product = rule.get("SMARTS").split(">>")
        if product.startswith("(") and product.endswith(")"):
            # the macro products are grouped to mean "one fragment"; chython parses the group
            # identically to the ungrouped form, so `single_product` carries that intent instead
            product = product[1:-1]
        try:
            theirs = _build_rule(
                reactant, product, _split_label_sets(rule.get("Labels"))[-1], base
            )
        except (ConversionError, DialectError, Exception) as exc:
            notes.append(
                f"note: {base} does not build from the macro file ({type(exc).__name__}: {exc})"
            )
            continue
        ours = query_labels_of(record["smarts"].split(">>", 1)[1])
        if len(theirs) == 1 and query_labels_of(theirs[0].split(">>", 1)[1]) != ours:
            notes.append(
                f"note: {base} upstream assigns {query_labels_of(theirs[0].split('>>', 1)[1])} "
                f"where the R twin assigns {ours} - taking the R twin"
            )
    return notes


def query_labels_of(template: str) -> dict[int, str]:
    return {
        n: a._label
        for n, a in smarts(template).atoms()
        if getattr(a, "_label", None) is not None
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_dir", help="Synt-On's config/ directory")
    parser.add_argument("--out", required=True, help="where the three JSON files go")
    parser.add_argument(
        "--check", action="store_true", help="verify the build before writing"
    )
    args = parser.parse_args(argv)

    config_dir, out = Path(args.config_dir), Path(args.out)
    built = build(config_dir)
    if args.check:
        problems = [p for p in check(built, config_dir) if not p.startswith("note:")]
        notes = [p for p in check(built, config_dir) if p.startswith("note:")]
        for note in notes:
            print(note)
        if problems:
            for problem in problems:
                print(f"FAIL {problem}", file=sys.stderr)
            return 1
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("bb_classes", built["bb_classes"]),
        ("bb_marks", built["bb_marks"]),
        ("rules", built["rules"]),
    ):
        (out / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(
        f"wrote {len(built['bb_classes'])} classes, {len(built['bb_marks'])} rule programs, "
        f"{len(built['rules']['disconnections'])} disconnection rules to {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
