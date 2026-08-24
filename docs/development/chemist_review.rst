Chemist review queue
====================

Open chemistry questions on the ring disconnection rules. Every rule whose provenance is
``llm`` was authored from the literature and tested against the code, but has **not** been
signed off by a practising chemist. This page is the queue.

Held rules
----------

Deliberately absent from ``rules.json``. They live in the curation record and ship only once
the question against each is answered.

.. list-table::
   :header-rows: 1
   :widths: 10 38 52

   * - Rule
     - Name
     - Why it is held
   * - ``R16.4a``
     - imidazole, N-substituted / amidine + alpha-halo ketone (SHIPPED RULE; DO NOT SHIP THE CURATED FORM EITHER)  [lane C, was R16.4a]
     - curator DO NOT SHIP - curated_C.md: "claims N-substituted imidazoles by the amidine route. DOES NOT ROUND-TRIP. Do not ship. Correct only for N-methyl." Report 9.3: the gate disagrees and is the weaker witness. Owner decision 2: believe the curator.
   * - ``R17.3``
     - Hantzsch pyrrole / Trofimov pyrrole (1,3-N,C-dinucleophile + 1,2-dielectrophile)  [lane A, was A3]
     - Q1 (report section 3) - A3: does Trofimov/Hantzsch pyrrole need a C3 or C4 EWG? Owner decision 2: held pending the chemist answer.
   * - ``R17.4``
     - thiophene from a 1,3-S,C-dinucleophile (thionitroacetamide / thioamide + alpha-halo ketone)  [lane A, was A4]
     - Q1 (report section 3) - A4: is the C3-EWG guard genuinely mandatory? The single most restrictive guard in the port. Owner decision 2: held pending the chemist answer.
   * - ``R17.5``
     - Knorr pyrrole (alpha-aminoketone + 1,3-dicarbonyl; also + DMAD)  [lane A, was A5]
     - Q1 (report section 3) - A5: must Knorr pyrrole C3 bear an EWG? Owner decision 2: held pending the chemist answer.
   * - ``R17.6``
     - Fiesselmann thiophene (thioglycolate + acetylenic ester / 1,3-dielectrophile)  [lane A, was A6]
     - Q1 (report section 3) - A6: is a 2-nitro / 2-sulfonyl thiophene a Fiesselmann product, or is C2 ester/nitrile/ketone only? Owner decision 2: held pending the chemist answer.
   * - ``R17.7``
     - oxa-Fiesselmann furan (glycolate/glyoxylate + 1,3-dielectrophile)  [lane A, was A7]
     - Q3 (report section 3) - A7 oxa-Fiesselmann furan is UNVERIFIED as a general method (10 USPTO reactions). Owner decision 2: held out of rules.json.
   * - ``R17.8``
     - aza-Fiesselmann pyrrole (glycine ester + 1,3-dielectrophile / chalcone route)  [lane A, was A8]
     - Q3 (report section 3) - A8 aza-Fiesselmann pyrrole is UNVERIFIED as a general method (44 USPTO reactions; the plain 1,3-diketone version unconfirmed). Owner decision 2: held out of rules.json.
   * - ``R17.10``
     - Hinsberg-type furan (diglycolate + 1,2-dicarbonyl)  [lane A, was A10]
     - Q3 (report section 3) - A10 Hinsberg furan: 0 USPTO reactions, 0 patents, 0 ChEMBL fires, curator UNVERIFIED. Owner decision 2 and report 9.7: held out of rules.json.
   * - ``R17.11``
     - Hinsberg-type pyrrole (iminodiacetate + 1,2-dicarbonyl)  [lane A, was A11]
     - Q3 (report section 3) - A11 Hinsberg pyrrole is UNVERIFIED, no worked example found. Owner decision 2: held out of rules.json.
   * - ``R17.15``
     - indole / Hemetsberger-Knittel (azidoacrylate thermolysis)  [lane A, was A16]
     - Q1 (report section 3) - A16: Hemetsberger C2 ester only, or admit 2-carboxamide and 2-nitrile? Owner decision 2: held pending the chemist answer.

Questions, grouped
------------------

Grouped so they can be answered in one sitting rather than rule by rule.

**Coverage note.** The `chemist_review_flags` were supplied in full for lane A's 21 rules and lane
B's first two. The other 73 rules' flags exist only in transient agent output and are **not on
disk** (§5, §9). The groups below are therefore complete for 23 rules and partial for the rest;
questions recovered from the lane markdowns and the audits are marked *(from lane notes)* or
*(new, from audit/adversary)*.

Q1. Is the activating group I made mandatory actually mandatory?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each of these guards refuses a whole substitution class. One yes/no each.

- **A4** — is a C3 EWG (NO₂ / CN / ester / ketone / sulfonyl) genuinely required, i.e. does an
  unactivated thioamide + α-bromoketone *always* give the Hantzsch **thiazole** rather than the
  thiophene? *This is the single most restrictive guard added anywhere in the port.* And is the EWG
  list complete — should C3 aryl, vinyl or CF₃ also count?
- **A6** — is a 2-nitro or 2-sulfonyl thiophene really a Fiesselmann product, or should C2 be
  narrowed to ester / nitrile / ketone only?
- **A5** — should C3 be *required* to bear an EWG (the 1,3-dicarbonyl always supplies one), or is an
  unactivated C3 ever a Knorr product?
- **A3** — does the rule need C3 or C4 to bear an EWG? Left open because Trofimov works on plain
  ketoximes.
- **A16** — Hemetsberger C2: ester only, or admit 2-carboxamide and 2-nitrile (azidoacetamide /
  azidoacetonitrile analogues)?

Q2. Should the exocyclic-heteroatom exclusion be widened or narrowed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **A1** — 3-hydroxypyrrole and 3-aminopyrrole still fire. Is a 2-hydroxy-1,4-dicarbonyl a real
  Paal-Knorr substrate? The α-amino version self-condenses to a pyrazine — exclude 3-amino too?
- **A1** — halogens are excluded at the α-carbon on the argument that a 2-halopyrrole implies an
  acyl halide. Correct, or should the rule still offer it since 2-halopyrroles are usually made by
  halogenating a pyrrole this rule could have built?
- **A12** — is a 3-alkoxy or 3-amino indole ever a Fischer product?
- *(new, from adversary)* **A12 / A14 / A15 / A19 and their lane-E twins accept N-amino and
  N-acetoxy azoles** — 1-aminoindole and 1-acetoxyindole both fire and round-trip. Lane B excluded
  exactly this with `!$(n-[#7]);!$(n-[#8])`. Adopt the four-token idiom family-wide?
- *(new, from lane C)* the `!$([c]!@[#7,#8,#16])` C2 guard on **C:R16.10/15/16** removes the
  **largest drug-relevant thiazole subclass** — sulfathiazole, famotidine, ritonavir, meloxicam,
  cefotaxime, dasatinib. After it the port has **no faithful retro-Hantzsch for any 2-aminothiazole**.
  Is that price acceptable?

Q3. Are these four routes real *general* methods, or a handful of special cases?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The harvest marked them UNVERIFIED and curation could not upgrade them. All four are medium
confidence and three are thinly corroborated in patents (§6b).

- **A7** oxa-Fiesselmann furan — sulfur original is a named reaction; the oxygen version is assembled
  case-by-case (closest: *Org. Lett.* 2000, doi 10.1021/ol0059652). USPTO: 10 reactions.
- **A8** aza-Fiesselmann pyrrole — the chalcone/glycine-ester electrocyclisation
  (doi 10.1021/jo5021823) is solid; **is the plain 1,3-diketone version real?** USPTO: 44 reactions.
- **A10** Hinsberg **furan** — reviews assert the extension; no worked example with a yield was
  found, and the guide's own section heading says "furans" while every product it draws is a
  thiophene. **USPTO: 0 reactions, 0 patents.**
- **A11** Hinsberg **pyrrole** — asserted in review literature, no worked example found. Does
  intramolecular Claisen condensation of the iminodiacetate kill the N-H case specifically?

**Consequence to decide with them:** A7 and A10 are the port's **only two furan rules**, and both
require an ester at C2. A plain 2,5-dialkylfuran therefore matches **no rule in the port at all**.
Ship medium-confidence rules, or declare furans out of scope until a verified route is found?

Q4. Should the emitted synthon be the reagent a CRO can actually order?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **A5** — free α-aminocarbonyls dimerise to pyrazines; cap the synthon as the HCl salt?
- **A8** — glycine ester hydrochloride must be freebased in situ; cap as the HCl salt?
- **A17** — the synthon is a free phenol, but *o*-alkynylanisoles are the practical substrate.
  Cap `O:nuc` as a methyl ether for this rule?
- **A18** — the synthon is a free thiophenol, which oxidises to the disulfide; the S-methyl
  thioanisole route dominates in practice. Cap `S:nuc` as methyl?
- *(from lane F)* the electrophile cap is misleading family-wide: `…CC[CH3_elec]` is **not** a methyl
  group but CH₂–LG / an aldehyde / a Michael acceptor, and for the four lactam rules the printed
  synthon reads as an aldehyde while the reagent to order is the amino **acid or ester**.

Q5. Regiochemistry — commit to one orientation, or emit both and let scoring decide?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Eight rules ask the same question: **A3** (Hantzsch vs Feist-Benary give *opposite* regiochemistry
on the same two reagents), **A9** and **A11** (unsymmetrical 1,2-diketones), **A12** (unsymmetrical
ketone → two enehydrazines; meta-substituted arylhydrazine → 4- and 6-substituted indoles),
**A14** (the *bulkier* alkyne substituent goes to C2; esters and Boc-amines are unreliable
directors), **A16** and **A19** (a meta-substituted arene gives two nitrene-insertion products),
**A21** (a meta-substituted anilide cyclises onto either ortho position).

Q6. Safety and scale — which of these need a hard flag rather than free offering?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **A4** — α-haloketones are genotoxic alkylators. GMP-suitability flag?
- **A2** — P₂S₅ reduces a ring carboxylate down to the ring carbon; P₄S₁₀/Lawesson also thionates
  amides and esters elsewhere in the molecule.
- **A9** — the oxalate variant uses Me₂SO₄.
- **A16** — the route isolates and heats an organic azide to 130–160 °C; Wikipedia's own assessment
  is that it is "not a popular reaction".
- **A19** — Cadogan burns 3–5 equiv P(OEt)₃ at 160 °C with a real exotherm.
- **A21** — stoichiometric AlCl₃ at 2–4 equiv, chlorinated solvent, large aqueous aluminium quench —
  and **A20 reaches the same targets with Fe/AcOH**.
- *(from lane F)* **R17.6** the thiomorpholine precursor is a **mustard**; propose only the
  telescoped/flow form (doi 10.1021/acs.oprd.2c00214), never a batch isolation.

Q7. Stereochemistry — three adjudications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **A20** — is oxindole C3 epimerised in practice under Fe/AcOH or acid-catalysed lactamisation? If
  yes, `stereo_spec` must say **MIXTURE**, not RETENTION, for the 3-substituted case. *(The curator
  flagged this against their own record; lane E's twin says RETENTION with no caveat — see §5.)*
- **A21** — is the racemic call right for every variant, including the Gassman *t*-BuOCl/methylthio
  route (JACS 1974, 96, 5495)?
- **D:R16.17 Pictet-Spengler** — the C1/C3 relative configuration (cis-1,3 kinetic, trans-1,3
  thermodynamic, attributed to Bailey and Cook) is recorded **UNVERIFIED, primary source not
  opened**. This is the port's only outstanding *literature* gap in stereochemistry (§5).

Q8. Four aggressive scope limits the curators flagged for adjudication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **A12** — deleting `[n;D3]` from shipped Fischer. Argument: in a five-ring with four aromatic
  carbons and one aromatic nitrogen the nitrogen is pyrrole-type by construction, so `D3` carried no
  information. Is there any real aromatic five-ring where that reasoning fails?
- **A13** — `[n;h1]` restricts Leimgruber-Batcho/Cadogan to **N-H** indoles because the nitrogen
  arrives by nitro reduction. Any variant that gives an N-alkyl indole directly?
- **A14** — `[c;h0]` on **both** C2 and C3 (terminal alkynes are not Larock substrates). Should a
  silyl-protected terminal alkyne → 2-silylindole → desilylation be offered as a two-step path?
- **A15** — `[c;h0]` at C2 ("essentially confined to 2-alkyl and 2-aryl indoles"). Is
  2-unsubstituted indole from N-formyl-*o*-toluidine common enough that this costs real hits?

Q9. Whole-molecule conditions that could not be encoded as a positional SMARTS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each is a real scope limit the rule does **not** carry. Encode with an explicit substituent list, or
leave to conditions/scoring?

- **A15** Madelung — base-sensitive esters, nitriles and aryl halides do not survive molten alkoxide
  at 200–400 °C; the Houlihan/Smith alternatives need cryogenic stoichiometric organolithium.
- **A21** Stolle — electron-poor anilides do not cyclise; NO₂ / CF₃ ortho or para block it outright.
- **A13** — nitro-reduction chemoselectivity: alkenes, benzyl ethers, aryl halides and nitriles
  compete.
- **A2** — a free amide or ester elsewhere is thionated by the same reagent.

Q10. Ownership and coverage decisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **A1** — N-amino pyrroles (hydrazine + 1,4-diketone) are deliberately **in** scope; confirm, and
  confirm the N–N guard should stay absent.
- **A2** — the halide-free 1,3-diyne + NaSH variant maps to the same two atoms. Does one rule id
  hide a reagent-class distinction the planner needs?
- **A9** — split the thiazole + bis(TMS)acetylene retro-Diels-Alder route (92%, *J. Org. Chem.* 62,
  1940 (1997)) into its own id? The ester requirement here excludes it.
- **A14** — Bartoli maps to the same two bonds but **fails without an ortho substituent** (it is
  really a 7-substituted-indole route). Split it out with a positive C7 requirement?
- **A14 / A19** — these two deliberately permit N-Ts / N-Ac where the rest of the family forbids
  them, because Larock runs on N-Ts and N-Ac 2-haloanilines and the Buchwald carbazole variant runs
  on 2-acetamidobiphenyl. Confirm they should be the sole owners of that space.
- **B:R16.1** — 1-acyl triazoles are currently allowed. Do acyl azides cycloadd cleanly, or do they
  Curtius-rearrange fast enough that a 1-acyltriazole should be traced to N-acylation instead?
- **B:R16.1** — **benzotriazole is owned by nothing** in the R16 set. Give it its own rule
  (*o*-phenylenediamine + HNO₂)?
- *(from lane E)* four declared gaps: **2-aminobenzoxazole**, **2-aminobenzothiazole**,
  **dibenzofuran**, **dibenzothiophene** have no owner. The first two need a C1 donor at the amidine
  level (BrCN, Hugerschoff), not the aldehyde the rules assume.
- *(from lane D)* **monocyclic pyrimidine cannot have a rule in this dialect at all** (the two
  amidine nitrogens are genuinely equivalent; both label orderings fail). **Cinnoline**,
  **2-pyridone/Guareschi-Thorpe** and **4-quinolone/Conrad-Limpach** have no owner.
- *(from lane F)* **ring closure by N-alkylation of an amide/lactam/carbamate N** is refused by every
  F rule so that "N is a nucleophile" means one thing family-wide; it needs a dedicated id (R17.16).
  **Imides** (glutarimide, thalidomide's ring) and **benzo-fused saturated N-heterocycles**
  (THIQ, tetrahydroquinoline, indoline, chromane) are cross-family merge decisions.
- *(new, from adversary)* **are azaindoles and 7-deazapurines in Fischer/Leimgruber-Batcho scope?**
  See §7 — the fusion guard admits them today.
- *(new, from adversary)* **what is the canonical form of an N-H azinone?** See §7 — the pipeline
  silently rewrites 2-pyridone, carbostyril, 4-quinolone and pyrimidin-4(3H)-one to the aromatic
  hydroxy-azine before any rule sees them.

---

Per-rule flags
--------------

Every question the curators could not settle from the literature, by rule. The full record,
including per-guard rationale, ``smirks_stereo`` and ``stereo_spec``, is in
``research/synthon/curation/curated_rules.json``.

``R16.1a`` — 1,2,3-triazole, N1-substituted / azide + alkyne (CuAAC, RuAAC, SPAAC, thermal Huisgen)  [lane B, was R16.1]
   * Acyl azides: 1-acyl-1,2,3-triazoles are currently ALLOWED by !$(n-[#7]);!$(n-[#8]). Do acyl
     azides cycloadd cleanly under CuAAC, or do they Curtius-rearrange fast enough that a
     1-acyltriazole should be traced to N-acylation of the N-H ring instead? If the latter, add
     !$(n-[#6]=[#8]).
   * Should benzotriazole get its own rule (o-phenylenediamine + HNO2)? It is now claimed by
     nothing in the R16 set.
   * The rule does not distinguish CuAAC from RuAAC from SPAAC. Is a condition-layer flag needed
     so a planner does not propose a 1,5-triazole and then order Cu(I)?
   * N1-alkyl where the alkyl is tertiary/neopentyl: tertiary azides are made and clicked, but is
     the corresponding building block realistically stocked?

``R16.1b`` — 1,2,3-triazole, N-unsubstituted (1H) / azide (NaN3, TMS-N3) + alkyne [n;h1] twin of R16.1  [lane B, was R16.1b]
   * Direct NaN3 + terminal alkyne to the N-H triazole normally needs an N-protecting azide
     surrogate (TMS-N3, or NaN3 with a Cu source then acidic workup). Should this rule carry a
     note that the reagent is not literally 'HN3', or is that a condition-layer concern?
   * Is a 4,5-disubstituted N-H triazole (from an internal alkyne, thermal only, no Cu) still in
     scope for this rule, given the harsher conditions and poor regiocontrol?
   * 5-substituted vs 4-substituted N-H triazole are the same compound by tautomerism — confirm
     the chython canonical form is the one Enamine's catalogue is normalised to, or half the
     stock will silently fail lookup.

``R16.2a`` — tetrazole, 1,5-disubstituted / organic azide + nitrile  [lane B, was R16.2]
   * Sulfonyl azides + nitriles: 1-sulfonyltetrazoles are allowed by the current guards. Is that
     a real preparative route, or should sulfonyl be excluded here even though it must stay in
     R16.1?
   * 5-Amino tetrazoles come from cyanamide (N#C-NH2) + azide — a genuine nitrile. Confirm the
     rule should keep claiming them, i.e. that cyanamide-derived synthons are acceptable building
     blocks.
   * Very hindered or very electron-rich nitriles react slowly; should the rule carry a
     downweight, or does that belong entirely in the condition layer?

``R16.2b`` — tetrazole, 5-substituted N-unsubstituted (1H) / NaN3 + nitrile, [n;h1] twin of R16.2  [lane B, was R16.2b]
   * Losartan is disconnected with the whole biphenyl-imidazole kept on the nitrile fragment. Is
     4'-cyanobiphenyl-2-yl the reagent a chemist would actually expect at this step (it is on the
     real route), or does the port need to cut the biaryl first?
   * Confirm that stocked tetrazoles are catalogued as the neutral N-H parent rather than the
     sodium salt, or the +0 guard will lose stock matches.
   * 5-Aryl tetrazoles with an ortho substituent (the sartan case) are sterically slow with
     NaN3/ZnBr2 — is the standard fix (Bu2SnO/toluene, or the TMS-N3 route) close enough to keep
     this as one rule?

``R16.3a`` — pyrazole, N1-substituted / Knorr (substituted hydrazine + 1,3-dicarbonyl, enaminone or ynone)  [lane B, was R16.3]
   * PYRAZOLONES ARE UNREACHABLE. chython keeps 5-hydroxypyrazole / pyrazol-5(4H)-one non-
     aromatic, so neither the shipped nor the guarded rule fires on edaravone, antipyrine,
     metamizole or phenylbutazone — one of the largest hydrazine + beta-ketoester classes in
     pharma. Does the port want a separate non-aromatic-LHS pyrazolone rule?
   * Acylhydrazides: 1-acylpyrazoles are currently allowed (semicarbazide/acylhydrazide +
     1,3-diketone is a real reaction). Confirm, or move them to N-acylation of the R16.3c
     product.
   * Tosylhydrazide + 1,3-diketone: does the N-tosyl group survive to give an isolable
     1-tosylpyrazole, or does it always eliminate — i.e. is the 1-tosylpyrazole positive control
     a real target class?
   * The regiochemical mixture from an unsymmetrical 1,3-dicarbonyl + a substituted hydrazine is
     not modelled. Should the rule be downweighted when C3 and C5 substituents differ?

``R16.3b`` — pyrazole, N-unsubstituted (1H) / Knorr with hydrazine hydrate, [n;h1] twin of R16.3  [lane B, was R16.3c]
   * N-H pyrazoles are acidophobic and polymerise in strong acid (per the harvest's MGU-guide
     citation). Should the rule carry a 'weak acid catalyst only' condition note that the
     N-substituted twin does not need?
   * Confirm that chython's canonical tautomer for 3-substituted-1H-pyrazoles matches the form
     Enamine's catalogue is normalised to; if not, half the stock lookups fail silently.
   * Hydrazine hydrate handling (toxicity, kg scale): is that enough of a debit to downweight
     this rule against R16.3 with a pre-formed arylhydrazine?

``R16.4a`` — imidazole, N-substituted / amidine + alpha-halo ketone (SHIPPED RULE; DO NOT SHIP THE CURATED FORM EITHER)  [lane C, was R16.4a]
   * Is the N-alkylamidine + alpha-halo ketone route to a 1-alkylimidazole worth keeping at all,
     or is the real bench route always make-the-NH-imidazole-then-alkylate? If the latter,
     retiring R16.4a costs nothing and R16.4b + R12-family N-alkylation covers it.
   * 1,2-disubstituted N-alkyl imidazoles (e.g. 1,2-dimethylimidazole, the ondansetron ring) now
     have NO disconnection at all: R16.4a is tautomer-broken and van Leusen cannot substitute C2.
     Is that an acceptable coverage hole, or is a Debus-type / N-alkylation-staged rule needed?
   * Does the R16.4a proposal 'amidine + alpha-halo ketone' ever appear in your own route data
     for an N-SUBSTITUTED imidazole, or only for NH ones?

``R16.4b`` — imidazole, N-unsubstituted (1H) / amidine + alpha-halo ketone  [lane C, was R16.4b]
   * A 2-aminoimidazole is disconnected here to free guanidine + alpha-halo ketone. Is that the
     route you would run, or do you always go through a protected/S-methylisothiourea guanidine
     equivalent that a stock lookup would need to recognise?
   * For a 4,5-unsymmetrically substituted NH-imidazole the enumerator legitimately returns both
     regiochemical assemblies (see the losartan-type output). Is the planner meant to see both,
     or should the more nucleophilic amidine nitrogen be preferred?
   * Does the rule need to exclude a 2-nitro / 2-sulfonyl imidazole, where the C2 carbon is not
     amidine-derived in practice?

``R16.5`` — isoxazole / hydroxylamine + 1,3-dicarbonyl (Claisen-type condensation)  [lane B, was R16.5]
   * Hydroxylamine + an unsymmetrical 1,3-dicarbonyl gives both 3- and 5-regioisomers and the
     ratio swings with pH and with which carbonyl is more electrophilic. Should the rule be
     downweighted when C3 and C5 substituents differ, the way the Knorr pyrazole case should be?
   * Does 1,2-benzisoxazole need its own rule (o-hydroxyaryl ketoxime cyclodehydration)? It is
     now claimed by nothing.
   * beta-Ketoester + NH2OH can stop at the isoxazol-5(4H)-one (an oxazolone) rather than
     aromatising — is the same chython non-aromatisation trap that hides pyrazolones also hiding
     isoxazolones here? Worth a direct check before shipping.

``R16.6a`` — pyridine / Kroehnke–Bohlmann-Rahtz (enamine + enone), orientation A  [lane D, was R16.6a]
   * R16.6a/b both fire on 2-chloropyridine and hand back a chloro-enone/chloro-imine synthon. Is
     a 2-halo azine ever built this way, or is it always made from the pyridone with POCl3 — in
     which case a deoxychlorination rule should own it and R16.6 needs !$(c[F,Cl,Br,I]) on the
     alpha carbons?
   * R16.6b keeps 3-hydroxypyridine, returning the alpha-hydroxy imine [NH_nuc]=C[CH2_nuc2]O. Is
     an alpha-hydroxy enamine a real, stockable Kroehnke partner, or should the enol guard extend
     onto the nuc2 carbon too?
   * Should the pyridine rules demand an EWG (ester, ketone, nitrile) on the C3/nuc2 carbon, as
     the classical Bohlmann-Rahtz beta-ketoester enamine has, or is an unactivated enamine a real
     reagent class here?

``R16.6b`` — pyridine / Kroehnke–Bohlmann-Rahtz, orientation B (mirror twin of R16.6a)  [lane D, was R16.6b]
   * Shipping a/b twins doubles the number of pyridine disconnections a planner sees. On a
     symmetric pyridine the two are identical and will need deduping downstream — is that
     acceptable, or should the twins be merged behind one rule id with two RHS?
   * Same 2-halo-azine question as R16.6a.
   * R16.6b keeps 3-hydroxypyridine as an alpha-hydroxy imine synthon. Real reagent or artefact?

``R16.7`` — pyridazine / hydrazine + 1,4-dicarbonyl; also phthalazine from an ortho-diacylarene  [lane D, was R16.7]
   * Is one rule right for two reactions with different oxidation bookkeeping? Pyridazine from a
     1,4-diketone + N2H4 goes through a 1,4-dihydropyridazine and needs a separate OXIDATION to
     aromatise; phthalazine from o-phthalaldehyde + N2H4 does not. If the planner must see that
     oxidation step, R16.7 has to split into R16.7a (monocyclic, x2 at 4/5) and R16.7b (benzo-
     fused, R2 at 4/5).
   * R16.7 keeps pyridazin-3-ol (i.e. pyridazin-3(2H)-one) and returns
     C([CH3_elec2])=C[CH2_elec2]O. The real synthesis of a pyridazin-3-one is from a gamma-keto
     ACID, not a 1,4-diketone. Is that synthon an acceptable stand-in, or is this a wrong hit
     that a chemist would reject?
   * Cinnoline is now excluded entirely. Is a cinnoline rule worth adding, or is that ring rare
     enough in the Enamine catalogue to ignore?

``R16.8`` — Fischer indole (SHIPPED RULE, re-guarded)  [lane A, was A12]
   * PLEASE CONFIRM THE D3 DELETION IS SAFE. My argument is structural: a five-ring with four
     aromatic carbons and one aromatic nitrogen forces a pyrrole-type nitrogen, so D3 was pure
     loss. Is there any aromatic five-ring in a real target where that reasoning fails?
   * An unsymmetrical ketone gives two Fischer regioisomers, and the rule offers only the
     traversal the automorphism filter kept. Is that acceptable, or does this rule also need
     automorphism_filter=False?
   * Meta-substituted arylhydrazines give 4- and 6-substituted indoles as a mixture. Should the
     rule flag that, or leave it to route scoring?
   * Excluding 2- and 3-heteroatom indoles: is a 3-alkoxy or 3-amino indole ever a Fischer
     product, or is my exclusion safe?

``R16.9`` — quinoline / Friedlaender (2-aminoaryl ketone + ketone)  [lane D, was R16.9]
   * R16.9 fires on 1,5- and 1,8-naphthyridine, handing back an aminopyridine-carbaldehyde plus a
     ketone. Is a Friedlaender on an electron-poor 3-aminopyridine-2-carbaldehyde a real, stocked
     reagent class, or should the amino-arene be restricted to a carbocyclic ring?
   * Should Friedlaender require the ketone partner to have an enolisable alpha-CH? The nuc2
     carbon can currently be fully substituted, which would make the aldol step impossible.
   * Unlike R16.13, R16.9 still fires on 3- and 4-hydroxyquinoline because all its labels are
     bivalent and H-cap to alcohols. Are 3-hydroxy- and 4-hydroxyquinoline genuinely accessible
     by Friedlaender, or is this a case where the mechanism, not the tautomer, should exclude
     them?

``R17.1`` — Paal-Knorr pyrrole (1,4-dicarbonyl + NH3/RNH2)  [lane A, was A1]
   * 3-Hydroxypyrrole still fires and round-trips with formula conserved. Is a
     2-hydroxy-1,4-dicarbonyl (acyloin-derived) a real Paal-Knorr substrate, or should C3/C4 also
     carry the no-exocyclic-heteroatom guard?
   * 3-Aminopyrrole still fires. The implied reagent is an alpha-amino-1,4-diketone, which the
     MGU guide warns self-condenses to a pyrazine (p-11). Exclude 3-amino too?
   * N-amino pyrroles (hydrazine + 1,4-diketone -> 1-aminopyrrole) are deliberately left IN
     scope. Confirm that is right and that the N-N guard should stay absent.
   * Halogens are in the alpha-carbon exclusion list on the argument that a 2-halopyrrole implies
     an acyl halide reagent. Correct, or should the rule still offer the disconnection since
     2-halopyrroles are usually made by halogenating a pyrrole this rule could have built?
   * NH-pyrroles are acidophobic and polymerise in strong acid (guide p-23). Should the rule
     carry any penalty/flag for NH targets, or is that purely a conditions matter left to the
     planner?

``R17.2`` — Paal-Knorr thiophene (1,4-dicarbonyl + P4S10/Lawesson; also SH- + 1,3-diyne)  [lane A, was A2]
   * P4S10/Lawesson also thionates amides and esters elsewhere in the molecule (harvest pitfall).
     Should the rule carry a whole-molecule exclusion for a free amide/ester, or is that a
     conditions/route-scoring matter rather than a rule guard?
   * P2S5 reduces a ring carboxylate down to the ring carbon (guide p-04). Does that mean a
     thiophene-2-carboxylate target should be excluded from this rule, or is the Lawesson variant
     clean enough to leave it in?
   * The halide-free 1,3-diyne + NaSH variant maps to the same two atoms. Does keeping both under
     one rule id hide a reagent-class distinction the planner needs?

``R17.3`` — Hantzsch pyrrole / Trofimov pyrrole (1,3-N,C-dinucleophile + 1,2-dielectrophile)  [lane A, was A3]
   * The enumerator returns the target PLUS the C4/C5-reversed regioisomer because both carry the
     same (C,False,elec) key. The guide says Hantzsch and Feist-Benary give OPPOSITE
     regiochemistry on the same two reagents (p-14). Is emitting both orientations the right
     behaviour here, or should the rule commit to the Hantzsch orientation?
   * Trofimov N-vinylation is a common side reaction. Should N-vinyl pyrroles be excluded from
     the target side, or is that a yield matter?
   * Does this rule need a positive requirement that C3 or C4 bears an EWG (the enamine is
     normally derived from a beta-ketoester)? Left unguarded because Trofimov works on plain
     ketoximes.

``R17.4`` — thiophene from a 1,3-S,C-dinucleophile (thionitroacetamide / thioamide + alpha-halo ketone)  [lane A, was A4]
   * THE KEY QUESTION FOR THIS RULE: is the C3 EWG genuinely mandatory, i.e. does an unactivated
     thioamide + alpha-bromoketone always give the thiazole rather than the thiophene? The guard
     makes that a hard requirement and it is the single most restrictive thing I added in this
     scope.
   * Is my EWG list complete for this purpose - should a C3 aryl, vinyl or CF3 also count as
     sufficient activation?
   * S-alkylation must outrun N- and C-alkylation of the thioamide (harvest pitfall). Does the
     substrate need any further structural constraint to make that reliable, e.g. a tertiary
     amide nitrogen?
   * alpha-Haloketones are genotoxic alkylators; does this rule need a GMP-suitability flag
     rather than being offered freely?

``R17.5`` — Knorr pyrrole (alpha-aminoketone + 1,3-dicarbonyl; also + DMAD)  [lane A, was A5]
   * Free alpha-aminocarbonyls dimerise to pyrazines and must be used as salts or generated in
     situ (guide p-11). Should the emitted synthon be capped as the HCl salt rather than the free
     amine, so the stocked reagent is orderable?
   * The amine always condenses with the MORE electrophilic carbonyl of the 1,3-dicarbonyl (guide
     p-12). Should the rule encode a preference when the two dicarbonyl carbons differ (e.g.
     ketone over ester), or leave the regiochemistry to scoring?
   * Should the rule require an EWG at C3 (the 1,3-dicarbonyl always supplies one)? Left
     unguarded - confirm whether an unactivated C3 is ever a Knorr product.

``R17.6`` — Fiesselmann thiophene (thioglycolate + acetylenic ester / 1,3-dielectrophile)  [lane A, was A6]
   * The harvest measured that on a symmetric-LHS target (COC(=O)c1sc(C(=O)OC)cc1O) chython's
     automorphism filter emits only ONE traversal direction, and it picked the one that puts the
     3-OH on an sp2 carbon of fragment B, producing an enol that tautomerised and broke the round
     trip. Should this rule ship with automorphism_filter=False, or is halved coverage
     acceptable? (Same defect as R16.6; not a guard question.)
   * Is a 2-nitro or 2-sulfonyl thiophene really a Fiesselmann product, or should the C2
     requirement be narrowed to ester/nitrile/ketone only?
   * Does a free C3-OH survive the alkoxide conditions well enough that the
     3-hydroxythiophene-2-carboxylate disconnection should be offered without a caveat?

``R17.7`` — oxa-Fiesselmann furan (glycolate/glyoxylate + 1,3-dielectrophile)  [lane A, was A7]
   * MARKED UNVERIFIED BY THE HARVEST AND STILL UNVERIFIED: the sulfur original is a named
     reaction with a documented general scope, the oxygen version is assembled from case-by-case
     literature (closest instance: Org. Lett. 2000, 3-aminofuran-2-carboxylates from an alpha-
     cyanoketone + ethyl glyoxylate, doi 10.1021/ol0059652). Is the base-mediated glycolate +
     1,3-dielectrophile condensation a real general furan synthesis, or only a handful of special
     cases?
   * If it is only special cases, should the C2 requirement be narrowed from the EWG list to
     specifically an ester, and should C3 be required to bear NH2 or OH?
   * Given that A7 and A10 are the port's ONLY furan entries, is it acceptable to ship a medium-
     confidence rule here, or should the port simply declare furans out of scope until a verified
     route is found?

``R17.8`` — aza-Fiesselmann pyrrole (glycine ester + 1,3-dielectrophile / chalcone route)  [lane A, was A8]
   * MARKED UNVERIFIED-AS-A-GENERAL-METHOD by the harvest and still so. The chalcone + glycine
     ester electrocyclisation/oxidation (J. Org. Chem. 2015, doi 10.1021/jo5021823) is solid; the
     plain 1,3-diketone version is assembled from the enaminone literature. Is the 1,3-diketone
     version real?
   * The forward route needs an aromatising oxidation the rule does not represent, so the pathway
     is one step longer than it looks. Should the rule be penalised in scoring, or annotated?
   * Glycine ester hydrochloride must be freebased in situ. Should the emitted synthon be capped
     as the hydrochloride so the stocked reagent matches?

``R17.9`` — Hinsberg thiophene (thiodiglycolate + 1,2-dicarbonyl)  [lane A, was A9]
   * Should the thiazole + alkyne retro-Diels-Alder route (guide p-22, 92%, J. Org. Chem. 62,
     1940 (1997)) be split out as its own rule id, given the ester requirement here excludes it?
     It reaches 3,4-bis(silyl)thiophenes this rule cannot.
   * The Hinsberg mechanism runs through a delta-lactone and commonly gives mixed diester/half-
     acid products. Should the rule emit the half-acid as an alternative product, or is the
     diester the right canonical answer?
   * Unsymmetrical alpha-diketones give regioisomer mixtures. Confirm that emitting both
     orientations is the desired behaviour.
   * The oxalate variant uses Me2SO4. Should that variant carry a hard safety flag, or is it
     enough that MeI is an alternative?

``R17.10`` — Hinsberg-type furan (diglycolate + 1,2-dicarbonyl)  [lane A, was A10]
   * UNVERIFIED at the level of a named preparation: the Hinsberg reviews state the method was
     extended to furan, selenophene and pyrrole derivatives, but the harvest found no worked
     example with a yield for the furan case, and it noted that the MGU guide's own section
     heading says 'furans' while every product it draws is a thiophene - that heading is an
     error, not evidence. Does a real diglycolate + 1,2-diketone furan preparation exist?
   * Yields for the oxygen version are expected to be lower (weaker C-O-C nucleophile) and
     furan-2,5-diesters decarboxylate on hydrolysis. Should the rule be down-weighted rather than
     shipped at parity with A9?
   * Same question as A7: is a medium-confidence rule acceptable when it is one of only two furan
     entries in the whole port?

``R17.11`` — Hinsberg-type pyrrole (iminodiacetate + 1,2-dicarbonyl)  [lane A, was A11]
   * UNVERIFIED at the preparation level: the pyrrole extension is asserted in the Hinsberg
     review literature but the harvest found no worked example with a yield. Does one exist?
   * Competing intramolecular Claisen condensation of the iminodiacetate is a plausible side
     path. Does that kill the method for N-H iminodiacetates specifically, i.e. should the rule
     be restricted to N-substituted pyrroles?
   * Unsymmetrical diketones give regioisomers; confirm emitting both orientations is wanted.

``R17.12`` — indole / Leimgruber-Batcho (covers Cadogan-Sundberg)  [lane A, was A13]
   * MY MOST AGGRESSIVE GUARD - PLEASE ADJUDICATE: [n;h1] restricts this rule to N-H indoles on
     the grounds that the nitrogen is delivered by nitro reduction. Is there any Leimgruber-
     Batcho or Cadogan variant that gives an N-alkyl indole directly (e.g. reductive cyclisation
     followed by in-situ alkylation counted as one step)? If yes, drop h1.
   * 2-Substituted indoles need a ketone acetal (DMA-DMA) instead of DMF-DMA and are much less
     general. Should the rule be restricted to 2-unsubstituted indoles (c2 with h1), leaving
     2-substituted ones to Fischer/Madelung/Larock?
   * Nitro-reduction chemoselectivity: alkenes, benzyl ethers, aryl halides and nitriles compete.
     Should any of those carry a whole-molecule exclusion, or is that a conditions matter?
   * Does the substrate need an explicit requirement that C3 bears a substituent (the benzylic
     carbon of the nitrotoluene)? Plain indole from o-nitrotoluene works, so I left it open -
     confirm.

``R17.13`` — indole / Larock heteroannulation (Bartoli maps to the same two bonds)  [lane A, was A14]
   * Is [c;h0] on BOTH C2 and C3 the right reading of 'terminal alkynes are not substrates'? A
     silyl-protected terminal alkyne gives a 2-silylindole that is then desilylated - should the
     rule offer that two-step path for 2-unsubstituted indoles, or is leaving it to
     Fischer/Leimgruber-Batcho correct?
   * Regioselectivity: the bulkier alkyne substituent goes to C2. Should the rule encode any
     steric preference, or emit both orientations and let scoring decide?
   * Bartoli FAILS without an ortho substituent on the nitroarene (it is really a 7-substituted-
     indole route). Since Bartoli maps to the same two bonds, should it be split into its own
     rule with a positive requirement for a C7 substituent?
   * The rule deliberately permits N-Ts and N-Ac indoles where the rest of the family excludes
     them. Confirm Larock on N-acyl-2-iodoanilines is reliable enough to be the sole owner of
     that space.

``R17.14`` — indole / Madelung (and the Houlihan and Smith variants)  [lane A, was A15]
   * I added [c;h0] at C2 on the strength of 'essentially confined to 2-alkyl and 2-aryl
     indoles'. Is 2-unsubstituted indole from N-formyl-o-toluidine common enough that this guard
     costs real hits?
   * NOT ENCODED, needs a decision: base-sensitive functionality (esters, nitriles, aryl halides)
     does not survive the classic 200-400 C molten-alkoxide conditions, and the Houlihan/Smith
     alternatives need cryogenic stoichiometric organolithium. This is a whole-molecule condition
     I judged un-encodable as a positional SMARTS. Should the rule instead carry a hard exclusion
     for a free ester or nitrile anywhere in the target?
   * Electron-withdrawing groups on the benzo ring lower yield except at C5, and a bulky C6
     substituent kills the reaction. Worth encoding, or scoring?

``R17.15`` — indole / Hemetsberger-Knittel (azidoacrylate thermolysis)  [lane A, was A16]
   * Should the C2 requirement be an ester only, or should a 2-carboxamide / 2-nitrile also be
     admitted (azidoacetamide and azidoacetonitrile analogues)?
   * Safety: this route isolates and then heats an organic azide to 130-160 C. Wikipedia's own
     assessment is that it is 'not a popular reaction'. Should the rule be gated behind an
     explicit hazard flag or excluded from any route intended for scale?
   * The mechanism is formally unknown (azirines have been isolated; the nitrene is postulated).
     Does that affect how the port should weight this disconnection?
   * A meta-substituted arylaldehyde gives two regioisomeric indoles. Encode a preference, or
     leave to scoring?

``R17.16`` — benzofuran / 5-endo-dig cycloisomerisation of a 2-alkynylphenol  [lane A, was A17]
   * 5-endo-dig vs 5-exo-dig: the literature disagrees. Is the endo mode reliable enough for a
     2-substituted benzofuran target that this rule should be offered without qualification?
   * The synthon emitted is a FREE phenol, but o-alkynylanisoles are often the practical
     substrate and need either demethylation or an electrophilic (I2) closure. Should the
     leaving-group cap for O:nuc be a methyl ether here rather than H, so the stocked reagent
     matches?
   * The route is two precious-metal steps in sequence (Pd/Cu Sonogashira, then Au or Pd). Should
     that be reflected in scoring, or is the 100% atom-economical isomerisation step enough of a
     redeeming feature?
   * Internal alkynes bearing a propargylic OH divert to 3-hydroxy-dihydrobenzofurans. Worth a
     structural exclusion on the C3 substituent?

``R17.17`` — benzothiophene / 5-endo-dig cycloisomerisation of a 2-alkynylthiophenol  [lane A, was A18]
   * The rule emits a FREE thiophenol ([SH_nuc]), but free 2-alkynylthiophenols oxidise to
     disulfides, which is why the S-methyl thioanisole route dominates in practice - the emitted
     synthon is one demethylation away from the orderable reagent. Should the S:nuc leaving-group
     cap be a methyl group for this rule specifically?
   * Sulfur poisons Pd, so catalyst loadings are higher than in the oxygen series. Does that
     change whether this disconnection should be offered at parity with A17?
   * The most-used closure (I2 / ICl / PhSeBr) demethylates and cyclises in one step and installs
     a C3 halogen. Should the rule offer the 3-halo product as a separate downstream node?

``R17.18`` — carbazole / Cadogan (2-nitrobiphenyl + P(OEt)3) or Buchwald C-N of a 2-aminobiphenyl  [lane A, was A19]
   * The rule deliberately permits N-acyl carbazoles because the Buchwald variant uses a
     2-acetamidobiphenyl. Confirm that is right, and that an N-alkyl-2-aminobiphenyl also
     cyclises (i.e. that N-alkylcarbazoles are legitimately owned here rather than by
     N-alkylation of carbazole).
   * A meta-substituted 2-nitrobiphenyl gives two regioisomeric carbazoles (the nitrene inserts
     at either ortho position of the distal ring). Encode or score?
   * Sundberg showed beta,beta-disubstituted substrates rearrange and ring-expand instead of
     cyclising. Is there a structural pattern for that worth excluding?
   * Cadogan burns 3-5 equivalents of P(OEt)3 at 160 C with a real exotherm. Worth a scale flag?

``R17.19`` — oxindole (indolin-2-one) / intramolecular lactamisation of a 2-aminophenylacetic acid  [lane A, was A20]
   * STEREOCHEMISTRY, PLEASE CHECK HARDEST: I claim C3 is relayed with retention because no bond
     to it is broken, and I verified the machinery preserves it. But under the practical
     conditions (nitro reduction in Fe/AcOH or Zn/AcOH at reflux, or acid-catalysed
     lactamisation) is C3 epimerised in practice? If yes, should stereo_spec say MIXTURE rather
     than RETENTION for the 3-substituted case?
   * For a 3-alkylidene target, the alkene must be installed AFTER cyclisation (a Knoevenagel),
     because the nitro reduction would also reduce it. The retro rule as written hands back a
     3-alkylidene 2-aminophenyl acetaldehyde/acid, which is not the real reagent. Should the rule
     refuse 3-alkylidene targets and leave them to a separate Knoevenagel step, or is the current
     behaviour acceptable?
   * Over-reduction of a C3 substituent is a known failure. Any structural class worth excluding
     outright?

``R17.20`` — oxindole / Stolle intramolecular Friedel-Crafts of a 2-chloroacetanilide  [lane A, was A21]
   * THE ONE GUARD I COULD NOT ENCODE: electron-poor anilines do not cyclise, and strongly
     deactivating groups (NO2, CF3) ortho or para to the reacting position block the reaction
     outright. Positional recursive SMARTS for 'no strong EWG ortho or para to C3a' costs more
     legitimate hits than it saves. Should this be encoded anyway, and if so, with what exact
     substituent list?
   * Friedel-Crafts regiochemistry: a meta-substituted anilide cyclises onto either ortho
     position, giving 4- and 6-substituted oxindoles as a mixture. Encode a preference or score
     it?
   * Stoichiometric AlCl3 at 2-4 equivalents (the amide chelates it) plus a chlorinated solvent
     and a large aqueous aluminium quench. Should this rule be down-weighted for scale, given A20
     reaches the same targets with Fe/AcOH?
   * Is my racemic call right for every variant, or does the Gassman t-BuOCl/methylthio route
     (JACS 1974, 96, 5495) proceed with any stereochemical difference at C3?

``R17.30`` — pyrazole, N1-substituted / nitrile imine (hydrazonoyl halide) + alkyne or alkene [3+2]  [lane B, was R16.3b]
   * The alkene route needs a separate oxidation step that the synthon formalism does not
     represent. Should this rule be restricted to the alkyne dipolarophile, or is a two-step
     (cycloadd + oxidise) route acceptable to the planner?
   * Nitrile imines dimerise unless the dipolarophile is in excess. Does that make this a low-
     yield rule that should be downweighted relative to R16.3?
   * Hydrazonoyl chlorides are made from hydrazones with NCS/Cl2 — are they catalogued building
     blocks at Enamine, or does the port need to cut one step further back to the hydrazone?
   * Cyclooctyne dipolarophiles: the [c;R1] guard blocks a strained-cycloalkyne route to a fused
     pyrazole. Real omission or acceptable?

``R17.31`` — isoxazole / nitrile oxide + alkyne (or alkene) [3+2]  [lane B, was R16.5b]
   * Nitrile oxides are energetic and must never be accumulated. Is a rule that proposes them
     acceptable at scale, or should it carry a hard scale ceiling in the condition layer?
   * The alkene route stops at the isoxazoline and needs a separate dehydrogenation. Restrict
     this rule to alkyne dipolarophiles, or accept the two-step route?
   * Aldoxime + NCS/NaOCl is the standard dipole precursor. Are hydroximoyl chlorides catalogued,
     or must the port cut back to the aldoxime or the aldehyde?
   * 3-Alkoxy and 3-amino isoxazoles are currently allowed at C3 (the nitrile-oxide carbon). Are
     alkoxy/amino nitrile oxides real reagents, or should C3 heteroatom substitution be excluded
     here and left to R16.5?

``R17.32a`` — 1,2,4-triazole, 4H / N4-substituted / Pellizzari (acylhydrazide + amidine or amide)  [lane B, was R16.B1]
   * Bug 2 costs this rule one of its two regiochemical disconnections. Ship a mirror twin (same
     LHS, RHS labels moved to the swapped positions) or set automorphism_filter=False on the
     record? Needs an integrator decision, not a chemist one — but a chemist should confirm both
     pairings are worth having.
   * Is the free amidine or the imidate hydrochloride the realistic stocked reagent? The synthon
     is written as an H-capped secondary amine, which a chemist may not recognise as an amidine.
   * N4-amino is allowed. Confirm 4-amino-4H-1,2,4-triazoles really do come from a
     hydrazide/hydrazine amine component rather than by N-amination of the N-H ring — the guard
     choice turns on this.
   * Does an alpha-chiral acylhydrazide survive 140-160 C neat, or must this rule be flagged as
     racemising for amino-acid-derived inputs?

``R17.32b`` — 1,2,4-triazole, 4H, N-unsubstituted / Pellizzari, [n;h1] twin of R16.B1  [lane B, was R16.B2]
   * Same bug-2 decision as R16.B1: mirror twin or automorphism_filter=False. For the N-H case
     the two disconnections are the two ways of assigning which substituent came from the
     hydrazide — a chemist should confirm both are worth proposing.
   * Confirm the amidine free base (not the hydrochloride) is what is catalogued, since the rule
     emits an H-capped primary amine synthon.
   * Free-NH triazoles are strong H-bond donors and often poorly soluble; does that change which
     of the two disconnections a chemist would actually run?

``R17.33`` — 1,2,4-triazole, 1H / N1-substituted / nitrile imine + nitrile [3+2]  [lane B, was R16.B3]
   * Padwa reports this mainly for tethered/intramolecular cases and activated nitriles, not as a
     general intermolecular method. Should the rule be restricted to electron-poor nitriles (a
     !$(...) on C5's substituent), or downweighted in scoring instead?
   * The intermolecular version uses the nitrile as solvent in large excess — is that acceptable
     atom economy for a route the planner will propose, or should the rule be marked
     intramolecular-only?
   * Prefer R16.B1 (Pellizzari) for general scope? A chemist should say whether this rule earns
     its place at all, or only for the N1-regiochemistry it uniquely delivers.

``R17.34`` — 1,2,4-oxadiazole / nitrile oxide + nitrile [3+2]  [lane B, was R16.B4]
   * THE BIG ONE: the amidoxime route dominates real practice and is inexpressible, so this rule
     will propose a low-yield, high-temperature nitrile-oxide route for drugs actually made by
     Tiemann-Kruger. Is having ANY handle on the ring worth the wrong-route risk, or should the
     rule be shipped disabled-by-default with a note pointing at the amidoxime step?
   * Should the rule require an electron-poor C5 substituent (the only nitriles that trap the
     dipole efficiently)? Concretely: does an unactivated aryl nitrile like benzonitrile give
     usable yields, or never?
   * Ataluren disconnects to 2-cyanobenzoic acid + 4-fluorobenzaldoxime. Is the free carboxylic
     acid compatible with in-situ nitrile-oxide generation (NCS/Et3N), or does it need
     protection?

``R17.35`` — 1,2,4-thiadiazole / nitrile sulfide + nitrile [3+2]  [lane B, was R16.B5]
   * 190 C thermolysis with a large nitrile excess and a chlorocarbonylsulfenyl-chloride-derived
     precursor: is this ever a route a chemist would run, or should the rule ship disabled and
     exist only so the ring is not invisible?
   * The rule will happily propose an UNACTIVATED nitrile. Should C5 be constrained to bear an
     EWG, and if so what is the smallest defensible SMARTS for 'electron-poor nitrile'?
   * Is the [SH_nuc]N=[CH_elec]R fragment recognisable to a chemist as a nitrile-sulfide
     precursor, or does it read as a nonsense thiooxime? A naming/annotation question with real
     usability consequences.

``R17.36`` — 1,3,4-thiadiazole / nitrile imine + C=S dipolarophile (dithioester, thioamide) [3+2] with beta-elimination  [lane B, was R16.B6]
   * The acetazolamide disconnection emits a sulfamoyl-methanethiol synthon (O=S(=O)(CS)N). Is
     that a real, orderable building block, or is acetazolamide actually made another way
     entirely — this is the single most load-bearing example for the rule's credibility.
   * An aromatic 2,5-disubstituted product implies an N-unsubstituted nitrile imine. Is that
     reagent class practically available, or is the aromatic rule effectively unusable and only
     the N3-aryl dihydro product real?
   * Bug 2 drops one of the two substituent pairings. A chemist should say whether both readings
     (which half is the dithioester) are worth proposing, before the integrator decides on a
     mirror twin.
   * The Lawesson's/P2S5 one-pot alternative is much more common than the nitrile-imine route. Is
     the port better served by a rule that models the hydrazide + Lawesson's route instead, even
     if the cut is tautomer-blocked today?

``R17.40`` — Hantzsch thiazole, ring-opened S-vinyl thioimidate form (N3-C4)  [lane C, was R16.10]
   * THE SCOPE CALL: excluding 2-aminothiazoles removes the largest drug subclass this ring has.
     Confirm you want correctness over coverage here, or authorise the join/close_ring proton-
     migration fix that would restore it (and would also unblock two-fragment Hantzsch and
     Robinson-Gabriel).
   * The guard is broader than the failure needs: 2-(dimethylamino)thiazole has no mobile proton
     and is excluded anyway. Should it be narrowed to exclude only C2 heteroatoms bearing an H,
     i.e. !$([c]!@[#7;!H0,#8;!H0,#16;!H0])?
   * The fragment is an S-alkyl thioimidate held at product bond order, so the stock lookup sees
     a vinyl sulfide, not the thioamide + alpha-halo ketone a chemist would order. Does the
     synthon->BB file already map that, or does this rule need a leaving-group note?
   * Does the rule need to exclude a 4- or 5-nitro thiazole, where the alpha-halo ketone
     precursor would not survive the condensation?

``R17.41`` — van Leusen oxazole (TosMIC + aldehyde)  [lane C, was R16.11]
   * Confirm the C2-H constraint: is there any TosMIC-type reagent in your stock that puts a
     substituent on the ISOCYANIDE carbon rather than the alpha carbon? If so the h1 guard is too
     tight.
   * The port renders the aldehyde fragment as an alcohol ([CH2_elec2](C)[OH_nuc]) because it
     holds product bond orders, and only the C:elec2 -> '=O' leaving-group cap recovers the
     aldehyde. Confirm the synthon->BB file makes that mapping, otherwise the planner will shop
     for the wrong reagent.
   * Enolisable and hindered aldehydes give poor van Leusen yields and ketones largely fail.
     Should the rule carry a bulk/enolisability exclusion on the C5 substituent, or is that a
     scoring concern rather than a rule concern?

``R17.42`` — van Leusen imidazole (TosMIC + aldimine)  [lane C, was R16.12]
   * This rule now carries the whole N-substituted imidazole space because R16.4a is tautomer-
     broken. Is van Leusen an acceptable single answer for N-alkylimidazoles in your route data,
     or does the planner need the amidine route restored (which requires the close_ring proton-
     migration fix)?
   * Confirm the C2-H constraint against your own TosMIC reagent list, same question as R16.11.
   * The N1 substituent comes from the amine of the aldimine and the C5 substituent from the
     aldehyde. Is the regiochemistry ever reversed in practice with a hindered amine, which would
     make the rule's fixed assignment wrong?

``R17.43`` — van Leusen thiazole (TosMIC + isothiocyanate or thiocarbonyl)  [lane C, was R16.13]
   * The harvest rated this rule medium because the 5-C-substituted (thioester / dithioester /
     thiocarboxylic acid) branch could not be pinned to a review, only the isothiocyanate branch.
     Confirm whether the 5-aryl/5-alkyl case is real enough to ship, or whether the rule should
     be narrowed to 5-amino only with $([c]!@[#7]) on C5.
   * If narrowed to 5-amino only, R16.13 and R16.14 would both own 5-aminothiazoles and the
     partition would need restating. Which do you prefer?
   * The harvest's drug-relevance note claims this reaches 'the dasatinib chemotype
     (2-aminothiazole-5-carboxamide)'. It does not: dasatinib's exocyclic amine is at C2 and its
     C5 bears a carboxamide CARBON, so neither R16.13 nor R16.14 fires on it. Confirm and correct
     the rule note before it reaches a route report.

``R17.44`` — Cook-Heilbron 5-aminothiazole (alpha-aminonitrile + CS2 / isothiocyanate / dithioester)  [lane C, was R16.14]
   * The X3 on the C5 nitrogen admits a 5-NHR and a 5-NR2. Cook-Heilbron delivers a primary 5-NH2
     from the nitrile; a 5-NHR would have to be alkylated afterwards. Should the guard be
     tightened to a primary amine ($([c]!@[NH2])), or is the post-alkylation staging acceptable?
   * The exocyclic 5-amino may in practice be acylated (5-acetamido) in a real target. As written
     the rule fires on that too, because X3 admits an amide nitrogen. Is that wanted, or should
     !$([c]!@[#7][#6]=[O]) be added?
   * The synthon renders the nitrile as an enamine at product bond order and the sulfur
     electrophile as a thiol; a stock lookup must map C:elec2 -> '=O'/'=S' and C:elec -> Cl.
     Confirm those caps exist before shipping.
   * alpha-Aminonitriles come from Strecker chemistry, so HCN handling sits upstream of every use
     of this rule. Should that be surfaced as a rule-level warning in the route report?

``R17.45`` — oxazole C2-O1 cyclodehydration (acyloin + nitrile / imidoyl electrophile) — NOT Robinson-Gabriel  [lane C, was R16.15]
   * The harvest warned the enol fragment is tautomer-fragile and asked for re-validation on
     every new substitution pattern. I ran 10 more patterns (C5-H, C4-H, 5-Cl, 5-CO2H, 4-CO2Me,
     4,5-dialkyl, 4,5-diaryl) and all 10 conserved formula. Is that enough coverage, or should
     validate.labelled_atoms_survive_canonicalisation be wired in as a runtime check for this
     rule specifically?
   * The rule NAME must not say Robinson-Gabriel. Confirm the name you want, since a route report
     that swaps the two changes which reagent oxygen has to be present.
   * Conditions are stoichiometric strong acid (H2SO4/TfOH/BF3) or POCl3/TFAA. Should the rule
     carry an acid-sensitivity exclusion on the C4/C5 substituents (e.g. Boc, acetal, tert-butyl
     ester), or is that a scoring concern?

``R17.46`` — oxazole, ring-opened O-vinyl imidate form (N3-C4)  [lane C, was R16.16]
   * This rule produces one fragment and no purchasable material, so its only value is as a DAG-
     opening step. Confirm the scoring treats a one-fragment ring-opening as progress rather than
     as a completed disconnection, otherwise it will inflate pathway counts without helping.
   * The practical staging (NH4OAc/AcOH reflux on an alpha-acyloxy ketone, the Blumlein-Lewy
     manifold) is a different reagent set from the O-vinyl imidate the fragment literally spells.
     Is the imidate rendering acceptable for a stock lookup, or should the rule carry a named-
     conditions note?
   * Same narrow-the-2-amino-guard question as R16.10.

``R17.47`` — 2-oxazoline (4,5-dihydro-1,3-oxazole) from a 1,2-amino alcohol, O1-C2  [lane C, was R16.17]
   * CONFIRM THE INVERSION CLAIM: does Deoxo-Fluor / DAST / Burgess cyclodehydration of a beta-
     hydroxy amide invert the carbinol carbon (C5) while the ZnCl2 + nitrile and acid routes
     retain it? I am asserting this from the oxazoline/thiazoline natural-product literature but
     did not re-fetch the primary papers in this session. This is the single most checkable
     stereochemical claim in my scope.
   * 2-Oxazolines are very often a carboxylic-acid PROTECTING GROUP rather than a target ring.
     Should the rule be scored down, or gated, when the oxazoline is the only ring in the
     molecule and C4/C5 are unsubstituted (i.e. it is plain ethanolamine-derived)?
   * Does a free NH or OH elsewhere in the substrate survive the ZnCl2/nitrile conditions well
     enough that this rule should fire on such targets, or does it need an exclusion?
   * The azlactone (oxazol-5(4H)-one) is now excluded and owned by nobody. Is an Erlenmeyer rule
     wanted?

``R17.48`` — 2-thiazoline (4,5-dihydro-1,3-thiazole) from a 1,2-amino thiol, S1-C2  [lane C, was R16.18]
   * How badly does C4 epimerise in practice under the common cyclodehydration conditions? If it
     is routine, a forward rule must emit a mixture and the retro must not promise enantiopure
     cysteine.
   * CONFIRM the inversion-at-C5 claim for the beta-hydroxy thioamide route, same question as
     R16.17. UNVERIFIED-IN-SESSION.
   * 2-Thiazolines oxidise/aromatise to thiazoles under some conditions, so a thiazoline target
     can be over-oxidised in the forward direction. Should the rule warn, or is that scoring?
   * Thiols oxidise to disulfides; the amino-thiol fragment will need a degassed or reducing
     workup. Is that worth surfacing at rule level?

``R17.49`` — 2-imidazoline (4,5-dihydro-1H-imidazole) from a 1,2-diamine, N1-C2  [lane C, was R16.19]
   * Is the N1-acyl exclusion right? An N-acyl-2-imidazoline may in practice be made by acylating
     the imidazoline afterwards, in which case excluding it here is correct and the acylation is
     a separate R1-family step. Confirm.
   * For an unsymmetrical 1,2-diamine the rule offers both N1/N3 assignments. Should the more
     nucleophilic (less hindered, more basic) nitrogen be forced to become N3, or is offering
     both correct for a planner?
   * Clonidine is disconnected here to an N-aryl-substituted formamidine plus ethylenediamine. Is
     that the route you would run, or is the aryl guanidine / carbodiimide route the real one?
     The rule reaches the same target either way but the fragment a stock lookup sees differs.
   * The neat-acid route runs at 150-200 C. Should the rule carry a thermal-stability exclusion
     on the C4/C5 substituents?

``R17.50`` — 2-imidazoline from an amidine + 1,2-dielectrophile, N1-C5 + N3-C4  [lane C, was R16.20]
   * Is the N1-H restriction acceptable, given R16.19 covers N1-substituted imidazolines by the
     diamine route? If a chemist genuinely runs N-alkylamidine + dibromoethane, the rule cannot
     express it and the close_ring proton-migration fix is the only way in.
   * The synthon renders the dielectrophile as ethane with two C:elec labels, which
     leaving_groups caps as 1,2-dichloroethane. Confirm the stock lookup accepts
     1,2-dibromoethane, a 1,2-ditosylate, ethylene carbonate and a cyclic sulfate as equivalent,
     otherwise the rule proposes a reagent nobody uses (1,2-dibromoethane is a suspected
     carcinogen and would be avoided on scale anyway).
   * The SN2 inversion claim is textbook, but the practical question is whether a chiral
     1,2-dielectrophile is ever used here at all. If double alkylation of an amidine with an
     enantiopure cyclic sulfate is not a real route, the whole stereo_spec for this rule is moot
     and 4-substituted imidazolines should come from R16.19 instead. Please rule on this.
   * Double N-alkylation competes with mono-alkylation and polymerisation and needs high
     dilution. Should the rule be scored down relative to R16.19?

``R17.55a`` — pyrazine / self-condensation of alpha-aminoketones (Staedel-Rugheimer, Gutknecht, Gastaldi), orientation A  [lane D, was R16.10a]
   * The Staedel-Rugheimer/Gastaldi pyrazine is a DIMERISATION of one alpha-aminoketone.
     R16.10a/b return two different halves, i.e. a MIXED condensation, which in practice gives a
     statistical mixture of three pyrazines. Should the rule be restricted to symmetric
     pyrazines, or is the mixed route used at all?
   * 2-hydroxy- and 2-aminopyrazine survive only because the guards force chython to pick a
     particular mapping. If the guards are ever relaxed, the other mapping returns a broken
     synthon. Is the pyrazine ring worth this fragility, or should hydroxy/amino pyrazines be
     excluded outright to be safe?
   * Every alpha-aminoketone is prone to self-condensation on standing — the reaction is
     notoriously low-yielding and needs an oxidant to aromatise the dihydropyrazine. Should this
     rule carry a yield/feasibility penalty rather than be treated like a Suzuki?

``R17.55b`` — pyrazine / alpha-aminoketone self-condensation, orientation B (mirror twin of R16.10a)  [lane D, was R16.10b]
   * Same mixed-vs-symmetric-condensation question as R16.10a.
   * Shipping a/b twins for pyrazine doubles the disconnection count for a ring that is usually
     bought rather than built. Is the pyrazine rule worth shipping at all?

``R17.56`` — quinoxaline / o-phenylenediamine + 1,2-dicarbonyl (and pteridine by the Isay route)  [lane D, was R16.11]
   * The 1,2-dicarbonyl synthon for unsubstituted quinoxaline is [CH3_elec2][CH3_elec2] — glyoxal
     in product bond orders, only TWO heavy atoms. _accept passes it (2 heavy + 4 attachment
     points = 6 >= 3), but will the planner's stock lookup resolve it to glyoxal, or does it need
     an explicit entry?
   * R16.11 fires on pteridine (Isay route from a 4,5-diaminopyrimidine). Does a
     4,5-diaminopyrimidine survive the 1,2-dicarbonyl condensation without competing acylation at
     the ring nitrogens, and is the regiochemistry (6- vs 7-substituted pteridine) controlled?
   * Quinoxalin-2-ol is kept, returning [CH3_elec2][CH2_elec2]O — glycolaldehyde in product bond
     orders. Is a quinoxalin-2(1H)-one really made from o-phenylenediamine + glyoxal, or from an
     alpha-keto ACID/ester (which this synthon does not represent)?

``R17.57a`` — quinazoline / quinazolin-4(3H)-one, N3-H / Niementowski (anthranilic acid + amide)  [lane D, was R16.12a]
   * R16.12a fires on purine, returning a 4,5-diaminoimidazole plus formamide — that is the
     Traube purine synthesis, correct chemistry but very different conditions from Niementowski.
     Should purine get its own rule id so the condition metadata is honest?
   * The 2,4-diaminoquinazoline (prazosin) hit returns N[CH2_elec2][NH2_nuc2], i.e. guanidine in
     product bond orders, plus an anthranilonitrile-like half. Is guanidine + 2-aminobenzonitrile
     the real prazosin-core route, and does that synthon match the stocked reagent?
   * The rule relies on chython aromatising quinazolin-4(3H)-one to 4-hydroxyquinazoline. If a
     future chython changes that standardisation, R16.12a silently stops firing on the whole
     4(3H)-one class. Is a non-aromatic fallback LHS worth authoring now?

``R17.57b`` — quinazolin-4(3H)-one, N3-substituted / Niementowski via acylanthranil  [lane D, was R16.12b]
   * Under the thermal acylanthranil route (typically 120-140 C) is a 3-(2-substituted-
     aryl)quinazolin-4(3H)-one configurationally stable ON THE TIMESCALE OF THE REACTION, or does
     it racemise in the pot? The 112.7-140.8 kJ/mol barriers were measured by racemisation of
     resolved material, so the half-life at reaction temperature needs a chemist's read before
     any stereo variant is shipped.
   * Should R16.12b's N3 substituent be restricted to carbon? An N3-amino (hydrazide-derived)
     quinazolinone is a different reagent class and the current [#7;A;D3;+0] would accept it.
   * The synthon for the aryl half is an anthranilALDEHYDE (c1cccc(c1[CH_elec]=O)[NH2_nuc2])
     rather than the acylanthranil or isatoic anhydride actually used. Is that an acceptable
     stand-in for the planner, or does the rule need a leaving-group-aware cap?

``R17.58`` — quinoline / Combes (aniline + 1,3-diketone); also Skraup, Doebner-von Miller and Knorr  [lane D, was R16.13]
   * R16.13 now keeps carbostyril (Knorr, 2-OH) and refuses 4-hydroxyquinoline (Conrad-Limpach).
     Both are real, high-value chemistry. Is losing Conrad-Limpach acceptable, or should the port
     ship a 4-methoxy/4-chloro surrogate rule that a planner can chain with a
     demethylation/hydrolysis?
   * R16.13 fires on 1,5-naphthyridine, i.e. Skraup on 3-aminopyridine. Skraup on electron-poor
     anilines is notoriously low-yielding and violent. Keep, or restrict the aniline to a
     carbocyclic ring?
   * The Friedel-Crafts site is written [c_nuc:5] with no requirement that it be unsubstituted or
     activated. Should the rule demand an H or an EDG there, given Combes fails on deactivated
     anilines?

``R17.59`` — isoquinoline / Bischler-Napieralski, aromatic product (one-bond cut, C1-C8a)  [lane D, was R16.14]
   * Bischler-Napieralski on an ELECTRON-POOR arene generally fails; the papaverine 6,7-dimethoxy
     pattern is not decoration, it is a requirement. Should R16.14/R16.15 carry a positive
     requirement for an EDG ortho or para to the cyclisation carbon, and if so what is the
     fairest SMARTS cut-off?
   * R16.14 is the Bischler-Napieralski PLUS an implicit dehydrogenation (Pd/C or DDQ). Is
     shipping that as one rule acceptable, or must the planner see the oxidation as its own step
     (in which case R16.15 is the only honest B-N rule and R16.14 should be withdrawn)?
   * The rule allows any substituent at C3 and C4. Real B-N products are unsubstituted at C4
     unless the phenethylamine was. Is the extra scope harmful?

``R17.60`` — 3,4-dihydroisoquinoline / Bischler-Napieralski, the literal one-step product  [lane D, was R16.15]
   * R16.15 keeps 1-methoxy-3,4-dihydroisoquinoline (a cyclic imidate). Is that a real Bischler-
     Napieralski product — from a carbamate/carbonate — or should the C1 guard widen from
     !$([#6](=[#7])[#7]) to also exclude O?
   * Same electron-density question as R16.14: B-N needs an activated (usually 3,4-dialkoxy)
     arene. Should the rule demand an EDG ortho/para to the cyclisation carbon?
   * The rule accepts any C3/C4 substitution. Bischler-Napieralski on a C3-substituted (alpha-
     branched) phenethylamide is much slower. Worth a guard, or leave it to the scoring?

``R17.61`` — isoquinoline / Pomeranz-Fritsch (aryl aldehyde + aminoacetaldehyde acetal)  [lane D, was R16.16]
   * The classical Pomeranz-Fritsch gives isoquinoline UNSUBSTITUTED at C1; a 1-substituted
     product needs the Schlittler-Muller variant (arylmethylamine + glyoxal hemiacetal), which
     forms the SAME two-atom pair but from different reagents. R16.16 currently fires on the
     1-methyl papaverine core. Should the rule demand C1-H (h1 at position 1), with Schlittler-
     Muller split off as its own rule, or is one rule honest?
   * Pomeranz-Fritsch needs strongly acidic cyclisation (H2SO4, or the Bobbitt/Jackson
     modifications) and fails on electron-poor arenes. Same EDG-requirement question as R16.14.
   * R16.14 and R16.16 will both fire on every aromatic isoquinoline, giving the planner two
     disconnections of one bond each. Is that the intended behaviour, or should one be preferred
     by score?

``R17.62`` — 1,2,3,4-tetrahydroisoquinoline / one-bond N-acyliminium cyclisation (Pictet-Spengler surrogate)  [lane D, was R16.17]
   * THE BIG ONE: R16.17's synthon is a SECONDARY AMINE, ArCH2CH2-NH-CHR2, which is NOT a reagent
     for this reaction — the real pair is the primary amine plus the aldehyde/ketone, joined by a
     condensation the enumerator cannot express. Is a planner allowed to order that secondary
     amine at all, or must the rule be tagged so it is never treated as purchasable?
   * R16.17 fires on N-acetyl-THIQ (the genuine N-acyliminium case) AND on N-methyl-THIQ, where
     an N-alkyl iminium is far less electrophilic and the cyclisation usually needs superacid.
     Should N2 be restricted to an amide/carbamate ([#7;$([#7]C=[#8])])?
   * In the substrate classes Enamine actually stocks, is C1 racemic by default, or is there
     enough 1,3-induction from a C3 substituent that a diastereoselectivity should be asserted?
     This decides whether the stereo twins ship together (racemic) or singly.
   * Does the spiro-THIQ hit (quaternary C1 from a cyclohexanone-derived iminium) actually work?
     Ketone-derived Pictet-Spenglers are much harder than aldehyde-derived ones.

``R17.70`` — benzimidazole / Phillips (o-phenylenediamine + aldehyde or carboxylic acid), one-bond N1-C2 cut  [lane E, was R16.14]
   * The Phillips condensation with a carboxylic acid needs 4N HCl at reflux (or PPA); with an
     aldehyde it needs an oxidant (air, Na2S2O5, DDQ) because the first-formed aminal must be
     oxidised to the benzimidazole. The rule does not distinguish acid from aldehyde. Does the
     planner need that distinction to price the step?
   * Purine and adenine are claimed here (Traube from a 4,5-diaminopyrimidine + formic acid).
     Legitimate, or should the rule be restricted to a carbocyclic benzo ring so purines get
     their own rule?
   * For an UNSYMMETRICALLY substituted o-phenylenediamine, both regiochemistries of the mono-
     imine are chemically equivalent because the cyclisation does not care which NH2 condensed
     first. I pinned n3 with h0 so only one is emitted. Is one enough, or does downstream reagent
     lookup need both?
   * Does anything in the 2-position genuinely come from an aldehyde while also bearing a
     heteroatom - e.g. 2-(alkoxymethyl)benzimidazoles like omeprazole's core? Those have CH2-O,
     so the heteroatom is one atom further out and the guard correctly lets them through, but
     please confirm.

``R17.71`` — benzimidazole / amidine + 1,2-dihaloarene (benzo analogue of the shipped R16.4)  [lane E, was R16.15]
   * Double SNAr of an amidine onto a 1,2-dihaloarene needs the arene ACTIVATED (o/p-nitro, or a
     heteroarene) or a Pd/Cu catalyst. The rule claims plain 1,2-dibromobenzene. Should there be
     an activation guard, or is Buchwald-Hartwig coverage enough to justify leaving it open?
   * The benzyne reading of the same synthon (two adjacent c_elec) is chemically different from
     the dihaloarene reading. Does the leaving-group table's c:elec -> Br give the right reagent
     for both?
   * On albendazole the round trip is raw False / canonical True (bug 5, the enumerator emits the
     exocyclic-imine tautomer without standardising). Formula is conserved. Is that acceptable to
     ship, or should the rule wait for a standardize() in Enumerator._close?
   * Purines are claimed here (4,5-dihalopyrimidine + amidine). Real, but is it a route anyone
     uses at scale?

``R17.72`` — benzoxazole / o-aminophenol + aldehyde or carboxylic acid, one-bond O1-C2 cut  [lane E, was R16.16]
   * 2-aminobenzoxazole is now owned by nothing. Should I have written a rule for the BrCN route
     even though the two-bond form is C1-insertion-blocked, i.e. as a one-bond O1-C2 cut giving
     an O-aryl isourea synthon? I judged that synthon unstocked and probably tautomer-fragile,
     but you may disagree.
   * Is an alpha-chiral 2-substituent (from an alpha-chiral acid) epimerised under the usual
     cyclodehydration conditions? If yes, the stereo_spec should say the relay is unreliable
     rather than retentive.
   * Does the aldehyde route need an explicit oxidant flag (DDQ, air/Cu) to be priced correctly?
   * The rule claims aza-benzoxazoles (oxazolo-fused pyridines) via the aromatic-fusion guard.
     Are those made the same way from aminohydroxypyridines?

``R17.73`` — benzothiazole / o-aminothiophenol + aldehyde or carboxylic acid, one-bond S1-C2 cut  [lane E, was R16.17]
   * 2-aminobenzothiazole (riluzole) is now owned by nothing. Given how common that scaffold is,
     is a Hugerschoff rule worth writing even though the natural two-bond form is C1-insertion-
     blocked?
   * o-Aminothiophenols oxidise to disulfides on standing and are usually bought as the disulfide
     or the hydrochloride. Does the synthon S:nuc -> H capping recover an orderable reagent, or
     should the disulfide be the recovered form?
   * The aldehyde route needs an oxidant; the acid route needs PPA or a coupling agent. Same
     pricing question as R16.16.
   * Should 2-mercaptobenzothiazole (from aniline + CS2) be called out as a separate excluded
     class, or is it covered by the same exocyclic-heteroatom guard note?

``R17.74`` — 1H-indazole / o-fluoroaryl aldehyde or ketone + hydrazine  [lane E, was R16.20]
   * I deliberately left h0 OFF the N2 pin because chython's canonical tautomer for an N-H
     indazole is substrate-dependent (some print 1H, some 2H). That means the rule sometimes
     labels the two hydrazine nitrogens the other way round - harmless because hydrazine is
     symmetric, but please confirm you are happy with a guard that is correct only because of a
     symmetry accident.
   * N-H indazoles round-trip canonical-True but raw-False (bug 5, the same chython tautomer
     flip). Formula is conserved. Acceptable to ship?
   * The SNAr needs the aryl halide activated - o-fluoro plus the ortho carbonyl is the classic
     combination. Should the rule demand fluorine specifically rather than the generic c_elec,
     which the leaving-group table caps as Br?
   * Pyrazolo[3,4-d]pyrimidines (allopurinol-like) are claimed here. Is
     4-chloro-5-formylpyrimidine + hydrazine the route people actually use, or should heteroaryl-
     fused cases be split out?

``R17.80`` — piperidine / intramolecular C-N ring closure (reductive amination, N-alkylation, aza-Michael, Mitsunobu, hydroamination)  [lane F, was R17.1]
   * We keep the N-sulfonyl nitrogen as a legal nucleophile (N-mesylpiperidine still fires 2)
     because the bis-sulfonamide is the literature route for piperazine and diazepane, but we
     refuse the N-acyl nitrogen. Is that line right for piperidine specifically, where the
     sulfonamide route is much less common than for the two-nitrogen rings?
   * For 2,6-disubstituted piperidines closed by intramolecular reductive amination, what
     cis:trans ratio should the rule advertise, and does it invert when the intermediate is an
     N-acyliminium rather than an iminium?
   * The rule fires on bridged bicyclics (tropine, 3-azaspiro[5.5]undecane both keep 2 cuts) but
     Enumerator.ring_size() reports the SMALLEST ring the new bond closes - 5 for tropine, not 6
     - so a bicyclic whose smallest bridge is 4 or >7 is dropped without a message. Should the
     rule claim bridged systems at all, or carry an explicit warning?
   * Does this disconnection survive a free indole N-H or a free carboxylic acid elsewhere in the
     substrate under the K2CO3/MeCN N-alkylation conditions it implies, or does the rule need an
     exclusion for a competing acidic N-H?

``R17.81`` — pyrrolidine / intramolecular C-N ring closure  [lane F, was R17.2]
   * Proline still fires 2 cuts. The build-vs-buy verdict says the rule should NOT earn its place
     on anything proline-derived. Should the rule carry an exclusion for a C2
     carboxyl/carboxamide (!$([#6](-@[#7])[#6](=[#8])[#8,#7])), or is suppressing catalogue hits
     a job for the stock filter rather than the SMARTS?
   * Over-alkylation to the quaternary ammonium is the standard failure of this closure when N is
     already tertiary. Is one equivalent of base and low temperature enough that we should keep
     firing on N,N-disubstituted targets, or should the rule prefer the N-H case?
   * Does 5-exo-tet closure tolerate a 3,3-difluoro substitution pattern (we keep
     3,3-difluoropyrrolidine at 2 cuts) without elimination, or is the fluorinated precursor a
     false suggestion?

``R17.82a`` — morpholine / C-O ring closure (diethanolamine cyclodehydration, haloether or epoxide closure)  [lane F, was R17.3]
   * On an unsymmetrical morpholine one of the two reported C-O cuts requires a secondary or
     benzylic electrophile, with real SN1 and racemisation risk. Should the rule refuse the
     benzylic cut outright (!$([#6](-@[#8])[c])), or report both and let a human choose?
   * The unprotected secondary amine in the acyclic precursor competes for the electrophilic
     carbon and gives piperazine/aziridinium chemistry. The rule proposes the free amino alcohol.
     Should it instead be flagged as requiring N-protection, and if so which protecting group
     does a CRO default to here?
   * Does 2-arylmorpholine formation by intramolecular epoxide opening reliably go 6-exo, or does
     5-exo (oxetane/THF) compete enough that the disconnection is misleading?

``R17.82b`` — morpholine / C-N ring closure  [lane F, was R17.3b]
   * Aziridinium formation from the beta-haloamine precursor scrambles C2/C3 substitution in
     reality and the model cannot see it. Is that a reason to refuse the cut where the
     electrophile is beta to the ring oxygen, or only a reason to warn?
   * Intramolecular O-alkylation (oxetane/THF formation) competes when the amine is protected. At
     what ring size does that competition become the major pathway, i.e. is the 6-exo C-N closure
     reliably preferred?
   * Is the free 2-(2-haloethoxy)ethylamine a proposable reagent at CRO scale, or must the rule
     always hand over the N-protected form?

``R17.83`` — morpholine / 1,2-amino alcohol + 1,2-bis-electrophile (two bonds, on two DIFFERENT heteroatoms)  [lane F, was R17.4]
   * The regiochemical over-generation is visible in the test output: cut#0 rebuilds to the
     target AND to the 2-aryl/3-aryl regioisomer. Real chemistry is regiodefined (styrene oxide
     opens at the terminal CH2, chloroacetyl chloride acylates N first). Should the rule emit
     only the chemically-controlled regiochemistry, and if so on what rule can that be decided
     without knowing the reagent?
   * The bis-electrophile fragment prints as [CH3_elec][CH3_elec] - two heavy atoms. Is that
     legible enough to a CRO as 'a two-carbon 1,2-bis-electrophile: dibromoethane, ethylene
     sulfate, ethylene ditosylate, or an epoxide', or does it need a rule-specific reagent
     annotation?
   * For a 2-aryl morpholine built from an enantiopure amino alcohol, is any epimerisation
     observed at the carbinol under the KOtBu/NaH closure, or is retention effectively
     quantitative?

``R17.84a`` — piperazine / 1,2-diamine + 1,2-bis-electrophile (two bonds, on two DIFFERENT nitrogens)  [lane F, was R17.5]
   * We keep N-sulfonyl and N-aryl nitrogens as nucleophiles but refuse N-Boc. The literature
     route protects with sulfonyl precisely because sulfonamide anions alkylate; is a Boc-
     piperazine nitrogen ever a practical ring-closing nucleophile, or is refusing it
     unambiguously right?
   * For a very electron-poor N-heteroaryl piperazine (2-pyrimidinyl, 2-pyridyl), is that
     nitrogen still a usable nucleophile for the ring-closing alkylation, or should the rule
     exclude N-(electron-poor heteroaryl)?
   * For 2,5- and 2,6-disubstituted piperazines built from an amino-acid-derived diamine, does
     the cis/trans relationship come from the diamine alone, or does the double alkylation set
     the second centre?

``R17.84b`` — piperazine / intramolecular C-N ring closure (one bond)  [lane F, was R17.5b]
   * Four cuts are reported on a symmetric target and two of them are duplicates by symmetry.
     Should the rule deduplicate on the fragment string, or is reporting all four traversals
     wanted for asymmetric rings?
   * The linear triamine precursor oligomerises without high dilution or N-protection. Is
     proposing the free triamine acceptable, or must the rule always hand over the bis-
     sulfonamide?
   * For a 2-substituted piperazine, does the C-N closure at the substituted carbon go cleanly
     SN2 with inversion, or does the aziridinium intermediate dominate enough that the
     disconnection is unreliable there?

``R17.85a`` — thiomorpholine / 1,2-amino thiol + 1,2-bis-electrophile (two bonds)  [lane F, was R17.6]
   * SAFETY: the linear intermediate on this disconnection is a MUSTARD. Should the rule refuse
     to emit the batch route entirely and only ever propose the telescoped/flow form, and is that
     a rule flag or a report-layer flag?
   * Thiolate S-alkylation always outruns N-alkylation, so with an unsymmetrical bis-electrophile
     only one of the two reported regiochemistries is real. Can the rule encode 'S attacks the
     less hindered carbon' or must a human filter?
   * Thiol oxidation to the disulfide is the standard yield killer here. Does that make the free
     amino thiol an unacceptable reagent to propose, or is degassing sufficient?

``R17.85b`` — thiomorpholine / C-S ring closure (one bond)  [lane F, was R17.6b]
   * The rule now refuses thiomorpholine 1,1-dioxide entirely on the sulfur side, which is
     chemically right, but the dioxide is far more common in drugs than the sulfide. Is the
     correct answer a separate FGI rule (sulfide -> sulfone) rather than a ring rule, and does
     the port have anywhere to put it?
   * Same mustard safety flag as R17.6 when the precursor is a beta-chloroethyl sulfide. Should
     the one-bond rule inherit that flag?
   * On an unsymmetrical thiomorpholine one C-S cut needs a secondary or benzylic electrophile.
     Is thiolate nucleophilicity high enough that SN2 still wins there, or does racemisation make
     that cut unusable?

``R17.86a`` — 1,4-diazepane (homopiperazine) / 1,2-diamine + 1,3-bis-electrophile (two bonds)  [lane F, was R17.7]
   * 7-exo-tet is entropically disfavoured and oligomerisation is the competing pathway. Should
     the rule be gated behind a high-dilution warning, or is the bis-sulfonamide activation
     enough that a CRO can run it as written?
   * The test target is fasudil, whose sulfonamide nitrogen we treat as a nucleophile. Is that
     right for the ring-CLOSING step, or does the real route sulfonylate after closure?
   * A substituted 1,3-bis-electrophile would set the bridge configuration, but this rule's caps
     erase it. Is the bridge-substituted homopiperazine common enough in practice to need the
     one-bond rule R17.7b instead, or is it a non-case?

``R17.86b`` — 1,4-diazepane / intramolecular C-N ring closure (one bond)  [lane F, was R17.7b]
   * All four cuts are reported as equally plausible, but 7-exo-tet closures differ enormously in
     rate depending on which bond you form. Which of the four is the one a CRO would actually run
     on a C-substituted homopiperazine?
   * The rule treats the aryl-sulfonamide nitrogen of fasudil as a nucleophile. Correct for the
     Ts/Ns-protected synthetic intermediate; is it still correct when the sulfonyl group is the
     final aryl sulfonyl of the drug, i.e. would anyone install that before closing the ring?
   * Does the free (unprotected) linear triamine ever close usefully at high dilution, or must
     the rule always require N-activation?

``R17.87`` — delta-lactam (piperidin-2-one) / amino acid or amino ester lactamisation  [lane F, was R17.9]
   * Imides (glutarimide, thalidomide's ring) currently have NO owner: the all-z1 pin refuses
     them here and nothing else claims them. Cyclodehydration of a glutaramic acid to a
     glutarimide is a real reaction - does it deserve its own id, and if so is the amide-nitrogen
     nucleophile acceptable there when we refuse it everywhere else?
   * The printed synthon MISLEADS: the H-cap makes the fragment read as an ALDEHYDE
     (...CC[CH_elec]=O, i.e. 5-aminopentanal) while the reagent to order is the amino ACID or
     ESTER. Is an electrophile cap of OH or OEt the right fix, and which one would a CRO rather
     see?
   * For a C3-substituted delta-lactam, how much epimerisation alpha to the carbonyl is actually
     seen under thermal cyclisation, and is the coupling-agent route reliably retentive?
   * A piperidin-2-one is not a piperidine - reduction (LiAlH4, BH3.THF, Red-Al) is an FGI
     outside the formalism. Should the rule's report say so explicitly every time, or is that a
     planner-level concern?

``R17.88`` — gamma-lactam (pyrrolidin-2-one) / amino acid or amino ester lactamisation  [lane F, was R17.10]
   * Same aldehyde mis-cap as R17.9: the printed synthon is a 4-aminobutanal, the reagent is the
     gamma-amino acid/ester. Confirm that OH or OEt is the cap a CRO wants.
   * For (R)-rolipram specifically, the literature route makes the gamma-nitro ester by
     asymmetric Michael addition and then closes on reduction. Does the rule need to say that the
     closure step is stereochemically silent and the enantiocontrol lives entirely in the Michael
     step?
   * Are there gamma-lactams whose C3 substituent survives thermal cyclisation without
     epimerisation, i.e. is the alpha-epimerisation flag substrate-dependent enough to need a
     substructure test rather than a blanket warning?

``R17.89`` — 1,2,3,6-tetrahydropyridine / ring-closing metathesis of an N-tethered diene  [lane F, was R17.11]
   * The synthon MIS-CAPS both olefin termini: the fragment prints as a saturated chain while the
     real reagent is the diene, one carbon longer at each end, with those two carbons leaving as
     ethylene. The port already accepts this abstraction for R11.2. Is it acceptable to a CRO
     here, or does the RCM rule need its own reagent annotation?
   * A free basic amine poisons Ru, so the substrate must be N-protected, but the target is often
     the free amine. Should the rule refuse the free-NH target (currently kept, 1 match) and
     force the planner to route through the carbamate, or report it with a protection note?
   * Should the rule also cover the 5-ring (3-pyrroline from diallylamine, an Organic Syntheses
     prep) and the 7-ring, which the current pin excludes by atom count? If so, is one rule per
     ring size right, or one rule with a size-agnostic pin?
   * For a substrate with a 1,1-disubstituted olefin terminus, Grubbs II is slow or fails. Is
     that common enough in this ring family to justify an exclusion?

``R17.90`` — morpholin-3-one / 1,2-amino alcohol + haloacetate (the reducible morpholine precursor)  [lane F, was R17.13]
   * This rule cuts the AMIDE bond and hands over the O-alkylated amino ester, while R17.G cuts
     the O-C bond of the same molecule and hands over the chloroacetamide. The real route runs
     both steps in that order. Should the two be linked so a planner reports them as one two-step
     sequence rather than two independent disconnections?
   * Aldehyde mis-cap again: the printed -OCH2CHO should read as the glycolate ester, the
     chloroacetate, or the acid chloride. Which of the three does a CRO want to see?
   * The rule ends at the morpholin-3-one; getting to the morpholine needs BH3 or LiAlH4. Is it
     acceptable to present a lactam as a route to a saturated amine, given the formalism cannot
     express the reduction?

``R17.91`` — piperazin-2-one / 1,2-diamine + haloacetate (the reducible piperazine precursor)  [lane F, was R17.14]
   * Chemoselectivity between the two nitrogens of an UNSYMMETRICAL diamine requires orthogonal
     protection; the tool proposes the unprotected diamine, which in practice bis-acylates.
     Should the rule refuse to fire when both nitrogens are free N-H, or is that too aggressive?
   * Which asymmetric catalytic one-pot entries to C3-stereodefined piperazin-2-ones are actually
     practical at CRO scale, and what enantioselectivity should the rule advertise?
   * Aldehyde mis-cap: the printed fragment should read as the haloacetate or the glycine
     derivative. Same question as R17.9/R17.10/R17.13.

``R17.92`` — piperidine / aza-annulation: amine-bearing C-nucleophile + 1,3-bis-electrophile (two bonds, N-C and C-C)  [lane F, was R17.15]
   * The carbanion-stabiliser guard now allows a BENZYLIC centre. Is a benzylic anion basic
     enough for a practical intramolecular alkylation at CRO scale, or should aryl be dropped
     from the OR-list leaving only carbonyl, nitrile, nitro and sulfone?
   * The harvest's own verdict is 'ship this one LAST, or not at all'. With the stabiliser guard
     in place the noise is measurably down (3-methylpiperidine 2 -> 0). Is that enough to justify
     shipping, or is the honest answer still to leave it out of the released set?
   * For the Mannich/Michael cascade route, does the 2,6-cis preference hold generally, and is
     there an asymmetric catalytic version a CRO would actually run?

``R17.93`` — GENERIC saturated heterocycle / one-bond heteroatom-carbon ring closure (the declared residual)  [lane F, was R17.G]
   * Benzo-fused saturated heterocycles (THIQ, tetrahydroquinoline, indoline, chromane) are
     claimed by this rule today and partly claimed by the benzo-fused harvest family. Who owns
     them, and if the benzo family takes THIQ, does anyone want chromane and dihydrobenzofuran?
   * Intramolecular N-alkylation of a lactam/sulfonamide nitrogen to close a ring is now refused
     everywhere in family F. It is a real reaction (upstream R6.1 owns the acyclic version). Is a
     dedicated ring rule worth adding, and would a chemist expect it for benzolactam-type
     targets?
   * The rule now claims 4-piperidone, 3-piperidone and the tetrahydropyridine C-N bonds because
     the named rules' all-z1 pins refuse them. Is 'the generic rule owns the ring-ketone
     variants' an acceptable report to a CRO, or should R17.1's pin be relaxed at positions 3-5
     so the named rule keeps them?
   * The rule cannot name its reaction and therefore cannot state a retention/inversion constant.
     Is reporting the disconnection with an explicitly UNDETERMINED stereochemical outcome
     useful, or actively misleading?

``DROPPED:E:R16.8`` — R16.8 indole / Fischer (arylhydrazine + enolisable ketone)  [lane E, was R16.8]
   * Should the rule refuse 2,3-UNSUBSTITUTED indoles? Fischer with acetaldehyde is notoriously
     low-yielding and indole itself is made industrially from aniline + ethylene glycol. I
     deliberately did NOT encode this because it is a yield boundary rather than a scope
     boundary, and the brief says silently dropping a legitimate hit is worse than no guard. Do
     you want [c;!$([c;h1]:[c;h1]):2] added?
   * Is 7-azaindole (and azaindoles generally) legitimately Fischer? The rule currently claims
     them because the fusion guard only asks that the fused ring be aromatic, not carbocyclic.
     Fischer on 2-hydrazinopyridines is known but poor. Should the fusion carbons be pinned to a
     carbocyclic ring instead?
   * Does the Borsche-Drechsel route (cyclohexanone phenylhydrazone) to FULLY AROMATIC carbazole
     need its own rule? It is excluded here because the ring synthon carries product bond orders,
     so the 'cyclohexanone' collapses onto the same benzyne synthon R16.23 already emits. I
     judged that a duplicate, not a gap.
   * Is the meta-substituted-arylhydrazine regiochemical ambiguity (4- vs 6-substituted indole)
     something the planner should be warned about at rule level, or is it acceptable to leave it
     to the chemist reading the route?

``DROPPED:E:R16.10`` — R16.10 indole / Leimgruber-Batcho (and Cadogan-Sundberg from an o-nitrostyrene)  [lane E, was R16.10]
   * The rule fuses two reactions under one id: Leimgruber-Batcho proper (o-nitrotoluene + DMF-
     DMA, always gives C2-H) and Cadogan-Sundberg (o-nitrostyrene, C2 can be substituted). Do you
     want them split as R16.10a ([c;h1:2], the true L-B) and R16.10b (the Cadogan), so that the
     reagent class recovered downstream is unambiguous?
   * The synthon is the reduced o-AMINOstyrene, not the o-nitrostyrene actually bought. Is that
     the right level of abstraction for reagent lookup here, or should the nitro form be
     recovered by a leaving-group rule?
   * Is the E/Z-indifference claim for 2-nitrostilbene Cadogan cyclisation solid enough to encode
     as 'geometry undefined', or do you know substrates where only one isomer cyclises?
   * Should tetrahydrocarbazole really be claimed here? 2-(cyclohex-1-enyl)nitrobenzene reductive
     cyclisation is real, but Fischer (R16.8) is how anyone actually makes it. Both rules
     currently claim it.

``DROPPED:E:R16.11`` — R16.11 indole / Larock heteroannulation (o-haloaniline + internal alkyne; Bartoli maps to the same two bonds)  [lane E, was R16.11]
   * Is [c;!h1:2] too strict? Larock with a TMS-alkyne followed by protodesilylation is a
     standard workaround to 3-substituted-2-H indoles, but that is two steps and I judged it
     outside a one-step disconnection. Do you agree?
   * Larock is regioselective, not regiospecific - with two similarly sized alkyne substituents
     the 2,3-selectivity degrades. The rule emits only one regiochemistry per mapping. Should it
     emit both?
   * The classic Larock needs the aniline N-H free or as N-Ts/N-Ac. The rule claims N-alkyl
     indoles too (N-alkyl-o-haloanilines are competent but slower). Is that acceptable?
   * Bartoli is folded into this rule because it forms the same two bonds, but its reagent is a
     nitroarene plus 3 equivalents of vinyl Grignard and it requires an ORTHO substituent on the
     nitroarene to work at all. Should Bartoli be split out with an ortho-substitution guard?

``DROPPED:E:R16.12`` — R16.12 indole / Madelung (with the Houlihan and Smith variants)  [lane E, was R16.12]
   * Classical Madelung (NaNH2, 250 C) tolerates almost nothing. Should the rule carry a base-
     sensitivity exclusion - no ester, no nitrile, no nitro, no ketone - on the ground that it
     will simply not work, or is that a conditions matter for the route scorer rather than the
     rule?
   * The rule currently claims 3-substituted and 2,3-unsubstituted indoles. Madelung from a
     formanilide (giving indole itself) is reported but poor. Restrict to C2-substituted?
   * Is the Houlihan variant close enough to share the id, or does its much wider functional-
     group tolerance deserve a separate rule so the planner can prefer it?

``DROPPED:E:R16.13`` — R16.13 indole-2-carboxylate / Hemetsberger-Knittel (alpha-azidoacrylate thermolysis)  [lane E, was R16.13]
   * Is the EWG list right? I allowed any exocyclic C=O (ester, amide, ketone) and C#N. The
     literature substrate is overwhelmingly the ESTER. Should ketone and nitrile be dropped, or
     are 2-acylindoles and indole-2-carbonitriles genuinely accessible this way?
   * Should the E/Z-convergence claim for azidocinnamate thermolysis be checked? If only one
     geometry cyclises efficiently, the synthon needs a geometry after all and this is the one
     rule in my scope where smirks_stereo would differ from smirks_after.
   * Hemetsberger-Knittel needs 130-160 C in a high-boiling solvent (or photolysis). Any
     substrate class you would rule out on thermal grounds?
   * Does the reaction tolerate an electron-poor arene? The nitrene inserts into an arene C-H, so
     a strongly deactivated ring may fail. Worth a guard on the benzo ring?

``DROPPED:E:R16.18`` — R16.18 benzofuran / 5-endo-dig cycloisomerisation of a 2-alkynylphenol  [lane E, was R16.18]
   * Should dibenzofuran get its own rule (the oxygen analogue of R16.23)? It is currently owned
     by nothing in the port.
   * The synthon is an alkene where the reagent is an alkyne. Does the reagent-recovery layer
     know to look up the 2-alkynylphenol rather than the 2-alkenylphenol? If not, this rule will
     fail reagent lookup even though the disconnection is right.
   * 3-Acyl benzofurans (benzbromarone, amiodarone) are usually made by acylating the preformed
     benzofuran, not by cyclising a 2-alkynylphenol that already carries the ketone. Does the
     rule propose an order of operations that no one would actually run?
   * Is the 2-amino/2-alkoxy exclusion right, or are 2-aminobenzofurans accessible from an
     ynamide often enough to be worth owning?

``DROPPED:E:R16.19`` — R16.19 benzo[b]thiophene / 5-endo-dig cycloisomerisation of a 2-alkynylthiophenol  [lane E, was R16.19]
   * Should dibenzothiophene get its own rule (the sulfur analogue of R16.23)? Currently unowned.
   * The rule also fires on 4H-thieno[3,2-b]indole (a thiophene fused to a pyrrole), giving a
     formula-conserving round trip. Is a 2-alkynyl-pyrrolyl-thiol a sane reagent, or should the
     fused partner be restricted to a carbocyclic or six-membered arene?
   * Free thiophenols are malodorous and air-sensitive; the practical reagent is usually the
     thioacetate or the disulfide. Same reagent-recovery question as R16.17.
   * Raloxifene's actual manufacture builds the benzothiophene by a different route (thioanisole
     + alpha-bromoketone, then Friedel-Crafts). Does that route need its own rule, or is the
     alkyne disconnection close enough for planning?

``DROPPED:E:R16.21`` — R16.21 oxindole (indolin-2-one) / intramolecular lactamisation of a 2-aminophenylacetic acid  [lane E, was R16.21]
   * Is C3 epimerisation under lactamisation conditions bad enough that the stereo_spec should
     say 'relayed, but not reliably' rather than 'relayed with retention'? This is the single
     most consequential stereochemical question in my scope and I could not settle it from
     general knowledge.
   * The rule claims isatin (C3 is a ketone), i.e. it proposes lactamising
     2-(2-aminophenyl)-2-oxoacetic acid. Chemically plausible but nobody makes isatin that way
     (Sandmeyer, or Stolle - R16.22). Should isatin be excluded here and left to R16.22?
   * It also claims 3-hydroxyoxindole (from a 2-aminomandelic acid). Real, or should 3-OH be
     excluded the way it is in R16.22?
   * Does a 3-alkylidene oxindole really survive the lactamisation step, or is the alkylidene
     always installed after the ring is closed? If the latter, this rule proposes a sequence
     nobody runs and the alkylidene case should be handed to a Knoevenagel rule first.

``DROPPED:E:R16.22`` — R16.22 oxindole / Stolle intramolecular Friedel-Crafts of a 2-haloacetanilide  [lane E, was R16.22]
   * Stolle needs AlCl3 at 150-200 C in a melt or high-boiling solvent. Should base- or Lewis-
     acid-sensitive substrates be excluded outright, or is that the route scorer's job?
   * The Friedel-Crafts attacks the arene ORTHO to the amide. A meta-substituted anilide gives
     two regioisomers and a strongly deactivated arene (nitro, multiple halogens) may not cyclise
     at all. Should the benzo ring carry an electron-density guard?
   * I kept isatin deliberately (Stolle's oxalyl chloride route). Is that the right call, and
     does the C/F/elec label with leaving group Cl recover oxalyl chloride correctly, or does it
     recover chloroacetyl chloride and silently propose the wrong reagent?
   * 3-spiro oxindoles are claimed (Friedel-Crafts of a 1-halocycloalkanecarboxanilide, i.e.
     through a tertiary cation). Does that actually work, or does the tertiary halide just
     eliminate?

``DROPPED:E:R16.23`` — R16.23 carbazole / Cadogan (2-nitrobiphenyl + P(OEt)3) or Buchwald C-N of a 2-aminobiphenyl  [lane E, was R16.23]
   * BUG 2 HITS THIS RULE and I could not fix it honestly. The LHS ring
     [n]1[c;R2][c;R2][c;R2][c;R2]1 is mirror-symmetric while the RHS is not, so chython's
     automorphism filter (which dedupes on the SET of matched atoms) keeps only one of the two
     genuinely different disconnections on an UNSYMMETRICAL carbazole. Measured on
     3-methoxycarbazole: automorphism_filter=True gives 1 hit, False gives 2, and the two
     fragments differ (aminate from the methoxy-bearing ring, or from the plain ring). There is
     no chemically honest way to break the LHS symmetry. Do you want the caller to run this rule
     with automorphism_filter=False?
   * Cadogan (P(OEt)3, 150 C) and Buchwald-Hartwig on a 2-halo-2'-aminobiphenyl are different
     reactions with different tolerances sharing one id. Worth splitting?
   * The rule claims beta-carbolines (harmine) and ellipticine. Cadogan on the corresponding
     nitro-biaryl is plausible but those alkaloids are made by Pictet-Spengler or Diels-Alder
     routes. Is claiming them useful or noise?
   * It also claims 4H-thieno[3,2-b]indole via the aromatic-fusion guard. Is a
     2-amino-3-(2-halophenyl)thiophene a sane reagent?

