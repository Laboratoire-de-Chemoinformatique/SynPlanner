"""Building-block synthonisation: 147 rule programs, 4 execution strategies, the keepPG policy."""

from dataclasses import dataclass

from chython import smarts, smiles, synthon_smiles
from chython.containers import MoleculeContainer, SynthonContainer

from synplan.chem.utils import safe_canonicalization
from synplan.enumeration.synthon.classify import BBClassifier, SynthonDataError
from synplan.enumeration.synthon.config import SynthonConfig, load_data
from synplan.enumeration.synthon.reactor import SynthonTransformer

# solvents and counterions dropped from a multi-component input, canonicalised once at import
# (the reference re-canonicalises this list on every call)
_SOLVENTS = frozenset(
    str(safe_canonicalization(smiles(s)))
    for s in (
        "OC(=O)C(=O)O",
        "CC(=O)O",
        "OS(=O)(=O)O",
        "[O-]Cl(=O)(=O)=O",
        "OP(=O)(O)O",
        "OC(=O)C(F)(F)F",
        "OS(=O)(=O)C(F)(F)F",
        "OC(=O)O",
        "[O-]S(=O)(=O)C(F)(F)F",
        "OC=O",
        r"OC(=O)/C=C\C(=O)O",
        "[O-]C(=O)C(F)(F)F",
        "OC(=O)/C=C/C(=O)O",
    )
)
# a 4-component ester mixture the reference rejects outright
_JUNK_MIXTURE = "[#6]-[#6]-[#8]-[#6].[#6]-[#8]-[#6](-[#6])=O.[#6]-[#8]-[#6](-[#6])=O.[#6]-[#8]-[#6](-[#6])=O"
# a poly-functional class whose second functionality is one of these IS a protecting group, so
# the protected intermediate can be thrown away
_PG_TOKENS = ("Nboc", "Ncbz", "Nfmoc", "Ester", "TFAc")
# ponytail: one unexplained upstream exemption from the mono-class skip; kept verbatim because
# no reason is recorded anywhere. Upgrade path: ask a chemist, then delete or generalise.
_MONO_SKIP_EXEMPT = frozenset({"Bifunctional_NbnDi_Amines"})

_AZOLE = "[nH;r5;!$(nc=O)]"
_AZOLE_RULE = "[n;H1;r5:1]>>[n_nuc:1]"
_BIACID_QUERY = (
    "[O;$([O]=[C]([#6])[OD1])].[O;$([O]([CH3])[C]([#6])=[O]),$([O]([CH2][CH3])[C]([#6])=[O]),"
    "$([O]([CH2]c1[cH][cH][cH][cH][cH]1)[C]([#6])=[O]),"
    "$([O]([C]([CH3])([CH3])[CH3])[C]([#6])=[O]),$([O]([CH2][CH]=[CH2])[C]([#6])=[O])]"
)
_BIACID_RULE = (
    "[O;$([O]([C])[C]([#6])=[O]):1][C;$([CH3]),$([CH2][CH3]),$([CH2]c1[cH][cH][cH][cH][cH]1),"
    "$([C]([CH3])([CH3])[CH3]),$([CH2][CH]=[CH2]):2]>>[OH;+0:1]"
)
# the Esters_Esters fallback the reference hard-codes inline when nothing else fired
_ESTER_FALLBACK = (
    "[C;$([C](=[O])[#6]):1][O:2]>>[C;+0_elec:1]",
    "[O;!R;$([O]([C](=[O])[#6])[$([CX4]),$([c])]):1][C;$([C](=[O])):2]>>[O;+0_nuc:1]",
)

Synthons = dict[str, set[str]]


@dataclass(frozen=True, slots=True)
class Step:
    """One `|`-separated step of a rule program. `variants` are its alternative labellings."""

    variants: tuple[SynthonTransformer, ...]
    is_pg_removal: bool
    keeps_protected: bool


@dataclass(frozen=True, slots=True)
class Program:
    name: str
    strategy: str
    func: int
    first_as_prep: bool
    steps: tuple[Step, ...]


def _parse(smi: str) -> SynthonContainer:
    return safe_canonicalization(synthon_smiles(smi))


def _key(product: SynthonContainer) -> str:
    """Raw transformer output is not canonical, and `Fragmenter` only ever looks up canonical
    synthons, so a raw key is an entry the stock can never serve."""
    return str(safe_canonicalization(product))


def _merge(into: Synthons, other: Synthons) -> Synthons:
    for key, classes in other.items():
        into.setdefault(key, set()).update(classes)
    return into


def _apply(step: Step, molecule: SynthonContainer) -> list[SynthonContainer]:
    return [product for rule in step.variants for product in rule(molecule)]


def _keep(product: SynthonContainer, before: int) -> bool:
    """A step must ADD a label when its input already carried one."""
    return not before or len(product.synthon_labels) > before


def _one_azole_nucleophile(product: SynthonContainer) -> bool:
    """An azole ring may give up only one N-H. chython's tautomer standardisation moves that H off
    the labelled nitrogen, so `[n;H1;r5]` matches the same site a second time and the double label
    reassembles to a quaternary N."""
    labelled = {
        n for n in product.synthon_labels if product.atom(n).atomic_symbol == "N"
    }
    return all(len(labelled.intersection(ring)) < 2 for ring in product.sssr)


def _normal(
    program: Program,
    steps: tuple[Step, ...],
    molecule: SynthonContainer,
    classes: set[str],
    func: int,
    used: frozenset[int] = frozenset(),
) -> Synthons:
    """Each step applied independently; `func` controls how deep the composition goes.

    Two upstream defects are fixed here: the recursion followed `labledSynthon[0]` — an arbitrary
    hash-order pick — and `usedInds` was a mutable default shared sideways across sibling branches,
    so a step consumed in one branch was blocked for every later sibling.
    """
    out: Synthons = {}
    before = len(molecule.synthon_labels)
    for index, step in enumerate(steps):
        if index in used:
            continue
        for product in _apply(step, molecule):
            if not _keep(product, before):
                continue
            out.setdefault(_key(product), set()).update(classes | {program.name})
            if func > 1:
                _merge(
                    out,
                    _normal(program, steps, product, classes, func - 1, used | {index}),
                )
    return out


def _first_as_prep(
    program: Program,
    steps: tuple[Step, ...],
    molecule: SynthonContainer,
    classes: set[str],
    func: int,
) -> Synthons:
    """Step 0 is a preparation transform whose product feeds `_normal`; falls through when it
    does not fire."""
    products = _apply(steps[0], molecule)
    if not products:
        if len(steps) > 1:
            return _first_as_prep(program, steps[1:], molecule, classes, func)
        return {str(molecule): set(classes)}
    prepared: Synthons = {}
    for product in products:
        prepared.setdefault(_key(product), set()).update(classes | {program.name})
    if len(steps) == 1:
        return prepared
    out = dict(prepared)
    for key in prepared:
        _merge(out, _normal(program, steps[1:], _parse(key), classes, max(func - 1, 1)))
    return out


def _polymer(
    program: Program,
    steps: tuple[Step, ...],
    molecule: SynthonContainer,
    classes: set[str],
    func: int,
) -> Synthons:
    """One rule applied to fixpoint: strips all N copies of an N-fold repeated group, then emits
    only the terminal molecules."""
    out: Synthons = {}
    for step in steps:
        frontier = {str(molecule): set(classes)}
        seen: set[str] = set()
        while frontier:
            nxt: Synthons = {}
            for key, inherited in frontier.items():
                if key in seen:
                    continue
                seen.add(key)
                parent = _parse(key)
                products = [
                    p
                    for p in _apply(step, parent)
                    if _keep(p, len(parent.synthon_labels))
                ]
                if not products:
                    if key != str(
                        molecule
                    ):  # a terminal molecule, not the untouched input
                        out.setdefault(key, set()).update(inherited | {program.name})
                    continue
                for product in products:
                    nxt.setdefault(_key(product), set()).update(
                        inherited | {program.name}
                    )
            frontier = nxt
    return out


def _protecting_group(
    program: Program,
    steps: tuple[Step, ...],
    molecule: SynthonContainer,
    classes: set[str],
    func: int,
    keep_pg: bool,
) -> Synthons:
    """Up to three stages separated by the protecting-group removal steps: label, deprotect,
    label, deprotect, label."""
    sections, current = [], []
    for step in steps:
        if step.is_pg_removal:
            sections.append((tuple(current), step))
            current = []
        else:
            current.append(step)
    sections.append((tuple(current), None))

    first, removal = sections[0]
    two_pgs = len(sections) == 3
    if program.first_as_prep:
        staged = _first_as_prep(program, first, molecule, classes, func)
    elif (
        "Ester" in program.name and "Acid" in program.name and func == 3 and not two_pgs
    ):
        staged = _normal(program, first, molecule, classes, 3)
    elif func == 2 or (func == 3 and two_pgs):
        staged = _normal(program, first, molecule, classes, 1)
    else:
        staged = _normal(program, first, molecule, classes, 2)

    out: Synthons = {}
    if removal is None:
        return staged
    if keep_pg or removal.keeps_protected:
        _merge(out, staged)

    without_pg: Synthons = {}
    for key, inherited in staged.items():
        parent = _parse(key)
        # only deprotect synthons where every unprotected group already became a synthon
        if (
            func == 3
            and not two_pgs
            and len(parent.synthon_labels) < 2
            and "Ester" not in program.name
            and "AlkyneCH" not in program.name
        ):
            continue
        for product in _apply(removal, parent):
            without_pg.setdefault(_key(product), set()).update(
                inherited | {program.name}
            )

    second, second_removal = sections[1]
    between = dict(without_pg)
    for key, inherited in without_pg.items():
        _merge(between, _normal(program, second, _parse(key), inherited, 1))

    # upstream re-points `PGremovalRule` at the SECOND removal before this test
    if not two_pgs or keep_pg or second_removal.keeps_protected:
        _merge(out, without_pg)
        _merge(out, between)
    if not two_pgs:
        return out

    without_two: Synthons = {}
    for key, inherited in between.items():
        for product in _apply(second_removal, _parse(key)):
            without_two.setdefault(_key(product), set()).update(
                inherited | {program.name}
            )
    _merge(out, without_two)
    last, _ = sections[2]
    for key, inherited in without_two.items():
        _merge(out, _normal(program, last, _parse(key), inherited, 1))
    return out


class BBSynthoniser:
    """Turns one building block into its synthons, keyed on the labelled canonical SMILES."""

    def __init__(
        self,
        config: SynthonConfig | None = None,
        classifier: BBClassifier | None = None,
    ) -> None:
        self.config = config or SynthonConfig()
        self.classifier = classifier or BBClassifier(self.config)
        self.programs = {p.name: p for p in self._load()}
        self._junk = smarts(_JUNK_MIXTURE)
        self._azole = smarts(_AZOLE)
        self._azole_rule = SynthonTransformer.from_smarts(_AZOLE_RULE)
        self._biacid = smarts(_BIACID_QUERY)
        self._biacid_rule = SynthonTransformer.from_smarts(_BIACID_RULE)
        self._ester_fallback = tuple(
            SynthonTransformer.from_smarts(r) for r in _ESTER_FALLBACK
        )

    def _load(self) -> list[Program]:
        records = load_data(self.config.marks_path)
        if len(records) != 147:
            raise SynthonDataError(f"{len(records)} rule programs, expected 147")
        out = []
        for record in records:
            steps = tuple(
                Step(
                    tuple(SynthonTransformer.from_smarts(v) for v in step["variants"]),
                    step["is_pg_removal"],
                    step["keeps_protected"],
                )
                for step in record["steps"]
            )
            out.append(
                Program(
                    record["name"],
                    record["strategy"],
                    record["func"],
                    record["first_as_prep"],
                    steps,
                )
            )
        return out

    def synthonise(
        self, molecule: MoleculeContainer, classes: list[str] | None = None
    ) -> tuple[Synthons, bool]:
        """(synthons, keep_pg_was_forced). The molecule must already be canonicalised."""
        if self._junk.is_substructure(molecule):
            return {}, False
        if classes is None:
            classes = [
                c
                for c in self.classifier.classify(molecule)
                if "MedChemHighlights" not in c and "DEL" not in c
            ]
        if not classes:
            return {}, False

        poly = [c for c in classes if "Bifunctional" in c or "Trifunctional" in c]
        # the undocumented branch: a poly-functional BB whose second functionality is not a
        # protecting group has nothing to deprotect, so the protected form is kept regardless of
        # the caller's keepPG. Named and reported instead of silently overriding.
        forced = bool(poly) and any(not any(t in c for t in _PG_TOKENS) for c in poly)
        keep_pg = self.config.keep_protecting_groups or forced

        working = {str(molecule): set()}
        mono_results: Synthons = {}
        skip_tokens = {t for name in poly for t in name.split("_")} if poly else set()
        exempt = bool(_MONO_SKIP_EXEMPT & set(poly))
        for name in classes:
            if name in poly:
                continue
            if skip_tokens and not exempt and any(t in name for t in skip_tokens):
                continue
            for key in list(working):
                _merge(
                    mono_results, self._run(name, _parse(key), working[key], keep_pg)
                )
            for key, value in mono_results.items():
                working.setdefault(key, set()).update(value)

        final: Synthons = {}
        if keep_pg or not poly:
            _merge(final, mono_results)
        for i, name in enumerate(poly):
            extra: Synthons = {}
            for key in list(working):
                produced = self._run(name, _parse(key), working[key], keep_pg)
                _merge(final, produced)
                if i < len(poly) - 1:
                    _merge(
                        extra, {k: v for k, v in produced.items() if k not in working}
                    )
            for key, value in extra.items():
                working.setdefault(key, set()).update(value)

        self._post_hooks(molecule, classes, final, keep_pg)
        return final, forced

    def _run(
        self, name: str, molecule: SynthonContainer, inherited: set[str], keep_pg: bool
    ) -> Synthons:
        program = self.programs.get(name)
        if program is None:
            return {}
        if program.strategy == "polymer":
            return _polymer(program, program.steps, molecule, inherited, program.func)
        if program.strategy == "protecting_group":
            return _protecting_group(
                program, program.steps, molecule, inherited, program.func, keep_pg
            )
        if program.strategy == "first_as_prep":
            return _first_as_prep(
                program, program.steps, molecule, inherited, program.func
            )
        return _normal(program, program.steps, molecule, inherited, program.func)

    def _post_hooks(
        self,
        molecule: MoleculeContainer,
        classes: list[str],
        final: Synthons,
        keep_pg: bool,
    ) -> None:
        """Four hooks, in order. No framework: four calls."""
        for name in classes:
            if "Trifunctional" in name and "Ester" in name and "Acid" in name:
                _merge(final, self._biacid_hook(final, name))
        _merge(final, self._azoles_hook(final))
        if not final and "Esters_Esters" in classes:
            fallback: Synthons = {}
            for rule in self._ester_fallback:
                for product in rule(_parse(str(molecule))):
                    fallback.setdefault(_key(product), set()).add("Esters_Esters")
            _merge(final, fallback)
        if "Ketones_Ketones" in classes:
            _merge(final, self._ketones_hook(final, keep_pg))

    def _biacid_hook(self, synthons: Synthons, name: str) -> Synthons:
        out: Synthons = {}
        for key, inherited in synthons.items():
            parent = _parse(key)
            if not self._biacid.is_substructure(parent):
                continue
            for product in self._biacid_rule(parent):
                if _keep(product, len(parent.synthon_labels)):
                    out.setdefault(_key(product), set()).update(inherited | {name})
        return out

    def _azoles_hook(self, synthons: Synthons) -> Synthons:
        if not synthons:
            return {}
        parsed = {key: _parse(key) for key in synthons}
        most = max(len(m.synthon_labels) for m in parsed.values())
        out: Synthons = {}
        for key, parent in parsed.items():
            if len(parent.synthon_labels) != most or not self._azole.is_substructure(
                parent
            ):
                continue
            for product in self._azole_rule(parent):
                if _keep(
                    product, len(parent.synthon_labels)
                ) and _one_azole_nucleophile(product):
                    out.setdefault(_key(product), set()).update(
                        synthons[key] | {"nHAzoles_nHAzoles"}
                    )
        return out

    def _ketones_hook(self, synthons: Synthons, keep_pg: bool) -> Synthons:
        """The one place a synthon re-enters the classifier — label-blind, via unlabelled()."""
        out: Synthons = {}
        for key, inherited in list(synthons.items()):
            if "Ketones_Ketones" not in inherited:
                continue
            parent = _parse(key)
            for name in self.classifier.classify(parent.unlabelled()):
                if "Alcohols" not in name:
                    continue
                produced = self._run(name, parent, inherited, keep_pg)
                for new_key, classes in produced.items():
                    if new_key not in synthons:
                        out.setdefault(new_key, set()).update(inherited | classes)
        return out

    def synthonise_smiles(self, smi: str) -> dict[str, dict]:
        """Component-aware entry point. Returns {synthon SMILES: {classes, component}}.

        One bad catalogue line must not take a worker down with it: 189k rows contain molecules
        chython cannot kekulise, and upstream calls `exit()` in four places, which is fatal
        inside a ProcessPoolExecutor.
        """
        out: dict[str, dict] = {}
        try:
            whole = smiles(smi)
            if not isinstance(whole, MoleculeContainer):
                return out
            components = list(whole.split())
        except Exception:
            return out
        if len(components) > self.config.max_components:
            return out
        for index, component in enumerate(components):
            try:
                molecule = safe_canonicalization(component)
                if (
                    len(components) > 1
                    and self.config.ignore_solvents
                    and str(molecule) in _SOLVENTS
                ):
                    continue
                produced, forced = self.synthonise(molecule)
            except Exception:  # a per-molecule failure is data, not a bug
                continue
            for key, classes in produced.items():
                record = out.setdefault(
                    key,
                    {"classes": set(), "component": index, "forced_keep_pg": forced},
                )
                record["classes"].update(classes)
        return out


def synthonise_batch(batch: list[str]) -> list[tuple[str, dict[str, dict]]]:
    """Picklable worker entry point: a batch of SMILES in, their synthons out.

    Batched because 189k items is ~380 tasks at 500 per batch, the same shape
    `standardize_smiles_batch` already uses. `_WORKER` costs 0.3 s per process, once.
    """
    return [(smi, _WORKER.synthonise_smiles(smi)) for smi in batch]


def classify_batch(batch: list[str]) -> list[tuple[str, list[str] | None]]:
    return [(smi, _WORKER.classifier.classify_smiles(smi)) for smi in batch]


_WORKER: BBSynthoniser | None = None


def init_worker(config: dict | None = None) -> None:
    """ProcessPoolExecutor initializer: build the 2401 compiled queries once per process.

    Neither BBSynthoniser nor a compiled QueryContainer can cross the process boundary, so the
    worker holds them in a module global instead.
    """
    global _WORKER
    _WORKER = BBSynthoniser(
        SynthonConfig.from_dict(config) if config else SynthonConfig()
    )


__all__ = [
    "BBSynthoniser",
    "Program",
    "Step",
    "classify_batch",
    "init_worker",
    "synthonise_batch",
]
