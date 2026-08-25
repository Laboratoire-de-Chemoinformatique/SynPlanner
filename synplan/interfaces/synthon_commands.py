"""Library functions behind the five synthon CLI commands. `interfaces/cli.py` attaches them."""

from collections.abc import Sized
from pathlib import Path

from chython import smiles, synthon_smiles
from chython.containers import MoleculeContainer, ReactionContainer

import synplan.chem.synthon.synthonise as _synthonise_workers
from synplan.chem.scaffolds import murcko_scaffold
from synplan.chem.synthon.config import SynthonConfig
from synplan.chem.synthon.coverage import (
    classify_coverage,
    load_coverage_rules,
)
from synplan.chem.synthon.enumerate import Enumerator
from synplan.chem.synthon.fragment import Fragmenter
from synplan.chem.synthon.stock import (
    SynthonRecord,
    load_synthon_stock,
    write_synthon_stock,
)
from synplan.chem.synthon.synthonise import (
    classify_batch,
    init_worker,
    synthonise_batch,
)
from synplan.chem.utils import safe_canonicalization
from synplan.interfaces.synthon_audit import (
    AuditError,
    AuditOutcome,
    AuditRun,
    InputRecord,
    iter_molecule_records,
    iter_pathway_records,
)
from synplan.utils.files import split_smiles_record
from synplan.utils.parallel import chunked, process_pool_map_stream

_BATCH_SIZE = 500


def _record_name(record: InputRecord) -> str:
    if record.headered:
        for name, value in zip(record.metadata_names, record.metadata):
            if name.casefold() in {"name", "id"}:
                return value
    return record.metadata[0] if record.metadata else ""


def _records(path: str):
    """Yield valid molecule rows under the strict TAB/headered-TSV contract."""
    for record in iter_molecule_records(path):
        if record.format_error is None:
            yield record.chemistry, _record_name(record)


def _guard(input_file: str, output_file: str) -> None:
    """Every command opens its output before it has read all of its input, and `w` truncates."""
    if Path(input_file).resolve() == Path(output_file).resolve():
        raise ValueError("input_file name and output_file name cannot be the same.")


def _exception(
    record: InputRecord,
    stage: str,
    error_type: str,
    error: Exception | str,
    *,
    component: int | None = None,
) -> AuditOutcome:
    message = str(error) or repr(error)
    context = {"component": component} if component is not None else {}
    return AuditOutcome(
        record,
        "processing_error",
        detail=message[:2000],
        errors=(AuditError(stage, error_type, message[:2000], context),),
    )


def _format_error(record: InputRecord, stage: str) -> AuditOutcome | None:
    if record.format_error is None:
        return None
    return _exception(record, stage, "input_format_error", record.format_error)


def _components(record: InputRecord) -> list[MoleculeContainer]:
    molecule = smiles(record.chemistry)
    if not isinstance(molecule, MoleculeContainer):
        raise TypeError(f"expected MoleculeContainer, got {type(molecule).__name__}")
    return list(molecule.split())


def _active_synthoniser():
    worker = _synthonise_workers._WORKER
    if worker is None:
        raise RuntimeError("synthon worker was not initialized")
    return worker


def _classify_audit_record(record: InputRecord) -> AuditOutcome:
    stage = "bb_classifying"
    if outcome := _format_error(record, stage):
        return outcome
    try:
        components = _components(record)
    except Exception as error:
        return _exception(record, stage, "parse_error", error)

    classifier = _active_synthoniser().classifier
    found: set[str] = set()
    evaluated = 0
    errors: list[AuditError] = []
    for index, component in enumerate(components):
        try:
            molecule = safe_canonicalization(component)
        except Exception as error:
            message = str(error) or repr(error)
            errors.append(
                AuditError(
                    stage,
                    "canonicalization_error",
                    message[:2000],
                    {"component": index},
                )
            )
            continue
        try:
            found.update(classifier.classify(molecule))
            evaluated += 1
        except Exception as error:
            message = str(error) or repr(error)
            errors.append(
                AuditError(
                    stage,
                    "classification_error",
                    message[:2000],
                    {"component": index},
                )
            )
    if not evaluated:
        return AuditOutcome(
            record,
            "processing_error",
            detail="no component could be canonicalized and classified",
            errors=tuple(errors),
        )
    classes = tuple(name for name, *_ in classifier.classes if name in found)
    if not classes:
        return AuditOutcome(
            record,
            "unclassified",
            detail="the record parsed successfully but no Synt-On class matched",
            errors=tuple(errors),
        )
    row = f"{record.chemistry}\t{_record_name(record)}\t{'+'.join(classes)}"
    return AuditOutcome(
        record,
        "classified",
        output_rows=(row,),
        detail=f"{len(classes)} classes matched",
        errors=tuple(errors),
    )


def _classify_audit_batch(batch: list[InputRecord]) -> list[AuditOutcome]:
    return [_classify_audit_record(record) for record in batch]


def _synthonise_audit_record(record: InputRecord) -> AuditOutcome:
    stage = "bb_synthonizing"
    if outcome := _format_error(record, stage):
        return outcome
    try:
        components = _components(record)
    except Exception as error:
        return _exception(record, stage, "parse_error", error)

    worker = _active_synthoniser()
    config = worker.config
    if len(components) > config.max_components:
        return AuditOutcome(
            record,
            "max_components",
            detail=(
                f"{len(components)} components exceeds "
                f"max_components={config.max_components}"
            ),
        )

    produced_by_key: dict[str, dict] = {}
    canonicalized = non_solvent = classified = transformed = 0
    eligible_class_found = False
    errors: list[AuditError] = []
    for index, component in enumerate(components):
        try:
            molecule = safe_canonicalization(component)
            canonicalized += 1
        except Exception as error:
            message = str(error) or repr(error)
            errors.append(
                AuditError(
                    stage,
                    "canonicalization_error",
                    message[:2000],
                    {"component": index},
                )
            )
            continue
        if (
            len(components) > 1
            and config.ignore_solvents
            and str(molecule) in _synthonise_workers._SOLVENTS
        ):
            continue
        non_solvent += 1
        try:
            classes = [
                name
                for name in worker.classifier.classify(molecule)
                if "MedChemHighlights" not in name and "DEL" not in name
            ]
            classified += 1
        except Exception as error:
            message = str(error) or repr(error)
            errors.append(
                AuditError(
                    stage,
                    "classification_error",
                    message[:2000],
                    {"component": index},
                )
            )
            continue
        if not classes:
            continue
        eligible_class_found = True
        try:
            produced, forced = worker.synthonise(molecule, classes)
            transformed += 1
        except Exception as error:
            message = str(error) or repr(error)
            errors.append(
                AuditError(
                    stage,
                    "transformation_error",
                    message[:2000],
                    {"component": index},
                )
            )
            continue
        for synthon, class_names in produced.items():
            entry = produced_by_key.setdefault(
                synthon,
                {"classes": set(), "component": index, "forced_keep_pg": forced},
            )
            entry["classes"].update(class_names)
            entry["forced_keep_pg"] = entry["forced_keep_pg"] or forced

    if produced_by_key:
        rows = tuple(
            SynthonRecord(
                synthon,
                (record.chemistry,),
                tuple(sorted(entry["classes"])),
                int(entry["component"]),
            ).line()
            for synthon, entry in sorted(produced_by_key.items())
        )
        forced = any(entry["forced_keep_pg"] for entry in produced_by_key.values())
        return AuditOutcome(
            record,
            "synthonised",
            output_rows=rows,
            detail=f"{len(rows)} synthons produced",
            errors=tuple(errors),
            metrics={"forced_keep_pg": forced},
        )
    if not canonicalized:
        status, detail = "processing_error", "no component could be canonicalized"
    elif not non_solvent:
        status, detail = (
            "unclassified",
            "all parsed components were ignored as solvents",
        )
    elif not classified:
        status, detail = "processing_error", "classification failed for every component"
    elif not eligible_class_found:
        status, detail = "unclassified", "no eligible Synt-On class matched"
    elif not transformed:
        status, detail = (
            "processing_error",
            "synthon transformation failed for every eligible component",
        )
    else:
        status, detail = (
            "no_synthon",
            "eligible classes matched but no rule produced output",
        )
    return AuditOutcome(record, status, detail=detail, errors=tuple(errors))


def _synthonise_audit_batch(batch: list[InputRecord]) -> list[AuditOutcome]:
    return [_synthonise_audit_record(record) for record in batch]


def _map(items, worker, config: SynthonConfig, *, ordered: bool = False):
    """Stream batched results, optionally preserving input order for an audit."""
    batches = chunked(items, _BATCH_SIZE)
    serial = config.num_workers <= 1 or (
        not ordered and isinstance(items, Sized) and len(items) < _BATCH_SIZE
    )
    if serial:
        init_worker(config.to_dict())
        for batch in batches:
            yield from worker(batch)
        return
    for batch in process_pool_map_stream(
        batches,
        worker,
        max_workers=config.num_workers,
        initializer=init_worker,
        initargs=(config.to_dict(),),
        max_tasks_per_child=50,
        timeout=0,
        ordered=ordered,
    ):
        yield from batch


def _provenance(config: SynthonConfig, *names: str, stock: str | None = None):
    paths = {name: getattr(config, name) for name in names}
    if stock is not None:
        paths["stock"] = stock
    return paths


def classify_file(
    input_file: str, output_file: str, config: SynthonConfig | None = None
) -> int:
    """`<bb_smiles>\\t<name>\\t<class>+<class>`, one line per input row."""
    _guard(input_file, output_file)
    config = config or SynthonConfig()
    if config.write_audit_files:
        with AuditRun(
            "bb_classifying",
            input_file,
            output_file,
            config,
            provenance_files=_provenance(config, "classes_path"),
        ) as audit:
            outcomes = _map(
                iter_molecule_records(input_file),
                _classify_audit_batch,
                config,
                ordered=True,
            )
            for outcome in outcomes:
                audit.write(outcome)
        return audit.counters["output_rows"]

    names: dict[str, list[str]] = {}
    for smi, name in _records(input_file):
        names.setdefault(smi, []).append(name)
    written = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for smi, classes in _map(list(names), classify_batch, config):
            if not classes:
                continue
            for name in names[smi]:
                out.write(f"{smi}\t{name}\t{'+'.join(classes)}\n")
                written += 1
    return written


def synthonise_file(
    input_file: str,
    output_file: str,
    config: SynthonConfig | None = None,
    keep_pg: bool | None = None,
) -> tuple[int, int]:
    """The stock file, plus the number of BBs whose keepPG setting was forced."""
    _guard(input_file, output_file)
    config = config or SynthonConfig()
    if keep_pg is not None:
        config = config.model_copy(update={"keep_protecting_groups": keep_pg})
    if config.write_audit_files:
        with AuditRun(
            "bb_synthonizing",
            input_file,
            output_file,
            config,
            provenance_files=_provenance(config, "classes_path", "marks_path"),
        ) as audit:
            outcomes = _map(
                iter_molecule_records(input_file),
                _synthonise_audit_batch,
                config,
                ordered=True,
            )
            for outcome in outcomes:
                audit.write(outcome)
        return audit.counters["output_rows"], audit.counters["forced_keep_pg"]

    written = forced = 0
    smiles_in = [s for s, _ in _records(input_file)]

    def stream():
        nonlocal written, forced
        for smi, produced in _map(smiles_in, synthonise_batch, config):
            if any(record["forced_keep_pg"] for record in produced.values()):
                forced += 1
            for synthon, record in produced.items():
                written += 1
                yield SynthonRecord(
                    synthon,
                    (smi,),
                    tuple(sorted(record["classes"])),
                    record["component"],
                )

    write_synthon_stock(output_file, stream())
    return written, forced


def _fragment_outcome(record: InputRecord, fragmenter: Fragmenter) -> AuditOutcome:
    stage = "synthon_fragment"
    if outcome := _format_error(record, stage):
        return outcome
    try:
        molecule = safe_canonicalization(smiles(record.chemistry))
    except Exception as error:
        return _exception(record, stage, "parse_error", error)
    try:
        dag = fragmenter.fragment(molecule)
    except Exception as error:
        return _exception(record, stage, "fragmentation_error", error)
    rows = tuple(
        f"{record.chemistry}\t{'|'.join(pathway.rules)}\t"
        f"{'.'.join(pathway.key)}\t{pathway.depth}\t{pathway.availability:.4f}"
        for pathway in dag.best_available()
    )
    if not rows:
        return AuditOutcome(
            record, "no_pathways", detail="no disconnection pathway matched"
        )
    return AuditOutcome(
        record,
        "fragmented",
        output_rows=rows,
        detail=f"{len(rows)} pathways produced",
    )


def fragment_file(
    input_file: str,
    output_file: str,
    stock_file: str | None = None,
    config: SynthonConfig | None = None,
) -> int:
    """`<target>\\t<pathway id>\\t<synthon>.<synthon>\\t<depth>\\t<availability>`."""
    _guard(input_file, output_file)
    config = config or SynthonConfig()
    if config.write_audit_files:
        with AuditRun(
            "synthon_fragment",
            input_file,
            output_file,
            config,
            provenance_files=_provenance(config, "rules_path", stock=stock_file),
        ) as audit:
            stock = load_synthon_stock(stock_file, config) if stock_file else {}
            fragmenter = Fragmenter(config, stock)
            for record in iter_molecule_records(input_file):
                audit.write(_fragment_outcome(record, fragmenter))
        return audit.counters["output_rows"]

    stock = load_synthon_stock(stock_file, config) if stock_file else {}
    fragmenter = Fragmenter(config, stock)
    written = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for smi, _name in _records(input_file):
            try:
                dag = fragmenter.fragment(safe_canonicalization(smiles(smi)))
            except Exception:
                continue
            for pathway in dag.best_available():
                out.write(
                    f"{smi}\t{'|'.join(pathway.rules)}\t{'.'.join(pathway.key)}\t"
                    f"{pathway.depth}\t{pathway.availability:.4f}\n"
                )
                written += 1
    return written


def _enumerate_outcome(
    record: InputRecord,
    enumerator: Enumerator,
    stock,
    config: SynthonConfig,
) -> AuditOutcome:
    stage = "synthon_enumerate"
    if outcome := _format_error(record, stage):
        return outcome
    target, pathway_id, encoded = record.fields[:3]
    synthons = tuple(encoded.split("."))
    try:
        slots = stock.slots(synthons, config)
    except Exception as error:
        return _exception(record, stage, "enumeration_error", error)
    missing = [synthon for synthon in synthons if not slots.get(synthon)]
    products = []
    errors: list[AuditError] = []
    try:
        products.extend(enumerator.enumerate_analogues(synthons, slots))
    except Exception as error:
        if not products:
            return _exception(record, stage, "enumeration_error", error)
        message = str(error) or repr(error)
        errors.append(AuditError(stage, "enumeration_error", message[:2000]))
    if not products:
        if missing:
            return AuditOutcome(
                record,
                "missing_stock_slots",
                detail=f"{len(missing)} pathway slots have no stock candidate",
            )
        return AuditOutcome(record, "no_products", detail="no product was enumerated")
    rows = tuple(
        f"{product}\t{target}\t{pathway_id}\t{'+'.join(synthons)}"
        for product in products
    )
    return AuditOutcome(
        record,
        "enumerated",
        output_rows=rows,
        detail=f"{len(rows)} products produced",
        errors=tuple(errors),
    )


def enumerate_file(
    input_file: str,
    output_file: str,
    stock_file: str,
    config: SynthonConfig | None = None,
) -> int:
    """Enumerate products from the fixed five-column fragmentation TSV."""
    _guard(input_file, output_file)
    config = config or SynthonConfig()
    if config.write_audit_files:
        with AuditRun(
            "synthon_enumerate",
            input_file,
            output_file,
            config,
            provenance_files=_provenance(config, "rules_path", stock=stock_file),
        ) as audit:
            stock = load_synthon_stock(stock_file, config)
            enumerator = Enumerator(config)
            for record in iter_pathway_records(input_file):
                audit.write(_enumerate_outcome(record, enumerator, stock, config))
        return audit.counters["output_rows"]

    stock = load_synthon_stock(stock_file, config)
    enumerator = Enumerator(config)
    written = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for record in iter_pathway_records(input_file):
            if record.format_error is not None:
                continue
            target, pathway_id, encoded = record.fields[:3]
            synthons = tuple(encoded.split("."))
            try:
                slots = stock.slots(synthons, config)
                products = enumerator.enumerate_analogues(synthons, slots)
                for product in products:
                    out.write(
                        f"{product}\t{target}\t{pathway_id}\t{'+'.join(synthons)}\n"
                    )
                    written += 1
            except Exception:
                continue
    return written


def _scaffold_outcome(record: InputRecord) -> AuditOutcome:
    stage = "bb_scaffolds"
    if outcome := _format_error(record, stage):
        return outcome
    try:
        molecule = safe_canonicalization(smiles(record.chemistry))
        scaffold = murcko_scaffold(molecule)
    except Exception as error:
        return _exception(record, stage, "scaffold_error", error)
    return AuditOutcome(
        record,
        "scaffolded",
        output_rows=(f"{record.chemistry}\t{scaffold}",),
        detail="scaffold produced",
    )


def scaffolds_file(
    input_file: str, output_file: str, config: SynthonConfig | None = None
) -> int:
    """`<bb_smiles>\\t<scaffold smiles | linearMolecule>`."""
    _guard(input_file, output_file)
    config = config or SynthonConfig()
    if config.write_audit_files:
        with AuditRun("bb_scaffolds", input_file, output_file, config) as audit:
            for record in iter_molecule_records(input_file):
                audit.write(_scaffold_outcome(record))
        return audit.counters["output_rows"]

    written = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for smi, _name in _records(input_file):
            try:
                molecule = safe_canonicalization(smiles(smi))
                scaffold = murcko_scaffold(molecule)
            except Exception:
                continue
            out.write(f"{smi}\t{scaffold}\n")
            written += 1
    return written


def coverage_file(
    input_file: str,
    output_file: str,
    config: SynthonConfig | None = None,
    *,
    keep: str = "uncovered",
) -> tuple[int, int]:
    """Split a mapped-reaction file on synthon coverage; returns (written, read).

    Kept lines are copied verbatim, metadata columns and all, so the output is still the input
    file's format. A record that will not parse is treated as uncovered: dropping training data
    on a valence quirk is the worse error.
    """
    _guard(input_file, output_file)
    rules = load_coverage_rules(config or SynthonConfig())
    wanted = keep == "covered"
    written = read = 0
    with (
        open(input_file, encoding="utf-8") as source,
        open(output_file, "w", encoding="utf-8") as out,
    ):
        for line in source:
            record, _ = split_smiles_record(line)
            if not record:
                continue
            read += 1
            try:
                reaction = smiles(record)
                covered = (
                    isinstance(reaction, ReactionContainer)
                    and classify_coverage(reaction, rules).covered
                )
            except Exception:
                covered = False
            if covered is wanted:
                out.write(line if line.endswith("\n") else line + "\n")
                written += 1
    return written, read


def read_stock_synthons(stock_file: str) -> list[str]:
    """The stocked synthons as parsed containers' canonical SMILES."""
    return [str(synthon_smiles(s)) for s in load_synthon_stock(stock_file)]


__all__ = [
    "classify_file",
    "coverage_file",
    "enumerate_file",
    "fragment_file",
    "read_stock_synthons",
    "scaffolds_file",
    "synthonise_file",
]
