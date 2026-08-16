"""Library functions behind the five synthon CLI commands. `interfaces/cli.py` attaches them."""

from pathlib import Path

from chython import smiles, synthon_smiles

from synplan.chem.scaffolds import murcko_scaffold
from synplan.chem.utils import safe_canonicalization
from synplan.enumeration.synthon.config import SynthonConfig
from synplan.enumeration.synthon.enumeration import Enumerator
from synplan.enumeration.synthon.fragment import Fragmenter
from synplan.enumeration.synthon.stock import (
    SynthonRecord,
    load_synthon_stock,
    write_synthon_stock,
)
from synplan.enumeration.synthon.synthonise import (
    classify_batch,
    init_worker,
    synthonise_batch,
)
from synplan.utils.files import iter_smiles_records
from synplan.utils.parallel import chunked, process_pool_map_stream


def _records(path: str):
    for line in iter_smiles_records(path):
        fields = line.split()
        yield fields[0], (fields[1] if len(fields) > 1 else "")


def _guard(input_file: str, output_file: str) -> None:
    """Every command opens its output before it has read all of its input, and `w` truncates."""
    if Path(input_file).resolve() == Path(output_file).resolve():
        raise ValueError("input_file name and output_file name cannot be the same.")


def classify_file(
    input_file: str, output_file: str, config: SynthonConfig | None = None
) -> int:
    """`<bb_smiles>\\t<name>\\t<class>+<class>`, one line per INPUT ROW."""
    _guard(input_file, output_file)
    config = config or SynthonConfig()
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


def _map(items: list[str], worker, config: SynthonConfig):
    """Stream results, in one process or many. Results are unordered by design."""
    if config.num_workers <= 1 or len(items) < 500:
        init_worker(config.to_dict())
        for batch in chunked(items, 500):
            yield from worker(batch)
        return
    for batch in process_pool_map_stream(
        chunked(items, 500),
        worker,
        max_workers=config.num_workers,
        initializer=init_worker,
        initargs=(config.to_dict(),),
        max_tasks_per_child=50,
        timeout=0,
    ):
        yield from batch


def synthonise_file(
    input_file: str,
    output_file: str,
    config: SynthonConfig | None = None,
    keep_pg: bool | None = None,
) -> tuple[int, int]:
    """The stock file, plus a count of how many BBs had keepPG FORCED on against the setting."""
    _guard(input_file, output_file)
    config = config or SynthonConfig()
    if keep_pg is not None:
        config = config.model_copy(update={"keep_protecting_groups": keep_pg})
    written = forced = 0
    smiles_in = [s for s, _ in _records(input_file)]

    def stream():
        nonlocal written, forced
        for smi, produced in _map(smiles_in, synthonise_batch, config):
            if any(r["forced_keep_pg"] for r in produced.values()):
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


def fragment_file(
    input_file: str,
    output_file: str,
    stock_file: str | None = None,
    config: SynthonConfig | None = None,
) -> int:
    """`<target>\\t<pathway id>\\t<synthon>.<synthon>\\t<depth>\\t<availability>`."""
    _guard(input_file, output_file)
    stock = load_synthon_stock(stock_file) if stock_file else {}
    fragmenter = Fragmenter(config, stock)
    written = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for smi, _name in _records(input_file):
            try:
                dag = fragmenter.fragment(safe_canonicalization(smiles(smi)))
            except Exception:  # one unkekulisable row must not truncate the whole batch
                continue
            for pathway in dag.best_available():
                out.write(
                    f"{smi}\t{'|'.join(pathway.rules)}\t{'.'.join(pathway.key)}\t"
                    f"{pathway.depth}\t{pathway.availability:.4f}\n"
                )
                written += 1
    return written


def enumerate_file(
    input_file: str,
    output_file: str,
    stock_file: str,
    config: SynthonConfig | None = None,
) -> int:
    """`<product>\\t<target>\\t<pathway id>\\t<synthon>+<synthon>` from a fragmentation TSV.

    The pathway id is a rule signature shared across targets, so without the target column a
    product cannot be joined back to the molecule it was analogised from.
    """
    _guard(input_file, output_file)
    config = config or SynthonConfig()
    stock = load_synthon_stock(stock_file)
    enumerator = Enumerator(config)
    written = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for line in iter_smiles_records(input_file):
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            target, pathway_id, synthons = fields[0], fields[1], fields[2].split(".")
            slots = {s: [s] if s in stock else [] for s in synthons}
            try:
                products = list(enumerator.enumerate_analogues(synthons, slots))
            except Exception:  # one unparsable row must not truncate the whole batch
                continue
            for product in products:
                out.write(f"{product}\t{target}\t{pathway_id}\t{'+'.join(synthons)}\n")
                written += 1
    return written


def scaffolds_file(input_file: str, output_file: str) -> int:
    """`<bb_smiles>\\t<scaffold smiles | linearMolecule>`."""
    _guard(input_file, output_file)
    written = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for smi, _name in _records(input_file):
            try:
                molecule = safe_canonicalization(smiles(smi))
            except Exception:
                continue
            out.write(f"{smi}\t{murcko_scaffold(molecule)}\n")
            written += 1
    return written


def read_stock_synthons(stock_file: str) -> list[str]:
    """The stocked synthons as parsed containers' canonical SMILES."""
    return [str(synthon_smiles(s)) for s in load_synthon_stock(stock_file)]


__all__ = [
    "classify_file",
    "enumerate_file",
    "fragment_file",
    "read_stock_synthons",
    "scaffolds_file",
    "synthonise_file",
]
