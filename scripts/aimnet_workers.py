"""Optional isolated CPU workers for independent molecular calculations.

Workers never write the shared molecular cache. Each result has its own atomic
job checkpoint; the parent assessor remains the only molecular-cache writer.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from synplan.chem.reaction.score.aimnet.cache import symmetry_error
from synplan.chem.reaction.score.aimnet.config import (
    AIMNetConfig,
    MolecularCalculationError,
    normalize_config,
)
from synplan.chem.reaction.score.aimnet.molecular import (
    PROTON_REFERENCE,
    AIMNetMolecularCalculator,
)
from synplan.chem.reaction.score.aimnet.preparation import (
    prepare_reaction,
    stereoisomers,
)
from synplan.chem.reaction.score.aimnet.species import is_proton
from synplan.utils.files import read_json, write_json_atomic

WORKER_CHECKPOINT_VERSION = 2

_backend = None


def _initialize(config, threads):
    import torch

    global _backend
    torch.set_num_threads(threads)
    _backend = AIMNetMolecularCalculator(AIMNetConfig.model_validate(config))
    _backend.identity()


def _identity():
    assert _backend is not None, "Worker calculator was not initialized"
    return _backend.identity()


def _calculate(smiles, checkpoint):
    assert _backend is not None, "Worker calculator was not initialized"
    try:
        result = _backend.calculate(smiles)
        if problem := symmetry_error(smiles, result, _backend.config):
            raise ValueError(problem)
        record = {"result": result}
    except MolecularCalculationError as exc:
        record = {
            "failure": exc.to_dict(),
            "conformer_failures": exc.conformer_failures,
        }
    write_json_atomic(checkpoint, record)
    return record


class CPUMolecularWorkers:
    """MolecularCalculator adapter with eager independent-species scheduling."""

    def __init__(self, config, output, workers, threads):
        if config.device != "cpu":
            raise ValueError("Multiple molecular workers currently support CPU only")
        if workers < 1 or threads < 1:
            raise ValueError("Worker and thread counts must be positive")
        config = self.config = normalize_config(config)
        self._thread_environment = {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        }
        for name in self._thread_environment:
            os.environ[name] = str(threads)
        self.pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(config.model_dump(mode="json"), threads),
        )
        self.output = Path(output)
        self.futures = {}
        self._identity = None
        self.cached = {}

    def identity(self):
        if self._identity is None:
            self._identity = self.pool.submit(_identity).result()
        return self._identity

    def _path(self, smiles):
        data = {
            "version": WORKER_CHECKPOINT_VERSION,
            "config": self.config.model_dump(mode="json"),
            "calculator": self.identity(),
            "smiles": smiles,
            "proton_reference": PROTON_REFERENCE,
        }
        key = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return self.output / f"{key}.json"

    def prime(self, reactions, cache):
        # Match the existing assessor cache identity, including settings, model
        # hashes and versions. Do not schedule already successful molecules.
        for block in cache.get("calculations", {}).values():
            identity = block.get("identity", {})
            if (
                identity.get("config") == self.config.molecular_settings()
                and identity.get("calculator") == self.identity()
                and identity.get("assessment_version") == 1
            ):
                self.cached.update(block["molecules"])
        seen = set()
        for reaction in reactions:
            try:
                _, reactants, products = prepare_reaction(
                    reaction,
                    self.config.min_confidence,
                    infer_balance=self.config.balance_policy == "infer",
                )
            except MolecularCalculationError:
                continue  # The assessor records the exact reaction failure.
            for species in dict.fromkeys([*reactants, *products]):
                for smiles in (
                    stereoisomers(species)
                    if self.config.stereo_policy == "enumerate"
                    else (species,)
                ):
                    if smiles in seen:
                        continue
                    seen.add(smiles)
                    cached = self.cached.get(smiles)
                    if (
                        cached is not None
                        and not symmetry_error(smiles, cached, self.config)
                        and (
                            not is_proton(smiles)
                            or cached.get("electronic_energy_reference")
                            == PROTON_REFERENCE
                        )
                    ):
                        continue
                    self._submit(smiles)
        return len(seen), len(self.futures)

    def _submit(self, smiles):
        path = self._path(smiles)
        if path.exists():
            record = read_json(path)
            result = record.get("result")
            if result is not None and symmetry_error(smiles, result, self.config):
                path.unlink()
        if smiles not in self.futures and not path.exists():
            self.futures[smiles] = self.pool.submit(_calculate, smiles, path)

    def calculate(self, smiles):
        self._submit(smiles)
        future = self.futures.get(smiles)
        record = (
            future.result() if future is not None else read_json(self._path(smiles))
        )
        if "failure" in record:
            error = record["failure"]
            exc = MolecularCalculationError(
                error["stage"],
                error["reason"],
                error["detail"],
                category=error["category"],
            )
            exc.conformer_failures = record["conformer_failures"]
            raise exc
        if problem := symmetry_error(smiles, record["result"], self.config):
            raise ValueError(problem)
        return record["result"]

    def close(self):
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        finally:
            for name, value in self._thread_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
