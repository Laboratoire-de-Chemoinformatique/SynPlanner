"""Reaction atom-atom mapping via neural attention (GPU-accelerated).

Three-stage concurrent pipeline:
  parse (ProcessPool) → GPU inference (main thread) → map + write (ProcessPool)
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import chain, count
from math import inf
from pathlib import Path
from queue import Queue
from typing import Any

import torch
from numpy import argmax, array, concatenate, isclose, ix_, nonzero, ones, unravel_index, zeros
from scipy.linalg import block_diag
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from chython import smiles as parse_smiles
from chython.containers import ReactionContainer
from chytorch.utils.data import ReactionEncoderDataset, collate_encoded_reactions
from chytorch.zoo.rxnmap import Model

from synplan.utils.config import ConfigABC
from synplan.utils.files import count_smiles_records
from synplan.utils.parallel import default_num_workers, select_device

logger = logging.getLogger(__name__)

_RECYCLE_WORKERS = sys.version_info >= (3, 11)  # max_tasks_per_child added in 3.11


# -- Configuration -----------------------------------------------------------

@dataclass
class MappingConfig(ConfigABC):
    """Configuration for the GPU reaction-mapping pipeline.

    :param batch_size: Batch size for GPU inference.
    :param chunk_size: Lines per streaming chunk.
    :param device: ``"cuda"``, ``"mps"``, ``"cpu"`` or *None* (auto-detect).
    :param no_amp: Disable automatic mixed precision.
    """

    batch_size: int = 16
    chunk_size: int = 5000
    device: str | None = None
    no_amp: bool = False

    @staticmethod
    def from_dict(config_dict: dict[str, Any]) -> MappingConfig:
        return MappingConfig(**config_dict)

    @staticmethod
    def from_yaml(file_path: str) -> MappingConfig:
        import yaml

        with open(file_path, encoding="utf-8") as f:
            return MappingConfig.from_dict(yaml.safe_load(f) or {})

    def _validate_params(self, params: dict[str, Any]) -> None:
        if params["batch_size"] < 1:
            raise ValueError("batch_size must be >= 1")
        if params["chunk_size"] < 1:
            raise ValueError("chunk_size must be >= 1")
        dev = params.get("device")
        if dev is not None and dev not in ("cuda", "mps", "cpu"):
            raise ValueError(
                f"device must be 'cuda', 'mps', 'cpu', or None; got {dev!r}"
            )


# -- Worker functions (module-level for ProcessPool pickling) ----------------

def _parse_one(line: str):
    """Parse a single SMILES line → ``(ReactionContainer | None, error | None)``."""
    try:
        smi = line.split("\t")[0]
        rxn = parse_smiles(smi)
        if rxn is None:
            return None, "parse returned None"
        if not isinstance(rxn, ReactionContainer):
            return None, "not a reaction SMILES"
        return rxn, None
    except Exception as e:
        return None, f"parse: {e}"


def _map_and_format(args):
    """Map reaction with pre-computed attention → ``(smiles | None, error | None)``."""
    rxn, am = args
    try:
        _map_reaction(rxn, am)
        return format(rxn, "m"), None
    except Exception as e:
        return None, f"mapping: {e}"


def _fix_dupes(molecules, counter):
    """Remap atoms if indices collide across *molecules*."""
    atoms = [n for m in molecules for n in m]
    if len(atoms) != len(set(atoms)):
        for m in molecules:
            m.remap({n: next(counter) for n in m})


def _side_info(molecules):
    """Build adjacency matrix, atomic numbers, and role-mask bits for one side.

    Returns ``(atom_map, adj, atomic_numbers, mask_bits)``.
    """
    atom_map = [n for m in molecules for n in m]
    adj_blocks = [m.adjacency_matrix() for m in molecules]
    adj = block_diag(*adj_blocks).astype(bool) if adj_blocks else zeros((0, 0), dtype=bool)
    atomic_nums = array(
        [a.atomic_number for m in molecules for _, a in m.atoms()], dtype=int,
    )
    # Role mask: [sep, sep, atoms..., sep, atoms..., ...]
    mask = concatenate(
        [[False], *[[False] + [True] * len(m) for m in molecules]]
    ) if molecules else array([False])
    return atom_map, adj, atomic_nums, mask


def _map_reaction(rxn, am, multiplier=1.75):
    """Apply atom-atom mapping using a pre-computed attention matrix."""
    # Deduplicate atom indices within each side.
    c = count(1)
    _fix_dupes(list(chain(rxn.reactants, rxn.reagents)), c)
    _fix_dupes(list(rxn.products), c)
    fixed = next(c) != 1
    if fixed:
        rxn.flush_cache()

    r_map, r_adj, r_atoms, r_mask = _side_info(list(rxn.reactants))
    p_map, p_adj, p_atoms, p_mask = _side_info(list(rxn.products))
    rg_map = [n for m in rxn.reagents for n in m]
    ra, pa = len(r_map), len(p_map)

    if not ra or not pa:
        return False

    # Build padded role masks for the combined attention matrix.
    pad_r = zeros(len(p_mask) - 1, dtype=bool)
    pad_p = zeros(len(r_mask), dtype=bool)
    ram = concatenate([r_mask, pad_r])
    pam = concatenate([pad_p, p_mask[1:]])

    # Extract cross-attention filtered by matching element types.
    am = (am[ix_(pam, ram)] + am[ix_(ram, pam)].T) * (p_atoms[:, None] == r_atoms)

    # Greedy matching: pick highest-attention pairs, boosting neighbours.
    mapping = {}
    scope = zeros(pa, dtype=bool)
    seen = ones(pa, dtype=bool)

    for x in range(pa):
        if not x:
            i, j = unravel_index(argmax(am), am.shape)
        else:
            ams = am[scope]
            if ams.size:
                i, j = unravel_index(argmax(ams), ams.shape)
                i = nonzero(scope)[0][i]
            else:
                i, j = unravel_index(argmax(am), am.shape)

        if isclose(am[i, j], 0.0):
            for n in set(p_map).difference(mapping):
                mapping[n] = 0
            break

        mapping[p_map[i]] = r_map[j]
        am[ix_(p_adj[i], r_adj[j])] *= multiplier
        am[i] = am[:, j] = 0
        seen[i] = False
        scope[i] = False
        scope[p_adj[i] & seen] = True

    # Apply the discovered mapping back to the reaction.
    if any(n != m for n, m in mapping.items()):
        if fixed:
            r_mapping = {n: n for n in r_map}
        else:
            r_mapping = {m: n for n, m in enumerate(r_map, 1)}
            for m in rxn.reactants:
                m.remap(r_mapping)

        nm = max(r_mapping.values()) if r_mapping else 0
        p_mapping = {}
        for n, m in mapping.items():
            if m_val := r_mapping.get(m):
                p_mapping[n] = m_val
            else:
                nm += 1
                p_mapping[n] = nm
        for m in rxn.products:
            m.remap(p_mapping)

        if not fixed:
            rg_mapping = {m: n for n, m in enumerate(rg_map, nm + 1)}
            for m in rxn.reagents:
                m.remap(rg_mapping)

        rxn.flush_cache()
        fixed = True

    fixed = rxn.fix_groups_mapping() or fixed
    fixed = rxn.fix_mapping() or fixed
    return fixed


# -- GPU inference -----------------------------------------------------------

def _batched_attention(model, reactions, batch_size, device, use_amp=False):
    """Compute attention matrices for *reactions* using batched GPU inference."""
    if not reactions:
        return []

    # Sort by size for efficient batching, then restore original order.
    sizes = [
        sum(len(m) for m in rxn.reactants) + sum(len(m) for m in rxn.products)
        for rxn in reactions
    ]
    order = sorted(range(len(reactions)), key=lambda i: sizes[i])
    inv = [0] * len(reactions)
    for new, old in enumerate(order):
        inv[old] = new

    attns: list = [None] * len(reactions)
    amp_ctx = torch.autocast(device.type) if use_amp else nullcontext()

    ds = ReactionEncoderDataset([reactions[i] for i in order])
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_encoded_reactions,
    )

    pos = 0
    for batch in dl:
        batch = batch.to(device)
        atoms, neighbors, distances, roles = batch
        n = atoms.size(1)
        bs = atoms.size(0)

        d_mask = torch.zeros_like(roles, dtype=torch.float).masked_fill_(
            roles == 0, -inf
        )
        d_mask = d_mask.view(-1, 1, 1, n).expand(-1, model.nhead, n, -1)

        with amp_ctx:
            x = model.molecule_encoder(
                (atoms, neighbors, distances)
            ) * (roles > 1).unsqueeze_(-1)
            x = x + model.role_encoder(roles)
            for lr in model.layers[:-1]:
                x, _ = lr(x, d_mask)
            _, attn = model.layers[-1](
                x, d_mask, need_embedding=False, need_weights=True
            )

        for j in range(bs):
            attns[pos + j] = attn[j].float().cpu().numpy()
        pos += bs
        del batch, atoms, neighbors, distances, roles, d_mask, x, attn

    # Free GPU memory between chunks.
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return [attns[inv[i]] for i in range(len(reactions))]


# -- I/O helpers -------------------------------------------------------------

def _read_chunks(path, chunk_size):
    """Yield ``(offset, lines)`` chunks without loading the whole file."""
    chunk, offset = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                chunk.append(stripped)
                if len(chunk) == chunk_size:
                    yield offset, chunk
                    offset += len(chunk)
                    chunk = []
    if chunk:
        yield offset, chunk


# -- Main pipeline -----------------------------------------------------------

def map_reactions_from_file(
    config: MappingConfig,
    input_reaction_data_path: str | Path,
    mapped_reaction_data_path: str | Path = "reaction_data_mapped.smi",
    *,
    num_workers: int = 0,
    silent: bool = True,
    ignore_errors: bool = False,
    error_file_path: str | Path | None = None,
) -> None:
    """Map reactions from file using a 3-stage GPU pipeline.

    :param config: Mapping configuration.
    :param input_reaction_data_path: Input reaction SMILES file.
    :param mapped_reaction_data_path: Output file for mapped reactions.
    :param num_workers: CPU worker processes (0 = auto).
    :param silent: Suppress progress bar.
    :param ignore_errors: Failed reactions pass through as original SMILES.
    :param error_file_path: TSV for failures (default: ``<output>.errors.tsv``).
    """
    input_path = Path(input_reaction_data_path)
    output_path = Path(mapped_reaction_data_path)

    dev = select_device(config.device)
    use_amp = not config.no_amp and dev.type in ("cuda", "mps")
    workers = num_workers if num_workers > 0 else default_num_workers()

    _error_path: Path | None = None
    if error_file_path is not None:
        _error_path = Path(error_file_path)
    elif ignore_errors:
        _error_path = output_path.with_suffix(".errors.tsv")

    logger.info("Device: %s  AMP: %s  Workers: %d", dev, use_amp, workers)

    total = count_smiles_records(input_path)
    logger.info("Total: %d reactions", total)

    model = Model()
    model.to(dev).eval()

    # Bounded queues connect the three stages.
    parsed_q: Queue = Queue(maxsize=4)
    mapped_q: Queue = Queue(maxsize=4)
    stop = threading.Event()
    errors: dict[str, Exception | None] = {
        "parser": None, "writer": None, "gpu": None,
    }
    stats = {"mapped": 0, "failed": 0}

    _TIMEOUT = 2.0  # polling interval for stop-event checks

    def _put(q, item):
        """Put with periodic stop-event polling."""
        while not stop.is_set():
            try:
                q.put(item, timeout=_TIMEOUT)
                return
            except Exception:
                continue

    # -- Stage 1: read + parse -----------------------------------------------
    def _parser(pool):
        try:
            for offset, lines in _read_chunks(input_path, config.chunk_size):
                if stop.is_set():
                    break
                results = list(pool.map(_parse_one, lines, chunksize=64))
                rxns, idx, fails = [], [], []
                for i, (rxn, err) in enumerate(results):
                    if rxn is not None:
                        rxns.append(rxn)
                        idx.append(i)
                    else:
                        fails.append((i, lines[i], err))
                _put(parsed_q, (offset, lines, rxns, idx, fails))
        except Exception as e:
            errors["parser"] = e
            stop.set()
        finally:
            parsed_q.put(None)

    # -- Stage 3: map + format + write ---------------------------------------
    def _writer(pool, out, fail_out, pbar):
        try:
            while (item := mapped_q.get()) is not None:
                offset, lines, rxns, attns, idx, fails = item

                results = (
                    list(pool.map(_map_and_format, zip(rxns, attns), chunksize=16))
                    if rxns else []
                )

                out_lines = list(lines)
                for ri, (smi, err) in enumerate(results):
                    ci = idx[ri]
                    if smi is not None:
                        parts = lines[ci].split("\t", 1)
                        out_lines[ci] = smi + ("\t" + parts[1] if len(parts) > 1 else "")
                    else:
                        fails.append((ci, lines[ci], err))

                for ln in out_lines:
                    out.write(ln + "\n")

                fail_set = {i for i, _, _ in fails}
                stats["failed"] += len(fail_set)
                stats["mapped"] += len(lines) - len(fail_set)

                if fail_out:
                    for fi, fl, fe in fails:
                        fail_out.write(f"{offset + fi}\t{fl}\t{fe}\n")

                pbar.update(len(lines))
        except Exception as e:
            errors["writer"] = e
            stop.set()

    # -- Run pipeline --------------------------------------------------------
    out = fail_out = None
    t0 = time.perf_counter()
    try:
        out = open(output_path, "w", encoding="utf-8")
        if _error_path:
            fail_out = open(_error_path, "w", encoding="utf-8")

        # Recycle workers periodically to free accumulated memory.
        pool_kw = {"max_tasks_per_child": 1000} if _RECYCLE_WORKERS else {}

        with (
            ProcessPoolExecutor(max_workers=workers, **pool_kw) as pool,
            tqdm(total=total, desc="Mapping", unit="rxn", disable=silent) as pbar,
        ):
            thr_p = threading.Thread(target=_parser, args=(pool,), daemon=True)
            thr_w = threading.Thread(
                target=_writer, args=(pool, out, fail_out, pbar), daemon=True,
            )
            thr_p.start()
            thr_w.start()

            # Stage 2 (main thread): GPU inference
            try:
                while not stop.is_set():
                    item = parsed_q.get()
                    if item is None:
                        break
                    offset, lines, rxns, idx, fails = item

                    if rxns:
                        try:
                            with torch.no_grad():
                                attns = _batched_attention(
                                    model, rxns, config.batch_size, dev, use_amp,
                                )
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            logger.warning("GPU OOM (%d rxns)", len(rxns))
                            for i in idx:
                                fails.append((i, lines[i], "GPU out of memory"))
                            rxns, idx, attns = [], [], []
                    else:
                        attns = []

                    _put(mapped_q, (offset, lines, rxns, attns, idx, fails))
            except Exception as e:
                errors["gpu"] = e
                stop.set()
            finally:
                mapped_q.put(None)
                thr_p.join(timeout=10)
                thr_w.join(timeout=10)
    finally:
        if out:
            out.close()
        if fail_out:
            fail_out.close()

    elapsed = time.perf_counter() - t0
    for exc in errors.values():
        if exc is not None:
            raise exc

    throughput = total / elapsed if elapsed > 0 else 0
    summary = (
        f"Mapped {stats['mapped']}, failed {stats['failed']} "
        f"in {elapsed:.1f}s ({throughput:.0f} rxn/s)"
    )
    print(summary)
    logger.info(summary)
    if _error_path and stats["failed"]:
        logger.info("Errors written to: %s", _error_path)
