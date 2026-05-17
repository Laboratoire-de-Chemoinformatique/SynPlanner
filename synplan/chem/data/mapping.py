"""Reaction atom-atom mapping via neural attention (GPU-accelerated).

Three-stage concurrent pipeline:
  parse (ProcessPool) → GPU inference (main thread) → map + write (ProcessPool)
"""

import logging
import os
import signal
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import nullcontext
from itertools import chain, count
from math import inf
from pathlib import Path
from queue import Queue
from typing import Literal

import torch
from chython.containers import ReactionContainer
from chytorch.utils.data import ReactionEncoderDataset, collate_encoded_reactions
from chytorch.zoo.rxnmap import Model
from numpy import (
    argmax,
    array,
    concatenate,
    isclose,
    ix_,
    nonzero,
    ones,
    unravel_index,
    zeros,
)
from pydantic import Field
from scipy.linalg import block_diag
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from synplan.utils.config import BaseConfigModel
from synplan.utils.files import (
    RawReactionReader,
    extract_origin_fields,
    init_parse_worker,
    parse_error_message,
    parse_one,
    tsv_safe,
    write_error_row,
    write_error_tsv_header,
)
from synplan.utils.parallel import default_num_workers, select_device

logger = logging.getLogger(__name__)


# -- Configuration -----------------------------------------------------------


class MappingConfig(BaseConfigModel):
    """Configuration for the GPU reaction-mapping pipeline.

    :param batch_size: Batch size for GPU inference.
    :param chunk_size: Lines per streaming chunk.
    :param device: ``"cuda"``, ``"mps"``, ``"cpu"`` or *None* (auto-detect).
    :param no_amp: Disable automatic mixed precision.
    :param worker_timeout: Timeout in seconds per worker task (parse/map chunk).
        If a chunk takes longer, it is skipped and reactions are logged as failed.
        Set to 0 to disable.
    """

    batch_size: int = Field(default=16, ge=1)
    chunk_size: int = Field(default=5000, ge=1)
    device: Literal["cuda", "mps", "cpu"] | None = None
    no_amp: bool = False
    worker_timeout: int = Field(default=120, ge=0)


# -- Worker functions (module-level for ProcessPool pickling) ----------------


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
    adj = (
        block_diag(*adj_blocks).astype(bool)
        if adj_blocks
        else zeros((0, 0), dtype=bool)
    )
    atomic_nums = array(
        [a.atomic_number for m in molecules for _, a in m.atoms()],
        dtype=int,
    )
    # Role mask: [sep, sep, atoms..., sep, atoms..., ...]
    mask = (
        concatenate([[False], *[[False] + [True] * len(m) for m in molecules]])
        if molecules
        else array([False])
    )
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


# -- Single-reaction helper --------------------------------------------------


def map_reaction(
    rxn: ReactionContainer,
    *,
    device: str | None = None,
    no_amp: bool = False,
) -> ReactionContainer:
    """Map a single reaction using GPU-accelerated neural attention.

    :param rxn: Reaction to map.
    :param device: ``"cuda"``, ``"mps"``, ``"cpu"`` or *None* (auto-detect).
    :param no_amp: Disable automatic mixed precision.
    :returns: The same reaction object with atom-atom mapping applied.
    """
    dev = select_device(device)
    use_amp = not no_amp and dev.type in ("cuda", "mps")

    model = Model()
    model.to(dev).eval()

    with torch.no_grad():
        attns = _batched_attention(
            model, [rxn], batch_size=1, device=dev, use_amp=use_amp
        )

    _map_reaction(rxn, attns[0])
    return rxn


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
        ds,
        batch_size=batch_size,
        shuffle=False,
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
            x = model.molecule_encoder((atoms, neighbors, distances)) * (
                roles > 1
            ).unsqueeze_(-1)
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
    worker_timeout = config.worker_timeout or None  # 0 means no timeout

    _error_path: Path | None = None
    if error_file_path is not None:
        _error_path = Path(error_file_path)
    elif ignore_errors:
        _error_path = output_path.with_suffix(".errors.tsv")

    logger.info("Device: %s  AMP: %s  Workers: %d", dev, use_amp, workers)

    raw_reader = RawReactionReader(input_path)
    fmt = raw_reader.format

    total = raw_reader.count()
    logger.info("Total: %d reactions", total)

    model = Model()
    model.to(dev).eval()

    # Bounded queues connect the three stages.
    parsed_q: Queue = Queue(maxsize=4)
    mapped_q: Queue = Queue(maxsize=4)
    stop = threading.Event()
    errors: dict[str, Exception | None] = {
        "parser": None,
        "writer": None,
        "gpu": None,
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
            for offset, items in raw_reader.iter_chunks(config.chunk_size):
                if stop.is_set():
                    break
                futures = [pool.submit(parse_one, item) for item in items]
                rxns, idx, fails = [], [], []
                for i, fut in enumerate(futures):
                    try:
                        rxn, err = fut.result(timeout=worker_timeout)
                        if rxn is not None:
                            rxns.append(rxn)
                            idx.append(i)
                        else:
                            fails.append((i, items[i], err))
                    except FuturesTimeoutError:
                        fut.cancel()
                        fails.append((i, items[i], "parse timeout"))
                    except Exception as e:
                        fails.append((i, items[i], f"parse worker: {e}"))
                _put(parsed_q, (offset, items, rxns, idx, fails))
        except Exception as e:
            errors["parser"] = e
            stop.set()
        finally:
            parsed_q.put(None)

    # -- Stage 3: map + format + write ---------------------------------------
    def _writer(pool, out, fail_out, pbar):
        try:
            while (item := mapped_q.get()) is not None:
                offset, raw_items, rxns, attns, idx, fails = item

                if rxns:
                    futures = [
                        pool.submit(_map_and_format, (rxn, attn))
                        for rxn, attn in zip(rxns, attns)
                    ]
                    results = []
                    for fi, fut in enumerate(futures):
                        try:
                            results.append(fut.result(timeout=worker_timeout))
                        except FuturesTimeoutError:
                            fut.cancel()
                            results.append((None, "mapping timeout"))
                            logger.warning(
                                "Mapping timeout for reaction at offset %d+%d",
                                offset,
                                idx[fi],
                            )
                        except Exception as e:
                            results.append((None, f"mapping worker: {e}"))
                else:
                    results = []

                n_ok = 0
                for ri, (smi, err) in enumerate(results):
                    ci = idx[ri]
                    if smi is not None:
                        out.write(smi + "\n")
                        n_ok += 1
                    else:
                        fails.append((ci, raw_items[ci], err))

                stats["failed"] += len(fails)
                stats["mapped"] += n_ok

                if fail_out:
                    for _fi, fl, fe in fails:
                        # Truncate RDF blocks to first line; SMILES records pass
                        # through extract_origin_fields which splits off
                        # tab-separated source columns into ``source_info``.
                        fl_short = fl.split("\n")[0] if "\n" in fl else fl
                        original, source_info = extract_origin_fields(
                            fl_short, fmt="smi"
                        )
                        stage, error_type, message = parse_error_message(
                            tsv_safe(str(fe)), default_stage="mapping"
                        )
                        write_error_row(
                            fail_out,
                            original,
                            source_info,
                            stage,
                            error_type,
                            message,
                        )

                pbar.update(len(raw_items))
        except Exception as e:
            errors["writer"] = e
            stop.set()

    # -- Signal handling for graceful shutdown --------------------------------
    _original_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, frame):
        logger.warning("SIGTERM received — stopping mapping pipeline")
        stop.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # -- Run pipeline --------------------------------------------------------
    out = fail_out = None
    t0 = time.perf_counter()
    try:
        out = open(output_path, "w", encoding="utf-8")
        if _error_path:
            fail_out = open(_error_path, "w", encoding="utf-8")
            write_error_tsv_header(fail_out)

        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_parse_worker,
            initargs=(fmt,),
        )
        pbar = tqdm(total=total, desc="Mapping", unit="rxn", disable=silent)

        try:
            thr_p = threading.Thread(target=_parser, args=(pool,), daemon=True)
            thr_w = threading.Thread(
                target=_writer,
                args=(pool, out, fail_out, pbar),
                daemon=True,
            )
            thr_p.start()
            thr_w.start()

            # Stage 2 (main thread): GPU inference
            try:
                while not stop.is_set():
                    item = parsed_q.get()
                    if item is None:
                        break
                    offset, raw_items, rxns, idx, fails = item

                    if rxns:
                        try:
                            with torch.no_grad():
                                attns = _batched_attention(
                                    model,
                                    rxns,
                                    config.batch_size,
                                    dev,
                                    use_amp,
                                )
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            logger.warning("GPU OOM (%d rxns)", len(rxns))
                            for i in idx:
                                fails.append((i, raw_items[i], "GPU out of memory"))
                            rxns, idx, attns = [], [], []
                    else:
                        attns = []

                    _put(mapped_q, (offset, raw_items, rxns, attns, idx, fails))
            except Exception as e:
                errors["gpu"] = e
                stop.set()
            finally:
                mapped_q.put(None)
                thr_p.join(timeout=10)
                thr_w.join(timeout=10)
        finally:
            pbar.close()
            # Collect worker PIDs before shutdown so we can force-kill if needed
            worker_pids = [p.pid for p in pool._processes.values() if p.pid is not None]
            pool.shutdown(wait=False, cancel_futures=True)
            # Force-kill lingering workers to avoid atexit deadlock (gh-115634)
            import contextlib

            for pid in worker_pids:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.kill(pid, signal.SIGKILL)

    finally:
        if out:
            out.close()
        if fail_out:
            fail_out.close()
        # Restore original signal handler
        signal.signal(signal.SIGTERM, _original_sigterm)

    elapsed = time.perf_counter() - t0

    # Only raise if NOT a graceful stop (SIGTERM)
    if not stop.is_set():
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
