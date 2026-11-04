"""Model-specific beam search; scores and stopping rules retain upstream behavior.

Adapted from chemformer-workflows / MolecularAI Chemformer (Apache-2.0),
Microsoft RetroChimera v1.3.0 / OpenNMT-py 3.5.1 (MIT), and ReactionT5v2
(Tatsuya Sagawa et al., MIT). T5 beam semantics follow Transformers 4.40.2.
Copyright (c) Microsoft Corporation and contributors; 2017-Present OpenNMT.
Modifications: unified inference plumbing with model-specific generation rules.
See the repository LICENSE for the full third-party terms and attribution.
"""

import numpy as np


def _topk(values, count):
    # Preserve the original sampler's stable ordering for tied scores.
    indices = np.argsort(-values, axis=-1, kind="stable")[..., :count]
    return np.take_along_axis(values, indices, axis=-1), indices


def chemformer(self, reactants, beam_width):
    ids, mask = self.tokenizer.encode(reactants)
    memory = self.encoder.run(None, {"input_ids": ids, "padding_mask": mask})[0]
    batch, vocabulary = len(ids), len(self.tokenizer.tokens)
    start, end, pad = (self.tokenizer.special[k] for k in ("start", "end", "pad"))
    prefix = np.full((batch, 1), start, dtype=np.int64)
    scores = None
    done = np.zeros(batch, dtype=bool)
    past, parents = None, None
    for position in range(self.max_length - 2):
        if done.all():
            break
        if self.kv_cache:
            actions, past = self._decode_cached(prefix, memory, mask, past, parents)
        else:
            actions = self._decode_full_prefix(prefix, memory, mask)
        if position == 0:
            scores, tokens = _topk(actions, beam_width)
            parents = np.repeat(np.arange(batch, dtype=np.int64), beam_width)
            memory = np.repeat(memory, beam_width, axis=0)
            mask = np.repeat(mask, beam_width, axis=0)
            prefix = np.repeat(prefix, beam_width, axis=0)
            prefix = np.concatenate((prefix, tokens.reshape(-1, 1)), axis=1)
            scores = scores.reshape(-1, 1)
        else:
            actions[done, :] = min(float(scores.min()), float(actions.min())) - 1
            actions[done, end] = 0
            combined = (scores + actions).reshape(batch, beam_width * vocabulary)
            scores, indices = _topk(combined, beam_width)
            parents, tokens = indices // vocabulary, indices % vocabulary
            prefix = prefix.reshape(batch, beam_width, -1)[
                np.arange(batch)[:, None], parents
            ].reshape(batch * beam_width, -1)
            parents = (parents + np.arange(batch)[:, None] * beam_width).reshape(-1)
            prefix = np.concatenate((prefix, tokens.reshape(-1, 1)), axis=1)
            scores = scores.reshape(-1, 1)
        done = np.any(prefix == end, axis=1)
    tail = np.where(done, pad, end).astype(np.int64)
    if not done.all():
        if self.kv_cache:
            actions, past = self._decode_cached(prefix, memory, mask, past, parents)
        else:
            actions = self._decode_full_prefix(prefix, memory, mask)
        scores += (actions[np.arange(len(prefix)), tail] * (~done)).reshape(-1, 1)
    prefix = np.concatenate((prefix, tail[:, None]), axis=1)
    predictions = (
        np.asarray(self.tokenizer.decode(prefix)).reshape(batch, beam_width).tolist()
    )
    scores = scores.reshape(batch, beam_width).tolist()
    return [
        {"reactants": s, "predictions": p, "log_likelihoods": ll}
        for s, p, ll in zip(reactants, predictions, scores)
    ]


def chimera(self, reactants, beam_width):
    """Raw beams, including finished hypotheses subsequently pruned upstream."""
    ids, mask = self.tokenizer.encode(reactants)
    memory = self._encode(ids, mask)
    active = np.arange(len(ids))
    prefix = np.zeros((len(ids), 1), dtype=np.int64)
    scores = np.zeros((len(ids), 1), dtype=np.float32)
    history = [[] for _ in ids]
    past, parents = None, None
    vocabulary = len(self.tokenizer.tokens)
    for position in range(self.max_generated_length):
        actions, past = self._decode(
            prefix[:, -1:], mask, memory, past, parents, position=position
        )
        ended = prefix[:, -1] == 1
        actions[ended] = -1e5
        actions[ended, 1] = 0
        combined = (actions + scores.reshape(-1, 1)).reshape(len(active), -1)
        indices = np.argsort(-combined, axis=1, kind="stable")[:, :beam_width]
        scores = np.take_along_axis(combined, indices, axis=1)
        previous_width = len(prefix) // len(active)
        parents = (
            indices // vocabulary + np.arange(len(active))[:, None] * previous_width
        ).reshape(-1)
        tokens = (indices % vocabulary).reshape(-1, 1)
        prefix = np.concatenate((prefix[parents], tokens), axis=1)
        mask = mask[parents]
        finished = tokens.reshape(len(active), beam_width) == 1
        if position + 1 == self.max_generated_length:
            finished[:] = True
        sequences = prefix.reshape(len(active), beam_width, -1)
        keep = []
        for index, original in enumerate(active):
            for beam in np.flatnonzero(finished[index]):
                sequence = sequences[index, beam]
                if sequence[-2] != 1:
                    history[original].append(
                        (float(scores[index, beam]), sequence[1:].copy())
                    )
            if not (finished[index].all() and len(history[original]) >= beam_width):
                keep.append(index)
        if not keep:
            break
        active = active[keep]
        scores = scores[keep]
        prefix = sequences[keep].reshape(-1, prefix.shape[-1])
        mask = mask.reshape(-1, beam_width, mask.shape[-1])[keep].reshape(
            len(prefix), -1
        )
        parents = parents.reshape(-1, beam_width)[keep].reshape(-1)
    result = []
    for value, candidates in zip(reactants, history, strict=True):
        candidates.sort(key=lambda item: -item[0])
        selected = candidates[:beam_width]
        result.append(
            {
                "reactants": value,
                "predictions": self.tokenizer.decode([row for _, row in selected]),
                "log_likelihoods": [score for score, _ in selected],
            }
        )
    return result


def reactiont5(decode, mask, beam_width, max_length, *, length_penalty=1.0):
    """T5 beam control, shared by both runtimes; start token 0 and EOS 1.

    Initial decoding runs once per input, then the decoder expands/reorders its
    cached states with the selected parents. Finished inputs leave the batch.
    ``max_length`` includes the decoder start token, as in Transformers.
    """
    active = np.arange(len(mask))
    prefix = np.zeros((len(mask), 1), dtype=np.int64)
    scores = np.zeros((len(mask), 1), dtype=np.float32)
    hypotheses = [[] for _ in mask]
    past, parents = None, None
    while prefix.shape[1] < max_length:
        logits, past = decode(prefix[:, -1:], mask, past, parents)
        logits = logits - logits.max(axis=-1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        vocabulary = logits.shape[-1]
        combined = (log_probs + scores.reshape(-1, 1)).reshape(len(active), -1)
        order = np.argsort(-combined, axis=1, kind="stable")[:, : 2 * beam_width]
        previous_width = len(prefix) // len(active)
        next_parents, next_tokens, next_scores, keep = [], [], [], []
        # Includes the candidate token, excludes the decoder start.
        generated_length = prefix.shape[1]
        for index, original in enumerate(active):
            completed = hypotheses[original]
            alive = []
            for rank, flat in enumerate(order[index]):
                parent = index * previous_width + int(flat) // vocabulary
                token = int(flat) % vocabulary
                score = float(combined[index, flat])
                if token == 1:
                    if rank < beam_width:
                        normalized = score / generated_length**length_penalty
                        if len(completed) < beam_width or normalized > completed[-1][0]:
                            completed.append(
                                (normalized, score, np.append(prefix[parent], token))
                            )
                            completed.sort(key=lambda item: -item[0])
                            del completed[beam_width:]
                else:
                    alive.append((parent, token, score))
                if len(alive) == beam_width:
                    break
            best_possible = (
                float(combined[index, order[index, 0]])
                / generated_length**length_penalty
            )
            if len(completed) == beam_width and completed[-1][0] >= best_possible:
                continue
            keep.append(index)
            for parent, token, score in alive:
                next_parents.append(parent)
                next_tokens.append(token)
                next_scores.append(score)
        if not keep:
            active = active[:0]
            break
        parents = np.asarray(next_parents, dtype=np.int64)
        prefix = np.concatenate(
            (prefix[parents], np.asarray(next_tokens)[:, None]), axis=1
        )
        mask = mask[parents]
        scores = np.asarray(next_scores, dtype=np.float32).reshape(
            len(keep), beam_width
        )
        active = active[keep]
    for index, original in enumerate(active):
        for beam, score in enumerate(scores[index]):
            row = prefix[index * beam_width + beam]
            hypotheses[original].append(
                (float(score) / (len(row) - 1) ** length_penalty, float(score), row)
            )
        hypotheses[original].sort(key=lambda item: -item[0])
        del hypotheses[original][beam_width:]
    return hypotheses
