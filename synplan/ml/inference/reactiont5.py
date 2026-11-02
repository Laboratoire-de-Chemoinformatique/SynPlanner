"""ReactionT5v2 USPTO-MIT: blank-reagent forward prediction with ONNX by default.

The tokenizer and checkpoint are from sagawa/ReactionT5v2-forward-USPTO_MIT
(MIT, Tatsuya Sagawa et al.). Beam scoring follows Transformers 4.40.2's
BeamSearchScorer semantics: length-normalized hypotheses, EOS admission among
the first num_beams candidates, and its default heuristic early stopping.
The sequence log likelihood returned here is the unnormalized sum; shared
augmentation ranking scores are reported separately.
"""

import json
from pathlib import Path

import numpy as np

from .forward import predict_augmented, prediction_identity

INPUT_TEMPLATE = "REACTANT:{}REAGENT: "
GENERATION_VERSION = 1


class _Tokenizer:
    def __init__(self, path, max_length):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.max_length = max_length

    def encode(self, reactants):
        rows = self.tokenizer.encode_batch(
            [INPUT_TEMPLATE.format(s) for s in reactants]
        )
        length = max(len(row.ids) for row in rows)
        if length > self.max_length:
            raise ValueError(
                f"Input has {length} tokens; maximum is {self.max_length}. No truncation applied."
            )
        ids = np.zeros((len(rows), length), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, : len(row.ids)] = row.ids
            mask[i, : len(row.ids)] = 1
        return ids, mask

    def decode(self, ids):
        return (
            self.tokenizer.decode(list(map(int, ids)), skip_special_tokens=True)
            .replace(" ", "")
            .rstrip(".")
        )


def _beam_search(decode, mask, beam_width, max_length, *, length_penalty=1.0):
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


class ReactionT5:
    """Portable USPTO-MIT bundle; optional explicit ``backend='ckpt'``.

    Both runtimes use the exported tokenizer, blank reagents, identical beam
    control and shared augmentation. The ONNX path imports neither torch nor
    Transformers. CUDA must be explicitly requested and successfully initialized.
    """

    def __init__(
        self,
        model_dir,
        device="cpu",
        threads=4,
        *,
        backend="onnx",
        augmentation_size=10,
        seed=0,
    ):
        if device not in ("cpu", "cuda") or backend not in ("onnx", "ckpt"):
            raise ValueError("Choose cpu/cuda and onnx/ckpt")
        if threads < 1 or not 1 <= augmentation_size <= 10:
            raise ValueError("Choose positive threads and 1-10 augmentations")
        self.directory, self.device, self.backend = Path(model_dir), device, backend
        self.augmentation_size, self.seed = augmentation_size, seed
        self.metadata = json.loads(
            (self.directory / "model.json").read_text(encoding="utf-8")
        )
        self.max_length = self.metadata["inference"]["max_input_length"]
        self.max_generated_length = self.metadata["inference"]["max_length"]
        self.length_penalty = self.metadata["inference"]["length_penalty"]
        self.tokenizer = _Tokenizer(
            self.directory / "source/tokenizer.json", self.max_length
        )
        self.runtime_files = ["model.json", "source/tokenizer.json"]
        layers = self.metadata["architecture"]["num_decoder_layers"]
        self.cache_names = [
            f"past_{i}_{kind}"
            for i in range(layers)
            for kind in ("key", "value", "cross_key", "cross_value")
        ]
        if backend == "ckpt":
            import torch

            from ._reactiont5_torch import DecoderGraph, EncoderGraph, load_network

            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device is unavailable")
            torch.set_num_threads(threads)
            self.network = load_network(self.directory / "source", device)
            self.encoder = EncoderGraph(self.network)
            self.decoder = DecoderGraph(self.network)
            self.cached_decoder = DecoderGraph(self.network, cached=True)
            self.runtime_files.extend(
                ["source/config.json", "source/model.safetensors"]
            )
            self.providers = [device]
        else:
            import onnxruntime as ort

            provider = (
                "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
            )
            if provider not in ort.get_available_providers():
                raise RuntimeError(f"Requested provider is unavailable: {provider}")
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
            providers = (
                [
                    (
                        provider,
                        {"use_tf32": "0", "arena_extend_strategy": "kSameAsRequested"},
                    )
                ]
                if device == "cuda"
                else [provider]
            )
            self.arena_cleanup = ort.RunOptions()
            self.arena_cleanup.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage",
                "gpu:0" if device == "cuda" else "cpu:0",
            )
            self.sessions = {}
            for name, files in self.metadata["onnx_files"].items():
                session = ort.InferenceSession(
                    str(self.directory / files[0]),
                    sess_options=options,
                    providers=providers,
                )
                if provider not in session.get_providers():
                    raise RuntimeError(
                        f"Requested provider failed to initialize: {provider}"
                    )
                self.sessions[name] = session
                self.runtime_files.extend(files)
            self.providers = self.sessions["decoder_with_past"].get_providers()

    def prediction_identity(self, *, beam_width=20, num_results=10):
        from importlib.metadata import version

        identity = prediction_identity(
            self, beam_width=beam_width, num_results=num_results
        )
        identity["generation"] = {
            "version": GENERATION_VERSION,
            "input_template": INPUT_TEMPLATE,
            "length_penalty": self.length_penalty,
            "early_stopping": False,
            "min_length": 1,
            "decoder_start_token_id": 0,
            "eos_token_id": 1,
        }
        identity["runtime"]["tokenizers"] = version("tokenizers")
        if self.backend == "ckpt":
            identity["runtime"]["transformers"] = version("transformers")
        return identity

    def predict(self, reactants, beam_width=20, batch_size=16, *, num_results=10):
        if not 1 <= beam_width < self.metadata["architecture"]["vocab_size"]:
            raise ValueError(
                "Beam width must be positive and smaller than vocabulary size"
            )
        return predict_augmented(
            self, reactants, beam_width, batch_size, num_results=num_results
        )

    def _run_onnx(self, name, inputs):
        session = self.sessions[name]
        binding = session.io_binding()
        for key, value in inputs.items():
            if isinstance(value, np.ndarray):
                binding.bind_cpu_input(key, value)
            else:
                binding.bind_ortvalue_input(key, value)
        for output in session.get_outputs():
            binding.bind_output(output.name, self.device)
        # Variable-length K/V states otherwise leave large, unused arena regions
        # behind across a collection. Periodic cleanup keeps live caches on device.
        cleanup = (
            name != "decoder_with_past" or inputs["past_0_key"].shape()[2] % 16 == 1
        )
        session.run_with_iobinding(binding, self.arena_cleanup if cleanup else None)
        return binding.get_outputs()

    def _encode(self, ids, mask):
        if self.backend == "onnx":
            return self._run_onnx(
                "encoder", {"input_ids": ids, "attention_mask": mask}
            )[0]
        import torch

        with torch.inference_mode():
            return self.encoder(
                torch.as_tensor(ids, device=self.device),
                torch.as_tensor(mask, device=self.device),
            )

    def _decode(self, ids, mask, memory, past=None, parents=None):
        if self.backend == "onnx":
            inputs = {"input_ids": ids, "attention_mask": mask}
            if past is None:
                name = "decoder"
                inputs["memory"] = memory
            else:
                name = "decoder_with_past"
                inputs.update(zip(self.cache_names, past))
                inputs["beam_indices"] = parents
            output = self._run_onnx(name, inputs)
            return output[0].numpy(), output[1:]
        import torch

        with torch.inference_mode():
            ids, mask = (
                torch.as_tensor(value, device=self.device) for value in (ids, mask)
            )
            if past is None:
                output = self.decoder(ids, mask, memory)
            else:
                output = self.cached_decoder(
                    ids, mask, torch.as_tensor(parents, device=self.device), *past
                )
            return output[0].cpu().numpy(), output[1:]

    def _predict_batch(self, reactants, beam_width):
        ids, mask = self.tokenizer.encode(reactants)
        memory = self._encode(ids, mask)
        hypotheses = _beam_search(
            lambda ids, mask, past, parents: self._decode(
                ids, mask, memory, past, parents
            ),
            mask,
            beam_width,
            self.max_generated_length,
            length_penalty=self.length_penalty,
        )
        return [
            {
                "reactants": value,
                "predictions": [self.tokenizer.decode(row) for _, _, row in beams],
                "log_likelihoods": [score for _, score, _ in beams],
                "sequence_scores": [score for score, _, _ in beams],
            }
            for value, beams in zip(reactants, hypotheses, strict=True)
        ]
