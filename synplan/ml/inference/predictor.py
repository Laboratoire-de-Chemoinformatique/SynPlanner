"""Unified forward-model loading and execution.

Adapted from chemformer-workflows / MolecularAI Chemformer (Apache-2.0),
Microsoft RetroChimera v1.3.0 / OpenNMT-py 3.5.1 (MIT), and ReactionT5v2
(Tatsuya Sagawa et al., MIT). T5 beam semantics follow Transformers 4.40.2.
Copyright (c) Microsoft Corporation and contributors; 2017-Present OpenNMT.
Modifications: unified inference plumbing with model-specific generation rules.
See the repository LICENSE for the full third-party terms and attribution.
"""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import beam_search
from .forward import predict_augmented, prediction_identity
from .tokenizer import INPUT_TEMPLATE, Tokenizer


class ForwardPredictor:
    """Read a portable model bundle and predict products without training code.

    Select ``chemformer``, ``chimera`` or ``reactiont5`` explicitly. ONNX is the
    default; Chimera and ReactionT5 also support ``backend='ckpt'``. CPU uses
    four threads by default. CUDA must be requested and initialize successfully.
    ``decoder_batch_size`` limits Chemformer beam rows per decoder invocation.
    """

    def __init__(
        self,
        model_dir,
        device="cpu",
        threads=4,
        *,
        model_type="chemformer",
        backend="onnx",
        decoder_batch_size=64,
        augmentation_size=10,
        seed=0,
    ):
        if model_type not in ("chemformer", "chimera", "reactiont5"):
            raise ValueError("Choose chemformer, chimera or reactiont5")
        if device not in ("cpu", "cuda") or backend not in ("onnx", "ckpt"):
            raise ValueError("Choose cpu/cuda and onnx/ckpt")
        if threads < 1 or decoder_batch_size < 1 or not 1 <= augmentation_size <= 10:
            raise ValueError(
                "Choose positive thread and decoder batch counts, and 1-10 augmentations"
            )
        if model_type == "chemformer" and backend != "onnx":
            raise ValueError("Chemformer currently supports only the ONNX backend")
        self.directory, self.device, self.backend = Path(model_dir), device, backend
        self.model_type = model_type
        # Keep persisted cache identifiers independent of Python class names.
        self.identity_name = {
            "chemformer": "OnnxChemformer",
            "chimera": "ForwardChimera",
            "reactiont5": "ReactionT5",
        }[model_type]
        self.augmentation_size, self.seed = augmentation_size, seed
        self.metadata = json.loads(
            (self.directory / "model.json").read_text(encoding="utf-8")
        )
        architecture = self.metadata["architecture"]
        self.kv_cache = (
            self.metadata.get("kv_cache", False) if model_type == "chemformer" else True
        )
        if model_type == "chemformer":
            self.max_length = architecture["max_seq_len"]
            self.decoder_batch_size = decoder_batch_size
            self.runtime_files = [
                "encoder.onnx",
                "decoder.onnx",
                "vocabulary.json",
                "model.json",
            ]
            if self.kv_cache:
                self.runtime_files.append("decoder_with_past.onnx")
            layers = architecture.get("num_layers", 0)
        elif model_type == "chimera":
            self.max_length = architecture["max_seq_len"]
            self.max_generated_length = self.metadata["inference"][
                "max_generated_seq_len"
            ]
            self.runtime_files = ["model.json", "source/vocab.txt"]
            layers = architecture["n_layers"]
        else:
            self.max_length = self.metadata["inference"]["max_input_length"]
            self.max_generated_length = self.metadata["inference"]["max_length"]
            self.length_penalty = self.metadata["inference"]["length_penalty"]
            self.runtime_files = ["model.json", "source/tokenizer.json"]
            layers = architecture["num_decoder_layers"]
        self.tokenizer = Tokenizer(self.directory, model_type, self.max_length)
        self.cache_names = [
            f"past_{i}_{kind}"
            for i in range(layers)
            for kind in ("key", "value", "cross_key", "cross_value")
        ]
        if backend == "onnx":
            self._load_onnx(threads)
        else:
            self._load_checkpoint(threads)

    def _load_onnx(self, threads):
        import onnxruntime as ort

        provider = (
            "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
        )
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"Requested provider is unavailable: {provider}")
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        provider_options = {"use_tf32": "0"}
        if self.model_type == "reactiont5":
            provider_options["arena_extend_strategy"] = "kSameAsRequested"
            self.arena_cleanup = ort.RunOptions()
            self.arena_cleanup.add_run_config_entry(
                "memory.enable_memory_arena_shrinkage",
                "gpu:0" if self.device == "cuda" else "cpu:0",
            )
            files = self.metadata["onnx_files"]
        else:
            names = ["encoder", "decoder"]
            if self.kv_cache:
                names.append("decoder_with_past")
            files = {name: [f"{name}.onnx"] for name in names}
        providers = (
            [(provider, provider_options)] if self.device == "cuda" else [provider]
        )
        self.sessions = {}
        for name, paths in files.items():
            session = ort.InferenceSession(
                str(self.directory / paths[0]),
                sess_options=options,
                providers=providers,
            )
            if provider not in session.get_providers():
                raise RuntimeError(
                    f"Requested provider failed to initialize: {provider}; check CUDA/cuDNN libraries"
                )
            self.sessions[name] = session
            if self.model_type != "chemformer":
                self.runtime_files.extend(paths)
        if self.model_type == "chemformer":
            self.encoder, self.decoder = (
                self.sessions["encoder"],
                self.sessions["decoder"],
            )
            if self.kv_cache:
                self.decoder_with_past = self.sessions["decoder_with_past"]
            self.providers = {
                name: session.get_providers() for name, session in self.sessions.items()
            }
        else:
            self.providers = self.sessions["decoder_with_past"].get_providers()

    def _load_checkpoint(self, threads):
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        torch.set_num_threads(threads)
        if self.model_type == "chimera":
            from ._chimera_torch import ChimeraNetwork

            self.network = ChimeraNetwork.load(
                self.directory / "source/combined.ckpt", self.device
            )
            self.runtime_files.append("source/combined.ckpt")
        else:
            from ._reactiont5_torch import DecoderGraph, EncoderGraph, load_network

            self.network = load_network(self.directory / "source", self.device)
            self.encoder = EncoderGraph(self.network)
            self.decoder = DecoderGraph(self.network)
            self.cached_decoder = DecoderGraph(self.network, cached=True)
            self.runtime_files.extend(
                ["source/config.json", "source/model.safetensors"]
            )
        self.providers = [self.device]

    def prediction_identity(self, *, beam_width=20, num_results=10):
        identity = prediction_identity(
            self, beam_width=beam_width, num_results=num_results
        )
        if self.model_type == "reactiont5":
            identity["generation"] = {
                "version": 1,
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
        """Return ordered distinct products, likelihoods and augmentation scores."""
        if self.model_type == "reactiont5":
            if not 1 <= beam_width < self.metadata["architecture"]["vocab_size"]:
                raise ValueError(
                    "Beam width must be positive and smaller than vocabulary size"
                )
        elif not 1 <= beam_width <= len(self.tokenizer.tokens):
            raise ValueError("Beam width must be between 1 and vocabulary size")
        return predict_augmented(
            self, reactants, beam_width, batch_size, num_results=num_results
        )

    def _predict_batch(self, reactants, beam_width):
        if self.model_type == "chemformer":
            return beam_search.chemformer(self, reactants, beam_width)
        if self.model_type == "chimera":
            return beam_search.chimera(self, reactants, beam_width)
        ids, padding = self.tokenizer.encode(reactants)
        mask = (~padding).astype(np.int64)
        memory = self._encode(ids, mask)
        hypotheses = beam_search.reactiont5(
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
                "predictions": self.tokenizer.decode([row for _, _, row in beams]),
                "log_likelihoods": [score for _, score, _ in beams],
                "sequence_scores": [score for score, _, _ in beams],
            }
            for value, beams in zip(reactants, hypotheses, strict=True)
        ]

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
        if self.model_type == "reactiont5":
            # Release unused arena regions periodically without moving live K/V.
            cleanup = (
                name != "decoder_with_past" or inputs["past_0_key"].shape()[2] % 16 == 1
            )
            session.run_with_iobinding(binding, self.arena_cleanup if cleanup else None)
        else:
            session.run_with_iobinding(binding)
        return binding.get_outputs()

    def _encode(self, ids, mask):
        if self.backend == "onnx":
            mask_name = (
                "attention_mask" if self.model_type == "reactiont5" else "padding_mask"
            )
            return self._run_onnx("encoder", {"input_ids": ids, mask_name: mask})[0]
        import torch

        with torch.inference_mode():
            ids, mask = (
                torch.as_tensor(value, device=self.device) for value in (ids, mask)
            )
            if self.model_type == "chimera":
                return self.network.encode(ids, mask)
            return self.encoder(ids, mask)

    def _decode(
        self, ids, mask, memory=None, past=None, parents=None, *, position=None
    ):
        if self.backend == "onnx":
            mask_name = (
                "attention_mask" if self.model_type == "reactiont5" else "padding_mask"
            )
            inputs = {"input_ids": ids, mask_name: mask}
            name = "decoder" if past is None else "decoder_with_past"
            if past is None:
                inputs["memory"] = memory
            else:
                inputs.update(zip(self.cache_names, past, strict=True))
                inputs["beam_indices"] = parents
                if self.model_type == "chimera":
                    inputs["position"] = np.asarray(position, dtype=np.int64)
            output = self._run_onnx(name, inputs)
            return output[0].numpy(), output[1:]
        import torch

        with torch.inference_mode():
            ids, mask = (
                torch.as_tensor(value, device=self.device) for value in (ids, mask)
            )
            parents = (
                None
                if parents is None
                else torch.as_tensor(parents, device=self.device)
            )
            if self.model_type == "chimera":
                output = self.network.decode(
                    ids, mask, position, memory=memory, past=past, beam_indices=parents
                )
            elif past is None:
                output = self.decoder(ids, mask, memory)
            else:
                output = self.cached_decoder(ids, mask, parents, *past)
            return output[0].cpu().numpy(), output[1:]

    def _decode_full_prefix(self, prefix, memory, mask):
        return np.concatenate(
            [
                self.decoder.run(
                    None,
                    {
                        "decoder_input_ids": prefix[
                            start : start + self.decoder_batch_size
                        ],
                        "memory": memory[start : start + self.decoder_batch_size],
                        "memory_padding_mask": mask[
                            start : start + self.decoder_batch_size
                        ],
                    },
                )[0]
                for start in range(0, len(prefix), self.decoder_batch_size)
            ],
            axis=0,
        )

    def _decode_cached(self, prefix, memory, mask, past, parents):
        """Single-token decoding; each chunk gathers its parents inside ONNX."""
        name = "decoder" if past is None else "decoder_with_past"
        chunks = []
        for start in range(0, len(prefix), self.decoder_batch_size):
            rows = slice(start, start + self.decoder_batch_size)
            inputs = {
                "input_ids": np.ascontiguousarray(prefix[rows, -1:]),
                "padding_mask": mask[rows],
            }
            if past is None:
                inputs["memory"] = memory[rows]
            else:
                inputs.update(zip(self.cache_names, past, strict=True))
                inputs["beam_indices"] = parents[rows]
                inputs["position"] = np.asarray(prefix.shape[1] - 1, dtype=np.int64)
            chunks.append(self._run_onnx(name, inputs))
        scores = np.concatenate([chunk[0].numpy() for chunk in chunks])
        if len(chunks) == 1:
            return scores, chunks[0][1:]
        # CPU OrtValue.numpy() exposes the same memory. Only joining chunk outputs
        # allocates a new array; parent gathering remains inside the cached graph.
        state = [
            np.concatenate([chunk[i].numpy() for chunk in chunks])
            for i in range(1, len(chunks[0]))
        ]
        return scores, state
