"""ForwardChimera DeNovo inference without RetroChimera or Lightning.

The incremental beam search follows RetroChimera v1.3.0 / OpenNMT-py 3.5.1.
Copyright (c) Microsoft Corporation and contributors; 2017-Present OpenNMT.
MIT License: permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
"AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Modifications: NumPy beam control, optional standalone PyTorch network, ONNX
KV-cache binding and reordering, and shared stereo-free prediction pipeline.
"""

import json
import re
from pathlib import Path

import numpy as np

from .forward import predict_augmented

TOKEN_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class _Tokenizer:
    def __init__(self, path, max_length):
        self.tokens = path.read_text().splitlines()
        self.ids = {token: index for index, token in enumerate(self.tokens)}
        self.regex = re.compile(TOKEN_PATTERN)
        self.max_length = max_length

    def encode(self, values):
        rows = []
        for value in values:
            tokens = self.regex.findall(value)
            if not value or "".join(tokens) != value:
                raise ValueError("Input could not be tokenized completely")
            if len(tokens) > self.max_length:
                raise ValueError(
                    f"Input has {len(tokens)} tokens; maximum is {self.max_length}. No truncation applied."
                )
            rows.append([self.ids.get(token, 3) for token in tokens])
        ids = np.full((len(rows), max(map(len, rows))), 2, dtype=np.int64)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = row
        return ids, ids == 2

    def decode(self, ids):
        if len(ids) and ids[0] == 0:
            ids = ids[1:]
        result = []
        for index in ids:
            if index == 1:
                break
            result.append(self.tokens[index])
        return "".join(result)


class ForwardChimera:
    """Portable model directory; ONNX by default, explicit ``backend='ckpt'``.

    PyTorch is imported only for checkpoint inference. CUDA must be explicitly
    requested and available; neither backend silently falls back to CPU.
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
        self.metadata = json.loads((self.directory / "model.json").read_text())
        self.max_length = self.metadata["architecture"]["max_seq_len"]
        self.max_generated_length = self.metadata["inference"]["max_generated_seq_len"]
        self.tokenizer = _Tokenizer(
            self.directory / "source/vocab.txt", self.max_length
        )
        self.runtime_files = ["model.json", "source/vocab.txt"]
        if backend == "ckpt":
            import torch

            from ._chimera_torch import ChimeraNetwork

            torch.set_num_threads(threads)
            self.network = ChimeraNetwork.load(
                self.directory / "source/combined.ckpt", device
            )
            self.runtime_files.append("source/combined.ckpt")
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
                [(provider, {"use_tf32": "0"})] if device == "cuda" else [provider]
            )
            self.sessions = {}
            for name in ("encoder", "decoder", "decoder_with_past"):
                filename = f"{name}.onnx"
                session = ort.InferenceSession(
                    str(self.directory / filename),
                    sess_options=options,
                    providers=providers,
                )
                if provider not in session.get_providers():
                    raise RuntimeError(
                        f"Requested provider failed to initialize: {provider}"
                    )
                self.sessions[name] = session
                self.runtime_files.append(filename)
            self.providers = self.sessions["decoder_with_past"].get_providers()
        layers = self.metadata["architecture"]["n_layers"]
        self.cache_names = [
            f"past_{i}_{kind}"
            for i in range(layers)
            for kind in ("key", "value", "cross_key", "cross_value")
        ]

    def predict(self, reactants, beam_width=20, batch_size=16, *, num_results=10):
        if not 1 <= beam_width <= len(self.tokenizer.tokens):
            raise ValueError("Beam width must be between 1 and vocabulary size")
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
        session.run_with_iobinding(binding)
        return binding.get_outputs()

    def _encode(self, ids, mask):
        if self.backend == "onnx":
            return self._run_onnx("encoder", {"input_ids": ids, "padding_mask": mask})[
                0
            ]
        import torch

        with torch.inference_mode():
            return self.network.encode(
                torch.as_tensor(ids, device=self.device),
                torch.as_tensor(mask, device=self.device),
            )

    def _decode(self, ids, mask, position, memory=None, past=None, parents=None):
        if self.backend == "onnx":
            inputs = {"input_ids": ids, "padding_mask": mask}
            if past is None:
                inputs["memory"] = memory
                name = "decoder"
            else:
                inputs.update(zip(self.cache_names, past))
                inputs.update(
                    position=np.asarray(position, dtype=np.int64), beam_indices=parents
                )
                name = "decoder_with_past"
            output = self._run_onnx(name, inputs)
            return output[0].numpy(), output[1:]
        import torch

        with torch.inference_mode():
            output = self.network.decode(
                torch.as_tensor(ids, device=self.device),
                torch.as_tensor(mask, device=self.device),
                position,
                memory=memory,
                past=past,
                beam_indices=None
                if parents is None
                else torch.as_tensor(parents, device=self.device),
            )
            return output[0].cpu().numpy(), output[1:]

    def _predict_batch(self, reactants, beam_width):
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
                prefix[:, -1:], mask, position, memory, past, parents
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
                    "predictions": [self.tokenizer.decode(row) for _, row in selected],
                    "log_likelihoods": [score for score, _ in selected],
                }
            )
        return result
