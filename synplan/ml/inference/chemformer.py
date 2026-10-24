"""Chemformer forward prediction from an encoder/decoder ONNX bundle.

synplan/ml/inference/chemformer.py adapts the portable ONNX inference wrapper from
chemformer-workflows (Apache License, Version 2.0).
Original source: chemformer_workflows/integrations/onnx_inference.py, distributed
with the SynFwd-1 model as predict.py.
Changes: chython-based mapping removal, a batched Python API, and removal of the
standalone command-line interface.

Chemformer architecture and vocabulary: MolecularAI/Chemformer, Apache-2.0.
V2 implementation: MolecularAI/aizynthmodels, Apache-2.0.
Export adapter and portable inference wrapper: chemformer-workflows, Apache-2.0.
The full Apache License, Version 2.0 follows.
"""

#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright [yyyy] [name of copyright owner]
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from synplan.ml.inference.forward import canonical_reactants, predict_augmented

__all__ = ["OnnxChemformer", "canonical_reactants"]


class _Tokenizer:
    def __init__(self, vocabulary: Path, max_length: int):
        data = json.loads(vocabulary.read_text())
        self.tokens = data["vocabulary"]
        self.ids = {token: i for i, token in enumerate(self.tokens)}
        self.special = {
            name: self.ids[token]
            for name, token in data["properties"]["special_tokens"].items()
        }
        self.regex = re.compile(data["properties"]["regex"])
        self.max_length = max_length

    def encode(self, reactants):
        encoded = []
        for value in reactants:
            tokens = self.regex.findall(value)
            if not value or "".join(tokens) != value:
                raise ValueError("Input could not be tokenized completely")
            ids = (
                [self.special["start"]]
                + [self.ids.get(t, self.special["unknown"]) for t in tokens]
                + [self.special["end"]]
            )
            if len(ids) > self.max_length:
                raise ValueError(
                    f"Input has {len(ids)} tokens; maximum is {self.max_length}. "
                    "No truncation applied."
                )
            encoded.append(ids)
        width = max(map(len, encoded))
        array = np.full((len(encoded), width), self.special["pad"], dtype=np.int64)
        mask = np.ones(array.shape, dtype=bool)
        for i, ids in enumerate(encoded):
            array[i, : len(ids)] = ids
            mask[i, : len(ids)] = False
        return array, mask

    def decode(self, rows):
        result = []
        controls = {self.special[k] for k in ("pad", "start", "end")}
        for row in rows:
            ids = list(map(int, row))
            if self.special["end"] in ids:
                ids = ids[: ids.index(self.special["end"]) + 1]
            while ids and ids[0] in controls:
                ids.pop(0)
            while ids and ids[-1] in controls:
                ids.pop()
            result.append("".join(self.tokens[i] for i in ids))
        return result


def _topk(values, count):
    # Preserve the original sampler's stable ordering for tied scores.
    indices = np.argsort(-values, axis=-1, kind="stable")[..., :count]
    return np.take_along_axis(values, indices, axis=-1), indices


class OnnxChemformer:
    """Forward prediction using the portable Chemformer encoder/decoder format.

    ``model_dir`` contains encoder.onnx, decoder.onnx, vocabulary.json and
    model.json. Sessions are loaded once; no Python code from the bundle is run.
    CPU needs only SynPlanner's base dependencies. CUDA requires a separately
    installed ONNX Runtime CUDA provider and fails if it cannot initialize.

    :param model_dir: Local model bundle directory, not a single ONNX graph.
    :param device: ``"cpu"`` (default) or ``"cuda"``.
    :param threads: ONNX Runtime threads per session.
    :param decoder_batch_size: Maximum beam rows per decoder invocation.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        threads: int = 4,
        decoder_batch_size: int = 64,
        augmentation_size: int = 10,
        seed: int = 0,
    ):
        import onnxruntime as ort

        if (
            device not in ("cpu", "cuda")
            or threads < 1
            or decoder_batch_size < 1
            or not 1 <= augmentation_size <= 10
        ):
            raise ValueError(
                "Choose cpu/cuda, positive thread and decoder batch counts, and 1-10 augmentations"
            )
        self.augmentation_size = augmentation_size
        self.seed = seed
        self.backend = "onnx"
        self.runtime_files = (
            "encoder.onnx",
            "decoder.onnx",
            "vocabulary.json",
            "model.json",
        )
        self.directory = Path(model_dir)
        self.metadata = json.loads((self.directory / "model.json").read_text())
        self.max_length = self.metadata["architecture"]["max_seq_len"]
        self.tokenizer = _Tokenizer(self.directory / "vocabulary.json", self.max_length)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        provider = (
            "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        )
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"Requested provider is unavailable: {provider}")
        providers = [(provider, {"use_tf32": "0"})] if device == "cuda" else [provider]
        self.encoder = ort.InferenceSession(
            str(self.directory / "encoder.onnx"),
            sess_options=options,
            providers=providers,
        )
        self.decoder = ort.InferenceSession(
            str(self.directory / "decoder.onnx"),
            sess_options=options,
            providers=providers,
        )
        if any(
            provider not in session.get_providers()
            for session in (self.encoder, self.decoder)
        ):
            raise RuntimeError(
                f"Requested provider failed to initialize: {provider}; "
                "check CUDA/cuDNN libraries"
            )
        self.decoder_batch_size = decoder_batch_size
        self.providers = {
            name: getattr(self, name).get_providers() for name in ("encoder", "decoder")
        }

    def predict(
        self,
        reactants: Sequence[str],
        beam_width: int = 20,
        batch_size: int = 16,
        *,
        num_results: int = 10,
    ) -> list[dict]:
        """Return up to ten stereo-free unique products from augmented beam search.

        ``beam_width`` applies to each rooted input; ``num_results`` limits the
        final list. Likelihoods belong to representative beams, ranking scores
        aggregate ranks across augmentations. See methods/forward.rst.
        """
        if not 1 <= beam_width <= len(self.tokenizer.tokens):
            raise ValueError("Beam width must be between 1 and vocabulary size")
        return predict_augmented(
            self, reactants, beam_width, batch_size, num_results=num_results
        )

    def _decode(self, prefix, memory, mask):
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

    def _predict_batch(self, reactants, beam_width):
        ids, mask = self.tokenizer.encode(reactants)
        memory = self.encoder.run(None, {"input_ids": ids, "padding_mask": mask})[0]
        batch, vocabulary = len(ids), len(self.tokenizer.tokens)
        start, end, pad = (self.tokenizer.special[k] for k in ("start", "end", "pad"))
        prefix = np.full((batch, 1), start, dtype=np.int64)
        scores = None
        done = np.zeros(batch, dtype=bool)
        for position in range(self.max_length - 2):
            if done.all():
                break
            actions = self._decode(prefix, memory, mask)
            if position == 0:
                scores, tokens = _topk(actions, beam_width)
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
                prefix = np.concatenate((prefix, tokens.reshape(-1, 1)), axis=1)
                scores = scores.reshape(-1, 1)
            done = np.any(prefix == end, axis=1)
        tail = np.where(done, pad, end).astype(np.int64)
        if not done.all():
            actions = self._decode(prefix, memory, mask)
            scores += (actions[np.arange(len(prefix)), tail] * (~done)).reshape(-1, 1)
        prefix = np.concatenate((prefix, tail[:, None]), axis=1)
        predictions = (
            np.asarray(self.tokenizer.decode(prefix))
            .reshape(batch, beam_width)
            .tolist()
        )
        scores = scores.reshape(batch, beam_width).tolist()
        return [
            {"reactants": s, "predictions": p, "log_likelihoods": ll}
            for s, p, ll in zip(reactants, predictions, scores)
        ]
