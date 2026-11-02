#!/usr/bin/env python3
"""Export cached Chemformer decoders alongside an unchanged portable encoder.

Requires SynPlanner[forward-export]. The source bundle and checkpoint are read
only; neither Chemformer nor Lightning needs to be installed.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from scripts.export_forward_chimera import manifest


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(source, checkpoint, destination):
    import onnx
    import torch

    from synplan.ml.inference._chemformer_torch import (
        CachedDecoderGraph,
        ChemformerDecoder,
        InitialDecoderGraph,
    )

    metadata = json.loads((source / "model.json").read_text(encoding="utf-8"))
    if metadata.get("kv_cache"):
        raise ValueError("Source must be the original full-prefix bundle")
    if sha256(checkpoint) != metadata["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match the source bundle")
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = ChemformerDecoder.load(checkpoint, metadata["architecture"])
    for name in ("encoder.onnx", "vocabulary.json", "LICENSE", "NOTICE.md"):
        shutil.copyfile(source / name, destination / name)
    beginning = torch.ones((2, 1), dtype=torch.long)
    memory = torch.randn(2, 7, model.width)
    mask = torch.zeros((2, 7), dtype=torch.bool)
    names = [
        f"{i}_{kind}"
        for i in range(metadata["architecture"]["num_layers"])
        for kind in ("key", "value", "cross_key", "cross_value")
    ]
    present, past = ["present_" + n for n in names], ["past_" + n for n in names]
    axes = {
        "input_ids": {0: "batch"},
        "memory": {0: "batch", 1: "source_length"},
        "padding_mask": {0: "batch", 1: "source_length"},
        "log_probs": {0: "batch"},
        **{
            name: {
                0: "batch",
                2: "source_length" if "cross" in name else "present_length",
            }
            for name in present
        },
    }
    with torch.inference_mode():
        state = model.decode(beginning, mask, 0, memory=memory)[1:]
        torch.onnx.export(
            InitialDecoderGraph(model),
            (beginning, memory, mask),
            str(destination / "decoder.onnx"),
            input_names=["input_ids", "memory", "padding_mask"],
            output_names=["log_probs", *present],
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False,
        )
        del axes["memory"]
        axes["beam_indices"] = {0: "batch"}
        axes.update(
            {
                name: {
                    0: "past_batch",
                    2: "source_length" if "cross" in name else "past_length",
                }
                for name in past
            }
        )
        torch.onnx.export(
            CachedDecoderGraph(model),
            (beginning, mask, torch.tensor(1), torch.tensor([1, 0]), *state),
            str(destination / "decoder_with_past.onnx"),
            input_names=[
                "input_ids",
                "padding_mask",
                "position",
                "beam_indices",
                *past,
            ],
            output_names=["log_probs", *present],
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False,
        )
    for name in ("encoder.onnx", "decoder.onnx", "decoder_with_past.onnx"):
        onnx.checker.check_model(str(destination / name), full_check=True)
    metadata.update(
        kv_cache=True,
        format_version=2,
        dynamic_axes=["batch", "source_length", "past_batch", "past_length"],
        export={
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "opset": 17,
            "precision": "float32",
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "encoder_sha256": sha256(source / "encoder.onnx"),
            "script": "scripts/export_chemformer.py",
        },
    )
    (destination / "model.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    original_provenance = (
        (source / "README.md")
        .read_text(encoding="utf-8")
        .partition("## Provenance\n")[2]
        .partition("## Original SynPlanner adapter validation")[0]
    )
    (destination / "README.md").write_text(
        "# SynFwd-1: cached ONNX decoding\n\n"
        "The same Chemformer v2 BART checkpoint trained on `uspto_synplan_final` "
        "(CGR-Morgan harsh split, final-cgr-20261016-01/harsh_cgr, epoch 25). "
        "Source revision and checkpoint SHA-256 are recorded in model.json.\n\n"
        "Runtime files: encoder.onnx, decoder.onnx, decoder_with_past.onnx, "
        "vocabulary.json, model.json. The encoder is copied byte-for-byte from "
        "the original bundle. Decoder weights are unchanged; self- and cross-"
        "attention keys/values are reused between tokens. LICENSE and NOTICE.md "
        "retain the original Apache-2.0 attribution. No executable assets.\n\n"
        "```python\nfrom synplan.ml.inference.chemformer import OnnxChemformer\n"
        "model = OnnxChemformer('synplan_data/forward/synfwd-1/v2', threads=4)\n"
        "products = model.predict(['CC(=O)Cl.N'], batch_size=1)\n```\n\n"
        "Input: reactants-only, dot-separated SMILES. Existing mapping removal, "
        "stereo/isotope preservation, ten input augmentations, twenty raw beams, "
        "stereo-free identity grouping and final ten-product ranking are unchanged. "
        "Outputs contain products, representative sequence likelihoods and ranking "
        "scores. CPU ONNX inference needs no PyTorch or Chemformer checkout.\n\n"
        "Export checks ONNX graph validity. Numerical and prediction parity must "
        "be checked separately with tests/integration/test_chemformer_cache.py; "
        "export alone is not a claim of validated parity.\n"
        + (
            "\n## Original model provenance\n" + original_provenance
            if original_provenance
            else ""
        ),
        encoding="utf-8",
    )
    manifest(destination)
    print(f"Exported cached bundle: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Original two-graph bundle"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New bundle directory"
    )
    args = parser.parse_args()
    export(args.source, args.checkpoint, args.output)
