#!/usr/bin/env python3
"""Download the pinned ReactionT5v2 USPTO-MIT release and export cached T5 ONNX graphs.

Requires the reactiont5-export extra. The downloaded directory contains data only.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from scripts.export_forward_chimera import manifest

REPOSITORY = "sagawa/ReactionT5v2-forward-USPTO_MIT"
REVISION = "260cf4aa7e6f55b1a9db1e328f48701cd3fd7cfc"
SOURCE_REVISION = "76eb08068e10fe255cae5d563a91e1c1e9abac54"
WEIGHTS_SHA256 = "b5fdde934cb203742547527154c3af33560a143e606118dcea81076f0e4ab458"
SOURCE_FILES = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "README.md",
]


def download(directory):
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPOSITORY,
        revision=REVISION,
        local_dir=directory / "source",
        allow_patterns=SOURCE_FILES,
    )
    for url, filename in [
        (
            f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true",
            "release.json",
        ),
        (
            f"https://raw.githubusercontent.com/sagawatatsuya/ReactionT5v2/{SOURCE_REVISION}/LICENSE.txt",
            "LICENSE.txt",
        ),
    ]:
        with urllib.request.urlopen(url) as response:
            (directory / filename).write_bytes(response.read())


def verify_checkpoint(directory):
    with (directory / "source/model.safetensors").open("rb") as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
        digest = checksum.hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError("Released checkpoint SHA-256 mismatch")


def export(directory):
    import onnx
    import torch
    import transformers

    from synplan.ml.inference._reactiont5_torch import (
        DecoderGraph,
        EncoderGraph,
        load_network,
    )

    verify_checkpoint(directory)
    torch.set_num_threads(4)
    model = load_network(directory / "source")
    ids = torch.tensor([[266, 8, 9, 267, 1], [266, 12, 267, 1, 0]])
    mask = (ids != 0).long()
    beginning = torch.zeros((2, 1), dtype=torch.long)
    suffixes = [
        f"{i}_{kind}"
        for i in range(model.config.num_decoder_layers)
        for kind in ("key", "value", "cross_key", "cross_value")
    ]
    past, present = (
        [prefix + suffix for suffix in suffixes] for prefix in ("past_", "present_")
    )
    encoder, decoder, cached = (
        EncoderGraph(model),
        DecoderGraph(model),
        DecoderGraph(model, cached=True),
    )
    with torch.inference_mode():
        memory = encoder(ids, mask)
        state = decoder(beginning, mask, memory)[1:]
        torch.onnx.export(
            encoder,
            (ids, mask),
            str(directory / "encoder.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["memory"],
            dynamic_axes={
                name: {0: "batch", 1: "source_length"}
                for name in ("input_ids", "attention_mask", "memory")
            },
            opset_version=17,
            dynamo=False,
        )
        axes = {
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch", 1: "source_length"},
            "memory": {0: "batch", 1: "source_length"},
            "logits": {0: "batch"},
            **{
                name: {
                    0: "batch",
                    2: "source_length" if "cross" in name else "present_length",
                }
                for name in present
            },
        }
        torch.onnx.export(
            decoder,
            (beginning, mask, memory),
            str(directory / "decoder.onnx"),
            input_names=["input_ids", "attention_mask", "memory"],
            output_names=["logits", *present],
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
            cached,
            (beginning, mask, torch.tensor([1, 0]), *state),
            str(directory / "decoder_with_past.onnx"),
            input_names=["input_ids", "attention_mask", "beam_indices", *past],
            output_names=["logits", *present],
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False,
        )
    files = {}
    for name in ("encoder", "decoder", "decoder_with_past"):
        filename = f"{name}.onnx"
        graph = onnx.load(directory / filename, load_external_data=False)
        external = {
            entry.value
            for tensor in graph.graph.initializer
            for entry in tensor.external_data
            if entry.key == "location"
        }
        files[name] = [filename, *sorted(external)]
    metadata = {
        "model": "ReactionT5v2-forward-USPTO_MIT",
        "format_version": 1,
        "architecture": model.config.to_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "inference": {
            "max_input_length": 400,
            "max_length": 300,
            "min_length": 1,
            "length_penalty": 1.0,
            "early_stopping": False,
            "input_template": "REACTANT:{}REAGENT: ",
            "beam_width": 20,
            "augmentation_size": 10,
            "num_results": 10,
            "seed": 0,
            "limits_source": "ReactionT5v2 task_forward/prediction.py defaults; output limit includes decoder start",
        },
        "source": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "weights_sha256": WEIGHTS_SHA256,
            "code_revision": SOURCE_REVISION,
            "license": "MIT",
            "card_note": "The card introduction says USPOT_50k; repository name, fine-tuning section and example command specify USPTO_MIT.",
        },
        "export": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
            "opset": 17,
            "dtype": "float32",
        },
        "onnx_files": files,
    }
    (directory / "model.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "README.md").write_text(
        "# ReactionT5v2 USPTO-MIT for SynPlanner\n\n"
        f"Checkpoint: https://huggingface.co/{REPOSITORY}/tree/{REVISION}\n\n"
        "ONNX: encoder.onnx, decoder.onnx, decoder_with_past.onnx, model.json, source/tokenizer.json, plus any external data listed in model.json. "
        "Checkpoint inference additionally uses source/config.json and source/model.safetensors.\n\n"
        "Use `ForwardPredictor(directory, model_type='reactiont5')` from `synplan.ml.inference`; specify backend='ckpt' explicitly for Transformers. "
        "Install SynPlanner[reactiont5] for ONNX or [reactiont5-checkpoint] for checkpoint inference.\n\n"
        "Input is unmapped, stereochemistry-preserving reactant SMILES, formatted as `REACTANT:<SMILES>REAGENT: ` with blank reagents. "
        "Up to ten deterministic rooted inputs, twenty beams each, stereo-free full-InChIKey grouping and best-rank/support reranking to ten distinct products. "
        "Sequence log likelihoods are unnormalized sums; ranking scores are separate. Neither measures experimental success.\n\n"
        "Limits follow the reference prediction script: input 400 tokens (error rather than truncation), output 300 tokens including decoder start. "
        "The checkpoint generation_config.json's 150-token default is explicitly overridden. No extra tautomer normalization.\n\n"
        "Provenance: Tatsuya Sagawa et al., ReactionT5v2, MIT; T5 v1.1 base → CompoundT5 → ORD forward training → USPTO-MIT fine-tuning. "
        "The source model card mentions USPOT_50k in its introduction despite identifying USPTO-MIT in the fine-tuning section; see source/README.md verbatim. "
        "The neural architecture is implemented by Hugging Face Transformers (Apache-2.0). No upstream executable code is shipped in the asset directory.\n\n"
        "See validation.json for numerical and raw-beam comparisons, and manifest.json for SHA-256 hashes.\n",
        encoding="utf-8",
    )
    manifest(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.model.mkdir(parents=True, exist_ok=True)
    if args.download:
        download(args.model)
    export(args.model)


if __name__ == "__main__":
    main()
