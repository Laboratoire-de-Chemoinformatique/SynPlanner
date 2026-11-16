#!/usr/bin/env python3
"""Export the released Chimera checkpoint; requires optional torch and onnx."""

import argparse
import hashlib
import json
from pathlib import Path

REVISION = "bf5ec59eec9ef32911d5167bf7528e3648622be1"


def manifest(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
    (directory / "manifest.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n"
    )


def download(directory):
    """Retain the Figshare release verbatim and verify its published MD5."""
    import urllib.request
    import zipfile

    source = directory / "source"
    source.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        "https://api.figshare.com/v2/articles/32994914"
    ) as response:
        metadata = json.load(response)
    (directory / "figshare.json").write_text(json.dumps(metadata, indent=2) + "\n")
    asset = next(item for item in metadata["files"] if item["id"] == 66654872)
    archive = source / asset["name"]
    if not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        urllib.request.urlretrieve(asset["download_url"], temporary)
        temporary.replace(archive)
    digest = hashlib.md5()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != asset["supplied_md5"]:
        raise ValueError("Figshare archive MD5 mismatch")
    with zipfile.ZipFile(archive) as bundle:
        for name in ("combined.ckpt", "vocab.txt"):
            if not (source / name).exists():
                with (
                    bundle.open(name) as incoming,
                    (source / name).open("wb") as target,
                ):
                    import shutil

                    shutil.copyfileobj(incoming, target)
    with urllib.request.urlopen(
        f"https://raw.githubusercontent.com/microsoft/retrochimera/{REVISION}/LICENSE"
    ) as response:
        (directory / "LICENSE").write_bytes(response.read())


def export(directory):
    import torch

    from synplan.ml.inference._chimera_torch import (
        CachedDecoderGraph,
        ChimeraNetwork,
        EncoderGraph,
        InitialDecoderGraph,
    )

    torch.set_num_threads(4)
    model = ChimeraNetwork.load(directory / "source/combined.ckpt")
    metadata = {
        "model": "ForwardChimeraDeNovo-Pistachio",
        "format_version": 1,
        "architecture": model.config,
        "inference": {
            "max_generated_seq_len": 512,
            "beam_width": 20,
            "augmentation_size": 10,
            "num_results": 10,
            "seed": 0,
            "pipeline_version": 2,
            "identity": "full-standard-inchikey-ignore-stereo",
            "ranking": "clean-best-rank-then-reciprocal-support-stable-first",
        },
        "source": {
            "article": 32994914,
            "file": 66654872,
            "release": "retrochimera-v1.3.0",
            "revision": REVISION,
            "url": "https://github.com/microsoft/retrochimera",
            "license": "MIT",
        },
        "export": {"torch": torch.__version__, "opset": 17, "dtype": "float32"},
    }
    (directory / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    ids = torch.tensor([[4, 5, 6, 2], [6, 4, 5, 6]])
    mask = ids == 2
    beginning = torch.zeros((2, 1), dtype=torch.long)
    names = [
        f"{i}_{kind}"
        for i in range(model.config["n_layers"])
        for kind in ("key", "value", "cross_key", "cross_value")
    ]
    present = ["present_" + name for name in names]
    past = ["past_" + name for name in names]
    with torch.inference_mode():
        memory = model.encode(ids, mask)
        state = model.decode(beginning, mask, 0, memory=memory)[1:]
        torch.onnx.export(
            EncoderGraph(model),
            (ids, mask),
            str(directory / "encoder.onnx"),
            input_names=["input_ids", "padding_mask"],
            output_names=["memory"],
            dynamic_axes={
                name: {0: "batch", 1: "source_length"}
                for name in ("input_ids", "padding_mask", "memory")
            },
            opset_version=17,
            dynamo=False,
        )
        axes = {
            "input_ids": {0: "batch"},
            "memory": {0: "batch", 1: "source_length"},
            "padding_mask": {0: "batch", 1: "source_length"},
            "log_probs": {0: "batch"},
        }
        axes.update(
            {
                name: {
                    0: "batch",
                    2: "source_length" if "cross" in name else "present_length",
                }
                for name in present
            }
        )
        torch.onnx.export(
            InitialDecoderGraph(model),
            (beginning, memory, mask),
            str(directory / "decoder.onnx"),
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
            str(directory / "decoder_with_past.onnx"),
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
    manifest(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch and verify the original Figshare archive first",
    )
    args = parser.parse_args()
    if args.download:
        download(args.model)
    export(args.model)
