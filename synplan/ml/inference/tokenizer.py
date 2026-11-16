"""Batch tokenization preserving each forward model's trained conventions.

Adapted from chemformer-workflows / MolecularAI Chemformer (Apache-2.0),
Microsoft RetroChimera v1.3.0 (MIT; Copyright (c) Microsoft Corporation and
contributors), and ReactionT5v2 (MIT; Tatsuya Sagawa et al.).
Modifications: shared padding, length checks and batch decoding; native tokens,
input templates and output cleanup are preserved. See the repository LICENSE.
"""

import json
import re
from pathlib import Path

import numpy as np

INPUT_TEMPLATE = "REACTANT:{}REAGENT: "
TOKEN_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class Tokenizer:
    """Model-native tokens with a common batch API; True masks padding.

    Inputs are already preprocessed SMILES spellings. No canonicalization or
    truncation is performed here. ReactionT5 alone imports ``tokenizers``.
    """

    def __init__(self, model_dir, model_type, max_length):
        directory = Path(model_dir)
        self.model_type, self.max_length = model_type, max_length
        if model_type == "reactiont5":
            from tokenizers import Tokenizer as NativeTokenizer

            self.tokenizer = NativeTokenizer.from_file(
                str(directory / "source/tokenizer.json")
            )
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
            self.special = {"pad": 0, "start": 0, "end": 1}
        elif model_type == "chemformer":
            data = json.loads(
                (directory / "vocabulary.json").read_text(encoding="utf-8")
            )
            self.tokens = data["vocabulary"]
            self.ids = {token: i for i, token in enumerate(self.tokens)}
            self.special = {
                name: self.ids[token]
                for name, token in data["properties"]["special_tokens"].items()
            }
            self.regex = re.compile(data["properties"]["regex"])
        elif model_type == "chimera":
            self.tokens = (
                (directory / "source/vocab.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.ids = {token: i for i, token in enumerate(self.tokens)}
            self.special = {"start": 0, "end": 1, "pad": 2, "unknown": 3}
            self.regex = re.compile(TOKEN_PATTERN)
        else:
            raise ValueError("Choose chemformer, chimera or reactiont5")

    def encode(self, reactants):
        """Return int64 token IDs and a boolean padding mask for a batch."""
        if self.model_type == "reactiont5":
            rows = [
                row.ids
                for row in self.tokenizer.encode_batch(
                    [INPUT_TEMPLATE.format(value) for value in reactants]
                )
            ]
        else:
            rows = []
            for value in reactants:
                tokens = self.regex.findall(value)
                if not value or "".join(tokens) != value:
                    raise ValueError("Input could not be tokenized completely")
                row = [self.ids.get(token, self.special["unknown"]) for token in tokens]
                if self.model_type == "chemformer":
                    row = [self.special["start"], *row, self.special["end"]]
                rows.append(row)
        width = max(map(len, rows))
        if width > self.max_length:
            raise ValueError(
                f"Input has {width} tokens; maximum is {self.max_length}. No truncation applied."
            )
        ids = np.full((len(rows), width), self.special["pad"], dtype=np.int64)
        mask = np.ones(ids.shape, dtype=bool)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = row
            mask[index, : len(row)] = False
        if self.model_type == "chimera":
            # Its native mask is defined by token identity, not sequence length.
            mask = ids == self.special["pad"]
        return ids, mask

    def decode(self, rows):
        """Decode a batch, retaining each model's special-token semantics."""
        result = []
        for row in rows:
            ids = list(map(int, row))
            if self.model_type == "reactiont5":
                result.append(
                    self.tokenizer.decode(ids, skip_special_tokens=True)
                    .replace(" ", "")
                    .rstrip(".")
                )
                continue
            if self.model_type == "chemformer":
                controls = {self.special[k] for k in ("pad", "start", "end")}
                if self.special["end"] in ids:
                    ids = ids[: ids.index(self.special["end"]) + 1]
                while ids and ids[0] in controls:
                    ids.pop(0)
                while ids and ids[-1] in controls:
                    ids.pop()
            else:
                if ids and ids[0] == self.special["start"]:
                    ids = ids[1:]
                if self.special["end"] in ids:
                    ids = ids[: ids.index(self.special["end"])]
            result.append("".join(self.tokens[i] for i in ids))
        return result
