"""Optional incremental Chemformer decoder for ONNX export and parity checks.

Adapted from MolecularAI/aizynthmodels Chemformer v2 (revision
f9262c2c44720a194ae67a37c35b411b20aec2c8) and chemformer-workflows' portable
attention. Apache-2.0; the full license and attribution are in chemformer.py.
Modifications: inference-only decoder, direct state-dictionary loading, cached
self/cross attention and beam-parent gathering. No training package is imported.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _project(attention, value, index):
    width, heads = attention.embed_dim, attention.num_heads
    weight = attention.in_proj_weight[index * width : (index + 1) * width]
    bias = attention.in_proj_bias[index * width : (index + 1) * width]
    return (
        F.linear(value, weight, bias)
        .reshape(value.shape[0], value.shape[1], heads, width // heads)
        .transpose(1, 2)
    )


def _attention(attention, value, keys, values, mask=None):
    query = _project(attention, value, 0)
    scores = torch.matmul(query, keys.transpose(-1, -2)) * (attention.head_dim**-0.5)
    if mask is not None:
        scores = scores.masked_fill(mask[:, None, None, :], -float("inf"))
    result = torch.matmul(torch.softmax(scores, dim=-1), values)
    result = result.transpose(1, 2).reshape(
        value.shape[0], value.shape[1], attention.embed_dim
    )
    return attention.out_proj(result)


class ChemformerDecoder(nn.Module):
    """The released pre-norm GELU decoder, with unchanged trained tensor names."""

    def __init__(self, architecture):
        super().__init__()
        self.width = architecture["d_model"]
        self.emb = nn.Embedding(
            architecture["vocabulary_size"], self.width, architecture["pad_token_idx"]
        )
        self.register_buffer(
            "pos_emb", torch.zeros(architecture["max_seq_len"], self.width)
        )
        layer = nn.TransformerDecoderLayer(
            self.width,
            architecture["num_heads"],
            architecture["d_feedforward"],
            dropout=0.0,
            activation=architecture["activation"],
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, architecture["num_layers"], norm=nn.LayerNorm(self.width)
        )
        self.token_fc = nn.Linear(self.width, architecture["vocabulary_size"])

    @classmethod
    def load(cls, checkpoint, architecture):
        # The original Lightning checkpoint also records a scheduler class.
        with torch.serialization.safe_globals(
            [getattr, torch.optim.lr_scheduler.OneCycleLR]
        ):
            data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = cls(architecture)
        model.load_state_dict(
            {
                k: v
                for k, v in data["state_dict"].items()
                if not k.startswith("encoder.")
            },
            strict=True,
        )
        return model.eval()

    def decode(self, ids, mask, position, memory=None, past=None, parents=None):
        value = self.emb(ids) * math.sqrt(self.width)
        value = value + self.pos_emb[position].reshape(1, 1, -1)
        present = []
        for index, layer in enumerate(self.decoder.layers):
            norm = layer.norm1(value)
            keys, values = (_project(layer.self_attn, norm, i) for i in (1, 2))
            if past is None:
                cross_keys, cross_values = (
                    _project(layer.multihead_attn, memory, i) for i in (1, 2)
                )
            else:
                old_keys, old_values, cross_keys, cross_values = (
                    tensor.index_select(0, parents)
                    for tensor in past[index * 4 : index * 4 + 4]
                )
                keys = torch.cat((old_keys, keys), dim=2)
                values = torch.cat((old_values, values), dim=2)
            value = value + _attention(layer.self_attn, norm, keys, values)
            value = value + _attention(
                layer.multihead_attn, layer.norm2(value), cross_keys, cross_values, mask
            )
            value = value + layer.linear2(
                layer.activation(layer.linear1(layer.norm3(value)))
            )
            present.extend((keys, values, cross_keys, cross_values))
        logits = self.token_fc(self.decoder.norm(value))[:, 0]
        return (F.log_softmax(logits, dim=-1), *present)


class InitialDecoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # nn.Module types forward as a mutable callable; overriding it is its API.
    def forward(self, input_ids, memory, padding_mask):  # ty: ignore[invalid-mutable-override]
        return self.model.decode(input_ids, padding_mask, 0, memory=memory)


class CachedDecoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    # nn.Module types forward as a mutable callable; overriding it is its API.
    def forward(self, input_ids, padding_mask, position, beam_indices, *past):  # ty: ignore[invalid-mutable-override]
        return self.model.decode(
            input_ids, padding_mask, position, past=past, parents=beam_indices
        )
