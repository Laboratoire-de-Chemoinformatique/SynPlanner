"""Inference-only released ForwardChimera architecture and ONNX export graphs.

Attention, RMSNorm and gated feed-forward blocks are adapted from
Microsoft RetroChimera v1.3.0 / OpenNMT-py 3.5.1 (bf5ec59eec9ef32911d5167bf7528e3648622be1).
Copyright (c) Microsoft Corporation and contributors; 2017-Present OpenNMT. MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
Modifications: removed training and unused architecture variants; explicit KV
state, beam reordering and standalone encoder/decoder export interfaces.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value):
        value = value.float()
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            * self.weight
        )


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.depth = heads, width // heads
        self.linear_keys = nn.Linear(width, width)
        self.linear_values = nn.Linear(width, width)
        self.linear_query = nn.Linear(width, width)
        self.final_linear = nn.Linear(width, width)

    def split(self, value):
        return value.reshape(
            value.shape[0], value.shape[1], self.heads, self.depth
        ).transpose(1, 2)

    def keys_values(self, value):
        return self.split(self.linear_keys(value)), self.split(
            self.linear_values(value)
        )

    def forward(self, query, keys, values, mask=None):
        queries = self.split(self.linear_query(query)) / math.sqrt(self.depth)
        scores = queries @ keys.transpose(-1, -2)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :], -1e18)
        output = scores.softmax(-1) @ values
        return self.final_linear(output.transpose(1, 2).reshape(query.shape))


class FeedForward(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.layer_norm = RMSNorm(width)
        self.w_1, self.w_2, self.w_3 = (
            nn.Linear(width, hidden),
            nn.Linear(hidden, width),
            nn.Linear(width, hidden),
        )

    def forward(self, value):
        norm = self.layer_norm(value)
        return value + self.w_2(F.silu(self.w_1(norm)) * self.w_3(norm))


class EncoderLayer(nn.Module):
    def __init__(self, width, hidden, heads):
        super().__init__()
        self.layer_norm = RMSNorm(width)
        self.self_attn = Attention(width, heads)
        self.feed_forward = FeedForward(width, hidden)

    def forward(self, value, mask):
        norm = self.layer_norm(value)
        keys, values = self.self_attn.keys_values(norm)
        return self.feed_forward(value + self.self_attn(norm, keys, values, mask))


class DecoderLayer(nn.Module):
    def __init__(self, width, hidden, heads):
        super().__init__()
        self.layer_norm_1, self.layer_norm_2 = RMSNorm(width), RMSNorm(width)
        self.self_attn, self.context_attn = (
            Attention(width, heads),
            Attention(width, heads),
        )
        self.feed_forward = FeedForward(width, hidden)

    def forward(self, value, memory, mask, past=None):
        norm = self.layer_norm_1(value)
        keys, values = self.self_attn.keys_values(norm)
        if past is None:
            cross_keys, cross_values = self.context_attn.keys_values(memory)
        else:
            old_keys, old_values, cross_keys, cross_values = past
            keys = torch.cat((old_keys, keys), dim=2)
            values = torch.cat((old_values, values), dim=2)
        value = value + self.self_attn(norm, keys, values)
        value = value + self.context_attn(
            self.layer_norm_2(value), cross_keys, cross_values, mask
        )
        return self.feed_forward(value), (keys, values, cross_keys, cross_values)


class ChimeraNetwork(nn.Module):
    """Inference-only architecture for the released Pistachio forward checkpoint."""

    def __init__(self, config, vocabulary_size):
        super().__init__()
        self.config = config
        width, hidden, heads = (
            config["hidden_dim"],
            config["feedforward_dim"],
            config["n_heads"],
        )
        layers = config["n_layers"]
        if (
            config["activation"],
            config["layer_norm"],
            config["num_kv"],
            config["share_encoder_decoder_input_embedding"],
            config["parallel_residual"],
            config["positional_encoding_type"],
        ) != ("silu", "rms", heads, False, False, "SinusoidalInterleaved"):
            raise ValueError("Unsupported Chimera checkpoint architecture")
        self.encoder_token_embedding = nn.Embedding(
            vocabulary_size, width, padding_idx=2
        )
        self.decoder_token_embedding = nn.Embedding(
            vocabulary_size, width, padding_idx=2
        )
        self.register_buffer(
            "positional_encoding", torch.zeros(config["max_seq_len"], width)
        )
        self.encoder = nn.Module()
        self.encoder.transformer = nn.ModuleList(
            EncoderLayer(width, hidden, heads) for _ in range(layers)
        )
        self.encoder.layer_norm = RMSNorm(width)
        self.decoder = nn.Module()
        self.decoder.transformer_layers = nn.ModuleList(
            DecoderLayer(width, hidden, heads) for _ in range(layers)
        )
        self.decoder.layer_norm = RMSNorm(width)
        self.token_fc = nn.Linear(width, vocabulary_size)

    @classmethod
    def load(cls, checkpoint, device="cpu"):
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = cls(
            data["hyper_parameters"],
            data["state_dict"]["encoder_token_embedding.weight"].shape[0],
        )
        model.load_state_dict(data["state_dict"], strict=True)
        return model.eval().to(device)

    def encode(self, ids, mask):
        value = self.encoder_token_embedding(ids) * math.sqrt(self.config["hidden_dim"])
        value = value + self.positional_encoding[: ids.shape[1]][None]
        for layer in self.encoder.transformer:
            value = layer(value, mask)
        return self.encoder.layer_norm(value)

    def decode(self, ids, mask, position, memory=None, past=None, beam_indices=None):
        value = self.decoder_token_embedding(ids) * math.sqrt(self.config["hidden_dim"])
        value = value + self.positional_encoding[position].reshape(1, 1, -1)
        present = []
        for i, layer in enumerate(self.decoder.transformer_layers):
            cached = (
                None
                if past is None
                else tuple(
                    tensor.index_select(0, beam_indices)
                    for tensor in past[i * 4 : i * 4 + 4]
                )
            )
            value, cache = layer(value, memory, mask, cached)
            present.extend(cache)
        logits = self.token_fc(self.decoder.layer_norm(value))[:, 0]
        return (F.log_softmax(logits, dim=-1), *present)


class EncoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, padding_mask):
        return self.model.encode(input_ids, padding_mask)


class InitialDecoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, memory, padding_mask):
        return self.model.decode(input_ids, padding_mask, 0, memory=memory)


class CachedDecoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, padding_mask, position, beam_indices, *past):
        return self.model.decode(
            input_ids, padding_mask, position, past=past, beam_indices=beam_indices
        )
