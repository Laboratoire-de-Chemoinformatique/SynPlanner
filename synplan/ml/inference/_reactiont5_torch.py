"""Optional wrappers around Hugging Face T5 for checkpoint inference and ONNX export.

ReactionT5v2: Tatsuya Sagawa et al., MIT; Transformers: Hugging Face, Apache-2.0.
The architecture is provided by Transformers, not copied or loaded from the assets.
"""

import torch
from torch import nn
from transformers import T5ForConditionalGeneration


def load_network(directory, device="cpu"):
    return (
        T5ForConditionalGeneration.from_pretrained(
            directory, local_files_only=True, use_safetensors=True
        )
        .eval()
        .to(device)
    )


class EncoderGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    # nn.Module types forward as a mutable callable; overriding it is its API.
    def forward(self, input_ids, attention_mask):  # ty: ignore[invalid-mutable-override]
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )[0]


class DecoderGraph(nn.Module):
    def __init__(self, model, *, cached=False):
        super().__init__()
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        self.cached = cached
        self.width = model.config.d_model
        self.scale = self.width**-0.5 if model.config.tie_word_embeddings else 1.0

    # nn.Module types forward as a mutable callable; overriding it is its API.
    def forward(self, input_ids, attention_mask, memory_or_parents, *past):  # ty: ignore[invalid-mutable-override]
        if self.cached:
            past = tuple(value.index_select(0, memory_or_parents) for value in past)
            state = tuple(tuple(past[i : i + 4]) for i in range(0, len(past), 4))
            # Cross-attention uses its cached K/V; only the source shape is needed.
            memory = torch.zeros(
                (input_ids.shape[0], attention_mask.shape[1], self.width),
                dtype=self.lm_head.weight.dtype,
                device=input_ids.device,
            )
        else:
            state, memory = None, memory_or_parents
        hidden, present = self.decoder(
            input_ids=input_ids,
            attention_mask=None,
            encoder_hidden_states=memory,
            encoder_attention_mask=attention_mask,
            past_key_values=state,
            use_cache=True,
            return_dict=False,
        )[:2]
        logits = self.lm_head(hidden[:, -1] * self.scale)
        return (logits, *(value for layer in present for value in layer))
