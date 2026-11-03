"""Extract residual stream activations from a Hugging Face causal LM.

Layer convention: layer L is the residual stream after transformer block L,
which is `hidden_states[L]` in Hugging Face (`hidden_states[0]` is the
embedding output).

Prefix token: some models (for example Qwen2.5) do not add a start token. Then
the first token of the prompt is an "attention sink" with a very large norm.
If the target token is the first token, its activation is not comparable with
the others. To prevent this, we put one prefix token before every prompt. The
prefix is added as a token id, so the prompt text and its character spans do
not change. A target token at index 0 is an error.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from typicality_linearity.activation_cache import ExtractedActivations
from typicality_linearity.token_positions import (
    target_char_index,
    token_index_at_char,
)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def choose_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding keeps the token indices of each prompt the same as
    # without padding. The causal mask stops real tokens from seeing pads.
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPES[dtype])
    model.to(device)
    model.eval()
    return tokenizer, model


def check_layers(model, layers):
    n_layers = model.config.num_hidden_layers
    for layer in layers:
        if layer < 0 or layer > n_layers:
            raise ValueError(f"Layer {layer} is not in 0..{n_layers}")


def prefix_token_id(tokenizer, prefix_token):
    """Return the id of prefix_token, or None if prefix_token is None."""
    if prefix_token is None:
        return None
    token_id = tokenizer.convert_tokens_to_ids(prefix_token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Prefix token {prefix_token!r} is not in the vocabulary")
    return token_id


def prepend_token(input_ids, attention_mask, token_id):
    """Put token_id at index 0 of each row. Return the new tensors."""
    n_rows = input_ids.shape[0]
    prefix_ids = torch.full((n_rows, 1), token_id, dtype=input_ids.dtype)
    prefix_mask = torch.ones((n_rows, 1), dtype=attention_mask.dtype)
    new_ids = torch.cat([prefix_ids, input_ids], dim=1)
    new_mask = torch.cat([prefix_mask, attention_mask], dim=1)
    return new_ids, new_mask


def find_token_indices(prompts, item_texts, offsets_batch, positions, shift=0):
    """Return {position: [token index for each prompt]}.

    shift is added to each index (1 if a prefix token was put before the prompt).
    """
    indices = {}
    for position in positions:
        indices[position] = []
        for prompt, item_text, offsets in zip(prompts, item_texts, offsets_batch):
            char_index = target_char_index(prompt, item_text, position)
            token_index = token_index_at_char(offsets, char_index) + shift
            if token_index == 0:
                raise ValueError(
                    f"The {position!r} token of {prompt!r} is the first token. "
                    "Its activation is an attention sink. Set a prefix_token."
                )
            indices[position].append(token_index)
    return indices


def extract_batch(
    model, tokenizer, prompts, item_texts, layers, positions, device, prefix_id=None
):
    encoded = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets_batch = encoded.pop("offset_mapping").tolist()
    shift = 0
    if prefix_id is not None:
        encoded["input_ids"], encoded["attention_mask"] = prepend_token(
            encoded["input_ids"], encoded["attention_mask"], prefix_id
        )
        shift = 1
    token_indices = find_token_indices(
        prompts, item_texts, offsets_batch, positions, shift
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    with torch.no_grad():
        output = model(**encoded, output_hidden_states=True)

    # Shape (n_layers, batch, seq, d_model)
    hidden = torch.stack([output.hidden_states[layer] for layer in layers])
    batch_rows = torch.arange(len(prompts))

    activations = {}
    spans = {}
    texts = {}
    for position in positions:
        index = torch.tensor(token_indices[position])
        # Shape (batch, n_layers, d_model)
        picked = hidden[:, batch_rows, index.to(hidden.device), :].transpose(0, 1)
        activations[position] = picked.float().cpu().numpy()
        spans[position] = []
        texts[position] = []
        for row, token_index in enumerate(token_indices[position]):
            start, end = offsets_batch[row][token_index - shift]
            spans[position].append((start, end))
            texts[position].append(prompts[row][start:end])
    return activations, spans, texts


def extract_activations(
    model,
    tokenizer,
    prompts,
    item_texts,
    layers,
    positions,
    batch_size,
    device,
    prefix_token=None,
):
    check_layers(model, layers)
    prefix_id = prefix_token_id(tokenizer, prefix_token)
    activation_parts = {position: [] for position in positions}
    token_spans = {position: [] for position in positions}
    token_texts = {position: [] for position in positions}

    for start in range(0, len(prompts), batch_size):
        end = start + batch_size
        activations, spans, texts = extract_batch(
            model,
            tokenizer,
            prompts[start:end],
            item_texts[start:end],
            layers,
            positions,
            device,
            prefix_id,
        )
        for position in positions:
            activation_parts[position].append(activations[position])
            token_spans[position].extend(spans[position])
            token_texts[position].extend(texts[position])

    stacked = {}
    for position in positions:
        stacked[position] = np.concatenate(activation_parts[position])
    return ExtractedActivations(stacked, token_spans, token_texts)
