"""Steering along a typicality direction.

The steering vector is added to the residual stream at layer L, which is
`hidden_states[L]` (the output of decoder block L - 1, see activations.py):

    h <- h + alpha * norm_ref * u

u is a unit direction. norm_ref is the mean norm of the unsteered residual
stream at layer L over the steered positions, so alpha is a fraction of the
usual activation norm. The prefix token (index 0) is never steered: its
activation is an attention sink with a very large norm.

The general typicality direction is the mean of the unit difference-of-means
directions over all categories and training prompts of one setting, as in
mean_direction.py (`mean_all`), then normalised.
"""

import csv

import numpy as np
import torch

STEER_MODES = ("item", "all", "last")


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalise a zero vector")
    return vector / norm


def best_setting(summary_csv, statistic, metric):
    """Return the summary row (dict) with the highest value of statistic."""
    with open(summary_csv) as f:
        rows = [row for row in csv.DictReader(f) if row["metric"] == metric]
    if not rows:
        raise ValueError(f"No rows with metric {metric!r} in {summary_csv}")
    return max(rows, key=lambda row: float(row[statistic]))


def general_direction(weights, setting, baseline):
    """Return (unit direction, list of keys used).

    weights: mapping with keys "<setting>/<baseline>/<category>/<prompt>".
    The direction is the mean of the unit directions of all categories and
    prompts, then normalised. For difference-of-means probes, the weights are
    the direction in activation space (scale 1).
    """
    prefix = f"{setting}/{baseline}/"
    keys = sorted(key for key in weights if key.startswith(prefix))
    if not keys:
        raise ValueError(f"No directions for {prefix!r}")
    mean = np.mean([unit(weights[key]) for key in keys], axis=0)
    return unit(mean), keys


def random_directions(n_directions, d_model, seed):
    """Return an array (n_directions, d_model) of random unit directions."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n_directions, d_model))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def steer_mask(attention_mask, mode, item_index=None, skip_first=True):
    """Return a float mask (batch, seq) of the positions to steer.

    mode "item": the token at item_index (the last token of the item). item_index
    is one int for all rows, or one int for each row. mode "all": all real
    tokens. mode "last": the last real token of each row. skip_first: do not
    steer index 0 (the prefix token).
    """
    if mode not in STEER_MODES:
        raise ValueError(f"Unknown steer mode {mode!r}, use one of {STEER_MODES}")
    n_rows = attention_mask.shape[0]
    mask = attention_mask.clone().float()
    if mode == "item":
        if item_index is None:
            raise ValueError("Steer mode 'item' needs item_index")
        indices = torch.as_tensor(item_index).reshape(-1).expand(n_rows)
        if skip_first and bool((indices == 0).any()):
            raise ValueError("The item token is the prefix token")
        mask = torch.zeros_like(mask)
        mask[torch.arange(n_rows), indices] = 1.0
    if mode == "last":
        last = attention_mask.sum(dim=1) - 1
        mask = torch.zeros_like(mask)
        mask[torch.arange(n_rows), last] = 1.0
    if skip_first:
        mask[:, 0] = 0.0
    return mask


class SteeringHook:
    """Forward hook that adds mask * vector to the output of a decoder block.

    Set `vector` and `mask` (tensor (batch, seq)) before each forward pass.
    vector is a tensor (d_model,) for all rows, or (batch, d_model) for one
    vector in each row. vector None means no steering.
    """

    def __init__(self):
        self.vector = None
        self.mask = None

    def __call__(self, module, inputs, output):
        if self.vector is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        vector = self.vector.to(hidden.dtype)
        if vector.dim() == 2:
            vector = vector[:, None, :]
        added = self.mask[:, :, None].to(hidden.dtype) * vector
        hidden = hidden + added
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden


def decoder_layers(model):
    """Return the list of decoder blocks of a Hugging Face causal LM."""
    return model.model.layers


def register_steering(model, layer):
    """Add a SteeringHook at the residual stream of layer (hidden_states[layer]).

    Return (hook, handle). Call handle.remove() at the end.
    """
    n_layers = len(decoder_layers(model))
    if layer < 1 or layer > n_layers:
        raise ValueError(f"Steering layer {layer} is not in 1..{n_layers}")
    hook = SteeringHook()
    handle = decoder_layers(model)[layer - 1].register_forward_hook(hook)
    return hook, handle


def pole_token_ids(tokenizer, pole_words):
    """Return {word: token id} of " word". Each word must be one token."""
    ids = {}
    for word in pole_words:
        token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Pole word {word!r} is {len(token_ids)} tokens, not 1")
        ids[word] = token_ids[0]
    return ids


def logit_difference(logits, typical_ids, atypical_ids):
    """Return mean logit of the typical poles - mean logit of the atypical poles.

    logits: array (..., vocab).
    """
    logits = np.asarray(logits)
    typical = logits[..., typical_ids].mean(axis=-1)
    atypical = logits[..., atypical_ids].mean(axis=-1)
    return typical - atypical
