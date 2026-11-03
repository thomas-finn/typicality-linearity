"""Cache extracted activations on disk so that runs can use them again.

Each entry is one prompt format. The cache key is a hash of all the inputs
that change the activations: model, dtype, layers, positions, prefix token,
and the exact prompt and item texts. If one of these changes, the key changes, so a stale
entry is never used.

Files in the cache directory:
- <prompt_name>_<key>.npz: activations, token spans and token texts
- <prompt_name>_<key>.json: the key inputs and how the entry was made
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


@dataclass
class ExtractedActivations:
    # position name -> array with shape (n_prompts, n_layers, d_model)
    activations: dict
    # position name -> list of (start, end) character spans of the token
    token_spans: dict
    # position name -> list of token strings
    token_texts: dict


def hash_texts(texts):
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def key_inputs(model, dtype, layers, positions, prompts, item_texts, prefix_token=None):
    return {
        "model": model,
        "dtype": dtype,
        "layers": [int(layer) for layer in layers],
        "positions": list(positions),
        "prefix_token": prefix_token,
        "n_prompts": len(prompts),
        "prompts_sha256": hash_texts(prompts),
        "item_texts_sha256": hash_texts(item_texts),
    }


def cache_key(inputs):
    text = json.dumps(inputs, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def entry_paths(cache_dir, prompt_name, key):
    stem = Path(cache_dir) / f"{prompt_name}_{key}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def load_entry(cache_dir, prompt_name, key, positions):
    """Return ExtractedActivations, or None if the entry does not exist."""
    npz_path, json_path = entry_paths(cache_dir, prompt_name, key)
    if not npz_path.exists() or not json_path.exists():
        return None
    activations = {}
    token_spans = {}
    token_texts = {}
    with np.load(npz_path) as data:
        for position in positions:
            activations[position] = data[f"activations_{position}"]
            spans = data[f"spans_{position}"]
            token_spans[position] = [(int(s), int(e)) for s, e in spans]
            token_texts[position] = [str(text) for text in data[f"texts_{position}"]]
    return ExtractedActivations(activations, token_spans, token_texts)


def write_atomic(path, write_function):
    # Write to a temporary file, then rename. An interrupted write does not
    # leave a broken entry.
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as f:
        write_function(f)
    os.replace(temporary, path)


def save_entry(cache_dir, prompt_name, key, inputs, extracted, extra_info=None):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    npz_path, json_path = entry_paths(cache_dir, prompt_name, key)

    arrays = {}
    for position, values in extracted.activations.items():
        arrays[f"activations_{position}"] = values
        arrays[f"spans_{position}"] = np.array(
            extracted.token_spans[position], dtype=np.int64
        )
        arrays[f"texts_{position}"] = np.array(extracted.token_texts[position])
    write_atomic(npz_path, lambda f: np.savez(f, **arrays))

    info = {
        "prompt_name": prompt_name,
        "key": key,
        "key_inputs": inputs,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if extra_info:
        info.update(extra_info)
    text = json.dumps(info, indent=2).encode("utf-8")
    # The JSON is written last: an entry is only complete when it exists.
    write_atomic(json_path, lambda f: f.write(text))
    return npz_path
