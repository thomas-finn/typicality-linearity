"""Steering experiment: does adding the general typicality direction change
the logits of typical and atypical pole words?

Prompts: one prompt for each Rosch item, made from the template. {category}
is the category with its determiner ("a fruit", "clothing"), {item} is the item
noun phrase (see item_noun_phrase), {be} is "is" or "are". BLANK is filled with
each pole word, for example

    As an example of a fruit, the apple is fairly typical

The logit of a pole word is read at the token before it. All pole words are one
token and follow the same context, so one forward pass of the context gives
all of them. This is checked for each prompt.

Direction: the setting with the best `selection_statistic` in the summary.csv
of the source run (for example mean_all_eval: the mean direction over all
categories, evaluated on the evaluation prompts). See steering.py.

Steering positions (steer_modes): "item" is the last token of the item in each
prompt. "all" and "last" are also supported (see steering.steer_mask). The
added vector is alpha x (norm of the unsteered residual stream of that prompt
at the steered positions) x unit direction.

Controls: random unit directions with the same steering constants.

Typicality third: the items of each category are sorted by typicality
(-rating_zscore) and split into 3 groups of almost equal size. 0 is the least
typical third, 2 the most typical third.

Outputs go in `<output_dir>/<run_id>/`:
- config.yaml, metadata.json
- direction.npy: the unit steering direction
- items.csv: one row per item, with its prompt, steered token and third
- results.csv: one row per direction, steer mode, steering constant and item,
  with the logit of each pole word and the logit difference
- top_tokens.csv: the top next tokens for the typicality direction
- <html_name>: the plots, with a category selector
"""

import csv
import traceback
from pathlib import Path

import numpy as np
import yaml

from typicality_linearity.rosch_prompts import (
    ARTICLE,
    CATEGORY_FORMS,
    item_phrase,
)
from typicality_linearity.run_metadata import (
    make_run_id,
    now_utc,
    set_seed,
    start_metadata,
    write_metadata,
)
from typicality_linearity.steering import (
    STEER_MODES,
    best_setting,
    general_direction,
    logit_difference,
    pole_token_ids,
    random_directions,
    register_steering,
    steer_mask,
)
from typicality_linearity.steering_plot import write_html
from typicality_linearity.token_positions import (
    target_char_index,
    token_index_at_char,
)

REQUIRED_KEYS = (
    "experiment",
    "model",
    "source_run_dir",
    "selection_statistic",
    "selection_metric",
    "prompts_csv",
    "template",
    "typical_poles",
    "atypical_poles",
    "layer",
    "steer_modes",
    "steering_constants",
    "n_random_directions",
    "n_top_tokens",
    "batch_size",
    "device",
    "dtype",
    "prefix_token",
    "seed",
    "categories",
    "max_rows",
    "output_dir",
    "html_name",
)

BLANK = "BLANK"
TYPICALITY = "typicality"
N_THIRDS = 3


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    check_config(config)
    return config


def check_config(config):
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if config["template"].count(BLANK) != 1:
        raise ValueError(f"The template must contain {BLANK} once")
    if not config["template"].split(BLANK)[0].endswith(" "):
        raise ValueError("BLANK must follow a space")
    for mode in config["steer_modes"]:
        if mode not in STEER_MODES:
            raise ValueError(f"Unknown steer mode {mode!r}")
    if 0.0 not in [float(c) for c in config["steering_constants"]]:
        raise ValueError("steering_constants must contain 0 (no steering)")


def item_noun_phrase(phrase):
    """Return the item with "the" for count nouns and plurals, and without a
    determiner for mass nouns and activities ("spinach", "swimming")."""
    if phrase.determiner == ARTICLE or phrase.plural:
        return "the " + phrase.text
    return phrase.text


def make_context(template, category, item):
    """Return (context before BLANK without the trailing space, item text)."""
    phrase = item_phrase(item, category)
    filled = template.format(
        category=CATEGORY_FORMS[category].with_determiner,
        item=item_noun_phrase(phrase),
        be=phrase.verb_be(),
    )
    context = filled.split(BLANK)[0][:-1]
    return context, phrase.text


def typicality_thirds(categories, typicality):
    """Return an int array: the typicality third of each item in its category
    (0 least typical, 2 most typical)."""
    categories = np.asarray(categories)
    typicality = np.asarray(typicality, dtype=np.float64)
    thirds = np.empty(len(categories), dtype=int)
    for category in np.unique(categories):
        rows = np.flatnonzero(categories == category)
        order = rows[np.argsort(typicality[rows], kind="stable")]
        for rank, row in enumerate(order):
            thirds[row] = rank * N_THIRDS // len(order)
    return thirds


def read_items(config):
    """Return a list of item dicts from prompts_csv, for the categories in
    config["categories"] (None: all categories)."""
    with open(config["prompts_csv"]) as f:
        rows = list(csv.DictReader(f))
    if config["categories"] is not None:
        rows = [row for row in rows if row["category"] in config["categories"]]
        if not rows:
            raise ValueError(f"No items in categories {config['categories']}")
    if config["max_rows"] is not None:
        rows = rows[: config["max_rows"]]
    items = []
    for row in rows:
        context, item_text = make_context(
            config["template"], row["category"], row["item"]
        )
        items.append(
            {
                "item": row["item"],
                "category": row["category"],
                "rating_zscore": float(row["rating_zscore"]),
                "typicality": -float(row["rating_zscore"]),
                "is_ambiguous_extended": row["is_ambiguous_extended"] == "True",
                "context": context,
                "item_text": item_text,
            }
        )
    thirds = typicality_thirds(
        [item["category"] for item in items], [item["typicality"] for item in items]
    )
    for item, third in zip(items, thirds):
        item["third"] = int(third)
    return items


def write_csv(rows, path):
    rows = list(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def check_context_is_prefix(tokenizer, context, words):
    """Check that the tokens of the context are the first tokens of each filled
    prompt, so the logit at the end of the context is the logit of the pole
    word in the filled prompt."""
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    for word in words:
        prompt_ids = tokenizer.encode(context + " " + word, add_special_tokens=False)
        if prompt_ids[: len(context_ids)] != context_ids:
            raise ValueError(f"{context!r} is not a token prefix with {word!r}")


def load_source_direction(config):
    source = Path(config["source_run_dir"])
    row = best_setting(
        source / "summary.csv",
        config["selection_statistic"],
        config["selection_metric"],
    )
    weights = np.load(source / "probe_weights.npz")
    direction, keys = general_direction(weights, row["setting"], row["baseline"])
    selection = {
        "setting": row["setting"],
        "baseline": row["baseline"],
        "statistic": config["selection_statistic"],
        "metric": config["selection_metric"],
        "value": float(row[config["selection_statistic"]]),
        "n_directions_averaged": len(keys),
    }
    return direction, selection


def encode_batch(tokenizer, contexts, item_texts, prefix_token, device):
    """Return (input_ids, attention_mask, item token indices).

    Right padding. The item token index is the last token of the item in each
    context, with the prefix token counted.
    """
    import torch

    from typicality_linearity.activations import prefix_token_id, prepend_token

    tokenizer.padding_side = "right"
    encoded = tokenizer(
        contexts, padding=True, return_tensors="pt", return_offsets_mapping=True
    )
    offsets = encoded["offset_mapping"].tolist()
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    item_index = []
    for context, item_text, row_offsets in zip(contexts, item_texts, offsets):
        char_index = target_char_index(context, item_text, "item")
        item_index.append(token_index_at_char(row_offsets, char_index))
    item_index = torch.tensor(item_index)
    prefix_id = prefix_token_id(tokenizer, prefix_token)
    if prefix_id is not None:
        input_ids, attention_mask = prepend_token(input_ids, attention_mask, prefix_id)
        item_index = item_index + 1
    return input_ids.to(device), attention_mask.to(device), item_index


def forward(model, input_ids, attention_mask, pole_ids, n_top, layer=None):
    """Run the model. Return (pole logits (batch, n_poles), top token ids
    (batch, n_top), top logits (batch, n_top), hidden_states[layer] or None).

    Only the logits of the last real token are used. The pole and top logits
    are selected on the device, so the full vocabulary is not copied.
    """
    import torch

    # Only the logits of the last real token of each row are computed
    # (logits_to_keep, shape (batch, 1)). The full logits are batch x seq x
    # vocab, which is too large for memory.
    last = attention_mask.sum(dim=1) - 1
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=layer is not None,
            logits_to_keep=last[:, None],
        )
    logits = output.logits[:, 0].float()
    pole_logits = logits[:, pole_ids].cpu().numpy()
    top = torch.topk(logits, n_top, dim=1)
    hidden = None
    if layer is not None:
        hidden = output.hidden_states[layer].float()
    return (
        pole_logits,
        top.indices.cpu().numpy(),
        top.values.cpu().numpy(),
        hidden,
    )


def row_norms(hidden, mask):
    """Return the mean norm of hidden over the masked positions of each row."""
    norms = hidden.norm(dim=-1)
    mask = mask.to(norms.device)
    return (norms * mask).sum(dim=1) / mask.sum(dim=1)


def sweep_batch(config, model, tokenizer, batch, directions, ids, device, hook):
    """Return (result rows, top token rows, token info) for one batch of items."""
    import torch

    poles = list(ids)
    pole_ids = torch.tensor([ids[w] for w in poles], device=device)
    # Columns of the pole logits.
    typical_ids = [poles.index(w) for w in config["typical_poles"]]
    atypical_ids = [poles.index(w) for w in config["atypical_poles"]]
    n_top = config["n_top_tokens"]
    contexts = [item["context"] for item in batch]
    for context in contexts:
        check_context_is_prefix(tokenizer, context, poles)
    input_ids, attention_mask, item_index = encode_batch(
        tokenizer,
        contexts,
        [item["item_text"] for item in batch],
        config["prefix_token"],
        device,
    )
    token_info = [
        {
            "item_token_index": int(index),
            "item_token": tokenizer.decode([int(input_ids[row, index])]),
        }
        for row, index in enumerate(item_index)
    ]

    hook.vector = None
    unsteered = forward(
        model, input_ids, attention_mask, pole_ids, n_top, config["layer"]
    )
    hidden = unsteered[3]
    base_difference = logit_difference(unsteered[0], typical_ids, atypical_ids)

    rows = []
    top_rows = []
    for mode in config["steer_modes"]:
        mask = steer_mask(attention_mask.cpu(), mode, item_index).to(device)
        norms = row_norms(hidden, mask)
        for info, norm in zip(token_info, norms.tolist()):
            info[f"norm_{mode}"] = norm
        hook.mask = mask
        for name, direction in directions.items():
            vector = torch.tensor(direction, dtype=torch.float32, device=device)
            for constant in config["steering_constants"]:
                constant = float(constant)
                if constant == 0:
                    logits, top_ids, top_values, _ = unsteered
                else:
                    hook.vector = constant * norms[:, None] * vector[None, :]
                    logits, top_ids, top_values, _ = forward(
                        model, input_ids, attention_mask, pole_ids, n_top
                    )
                    hook.vector = None
                difference = logit_difference(logits, typical_ids, atypical_ids)
                for row, item in enumerate(batch):
                    record = {
                        "direction": name,
                        "steer_mode": mode,
                        "constant": constant,
                        "item": item["item"],
                        "category": item["category"],
                        "logit_difference": float(difference[row]),
                        "change": float(difference[row] - base_difference[row]),
                    }
                    for column, word in enumerate(poles):
                        record[f"logit_{word}"] = float(logits[row, column])
                    rows.append(record)
                    if name != TYPICALITY:
                        continue
                    for rank in range(n_top):
                        text = tokenizer.decode([int(top_ids[row, rank])])
                        value = float(top_values[row, rank])
                        top_rows.append(
                            {
                                "steer_mode": mode,
                                "constant": constant,
                                "item": item["item"],
                                "category": item["category"],
                                "rank": rank + 1,
                                "token": text,
                                "logit": value,
                            }
                        )
    return rows, top_rows, token_info


def sweep(config, model, tokenizer, items, directions, device):
    """Return (result rows, top token rows). Adds the steered token and the
    reference norms to each item dict."""
    poles = list(config["typical_poles"]) + list(config["atypical_poles"])
    ids = pole_token_ids(tokenizer, poles)
    hook, handle = register_steering(model, config["layer"])
    rows = []
    top_rows = []
    batch_size = config["batch_size"]
    try:
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            print(f"Items {start + 1}-{start + len(batch)} of {len(items)}")
            batch_rows, batch_top, token_info = sweep_batch(
                config, model, tokenizer, batch, directions, ids, device, hook
            )
            rows.extend(batch_rows)
            top_rows.extend(batch_top)
            for item, info in zip(batch, token_info):
                item.update(info)
    finally:
        handle.remove()
    return rows, top_rows


def run(config, config_path):
    from typicality_linearity.activations import choose_device, load_model

    run_id = make_run_id(config["experiment"])
    run_dir = Path(config["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    metadata = start_metadata(run_id, config, config_path, run_dir)
    write_metadata(metadata, run_dir / "metadata.json")
    try:
        set_seed(config["seed"])
        direction, selection = load_source_direction(config)
        metadata["selection"] = selection
        directions = {TYPICALITY: direction}
        randoms = random_directions(
            config["n_random_directions"], len(direction), config["seed"]
        )
        for k, vector in enumerate(randoms):
            directions[f"random_{k}"] = vector
        np.save(run_dir / "direction.npy", direction.astype(np.float32))

        items = read_items(config)
        device = choose_device(config["device"])
        metadata["device"] = device
        tokenizer, model = load_model(config["model"], device, config["dtype"])
        rows, top_rows = sweep(config, model, tokenizer, items, directions, device)

        write_csv(items, run_dir / "items.csv")
        write_csv(rows, run_dir / "results.csv")
        write_csv(top_rows, run_dir / "top_tokens.csv")
        write_html(
            config,
            selection,
            items,
            rows,
            top_rows,
            run_id,
            run_dir / config["html_name"],
        )
        metadata["status"] = "completed"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = "".join(traceback.format_exception(error))
        raise
    finally:
        metadata["finished_at"] = now_utc()
        write_metadata(metadata, run_dir / "metadata.json")
    return run_dir
