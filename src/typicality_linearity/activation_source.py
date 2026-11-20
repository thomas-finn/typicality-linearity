"""Get activations from the activation cache, or from the model.

The model is loaded only the first time an entry is not in the cache (see
LazyModel). A run that finds all entries in the cache does not need torch.
See activation_cache.py for the cache key and file layout.
"""

from typicality_linearity.activation_cache import (
    cache_key,
    key_inputs,
    load_entry,
    model_cache_dir,
    save_entry,
)


class LazyModel:
    """Load the model the first time it is necessary, then keep it."""

    def __init__(self, config, metadata):
        self.config = config
        self.metadata = metadata
        self.tokenizer = None
        self.model = None
        self.device = None

    def extract(self, prompts, item_texts):
        # Import here so that a run from the cache does not need torch.
        from typicality_linearity.activations import (
            choose_device,
            extract_activations,
            load_model,
        )

        if self.model is None:
            self.device = choose_device(self.config["device"])
            self.metadata["device"] = self.device
            self.tokenizer, self.model = load_model(
                self.config["model"], self.device, self.config["dtype"]
            )
        return extract_activations(
            self.model,
            self.tokenizer,
            prompts,
            item_texts,
            self.config["layers"],
            self.config["positions"],
            self.config["batch_size"],
            self.device,
            self.config["prefix_token"],
        )


def get_activations(config, lazy_model, prompt_name, prompts, item_texts):
    """Return (ExtractedActivations, source). Source is "cache" or "model"."""
    cache_dir = config["activation_cache_dir"]
    if cache_dir is None:
        return lazy_model.extract(prompts, item_texts), "model"

    inputs = key_inputs(
        config["model"],
        config["dtype"],
        config["layers"],
        config["positions"],
        prompts,
        item_texts,
        config["prefix_token"],
    )
    key = cache_key(inputs)
    cache_dir = model_cache_dir(cache_dir, config["model"])
    cached = load_entry(cache_dir, prompt_name, key, config["positions"])
    if cached is not None:
        return cached, "cache"

    extracted = lazy_model.extract(prompts, item_texts)
    extra_info = {
        "device": lazy_model.device,
        "batch_size": config["batch_size"],
        "experiment": config["experiment"],
    }
    save_entry(cache_dir, prompt_name, key, inputs, extracted, extra_info)
    return extracted, "model"
