import numpy as np

from typicality_linearity.activation_cache import (
    ExtractedActivations,
    cache_key,
    key_inputs,
    load_entry,
    save_entry,
)

POSITIONS = ["item", "period"]


def make_inputs(prompts):
    return key_inputs("model", "float32", [10, 15], POSITIONS, prompts, ["doll"])


def make_extracted():
    rng = np.random.default_rng(0)
    return ExtractedActivations(
        activations={
            p: rng.normal(size=(1, 2, 8)).astype(np.float32) for p in POSITIONS
        },
        token_spans={"item": [(2, 6)], "period": [(15, 16)]},
        token_texts={"item": [" doll"], "period": ["."]},
    )


def test_key_changes_when_prompts_change():
    key_a = cache_key(make_inputs(["A doll is a toy."]))
    key_b = cache_key(make_inputs(["A doll is a toy!"]))
    assert key_a != key_b
    assert key_a == cache_key(make_inputs(["A doll is a toy."]))


def test_key_does_not_join_prompts_ambiguously():
    a = key_inputs("m", "float32", [1], POSITIONS, ["ab", "c"], ["x"])
    b = key_inputs("m", "float32", [1], POSITIONS, ["a", "bc"], ["x"])
    assert cache_key(a) != cache_key(b)


def test_missing_entry_returns_none(tmp_path):
    assert load_entry(tmp_path, "rosch_simple", "abc", POSITIONS) is None


def test_save_then_load_gives_same_values(tmp_path):
    inputs = make_inputs(["A doll is a toy."])
    key = cache_key(inputs)
    extracted = make_extracted()
    save_entry(tmp_path, "rosch_example", key, inputs, extracted)

    loaded = load_entry(tmp_path, "rosch_example", key, POSITIONS)
    for position in POSITIONS:
        assert np.array_equal(
            loaded.activations[position], extracted.activations[position]
        )
        assert loaded.token_spans[position] == extracted.token_spans[position]
        assert loaded.token_texts[position] == extracted.token_texts[position]
    assert not list(tmp_path.glob("*.tmp"))


def test_key_changes_with_prefix_token():
    a = key_inputs("m", "float32", [1], POSITIONS, ["a"], ["x"], None)
    b = key_inputs("m", "float32", [1], POSITIONS, ["a"], ["x"], "<|endoftext|>")
    assert cache_key(a) != cache_key(b)
