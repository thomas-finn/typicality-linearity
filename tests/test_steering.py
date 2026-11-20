import csv

import numpy as np
import pytest
import torch

from typicality_linearity.steering import (
    best_setting,
    general_direction,
    logit_difference,
    random_directions,
    register_steering,
    steer_mask,
)
from typicality_linearity.steering_experiment import (
    check_config,
    make_context,
    typicality_thirds,
)
from typicality_linearity.steering_plot import build_plot_data, render_html


class Block(torch.nn.Module):
    def forward(self, x):
        return x * 2


class Inner(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block() for _ in range(n_layers)])


class TinyModel(torch.nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.model = Inner(n_layers)

    def forward(self, x):
        hidden = [x]
        for layer in self.model.layers:
            x = layer(x)
            hidden.append(x)
        return hidden


def test_steer_mask_all_skips_prefix_and_padding():
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    mask = steer_mask(attention, "all")
    assert mask.tolist() == [[0, 1, 1, 0], [0, 1, 1, 1]]


def test_steer_mask_last():
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    mask = steer_mask(attention, "last")
    assert mask.tolist() == [[0, 0, 1, 0], [0, 0, 0, 1]]


def test_hook_changes_hidden_state_of_layer_only():
    model = TinyModel()
    x = torch.ones(1, 3, 2)
    hook, handle = register_steering(model, layer=2)
    hook.vector = torch.tensor([1.0, -1.0])
    hook.mask = steer_mask(torch.ones(1, 3), "all")
    hidden = model(x)
    handle.remove()
    # hidden[1] is not steered. hidden[2] is steered at positions 1 and 2.
    assert torch.equal(hidden[1], 2 * x)
    expected = 4 * x
    expected[0, 1:] += torch.tensor([1.0, -1.0])
    assert torch.equal(hidden[2], expected)
    # The steering carries to the next layer.
    assert torch.equal(hidden[3], 2 * expected)
    # No vector: no change. After remove: no hook.
    assert torch.equal(model(x)[2], 4 * x)


def test_hook_no_vector_is_identity():
    model = TinyModel()
    _hook, handle = register_steering(model, layer=1)
    x = torch.randn(2, 4, 3)
    assert torch.equal(model(x)[1], 2 * x)
    handle.remove()


def test_register_rejects_bad_layer():
    with pytest.raises(ValueError):
        register_steering(TinyModel(3), 0)
    with pytest.raises(ValueError):
        register_steering(TinyModel(3), 4)


def test_general_direction_is_normalised_mean_of_unit_directions():
    weights = {
        "s/mc/a/p1": np.array([2.0, 0.0]),
        "s/mc/b/p1": np.array([0.0, 5.0]),
        "s/none/a/p1": np.array([-1.0, 0.0]),
        "t/mc/a/p1": np.array([-1.0, 0.0]),
    }
    direction, keys = general_direction(weights, "s", "mc")
    assert keys == ["s/mc/a/p1", "s/mc/b/p1"]
    assert np.allclose(direction, np.array([1.0, 1.0]) / np.sqrt(2))


def test_best_setting(tmp_path):
    path = tmp_path / "summary.csv"
    rows = [
        {"setting": "a", "baseline": "none", "metric": "spearman", "x": "0.3"},
        {"setting": "b", "baseline": "mc", "metric": "spearman", "x": "0.5"},
        {"setting": "c", "baseline": "mc", "metric": "pearson", "x": "0.9"},
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert best_setting(path, "x", "spearman")["setting"] == "b"


def test_logit_difference():
    logits = np.array([[1.0, 3.0, 0.0, -2.0], [0.0, 0.0, 0.0, 0.0]])
    assert np.allclose(logit_difference(logits, [0, 1], [2, 3]), [3.0, 0.0])


def test_random_directions_are_unit_and_seeded():
    a = random_directions(3, 10, seed=0)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, random_directions(3, 10, seed=0))


TEMPLATE = "As an example of {category}, {item} {be} fairly BLANK"


def test_make_context_forms():
    assert make_context(TEMPLATE, "fruit", "apple") == (
        "As an example of a fruit, the apple is fairly",
        "apple",
    )
    assert make_context(TEMPLATE, "clothing", "pants")[0].startswith(
        "As an example of clothing, the pants"
    )
    assert make_context(TEMPLATE, "vegetable", "spinach")[0] == (
        "As an example of a vegetable, spinach is fairly"
    )
    assert make_context(TEMPLATE, "vegetable", "brussels sprouts") == (
        "As an example of a vegetable, the Brussels sprouts are fairly",
        "Brussels sprouts",
    )
    assert make_context(TEMPLATE, "sport", "swimming")[0] == (
        "As an example of a sport, swimming is fairly"
    )


def test_typicality_thirds():
    categories = ["a"] * 6 + ["b"] * 3
    typicality = [5, 0, 4, 1, 3, 2, 0.5, 0.1, 0.9]
    thirds = typicality_thirds(categories, typicality)
    assert thirds.tolist() == [2, 0, 2, 0, 1, 1, 1, 0, 2]


def minimal_config():
    return {
        "experiment": "e",
        "model": "m",
        "source_run_dir": "s",
        "selection_statistic": "x",
        "selection_metric": "spearman",
        "prompts_csv": "p.csv",
        "categories": None,
        "max_rows": None,
        "batch_size": 2,
        "template": TEMPLATE,
        "typical_poles": ["typical"],
        "atypical_poles": ["odd"],
        "layer": 1,
        "steer_modes": ["all"],
        "steering_constants": [-1.0, 0.0, 1.0],
        "n_random_directions": 1,
        "n_top_tokens": 2,
        "device": "cpu",
        "dtype": "float32",
        "prefix_token": None,
        "seed": 0,
        "output_dir": "o",
        "html_name": "h.html",
    }


def test_check_config():
    config = minimal_config()
    check_config(config)
    config["steering_constants"] = [1.0]
    with pytest.raises(ValueError):
        check_config(config)


def test_plot_data_and_html():
    config = minimal_config()
    items = [
        {
            "item": "apple",
            "category": "fruit",
            "typicality": 1.5,
            "third": 2,
            "context": "As an example of a fruit, the apple is fairly",
        }
    ]
    rows = []
    for direction in ["typicality", "random_0"]:
        for constant in config["steering_constants"]:
            rows.append(
                {
                    "direction": direction,
                    "steer_mode": "all",
                    "constant": constant,
                    "item": "apple",
                    "category": "fruit",
                    "logit_difference": 1.0 + constant,
                    "change": constant,
                    "logit_typical": 2.0,
                    "logit_odd": 1.0,
                }
            )
    top = [
        {
            "steer_mode": "all",
            "constant": 0.0,
            "item": "apple",
            "category": "fruit",
            "rank": 1,
            "token": "</script>",
            "logit": 1.0,
        }
    ]
    selection = {
        "setting": "s",
        "baseline": "mc",
        "statistic": "x",
        "metric": "spearman",
        "value": 0.5,
        "n_directions_averaged": 2,
    }
    data = build_plot_data(config, selection, items, rows, top)
    mode = data["modes"]["all"]
    assert data["zero"] == 1
    assert mode["typicality"][0] == [0.0, 1.0, 2.0]
    assert mode["random"][0][0] == [-1.0, 0.0, 1.0]
    assert mode["tops"][0][1] == ["</script>"]
    page = render_html(data, "run")
    assert page.count("</script>") == 2  # the Plotly tag and the page script


def test_read_items_category_filter(tmp_path):
    from typicality_linearity.steering_experiment import read_items

    path = tmp_path / "p.csv"
    path.write_text(
        "item,category,rating_zscore,is_ambiguous_extended\n"
        "gun,weapon,-1.0,False\n"
        "apple,fruit,-1.0,False\n"
        "knife,weapon,0.5,False\n"
    )
    config = minimal_config()
    config["prompts_csv"] = str(path)
    config["categories"] = ["weapon"]
    items = read_items(config)
    assert [item["item"] for item in items] == ["gun", "knife"]
    assert [item["third"] for item in items] == [1, 0]
    config["categories"] = ["tool"]
    with pytest.raises(ValueError):
        read_items(config)
