import pytest
import torch

from typicality_linearity.activations import find_token_indices, prepend_token


def test_prepend_token_puts_token_first():
    ids = torch.tensor([[5, 6, 0], [7, 8, 9]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    new_ids, new_mask = prepend_token(ids, mask, 42)
    assert new_ids.tolist() == [[42, 5, 6, 0], [42, 7, 8, 9]]
    assert new_mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]


def test_find_token_indices_rejects_first_token():
    # "Doll: toy." with "Doll" as token 0.
    offsets = [[(0, 4), (4, 5), (5, 9), (9, 10)]]
    with pytest.raises(ValueError, match="attention sink"):
        find_token_indices(["Doll: toy."], ["doll"], offsets, ["item"])


def test_find_token_indices_shift_for_prefix():
    offsets = [[(0, 4), (4, 5), (5, 9), (9, 10)]]
    indices = find_token_indices(
        ["Doll: toy."], ["doll"], offsets, ["item", "period"], shift=1
    )
    assert indices == {"item": [1], "period": [4]}
