import numpy as np
import pytest

from typicality_linearity.probes import (
    fit_diff_of_means,
    make_folds,
    pearson,
    spearman,
)


def test_constant_input_is_nan_without_a_warning():
    import warnings

    a = np.array([1.0, 2.0, 3.0, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(pearson(a, np.ones(4)))
        assert np.isnan(spearman(np.ones(4), a))


def test_spearman_and_pearson():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a**3) == pytest.approx(1.0)
    assert pearson(a, a**3) < 1.0
    assert pearson(a, -a) == pytest.approx(-1.0)
    assert np.isnan(pearson(a, np.ones(4)))


def test_diff_of_means_finds_typicality_direction():
    rng = np.random.default_rng(0)
    direction = np.zeros(20)
    direction[3] = 1.0
    y = rng.normal(size=60)
    x = y[:, None] * direction + rng.normal(size=(60, 20)) * 0.1
    direction = fit_diff_of_means(x, y)
    assert np.argmax(np.abs(direction)) == 3
    assert direction[3] > 0
    assert spearman(x @ direction, y) > 0.9


def test_make_folds_are_per_category_balanced_and_repeatable():
    categories = ["a"] * 23 + ["b"] * 17
    values = np.arange(40.0)
    folds = make_folds(categories, values, 5, seed=0)
    assert np.array_equal(folds, make_folds(categories, values, 5, seed=0))
    for category, n in (("a", 23), ("b", 17)):
        counts = np.bincount(folds[np.array(categories) == category], minlength=5)
        assert counts.sum() == n
        assert counts.max() - counts.min() <= 1
