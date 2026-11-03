"""Baselines that are subtracted from activations before PCA.

Mean contrast: for item i rated in category c, the result is

    a(i, c) - mean over c' != c of a(i, c')

The rows come from `rosch_mean_contrast.csv`: one row for each item and each
prompt category. A group is all rows of one (item, item_category) pair. The
target row of a group has `is_item_category` true. The other rows of the group
give the baseline. All other categories are used, also categories in which the
item is rated (for example "tomato" in "vegetable" when c is "fruit").
"""

import numpy as np

BASELINES = (None, "mean_contrast")


def is_true(value):
    return str(value) == "True"


def mean_contrast_groups(rows):
    """Return a list of (target row index, [other row indices]), one per group.

    The order is the order of the target rows in `rows`.
    """
    group_rows = {}
    for index, row in enumerate(rows):
        key = (row["item"], row["item_category"])
        group_rows.setdefault(key, []).append(index)

    groups = []
    for key, indices in group_rows.items():
        targets = [i for i in indices if is_true(rows[i]["is_item_category"])]
        others = [i for i in indices if not is_true(rows[i]["is_item_category"])]
        if len(targets) != 1:
            raise ValueError(f"Group {key} has {len(targets)} target rows, not 1")
        if len(others) == 0:
            raise ValueError(f"Group {key} has no rows for other categories")
        groups.append((targets[0], others))
    groups.sort(key=lambda group: group[0])
    return groups


def mean_contrast(x, groups):
    """Return x[target] - mean(x[others]) for each group.

    x: array with shape (n_rows, ...). The result has shape (n_groups, ...).
    """
    result = np.empty((len(groups),) + x.shape[1:], dtype=x.dtype)
    for k, (target, others) in enumerate(groups):
        result[k] = x[target] - x[others].mean(axis=0)
    return result


def target_rows(rows, baseline):
    """Return the indices of the rows that become points in the plots."""
    if baseline is None:
        return list(range(len(rows)))
    if baseline == "mean_contrast":
        return [target for target, _ in mean_contrast_groups(rows)]
    raise ValueError(f"Unknown baseline {baseline!r}. Use one of {BASELINES}")


def apply_baseline(activations, rows, baseline):
    """Return {position: activations of the target rows, baseline subtracted}.

    activations: {position: array with shape (n_rows, n_layers, d_model)}
    """
    if baseline is None:
        return activations
    if baseline == "mean_contrast":
        groups = mean_contrast_groups(rows)
        return {
            position: mean_contrast(values, groups)
            for position, values in activations.items()
        }
    raise ValueError(f"Unknown baseline {baseline!r}. Use one of {BASELINES}")
