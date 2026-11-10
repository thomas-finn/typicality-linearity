"""Baselines that are subtracted from activations before PCA.

Mean contrast: the target row of a group minus the mean of the other rows of
the group. A group is all rows of one (item, item_category) pair. Two layouts
of rows are supported:

- Category contrast (`rosch_mean_cat_contrast.csv`): one row for each item and
  each prompt category. For item i rated in category c, the result is

      a(i, c) - mean over c' != c of a(i, c')

  The target row has `is_item_category` true. All other categories are used,
  also categories in which the item is rated (for example "tomato" in
  "vegetable" when c is "fruit").

- Item contrast (`rosch_mean_item_contrast.csv`): the category c is fixed and
  the item changes. For item i rated in c, the result is

      a(i, c) - mean over contrast items j of a(j, c)

  The target row has `is_anchor` true.
"""

import numpy as np

BASELINES = (None, "mean_contrast")


def is_true(value):
    return str(value) == "True"


# Columns that mark the target row of a group, one for each layout.
TARGET_COLUMNS = ("is_item_category", "is_anchor")


def target_column(rows):
    """Return the column that marks the target rows: is_item_category or is_anchor."""
    found = [column for column in TARGET_COLUMNS if column in rows[0]]
    if len(found) != 1:
        raise ValueError(
            f"Rows must have exactly one of the columns {TARGET_COLUMNS}, found {found}"
        )
    return found[0]


def mean_contrast_groups(rows):
    """Return a list of (target row index, [other row indices]), one per group.

    The target row has `is_item_category` or `is_anchor` true (see
    target_column). The order is the order of the target rows in `rows`.
    """
    column = target_column(rows)
    group_rows = {}
    for index, row in enumerate(rows):
        key = (row["item"], row["item_category"])
        group_rows.setdefault(key, []).append(index)

    groups = []
    for key, indices in group_rows.items():
        targets = [i for i in indices if is_true(rows[i][column])]
        others = [i for i in indices if not is_true(rows[i][column])]
        if len(targets) != 1:
            raise ValueError(f"Group {key} has {len(targets)} target rows, not 1")
        if len(others) == 0:
            raise ValueError(f"Group {key} has no contrast rows")
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
