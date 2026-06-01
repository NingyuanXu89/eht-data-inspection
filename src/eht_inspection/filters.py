"""Reusable DataFrame filtering helpers."""

import operator

OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "in": lambda s, v: s.isin(v),
    "not in": lambda s, v: ~s.isin(v),
}


def filter_df(df, filters):
    """
    Generic DataFrame filter.

    Parameters
    ----------
    df : pandas.DataFrame
    filters : dict
        Example:
        {
            "baseline": ("==", "AA"),
            "polarization": ("==", "RR"),
            "snr": (">=", 10)
        }

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe.
    """
    mask = True
    for col, condition in filters.items():
        op, value = condition
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe")
        if op not in OPS:
            raise ValueError(f"Unsupported operation '{op}'")
        mask = mask & OPS[op](df[col], value)
    return df[mask]


def combine_masks(*masks, how="and"):
    """
    Combine boolean masks with logical AND or OR.

    Parameters
    ----------
    masks
        Boolean scalars or array-like masks with matching shapes.
    how
        ``"and"`` for intersection or ``"or"`` for union.

    Returns
    -------
    scalar or array-like
        Combined mask with the same shape as the inputs.
    """
    if not masks:
        raise ValueError("At least one mask is required.")
    if how not in {"and", "or"}:
        raise ValueError("how must be either 'and' or 'or'.")
    out = masks[0]
    for mask in masks[1:]:
        if how == "and":
            out = out & mask
        else:
            out = out | mask
    return out


def invert_mask(mask):
    """
    Invert a boolean mask.

    Parameters
    ----------
    mask
        Boolean scalar, pandas Series, or NumPy array.

    Returns
    -------
    scalar or array-like
        Logical negation of ``mask``.
    """
    return ~mask


def equality_filters(**kwargs):
    """
    Build a ``filter_df``-compatible equality-filter dictionary.

    Returns
    -------
    dict[str, tuple[str, Any]]
        Mapping from column name to ``("==", value)``.
    """
    return {column: ("==", value) for column, value in kwargs.items()}
