"""`build_level` in pure numpy — the bucket fold, written from the specification.

This is the fold spec/v1 defines, expressed once so the conformance data has a
single source. It is deliberately the slow, obvious version: a conformance
corpus is not the place for a clever implementation, and every rule below is
one a reader has to get right.

## What is specified, and therefore what this implements

* Numeric mode produces `[min, max, first, last]` in that column order.
* NaN is skipped in `min` and `max`. An all-NaN bucket yields four NaNs, with
  positions `[0, 0]`. `first` and `last` are the literal first and last
  elements of the bucket and may themselves be NaN.
* The comparison is strict `<` and `>`, so on a tie the **first** occurrence
  wins and its bucket-relative index is what the positions carry.
* Bitfield mode produces `[OR, AND, first, last]`, on the two's-complement bit
  pattern including the sign bit, and is defined only for integer dtypes.
* The ragged tail is folded, not dropped: the final bucket is short and is
  computed over exactly the elements it has.
* `from_raw=0` folds an `(N, 4)` level: `min` over the children's `min` column,
  `max` over their `max` column, `first` from the first child's `first`, `last`
  from the last child's `last`. Positions index the child, not the raw sample —
  chaining to a raw index is the caller's job (`_tslod_build.build_pyramid`).
"""

from __future__ import annotations

import numpy as np


def _numeric_group(block: np.ndarray, from_raw: bool):
    """Fold one bucket. `block` is `(k,)` raw or `(k, 4)` tuples."""
    if from_raw:
        mins = maxs = block
        first, last = block[0], block[-1]
    else:
        mins, maxs = block[:, 0], block[:, 1]
        first, last = block[0, 2], block[-1, 3]

    if np.issubdtype(block.dtype, np.floating):
        nan_min = np.isnan(mins)
        nan_max = np.isnan(maxs)
        # The two columns are resolved INDEPENDENTLY. Folding stored tuples,
        # the min column being all NaN says nothing about the max column, and
        # gating one on the other couples them exactly where the rule says
        # they do not touch. An all-NaN column falls back to its own first
        # child, at position 0, and the NaN that lands there is that child's
        # own — payload preserved, never a manufactured canonical one.
        #
        # +inf / -inf sentinels rather than np.nanargmin: argmin returns the
        # FIRST minimum, which is exactly the strict-`<` tie-break, and a
        # column whose real values are all +inf still resolves correctly
        # because the sentinel only replaces NaN.
        if nan_min.all():
            i_min = 0
        else:
            i_min = int(np.argmin(np.where(nan_min, np.inf, mins)))
        if nan_max.all():
            i_max = 0
        else:
            i_max = int(np.argmax(np.where(nan_max, -np.inf, maxs)))
    else:
        i_min = int(np.argmin(mins))
        i_max = int(np.argmax(maxs))
    return mins[i_min], maxs[i_max], first, last, i_min, i_max


def build_level(values: np.ndarray, branching_factor: int, from_raw: int,
                aggregation_mode: int = 0, positions: bool = False):
    """The fold, as spec/v1 defines it."""
    bf = int(branching_factor)
    if bf < 2:
        raise ValueError(f"branching_factor must be >= 2, got {bf}")
    arr = np.ascontiguousarray(values)
    if arr.size == 0:
        raise ValueError("empty array")
    if aggregation_mode == 1:
        if np.issubdtype(arr.dtype, np.floating):
            raise TypeError("bitfield aggregation_mode with a float dtype")
        if positions:
            raise ValueError("positions are not supported for bitfield aggregation")

    n = arr.shape[0]
    n_out = -(-n // bf)
    out = np.empty((n_out, 4), dtype=arr.dtype)
    pos = np.empty((n_out, 2), dtype=np.int64)

    for b in range(n_out):
        block = arr[b * bf:(b + 1) * bf]
        if aggregation_mode == 1:
            if from_raw:
                lanes = block
                first, last = block[0], block[-1]
            else:
                lanes_or, lanes_and = block[:, 0], block[:, 1]
                first, last = block[0, 2], block[-1, 3]
            if from_raw:
                acc_or = np.bitwise_or.reduce(lanes)
                acc_and = np.bitwise_and.reduce(lanes)
            else:
                acc_or = np.bitwise_or.reduce(lanes_or)
                acc_and = np.bitwise_and.reduce(lanes_and)
            out[b] = (acc_or, acc_and, first, last)
            pos[b] = (0, 0)
        else:
            mn, mx, first, last, i_min, i_max = _numeric_group(block, bool(from_raw))
            out[b] = (mn, mx, first, last)
            pos[b] = (i_min, i_max)

    return (out, pos) if positions else out
