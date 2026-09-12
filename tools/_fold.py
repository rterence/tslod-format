"""`build_level` in pure numpy — the bucket fold, written from the specification.

This is the fold spec/v1 defines, expressed once so the conformance data has a
single source. It is deliberately the slow, obvious version: a conformance
corpus is not the place for a clever implementation, and every rule below is
one a reader has to get right.

## What is specified, and therefore what this implements

* Numeric mode produces `[min, max, first, mid, last]` in that column order.
* Every column folds from the row below, which is what lets a level be rebuilt
  from the level below without reading a raw sample:
  - `min` and `max` over the children's `min` and `max` columns;
  - `first` is the first child's `first` and `last` the last child's `last`;
  - `mid` is the **first field of child ⌊k/2⌋** of the bucket's `k` existing
    children — never the sample at the bucket's middle index. At level 1 a child
    is one raw sample, so `mid` is element `⌊k/2⌋` of the bucket.
* NaN is skipped in `min` and `max`, each column on its own. A column that is
  all NaN yields its first entry, at position 0 — that entry's own NaN, payload
  and all. `first`, `mid` and `last` are copied from the child the rule names
  and may be NaN while `min` and `max` are finite.
* The comparison is strict `<` and `>`, so on a tie the **first** occurrence
  wins and its bucket-relative index is what the positions carry. `first`, `mid`
  and `last` are named by the geometry, so no comparison decides them.
* Bitfield mode produces `[OR, AND]`, on the two's-complement bit pattern
  including the sign bit, and is defined only for integer dtypes. A mask bucket
  carries no `first`, `mid` or `last`.
* The ragged tail is folded, not dropped: the final bucket is short, is computed
  over exactly the elements it has, and its `k` is that count — which is what
  makes its `mid` a different child from a full bucket's.
* `from_raw=0` folds an `(N, 5)` level. Positions index the child, not the raw
  sample — chaining to a raw index is the caller's job
  (`_tslod_build.build_pyramid`).
"""

from __future__ import annotations

import numpy as np


def _numeric_group(block: np.ndarray, from_raw: bool):
    """Fold one bucket. `block` is `(k,)` raw or `(k, 5)` tuples."""
    k = block.shape[0]
    if from_raw:
        mins = maxs = block
        # A child is one raw sample, so its "first field" is the sample itself.
        first, mid, last = block[0], block[k // 2], block[k - 1]
    else:
        mins, maxs = block[:, 0], block[:, 1]
        # mid is the FIRST field of child k // 2 — column 2, not column 3. The
        # child's own mid describes the child's centre, not this bucket's.
        first, mid, last = block[0, 2], block[k // 2, 2], block[k - 1, 4]

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
    return mins[i_min], maxs[i_max], first, mid, last, i_min, i_max


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
    out = np.empty((n_out, 2 if aggregation_mode == 1 else 5), dtype=arr.dtype)
    pos = np.empty((n_out, 2), dtype=np.int64)

    for b in range(n_out):
        block = arr[b * bf:(b + 1) * bf]
        if aggregation_mode == 1:
            if from_raw:
                lanes_or = lanes_and = block
            else:
                lanes_or, lanes_and = block[:, 0], block[:, 1]
            out[b] = (np.bitwise_or.reduce(lanes_or), np.bitwise_and.reduce(lanes_and))
            pos[b] = (0, 0)
        else:
            mn, mx, first, mid, last, i_min, i_max = _numeric_group(
                block, bool(from_raw))
            out[b] = (mn, mx, first, mid, last)
            pos[b] = (i_min, i_max)

    return (out, pos) if positions else out
