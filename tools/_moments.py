"""The moment fold — one implementation, shared by everything that writes it.

Every numeric value bucket at level >= 1 carries `(count, mean, M2, M3, M4)`
as a third stream, `(N, 5)` float64 row-major.

This module is the only place the fold is written. The file builder and every
generator import it, and so must any tool outside this repository that writes
version-1 files, because the fold's output is a byte-for-byte format obligation
and a second implementation of it would be a second answer.

## The order, which is normative

Floating-point addition is not associative, so the fold shape leaks into the
answer (measured: left-to-right and balanced-tree folds of the same buckets
differ in their last bits). The ruling therefore fixes the order at **two**
levels, so that a per-block parallel engine and a serial reader are
bit-identical:

* **Stored** moments of a level-k bucket are the strict left-to-right
  Chan/Pébay fold over its `branching_factor` children in ascending index.
  At level 1 a child is one raw sample; above it, a child is a level-(k-1)
  bucket's stored moment tuple.
* **A range query's** merged moments are the strict left-to-right fold within
  each block in ascending bucket index, then the strict left-to-right fold of
  the per-block results in ascending block index.

A single-level rule would not have made the parallel case identical, which is
why there are two.

## Only the four operations IEEE-754 rounds exactly

The order alone is not enough to make the fold reproducible. `+`, `-`, `*` and
`/` on float64 are correctly rounded and give the same bits on every IEEE-754
machine; `pow` does not, and is not required to. Raising a value to the third
or fourth power with `**` calls the platform's `pow`, whose last bit differs
between one C library and another, so the same samples folded in the same order
produced different files on two machines.

Every power in the merge is therefore written as multiplication: `d3` and `d4`
from `d2`, and `n3` from `n2`. `x * x` and `x ** 2` agree, but the association
is written out anyway so the expression says what it computes. An implementation
in another language must do the same and must not reach for a `powi`, `powf` or
`cbrt`: the stored moments are a byte-for-byte obligation, and only the four
exact operations can carry one.

## NaN

`count` is the number of **non-NaN** elements and the moments are over the
non-NaN elements. An all-NaN bucket has `count = 0` and all four moments as the
**canonical quiet NaN** (`0x7FF8000000000000`), consistent with the fold semantics's rule that
a manufactured NaN is the canonical one. Integers have no NaN, so `count` is
always the full bucket there.

An implementation that counts NaN elements instead poisons every statistic
above them, permanently and silently: once an accumulator's mean is NaN, every
merge above it is NaN too, and the pyramid reports nothing about a channel that
had one bad sample in a billion.
"""

from __future__ import annotations

import numpy as np

#: The canonical quiet NaN a kernel manufactures.
QNAN64 = np.frombuffer(np.uint64(0x7FF8000000000000).tobytes(), dtype=np.float64)[0]

#: Column order of the moment stream. Row-major: the five values of bucket 0,
#: then the five of bucket 1 — never planar.
COLUMNS = ("count", "mean", "M2", "M3", "M4")


def merge_arrays(a, b):
    """Elementwise Chan/Pébay pairwise merge of two arrays of moment tuples.

    `a` and `b` are each `(N, 5)` float64. Returns `(N, 5)`.

    Vectorised across buckets, **not** across the fold: every bucket advances
    through its own children in the same left-to-right order, one step at a
    time, so the result is bit-identical to folding each bucket separately in a
    Python loop. That identity is asserted in `corpus/runner` and re-checked by
    the generators; it is what makes a fast writer and a simple
    reader agree.
    """
    ca, ma, m2a, m3a, m4a = (a[:, i] for i in range(5))
    cb, mb, m2b, m3b, m4b = (b[:, i] for i in range(5))
    n = ca + cb

    # Guard the denominators so the discarded branches of the selects below
    # cannot raise; `np.where` picks, it does not accumulate, so a garbage
    # value in an unselected lane never reaches the result.
    safe_n = np.where(n > 0, n, 1.0)
    delta = mb - ma
    d2 = delta * delta
    d3 = d2 * delta
    d4 = d2 * d2
    n2 = safe_n * safe_n
    n3 = n2 * safe_n

    m2 = m2a + m2b + d2 * ca * cb / safe_n
    m3 = (m3a + m3b
          + d3 * ca * cb * (ca - cb) / n2
          + 3.0 * delta * (ca * m2b - cb * m2a) / safe_n)
    m4 = (m4a + m4b
          + d4 * ca * cb * (ca * ca - ca * cb + cb * cb) / n3
          + 6.0 * d2 * (ca * ca * m2b + cb * cb * m2a) / n2
          + 4.0 * delta * (ca * m3b - cb * m3a) / safe_n)
    mean = (ca * ma + cb * mb) / safe_n

    out = np.empty_like(a)
    out[:, 0] = n
    # Identity branches, in the order the scalar merge takes them: an empty
    # right operand leaves the accumulator alone; an empty accumulator is
    # replaced wholesale.
    b_empty = cb == 0
    a_empty = ca == 0
    for i, col in enumerate((mean, m2, m3, m4), start=1):
        col = np.where(a_empty, b[:, i], col)
        col = np.where(b_empty, a[:, i], col)
        out[:, i] = col
    both_empty = a_empty & b_empty
    if np.any(both_empty):
        out[both_empty, 1:] = QNAN64
    return out


def singles(values: np.ndarray) -> np.ndarray:
    """Raw samples -> `(N, 5)` moment tuples, one per sample.

    A non-NaN sample is `(1, x, 0, 0, 0)`; a NaN sample is
    `(0, qNaN, qNaN, qNaN, qNaN)` — count 0, so it is the identity of the merge
    and contributes nothing, which is what "NaN is skipped" means here.
    """
    x = np.asarray(values, dtype=np.float64)
    out = np.empty((len(x), 5), dtype=np.float64)
    valid = ~np.isnan(x)
    out[:, 0] = valid.astype(np.float64)
    out[:, 1] = np.where(valid, x, QNAN64)
    out[:, 2:] = 0.0
    out[~valid, 1:] = QNAN64
    return out


def fold_children(children: np.ndarray, k: int) -> np.ndarray:
    """Fold `(M, 5)` children into `(ceil(M/k), 5)` parents, left to right.

    The parents advance in lockstep — step `i` merges every parent's `i`-th
    child — so each parent sees its own children in ascending index and the
    result is the strict left-to-right fold the ruling names.
    """
    m = children.shape[0]
    n_out = -(-m // k)
    pad = n_out * k - m
    if pad:
        empty = np.empty((pad, 5), dtype=np.float64)
        empty[:, 0] = 0.0
        empty[:, 1:] = QNAN64
        children = np.concatenate([children, empty], axis=0)
    grid = children.reshape(n_out, k, 5)

    acc = np.ascontiguousarray(grid[:, 0, :])
    for i in range(1, k):
        acc = merge_arrays(acc, np.ascontiguousarray(grid[:, i, :]))
    return np.ascontiguousarray(acc)


def level1_from_raw(values: np.ndarray, bf: int) -> np.ndarray:
    """Level-1 stored moments: fold `bf` raw samples per bucket, left to right."""
    return fold_children(singles(values), bf)


def next_level(parent_children: np.ndarray, bf: int) -> np.ndarray:
    """Level k+1 stored moments from level k's, folding `bf` children."""
    return fold_children(parent_children, bf)


def fold_serial(parts) -> np.ndarray:
    """The scalar reference: strict left-to-right over a list of `(5,)` tuples.

    Used to verify `merge_arrays`' lockstep vectorisation is bit-identical.
    """
    acc = np.asarray(parts[0], dtype=np.float64).reshape(1, 5)
    for p in parts[1:]:
        acc = merge_arrays(acc, np.asarray(p, dtype=np.float64).reshape(1, 5))
    return acc.reshape(5)


def range_merge(blocks) -> np.ndarray:
    """A range query's merged moments, under the ruled TWO-level order.

    `blocks` is a sequence of `(N_b, 5)` arrays, in ascending block index, each
    holding that block's buckets in ascending bucket index. Within a block the
    fold is left-to-right; the per-block results are then folded left-to-right.

    The two levels are the point: a parallel engine folds each block on its own
    thread and combines the results in block order, and a serial reader folding
    every bucket left-to-right through the whole range gets the SAME bits only
    because the block boundary is part of the rule.
    """
    per_block = []
    for blk in blocks:
        arr = np.ascontiguousarray(np.asarray(blk, dtype=np.float64))
        if arr.shape[0] == 0:
            continue
        acc = arr[0:1, :].copy()
        for i in range(1, arr.shape[0]):
            acc = merge_arrays(acc, arr[i:i + 1, :])
        per_block.append(acc)
    if not per_block:
        out = np.empty((1, 5), dtype=np.float64)
        out[:, 0] = 0.0
        out[:, 1:] = QNAN64
        return out.reshape(5)
    acc = per_block[0]
    for nxt in per_block[1:]:
        acc = merge_arrays(acc, nxt)
    return acc.reshape(5)
