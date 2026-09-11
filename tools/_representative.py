"""The representative — one sample per bucket, written from the specification.

This is the selection spec/v1 defines under **The representative**, expressed
once so the conformance data has a single source. Like `_fold.py` it is
deliberately the slow, obvious version: a scalar walk in float64, bucket by
bucket, in the order and the association the specification fixes. A vectorised
version would have to prove that it sums left to right and compares strictly;
this one does both in plain sight.

## What is specified, and therefore what this implements

* Every numeric bucket at level L >= 1 carries one representative, chosen from
  the channel's RAW samples at every level — never from the level below's
  representatives. Bucket j covers raw samples `[j*S, (j+1)*S)` with
  `S = BF**L`, anchored at sample 0; a ragged last bucket covers the samples
  that exist.
* Coordinates. `x_i` is the sample index at fixed rate, and at variable rate the
  stored timestamp minus the group's `start_timestamp`, an exact integer
  difference. `y_i` is the value. Each becomes a float64 by round-to-nearest,
  ties-to-even, which is what Python's `float()` of an integer does.
* Anchors. A bucket's anchor is the float64 sum of `x` and the float64 sum of
  `y` over its samples whose `y` is not NaN, each accumulated left to right in
  ascending index, divided by how many there are. With none, the bucket is
  invalid and its anchor is NaN.
* Edges. The level's first bucket takes sample 0 and its last bucket the
  channel's last sample, unconditionally; a level of one bucket takes sample 0.
* The triangle. Every other bucket takes, among its samples whose `y` is not
  NaN, the one maximising `|(pax - nax) * (y - pay) - (pax - x) * (nay - pay)|`
  in exactly that association, `(pax, pay)` being the previous bucket's anchor
  and `(nax, nay)` the next one's. A running maximum seeded at -1 is replaced
  only by a strictly greater area, so the first of two equal areas wins and a
  NaN area never does. With no winner the bucket takes its first sample whose
  `y` is not NaN, and with no such sample its first sample as stored.

What is stored is the chosen sample itself: its value in the channel's dtype,
bit for bit, and its time by the one time rule. That is the caller's business
(`_tslod_build.build`), so this module returns raw indices and nothing else.

`np.sum` and `np.mean` are not used for the anchors, and must not be: numpy
sums a float array pairwise rather than left to right, and the last bit of an
anchor is enough to change which sample a bucket stores.
"""

from __future__ import annotations

import numpy as np

NAN = float("nan")


def coordinates(values, timestamps=None, start_timestamp: int = 0):
    """`(x, y)`, one float64 each per raw sample, as lists of Python floats.

    `timestamps` is None at fixed rate, where `x` is the sample index. At
    variable rate it holds every sample's stored timestamp, and `x` is that
    timestamp minus `start_timestamp` — subtracted as integers, so exactly, and
    only then rounded to float64.
    """
    y = [float(v) for v in np.ascontiguousarray(values).tolist()]
    if timestamps is None:
        x = [float(i) for i in range(len(y))]
    else:
        start = int(start_timestamp)
        x = [float(int(t) - start) for t in np.asarray(timestamps).tolist()]
        if len(x) != len(y):
            raise ValueError(f"{len(x)} timestamps for {len(y)} samples")
    return x, y


def bucket_anchor(xs, ys) -> tuple[float, float]:
    """The anchor `(ax, ay)` of one bucket's samples; NaN for an invalid bucket."""
    sum_x = sum_y = 0.0
    count = 0
    for x, y in zip(xs, ys):
        if y != y:                      # NaN: not a sample the anchor sees
            continue
        sum_x += x
        sum_y += y
        count += 1
    if count == 0:
        return NAN, NAN
    return sum_x / count, sum_y / count


def bucket_representative(xs, ys, prev_anchor, next_anchor) -> int:
    """The bucket-relative index of one bucket's representative, by the triangle.

    `xs` and `ys` are the bucket's own samples' coordinates, and the two anchors
    are its neighbours'. This is the rule for every bucket but a level's first
    and last, which take the channel's endpoints without asking.
    """
    pax, pay = prev_anchor
    nax, nay = next_anchor
    best = -1.0
    chosen = None
    for k, (x, y) in enumerate(zip(xs, ys)):
        if y != y:
            continue
        area = abs((pax - nax) * (y - pay) - (pax - x) * (nay - pay))
        if area > best:                 # strict, and False for a NaN area
            best = area
            chosen = k
    if chosen is not None:
        return chosen
    for k, y in enumerate(ys):          # no area won: first sample that is not NaN
        if y == y:
            return k
    return 0                            # no valid sample: the first as stored


def level_representatives(values, span: int, timestamps=None,
                          start_timestamp: int = 0) -> np.ndarray:
    """The raw index of every bucket's representative at one level.

    `values` is the whole channel's raw samples and `span` the level's bucket
    span, `branching_factor ** level`. Every bucket but the two edges is
    `bucket_representative` over its own samples and its neighbours' anchors,
    so the whole-channel rule is expressed through the one-bucket rule.
    """
    span = int(span)
    if span < 2:
        raise ValueError(f"a level >= 1 has a bucket span of at least 2, got {span}")
    x, y = coordinates(values, timestamps, start_timestamp)
    n = len(y)
    if n == 0:
        raise ValueError("an empty channel has no level above level 0")
    n_buckets = -(-n // span)
    bounds = [(j * span, min((j + 1) * span, n)) for j in range(n_buckets)]
    anchors = [bucket_anchor(x[a:b], y[a:b]) for a, b in bounds]

    out = np.empty(n_buckets, dtype=np.int64)
    for j, (a, b) in enumerate(bounds):
        if j == 0:
            out[j] = 0
        elif j == n_buckets - 1:
            out[j] = n - 1
        else:
            out[j] = a + bucket_representative(x[a:b], y[a:b],
                                               anchors[j - 1], anchors[j + 1])
    return out
